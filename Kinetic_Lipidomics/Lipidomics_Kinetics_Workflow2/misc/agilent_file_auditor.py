# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:42:30 2025
This program provides and easy way of getting the file-run order and data aquisition method
for your agilent (.d) files. Just select the folder with your .d files and it will 
output a .csv with the current file name, what the original sample name was, the vial position
the run date-time and LC-MS method of aquisition. 

"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Regex that matches positions like P1-A1, P2-F9 (case-insensitive)
POS_RE = re.compile(r"^\s*P(?P<tray>[12])-(?P<row>[A-Fa-f])(?P<col>[1-9])\s*$")

# Precompute row letter -> 0..5 (A..F)
ROW_INDEX = {chr(ord('A') + i): i for i in range(6)}  # {'A':0, ... 'F':5}

def vial_number_from_position(pos: str) -> int | None:
    """
    Convert 'P1-A1' to vial number using:
      - P1 first (1..54), then P2 (55..108)
      - 6 rows (A..F), 9 columns (1..9)
      - row-major numbering: A1..A9 = 1..9, B1 = 10, ..., F9 = 54
    Returns None if pos is invalid.
    """
    if not isinstance(pos, str):
        return None
    m = POS_RE.match(pos)
    if not m:
        return None

    tray = int(m.group("tray"))               # 1 or 2
    row_letter = m.group("row").upper()       # 'A'..'F'
    col = int(m.group("col"))                 # 1..9

    # Map row to 0..5
    r = ROW_INDEX.get(row_letter)
    if r is None or not (1 <= col <= 9) or tray not in (1, 2):
        return None

    # Per-tray offset: 0 for P1, 54 for P2
    tray_offset = (tray - 1) * (6 * 9)  # 54 per tray
    # Row-major index within tray: r*9 + (col-1), then +1 for 1-based
    return tray_offset + r * 9 + (col - 1) + 1

def extract_sample_info(root_dir):
    results = []

    for folder in Path(root_dir).iterdir():
        if not folder.is_dir():
            continue

        xml_path = folder / "AcqData" / "sample_info.xml"
        if not xml_path.exists():
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            sample_name = None
            vial_position = None
            acquisition_time = None
            method_file = None

            # Walk the XML as DisplayName/Value pairs
            elems = list(root.iter())
            for i, elem in enumerate(elems):
                if elem.tag == "DisplayName":
                    label = elem.text or ""
                    if i + 1 < len(elems) and elems[i + 1].tag == "Value":
                        val = (elems[i + 1].text or "").strip()
                        if label == "Data File":
                            sample_path = val
                            sample_name = os.path.basename(sample_path)
                        elif label == "Sample Position":
                            vial_position = val
                        elif label == "Acquisition Time":
                            acquisition_time = val
                        elif label == "Method":
                            method_file = val

            vial_number = vial_number_from_position(vial_position)

            # Append a row if we got at least one of the key fields
            if sample_name or vial_position:
                results.append({
                    "File_Name": folder.name,        # actual folder name
                    "Sample_name": sample_name,
                    "Vial_Position": vial_position,
                    "Vial_Number": vial_number,
                    "Acquisition_Time": acquisition_time,
                    "Method": method_file
                })

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")

    return pd.DataFrame(results, columns=[
        "File_Name", "Sample_name", "Vial_Position", "Vial_Number",
        "Acquisition_Time", "Method"
    ])


if __name__ == "__main__":
    # Tkinter directory picker
    root = tk.Tk()
    root.withdraw()
    root_directory = filedialog.askdirectory(
        title="Select the directory containing sample subfolders"
    )
    if not root_directory:
        print("No directory selected. Exiting.")
        raise SystemExit(0)

    df = extract_sample_info(root_directory)
    df = df.dropna(subset = ['Vial_Number'])
    #df = df.drop_duplicates(subset = ['Vial_Number'], keep = 'first')
    df = df.sort_values(['Vial_Number'])
    print(df)

    out_path = Path(root_directory) / "sample_info_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")
