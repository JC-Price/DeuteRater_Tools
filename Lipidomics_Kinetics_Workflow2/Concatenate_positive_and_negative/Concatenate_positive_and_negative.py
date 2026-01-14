#!/usr/bin/env python3
"""
Concatenate two tabular files (CSV, TSV, or Excel).
- Prompts first for the NEGATIVE input file.
- Prompts second for the POSITIVE input file.
- Adds a 'polarity' column indicating "negative" or "positive".
- Prompts for the output file location.
- Saves the concatenated DataFrame (rows stacked, index reset).

Requires: pandas ≥ 1.0
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def read_table(path: str) -> pd.DataFrame:
    """Load CSV/TSV/Excel based on extension."""
    ext = path.lower().split('.')[-1]
    if ext in {"csv", "tsv"}:
        sep = "," if ext == "csv" else "\t"
        return pd.read_csv(path, sep=sep)
    else:  # assume Excel
        return pd.read_excel(path)

def main() -> None:
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # --- select the two source files ---
    neg_file = filedialog.askopenfilename(title="Select NEGATIVE data file")
    if not neg_file:
        messagebox.showwarning("Cancelled", "No negative file selected. Bye!")
        return

    pos_file = filedialog.askopenfilename(title="Select POSITIVE data file")
    if not pos_file:
        messagebox.showwarning("Cancelled", "No positive file selected. Bye!")
        return

    # --- read data ---
    df_neg = read_table(neg_file)
    df_pos = read_table(pos_file)

    # --- add polarity column ---
    df_neg["polarity"] = "negative"
    df_pos["polarity"] = "positive"

    # --- concatenate across rows ---
    combined = pd.concat([df_neg, df_pos], ignore_index=True)

    # --- choose output destination ---
    out_path = filedialog.asksaveasfilename(
        title="Save concatenated file as…",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")]
    )
    if not out_path:
        messagebox.showwarning("Cancelled", "No output path chosen. Bye!")
        return

    # --- save based on extension ---
    if out_path.lower().endswith(".xlsx"):
        combined.to_excel(out_path, index=False)
    else:
        combined.to_csv(out_path, index=False)

    messagebox.showinfo("Success", f"Concatenated file saved to:\n{out_path}")

if __name__ == "__main__":
    main()
