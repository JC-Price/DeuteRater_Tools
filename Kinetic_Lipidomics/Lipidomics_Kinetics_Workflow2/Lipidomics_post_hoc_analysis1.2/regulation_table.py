#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
regulation_table.py
Dual‑mode version:
 - If run directly: uses Tkinter GUI (or accepts CLI filepath)
 - If imported: GUI disabled; expose process_table_from_path(filepath)
"""

import pandas as pd
import os


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_first_number(val):
    try:
        if pd.isna(val):
            return None
        first = str(val).split(",")[0]
        return int(first)
    except:
        return None


def assign_regulation(row):
    """Determine regulation type based on abundance and rate."""
    dAbun = row.get("Abundance Mean Diff", None)
    dTurn = row.get("Rate Mean Diff", None)

    if pd.isna(dAbun) or pd.isna(dTurn):
        return ""

    if dAbun > 0 and dTurn > 0:
        return "↑Synthesis"
    elif dAbun > 0 and dTurn < 0:
        return "↓Degradation"
    elif dAbun < 0 and dTurn > 0:
        return "↑Degradation"
    elif dAbun < 0 and dTurn < 0:
        return "↓Synthesis"

    return ""


def flux_change_arrow(val):
    if pd.isna(val):
        return "None"
    if round(val, 2) > 0:
        return "↑"
    if round(val, 2) < 0:
        return "↓"
    return "0"


# ------------------------------------------------------------
# Main processing logic — NO TKINTER HERE
# ------------------------------------------------------------

def process_table_from_path(filepath: str):
    """
    Headless (non-GUI) processor.
    When imported and called programmatically, Tk is not used.
    """

    df = pd.read_csv(filepath)

    # ---- Your original logic begins ----

    filtered = df[df["Metric"].isin(["Abundance", "Rate", "Flux"])]

    cols = [
        "Plot_Group", "Comparison", "Metric",
        "Mean_Diff_All", "t_All", "p_All", "N_All"
    ]

    out = filtered[cols]

    pivot = out.pivot_table(
        index=["Plot_Group", "Comparison"],
        columns="Metric",
        aggfunc="first"
    )

    pivot.columns = ["_".join(col[::-1]) for col in pivot.columns]
    pivot.columns = [c.replace("_All", "") for c in pivot.columns]
    pivot = pivot.reset_index()

    # Rename Flux_t → Flux_chi2
    newcols = {}
    for c in pivot.columns:
        new = c
        if "Flux_t" in c:
            new = c.replace("Flux_t", "Flux_chi2")
        new = new.replace("_", " ")
        newcols[c] = new

    cute = pivot.rename(columns=newcols)

    # Filtering
    if "Flux Mean Diff" in cute.columns:
        cute = cute[~cute["Flux Mean Diff"].isna()]
    if "Flux p" in cute.columns:
        cute = cute[cute["Flux p"] < 0.05]
    if "Flux N" in cute.columns:
        cute["Flux N numeric"] = cute["Flux N"].apply(extract_first_number)
        cute = cute[cute["Flux N numeric"].fillna(0) >= 10]
    if "Rate N" in cute.columns:
        cute["Rate N numeric"] = cute["Rate N"].apply(extract_first_number)
        mask_rate_ok = (
            cute["Rate N numeric"].isna()
            | (cute["Rate N numeric"] >= 10)
        )
        cute = cute[mask_rate_ok]

    cute = cute.drop(columns=[c for c in cute.columns if "numeric" in c], errors="ignore")

    # Flux Change
    cute["Flux Change"] = cute["Flux Mean Diff"].apply(flux_change_arrow)

    # Regulation
    cute["Regulation"] = cute.apply(assign_regulation, axis=1)

    # Remove Flux N column entirely
    cute = cute.drop(columns=[c for c in cute.columns if c == "Flux N"], errors="ignore")

    base_cols = ["Plot Group", "Comparison"]
    diff_cols = [
        "Flux Mean Diff",
        "Abundance Mean Diff",
        "Rate Mean Diff",
    ]
    p_cols = [
        "Flux p",
        "Abundance p",
        "Rate p",
    ]
    stat_cols = [
        "Flux chi2",
        "Abundance t",
        "Rate t",
    ]
    n_cols = [
        "Abundance N",
        "Rate N",
    ]

    diff_cols = [c for c in diff_cols if c in cute.columns]
    p_cols = [c for c in p_cols if c in cute.columns]
    stat_cols = [c for c in stat_cols if c in cute.columns]
    n_cols = [c for c in n_cols if c in cute.columns]

    ordered_cols = (
        base_cols +
        diff_cols +
        ["Flux Change", "Regulation"] +
        p_cols +
        stat_cols +
        n_cols
    )

    cute = cute[ordered_cols]
    cute = cute.sort_values(by=["Comparison", "Plot Group"])

    # ----- Rounding for final paper-ready table -----

    def format_p(val):
        try:
            if pd.isna(val):
                return val
            v = float(val)
            if v < 0.001:
                return f"{v:.2e}"
            return v   # no rounding
        except:
            return val

    def round_or_none(val, ndigits):
        if pd.isna(val):
            return val
        try:
            return round(float(val), ndigits)
        except:
            return val

    # Mean diffs → 3 decimals
    for col in ["Flux Mean Diff", "Abundance Mean Diff", "Rate Mean Diff"]:
        if col in cute.columns:
            cute[col] = cute[col].apply(lambda x: round_or_none(x, 3))

    # p-values
    for col in ["Flux p", "Abundance p", "Rate p"]:
        if col in cute.columns:
            cute[col] = cute[col].apply(format_p)

    # test stats → 2 decimals
    for col in ["Flux chi2", "Abundance t", "Rate t"]:
        if col in cute.columns:
            cute[col] = cute[col].apply(lambda x: round_or_none(x, 2))

    # recompute arrow after rounding
    if "Flux Mean Diff" in cute.columns:
        cute["Flux Change"] = cute["Flux Mean Diff"].apply(flux_change_arrow)

    # ----- Save output -----

    outdir = os.path.dirname(filepath)
    outfile = os.path.join(outdir, "table_of_regulation_fullwide_cute.csv")
    cute.to_csv(outfile, index=False, encoding="utf-8-sig")

    print(f"[RegTable] Saved: {outfile}")
    return outfile


# ------------------------------------------------------------
# GUI (only used when run directly)
# ------------------------------------------------------------

def process_table_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(
        title="Select paired test statstics csv",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not filepath:
        messagebox.showerror("Error", "No file selected.")
        return

    outfile = process_table_from_path(filepath)
    messagebox.showinfo("Success", f"Saved:\n{outfile}")


# ------------------------------------------------------------
# Dual‑mode main block
# ------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # If called as: python regulation_table.py file.csv
    if len(sys.argv) == 2:
        process_table_from_path(sys.argv[1])
    else:
        # No args → open GUI file picker
        process_table_gui()