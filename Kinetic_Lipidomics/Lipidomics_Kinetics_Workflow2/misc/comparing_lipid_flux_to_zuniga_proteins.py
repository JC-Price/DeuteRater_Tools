# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:06:25 2026

@author: Brigham Young Univ
"""

import pandas as pd
import numpy as np
import re
import os
from tkinter import Tk, filedialog

# -------------------------
# Helpers
# -------------------------

def clean_accession(x):
    if pd.isna(x):
        return None
    return re.split(r"[|\s]", str(x))[0]

def normalize_area_col(col, suffix=None):
    """
    Strip any trailing _XX suffix from Area column names.
    If suffix is provided, strips that specific suffix.
    Otherwise strips anything matching _[A-Z]+ at the end.
    This is robust to typos where e.g. a BM file has _BC suffixes on columns.
    """
    if suffix:
        pattern = re.escape(f"_{suffix}") + r"$"
    else:
        pattern = r"_[A-Z]+$"
    return re.sub(pattern, "", col)

def extract_genotype(col):
    m = re.search(r"Area\s+(A[234])", col)
    return m.group(1) if m else None

def compute_log2fc(row, num_cols, den_cols, min_reps=2, pseudocount=1.0):
    num = pd.to_numeric(row[num_cols], errors="coerce")
    den = pd.to_numeric(row[den_cols], errors="coerce")

    num[num <= 0] = np.nan
    den[den <= 0] = np.nan

    if num.count() < min_reps or den.count() < min_reps:
        return np.nan

    return np.nanmean(np.log2(num + pseudocount)) - np.nanmean(np.log2(den + pseudocount))


# -------------------------
# Tkinter file selection
# -------------------------

root = Tk()
root.withdraw()

print("Select one or more BC / BM proteomics CSV files")
proteomics_files = filedialog.askopenfilenames(
    title="Select BC/BM proteomics CSV files",
    filetypes=[("CSV files", "*.csv")]
)

if not proteomics_files:
    raise RuntimeError("No proteomics files selected.")

print("Select flux CSV file")
flux_file = filedialog.askopenfilename(
    title="Select flux CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not flux_file:
    raise RuntimeError("No flux file selected.")

# -------------------------
# Load & normalize proteomics
# -------------------------

tables = []

for f in proteomics_files:
    df = pd.read_csv(f)
    df["PrimaryAccession"] = df["Accession"].map(clean_accession)

    area_cols = [c for c in df.columns if c.startswith("Area ")]

    # Derive suffix from filename (e.g. "M1BC_for_Quant.csv" -> "BC")
    # so we strip the correct suffix regardless of what's in the column names
    fname_match = re.search(r"(BC|BM)", os.path.basename(f), re.IGNORECASE)
    fname_suffix = fname_match.group(1).upper() if fname_match else None

    rename_map = {c: normalize_area_col(c, suffix=fname_suffix) for c in area_cols}
    df = df.rename(columns=rename_map)

    area_cols = list(rename_map.values())
    df = df[["PrimaryAccession"] + area_cols]

    df_grp = df.groupby("PrimaryAccession", as_index=False).sum(numeric_only=True)
    tables.append(df_grp)

# -------------------------
# Merge all BC/BM tables (sum areas)
# -------------------------

combined = tables[0]
for t in tables[1:]:
    combined = combined.merge(t, on="PrimaryAccession", how="outer", suffixes=("", "_y"))
    for c in list(combined.columns):
        if c.endswith("_y"):
            base = c[:-2]
            combined[base] = combined[base].fillna(0) + combined[c].fillna(0)
            combined.drop(columns=c, inplace=True)

combined.fillna(0, inplace=True)

# -------------------------
# Identify genotype columns
# -------------------------

area_cols = [c for c in combined.columns if c.startswith("Area ")]

geno_cols = {"A2": [], "A3": [], "A4": []}
for c in area_cols:
    g = extract_genotype(c)
    if g in geno_cols:
        geno_cols[g].append(c)

# -------------------------
# Replicate counts
# -------------------------

combined["n_A2"] = combined[geno_cols["A2"]].gt(0).sum(axis=1)
combined["n_A3"] = combined[geno_cols["A3"]].gt(0).sum(axis=1)
combined["n_A4"] = combined[geno_cols["A4"]].gt(0).sum(axis=1)

# -------------------------
# Compute proteomics FCs
# -------------------------

combined["log2FC_A2_vs_A3"] = combined.apply(
    lambda r: compute_log2fc(r, geno_cols["A2"], geno_cols["A3"]),
    axis=1
)

combined["log2FC_A4_vs_A3"] = combined.apply(
    lambda r: compute_log2fc(r, geno_cols["A4"], geno_cols["A3"]),
    axis=1
)

# -------------------------
# Load & process flux data
# -------------------------

flux = pd.read_csv(flux_file)
flux["PrimaryAccession"] = flux["Accession_Mouse"].map(clean_accession)
flux["log2FC_Flux"] = np.log2(flux["Edge_FC"])

comparison_map = {
    "APOE2 vs APOE3": "log2FC_A2_vs_A3",
    "APOE4 vs APOE3": "log2FC_A4_vs_A3",
}

flux["Proteomics_FC_Column"] = flux["Comparison"].map(comparison_map)
flux = flux.dropna(subset=["Proteomics_FC_Column"])

# -------------------------
# Comparison-aware merge (n-restricted)
# -------------------------

rows = []

for _, f in flux.iterrows():
    acc = f["PrimaryAccession"]
    fc_col = f["Proteomics_FC_Column"]

    p = combined[combined["PrimaryAccession"] == acc]
    if p.empty:
        continue

    for _, pr in p.iterrows():
        if pr["n_A3"] < 2:
            continue
        if fc_col == "log2FC_A2_vs_A3" and pr["n_A2"] < 2:
            continue
        if fc_col == "log2FC_A4_vs_A3" and pr["n_A4"] < 2:
            continue

        rows.append({
            "PrimaryAccession": acc,
            "Comparison": f["Comparison"],
            "Proteomics_log2FC": pr[fc_col],
            "Flux_log2FC": f["log2FC_Flux"],
            "n_A2": pr["n_A2"],
            "n_A3": pr["n_A3"],
            "n_A4": pr["n_A4"],
        })

final_df = pd.DataFrame(rows)

# -------------------------
# Output
# -------------------------

out_dir = os.path.dirname(flux_file)
out_path = os.path.join(out_dir, "proteomics_flux_comparison.csv")
final_df.to_csv(out_path, index=False)

print(f"✅ Finished. Output written to:\n{out_path}")