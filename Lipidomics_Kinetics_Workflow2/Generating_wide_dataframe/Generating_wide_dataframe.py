
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Rates → Fraction-New → Concatenated ID file
=====================================

• Rates joins Fraction-New by analyte_id ↔ Lipid Unique Identifier
• Fraction-New joins Concatenated ID file by Alignment ID
• Outputs one master table keyed by Alignment ID
"""

from __future__ import annotations
import os, re, sys, ast, warnings
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Iterable, Optional
import numpy as np
import pandas as pd

# ───────────────────────────── Helpers ──────────────────────────────
def _safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        return series


def coerce_numeric_columns(
    df: pd.DataFrame, skip: Iterable[str] = ("abundances", "mzml_path")
) -> pd.DataFrame:
    for col in df.columns:
        if col in skip:
            continue
        cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = _safe_to_numeric(cleaned)
    return df


def add_abundance_dr_columns(
    df: pd.DataFrame,
    *,
    align_col: str = "Alignment ID",
    path_col: str = "mzml_path",
    abundance_col: str = "abundances",
) -> pd.DataFrame:
    """Add one DeuteRater abundance column per mzML file; pivot index = (Alignment ID, mzML)."""
    df = df.copy()
    if path_col not in df.columns:
        return df

    df["file_name"] = df[path_col].apply(os.path.basename)

    def _sum_tuple(cell):
        # Treat missing/unparseable as NaN so means/SD and replicate counts behave correctly
        if pd.isna(cell):
            return np.nan
        try:
            items = ast.literal_eval(cell)
            if isinstance(items, (list, tuple)):
                return sum(map(float, items))
        except Exception:
            pass
        try:
            return float(cell)
        except Exception:
            return np.nan

    df["_abund_sum"] = df[abundance_col].apply(_sum_tuple)

    wide = (
        df.pivot_table(
            index=[align_col, "file_name"],
            values="_abund_sum",
            aggfunc="first"
        )
        .reset_index()
        .assign(col_name=lambda d: d["file_name"].map(lambda fn: f"Abundance_{os.path.splitext(fn)[0]}_DR"))
    )

    df_dr = wide.pivot(index=align_col, columns="col_name", values="_abund_sum").reset_index()
    return df.merge(df_dr, on=align_col, how="left").drop(columns=["_abund_sum", "file_name"], errors="ignore")


def _find_col_variant(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None:
        return None
    lowercols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace("_", " ").replace("-", " ")
        for lc, actual in lowercols.items():
            if lc == key or lc.replace("_", " ") == key or lc.replace("-", " ") == key:
                return actual
    return None


def flatten_cols(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            out.append(f"{c[0]}_{c[1]}" if c[1] else str(c[0]))
        else:
            out.append(str(c))
    return out


def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep="," if path.lower().endswith(".csv") else "\t")

def get_element_counts(formula: str):
    def grab(elem):
        m = re.search(rf"{elem}(\d+)", formula)
        return int(m.group(1)) if m else (1 if re.search(rf"{elem}(?![a-z])", formula) else 0)

    return grab("C"), grab("H"), grab("N"), grab("O"), grab("S")

# ───────────────────────────── Main ──────────────────────────────
def main() -> None:
    tk.Tk().withdraw()

    exp_raw = simpledialog.askstring(
        "Experiments",
        "Enter experiment/group codes, comma-separated (e.g. A2,A3,A4)"
    )
    if not exp_raw:
        messagebox.showinfo("Cancelled", "No experiments entered.")
        sys.exit()
    EXPERIMENTS = [e.strip() for e in exp_raw.split(",") if e.strip()]

    fn_path = filedialog.askopenfilename(title="Select *Fraction-New* table")
    if not fn_path:
        sys.exit()
    rates_path = filedialog.askopenfilename(title="Select *Rates* table")
    if not rates_path:
        sys.exit()
    id_path = filedialog.askopenfilename(title="Select *Concatenated ID file* table")
    if not id_path:
        sys.exit()

    frac = read_any(fn_path)
    rates = read_any(rates_path)
    id_df = read_any(id_path)

    # Fallback, in case n_value_SE not present currently
    if "n_value_SE" not in frac.columns:
        if "BA_nL_SE" in frac.columns:
            frac["n_value_SE"] = frac["BA_nL_SE"]
        else:
            # optional: create as all-NA so downstream code doesn't break
            frac["n_value_SE"] = pd.NA

    frac_lui_col = _find_col_variant(frac, ["Lipid Unique Identifier", "lui", "lipid_unique_identifier"])
    frac_align_col = _find_col_variant(frac, ["Alignment ID", "alignment id", "alignment_id"])
    rates_analyte_col = _find_col_variant(rates, ["analyte_id", "analyte id", "analyteid"])
    rates_group_col = _find_col_variant(rates, ["group_name", "group name"])
    id_align_col = _find_col_variant(id_df, ["Alignment ID", "alignment id", "alignment_id"])
    frac_nH_col = _find_col_variant(frac, ['nH'])

    if not frac_lui_col or not frac_align_col or not rates_analyte_col:
        raise KeyError("Missing critical columns in input tables.")

    # normalize Alignment IDs to avoid merge mismatches
    frac[frac_align_col] = (
        frac[frac_align_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    # ─── DR Abundance columns ───
    frac = add_abundance_dr_columns(frac, align_col=frac_align_col)
    frac = coerce_numeric_columns(frac)

    # ─── Build one melt-pivot wide Fraction table ───
    metrics = [
        "n_value", "cv", "n_value_stddev", "num_nv_time_points", "Median_FN_stddev",
        "n_val_lower_margin", "n_val_upper_margin", "All_n_values", "N_value_time_points",
        "Filtered_out_N_values", "Filtered_out_N_value_time_points", "Reason_for_filtering", "abundances",
        # Added Fraction-New BA fields
        "BA_rate", "BA_Asyn", "BA_nL_SD", "BA_rate_SD", "BA_Asyn_SD",
        "n_value_SD", "n_value_dof",
        "bootstrap_nL", "bootstrap_rate", "bootstrap_Asyn",
        "null_bootstrap_nL", "null_bootstrap_rate", "null_bootstrap_Asyn"  # <- fixed typo here
    ]

    if "sample_group" not in frac.columns:
        raise KeyError("Fraction-New missing required 'sample_group' column.")

    frac_long = frac.melt(
        id_vars=[frac_align_col, "sample_group"],
        value_vars=[m for m in metrics if m in frac.columns],
        var_name="metric",
        value_name="value"
    )
    # Keep null_bootstrap_* WITHOUT a group suffix; everything else gets "_{sample_group}"
    is_null = frac_long["metric"].str.startswith("null_bootstrap_")
    frac_long["col_id"] = np.where(
        is_null,
        frac_long["metric"],  # no suffix for null bootstraps
        frac_long["metric"] + "_" + frac_long["sample_group"]
    )

    frac_wide = (
        frac_long.pivot_table(index=frac_align_col, columns="col_id", values="value", aggfunc="first")
        .reset_index()
    )

    # ---------- FIXED: robust keep_cols / frac_keep construction ----------
    # keep ID + formula + DR columns (filter out None and missing names)
    candidate_keep = {
        frac_align_col,            # required and validated
        frac_lui_col,              # required and validated
        "Formula",                 # optional
        frac_nH_col,               # may be None
    }
    keep_cols = [c for c in candidate_keep if c is not None and c in frac.columns]

    # DR columns (if any)
    dr_cols = [c for c in frac.columns if c.startswith("Abundance_") and c.endswith("_DR")]

    # Optional diagnostics
    missing_advisories = []
    if frac_nH_col is None or frac_nH_col not in frac.columns:
        missing_advisories.append("nH")
    if "Formula" not in frac.columns:
        missing_advisories.append("Formula")
    if missing_advisories:
        warnings.warn(f"[merge] Optional columns not found in Fraction-New: {', '.join(missing_advisories)}")

    if not keep_cols and not dr_cols:
        # Minimal frame with just the key to allow the merge to proceed
        frac_keep = pd.DataFrame({frac_align_col: frac[frac_align_col].drop_duplicates()})
    else:
        # Deduplicate and preserve order; ensure the key is included first
        subset_cols = list(dict.fromkeys([frac_align_col] + keep_cols + dr_cols))
        frac_keep = frac[subset_cols].drop_duplicates(subset=[frac_align_col])

    # Merge these back
    frac_wide = frac_wide.merge(frac_keep, on=frac_align_col, how="left")
    # ---------- END FIX ----------

    # ─── Rates ───
    num_cols = [
        "Abundance rate", "Abundance asymptote", "Abundance 95pct_confidence",
        "Abundance half life", "Abundance R2", "Abundance files observed in",
        "Abundance num_measurements", "Abundance num_time_points",
        "Abundance SE_K", "Abundance SE_A", "Abundance dof",
        "Abundance 95pct_confidence_K", "Abundance 95pct_confidence_A"
    ]
    # the list-like columns we want to preserve (carry through as object dtype)
    list_cols = [c for c in ["rate_graph_time_points_x", "normed_isotope_data_y"] if c in rates.columns]

    # Only coerce numeric for numeric columns
    for c in (set(num_cols) & set(rates.columns)):
        rates[c] = pd.to_numeric(rates[c], errors="coerce")

    # Pivot values should include numeric + list columns (list columns remain object dtype)
    pivot_vals = [c for c in (num_cols + list_cols) if c in rates.columns]

    if rates_group_col and rates_group_col in rates.columns and pivot_vals:
        rates_wide = (
            rates.pivot_table(
                index=rates_analyte_col,
                columns=rates_group_col,
                values=pivot_vals,
                aggfunc="first"
            )
            .reset_index()
        )
        if isinstance(rates_wide.columns, pd.MultiIndex):
            rates_wide.columns = flatten_cols(rates_wide.columns)
    else:
        rates_wide = rates.copy()

    # ─── Merge Fractions + Rates + Concatenated ID file ───
    merged_rf = frac_wide.merge(
        rates_wide, left_on=frac_lui_col, right_on=rates_analyte_col, how="outer"
    )
    if id_align_col in id_df.columns:
        merged = merged_rf.merge(
            id_df, left_on=frac_align_col, right_on=id_align_col, how="left"
        )
    else:
        merged = merged_rf

    # ─── Clean up duplicate merge suffixes, but preserve specific list-columns ───
    # We restore the classic behavior of dropping merge-created _x/_y duplicates,
    # except we protect two real columns that legitimately end with _x or _y.
    protected_cols = {"rate_graph_time_points_x", "normed_isotope_data_y"}

    # If both Lipid Unique Identifier variants exist, keep the _x version and rename to canonical
    if "Lipid Unique Identifier_x" in merged.columns and "Lipid Unique Identifier_y" in merged.columns:
        merged = merged.drop(columns=["Lipid Unique Identifier_y"])
        merged = merged.rename(columns={"Lipid Unique Identifier_x": "Lipid Unique Identifier"})

    # If both Formula variants exist, keep the _x version and rename to canonical
    if "Formula_x" in merged.columns and "Formula_y" in merged.columns:
        merged = merged.drop(columns=["Formula_y"])
        merged = merged.rename(columns={"Formula_x": "Formula"})

    # Prefer the Fraction-New copy (_x) and canonicalize to "nH"
    if "nH_x" in merged.columns and "nH_y" in merged.columns:
        merged = merged.drop(columns=["nH_y"])
        merged = merged.rename(columns={"nH_x": "nH"})
    elif "nH_x" in merged.columns:
        merged = merged.rename(columns={"nH_x": "nH"})
    elif "nH_y" in merged.columns:
        # If only _y exists (e.g., came from Rates or Concatenated ID file), keep it but canonicalize
        merged = merged.rename(columns={"nH_y": "nH"})

    # Now drop other _x/_y columns created by the merge — but do not drop protected columns
    # (Add nH_x/nH_y to protected to be extra safe in case the rename above hasn’t fired yet.)
    protected_cols.update({"nH_x", "nH_y"})

    # Protect null bootstraps explicitly (unsuffixed form)
    protected_cols.update({
        "null_bootstrap_nL",
        "null_bootstrap_rate",
        "null_bootstrap_Asyn"
    })

    # Drop any other _x/_y columns created by merge — but do not drop protected columns
    drop_cols = [
        col for col in merged.columns
        if (col.endswith("_x") or col.endswith("_y")) and col not in protected_cols
    ]
    

    if "nH" not in merged.columns:
        merged["nH"] = merged["Adduct_cf"].apply(
            lambda cf: int(get_element_counts(str(cf))[1]) if pd.notna(cf) else np.nan
        )
    

    
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # ─── Per-experiment stats (placeholder loop retained for future use) ───
    for exp in EXPERIMENTS:
        abd_cols = [c for c in merged.columns if c.endswith(f"_{exp}") and "Abundance" in c and "DR" not in c]
        dr_cols_exp = [
            c for c in merged.columns
            if c.startswith("Abundance_") and c.endswith("DR") and re.search(re.escape(exp), c, flags=re.I)
        ]
        # (No-op here; keep for future metrics if needed)

    # Move Lipid Unique Identifier to be the 2nd column (if present)
    cols = list(merged.columns)
    if 'Lipid Unique Identifier' in cols:
        cols.insert(1, cols.pop(cols.index('Lipid Unique Identifier')))
        merged = merged[cols]

    # ─── Write ───
    outfile = os.path.join(os.path.dirname(fn_path) or ".", "merged_results.csv")
    merged.to_csv(outfile, index=False)
    messagebox.showinfo("Done", f"Merged table saved to {outfile}")
    print("Done. Wrote merged table to:", outfile)


if __name__ == "__main__":
    main()
