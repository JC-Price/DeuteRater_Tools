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
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


# ───────────────────────────── Helpers ──────────────────────────────
def _safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        return series


def coerce_numeric_columns(df: pd.DataFrame, skip: Iterable[str] = ("abundances", "mzml_path")) -> pd.DataFrame:
    for col in df.columns:
        if col in skip:
            continue
        cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = _safe_to_numeric(cleaned)
    return df


def add_abundance_dr_columns(df: pd.DataFrame,
    *, align_col: str = "Alignment ID",
    path_col: str = "mzml_path",
    abundance_col: str = "abundances",
) -> pd.DataFrame:
    """Add one DR-corrected abundance column per mzML file; pivot index = (Alignment ID, mzML)."""
    df = df.copy()
    if path_col not in df.columns:
        return df

    df["file_name"] = df[path_col].apply(os.path.basename)

    def _sum_tuple(cell):
        if pd.isna(cell):
            return 0.0
        try:
            items = ast.literal_eval(cell)
            if isinstance(items, (list, tuple)):
                return sum(map(float, items))
        except Exception:
            pass
        try:
            return float(cell)
        except Exception:
            return 0.0

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


def bootstrap_ci(arr: np.ndarray, n_bootstrap: int = 1_000, ci: int = 95) -> tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    meds = np.array([np.nanmedian(np.random.choice(arr, size=arr.size, replace=True)) for _ in range(n_bootstrap)])
    alpha = (100 - ci) / 2
    return tuple(np.percentile(meds, [alpha, 100 - alpha]))


def _find_col_variant(df: pd.DataFrame, candidates: list[str]) -> str | None:
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


# ───────────────────────────── Main ──────────────────────────────
def main() -> None:
    tk.Tk().withdraw()

    exp_raw = simpledialog.askstring("Experiments",
                                     "Enter experiment/group codes, comma-separated (e.g. A2,A3,A4)")
    if not exp_raw:
        messagebox.showinfo("Cancelled", "No experiments entered.")
        sys.exit()
    EXPERIMENTS = [e.strip() for e in exp_raw.split(",") if e.strip()]

    fn_path = filedialog.askopenfilename(title="Select *Fraction-New* table")
    if not fn_path: sys.exit()
    rates_path = filedialog.askopenfilename(title="Select *Rates* table")
    if not rates_path: sys.exit()
    id_path = filedialog.askopenfilename(title="Select *Concatenated ID file* table")
    if not id_path: sys.exit()

    frac = read_any(fn_path)
    rates = read_any(rates_path)
    id_df = read_any(id_path)

    frac_lui_col = _find_col_variant(frac, ["Lipid Unique Identifier", "lui", "lipid_unique_identifier"])
    frac_align_col = _find_col_variant(frac, ["Alignment ID", "alignment id", "alignment_id"])
    rates_analyte_col = _find_col_variant(rates, ["analyte_id", "analyte id", "analyteid"])
    rates_group_col = _find_col_variant(rates, ["group_name", "group name"])
    id_align_col = _find_col_variant(id_df, ["Alignment ID", "alignment id", "alignment_id"])
    frac_nH_col = _find_col_variant(frac, ['nH'])

    if not frac_lui_col or not frac_align_col or not rates_analyte_col:
        raise KeyError("Missing critical columns in input tables.")

    # normalize Alignment IDs to avoid merge mismatches
    frac[frac_align_col] = frac[frac_align_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # ─── DR Abundance columns ───
    frac = add_abundance_dr_columns(frac, align_col=frac_align_col)
    frac = coerce_numeric_columns(frac)

    # ─── Build one melt-pivot wide Fraction table ───
    metrics = [
        "n_value","cv","n_value_stddev","num_nv_time_points","Median_FN_stddev",
        "n_val_lower_margin","n_val_upper_margin","All_n_values","N_value_time_points",
        "Filtered_out_N_values","Filtered_out_N_value_time_points","Reason_for_filtering","abundances"
    ]
    if "sample_group" not in frac.columns:
        raise KeyError("Fraction-New missing required 'sample_group' column.")

    frac_long = frac.melt(
        id_vars=[frac_align_col, "sample_group"],
        value_vars=[m for m in metrics if m in frac.columns],
        var_name="metric",
        value_name="value"
    )
    frac_long["col_id"] = frac_long["metric"] + "_" + frac_long["sample_group"]

    frac_wide = (
        frac_long.pivot_table(index=frac_align_col, columns="col_id", values="value", aggfunc="first")
        .reset_index()
    )

    # keep ID + formula + DR columns
    keep_cols = [frac_align_col, frac_lui_col, "Formula", frac_nH_col]
    dr_cols = [c for c in frac.columns if c.startswith("Abundance_") and c.endswith("_DR")]
    frac_keep = frac[keep_cols + dr_cols].drop_duplicates(subset=[frac_align_col])

    frac_wide = frac_wide.merge(frac_keep, on=frac_align_col, how="left")

    # ─── Rates ───
    num_cols = [
        "Abundance rate", "Abundance asymptote", "Abundance 95pct_confidence",
        "Abundance half life", "Abundance R2", "Abundance files observed in",
        "Abundance num_measurements", "Abundance num_time_points",
        "Abundance std_error", "Abundance std_error_A", "Abundance dof",
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
    merged_rf = frac_wide.merge(rates_wide, left_on=frac_lui_col, right_on=rates_analyte_col, how="outer")
    if id_align_col in id_df.columns:
        merged = merged_rf.merge(id_df, left_on=frac_align_col, right_on=id_align_col, how="left")
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
    
    drop_cols = [
        col for col in merged.columns
        if (col.endswith("_x") or col.endswith("_y")) and col not in protected_cols
    ]
    merged = merged.drop(columns=drop_cols, errors="ignore")
        
    

    # Drop any other _x/_y columns created by merge — but do not drop protected columns
    drop_cols = [
        col for col in merged.columns
        if (col.endswith("_x") or col.endswith("_y")) and col not in protected_cols
    ]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # ─── Per-experiment stats ───
    for exp in EXPERIMENTS:
        abd_cols = [c for c in merged.columns if c.endswith(f"_{exp}") and "Abundance" in c and "DR" not in c]
        dr_cols_exp = [c for c in merged.columns if c.startswith("Abundance_") and c.endswith("_DR")
                       and re.search(re.escape(exp), c, flags=re.I)]

        def _stats(cols, tag=""):
            if not cols:
                return
            arr = merged[cols].apply(pd.to_numeric, errors="coerce")
            merged[f"{exp}{tag}_median_abundance"] = arr.median(axis=1)
            merged[f"{exp}{tag}_abundance_stddev"] = arr.std(axis=1)
            merged[f"{exp}{tag}_abundance_RSD"] = (
                merged[f"{exp}{tag}_abundance_stddev"] / merged[f"{exp}{tag}_median_abundance"]
            )
            merged[[f"Abundance_lower_margin{tag}_{exp}", f"Abundance_upper_margin{tag}_{exp}"]] = (
                arr.apply(lambda r: bootstrap_ci(r.dropna().values), axis=1).apply(pd.Series)
            )

        _stats(abd_cols, "")
        _stats(dr_cols_exp, "_DR")

    # ─── p-values + FDR ───
    is_abn_col = lambda c: str(c).startswith("Abundance_") and all(x not in c for x in ["_median","_std","_RSD","_margin"])
    raw_cols = [c for c in merged.columns if is_abn_col(c) and "_DR" not in c]
    dr_cols = [c for c in merged.columns if is_abn_col(c) and "_DR" in c]

    abn_pvals, dr_pvals = [], []
    for _, row in merged.iterrows():
        a_raw = row[raw_cols].astype(float).dropna().values if raw_cols else np.array([])
        a_dr = row[dr_cols].astype(float).dropna().values if dr_cols else np.array([])
        if a_raw.size >= 2 and a_dr.size >= 2:
            p_pair = stats.ttest_rel(a_raw, a_dr, nan_policy="omit").pvalue if a_raw.size == a_dr.size else np.nan
            p_welch = stats.ttest_ind(a_raw, a_dr, equal_var=False, nan_policy="omit").pvalue
        else:
            p_pair = p_welch = np.nan
        abn_pvals.append(p_pair)
        dr_pvals.append(p_welch)

    merged["Abn-p"], merged["DR_Abn-p"] = abn_pvals, dr_pvals

    def _bh(p: pd.Series) -> np.ndarray:
        if multipletests is not None:
            return multipletests(p, method="fdr_bh")[1]
        p = np.asarray(p, float)
        n = p.size
        order = np.argsort(p)
        ranked = p[order]
        adj = np.empty_like(ranked)
        scaling = n / np.arange(1, n + 1)
        adj[-1] = ranked[-1]
        for i in range(n - 2, -1, -1):
            adj[i] = min(ranked[i] * scaling[i], adj[i + 1])
        out = np.empty_like(adj)
        out[order] = adj
        return out

    merged["Abn-p-BH"] = _bh(merged["Abn-p"].fillna(1.0))
    merged["DR_Abn-p-BH"] = _bh(merged["DR_Abn-p"].fillna(1.0))

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
