
#!/usr/bin/env python3
"""
Standardize lipid IDs in negative and positive modes.

Uses Tkinter to ask you for:
  • a positive‑mode CSV
  • a negative‑mode CSV
  • where to save the standardized positive CSV
  • where to save the standardized negative CSV
  • number of seconds overlap required for standardization (default: 8)
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd

from typing import Tuple


# -------------------------------
# Helpers
# -------------------------------
def label_standards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean column 'D7-standard' indicating whether 'cf' contains 'D7'.
    Raises KeyError if 'cf' is missing.
    """
    if "cf" not in df.columns:
        raise KeyError("Column 'cf' not found in dataframe.")
    df["D7-standard"] = df["cf"].astype(str).str.contains("D7", regex=False)
    return df


def convert_D_to_H(formula: str) -> str:
    """
    Convert any 'D' (deuterium) element count into 'H' (hydrogen) by summing H and D counts.
    Example: 'C6H2D5O3' -> 'C6H7O3'
    Leaves non-string or blank values unchanged.
    """
    import re

    if not isinstance(formula, str) or not formula.strip():
        return formula

    # Tokenize chemical formula like C6, H2, D5, O3...
    tokens = re.findall(r"[A-Z][a-z]?\d*", formula)

    elements = []
    h_count = 0
    d_count = 0

    for tok in tokens:
        elem = re.match(r"[A-Za-z]+", tok).group()
        num_match = re.search(r"\d+", tok)
        count = int(num_match.group()) if num_match else 1

        if elem == "H":
            h_count += count
            elements.append(("H_slot", None))
        elif elem == "D":
            d_count += count
            elements.append(("D_slot", None))
        else:
            elements.append((elem, count))

    total_H = h_count + d_count

    output = []
    used_h_slot = False

    for elem, count in elements:
        if elem in ("H_slot", "D_slot"):
            if not used_h_slot:
                output.append("H" if total_H == 1 else f"H{total_H}")
                used_h_slot = True
            continue

        output.append(elem if count == 1 else f"{elem}{count}")

    # If there were only D entries (no explicit H slot captured), add H total
    if (h_count == 0 and d_count > 0) and not used_h_slot:
        output.append("H" if total_H == 1 else f"H{total_H}")

    return "".join(output)


def _ensure_sum_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a 'Sum Composition' column.

    Logic:
      - If 'Lipid Unique Identifier' contains '|', use the part before '|'.
      - Else if 'Lipid Name' is present and not NaN, use it.
      - Else fallback to 'Lipid Unique Identifier' as string (or NaN if missing).
    """
    df = df.copy()

    if 'Sum Composition' in df.columns:
        return df

    if 'Lipid Unique Identifier' not in df.columns:
        raise KeyError("'Lipid Unique Identifier' column not found in dataframe.")

    def derive(row):
        lui = row.get('Lipid Unique Identifier')
        ln = row.get('Lipid Name')
        if isinstance(lui, str) and '|' in lui:
            return lui.split('|')[0].strip()
        if pd.notna(ln):
            return str(ln).strip()
        return str(lui).strip() if pd.notna(lui) else np.nan

    df['Sum Composition'] = df.apply(derive, axis=1)
    return df


# -------------------------------
# Standardization logic
# -------------------------------

import numpy as np
import pandas as pd
from typing import Tuple


def standardize_within_df(
    df: pd.DataFrame,
    rt_tol_sec: float = 8.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Standardize 'Lipid Unique Identifier' *within a single dataframe* and
    also unify 'Precursor Retention Time (sec)' and 'Precursor Retention Time (min)'
    to the chosen representative row's RT within each batch.

    Rules:
      1) Only unify rows with the same 'Sum Composition'.
      2) Only unify if retention time difference <= rt_tol_sec.
      3) If IDs differ:
         - Prefer the longer identifier string.
         - If equal length, choose the earliest occurrence in the original order.

    Returns a NEW dataframe (does not modify input in place).
    """
    # Ensure Sum Composition is present
    df = _ensure_sum_composition(df)

    # Ensure the minute column exists (derived from seconds if missing)
    if 'Precursor Retention Time (min)' not in df.columns:
        if 'Precursor Retention Time (sec)' not in df.columns:
            raise KeyError("'Precursor Retention Time (sec)' column not found.")
        df['Precursor Retention Time (min)'] = pd.to_numeric(
            df['Precursor Retention Time (sec)'], errors='coerce'
        ) / 60.0

    df = df.reset_index(drop=True).copy()
    df['_OriginalIndex'] = df.index  # stable for tie-breaks

    # Process each composition independently
    for comp, group in df.groupby('Sum Composition', sort=False):
        if len(group) <= 1:
            continue

        group_sorted = group.sort_values('Precursor Retention Time (sec)').copy()

        # Safe numeric RTs
        rts = pd.to_numeric(group_sorted['Precursor Retention Time (sec)'], errors='coerce').values
        idx = group_sorted['_OriginalIndex'].values

        # Keep only valid RT rows
        valid = ~np.isnan(rts)
        rts, idx = rts[valid], idx[valid]
        if len(rts) == 0:
            continue

        # Sliding window by RT
        start = 0
        while start < len(rts):
            batch = [idx[start]]
            end = start + 1
            while end < len(rts) and (rts[end] - rts[start] <= rt_tol_sec):
                batch.append(idx[end])
                end += 1

            # Choose canonical ID & RT for this batch
            sub = df.loc[batch].copy()
            sub['ID_len'] = (
                sub['Lipid Unique Identifier']
                .fillna("")
                .astype(str)
                .str.len()
            )
            # Prefer longer ID; tie-break on earliest original index
            chosen_row = sub.sort_values(['ID_len', '_OriginalIndex'], ascending=[False, True]).iloc[0]
            chosen_id = chosen_row['Lipid Unique Identifier']
            chosen_rt_sec = pd.to_numeric(chosen_row['Precursor Retention Time (sec)'], errors='coerce')
            chosen_rt_min = float(chosen_rt_sec) / 60.0 if pd.notna(chosen_rt_sec) else np.nan

            # Apply unified ID + RT to all rows in the batch
            df.loc[batch, 'Lipid Unique Identifier'] = chosen_id
            df.loc[batch, 'Precursor Retention Time (sec)'] = chosen_rt_sec
            df.loc[batch, 'Precursor Retention Time (min)'] = chosen_rt_min

            # Optional logging
            if verbose:
                for i in batch:
                    row = df.loc[i]
                    print(
                        f"Match found:\n"
                        f"  ID: {row['Lipid Unique Identifier']}\n"
                        f"  RT(sec): {row['Precursor Retention Time (sec)']}\n"
                        f"  Sum Composition: {comp}\n"
                        f"  Unified ID: {chosen_id}\n"
                        f"  Unified RT(sec): {chosen_rt_sec}\n"
                    )

            start = end

    return df.drop(columns=['_OriginalIndex'])





def standardize_within_df(
    df: pd.DataFrame,
    rt_tol_sec: float = 8.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Standardize 'Lipid Unique Identifier' *within a single dataframe* and
    also unify 'Precursor Retention Time (sec)' and 'Precursor Retention Time (min)'
    to the chosen representative row's RT within each batch.

    Rules:
      1) Only unify rows with the same 'Sum Composition'.
      2) Only unify if retention time difference <= rt_tol_sec.
      3) If IDs differ:
         - Prefer the longer identifier string.
         - If equal length, choose the earliest occurrence in the original order.

    Returns a NEW dataframe (does not modify input in place).
    """
    # Ensure Sum Composition is present
    df = _ensure_sum_composition(df)

    # Ensure the minute column exists (derived from seconds if missing)
    if 'Precursor Retention Time (min)' not in df.columns:
        if 'Precursor Retention Time (sec)' not in df.columns:
            raise KeyError("'Precursor Retention Time (sec)' column not found.")
        df['Precursor Retention Time (min)'] = pd.to_numeric(
            df['Precursor Retention Time (sec)'], errors='coerce'
        ) / 60.0

    df = df.reset_index(drop=True).copy()
    df['_OriginalIndex'] = df.index  # stable for tie-breaks

    # Process each composition independently
    for comp, group in df.groupby('Sum Composition', sort=False):
        if len(group) <= 1:
            continue

        group_sorted = group.sort_values('Precursor Retention Time (sec)').copy()

        # Safe numeric RTs
        rts = pd.to_numeric(group_sorted['Precursor Retention Time (sec)'], errors='coerce').values
        idx = group_sorted['_OriginalIndex'].values

        # Keep only valid RT rows
        valid = ~np.isnan(rts)
        rts, idx = rts[valid], idx[valid]
        if len(rts) == 0:
            continue

        # Sliding window by RT
        start = 0
        while start < len(rts):
            batch = [idx[start]]
            end = start + 1
            while end < len(rts) and (rts[end] - rts[start] <= rt_tol_sec):
                batch.append(idx[end])
                end += 1

            # Choose canonical ID & RT for this batch
            sub = df.loc[batch].copy()
            sub['ID_len'] = (
                sub['Lipid Unique Identifier']
                .fillna("")
                .astype(str)
                .str.len()
            )
            # Prefer longer ID; tie-break on earliest original index
            chosen_row = sub.sort_values(['ID_len', '_OriginalIndex'], ascending=[False, True]).iloc[0]
            chosen_id = chosen_row['Lipid Unique Identifier']
            chosen_rt_sec = pd.to_numeric(chosen_row['Precursor Retention Time (sec)'], errors='coerce')
            chosen_rt_min = float(chosen_rt_sec) / 60.0 if pd.notna(chosen_rt_sec) else np.nan

            # Apply unified ID + RT to all rows in the batch
            df.loc[batch, 'Lipid Unique Identifier'] = chosen_id
            df.loc[batch, 'Precursor Retention Time (sec)'] = chosen_rt_sec
            df.loc[batch, 'Precursor Retention Time (min)'] = chosen_rt_min

            # Optional logging
            if verbose:
                for i in batch:
                    row = df.loc[i]
                    print(
                        f"Match found:\n"
                        f"  ID: {row['Lipid Unique Identifier']}\n"
                        f"  RT(sec): {row['Precursor Retention Time (sec)']}\n"
                        f"  Sum Composition: {comp}\n"
                        f"  Unified ID: {chosen_id}\n"
                        f"  Unified RT(sec): {chosen_rt_sec}\n"
                    )

            start = end

    return df.drop(columns=['_OriginalIndex'])


def standardize_modes(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    rt_tol_sec: float = 8.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize lipid IDs within POS and NEG dataframes and then unify IDs AND RTs across modes.

    Steps:
      1) Ensure 'Sum Composition' exists in both frames.
      2) Run within-mode standardization (per composition, sliding window in RT).
      3) Assign unique global indices to ensure safe merging.
      4) For compositions present in both modes:
         - Concatenate POS+NEG for that composition.
         - Group by sliding RT window.
         - Pick canonical ID (longest; tie -> earliest original order).
         - Write chosen ID and RT (sec/min) back to the corresponding rows in POS/NEG.
      5) Return the updated POS and NEG dataframes.
    """
    # 1) Ensure 'Sum Composition'
    pos = _ensure_sum_composition(pos_df)
    neg = _ensure_sum_composition(neg_df)

    # Ensure the minute column exists (derived from seconds if missing)
    for df in (pos, neg):
        if 'Precursor Retention Time (min)' not in df.columns:
            if 'Precursor Retention Time (sec)' not in df.columns:
                raise KeyError("'Precursor Retention Time (sec)' column not found.")
            df['Precursor Retention Time (min)'] = pd.to_numeric(
                df['Precursor Retention Time (sec)'], errors='coerce'
            ) / 60.0

    # 2) Within-mode standardization (IDs + RTs)
    pos = standardize_within_df(pos, rt_tol_sec=rt_tol_sec, verbose=False)
    neg = standardize_within_df(neg, rt_tol_sec=rt_tol_sec, verbose=False)

    # 3) Stable global indices
    pos = pos.reset_index(drop=True).copy()
    neg = neg.reset_index(drop=True).copy()

    pos['_GlobalIndex'] = np.arange(len(pos))
    neg['_GlobalIndex'] = np.arange(len(pos), len(pos) + len(neg))
    pos['_OriginalIndex'] = pos.index  # deterministic tie-break
    neg['_OriginalIndex'] = neg.index

    # 4) Cross-mode unification for shared compositions
    shared_comps = set(pos['Sum Composition']).intersection(neg['Sum Composition'])

    for comp in shared_comps:
        pos_group = pos[pos['Sum Composition'] == comp]
        neg_group = neg[neg['Sum Composition'] == comp]

        combined = pd.concat([pos_group, neg_group], ignore_index=False).sort_values('Precursor Retention Time (sec)')
        rts = pd.to_numeric(combined['Precursor Retention Time (sec)'], errors='coerce').values
        gidx = combined['_GlobalIndex'].values

        valid = ~np.isnan(rts)
        rts, gidx = rts[valid], gidx[valid]
        if len(rts) == 0:
            continue

        start = 0
        while start < len(rts):
            batch = [gidx[start]]
            end = start + 1
            while end < len(rts) and (rts[end] - rts[start] <= rt_tol_sec):
                batch.append(gidx[end])
                end += 1

            sub = combined[combined['_GlobalIndex'].isin(batch)].copy()
            sub['ID_len'] = (
                sub['Lipid Unique Identifier']
                .fillna("")
                .astype(str)
                .str.len()
            )

            # Canonical selection (longest ID; tie -> earliest original order)
            chosen_row = sub.sort_values(['ID_len', '_OriginalIndex'], ascending=[False, True]).iloc[0]
            chosen_id = chosen_row['Lipid Unique Identifier']
            chosen_rt_sec = pd.to_numeric(chosen_row['Precursor Retention Time (sec)'], errors='coerce')
            chosen_rt_min = float(chosen_rt_sec) / 60.0 if pd.notna(chosen_rt_sec) else np.nan

            # Optional logging (restored print style)
            if verbose:
                for _, row in sub.iterrows():
                    # Determine mode from global index range
                    gi = int(row['_GlobalIndex'])
                    mode = "Positive" if gi < len(pos) else "Negative"
                    print(
                        f"Match found:\n"
                        f"  {mode} ID: {row['Lipid Unique Identifier']}, "
                        f"RT(sec): {row['Precursor Retention Time (sec)']}\n"
                        f"  Sum Composition: {comp}\n"
                        f"  Unified ID: {chosen_id}\n"
                        f"  Unified RT(sec): {chosen_rt_sec}\n"
                    )

            # Apply unified ID + RT back to POS/NEG via global index masks
            pos_mask = pos['_GlobalIndex'].isin(batch)
            neg_mask = neg['_GlobalIndex'].isin(batch)
            pos.loc[pos_mask, 'Lipid Unique Identifier'] = chosen_id
            neg.loc[neg_mask, 'Lipid Unique Identifier'] = chosen_id

            pos.loc[pos_mask, 'Precursor Retention Time (sec)'] = chosen_rt_sec
            neg.loc[neg_mask, 'Precursor Retention Time (sec)'] = chosen_rt_sec

            pos.loc[pos_mask, 'Precursor Retention Time (min)'] = chosen_rt_min
            neg.loc[neg_mask, 'Precursor Retention Time (min)'] = chosen_rt_min

            start = end

    # 5) Cleanup helper columns
    pos = pos.drop(columns=['_GlobalIndex', '_OriginalIndex'])
    neg = neg.drop(columns=['_GlobalIndex', '_OriginalIndex'])

    return pos, neg




# -------------------------------
# Main script (Tkinter I/O)
# -------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # hide the empty main window

    # Ask user for RT tolerance (seconds) with default = 8
    rt_tol_sec = simpledialog.askfloat(
        "RT Overlap",
        "Enter the number of seconds overlap required for standardization:",
        minvalue=0.0,
        initialvalue=8.0  # default value
    )
    if rt_tol_sec is None:
        messagebox.showwarning("Canceled", "No overlap time entered.")
        sys.exit(1)

    # 1) Pick positive-mode input
    pos_path = filedialog.askopenfilename(
        title="Select POSITIVE‑mode CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not pos_path:
        messagebox.showwarning("Canceled", "No positive CSV selected.")
        sys.exit(1)

    # 2) Pick negative-mode input
    neg_path = filedialog.askopenfilename(
        title="Select NEGATIVE‑mode CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not neg_path:
        messagebox.showwarning("Canceled", "No negative CSV selected.")
        sys.exit(1)
        
        
    # 3) Pick where to save standardized negative CSV
    out_neg = filedialog.asksaveasfilename(
        title="Save standardized NEGATIVE CSV as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not out_neg:
        messagebox.showwarning("Canceled", "No output file chosen for negative CSV.")
        sys.exit(1)

    # 4) Pick where to save standardized positive CSV
    out_pos = filedialog.asksaveasfilename(
        title="Save standardized POSITIVE CSV as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not out_pos:
        messagebox.showwarning("Canceled", "No output file chosen for positive CSV.")
        sys.exit(1)



    # Load CSVs
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)

    # Standardize within and across modes (returns NEW dataframes)
    pos_df, neg_df = standardize_modes(pos_df, neg_df, rt_tol_sec=rt_tol_sec, verbose=False)

    # 1) Label D7 standards in 'cf'
    pos_df = label_standards(pos_df)
    neg_df = label_standards(neg_df)

    # 2) Convert D → H in formula-like columns
    for df in (pos_df, neg_df):
        for col in ("cf", "Adduct_cf"):
            if col in df.columns:
                df[col] = df[col].apply(convert_D_to_H)

    # Save outputs
    pos_df.to_csv(out_pos, index=False)
    neg_df.to_csv(out_neg, index=False)

    messagebox.showinfo(
        "Done",
        "Standardization complete!\n\n"
        f"Positive → {out_pos}\nNegative → {out_neg}\n"
        f"RT Overlap used: {rt_tol_sec} sec"
    )
