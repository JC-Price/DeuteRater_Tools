# -*- coding: utf-8 -*-
"""Creating_tidy_regression_matrix_with_CIs.py

Created on Wed Sep 3 13:11:45 2025
@author: Brigham Young Univ

Update (Jan 2026): Pre-file-dialog basic filtering window
---------------------------------------------------------
Before the input file browser opens, this script now shows a small Tk
window titled:

    Basic Filtering For Tidy Regression Dataframe

These filters are enforced on the WIDE dataframe BEFORE the tidy
(long) dataframe is created:

1) Exponential decay R² cutoff (default 0.6)
   - References per-sample columns named:
       Abundance R2_<sample_token>   OR   <sample_token>_Abundance R2
   - If R² < cutoff for a given sample, then for that sample we blank
     out (set to NA) the following per-sample metrics (so they drop
     during tidy creation):
       Abundance rate_<sample_token>
       Abundance asymptote_<sample_token>

2) Checkbox (default checked): n_L (n_value) < nH (hydrogen count)
   - References:
       nH   and   n_value_<sample_token>
   - If enabled, rows where n_value_<sample_token> >= nH are set to NA
     for that n_value column.

Notes:
- Filtering is applied by setting failing values to NA; the downstream
  melt/long step then drops NA values as usual.
"""

from __future__ import annotations

import re, json, ast
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import pandas as pd

import numpy as np

# ───────────────────────── Config ─────────────────────────


METRIC_TOKENS = [
    "DR_median_abundance",
    "n_value",
    "Abundance rate",
    "Abundance asymptote",

    # --- ADD: BA metrics ---
    "BA_rate",
    "BA_Asyn",

    # --- SE tokens so they survive normalization ---
    "n_value_SE",            # n_value_SE_{group}
    "DR_Abundance_SE",       # {group}_DR_Abundance_SE or DR_Abundance_SE_{group}
    "Abundance SE_K",        # Abundance SE_K_{group}
    "Abundance SE_A",        # Abundance SE_A_{group}

    # --- ADD: BA SE tokens ---
    "BA_rate_SE",
    "BA_Asyn_SE",
]



TOKENS_RE = "|".join(re.escape(t) for t in METRIC_TOKENS)

# SAMPLE_ID: allow letters, numbers, dash, dot (NO underscore)
SAMPLE_ID_RE = r"[A-Za-z0-9][A-Za-z0-9\-.]*"

# Treat these numeric values as missing and drop
SENTINEL_BAD_VALUES = {-4}

# Aux column tokens for filtering
R2_TOKEN = "Abundance R2"   # 'Abundance R2_<sample>' or '<sample>_Abundance R2'
NH_TOKEN = "nH"            # hydrogen count column

# ───────────────────────── Helpers ─────────────────────────

def normalize_headers_strict(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize per-sample metric columns to metric_sample form.

    Keeps ONLY columns that exactly match either:
        <METRIC_TOKEN>_<SAMPLE_ID>
        <SAMPLE_ID>_<METRIC_TOKEN>
    where METRIC_TOKEN is one of METRIC_TOKENS.

    Returns (renamed_df, detected_cols) where detected_cols are the
    normalized column names.
    """
    rename_map, detected = {}, []
    p1 = re.compile(rf"^(?P<metric>{TOKENS_RE})_(?P<sample>{SAMPLE_ID_RE})$", re.IGNORECASE)
    p2 = re.compile(rf"^(?P<sample>{SAMPLE_ID_RE})_(?P<metric>{TOKENS_RE})$", re.IGNORECASE)

    for col in df.columns:
        col_s = str(col)
        m = p1.match(col_s) or p2.match(col_s)
        if not m:
            continue
        metric = m.group("metric")
        sample = m.group("sample")
        norm = f"{metric}_{sample}"
        detected.append(col_s)
        if col_s != norm:
            rename_map[col_s] = norm

    if not detected:
        return df, []

    df2 = df.rename(columns=rename_map)
    detected_norm = [rename_map.get(str(c), str(c)) for c in detected]
    return df2, detected_norm


def normalize_aux_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize auxiliary columns used for filtering.

    - Abundance R2 columns → normalized to 'Abundance R2_<sample>'
      Accepts either 'Abundance R2_<sample>' or '<sample>_Abundance R2'.

    - nH column → normalized case-insensitively to 'nH' if present.

    Returns (df_renamed, info)
      info['r2_cols'] = list of normalized R2 columns
      info['nh_col']  = 'nH' if present else None
    """
    rename_map = {}
    r2_cols = []

    p_r2_1 = re.compile(rf"^{re.escape(R2_TOKEN)}_(?P<sample>{SAMPLE_ID_RE})$", re.IGNORECASE)
    p_r2_2 = re.compile(rf"^(?P<sample>{SAMPLE_ID_RE})_{re.escape(R2_TOKEN)}$", re.IGNORECASE)

    for col in list(df.columns):
        s = str(col)
        m = p_r2_1.match(s) or p_r2_2.match(s)
        if m:
            sample = m.group('sample')
            norm = f"{R2_TOKEN}_{sample}"
            r2_cols.append(norm)
            if s != norm:
                rename_map[s] = norm

    nh_col = None
    for col in df.columns:
        if str(col).strip().lower() == NH_TOKEN.lower():
            nh_col = NH_TOKEN
            if str(col) != NH_TOKEN:
                rename_map[str(col)] = NH_TOKEN
            break

    df2 = df.rename(columns=rename_map) if rename_map else df
    r2_cols = [rename_map.get(c, c) for c in r2_cols]

    return df2, {'r2_cols': sorted(set(r2_cols)), 'nh_col': nh_col}


def parse_interpretation(raw: str | None) -> dict:
    """Accept Python dict or JSON. Return {} if blank."""
    if not raw or not raw.strip():
        return {}
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    val = json.loads(raw)
    if isinstance(val, dict):
        return val
    raise ValueError("Provide a Python dict or JSON object.")


def map_from_substring(sample_id: str, mapping: dict[str, str]) -> str | None:
    """Boundary-aware, longest-token-first substring matching."""
    s = str(sample_id)
    keys = sorted(mapping.keys(), key=len, reverse=True)
    for k in keys:
        if re.search(rf"(?<!\w){re.escape(str(k))}(?!\w)", s, re.IGNORECASE):
            return mapping[k]
    for k in keys:
        if str(k).lower() in s.lower():
            return mapping[k]
    return None


def coerce_dtype(series: pd.Series, dtype: str | None, order: list | None):
    if not dtype:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().all():
            return num
        out = series.astype("category")
        if order and set(order).issuperset(set(out.dropna().unique())):
            out = out.cat.reorder_categories(order, ordered=True)
        return out

    dtype = dtype.lower()
    if dtype in ("int", "integer"):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if dtype in ("float", "numeric", "number"):
        return pd.to_numeric(series, errors="coerce")
    if dtype in ("category", "categorical", "cat"):
        out = series.astype("category")
        if order and set(order).issuperset(set(out.dropna().unique())):
            out = out.cat.reorder_categories(order, ordered=True)
        return out
    if dtype in ("string", "str"):
        return series.astype("string")
    return series


def build_variable(df_like: pd.DataFrame, var: str, spec):
    """Build interpreted variable from sample_id using mapping or regex extraction."""
    if isinstance(spec, dict) and any(k in spec for k in ("map", "regex", "type", "order", "missing", "group")):
        mapping = spec.get("map", None)
        regex = spec.get("regex", None)
        group = spec.get("group", 1)
        dtype = spec.get("type", None)
        order = spec.get("order", None)
        missing = spec.get("missing", None)
    else:
        mapping, regex, group, dtype, order, missing = (spec if isinstance(spec, dict) else None), None, 1, None, None, None

    out = pd.Series([None] * len(df_like), index=df_like.index, dtype="object")

    if regex:
        pattern = re.compile(str(regex))

        def extract(s):
            m = pattern.search(str(s))
            if not m:
                return None
            if isinstance(group, int):
                return m.group(group) if (m.lastindex or 0) >= group else None
            try:
                return m.group(str(group))
            except Exception:
                return None

        extracted = df_like["sample_id"].apply(extract)
        out = extracted.map(lambda x: mapping.get(str(x), mapping.get(x)) if (mapping and x is not None) else x)
    elif mapping:
        out = df_like["sample_id"].apply(lambda s: map_from_substring(s, mapping))

    if missing is not None:
        out = out.fillna(missing)

    return coerce_dtype(pd.Series(out, index=df_like.index), dtype, order)



def build_se_long_from_wide(wide_src: pd.DataFrame, sample_ids, REV_SE_MAP) -> pd.DataFrame:
    """
    Produce a long table with columns:
        ['Alignment ID', 'sample_id', 'metric', 'se']
    from SE columns that exist in `wide_src` after normalization.
    """
    # Candidate SE column names by sample
    se_tokens = list(REV_SE_MAP.keys())  # ["n_value_SE", "DR_Abundance_SE", "Abundance SE_K", "Abundance SE_A"]
    # Use only SE columns that actually exist
    se_cols = []
    for tok in se_tokens:
        for s in sample_ids:
            name = f"{tok}_{s}"  # normalized form
            if name in wide_src.columns:
                se_cols.append(name)

    if not se_cols:
        # Return empty; caller can handle (e.g., leave se as NA)
        return pd.DataFrame(columns=["Alignment ID", "sample_id", "metric", "se"])

    se_wide = wide_src[["Alignment ID"] + se_cols].copy()

    # Melt to long
    se_long = se_wide.melt(
        id_vars=["Alignment ID"],
        value_vars=se_cols,
        var_name="se_token_sample",
        value_name="se"
    )

    # Extract (se_token, sample_id) by splitting on the LAST underscore
    # Works with tokens that have spaces, e.g., "Abundance SE_K_A3"
    token_and_sample = se_long["se_token_sample"].str.rsplit("_", n=1, expand=True)
    se_long["se_token"] = token_and_sample[0]      # e.g., "Abundance SE_K"
    se_long["sample_id"] = token_and_sample[1]     # e.g., "A3"

    # Map se_token -> value metric
    se_long["metric"] = se_long["se_token"].map(REV_SE_MAP)

    # Keep only needed columns and coerce se to numeric
    se_long = se_long[["Alignment ID", "sample_id", "metric", "se"]]
    se_long["se"] = pd.to_numeric(se_long["se"], errors="coerce")

    return se_long



def tidy_with_se(long: pd.DataFrame, interp: dict, wide_src: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy dataframe using SEs instead of CIs.

    Inputs:
      - long: melted values with columns ['Alignment ID', ('Ontology'), 'metric', 'sample_id', 'value']
      - interp: interpretation dict
      - wide_src: the wide dataframe (filtered) that still contains the SE columns
                  after header normalization (IMPORTANT)
    Output:
      - ['Alignment ID', ('Ontology'), 'metric', 'value', 'se', ...interpreted vars]
    """
    

    SE_MAP = {
        "n_value": "n_value_SE",
        "DR_median_abundance": "DR_Abundance_SE",
        "Abundance rate": "Abundance SE_K",
        "Abundance asymptote": "Abundance SE_A",
    
        # --- ADD: BA metrics -> BA SE tokens ---
        "BA_rate": "BA_rate_SE",
        "BA_Asyn": "BA_Asyn_SE",
    }


    REV_SE_MAP = {v: k for k, v in SE_MAP.items()}
    
    # Keep only metrics we care about
    keep_metrics = set(SE_MAP.keys())
    long = long[long["metric"].isin(keep_metrics)].copy()

    # Build SE-long table (one-time melt of SE columns)
    sample_ids = sorted(long["sample_id"].dropna().astype(str).unique().tolist())
    se_long = build_se_long_from_wide(wide_src, sample_ids,REV_SE_MAP)

    # Attach SE by Alignment ID + metric + sample_id
    long = long.merge(se_long, on=["Alignment ID", "metric", "sample_id"], how="left")

    # Apply interpretation variables (optional)
    if interp:
        keys = ["Alignment ID", "sample_id"]
        base = long[keys].drop_duplicates()
        enrich = base.copy()
        for var, spec in interp.items():
            try:
                enrich[var] = build_variable(enrich, var, spec)
            except Exception as e:
                messagebox.showwarning("Variable skipped", f"Could not build variable '{var}': {e}")
        long = long.merge(enrich, on=keys, how="left")

    # Final order
    col_order = ["Alignment ID"]
    if "Ontology" in long.columns:
        col_order.append("Ontology")
    col_order += ["metric", "value", "se"]

    extras = [c for c in long.columns if c not in col_order + ["metric_sample"]]
    long = long[col_order + extras]

    return long




# ───────────────────────── Filtering UI + Logic ─────────────────────────


def show_basic_filter_window(root: tk.Tk) -> dict:
    """Show the small filter window BEFORE file dialogs.

    IMPORTANT: This uses the main root window (not a Toplevel) to avoid
    platform quirks where a withdrawn root prevents the popup from being
    visible.
    """
    settings = {
        "r2_cutoff": 0.6,
        "enforce_nL_lt_nH": True,
        "max_asymptote": 2.0,   # NEW
        "cancelled": True,
    }

    root.deiconify()
    root.title("Basic Filtering For Tidy Regression Dataframe")
    root.resizable(False, False)

    pad = 12
    frm = tk.Frame(root, padx=pad, pady=10)
    frm.pack(fill="both", expand=True)

    tk.Label(
        frm,
        text="These will be enforced in the wide dataframe before the tidy dataframe creation.",
        justify="left"
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    # --- R2 cutoff ---
    tk.Label(frm, text="Exponential decay R² cutoff (fractional turnover rate + asymptote):").grid(
        row=1, column=0, sticky="w"
    )

    r2_var = tk.StringVar(value=str(settings["r2_cutoff"]))
    tk.Entry(frm, textvariable=r2_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0))

    tk.Label(frm, text="references: Abundance R2_<sample_token>").grid(
        row=1, column=2, sticky="w", padx=(10, 0)
    )

    # --- NEW: Max asymptote ---
    tk.Label(frm, text="Maximum asymptote (A):").grid(row=2, column=0, sticky="w", pady=(10, 0))
    amax_var = tk.StringVar(value=str(settings["max_asymptote"]))
    tk.Entry(frm, textvariable=amax_var, width=10).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(10, 0))
    tk.Label(frm, text="references: Abundance asymptote_<sample_token>").grid(
        row=2, column=2, sticky="w", padx=(10, 0), pady=(10, 0)
    )

    # --- nL < nH checkbox ---
    enforce_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        frm,
        text="n_L (n_value) < nH (hydrogen count)   references: nH and n_value_<sample_token>",
        variable=enforce_var
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 12))

    btns = tk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=3, sticky="e")

    def do_cancel():
        settings["cancelled"] = True
        root.quit()

    def do_continue():
        # Validate R2 cutoff
        try:
            val = float(r2_var.get())
            if not (0.0 <= val <= 1.0):
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid R² cutoff", "Enter a number between 0 and 1.")
            return

        # Validate max asymptote
        try:
            amax = float(amax_var.get())
            if amax <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid maximum asymptote", "Enter a number > 0 (e.g., 2).")
            return

        settings["r2_cutoff"] = val
        settings["enforce_nL_lt_nH"] = bool(enforce_var.get())
        settings["max_asymptote"] = amax   # NEW
        settings["cancelled"] = False
        root.quit()

    tk.Button(btns, text="Cancel", command=do_cancel).pack(side="right", padx=(6, 0))
    tk.Button(btns, text="Continue", command=do_continue).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", do_cancel)
    root.mainloop()

    root.withdraw()
    return settings






def apply_basic_filters_wide(
    df: pd.DataFrame,
    detected_metric_cols: list[str],
    aux: dict,
    r2_cutoff: float,
    enforce_nL_lt_nH: bool,
    report: list[str] | None = None,
    show_summary_popup: bool = False,
    print_summary: bool = True,
    asymptote_max: float | None = None,     # define what "big" means
    drop_rows: bool = False,                # True = drop entire rows; False = set fields to NaN
) -> pd.DataFrame:
    """
    Apply filters to a wide dataframe.

    Behavior:
      - drop_rows=False (default): sets failing *cells* to np.nan (does not remove rows)
      - drop_rows=True: drops entire rows that fail any rule

    Assumptions:
      - Global constants exist: SAMPLE_ID_RE and R2_TOKEN
        e.g. SAMPLE_ID_RE = r"...", R2_TOKEN = "Abundance R2"
    """
    out = df.copy()

    ALIGN_COL = "Alignment ID"  # change if your column name differs

    def tell(msg: str):
        if report is not None:
            report.append(msg)

    def safe_messagebox(method: str, title: str, msg: str):
        if not show_summary_popup:
            return
        try:
            getattr(messagebox, method)(title, msg)
        except Exception:
            # If tkinter isn't available, just ignore popup requests
            pass

    def alignment_ids(mask: pd.Series, max_show: int = 20) -> str:
        """Compact Alignment ID string for rows where mask is True."""
        if ALIGN_COL not in out.columns:
            return f"(no '{ALIGN_COL}' column)"
        ids = out.loc[mask, ALIGN_COL].dropna().astype(str).unique().tolist()
        if not ids:
            return "(Alignment IDs: none/NA)"
        preview = ids[:max_show]
        extra = len(ids) - len(preview)
        if extra > 0:
            return f"(Alignment IDs: {', '.join(preview)} ... +{extra} more)"
        return f"(Alignment IDs: {', '.join(preview)})"

    # -----------------------------
    # Sample token detection (more robust)
    # -----------------------------
    # 1) From detected_metric_cols (your original approach)
    samples: set[str] = set()
    sample_re = None
    try:
        sample_re = re.compile(SAMPLE_ID_RE)
    except Exception:
        # If SAMPLE_ID_RE isn't defined or invalid, accept any trailing token
        sample_re = re.compile(r".+")

    for c in (detected_metric_cols or []):
        c = str(c)
        if "_" not in c:
            continue
        token = c.split("_")[-1]
        if sample_re.fullmatch(token):
            samples.add(token)

    # 2) Also infer samples from actual dataframe columns to avoid "samples is empty"
    #    This prevents "nothing happens" when detected_metric_cols is incomplete.
    def infer_samples_from_df(cols) -> set[str]:
        pats = [
            r"^Abundance rate_(.+)$",
            r"^Abundance asymptote_(.+)$",
            r"^n_value_(.+)$",
        ]
        try:
            pats.append(rf"^{re.escape(R2_TOKEN)}_(.+)$")
        except Exception:
            # If R2_TOKEN isn't defined, skip that pattern
            pass

        found = set()
        for col in map(str, cols):
            for p in pats:
                m = re.match(p, col)
                if m:
                    tok = m.group(1)
                    if sample_re.fullmatch(tok):
                        found.add(tok)
        return found

    samples |= infer_samples_from_df(out.columns)

    # If requested, track rows to drop (only used when drop_rows=True)
    rows_to_drop = pd.Series(False, index=out.index)

    # -----------------------------
    # Helper: ensure columns accept NaN
    # -----------------------------
    def ensure_numeric(colname: str):
        """Coerce a column to numeric so np.nan assignments are reliable."""
        if colname in out.columns:
            out[colname] = pd.to_numeric(out[colname], errors="coerce")

    # -----------------------------
    # 1) R² cutoff OR big-asymptote gating (combined exclusion)
    # -----------------------------
    # Do NOT gate this on aux["r2_cols"] because that can be empty even when columns exist.
    # Instead: run per-sample and only apply masks when the needed columns exist.
    if not samples:
        warn = "No sample tokens were detected from columns; no per-sample filters were applied."
        if print_summary:
            print(warn)
        tell(warn)
    else:
        for sample in sorted(samples):
            # Build column names
            try:
                r2_col = f"{R2_TOKEN}_{sample}"
            except Exception:
                r2_col = None

            asym_col = f"Abundance asymptote_{sample}"

            r2_bad = pd.Series(False, index=out.index)
            asym_bad = pd.Series(False, index=out.index)

            # R2 gating
            if r2_col and r2_col in out.columns:
                r2 = pd.to_numeric(out[r2_col], errors="coerce")
                r2_bad = r2.notna() & (r2 < r2_cutoff)

            # Asymptote gating (only if user provided a threshold)
            if asymptote_max is not None and asym_col in out.columns:
                asym = pd.to_numeric(out[asym_col], errors="coerce")
                asym_bad = asym.notna() & (asym > asymptote_max)

            combined_bad = r2_bad | asym_bad

            if not combined_bad.any():
                continue

            n_combined = int(combined_bad.sum())
            n_r2 = int(r2_bad.sum())
            n_asym = int(asym_bad.sum())
            aid_str = alignment_ids(combined_bad)

            if drop_rows:
                rows_to_drop |= combined_bad
                msg = (
                    f"EXCLUDING ROWS for sample '{sample}': {n_combined} rows "
                    f"(R² bad: {n_r2}, asymptote big: {n_asym}) {aid_str}"
                )
                if print_summary:
                    print(msg)
                tell(msg)
            else:
                # ✅ NaN-out individual cells (NOT pd.NA)
                changed_targets = []
                for target in (f"Abundance rate_{sample}", f"Abundance asymptote_{sample}"):
                    if target in out.columns:
                        ensure_numeric(target)
                        newly_set = int(out.loc[combined_bad, target].notna().sum())
                        out.loc[combined_bad, target] = np.nan
                        changed_targets.append(f"{target} (newly NaN: {newly_set})")

                        if print_summary:
                            print(
                                f"{target} set to NaN if (R²<{r2_cutoff} OR asym>{asymptote_max}) "
                                f"(newly NaN: {newly_set}) {aid_str}"
                            )

                if changed_targets:
                    tell(
                        f"R²/asymptote gating for sample '{sample}': "
                        f"{n_combined} rows affected (R² bad: {n_r2}, asymptote big: {n_asym}) -> "
                        + "; ".join(changed_targets)
                        + f" {aid_str}"
                    )

    # -----------------------------
    # 2) n_L (n_value) < nH enforcement
    # -----------------------------
    if enforce_nL_lt_nH:
        nh_col = aux.get("nh_col")
        if nh_col and nh_col in out.columns:
            nh = pd.to_numeric(out[nh_col], errors="coerce")

            for sample in sorted(samples):
                ncol = f"n_value_{sample}"
                if ncol not in out.columns:
                    continue

                nval = pd.to_numeric(out[ncol], errors="coerce")
                bad = nh.notna() & nval.notna() & (nval >= nh)

                if not bad.any():
                    continue

                n_bad = int(bad.sum())
                aid_str = alignment_ids(bad)

                if drop_rows:
                    rows_to_drop |= bad
                    msg = f"EXCLUDING ROWS for sample '{sample}': n_L<nH violated in {n_bad} rows {aid_str}"
                    if print_summary:
                        print(msg)
                    tell(msg)
                else:
                    ensure_numeric(ncol)
                    newly_set = int(out.loc[bad, ncol].notna().sum())
                    out.loc[bad, ncol] = np.nan

                    msg = (
                        f"n_L < nH: {ncol} set to NaN (newly NaN: {newly_set}), "
                        f"rows: {n_bad} for sample '{sample}' {aid_str}"
                    )
                    if print_summary:
                        print(msg)
                    tell(msg)
        else:
            msg = (
                "n_L < nH filter skipped: enforce_nL_lt_nH=True, "
                "but aux['nh_col'] was missing or not in dataframe."
            )
            if print_summary:
                print(msg)
            tell(msg)

    # -----------------------------
    # Drop rows once at the end (only if drop_rows=True)
    # -----------------------------
    if drop_rows and rows_to_drop.any():
        aid_str = alignment_ids(rows_to_drop)
        n_drop = int(rows_to_drop.sum())
        out = out.loc[~rows_to_drop].copy()
        msg = f"Dropped {n_drop} total rows after all filters {aid_str}"
        if print_summary:
            print(msg)
        tell(msg)

    # -----------------------------
    # Summary
    # -----------------------------
    if report is not None and report:
        summary = "\n".join(report)
        if print_summary:
            print(summary)
        safe_messagebox("showinfo", "Filters applied", summary)

    return out



# ───────────────────────── Main ─────────────────────────

def main() -> None:
    root = tk.Tk()

    # 0) Pre-file-dialog filter UI
    settings = show_basic_filter_window(root)
    if settings.get("cancelled"):
        messagebox.showinfo("Cancelled", "Operation cancelled.")
        root.destroy()
        return

    # 1) File browser
    in_path = filedialog.askopenfilename(
        title="Select the wide lipidomics CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not in_path:
        messagebox.showinfo("No file selected", "Operation cancelled.")
        root.destroy()
        return

    df = pd.read_csv(in_path)
    if "Alignment ID" not in df.columns:
        messagebox.showerror("Missing column", "'Alignment ID' column not found.")
        root.destroy()
        return

    # 2) Strict metric header normalization
    df, detected_cols = normalize_headers_strict(df)

    # 3) Normalize aux columns needed for filtering (R2, nH)
    df, aux = normalize_aux_columns(df)

    extra_id_cols = ["Ontology"] if "Ontology" in df.columns else []

    # Ontology uniqueness warning (unchanged)
    if "Ontology" in df.columns:
        nuniq = df.groupby("Alignment ID", dropna=False)["Ontology"].nunique()
        bad = nuniq[nuniq > 1]
        if not bad.empty:
            messagebox.showwarning(
                "Ontology mismatch",
                f"'Ontology' differs within the same Alignment ID for {len(bad)} IDs.\n"
                "Keeping the first occurrence per Alignment ID."
            )

    # 4) Keep only needed columns, including aux filtering columns
    cols_keep = ["Alignment ID"] + extra_id_cols + detected_cols
    for c in (aux.get("r2_cols") or []):
        if c in df.columns and c not in cols_keep:
            cols_keep.append(c)
    if aux.get("nh_col") and aux["nh_col"] in df.columns and aux["nh_col"] not in cols_keep:
        cols_keep.append(aux["nh_col"])

    df = df[cols_keep].copy().drop_duplicates(subset=["Alignment ID"], keep="first")
    df.columns = df.columns.map(str)

    # 5) Apply wide filters BEFORE tidy creation

    df = apply_basic_filters_wide(
        df,
        detected_metric_cols=detected_cols,
        aux=aux,
        r2_cutoff=float(settings["r2_cutoff"]),
        enforce_nL_lt_nH=bool(settings["enforce_nL_lt_nH"]),
        asymptote_max=float(settings["max_asymptote"]),   # ✅ THIS WAS MISSING
    )


    # 6) Wide → long (melt ONLY metric columns)
    id_vars = ["Alignment ID"] + extra_id_cols
    long = df.melt(id_vars=id_vars, value_vars=detected_cols, var_name="metric_sample", value_name="value")

    # Strict extraction of metric and sample_id
    m1 = long["metric_sample"].str.extract(
        rf"^(?P<metric>{TOKENS_RE})_(?P<sample_id>{SAMPLE_ID_RE})$", expand=True
    )
    m2 = long["metric_sample"].str.extract(
        rf"^(?P<sample_id2>{SAMPLE_ID_RE})_(?P<metric2>{TOKENS_RE})$", expand=True
    )

    long["metric"] = m1["metric"].where(m1["metric"].notna(), m2["metric2"])
    long["sample_id"] = m1["sample_id"].where(m1["sample_id"].notna(), m2["sample_id2"])
    long = long.dropna(subset=["metric", "sample_id"])

    # Clean values and drop sentinel values
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if SENTINEL_BAD_VALUES:
        long = long[~long["value"].isin(SENTINEL_BAD_VALUES)]
    long = long.dropna(subset=["value"])

    # Interpretation dictionary (Python dict or JSON)
    example = (
        '{\n'
        '  "genotype": {"map": {"A2": "APOE2", "A3": "APOE3", "A4": "APOE4"}, '
        '"type": "category", "order": ["APOE3", "APOE2", "APOE4"]}\n'
        '}'
    )

    raw = simpledialog.askstring(
        "Interpretation dictionary",
        "Paste a Python dict or JSON describing how to interpret sample_id into variables.\n"
        "Top-level keys become columns. Use 'map' for tokens or 'regex' to extract.\n\n"
        "EXAMPLE:\n" + example,
        initialvalue=example
    )

    try:
        interp = parse_interpretation(raw)
    except Exception as e:
        messagebox.showerror("Parse error", f"Could not parse the dictionary:\n{e}")
        interp = {}

    tidy = tidy_with_se(long, interp, df)

    out_path = filedialog.asksaveasfilename(
        title="Save tidy dataframe as…",
        defaultextension=".csv",
        initialfile="tidy_lipid_regression_with_ci.csv",
        filetypes=[("CSV files", "*.csv")]
    )

    if out_path:
        tidy.to_csv(out_path, index=False)
        messagebox.showinfo(
            "Done",
            f"Tidy data written to:\n{out_path}\n\nColumns: {', '.join(tidy.columns)}"
        )
    else:
        messagebox.showinfo("Cancelled", "No output saved; the tidy dataframe remains in memory as 'tidy'.")

    root.destroy()


if __name__ == "__main__":
    main()
