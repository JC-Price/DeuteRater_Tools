# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 13:11:45 2025

@author: Brigham Young Univ
"""

#!/usr/bin/env python3
# tidy_regression_prep_with_ci_bounds.py
#
# Strictly keeps ONLY columns that exactly match:
#   <METRIC_TOKEN>_<SAMPLE_ID>  OR  <SAMPLE_ID>_<METRIC_TOKEN>
# where METRIC_TOKEN ∈ {
#   "DR_median_abundance",
#   "n_value",
#   "Abundance rate",
#   "Abundance asymptote",
#   "median_abundance",
#   # plus CI/bounds tokens found in your dataset:
#   "n_val_lower_margin",
#   "n_val_upper_margin",
#   "Abundance_lower_margin",
#   "Abundance_upper_margin",
#   "Abundance_lower_margin_DR",
#   "Abundance_upper_margin_DR",
#   "Abundance 95pct_confidence_A",  # half-width for A (asymptote)
#   "Abundance 95pct_confidence_K",  # half-width for k (rate)
#   "Abundance 95pct_confidence",    # generic half-width (fallback)
# }
# and SAMPLE_ID matches [A-Za-z0-9.-]+  (edit SAMPLE_ID_RE if needed)
#
# Pipeline:
#  • Wide → long (Alignment ID, metric, sample_id, value)
#  • Drop sentinel -4
#  • Let user provide ONE interpretation DICTIONARY (Python dict or JSON)
#  • Compute ci_lower/ci_upper per metric (from explicit bounds or half-widths)
#  • Save tidy CSV: [Alignment ID] [Ontology?] metric value ci_lower ci_upper [interpreted vars...]

from __future__ import annotations
import re, json, ast
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd

# ───────────────────────── Config ─────────────────────────
METRIC_TOKENS = [
    "DR_median_abundance",
    "n_value",
    "Abundance rate",
    "Abundance asymptote",
    "median_abundance",
    # ---- CI/bounds tokens present in your file ----
    "n_val_lower_margin",
    "n_val_upper_margin",
    "Abundance_lower_margin",
    "Abundance_upper_margin",
    "Abundance_lower_margin_DR",
    "Abundance_upper_margin_DR",
    "Abundance 95pct_confidence_A",   # half-width for A (asymptote)
    "Abundance 95pct_confidence_K",   # half-width for k (rate)
    "Abundance 95pct_confidence",     # generic half-width (fallback)
]
TOKENS_RE = "|".join(re.escape(t) for t in METRIC_TOKENS)

# SAMPLE_ID: allow letters, numbers, dash, dot (NO underscore)
SAMPLE_ID_RE = r"[A-Za-z0-9][A-Za-z0-9\-.]*"

# Treat these numeric values as missing and drop
SENTINEL_BAD_VALUES = {-4}

# ───────────────────────── Helpers ─────────────────────────
def normalize_headers_strict(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Detect per-sample metric columns ONLY if they are EXACTLY one of:
        <METRIC_TOKEN>_<SAMPLE_ID>
        <SAMPLE_ID>_<METRIC_TOKEN>
    Returns (renamed_df, detected_cols) where all detected headers are normalized to metric_sample.
    """
    rename_map, detected = {}, []
    p1 = re.compile(rf"^(?P<metric>{TOKENS_RE})_(?P<sample>{SAMPLE_ID_RE})$", re.IGNORECASE)
    p2 = re.compile(rf"^(?P<sample>{SAMPLE_ID_RE})_(?P<metric>{TOKENS_RE})$", re.IGNORECASE)

    for col in df.columns:
        m = p1.match(col) or p2.match(col)
        if not m:
            continue
        metric = m.group("metric")
        sample = m.group("sample")
        norm = f"{metric}_{sample}"
        detected.append(col)
        if col != norm:
            rename_map[col] = norm

    if not detected:
        return df, []

    df2 = df.rename(columns=rename_map)
    return df2, [rename_map.get(c, c) for c in detected]


def parse_interpretation(raw: str|None) -> dict:
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


def coerce_dtype(series: pd.Series, dtype: str|None, order: list|None):
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
    """
    Build interpreted variable from sample_id using mapping or regex extraction.

    spec formats:
      - mapping dict (tokens->value), OR
      - dict with optional keys: map, regex, group, type, order, missing
      • If both regex & map: regex extract first → then map token
    """
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


def tidy_with_ci_bounds(long: pd.DataFrame, interp: dict) -> pd.DataFrame:
    """
    From 'long' (Alignment ID, Ontology?, metric, sample_id, value), compute
    per-row ci_lower / ci_upper for each point-estimate metric and return tidy.
    """
    # Keep 1 Ontology per Alignment ID if present
    ont = None
    if "Ontology" in long.columns:
        ont = (long[["Alignment ID", "Ontology"]]
               .dropna(subset=["Alignment ID"])
               .drop_duplicates(subset=["Alignment ID"], keep="first"))

    # Wide table with all metrics per (Alignment ID, sample_id)
    wide = (long.pivot_table(index=["Alignment ID", "sample_id"],
                             columns="metric", values="value", aggfunc="first")
                 .reset_index())

    # Attach Ontology back if present
    if ont is not None:
        wide = wide.merge(ont, on="Alignment ID", how="left")

    # Build interpreted variables on this wide frame
    if interp:
        for var, spec in interp.items():
            try:
                wide[var] = build_variable(wide, var, spec)
            except Exception as e:
                messagebox.showwarning("Variable skipped", f"Could not build variable '{var}': {e}")

    # CI mapping: explicit lower/upper tokens or half-width fallbacks
    CI_MAP = {
        "n_value": ("n_val_lower_margin", "n_val_upper_margin", None),
        "median_abundance": ("Abundance_lower_margin", "Abundance_upper_margin", "Abundance 95pct_confidence"),
        "DR_median_abundance": ("Abundance_lower_margin_DR", "Abundance_upper_margin_DR", None),
        "Abundance asymptote": (None, None, "Abundance 95pct_confidence_A"),
        "Abundance rate": (None, None, "Abundance 95pct_confidence_K"),
    }
    BASES = list(CI_MAP.keys())

    pieces = []
    for base in BASES:
        if base not in wide.columns:
            continue

        lower_tok, upper_tok, half_tok = CI_MAP[base]
        val = pd.to_numeric(wide[base], errors="coerce")

        lo = up = None
        if lower_tok and (lower_tok in wide.columns):
            lo = pd.to_numeric(wide[lower_tok], errors="coerce")
        if upper_tok and (upper_tok in wide.columns):
            up = pd.to_numeric(wide[upper_tok], errors="coerce")

        # If margins missing and a half-width exists, compute bounds
        if (lo is None or up is None) and half_tok and (half_tok in wide.columns):
            half = pd.to_numeric(wide[half_tok], errors="coerce")
            lo = val - half
            up = val + half

        out = wide.copy()
        out["metric"] = base
        out["value"] = val
        out["ci_lower"] = lo if lo is not None else pd.NA
        out["ci_upper"] = up if up is not None else pd.NA
        out = out[out["value"].notna()]

        # Column ordering: ID, Ontology?, metric/value/ci*, interpreted variables at end
        keep_core = ["Alignment ID", "metric", "value", "ci_lower", "ci_upper"]
        if "Ontology" in wide.columns:
            keep_core.insert(1, "Ontology")

        # Interpreted variables = everything that is NOT a metric column or sample_id/etc.
        drop_known = set(BASES + [
            "Alignment ID", "sample_id", "Ontology",
            "n_val_lower_margin", "n_val_upper_margin",
            "Abundance_lower_margin", "Abundance_upper_margin",
            "Abundance_lower_margin_DR", "Abundance_upper_margin_DR",
            "Abundance 95pct_confidence",
            "Abundance 95pct_confidence_A",
            "Abundance 95pct_confidence_K",
        ])
        extras = [c for c in wide.columns if c not in drop_known]

        pieces.append(out[keep_core + [c for c in extras if c not in keep_core]])

    if not pieces:
        # Fallback: return the original long (no base metrics found)
        base_cols = ["Alignment ID"] + (["Ontology"] if "Ontology" in long.columns else []) + ["metric", "value"]
        extras = [c for c in long.columns if c not in base_cols + ["metric_sample", "sample_id"]]
        fallback = long[base_cols + extras].copy()
        fallback["ci_lower"] = pd.NA
        fallback["ci_upper"] = pd.NA
        return fallback

    tidy = pd.concat(pieces, ignore_index=True)

    # Final ordering: drop sample_id (kept implicit after interpretation)
    col_order = ["Alignment ID"]
    if "Ontology" in tidy.columns:
        col_order += ["Ontology"]
    col_order += ["metric", "value", "ci_lower", "ci_upper"]
    extras = [c for c in tidy.columns if c not in col_order and c != "sample_id"]
    tidy = tidy[col_order + extras]

    return tidy

# ───────────────────────── Main ─────────────────────────
def main() -> None:
    root = tk.Tk(); root.withdraw()

    in_path = filedialog.askopenfilename(
        title="Select the wide lipidomics CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not in_path:
        messagebox.showinfo("No file selected", "Operation cancelled."); return

    df = pd.read_csv(in_path)
    if "Alignment ID" not in df.columns:
        messagebox.showerror("Missing column", "'Alignment ID' column not found."); return

    # Detect ONLY exact metric_sample / sample_metric matches (strict)
    df, detected_cols = normalize_headers_strict(df)

    extra_id_cols = ["Ontology"] if "Ontology" in df.columns else []

    # Optional: sanity-check Ontology is unique per Alignment ID
    if "Ontology" in df.columns:
        nuniq = df.groupby("Alignment ID", dropna=False)["Ontology"].nunique()
        bad = nuniq[nuniq > 1]
        if not bad.empty:
            messagebox.showwarning(
                "Ontology mismatch",
                f"'Ontology' differs within the same Alignment ID for {len(bad)} IDs.\n"
                "Keeping the first occurrence per Alignment ID."
            )

    cols_keep = ["Alignment ID"] + extra_id_cols + detected_cols
    df = df[cols_keep].copy().drop_duplicates(subset=["Alignment ID"], keep="first")
    df.columns = df.columns.map(str)

    # Wide → long
    id_vars = ["Alignment ID"] + extra_id_cols
    long = df.melt(id_vars=id_vars, var_name="metric_sample", value_name="value")

    # Token-anchored extraction with STRICT sample_id (matches exactly)
    m1 = long["metric_sample"].str.extract(
        rf"^(?P<metric>{TOKENS_RE})_(?P<sample_id>{SAMPLE_ID_RE})$", expand=True
    )
    m2 = long["metric_sample"].str.extract(
        rf"^(?P<sample_id2>{SAMPLE_ID_RE})_(?P<metric2>{TOKENS_RE})$", expand=True
    )
    long["metric"]    = m1["metric"].where(m1["metric"].notna(), m2["metric2"])
    long["sample_id"] = m1["sample_id"].where(m1["sample_id"].notna(), m2["sample_id2"])
    long = long.dropna(subset=["metric", "sample_id"])

    # Clean values, drop sentinels
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if SENTINEL_BAD_VALUES:
        long = long[~long["value"].isin(SENTINEL_BAD_VALUES)]
    long = long.dropna(subset=["value"])

    # Interpretation dictionary (Python dict or JSON)
    example = (
        '{\n'
        '  "genotype": {"map":{"A2":"APOE2","A3":"APOE3","A4":"APOE4"}, "type":"category", "order":["APOE3","APOE2","APOE4"]}\n'
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

    # Build tidy with lower/upper CI bounds as new columns
    tidy = tidy_with_ci_bounds(long, interp)

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
            f"Tidy data written to:\n{out_path}\n\n"
            f"Columns: {', '.join(tidy.columns)}"
        )
    else:
        messagebox.showinfo("Cancelled", "No output saved; the tidy dataframe remains in memory as 'tidy'.")

if __name__ == "__main__":
    main()
