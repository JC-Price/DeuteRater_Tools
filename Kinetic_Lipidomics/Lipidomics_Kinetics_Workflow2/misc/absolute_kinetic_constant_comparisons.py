import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ============================
# 1) Pick input CSV via Tkinter
# ============================
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select final_dataframe.csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
root.destroy()

if not file_path:
    raise SystemExit("No file selected.")

input_dir = os.path.dirname(file_path)

# ============================
# 2) Load & de-duplicate
# ============================
df = pd.read_csv(file_path)
if "Alignment ID" not in df.columns:
    raise RuntimeError("'Alignment ID' column not found.")
df = df.drop_duplicates(subset=["Alignment ID"]).copy()

# ============================
# 3) Clean Ontology column
# ============================
if "Ontology" not in df.columns:
    alt = [c for c in df.columns if c.strip().lower() == "ontology"]
    if alt:
        onto_col = alt[0]
    else:
        raise RuntimeError("No Ontology column found.")
else:
    onto_col = "Ontology"

df["_Ontology_"] = (
    df[onto_col]
      .astype(str).str.strip()
      .replace({"": np.nan, "nan": np.nan, "None": np.nan})
)

# ============================
# 4) High-order membership (skip Standards) — EXPLODE
# ============================
high_order_all = [
    "Ethers",
    "glycerophospholipids",
    "glycerolipids",
    "sphingolipids",
    "lysos",
    "Kennedy_lipids",
    "neutral_lipids",
    "ionic_lipids",
]
existing_high = [g for g in high_order_all if g in df.columns]

rows = []
for _, row in df.iterrows():
    aid = row["Alignment ID"]

    # High-order memberships (one record per True flag)
    for g in existing_high:
        try:
            if bool(row[g]):
                rows.append({"Alignment ID": aid, "GroupType": "HighOrder", "GroupName": g})
        except Exception:
            pass

    # Ontology membership (always add if present)
    if pd.notna(row["_Ontology_"]):
        rows.append({"Alignment ID": aid, "GroupType": "Ontology", "GroupName": str(row["_Ontology_"])})

mem = pd.DataFrame(rows)
if mem.empty:
    raise SystemExit("No High-order or Ontology memberships found to plot.")

def pretty_label(s: str) -> str:
    s = str(s or "").strip().replace("_", " ")
    return s[:1].upper() + s[1:] if s else s

mem["GroupPretty"] = mem["GroupName"].map(pretty_label)

# ============================
# 5) Helpers for suffix-aware QC & plotting
# ============================
def _norm(s: str) -> str:
    """Normalize col name for tolerant matching (spaces/underscores/case)."""
    return re.sub(r"[^a-z0-9_]+", "_", str(s).strip().lower())

norm_map = {_norm(c): c for c in df.columns}

def find_col_for_base_suffix(base: str, suffix: str):
    """
    Find a column equivalent to f'{base}_{suffix}', tolerating spaces/underscores/case.
    """
    nb, ns = _norm(base), _norm(suffix)

    # 1) Direct candidate "<base>_<suffix>"
    direct = f"{nb}_{ns}"
    if direct in norm_map:
        return norm_map[direct]

    # 2) Startswith base & endswith _suffix
    for nk, real in norm_map.items():
        if nk.startswith(nb) and nk.endswith(f"_{ns}"):
            return real

    # 3) Both tokens present, in order
    for nk, real in norm_map.items():
        i, j = nk.find(nb), nk.rfind(ns)
        if i != -1 and j != -1 and i < j:
            return real

    return None

# QC filters (applied per suffix, used in both figures)
FILTERS = {
    "Abundance rate":      {"min": 0.001, "max": 1},
    "Abundance R2":        {"min": 0.6},
    "Abundance asymptote": {"min": 0, "max": 1.2},
}

def build_long_for_metric(metric_base: str,
                          rx_pattern: re.Pattern,
                          min_n_any_genotype: int = 10):
    """
    Build long-format DF for a given metric base (e.g., 'Abundance rate' or 'Abundance asymptote').
    - Detects value columns with suffix (e.g., 'Abundance rate_A2'), applies per-suffix QC.
    - EXPLODES membership so each lipid contributes to High-order and Ontology.
    - Keep a group if ANY genotype has n >= min_n_any_genotype; others can be < 10.
    Returns: long_df, counts_by_geno (DataFrame), totals (Series), geno_order (list)
    """
    # 1) Detect value columns for this metric (capture suffix)
    value_cols, suffixes_unique = [], []
    for col in df.columns:
        m = rx_pattern.match(col)
        if not m:
            continue
        suf = (m.group(1) or "").strip()
        if not suf:
            continue
        value_cols.append(col)
        if suf not in suffixes_unique:
            suffixes_unique.append(suf)

    if not value_cols:
        suspects = [c for c in df.columns if metric_base.lower() in c.lower()]
        raise RuntimeError(
            f"No '{metric_base}_{{suffix}}' columns detected. "
            f"Found these candidates: {suspects}"
        )

    # 2) Build QC-filtered long-form records
    long_parts = []
    for val_col in value_cols:
        m = rx_pattern.match(val_col)
        if not m:
            print(f"[warn] Skipping malformed {metric_base} column: {val_col}")
            continue
        suf = (m.group(1) or "").strip()
        if not suf:
            print(f"[warn] Empty suffix in column: {val_col} — skipping.")
            continue

        # QC mask for this suffix
        mask = pd.Series(True, index=df.index, dtype=bool)
        missing_filters = []

        for base, rule in FILTERS.items():
            qc_col = find_col_for_base_suffix(base, suf)
            if not qc_col:
                missing_filters.append(f"{base}_{suf}")
                continue

            vals = pd.to_numeric(df[qc_col], errors="coerce")
            cond = vals.notna()
            if "min" in rule:
                cond &= vals >= rule["min"]
            if "max" in rule:
                cond &= vals <= rule["max"]
            mask &= cond

        if missing_filters:
            print(f"[info] {val_col}: skipping missing QC columns: {', '.join(missing_filters)}")

        pass_df = df.loc[mask, ["Alignment ID", val_col]].copy()
        if pass_df.empty:
            print(f"[warn] No data left for {val_col} after QC filters — skipping.")
            continue

        pass_df.rename(columns={val_col: "Delta"}, inplace=True)
        pass_df["Genotype"] = suf

        # Merge with exploded memberships (High-order + Ontology)
        merged = pass_df.merge(mem, on="Alignment ID", how="inner")
        merged = merged.dropna(subset=["Delta", "GroupPretty"])

        long_parts.append(merged[["GroupType", "GroupPretty", "Delta", "Genotype"]])

    if not long_parts:
        raise SystemExit(f"No data remains for '{metric_base}' after applying QC filters.")

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["Delta"] = pd.to_numeric(long_df["Delta"], errors="coerce")
    long_df = long_df.dropna(subset=["GroupPretty", "Delta"])

    # 3) Per-genotype and total counts
    geno_order = sorted(long_df["Genotype"].unique().tolist())
    counts_by_g = (
        long_df.groupby(["GroupPretty", "Genotype"])["Delta"].count()
               .unstack(fill_value=0)
               .reindex(columns=geno_order, fill_value=0)
    )
    totals = counts_by_g.sum(axis=1)

    # 4) Keep a group if ANY genotype has n >= min_n_any_genotype
    keep_groups_idx = counts_by_g.max(axis=1) >= min_n_any_genotype
    keep_groups = counts_by_g.index[keep_groups_idx].tolist()

    long_df = long_df[long_df["GroupPretty"].isin(keep_groups)].copy()
    counts_by_g = counts_by_g.loc[keep_groups]
    totals = totals.loc[keep_groups]

    if long_df.empty:
        raise SystemExit(
            f"All groups had every genotype with n < {min_n_any_genotype} for '{metric_base}' after QC; nothing to plot."
        )

    return long_df, counts_by_g, totals, geno_order

def plot_and_save(long_df: pd.DataFrame,
                  counts_by_g: pd.DataFrame,
                  totals: pd.Series,
                  geno_order: list,
                  metric_title: str,
                  svg_name: str,
                  png_name: str):
    """
    Order groups (High-order first, then Ontology), build labels with per-genotype counts + total, plot, save.
    """
    pretty_high = [pretty_label(g) for g in existing_high]
    present_groups = set(long_df["GroupPretty"].dropna().unique().tolist())

    hi_present = [g for g in pretty_high if g in present_groups]
    onto_present = sorted([g for g in present_groups if g not in hi_present])  # alphabetical
    order = hi_present + onto_present

    # Reindex counts to match order
    counts_by_g = counts_by_g.reindex(order, fill_value=0)
    totals = totals.reindex(order, fill_value=0)

    # Build label with per-genotype counts and total, e.g., TG (A2=12 | A3=7 | A4=0; n=19)
    def label_for(group):
        parts = [f"{g}={int(counts_by_g.loc[group, g])}" for g in geno_order]
        return f"{group} (" + " | ".join(parts) + f"; n={int(totals.loc[group])})"

    order_labels = [label_for(g) for g in order]
    label_map = dict(zip(order, order_labels))

    long_df = long_df.copy()
    long_df["GroupLabel"] = long_df["GroupPretty"].map(label_map)
    long_df["GroupLabel"] = pd.Categorical(long_df["GroupLabel"], categories=order_labels, ordered=True)

    n_groups = len(order_labels)
    fig_h = max(6.0, 0.40 * n_groups)

    plt.figure(figsize=(12, fig_h))
    palette = sns.color_palette("colorblind", n_colors=len(geno_order))

    ax = sns.boxplot(
        data=long_df,
        x="Delta",
        y="GroupLabel",   # includes per-genotype and total counts
        hue="Genotype",
        hue_order=geno_order,
        order=order_labels,
        orient="h",
        showfliers=False,
        palette=palette
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{metric_title} — High-order (top) + Ontology (below), QC-filtered (keep if any genotype n≥10)")
    ax.set_xlabel(f"{metric_title} (Δ)")
    ax.set_ylabel("Group — counts reflect post‑QC data (per genotype and total)")
    ax.legend(title="Genotype", frameon=False)

    plt.tight_layout()

    svg_path = os.path.join(input_dir, svg_name)
    png_path = os.path.join(input_dir, png_name)

    plt.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved SVG to:", svg_path)
    print("Saved PNG to:", png_path)

# ============================
# 6) Build + Plot: Abundance rate
# ============================
RX_RATE = re.compile(r"(?i)^\s*abundance\s*rate(?:\s+|_)+(.+)\s*$")   # capture suffix after space/underscore(s)
long_rate, counts_rate, totals_rate, geno_order_rate = build_long_for_metric(
    metric_base="Abundance rate",
    rx_pattern=RX_RATE,
    min_n_any_genotype=10
)
plot_and_save(
    long_rate, counts_rate, totals_rate, geno_order_rate,
    metric_title="Abundance rate",
    svg_name="combined_highorder_plus_ontology_RATE_QC_counts_anyGenN10.svg",
    png_name="combined_highorder_plus_ontology_RATE_QC_counts_anyGenN10.png"
)

# ============================
# 7) Build + Plot: Abundance asymptote
# ============================
RX_ASYM = re.compile(r"(?i)^\s*abundance\s*asymptote(?:\s+|_)+(.+)\s*$")  # fixed regex (no double-escaping)
long_asym, counts_asym, totals_asym, geno_order_asym = build_long_for_metric(
    metric_base="Abundance asymptote",
    rx_pattern=RX_ASYM,
    min_n_any_genotype=10
)
plot_and_save(
    long_asym, counts_asym, totals_asym, geno_order_asym,
    metric_title="Abundance asymptote",
    svg_name="combined_highorder_plus_ontology_ASYMPTOTE_QC_counts_anyGenN10.svg",
    png_name="combined_highorder_plus_ontology_ASYMPTOTE_QC_counts_anyGenN10.png"
)

print("Done.")