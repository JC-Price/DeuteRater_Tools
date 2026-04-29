# -*- coding: utf-8 -*-
"""
Two-panel plot that NORMALIZES first (per species × adduct), then averages adducts per species,
then aggregates to class per genotype, and colors bars by class.

Workflow:
1) Normalize per (Alignment ID, Ontology, adduct) across genotypes:
   value_norm = value / max(value over genotypes)  ;  se_norm = se / max(...)
2) Average across adducts within each species × genotype × class using inverse-variance weights.
3) Sum species → class per genotype; combine SEs in quadrature.
4) Plot two panels: (PE vs LPE) and (TG vs DG), classes colored (PE/TG dark; LPE/DG lighter).

Requires: pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# -----------------------------
# 1) Config
# -----------------------------
RAW_CSV = "tidy_lipid_regression_with_ci.csv"  # change path as needed

# Panels: left is PE/LPE, right is TG/DG
PANEL_SPECS = [
    ("PE vs LPE", ["PE", "LPE"]),
    ("TG vs DG",  ["TG", "DG"]),
]

# Preferred genotype order (others will be appended alphabetically)
PREFERRED_GENOTYPE_ORDER = ["APOE2", "APOE3", "APOE4"]

# Class colors (LPE/DG are lighter shades of their base hues)
BASE_PE = "#1f77b4"   # blue
BASE_TG = "#ff7f0e"   # orange
LIGHTEN_FACTOR = 0.35  # 0 = original, 1 = white

# Optional: after all aggregation, normalize each panel to [0,1] (or to 100%)
NORMALIZE_PER_PANEL = False
NORMALIZE_TO_PERCENT = False


# -----------------------------
# 2) Helpers
# -----------------------------
def lighten_color(hex_or_rgb, factor=0.35):
    """Lighten a color by blending it with white (factor in [0,1])."""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(hex_or_rgb))
    white = np.array([1.0, 1.0, 1.0])
    return tuple((1 - factor) * rgb + factor * white)

CLASS_COLORS = {
    "PE":  BASE_PE,
    "LPE": lighten_color(BASE_PE, LIGHTEN_FACTOR),
    "TG":  BASE_TG,
    "DG":  lighten_color(BASE_TG, LIGHTEN_FACTOR),
}

def order_genotypes(values: pd.Series) -> list:
    """Stable genotype order: preferred first, then others alphabetically."""
    present = pd.Index(values.astype(str).unique()).tolist()
    preferred = [g for g in PREFERRED_GENOTYPE_ORDER if g in present]
    others = sorted([g for g in present if g not in PREFERRED_GENOTYPE_ORDER])
    return preferred + others

def combine_se_quadrature(series) -> float:
    """Combine standard errors in quadrature, ignoring NaNs."""
    s = pd.to_numeric(series, errors="coerce")
    return float(np.sqrt(np.nansum((s ** 2).values)))

def weighted_mean_with_se(values: pd.Series, ses: pd.Series) -> tuple:
    """
    Inverse-variance weighted mean across items.
    Returns (mean, SE_of_weighted_mean).
    Falls back to simple mean & SE from sample variance if SEs are missing.
    """
    v = pd.to_numeric(values, errors="coerce")
    s = pd.to_numeric(ses, errors="coerce")

    mask = np.isfinite(s) & (s > 0)
    if mask.sum() >= 1:
        w = 1.0 / (s[mask] ** 2)
        wsum = w.sum()
        if wsum > 0:
            wmean = float((w * v[mask]).sum() / wsum)
            se_wmean = float(1.0 / np.sqrt(wsum))
            return wmean, se_wmean

    vv = v[np.isfinite(v)]
    n = vv.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        # single obs: use its value; if any finite se, use mean of those; else 0
        return float(vv.iloc[0]), float(s[np.isfinite(s)].mean() if np.isfinite(s).any() else 0.0)

    mean = float(vv.mean())
    se = float(vv.std(ddof=1) / np.sqrt(n))
    return mean, se


# -----------------------------
# 3) Load & filter
# -----------------------------
df = pd.read_csv(RAW_CSV).drop_duplicates().copy()
df = df[df["metric"] == "abundance_mean"].copy()

# Ensure numeric
df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
df["se"] = pd.to_numeric(df["se"], errors="coerce")

# Keep only the classes used in our panels
classes_needed = {cls for _, classes in PANEL_SPECS for cls in classes}
df = df[df["Ontology"].astype(str).isin(classes_needed)].copy()
if df.empty:
    raise SystemExit("No rows for the requested classes. Check PANEL_SPECS and input data.")

# We expect at least: Alignment ID, Ontology, genotype, value, se, and optionally adduct.
# Adduct can be missing now; normalization is per species×adduct if present, else treat as one “adduct”.
has_adduct = "adduct" in df.columns
if not has_adduct:
    # fabricate a single-adduct column so the pipeline still works
    df["adduct"] = "na_adduct"


# -----------------------------
# 4) NORMALIZE FIRST: per (Alignment ID, Ontology, adduct) across genotypes
# -----------------------------
norm_parts = []
group_keys = ["Alignment ID", "Ontology", "adduct"]
for (sp, ont, ad), g in df.groupby(group_keys, dropna=False):
    max_val = pd.to_numeric(g["value"], errors="coerce").max()
    scale = max_val if np.isfinite(max_val) and max_val > 0 else 1.0

    tmp = g.copy()
    tmp["value_norm"] = pd.to_numeric(tmp["value"], errors="coerce").fillna(0.0) / scale
    tmp["se_norm"] = pd.to_numeric(tmp["se"], errors="coerce").fillna(0.0) / scale
    norm_parts.append(tmp)

norm_df = pd.concat(norm_parts, ignore_index=True)

# -----------------------------
# 5) Average adducts per species (after normalization)
# -----------------------------
# Now average across "adduct" within each (Alignment ID, Ontology, genotype)
species_avg = (
    norm_df.groupby(["Alignment ID", "Ontology", "genotype"], dropna=False)
           .apply(lambda g: pd.Series(
               dict(
                   value_avg=weighted_mean_with_se(g["value_norm"], g["se_norm"])[0],
                   se_avg=weighted_mean_with_se(g["value_norm"], g["se_norm"])[1],
               )
           ))
           .reset_index()
)

# -----------------------------
# 6) Aggregate species → class per genotype (IVW across species; NO SUMMING)
# -----------------------------

def ivw_mean_and_se_for_species(group: pd.DataFrame) -> pd.Series:
    """
    Inverse-variance weighted mean across species for a given (Ontology, genotype).
    Inputs are the adduct-averaged, normalized species values:
      - group['value_avg']: species means (already normalized per species×adduct and IVW-averaged across adducts)
      - group['se_avg']:    their SEs
    Returns:
      value_sum   -> the IVW mean across species (keeps your downstream column names)
      se_combined -> SE of the IVW mean = 1/sqrt(sum(w))
      n_species   -> number of unique species contributing
    """
    v = pd.to_numeric(group["value_avg"], errors="coerce")
    s = pd.to_numeric(group["se_avg"], errors="coerce")

    # Keep finite, positive SEs for IVW
    mask = np.isfinite(v) & np.isfinite(s) & (s > 0)
    n_species = int(group["Alignment ID"].nunique())

    if mask.sum() >= 1:
        # Optional small floor to prevent dominance by near-zero SEs
        eps = 1e-12
        s_eff = np.maximum(s[mask].values, eps)
        w = 1.0 / (s_eff ** 2)
        wsum = float(w.sum())
        wmean = float(np.sum(w * v[mask].values) / wsum) if wsum > 0 else float(np.nan)
        se_wmean = float(1.0 / np.sqrt(wsum)) if wsum > 0 else float(np.nan)
        return pd.Series({
            "value_sum": wmean,      # preserves downstream column name
            "se_combined": se_wmean,
            "n_species": n_species,
        })

    # Fallbacks if SEs are missing/non-positive:
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return pd.Series({"value_sum": np.nan, "se_combined": np.nan, "n_species": n_species})
    if vv.size == 1:
        # Single species available; carry its value and SE (if any)
        single_se = float(s[np.isfinite(s)].mean()) if np.isfinite(s).any() else np.nan
        return pd.Series({"value_sum": float(vv.iloc[0]), "se_combined": single_se, "n_species": n_species})

    # If multiple values but no usable SEs, use simple mean and its SE from sample variance
    mean = float(vv.mean())
    se = float(vv.std(ddof=1) / np.sqrt(vv.size)) if vv.size > 1 else np.nan
    return pd.Series({"value_sum": mean, "se_combined": se, "n_species": n_species})


class_agg = (
    species_avg.groupby(["Ontology", "genotype"], dropna=False)
               .apply(ivw_mean_and_se_for_species)
               .reset_index()
)


# Genotype order
geno_order = order_genotypes(class_agg["genotype"])
class_agg["genotype"] = pd.Categorical(class_agg["genotype"], categories=geno_order, ordered=True)
class_agg = class_agg.sort_values(["Ontology", "genotype"]).reset_index(drop=True)

# -----------------------------
# 7) (Optional) Panel normalization after aggregation
# -----------------------------
plot_df = class_agg.copy()
if NORMALIZE_PER_PANEL:
    norm_blocks = []
    for title, classes in PANEL_SPECS:
        block = plot_df[plot_df["Ontology"].isin(classes)].copy()
        if block.empty:
            continue
        max_val = pd.to_numeric(block["value_sum"], errors="coerce").max()
        scale = max_val if np.isfinite(max_val) and max_val > 0 else 1.0
        block["value_plot"] = block["value_sum"] / scale
        block["se_plot"] = pd.to_numeric(block["se_combined"], errors="coerce").fillna(0.0) / scale
        block["panel"] = title
        norm_blocks.append(block)
    plot_df = pd.concat(norm_blocks, ignore_index=True)

    if NORMALIZE_TO_PERCENT:
        plot_df["value_plot"] *= 100.0
        plot_df["se_plot"] *= 100.0
        y_label = "Normalized abundance (panel max = 100%)"
    else:
        y_label = "Normalized abundance (panel max = 1.0)"
else:
    plot_df["value_plot"] = plot_df["value_sum"]
    plot_df["se_plot"] = plot_df["se_combined"]
    y_label = "Abundance (species‑normalized, adduct‑averaged)"

# -----------------------------
# 8) Plotting (two panels)
# -----------------------------
fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(14, 4.8), squeeze=False)

# Figure‑level legend handles (classes)
legend_handles = [
    Patch(facecolor=CLASS_COLORS["PE"],  edgecolor="black", label="PE"),
    Patch(facecolor=CLASS_COLORS["LPE"], edgecolor="black", label="LPE"),
    Patch(facecolor=CLASS_COLORS["TG"],  edgecolor="black", label="TG"),
    Patch(facecolor=CLASS_COLORS["DG"],  edgecolor="black", label="DG"),
]


for idx, (panel_title, classes) in enumerate(PANEL_SPECS):
    ax = axes[0][idx]
    sub = plot_df[plot_df["Ontology"].isin(classes)].copy()
    if sub.empty:
        ax.set_axis_off()
        continue

    x = np.arange(len(geno_order))
    n_classes = len(classes)
    width = 0.8 / max(1, n_classes)

    # --- Bars with errors ---
    for j, cls in enumerate(classes):
        ss = (
            sub[sub["Ontology"] == cls]
            .set_index("genotype")[["value_plot", "se_plot"]]
            .reindex(geno_order)
        )
        heights = ss["value_plot"].fillna(0.0).values
        errs = ss["se_plot"].replace({np.nan: 0.0}).values

        ax.bar(
            x + (j - (n_classes - 1) / 2) * width,
            heights,
            width,
            yerr=errs,
            capsize=4,
            color=CLASS_COLORS.get(cls, "#999999"),
            edgecolor="black",
            label=cls,
        )

    # --- Compute SD across genotypes of the difference between the two classes ---
    sd_text = ""
    if len(classes) == 2:
        cls_a, cls_b = classes
        # Pivot to genotype × class for value_plot
        pv = (
            sub.pivot_table(index="genotype", columns="Ontology", values="value_plot", aggfunc="first")
            .reindex(geno_order)
        )
        if cls_a in pv.columns and cls_b in pv.columns:
            diffs = (pd.to_numeric(pv[cls_a], errors="coerce")
                     - pd.to_numeric(pv[cls_b], errors="coerce")).astype(float)
            # Sample SD across genotypes, ignoring NaNs; only if at least 2 valid points
            valid = diffs[np.isfinite(diffs)]
            if valid.size >= 2:
                sd_val = float(np.nanstd(valid, ddof=1))
                sd_text = f" (SD = {sd_val:.3f})"
            elif valid.size == 1:
                sd_text = f" ({cls_a}-{cls_b} defined for 1 genotype; SD n/a)"
            else:
                sd_text = " (insufficient data for SD)"
        else:
            sd_text = " (insufficient data for SD)"

    ax.set_title(f"{panel_title}{sd_text}")
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(geno_order)
    ax.set_ylabel(y_label if idx == 0 else "")
    ax.grid(axis="y", alpha=0.3)

# Figure‑level legend (classes)
fig.legend(
    handles=legend_handles,
    title="Class",
    frameon=False,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.02),
)

fig.tight_layout()

# Save next to the CSV in a 'plots' subfolder
data_dir = Path(RAW_CSV).resolve().parent
out_dir = data_dir 
out_dir.mkdir(parents=True, exist_ok=True)

out_svg = out_dir / "class_panels_normFirst_avgAdducts.svg"
out_png = out_dir / "class_panels_normFirst_avgAdducts.png"
fig.savefig(out_svg, format="svg", bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved: {out_svg}")
print(f"Saved: {out_png}")
