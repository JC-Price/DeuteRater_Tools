# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:42:51 2026

@author: Brigham Young Univ
"""
#!/usr/bin/env python3
# lipid_deconvolution.py
# Lipid component deconvolution with:
#  - parsing-based eligibility filter (_components_from_row)
#  - QCM-based filtering (same logic as peptide version)
#  - jackknife + Monte Carlo ridge solver (allow negative contributions)
#  - Passing–Bablok + Bland–Altman using empirical component medians
#
# INPUT COLUMNS REQUIRED:
#   Alignment ID, Ontology, metric, value, se, genotype
#
# Notes:
# - Uses Sequence column to store Alignment ID after rename
# - Lipid identity key is (Ontology, Sequence, group_name)

# Standard library imports
import os
import sys
import re
import math
from collections import Counter
from datetime import datetime

# GUI imports
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.stats import truncnorm
from sklearn.metrics import r2_score

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Random generator
from numpy.random import default_rng

# Custom modules
from parsing import _components_from_row


# Create one hidden root
root = tk.Tk()
root.withdraw()  # hide main root window

# Pass root as parent to your Toplevel GUI
def get_user_settings(parent):
    gui = tk.Toplevel(parent)
    gui.grab_set()
    gui.title("Lipid Deconvolution QC Settings")

    # Abort handler
    def abort_program():
        gui.destroy()
        sys.exit("GUI closed by user — aborting.")

    gui.protocol("WM_DELETE_WINDOW", abort_program)

    # ----------------------------
    # Storage for final values
    # ----------------------------
    final_settings = {}

    # ----------------------------
    # Default numeric values
    # ----------------------------
    defaults = {
        "k_low": 0.005,
        "k_high": 3,
        "A_low": 0.005,
        "A_high": 3,
        "n_low": 0.005,
        "n_high": 3,
        "rare_pct": 3,
        "mc_iters": 4000
    }

    vars_dict = {k: tk.StringVar(value=str(v)) for k, v in defaults.items()}

    show_common_var = tk.BooleanVar(value=True)
    vars_dict["show_common_components"] = show_common_var

    # ----------------------------
    # Layout
    # ----------------------------
    frm = ttk.Frame(gui, padding=16)
    frm.grid()

    ttk.Label(
        frm,
        text="Enter Standard Error Ranges for Binomial Fit Metrics",
        font=("Segoe UI", 11, "bold")
    ).grid(column=0, row=0, columnspan=4, pady=(0, 15))

    ttk.Label(frm, text="Metric").grid(column=0, row=1)
    ttk.Label(frm, text="Low").grid(column=1, row=1)
    ttk.Label(frm, text="High").grid(column=2, row=1)

    rows = [
        ("k SE", "k_low", "k_high"),
        ("Asyn SE", "A_low", "A_high"),
        ("n SE", "n_low", "n_high"),
    ]

    current_row = 2
    for label, low_key, high_key in rows:
        ttk.Label(frm, text=label + ":").grid(
            column=0, row=current_row, sticky="e", padx=5, pady=4
        )
        ttk.Entry(frm, textvariable=vars_dict[low_key], width=10).grid(
            column=1, row=current_row
        )
        ttk.Entry(frm, textvariable=vars_dict[high_key], width=10).grid(
            column=2, row=current_row
        )
        current_row += 1

    # ----------------------------
    # Rare component cutoff
    # ----------------------------
    ttk.Label(
        frm,
        text="Component min % (nL deconvolution):"
    ).grid(column=0, row=current_row, sticky="e", pady=10)

    ttk.Entry(frm, textvariable=vars_dict["rare_pct"], width=10).grid(
        column=1, row=current_row
    )

    current_row += 1

    # ----------------------------
    # Monte Carlo iterations
    # ----------------------------
    ttk.Label(
        frm,
        text="Monte Carlo iterations (per genotype):"
    ).grid(column=0, row=current_row, sticky="e", pady=10)

    ttk.Entry(
        frm,
        textvariable=vars_dict["mc_iters"],
        width=10
    ).grid(column=1, row=current_row)

    current_row += 1

    # ----------------------------
    # Display option
    # ----------------------------
    ttk.Checkbutton(
        frm,
        text="Only show components present in ALL biological groups",
        variable=show_common_var
    ).grid(
        column=0, row=current_row, columnspan=3, sticky="w", pady=(10, 10)
    )

    current_row += 1

    # ----------------------------
    # Literature input
    # ----------------------------
    ttk.Label(
        frm,
        text="Literature component reference values\n"
             "(one per line: COMPONENT = VALUE)",
        font=("Segoe UI", 10, "bold")
    ).grid(
        column=0, row=current_row, columnspan=3, sticky="w", pady=(10, 4)
    )

    current_row += 1

    lit_text = tk.Text(frm, width=40, height=6)
    lit_text.grid(column=0, row=current_row, columnspan=3, sticky="w")

    lit_text.insert(
        "1.0",
        "16:0 = 17\n"
        "18:0 = 20\n"
        "20:4 = 6\n"
    )

    current_row += 1

    # ----------------------------
    # Submit handler
    # ----------------------------
    def submit():

        # Parse numeric + checkbox values
        for k, v in vars_dict.items():
            if isinstance(v, tk.BooleanVar):
                final_settings[k] = bool(v.get())
            elif k == "mc_iters":
                final_settings[k] = int(float(v.get()))
            else:
                final_settings[k] = float(v.get())

        # Parse literature entries
        literature_values = {}
        raw_lines = lit_text.get("1.0", "end-1c").splitlines()

        for line in raw_lines:
            if not line.strip():
                continue
            if "=" not in line:
                print(f"⚠️ Skipping malformed literature line: '{line}'")
                continue

            key, val = line.split("=", 1)
            key = key.strip()

            try:
                literature_values[key] = float(val.strip())
            except ValueError:
                print(f"⚠️ Skipping non-numeric literature value: '{line}'")

        final_settings["literature_values"] = literature_values

        gui.quit()
        gui.destroy()

    ttk.Button(frm, text="OK", command=submit).grid(
        column=0, row=current_row, columnspan=3, pady=15
    )

    gui.mainloop()
    return final_settings



# Retrieve user inputs
settings = get_user_settings(root)

# File dialog uses the same root
file_path = filedialog.askopenfilename(
    parent=root,  # ensures it belongs to the hidden root
    title="Select tidy CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

# Apply values
k_low   = settings["k_low"]
k_high  = settings["k_high"]
A_low   = settings["A_low"]
A_high  = settings["A_high"]
n_low   = settings["n_low"]
n_high  = settings["n_high"]
rare_pct = settings["rare_pct"]
rare_fraction = rare_pct / 100.0
show_common_components = settings["show_common_components"]
LITERATURE_COMPONENT_VALUES = settings["literature_values"]
mc_iters = settings["mc_iters"]

z_k_cut    = (k_low,  k_high)
z_asyn_cut = (A_low,  A_high)
z_n_cut    = (n_low,  n_high)

print("\n=== User-defined QC cutoffs ===")
print("k SE range:     ", z_k_cut)
print("Asyn SE range:  ", z_asyn_cut)
print("n SE range:     ", z_n_cut)
print(f"Rare component threshold: {rare_pct}%  (fraction={rare_fraction})")



if not file_path:
    raise SystemExit("No file selected.")

df = pd.read_csv(file_path)

# ----------------------------
# Normalize column names
# ----------------------------
df = df.rename(columns={
    "Alignment ID": "Sequence",
    "genotype": "group_name"
})

expected = ["Sequence", "Ontology", "metric", "value", "se", "group_name"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise SystemExit(f"ERROR: Missing expected columns: {missing}")

# Ensure numeric columns
for col in ["value", "se"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop invalid SE rows
bad_se_mask = (~np.isfinite(df["se"])) | (df["se"] <= 0)
n_bad = int(bad_se_mask.sum())
if n_bad > 0:
    print(f"⚠️ Dropping {n_bad} rows with invalid SE (<=0 or NaN)")
    df = df.loc[~bad_se_mask].copy()

# Drop invalid IDs/values
df = df.dropna(subset=["Sequence", "value"]).copy()
df["Sequence"] = df["Sequence"].astype(str)
df["Ontology"] = df["Ontology"].astype(str)

GROUPS = sorted(df["group_name"].dropna().unique().tolist())
print("Detected genotypes:", GROUPS)

# ----------------------------
# SE-only uncertainty handling
# ----------------------------
if df["se"].isna().any() or (df["se"] <= 0).any():
    raise SystemExit("ERROR: SE must be present and > 0 for all rows")

df["sd_pref"] = df["se"].to_numpy(float)
df_all = df.copy()

print("\n✅ Data ready (SE-only):")
print(df_all.head(6).to_string(index=False))


# ----------------------------
# Sampling & solver helpers
# ----------------------------
def sample_from_sd(med, sd, lower_bound=-np.inf, upper_bound=np.inf, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return float(np.clip(med, lower_bound, upper_bound))
    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        a, b = (lower_bound - med) / sd, (upper_bound - med) / sd
        return float(truncnorm.rvs(a, b, loc=med, scale=sd, random_state=rng))
    return float(rng.normal(loc=med, scale=sd))


def solve_unconstrained_ridge(A, y, l2_reg=1e-6):
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    n = A.shape[1]
    A_aug = np.vstack([A, np.sqrt(l2_reg) * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    x, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    return x




def run_mc_allowneg(
    A,
    medians,
    ses,
    mc_iters=2000,
    seed=0,
    l2_reg=1e-8,
    show_progress=True
):
    """
    Pure Monte Carlo ridge solver

    Returns
    -------
    results : ndarray (mc_iters × n_components)
    diagnostics : dict
    """


    rng = default_rng(seed)

    A = np.asarray(A, float)
    medians = np.asarray(medians, float)
    ses = np.asarray(ses, float)

    if np.any(~np.isfinite(ses)) or np.any(ses <= 0):
        raise ValueError("All SEs must be finite and > 0")

    results = []
    diagnostics = {
        "mc_iters": mc_iters,
        "solver_fails": 0,
        "succeeded": 0
    }

    iterator = range(mc_iters)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="Monte Carlo")

    for _ in iterator:
        sampled = rng.normal(medians, ses)

        try:
            x = solve_unconstrained_ridge(A, sampled, l2_reg=l2_reg)
            if np.all(np.isfinite(x)):
                results.append(x)
                diagnostics["succeeded"] += 1
            else:
                diagnostics["solver_fails"] += 1
        except Exception:
            diagnostics["solver_fails"] += 1

    return np.asarray(results), diagnostics



# ----------------------------
# Barplot of inferred lipid component contributions (per genotype)
# ----------------------------


_FA_PATTERN = re.compile(r"^(\d+):(\d+)$")

def sort_components_fa_first(components):
    """
    Sort components such that:
      1) FA components (A:B) first, sorted by A then B
      2) All other components alphabetically after
    """
    fa = []
    non_fa = []

    for c in components:
        m = _FA_PATTERN.match(c)
        if m:
            A = int(m.group(1))
            B = int(m.group(2))
            fa.append((A, B, c))
        else:
            non_fa.append(c)

    fa_sorted = [c for _, _, c in sorted(fa, key=lambda x: (x[0], x[1]))]
    non_fa_sorted = sorted(non_fa)

    return fa_sorted + non_fa_sorted


    


def plot_component_barplot(
    group_summaries,
    iter_dir,
    tag,
    top_n=30,
    show_common=True,
    literature_values=None,
):
    """
    group_summaries: dict[group] -> DataFrame indexed by component
                     with columns: Median, Lower, Upper
    literature_values: dict[component] -> scalar (optional)
    """
   

    groups_list = list(group_summaries.keys())
    if not groups_list:
        return

    # --------------------------------------------------
    # Component universe (Monte Carlo ONLY)
    # --------------------------------------------------
    comp_sets = [set(group_summaries[g].index) for g in groups_list]

    if show_common:
        all_components = sorted(set.intersection(*comp_sets))
    else:
        all_components = sorted(set().union(*comp_sets))

    if not all_components:
        print("⚠️ No components found after applying group intersection rule.")
        return

    # --------------------------------------------------
    # Build Median / CI matrices
    # --------------------------------------------------
    n_comp = len(all_components)
    n_grp = len(groups_list)

    med_mat = np.full((n_comp, n_grp), np.nan)
    lo_mat  = np.full_like(med_mat, np.nan)
    hi_mat  = np.full_like(med_mat, np.nan)

    for j, g in enumerate(groups_list):
        s = group_summaries[g].reindex(all_components)
        med_mat[:, j] = s["Median"].to_numpy(float)
        lo_mat[:, j]  = s["Lower"].to_numpy(float)
        hi_mat[:, j]  = s["Upper"].to_numpy(float)

    # --------------------------------------------------
    # Select top-N by variability
    # --------------------------------------------------
    variability = np.nanstd(med_mat, axis=1)
    order = np.argsort(variability)[::-1]
    keep = order[:min(top_n, len(order))]

    # Components selected by variability
    comps_raw = [all_components[i] for i in keep]

    # FA-first ordering of selected components
    comps = sort_components_fa_first(comps_raw)

    # Reorder matrices to match FA order
    idx = [comps_raw.index(c) for c in comps]
    med_mat = med_mat[keep, :][idx, :]
    lo_mat  = lo_mat[keep, :][idx, :]
    hi_mat  = hi_mat[keep, :][idx, :]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(max(10, 0.35 * len(comps)), 5), dpi=150)
    plt.gcf().set_size_inches(max(10, 0.35 * len(comps)), 6, forward=True)

    x = np.arange(len(comps))

    # Total cluster width (kept consistent with your prior logic)
    bar_width = 0.8 / n_grp
    colors = plt.cm.tab10.colors

    # --------------------------------------------------
    # BEST PRACTICE: center grouped bars around x
    # --------------------------------------------------
    # Offsets are symmetric around 0 (so the cluster is centered at x)
    offsets = (np.arange(n_grp) - (n_grp - 1) / 2.0) * bar_width

    # --- Literature bars (optional) centered at x (behind the cluster) ---
    if literature_values is not None:
        lit_vals = np.array([literature_values.get(c, np.nan) for c in comps], dtype=float)
        ok_lit = np.isfinite(lit_vals)

        plt.bar(
            x[ok_lit],                 # centered at x
            lit_vals[ok_lit],
            width=0.8,                 # spans the whole cluster
            color="lightgray",
            edgecolor="black",
            linewidth=0.8,
            zorder=1,
            label="Literature",
            alpha=0.70,                # helps foreground stand out
        )

    # --- Group bars + error bars (centered around x) ---
    for j, g in enumerate(groups_list):
        med = med_mat[:, j]
        low = lo_mat[:, j]
        high = hi_mat[:, j]

        ok = np.isfinite(med) & np.isfinite(low) & np.isfinite(high)
        if not np.any(ok):
            continue

        err = np.vstack([
            med[ok] - low[ok],
            high[ok] - med[ok],
        ])

        xpos = x[ok] + offsets[j]

        plt.bar(
            xpos,
            med[ok],
            width=bar_width * 0.85,
            color=colors[j % len(colors)],
            edgecolor="black",
            zorder=2,
            label=str(g),
        )

        plt.errorbar(
            xpos,
            med[ok],
            yerr=err,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=6,
            capthick=1.2,
            zorder=5,
        )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    # xticks centered at x (no extra shift needed now)
    plt.xticks(
        x,
        comps,
        rotation=60,
        ha="right",
    )

    plt.ylabel(r"Estimated $n_{L}$ contribution")
    plt.title(f"Top {len(comps)} lipid component contributions\n{tag}")
    plt.grid(axis="y", alpha=0.3, zorder=0)
    plt.legend(fontsize=8)

    # --- y-axis headroom + more graduations + minor gridlines ---
    ax = plt.gca()

    # Determine overall max from upper CI and (optionally) literature bars
    y_max = np.nanmax(hi_mat) if np.any(np.isfinite(hi_mat)) else np.nan

    if literature_values is not None:
        lit_vals_for_ylim = np.array([literature_values.get(c, np.nan) for c in comps], dtype=float)
        if np.any(np.isfinite(lit_vals_for_ylim)):
            if np.isfinite(y_max):
                y_max = max(y_max, np.nanmax(lit_vals_for_ylim))
            else:
                y_max = np.nanmax(lit_vals_for_ylim)

    # Round up to a "nice" y-limit with headroom
    if np.isfinite(y_max) and y_max > 0:
        y_top = y_max
        m = 10 ** math.floor(math.log10(y_top))
        step = m / 2  # 0.5*10^k style rounding
        nice_top = math.ceil(y_top / step) * step
        ax.set_ylim(0, nice_top)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(axis="y", which="major", alpha=0.30, zorder=0)
    ax.grid(axis="y", which="minor", alpha=0.12, zorder=0)

    plt.tight_layout()

    outpath = os.path.join(iter_dir, f"bar_components_{tag}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✅ Saved barplot (FA-ordered): {outpath}")




def drop_rare_components(components, A, comp_names, threshold=0.01):
    """
    Remove components that appear in <1% of lipids AND remove rows of A 
    where those components appear.

    Parameters
    ----------
    components : list[list[str]]
        components[i] is the list of component names in lipid i.

    A : np.ndarray
        Design matrix, shape (n_lipids, n_components).

    comp_names : list[str]
        Names of components in the same order as columns of A.

    threshold : float
        Minimum fraction of lipids a component must appear in.

    Returns
    -------
    A_new : np.ndarray
        Filtered design matrix.

    components_new : list[list[str]]
        Filtered per-lipid component lists.

    comp_names_new : list[str]
        Filtered component name list.

    info : dict
        Metadata about which components were removed.
    """

    n_lipids = len(components)

    # Count how many lipids contain each component
    comp_counts = Counter()
    for comp_list in components:
        comp_counts.update(set(comp_list))   # use set to avoid double-counting

    # Determine rare components
    rare_components = {c for c, cnt in comp_counts.items()
                       if cnt / n_lipids < threshold}

    # Rows to drop: any lipid containing a rare component
    rows_to_keep = [
        i for i, comp_list in enumerate(components)
        if not any(c in rare_components for c in comp_list)
    ]

    # Columns to drop: the rare components themselves
    cols_to_keep = [
        j for j, c in enumerate(comp_names)
        if c not in rare_components
    ]

    # Apply filtering
    A_new = A[np.array(rows_to_keep)[:, None], cols_to_keep]
    components_new = [components[i] for i in rows_to_keep]
    comp_names_new = [comp_names[j] for j in cols_to_keep]

    info = {
        "removed_components": sorted(list(rare_components)),
        "n_removed_components": len(rare_components),
        "n_removed_rows": n_lipids - len(rows_to_keep),
        "n_original_rows": n_lipids,
        "n_original_components": len(comp_names),
        "n_new_rows": len(rows_to_keep),
        "n_new_components": len(comp_names_new)
    }

    return A_new, components_new, comp_names_new, info





# ----------------------------
# Output folder (FIXED ORDER)
# ----------------------------
base = os.path.splitext(os.path.basename(file_path))[0]
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_root = os.path.join(os.path.dirname(file_path), f"lipid_deconv_out_{base}_{timestamp}")
os.makedirs(out_root, exist_ok=True)
print(f"\nAll outputs will be saved in:\n{out_root}")


# ----------------------------
# Lipid parsing / eligibility filter
# ----------------------------
restrict_classes = ('PE', 'PC', 'PI', 'PS', 'PG', 'PA',
                    'DG', 'TG', 'LPE', 'LPC', 'LPA',
                    'LPG', 'LPI', 'LPS')
exclude_ether = True

reasons = Counter()
components_list = []

for _, r in df_all.iterrows():
    ont = str(r["Ontology"])
    aid = str(r["Sequence"])
    comps = _components_from_row(
        ontology=ont,
        alignment_id=aid,
        restrict_classes=restrict_classes,
        exclude_ether=exclude_ether
    )

    if comps is None:
        if exclude_ether and ont.startswith("Ether"):
            reasons["excluded_ether"] += 1
        elif restrict_classes and (ont not in restrict_classes and ont not in [f"Ether{x}" for x in restrict_classes]):
            reasons["ontology_not_allowed"] += 1
        else:
            reasons["failed_parse_or_missing_species"] += 1
        components_list.append(None)
    else:
        components_list.append(comps)

df_all["components"] = components_list



before = len(df_all)
df_all = df_all[df_all["components"].notna()].copy()
after = len(df_all)

print(f"\n✅ Lipid parsing filter: kept {after}/{before} rows")
print("Drop reasons:", dict(reasons))

pd.DataFrame.from_dict(reasons, orient="index", columns=["count"]).to_csv(
    os.path.join(out_root, "lipid_parsing_drop_reasons.csv")
)

# ----------------------------
# Define QC cutoff sweeps
# Optional: drop padding component token from modeling
DROP_PAD_TOKEN = True
PAD_TOKEN = "0:0"

results_summary = []

tag = f"k{z_k_cut[1]}_A{z_asyn_cut[1]}_n{z_n_cut[1]}"
iter_dir = os.path.join(out_root, tag)
os.makedirs(iter_dir, exist_ok=True)

print(f"\n=== QC filtering pass {tag} ===")

# ----------------------------
# QCM-based filtering
# ----------------------------
filtered_keys = []

# IMPORTANT: lipid key includes Ontology
for (ont, seq, grp), sub in df_all.groupby(["Ontology", "Sequence", "group_name"]):

    se_k_vals = sub.loc[sub["metric"].str.lower() == "k", "se"].dropna()
    se_A_vals = sub.loc[sub["metric"].str.lower() == "asyn", "se"].dropna()
    se_n_vals = sub.loc[sub["metric"].str.lower().isin(["n", "n_value"]), "se"].dropna()

    # Require n_value SE
    if se_n_vals.empty:
        continue

    ignore_Asyn = False
    if not se_A_vals.empty and (se_A_vals == 0).all():
        ignore_Asyn = True

    if ignore_Asyn:
        se_for_global = pd.concat([se_k_vals, se_n_vals])
    else:
        se_for_global = pd.concat([se_k_vals, se_n_vals, se_A_vals])


    qc_k = se_k_vals.max() if not se_k_vals.empty else np.nan
    qc_A = se_A_vals.max() if not se_A_vals.empty else np.nan
    qc_n = se_n_vals.max()

    k_ok = (not np.isfinite(qc_k)) or (qc_k <= z_k_cut[1] and qc_k >= z_k_cut[0])
    A_ok = True if ignore_Asyn else ((not np.isfinite(qc_A)) or (qc_A <= z_asyn_cut[1] and qc_A >= z_asyn_cut[0]))
    n_ok = (qc_n <= z_n_cut[1] and qc_n >= z_n_cut[0])

    if k_ok and A_ok and n_ok:
        filtered_keys.append((ont, seq, grp))

n_keys = len(filtered_keys)
print(f"→ {n_keys} lipid×group combos passed QC")

if n_keys == 0:
    open(os.path.join(iter_dir, f"EMPTY_{tag}.txt"), "w").write(
        "No lipid×group combos passed QC.\n"
    )
    results_summary.append({
        "z_k_cut": z_k_cut,
        "z_Asyn_cut": z_asyn_cut,
        "z_n_cut": z_n_cut,
        "n_keys": 0,
        "n_points": 0,
        "mean_r2": np.nan
    })
else:
    idx_pass = pd.MultiIndex.from_tuples(filtered_keys, names=["Ontology", "Sequence", "group_name"])
    df_iter = df_all.set_index(["Ontology", "Sequence", "group_name"]).loc[
        lambda x: x.index.isin(idx_pass)
    ].reset_index()

    # Save QC stats
    summary_qc = df_iter.groupby("metric")["se"].agg(["mean", "std", "min", "max"])
    open(os.path.join(iter_dir, f"QC_summary_{tag}.txt"), "w").write(str(summary_qc.round(3)))

    # ----------------------------
    # Extract n_value rows only
    # ----------------------------
    df_n = df_iter[df_iter["metric"].str.lower() == "n_value"].copy()
    df_n = df_n.rename(columns={"value": "n_value", "se": "n_val_se"})
    for col in ["n_value", "n_val_se"]:
        df_n[col] = pd.to_numeric(df_n[col], errors="coerce")

    n_points = len(df_n)
    print(f"n_value rows retained: {n_points}")

    # ----------------------------
    # filter by nL > 10
    # ----------------------------
    before_nl = len(df_n)
    df_n = df_n[df_n["n_value"] > 10].copy()
    after_nl = len(df_n)
    print(f"Filtering nL > 10 kept {after_nl}/{before_nl}")

    if after_nl == 0:
        open(os.path.join(iter_dir, f"EMPTY_nL_{tag}.txt"), "w").write(
            "No lipids with nL > 10.\n"
        )
        results_summary.append({
            "z_k_cut": z_k_cut,
            "z_Asyn_cut": z_asyn_cut,
            "z_n_cut": z_n_cut,
            "n_keys": n_keys,
            "n_points": before_nl,
            "mean_r2": np.nan
        })
    else:
        # ----------------------------
        # Component universe
        # ----------------------------
        ALL_COMPONENTS = sorted({c for comps in df_n["components"] for c in comps})
        if DROP_PAD_TOKEN and PAD_TOKEN in ALL_COMPONENTS:
            ALL_COMPONENTS = [c for c in ALL_COMPONENTS if c != PAD_TOKEN]

        print(f"Components: {len(ALL_COMPONENTS)} (showing up to 30)")
        print(", ".join(ALL_COMPONENTS[:30]) + (" ..." if len(ALL_COMPONENTS) > 30 else ""))

        # ----------------------------
        # Per-genotype MC estimation
        # ----------------------------
        group_summaries = {}
        diag_all = {}
        mean_r2s = []

        for grp in sorted(df_n["group_name"].unique()):
            print(f"  Running MC for genotype {grp}")

            sub = df_n[df_n["group_name"] == grp].copy()
            if sub.empty:
                continue

            n_obs = len(sub)
            A = np.zeros((n_obs, len(ALL_COMPONENTS)), int)

            # speed-up tip: precompute component->index
            comp_to_idx = {c:i for i,c in enumerate(ALL_COMPONENTS)}

            for i, comps in enumerate(sub["components"]):
                if DROP_PAD_TOKEN:
                    comps = [c for c in comps if c != PAD_TOKEN]
                for c, cnt in Counter(comps).items():
                    j = comp_to_idx.get(c)
                    if j is not None:
                        A[i, j] = cnt
                        
            

            # ----------------------------------------
            # Drop rare components (< threshold)
            # ----------------------------------------
            A, _, new_comp_names, rare_info = drop_rare_components(
                components=list(sub["components"]),
                A=A,
                comp_names=ALL_COMPONENTS,
                threshold=0.05
            )
            
            print(f"Removed {rare_info['n_removed_components']} rare components")
            print(f"Removed {rare_info['n_removed_rows']} lipid rows")
            
            # ----------------------------------------
            # Enforce matrix-based prevalence (AUTHORITATIVE)
            # ----------------------------------------
            col_sums = A.sum(axis=0)
            cols_to_keep = col_sums > 0
            
            A = A[:, cols_to_keep]
            new_comp_names = [c for c, keep in zip(new_comp_names, cols_to_keep) if keep]
            
            print(f"Dropped {(~cols_to_keep).sum()} zero-prevalence components")
            
            # ----------------------------------------
            # Solver-aligned row indices
            # ----------------------------------------
            rows_kept = np.where(A.sum(axis=1) >= 0)[0]  # all rows retained by A
            
            # Extract medians / SEs IN THE SAME ORDER AS A
            medians = sub.iloc[rows_kept]["n_value"].to_numpy(float)
            ses     = sub.iloc[rows_kept]["n_val_se"].to_numpy(float)
            
            # ----------------------------------------
            # Export EXACT solver matrix
            # ----------------------------------------
            decision_df = pd.DataFrame(A, columns=new_comp_names)
            
            decision_df.insert(0, "group_name", grp)
            decision_df.insert(1, "Ontology", sub.iloc[rows_kept]["Ontology"].values)
            decision_df.insert(2, "Sequence", sub.iloc[rows_kept]["Sequence"].values)
            decision_df.insert(3, "n_value", medians)
            decision_df.insert(4, "n_val_se", ses)
            
            decision_out = os.path.join(iter_dir, f"decision_matrix_{grp}_{tag}.csv")
            decision_df.to_csv(decision_out, index=False)
            
            print(f"✅ Exported decision matrix for {grp}")
            

        


            # Filter medians and ses to match the rows kept in A
            row_mask = [i for i, comp_list in enumerate(sub["components"])
                        if not any(c in rare_info["removed_components"] for c in comp_list)]
            
            medians = sub["n_value"].to_numpy(float)[row_mask]
            ses     = sub["n_val_se"].to_numpy(float)[row_mask]



            
            results_obs, diag = run_mc_allowneg(
                A,
                medians,
                ses,
                mc_iters=mc_iters,
                seed=0,
                l2_reg=1e-8
            )


            diag_all[grp] = diag


            if results_obs.size == 0:
                summary = pd.DataFrame({
                    "Component": new_comp_names,
                    "Median": np.nan,
                    "Lower": np.nan,
                    "Upper": np.nan
                }).set_index("Component")
            else:
                summary = pd.DataFrame({
                    "Component": new_comp_names,
                    "Median": np.median(results_obs, axis=0),
                    "Lower":  np.percentile(results_obs, 2.5, axis=0),
                    "Upper":  np.percentile(results_obs, 97.5, axis=0)
                }).set_index("Component")


            
            output_path = os.path.join(iter_dir, f"component_summary_{grp}_{tag}.csv")
            
            os.makedirs(iter_dir, exist_ok=True)
            summary.to_csv(output_path)

            group_summaries[grp] = summary

            beta = summary["Median"].to_numpy(float)
            pred = A.dot(beta)
            mask = np.isfinite(pred) & np.isfinite(medians)
            if mask.sum() >= 3:
                mean_r2s.append(r2_score(medians[mask], pred[mask]))

        if not group_summaries:
            open(os.path.join(iter_dir, f"EMPTY_SUMMARY_{tag}.txt"), "w").write("No genotype summaries.\n")
        else:
            summary_combined = pd.concat(group_summaries, axis=1)  # MultiIndex columns
            summary_combined.to_csv(os.path.join(iter_dir, f"combined_component_summary_{tag}.csv"))

            # make barplot
            
        plot_component_barplot(
            group_summaries,
            iter_dir,
            tag,
            top_n=30,
            show_common=show_common_components,
            literature_values=LITERATURE_COMPONENT_VALUES
        )


            

# ----------------------------
# Save final sweep table
# ----------------------------
results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(out_root, "qc_filter_grid_summary.csv"), index=False)

# ----------------------------
# Export user-defined settings
# ----------------------------
settings_outfile = os.path.join(out_root, "user_qc_settings.txt")

with open(settings_outfile, "w", encoding="utf-8") as f:
    f.write("Lipid Deconvolution QC Settings\n")
    f.write(f"Export timestamp: {datetime.now()}\n\n")
    for key, val in settings.items():
        if isinstance(val, dict):
            f.write(f"{key}:\n")
            for k2, v2 in val.items():
                f.write(f"  {k2} = {v2}\n")
        else:
            f.write(f"{key} = {val}\n")

print(f"\n✅ User settings exported to: {settings_outfile}")

print("\nSaved QC sweep summary:")
print(results_df.head(10).to_string(index=False))

