#!/usr/bin/env python3
# peptide_deconvolution.py (allow negative contributions option)
"""
Peptide deconvolution with optional negative AA contributions.

- If allow_negative=True: use ridge-regularized unconstrained solves (np.linalg.lstsq on augmented system).
- If allow_negative=False: use bounded non-negative solves via scipy.optimize.lsq_linear.
- Sampling: if sampling bounds are finite, use truncated normal; if unbounded, use normal.
"""
import os
import sys
from collections import Counter

import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import truncnorm
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score
# add this import near the top with your other imports
from sklearn.linear_model import LinearRegression






# ----------------------------
# File load (GUI)
# ----------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select tidy n-value CSV (e.g. rate_by_sequence_filtered_intersection_tidy.csv)",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not file_path:
    raise SystemExit("No file selected.")
df = pd.read_csv(file_path)

# ----------------------------
# Validate required columns
# ----------------------------
if "metric" not in df.columns or "genotype" not in df.columns:
    raise SystemExit("Expected columns 'metric' and 'genotype' not found.")

# Normalize column names we expect
df = df.rename(columns={
    "Alignment ID": "Sequence",
    "value": "value",
    "ci_lower": "ci_lower",
    "ci_upper": "ci_upper",
    "genotype": "group_name",
})

# If SE column exists under other names, try to detect them
if "se" not in df.columns:
    for candidate in ["SE nL", "SE_nL", "se_nl", "se", "SE"]:
        if candidate in df.columns:
            df["se"] = df[candidate].astype(float)
            break

# ----------------------------
# Drop invalids and validate sequences
# ----------------------------
df = df.dropna(subset=["Sequence", "value"]).copy()
df["Sequence"] = df["Sequence"].astype(str)
valid_aas = set("ARNDCEQGHILKMFPSTWYV" + "arndceqghilkmfpstwyv")
df = df[df["Sequence"].apply(lambda s: all(ch in valid_aas for ch in s))].copy()

# Auto-detect groups
GROUPS = sorted(df["group_name"].dropna().unique().tolist())
print(f"Detected genotypes: {GROUPS}")

# ----------------------------
# Helpers
# ----------------------------
def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return np.nan

def ci_to_sd(lo, med, hi, ci_level=0.95):
    """Convert CI bounds to symmetric sd approximation via (hi-lo)/(2*z)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    z = {0.68:1.0, 0.90:1.645, 0.95:1.96, 0.99:2.575}.get(ci_level, 1.96)
    return float(max(1e-8, (hi - lo) / (2.0 * z)))

# ----------------------------
# Compute CI width & frac_uncert (prefer CI if present, otherwise use SE -> 95% CI)
# ----------------------------
df["ci_lower"] = df.get("ci_lower", pd.Series(np.nan, index=df.index)).apply(safe_float)
df["ci_upper"] = df.get("ci_upper", pd.Series(np.nan, index=df.index)).apply(safe_float)
df["se"] = df.get("se", pd.Series(np.nan, index=df.index)).apply(safe_float)

Z95 = 1.96
ci_lower_fallback = df["value"] - Z95 * df["se"]
ci_upper_fallback = df["value"] + Z95 * df["se"]
df["ci_lower"] = np.where(df["ci_lower"].notna(), df["ci_lower"], ci_lower_fallback)
df["ci_upper"] = np.where(df["ci_upper"].notna(), df["ci_upper"], ci_upper_fallback)

df["ci_width"] = (df["ci_upper"] - df["ci_lower"]).replace([np.inf, -np.inf], np.nan)
df["frac_uncert"] = df["ci_width"].abs() / df["value"].abs().replace(0, np.nan)

# ----------------------------
# Robust MAD z (index-preserving)
# ----------------------------
def robust_z_series(s: pd.Series) -> pd.Series:
    """Return robust z-scores (0.6745*(x-med)/mad) for a pandas Series. Preserves index."""
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad





df["z_k"] = np.nan
df["z_Asyn"] = np.nan
df["z_n"] = np.nan

for grp, sub_grp in df.groupby("group_name"):
    mask_k = (sub_grp["metric"] == "k")
    if mask_k.any():
        z_k = robust_z_series(sub_grp.loc[mask_k, "frac_uncert"])
        df.loc[z_k.index, "z_k"] = z_k
    mask_as = (sub_grp["metric"] == "Asyn")
    if mask_as.any():
        z_as = robust_z_series(sub_grp.loc[mask_as, "frac_uncert"])
        df.loc[z_as.index, "z_Asyn"] = z_as
    mask_n = (sub_grp["metric"] == "n_value")
    if mask_n.any():
        z_n = robust_z_series(sub_grp.loc[mask_n, "frac_uncert"])
        df.loc[z_n.index, "z_n"] = z_n

# ----------------------------
# Filtering criteria (safe)
# ----------------------------
Z_CUTOFF = 1
filtered_pairs = []

for (seq, grp), sub in df.groupby(["Sequence", "group_name"]):
    asyn_sub = sub[sub["metric"] == "Asyn"]
    asyn_vals = asyn_sub["value"].dropna().unique()
    asyn_span = 0.0
    if not asyn_sub.empty:
        lo = asyn_sub["ci_lower"].min(skipna=True)
        hi = asyn_sub["ci_upper"].max(skipna=True)
        if pd.notna(lo) and pd.notna(hi):
            asyn_span = hi - lo

    if (len(asyn_vals) <= 1) or (asyn_span >= 1.8):
        asyn_good = True
    else:
        z_asyn_vals = sub.loc[sub["metric"] == "Asyn", "z_Asyn"].dropna().abs()
        asyn_good = (z_asyn_vals.max() < Z_CUTOFF) if (not z_asyn_vals.empty) else True

    z_k_vals = sub.loc[sub["metric"] == "k", "z_k"].dropna().abs()
    k_good = (z_k_vals.max() < Z_CUTOFF) if (not z_k_vals.empty) else True

    z_n_vals = sub.loc[sub["metric"] == "n_value", "z_n"].dropna().abs()
    n_good = (z_n_vals.max() < Z_CUTOFF) if (not z_n_vals.empty) else False

    if n_good and (k_good or asyn_good):
        filtered_pairs.append((seq, grp))

idx = pd.MultiIndex.from_tuples(filtered_pairs, names=["Sequence", "group_name"])
if len(filtered_pairs) == 0:
    raise SystemExit("No peptide×group combos passed QC — check thresholds or input.")
df = df.set_index(["Sequence", "group_name"]).loc[df.set_index(["Sequence", "group_name"]).index.isin(idx)].reset_index()

print(f"Passed MAD-based QC for {len(filtered_pairs)} peptide × genotype combinations ({len(df)} total metric rows retained).")
print("\nQC summary by metric:")
summary_qc = df.groupby("metric")[["ci_width", "frac_uncert"]].agg(["mean", "std", "min", "max"])
print(summary_qc.round(3))

# ----------------------------
# Keep only n_value rows
# ----------------------------
df_n = df[df["metric"].str.lower() == "n_value"].copy()
df_n = df_n.rename(columns={
    "value": "n_value",
    "ci_lower": "n_val_lower_margin",
    "ci_upper": "n_val_upper_margin",
    "se": "n_val_se"
})





df_n["n_value"] = df_n["n_value"].astype(float)
df_n["n_val_lower_margin"] = df_n["n_val_lower_margin"].astype(float)
df_n["n_val_upper_margin"] = df_n["n_val_upper_margin"].astype(float)
if "n_val_se" not in df_n.columns:
    df_n["n_val_se"] = np.nan
else:
    df_n["n_val_se"] = df_n["n_val_se"].astype(float)

# ----------------------------
# Sampling & solver helpers (support negatives)
# ----------------------------
def sample_from_sd(med, sd, lower_bound=-np.inf, upper_bound=np.inf, rng=None):
    """Sample from normal if bounds infinite, else truncated normal."""
    if rng is None:
        rng = np.random.default_rng()
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return float(np.clip(med, lower_bound, upper_bound))
    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        a, b = (lower_bound - med) / sd, (upper_bound - med) / sd
        return float(truncnorm.rvs(a, b, loc=med, scale=sd, random_state=rng))
    # unbounded on one or both sides -> normal
    return float(rng.normal(loc=med, scale=sd))

def solve_unconstrained_ridge(A, y, l2_reg=1e-6):
    """Solve (A^T A + lambda I) x = A^T y via lstsq on augmented system (ridge)."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n = A.shape[1]
    if l2_reg and l2_reg > 0:
        A_aug = np.vstack([A, np.sqrt(l2_reg) * np.eye(n)])
        y_aug = np.concatenate([y, np.zeros(n)])
    else:
        A_aug, y_aug = A, y
    # use lstsq (allows negatives)
    x, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    return x

def solve_bounded_nn(A, y, lb=0.0, ub=np.inf, l2_reg=1e-8):
    """Bounded solver using lsq_linear (for non-negative mode)."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n = A.shape[1]
    if l2_reg and l2_reg > 0:
        A_aug = np.vstack([A, np.sqrt(l2_reg) * np.eye(n)])
        y_aug = np.concatenate([y, np.zeros(n)])
    else:
        A_aug, y_aug = A, y
    bounds = (np.full(n, lb), np.full(n, ub if np.isfinite(ub) else 1e12))
    try:
        res = lsq_linear(A_aug, y_aug, bounds=bounds, lsmr_tol='auto', max_iter=2000)
        if res.success:
            return res.x, True
        x, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        return x, False
    except Exception:
        x, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        return x, False

# add this near your other imports at the top of the file:
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback if tqdm not installed

# Improved jackknife MC supporting negative contributions (with tqdm progress)
def run_jackknife_mc_allowneg(A, medians, lo_bounds, hi_bounds, ses,
                              num_jackknifes=100, drop_fraction=0.2,
                              mc_per_boot=200, seed=0,
                              allow_negative=False, lb=0.0, ub=50.0,
                              l2_reg=1e-8, skip_underdetermined=True,
                              show_progress=True):
    """
    A: (n_pep, n_aa)
    medians, lo_bounds, hi_bounds, ses: length n_pep
    allow_negative: if True -> unconstrained ridge solves; else bounded NNLS solves.
    lb/ub used for sampling bounds and bounded solver when allow_negative=False.

    show_progress: if True and tqdm is available, show nested progress bars.
    """
    rng = np.random.default_rng(seed)
    n_pep, n_aa = A.shape
    all_results = []
    diagnostics = {"attempts": 0, "succeeded": 0, "skipped_underd": 0, "solver_fails": 0}

    sd_from_ci = np.array([ci_to_sd(lo, m, hi) for lo, m, hi in zip(lo_bounds, medians, hi_bounds)], dtype=float)
    sd_array = np.where(np.isfinite(ses) & (ses > 0), ses, sd_from_ci)
    sd_array = np.where(np.isfinite(sd_array), sd_array, 1e-6)

    use_tqdm = (show_progress and (tqdm is not None))
    outer_iter = range(num_jackknifes)
    if use_tqdm:
        outer_iter = tqdm(outer_iter, desc="Jackknifes", unit="jk")

    for b in outer_iter:
        diagnostics["attempts"] += 1
        keep_idx = rng.choice(n_pep, size=max(1, int(n_pep * (1 - drop_fraction))), replace=False)
        A_sub = A[keep_idx, :]
        med_sub = medians[keep_idx]
        sd_sub = sd_array[keep_idx]

        # skip underdetermined draws
        if skip_underdetermined and (A_sub.shape[0] <= A_sub.shape[1]):
            diagnostics["skipped_underd"] += 1
            continue

        inner_iter = range(mc_per_boot)
        if use_tqdm:
            # show a short, ephemeral inner bar to avoid clutter (leave=False)
            inner_iter = tqdm(inner_iter, desc=f"MC jk#{b+1}", leave=False, unit="mc")

        for _ in inner_iter:
            if allow_negative:
                sampled = np.array([sample_from_sd(med_sub[i], sd_sub[i], lower_bound=-np.inf, upper_bound=np.inf, rng=rng)
                                    for i in range(len(med_sub))], dtype=float)
                # unconstrained ridge solve
                try:
                    x = solve_unconstrained_ridge(A_sub, sampled, l2_reg=l2_reg)
                    if np.all(np.isfinite(x)):
                        all_results.append(x)
                        diagnostics["succeeded"] += 1
                    else:
                        diagnostics["solver_fails"] += 1
                except Exception:
                    diagnostics["solver_fails"] += 1
            else:
                sampled = np.array([sample_from_sd(med_sub[i], sd_sub[i], lower_bound=lb, upper_bound=ub, rng=rng)
                                    for i in range(len(med_sub))], dtype=float)
                x, ok = solve_bounded_nn(A_sub, sampled, lb=lb, ub=ub, l2_reg=l2_reg)
                if ok and np.all(np.isfinite(x)):
                    all_results.append(x)
                    diagnostics["succeeded"] += 1
                else:
                    diagnostics["solver_fails"] += 1

    results = np.asarray(all_results, dtype=float) if len(all_results) > 0 else np.empty((0, n_aa))
    return results, diagnostics


# ----------------------------
# Build global amino acid order
# ----------------------------
ALL_AAS = sorted(set("".join(df_n["Sequence"].str.upper())))
if len(ALL_AAS) == 0:
    raise SystemExit("No amino acids found after filtering.")
print("Amino acids (global order):", "".join(ALL_AAS))

# ----------------------------
# Run per-genotype MC estimation (allow negatives by default)
# ----------------------------
group_summaries = {}
ALLOW_NEGATIVE = True   # <--- set True to allow negative AA contributions; False enforces non-negative bounds

for grp in GROUPS:
    print(f"\n=== Running jackknife for genotype: {grp} ===")
    sub = df_n[df_n["group_name"] == grp].copy()
    if sub.empty:
        print(f"  no n_value rows for {grp}, skipping")
        continue

    n_pep = len(sub)
    A = np.zeros((n_pep, len(ALL_AAS)), dtype=int)
    for i, seq in enumerate(sub["Sequence"].str.upper()):
        counts = Counter(seq)
        for aa, cnt in counts.items():
            if aa in ALL_AAS:
                A[i, ALL_AAS.index(aa)] = cnt

    medians = sub["n_value"].to_numpy(dtype=float)
    lo_bounds = sub["n_val_lower_margin"].to_numpy(dtype=float)
    hi_bounds = sub["n_val_upper_margin"].to_numpy(dtype=float)
    ses = sub.get("n_val_se", pd.Series(np.nan, index=sub.index)).to_numpy(dtype=float)

    results_obs, diag = run_jackknife_mc_allowneg(
        A, medians, lo_bounds, hi_bounds, ses,
        num_jackknifes=20, mc_per_boot=200, seed=0,
        allow_negative=ALLOW_NEGATIVE, lb=0.0, ub=50.0, l2_reg=1e-8, skip_underdetermined=True
    )
    print("  diagnostics:", diag)
    if results_obs.size == 0:
        print(f"  No successful MC results for {grp} — filling NaNs.")
        summary = pd.DataFrame({
            "AminoAcid": ALL_AAS,
            "Median": [np.nan] * len(ALL_AAS),
            "Lower": [np.nan] * len(ALL_AAS),
            "Upper": [np.nan] * len(ALL_AAS)
        }).set_index("AminoAcid")
    else:
        summary = pd.DataFrame({
            "AminoAcid": ALL_AAS,
            "Median": np.median(results_obs, axis=0),
            "Lower": np.percentile(results_obs, 2.5, axis=0),
            "Upper": np.percentile(results_obs, 97.5, axis=0)
        }).set_index("AminoAcid")

    group_summaries[grp] = summary

# ----------------------------
# Combine summaries and add theoretical
# ----------------------------
if len(group_summaries) == 0:
    raise SystemExit("No genotype summaries produced.")
summary_combined = pd.concat(group_summaries, axis=1)
print("\nCombined summary:")
print(summary_combined.round(3))

theoretical_nL = {
    "A": 4.00, "C": 1.62, "D": 1.89, "E": 3.95, "F": 0.32,
    "G": 2.06, "H": 2.88, "I": 1.00, "K": 0.54, "L": 0.69,
    "M": 1.12, "N": 1.89, "P": 2.59, "Q": 3.95, "R": 3.34,
    "S": 2.61, "T": 0.20, "V": 0.56, "W": 0.08, "Y": 0.42
}
summary_combined[("Theoretical", "Median")] = [theoretical_nL.get(aa, np.nan) for aa in summary_combined.index]
summary_combined[("Theoretical", "Lower")] = np.nan
summary_combined[("Theoretical", "Upper")] = np.nan


def passing_bablok(x, y):
    """Nonparametric Passing–Bablok regression returning slope, intercept, and 95% CIs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    slopes = np.sort(slopes)
    m = np.median(slopes)
    intercepts = y - m * x
    b = np.median(intercepts)

    # rank-based CI
    M = len(slopes)
    z = 1.96  # 95%
    k = int(np.floor((M - z * np.sqrt(M)) / 2))
    l = int(np.ceil((M + z * np.sqrt(M)) / 2)) - 1
    slope_ci = (slopes[max(0, k)], slopes[min(l, M - 1)])

    intercepts_low = y - slope_ci[0] * x
    intercepts_high = y - slope_ci[1] * x
    intercept_ci = (np.median(intercepts_low), np.median(intercepts_high))
    return m, b, slope_ci, intercept_ci


# Replace this entire per-group plotting section with the following:
for grp in group_summaries.keys():
    sub = df_n[df_n["group_name"] == grp].copy()
    if sub.empty:
        continue

    n_pep_sub = len(sub)
    A_sub = np.zeros((n_pep_sub, len(ALL_AAS)), dtype=int)
    for i, seq in enumerate(sub["Sequence"].str.upper()):
        cnts = Counter(seq)
        for aa, cnt in cnts.items():
            if aa in ALL_AAS:
                A_sub[i, ALL_AAS.index(aa)] = cnt

    aa_med_vec = np.array([theoretical_nL.get(aa, np.nan) for aa in ALL_AAS], dtype=float)
    predicted_arr = A_sub.dot(aa_med_vec)
    sub = sub.reset_index(drop=True).assign(predicted_nL=predicted_arr)
    sub = sub.dropna(subset=["n_value", "predicted_nL"])

    x = sub["predicted_nL"].to_numpy(float)
    y = sub["n_value"].to_numpy(float)
    n_pts = len(x)
    if n_pts < 3:
        print(f"[WARN] Not enough points for Passing–Bablok ({grp})")
        continue

    # --- Passing–Bablok regression ---
    slope, intercept, slope_ci, intercept_ci = passing_bablok(x, y)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)

    # --- Passing–Bablok plot ---
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept
    y_low = slope_ci[0] * x_fit + intercept_ci[0]
    y_high = slope_ci[1] * x_fit + intercept_ci[1]

    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(x, y, s=30, alpha=0.8, label="Data", facecolors="none", edgecolor="blue")
    plt.plot(x_fit, y_fit, "r-", label=f"Passing–Bablok fit: y={slope:.3f}x+{intercept:.3f}")
    plt.fill_between(x_fit, y_low, y_high, color="lightgray", alpha=0.5, label="95% CI")
    plt.plot(x_fit, x_fit, "--", color="gray", lw=1, label="Identity (y=x)")
    plt.xlabel("Predicted nₗ (sum of amino acids)")
    plt.ylabel("Measured nₗ (empirical)")
    plt.title(f"{grp} — Passing–Bablok Regression\n"
              f"Slope={slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}] | "
              f"Intercept={intercept:.3f} | R²={r2:.3f}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Bland–Altman residuals ---
    mean_vals = (x + y) / 2
    diff_vals = y - x
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals)
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff
    n_out = np.sum((diff_vals > upper) | (diff_vals < lower))
    pct_out = 100 * n_out / len(diff_vals)

    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(mean_vals, diff_vals, color="purple", facecolors="none", label="Data")
    plt.axhline(mean_diff, color="red", ls="--", label=f"Mean diff={mean_diff:.3f}")
    plt.axhline(upper, color="gray", ls="--", label=f"+1.96 SD={upper:.3f}")
    plt.axhline(lower, color="gray", ls="--", label=f"-1.96 SD={lower:.3f}")
    plt.xlabel("Mean of predicted and measured nₗ")
    plt.ylabel("Difference (measured – predicted)")
    plt.title(f"{grp} — Bland–Altman Analysis\n"
              f"Mean diff={mean_diff:.3f}, SD={sd_diff:.3f}, "
              f"Outside limits={pct_out:.1f}%")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n{grp} Passing–Bablok results:")
    print(f"  slope = {slope:.4f} (95% CI {slope_ci[0]:.4f}–{slope_ci[1]:.4f})")
    print(f"  intercept = {intercept:.4f} (95% CI {intercept_ci[0]:.4f}–{intercept_ci[1]:.4f})")
    print(f"  R² = {r2:.4f}")
    print(f"  Bland–Altman mean diff = {mean_diff:.4f}, SD = {sd_diff:.4f}, "
          f"{n_out}/{len(diff_vals)} points ({pct_out:.2f}%) outside limits.")



# ----------------------------
# Barplot of empirical n-values per amino acid (with essential AAs in red)
# ----------------------------
import matplotlib as mpl

plt.figure(figsize=(12, 5), dpi=150)

# Essential amino acids (human)
ESSENTIAL_AAS = {"H", "I", "L", "K", "M", "F", "T", "W", "V", "Y"}
# Nonessential AAs get default gray/blue hues later

groups = list(group_summaries.keys())
n_groups = len(groups) + 1  # +1 for theoretical
x = np.arange(len(ALL_AAS))

# choose width so all groups fit comfortably
total_width = 0.8
width = total_width / n_groups

# color palette for empirical groups
colors = plt.cm.tab10.colors

# --- Plot empirical groups ---
for i, grp in enumerate(groups):
    summary = group_summaries[grp].reindex(ALL_AAS)
    med = summary["Median"].to_numpy(float)
    low = summary["Lower"].to_numpy(float)
    high = summary["Upper"].to_numpy(float)

    err_lower = np.nan_to_num(med - low, nan=0.0)
    err_upper = np.nan_to_num(high - med, nan=0.0)
    yerr = [err_lower, err_upper]

    # color code: red if essential, else palette color
    bar_colors = [
        "red" if aa in ESSENTIAL_AAS else colors[i % len(colors)]
        for aa in ALL_AAS
    ]

    plt.bar(
        x + i * width,
        med,
        width=width,
        color=bar_colors,
        alpha=0.9,
        label=f"{grp} (empirical)",
        yerr=yerr,
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
    )

# --- Plot theoretical values ---
theory_vals = np.array([theoretical_nL.get(aa, np.nan) for aa in ALL_AAS], dtype=float)
i_theory = len(groups)
plt.bar(
    x + i_theory * width,
    theory_vals,
    width=width,
    color="lightgray",
    label="Theoretical",
    edgecolor="black",
    linewidth=0.8,
    hatch="///",
    alpha=0.95,
)

# --- Formatting ---
center_offset = (n_groups - 1) / 2 * width
plt.xticks(x + center_offset, ALL_AAS)
plt.ylabel("Estimated nₗ per amino acid")
plt.title("Empirical amino acid nₗ values (Essential AAs in red) — Theoretical as bar")
plt.grid(axis="y", alpha=0.25)
plt.legend(fontsize=8, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
plt.tight_layout()
plt.show()


# x ticks centered under group cluster
center_offset = (n_groups - 1) / 2 * width
plt.xticks(x + center_offset, ALL_AAS)
plt.ylabel("Estimated nₗ per amino acid")
plt.title("Empirical amino acid nₗ values with 95% CI (per genotype) — Theoretical as bar")
plt.grid(axis="y", alpha=0.25)
plt.legend(fontsize=8, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
plt.tight_layout()
plt.show()
