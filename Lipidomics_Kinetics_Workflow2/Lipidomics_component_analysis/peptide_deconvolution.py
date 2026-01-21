#!/usr/bin/env python3
# peptide_deconvolution.py
# QCM-based filtering + negative AA contributions + 3D QC sweep
#
# INPUT COLUMNS REQUIRED (and only these are assumed):
# Alignment ID, Ontology, metric, value, se,genotype
#
# QC RULE (IMPORTANT):
# - REQUIRE n_value.qcm exists (finite)
# - Allow missing k/asyn qcm to PASS
# - If k/asyn qcm exists, enforce threshold

import os
from collections import Counter
from datetime import datetime

import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import truncnorm
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score


# ----------------------------
# File load (GUI)
# ----------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select tidy CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
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

expected = ["Sequence", "Ontology", "metric", "value", "se","group_name"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise SystemExit(f"ERROR: Missing expected columns: {missing}")

# Ensure numeric columns
for col in ["value", "se"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Coerce SE
df["se"] = pd.to_numeric(df["se"], errors="coerce")

# Filter invalid SE rows
bad_se_mask = (~np.isfinite(df["se"])) | (df["se"] <= 0)

n_bad = bad_se_mask.sum()
if n_bad > 0:
    print(f"⚠️ Dropping {n_bad} rows with invalid SE (<=0 or NaN)")
    df = df.loc[~bad_se_mask].copy()
# ----------------------------
# Drop invalid sequences
# ----------------------------
df = df.dropna(subset=["Sequence", "value"]).copy()
df["Sequence"] = df["Sequence"].astype(str)
# Filter rows


valid_aas = set("ARNDCEQGHILKMFPSTWYVarndceqghilkmfpstwyv")
df = df[df["Sequence"].apply(lambda s: all(ch in valid_aas for ch in s))].copy()

GROUPS = sorted(df["group_name"].dropna().unique().tolist())
print("Detected genotypes:", GROUPS)


# ----------------------------
# SE-only uncertainty handling (CI-free)
# ----------------------------

df["se"] = pd.to_numeric(df["se"], errors="coerce")

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
    A_aug = np.vstack([A, np.sqrt(l2_reg)*np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    x, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    return x

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def run_jackknife_mc_allowneg(
    A, medians, ses,
    num_jackknifes=20, drop_fraction=0.2,
    mc_per_boot=200, seed=0,
    allow_negative=True,
    l2_reg=1e-8,
    skip_underdetermined=True,
    show_progress=True
):
    rng = np.random.default_rng(seed)
    n_pep, n_aa = A.shape
    all_results = []
    diagnostics = {"attempts": 0, "succeeded": 0, "skipped_underd": 0, "solver_fails": 0}

    sd_array = np.asarray(ses, float)
    if not np.all(np.isfinite(sd_array)) or np.any(sd_array <= 0):
        raise ValueError("SE must be finite and > 0 for all rows")

    use_tqdm = (show_progress and (tqdm is not None))
    outer_iter = range(num_jackknifes)
    if use_tqdm:
        outer_iter = tqdm(outer_iter, desc="Jackknifes")

    for b in outer_iter:
        diagnostics["attempts"] += 1

        keep_idx = rng.choice(n_pep, size=max(1, int(n_pep*(1-drop_fraction))), replace=False)
        A_sub = A[keep_idx, :]
        med_sub = medians[keep_idx]
        sd_sub = sd_array[keep_idx]

        if skip_underdetermined and (A_sub.shape[0] <= A_sub.shape[1]):
            diagnostics["skipped_underd"] += 1
            continue

        inner_iter = range(mc_per_boot)
        if use_tqdm:
            inner_iter = tqdm(inner_iter, desc=f"MC jk#{b+1}", leave=False)

        for _ in inner_iter:
            sampled = rng.normal(loc=med_sub, scale=sd_sub)
            try:
                x = solve_unconstrained_ridge(A_sub, sampled, l2_reg=l2_reg)
                if np.all(np.isfinite(x)):
                    all_results.append(x)
                    diagnostics["succeeded"] += 1
                else:
                    diagnostics["solver_fails"] += 1
            except Exception:
                diagnostics["solver_fails"] += 1

    results = np.asarray(all_results) if all_results else np.empty((0, n_aa))
    return results, diagnostics

# ----------------------------
# Output folder
# ----------------------------
base = os.path.splitext(os.path.basename(file_path))[0]
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_root = os.path.join(os.path.dirname(file_path), f"peptide_deconv_out_{base}_{timestamp}")
os.makedirs(out_root, exist_ok=True)
print(f"\nAll outputs will be saved in:\n{out_root}")

# ----------------------------
# Define QC cutoff sweeps (QCM thresholds)
# ----------------------------
SE_K_RANGE    = [0.5]
SE_ASYN_RANGE = [0.5]
SE_N_RANGE    = [0.5]




# ----------------------------
# theoretical AA nL values
# ----------------------------
theoretical_nL_master = {
    "A": 4.00, "C": 1.62, "D": 1.89, "E": 3.95, "F": 0.32,
    "G": 2.06, "H": 2.88, "I": 1.00, "K": 0.54, "L": 0.69,
    "M": 1.12, "N": 1.89, "P": 2.59, "Q": 3.95, "R": 3.34,
    "S": 2.61, "T": 0.20, "V": 0.56, "W": 0.08, "Y": 0.42
}

results_summary = []

# ----------------------------
# Begin QC sweep
# ----------------------------
total_tests = len(SE_K_RANGE)*len(SE_ASYN_RANGE)*len(SE_N_RANGE)
counter = 0
for z_k_cut in SE_K_RANGE:
    for z_asyn_cut in SE_ASYN_RANGE:
        for z_n_cut in SE_N_RANGE:
            counter +=1 

            tag = f"qcK{z_k_cut}_qcA{z_asyn_cut}_qcN{z_n_cut}"
            iter_dir = os.path.join(out_root, tag)
            os.makedirs(iter_dir, exist_ok=True)

            print(f"\n=== QC filtering pass {tag} {counter}/{total_tests} ===")

            # ----------------------------
            # QCM-based filtering (FIXED LOGIC)
            # ----------------------------
            filtered_pairs = []

            for (seq, grp), sub in df_all.groupby(["Sequence", "group_name"]):
            
                # Extract SE by metric
                se_k_vals = sub.loc[sub["metric"].str.lower() == "k", "se"].dropna()
                se_A_vals = sub.loc[sub["metric"].str.lower() == "asyn", "se"].dropna()
                se_n_vals = sub.loc[sub["metric"].str.lower().isin(["n", "n_value"]), "se"].dropna()
            
                # Require n_value SE
                if se_n_vals.empty:
                    continue
            
                # Determine whether Asyn should be ignored
                ignore_Asyn = False
                if not se_A_vals.empty:
                    if (se_A_vals == 0).all():
                        ignore_Asyn = True
            
                # Build SE list for global rule
                if ignore_Asyn:
                    # ignore Asyn SE completely
                    se_for_global = pd.concat([se_k_vals, se_n_vals])
                else:
                    se_for_global = pd.concat([se_k_vals, se_n_vals, se_A_vals])
            
                # Global rule: reject if ANY relevant SE < 0.001
                if (se_for_global < 0.005).any():
                    continue
            
                # Metric-specific QC threshold extraction
                qc_k = se_k_vals.max() if not se_k_vals.empty else np.nan
                qc_A = se_A_vals.max() if not se_A_vals.empty else np.nan
                qc_n = se_n_vals.max()
            
                # Apply thresholds
                k_ok = (not np.isfinite(qc_k)) or (qc_k <= z_k_cut)
            
                if ignore_Asyn:
                    A_ok = True
                else:
                    A_ok = (not np.isfinite(qc_A)) or (qc_A <= z_asyn_cut)
            
                n_ok = qc_n <= z_n_cut
            
                if k_ok and A_ok and n_ok:
                    filtered_pairs.append((seq, grp))


            n_pairs = len(filtered_pairs)
            print(f"→ {n_pairs} peptide×group combos passed QC")

            if n_pairs == 0:
                open(os.path.join(iter_dir, f"EMPTY_{tag}.txt"), "w").write(
                    "No peptide×group combos passed QC.\n"
                )
                results_summary.append({
                    "z_k_cut": z_k_cut,
                    "z_Asyn_cut": z_asyn_cut,
                    "z_n_cut": z_n_cut,
                    "n_pairs": 0,
                    "n_points": 0,
                    "mean_r2": np.nan
                })
                continue

            idx_pass = pd.MultiIndex.from_tuples(filtered_pairs)
            df_iter = df_all.set_index(["Sequence","group_name"]).loc[
                lambda x: x.index.isin(idx_pass)
            ].reset_index()

            # Save QC stats
            summary_qc = df_iter.groupby("metric")["se"].agg(["mean","std","min","max"])

            open(os.path.join(iter_dir, f"QC_summary_{tag}.txt"), "w").write(str(summary_qc.round(3)))

            # ----------------------------
            # Extract n_value rows only
            # ----------------------------
            df_n = df_iter[df_iter["metric"].str.lower()=="n_value"].copy()
            df_n = df_n.rename(columns={
                "value": "n_value",
                "se": "n_val_se"
            })


            for col in ["n_value","n_val_se"]:
                df_n[col] = pd.to_numeric(df_n[col], errors="coerce")


            n_points = len(df_n)
            print(f"n_value rows retained: {n_points}")

            # ----------------------------
            # filter by nL > 10
            # ----------------------------
            before = len(df_n)
            df_n = df_n[df_n["n_value"] > 10].copy()
            after = len(df_n)
            print(f"Filtering nL > 10 kept {after}/{before}")

            if after == 0:
                open(os.path.join(iter_dir, f"EMPTY_nL_{tag}.txt"), "w").write(
                    "No peptides with nL > 10.\n"
                )
                results_summary.append({
                    "z_k_cut": z_k_cut,
                    "z_Asyn_cut": z_asyn_cut,
                    "z_n_cut": z_n_cut,
                    "n_pairs": n_pairs,
                    "n_points": before,
                    "mean_r2": np.nan
                })
                continue

            # ----------------------------
            # AA ordering
            # ----------------------------
            ALL_AAS = sorted(set("".join(df_n["Sequence"].str.upper())))
            print("Amino acids:", "".join(ALL_AAS))

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

                n_pep = len(sub)
                A = np.zeros((n_pep, len(ALL_AAS)), int)
                for i, seq in enumerate(sub["Sequence"].str.upper()):
                    for aa, cnt in Counter(seq).items():
                        if aa in ALL_AAS:
                            A[i, ALL_AAS.index(aa)] = cnt

                medians   = sub["n_value"].to_numpy(float)
                ses       = sub["n_val_se"].to_numpy(float)

                results_obs, diag = run_jackknife_mc_allowneg(
                    A,
                    medians,
                    ses,
                    num_jackknifes=20,
                    mc_per_boot=200,
                    seed=0,
                    allow_negative=True,
                    l2_reg=1e-8
                )

                diag_all[grp] = diag

                if results_obs.size == 0:
                    summary = pd.DataFrame({
                        "AminoAcid": ALL_AAS,
                        "Median": np.nan,
                        "Lower": np.nan,
                        "Upper": np.nan
                    }).set_index("AminoAcid")
                else:
                    summary = pd.DataFrame({
                        "AminoAcid": ALL_AAS,
                        "Median": np.median(results_obs, axis=0),
                        "Lower":  np.percentile(results_obs, 2.5, axis=0),
                        "Upper":  np.percentile(results_obs,97.5, axis=0)
                    }).set_index("AminoAcid")

                group_summaries[grp] = summary

            if not group_summaries:
                open(os.path.join(iter_dir, f"EMPTY_SUMMARY_{tag}.txt"), "w").write("No genotype summaries.\n")
                continue

            # ----------------------------
            # Combine summaries + theoretical
            # ----------------------------
            summary_combined = pd.concat(group_summaries, axis=1)
            summary_combined[("Theoretical","Median")] = [
                theoretical_nL_master.get(aa, np.nan) for aa in summary_combined.index
            ]
            summary_combined[("Theoretical","Lower")] = np.nan
            summary_combined[("Theoretical","Upper")] = np.nan
            summary_combined.to_csv(os.path.join(iter_dir, f"combined_summary_{tag}.csv"))
            # ----------------------------
            # Passing–Bablok & Bland–Altman
            # ----------------------------
            
            def passing_bablok(x, y):
                x, y = np.asarray(x,float), np.asarray(y,float)
                n=len(x)
                slopes=[]
                for i in range(n-1):
                    for j in range(i+1,n):
                        if x[j]!=x[i]:
                            slopes.append((y[j]-y[i])/(x[j]-x[i]))
                if not slopes:
                    return np.nan,np.nan,(np.nan,np.nan),(np.nan,np.nan)
                slopes=np.sort(slopes)
                m=np.median(slopes)
                intercepts=y-m*x
                b=np.median(intercepts)
                M=len(slopes)
                z=1.96
                k=int(np.floor((M - z*np.sqrt(M))/2))
                l=int(np.ceil((M + z*np.sqrt(M))/2))-1
                k=max(0,k); l=min(l,M-1)
                slope_ci=(slopes[k],slopes[l])
                intercepts_low=y - slope_ci[0]*x
                intercepts_high=y - slope_ci[1]*x
                intercept_ci=(np.median(intercepts_low), np.median(intercepts_high))
                return m,b,slope_ci,intercept_ci
            
            mean_r2s = []
            
            for grp, summary in group_summaries.items():
                sub = df_n[df_n["group_name"]==grp].copy()
                if sub.empty:
                    continue
            
                # Build design matrix
                n_pep_sub=len(sub)
                A_sub=np.zeros((n_pep_sub,len(ALL_AAS)),int)
                for i,seq in enumerate(sub["Sequence"].str.upper()):
                    for aa,cnt in Counter(seq).items():
                        if aa in ALL_AAS:
                            A_sub[i,ALL_AAS.index(aa)] = cnt
            
                aa_med_vec = np.array([theoretical_nL_master.get(aa,np.nan) for aa in ALL_AAS])
                predicted = A_sub.dot(aa_med_vec)
            
                sub = sub.reset_index(drop=True)
                sub["predicted_nL"] = predicted
                sub = sub.dropna(subset=["n_value","predicted_nL"])
            
                x = sub["predicted_nL"].to_numpy(float)
                y = sub["n_value"].to_numpy(float)
            
                if len(x) < 3:
                    print(f"[WARN] insufficient points for PB regression in {grp}")
                    continue
            
                slope,intercept,slope_ci,intercept_ci = passing_bablok(x,y)
                y_pred = slope*x + intercept
                r2 = r2_score(y,y_pred)
                if np.isfinite(r2):
                    mean_r2s.append(r2)
            
                # ---- Passing–Bablok plot ----
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope*x_fit + intercept
                y_low = slope_ci[0]*x_fit + intercept_ci[0]
                y_high = slope_ci[1]*x_fit + intercept_ci[1]
            
                plt.figure(figsize=(6,6),dpi=150)
                plt.scatter(x, y, facecolors="none", edgecolors="blue")
                plt.plot(x_fit, y_fit, "r-")
                plt.fill_between(x_fit, y_low, y_high, color="lightgray", alpha=0.5)
                plt.plot(x_fit, x_fit, "--", color="gray")
            
                plt.xlabel("Predicted nL")
                plt.ylabel("Measured nL")
                plt.title(f"{grp} Passing–Bablok\nR2={r2:.3f}\n{tag}")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(iter_dir, f"PB_{grp}_{tag}.png"), dpi=300)
                plt.close()
            
                # ---- Bland–Altman ----
                mean_vals = (x + y) / 2
                diff_vals = y - x
                mean_diff = np.mean(diff_vals)
                sd_diff = np.std(diff_vals)
                upper = mean_diff + 1.96*sd_diff
                lower = mean_diff - 1.96*sd_diff
            
                plt.figure(figsize=(6,6),dpi=150)
                plt.scatter(mean_vals, diff_vals, facecolors="none", edgecolors="purple")
                plt.axhline(mean_diff, color="red", ls="--")
                plt.axhline(upper, color="gray", ls="--")
                plt.axhline(lower, color="gray", ls="--")
                plt.xlabel("Mean predicted/measured")
                plt.ylabel("Measured – predicted")
                plt.title(f"{grp} Bland–Altman\n{tag}")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(iter_dir, f"BA_{grp}_{tag}.png"), dpi=300)
                plt.close()
            
            
            # ----------------------------
            # barplot empirical vs theoretical
            # ----------------------------
            ESSENTIAL_AAS = {"H","I","L","K","M","F","T","W","V","Y"}
            
            plt.figure(figsize=(12,5),dpi=150)
            groups_list = list(group_summaries.keys())
            n_groups = len(groups_list) + 1
            x_idx = np.arange(len(ALL_AAS))
            total_width = 0.8
            width = total_width / n_groups
            colors = plt.cm.tab10.colors
            
            for i,grp in enumerate(groups_list):
                summary = group_summaries[grp].reindex(ALL_AAS)
                med = summary["Median"].to_numpy(float)
                low = summary["Lower"].to_numpy(float)
                high = summary["Upper"].to_numpy(float)
                err = [med-low, high-med]
            
                bar_colors = ["red" if aa in ESSENTIAL_AAS else colors[i%len(colors)] for aa in ALL_AAS]
            
                plt.bar(x_idx + i*width, med, width=width, color=bar_colors,
                        yerr=err, capsize=3, edgecolor="black", label=f"{grp}")
            
            theory_vals = np.array([theoretical_nL_master.get(aa,np.nan) for aa in ALL_AAS])
            plt.bar(x_idx + len(groups_list)*width, theory_vals,
                    width=width, color="lightgray", edgecolor="black",
                    hatch="///", label="Theoretical")
            
            plt.xticks(x_idx + (n_groups-1)/2*width, ALL_AAS)
            plt.ylabel("Estimated nL per AA")
            plt.title(f"Empirical vs Theoretical AA nL values\n{tag}")
            plt.legend(fontsize=8)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(iter_dir, f"bar_empirical_vs_theoretical_{tag}.png"), dpi=300)
            plt.close()


            # ----------------------------
            # Quick R2 check vs theoretical (simple, fast)
            # ----------------------------
            for grp in group_summaries.keys():
                sub = df_n[df_n["group_name"]==grp].copy()
                if len(sub) < 3:
                    continue

                A_sub = np.zeros((len(sub), len(ALL_AAS)), int)
                for i, seq in enumerate(sub["Sequence"].str.upper()):
                    for aa, cnt in Counter(seq).items():
                        if aa in ALL_AAS:
                            A_sub[i, ALL_AAS.index(aa)] = cnt

                aa_med_vec = np.array([theoretical_nL_master.get(aa, np.nan) for aa in ALL_AAS])
                predicted = A_sub.dot(aa_med_vec)

                y = sub["n_value"].to_numpy(float)
                mask = np.isfinite(predicted) & np.isfinite(y)
                if mask.sum() >= 3:
                    r2 = r2_score(y[mask], predicted[mask])
                    mean_r2s.append(r2)

            mean_r2 = float(np.nanmean(mean_r2s)) if mean_r2s else np.nan

            results_summary.append({
                "z_k_cut": z_k_cut,
                "z_Asyn_cut": z_asyn_cut,
                "z_n_cut": z_n_cut,
                "n_pairs": n_pairs,
                "n_points": len(df_n),
                "mean_r2": mean_r2
            })

            # Save diagnostics
            pd.DataFrame.from_dict(diag_all, orient="index").to_csv(
                os.path.join(iter_dir, f"solver_diagnostics_{tag}.csv")
            )

# ----------------------------
# Save final sweep table
# ----------------------------
results_df = pd.DataFrame(results_summary)
results_df.sort_values(["mean_r2","n_points"], ascending=[False, False], inplace=True)
results_df.to_csv(os.path.join(out_root, "qc_filter_grid_summary.csv"), index=False)

print("\nSaved QC sweep summary:")
print(results_df.head(10).to_string(index=False))
