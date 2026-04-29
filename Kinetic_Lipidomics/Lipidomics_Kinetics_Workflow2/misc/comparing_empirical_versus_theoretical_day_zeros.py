# -*- coding: utf-8 -*-
"""
Two‑file Relative Deviation comparison workflow (2‑panel version)
Created March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust.scale import mad
import tkinter as tk
from tkinter import filedialog
import os

# --------------------------------------------------------------
# ---------- Helper Functions ----------------------------------
# --------------------------------------------------------------

def parse_list(x):
    return np.array([float(i.strip()) for i in str(x).split(",")])

def pad(arr, max_len):
    return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

def process_df(df):
    """Process a dataframe: filtering, deduplication, parsing, padding."""
    
    # Filter time == 0
    df = df[df["time"] == 0].copy()
    
    # Drop duplicates
    df = df.drop_duplicates(
        subset=["Metabolite name", "Adduct type", "sample_group"],
        keep="first"
    )

    # Parse lists
    df["emp"]  = df["normalized_empirical_abundances"].apply(parse_list)
    df["theo"] = df["Theoretical Unlabeled Normalized Abundances"].apply(parse_list)

    # Max neutromer length
    max_len = df["emp"].apply(len).max()

    # Padding
    df["emp_pad"]  = df["emp"].apply(lambda x: pad(x, max_len))
    df["theo_pad"] = df["theo"].apply(lambda x: pad(x, max_len))


    return df, max_len


def compute_relative_deviation(df, max_len, eps=1e-12):
    """Compute neutromer‑wise relative deviation (MAD of fractional differences)."""
    
    results = {}

    for sg, grp in df.groupby("sample_group"):
        emp = np.vstack(grp["emp_pad"])
        theo = np.vstack(grp["theo_pad"])

        # Relative deviation
        diff = (emp - theo) / (theo + eps)

        rd_vals = np.array([
            mad(col[~np.isnan(col)], axis=0)
            for col in diff.T
        ])

        results[sg] = rd_vals

    return results


# --------------------------------------------------------------
# ---------- File Loading --------------------------------------
# --------------------------------------------------------------

root = tk.Tk()
root.withdraw()

print("Select FIRST TSV file...")
file1 = filedialog.askopenfilename(
    title="Select FIRST TSV",
    filetypes=[("TSV files", "*.tsv")]
)

print("Select SECOND TSV file...")
file2 = filedialog.askopenfilename(
    title="Select SECOND TSV",
    filetypes=[("TSV files", "*.tsv")]
)

if not file1 or not file2:
    raise SystemExit("Both files must be selected.")

def detect_sep(path):
    return "\t" if path.lower().endswith(".tsv") else ","


df1 = pd.read_csv(file1, sep=detect_sep(file1))
df2 = pd.read_csv(file2, sep=detect_sep(file2))


df1 = df1[df1["Average Rt(min)"] < 20]
df2 = df2[df2["Average Rt(min)"] < 20]


# --------------------------------------------------------------
# ---------- Process Data --------------------------------------
# --------------------------------------------------------------

df1_proc, max1 = process_df(df1)
df2_proc, max2 = process_df(df2)

# --------------------------------------------------------------
# ---------- Shared Species ------------------------------------
# --------------------------------------------------------------

def common_species_across_groups(df, groups=("A2", "A3", "A4")):
    group_sets = []
    for g in groups:
        grp = df[df["sample_group"] == g]
        group_sets.append(set(zip(grp["Metabolite name"], grp["Adduct type"])))
    return set.intersection(*group_sets)

species1 = common_species_across_groups(df1_proc)
species2 = common_species_across_groups(df2_proc)

common_species = species1.intersection(species2)

df1_common = df1_proc[
    df1_proc.apply(lambda r: (r["Metabolite name"], r["Adduct type"]) in common_species, axis=1)
]

df2_common = df2_proc[
    df2_proc.apply(lambda r: (r["Metabolite name"], r["Adduct type"]) in common_species, axis=1)
]

# --------------------------------------------------------------
# ---------- Unified Neutromer Padding -------------------------
# --------------------------------------------------------------

max_len = max(max1, max2)

df1_common["emp_pad"]  = df1_common["emp"].apply(lambda x: pad(x, max_len))
df1_common["theo_pad"] = df1_common["theo"].apply(lambda x: pad(x, max_len))

df2_common["emp_pad"]  = df2_common["emp"].apply(lambda x: pad(x, max_len))
df2_common["theo_pad"] = df2_common["theo"].apply(lambda x: pad(x, max_len))

# --------------------------------------------------------------
# ---------- Compute Relative Deviation -------------------------
# --------------------------------------------------------------

rd1 = compute_relative_deviation(df1_common, max_len)
rd2 = compute_relative_deviation(df2_common, max_len)


# --------------------------------------------------------------
# ---------- Compute AVERAGE Relative Deviation per genotype ---
# --------------------------------------------------------------

avg_rd1 = {sg: np.nanmean(vals) for sg, vals in rd1.items()}
avg_rd2 = {sg: np.nanmean(vals) for sg, vals in rd2.items()}

print("\nAverage Relative Deviation (File 1):")
for sg, val in avg_rd1.items():
    print(f"  {sg}: {val:.5f}")

print("\nAverage Relative Deviation (File 2):")
for sg, val in avg_rd2.items():
    print(f"  {sg}: {val:.5f}")


# --------------------------------------------------------------
# ---------- Two‑Panel Plot (Side-by-Side, 12‑point font) -------
# --------------------------------------------------------------

plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # shared y-axis
colors = ["teal", "orange", "magenta"]

# x-axis ticks as integers with M-subscript labels
x_vals = np.arange(max_len)
x_labels = [f"M$_{{{x}}}$" for x in x_vals]   # M₀, M₁, M₂ ...

# ---- Panel A (File 1) ----
ax = axes[0]
for i, (sg, vals) in enumerate(rd1.items()):
    ax.plot(x_vals, vals, color=colors[i % 3], linewidth=2, label=sg)

ax.set_xlabel("Neutromer")
ax.set_ylabel("Relative Deviation")
ax.set_xticks(x_vals)
ax.set_xticklabels(x_labels)

# Panel label
ax.text(0.02, 0.95, "A", transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

ax.legend(fontsize=12)

# ---- Panel B (File 2) ----
ax = axes[1]
for i, (sg, vals) in enumerate(rd2.items()):
    ax.plot(x_vals, vals, color=colors[i % 3], linewidth=2, label=sg)

ax.set_xlabel("Neutromer")
ax.set_xticks(x_vals)
ax.set_xticklabels(x_labels)

# Panel label
ax.text(0.02, 0.95, "B", transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

ax.legend(fontsize=12)

plt.tight_layout()


# --------------------------------------------------------------
# ---------- Save Output ---------------------------------------
# --------------------------------------------------------------

out_path = os.path.join(os.path.dirname(file1), "RelativeDeviation_comparison_2panel.svg")
plt.savefig(out_path, format="svg")

print(f"\nSaved 2‑panel Relative Deviation SVG to:\n{out_path}\n")