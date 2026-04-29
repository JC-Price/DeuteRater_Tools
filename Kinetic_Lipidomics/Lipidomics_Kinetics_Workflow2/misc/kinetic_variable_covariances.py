#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended slope-surface export for all 6 parameter pairs using Wald covariance.
Input file is the binomial_results file from DeuteRater
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================
# GLOBAL FONT SETTINGS (~12 pt)
# =========================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})


# =========================================================
# FILE PICKER
# =========================================================
def pick_df():
    root = tk.Tk()
    root.withdraw()
    root.update()

    path = filedialog.askopenfilename(
        title="Select results file",
        filetypes=[
            ("TSV", "*.tsv"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx"),
            ("All files", "*.*")
        ]
    )

    root.update()
    root.destroy()

    if not path:
        print("No file selected.")
        sys.exit(0)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return path, df


# =========================================================
# COMPUTE ALL SLOPES
# =========================================================
def compute_slopes(df,
                   flag_col="Flag",
                   pcov_col="BA_pcov",
                   nL_col="BA_nL",
                   rate_col="BA_rate",
                   A_col="BA_Asyn"):

    mask_pass = df[flag_col].astype(str).str.startswith("PASS") if flag_col in df.columns else pd.Series(True, index=df.index)

    sub = df.loc[mask_pass, [nL_col, rate_col, A_col, pcov_col]].dropna().copy()
    if sub.empty:
        return sub

    dA_dnL, drate_dnL, dnL_drate = [], [], []
    dA_drate, dnL_dA, drate_dA = [], [], []

    for s in sub[pcov_col]:
        try:
            M = np.array(json.loads(s), dtype=float)

            var_nL   = M[0, 0]
            var_rate = M[1, 1]
            var_A    = M[2, 2]

            cov_nL_rate = M[0, 1]
            cov_nL_A    = M[0, 2]
            cov_rate_A  = M[1, 2]

            dA_dnL.append(cov_nL_A / var_nL if var_nL > 0 else np.nan)
            drate_dnL.append(cov_nL_rate / var_nL if var_nL > 0 else np.nan)
            dnL_drate.append(cov_nL_rate / var_rate if var_rate > 0 else np.nan)
            dA_drate.append(cov_rate_A / var_rate if var_rate > 0 else np.nan)
            dnL_dA.append(cov_nL_A / var_A if var_A > 0 else np.nan)
            drate_dA.append(cov_rate_A / var_A if var_A > 0 else np.nan)

        except Exception:
            dA_dnL.append(np.nan)
            drate_dnL.append(np.nan)
            dnL_drate.append(np.nan)
            dA_drate.append(np.nan)
            dnL_dA.append(np.nan)
            drate_dA.append(np.nan)

    sub["dA_dnL"] = dA_dnL
    sub["drate_dnL"] = drate_dnL
    sub["dnL_drate"] = dnL_drate
    sub["dA_drate"] = dA_drate
    sub["dnL_dA"] = dnL_dA
    sub["drate_dA"] = drate_dA

    return sub.replace([np.inf, -np.inf], np.nan)


# =========================================================
# SAFE FILE NAME
# =========================================================
def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# =========================================================
# MATPLOTLIB SLOPE SURFACE (SVG)
# =========================================================
def make_surface_svg(df, base, x_col, y_col, slope_col,
                     x_label, y_label, slope_label,
                     nx=40, ny=30, min_count=10,
                     clip_scale=True):

    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    z = df[slope_col].to_numpy(float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    x_edges = np.linspace(np.min(x), np.max(x), nx + 1)
    y_edges = np.linspace(np.min(y), np.max(y), ny + 1)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1

    grid = pd.DataFrame({"xbin": xi, "ybin": yi, "z": z})
    grid = grid[(grid.xbin >= 0) & (grid.xbin < nx)
                & (grid.ybin >= 0) & (grid.ybin < ny)]

    med = grid.groupby(["ybin", "xbin"])["z"].median()
    cnt = grid.groupby(["ybin", "xbin"]).size()

    Z = np.full((ny, nx), np.nan)
    C = np.zeros((ny, nx), int)

    for (iy, ix), val in med.items():
        Z[iy, ix] = val
        C[iy, ix] = cnt.loc[(iy, ix)]

    Z[C < min_count] = np.nan

    if np.isfinite(Z).any():
        vabs = np.nanpercentile(np.abs(Z[np.isfinite(Z)]), 95)
        if not np.isfinite(vabs) or vabs == 0:
            vabs = 10.0
    else:
        vabs = 10.0

    if clip_scale:
        vabs = min(vabs, 10.0)

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{slope_label} across ({x_label}, {y_label})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(slope_label, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    out_svg = f"{base}_{safe_name(slope_col)}.svg"
    plt.savefig(out_svg, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return out_svg


# =========================================================
# MAIN
# =========================================================
def main():
    in_path, df = pick_df()
    base, _ = os.path.splitext(in_path)

    sub = compute_slopes(df)
    if sub.empty:
        print("No valid rows.")
        return

    surfaces = []

    def is_unclipped(slope_col):
        return slope_col.startswith("dnL_")

    # ✅ FINAL LABELS (mathtext + subscripts)
    requests = [
        ("BA_nL",  "BA_Asyn", "dA_dnL",    r"$n_L$",   r"$A_{syn}$", r"$\partial A_{syn} / \partial n_L$"),
        ("BA_nL",  "BA_rate", "drate_dnL", r"$n_L$",   r"$rate$",    r"$\partial rate / \partial n_L$"),
        ("BA_rate","BA_nL",   "dnL_drate", r"$rate$",  r"$n_L$",     r"$\partial n_L / \partial rate$"),
        ("BA_rate","BA_Asyn", "dA_drate",  r"$rate$",  r"$A_{syn}$", r"$\partial A_{syn} / \partial rate$"),
        ("BA_Asyn","BA_nL",   "dnL_dA",    r"$A_{syn}$", r"$n_L$",   r"$\partial n_L / \partial A_{syn}$"),
        ("BA_Asyn","BA_rate", "drate_dA",  r"$A_{syn}$", r"$rate$",  r"$\partial rate / \partial A_{syn}$"),
    ]

    for xcol, ycol, scol, xlabel, ylabel, slabel in requests:
        clip_flag = not is_unclipped(scol)

        surfaces.append(make_surface_svg(
            sub, base,
            xcol, ycol, scol,
            xlabel, ylabel, slabel,
            clip_scale=clip_flag
        ))

    print("\nGenerated SVG files:")
    for s in surfaces:
        print("  ", s)


if __name__ == "__main__":
    main()