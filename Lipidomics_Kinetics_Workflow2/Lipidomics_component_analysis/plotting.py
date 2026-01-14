"""This code was authored by Coleman Nielsen, with support from ChatGPT"""
# ================================================================
# PLOTTING MODULE (6. plotting)
#
# This module handles all visualization of component n-values:
#   - Scatter plots with paired t-tests
#   - Conformity plots (baseline vs other genotypes)
#   - Bar plots of jackknifeped component estimates
#   - Special comparison plots for palmitate/stearate vs. literature
#
# Functions:
#   _build_component_styles()
#       - Assigns deterministic (marker, color) pairs to categories.
#
#   scatter_ttest_components()
#       - Scatter plot of paired component values.
#       - Shows included vs excluded components.
#       - Runs correlation-adjusted t-test (2. stats_tests).
#       - Annotates plot with summary statistics.
#
#   conformity_plots_from_components()
#       - For a chosen baseline genotype, builds a grid of
#         scatter plots vs every other genotype.
#
#   linear_algebra_plot_from_tidy()
#       - End-to-end pipeline:
#           (A) Drop uninformative rows (CI=0).
#           (B) Call jackknife (5. jackknife) to estimate n-values.
#           (C) Restrict to components derived from lipids common
#               to all genotypes.
#           (D) Plot bar chart of median n-values + error bars.
#           (E) Save component table (with MAD-z score).
#
#   plot_palmitate_stearate_comparison()
#       - Uses existing comp_df (no recalculation).
#       - Restricts to palmitate (16:0) and stearate (18:0).
#       - Compares jackknife estimates to Lee et al. (1994)
#         empirical values and theoretical maxima.
#
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.lines import Line2D
from tkinter import filedialog

from stats_test import adjusted_ttest        # (2. stats_tests)
from jackknife import jackknife_component_n_values_from_tidy  # (5. jackknife)

# --- global plot style ---
from matplotlib import rcParams

from histograms import plot_fatty_acid_histograms_separate
rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16,
})

# --- per-component styles ---
_MARKERS = ['o','s','D','^','v','<','>','*','p','h','X','P','H','d',
            '1','2','3','4','+','x','|','_','.']
_COLORS  = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple',
            'tab:brown','tab:olive','tab:cyan','magenta','goldenrod',
            'teal','slategray']

# ------------------------------------------------------------
# (6a) Marker/color assignment
# ------------------------------------------------------------
def _build_component_styles(categories: list[str]) -> dict[str, dict]:
    """
    Deterministic mapping: each category gets a (marker, color) pair.
    Cycles through the Cartesian product of MARKERS × COLORS.
    """
    pairs = list(product(_MARKERS, _COLORS))
    styles = {}
    for i, c in enumerate(categories):
        m, col = pairs[i % len(pairs)]
        styles[c] = {'marker': m, 'color': col}
    return styles


# ------------------------------------------------------------
# (6b) Scatter plot with t-test annotation
# ------------------------------------------------------------
def scatter_ttest_components(
    ax,
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    x_col: str,
    y_col: str,
    category_col: str = "Component",
    point_size: float = 32.0,
    comparison_label: str | None = None,
    axis_titles: list[str] | None = None,
    x_lim: float | None = None,
    y_lim: float | None = None,
    show_legend: bool = True,
    legend_outside: bool = True,
    legend_ncol: int = 6,
    legend_fontsize: int = 16,
) -> str:
    """
    Scatter plot of paired component values with correlation-adjusted t-test summary.
    - df_all     = full dataframe of components
    - df_filtered= subset (included components)
    - x_col, y_col = identifiers to compare
    """
    # (B1) Ensure numeric
    for col in [x_col, y_col]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")
    df_all = df_all.dropna(subset=[x_col, y_col])
    df_filtered = df_filtered.dropna(subset=[x_col, y_col])

    # (B2) Run paired t-tests
    if len(df_all) > 1:
        mean_diff_all, t_all, p_all, rho = adjusted_ttest(df_all, x_col, y_col)
    else:
        t_all, p_all, mean_diff_all = np.nan, np.nan, np.nan
    if len(df_filtered) > 1:
        from scipy import stats
        t_f, p_f = stats.ttest_rel(df_filtered[y_col], df_filtered[x_col])
        mean_diff_f = (df_filtered[y_col] - df_filtered[x_col]).mean()
    else:
        t_f, p_f, mean_diff_f = np.nan, np.nan, np.nan

    # (B3) Split excluded vs included
    excluded = df_all.loc[~df_all.index.isin(df_filtered.index)]

    # (B4) Styles per component
    cats = list(pd.unique(df_filtered[category_col].dropna())) if category_col in df_filtered.columns else []
    styles = _build_component_styles(cats) if cats else {}

    # (B5) Plot excluded (grey) + included (colored)
    ax.scatter(excluded[x_col], excluded[y_col],
               s=point_size, color="lightgrey", marker='o', linewidths=0.0)
    if cats:
        for c in cats:
            sub = df_filtered[df_filtered[category_col] == c]
            if not sub.empty:
                st = styles[c]
                ax.scatter(sub[x_col], sub[y_col],
                           s=point_size, marker=st['marker'], color=st['color'], linewidths=0.0)
    else:
        ax.scatter(df_filtered[x_col], df_filtered[y_col],
                   s=point_size, alpha=0.9, linewidths=0.0)

    # (B6) Identity line
    if x_lim is not None: ax.set_xlim(0, x_lim)
    if y_lim is not None: ax.set_ylim(0, y_lim)
    if x_lim is None and y_lim is None:
        ax.relim(); ax.autoscale_view()
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        rng = max(x1-x0, y1-y0); xc = 0.5*(x0+x1); yc = 0.5*(y0+y1)
        ax.set_xlim(xc - rng/2, xc + rng/2); ax.set_ylim(yc - rng/2, yc + rng/2)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="grey", linewidth=1, zorder=0)

    # (B7) Labels
    if axis_titles and len(axis_titles) >= 2:
        ax.set_xlabel(f"{x_col} {axis_titles[0]}")
        ax.set_ylabel(f"{y_col} {axis_titles[1]}")
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel("n_sub_L")

    # (B8) Title with t-test summary
    def _fmt(val: float, small="{:.2e}", large="{:.2f}"):
        try:
            return small.format(val) if np.isfinite(val) and abs(val) < 1e-2 else large.format(val)
        except Exception:
            return str(val)
    all_str = (
        f"$\\bar{{d}}$ = {_fmt(mean_diff_all)}, "
        f"t = {_fmt(t_all, '{:.2f}', '{:.2f}')}, "
        f"p = {_fmt(p_all, '{:.2e}', '{:.3f}')}"
    )
    summary = f"{all_str}"
    ax.set_title(f"{comparison_label}\n{summary}" if comparison_label else summary, fontsize=16)

    # (B9) Legend
    if show_legend and cats:
        handles = [Line2D([0],[0], marker=styles[c]['marker'], linestyle='',
                          color=styles[c]['color'], label=c,
                          markersize=np.sqrt(point_size)/1.5)
                   for c in cats]
        if legend_outside:
            ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.30),
                      frameon=True, ncol=legend_ncol, fontsize=legend_fontsize)
        else:
            ax.legend(handles=handles, loc='best', frameon=True, ncol=legend_ncol,
                      fontsize=legend_fontsize)
    return summary


# ------------------------------------------------------------
# (6c) Conformity plots
# ------------------------------------------------------------
def conformity_plots_from_components(df, comp_df, identifiers, baseline, num_simulations=500, seed=0):
    """
    Build conformity plots (paired t-tests) of component n-values:
    baseline vs each other identifier.
    """
    n = len(identifiers) - 1
    ncols = 2; nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), dpi=200)
    axs = np.atleast_1d(axs).ravel()

    panel = 0
    for ident in identifiers:
        if ident == baseline: continue
        ax = axs[panel]

        # Construct paired dataframe
        df_all = pd.DataFrame({
            f"{baseline}": comp_df[f"n_value_{baseline}_median"],
            f"{ident}": comp_df[f"n_value_{ident}_median"],
            "Component": comp_df["Component"]
        })
        df_filtered = df_all.copy()

        scatter_ttest_components(
            ax,
            df_all=df_all,
            df_filtered=df_filtered,
            x_col=baseline,
            y_col=ident,
            category_col="Component",
            comparison_label=f"{ident} vs {baseline}",
            axis_titles=[r'$n_L$', r'$n_L$'],
            show_legend=True,
            legend_outside=True,
            legend_ncol=6,
            legend_fontsize=16
        )
        panel += 1

    # Delete unused axes
    for j in range(panel, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Component conformity plots (baseline = {baseline})", fontsize=16)
    fig.tight_layout()
    return fig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from stats_test import adjusted_ttest

def conformity_plots_from_components(df, comp_df, identifiers, baseline):
    """
    Build conformity scatter plots comparing baseline vs other genotypes.
    Enhancements:
      - Outliers (|median| >= 3) are still plotted in grey.
      - Robust linear fit using HuberRegressor (outliers downweighted).
      - R² and t-test results included in the title.
    """
    figs = []
    baseline_col = f"n_value_{baseline}_median"

    for ident in identifiers:
        if ident == baseline:
            continue

        compare_col = f"n_value_{ident}_median"
        if baseline_col not in comp_df or compare_col not in comp_df:
            continue

        # --- Collect data
        x_all = comp_df[baseline_col].to_numpy(float)
        y_all = comp_df[compare_col].to_numpy(float)

        # Mask NaNs
        mask_valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        x_all, y_all = x_all[mask_valid], y_all[mask_valid]

        if len(x_all) < 2:
            continue

        # --- T-test (all data, including outliers)
        d_mean, t_stat, p_val, rho = adjusted_ttest(
            comp_df[mask_valid],
            baseline_col,
            compare_col
        )

        # --- Outlier detection (for marking only, not exclusion)
        mask_outliers = (np.abs(x_all) >= 3) | (np.abs(y_all) >= 3)

        # --- Build figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot main points
        ax.scatter(x_all[~mask_outliers], y_all[~mask_outliers],
                   color="blue", alpha=0.7, label="Components")

        # Plot outliers in grey
        if np.any(mask_outliers):
            ax.scatter(x_all[mask_outliers], y_all[mask_outliers],
                       color="grey", alpha=0.5, label="Outliers (downweighted)")

        # Reference line y = x
        lims = [min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())]
        ax.plot(lims, lims, 'k--', lw=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # --- Robust regression with HuberRegressor
        X = x_all.reshape(-1, 1)
        huber = HuberRegressor().fit(X, y_all)
        slope = huber.coef_[0]
        intercept = huber.intercept_

        # Predicted line
        xx = np.linspace(min(lims), max(lims), 200).reshape(-1, 1)
        yy = huber.predict(xx)
        ax.plot(xx, yy, color="red", lw=2, label="Huber fit")

        # Compute R² manually
        y_pred = huber.predict(X)
        ss_res = np.sum((y_all - y_pred) ** 2)
        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # --- Titles & labels
        reg_line = f"Linear fit (Huber) R² = {r2:.3f}"
        ttest_line = f"T-test: Δ={d_mean:.2f}, t={t_stat:.2f}, p={p_val:.3g}"
        ax.set_title(f"{ident} vs {baseline}\n{reg_line}\n{ttest_line}")
        ax.set_xlabel(f"{baseline} median n-value")
        ax.set_ylabel(f"{ident} median n-value")

        # --- Legend
        ax.legend()

        figs.append(fig)

    return figs


import numpy as np
import matplotlib.pyplot as plt
from stats_test import adjusted_ttest

def conformity_plots_from_components(df, comp_df, identifiers, baseline, mad_thresh=3.5):
    """
    Multi-panel conformity scatter plots (baseline vs each other genotype).

    Features:
      - Outliers detected by Median Absolute Deviation (MAD).
      - Outliers plotted in grey, excluded from regression fit.
      - Linear regression fit (ordinary least squares) on non-outliers.
      - R² and t-test results included in subplot titles.
      - T-test always uses ALL points (outliers included).
    """
    n = len(identifiers) - 1
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), dpi=200)
    axs = np.atleast_1d(axs).ravel()

    panel = 0
    baseline_col = f"n_value_{baseline}_median"

    for ident in identifiers:
        if ident == baseline:
            continue
        compare_col = f"n_value_{ident}_median"
        if baseline_col not in comp_df or compare_col not in comp_df:
            continue

        ax = axs[panel]
        panel += 1

        # --- Collect data
        x_all = comp_df[baseline_col].to_numpy(float)
        y_all = comp_df[compare_col].to_numpy(float)
        mask_valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        x_all, y_all = x_all[mask_valid], y_all[mask_valid]

        if len(x_all) < 2:
            ax.set_axis_off()
            continue

        # --- T-test (all data, including outliers)
        d_mean, t_stat, p_val, rho = adjusted_ttest(
            comp_df[mask_valid], baseline_col, compare_col
        )

        # --- Outlier detection (MAD)
        diffs = y_all - x_all
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        if mad == 0:
            mask_outliers = np.zeros_like(diffs, dtype=bool)
        else:
            mad_z = 0.6745 * (diffs - med) / mad
            mask_outliers = np.abs(mad_z) > mad_thresh

        # --- Plot points
        ax.scatter(x_all[~mask_outliers], y_all[~mask_outliers],
                   color="blue", alpha=0.7, label="Components")
        if np.any(mask_outliers):
            ax.scatter(x_all[mask_outliers], y_all[mask_outliers],
                       color="grey", alpha=0.5, label="Outliers")

        # --- Reference line y = x
        lims = [min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())]
        ax.plot(lims, lims, 'k--', lw=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # --- Linear regression (exclude outliers)
        if np.sum(~mask_outliers) > 1:
            slope, intercept = np.polyfit(x_all[~mask_outliers], y_all[~mask_outliers], 1)
            xx = np.linspace(min(lims), max(lims), 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color="red", lw=2, label="OLS fit")

            y_pred = slope * x_all[~mask_outliers] + intercept
            ss_res = np.sum((y_all[~mask_outliers] - y_pred) ** 2)
            ss_tot = np.sum((y_all[~mask_outliers] - np.mean(y_all[~mask_outliers])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan

        # --- Titles & labels
        reg_line = f"OLS R²={r2:.3f}" if np.isfinite(r2) else "Fit skipped"
        ttest_line = f"Δ={d_mean:.2f}, t={t_stat:.2f}, p={p_val:.3g}"
        ax.set_title(f"{ident} vs {baseline}\n{reg_line}, {ttest_line}")
        ax.set_xlabel(rf"{baseline} $n_L$")
        ax.set_ylabel(rf"{ident} $n_L$")
        ax.legend()

    # --- Remove unused panels
    for j in range(panel, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Component conformity plots (baseline = {baseline})", fontsize=16)
    fig.tight_layout()
    return fig


import numpy as np
import matplotlib.pyplot as plt
from stats_test import adjusted_ttest

def conformity_plots_from_components(df, comp_df, identifiers, baseline, mad_thresh=3.5):
    """
    Multi-panel conformity scatter plots (baseline vs each other genotype).

    Features:
      - Outliers detected by Median Absolute Deviation (MAD).
      - Outliers plotted in grey, excluded from regression fit.
      - Linear regression fit (ordinary least squares) on non-outliers.
      - R² and t-test results included in subplot titles.
      - T-test always uses ALL points (outliers included).
    """
    n = len(identifiers) - 1
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), dpi=200)
    axs = np.atleast_1d(axs).ravel()

    panel = 0
    baseline_col = f"n_value_{baseline}_median"

    for ident in identifiers:
        if ident == baseline:
            continue
        compare_col = f"n_value_{ident}_median"
        if baseline_col not in comp_df or compare_col not in comp_df:
            continue

        ax = axs[panel]
        panel += 1

        # --- Collect data
        x_all = comp_df[baseline_col].to_numpy(float)
        y_all = comp_df[compare_col].to_numpy(float)
        mask_valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        x_all, y_all = x_all[mask_valid], y_all[mask_valid]

        if len(x_all) < 2:
            ax.set_axis_off()
            continue

        # --- T-test (all data, including outliers)
        d_mean, t_stat, p_val, rho = adjusted_ttest(
            comp_df[mask_valid], baseline_col, compare_col
        )

        # --- Outlier detection (MAD)
        diffs = y_all - x_all
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        if mad == 0:
            mask_outliers = np.zeros_like(diffs, dtype=bool)
        else:
            mad_z = 0.6745 * (diffs - med) / mad
            mask_outliers = np.abs(mad_z) > mad_thresh

        # --- Plot points
        ax.scatter(x_all[~mask_outliers], y_all[~mask_outliers],
                   color="blue", alpha=0.7, label="Components")
        if np.any(mask_outliers):
            ax.scatter(x_all[mask_outliers], y_all[mask_outliers],
                       color="grey", alpha=0.5, label="Outliers")

        # --- Reference line y = x
        lims = [min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())]
        ax.plot(lims, lims, 'k--', lw=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # --- Linear regression (exclude outliers)
        if np.sum(~mask_outliers) > 1:
            slope, intercept = np.polyfit(x_all[~mask_outliers], y_all[~mask_outliers], 1)
            xx = np.linspace(min(lims), max(lims), 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color="red", lw=2, label="OLS fit")

            y_pred = slope * x_all[~mask_outliers] + intercept
            ss_res = np.sum((y_all[~mask_outliers] - y_pred) ** 2)
            ss_tot = np.sum((y_all[~mask_outliers] - np.mean(y_all[~mask_outliers])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan

        # --- Titles & labels
        reg_line = f"OLS R²={r2:.3f}" if np.isfinite(r2) else "Fit skipped"
        ttest_line = f"Δ={d_mean:.2f}, t={t_stat:.2f}, p={p_val:.3g}"
        ax.set_title(f"{ident} vs {baseline}\n{reg_line}, {ttest_line}")
        ax.set_xlabel(f"{baseline} median n-value")
        ax.set_ylabel(f"{ident} median n-value")
        ax.legend()

    # --- Remove unused panels
    for j in range(panel, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"Component conformity plots (baseline = {baseline})", fontsize=16)
    fig.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
from stats_test import adjusted_ttest

def conformity_plots_from_components(df, comp_df, identifiers, baseline, mad_thresh=3.5):
    """
    Multi-panel conformity scatter plots (baseline vs each other genotype).

    Features:
      - Outliers detected by Median Absolute Deviation (MAD).
      - Outliers plotted in grey, excluded from regression fit.
      - Linear regression fit (ordinary least squares) on non-outliers.
      - R² and t-test results included in subplot titles.
      - T-test always uses ALL points (outliers included).
      - Axis labels show n_L (subscript L).
    """
    n = len(identifiers) - 1
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), dpi=200)
    axs = np.atleast_1d(axs).ravel()

    panel = 0
    baseline_col = f"n_value_{baseline}_median"

    for ident in identifiers:
        if ident == baseline:
            continue
        compare_col = f"n_value_{ident}_median"
        if baseline_col not in comp_df or compare_col not in comp_df:
            continue

        ax = axs[panel]
        panel += 1

        # --- Collect data
        x_all = comp_df[baseline_col].to_numpy(float)
        y_all = comp_df[compare_col].to_numpy(float)
        mask_valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        x_all, y_all = x_all[mask_valid], y_all[mask_valid]

        if len(x_all) < 2:
            ax.set_axis_off()
            continue

        # --- T-test (all data, including outliers)
        d_mean, t_stat, p_val, rho = adjusted_ttest(
            comp_df[mask_valid], baseline_col, compare_col
        )

        # --- Outlier detection (MAD)
        diffs = y_all - x_all
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        if mad == 0:
            mask_outliers = np.zeros_like(diffs, dtype=bool)
        else:
            mad_z = 0.6745 * (diffs - med) / mad
            mask_outliers = np.abs(mad_z) > mad_thresh

        # --- Plot points
        ax.scatter(x_all[~mask_outliers], y_all[~mask_outliers],
                   color="blue", alpha=0.7, label="Components")
        if np.any(mask_outliers):
            ax.scatter(x_all[mask_outliers], y_all[mask_outliers],
                       color="grey", alpha=0.5, label="Outliers")

        # --- Reference line y = x
        lims = [min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())]
        ax.plot(lims, lims, 'k--', lw=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # --- Linear regression (exclude outliers)
        if np.sum(~mask_outliers) > 1:
            slope, intercept = np.polyfit(x_all[~mask_outliers], y_all[~mask_outliers], 1)
            xx = np.linspace(min(lims), max(lims), 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color="red", lw=2, label="OLS fit")

            y_pred = slope * x_all[~mask_outliers] + intercept
            ss_res = np.sum((y_all[~mask_outliers] - y_pred) ** 2)
            ss_tot = np.sum((y_all[~mask_outliers] - np.mean(y_all[~mask_outliers])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan

        # --- Titles & labels
        reg_line = f"OLS R²={r2:.3f}" if np.isfinite(r2) else "Fit skipped"
        ttest_line = fr"$\bar{{d}}$={d_mean:.2f}, t={t_stat:.2f}, p={p_val:.3g}"
        ax.set_title(f"{ident} vs {baseline}\n{reg_line}, {ttest_line}")
        ax.set_xlabel(rf"{baseline} $n_L$")
        ax.set_ylabel(rf"{ident} $n_L$")
        ax.legend()

    # --- Remove unused panels
    for j in range(panel, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(rf"Component conformity plots (baseline = {baseline} $n_L$)", fontsize=16)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# (6d) Bar plots of component n-values
# ------------------------------------------------------------
def linear_algebra_plot_from_tidy(
    df: pd.DataFrame,
    identifiers: list[str],
    y_max: float | None = None,
    title: str | None = None,
    num_jackknifes: int = 1000,
    monte_carlo_per_jackknife: int = 50,
    seed: int | None = 0,
    restrict_classes=('PE','PC','PI','PS','PG','PA','DG','TG','LPE','LPC','LPA','LPG','LPI','LPS'),
    ax=None
):
    """
    Estimate component n-values (via jackknife), then filter to only
    components present in lipids shared across all genotypes.
    Finally, plot bar chart of median + CI.
    """
    # (D1) Drop uninformative CI=0 rows
    df = df[~((df["ci_upper"] == 0) & (df["ci_lower"] == 0))]

    # (D2) jackknife estimation
    comp_df, sdf0, sims_by_ident = jackknife_component_n_values_from_tidy(
    df=df,
    identifiers=['APOE2', 'APOE3', 'APOE4'],
    num_jackknifes=1000,
    num_simulations=50,
    return_sims=True,
    seed=0
)

    figs = plot_fatty_acid_histograms_separate(sims_by_ident)
    for fa, fig in figs.items():
        fig.savefig(f"{fa.replace(':','-')}_overlay_hist.svg", dpi=300)


    # (D3) Determine lipids shared across all genotypes
    key_col = 'dedup_key' if 'dedup_key' in sdf0.columns else '__lipid_key'
    if key_col not in sdf0.columns:
        sdf0[key_col] = (
            sdf0['Alignment ID'].astype(str)
                .str.replace(r'\(\s*[Mm][^)]*\)', '', regex=True)
                .str.replace('/0:0', '', regex=False)
                .str.split('|').str[-1].str.strip()
        )
    present = sdf0[[key_col, 'genotype']].drop_duplicates()
    common_keys = set(
        present.groupby(key_col)['genotype'].nunique()
               .pipe(lambda s: s[s == len(identifiers)]).index
    )

    # (D4) Restrict to common lipids
    sdf_common = sdf0[sdf0[key_col].isin(common_keys)].copy()
    if 'Components' not in sdf_common.columns:
        raise ValueError("Expected 'Components' column missing. Make sure jackknife was run first.")
    sdf_common = sdf_common[sdf_common['Components'].notna()]
    components_from_common_lipids = {c for comps in sdf_common['Components'] for c in comps}
    comp_df = comp_df[comp_df['Component'].isin(components_from_common_lipids)].reset_index(drop=True)

    # (D5) Handle empty case
    if comp_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No components from lipids common to all genotypes.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, comp_df

    # (D6) Build bar plot
    n_comps = len(comp_df)
    fig_width = max(12, n_comps * 0.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)
    bar_w = 0.8 / max(1, len(identifiers))

    for i, ident in enumerate(identifiers):
        means = pd.to_numeric(comp_df[f'n_value_{ident}_median'], errors='coerce').to_numpy(float)
        q_lo  = pd.to_numeric(comp_df[f'n_value_{ident}_lower_quantile'], errors='coerce').to_numpy(float)
        q_hi  = pd.to_numeric(comp_df[f'n_value_{ident}_upper_quantile'], errors='coerce').to_numpy(float)

        lo = np.clip(means - q_lo, 0, None)
        hi_raw = np.clip(q_hi - means, 0, None)
        hi = np.minimum(hi_raw, np.maximum(0.0, 100.0 - means))

        x_pos = idx + (i - (len(identifiers)-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, yerr=[lo, hi], capsize=3, label=ident)

    ax.set_xlabel("Component")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_xticklabels(comp_df["Component"], rotation=90, ha="center")
    if title: ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)  # <-- Add this line to force lower y-limit to 0

    # (D7) Save results table
    save_table = filedialog.asksaveasfilename(
        title="Save component n-value table as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if save_table:
        mean_cols = [f'n_value_{i}_median' for i in identifiers]
        means_flat = comp_df[mean_cols].to_numpy(float).flatten()
        median = np.median(means_flat)
        mad = np.median(np.abs(means_flat - median))
        if mad == 0:
            comp_df['MAD_z'] = 0.0
        else:
            row_z = np.abs(comp_df[mean_cols] - median) / mad
            comp_df['MAD_z'] = row_z.max(axis=1)
        comp_df.to_csv(save_table, index=False)
        print("Table (filtered to common-lipid components) saved to:", save_table)

    return fig, ax, comp_df


def linear_algebra_plot_from_tidy(
    df: pd.DataFrame,
    identifiers: list[str],
    y_max: float | None = None,
    title: str | None = None,
    num_jackknifes: int = 1000,
    monte_carlo_per_jackknife: int = 50,
    seed: int | None = 0,
    restrict_classes=('PE','PC','PI','PS','PG','PA','DG','TG','LPE','LPC','LPA','LPG','LPI','LPS'),
    ax=None
):
    """
    Estimate component n-values (via jackknife), then filter to only
    components present in lipids shared across all genotypes.
    Finally, plot bar chart of median + CI.

    NOTE: This implementation does NOT mutate comp_df for plotting.
          All plotting sanitization is performed on a local copy (comp_plot).
    """
    # (D1) Drop uninformative CI=0 rows (input df)
    df = df[~((df["ci_upper"] == 0) & (df["ci_lower"] == 0))]
    print('Length of DF')
    print(len(df))

    # (D2) jackknife estimation (unchanged)
    comp_df, sdf0, sims_by_ident = jackknife_component_n_values_from_tidy(
        df=df,
        identifiers=['APOE2', 'APOE3', 'APOE4'],
        num_jackknifes=num_jackknifes,
        num_simulations=monte_carlo_per_jackknife,
        return_sims=True,
        seed=seed
    )

    # optional saving of per-FA histograms (unchanged)
    figs = plot_fatty_acid_histograms_separate(sims_by_ident)
    for fa, fig in figs.items():
        try:
            fig.savefig(f"{fa.replace(':','-')}_overlay_hist.svg", dpi=300)
        except Exception:
            pass

    # (D3) Determine lipids shared across all genotypes
    key_col = 'dedup_key' if 'dedup_key' in sdf0.columns else '__lipid_key'
    if key_col not in sdf0.columns:
        sdf0[key_col] = (
            sdf0['Alignment ID'].astype(str)
                .str.replace(r'\(\s*[Mm][^)]*\)', '', regex=True)
                .str.replace('/0:0', '', regex=False)
                .str.split('|').str[-1].str.strip()
        )
    present = sdf0[[key_col, 'genotype']].drop_duplicates()
    common_keys = set(
        present.groupby(key_col)['genotype'].nunique()
               .pipe(lambda s: s[s == len(identifiers)]).index
    )

    # (D4) Restrict to common lipids and build component list
    sdf_common = sdf0[sdf0[key_col].isin(common_keys)].copy()
    if 'Components' not in sdf_common.columns:
        raise ValueError("Expected 'Components' column missing. Make sure jackknife was run first.")
    sdf_common = sdf_common[sdf_common['Components'].notna()]
    components_from_common_lipids = {c for comps in sdf_common['Components'] for c in comps}
    comp_df = comp_df[comp_df['Component'].isin(components_from_common_lipids)].reset_index(drop=True)

    # (D5) Handle empty case
    if comp_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No components from lipids common to all genotypes.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, comp_df

    # ---------------------------
    # (D6) Build bar plot (PLOTTING-ONLY: do not mutate comp_df)
    # ---------------------------
    # make a lightweight plotting copy (we will not write back to comp_df)
    comp_plot = comp_df.copy()

    # Ensure required cols exist in comp_plot (error early if not)
    mean_cols = [f'n_value_{i}_median' for i in identifiers]
    lo_cols   = [f'n_value_{i}_lower_quantile' for i in identifiers]
    hi_cols   = [f'n_value_{i}_upper_quantile' for i in identifiers]
    for col in mean_cols + lo_cols + hi_cols:
        if col not in comp_plot.columns:
            raise KeyError(f"Expected column '{col}' missing from comp_df")

    # figure sizing & optional limit on number of components displayed (non-destructive)
    n_comps = len(comp_plot)
    max_components_to_show = 200  # change if you want to show more
    if n_comps > max_components_to_show:
        # choose top components by max median across genotypes (display-only)
        comp_plot['max_median_display'] = comp_plot[mean_cols].max(axis=1)
        comp_plot = comp_plot.sort_values('max_median_display', ascending=False).head(max_components_to_show).reset_index(drop=True)
        n_comps = len(comp_plot)

    fig_width = max(12, n_comps * 0.45)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)
    bar_w = 0.8 / max(1, len(identifiers))

    # Plot using local numpy arrays only (comp_plot is not written back to comp_df)
    overall_top = 0.0
    for i, ident in enumerate(identifiers):
        med_col = f'n_value_{ident}_median'
        lo_col  = f'n_value_{ident}_lower_quantile'
        hi_col  = f'n_value_{ident}_upper_quantile'

        # read-to-local and coerce numeric for plotting only
        means = pd.to_numeric(comp_plot[med_col], errors='coerce').to_numpy(dtype=float)
        q_lo  = pd.to_numeric(comp_plot[lo_col],  errors='coerce').to_numpy(dtype=float)
        q_hi  = pd.to_numeric(comp_plot[hi_col],  errors='coerce').to_numpy(dtype=float)

        # display-only sanitization (no writes to comp_df):
        means = np.nan_to_num(means, nan=0.0)
        q_lo  = np.nan_to_num(q_lo,  nan=0.0)
        q_hi  = np.nan_to_num(q_hi,  nan=0.0)

        # distances for asymmetric errorbars (non-negative)
        lo = np.clip(means - q_lo, 0.0, None)
        lo = np.minimum(lo, means)   # ensure downward error doesn't push below 0
        hi = np.clip(q_hi - means, 0.0, None)

        # update overall_top from plotted arrays (means + hi)
        if np.isfinite((means + hi).max()):
            overall_top = max(overall_top, float(np.nanmax(means + hi)))

        # x positions and bar drawing
        x_pos = idx + (i - (len(identifiers)-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, label=ident,
               edgecolor='black', linewidth=0.25, zorder=3, alpha=0.95)

        # draw errorbars separately (ensures correct handling of asymmetric yerr)
        ax.errorbar(x_pos, means, yerr=(lo, hi), fmt='none',
                    ecolor='#555555', elinewidth=1.0, capsize=3, alpha=0.85, zorder=2)

    # formatting
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_xticklabels(comp_plot["Component"], rotation=90, ha="center", fontsize=8)
    if title:
        ax.set_title(title)

    # set final y-limits AFTER plotting so autoscaling won't override them
    if y_max is not None:
        top = float(y_max)
    else:
        # fallback: use overall_top computed from means+hi, else small positive so bars are visible
        top = overall_top if overall_top > 0 else 0.06

    ax.set_ylim(0, top * 1.12)
    ax.set_autoscale_on(False)  # lock limits so following ops don't change them

    # legend dedupe and layout
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    fig.tight_layout()

    # (D7) Save results table (keeps original behavior; comp_df is not modified)
    save_table = filedialog.asksaveasfilename(
        title="Save component n-value table as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if save_table:
        mean_cols = [f'n_value_{i}_median' for i in identifiers]
        means_flat = comp_df[mean_cols].to_numpy(float).flatten()
        median = np.median(means_flat)
        mad = np.median(np.abs(means_flat - median))
        if mad == 0:
            comp_df['MAD_z'] = 0.0
        else:
            row_z = np.abs(comp_df[mean_cols] - median) / mad
            comp_df['MAD_z'] = row_z.max(axis=1)
        comp_df.to_csv(save_table, index=False)
        print("Table (filtered to common-lipid components) saved to:", save_table)

    return fig, ax, comp_df


# ------------------------------------------------------------
# (6e) Special comparison: palmitate vs stearate
# ------------------------------------------------------------
def plot_palmitate_stearate_comparison(
    comp_df: pd.DataFrame,
    identifiers: list[str],
    ax=None
):
    """
    Bar plot for palmitate (16:0) and stearate (18:0) only,
    using already-calculated comp_df (no recalculation).
    Shows jackknife estimates per genotype plus literature
    reference bars (Lee et al. 1994 and theoretical maxima).
    """
    subset = comp_df[comp_df['Component'].isin(['16:0', '18:0', '20:4'])].copy()

    # Require both present
    if not set(['16:0', '18:0', '20:4']).issubset(set(subset['Component'])):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "16:0 and 18:0 not present in comp_df.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    n_comps = len(subset)
    fig_width = max(8, n_comps * 2.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)

    # total number of bar groups: genotypes + 2 refs
    n_groups = len(identifiers) + 2
    bar_w = 0.8 / max(1, n_groups)

    # Plot jackknife means + CI per genotype
    for i, ident in enumerate(identifiers):
        means = pd.to_numeric(subset[f'n_value_{ident}_median'], errors='coerce').to_numpy(float)
        q_lo  = pd.to_numeric(subset[f'n_value_{ident}_lower_quantile'], errors='coerce').to_numpy(float)
        q_hi  = pd.to_numeric(subset[f'n_value_{ident}_upper_quantile'], errors='coerce').to_numpy(float)

        lo = np.clip(means - q_lo, 0, None)
        hi_raw = np.clip(q_hi - means, 0, None)
        hi = np.minimum(hi_raw, np.maximum(0.0, 100.0 - means))

        x_pos = idx + (i - (n_groups-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, yerr=[lo, hi], capsize=3, label=ident)

    # Add reference bars
    ref_vals = {
        '16:0': {'Lee1994': 17, 'Theoretical': 21},
        '18:0': {'Lee1994': 20, 'Theoretical': 24},
        '20:4': {'Carson2017': 6, 'Theoretical':7}
    }
    ref_colors = {'Lee1994': 'magenta', 'Theoretical': 'dimgray', 'Carson2017': '#6A3D9A'}

    for j, comp in enumerate(subset['Component']):
        for k, (label, val) in enumerate(ref_vals[comp].items()):
            offset = len(identifiers) + k  # place refs after genotypes
            x_pos = j + (offset - (n_groups-1)/2) * bar_w
            ax.bar(x_pos, val, bar_w, color=ref_colors[label], label=label if j == 0 else None)

    # Format axes
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_ylim(0,29)
    ax.set_xticklabels(subset["Component"], rotation=0, ha="center")
    ax.legend()
    ax.set_title("Palmitate (16:0), Stearate (18:0) and Arachidonic acid (20:4) compariosn")

    return fig, ax



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_theoretical_vs_empirical(
    comp_df: pd.DataFrame,
    identifiers: list[str],
    precursor_type: str = "de novo",
    ax=None,
    title: str = "Theoretical vs Empirical n-values"
):
    """
    Compare empirical jackknife n-values (comp_df) against theoretical
    predictions (theory_nL.csv) using a grouped bar chart.

    Parameters
    ----------
    comp_df : DataFrame
        Must have columns: 'Component', n_value_{ident}_median,
        n_value_{ident}_lower_quantile, n_value_{ident}_upper_quantile.
    identifiers : list[str]
        Genotypes present in comp_df.
    precursor_type : str, default="de novo"
        Which predictions to use: "de novo" or "dietary".
    ax : matplotlib Axes, optional
    title : str
    """

    # --- Load theory_nL.csv from same directory ---
    csv_path = os.path.join(os.path.dirname(__file__), "theory_nL.csv")
    theory_df = pd.read_csv(csv_path)

    # --- Choose columns based on precursor type ---
    prefix = " (de novo precursor)" if precursor_type == "de novo" else " (dietary precursor)"
    avg_col = f"n_avg{prefix}"
    lo_col = f"estimated_n_val_low_end{prefix}"
    hi_col = f"estimated_n_val_high_end{prefix}"

    # --- Align theoretical predictions to empirical data ---
    merged = comp_df.merge(
        theory_df,
        left_on="Component",       # from comp_df
        right_on="FA notation",    # from theory_nL.csv
        how="inner"
    )

    if merged.empty:
        raise ValueError("No overlap between comp_df.Component and theory_nL.csv['FA notation']")

    n_comps = len(merged)
    n_groups = len(identifiers) + 1  # empirical + theoretical
    fig_width = max(12, n_comps * 0.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)
    bar_w = 0.8 / n_groups

    # --- Plot empirical jackknife estimates ---
    for i, ident in enumerate(identifiers):
        means = pd.to_numeric(merged[f'n_value_{ident}_median'], errors="coerce").to_numpy(float)
        q_lo  = pd.to_numeric(merged[f'n_value_{ident}_lower_quantile'], errors="coerce").to_numpy(float)
        q_hi  = pd.to_numeric(merged[f'n_value_{ident}_upper_quantile'], errors="coerce").to_numpy(float)

        lo = np.clip(means - q_lo, 0, None)
        hi = np.clip(q_hi - means, 0, None)

        x_pos = idx + (i - (n_groups-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, yerr=[lo, hi],
               capsize=3, label=f"Empirical {ident}")

    # --- Plot theoretical predictions ---
    theory_means = pd.to_numeric(merged[avg_col], errors="coerce").to_numpy(float)
    theory_low   = pd.to_numeric(merged[lo_col], errors="coerce").to_numpy(float)
    theory_high  = pd.to_numeric(merged[hi_col], errors="coerce").to_numpy(float)

    # Error bars = absolute distance from mean (force non-negative)
    theory_lo = np.clip(theory_means - theory_low, 0, None)
    theory_hi = np.clip(theory_high - theory_means, 0, None)

    x_pos = idx + (len(identifiers) - (n_groups-1)/2) * bar_w
    ax.bar(x_pos, theory_means, bar_w, yerr=[theory_lo, theory_hi],
           capsize=3, color="dimgray", alpha=0.7, label=f"Theoretical ({precursor_type})")

    # --- Formatting ---
    ax.set_xlabel("Fatty Acid")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_xticklabels(merged["Component"], rotation=90, ha="center")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)

    return fig, ax, merged

def linear_algebra_plot_from_tidy(
    df: pd.DataFrame,
    identifiers: list[str],
    y_max: float | None = None,
    title: str | None = None,
    num_jackknifes: int = 1000,
    monte_carlo_per_jackknife: int = 50,
    seed: int | None = 0,
    restrict_classes=('PE','PC','PI','PS','PG','PA','DG','TG','LPE','LPC','LPA','LPG','LPI','LPS'),
    ax=None
):
    """
    Estimate component n-values (via jackknife), then filter to only
    components present in lipids shared across all genotypes.
    Finally, plot bar chart of median + CI.

    This implementation is plotting-only non-destructive: comp_df is never
    mutated for sanitization purposes; local display arrays are used instead.
    """
    # (D1) Drop uninformative CI=0 rows
    df = df[~((df["ci_upper"] == 0) & (df["ci_lower"] == 0))]

    # (D2) jackknife estimation (use provided identifiers)
    comp_df, sdf0, sims_by_ident = jackknife_component_n_values_from_tidy(
        df=df,
        identifiers=identifiers,
        num_jackknifes=num_jackknifes,
        num_simulations=monte_carlo_per_jackknife,
        return_sims=True,
        seed=seed
    )

    # optional FA histograms (best-effort; ignore save errors)
    figs = plot_fatty_acid_histograms_separate(sims_by_ident)
    for fa, fig in figs.items():
        try:
            fig.savefig(f"{fa.replace(':','-')}_overlay_hist.svg", dpi=300)
        except Exception:
            pass

    # (D3) Determine lipids shared across all genotypes
    key_col = 'dedup_key' if 'dedup_key' in sdf0.columns else '__lipid_key'
    if key_col not in sdf0.columns:
        sdf0[key_col] = (
            sdf0['Alignment ID'].astype(str)
                .str.replace(r'\(\s*[Mm][^)]*\)', '', regex=True)
                .str.replace('/0:0', '', regex=False)
                .str.split('|').str[-1].str.strip()
        )
    present = sdf0[[key_col, 'genotype']].drop_duplicates()
    common_keys = set(
        present.groupby(key_col)['genotype'].nunique()
               .pipe(lambda s: s[s == len(identifiers)]).index
    )

    # (D4) Restrict to common lipids
    sdf_common = sdf0[sdf0[key_col].isin(common_keys)].copy()
    if 'Components' not in sdf_common.columns:
        raise ValueError("Expected 'Components' column missing. Make sure jackknife was run first.")
    sdf_common = sdf_common[sdf_common['Components'].notna()]
    components_from_common_lipids = {c for comps in sdf_common['Components'] for c in comps}
    comp_df = comp_df[comp_df['Component'].isin(components_from_common_lipids)].reset_index(drop=True)

    # (D5) Handle empty case
    if comp_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No components from lipids common to all genotypes.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, comp_df

    # ----------------------------
    # (D6) Build bar plot  -- DROP-IN (replication of working function style)
    # ----------------------------
    # operate on a plotting copy (do NOT mutate comp_df)
    comp_plot = comp_df.copy()

    # Validate expected columns exist
    mean_cols = [f'n_value_{i}_median' for i in identifiers]
    lo_cols   = [f'n_value_{i}_lower_quantile' for i in identifiers]
    hi_cols   = [f'n_value_{i}_upper_quantile' for i in identifiers]
    for col in mean_cols + lo_cols + hi_cols:
        if col not in comp_plot.columns:
            raise KeyError(f"Expected column '{col}' missing from comp_df")

    # If too many components, display top-K by max median for readability (display-only)
    n_comps = len(comp_plot)
    max_display = 200
    if n_comps > max_display:
        comp_plot['max_median_display'] = comp_plot[mean_cols].max(axis=1)
        comp_plot = comp_plot.sort_values('max_median_display', ascending=False).head(max_display).reset_index(drop=True)
        n_comps = len(comp_plot)

    # Match working function grouping: include an extra slot for theoretical bars if needed
    n_groups = len(identifiers) + 1
    fig_width = max(12, n_comps * 0.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)
    bar_w = 0.8 / n_groups

    # plotting-only arrays and robust sanitization (do not write back to comp_df)
    overall_top = 0.0
    for i, ident in enumerate(identifiers):
        med_col = f'n_value_{ident}_median'
        lo_col  = f'n_value_{ident}_lower_quantile'
        hi_col  = f'n_value_{ident}_upper_quantile'

        # read & coerce to local numeric arrays
        means = pd.to_numeric(comp_plot[med_col], errors='coerce').to_numpy(dtype=float)
        q_lo  = pd.to_numeric(comp_plot[lo_col],  errors='coerce').to_numpy(dtype=float)
        q_hi  = pd.to_numeric(comp_plot[hi_col],  errors='coerce').to_numpy(dtype=float)

        # display-only sanitization: NaN/inf -> 0 (no mutation of comp_plot/comp_df)
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        q_lo  = np.nan_to_num(q_lo,  nan=0.0, posinf=0.0, neginf=0.0)
        q_hi  = np.nan_to_num(q_hi,  nan=0.0, posinf=0.0, neginf=0.0)

        # compute yerr distances using the working-function pattern
        lo = means - q_lo
        hi = q_hi - means

        # robustly enforce non-negativity (strip tiny negative rounding errors)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        lo = np.clip(lo, 0.0, None)
        hi = np.clip(hi, 0.0, None)
        eps = 1e-12
        lo[np.abs(lo) < eps] = 0.0
        hi[np.abs(hi) < eps] = 0.0

        # optional debug (uncomment to help debugging without changing data)
        # if (lo < 0).any() or (hi < 0).any():
        #     print(f"DEBUG negative yerr for {ident}: lo.min={lo.min()}, hi.min={hi.min()}")

        # draw bars exactly like the working function
        x_pos = idx + (i - (n_groups-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, yerr=[lo, hi], capsize=3, label=f"Empirical {ident}")

        # update overall_top so we can set a reasonable y-lim AFTER plotting
        if np.isfinite((means + hi).max()):
            overall_top = max(overall_top, float(np.nanmax(means + hi)))

    # formatting similar to working function
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_xticklabels(comp_plot["Component"], rotation=90, ha="center")
    if title:
        ax.set_title(title)
    ax.legend()

    # set final y-limits AFTER plotting (so autoscale won't override) and lock them
    if y_max is not None:
        top = float(y_max)
    else:
        top = overall_top if overall_top > 0 else 0.06

    ax.set_ylim(0, top * 1.12)
    ax.set_autoscale_on(False)

    fig.tight_layout()

    # (D7) Save results table (preserve original behavior; comp_df unchanged)
    save_table = filedialog.asksaveasfilename(
        title="Save component n-value table as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if save_table:
        mean_cols = [f'n_value_{i}_median' for i in identifiers]
        means_flat = comp_df[mean_cols].to_numpy(float).flatten()
        median = np.median(means_flat)
        mad = np.median(np.abs(means_flat - median))
        if mad == 0:
            comp_df['MAD_z'] = 0.0
        else:
            row_z = np.abs(comp_df[mean_cols] - median) / mad
            comp_df['MAD_z'] = row_z.max(axis=1)
        comp_df.to_csv(save_table, index=False)
        print("Table (filtered to common-lipid components) saved to:", save_table)

    return fig, ax, comp_df


def linear_algebra_plot_from_tidy(
    df: pd.DataFrame,
    identifiers: list[str],
    y_max: float | None = None,
    title: str | None = None,
    num_jackknifes: int = 1000,
    monte_carlo_per_jackknife: int = 50,
    seed: int | None = 0,
    restrict_classes=('PE','PC','PI','PS','PG','PA','DG','TG','LPE','LPC','LPA','LPG','LPI','LPS'),
    ax=None
):
    """
    Estimate component n-values (via jackknife), then filter to only
    components present in lipids shared across all genotypes.
    Finally, plot bar chart of median + CI.

    This version additionally removes any component for which any genotype's
    confidence-interval width (upper_quantile - lower_quantile) > 100.
    """
    
    print(len(df))
    print(df.head())
    # (D1) Drop uninformative CI=0 rows
    df = df[~((df["ci_upper"] == 0) & (df["ci_lower"] == 0))]
    print(len(df))
    print(df.head())

    # (D2) jackknife estimation (use provided identifiers)
    comp_df, sdf0, sims_by_ident = jackknife_component_n_values_from_tidy(
        df=df,
        identifiers=identifiers,
        num_jackknifes=num_jackknifes,
        num_simulations=monte_carlo_per_jackknife,
        return_sims=True,
        seed=seed
    )
    
    print(len(comp_df))

    # optional FA histograms (best-effort; ignore save errors)
    figs = plot_fatty_acid_histograms_separate(sims_by_ident)
    for fa, fig in figs.items():
        try:
            fig.savefig(f"{fa.replace(':','-')}_overlay_hist.svg", dpi=300)
        except Exception:
            pass

    # (D3) Determine lipids shared across all genotypes
    key_col = 'dedup_key' if 'dedup_key' in sdf0.columns else '__lipid_key'
    if key_col not in sdf0.columns:
        sdf0[key_col] = (
            sdf0['Alignment ID'].astype(str)
                .str.replace(r'\(\s*[Mm][^)]*\)', '', regex=True)
                .str.replace('/0:0', '', regex=False)
                .str.split('|').str[-1].str.strip()
        )
    present = sdf0[[key_col, 'genotype']].drop_duplicates()
    common_keys = set(
        present.groupby(key_col)['genotype'].nunique()
               .pipe(lambda s: s[s == len(identifiers)]).index
    )

    # (D4) Restrict to common lipids
    sdf_common = sdf0[sdf0[key_col].isin(common_keys)].copy()
    if 'Components' not in sdf_common.columns:
        raise ValueError("Expected 'Components' column missing. Make sure jackknife was run first.")
    sdf_common = sdf_common[sdf_common['Components'].notna()]
    components_from_common_lipids = {c for comps in sdf_common['Components'] for c in comps}
    comp_df = comp_df[comp_df['Component'].isin(components_from_common_lipids)].reset_index(drop=True)

    # (NEW) Remove components with CI width > 100 for any genotype
    # compute widths per genotype (display-safe numeric coercion)
    width_flags = []
    for ident in identifiers:
        lo_col = f'n_value_{ident}_lower_quantile'
        hi_col = f'n_value_{ident}_upper_quantile'
        if lo_col not in comp_df.columns or hi_col not in comp_df.columns:
            # if expected columns are missing, skip width filtering for that identifier
            width_flags.append(np.zeros(len(comp_df), dtype=bool))
            continue
        lo_vals = pd.to_numeric(comp_df[lo_col], errors='coerce').to_numpy(dtype=float)
        hi_vals = pd.to_numeric(comp_df[hi_col], errors='coerce').to_numpy(dtype=float)
        # compute width, treat NaN as large so they get removed conservatively (optional)
        widths = np.nan_to_num(hi_vals - lo_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)
        width_flags.append(widths > 100.0)

    if width_flags:
        # combine flags: remove component if any genotype had width > 100
        remove_mask = np.logical_or.reduce(width_flags)
        n_removed = int(remove_mask.sum())
        if n_removed > 0:
            print(f"Removing {n_removed} components with CI width > 100 for at least one genotype.")
            comp_df = comp_df.loc[~remove_mask].reset_index(drop=True)
        else:
            # no removals
            pass

    # (D5) Handle empty case after filtering
    if comp_df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No components from lipids common to all genotypes (after filtering).",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, comp_df

    # ----------------------------
    # (D6) Build bar plot  -- DROP-IN (replication of working function style)
    # ----------------------------
    # operate on a plotting copy (do NOT mutate comp_df further)
    comp_plot = comp_df.copy()

    # Validate expected columns exist
    mean_cols = [f'n_value_{i}_median' for i in identifiers]
    lo_cols   = [f'n_value_{i}_lower_quantile' for i in identifiers]
    hi_cols   = [f'n_value_{i}_upper_quantile' for i in identifiers]
    for col in mean_cols + lo_cols + hi_cols:
        if col not in comp_plot.columns:
            raise KeyError(f"Expected column '{col}' missing from comp_df")

    # If too many components, display top-K by max median for readability (display-only)
    n_comps = len(comp_plot)
    max_display = 200
    if n_comps > max_display:
        comp_plot['max_median_display'] = comp_plot[mean_cols].max(axis=1)
        comp_plot = comp_plot.sort_values('max_median_display', ascending=False).head(max_display).reset_index(drop=True)
        n_comps = len(comp_plot)

    # Match working function grouping: include an extra slot for theoretical bars if needed
    n_groups = len(identifiers) + 1
    fig_width = max(12, n_comps * 0.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    else:
        fig = ax.figure

    idx = np.arange(n_comps)
    bar_w = 0.8 / n_groups

    # plotting-only arrays and robust sanitization (do not write back to comp_df)
    overall_top = 0.0
    for i, ident in enumerate(identifiers):
        med_col = f'n_value_{ident}_median'
        lo_col  = f'n_value_{ident}_lower_quantile'
        hi_col  = f'n_value_{ident}_upper_quantile'

        # read & coerce to local numeric arrays
        means = pd.to_numeric(comp_plot[med_col], errors='coerce').to_numpy(dtype=float)
        q_lo  = pd.to_numeric(comp_plot[lo_col],  errors='coerce').to_numpy(dtype=float)
        q_hi  = pd.to_numeric(comp_plot[hi_col],  errors='coerce').to_numpy(dtype=float)

        # display-only sanitization: NaN/inf -> 0 (no mutation of comp_plot/comp_df)
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        q_lo  = np.nan_to_num(q_lo,  nan=0.0, posinf=0.0, neginf=0.0)
        q_hi  = np.nan_to_num(q_hi,  nan=0.0, posinf=0.0, neginf=0.0)

        # compute yerr distances using the working-function pattern
        lo = means - q_lo
        hi = q_hi - means

        # robustly enforce non-negativity (strip tiny negative rounding errors)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        lo = np.clip(lo, 0.0, None)
        hi = np.clip(hi, 0.0, None)
        eps = 1e-12
        lo[np.abs(lo) < eps] = 0.0
        hi[np.abs(hi) < eps] = 0.0

        x_pos = idx + (i - (n_groups-1)/2) * bar_w
        ax.bar(x_pos, means, bar_w, yerr=[lo, hi], capsize=3, label=f"Empirical {ident}")

        # update overall_top so we can set a reasonable y-lim AFTER plotting
        if np.isfinite((means + hi).max()):
            overall_top = max(overall_top, float(np.nanmax(means + hi)))

    # formatting similar to working function
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$n_L$")
    ax.set_xticks(idx)
    ax.set_xticklabels(comp_plot["Component"], rotation=90, ha="center")
    if title:
        ax.set_title(title)
    ax.legend()

    # set final y-limits AFTER plotting (so autoscale won't override) and lock them
    if y_max is not None:
        top = float(y_max)
    else:
        top = overall_top if overall_top > 0 else 0.06

    ax.set_ylim(0, top * 1.12)
    ax.set_autoscale_on(False)

    fig.tight_layout()

    # (D7) Save results table (preserve original behavior; comp_df is now filtered)
    save_table = filedialog.asksaveasfilename(
        title="Save component n-value table as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if save_table:
        mean_cols = [f'n_value_{i}_median' for i in identifiers]
        means_flat = comp_df[mean_cols].to_numpy(float).flatten()
        median = np.median(means_flat)
        mad = np.median(np.abs(means_flat - median))
        if mad == 0:
            comp_df['MAD_z'] = 0.0
        else:
            row_z = np.abs(comp_df[mean_cols] - median) / mad
            comp_df['MAD_z'] = row_z.max(axis=1)
        comp_df.to_csv(save_table, index=False)
        print("Table (filtered to common-lipid components) saved to:", save_table)

    return fig, ax, comp_df


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parsing import _components_from_row  # Uses your existing lipid parser

def plot_theoretical_vs_empirical(
    comp_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    identifiers: list[str],
    theoretical_dict: dict[str, float] | None = None
):
    
    Compare empirical component nL values to theoretical expectations,
    applying a correction for dietary dilution based on lipid asymptote (A).

    For each component, the mean asymptote is computed from all lipids that
    contain that component (unweighted average).

    Parameters
    ----------
    comp_df : DataFrame
        Component-level output from linear_algebra_plot_from_tidy().
        Must include columns ['component', 'n_value_median'].

    regression_df : DataFrame
        Regression matrix containing rows where 'metric' == 'Abundance asymptote'.
        Must include columns ['metric', 'value', 'Lipid Unique Identifier', 'Ontology'].

    identifiers : list[str]
        Genotypes or experimental condition labels (for legend titles, if needed).

    theoretical_dict : dict[str, float], optional
        Mapping from component name → theoretical nL value.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    
    df = comp_df.copy()

    # --- Extract asymptote rows (Abundance asymptote only) ---
    asymptote_df = regression_df.loc[
        regression_df["metric"] == "Abundance asymptote"
    ].copy()
    asymptote_df = asymptote_df.rename(columns={"value": "A"})

    # --- Parse components from each lipid ---
    component_records = []
    for _, row in asymptote_df.iterrows():
        lipid_id = row.get("Lipid Unique Identifier", row.get("Alignment ID"))
        ontology = row.get("Ontology", None)
        if pd.isna(lipid_id) or pd.isna(ontology):
            continue
        try:
            components = _components_from_row(ontology, lipid_id)
        except Exception:
            components = []
        for c in components:
            component_records.append({"component": c, "A": row["A"]})

    # --- Build dataframe of component–asymptote mappings ---
    component_df = pd.DataFrame(component_records)
    if component_df.empty:
        raise ValueError("No component–asymptote mappings could be parsed.")

    # --- Compute mean asymptote per component (unweighted) ---
    asym_means = (
        component_df.groupby("component", dropna=False)["A"]
        .mean()
        .rename("A_mean")
    )

    # --- Merge mean asymptote with component-level nL results ---
    df = df.merge(asym_means, on="component", how="left")

    # --- Apply correction: divide by mean asymptote ---
    df["n_value_corrected"] = df["n_value_median"] / df["A_mean"]
    df["n_value_corrected"] = df["n_value_corrected"].replace([np.inf, -np.inf], np.nan)

    # --- Add theoretical values (if provided) ---
    if theoretical_dict:
        df["n_value_theoretical"] = df["component"].map(theoretical_dict)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["n_value_median"], width, label="Empirical", color="skyblue")
    ax.bar(x, df["n_value_corrected"], width, label="Asymptote-corrected", color="steelblue")

    if "n_value_theoretical" in df:
        ax.bar(x + width, df["n_value_theoretical"], width,
               label="Theoretical", color="lightgray")

    ax.set_xticks(x)
    ax.set_xticklabels(df["component"], rotation=45, ha="right")
    ax.set_ylabel(r"$n_L$")
    ax.set_title("Empirical vs Asymptote-corrected Component $n_L$")
    ax.legend()
    plt.tight_layout()
    return fig, ax

"""

