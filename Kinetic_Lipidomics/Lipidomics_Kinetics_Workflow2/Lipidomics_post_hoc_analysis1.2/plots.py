"""
plots.py
--------
Visualization and statistical utilities for DeuteRater analyses.

This module centralizes plotting, filtering, and styling logic used to
generate volcano plots, conformity scatterplots, and homeostasis panels.
It also contains helpers for robust column resolution, ontology-based
styling
These utilities are called by higher-level orchestration in `main.py`.
"""

from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import gc
from typing import Dict, Any, List, Iterable, Optional, Tuple, Union
import itertools
import os
import math
import difflib
import re as _re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from scipy import stats
import seaborn as sns  # used in homeostasis plots
from scipy.stats import wilcoxon


try:
    import mpld3
    from mpld3 import plugins
    _HAS_MPLD3 = True
    print('mpld3 found, interactive htmls available.')
except Exception:
    _HAS_MPLD3 = False
    print('mpld3 not found, interactive htmls unavailable.')


# ----------------------------------------------------------------------
# Global matplotlib defaults for publication-quality output
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 18,          # Base font size
    "axes.titlesize": 18,     # Title font size
    "axes.labelsize": 18,     # X and Y label size
    "xtick.labelsize": 18,    # X tick label size
    "ytick.labelsize": 18,    # Y tick label size
    "legend.fontsize": 18,    # Legend text
})


def plot_homeostasis_y(experiment, ax, df, x_col, y_col,
                       x_size, y_size,
                       ontology_styles, category_column,
                       significance_column1, significance_column2,
                       axis_titles=None,
                       show_significance_outlines: bool = False):

    def _radius(v):
        try:
            # tuple/list case: use max(|lo|, |hi|) so we stay symmetric around 0
            if isinstance(v, (tuple, list)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                return max(abs(lo), abs(hi))
            # scalar case
            return float(v)
        except Exception:
            # Fallback if something odd slips through
            return 5.0

    # Compute a single square radius R from both inputs
    Rx = _radius(x_size) if x_size is not None else 5.0
    Ry = _radius(y_size) if y_size is not None else 5.0
    R  = max(Rx, Ry) if (np.isfinite(Rx) and np.isfinite(Ry)) else 5.0

    # Minimal background frame
    x_bg = np.linspace(-R, R, 400)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    ax.plot(x_bg, -x_bg, color="green", linestyle="-", linewidth=2.0, zorder=1)

    # Optional significance outlines
    if show_significance_outlines:
        def determine_edge_color(row_data):
            if row_data[significance_column1] and row_data[significance_column2]:
                return 'magenta'
            elif row_data[significance_column1] or row_data[significance_column2]:
                return 'orange'
            else:
                return 'grey'
        for _, row_data in df.iterrows():
            ax.plot(row_data[x_col], row_data[y_col], 'o',
                    markerfacecolor='none',
                    markeredgecolor=determine_edge_color(row_data),
                    markersize=10, zorder=2)

    # Ontology-colored scatter
    if ontology_styles and (category_column in df.columns):
        for ontology, style in ontology_styles.items():
            subset = df[df[category_column] == ontology]
            if not subset.empty:
                sns.scatterplot(
                    data=subset, x=x_col, y=y_col, ax=ax,
                    color=style.get('color', None),
                    marker=style.get('marker', 'o'),
                    zorder=3
                )
    else:
        # Fallback single-color scatter if styles or category not provided
        if not df.empty:
            ax.scatter(df[x_col], df[y_col], s=28, color='tab:blue', zorder=3)

    # Axis formatting: use the derived R for a square frame
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    if axis_titles and len(axis_titles) >= 2:
        ax.set_xlabel(axis_titles[0])
        ax.set_ylabel(axis_titles[1])

    return ''

# ──────────────────────────────────────────────────────────────────────
# Globals & helpers
# ──────────────────────────────────────────────────────────────────────
# Global dataframe to collect paired t-test statistics across plots
statistics_df = pd.DataFrame(columns=["Comparison", "p_all", "p_filtered"])


def _normalize(s: str) -> str:
    """Normalize a string by lowering case, replacing spaces/dashes, and keeping alphanumerics."""
    s = s.replace('-', '_').replace(' ', '_')
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch == '_')


_SYNONYM_SWAPS: Tuple[Tuple[str, str], ...] = (
    ("value", "val"), ("val", "value"),
    ("abundance_95pct_confidence_k", "%abundance_95pct_confidence_k"),
)
_DEF_SUFFIXES = ("_control", "_experiment")


def resolve_column_name(df: pd.DataFrame, target: str) -> str:

    cols = list(df.columns)
    if target in df.columns:
        return target

    norm_map = {_normalize(c): c for c in cols}
    tnorm = _normalize(target)

    if tnorm in norm_map:
        return norm_map[tnorm]

    candidates = {tnorm}
    for a, b in _SYNONYM_SWAPS:
        candidates.add(tnorm.replace(a, b))
    for suf in _DEF_SUFFIXES:
        if not tnorm.endswith(suf):
            candidates.add(tnorm + suf)
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]

    suggestion_keys = difflib.get_close_matches(tnorm, list(norm_map.keys()), n=8, cutoff=0.6)
    suggestions = [norm_map[k] for k in suggestion_keys]
    raise KeyError(f"Column '{target}' not found. Close matches: {suggestions[:8]}")



import re
import operator


def filter_dataframe(
    df: pd.DataFrame,
    columns_to_nan: Optional[List[str]] = None,
    **filters
) -> pd.DataFrame:
    """
    Apply flexible row-level filters to a DataFrame.

    Supports:
      - Numeric comparisons: '< 1', '>= 0.05', '> other_column'
      - Dict ranges with scalars or columns:
            {'min': 0, 'max': 'ref_col', 'min_op': '>=', 'max_op': '<'}
      - 'in:' and 'not in:' substring filters for text columns
      - Bool and exact-value matches
      - Column-to-column comparisons
    """
    df = df.copy()
    combined = pd.Series(True, index=df.index)

    ops = {
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
        '=': operator.eq,
    }

    def _rhs(value):
        """Return numeric Series if value is a column, else scalar."""
        if isinstance(value, str) and value in df.columns:
            return pd.to_numeric(df[value], errors="coerce")
        return value

    for col, condition in filters.items():
        if col not in df.columns:
            continue

        mask = pd.Series(True, index=df.index)
        col_num = pd.to_numeric(df[col], errors="coerce")

        # --- Numeric range dict ---
        if isinstance(condition, dict):
            min_val = _rhs(condition.get("min", float("-inf")))
            max_val = _rhs(condition.get("max", float("inf")))
            min_op = condition.get("min_op", ">=")
            max_op = condition.get("max_op", "<=")

            if isinstance(min_val, pd.Series):
                mask &= ops[min_op](col_num, min_val)
            else:
                mask &= ops[min_op](col_num, float(min_val))

            if isinstance(max_val, pd.Series):
                mask &= ops[max_op](col_num, max_val)
            else:
                mask &= ops[max_op](col_num, float(max_val))

        # --- Boolean exact ---
        elif isinstance(condition, bool):
            mask &= (df[col] == condition)

        # --- String-based filters ---
        elif isinstance(condition, str):
            cond = condition.strip()
            low = cond.lower()

            # --- text inclusion ---
            if low.startswith("in:"):
                substrings = cond.split(":", 1)[1].split(",")
                rx = "|".join(map(lambda s: str(s).strip(), substrings))
                mask &= df[col].astype(str).str.contains(rx, na=False)

            elif low.startswith("not in:"):
                substrings = cond.split(":", 1)[1].split(",")
                rx = "|".join(map(lambda s: str(s).strip(), substrings))
                mask &= ~df[col].astype(str).str.contains(rx, na=False)

            # --- numeric / column comparisons ---
            elif re.match(r"^\s*[<>]=?|=\s*[\w\.-]+\s*$", cond):
                for sym, fn in ops.items():
                    if cond.startswith(sym):
                        rhs_raw = cond[len(sym):].strip()
                        rhs = _rhs(rhs_raw)

                        if isinstance(rhs, pd.Series):
                            mask &= fn(col_num, rhs)
                        else:
                            try:
                                mask &= fn(col_num, float(rhs))
                            except ValueError:
                                mask &= False
                        break

            # --- exact string equality ---
            else:
                mask &= (df[col].astype(str) == cond)

        # --- Fallback ---
        else:
            mask &= (df[col] == condition)

        combined &= mask

    # --- Output handling ---
    if columns_to_nan is None:
        return df.loc[combined]

    eff_cols = set(columns_to_nan).union(filters.keys())
    df.loc[~combined, list(eff_cols)] = np.nan
    return df


# ──────────────────────────────────────────────────────────────────────
# Ontology styling & legends
# ──────────────────────────────────────────────────────────────────────
_MARKERS = ['o','s','D','^','v','<','>','*','p','h','X','P','H','d',
            '1','2','3','4','+','x']
_COLORS  = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple',
            'tab:brown','tab:olive','tab:cyan','magenta','goldenrod',
            'teal','slategray']

# GLOBAL list: ensures ontology order is stable forever
# GLOBAL cache: ensures every ontology keeps the same color+marker forever
GLOBAL_ONTOLOGY_STYLES = {}

def build_ontology_styles(categories: Iterable[Any]) -> Dict[str, Dict[str, str]]:
    global GLOBAL_ONTOLOGY_STYLES

    # Prepare ontology strings
    cats = [str(c) for c in categories if pd.notna(c)]

    # --- YOUR EXISTING COLOR GROUPING LOGIC ---
    teal = [c for c in cats if 'd7' in c.lower()]
    blue = [c for c in cats if ('ether' in c.lower() or 'O-' in c or 'P-' in c) and c not in teal]
    red_tokens = ['PA','PC','PI','PE','PS','PG','CL']
    red = [c for c in cats if any(t in c for t in red_tokens) and c not in teal + blue]
    green = [c for c in cats if (('Cer' in c) or ('SM' in c)) and c not in red + blue + teal]
    magenta = [c for c in cats if any(t in c for t in ['MG','DG','TG']) and c not in teal + blue]
    other = [c for c in cats if c not in teal + red + blue + green + magenta]

    # Ordered categories using your logic
    ordered = red + blue + teal + green + magenta + other

    # Marker cycle
    safe_markers = _MARKERS
    marker_cycle = itertools.cycle(safe_markers)

    # Assign styles
    styles = {}
    for ont in ordered:
        # If ontology already seen, reuse style
        if ont in GLOBAL_ONTOLOGY_STYLES:
            styles[ont] = GLOBAL_ONTOLOGY_STYLES[ont]
            continue

        # Determine color based on your rules
        if ont in red:
            col = 'tab:red'
        elif ont in blue:
            col = 'tab:blue'
        elif ont in teal:
            col = 'teal'
        elif ont in green:
            col = 'tab:green'
        elif ont in magenta:
            col = 'magenta'
        else:
            col = next(c for c in _COLORS if c not in ['tab:red','tab:blue','tab:green','magenta','teal'])

        # Assign first unused marker
        mk = next(marker_cycle)

        GLOBAL_ONTOLOGY_STYLES[ont] = {"color": col, "marker": mk}
        styles[ont] = GLOBAL_ONTOLOGY_STYLES[ont]

    return styles


def legend_handles_from_styles(styles: Dict[str, Dict[str, str]],
                               markersize: int = 7) -> List[Line2D]:

    return [Line2D([0], [0],
                   marker=sty['marker'],
                   linestyle='',
                   markerfacecolor=sty['color'],
                   markeredgecolor=sty['color'],
                   label=name,
                   markersize=markersize)
            for name, sty in styles.items()]



# ───────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ───────────────────────────────────────────────────────────────────────────────


def volcano(ax,
            df: pd.DataFrame,
            x_col: str,
            y_col: str,
            FC_cut: float = 1.0,
            P_cut: float = 1.3,
            x_lim: Optional[Union[float, Tuple[float, float]]] = None,
            y_lim: Optional[Union[float, Tuple[float, float]]] = None,
            label: Optional[str] = None,
            category_col: str = 'Ontology',
            styles: Optional[Dict[str, Dict[str, str]]] = None,
            point_size: float = 28.0) -> None:

    # Validate columns
    if x_col not in df.columns or y_col not in df.columns:
        ax.text(0.5, 0.5, f'Missing {x_col} or {y_col}',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Clean data
    data = df.dropna(subset=[x_col, y_col]).copy()
    data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
    data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
    data = data.dropna(subset=[x_col, y_col])

    # Plot points (ontology-aware if available)
    if category_col in data.columns:
        cats = list(pd.unique(data[category_col].dropna()))
        styles = styles or build_ontology_styles(cats)
        for c in cats:
            sub = data[data[category_col] == c]
            if not sub.empty:
                ax.scatter(sub[x_col], sub[y_col],
                           s=point_size,
                           marker=styles[c].get('marker', 'o'),
                           color=styles[c].get('color', 'tab:blue'),
                           linewidths=0.0)
    else:
        ax.scatter(data[x_col], data[y_col], s=point_size, alpha=0.9)

    # Threshold lines
    ax.axhline(P_cut, linestyle='--', color='red', linewidth=1)
    ax.axvline( FC_cut, linestyle='--', color='red', linewidth=1)
    ax.axvline(-FC_cut, linestyle='--', color='red', linewidth=1)

    # -------- Axis ranges (scalar or tuple) with y-bottom forced to 0 --------
    def _apply_lim(ax, lim, is_x: bool, y_floor_zero: bool = False):

        if lim is None:
            return False
        if isinstance(lim, (tuple, list)) and len(lim) == 2:
            lo, hi = float(lim[0]), float(lim[1])
            if y_floor_zero and not is_x:
                lo = 0.0
            (ax.set_xlim if is_x else ax.set_ylim)(lo, hi)
            return True
        # Scalar radius
        R = abs(float(lim))
        if is_x:
            ax.set_xlim(-R, R)
        else:
            ax.set_ylim(0.0, R)
        return True

    x_applied = _apply_lim(ax, x_lim, is_x=True,  y_floor_zero=False)
    y_applied = _apply_lim(ax, y_lim, is_x=False, y_floor_zero=True)

    # If autoscaling (no limits passed), relim and then clamp y-bottom to 0
    if not x_applied or not y_applied:
        ax.relim()
        ax.autoscale_view()
        # Always force y >= 0 for volcano
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0.0, y0), y1)

    # Title
    if label:
        ax.set_title(label)




def _build_points_df(df, x_col, y_col, label, metric, plot_group, filtered):
    return pd.DataFrame({
        "Comparison": label,
        "Metric": metric,
        "Plot_Group": plot_group,
        "Filtered": filtered,
        "Delta": df[y_col] - df[x_col],
        "X": df[x_col],
        "Y": df[y_col],
    })




def _attach_mpld3_tooltips(ax, df, x_col, y_col, label_col='Species'):
   
    if not _HAS_MPLD3:
        return None

    # Find the most recent PathCollection (scatter) on this axes
    scatters = [c for c in ax.collections if hasattr(c, 'get_offsets')]
    if not scatters:
        return None
    scatter = scatters[-1]

    # Build labels (one per point), default to empty if missing
    labels = []
    if label_col in df.columns:
        labels = [str(v) if pd.notna(v) else "" for v in df[label_col].tolist()]
    else:
        # Fallback: show X,Y if no label column
        xs = pd.to_numeric(df[x_col], errors='coerce').fillna(pd.NA)
        ys = pd.to_numeric(df[y_col], errors='coerce').fillna(pd.NA)
        labels = [f"{x}, {y}" for x, y in zip(xs, ys)]

    tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
    plugins.connect(ax.figure, tooltip, plugins.Zoom(enabled=True), plugins.Reset())
    return tooltip





def scatter_ttest(
    ax,
    N_All: pd.DataFrame,
    N_Filtered: pd.DataFrame,
    x_col: str,
    y_col: str,
    category_col: str = 'Ontology',
    styles: Optional[Dict[str, Dict[str, str]]] = None,
    point_size: float = 32.0,
    axis_titles: Optional[List[str]] = None,
    Experiment_lower: Optional[str] = None,
    Experiment_upper: Optional[str] = None,
    Control_lower: Optional[str] = None,
    Control_upper: Optional[str] = None,
    x_lim: Optional[float] = None,
    y_lim: Optional[float] = None,
    comparison_label: Optional[str] = None,
    filters_applied: Optional[Dict[str, Any]] = None,
    metric_label: Optional[str] = None,
    mode: str = "paired",          # {"paired", "onesample_fc"}
    input_is_log2: bool = True,          # if False, we compute log2 internally for FCs
    alternative: str = "two-sided",      # {"two-sided","less","greater"}
    tiny: float = None,                  # numeric floor to avoid log2(0) if needed
) -> Dict[str, Any]:




    if tiny is None:
        tiny = np.finfo(float).tiny

    # ── Clean input tables (numeric only, drop NaNs for x/y) ─────────────────
    data_all = N_All.dropna(subset=[x_col, y_col]).copy()
    data_all[x_col] = pd.to_numeric(data_all[x_col], errors='coerce')
    data_all[y_col] = pd.to_numeric(data_all[y_col], errors='coerce')
    data_all = data_all.dropna(subset=[x_col, y_col])

    data_filt = N_Filtered.dropna(subset=[x_col, y_col]).copy()
    data_filt[x_col] = pd.to_numeric(data_filt[x_col], errors='coerce')
    data_filt[y_col] = pd.to_numeric(data_filt[y_col], errors='coerce')
    data_filt = data_filt.dropna(subset=[x_col, y_col])

    # ── Helper to compute stats block depending on mode ──────────────────────
    def _paired_stats(df_xy: pd.DataFrame):
        if len(df_xy) <= 1:
            return np.nan, np.nan, np.nan
        a = df_xy[y_col].values
        b = df_xy[x_col].values
        # Paired t-test (use alternative if available)
        try:
            res = stats.ttest_rel(a, b, alternative=alternative)
            t_val, p_val = float(res.statistic), float(res.pvalue)
        except TypeError:
            # older SciPy: no 'alternative' argument
            res = stats.ttest_rel(a, b)
            t_val, p_val = float(res.statistic), float(res.pvalue)
        mean_diff = float(np.nanmean(a - b))
        return t_val, p_val, mean_diff

    def _onesample_fc_stats(df_xy: pd.DataFrame):
        if len(df_xy) <= 1:
            return np.nan, np.nan, np.nan
        if input_is_log2:
            log2fc = (df_xy[y_col] - df_xy[x_col]).values
        else:
            # compute log2((y+tiny)/(x+tiny)) safely
            num = (df_xy[y_col].astype(float) + tiny).values
            den = (df_xy[x_col].astype(float) + tiny).values
            log2fc = np.log2(num / den)

        # One-sample t-test vs 0
        try:
            res = stats.ttest_1samp(log2fc, popmean=0.0, alternative=alternative, nan_policy="omit")
            t_val, p_val = float(res.statistic), float(res.pvalue)
        except TypeError:
            # older SciPy: no 'alternative' or 'nan_policy'
            res = stats.ttest_1samp(log2fc, 0.0)
            t_val, p_val = float(res.statistic), float(res.pvalue)
        mean_diff = float(np.nanmean(log2fc))
        return t_val, p_val, mean_diff

    # ── Compute statistics for all and filtered subsets ──────────────────────
    if mode.lower() == "paired":
        t_all, p_all, mean_diff_all = _paired_stats(data_all)
        t_f,   p_f,   mean_diff_f   = _paired_stats(data_filt)
        test_descriptor = f"paired t-test on {y_col} vs {x_col} (alternative='{alternative}')"
        mean_label = r"$\bar{d}$ (mean diff)"
    elif mode.lower() == "onesample_fc":
        t_all, p_all, mean_diff_all = _onesample_fc_stats(data_all)
        t_f,   p_f,   mean_diff_f   = _onesample_fc_stats(data_filt)
        test_descriptor = f"one-sample t-test on log2FC (H0: mean=0, alternative='{alternative}')"
        mean_label = r"$\overline{\Delta}$ (mean log$_2$FC)"
    else:
        raise ValueError("mode must be one of {'paired','onesample_fc'}")

    n_all, n_filt = len(data_all), len(data_filt)

    # ── Scatter Plot: visualize y vs x with y=x reference line ───────────────
    excluded = data_all.loc[~data_all.index.isin(data_filt.index)]
    if category_col in data_all.columns:
        cats = list(pd.unique(data_all[category_col].dropna()))
        styles = styles or build_ontology_styles(cats)
        for c in cats:
            sub_ex = excluded[excluded[category_col] == c]
            if not sub_ex.empty:
                ax.scatter(sub_ex[x_col], sub_ex[y_col],
                           s=point_size, marker=styles[c]['marker'],
                           color='lightgrey', linewidths=0.0)
        for c in cats:
            sub_in = data_filt[data_filt[category_col] == c]
            if not sub_in.empty:
                ax.scatter(sub_in[x_col], sub_in[y_col],
                           s=point_size, marker=styles[c]['marker'],
                           color=styles[c]['color'], linewidths=0.0)
    else:
        ax.scatter(excluded[x_col], excluded[y_col], s=point_size, color='lightgrey', linewidths=0.0)
        ax.scatter(data_filt[x_col], data_filt[y_col], s=point_size, alpha=0.9)

    # ── Axis Limits / Equalize ───────────────────────────────────────────────
    def _apply_bounds(ax, lim, is_x=True):
        """
        lim can be:
          • None: autoscale
          • float/int: treat as 'R' → symmetric [-R, +R]
          • (min, max) tuple: direct bounds
        """
        if lim is None:
            return False  # caller will autoscale
        if isinstance(lim, (tuple, list)) and len(lim) == 2:
            lo, hi = float(lim[0]), float(lim[1])
            if is_x: ax.set_xlim(lo, hi)
            else:    ax.set_ylim(lo, hi)
            return True
        # scalar → old behavior (symmetric radius)
        R = abs(float(lim))
        if is_x: ax.set_xlim(-R, R)
        else:    ax.set_ylim(-R, R)
        return True
    
    applied_x = _apply_bounds(ax, x_lim, is_x=True)
    applied_y = _apply_bounds(ax, y_lim, is_x=False)
    
    if not applied_x or not applied_y:
        # No global bounds provided → normal autoscale, then square the view
        ax.relim(); ax.autoscale_view()
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xr, yr = (x1 - x0), (y1 - y0)
        if np.isfinite(xr) and np.isfinite(yr):
            if xr > yr:
                yc, half = 0.5*(y0+y1), 0.5*xr; ax.set_ylim(yc-half, yc+half)
            else:
                xc, half = 0.5*(x0+x1), 0.5*yr; ax.set_xlim(xc-half, xc+half)

    # If limits are NOT provided, autoscale and then equalize to square
    if x_lim is None and y_lim is None:
        ax.relim(); ax.autoscale_view()
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xr, yr = (x1 - x0), (y1 - y0)
        if np.isfinite(xr) and np.isfinite(yr):
            if xr > yr:
                yc, half = 0.5*(y0+y1), 0.5*xr; ax.set_ylim(yc-half, yc+half)
            else:
                xc, half = 0.5*(x0+x1), 0.5*yr; ax.set_xlim(xc-half, xc+half)

    # y=x line (use current axis limits)
    try:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        lo, hi = min(x0, y0), max(x1, y1)
        ax.plot([lo, hi], [lo, hi], linestyle=':', color='grey', linewidth=1, zorder=0)
    except Exception:
        pass

    # ── Axis Labels ──────────────────────────────────────────────────────────
    exp_id, ctrl_id = None, None
    if isinstance(comparison_label, str) and ' vs ' in comparison_label:
        parts = comparison_label.split(' vs ', 1)
        if len(parts) == 2:
            exp_id, ctrl_id = parts[0], parts[1]

    if axis_titles and len(axis_titles) >= 2:
        if ctrl_id and exp_id:
            ax.set_xlabel(f"{ctrl_id} {axis_titles[1]}")
            ax.set_ylabel(f"{exp_id} {axis_titles[0]}")
        else:
            ax.set_xlabel(axis_titles[0])
            ax.set_ylabel(axis_titles[1])

    # ── In-plot LaTeX Summary ────────────────────────────────────────────────
    def _fmt(val: float, small_fmt="{:.2e}", large_fmt="{:.2f}") -> str:
        try:
            return small_fmt.format(val) if np.isfinite(val) and abs(val) < 1e-2 else large_fmt.format(val)
        except Exception:
            return str(val)

    all_str = (
        f"{mean_label} (all) = {_fmt(mean_diff_all)}, "
        f"t = {_fmt(t_all, '{:.2f}', '{:.2f}')}, "
        f"p = {_fmt(p_all, '{:.2e}', '{:.3f}')}"
    )
    filt_str = (
        f"{mean_label} (filtered) = {_fmt(mean_diff_f)}, "
        f"t = {_fmt(t_f, '{:.2f}', '{:.2f}')}, "
        f"p = {_fmt(p_f, '{:.2e}', '{:.3f}')}"
    )
    summary_text = f"{all_str}\n{filt_str}"

    # ── Metric Label Cleanup (unchanged) ─────────────────────────────────────
    def _short_metric(name: Optional[str]) -> str:
        if not isinstance(name, str):
            return ""
        name_lower = name.lower()
        if "n-value" in name_lower or "n_l" in name_lower or "nl" in name_lower:
            return "nL"
        if "abundance" in name_lower or "dr" in name_lower:
            return "Abundance"
        if "flux" in name_lower:
            return name_lower
        if "asymptote" in name_lower:
            return "Asymptote"
        if "rate" in name_lower:
            return "Rate"
        return name.strip()

    clean_metric = _short_metric(metric_label)

    # Keep your points collection logic unchanged
    points_all = _build_points_df(
        data_all,
        x_col, y_col,
        comparison_label,
        metric_label,
        plot_group=None,      # filled later
        filtered=False
    )
    points_filt = _build_points_df(
        data_filt,
        x_col, y_col,
        comparison_label,
        metric_label,
        plot_group=None,
        filtered=True
    )
    points_df = pd.concat([points_all, points_filt], ignore_index=True)

    # ── Return Stats Dict (keys unchanged to avoid breaking callers) ─────────
    stats_entry = {
        "Comparison": comparison_label or "",
        "Metric": clean_metric,
        "Mean_Diff_All": mean_diff_all,
        "Mean_Diff_Filtered": mean_diff_f,
        "t_All": t_all,
        "p_All": p_all,
        "t_Filtered": t_f,
        "p_Filtered": p_f,
        "N_All": n_all,
        "N_Filtered": n_filt,
        "Filters_Applied": str(filters_applied or {}),
        "Test": test_descriptor,
        "Mode": mode,
    }

    return summary_text, stats_entry, points_df

# ───────────────────────────────────────────────────────────────────────────────
# High-level driver
# ───────────────────────────────────────────────────────────────────────────────
def _replace_tokens(s: str, exp_id: str, ctrl_id: str) -> str:

    return s.replace('Experiment', exp_id).replace('Control', ctrl_id)


def _replace_tokens_in_filters(d: Dict[str, Any], exp_id: str, ctrl_id: str) -> Dict[str, Any]:

    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[_replace_tokens(k, exp_id, ctrl_id)] = v
    return out





def _autosize_limits(experiments: List[Any], x_col: str, y_col: str,
                     primary: Dict[str, Any], secondary: Dict[str, Any]) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:

    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for exp in experiments:
        df = exp.df.copy()
        x_i = _replace_tokens(x_col, exp.experimental_identifier, exp.control_identifier)
        y_i = _replace_tokens(y_col, exp.experimental_identifier, exp.control_identifier)

        pf = _replace_tokens_in_filters(primary,   exp.experimental_identifier, exp.control_identifier)
        sf = _replace_tokens_in_filters(secondary, exp.experimental_identifier, exp.control_identifier)

        if pf:
            df = filter_dataframe(df, columns_to_nan=[x_i, y_i], **pf)
        if sf:
            df = filter_dataframe(df, columns_to_nan=[x_i, y_i], **sf)

        if x_i in df and y_i in df:
            xv = pd.to_numeric(df[x_i], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            yv = pd.to_numeric(df[y_i], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if not xv.empty:
                xmins.append(float(xv.min()))
                xmaxs.append(float(xv.max()))
            if not yv.empty:
                ymins.append(float(yv.min()))
                ymaxs.append(float(yv.max()))

    if not xmins and not ymins:
        return None, None

    def _pad(lo: float, hi: float, frac: float = 0.10) -> Tuple[float,float]:
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return lo, hi
        pad = span * frac
        return lo - pad, hi + pad

    x_bounds = _pad(min(xmins), max(xmaxs)) if xmins and xmaxs else None
    y_bounds = _pad(min(ymins), max(ymaxs)) if ymins and ymaxs else None
    return x_bounds, y_bounds



def _apoe_label(name: str) -> str:

    if name in {"A2", "A3", "A4", "E2", "E3", "E4"}:
        return "APOE" + name[-1]
    return name


def re_safe_filename(s: str) -> str:

    s = _re.sub(r'[^\w\-.() ]+', '_', str(s))
    return s.strip().replace(' ', '_')



def create_plots(
    experiments: List[Any],
    x_col: str,
    y_col: str,
    analysis_type: str = 'volcano',
    title: Optional[str] = None,
    primary_filters: Optional[Dict[str, Any]] = None,
    secondary_filters: Optional[Dict[str, Any]] = None,
    x_size: Optional[float] = None,
    y_size: Optional[float] = None,
    FC_cut_off: float = 1.0,
    stats_cut_off: float = 1.3,
    output_dir: Optional[str] = None,
    category_column: str = 'Ontology',
    show_legend: bool = True,
    legend_outside: bool = True,
    legend_ncol: int = 6,
    legend_markersize: int = 7,
    legend_fontsize: int = 8,
    axis_titles: Optional[List[str]] = None,
    Experiment_lower: Optional[str] = None,
    Experiment_upper: Optional[str] = None,
    Control_lower: Optional[str] = None,
    Control_upper: Optional[str] = None,
    point_size: float = 32.0,
    columns_to_nan: Optional[List[str]] = None,
    significance_column1: Optional[str] = None,
    significance_column2: Optional[str] = None,
    hspace: float = 0.4,
    wspace: float = -0.4,
    suptitle_y: float = 0.95,
    top_with_legend: float = 0.90,
    top_no_legend: float = 0.965,
    left: float = 0.07,
    right: float = 0.98,
    bottom_min: float = 0.08,
    panel_w: float = 12,
    panel_h: float = 12.0,
    font_family: Optional[str] = None,
    font_size: Optional[float] = None,
    ensure_square: bool = True,
    ensure_same_axis: bool = False,
    panel_cols: int = 2, 
    drop_duplicates_by = None,
    plot_group: Optional[str] = None,
    modifiers = None,
    
    output_interactive_html: bool = True,     # NEW: also write interactive HTML
    tooltip_label_column: str = 'Lipid Unqiue Identifier',     # NEW: which column to show on hover

):
    """
    Create multi-panel plots (volcano / scatter_ttest / homeostasis) and
    return any gathered paired-test statistics.
    """
    statistics_records = []   # 🆕 Collect stats dictionaries here
    datapoint_records = []

    primary_filters = primary_filters or {}
    secondary_filters = secondary_filters or {}

  
    # Auto-size
    if x_size is None or y_size is None:
        xs, ys = _autosize_limits(experiments, x_col, y_col, primary_filters, secondary_filters)
        x_size = x_size if x_size is not None else xs
        y_size = y_size if y_size is not None else ys


    n = len(experiments)
    ncols = 1 if n <= 1 else panel_cols
    nrows = math.ceil(n / max(1, ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * max(1, nrows)), dpi=200)
    axs = np.atleast_1d(axs).ravel()
    fig.subplots_adjust(hspace=hspace, wspace=wspace, left=left, right=right)

    # Build category styles
    all_cats: List[Any] = []
    for exp in experiments:
        df = getattr(exp, 'df', None)
        if isinstance(df, pd.DataFrame) and category_column in df.columns:
            all_cats.extend(list(pd.unique(df[category_column].dropna())))
    styles = build_ontology_styles(sorted(set(all_cats))) if all_cats else None

    # Font
    fp = None
    if font_family or font_size:
        fp = FontProperties()
        if font_family: fp.set_family(font_family)
        if font_size: fp.set_size(font_size)

    # Per experiment
    for k, exp in enumerate(experiments):
        ax = axs[k]
        df_base = exp.df.copy()
        if drop_duplicates_by:
            df_base = df_base.drop_duplicates(subset=drop_duplicates_by, keep='first')

        # Column name resolution
        x_col_i = _replace_tokens(x_col, exp.experimental_identifier, exp.control_identifier)
        y_col_i = _replace_tokens(y_col, exp.experimental_identifier, exp.control_identifier)
        pf = _replace_tokens_in_filters(primary_filters, exp.experimental_identifier, exp.control_identifier)
        sf = _replace_tokens_in_filters(secondary_filters, exp.experimental_identifier, exp.control_identifier)

        df_primary = filter_dataframe(df_base, columns_to_nan=(columns_to_nan or [x_col_i, y_col_i]), **pf) if pf else df_base
        df_secondary = filter_dataframe(df_primary, columns_to_nan=(columns_to_nan or [x_col_i, y_col_i]), **sf) if sf else df_primary

        try:
            x_res = resolve_column_name(df_secondary, x_col_i)
            y_res = resolve_column_name(df_secondary, y_col_i)
        except KeyError as e:
            ax.text(0.5, 0.5, f"{e}", ha='center', va='center', wrap=True, transform=ax.transAxes)
            continue

        exp_label = _apoe_label(exp.experimental_identifier)
        ctl_label = _apoe_label(exp.control_identifier)
        label = f'{exp_label} vs {ctl_label}'

        if analysis_type == 'volcano':
            volcano(ax, df_secondary, x_res, y_res,
                    FC_cut=FC_cut_off, P_cut=stats_cut_off,
                    x_lim=x_size, y_lim=y_size, label=label,
                    category_col=category_column, styles=styles,
                    point_size=point_size)


        
        
        elif analysis_type == 'scatter_ttest':
            
            if not modifiers:
                modifiers = 'paired'
            
            # 🧩 New structured output
            summary, stat_dict, points_df = scatter_ttest(
                ax,
                N_All=df_primary,
                N_Filtered=df_secondary,
                x_col=x_res,
                y_col=y_res,
                category_col=category_column,
                styles=styles,
                point_size=point_size,
                axis_titles=axis_titles,
                Experiment_lower=Experiment_lower,
                Experiment_upper=Experiment_upper,
                Control_lower=Control_lower,
                Control_upper=Control_upper,
                x_lim=x_size,
                y_lim=y_size,
                comparison_label=label,
                filters_applied={**pf, **sf},
                metric_label=title or analysis_type,
                mode = modifiers
            )


            if isinstance(stat_dict, dict):
                # --- Ensure correct Plot_Group labeling ---
                if plot_group and isinstance(plot_group, str) and plot_group.strip():
                    group_label = plot_group.strip()
                elif title and isinstance(title, str) and title.strip():
                    group_label = title.strip()
                else:
                    if isinstance(label, str) and "_" in label:
                        group_label = label.split("_")[0]
                    else:
                        group_label = "Uncategorized"
            
                # ✅ Stats get Plot_Group
                stat_dict["Plot_Group"] = group_label
                ordered = {"Plot_Group": stat_dict.pop("Plot_Group")}
                ordered.update(stat_dict)
                statistics_records.append(ordered)
            
                # ✅ Datapoints ALSO get Plot_Group
                if isinstance(points_df, pd.DataFrame) and not points_df.empty:
                    points_df = points_df.copy()
                    points_df["Plot_Group"] = group_label
            
                datapoint_records.append(points_df)
    



            # Display summary on panel
            ax.set_title(summary,
                         pad=2.0, fontproperties=fp)

        elif analysis_type == 'homeostasis':
            sig1 = significance_column1 or "__sig1_tmp__"
            sig2 = significance_column2 or "__sig2_tmp__"
            if sig1 not in df_secondary.columns: df_secondary[sig1] = False
            if sig2 not in df_secondary.columns: df_secondary[sig2] = False

            _ = plot_homeostasis_y(
                experiment=exp,
                ax=ax,
                df=df_secondary.dropna(subset=[x_res, y_res]),
                x_col=x_res,
                y_col=y_res,
                x_size=(x_size or 1.0),
                y_size=(y_size or 1.0),
                ontology_styles=(styles or {}),
                category_column=category_column,
                significance_column1=sig1,
                significance_column2=sig2,
                axis_titles=axis_titles,
            )
            ax.set_title(label, pad=2.0, fontproperties=fp)

        else:
            ax.text(0.5, 0.5, f'Unknown analysis_type: {analysis_type}',
                    ha='center', va='center', transform=ax.transAxes)

        # Font
        if fp is not None:
            ax.xaxis.label.set_fontproperties(fp)
            ax.yaxis.label.set_fontproperties(fp)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontproperties(fp)

        # Geometry normalization
        if ensure_square:
            if hasattr(ax, "set_box_aspect"): ax.set_box_aspect(1)
            else: ax.set_aspect('equal', adjustable='box')
        if ensure_same_axis:
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            xr, yr = (x1-x0), (y1-y0)
            if xr > yr:
                yc, half = 0.5*(y0+y1), 0.5*xr; ax.set_ylim(yc-half, yc+half)
            elif yr > xr:
                xc, half = 0.5*(x0+x1), 0.5*yr; ax.set_xlim(xc-half, xc+half)

    # Remove unused axes
    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    # Global title & legend
    if title: fig.suptitle(title, y=suptitle_y, fontproperties=fp)
    if show_legend and styles:
        handles = legend_handles_from_styles(styles, markersize=legend_markersize)
        if legend_outside:
            n_items = len(handles)
            rows = math.ceil(n_items / max(1, legend_ncol))
            bottom_pad = max(bottom_min, 0.06 + 0.045 * rows)
            fig.subplots_adjust(bottom=bottom_pad, top=top_with_legend,
                                left=left, right=right, hspace=hspace, wspace=wspace)
            legend_kwargs = dict(ncol=legend_ncol, loc='lower center', bbox_to_anchor=(0.5, 0.02), frameon=True)
            legend_kwargs['prop' if fp else 'fontsize'] = fp or legend_fontsize
            fig.legend(handles=handles, **legend_kwargs)
        else:
            legend_kwargs = dict(ncol=legend_ncol, loc='best', frameon=True)
            legend_kwargs['prop' if fp else 'fontsize'] = fp or legend_fontsize
            axs[min(n-1, len(axs)-1)].legend(handles=handles, **legend_kwargs)
            fig.subplots_adjust(top=top_no_legend, left=left, right=right,
                                hspace=hspace, wspace=wspace)
    else:
        fig.subplots_adjust(top=top_no_legend, left=left, right=right,
                            hspace=hspace, wspace=wspace)

    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe = re_safe_filename(title or f'{analysis_type}_{x_col}_vs_{y_col}')
        out = os.path.join(output_dir, f'{safe}.svg')
        fig.savefig(out, format='svg')
        plt.close()
        gc.collect


    # 🆕 Return unified dataframe if any stats collected
    stats_df = pd.DataFrame(statistics_records)
    
    if datapoint_records:
        points_df = pd.concat(datapoint_records, ignore_index=True)
    else:
        # No scatter_ttest datapoints were generated (e.g., all panels skipped due to missing columns)
        points_df = pd.DataFrame(columns=["Comparison","Metric","Plot_Group","Filtered","Delta","X","Y"])
    
    return {
        "figure": fig,
        "statistics_df": stats_df,
        "datapoints_df": points_df,
    }

