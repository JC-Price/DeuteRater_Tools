#!/usr/bin/env python3
"""

This was produced by Copilot with Coleman's instructions. Needs to be scrutinized. 

Per-metric regression on a tidy CSV (must include columns: 'metric' and 'value').

Key features:
  - Option A: WITH intercept (grand mean) + Sum (effects) coding for categoricals:
        value ~ 1 + C(cat, Sum) + numeric
  - Ontology handling:
      * Run ALL rows into output/ALL/
      * Run each unique Ontology value into output/Ontology_<value>/
      * Ontology is always excluded as a predictor.
  - Robust SE (HC3) optional output
  - ANOVA optional output
  - Forest plot saved per metric (no interactive show)
  - IMPORTANT: Adds IMPLIED sum-coded level effects (e.g., APOE4) into forest plot + key,
    with CI computed from covariance matrix (robust covariance if robust result passed).

Notes on Sum coding:
  For a k-level factor, statsmodels will report k-1 coefficients; the last level is implied:
     beta_missing = -sum(beta_shown)
  This script adds that missing level back into the plot for interpretability.
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# Force non-interactive backend so saving always works (no GUI needed)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
ONTOLOGY_COL = "Ontology"

EXCLUDE_DEFAULT = {
    "Alignment ID", "metric", "value", "sample_id", "ci_lower", "ci_upper", ONTOLOGY_COL, 'se'
}

ADD_PAIRWISE_CATEGORICAL_INTERACTIONS = False
USE_ROBUST_SE = True
MIN_ROWS_PER_MODEL = 6

# Option A: include intercept (grand mean) with effects coding.
INCLUDE_INTERCEPT = True

# Add implied Sum-coded levels (recommended for showing APOE4, etc.)
ADD_IMPLIED_SUM_LEVELS_TO_PLOT = True


# Show numeric values (coef + CI) inside the legend/key under the forest plot
SHOW_NUMBERS_IN_KEY = True

# Formatting for numbers in key (used by pretty_term)
NUM_FMT = ".3f"

# If CI is “close enough” to symmetric, print as ± margin; else print as [L, U]
REL_SYMM_TOL = 0.10  # 10% relative toleranc





# ── Helpers ───────────────────────────────────────────────────────────────────
def q(name: str) -> str:
    """Quote a column name for patsy Q() (handles spaces/special chars)."""
    safe = str(name).replace('"', r'\"')
    return f'Q("{safe}")'


def _as_numeric_if_possible(s: pd.Series) -> tuple[pd.Series, bool]:
    """
    Try to coerce to numeric; return (coerced_series, success_flag).
    success_flag=True means every non-null original value converted to numeric.
    """
    coerced = pd.to_numeric(s, errors="coerce")
    ok = coerced.notna().sum() == s.notna().sum()
    return coerced, ok


def normalize_column_types(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    """Convert object columns to numeric if possible; else categorical."""
    for c in [c for c in df.columns if c not in exclude]:
        if isinstance(df[c].dtype, CategoricalDtype):
            continue
        if df[c].dtype == "object":
            coerced, ok = _as_numeric_if_possible(df[c])
            if ok:
                df[c] = coerced
            else:
                df[c] = df[c].astype("category")
    return df


def split_predictors(df: pd.DataFrame, exclude: set[str]) -> tuple[list[str], list[str]]:
    """Split predictors into categorical / numeric (excluding 'exclude')."""
    candidates = [c for c in df.columns if c not in exclude]
    cat_cols: list[str] = []
    num_cols: list[str] = []

    for c in candidates:
        if isinstance(df[c].dtype, CategoricalDtype):
            cat_cols.append(c)
            continue

        if df[c].dtype == "object":
            coerced, ok = _as_numeric_if_possible(df[c])
            if ok:
                df[c] = coerced
                num_cols.append(c)
            else:
                df[c] = df[c].astype("category")
                cat_cols.append(c)
        else:
            coerced, ok = _as_numeric_if_possible(df[c])
            if ok:
                df[c] = coerced
                num_cols.append(c)
            else:
                df[c] = df[c].astype("category")
                cat_cols.append(c)

    return cat_cols, num_cols


def drop_degenerate_predictors(
    sub: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str]
) -> tuple[list[str], list[str]]:
    """Remove predictors with no variation within this metric subset."""
    good_cat: list[str] = []
    good_num: list[str] = []

    for c in cat_cols:
        u = pd.Series(sub[c]).dropna().unique()
        if len(u) >= 2:
            good_cat.append(c)

    for c in num_cols:
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if s.nunique() >= 2 and float(s.var()) > 0.0:
            good_num.append(c)

    return good_cat, good_num


def build_formula(cat_cols: list[str], num_cols: list[str]) -> str:
    """
    Build a model formula using sum (effects) coding for categoricals.

    Option A (recommended): include intercept so it represents the grand mean.
      value ~ 1 + C(cat, Sum) + numeric
    """
    terms: list[str] = []
    terms.append("1" if INCLUDE_INTERCEPT else "0")
    terms += [f"C({q(c)}, Sum)" for c in cat_cols]
    terms += [q(c) for c in num_cols]

    if ADD_PAIRWISE_CATEGORICAL_INTERACTIONS and len(cat_cols) >= 2:
        for a, b in itertools.combinations(cat_cols, 2):
            terms.append(f"C({q(a)}, Sum):C({q(b)}, Sum)")

    rhs = " + ".join(terms) if terms else ("1" if INCLUDE_INTERCEPT else "0")
    return f"value ~ {rhs}"


def _superscript_int(n: int) -> str:
    """Map digits/signs to Unicode superscripts."""
    trans = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    return str(n).translate(trans)


def safe_folder_name(text: str, max_len: int = 80) -> str:
    """Sanitize a string to be safe as a folder name across OSes."""
    s = str(text).strip()
    # Replace path separators and forbidden characters
    s = "".join("_" if ch in '\\/:"*?<>|' else ch for ch in s)
    # Collapse whitespace
    s = " ".join(s.split())
    if not s:
        s = "UNKNOWN"
    return s[:max_len]


# ── Robust parsing for parameter names (NO REGEX) ──────────────────────────────
def _parse_c_term(term: str) -> tuple[str | None, str | None]:
    """
    Parse patsy categorical term names without regex.
    Expected patterns like:
      C(Q("var name"), Sum)[S.level]
      C(var, Sum)[level]
    Returns (var, level) or (None, None) if it doesn't look like a categorical term.
    """
    try:
        term = str(term)
        if not term.startswith("C("):
            return None, None
        if "[" not in term or "]" not in term:
            return None, None

        prefix, bracket = term.split("[", 1)
        level = bracket.rsplit("]", 1)[0]

        # Remove optional "S." prefix used by Sum coding sometimes
        if level.startswith("S."):
            level = level[2:]

        # prefix example: C(Q("My Col"), Sum)  OR  C(MyCol, Sum)
        if not prefix.endswith(")"):
            return None, None

        inner = prefix[len("C("):-1]  # inside the parentheses
        first_arg = inner.split(",", 1)[0].strip()  # Q("My Col") or MyCol

        if first_arg.startswith('Q("') and first_arg.endswith('")'):
            var = first_arg[3:-2]
        else:
            var = first_arg

        return str(var), str(level)
    except Exception:
        return None, None
    
    


def unique_subdir(base_dir: Path, preferred_name: str) -> Path:
    """
    Return a Path like base_dir/preferred_name or base_dir/preferred_name (1), (2), ...
    that does not exist yet.
    """
    base_dir = Path(base_dir)
    candidate = base_dir / preferred_name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        cand = base_dir / f"{preferred_name} ({i})"
        if not cand.exists():
            return cand
        i += 1


def _format_numbers_in_label(
    coef: float | None,
    lower: float | None,
    upper: float | None,
    num_fmt: str = NUM_FMT,
) -> str:
    """
    Format either ' = coef ± margin' (if symmetric) or ' = coef [L, U]'.
    If any of coef/lower/upper is None or non-finite, return '' (no numbers).
    """
    try:
        if coef is None or lower is None or upper is None:
            return ""
        if not (np.isfinite(coef) and np.isfinite(lower) and np.isfinite(upper)):
            return ""

        err_lo = coef - lower
        err_hi = upper - coef

        # Prefer ± if symmetric-ish; otherwise show [L, U]
        if err_lo > 0 and err_hi > 0:
            # Relative symmetry test: |err_lo - err_hi| <= tol * max(err_lo, err_hi)
            if abs(err_lo - err_hi) <= REL_SYMM_TOL * max(err_lo, err_hi):
                margin = 0.5 * (err_lo + err_hi)
                return f" = {coef:{num_fmt}} ± {margin:{num_fmt}}"
            else:
                return f" = {coef:{num_fmt}} [{lower:{num_fmt}}, {upper:{num_fmt}}]"
        else:
            return f" = {coef:{num_fmt}} [{lower:{num_fmt}}, {upper:{num_fmt}}]"
    except Exception:
        return ""


def pretty_term(
    name: str,
    coef: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> str:
    """
    Make patsy/statsmodels parameter names more readable.
    Optionally append numbers (coef ± margin or coef [L, U]) if provided.
    """
    name = str(name)

    # 1) Intercept -> "Intercept (grand mean)"
    if name == "Intercept":
        base = "Intercept (grand mean)"
        # Append numbers only if desired and provided
        return base + (_format_numbers_in_label(coef, lower, upper) if SHOW_NUMBERS_IN_KEY else "")

    # 2) Interactions -> format each side independently and join with ' × '
    if ":" in name:
        pieces = name.split(":")
        friendly_parts: list[str] = []
        for piece in pieces:
            var, lvl = _parse_c_term(piece)
            if var is not None:
                friendly_parts.append(f"{var}: {lvl}")
            else:
                friendly_parts.append(piece)
        base = " × ".join(friendly_parts) + " (interaction)"
        return base + (_format_numbers_in_label(coef, lower, upper) if SHOW_NUMBERS_IN_KEY else "")

    # 3) Categorical term -> "Var: Level"
    var, lvl = _parse_c_term(name)
    if var is not None:
        base = f"{var}: {lvl}"
        return base + (_format_numbers_in_label(coef, lower, upper) if SHOW_NUMBERS_IN_KEY else "")

    # 4) Otherwise return the name (likely numeric predictor)
    base = name
    return base + (_format_numbers_in_label(coef, lower, upper) if SHOW_NUMBERS_IN_KEY else "")


def _critical_value_975(result) -> float:
    """
    97.5% critical value for two-sided 95% CI.
    Use t if SciPy is available; otherwise fall back to 1.96.
    """
    df_resid = getattr(result, "df_resid", None)
    if df_resid is None or not np.isfinite(df_resid) or df_resid <= 0:
        return 1.96
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.975, df_resid))
    except Exception:
        return 1.96

def critical_value(result, alpha: float) -> float:
    """
    Two-sided critical value for the requested CI (1 - alpha).
    Prefer t with df_resid if SciPy is available; fall back to normal.
      alpha=0.10 -> ~1.64485
      alpha=0.05 -> ~1.95996
      alpha=0.01 -> ~2.57583
    """
    df_resid = getattr(result, "df_resid", None)
    # Fast path without SciPy (common CIs)
    normal_by_alpha = {
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
        0.01: 2.5758293035489004
    }
    try:
        # Try SciPy t critical if available and df makes sense
        if df_resid is not None and np.isfinite(df_resid) and df_resid > 0:
            try:
                from scipy.stats import t as student_t
                return float(student_t.ppf(1 - alpha / 2.0, df_resid))
            except Exception:
                pass
        # Fall back to normal approximation; use table if alpha matches
        if alpha in normal_by_alpha:
            return normal_by_alpha[alpha]
        # Generic normal fallback via approximations (no SciPy): use 1.96-like default
        return 1.959963984540054
    except Exception:
        return 1.959963984540054


def add_implied_sum_levels(
    result,
    forest: pd.DataFrame,
    sub_df: pd.DataFrame,
    cat_cols: list[str],
    alpha
) -> pd.DataFrame:
    """
    For each categorical predictor using Sum coding, statsmodels shows k-1 levels.
    Add the implied kth level with coef and CI computed from covariance matrix.
    Works with robust result objects too (uses result.cov_params()).
    """
    # Params and names
    exog_names = list(getattr(result.model, "exog_names", []))
    params = pd.Series(np.asarray(result.params), index=exog_names)

    cov_raw = result.cov_params()
    cov = pd.DataFrame(np.asarray(cov_raw), index=exog_names, columns=exog_names)

    crit = critical_value(result, alpha)

    new_rows = []

    for var in cat_cols:
        if var not in sub_df.columns:
            continue

        # Levels present in this modeled subset
        s = sub_df[var].dropna()
        if s.empty:
            continue

        if hasattr(sub_df[var].dtype, "categories"):
            levels = list(sub_df[var].dtype.categories)
        else:
            levels = sorted(s.unique().tolist())

        # Which levels appear in the estimated parameter list?
        var_param_names = []
        levels_in_params = []

        for pname in params.index:
            v, lvl = _parse_c_term(pname)
            if v == var and lvl is not None:
                var_param_names.append(pname)
                levels_in_params.append(lvl)

        if not var_param_names:
            continue

        missing = [lvl for lvl in levels if lvl not in levels_in_params]
        if len(missing) != 1:
            # If none missing, or more than one missing, skip (degenerate/unusual design)
            continue

        implied_lvl = missing[0]

        # implied coef: negative sum of shown level coefficients
        implied_coef = -float(params.loc[var_param_names].sum())

        # Var(implied) = 1' Cov 1 over shown params
        C = cov.loc[var_param_names, var_param_names].to_numpy()
        implied_var = float(np.ones(len(var_param_names)) @ C @ np.ones(len(var_param_names)))
        implied_se = math.sqrt(implied_var) if implied_var >= 0 else np.nan

        implied_lower = implied_coef - crit * implied_se
        implied_upper = implied_coef + crit * implied_se

        # Build a synthetic term name that pretty_term can parse
        implied_term = f'C({q(var)}, Sum)[S.{implied_lvl}]'

        new_rows.append((implied_term, implied_lower, implied_upper, implied_coef))

    if not new_rows:
        return forest

    add_df = pd.DataFrame(new_rows, columns=["term", "lower", "upper", "coef"]).set_index("term")
    forest2 = pd.concat([forest, add_df], axis=0)
    return forest2




import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from pathlib import Path

# --- UPDATED: regression equation can optionally include ±SE ------------------
def format_regression_equation(
    model_result,
    decimals: int = 3,
    max_terms: int = 12,
    se_by_name: pd.Series | None = None,
    show_se: bool = False,
) -> str:
    """
    Return a human-readable linear regression equation like:
        y = 1.23 + 0.456*x1 - 2.000*x2 + ...
    Optionally appends (±SE) to each term when show_se=True and se_by_name is provided.
    """
    try:
        y_name = getattr(model_result.model, "endog_names", None) or "y"
        if isinstance(y_name, (list, tuple)):
            y_name = ", ".join(map(str, y_name))
    except Exception:
        y_name = "y"

    try:
        names = list(getattr(model_result.model, "exog_names", []))
    except Exception:
        names = []
    params = pd.Series(np.asarray(model_result.params), index=names if names else None)

    intercept_names = [n for n in (params.index or []) if str(n).lower() == "const"]
    intercept = float(params[intercept_names[0]]) if intercept_names else 0.0

    rhs_terms = []
    for name, coef in params.items():
        if intercept_names and name == intercept_names[0]:
            continue
        term = str(name) if (name is not None and str(name).strip() != "") else "x"
        c = float(coef)
        sign = "+" if c >= 0 else "-"

        piece = f" {sign} {abs(c):.{decimals}f}*{term}"
        if show_se and se_by_name is not None and name in se_by_name.index:
            se_val = se_by_name.loc[name]
            if pd.notna(se_val) and np.isfinite(se_val):
                piece += f" (±{float(se_val):.{decimals}f})"
        rhs_terms.append(piece)

    overflow = False
    if len(rhs_terms) > max_terms:
        rhs_terms = rhs_terms[:max_terms]
        overflow = True

    eq = f"{y_name} = {intercept:.{decimals}f}"
    if rhs_terms:
        eq += "".join(rhs_terms)
    if overflow:
        eq += " + …"
    return eq
# -----------------------------------------------------------------------------


def _superscript_int(exp: int) -> str:
    """Render an integer as superscript text like 10⁶."""
    trans = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(exp).translate(trans)


def diagnostics_plots(
    model_result,
    model_name: str,
    save_to: Path,
    sub_df: pd.DataFrame,
    cat_cols: list[str],
    *,
    se_series: pd.Series | None = None,   # <--- NEW: accept external SEs by name
    alpha: float = 0.05,                   # <--- NEW: CI level
    prefer_se: bool = True,                # <--- NEW: prefer SE to compute CI
) -> None:
    """Coefficient forest plot with numeric labels and a key mapping numbers→terms."""

    FS = 12
    rc = {
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "figure.titlesize": FS,
    }

    with plt.rc_context(rc):
        names = list(getattr(model_result.model, "exog_names", []))
        params = pd.Series(np.asarray(model_result.params), index=names, name="coef")

        # --- NEW: obtain SEs (priority: se_series → model_result.bse) ----------
        se = None
        if se_series is not None:
            # If caller provides SEs, align them to params by name
            se = pd.Series(se_series, copy=False)
            if se.index is None or len(se.index) == 0:
                se.index = params.index
            se = se.reindex(params.index)

        if se is None:
            try:
                se = pd.Series(np.asarray(model_result.bse), index=params.index)
            except Exception:
                se = None
        # ----------------------------------------------------------------------

        # --- UPDATED: build CI, preferring SE if available ---------------------
        use_se = prefer_se and (se is not None) and se.notna().any()
        if use_se:
            # Normal-approx critical value (no SciPy dependency)
            # For alpha=0.05 → 1.9599639845
            z = float(np.abs(np.percentile(np.random.standard_normal(10_0000), 100*(1-alpha/2))))  # deterministic not guaranteed
            # To avoid randomness and avoid SciPy, just use constant:
            z = 1.959963984540054
            ci_df = pd.DataFrame({
                "lower": params - z * se,
                "upper": params + z * se,
                "se": se,
            })
        else:
            ci_raw = model_result.conf_int(alpha=alpha)
            if isinstance(ci_raw, np.ndarray):
                ci_df = pd.DataFrame(ci_raw, index=params.index, columns=["lower", "upper"])
            else:
                ci_df = ci_raw.rename(columns={0: "lower", 1: "upper"}).reindex(params.index)
            if se is not None:
                ci_df["se"] = se
        # ----------------------------------------------------------------------

        forest = ci_df.assign(coef=params)

        # Add implied levels so APOE4 etc. shows up
        if 'ADD_IMPLIED_SUM_LEVELS_TO_PLOT' in globals() and ADD_IMPLIED_SUM_LEVELS_TO_PLOT:
            forest = add_implied_sum_levels(model_result, forest, sub_df, cat_cols, alpha)

        forest = forest.sort_values("coef")
        n = len(forest)
        
        # Identify intercept rows (statsmodels often uses "Intercept" or "const")
        def _is_intercept_name(s: str) -> bool:
            s = str(s).strip()
            return s.lower() in {"intercept", "const"}
        
        forest_full = forest.copy()
        intercept_rows = [i for i in forest_full.index if _is_intercept_name(i)]
        forest_plot = forest_full.drop(index=intercept_rows, errors="ignore")
        
        # Sort and proceed with plotting using forest_plot only
        forest_plot = forest_plot.sort_values("coef")
        n = len(forest_plot)

        fig_height = max(3.4, 1.0 + 0.35 * n) * 1.3
        fig, ax = plt.subplots(figsize=(7.0, fig_height))

        xerr = np.vstack([
            forest["coef"] - forest["lower"],
            forest["upper"] - forest["coef"],
        ])

        ax.errorbar(
            forest["coef"], range(n), xerr=xerr,
            fmt="o", capsize=3, markersize=5, mec="black", mew=0.6
        )
        ax.axvline(0, color="k", linestyle="--")
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(i) for i in range(1, n + 1)])
        ax.invert_yaxis()

        ax.set_xlabel("Estimate (95% CI)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False, useMathText=False)
        ax.get_xaxis().get_offset_text().set_visible(False)

        x_abs_max = np.nanmax(np.abs(forest[["coef", "lower", "upper"]].to_numpy()))
        if np.isfinite(x_abs_max) and x_abs_max >= 1_000:
            exp = int(3 * math.floor(math.log10(x_abs_max) / 3.0))
            scale = 10.0 ** exp
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/scale:.1f}"))
            ax.set_xlabel(f"Estimate (95% CI) ×10{_superscript_int(exp)}")
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Title
        fig.suptitle(f"Effect sizes – {model_name}", ha="center", y=0.8)

        # NEW: Human-readable regression equation with optional ±SE
        try:
            se_for_eq = forest.get("se", None)
            eq_text = format_regression_equation(
                model_result, decimals=3, max_terms=12,
                se_by_name=se_for_eq, show_se=True
            )
            fig.text(
                0.02, 0.76, eq_text,
                ha="left", va="top",
                fontsize=FS, family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.25", alpha=0.9)
            )
        except Exception:
            pass

        # Bottom key (robust)
        term_labels = []
        for t, row in forest.iterrows():
            try:
                if 'SHOW_NUMBERS_IN_KEY' in globals() and SHOW_NUMBERS_IN_KEY:
                    # Optionally append SE in the key if available
                    if "se" in row.index and pd.notna(row["se"]):
                        label = f"{pretty_term(t, float(row['coef']), float(row['lower']), float(row['upper']))}"
                    else:
                        label = pretty_term(t, float(row["coef"]), float(row["lower"]), float(row["upper"]))
                    term_labels.append(label)
                else:
                    if "se" in row.index and pd.notna(row["se"]):
                        term_labels.append(f"{pretty_term(t)} (SE={row['se']:.3f})")
                    else:
                        term_labels.append(pretty_term(t))
            except Exception:
                term_labels.append(str(t))

        if n > 0:
            max_rows = 18
            ncols = max(1, min(4, int(math.ceil(n / max_rows))))
            rows_per_col = int(math.ceil(n / ncols))

            cols = []
            for c in range(ncols):
                start = c * rows_per_col
                end = min((c + 1) * rows_per_col, n)
                if start >= n:
                    break
                block = "\n".join(f"{i+1:>2}. {term_labels[i]}" for i in range(start, end))
                cols.append(block)

            fs_key = FS
            lines = max(len(b.splitlines()) for b in cols) + 1
            line_h_in = fs_key / 72.0 * 1.25
            _, fig_h = fig.get_size_inches()
            bottom_pad = min(0.50, max(0.18, (lines * line_h_in) / fig_h + 0.05))

            fig.tight_layout(rect=[0.0, bottom_pad, 1.0, 0.86])
            y_top = bottom_pad - 0.02
            fig.text(0.02, y_top, "Key:", ha="left", va="top",
                     weight="bold", fontsize=fs_key)

            xs = np.linspace(0.10, 0.98, len(cols), endpoint=True)
            for x, block in zip(xs, cols):
                fig.text(x, y_top, block, ha="left", va="top",
                         family="monospace", fontsize=fs_key)
        else:
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])

        out_path = save_to / f"{model_name}_coeff_forest.png"
        
        # --- NEW: Simple bottom equation line: y = [effect1] + [effect2] + intercept
# --- BOTTOM: Show biotypes (exactly the categorical columns) + intercept = y ---
        try:
            # biotypes are literally the columns in `cat_cols` (already filtered/excluded upstream)
            biotypes = [str(c) for c in cat_cols]

            if biotypes:
                lhs = " + ".join(biotypes)
                eq_groups = f"{lhs} + intercept = {model_name}"
            else:
                # No categorical predictors → just show intercept
                eq_groups = f"intercept = {model_name}"

            # Put the text near the bottom of the figure canvas (below plot/key).
            # If it looks too low/clipped in your environment, bump y to 0.02.
            fig.text(
                0.02, 0.012, eq_groups,
                ha="left", va="bottom",
                fontsize=FS, family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.8",
                          boxstyle="round,pad=0.25", alpha=0.95),
            )
        except Exception:
            pass

        
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def diagnostics_plots(
    model_result,
    model_name: str,
    save_to: Path,
    sub_df: pd.DataFrame,
    cat_cols: list[str],
    *,
    se_series: pd.Series | None = None,   # accept external SEs by name
    alpha: float = 0.05,                   # CI level
    prefer_se: bool = True,                # prefer SE to compute CI
) -> None:
    """Coefficient forest plot with numeric labels and a key mapping numbers→terms.

    This version hides the Intercept from the plotted points,
    keeps it in the key (marked "not plotted"),
    and prints Intercept value + CI in the bottom equation instead of the literal 'intercept'.
    """

    FS = 12
    rc = {
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "figure.titlesize": FS,
    }

    # --- local helpers (self-contained for drop-in) --------------------------------
    def _is_intercept_name(s: str) -> bool:
        s = str(s).strip()
        return s.lower() in {"intercept", "const"}

    def _format_value_with_ci(coef: float, lower: float, upper: float, num_fmt: str = NUM_FMT) -> str:
        """Return 'coef ± margin' if symmetric-ish; else 'coef [L, U]'."""
        try:
            if any((v is None) or (not np.isfinite(v)) for v in (coef, lower, upper)):
                return f"{coef:{num_fmt}}"
            err_lo = coef - lower
            err_hi = upper - coef
            if err_lo > 0 and err_hi > 0 and abs(err_lo - err_hi) <= REL_SYMM_TOL * max(err_lo, err_hi):
                margin = 0.5 * (err_lo + err_hi)
                return f"{coef:{num_fmt}} ± {margin:{num_fmt}}"
            return f"{coef:{num_fmt}} [{lower:{num_fmt}}, {upper:{num_fmt}}]"
        except Exception:
            return f"{coef:{num_fmt}}"
    # ------------------------------------------------------------------------------

    with plt.rc_context(rc):
        # Coefs
        names = list(getattr(model_result.model, "exog_names", []))
        params = pd.Series(np.asarray(model_result.params), index=names, name="coef")

        # --- SEs (priority: provided se_series → model_result.bse) -----------------
        se = None
        if se_series is not None:
            se = pd.Series(se_series, copy=False)
            if se.index is None or len(se.index) == 0:
                se.index = params.index
            se = se.reindex(params.index)

        if se is None:
            try:
                se = pd.Series(np.asarray(model_result.bse), index=params.index)
            except Exception:
                se = None

        # --- Build CI (prefer SE when available) -----------------------------------
        if prefer_se and (se is not None) and se.notna().any():
            crit = critical_value(model_result, alpha) if 'critical_value' in globals() else 1.959963984540054
            ci_df = pd.DataFrame(
                {
                    "lower": params - crit * se,
                    "upper": params + crit * se,
                    "se": se,
                },
                index=params.index,
            )
        else:
            ci_raw = model_result.conf_int(alpha=alpha)
            if isinstance(ci_raw, np.ndarray):
                ci_df = pd.DataFrame(ci_raw, index=params.index, columns=["lower", "upper"])
            else:
                ci_df = ci_raw.rename(columns={0: "lower", 1: "upper"}).reindex(params.index)
            if se is not None:
                ci_df["se"] = se

        forest = ci_df.assign(coef=params)

        # Add implied Sum-coded levels (e.g., to show the 'missing' APOE level)
        if 'ADD_IMPLIED_SUM_LEVELS_TO_PLOT' in globals() and ADD_IMPLIED_SUM_LEVELS_TO_PLOT:
            forest = add_implied_sum_levels(model_result, forest, sub_df, cat_cols, alpha)

        # Preserve a full copy for key and intercept usage
        forest_full = forest.copy()

        # Remove intercept from the plotted set only
        intercept_rows = [i for i in forest_full.index if _is_intercept_name(i)]
        forest_plot = forest_full.drop(index=intercept_rows, errors="ignore").sort_values("coef")
        n = len(forest_plot)

        # --- Figure & axes ----------------------------------------------------------
        fig_height = max(3.4, 1.0 + 0.35 * max(1, n)) * 1.3
        fig, ax = plt.subplots(figsize=(7.0, fig_height))

        if n > 0:
            xerr = np.vstack(
                [
                    forest_plot["coef"] - forest_plot["lower"],
                    forest_plot["upper"] - forest_plot["coef"],
                ]
            )
            ax.errorbar(
                forest_plot["coef"],
                range(n),
                xerr=xerr,
                fmt="o",
                capsize=3,
                markersize=5,
                mec="black",
                mew=0.6,
            )
            ax.set_yticks(range(n))
            ax.set_yticklabels([str(i) for i in range(1, n + 1)])
            ax.invert_yaxis()
        else:
            # No plotted points (e.g., only Intercept) – keep axes clean
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.axvline(0, color="k", linestyle="--")
        ax.set_xlabel("Estimate (95% CI)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False, useMathText=False)
        ax.get_xaxis().get_offset_text().set_visible(False)

        # X-axis formatter & optional SI scaling (compute from plotted terms if any,
        # else fall back to all terms so the scale is still sensible)
        source_for_scale = forest_plot if n > 0 else forest_full
        x_abs_max = np.nanmax(np.abs(source_for_scale[["coef", "lower", "upper"]].to_numpy()))
        if np.isfinite(x_abs_max) and x_abs_max >= 1_000:
            exp = int(3 * math.floor(math.log10(x_abs_max) / 3.0))
            scale = 10.0 ** exp
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/scale:.1f}"))
            ax.set_xlabel(f"Estimate (95% CI) ×10{_superscript_int(exp)}")
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        # Title
        fig.suptitle(f"Effect sizes – {model_name}", ha="center", y=0.8)

        # Human-readable regression equation (top, monospaced)
        try:
            se_for_eq = forest_full.get("se", None)
            eq_text = format_regression_equation(
                model_result, decimals=3, max_terms=12, se_by_name=se_for_eq, show_se=True
            )
            fig.text(
                0.02,
                0.76,
                eq_text,
                ha="left",
                va="top",
                fontsize=FS,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.25", alpha=0.9),
            )
        except Exception:
            pass

        # --- Build the key ---------------------------------------------------------
        # Numbered entries correspond ONLY to plotted points
        term_labels: list[str] = []
        for t, row in forest_plot.iterrows():
            try:
                if 'SHOW_NUMBERS_IN_KEY' in globals() and SHOW_NUMBERS_IN_KEY:
                    label = pretty_term(t, float(row["coef"]), float(row["lower"]), float(row["upper"]))
                else:
                    label = pretty_term(t)
                term_labels.append(label)
            except Exception:
                term_labels.append(str(t))


        # Layout the key
        if n > 0:
            max_rows = 18
            ncols = max(1, min(4, int(math.ceil(n / max_rows))))
            rows_per_col = int(math.ceil(n / ncols))

            cols: list[str] = []
            for c in range(ncols):
                start = c * rows_per_col
                end = min((c + 1) * rows_per_col, n)
                if start >= n:
                    break
                block = "\n".join(f"{i+1:>2}. {term_labels[i]}" for i in range(start, end))
                cols.append(block)

            fs_key = FS
            lines = max(len(b.splitlines()) for b in cols) + 1
            line_h_in = fs_key / 72.0 * 1.25
            _, fig_h = fig.get_size_inches()
            bottom_pad = min(0.50, max(0.18, (lines * line_h_in) / fig_h + 0.05))

            fig.tight_layout(rect=[0.0, bottom_pad, 1.0, 0.86])

            y_top = bottom_pad - 0.02
            fig.text(
                0.02, y_top, "Key:", ha="left", va="top", weight="bold", fontsize=fs_key
            )

            xs = np.linspace(0.10, 0.98, len(cols), endpoint=True)
            for x, block in zip(xs, cols):
                fig.text(
                    x, y_top, block, ha="left", va="top", family="monospace", fontsize=fs_key
                )

            intercept_block = False
            # Add the intercept (unnumbered) just below the key columns
            if intercept_block:
                extra_y = y_top - (lines * (fs_key / 72.0) * 1.25) - 0.008
                fig.text(0.10, extra_y, intercept_block, ha="left", va="top", family="monospace", fontsize=fs_key)
        else:
            # No plotted terms → simpler layout
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
            if intercept_block:
                fig.text(
                    0.02,
                    0.14,
                    "Key:",
                    ha="left",
                    va="top",
                    weight="bold",
                    fontsize=FS,
                )
                fig.text(
                    0.10,
                    0.12,
                    intercept_block,
                    ha="left",
                    va="top",
                    family="monospace",
                    fontsize=FS,
                )

        out_path = save_to / f"{model_name}_coeff_forest.png"

        # --- Bottom equation: show cat predictors + Intercept VALUE+CI ------------
        try:
            # Intercept text with CI
            intercept_text = None
            if intercept_rows:
                it = intercept_rows[0]
                try:
                    irow = forest_full.loc[it]
                    iv = float(irow["coef"])
                    il = float(irow["lower"])
                    iu = float(irow["upper"])
                    intercept_text = _format_value_with_ci(iv, il, iu, num_fmt=NUM_FMT)
                except Exception:
                    intercept_text = None

            if intercept_text is None:
                # Fallback to numeric intercept if CI unavailable
                try:
                    intercept_guess = None
                    for nm, val in params.items():
                        if _is_intercept_name(nm):
                            intercept_guess = float(val)
                            break
                    intercept_text = f"{intercept_guess:{NUM_FMT}}" if intercept_guess is not None else "intercept"
                except Exception:
                    intercept_text = "intercept"

            biotypes = [str(c) for c in cat_cols]
            if biotypes:
                lhs = " + ".join(biotypes)
                eq_groups = f"{lhs} + {intercept_text} = {model_name}"
            else:
                eq_groups = f"{intercept_text} = {model_name}"

            # Place near bottom of canvas (below plot/key)
            fig.text(
                0.02,
                0.012,
                eq_groups,
                ha="left",
                va="bottom",
                fontsize=FS,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.25", alpha=0.95),
            )
        except Exception:
            pass

        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def run_per_metric_analysis(df_in: pd.DataFrame, out_dir: Path, alpha: float, exclude: set[str]) -> tuple[int, list[str]]:
    """
    Run the per-metric loop on df_in; write outputs into out_dir.
    Returns: (metrics_ran_count, errors_list)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    ran = 0
    df = df_in.copy()

    if "metric" not in df.columns or "value" not in df.columns:
        return 0, ["Missing required columns: 'metric' and/or 'value'."]

    if not isinstance(df["metric"].dtype, CategoricalDtype):
        df["metric"] = df["metric"].astype("category")

    # Normalize predictor types (excluding excluded columns)
    df = normalize_column_types(df, exclude)

    # Only metrics present in this subset
    metrics = sorted(df["metric"].dropna().unique().tolist())

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()

        # Detect predictors on this subset
        cat_cols_all, num_cols_all = split_predictors(sub, exclude=exclude)
        cat_cols, num_cols = drop_degenerate_predictors(sub, cat_cols_all, num_cols_all)

        formula = build_formula(cat_cols, num_cols)
        needed_cols = ["value"] + cat_cols + num_cols
        sub2 = sub.dropna(subset=needed_cols)
        
        # --- sanity check ---
        v = pd.to_numeric(sub2["value"], errors="coerce")
        print(f"[{metric}] value summary: n={v.notna().sum()}, min={v.min()}, mean={v.mean()}, max={v.max()}")
        # --------------------


        if len(sub2) < MIN_ROWS_PER_MODEL:
            errors.append(f"[{metric}] insufficient rows after NA-drop (n={len(sub2)}); skipped.")
            continue

        # Fit model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.ols(formula, data=sub2).fit()
        except Exception as e:
            errors.append(f"[{metric}] fit failed: {e}")
            continue

        # Write OLS summary
        try:
            (out_dir / f"{metric}_ols_summary.txt").write_text(
                mdl.summary().as_text(),
                encoding="utf-8"
            )
        except Exception as e:
            errors.append(f"[{metric}] could not write OLS summary: {e}")

        # Robust summary + source for plotting
        coef_src = mdl
        if USE_ROBUST_SE:
            try:
                rob = mdl.get_robustcov_results(cov_type="HC3")
                (out_dir / f"{metric}_robust_summary.txt").write_text(
                    rob.summary().as_text(),
                    encoding="utf-8"
                )
                coef_src = rob
            except Exception as e:
                errors.append(f"[{metric}] robust SE failed: {e}")
                coef_src = mdl  # fall back

        # ANOVA (based on OLS)
        try:
            an = anova_lm(mdl, typ=2)
            an.to_csv(out_dir / f"{metric}_anova.csv")
        except Exception as e:
            errors.append(f"[{metric}] ANOVA failed: {e}")

        # Plot (saved to file)
        try:
            # Build SE series aligned with exog names (works for robust or OLS)
            try:
                exog_names = list(getattr(coef_src.model, "exog_names", []))
            except Exception:
                exog_names = None

            se_series = None
            try:
                bse = np.asarray(coef_src.bse)
                if exog_names and len(bse) == len(exog_names):
                    se_series = pd.Series(bse, index=exog_names)
                else:
                    # Fallback: try to align using the params index if available
                    se_index = exog_names if exog_names else getattr(coef_src.params, "index", None)
                    se_series = pd.Series(bse, index=se_index)
            except Exception:
                se_series = None  # diagnostics_plots will fall back to conf_int()

            diagnostics_plots(
                coef_src,                 # model (robust or OLS)
                str(metric),              # model name
                out_dir,                  # output directory
                sub2,                     # data subset
                cat_cols,                 # categorical columns
                se_series=se_series,      # <--- NEW: pass SEs for CI & annotations
                alpha=alpha,               # <--- NEW: CI level
                prefer_se=True            # <--- NEW: prefer SE-based CI
            )
        except Exception as e:
            errors.append(f"[{metric}] plotting failed: {e}")
        except Exception as e:
            errors.append(f"[{metric}] plotting failed: {e}")

        ran += 1
        print(f"{out_dir.name} :: {metric}: OK | formula: {formula}")

    if errors:
        (out_dir / "_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    return ran, errors

import tkinter as tk
from tkinter import messagebox

class CIChooser(tk.Tk):
    def __init__(self, default_ci=95.0):
        super().__init__()

        self.title("Choose Confidence Interval")
        self.resizable(False, False)

        # This will hold the chosen CI percent (float) or None if canceled
        self.result_ci = None

        self.var = tk.StringVar(value=f"{default_ci}")
        self.choice = tk.StringVar(value="95")

        frm = tk.Frame(self, padx=12, pady=10)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Select a confidence interval:").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        radio_frame = tk.Frame(frm)
        radio_frame.grid(row=1, column=0, columnspan=2, sticky="w")

        for text, val in [("90%", "90"), ("95% (default)", "95"), ("99%", "99"), ("Custom", "custom")]:
            tk.Radiobutton(
                radio_frame,
                text=text,
                value=val,
                variable=self.choice,
                command=self._sync_choice
            ).pack(anchor="w")

        custom_frame = tk.Frame(frm)
        custom_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        tk.Label(custom_frame, text="Custom (%): ").pack(side="left")
        self.entry = tk.Entry(custom_frame, width=7, textvariable=self.var, state="disabled")
        self.entry.pack(side="left")

        btns = tk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        tk.Button(btns, text="OK", width=10, command=self._ok).pack(side="right", padx=(6, 0))
        tk.Button(btns, text="Cancel", width=10, command=self._cancel).pack(side="right")

        # ✅ Correct Tk event names (NOT HTML-escaped)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())

        # Handle clicking the X
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._sync_choice()

        # --- Windows-friendly: center + bring to front ---
        self.update_idletasks()
        self._center_on_screen()

        # Bring to front reliably on Windows
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))
        self.after(0, self.focus_force)
        
        

    def _center_on_screen(self):
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"+{x}+{y}")

    def _sync_choice(self):
        """Enable/disable custom entry based on selection."""
        if self.choice.get() == "custom":
            self.entry.config(state="normal")
            self.entry.focus_set()
            self.entry.selection_range(0, tk.END)
        else:
            fixed = self.choice.get()
            if fixed in {"90", "95", "99"}:
                self.var.set(fixed)
            self.entry.config(state="disabled")

    def _ok(self):
        """Validate, store CI, then exit mainloop but keep root for dialogs."""
        s = (self.var.get() or "").strip()
        try:
            val = float(s)
            if not (0.0 < val < 100.0):
                raise ValueError
        except Exception:
            messagebox.showerror(
                "Invalid value",
                "Please enter a number between 0 and 100 (exclusive).",
                parent=self
            )
            return

        self.result_ci = val

        # ✅ Important: hide window + quit loop, DON'T destroy yet
        self.withdraw()
        self.quit()

    def _cancel(self):
        """Cancel selection: set None and close app."""
        self.result_ci = None
        self.quit()
        self.destroy()


# ── Main ─────────────────────────────────────────────────────────────────────
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd

def main() -> None:
    chooser = CIChooser(default_ci=95.0)
    chooser.mainloop()

    # If user canceled, app is destroyed already
    if chooser.result_ci is None:
        return

    ci_percent = chooser.result_ci
    alpha = 1.0 - (ci_percent / 100.0)
    ci_pct_str = f"{ci_percent:.1f}".rstrip("0").rstrip(".")

    # Use chooser as the hidden root for dialogs
    csv_path = filedialog.askopenfilename(
        parent=chooser,
        title="Select tidy CSV (must contain 'metric' and 'value')",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        chooser.destroy()
        return

    # Put outputs next to the CSV under a unique "regression output" folder
    csv_path = Path(csv_path)
    out_root = unique_subdir(csv_path.parent, "regression output")
    out_root.mkdir(parents=True, exist_ok=False)  # create the unique folder

    # Read data
    df = pd.read_csv(csv_path)

    if "metric" not in df.columns or "value" not in df.columns:
        messagebox.showerror(
            "Missing columns",
            "CSV must include at least 'metric' and 'value'.",
            parent=chooser
        )
        chooser.destroy()
        return

    exclude = set(EXCLUDE_DEFAULT)

    # ---- RUN ANALYSES (this is what was missing) ----
    all_dir = out_root / "ALL"
    ran_all, err_all = run_per_metric_analysis(df, all_dir, alpha=alpha, exclude=exclude)

    ran_onto_total = 0
    onto_folders = 0
    onto_missing = ONTOLOGY_COL not in df.columns

    if not onto_missing:
        onto_vals = sorted(pd.Series(df[ONTOLOGY_COL]).dropna().unique().tolist())
        for ov in onto_vals:
            sub = df[df[ONTOLOGY_COL] == ov].copy()
            folder = out_root / f"Ontology_{safe_folder_name(ov)}"
            ran_sub, _ = run_per_metric_analysis(sub, folder, alpha=alpha, exclude=exclude)
            ran_onto_total += ran_sub
            onto_folders += 1

    msg_lines = [
        f"ALL: processed {ran_all} metric-model(s).",
        f"Model: {'WITH intercept (grand mean)' if INCLUDE_INTERCEPT else 'NO intercept'} + Sum coding",
        f"Implied levels added to plot: {ADD_IMPLIED_SUM_LEVELS_TO_PLOT}",
        f"Confidence interval: {ci_pct_str}%",
        f"Outputs in: {out_root}",
    ]

    if onto_missing:
        msg_lines.append(f"Note: '{ONTOLOGY_COL}' column not found — skipped ontology folders.")
    else:
        msg_lines.append(f"Ontology folders created: {onto_folders}")
        msg_lines.append(f"Ontology folders total metric-model(s): {ran_onto_total}")

    if err_all:
        msg_lines.append("See ALL/_errors.txt for ALL-level notes.")
        messagebox.showwarning("Finished (with notes)", "\n".join(msg_lines), parent=chooser)
    else:
        messagebox.showinfo("Finished", "\n".join(msg_lines), parent=chooser)

    chooser.destroy()



if __name__ == "__main__":
    
    main()
