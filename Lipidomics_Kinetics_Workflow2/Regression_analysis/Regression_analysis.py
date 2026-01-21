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


# ── Config ────────────────────────────────────────────────────────────────────
ONTOLOGY_COL = "Ontology"

EXCLUDE_DEFAULT = {
    "Alignment ID", "metric", "value", "sample_id", "ci_lower", "ci_upper", ONTOLOGY_COL
}

ADD_PAIRWISE_CATEGORICAL_INTERACTIONS = False
USE_ROBUST_SE = True
MIN_ROWS_PER_MODEL = 6

# Option A: include intercept (grand mean) with effects coding.
INCLUDE_INTERCEPT = True

# Add implied Sum-coded levels (recommended for showing APOE4, etc.)
ADD_IMPLIED_SUM_LEVELS_TO_PLOT = True


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


def pretty_term(name: str) -> str:
    """Make patsy/statsmodels parameter names more readable."""
    name = str(name)
    if name == "Intercept":
        return "Intercept (grand mean)"
    if ":" in name:
        parts = []
        for piece in name.split(":"):
            var, lvl = _parse_c_term(piece)
            parts.append(f"{var} = {lvl}" if var is not None else piece)
        return " × ".join(parts) + " (interaction)"
    var, lvl = _parse_c_term(name)
    if var is not None:
        return f"{var} = {lvl} (deviation from grand mean)"
    return name


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


def add_implied_sum_levels(
    result,
    forest: pd.DataFrame,
    sub_df: pd.DataFrame,
    cat_cols: list[str],
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

    crit = _critical_value_975(result)

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


def diagnostics_plots(model_result, model_name: str, save_to: Path, sub_df: pd.DataFrame, cat_cols: list[str]) -> None:
    """Coefficient forest plot with numeric labels and a key mapping numbers->terms."""
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

        ci_raw = model_result.conf_int()
        if isinstance(ci_raw, np.ndarray):
            ci_df = pd.DataFrame(ci_raw, index=params.index, columns=["lower", "upper"])
        else:
            ci_df = ci_raw.rename(columns={0: "lower", 1: "upper"}).reindex(params.index)

        forest = ci_df.assign(coef=params)

        # Add implied levels so APOE4 etc. shows up
        if ADD_IMPLIED_SUM_LEVELS_TO_PLOT:
            forest = add_implied_sum_levels(model_result, forest, sub_df, cat_cols)

        forest = forest.sort_values("coef")
        n = len(forest)

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

        fig.suptitle(f"Effect sizes – {model_name}", ha="center", y=0.8)

        # Bottom key (make it robust—never let pretty formatting crash plotting)
        term_labels_raw = list(forest.index)
        term_labels = []
        for t in term_labels_raw:
            try:
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
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def run_per_metric_analysis(df_in: pd.DataFrame, out_dir: Path, exclude: set[str]) -> tuple[int, list[str]]:
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
            diagnostics_plots(coef_src, str(metric), out_dir, sub2, cat_cols)
        except Exception as e:
            errors.append(f"[{metric}] plotting failed: {e}")

        ran += 1
        print(f"{out_dir.name} :: {metric}: OK | formula: {formula}")

    if errors:
        (out_dir / "_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    return ran, errors


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    root = tk.Tk()
    root.withdraw()

    csv_path = filedialog.askopenfilename(
        title="Select tidy CSV (must contain 'metric' and 'value')",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        return

    out_root = filedialog.askdirectory(title="Select output root folder")
    if not out_root:
        return
    out_root = Path(out_root)

    df = pd.read_csv(csv_path)

    if "metric" not in df.columns or "value" not in df.columns:
        messagebox.showerror("Missing columns", "CSV must include at least 'metric' and 'value'.")
        return

    # Always exclude Ontology as a predictor
    exclude = set(EXCLUDE_DEFAULT)

    # 1) ALL analysis
    all_dir = out_root / "ALL"
    ran_all, err_all = run_per_metric_analysis(df, all_dir, exclude=exclude)

    # 2) Ontology-specific analyses
    ran_onto_total = 0
    onto_folders = 0
    onto_missing = ONTOLOGY_COL not in df.columns

    if not onto_missing:
        onto_vals = sorted(pd.Series(df[ONTOLOGY_COL]).dropna().unique().tolist())
        for ov in onto_vals:
            sub = df[df[ONTOLOGY_COL] == ov].copy()
            folder = out_root / f"Ontology_{safe_folder_name(ov)}"
            ran_sub, _ = run_per_metric_analysis(sub, folder, exclude=exclude)
            ran_onto_total += ran_sub
            onto_folders += 1

    msg_lines = [
        f"ALL: processed {ran_all} metric-model(s).",
        f"Model: {'WITH intercept (grand mean)' if INCLUDE_INTERCEPT else 'NO intercept'} + Sum coding",
        f"Implied levels added to plot: {ADD_IMPLIED_SUM_LEVELS_TO_PLOT}",
        f"Outputs in: {out_root}",
    ]

    if onto_missing:
        msg_lines.append(f"Note: '{ONTOLOGY_COL}' column not found — skipped ontology folders.")
    else:
        msg_lines.append(f"Ontology folders created: {onto_folders}")
        msg_lines.append(f"Ontology folders total metric-model(s): {ran_onto_total}")

    if err_all:
        msg_lines.append("See ALL/_errors.txt for ALL-level notes.")
        messagebox.showwarning("Finished (with notes)", "\n".join(msg_lines))
    else:
        messagebox.showinfo("Finished", "\n".join(msg_lines))


if __name__ == "__main__":
    main()
