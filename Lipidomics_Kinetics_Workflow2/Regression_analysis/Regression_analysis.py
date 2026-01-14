#!/usr/bin/env python3
# lipid_regression_comparative.py
#
# Comparative per-metric regression with a user-chosen reference:
#   • You choose a column (class) like "genotype" and a reference level like "A3"
#   • For each metric: value ~ C(class, Treatment(reference=ref)) + all other predictors
#   • Extract adjusted contrasts (each other level / reference) per metric
#   • Writes: OLS + HC3 summaries, coeff tables + CI, ANOVA (when feasible),
#             diagnostics PNGs, and one combined comparisons CSV.

from __future__ import annotations
import itertools
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import re
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import math



# ── Config ───────────────────────────────────────────────────────────────────
EXCLUDE_DEFAULT = {"Alignment ID", "metric", "value", "sample_id"}
ADD_PAIRWISE_CATEGORICAL_INTERACTIONS = False  # set True for all pairwise CxC
USE_ROBUST_SE = True                           # also write HC3 summaries
MIN_ROWS_PER_MODEL = 6                         # after NA-drop

# ── Helpers ──────────────────────────────────────────────────────────────────
def q(name: str) -> str:
    """Quote a column name for patsy Q() (handles spaces/special chars)."""
    safe = str(name).replace('"', r'\"')
    return f'Q("{safe}")'

def split_predictors(df: pd.DataFrame, target_class: str):
    """Split predictors into categorical / numeric (excluding EXCLUDE_DEFAULT and the target_class)."""
    candidates = [c for c in df.columns if c not in EXCLUDE_DEFAULT and c != target_class]
    cat_cols, num_cols = [], []
    for c in candidates:
        if isinstance(df[c].dtype, CategoricalDtype):
            cat_cols.append(c); continue
        if df[c].dtype == "object":
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().all():
                num_cols.append(c)
            else:
                df[c] = df[c].astype("category")
                cat_cols.append(c)
        else:
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().all():
                num_cols.append(c)
            else:
                df[c] = df[c].astype("category")
                cat_cols.append(c)
    return cat_cols, num_cols

def drop_degenerate_predictors(sub: pd.DataFrame, cat_cols, num_cols):
    """Remove predictors with no variation within this metric subset."""
    good_cat, good_num = [], []
    for c in cat_cols:
        u = pd.Series(sub[c]).dropna().unique()
        if len(u) >= 2:
            good_cat.append(c)
    for c in num_cols:
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if s.nunique() >= 2 and float(s.var()) > 0.0:
            good_num.append(c)
    return good_cat, good_num

ADD_INTERACTIONS_WITH_TARGET = False  # put near the other config flags

def build_formula(target_class: str, ref_level: str, cat_cols, num_cols) -> str:
    # No intercept -> allows one coefficient per level of target_class
    terms = [f"0 + C({q(target_class)})"]

    # Keep your other predictors as before
    terms += [f"C({q(c)})" for c in cat_cols]
    terms += [q(c) for c in num_cols]

    if ADD_INTERACTIONS_WITH_TARGET:
        for c in cat_cols:
            terms.append(f"C({q(target_class)}):C({q(c)})")

    if ADD_PAIRWISE_CATEGORICAL_INTERACTIONS and len(cat_cols) >= 2:
        for a, b in itertools.combinations(cat_cols, 2):
            terms.append(f"C({q(a)}):C({q(b)})")

    rhs = " + ".join(terms) if terms else "0"
    return f"value ~ {rhs}"



def _baseline_ref_string(model) -> str:
    import re, pandas as pd
    names = getattr(model.model, "exog_names", []) or []
    refs, levels = {}, {}

    # Parse explicit Treatment(reference=...) from exog names
    for name in names:
        m = re.search(
            r'C\((?:Q\("([^"]+)"\)|([^,]+)),\s*Treatment\(reference=(["\'])?([^"\')]+)\3\)\)\[T\.',
            name
        )
        if m:
            var = m.group(1) or m.group(2)
            refs[var] = m.group(4)

    # Collect seen levels for vars without explicit reference
    for name in names:
        m = re.search(r'C\((?:Q\("([^"]+)"\)|([^,]+)).*?\)\[T\.([^\]]+)\]$', name)
        if m:
            var = m.group(1) or m.group(2)
            lvl = m.group(3)
            levels.setdefault(var, set()).add(str(lvl))

    df = getattr(model.model.data, "frame", None)
    if df is not None:
        for var, seen in levels.items():
            if var not in refs and var in df.columns:
                col = df[var]
                if pd.api.types.is_categorical_dtype(col):
                    cats = list(col.cat.categories)
                else:
                    cats = sorted(map(str, pd.Series(col).dropna().unique()))
                missing = [x for x in cats if str(x) not in seen]
                if missing:
                    refs[var] = str(missing[0])

    return ", ".join(f"{k}={v}" for k, v in sorted(refs.items()))

def _superscript_int(n: int) -> str:
    # map digits/signs to Unicode superscripts
    trans = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    return str(n).translate(trans)


def pretty_term(name: str) -> str:
    if name == "Intercept":
        return "Intercept"

    # Match BOTH:
    #   C(Q("sex"))[T.Male]  (with intercept, treatment coding)
    #   C(Q("sex"))[Male]    (no intercept, full indicators)
    m = re.match(r'C\((?:Q\("([^"]+)"\)|([^,]+))(?:,.*)?\)\[(?:T\.)?([^\]]+)\]$', name)
    if m:
        var = m.group(1) or m.group(2)
        lvl = m.group(3)
        return f"{var} = {lvl}"

    # Interactions: split and prettify each piece
    parts = []
    for piece in name.split(":"):
        m2 = re.match(r'C\((?:Q\("([^"]+)"\)|([^,]+))(?:,.*)?\)\[(?:T\.)?([^\]]+)\]$', piece)
        if m2:
            var = m2.group(1) or m2.group(2)
            lvl = m2.group(3)
            parts.append(f"{var} = {lvl}")
    if parts:
        return " × ".join(parts) + " (interaction)"

    return name


def diagnostics_plots(model, model_name: str, save_to: Path) -> None:
    # ---- base font size used everywhere ----
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

    # Prefer a global pretty_term if it exists, but fall back to a local one
    pretty_global = globals().get("pretty_term", None)
    supers = globals().get("_superscript_int", None)

    def _superscript_int_local(n: int) -> str:
        trans = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
        return str(n).translate(trans)

    def pretty_local(name: str) -> str:
        # Intercept (may not exist with "0 + ..." models)
        if name == "Intercept":
            return "Intercept"

        # Match BOTH:
        #   C(Q("sex"))[T.Male]  (treatment coding with intercept)
        #   C(Q("sex"))[Male]    (no intercept / full indicators)
        m = re.match(r'C\((?:Q\("([^"]+)"\)|([^,]+))(?:,.*)?\)\[(?:T\.)?([^\]]+)\]$', name)
        if m:
            var = m.group(1) or m.group(2)
            lvl = m.group(3)
            return f"{var} = {lvl}"

        # Interactions like ...:...
        parts = []
        for piece in name.split(":"):
            m2 = re.match(r'C\((?:Q\("([^"]+)"\)|([^,]+))(?:,.*)?\)\[(?:T\.)?([^\]]+)\]$', piece)
            if m2:
                var = m2.group(1) or m2.group(2)
                lvl = m2.group(3)
                parts.append(f"{var} = {lvl}")
        if parts:
            return " × ".join(parts) + " (interaction)"

        return name

    pretty = pretty_global if callable(pretty_global) else pretty_local

    with plt.rc_context(rc):

        # ── Coefficient forest plot (numeric labels, bottom key) ──────────────
        names = getattr(model.model, "exog_names", None)
        params = pd.Series(np.asarray(model.params), index=names, name="coef")

        ci_raw = model.conf_int()
        if isinstance(ci_raw, np.ndarray):
            ci_df = pd.DataFrame(ci_raw, index=params.index, columns=["lower", "upper"])
        else:
            ci_df = (ci_raw.rename(columns={0: "lower", 1: "upper"})
                           .reindex(params.index))

        forest = ci_df.assign(coef=params).sort_values("coef")
        n = len(forest)

        fig_height = max(3.4, 1.0 + 0.35 * n) * 1.3  # 30% taller
        fig4, ax4 = plt.subplots(figsize=(7.0, fig_height))

        # error bars
        xerr = np.vstack([forest["coef"] - forest["lower"], forest["upper"] - forest["coef"]])
        ax4.errorbar(
            forest["coef"], range(n), xerr=xerr, fmt="o", capsize=3,
            markersize=5, mec="black", mew=0.6
        )
        ax4.axvline(0, color="k", linestyle="--")

        # numeric y ticks, reversed so 1 is at the top
        ax4.set_yticks(range(n))
        ax4.set_yticklabels([str(i) for i in range(1, n + 1)])
        ax4.invert_yaxis()
        ax4.tick_params(axis="both", labelsize=FS)

        # X axis: compact scientific with true superscripts; no offset text
        ax4.set_xlabel("Estimate (95% CI)", fontsize=FS)
        ax4.ticklabel_format(axis="x", style="plain", useOffset=False, useMathText=False)
        ax4.get_xaxis().get_offset_text().set_visible(False)

        x_abs_max = np.nanmax(np.abs(forest[["coef", "lower", "upper"]].to_numpy()))
        if np.isfinite(x_abs_max) and x_abs_max >= 1_000:
            exp = int(3 * math.floor(math.log10(x_abs_max) / 3.0))
            scale = 10.0 ** exp
            ax4.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/scale:.1f}"))
            sup = supers or _superscript_int_local
            ax4.set_xlabel(f"Estimate (95% CI)  ×10{sup(exp)}", fontsize=FS)
        else:
            ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # title (no baseline line; works for both intercept and no-intercept models)
        fig4.suptitle(f"Effect sizes – {model_name}", ha="center", y=0.8, fontsize=FS)

        # ----- bottom key: numbers -> pretty term names (multi-column) -----
        term_labels_raw = list(forest.index)
        term_labels = [pretty(t) for t in term_labels_raw]

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

            # dynamic bottom padding based on tallest column
            fs_key = FS
            lines = max(len(b.splitlines()) for b in cols) + 1  # +1 for "Key:"
            line_h_in = fs_key / 72.0 * 1.25
            _, fig_h = fig4.get_size_inches()
            bottom_pad = min(0.50, max(0.18, (lines * line_h_in) / fig_h + 0.05))

            fig4.tight_layout(rect=[0.0, bottom_pad, 1.0, 0.86])

            y_top = bottom_pad - 0.02
            fig4.text(0.02, y_top, "Key:", ha="left", va="top",
                      weight="bold", fontsize=fs_key)
            xs = np.linspace(0.10, 0.98, len(cols), endpoint=True)
            for x, block in zip(xs, cols):
                fig4.text(x, y_top, block, ha="left", va="top",
                          family="monospace", fontsize=fs_key)
        else:
            fig4.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])

        # save without clipping
        fig4.canvas.draw()
        fig4.savefig(save_to / f"{model_name}_coeff_forest.png",
                     dpi=300, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig4)




# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    root = tk.Tk(); root.withdraw()

    csv_path = filedialog.askopenfilename(
        title="Select tidy CSV (must contain 'metric' and 'value')",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        return
    out_dir = filedialog.askdirectory(title="Select output folder")
    if not out_dir:
        return
    out_dir = Path(out_dir)

    df = pd.read_csv(csv_path)

    if "metric" not in df.columns or "value" not in df.columns:
        messagebox.showerror("Missing columns", "CSV must include at least 'metric' and 'value'.")
        return
    if not isinstance(df["metric"].dtype, CategoricalDtype):
        df["metric"] = df["metric"].astype("category")

    # --- Ask for the class/column (case-insensitive match to columns)
    target_class_raw = simpledialog.askstring(
        "Class/column",
        "Enter the class/column to compare within (e.g., genotype, sex, treatment):"
    )
    if not target_class_raw:
        messagebox.showinfo("Cancelled", "No class/column provided."); return

    col_map = {c.lower(): c for c in df.columns}
    if target_class_raw.lower() not in col_map:
        messagebox.showerror("Not found",
                             f"Column '{target_class_raw}' not found.\n\nAvailable:\n" +
                             ", ".join(sorted(df.columns)))
        return
    target_class = col_map[target_class_raw.lower()]

    # Make target_class categorical if appropriate
    if not isinstance(df[target_class].dtype, CategoricalDtype):
        if df[target_class].dtype == "object":
            as_num = pd.to_numeric(df[target_class], errors="coerce")
            if not as_num.notna().all():
                df[target_class] = df[target_class].astype("category")
        if not isinstance(df[target_class].dtype, CategoricalDtype):
            df[target_class] = df[target_class].astype("category")

    # --- Build a safe case-insensitive level map (NO .astype(str).cat combo)
    vals = df[target_class]
    if isinstance(vals.dtype, CategoricalDtype):
        available_levels = [str(x) for x in vals.cat.categories]
    else:
        available_levels = sorted([str(x) for x in vals.dropna().unique()])
    lvl_map = {lvl.lower(): lvl for lvl in available_levels}

    # Ask for the reference level
    ref_raw = simpledialog.askstring(
        "Reference indicator",
        f"Enter the reference level for '{target_class}'.\n"
        f"Available (examples): {', '.join(available_levels[:10])}"
    )
    if not ref_raw:
        messagebox.showinfo("Cancelled", "No reference level provided."); return
    if ref_raw.lower() not in lvl_map:
        messagebox.showerror(
            "Reference not found",
            f"Reference '{ref_raw}' not found in '{target_class}'.\n"
            f"Available: {', '.join(available_levels)}"
        )
        return
    ref_level = lvl_map[ref_raw.lower()]

    # Normalize other predictors lightly
    for c in [c for c in df.columns if c not in EXCLUDE_DEFAULT]:
        if isinstance(df[c].dtype, CategoricalDtype):
            continue
        if df[c].dtype == "object":
            as_num = pd.to_numeric(df[c], errors="coerce")
            if not as_num.notna().all():
                df[c] = df[c].astype("category")

    metrics = list(df["metric"].cat.categories)
    errors = []
    ran = 0

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()

        # Ensure target_class acts categorical in this subset
        if not isinstance(sub[target_class].dtype, CategoricalDtype):
            sub[target_class] = sub[target_class].astype("category")

        # Need reference + at least one other level in this subset
        present_levels = [str(x) for x in sub[target_class].dropna().unique().tolist()]
        if ref_level not in present_levels:
            errors.append(f"[{metric}] reference '{ref_level}' not present in this metric; skipped.")
            continue
        if len(set(present_levels)) < 2:
            errors.append(f"[{metric}] only one '{target_class}' level present; skipped.")
            continue

        # Detect & prune other predictors on this subset
        cat_cols_all, num_cols_all = split_predictors(sub, target_class)
        cat_cols, num_cols = drop_degenerate_predictors(sub, cat_cols_all, num_cols_all)

        # Build comparative model
        formula = build_formula(target_class, ref_level, cat_cols, num_cols)

        # Drop NAs required for this model
        needed_cols = ["value", target_class] + cat_cols + num_cols
        sub2 = sub.dropna(subset=needed_cols)
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

        # Save summaries
        coef_src = mdl
        if USE_ROBUST_SE:
            try:
                rob = mdl.get_robustcov_results(cov_type="HC3")
                (out_dir / f"{metric}_robust_summary.txt").write_text(rob.summary().as_text(), encoding="utf-8")
                coef_src = rob
            except Exception as e:
                errors.append(f"[{metric}] robust SE failed: {e}")


        # ANOVA if model includes predictors
        try:
            if formula.strip() != "value ~ 1":
                an = anova_lm(mdl, typ=2)
                an.to_csv(out_dir / f"{metric}_anova.csv")
        except Exception as e:
            errors.append(f"[{metric}] ANOVA failed: {e}")

        # Diagnostics
        try:
            diagnostics_plots(coef_src, str(metric), out_dir)
        except Exception as e:
            errors.append(f"[{metric}] plotting failed: {e}")

        ran += 1
        print(f"{metric}: OK  |  formula: {formula}")


    # Wrap up
    if errors:
        (out_dir / "_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        messagebox.showwarning(
            "Finished (with notes)",
            f"Processed {ran}/{len(metrics)} metrics.\n"
            f"Outputs in: {out_dir}\n"
            f"Combined contrasts: comparisons_{target_class}_{ref_level}.csv\n"
            f"See _errors.txt for details."
        )
    else:
        messagebox.showinfo(
            "Finished",
            f"Processed {ran}/{len(metrics)} metrics.\n"
            f"Outputs in: {out_dir}\n"
            f"Combined contrasts: comparisons_{target_class}_{ref_level}.csv"
        )
        
        
    # Pop any figures (non-blocking)
    plt.show(block=False)
    
    
    

if __name__ == "__main__":
    main()
