import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os
import ast
from datetime import datetime
from matplotlib.lines import Line2D
import itertools

# =========================
# GLOBALS
# =========================
df = None
input_file_path = None
analyses = []

# =========================
# FILE LOADER
# =========================
def load_file():
    global df, input_file_path
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        return
    try:
        df = pd.read_csv(file_path)
        #df["Alignment ID"] = df["Alignment ID"].str.rsplit("_", n=1).str[0]
        input_file_path = file_path
        messagebox.showinfo("Success", f"Loaded file:\n{file_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# STATS UTILITIES
# =========================
import numpy as np
from scipy.stats import ttest_rel

def paired_ttest(x, y):
    """
    Standard paired t-test (two-sided).

    Returns
    -------
    t_stat : float
    p_value : float
    n : int
        Number of valid paired observations
    mean_diff : float
        Mean of (y - x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n < 2:
        return np.nan, np.nan, n, np.nan

    # Mean paired difference
    mean_diff = float(np.mean(y - x))

    res = ttest_rel(x, y)
    return float(res.statistic), float(res.pvalue), n, mean_diff

# =========================
# ONTOLOGY STYLES (ported from plots.py)
# =========================
_MARKERS = ['o','s','D','^','v','<','>','*','p','h','X','P','H','d','1','2','3','4','+','x']
_COLORS  = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple',
            'tab:brown','tab:olive','tab:cyan','magenta','goldenrod','teal','slategray']

def build_ontology_styles(categories):
    """
    Map ontology category -> {'color': str, 'marker': str}
    Heuristics:
      - red     : PA/PC/PI/PE/PS/PG/CL
      - blue    : ether/plasmalogen ('ether', 'O-', 'P-')
      - teal    : contains 'd7'
      - green   : Cer or SM
      - magenta : MG/DG/TG
      - others  : rotate remaining tab colors
    """
    cats = [str(c) for c in categories if pd.notna(c)]
    teal = [c for c in cats if 'd7' in c.lower()]
    blue = [c for c in cats if (('ether' in c.lower()) or ('O-' in c) or ('P-' in c)) and c not in teal]
    red_tokens = ['PA','PC','PI','PE','PS','PG','CL']
    red  = [c for c in cats if any(t in c for t in red_tokens) and c not in teal + blue]
    green = [c for c in cats if (('Cer' in c) or ('SM' in c)) and c not in red + blue + teal]
    magenta = [c for c in cats if any(t in c for t in ['MG','DG','TG']) and c not in teal + blue]
    other = [c for c in cats if c not in teal + red + blue + green + magenta]

    ordered = red + blue + teal + green + magenta + other
    marker_cycle = itertools.cycle(_MARKERS)
    default_cycle = itertools.cycle([c for c in _COLORS if c not in ['tab:red','tab:blue','tab:green','magenta','teal']])

    styles = {}
    for c in ordered:
        m = next(marker_cycle)
        if c in red:      col = 'tab:red'
        elif c in blue:   col = 'tab:blue'
        elif c in teal:   col = 'teal'
        elif c in green:  col = 'tab:green'
        elif c in magenta:col = 'magenta'
        else:             col = next(default_cycle)
        styles[c] = {'color': col, 'marker': m}
    return styles

def legend_handles_from_styles(styles, markersize=8):
    return [Line2D([0],[0],
                   marker=sty['marker'],
                   linestyle='',
                   markerfacecolor=sty['color'],
                   markeredgecolor=sty['color'],
                   label=name,
                   markersize=markersize)
            for name, sty in styles.items()]

# =========================
# CONFORMITY (IDENTITY) ANALYSIS
# =========================
def run_conformity_analysis(
    metrics_list,
    index_column,
    filter_col,
    filter_val,
    compare_col,
    group1,
    group2,
    data_override=None,
    log_transform=False
):
    """
    Build paired (group1, group2) values for each metric and compute paired t-tests.
    Returns
    -------
    results : dict with keys:
        'per_metric': DataFrame with columns
            [metric, ontology, n_pairs, mean_g1, mean_g2, mean_diff, std_diff, t_stat, p_value]
        'pairs': dict[metric] -> DataFrame with columns [index_column, group1, group2, diff, Ontology]
        'metric_to_ontology': dict
    """
    global df
    data = data_override if data_override is not None else df
    if data is None:
        messagebox.showerror("Error", "No file loaded.")
        return None

    # Validate columns
    required_cols = ["metric", "value", index_column, compare_col]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        messagebox.showerror("Error", f"Missing required column(s): {', '.join(missing)}")
        return None

    working = data.copy()

    # metrics_list: if empty -> all
    if not metrics_list:
        try:
            metrics_list = sorted(working["metric"].dropna().unique().tolist())
        except Exception:
            messagebox.showerror("Error", "Could not infer 'metrics' from dataframe.")
            return None

    # Filter metrics
    working = working[working["metric"].isin(metrics_list)]

    # Optional filter column
    if filter_col and str(filter_col).lower() != "all":
        if filter_col not in working.columns:
            messagebox.showerror("Error", f"Filter column '{filter_col}' not found in dataframe.")
            return None
        working = working[working[filter_col] == filter_val]

    # Keep only two groups of interest
    if compare_col not in working.columns:
        messagebox.showerror("Error", f"Compare column '{compare_col}' not found in dataframe.")
        return None
    working = working[working[compare_col].isin([group1, group2])]
    if working.empty:
        messagebox.showerror("Error", "Filtered dataframe is empty.")
        return None

    # Aggregate duplicates
    agg_df = (
        working.groupby([index_column, compare_col, "metric"], dropna=False)["value"].mean().reset_index()
    )

    # OPTIONAL log2 transform (only if all values >= 0)
    if log_transform and not (agg_df["value"] < 0).any():
        agg_df["value"] = np.log2(agg_df["value"] + 1.0)

    # Map metric -> Ontology (first seen) for the per-metric summary title/CSV
    if "Ontology" in working.columns:
        meta = (working[['metric', 'Ontology']]
                .dropna(subset=['metric'])
                .drop_duplicates())
        metric_to_ontology = dict(meta.groupby('metric')['Ontology'].first())
    else:
        metric_to_ontology = {}

    # Map (index, metric) -> Ontology to carry into each pairs table (per point)
    if "Ontology" in working.columns:
        onto_map_df = (
            working[[index_column, "metric", "Ontology"]]
            .dropna(subset=["metric"])
            .drop_duplicates(subset=[index_column, "metric"])
        )
        onto_map = {
            (r[index_column], r["metric"]): r["Ontology"]
            for _, r in onto_map_df.iterrows()
        }
    else:
        onto_map = {}

    # Build paired table per metric
    pairs_dict = {}
    rows = []
    for m in metrics_list:
        sub = agg_df[agg_df["metric"] == m]
        pivot = sub.pivot(index=index_column, columns=compare_col, values="value")

        # Keep only rows where both groups present
        if group1 not in pivot.columns or group2 not in pivot.columns:
            n_pairs = 0
            pairs = pd.DataFrame(columns=[index_column, str(group1), str(group2), "diff", "Ontology"])
        else:
            paired = pivot[[group1, group2]].dropna()
            n_pairs = paired.shape[0]
            pairs = paired.reset_index().rename(columns={group1: str(group1), group2: str(group2)})

            # per-sample difference and ontology
            if n_pairs > 0:
                pairs["diff"] = pairs[str(group2)] - pairs[str(group1)]
                # add Ontology per point
                pairs["Ontology"] = pairs[index_column].map(lambda k: onto_map.get((k, m), np.nan))
            else:
                pairs["diff"] = np.nan
                pairs["Ontology"] = np.nan

        if n_pairs >= 2:
            x = pairs[str(group1)].values
            y = pairs[str(group2)].values
            t_stat, p_val, n,d = paired_ttest(x, y)
            mean_diff = y - x
            mean_g1 = float(np.nanmean(x))
            mean_g2 = float(np.nanmean(y))
            mean_diff = float(np.nanmean(d))
            std_diff = float(np.nanstd(d, ddof=1)) if n > 1 else np.nan
        else:
            t_stat = np.nan
            p_val = np.nan
            mean_g1 = float(pairs[str(group1)].mean()) if n_pairs > 0 else np.nan
            mean_g2 = float(pairs[str(group2)].mean()) if n_pairs > 0 else np.nan
            mean_diff = float((pairs[str(group2)] - pairs[str(group1)]).mean()) if n_pairs > 0 else np.nan
            std_diff = float((pairs[str(group2)] - pairs[str(group1)]).std(ddof=1)) if n_pairs > 1 else np.nan

        rows.append({
            "metric": m,
            "ontology": metric_to_ontology.get(m, "Other"),
            "n_pairs": int(n_pairs),
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "t_stat": t_stat,
            "p_value": p_val,
            "difference":mean_diff
        })
        pairs_dict[m] = pairs

    per_metric_df = pd.DataFrame(rows)
    return {
        "per_metric": per_metric_df,
        "pairs": pairs_dict,
        "metric_to_ontology": metric_to_ontology,
    }

# =========================
# ANALYSIS WINDOW
# =========================
def open_analysis_window():
    if df is None:
        messagebox.showerror("Error", "Load a file first.")
        return

    window = tk.Toplevel(root)
    window.title("Add Conformity + Paired t-test Analysis")
    window.geometry("650x740")

    # Dictionary input
    tk.Label(
        window,
        text="Paste Parameter Dictionary:",
        font=("Arial", 10, "bold")
    ).pack(pady=5)
    dict_text = tk.Text(window, height=10, width=78)
    dict_text.pack(pady=5)

    # Form fields
    frame = tk.Frame(window)
    frame.pack(pady=10)

    fields = [
        ("Metrics (comma separated; leave blank for ALL)", "metrics"),
        ("Index Column (pairing key)", "index_column"),
        ("Filter Column", "filter_column"),
        ("Filter Value", "filter_value"),
        ("Compare Column", "compare_column"),
        ("Group 1 (x-axis)", "group1"),
        ("Group 2 (y-axis)", "group2"),
        ("Log2 transform non-negative values? (True/False)", "log_transform"),
    ]
    entries = {}
    for i, (label, key) in enumerate(fields):
        tk.Label(frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(frame, width=50)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    # Helper: load dict into fields
    def load_from_dict():
        try:
            user_dict = ast.literal_eval(dict_text.get("1.0", tk.END).strip())
            entries["metrics"].delete(0, tk.END)
            if "metrics" in user_dict:
                if isinstance(user_dict["metrics"], list):
                    entries["metrics"].insert(0, ", ".join(str(x) for x in user_dict["metrics"]))
                else:
                    entries["metrics"].insert(0, str(user_dict["metrics"]))
            for k in ["index_column", "filter_column", "filter_value", "compare_column", "group1", "group2", "log_transform"]:
                if k in user_dict:
                    entries[k].delete(0, tk.END)
                    entries[k].insert(0, str(user_dict[k]))
            messagebox.showinfo("Success", "Dictionary loaded into fields.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid dictionary format:\n{e}")

    tk.Button(window, text="Load From Dictionary", command=load_from_dict).pack(pady=5)

    def submit():
        # Parse dictionary
        user_dict = {}
        raw = dict_text.get("1.0", tk.END).strip()
        if raw:
            try:
                user_dict = ast.literal_eval(raw)
            except Exception as e:
                messagebox.showerror("Error", f"Invalid dictionary format:\n{e}")
                return

        # Gather parameters (dict has priority if present)
        metrics_input = entries["metrics"].get().strip()
        index_column = entries["index_column"].get().strip() or user_dict.get("index_column")
        filter_column = entries["filter_column"].get().strip() or user_dict.get("filter_column")
        filter_value = entries["filter_value"].get().strip() or user_dict.get("filter_value")
        compare_column = entries["compare_column"].get().strip() or user_dict.get("compare_column")
        group1 = entries["group1"].get().strip() or user_dict.get("group1")
        group2 = entries["group2"].get().strip() or user_dict.get("group2")
        log_transform_str = entries["log_transform"].get().strip() or str(user_dict.get("log_transform", "False"))
        log_transform = str(log_transform_str).lower() in ["true", "1", "yes", "y"]

        # Dict-only fields (optional reshaping)
        feature_id_column = user_dict.get("feature_id_column", "Alignment ID")
        single_measure = user_dict.get("measure", None)
        measure_list = user_dict.get("measure_list", None)

        # Reasonable defaults
        index_column = index_column or "sample_id"
        filter_column = (filter_column or "All")
        filter_value = (filter_value or "All")
        compare_column = compare_column or "group"

        working = df.copy()

        # Safety checks
        for col in [feature_id_column, "metric", "value", index_column, compare_column]:
            if col not in working.columns:
                messagebox.showerror("Error", f"Required column '{col}' not found in dataframe.")
                return
        if single_measure and measure_list:
            messagebox.showerror("Error", "Provide either 'measure' OR 'measure_list', not both.")
            return

        # Reshape according to modes
        if single_measure:
            working = working[working["metric"] == single_measure].copy()
            if working.empty:
                messagebox.showerror("Error", f"No rows found for measure '{single_measure}'.")
                return
            working["metric"] = working[feature_id_column]
        elif measure_list:
            working = working[working["metric"].isin(measure_list)].copy()
            if working.empty:
                messagebox.showerror("Error", f"No rows found for measures: {measure_list}")
                return
            working["metric"] = working[feature_id_column] + "__" + working["metric"]
        # else: use as-is

        if metrics_input:
            metrics_list = [m.strip() for m in metrics_input.split(",") if m.strip()]
        else:
            metrics_list = sorted(working["metric"].dropna().unique().tolist())

        result = run_conformity_analysis(
            metrics_list=metrics_list,
            index_column=index_column,
            filter_col=filter_column,
            filter_val=filter_value,
            compare_col=compare_column,
            group1=group1,
            group2=group2,
            data_override=working,
            log_transform=log_transform,
        )
        if result is None:
            return

        # Title string
        measures_for_title = []
        if single_measure:
            measures_for_title = [str(single_measure)]
        elif measure_list:
            measures_for_title = [str(m) for m in measure_list]
        if measures_for_title:
            measure_part = " + ".join(measures_for_title)
        else:
            if filter_column and filter_column.lower() != "all":
                measure_part = f"{filter_column} = {filter_value}"
            else:
                measure_part = "All metrics"

        title_str = f"{measure_part}\n{group1} vs {group2}\nindex={index_column}"

        analyses.append({
            "data": result,
            "title": title_str,
            "comparison": f"{group1} vs {group2}",
            "groups": (group1, group2),
            "index_column": index_column,
        })
        messagebox.showinfo("Success", "Analysis added.")
        window.destroy()

    # Buttons
    btns = tk.Frame(window)
    btns.pack(pady=10)
    tk.Button(btns, text="Run Analysis", command=submit, height=2, width=16).grid(row=0, column=0, padx=8)
    tk.Button(btns, text="Close", command=window.destroy, height=2, width=12).grid(row=0, column=1, padx=8)

    # Keyboard shortcut
    window.bind("<Control-Return>", lambda e: submit())

# =========================
# PLOTTING (per-point ontology)
# =========================
def plot_all_analyses(
    points_alpha=0.85,
    point_size=40,
    show_reg_line=False,
    annotate_p=True,
    max_cols=3,
    save_figs=True,
    dpi=300,
    show_only_significant=False,
    alpha_sig=0.05,
    save_pairs_csv=True,      # save per-metric pairs with diff
    legend_outside=True,      # put ontology legend under figure
):
    """
    For each added analysis:
    - Create a multi-panel identity (conformity) plot: y=group2, x=group1 per metric
    - Annotate paired t-test p-values on each subplot
    - Save figure + CSV summary next to the input CSV
    - Style each point by its own Ontology with a global legend built from cats used

    Fixes:
    - Force numeric dtype (pd.to_numeric) before np.isfinite / plotting
    - Robust guards for empty/NaN-only slices, regression, and subtitles
    """
    global input_file_path
    if len(analyses) == 0:
        messagebox.showerror("Error", "No analyses added.")
        return

    for i, analysis in enumerate(analyses, start=1):
        data = analysis["data"]
        per_metric = data["per_metric"].copy()
        pairs = data["pairs"]
        title = analysis.get("title", f"Analysis {i}")
        g1, g2 = analysis.get("groups", ("G1", "G2"))

        # Optional significance filter
        if show_only_significant:
            per_metric = per_metric[(per_metric["p_value"].notna()) & (per_metric["p_value"] < alpha_sig)]
            if per_metric.empty:
                messagebox.showwarning("No significant metrics", f"{title}: No metrics with p < {alpha_sig}.")
                continue

        metrics = per_metric["metric"].tolist()
        n = len(metrics)
        cols = min(max_cols, n) if n > 0 else 1
        rows = int(np.ceil(max(n, 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), dpi=dpi)
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        # Collect all ontology categories actually used (for a single legend per figure)
        legend_cats_global = set()

        for j, m in enumerate(metrics):
            ax = axes[j]
            # Ensure expected columns
            ptab = pairs.get(m, pd.DataFrame(columns=[analysis["index_column"], str(g1), str(g2), "diff", "Ontology"])).copy()
            for need in [str(g1), str(g2)]:
                if need not in ptab.columns:
                    ptab[need] = np.nan

            # Determine ontologies present in this panel
            if "Ontology" in ptab.columns and not ptab.empty:
                cats = sorted(pd.Series(ptab["Ontology"]).dropna().unique().tolist())
            else:
                cats = []

            # Build styles for the cats of this panel
            local_styles = build_ontology_styles(cats) if cats else {}

            # Accumulators for axis limits / regression
            combined_x = []
            combined_y = []

            if cats:
                # Plot each ontology subset with its own color/marker
                for c in cats:
                    sub = ptab[ptab["Ontology"] == c]
                    if sub.empty:
                        continue

                    # Force numeric arrays; bad parses -> NaN
                    x = pd.to_numeric(sub[str(g1)], errors='coerce').to_numpy(dtype=float)
                    y = pd.to_numeric(sub[str(g2)], errors='coerce').to_numpy(dtype=float)

                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]

                    if x.size == 0 or y.size == 0:
                        continue

                    ax.scatter(
                        x, y,
                        s=point_size, alpha=points_alpha,
                        color=local_styles[c]["color"], marker=local_styles[c]["marker"],
                        edgecolor="white", linewidth=0.5, label=str(c)
                    )
                    combined_x.append(x)
                    combined_y.append(y)

                legend_cats_global.update(cats)
            else:
                # Fallback: no ontology info -> single color
                x = pd.to_numeric(ptab.get(str(g1), pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
                y = pd.to_numeric(ptab.get(str(g2), pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if x.size > 0 and y.size > 0:
                    ax.scatter(
                        x, y,
                        s=point_size, alpha=points_alpha, color="tab:blue",
                        edgecolor="white", linewidth=0.5
                    )
                    combined_x.append(x)
                    combined_y.append(y)

            # If nothing finite to plot, label and skip axes work
            if len(combined_x) == 0 or len(combined_y) == 0:
                ax.set_title(f"{m}\n(no finite pairs)", fontsize=10)
                ax.set_xlabel(str(g1))
                ax.set_ylabel(str(g2))
                continue

            # Flatten combined arrays
            X = np.concatenate(combined_x)
            Y = np.concatenate(combined_y)

            # Identity line and limits (only if we still have finite data)
            if X.size > 0 and Y.size > 0:
                all_vals = np.concatenate([X, Y])
                if all_vals.size > 0:
                    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                    if not (np.isfinite(vmin) and np.isfinite(vmax)):
                        vmin, vmax = 0.0, 1.0
                    pad = 0.05 * (vmax - vmin + 1e-9)
                    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
                            color="gray", alpha=0.6, linestyle="--", linewidth=1)
                    ax.set_xlim(vmin - pad, vmax + pad)
                    ax.set_ylim(vmin - pad, vmax + pad)

            # Optional regression line
            if show_reg_line:
                xf = X[np.isfinite(X)]
                yf = Y[np.isfinite(Y)]
                if xf.size >= 2 and yf.size >= 2:
                    try:
                        coeffs = np.polyfit(xf, yf, 1)
                        xs = np.linspace(np.nanmin(xf), np.nanmax(xf), 50)
                        ys = coeffs[0] * xs + coeffs[1]
                        ax.plot(xs, ys, color="tab:orange", alpha=0.8, linewidth=1)
                    except Exception:
                        pass  # Be silent if polyfit has trouble with degenerate ranges

            # Titles + labels
            row = per_metric[per_metric["metric"] == m].iloc[0]
            pval = row.get("p_value", np.nan)
            d    = row.get("difference", np.nan)
            n_pairs = int(row["n_pairs"]) if pd.notna(row.get("n_pairs")) else 0
            tstat = row.get("t_stat", np.nan)

            if pd.notna(pval):
                # Safe formatting with nan guards
                d_txt = "NA" if not np.isfinite(d) else f"{float(d):.4g}"
                t_txt = "NA" if not np.isfinite(tstat) else f"{float(tstat):.2f}"
                p_txt = f"{float(pval):.3e}"
                subtitle = f"n={n_pairs}, d={d_txt}, t={t_txt}, p={p_txt}"
            else:
                subtitle = f"n={n_pairs}, t=NA, p=NA"

            ax.set_title(f"{m}\n{subtitle}", fontsize=10)
            ax.set_xlabel(str(g1))
            ax.set_ylabel(str(g2))

        # Remove any unused axes
        for k in range(len(metrics), len(axes)):
            fig.delaxes(axes[k])

        fig.suptitle(f"Conformity (Identity) Plots\n{title}", fontsize=12, y=0.98)

        # Legend: all categories used across panels in this figure
        if legend_cats_global:
            styles = build_ontology_styles(sorted(legend_cats_global))
            handles = legend_handles_from_styles(styles, markersize=8)
            if legend_outside:
                fig.legend(handles=handles, ncol=min(6, max(1, len(handles))),
                           loc='lower center', bbox_to_anchor=(0.5, 0.0))
                plt.tight_layout(rect=[0, 0.06, 1, 0.97])
            else:
                axes[min(len(metrics)-1, len(axes)-1)].legend(
                    handles=handles, ncol=min(3, len(handles)), loc='best'
                )
                plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        else:
            plt.tight_layout(rect=[0, 0.02, 1, 0.97])

        # Save next to CSV
        if save_figs and input_file_path:
            directory = os.path.dirname(input_file_path)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as SVG
            fig_path = os.path.join(directory, f"Conformity_{i}_{ts}.svg")
            fig.savefig(fig_path, format="svg", bbox_inches="tight")

            # Save per-metric stats as CSV
            csv_path = os.path.join(directory, f"Conformity_{i}_stats_{ts}.csv")
            per_metric.to_csv(csv_path, index=False)

            # Save per-pair tables (with diff and Ontology)
            if save_pairs_csv:
                pairs_dir = os.path.join(directory, f"Conformity_{i}_pairs_{ts}")
                os.makedirs(pairs_dir, exist_ok=True)
                for m, ptab in pairs.items():
                    if ptab is not None and not ptab.empty:
                        g1_str, g2_str = str(g1), str(g2)
                        if "diff" not in ptab.columns and g2_str in ptab.columns and g1_str in ptab.columns:
                            ptab = ptab.copy()
                            # Force numeric before computing diff
                            ptab[g1_str] = pd.to_numeric(ptab[g1_str], errors='coerce')
                            ptab[g2_str] = pd.to_numeric(ptab[g2_str], errors='coerce')
                            ptab["diff"] = ptab[g2_str] - ptab[g1_str]
                        safe_metric = str(m).replace(os.sep, "_")
                        ptab.to_csv(os.path.join(pairs_dir, f"{safe_metric}.csv"), index=False)

            messagebox.showinfo("Saved", f"Saved figure and stats to:\n{fig_path}\n{csv_path}")

        plt.show()


# =========================
# MAIN GUI
# =========================
root = tk.Tk()
root.title("Conformity + Paired t-test GUI")
root.geometry("420x300")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(root, text="Load CSV File", command=load_file, height=2).pack(pady=10)

# Provide a short hint label to guide the user on expected columns
hint = (
    "Expected columns: index, metric, value, compare_col (group).\n"
    "Use the parameter dictionary to map column names and groups.\n"
    "If your file includes 'Ontology', points are colored/shaped by lipid class."
)
lab = tk.Label(root, text=hint, fg="gray")
lab.pack(pady=2)

tk.Button(root, text="Add Analysis", command=open_analysis_window, height=2).pack(pady=10)
tk.Button(root, text="Plot & Save All Analyses", command=plot_all_analyses, height=2).pack(pady=10)

root.mainloop()
