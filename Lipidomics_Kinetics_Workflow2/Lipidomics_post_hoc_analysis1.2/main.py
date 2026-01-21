# -*- coding: utf-8 -*-
"""
Main part of the post hoc analysis code. Connects the gui and starts the plots.
"""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Set, Dict, Any
import pandas as pd
import ast
from prep import Experiment
from plots import create_plots 
from gui import launch_gui, set_on_analyze, close_window
import copy
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# OR: ignore by category (safer)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Pandas-specific PerformanceWarning
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)


GROUPS = ("A2","A3","A4")  # change if your group labels differ

def generate_comparison_tables(stats_df: pd.DataFrame, output_dir: str = ".", font_size: int = 10, alpha: float = 0.05):
    """
    Create one CSV per comparison with two files:
      • <comparison>_Ontology.csv        (rows: Plot_Group starting with 'Ontology')
      • <comparison>_HighOrder_All.csv   (rows: 'All' and Plot_Group starting with 'HighOrder')

    Row inclusion rule:
      • Include ONLY groups (lipids/ontologies) that have at least ONE metric with p_All <= alpha.
        Groups with no significant metrics are excluded entirely.

    Columns (wide format):
      Mean_Diff_<Metric>, t_<Metric>, p_<Metric>, N_<Metric>
      (the '_All' part is removed)

    Notes:
      • Metrics are grouped side-by-side (Δ/t/p/N neighbors per metric).
      • Metric labels are lightly normalized (e.g., 'N-value'→'nL', verbose Flux labels→'Flux').
      • Exports CSVs, not XLSX.
    """
    import os
    import pandas as pd

    required = {"Comparison", "Plot_Group", "Metric", "Mean_Diff_All", "t_All", "p_All", "N_All"}
    missing = required - set(stats_df.columns)
    if missing:
        print(f"[generate_comparison_tables] Missing columns: {missing}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = stats_df.copy()

    # Normalize metric names to compact forms
    metric_map = {
        "N-value Paired t-test and Conformity": "nL",
        "N-value": "nL",
        "nl": "nL",
        "DR-Abundance Paired t-test and Conformity": "Abundance",
        "Asymptote Paired t-test and Conformity": "Asymptote",
        "Rate Paired t-test and Conformity": "Rate",
        "Flux Paired t-test and Conformity": "Flux",
        "flux paired t-test and conformity": "Flux",
    }
    df["Metric"] = df["Metric"].map(lambda s: metric_map.get(str(s), str(s)))

    # Ensure p-values are numeric
    df["p_All"] = pd.to_numeric(df["p_All"], errors="coerce")

    def _filter_groups_with_sig(df_part: pd.DataFrame) -> pd.DataFrame:
        """Keep only Plot_Groups where ANY metric has p_All <= alpha."""
        if df_part.empty:
            return df_part
        keep_mask = df_part.groupby("Plot_Group")["p_All"].transform(lambda s: (s <= alpha).any())
        return df_part[keep_mask]

    def _build_summary(df_part: pd.DataFrame) -> pd.DataFrame:
        """Pivot to wide, strip `_All` from stats, group per-metric columns, rename Plot_Group→Group."""
        if df_part.empty:
            return pd.DataFrame()

        wide = df_part.pivot_table(
            index="Plot_Group",
            columns="Metric",
            values=["Mean_Diff_All", "t_All", "p_All", "N_All"],
            aggfunc="first",
        )

        # Flatten to 'stat_metric' then remove the `_All` segment
        flat_cols = []
        for stat, metric in wide.columns:
            name = f"{stat}_{metric}"  # e.g., Mean_Diff_All_Abundance
            name = (name
                    .replace("Mean_Diff_All_", "Mean_Diff_")
                    .replace("t_All_", "t_")
                    .replace("p_All_", "p_")
                    .replace("N_All_", "N_"))
            flat_cols.append(name)
        wide.columns = flat_cols
        wide.reset_index(inplace=True)

        # Order columns: Group, then (Mean_Diff, t, p, N) for each metric
        metrics = sorted({c.split("_", 1)[1] for c in wide.columns if c != "Plot_Group"})
        ordered = ["Plot_Group"]
        for m in metrics:
            for stat in ["Mean_Diff", "t", "p", "N"]:
                col = f"{stat}_{m}"
                if col in wide.columns:
                    ordered.append(col)
        wide = wide[ordered]

        wide.rename(columns={"Plot_Group": "Group"}, inplace=True)
        return wide.sort_values("Group")

    # Main loop by comparison
    for comparison, df_comp in df.groupby("Comparison"):
        comp_safe = comparison.replace(" vs ", "_vs_").replace(" ", "_")

        # Split groups
        df_high_raw = df_comp[
            df_comp["Plot_Group"].str.contains("HighOrder", na=False) | (df_comp["Plot_Group"] == "All")
        ]
        df_onto_raw = df_comp[df_comp["Plot_Group"].str.contains("Ontology", na=False)]

        # Filter to groups with ≥1 significant metric
        df_high = _filter_groups_with_sig(df_high_raw)
        df_onto = _filter_groups_with_sig(df_onto_raw)

        # Pivot & export
        high_table = _build_summary(df_high)
        onto_table = _build_summary(df_onto)

        if not onto_table.empty:
            onto_csv = os.path.join(output_dir, f"{comp_safe}_Ontology.csv")
            onto_table.to_csv(onto_csv, index=False)
            print(f"✅ Exported: {onto_csv}")
        else:
            print(f"ℹ️ No Ontology groups with p ≤ {alpha} for {comparison} — skipping Ontology CSV.")

        if not high_table.empty:
            high_csv = os.path.join(output_dir, f"{comp_safe}_HighOrder_All.csv")
            high_table.to_csv(high_csv, index=False)
            print(f"✅ Exported: {high_csv}")
        else:
            print(f"ℹ️ No HighOrder/All groups with p ≤ {alpha} for {comparison} — skipping HighOrder CSV.")

    print("[generate_comparison_tables] Finished exporting CSVs.")



def pivot_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Make wide table: index=analyte_id, columns=A2/A3/A4 for `value_col`."""
    w = df.pivot_table(index="analyte_id", columns="group_name",
                       values=value_col, aggfunc="first")
    for g in GROUPS:
        if g not in w.columns:
            w[g] = np.nan
    return w[list(GROUPS)]

def robust_z(s: pd.Series) -> tuple[pd.Series, float, float]:
    """MAD-based robust z-score (0.6745*MAD makes it ~std under normality)."""
    x = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index), med, mad
    z = 0.67448975 * (x - med) / mad
    return pd.Series(z, index=x.index), float(med), float(mad)


def versioned_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    base = path
    i = 2
    while True:
        candidate = f"{base} ({i})"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def build_final_dataframe(experiments: list) -> pd.DataFrame:
    frames = []
    for exp in experiments:
        df = exp.df.copy()
        df.insert(0, "Pair", f"{exp.experimental_identifier}_vs_{exp.control_identifier}")
        df.insert(1, "Experiment_ID", exp.experimental_identifier)
        df.insert(2, "Control_ID", exp.control_identifier)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def get_string_tuples(message: str) -> List[Tuple[str, str]]:
    import re
    while True:
        root = tk.Tk()
        root.withdraw()
        from tkinter.simpledialog import askstring
        user_input = askstring("ID pairs", message)
        root.destroy()
        if user_input is None:
            raise SystemExit("Canceled by user.")
        raw = user_input.strip()
        pairs = re.findall(r'\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)', raw)
        if len(pairs) >= 1:
            return [(a.strip(), b.strip()) for a,b in pairs]
        else:
            print("Invalid format. Please try again (e.g., (A2,A3),(A4,E3)).")


def select_csv_files(title: str = "Select CSV files") -> List[str]:
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files","*.*")]
    )
    root.destroy()
    if not file_paths:
        raise SystemExit("No files selected.")
    return list(file_paths)


def _parse_metrics_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    lines = []
    for line in text.splitlines():
        if "#" in line:
            line = line[:line.index("#")]
        if line.strip():
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return {}
    try:
        obj = ast.literal_eval(cleaned)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            obj = ast.literal_eval("{%s}" % cleaned)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _maybe_add_filters(kwargs: Dict[str, Any], metrics_item: Dict[str, Any], conformity: bool):
    if conformity:
        primary_text = metrics_item.get("primary", "")
        secondary_text = metrics_item.get("secondary", "")
        p = _parse_metrics_text(primary_text)
        s = _parse_metrics_text(secondary_text)
        if p:
            kwargs["primary_filters"] = p
        if s:
            kwargs["secondary_filters"] = s
    else:
        text = metrics_item.get("text", "")
        d = _parse_metrics_text(text)
        if d:
            kwargs["primary_filters"] = d
    if "custom_column" in metrics_item:
        kwargs["custom_analysis_column"] = metrics_item["custom_column"]



def _volcano_fixed_xy(section_name: str):
    s = section_name.lower()
    if s == "abundance":
        return ("log2_abn_FC", "-log10abnBH", dict(title="DR Abundance Volcano", x_size=4, y_size=4))

    if s == "rate":
        return ("log2_rate_FC", "-log10rate_P", dict(title="Rate Volcano", x_size=4, y_size=4))
    if s == "asymptote":
        return ("asymptote_difference", "-log10_asymptote_p", dict(title="Asymptote Volcano", x_size=0.75))
    if s in ("nl", "n l", "n_l", "n-value", "n-value".lower()):
        return ("n_val_fraction_difference", "-log10n_val_p", dict(title="N-value Volcano", x_size=2, y_size=4))
    if s == "flux":
        return ("log2_flux_FC", "-log10flux_p", dict(title="Total Flux Volcano", x_size=3, y_size=4))
    if s in ("synthesis flux", "synth_flux", "synth flux"):
        return ("log2_synth_flux_FC", "-log10synth_flux_p", dict(title="Synthesis Flux Volcano", x_size=3, y_size=4))
    if s in ("dietary flux", "diet_flux", "diet flux"):
        return ("log2_diet_flux_FC", "-log10diet_flux_p", dict(title="Dietary Flux Volcano", x_size=3, y_size=4))

    
    return None


def _conformity_fixed(section_name: str):
    s = section_name.lower()
    if s in ("nl", "n l", "n_l", "n-value"):
        return dict(
            x_col="n_value_Control",
            y_col="n_value_Experiment",
            title="N-value Paired t-test and Conformity",
            axis_titles=[r"$n_L$", r"$n_L$"],
            Experiment_lower="n_val_lower_margin_Experiment",
            Experiment_upper="n_val_upper_margin_Experiment",
            Control_lower="n_val_lower_margin_Control",
            Control_upper="n_val_upper_margin_Control",
            ensure_same_axis=True,
            drop_duplicates_by = 'Lipid Unique Identifier'
        )
    if s == "asymptote":
        return dict(
            x_col="Abundance asymptote_Control",
            y_col="Abundance asymptote_Experiment",
            title="Asymptote Paired t-test and Conformity",
            axis_titles=["Asymptote", "Asymptote"],
            Experiment_lower="Abundance 95pct_confidence_A_Experiment",
            Experiment_upper="Abundance 95pct_confidence_A_Experiment",
            Control_lower="Abundance 95pct_confidence_A_Control",
            Control_upper="Abundance 95pct_confidence_A_Control",
            ensure_same_axis=True,
            drop_duplicates_by = 'Lipid Unique Identifier'
        )
    if s == "abundance":
        return dict(
            x_col="Control_abundance_median",
            y_col="Experiment_abundance_median",
            title="Abundance Paired t-test",
            axis_titles=["Control Median Abundance", "Experiment Median Abundance"],
            ensure_same_axis=True,
        )


    if s == "rate":
        return dict(
            x_col="Abundance rate_Control",
            y_col="Abundance rate_Experiment",
            title="Rate Paired t-test and Conformity",
            axis_titles=["%Turnover/day", "%Turnover/day"],
            Experiment_lower="%Abundance 95pct_confidence_K_Experiment",
            Experiment_upper="%Abundance 95pct_confidence_K_Experiment",
            Control_lower="%Abundance 95pct_confidence_K_Control",
            Control_upper="%Abundance 95pct_confidence_K_Control",
            ensure_same_axis=True,  
            drop_duplicates_by = 'Lipid Unique Identifier'
        )
    
    if s == "flux":
        return dict(
            x_col="Flux_Control",
            y_col="Flux_Experiment",
            title="Flux Paired t-test and Conformity",
            axis_titles=["Flux (Control)", "Flux (Experiment)"],
            ensure_same_axis=True
        )

    if s in ("synthesis flux", "synth flux", "synth_flux"):
        return dict(
            x_col="synth_flux_Control",
            y_col="synth_flux_Experiment",
            title="Synthesis Flux Paired t-test and Conformity",
            axis_titles=["Synthesis Flux (Control)", "Synthesis Flux (Experiment)"],
            ensure_same_axis=True
        )

    if s in ("dietary flux", "diet flux", "diet_flux"):
        return dict(
            x_col="diet_flux_Control",
            y_col="diet_flux_Experiment",
            title="Dietary Flux Paired t-test and Conformity",
            axis_titles=["Dietary Flux (Control)", "Dietary Flux (Experiment)"],
            ensure_same_axis=True
        )

    
    
    return None



def run_plots_from_gui_config(cfg: Dict[str, Any], experiments, plots_dir) -> pd.DataFrame:

    collected_stats = []

    def _build_plot_groups(experiments):
        ontology_values = set()
        for exp in experiments:
            if 'Ontology' in getattr(exp, "df", pd.DataFrame()).columns:
                ontology_values.update(exp.df['Ontology'].dropna().astype(str).unique().tolist())
        ontology_list = sorted(ontology_values)

        higher_order = ['Standards', 'glycerolipids', 'lysos', 'Ethers',
                        'glycerophospholipids', 'sphingolipids']

        groups = []
        groups.append(("All", None))
        for v in ontology_list:
            groups.append((f"Ontology_{v}", {"Ontology": v}))
        for h in higher_order:
            # add only if any experiment contains that column
            if any(h in getattr(exp, "df", pd.DataFrame()).columns for exp in experiments):
                groups.append((f"HighOrder_{h}", {h: True}))
        return groups

    def _filter_experiments_for_group(experiments, extra_filter):
        out = []
        for exp in experiments:
            df = getattr(exp, "df", pd.DataFrame())
            if extra_filter is None:
                filtered_df = df.copy()
            else:
                mask = pd.Series(True, index=df.index)
                for col, val in extra_filter.items():
                    if col not in df.columns:
                        mask &= False
                        continue
    
                    # ✅ Inclusive match for ontology labels (e.g., PE matches LPE, EtherPE, etc.)
                    if col.lower() == "ontology":
                        mask &= df[col].astype(str).str.contains(str(val), case=False, regex=False)
                    # ✅ Boolean flags (e.g., HighOrder_glycerolipids)
                    elif val is True:
                        mask &= df[col].astype(bool)
                    # ✅ Default exact match for everything else
                    else:
                        mask &= (df[col].astype(str) == str(val))
    
                filtered_df = df.loc[mask].copy()
    
            new_exp = copy.copy(exp)
            setattr(new_exp, "df", filtered_df)
            out.append(new_exp)
        return out


    groups = _build_plot_groups(experiments)


    for group_name, extra_filter in groups:
        safe_name = str(group_name).replace(" ", "_")

        fed_experiments = _filter_experiments_for_group(experiments, extra_filter)
        any_rows = any((getattr(e, "df", pd.DataFrame()).shape[0] > 0) for e in fed_experiments)
        if not any_rows:
            print(f"[plot] skipping group '{group_name}' — no rows selected")
            continue

        group_plots_dir = os.path.join(plots_dir, safe_name)
        os.makedirs(group_plots_dir, exist_ok=True)

        # Homeostasis plot
        homeo = cfg.get("Homeostasis", {})
        if homeo.get("enabled"):
            create_plots(
                experiments=fed_experiments,
                analysis_type="homeostasis",
                x_col="log2_abn_FC",
                y_col="log2_rate_FC",
                title=f"Homeostasis: Log₂ FC Abundance vs Rate ({group_name})",
                axis_titles=["log₂ FC Abundance", "log₂ FC Rate"],
                output_dir=group_plots_dir,
                ensure_same_axis=True,
                significance_column1=homeo.get("significance_column1", "abn_Overall_significant"),
                significance_column2=homeo.get("significance_column2", "rate_Overall_significant"),
                drop_duplicates_by = 'Lipid Unique Identifier'
            )

        #Volcano plot
        volc = cfg.get("Volcano", {})
        if volc.get("enabled"):
            metrics = volc.get("metrics", volc.get("filters", {}))
            for section_name, item in (metrics or {}).items():
                if not item.get("enabled"):
                    continue
                fixed = _volcano_fixed_xy(section_name)
                if not fixed:
                    continue
                x_col, y_col, extras = fixed
                kwargs = dict(
                    experiments=fed_experiments,
                    analysis_type="volcano",
                    x_col=x_col,
                    y_col=y_col,
                    FC_cut_off=1.0,
                    stats_cut_off=1.3,
                    output_dir=group_plots_dir,
                    **extras,
                )

                _maybe_add_filters(kwargs, item, conformity=False)
                res = create_plots(**kwargs)
                if isinstance(res, dict) and isinstance(res.get("statistics_df"), pd.DataFrame):
                    collected_stats.append(res["statistics_df"])

        # Conformity plot and scatter-ttest
        conf = cfg.get("Conformity", {})
        if conf.get("enabled"):
            metrics = conf.get("metrics", conf.get("filters", {}))
            for section_name, item in (metrics or {}).items():
                if not item.get("enabled"):
                    continue
                fixed = _conformity_fixed(section_name)
                if not fixed:
                    # Allow YAML-defined plots (like "Other")
                    manual_keys = {k: v for k, v in item.items() if k in ("x_col", "y_col", "title", "axis_titles")}
                    if manual_keys:
                        fixed = manual_keys
                    else:
                        continue
                
                kwargs = dict(
                    experiments=fed_experiments,
                    analysis_type="scatter_ttest",
                    output_dir=group_plots_dir,
                    **fixed,
                )
                _maybe_add_filters(kwargs, item, conformity=True)
                res = create_plots(**kwargs, plot_group=group_name)
                if isinstance(res, dict) and isinstance(res.get("statistics_df"), pd.DataFrame):
                    collected_stats.append(res["statistics_df"])


    
    if collected_stats:
        stats_df = pd.concat(collected_stats, ignore_index=True)
        # Sort by p_All ascending (most significant first)
        if "p_All" in stats_df.columns:
            stats_df = stats_df.sort_values(by="p_All", ascending=True, na_position="last").reset_index(drop=True)
    else:
        stats_df = pd.DataFrame()
    
    return stats_df







def main():
    # 1) Let user pick CSVs
    file_paths = select_csv_files("Select one or more CSV(s) to analyze")

    # 2) Ask for experiment-control pairs
    pairs = get_string_tuples("Enter pairs like: (A2,A3),(A4,E3)")
    all_ids: Set[str] = set([x for tup in pairs for x in tup])

    # 3) Create an output folder next to the first file
    data_root = os.path.dirname(file_paths[0])
    parent_base = os.path.join(data_root, "Analysis")
    parent_dir = versioned_dir(parent_base)
    plots_dir = os.path.join(parent_dir, "plot_outputs")
    os.makedirs(plots_dir, exist_ok=False)

    # Containers to receive results built inside the GUI callback
    stats_container: Dict[str, Optional[pd.DataFrame]] = {"df": None}
    experiments_container: Dict[str, Optional[List[Experiment]]] = {"list": None}

    # Normalization dataframe (shared across experiments)
    NORMALIZATION_DF: Optional[pd.DataFrame] = None

    # --------------------------------------------------
    # GUI
    # --------------------------------------------------
    def establish_plots_with_gui():
        def on_analyze(config_tree: Dict[str, Any]):
            nonlocal NORMALIZATION_DF

            try:
                # ----------------------------------
                # mTIC abundance normalization
                # ----------------------------------
                do_norm = (
                    config_tree
                    .get("Settings", {})
                    .get("perform_abundance_normalization", False)
                )

                if do_norm:
                    norm_path = filedialog.askopenfilename(
                        title="Select normalization dataframe",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                    )

                    if not norm_path:
                        raise RuntimeError(
                            "Abundance normalization was enabled, but no normalization file was selected."
                        )

                    NORMALIZATION_DF = pd.read_csv(norm_path)

                    print(
                        f"[GUI] ✅ Abundance normalization enabled "
                        f"({len(NORMALIZATION_DF):,} rows loaded from {norm_path})"
                    )
                else:
                    NORMALIZATION_DF = None
                    print("[GUI] Abundance normalization DISABLED")

                # ----------------------------------
                # Standards normalization settings
                # ----------------------------------
                settings = config_tree.get("Settings", {})
                
                use_standards_norm = bool(
                    settings.get("use_standards_normalization", False)
                )
                
                standards_baseline_group = settings.get(
                    "standards_baseline_group",
                    "A3"
                )

                print(
                    f"[GUI] Standards normalization "
                    f"{'ENABLED' if use_standards_norm else 'DISABLED'}"
                )
                
                if use_standards_norm:
                    print(
                        f"[GUI] Standards baseline group: "
                        f"{standards_baseline_group}"
                    )
                

                # ----------------------------------
                # Build experiments
                # ----------------------------------
                experiments: List[Experiment] = []

                for idx, pair in enumerate(pairs, start=1):
                    exp = Experiment(
                        file_paths=file_paths,
                        pair=pair,
                        all_ids=all_ids,
                        number=idx,
                        total=len(pairs),
                        normalization_df=NORMALIZATION_DF,
                        normalize_by_standards= use_standards_norm,
                        baseline = standards_baseline_group
                    )
                    experiments.append(exp)

                experiments_container["list"] = experiments

                # ----------------------------------
                # Run plots according to GUI config
                # ----------------------------------
                stats_df = run_plots_from_gui_config(
                    config_tree,
                    experiments,
                    plots_dir
                )

                if isinstance(stats_df, dict) and "statistics_df" in stats_df:
                    stats_df = stats_df["statistics_df"]

                if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                    stats_container["df"] = stats_df
                    print(f"[GUI] ✅ Collected {len(stats_df):,} rows of statistical output.")
                else:
                    stats_container["df"] = None
                    print("[GUI] ⚠️ No statistical results were produced.")

                # ----------------------------------
                # Completion popup
                # ----------------------------------
                from gui import _ROOT
                from tkinter import messagebox

                if _ROOT is not None:
                    _ROOT.after(
                        0,
                        lambda msg=(
                            "✅ All selected plots and statistics have been generated.\n\n"
                            f"Plots saved to:\n{plots_dir}\n\n"
                            "Please close the window using the 'X' to finish."
                        ): messagebox.showinfo("Analysis Complete", msg)
                    )

                print(f"[GUI] ✅ Analysis complete. Plots saved to: {plots_dir}")

                # ----------------------------------
                # Write final merged dataframe
                # ----------------------------------
                final_df = build_final_dataframe(experiments)

                if not final_df.empty:
                    final_out = os.path.join(plots_dir, "final_results.csv")
                    final_df.to_csv(final_out, index=False)
                    print(f"[GUI] ✅ Final dataframe written to: {final_out}")
                else:
                    print("[GUI] ⚠️ Final dataframe empty — nothing written.")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Analysis failed: {e}")

        # Register and launch GUI
        set_on_analyze(on_analyze)
        launch_gui()

    # Launch GUI
    establish_plots_with_gui()

    # --------------------------------------------------
    # Post-GUI cleanup + exports
    # --------------------------------------------------
    statistics_df = stats_container.get("df", None)

    stats_csv = os.path.join(parent_dir, "paired_ttest_statistics.csv")
    try:
        if isinstance(statistics_df, pd.DataFrame) and not statistics_df.empty:
            generate_comparison_tables(statistics_df, output_dir=parent_dir)
            statistics_df.to_csv(stats_csv, index=False)
            print(f"Saved paired-t summary: {stats_csv}")
        else:
            print("No paired-t statistics to save.")
    except Exception as e:
        print(f"Warning: could not save statistics_df: {e}")

    experiments = experiments_container.get("list", None)
    if experiments:
        try:
            final_df = build_final_dataframe(experiments)
            final_csv = os.path.join(parent_dir, "final_dataframe.csv")
            final_df.to_csv(final_csv, index=False)
            print(f"Saved final dataframe: {final_csv}")
        except Exception as e:
            print(f"Warning: could not save final_df: {e}")
    else:
        print("No experiments were built. Skipping final dataframe save.")

    print(f"Done.\nParent folder: {parent_dir}\nPlots:         {plots_dir}")



if __name__ == '__main__':
    main()
