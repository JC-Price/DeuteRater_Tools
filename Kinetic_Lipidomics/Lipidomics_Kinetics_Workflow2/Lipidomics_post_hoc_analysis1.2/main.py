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
from gui import update_progress
import seaborn as sns
import re



import numpy as np
import pandas as pd
from scipy.stats import chi2
import regulation_table as reg



def passes_n_cutoff(n_value, cutoff=10):
    if pd.isna(n_value):
        return False

    # Single number
    if isinstance(n_value, (int, float, np.integer, np.floating)):
        return n_value >= cutoff

    # List or tuple
    if isinstance(n_value, (list, tuple)):
        try:
            return all(float(x) >= cutoff for x in n_value)
        except:
            return False

    # String like "12,10"
    if isinstance(n_value, str):
        parts = [p.strip() for p in n_value.split(",")]
        try:
            nums = [float(p) for p in parts]
            return all(n >= cutoff for n in nums)
        except:
            return False

    return False

def filter_stats_df(stats_df: pd.DataFrame,
                    alpha: float = 0.05, 
                    min_n: int = 10) -> pd.DataFrame:
    """
    Unified filtering logic for all metrics including Flux.
    
    Conditions:
      • p_All <= alpha  (must exist & be numeric)
      • N_All meets passes_n_cutoff()
      
    Returns a clean, reindexed dataframe.
    """
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()

    df = stats_df.copy()

    # Ensure p_All is numeric
    df["p_All"] = pd.to_numeric(df["p_All"], errors="coerce")

    # Apply N filtering (handles strings, lists, etc.)
    n_mask = df["N_All"].apply(lambda x: passes_n_cutoff(x, min_n))

    # p-value condition
    p_mask = df["p_All"] <= alpha

    # Final mask
    mask = p_mask & n_mask

    return df[mask].sort_values(
        by=["Comparison", "Metric"],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)


def add_fisher_flux(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Flux metric derived from Abundance × Rate with Fisher-combined p-values.
    Computes All and Filtered versions independently.
    Includes N_All and N_Filtered = 4 for Fisher tests.
    Returns a new dataframe with Flux rows appended.
    """

    df = stats_df.copy()

    # Extract Abundance and Rate rows
    abn = df[df["Metric"] == "Abundance"]
    rate = df[df["Metric"] == "Rate"]

    # Merge keys
    merge_keys = ["Comparison", "Plot_Group"]
    if "Lipid Unique Identifier" in df.columns:
        merge_keys.append("Lipid Unique Identifier")

    merged = abn.merge(
        rate,
        on=merge_keys,
        suffixes=("_abn", "_rate"),
        how="inner"
    )

    # 1) Flux mean differences
    merged["Mean_Diff_All"] = (
        merged["Mean_Diff_All_abn"] * merged["Mean_Diff_All_rate"]
    )
    merged["Mean_Diff_Filtered"] = (
        merged["Mean_Diff_Filtered_abn"] * merged["Mean_Diff_Filtered_rate"]
    )

    # 2) Fisher combined p-values
    def fisher_p(p1, p2):
        if p1 <= 0 or p2 <= 0 or np.isnan(p1) or np.isnan(p2):
            return np.nan, np.nan
        X = -2 * (np.log(p1) + np.log(p2))
        p = 1 - chi2.cdf(X, df=4)
        return X, p

    merged["X_All"], merged["p_All"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_All_abn"], r["p_All_rate"]),
        axis=1
    ))

    merged["X_Filtered"], merged["p_Filtered"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_Filtered_abn"], r["p_Filtered_rate"]),
        axis=1
    ))

    # 3) Fisher-based t statistics
    merged["t_All"] = np.sign(merged["Mean_Diff_All"]) * np.sqrt(merged["X_All"])
    merged["t_Filtered"] = np.sign(merged["Mean_Diff_Filtered"]) * np.sqrt(merged["X_Filtered"])

    # 4) Add degrees of freedom (always df = 4 for Fisher)
    merged["N_All"] = 4
    merged["N_Filtered"] = 4

    # 5) Additional metadata
    merged["Metric"] = "Flux"
    merged["Filters_Applied"] = "{}"
    merged["Test"] = "Fisher projection from Abundance + Rate"
    merged["Mode"] = "derived_flux"

    # 6) Ensure Flux rows include all original dataframe columns
    final_cols = df.columns
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[final_cols]

    # 7) Append back to original df
    return pd.concat([df, merged], ignore_index=True)

import numpy as np
import pandas as pd
from scipy.stats import chi2

def add_fisher_flux(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived Flux metric using Fisher's combined test.
    - N_All and N_Filtered become a single delimited string: "abundanceN,rateN"
    - Abundance always listed first
    """

    df = stats_df.copy()

    # Extract Abundance and Rate rows
    abn = df[df["Metric"] == "Abundance"]
    rate = df[df["Metric"] == "Rate"]

    # Merge keys
    merge_keys = ["Comparison", "Plot_Group"]
    if "Lipid Unique Identifier" in df.columns:
        merge_keys.append("Lipid Unique Identifier")

    merged = abn.merge(
        rate,
        on=merge_keys,
        suffixes=("_abn", "_rate"),
        how="inner"
    )

    # -----------------------------
    # 1) Flux mean differences
    # -----------------------------
    merged["Mean_Diff_All"] = (
        merged["Mean_Diff_All_abn"] * merged["Mean_Diff_All_rate"]
    )
    merged["Mean_Diff_Filtered"] = (
        merged["Mean_Diff_Filtered_abn"] * merged["Mean_Diff_Filtered_rate"]
    )

    # -----------------------------
    # 2) Fisher combined p-values
    # -----------------------------
    def fisher_p(p1, p2):
        if p1 <= 0 or p2 <= 0 or np.isnan(p1) or np.isnan(p2):
            return np.nan, np.nan
        X = -2 * (np.log(p1) + np.log(p2))
        p = 1 - chi2.cdf(X, df=4)
        return X, p

    merged["X_All"], merged["p_All"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_All_abn"], r["p_All_rate"]),
        axis=1
    ))

    merged["X_Filtered"], merged["p_Filtered"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_Filtered_abn"], r["p_Filtered_rate"]),
        axis=1
    ))

    # -----------------------------
    # 3) Fisher-based t statistics
    # -----------------------------
    merged["t_All"] = np.sign(merged["Mean_Diff_All"]) * np.sqrt(merged["X_All"])
    merged["t_Filtered"] = np.sign(merged["Mean_Diff_Filtered"]) * np.sqrt(merged["X_Filtered"])

    # -----------------------------
    # 4) Build SINGLE N columns (Abundance first)
    # -----------------------------
    merged["N_All"] = (
        merged["N_All_abn"].astype(int).astype(str)
        + "," +
        merged["N_All_rate"].astype(int).astype(str)
    )

    merged["N_Filtered"] = (
        merged["N_Filtered_abn"].astype(int).astype(str)
        + "," +
        merged["N_Filtered_rate"].astype(int).astype(str)
    )

    # -----------------------------
    # 5) Metadata
    # -----------------------------
    merged["Metric"] = "Flux"
    merged["Filters_Applied"] = "{}"
    merged["Test"] = "Fisher projection from Abundance + Rate"
    merged["Mode"] = "derived_flux"

    # -----------------------------
    # 6) Ensure the output matches the original schema
    # -----------------------------
    final_cols = df.columns
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[final_cols]

    # Append flux rows
    return pd.concat([df, merged], ignore_index=True)

def add_fisher_flux(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived Flux metric using proper log2 FC math + Fisher's combined test.

    Correct flux fold-change relationship:
        log2(FC_flux) = log2(FC_abundance) + log2(FC_rate)

    This replaces the incorrect previous implementation that multiplied
    mean differences instead of adding log2 fold-changes.
    """

    df = stats_df.copy()

    # Extract Abundance and Rate rows
    abn = df[df["Metric"] == "Abundance"]
    rate = df[df["Metric"] == "Rate"]

    # Merge keys
    merge_keys = ["Comparison", "Plot_Group"]
    if "Lipid Unique Identifier" in df.columns:
        merge_keys.append("Lipid Unique Identifier")

    merged = abn.merge(
        rate,
        on=merge_keys,
        suffixes=("_abn", "_rate"),
        how="inner",
    )

    # ---------------------------------------------------------
    # 1) Correct flux log2 fold-change math
    # ---------------------------------------------------------
    # Flux log2 FC = log2_FC_abn + log2_FC_rate
    merged["Mean_Diff_All"] = (
        merged["Mean_Diff_All_abn"] + merged["Mean_Diff_All_rate"]
    )

    merged["Mean_Diff_Filtered"] = (
        merged["Mean_Diff_Filtered_abn"] + merged["Mean_Diff_Filtered_rate"]
    )

    # Optional: linear FCs for debugging
    merged["FC_Flux_All"] = 2 ** merged["Mean_Diff_All"]
    merged["FC_Flux_Filtered"] = 2 ** merged["Mean_Diff_Filtered"]

    # ---------------------------------------------------------
    # 2) Fisher combined p-values (unchanged)
    # ---------------------------------------------------------
    def fisher_p(p1, p2):
        if p1 <= 0 or p2 <= 0 or np.isnan(p1) or np.isnan(p2):
            return np.nan, np.nan
        X = -2 * (np.log(p1) + np.log(p2))
        p = 1 - chi2.cdf(X, df=4)
        return X, p

    merged["X_All"], merged["p_All"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_All_abn"], r["p_All_rate"]),
        axis=1
    ))

    merged["X_Filtered"], merged["p_Filtered"] = zip(*merged.apply(
        lambda r: fisher_p(r["p_Filtered_abn"], r["p_Filtered_rate"]),
        axis=1
    ))

    # ---------------------------------------------------------
    # 3) Signed "t" = sign(log2 FC) * sqrt(Fisher X)
    # ---------------------------------------------------------
    merged["t_All"] = np.sign(merged["Mean_Diff_All"]) * np.sqrt(merged["X_All"])
    merged["t_Filtered"] = (
        np.sign(merged["Mean_Diff_Filtered"]) * np.sqrt(merged["X_Filtered"])
    )

    # ---------------------------------------------------------
    # 4) Build N columns (Abundance first)
    # ---------------------------------------------------------
    merged["N_All"] = (
        merged["N_All_abn"].astype(int).astype(str)
        + "," +
        merged["N_All_rate"].astype(int).astype(str)
    )
    merged["N_Filtered"] = (
        merged["N_Filtered_abn"].astype(int).astype(str)
        + "," +
        merged["N_Filtered_rate"].astype(int).astype(str)
    )

    # ---------------------------------------------------------
    # 5) Metadata
    # ---------------------------------------------------------
    merged["Metric"] = "Flux"
    merged["Filters_Applied"] = "{}"
    merged["Test"] = "Fisher projection from Abundance + Rate"
    merged["Mode"] = "derived_flux"

    # ---------------------------------------------------------
    # 6) Ensure final schema matches the original df
    # ---------------------------------------------------------
    final_cols = df.columns
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[final_cols]

    # Append flux rows
    return pd.concat([df, merged], ignore_index=True)



# OR: ignore by category (safer)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Pandas-specific PerformanceWarning
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

def safe_filename(s: str) -> str:
    """Make a string safe for filenames."""
    s = re.sub(r"[^\w\s-]", "", str(s)).strip()
    s = re.sub(r"[\s]+", "_", s)
    return s[:180]  # avoid super long filenames


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
        "Abundance Paired t-test and Conformity": "Abundance",
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
    
    final = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    final = final.loc[:, ~final.columns.duplicated()]  # drop hidden dupes
    final = final.copy()  # break block manager ties
    return final, frames



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
            print("Invalid format. Please try again (e.g., (A2,A3),(A4,A3)).")


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
        return ("log2_abn_FC", "-log10abnBH", dict(title="Abundance Volcano", x_size=4, y_size=4))

    if s == "rate":
        return ("log2_rate_FC", "-log10rate_P", dict(title="Rate Volcano", x_size=4, y_size=4))
    if s == "asymptote":
        return ("asymptote_difference", "-log10_asymptote_p", dict(title="Asymptote Volcano", x_size=0.75))

    if s in ("nl", "n l", "n_l", "n-value"):
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
            x_col="abundance_mean_log2_Control",
            y_col="abundance_mean_log2_Experiment",
            title="Abundance Single Sample Fold-Change t-test",
            axis_titles=["Log2(mean abundance)", "Log2(mean abundance)"],
            ensure_same_axis=True,
            modifiers = 'onesample_fc'
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












def run_plots_from_gui_config(cfg: Dict[str, Any], experiments, plots_dir) -> Dict[str, pd.DataFrame]:

    collected_stats = []
    collected_points = []

    def _build_plot_groups(experiments):
        ontology_values = set()
        for exp in experiments:
            if 'Ontology' in getattr(exp, "df", pd.DataFrame()).columns:
                ontology_values.update(
                    exp.df['Ontology'].dropna().astype(str).unique().tolist()
                )
        ontology_list = sorted(ontology_values)

        higher_order = [
            'Standards', 'glycerolipids', 'lysos', 'Ethers',
            'glycerophospholipids', 'sphingolipids', 'neutral_lipids', 'ionic_lipids', "Kennedy_lipids"
        ]

        groups = []
        groups.append(("All", None))

        for v in ontology_list:
            groups.append((f"Ontology_{v}", {"Ontology": v}))

        for h in higher_order:
            if any(h in getattr(exp, "df", pd.DataFrame()).columns for exp in experiments):
                groups.append((f"HighOrder_{h}", {h: True}))

        return groups
    
    
        
    def _build_plot_groups(experiments):
        """
        Build the list of plot/ttest groups:
          - 'All'
          - one group per literal Ontology value (exact match)
          - one group per classic higher-order
          - one group for every dynamic 'higher_order_*' column present
        """
        ontology_values = set()
        dynamic_high_orders = set()
        classic_high_orders = [
            'Standards', 'glycerolipids', 'lysos', 'Ethers',
            'glycerophospholipids', 'sphingolipids',
            'neutral_lipids', 'ionic_lipids', 'Kennedy_lipids', 'ethanolamines', 'serines', 'cholines', 'glycerols', 'inositols', 'phosphatidics'       ]
    
        for exp in experiments:
            df = getattr(exp, "df", pd.DataFrame())
            if 'Ontology' in df.columns:
                ontology_values.update(
                    df['Ontology'].dropna().astype(str).unique().tolist()
                )
            if isinstance(df, pd.DataFrame):
                dynamic_high_orders.update({
                    c for c in df.columns
                    if isinstance(c, str) and c.startswith("higher_order_")
                })
    
        ontology_list = sorted(ontology_values)
        higher_order = classic_high_orders + sorted(dynamic_high_orders)
    
        groups = []
        groups.append(("All", None))
        for v in ontology_list:
            # IMPORTANT: exact-equality Ontology filter (handled later)
            groups.append((f"Ontology_{v}", {"Ontology": v}))
        for h in higher_order:
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

                    if col.lower() == "ontology":
                        # Literal class boundary: exact (case-insensitive) equality
                        lhs = df[col].astype(str).str.strip().str.upper()
                        rhs = str(val).strip().upper()
                        mask &= (lhs == rhs)


                    elif val is True:
                        mask &= df[col].astype(bool)
                    else:
                        mask &= (df[col].astype(str) == str(val))

                filtered_df = df.loc[mask].copy()

            new_exp = copy.copy(exp)
            setattr(new_exp, "df", filtered_df)
            out.append(new_exp)

        return out

    # -----------------------------------
    # Build groups
    # -----------------------------------
    groups = _build_plot_groups(experiments)

    # -----------------------------------
    # LOOP THROUGH GROUPS
    # -----------------------------------
    for group_name, extra_filter in groups:

        safe_name = str(group_name).replace(" ", "_")

        fed_experiments = _filter_experiments_for_group(experiments, extra_filter)

        any_rows = any(
            getattr(e, "df", pd.DataFrame()).shape[0] > 0
            for e in fed_experiments
        )

        if not any_rows:
            print(f"[plot] skipping group '{group_name}' — no rows selected")
            continue

        group_plots_dir = os.path.join(plots_dir, safe_name)
        os.makedirs(group_plots_dir, exist_ok=True)


        # -------------------------
        # Homeostasis
        # -------------------------
        homeo = cfg.get("Homeostasis", {})
        if homeo.get("enabled"):
            # --- Extract Homeostasis filter sections the same way Volcano does ---
            metrics = homeo.get("filters", {})
            primary_filters = {}
            secondary_filters = {}
            for section_name, item in (metrics or {}).items():
                if not item.get("enabled"):
                    continue
                # For Homeostasis, treat the section's text as primary filters
                text = item.get("text", "")
                p = _parse_metrics_text(text)
                if p:
                    primary_filters.update(p)
                # Optional support for "secondary"
                if "secondary" in item:
                    s = _parse_metrics_text(item.get("secondary", ""))
                    if s:
                        secondary_filters.update(s)
        
            # ---------------- Run Homeostasis plot with filters ----------------
            res = create_plots(
                experiments=fed_experiments,
                analysis_type="homeostasis",
                x_col="log2_abn_FC",
                y_col="log2_rate_FC",
                title=f"Homeostasis: Log₂ FC Abundance vs Rate ({group_name})",
                axis_titles=["log₂ FC Abundance", "log₂ FC Rate"],
                output_dir=group_plots_dir,
                ensure_same_axis=True,
                # Significance columns
                significance_column1=homeo.get("significance_column1", "abn_Overall_significant"),
                significance_column2=homeo.get("significance_column2", "rate_Overall_significant"),
                # Pass filter dictionaries into create_plots
                primary_filters=primary_filters,
                secondary_filters=secondary_filters,
                drop_duplicates_by='Lipid Unique Identifier',
                # Autoscale overrides (optional)
                x_size=2,
                y_size=2,
                # Ensure clean Plot_Group values (All / HighOrder_* / Ontology_*)
                plot_group=group_name,
            )
            if isinstance(res, dict) and isinstance(res.get("datapoints_df"), pd.DataFrame):
                collected_points.append(res["datapoints_df"])

        # -------------------------
        # Volcano
        # -------------------------
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

        # -------------------------
        # Conformity / scatter_ttest
        # -------------------------
        conf = cfg.get("Conformity", {})
        if conf.get("enabled"):
            metrics = conf.get("metrics", conf.get("filters", {}))

            for section_name, item in (metrics or {}).items():
                if not item.get("enabled"):
                    continue

                fixed = _conformity_fixed(section_name)

                if not fixed:
                    manual_keys = {
                        k: v for k, v in item.items()
                        if k in ("x_col", "y_col", "title", "axis_titles")
                    }
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
                    collected_points.append(res["datapoints_df"])

    # -----------------------------------
    # BUILD FINAL DATAFRAMES (OUTSIDE LOOP)
    # -----------------------------------
    
    # 1) Build the combined statistics dataframe
    if collected_stats:
        stats_df = pd.concat(collected_stats, ignore_index=True)
    else:
        stats_df = pd.DataFrame()
    
    # 2) Add Flux rows BEFORE any filtering
    stats_df = add_fisher_flux(stats_df)
    
    # 3) Sort stats_df by p-value if present
    if "p_All" in stats_df.columns:
        stats_df["p_All"] = pd.to_numeric(stats_df["p_All"], errors="coerce")
        stats_df = stats_df.sort_values(
            by="p_All",
            ascending=True,
            na_position="last"
        ).reset_index(drop=True)
    
    # 4) Apply unified filtering logic (p-value + N cutoff)
    def filter_stats_df(stats_df, alpha=0.05, min_n=10):
        if stats_df is None or stats_df.empty:
            return pd.DataFrame()
    
        df = stats_df.copy()
    
        # Ensure p_All is numeric
        df["p_All"] = pd.to_numeric(df["p_All"], errors="coerce")
    
        # Masks
        p_mask = df["p_All"] <= alpha
        n_mask = df["N_All"].apply(lambda x: passes_n_cutoff(x, min_n))
    
        mask = p_mask & n_mask
    
        return (
            df[mask]
            .sort_values(by=["Comparison", "Metric"],
                         ascending=[True, True],
                         na_position="last")
            .reset_index(drop=True)
        )
    
    # 5) Build the filtered statistics df (Flux included if it passes)
    stats_sorted_filtered_df = filter_stats_df(stats_df, alpha=0.05, min_n=10)
    
    # 6) Build consolidated datapoints df
    points_df = (
        pd.concat(collected_points, ignore_index=True)
        if collected_points else pd.DataFrame()
    )
    
    # 7) Return both final outputs
    return {
        "statistics_df": stats_df,
        "statistics_alpha0_05_minN10": stats_sorted_filtered_df,
        "datapoints_df": points_df,
    }



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
    datapoints_container: Dict[str, Optional[pd.DataFrame]] = {"df": None}

    # Normalization dataframe (shared across experiments)
    NORMALIZATION_DF: Optional[pd.DataFrame] = None

    # --------------------------------------------------
    # GUI
    # --------------------------------------------------
    def establish_plots_with_gui():
    
        def on_analyze(config_tree: Dict[str, Any]):
            nonlocal NORMALIZATION_DF
    
            try:
                # Progress: starting
                update_progress(0, 100, "Preparing analysis...")
    
                # ----------------------------------
                # Normalization (single-choice UI) + back-compat with old YAML keys
                # ----------------------------------
                settings = config_tree.get("Settings", {})
                norm_cfg = settings.get("normalization", {})
                method = str(norm_cfg.get("method", "none")).lower()
                baseline_from_cfg = norm_cfg.get("baseline", "A3")
    
                # Back-compat: map old booleans to the new single method if present
                if "perform_abundance_normalization" in settings or "use_standards_normalization" in settings:
                    old_mtic = bool(settings.get("perform_abundance_normalization", False))
                    old_std  = bool(settings.get("use_standards_normalization", False))
                    method = "standards" if old_std else ("mtic" if old_mtic else "none")
    
                do_norm = (method == "mtic")
                use_standards_norm = (method == "standards")
                standards_baseline_group = baseline_from_cfg if use_standards_norm else None
    
                # We no longer load an external file; mTIC is computed in Experiment/prep.
                NORMALIZATION_DF = None
    
                
                if method == "mtic":
                    print("[GUI] ✅ Normalization: mTIC (computed per polarity from abundances).")
                elif method == "standards":
                    print("[GUI] ✅ Normalization: Standards (robust log2 mapping per polarity).")
                    print(f"[GUI] Standards baseline group: {standards_baseline_group}")
                elif method == "zscore":
                    print("[GUI] ✅ Normalization: Z-score (per lipid across samples; stats use Δz).")
                elif method == "quantile":
                    print("[GUI] ✅ Normalization: Quantile (per polarity; proceeds with FC & t-tests).")
                else:
                    print("[GUI] Normalization: None (raw abundances).")

    
                # ----------------------------------
                # Build experiments
                # ----------------------------------
                update_progress(20, 100, "Building experiments...")
                experiments: List[Experiment] = []
                total_pairs = len(pairs)
    
                # Map 20% -> 50% across experiment builds.
                for idx, pair in enumerate(pairs, start=1):
                    # Increment progress during experiment building
                    build_phase_progress = 20 + int((idx / max(total_pairs, 1)) * 30)
                    update_progress(build_phase_progress, 100, f"Building experiment {idx}/{total_pairs}")
    
                    exp = Experiment(
                        file_paths=file_paths,
                        pair=pair,
                        all_ids=all_ids,
                        number=idx,
                        total=total_pairs,
                        normalization_df=None,
                        normalize_by_standards=use_standards_norm,
                        baseline=standards_baseline_group,
                        perform_mtic=do_norm,
                        norm_method=method   # <-- NEW
                    )
                    experiments.append(exp)
    
                experiments_container["list"] = experiments
    
                # ----------------------------------
                # Run plots according to GUI config
                # ----------------------------------
                update_progress(50, 100, "Starting plot generation...")
    
                results = run_plots_from_gui_config(
                    config_tree,
                    experiments,
                    plots_dir
                )
    
                stats_df = results.get("statistics_df")
                stats_sorted_filtered_df = results.get("statistics_alpha0_05_minN10")
                datapoints_df = results.get("datapoints_df")

    
                # Store statistics (into the container from main())
                if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                    stats_container["df"] = stats_df
                    print(f"[GUI] ✅ Collected {len(stats_df):,} rows of statistical output.")
                else:
                    stats_container["df"] = None
                    print("[GUI] ⚠️ No statistical results were produced.")
    
                # Store datapoints (into the container from main())
                if isinstance(datapoints_df, pd.DataFrame) and not datapoints_df.empty:
                    datapoints_container["df"] = datapoints_df
                    print(f"[GUI] ✅ Collected {len(datapoints_df):,} datapoints.")
                else:
                    datapoints_container["df"] = None
                    print("[GUI] ⚠️ No datapoints collected.")
    
                # ----------------------------------
                # (Moved up) Post-GUI exports: run BEFORE popup
                # ----------------------------------
                # Save cleaned datapoints & make boxplots
                statistics_df = stats_container.get("df", None)
                datapoints_all = datapoints_container.get("df", None)
    
                # Boxplot: Filtered vs Unfiltered (single clean block)
                if isinstance(datapoints_all, pd.DataFrame) and not datapoints_all.empty:
                    required_cols = {"Plot_Group", "Delta", "Filtered"}
                    if required_cols.issubset(datapoints_all.columns):
                        plot_df = datapoints_all.copy()
                
                        # Coerce Delta to numeric
                        plot_df["Delta"] = pd.to_numeric(plot_df["Delta"], errors="coerce")
                
                        # Clean Plot_Group
                        plot_df["Plot_Group"] = plot_df["Plot_Group"].astype(str).str.strip()
                        plot_df.loc[plot_df["Plot_Group"].isin(["", "nan", "None"]), "Plot_Group"] = np.nan
                
                        # --- Pretty label for Plot_Group (for box-plot display only) ---
                        def pretty_plot_group(s: str) -> str:
                            s = str(s or "").strip()
                            if not s:
                                return s
                            if s == "All":
                                return "All"
                            # Strip known prefixes
                            for pref in ("Ontology_", "HighOrder_"):
                                if s.startswith(pref):
                                    s = s[len(pref):]
                                    break
                            # Make it readable: underscores -> spaces
                            s = s.replace("_", " ")
                            # Uppercase ONLY the very first character, preserve rest as-is
                            # (If you want absolutely no casing change at all, comment the next line.)
                            s = s[:1].upper() + s[1:]
                            return s
                
                        plot_df["Group_Pretty"] = plot_df["Plot_Group"].map(pretty_plot_group)
                
                        # Normalize Filtered to boolean (or NaN)
                        def _to_bool(v):
                            s = str(v).strip().lower()
                            if s in ("true", "t", "1", "yes", "y"):
                                return True
                            if s in ("false", "f", "0", "no", "n"):
                                return False
                            return np.nan
                
                        plot_df["Filtered"] = plot_df["Filtered"].map(_to_bool)
                
                        # Drop rows missing essentials
                        plot_df = plot_df.dropna(subset=["Plot_Group", "Group_Pretty", "Delta", "Filtered"])
                
                        # Normalize metric naming to match paired t-tests
                        def clean_metric(m):
                            m = str(m).lower()
                            if "n-value" in m or "n_l" in m or "nl" in m:
                                return "nL"
                            if "abundance" in m or "dr" in m:
                                return "Abundance"
                            if "flux" in m:
                                return "Flux"
                            if "asymptote" in m:
                                return "Asymptote"
                            if "rate" in m:
                                return "Rate"
                            return m
                
                        plot_df["Metric"] = plot_df["Metric"].apply(clean_metric)
                
                        # Save the cleaned/plot-ready datapoints
                        cleaned_csv = os.path.join(parent_dir, "datapoints_for_boxplot.csv")
                        try:
                            if not plot_df.empty:
                                plot_df.to_csv(cleaned_csv, index=False)
                                print(f"Saved cleaned datapoints: {cleaned_csv}")
                        except Exception as e:
                            print(f"Warning: could not save cleaned datapoints: {e}")
                
                        # Helper to classify group type using the ORIGINAL Plot_Group
                        def _group_type(raw: str) -> str:
                            raw = str(raw or "")
                            if raw == "All":
                                return "All"
                            if raw.startswith("HighOrder_"):
                                return "HighOrder"
                            if raw.startswith("Ontology_"):
                                return "Ontology"
                            return "Other"
                
                        # Plot only if there is something to plot
                        for USE_FILTERED in (True, False):
                            # Ensure Filtered is boolean
                            plot_df2 = plot_df.copy()
                            if plot_df2["Filtered"].dtype == object:
                                plot_df2["Filtered"] = plot_df2["Filtered"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
                
                            # Subset based on the boolean toggle
                            subset_df = plot_df2[plot_df2["Filtered"] == USE_FILTERED].copy()
                
                            if subset_df.empty:
                                print(f"No rows found where Filtered == {USE_FILTERED}. Nothing to plot.")
                                continue
                
                            # Keep any Group × Metric that has at least ONE significant comparison,
                            # but include ALL comparisons for those groups/metrics.
                            if isinstance(stats_sorted_filtered_df, pd.DataFrame) and not stats_sorted_filtered_df.empty:
                
                                # Build a cleaned copy of sig pairs with pretty labels and cleaned metrics
                                sig_pairs = stats_sorted_filtered_df[["Plot_Group", "Metric"]].drop_duplicates().copy()
                
                                # Pretty-ize the Plot_Group for stats as well
                                sig_pairs["Group_Pretty"] = sig_pairs["Plot_Group"].map(pretty_plot_group)
                                sig_pairs["Metric"] = sig_pairs["Metric"].apply(clean_metric)
                
                                # Merge on pretty label + metric (not the raw Plot_Group)
                                subset_df = subset_df.merge(
                                    sig_pairs[["Group_Pretty", "Metric"]].drop_duplicates(),
                                    on=["Group_Pretty", "Metric"],
                                    how="inner"
                                )
                
                                # If nothing remains, skip
                                if subset_df.empty:
                                    print("No Group × Metric combinations had ANY significant comparisons.")
                                    continue
                
                                # Loop each metric panel
                                for metric, df_m in subset_df.groupby("Metric", dropna=False):
                                    if df_m.empty:
                                        continue
                
                                    # --- Custom y-axis order: All -> HighOrder -> Ontology -> Other; alpha within each by pretty ---
                                    order_df = (
                                        df_m[["Plot_Group", "Group_Pretty"]]
                                        .drop_duplicates()
                                        .copy()
                                    )
                                    order_df["_type"] = order_df["Plot_Group"].map(_group_type)
                                    _type_rank = {"All": 0, "HighOrder": 1, "Ontology": 2, "Other": 3}
                                    order_df["_rank"] = order_df["_type"].map(_type_rank).fillna(3).astype(int)
                
                                    # Sort by rank first, then by pretty label (stable sort keeps deterministic order)
                                    order_df = order_df.sort_values(["_rank", "Group_Pretty"], kind="mergesort")
                                    plot_group_order = order_df["Group_Pretty"].tolist()
                
                                    # Enforce this categorical order for plotting
                                    df_m = df_m.copy()
                                    df_m["Group_Pretty"] = pd.Categorical(
                                        df_m["Group_Pretty"], categories=plot_group_order, ordered=True
                                    )
                
                                    n_groups = len(plot_group_order)
                                    fig_h = max(6, 0.40 * n_groups)  # auto height scaling
                
                                    plt.figure(figsize=(12, fig_h))
                                    ax = sns.boxplot(
                                        data=df_m,
                                        x="Delta",
                                        y="Group_Pretty",
                                        order=plot_group_order,
                                        orient="h",
                                        hue="Comparison",
                                        showfliers=False
                                    )
                
                                    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
                
                                    ax.set_title(f"{metric}  |  Filtered = {USE_FILTERED}")
                                    ax.set_xlabel("Delta")
                                    ax.set_ylabel("Group")
                
                                    plt.tight_layout()
                
                                    metric_slug = safe_filename(metric)
                                    out_name = f"boxplot_{metric_slug}_Filtered_{USE_FILTERED}.png"
                                    out_path = os.path.join(parent_dir, out_name)
                                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                                    svg_path = out_path.replace(".png", ".svg")
                                    plt.savefig(svg_path, format="svg", bbox_inches="tight")
                
                                    plt.close()
                
                                    print(f"Saved: {out_path}")
                
                            else:
                                print("No significant comparisons with N >= 10.")
                                subset_df = pd.DataFrame()
                                
                                
                
                                            

                else:
                    print("No datapoints available for boxplots.")
                    
                    
                # Save stats summary + comparison tables
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
                    
                    
                
                # -----------------------------------------------
                # Create Regulation Table using regulation_table.py
                # -----------------------------------------------
                try:
                    stats_filtered_csv = os.path.join(parent_dir, "statistics_alpha0_05_minN10.csv")
                
                    if os.path.exists(stats_filtered_csv):
                        print("[RegTable] Generating regulation table...")
                        output_reg_table = reg.process_table_from_path(stats_filtered_csv)
                        print(f"[RegTable] Regulation table saved to: {output_reg_table}")
                    else:
                        print("[RegTable] No filtered statistics CSV found — skipping regulation table.")
                
                except Exception as e:
                    print(f"[RegTable] Error generating regulation table: {e}")

                    
                # Save sorted + filtered statistics
                sorted_filtered_csv = os.path.join(parent_dir, "statistics_alpha0_05_minN10.csv")
                
                try:
                    if isinstance(stats_sorted_filtered_df, pd.DataFrame) and not stats_sorted_filtered_df.empty:
                        stats_sorted_filtered_df.to_csv(sorted_filtered_csv, index=False)
                        print(f"Saved sorted/filtered stats: {sorted_filtered_csv}")
                    else:
                        print("No sorted/filtered statistics to save.")
                except Exception as e:
                    print(f"Warning: could not save sorted/filtered statistics: {e}")

    
                # Save final merged dataframes (both locations)
                final_df, frames = build_final_dataframe(experiments)
                if not final_df.empty:
                    # legacy path near plots
                    # summary path in parent folder
                    final_csv = os.path.join(parent_dir, "final_dataframe.csv")
                    final_df.to_csv(final_csv, index=False)
                    print(f"Saved final dataframe: {final_csv}")
                else:
                    print("[GUI] ⚠️ Final dataframe empty — nothing written.")
    
                # Final progress update BEFORE popup
                update_progress(100, 100, "Done.")
    
                # ----------------------------------
                # Completion popup (AFTER exports)
                # ----------------------------------
                from gui import _ROOT
                from tkinter import messagebox
    
                if _ROOT is not None:
                    _ROOT.after(
                        0,
                        lambda msg=(
                            "✅ All selected plots and statistics have been generated.\n\n"
                            f"Outputs saved to:\n{parent_dir}\n\n"
                            "You may close this window using the 'X'."
                        ): messagebox.showinfo("Analysis Complete", msg)
                    )
    
                print(f"[GUI] ✅ Analysis complete. Plots saved to: {plots_dir}")
    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Analysis failed: {e}")
                # Ensure the progress bar shows failure/completion state
                try:
                    update_progress(100, 100, "Failed")
                except Exception:
                    pass
    
        # Register and launch GUI
        set_on_analyze(on_analyze)
        launch_gui()

    # Launch GUI
    establish_plots_with_gui()

    # --------------------------------------------------
    # No post-GUI export block needed anymore
    # (we moved all exports inside on_analyze BEFORE the popup)
    # --------------------------------------------------
    print(f"Done.\nParent folder: {parent_dir}\nPlots:         {plots_dir}")



if __name__ == '__main__':
    main()
