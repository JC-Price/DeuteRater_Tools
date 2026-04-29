import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import ast

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
        input_file_path = file_path
        messagebox.showinfo("Success", f"Loaded file:\n{file_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# PCA FUNCTION
# =========================
def run_pca_analysis(
    metrics_list,
    index_column,
    filter_col,
    filter_val,
    compare_col,
    group1,
    group2,
    data_override=None,
    log_transform=True
):
    """
    Run a 2D PCA on a long-format dataframe, returning scores and loadings.

    Returns
    -------
    (result_df, explained_variance_ratio_, load_df)

    - result_df: index = samples, columns = ['PC1','PC2','group']
    - explained_variance_ratio_: array-like length 2
    - load_df: index = features (metrics), columns = ['PC1_load','PC2_load']
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    global df
    data = data_override if data_override is not None else df

    if data is None:
        messagebox.showerror("Error", "No file loaded.")
        return None

    # ---- Validate columns (do NOT require filter_col; it can be 'ALL') ----
    required_cols = ["metric", "value", index_column, compare_col]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        messagebox.showerror("Error", f"Missing required column(s): {', '.join(missing)}")
        return None

    # ---- Start from all rows; filter metrics, filter_col, and groups ----
    working = data.copy()

    # If user passed no metrics_list (or blank/None), include ALL metrics present
    if not metrics_list:
        try:
            metrics_list = sorted(working["metric"].dropna().unique().tolist())
        except Exception:
            messagebox.showerror("Error", "Could not infer 'metrics' from dataframe.")
            return None

    # Filter by selected metrics
    working = working[working["metric"].isin(metrics_list)]

    # Apply filter_col/value only if meaningful (not ALL/blank/None)
    if filter_col and str(filter_col).lower() != "all":
        if filter_col not in working.columns:
            messagebox.showerror("Error", f"Filter column '{filter_col}' not found in dataframe.")
            return None
        working = working[working[filter_col] == filter_val]

    # Filter by groups
    if compare_col not in working.columns:
        messagebox.showerror("Error", f"Compare column '{compare_col}' not found in dataframe.")
        return None
    working = working[working[compare_col].isin([group1, group2])]

    if working.empty:
        messagebox.showerror("Error", "Filtered dataframe is empty.")
        return None

    # ---- Aggregate (mean) in case there are duplicate rows per (index, metric) ----
    try:
        agg_df = (
            working
            .groupby([index_column, "metric"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
    except KeyError as e:
        messagebox.showerror("Error", f"Grouping failed: {e}")
        return None

    # ---- Pivot: rows=samples, cols=metrics ----
    pivot_df = agg_df.pivot(index=index_column, columns="metric", values="value")
    # Drop samples that are entirely NA
    pivot_df = pivot_df.dropna(how="all")

    # ---- Basic shape checks ----
    if pivot_df.shape[0] < 2:
        messagebox.showerror("Error", "Need at least 2 samples (rows) for PCA.")
        return None

    # Convert to numeric & clean
    pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce")
    pivot_df = pivot_df.replace([np.inf, -np.inf], np.nan)

    # ---- Optional log transform (safe only if no negatives) ----
    if log_transform:
        if not (pivot_df < 0).any().any():
            pivot_df = np.log2(pivot_df + 1.0)

    # ---- Impute missing values with column means ----
    pivot_df = pivot_df.fillna(pivot_df.mean(numeric_only=True))

    # ---- Drop constant (zero-variance) columns BEFORE scaling ----
    variances = pivot_df.var(axis=0, numeric_only=True)
    const_cols = variances[variances == 0].index.tolist()
    if const_cols:
        pivot_df = pivot_df.drop(columns=const_cols, errors="ignore")

    if pivot_df.shape[1] < 2:
        messagebox.showerror(
            "Error",
            f"Need at least 2 non-constant metrics for PCA. Found {pivot_df.shape[1]}"
        )
        return None

    # ---- Final numeric sanity check ----
    import numpy as np
    if not np.isfinite(pivot_df.values).all():
        messagebox.showerror("Error", "Invalid values remain after cleaning.")
        return None

    # ---- Standardize & PCA ----
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(pivot_df)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_scaled)

    # ---- Scores (samples) ----
    result_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=pivot_df.index)

    group_map = (
        working[[index_column, compare_col]]
        .drop_duplicates()
        .set_index(index_column)
    )
    # Subset group_map to match PCA rows (in case)
    group_map = group_map.loc[group_map.index.intersection(result_df.index)]

    result_df = result_df.join(group_map)
    result_df = result_df.rename(columns={compare_col: "group"})

    if "group" not in result_df.columns:
        messagebox.showerror("Error", "Group column failed to attach.")
        return None

    if result_df["group"].isna().any():
        messagebox.showwarning("Warning", "Some samples are missing group labels.")

    # ---- Loadings (features) ----
    # pca.components_.T: shape = (n_features, n_components)
    loadings = pca.components_.T
    features = pivot_df.columns
    load_df = pd.DataFrame(loadings, columns=["PC1_load", "PC2_load"], index=features)

    return result_df, pca.explained_variance_ratio_, load_df



# =========================
# ANALYSIS WINDOW
# =========================
def open_analysis_window():
    if df is None:
        messagebox.showerror("Error", "Load a file first.")
        return

    window = tk.Toplevel(root)
    window.title("Add PCA Analysis")
    window.geometry("600x700")

    # Dictionary input
    tk.Label(
        window,
        text="Paste Parameter Dictionary:",
        font=("Arial", 10, "bold")
    ).pack(pady=5)

    dict_text = tk.Text(window, height=10, width=72)
    dict_text.pack(pady=5)

    # Form fields (kept for convenience/back-compat)
    frame = tk.Frame(window)
    frame.pack(pady=10)

    fields = [
        ("Metrics (comma separated; leave blank for ALL)", "metrics"),
        ("Index Column (points in PCA)", "index_column"),
        ("Filter Column", "filter_column"),
        ("Filter Value", "filter_value"),
        ("Compare Column", "compare_column"),
        ("Group 1", "group1"),
        ("Group 2", "group2")
    ]

    entries = {}
    for i, (label, key) in enumerate(fields):
        tk.Label(frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(frame, width=45)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[key] = entry

    # Load dict into fields (optional helper)
    def load_from_dict():
        try:
            user_dict = ast.literal_eval(dict_text.get("1.0", tk.END).strip())

            # Optional fields in dict; only populate what exists
            entries["metrics"].delete(0, tk.END)
            if "metrics" in user_dict:
                if isinstance(user_dict["metrics"], list):
                    entries["metrics"].insert(0, ", ".join(str(x) for x in user_dict["metrics"]))
                else:
                    entries["metrics"].insert(0, str(user_dict["metrics"]))

            for k in ["index_column", "filter_column", "filter_value",
                      "compare_column", "group1", "group2"]:
                if k in user_dict:
                    entries[k].delete(0, tk.END)
                    entries[k].insert(0, str(user_dict[k]))

            messagebox.showinfo("Success", "Dictionary loaded into fields.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid dictionary format:\n{e}")

    tk.Button(window, text="Load From Dictionary", command=load_from_dict).pack(pady=5)

    # ---- RUN ANALYSIS (handles Mode A & Mode B) ----
    def submit():
        # 1) Parse dictionary (if provided)
        user_dict = {}
        raw = dict_text.get("1.0", tk.END).strip()
        if raw:
            try:
                user_dict = ast.literal_eval(raw)
            except Exception as e:
                messagebox.showerror("Error", f"Invalid dictionary format:\n{e}")
                return

        # 2) Gather UI + dict parameters
        # UI entries (back-compat; used if dict fields are missing)
        metrics_input = entries["metrics"].get().strip()
        index_column = entries["index_column"].get().strip() or user_dict.get("index_column")
        filter_column = entries["filter_column"].get().strip() or user_dict.get("filter_column")
        filter_value  = entries["filter_value"].get().strip()  or user_dict.get("filter_value")
        compare_column = entries["compare_column"].get().strip() or user_dict.get("compare_column")
        group1 = entries["group1"].get().strip() or user_dict.get("group1")
        group2 = entries["group2"].get().strip() or user_dict.get("group2")

        # Dict-only fields for the new flow
        feature_id_column = user_dict.get("feature_id_column", "Alignment ID")
        single_measure    = user_dict.get("measure", None)          # Mode A
        measure_list      = user_dict.get("measure_list", None)     # Mode B

        # Reasonable defaults if user left some UI fields empty but provided dict ones
        index_column   = index_column   or "sample_id"
        filter_column  = (filter_column or "All")
        filter_value   = (filter_value  or "All")
        compare_column = compare_column or "genotype"

        # 3) Build a working dataframe (data_override) based on dict modes
        working = df.copy()

        # Safety checks
        for col in [feature_id_column, "metric", "value", index_column, compare_column]:
            if col not in working.columns:
                messagebox.showerror("Error", f"Required column '{col}' not found in dataframe.")
                return

        # Exclusive modes check
        if single_measure and measure_list:
            messagebox.showerror("Error", "Provide either 'measure' OR 'measure_list', not both.")
            return

        # MODE A: single measure -> metric becomes lipid identity (feature)
        if single_measure:
            working = working[working["metric"] == single_measure].copy()
            if working.empty:
                messagebox.showerror("Error", f"No rows found for measure '{single_measure}'.")
                return
            working["metric"] = working[feature_id_column]

        # MODE B: multiple measures -> synthesize feature names as feature__measure
        elif measure_list:
            working = working[working["metric"].isin(measure_list)].copy()
            if working.empty:
                messagebox.showerror("Error", f"No rows found for measures: {measure_list}")
                return
            working["metric"] = working[feature_id_column] + "__" + working["metric"]

        # Else: Back-compat (assume user already prepped df['metric'] as features)
        # No change to working["metric"]

        # 4) Build metrics list (features) from UI field or ALL in working
        if metrics_input:
            metrics_list = [m.strip() for m in metrics_input.split(",") if m.strip()]
        else:
            # Use ALL available features from working
            if "metric" not in working.columns:
                messagebox.showerror("Error", "Working dataframe lacks 'metric' column.")
                return
            metrics_list = sorted(working["metric"].dropna().unique().tolist())

            result = run_pca_analysis(
            metrics_list=metrics_list,
            index_column=index_column,
            filter_col=filter_column,
            filter_val=filter_value,
            compare_col=compare_column,
            group1=group1,
            group2=group2,
            data_override=working,
            log_transform=True
        )
        
        if result is not None:
            # Accept both 2-tuple (back-compat) and 3-tuple (with loadings)
            if len(result) == 3:
                result_df, variance, load_df = result
            else:
                result_df, variance = result
                load_df = None
        
            # ---- BUILD A NICE TITLE STRING ----
            # Prefer dictionary-driven info (measure or measure_list); fall back to filter info.
            measures_for_title = []
            if single_measure:
                measures_for_title = [str(single_measure)]
            elif measure_list:
                measures_for_title = [str(m) for m in measure_list]
        
            if measures_for_title:
                measure_part = " + ".join(measures_for_title)
            else:
                # If user didn’t pass measure/measure_list, show filter column/value or "All metrics"
                if filter_column and filter_column.lower() != "all":
                    measure_part = f"{filter_column} = {filter_value}"
                else:
                    measure_part = "All metrics"
        
            # Include group comparison and sample index column for clarity
            title_str = f"{measure_part}  |  {group1} vs {group2}  |  index={index_column}"
        
            # ---- STORE ANALYSIS ----
            analyses.append({
                "data": result_df,
                "variance": variance,
                "loadings": load_df,          # used by the biplot to draw arrows/labels
                "title": title_str,
                "comparison": f"{group1} vs {group2}"
            })
        
            messagebox.showinfo("Success", "Analysis added.")
            window.destroy()

    # Buttons: Run + Close
    btns = tk.Frame(window)
    btns.pack(pady=10)
    tk.Button(btns, text="Run Analysis", command=submit, height=2, width=16).grid(row=0, column=0, padx=8)
    tk.Button(btns, text="Close", command=window.destroy, height=2, width=12).grid(row=0, column=1, padx=8)

    # Keyboard shortcut: Ctrl+Enter to run
    window.bind("<Control-Return>", lambda e: submit())
# =========================
# PLOT & SAVE
# =========================
def plot_all_analyses(
    show_loadings=True,
    show_loading_arrows=False,    # keep arrows off by default
    loading_point_size=12,
    loading_alpha=0.35,
    arrow_alpha=0.18,
    arrow_color="gray",
    sample_markersize=60,
    cmap_dict=None                 # e.g., {"APOE2":"tab:blue","APOE3":"tab:orange","APOE4":"tab:green"}
):
    """
    Plot all added analyses as biplots. If loadings are present and show_loadings=True,
    draw faint points at loading locations, colored by the group they align with most.

    How group color for loadings is decided:
      - Compute each group's centroid in score space (PC1, PC2).
      - Compute cosine similarity between a loading vector (from origin) and each centroid.
      - Assign the loading to the group with the highest similarity (if all similarities <= 0, it's assigned to the nearest-by-angle anyway).

    Parameters
    ----------
    show_loadings : bool
        Whether to overlay loading points.
    show_loading_arrows : bool
        Whether to also draw subtle arrows from origin to loading points.
    loading_point_size : int
        Marker size for loading dots.
    loading_alpha : float
        Transparency for loading dots (0..1).
    arrow_alpha : float
        Transparency for loading arrows (0..1).
    arrow_color : str
        Color for loading arrows (if shown).
    sample_markersize : int
        Size for sample scatter points.
    cmap_dict : dict or None
        Optional mapping from group name to color string for consistency across subplots.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    global input_file_path

    if len(analyses) == 0:
        messagebox.showerror("Error", "No analyses added.")
        return

    # Layout
    n = len(analyses)
    rows = 2
    cols = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 10))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Helper to get group colors
    def get_group_colors(groups):
        # Use provided cmap if given; else let matplotlib choose defaults
        if cmap_dict:
            return {g: cmap_dict.get(g, None) for g in groups}
        # fall back to default color cycle
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        col_map = {}
        for i, g in enumerate(groups):
            col_map[g] = default_colors[i % len(default_colors)] if default_colors else None
        return col_map

    for i, analysis in enumerate(analyses):
        ax = axes[i]
        result_df = analysis["data"]
        variance = analysis["variance"]
        load_df = analysis.get("loadings", None)

        # --- Scores (samples) ---
        if "group" not in result_df.columns:
            # If no groups, just plot samples in one color
            ax.scatter(result_df["PC1"], result_df["PC2"], s=sample_markersize, c="tab:blue", label="samples")
        else:
            groups = list(result_df["group"].dropna().unique())
            color_map = get_group_colors(groups)
            for g in groups:
                subset = result_df[result_df["group"] == g]
                ax.scatter(
                    subset["PC1"], subset["PC2"],
                    s=sample_markersize,
                    label=str(g),
                    color=color_map.get(g, None)
                )

        # Titles and axes
        ax.set_title(f"Analysis {i+1}: {analysis.get('title', '')}")
        ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.axvline(0, color="black", lw=0.5, alpha=0.4)

        # --- Loadings as faint points, colored by aligned group ---
        if show_loadings and load_df is not None and not load_df.empty:
            # Extract scores to scale loadings into the same visual range
            scores = result_df[["PC1", "PC2"]].values
            score_radius = np.sqrt((scores**2).sum(axis=1)).max()
            loads = load_df[["PC1_load", "PC2_load"]].values
            load_norm = np.sqrt((loads**2).sum(axis=1)).max()
            scale = (score_radius / load_norm * 0.9) if (load_norm and load_norm > 0) else 1.0

            # Compute group centroids in score space
            centroids = {}
            if "group" in result_df.columns and result_df["group"].notna().any():
                for g in result_df["group"].dropna().unique():
                    grp = result_df[result_df["group"] == g][["PC1", "PC2"]].values
                    if grp.shape[0] > 0:
                        centroids[g] = grp.mean(axis=0)
            # Ensure we have at least something; if not, treat all loads as neutral gray
            groups_for_loads = list(centroids.keys())

            # Assign each loading to the most aligned group (cosine similarity)
            assigned_groups = []
            load_points = loads * scale  # scaled to score space

            def cosine_sim(u, v):
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)
                if nu == 0 or nv == 0:
                    return -np.inf
                return float(np.dot(u, v) / (nu * nv))

            # Build color map for groups
            color_map = get_group_colors(groups_for_loads) if groups_for_loads else {}

            # Gather arrays for plotting by group
            group_to_points = {g: [] for g in groups_for_loads} if groups_for_loads else {}

            for j, vec in enumerate(load_points):
                if groups_for_loads:
                    # Compare to centroid directions
                    sims = {g: cosine_sim(vec, centroids[g]) for g in groups_for_loads}
                    best_g = max(sims, key=sims.get) if sims else None
                    group_to_points[best_g].append(vec)
                else:
                    # No groups available -> skip color assignment (will draw all in gray)
                    pass

            # Plot loading points grouped by assigned group
            if groups_for_loads:
                for g, pts in group_to_points.items():
                    if not pts:
                        continue
                    pts = np.array(pts)
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        s=loading_point_size,
                        color=color_map.get(g, "gray"),
                        alpha=loading_alpha,
                        edgecolor="none",
                        label=f"{g} (features)"
                    )
                    if show_loading_arrows:
                        for (x, y) in pts:
                            ax.arrow(0, 0, x, y,
                                     color=arrow_color, alpha=arrow_alpha,
                                     length_includes_head=True,
                                     head_width=0.02*scale, head_length=0.02*scale)
            else:
                # If no centroids, plot all loadings in gray
                ax.scatter(
                    load_points[:, 0], load_points[:, 1],
                    s=loading_point_size,
                    color="gray", alpha=loading_alpha,
                    edgecolor="none", label="features"
                )
                if show_loading_arrows:
                    for (x, y) in load_points:
                        ax.arrow(0, 0, x, y,
                                 color=arrow_color, alpha=arrow_alpha,
                                 length_includes_head=True,
                                 head_width=0.02*scale, head_length=0.02*scale)

            # Expand limits to include both scores and loadings
            xs = np.concatenate([scores[:, 0], load_points[:, 0]])
            ys = np.concatenate([scores[:, 1], load_points[:, 1]])
            pad_x = 0.1 * (xs.max() - xs.min() + 1e-9)
            pad_y = 0.1 * (ys.max() - ys.min() + 1e-9)
            ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
            ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

        ax.legend(loc="best", fontsize=8, frameon=False)

    # Remove extra axes if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save alongside the input file
    if input_file_path:
        directory = os.path.dirname(input_file_path)
        output_path = os.path.join(directory, "PCA_multi_analysis_biplot_points.png")
        fig.savefig(output_path, dpi=300)
        messagebox.showinfo("Saved", f"Figure saved to:\n{output_path}")

    plt.show()
    
    
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox

def plot_loadings_only(
    color_by="magnitude",         # "magnitude" | "group" | "single"
    show_labels=False,            # label top-N loadings by magnitude
    top_n_labels=20,
    cmap="viridis",               # used when color_by="magnitude"
    point_size=16,
    alpha=0.6,
    arrow=False,                  # keep arrows off by default
    arrow_alpha=0.15,
    arrow_color="gray",
    same_axes_limits=True         # same axis limits across subplots
):
    """
    Plot only the PCA loadings (PC1_load vs PC2_load) for each analysis.
    - color_by="magnitude": color by vector norm (sqrt(PC1^2 + PC2^2))
    - color_by="group": assign loadings to the closest group direction (cosine similarity)
    - color_by="single": single color for all loadings
    """
    if len(analyses) == 0:
        messagebox.showerror("Error", "No analyses added.")
        return

    # Prepare subplot grid
    n = len(analyses)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Precompute global limits if requested
    global_x = []
    global_y = []

    # Helper to compute cosine similarity
    def cosine_sim(u, v):
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return -np.inf
        return float(np.dot(u, v) / (nu * nv))

    def get_group_colors(groups):
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        col_map = {}
        for i, g in enumerate(groups):
            col_map[g] = default_colors[i % len(default_colors)] if default_colors else None
        return col_map

    # First pass to gather global mins/maxes if needed
    for i, analysis in enumerate(analyses):
        load_df = analysis.get("loadings", None)
        if load_df is None or load_df.empty:
            continue
        xs = load_df["PC1_load"].values
        ys = load_df["PC2_load"].values
        global_x.extend(xs)
        global_y.extend(ys)

    # Plot each analysis
    for i, analysis in enumerate(analyses):
        ax = axes[i]
        title = analysis.get("title", f"Analysis {i+1}")
        load_df = analysis.get("loadings", None)

        ax.set_title(f"Loadings Only: {title}", fontsize=11)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.axvline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel("PC1 loading")
        ax.set_ylabel("PC2 loading")

        if load_df is None or load_df.empty:
            ax.text(0.5, 0.5, "No loadings available", ha="center", va="center", transform=ax.transAxes)
            continue

        # Loadings
        L = load_df[["PC1_load", "PC2_load"]].values
        names = load_df.index.astype(str).tolist()

        # Compute colors
        if color_by == "magnitude":
            mags = np.linalg.norm(L, axis=1)
            sc = ax.scatter(L[:, 0], L[:, 1], c=mags, s=point_size, cmap=cmap, alpha=alpha, edgecolor="none")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Loading magnitude", rotation=90)
            label_color = "black"

        elif color_by == "group":
            # Compute sample centroids by group from the PCA scores
            result_df = analysis["data"]
            if "group" not in result_df.columns or result_df["group"].isna().all():
                # No groups; fall back to single color
                ax.scatter(L[:, 0], L[:, 1], color="tab:blue", s=point_size, alpha=alpha, edgecolor="none")
                label_color = "tab:blue"
            else:
                groups = list(result_df["group"].dropna().unique())
                color_map = get_group_colors(groups)
                centroids = {g: result_df[result_df["group"] == g][["PC1", "PC2"]].mean().values for g in groups}

                group_buckets = {g: [] for g in groups}
                for j, vec in enumerate(L):
                    sims = {g: cosine_sim(vec, centroids[g]) for g in groups}
                    best_g = max(sims, key=sims.get)
                    group_buckets[best_g].append((vec[0], vec[1], names[j]))

                for g, pts in group_buckets.items():
                    if not pts:
                        continue
                    pts_arr = np.array([(x, y) for (x, y, _) in pts])
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                               color=color_map.get(g, None), s=point_size, alpha=alpha, edgecolor="none", label=str(g))
                ax.legend(loc="best", fontsize=8, frameon=False)
                label_color = "black"

        else:  # "single"
            ax.scatter(L[:, 0], L[:, 1], color="tab:blue", s=point_size, alpha=alpha, edgecolor="none")
            label_color = "tab:blue"

        # Optional arrows from origin to loading points
        if arrow:
            for x, y in L:
                ax.arrow(0, 0, x, y, color="gray", alpha=arrow_alpha,
                         length_includes_head=True, head_width=0.02, head_length=0.04)

        # Optional labels for top-N by magnitude
        if show_labels:
            mags = np.linalg.norm(L, axis=1)
            order = np.argsort(mags)[::-1][:min(top_n_labels, len(mags))]
            for idx in order:
                ax.text(L[idx, 0], L[idx, 1], names[idx],
                        fontsize=8, color=label_color, ha="left", va="bottom")

    # Unify axes if requested
    if same_axes_limits and len(global_x) > 0:
        xmin, xmax = min(global_x), max(global_x)
        ymin, ymax = min(global_y), max(global_y)
        dx = 0.1 * (xmax - xmin + 1e-9)
        dy = 0.1 * (ymax - ymin + 1e-9)
        for ax in axes[:len(analyses)]:
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)

    # Remove extra axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
import os
from datetime import datetime

def plot_loadings_only(
    color_by="magnitude",         # "magnitude" | "group" | "single"
    show_labels=False,            # label top-N loadings by magnitude
    top_n_labels=20,
    cmap="viridis",               # used when color_by="magnitude"
    point_size=16,
    alpha=0.6,
    arrow=False,                  # keep arrows off by default
    arrow_alpha=0.15,
    arrow_color="gray",
    same_axes_limits=True,        # same axis limits across subplots
    save=True,                    # <— NEW: control saving
    dpi=300                       # <— NEW: PNG resolution
):
    """
    Plot only the PCA loadings (PC1_load vs PC2_load) for each analysis.
    If save=True and input_file_path is set, save the figure next to the CSV.
    """
    # Access global input_file_path and analyses
    global input_file_path, analyses

    if len(analyses) == 0:
        messagebox.showerror("Error", "No analyses added.")
        return

    # Prepare subplot grid
    n = len(analyses)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Precompute global limits if requested
    global_x = []
    global_y = []

    # Helper to compute cosine similarity
    def cosine_sim(u, v):
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return -np.inf
        return float(np.dot(u, v) / (nu * nv))

    def get_group_colors(groups):
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        col_map = {}
        for i, g in enumerate(groups):
            col_map[g] = default_colors[i % len(default_colors)] if default_colors else None
        return col_map

    # First pass to gather global mins/maxes if needed
    for i, analysis in enumerate(analyses):
        load_df = analysis.get("loadings", None)
        if load_df is None or load_df.empty:
            continue
        xs = load_df["PC1_load"].values
        ys = load_df["PC2_load"].values
        global_x.extend(xs)
        global_y.extend(ys)

    # Plot each analysis
    for i, analysis in enumerate(analyses):
        ax = axes[i]
        title = analysis.get("title", f"Analysis {i+1}")
        load_df = analysis.get("loadings", None)

        ax.set_title(f"Loadings Only: {title}", fontsize=11)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.axvline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel("PC1 loading")
        ax.set_ylabel("PC2 loading")

        if load_df is None or load_df.empty:
            ax.text(0.5, 0.5, "No loadings available", ha="center", va="center", transform=ax.transAxes)
            continue

        # Loadings
        L = load_df[["PC1_load", "PC2_load"]].values
        names = load_df.index.astype(str).tolist()

        # Color logic
        if color_by == "magnitude":
            mags = np.linalg.norm(L, axis=1)
            sc = ax.scatter(L[:, 0], L[:, 1], c=mags, s=point_size, cmap=cmap, alpha=alpha, edgecolor="none")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Loading magnitude", rotation=90)
            label_color = "black"

        elif color_by == "group":
            result_df = analysis["data"]
            if "group" not in result_df.columns or result_df["group"].isna().all():
                ax.scatter(L[:, 0], L[:, 1], color="tab:blue", s=point_size, alpha=alpha, edgecolor="none")
                label_color = "tab:blue"
            else:
                groups = list(result_df["group"].dropna().unique())
                color_map = get_group_colors(groups)
                centroids = {g: result_df[result_df["group"] == g][["PC1", "PC2"]].mean().values for g in groups}

                group_buckets = {g: [] for g in groups}
                for j, vec in enumerate(L):
                    sims = {g: cosine_sim(vec, centroids[g]) for g in groups}
                    best_g = max(sims, key=sims.get)
                    group_buckets[best_g].append((vec[0], vec[1], names[j]))

                for g, pts in group_buckets.items():
                    if not pts:
                        continue
                    pts_arr = np.array([(x, y) for (x, y, _) in pts])
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                               color=color_map.get(g, None), s=point_size, alpha=alpha, edgecolor="none", label=str(g))
                ax.legend(loc="best", fontsize=8, frameon=False)
                label_color = "black"

        else:  # "single"
            ax.scatter(L[:, 0], L[:, 1], color="tab:blue", s=point_size, alpha=alpha, edgecolor="none")
            label_color = "tab:blue"

        # Optional arrows from origin to loading points
        if arrow:
            for x, y in L:
                ax.arrow(0, 0, x, y, color="gray", alpha=arrow_alpha,
                         length_includes_head=True, head_width=0.02, head_length=0.04)

        # Optional labels for top-N by magnitude
        if show_labels:
            mags = np.linalg.norm(L, axis=1)
            order = np.argsort(mags)[::-1][:min(top_n_labels, len(mags))]
            for idx in order:
                ax.text(L[idx, 0], L[idx, 1], names[idx],
                        fontsize=8, color=label_color, ha="left", va="bottom")

    # Unify axes if requested
    if same_axes_limits and len(global_x) > 0:
        xmin, xmax = min(global_x), max(global_x)
        ymin, ymax = min(global_y), max(global_y)
        dx = 0.1 * (xmax - xmin + 1e-9)
        dy = 0.1 * (ymax - ymin + 1e-9)
        for ax in axes[:len(analyses)]:
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)

    # Remove extra axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # ===== SAVE NEXT TO THE LOADED CSV =====
    if save and input_file_path:
        directory = os.path.dirname(input_file_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"PCA_loadings_only_{ts}.png"
        output_path = os.path.join(directory, out_name)
        fig.savefig(output_path, dpi=dpi)
        messagebox.showinfo("Saved", f"Figure saved to:\n{output_path}")

    plt.show()

# =========================
# MAIN GUI
# =========================
root = tk.Tk()
root.title("Multi-Metric PCA GUI")
root.geometry("380x240")

tk.Button(root, text="Load CSV File", command=load_file, height=2).pack(pady=10)
tk.Button(root, text="Add Analysis", command=open_analysis_window, height=2).pack(pady=10)
tk.Button(root, text="Plot & Save All Analyses", command=plot_loadings_only, height=2).pack(pady=10)

root.mainloop()