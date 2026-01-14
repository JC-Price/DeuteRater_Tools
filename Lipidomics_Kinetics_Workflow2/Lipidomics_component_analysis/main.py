"""This code was authored by Coleman Nielsen, with support from ChatGPT"""
# ================================================================
# MAIN SCRIPT (7. main)
#
# This is the entry point for the lipid component analysis workflow.
# It ties together all the modular code:
#
#   1. sampling.py       → draws asymmetric Gaussian samples, runs MC fits
#   2. stats_tests.py    → provides correlation-adjusted paired t-test
#   3. design_matrix.py  → builds the design matrix for lipid components
#   4. parsing.py        → extracts fatty acid tokens and structural groups
#   5. jackknife.py      → runs jackknife resampling with Monte Carlo fits
#   6. plotting.py       → makes bar plots and conformity scatter plots
#   7. main.py           → user interface (file dialogs), data prep, orchestration
#
# Step-by-step flow:
#   (A) Ask the user to select an input CSV file.
#   (B) Load the CSV into a pandas DataFrame.
#   (C) Clean and deduplicate lipid identifiers (build `dedup_key`).
#   (D) Detect all unique genotypes present in the data.
#   (E) Call plotting.linear_algebra_plot_from_tidy() to:
#       - Run jackknife estimation (5. jackknife)
#       - Generate bar plots of component n-values (6. plotting)
#   (F) Ask the user where to save the bar plot.
#   (G) Prompt the user to select a baseline genotype.
#   (H) Call plotting.conformity_plots_from_components() to:
#       - Build scatter plots comparing baseline vs other genotypes.
#   (I) Ask the user where to save conformity plots.
#   (J) Show all figures interactively with plt.show().
#
# All statistical and mathematical heavy lifting is delegated to the
# supporting modules (1–6). This script focuses only on I/O and orchestration.
# ================================================================



import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import matplotlib.pyplot as plt

# (6. plotting) imports
from plotting import linear_algebra_plot_from_tidy, conformity_plots_from_components

if __name__ == "__main__":
    # (A) Create a hidden Tk root window for dialogs
    root = tk.Tk()
    root.withdraw()

    # (A) Ask user for CSV file
    input_path = filedialog.askopenfilename(
        title="Select tidy lipid regression CSV with CI",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not input_path:
        raise SystemExit("No file selected.")

    # (B) Load data
    
    
    df = pd.read_csv(input_path)
    # (B) Keep only n_value metric rows
    df = df[df["metric"].fillna("").str.strip().str.lower() == "n_value"].copy()
    print(len(df))
    print(df.head())
    # (C) Build deduplication key
    df['dedup_key'] = (
        df['Alignment ID']
          .astype(str)
          .str.replace(r'\(\s*[Mm][^)]*\)', '', regex=True)
          .str.replace('/0:0', '', regex=False)
          .str.split('|').str[0]
          .str.strip()
    )
    df = df.drop_duplicates(subset=['dedup_key', 'genotype'], keep='first')
    print(len(df))
    print(df.head())

    # (D) Detect unique identifiers
    identifiers = sorted(df["genotype"].dropna().unique())
    print("Identifiers found:", identifiers)

    # (E) Run component estimation + bar plotting
    fig, ax, comp_df = linear_algebra_plot_from_tidy(
        df=df,
        identifiers=identifiers,
        y_max=None,
        title=f"Component {r'$n_L$'} by Genotype",
        num_jackknifes=1000,
        monte_carlo_per_jackknife=500,
        seed=0
    )

    # (F) Ask where to save bar plot
    out_path = filedialog.asksaveasfilename(
        title="Save bar plot as...",
        defaultextension=".svg",
        filetypes=[("SVG image", "*.svg"), ("PNG image", "*.png"), ("PDF file", "*.pdf")]
    )
    if out_path:
        fig.savefig(out_path, dpi=300)
        print("Bar plot saved to:", out_path)

        # (G) Ask for baseline genotype
        baseline = simpledialog.askstring(
            "Baseline Selection",
            f"Available genotypes:\n{', '.join(identifiers)}\n\nEnter baseline genotype:"
        )
        if baseline in identifiers:
            print("Baseline selected:", baseline)

            # (H) Build conformity plots
            fig2 = conformity_plots_from_components(df, comp_df, identifiers, baseline)
            fig2.tight_layout()

            # (I) Ask where to save conformity plots
            out_path2 = filedialog.asksaveasfilename(
                title="Save conformity plot as...",
                defaultextension=".svg",
                filetypes=[("SVG image", "*.svg"), ("PNG image", "*.png"), ("PDF file", "*.pdf")]
            )
            if out_path2:
                fig2.savefig(out_path2, dpi=300)
                print("Conformity plot saved to:", out_path2)
                
    from plotting import plot_palmitate_stearate_comparison
    
    # After linear_algebra_plot_from_tidy()
    fig3, ax3 = plot_palmitate_stearate_comparison(comp_df, identifiers)
    
    out_path3 = filedialog.asksaveasfilename(
        title="Save 16:0 vs 18:0 comparison plot as...",
        defaultextension=".svg",
        filetypes=[("SVG image", "*.svg"), ("PNG image", "*.png"), ("PDF file", "*.pdf")]
    )
    if out_path3:
        fig3.savefig(out_path3, dpi=300)
        print("16:0 vs 18:0 comparison saved to:", out_path3)
        
        
    from plotting import plot_theoretical_vs_empirical  # <-- add to imports

    # ... inside __main__ after you have comp_df and identifiers ...
    
    # (K) Theoretical vs Empirical comparison
    fig4, ax4, merged = plot_theoretical_vs_empirical(
        comp_df=comp_df,
        identifiers=identifiers,
        precursor_type="de novo",   # or "dietary"
        title="Theoretical vs Empirical n-values"
    )
    
    out_path4 = filedialog.asksaveasfilename(
        title="Save theoretical vs empirical plot as...",
        defaultextension=".svg",
        filetypes=[("SVG image", "*.svg"), ("PNG image", "*.png"), ("PDF file", "*.pdf")]
    )
    if out_path4:
        fig4.savefig(out_path4, dpi=300)
        print("Theoretical vs Empirical plot saved to:", out_path4)
    


    # (J) Show results
    plt.show()


