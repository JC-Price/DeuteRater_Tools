import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
"""The input file for this is the Fractioin New file"""

def parse_abundance_string(s):
    if pd.isna(s):
        return []
    s = s.strip().strip("()")
    parts = [p.strip() for p in s.split(",")]
    return [float(x) for x in parts if x]


def process_file(filepath):
    df = pd.read_csv(filepath, sep="\t")

    # Normalize time to nearest official timepoint
    official_times = [0, 0.25, 1, 2, 4, 16, 32]

    def round_time(t):
        try:
            t = float(t)
            return min(official_times, key=lambda x: abs(x - t))
        except:
            return t

    df["time"] = df["time"].apply(round_time)

    # Parse abundance column
    df["abundance_list"] = df["abundances"].apply(parse_abundance_string)

    # Compute metrics
    df["abundance_sum"] = df["abundance_list"].apply(sum)
    df["first_abundance"] = df["abundance_list"].apply(
        lambda x: x[0] if len(x) > 0 else np.nan
    )

    # Group by Alignment ID + sample_group + time
    result = df.groupby(["Alignment ID", "sample_group", "time"]).agg(
        avg_abundance_sum=("abundance_sum", "mean"),
        avg_first_abundance=("first_abundance", "mean")
    ).reset_index()

    # Normalize within each Alignment ID + group across time
    def normalize_group(subdf):
        max_sum = subdf["avg_abundance_sum"].max()
        max_first = subdf["avg_first_abundance"].max()

        subdf["norm_abundance_sum"] = (
            subdf["avg_abundance_sum"] / max_sum if max_sum != 0 else 0
        )
        subdf["norm_first_abundance"] = (
            subdf["avg_first_abundance"] / max_first if max_first != 0 else 0
        )
        subdf["diff"] = subdf["norm_first_abundance"] - subdf["norm_abundance_sum"]

        return subdf

    result = result.groupby(["Alignment ID", "sample_group"], group_keys=False).apply(normalize_group)

    return result


def summarize_trend(result_df):
    return result_df.groupby(["sample_group", "time"]).agg(
        mean_diff=("diff", "mean"),
        sem_diff=("diff", "sem")
    ).reset_index()


# NEW: Compute under-prediction at Day 32
def compute_day32_underprediction(trend_df):
    day32 = trend_df[trend_df["time"] == 32]
    return day32[["sample_group", "mean_diff"]]





def plot_trend_svg(result_df, trend_df, output_path):
    plt.figure(figsize=(6, 4))

    # Custom color palette (magenta, orange, teal)
    colors = ["#D81B60", "#E69F00", "#009E73"]
    groups = list(trend_df["sample_group"].unique())
    color_map = {g: colors[i % len(colors)] for i, g in enumerate(groups)}

    # --------------------------------------------------
    # Plot mean lines + 95% CI shading (no individual lipid traces)
    # --------------------------------------------------
    for group in groups:
        sub = trend_df[trend_df["sample_group"] == group].sort_values("time")
        c = color_map[group]

        mean = sub["mean_diff"]
        sem = sub["sem_diff"]

        # 95% CI
        ci_upper = mean + 1.96 * sem
        ci_lower = mean - 1.96 * sem

        # Shaded CI band
        plt.fill_between(
            sub["time"],
            ci_lower,
            ci_upper,
            color=c,
            alpha=0.25,
            linewidth=0
        )

        # Mean line + markers
        plt.plot(
            sub["time"],
            mean,
            marker='o',
            color=c,
            linewidth=2,
            label=group
        )

    # --- Label underestimation at Day 32 ---
    day32 = trend_df[trend_df["time"] == 32]
    for _, row in day32.iterrows():
        group = row["sample_group"]
        yval = row["mean_diff"]
        c = color_map[group]

        plt.text(
            32 + 0.3,
            yval,
            f"{yval:.3f}",
            color=c,
            fontsize=8,
            va='center'
        )

    # Reference line
    plt.axhline(0, linestyle='--', linewidth=1)

    # Labels
    plt.xlabel("Time")
    plt.ylabel(plt.ylabel("Normalized Difference\n(M0 − Σ(M0–M2))"))
    plt.title("Increasing Underestimation from Monoisotopic Signal")

    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(output_path, format="svg")
    plt.close()
def select_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv")]
    )

    if not filepath:
        return

    try:
        result_df = process_file(filepath)
        trend_df = summarize_trend(result_df)

        # NEW: Compute Day 32 values
        day32_df = compute_day32_underprediction(trend_df)

        base, ext = os.path.splitext(filepath)

        # Save outputs
        processed_path = base + "_processed.csv"
        trend_path = base + "_trend.csv"
        day32_path = base + "_day32_underprediction.csv"
        svg_path = base + "_figure.svg"

        result_df.to_csv(processed_path, index=False)
        trend_df.to_csv(trend_path, index=False)
        day32_df.to_csv(day32_path, index=False)
        plot_trend_svg(result_df, trend_df, svg_path)

        messagebox.showinfo(
            "Success",
            f"Saved:\n{processed_path}\n{trend_path}\n{day32_path}\n{svg_path}"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    root.title("Alignment Time Analysis Tool")
    root.geometry("400x200")

    label = tk.Label(root, text="Select a data file to analyze", pady=20)
    label.pack()

    button = tk.Button(root, text="Browse File", command=select_file)
    button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
