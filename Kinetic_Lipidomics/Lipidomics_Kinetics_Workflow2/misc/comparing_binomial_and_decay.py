# -*- coding: utf-8 -*-
"""
Generate a 4‑panel plot with:
  • Conformity plots (Huber regression + mean residual):
        - Binomial rate vs Exponential decay rate
        - Binomial Asyn(syn subscript) vs Exponential decay Asyn

  • Bland–Altman plots (canonical: mean difference ± 1.96 SD), with genotype colors:
        - Binomial rate vs Exponential decay rate
        - Binomial Asyn(syn) vs Exponential decay Asyn

All filters are applied identically to both comparisons.

All text is 12‑point.

Usage:
    python make_ba_4panel.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from datetime import datetime

# =====================================================================
# CONFIG
# =====================================================================

HUBER_DELTA = 1.345

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12,
})

# =====================================================================
# ROBUST HELPERS
# =====================================================================

def mad(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def weighted_linreg(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]

    if x.size < 2:
        return float("nan"), float("nan")

    sw = np.sum(w)
    xw = np.sum(w * x)
    yw = np.sum(w * y)
    xxw = np.sum(w * x * x)
    xyw = np.sum(w * x * y)

    denom = sw * xxw - xw * xw
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")

    m = (sw * xyw - xw * yw) / denom
    b = (yw - m * xw) / sw
    return float(m), float(b)


def huber_regression(x, y, delta=HUBER_DELTA, max_iter=50, tol=1e-6):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if x.size < 2:
        return float("nan"), float("nan"), float("nan")

    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        m = 0.0
        b = float(np.median(y))

    for _ in range(max_iter):
        r = y - (m * x + b)
        s = mad(r)
        if not math.isfinite(s) or s < 1e-12:
            s = max(float(np.std(r, ddof=1)), 1e-6)

        t = r / s
        abs_t = np.abs(t)
        w = np.ones_like(t)
        w[abs_t > delta] = delta / abs_t[abs_t > delta]

        m_new, b_new = weighted_linreg(x, y, w)
        if not (math.isfinite(m_new) and math.isfinite(b_new)):
            return float("nan"), float("nan"), float("nan")

        if (abs(m_new - m) <= tol * (1 + abs(m)) and
            abs(b_new - b) <= tol * (1 + abs(b))):
            m, b = m_new, b_new
            break

        m, b = m_new, b_new

    return float(m), float(b), float(mad(y - (m * x + b)))


# =====================================================================
# PLOTTING: CONFORMITY PANEL  (WITH MEAN RESIDUAL)
# =====================================================================

def plot_conformity(ax, x, y, hue, title, xlab, ylab):
    palette = {"APOE2": "#0072B2", "APOE3": "#009E73", "APOE4": "#D55E00"}

    df = pd.DataFrame({"x": x, "y": y, "h": hue}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)

    for gt in ["APOE2", "APOE3", "APOE4"]:
        sub = df[df.h == gt]
        if not sub.empty:
            ax.scatter(
                sub.x, sub.y, s=26,
                color=palette.get(gt, "gray"),
                alpha=0.85, edgecolor="white", linewidth=0.4, label=str(gt)
            )

    m, b, _ = huber_regression(df.x.values, df.y.values)

    if math.isfinite(m):
        xx = np.linspace(np.min(df.x), np.max(df.x), 200)
        yy = m * xx + b

        ax.plot(xx, yy, color="black", lw=2)

        residuals = df.y.values - (m * df.x.values + b)
        res_mean = float(np.mean(residuals))

        ax.fill_between(xx, yy - res_mean, yy + res_mean,
                        color="0.4", alpha=0.18, linewidth=0)

        y_hat = m * df.x.values + b
        ss_res = np.sum((df.y.values - y_hat)**2)
        ss_tot = np.sum((df.y.values - np.mean(df.y.values))**2)
        r2 = float("nan") if ss_tot == 0 else 1 - ss_res/ss_tot

        subtitle = (
            f"y = {m:.3f}x + {b:.3f}   |   n = {n}   |   R² = {r2:.3f}\n"
            f"Mean(residuals) = {res_mean:.3f}"
        )
    else:
        subtitle = f"Insufficient points (n = {n})"
        res_mean = float("nan")

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.5, 1.04, subtitle, ha="center", va="bottom",
            transform=ax.transAxes)

    return n, m, b, res_mean


# =====================================================================
# PLOTTING: BLAND–ALTMAN PANEL  (UPDATED FOR GENOTYPE COLORS)
# =====================================================================

def bland_altman(ax, x, y, hue, title, xlab, ylab):
    palette = {"APOE2": "#0072B2", "APOE3": "#009E73", "APOE4": "#D55E00"}

    df = pd.DataFrame({"x": x, "y": y, "h": hue}).dropna()
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center")
        return float("nan"), float("nan"), float("nan")

    d = df.y - df.x
    m = (df.y + df.x) / 2

    mean_diff = float(np.mean(d))
    sd_diff = float(np.std(d, ddof=1))
    loa = 1.96 * sd_diff

    # genotype-colored points
    for gt in ["APOE2", "APOE3", "APOE4"]:
        sub = df[df.h == gt]
        if not sub.empty:
            ax.scatter(
                (sub.y + sub.x) / 2,
                (sub.y - sub.x),
                s=26,
                color=palette.get(gt, "gray"),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                label=str(gt)
            )

    ax.axhline(mean_diff, color="blue", lw=2, label="Mean difference")
    ax.axhline(mean_diff + loa, color="red", linestyle="--", lw=1.2, label="95% LoA")
    ax.axhline(mean_diff - loa, color="red", linestyle="--", lw=1.2)

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    return mean_diff, sd_diff, loa


# =====================================================================
# DATA PREP / FILTERS
# =====================================================================

def load_filtered(csv):
    df = pd.read_csv(csv)

    M_RATE = "BA_rate"
    M_ABUND_RATE = "Abundance rate"
    M_ASYN = "BA_Asyn"
    M_ASYM = "Abundance asymptote"
    M_NVAL = "n_value"

    keys = ["Alignment ID", "Ontology", "sample_id", "genotype"]

    def pick(metric, vname, sename):
        tmp = df[df.metric == metric][keys + ["value", "se"]].copy()
        tmp = tmp.rename(columns={"value": vname, "se": sename})
        return tmp

    d_rate  = pick(M_RATE,        "ba_rate",        "ba_se")
    d_abund = pick(M_ABUND_RATE,  "abundance_rate", "abundance_se")
    d_asyn  = pick(M_ASYN,        "ba_asyn",        "ba_asyn_se")
    d_asym  = pick(M_ASYM,        "asymptote",      "asymptote_se")
    d_nval  = pick(M_NVAL,        "n_value",        "n_value_se")

    common = (
        d_rate.merge(d_abund, on=keys, how="inner")
              .merge(d_asyn,  on=keys, how="inner")
              .merge(d_nval,  on=keys, how="inner")
    )

    mask = (
        (common.ba_rate <= 1) &
        (common.ba_se <= 3) &
        (common.ba_asyn_se <= 3) &
        (common.n_value_se <= 3) &
        (common.ba_asyn < 1)
    )

    base = common[mask].copy()

    p1 = base.dropna(subset=["abundance_rate", "ba_rate"]).copy()

    p2 = base.merge(d_asym[keys + ["asymptote"]], on=keys, how="inner")
    p2 = p2.dropna(subset=["asymptote", "ba_asyn"]).copy()

    return p1, p2


# =====================================================================
# MAIN
# =====================================================================

def main():
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select lipid CSV",
        filetypes=[("CSV","*.csv")]
    )
    root.update()
    root.destroy()

    if not csv_path:
        print("No file selected.")
        return

    outdir = os.path.dirname(csv_path)
    csv_name = os.path.basename(csv_path)

    plot1, plot2 = load_filtered(csv_path)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # -------------------------------------------------------
    # RATE COMPARISON PANEL
    # -------------------------------------------------------
    n1, m1, b1, mean_res1 = plot_conformity(
        axs[0,0],
        x=plot1.abundance_rate.values,
        y=plot1.ba_rate.values,
        hue=plot1.genotype.values,
        title="Binomial rate vs Exponential decay rate",
        xlab="Exponential decay rate",
        ylab="Binomial rate"
    )

    ba_stats1 = bland_altman(
        axs[0,1],
        x=plot1.abundance_rate.values,
        y=plot1.ba_rate.values,
        hue=plot1.genotype.values,
        title="Bland–Altman: Binomial rate vs Exponential decay rate",
        xlab="Mean of (Binomial rate and Exponential decay rate)",
        ylab="Difference (Binomial rate − Exponential decay rate)"
    )

    # -------------------------------------------------------
    # ASYN COMPARISON PANEL
    # -------------------------------------------------------
    n2, m2, b2, mean_res2 = plot_conformity(
        axs[1,0],
        x=plot2.asymptote.values,
        y=plot2.ba_asyn.values,
        hue=plot2.genotype.values,
        title="Binomial A$_{syn}$ vs Exponential decay A$_{syn}$",
        xlab="Exponential decay A$_{syn}$",
        ylab=r"Binomial A$_{syn}$"
    )

    ba_stats2 = bland_altman(
        axs[1,1],
        x=plot2.asymptote.values,
        y=plot2.ba_asyn.values,
        hue=plot2.genotype.values,
        title="Bland–Altman: Binomial A$_{syn}$ vs Exponential decay A$_{syn}$",
        xlab="Mean of (Binomial A$_{syn}$ and Exponential decay A$_{syn}$)",
        ylab="Difference (Binomial A$_{syn}$ − Exponential decay A$_{syn}$)"
    )

    # -------------------------------------------------------
    fig.tight_layout()
    out_png = os.path.join(outdir, "4panel_conformity_bland_altman.svg")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_png}")

    # -------------------------------------------------------
    # TEXT SUMMARY OUTPUT
    # -------------------------------------------------------
    txt_path = os.path.join(outdir, "4panel_results.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"Input file: {csv_name}")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("Filters applied:")
    lines.append("  BA_rate ≤ 1")
    lines.append("  BA_rate SE ≤ 3")
    lines.append("  BA_A$_{syn}$ SE ≤ 3")
    lines.append("  n_value SE ≤ 3")
    lines.append("  BA_A$_{syn}$ < 1")
    lines.append("")
    lines.append("Conformity plots use Huber regression with mean residual.")
    lines.append("Bland–Altman uses mean difference ± 1.96 SD.")
    lines.append("")

    # -----------------
    mean_diff1, sd1, loa1 = ba_stats1
    lines.append("PLOT 1 — Binomial rate vs Exponential decay rate")
    lines.append(f"  n = {n1}")
    lines.append(f"  Huber slope = {m1:.6f}, intercept = {b1:.6f}")
    lines.append(f"  Mean residual = {mean_res1:.6f}")
    lines.append(f"  BA mean diff = {mean_diff1:.6f}")
    lines.append(f"  BA SD(diff)  = {sd1:.6f}")
    lines.append(f"  BA LoA ±     = {loa1:.6f}")
    lines.append("")

    # -----------------
    mean_diff2, sd2, loa2 = ba_stats2
    lines.append("PLOT 2 — Binomial Asyn(syn) vs Exponential decay Asyn")
    lines.append(f"  n = {n2}")
    lines.append(f"  Huber slope = {m2:.6f}, intercept = {b2:.6f}")
    lines.append(f"  Mean residual = {mean_res2:.6f}")
    lines.append(f"  BA mean diff = {mean_diff2:.6f}")
    lines.append(f"  BA SD(diff)  = {sd2:.6f}")
    lines.append(f"  BA LoA ±     = {loa2:.6f}")
    lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote TXT summary: {txt_path}")


if __name__ == "__main__":
    main()