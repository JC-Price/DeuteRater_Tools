# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:08:44 2026

@author: Brigham Young Univ
"""
#!/usr/bin/env python3
"""
Generates lipid pathway fold-change network plots AND exports an edge flux table.

- Uses Tkinter to select the input CSV.
- Saves output PNGs/SVGs and edge_flux_table.csv to the SAME DIRECTORY as the input CSV.
- Adds LPC/LPE/LPS reversible branches and lyso-specific out sinks.
- Removes direct "out" directions for PE, PC, and PS.
Use ttest output from post-hoc analysis gui.
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ONTOLOGY_TO_NODE = {
    'Ontology_DG':  'A',
    'Ontology_PE':  'B',
    'Ontology_PC':  'C',
    'Ontology_PS':  'D',
    'Ontology_LPE': 'LPE',
    'Ontology_LPC': 'LPC',
    'Ontology_LPS': 'LPS',
}

METRIC      = 'Flux'
LOG2FC_COL  = 'Mean_Diff_All'
PVAL_COL    = 'p_All'
ALPHA       = 0.05
TOL         = 0.05

NODE_LABELS = {
    'In':      'In',
    'A':       'DG',
    'B':       'PE',
    'C':       'PC',
    'D':       'PS',
    'LPE':     'LPE',
    'LPC':     'LPC',
    'LPS':     'LPS',
    'A_out':   'DG\nout',
    'LPE_out': 'LPE\nout',
    'LPC_out': 'LPC\nout',
    'LPS_out': 'LPS\nout',
}

EDGES = [
    ('In',  'A',       'In→A'),
    ('A',   'B',       'A→B'),
    ('A',   'C',       'A→C'),
    ('A',   'A_out',   'A→out'),

    ('B',   'D',       'B→D'),
    ('C',   'D',       'C→D'),
    ('D',   'B',       'D→B'),

    ('B',   'C',       'B→C'),   # PE → PC via PEMT

    ('B',   'LPE',     'B→LPE'),
    ('LPE', 'B',       'LPE→B'),
    ('LPE', 'LPE_out', 'LPE→out'),

    ('C',   'LPC',     'C→LPC'),
    ('LPC', 'C',       'LPC→C'),
    ('LPC', 'LPC_out', 'LPC→out'),

    ('D',   'LPS',     'D→LPS'),
    ('LPS', 'D',       'LPS→D'),
    ('LPS', 'LPS_out', 'LPS→out'),
]

EDGE_ENZYMES = {
    'In→A':    'DG Import',
    'A→B':     'EPT',
    'A→C':     'CPT',
    'A→out':   'DG Export/Degradation',

    'B→D':     'PSS2',
    'C→D':     'PSS1',
    'D→B':     'PS Decarboxylase',

    'B→C':     'PEMT (SAM→SAH)',

    'B→LPE':   'PLA',
    'LPE→B':   'LPEAT',
    'LPE→out': 'LPE Export/Degradation',

    'C→LPC':   'PLA',
    'LPC→C':   'LPCAT',
    'LPC→out': 'LPC Export/Degradation',

    'D→LPS':   'PLA',
    'LPS→D':   'LPS Acyltransferase',
    'LPS→out': 'LPS Export/Degradation',
}

# UniProt accession numbers — Homo sapiens
# N/A = non-enzymatic transport/degradation step
# Gene symbols shown in comments for cross-reference
ENZYME_ACCESSIONS_HUMAN = {
    'DG Import':               'N/A',
    'EPT':                     'P48583',   # CEPT1  (ethanolaminephosphotransferase 1)
    'CPT':                     'P16880',   # CHPT1  (choline phosphotransferase 1; choline-specific)
    'DG Export/Degradation':   'N/A',
    'PSS2':                    'O14494',   # PTDSS2 (phosphatidylserine synthase 2)
    'PSS1':                    'P48651',   # PTDSS1 (phosphatidylserine synthase 1)
    'PS Decarboxylase':        'P0CG43',   # PISD   (phosphatidylserine decarboxylase)
    'PEMT (SAM→SAH)':          'Q9UBM1',   # PEMT   (phosphatidylethanolamine N-methyltransferase)
    'PLA':                     'P39877',   # PLA2G4A (cPLA2α; cytosolic, dominant remodelling isoform)
    'LPEAT':                   'Q6UWP7',   # LPEAT2 / AGPAT7 (lysophosphatidylethanolamine acyltransferase 2)
    'LPE Export/Degradation':  'N/A',
    'LPCAT':                   'Q86YP8',   # LPCAT1 (lysophosphatidylcholine acyltransferase 1)
    'LPC Export/Degradation':  'N/A',
    'LPS Acyltransferase':     'Q9Y259',   # AGPAT5 (lysophosphatidylserine acyltransferase / LPSAT)
    'LPS Export/Degradation':  'N/A',
}

# UniProt accession numbers — Mus musculus
ENZYME_ACCESSIONS_MOUSE = {
    'DG Import':               'N/A',
    'EPT':                     'Q9Z1N4',   # Cept1
    'CPT':                     'Q8BFC2',   # Chpt1
    'DG Export/Degradation':   'N/A',
    'PSS2':                    'Q9WU60',   # Ptdss2
    'PSS1':                    'Q9WU67',   # Ptdss1
    'PS Decarboxylase':        'Q9DCX4',   # Pisd
    'PEMT (SAM→SAH)':          'Q9Z1N3',   # Pemt
    'PLA':                     'P47713',   # Pla2g4a (cPLA2α)
    'LPEAT':                   'Q8BH22',   # Lpeat2 / Agpat7
    'LPE Export/Degradation':  'N/A',
    'LPCAT':                   'Q8BYI6',   # Lpcat1
    'LPC Export/Degradation':  'N/A',
    'LPS Acyltransferase':     'Q9Z1T2',   # Agpat5
    'LPS Export/Degradation':  'N/A',
}

POS = {
    'In':      (-1.5,  0.0),
    'A':       ( 0.0,  0.0),
    'B':       ( 2.0,  1.35),
    'C':       ( 2.0, -1.35),
    'D':       ( 4.4,  0.0),
    'LPE':     ( 3.2,  2.45),
    'LPC':     ( 3.2, -2.45),
    'LPS':     ( 6.0,  0.0),
    'A_out':   ( 0.9, -2.35),
    'LPE_out': ( 4.55,  3.00),
    'LPC_out': ( 4.55, -3.00),
    'LPS_out': ( 7.35,  0.0),
}

main_nodes = ['A', 'B', 'C', 'D', 'LPE', 'LPC', 'LPS']
sink_nodes = ['A_out', 'LPE_out', 'LPC_out', 'LPS_out']

NODE_R = 0.30
SINK_R = 0.16

BG        = '#f8f9fa'
NODE_FACE = '#ffffff'
NODE_EDGE = '#333333'
SINK_FACE = '#eeeeee'
SINK_EDGE = '#888888'
GREEN     = '#2ca02c'
RED       = '#d62728'
GRAY      = '#9e9e9e'
TEXT_DARK = '#222222'
SIG_RING  = '#e6a817'


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ratio_color(fc):
    if np.isnan(fc):
        return GRAY
    if fc > 1 + TOL:
        return GREEN
    if fc < 1 - TOL:
        return RED
    return GRAY


def infer_edge_fc(s, d, R_node, alpha_ratio=None, edge_key=None):
    if s == 'In' and d == 'A':
        return R_node.get('A', 1.0)
    if alpha_ratio is not None and edge_key in alpha_ratio:
        return alpha_ratio[edge_key] * R_node.get(s, 1.0)
    return R_node.get(s, 1.0)


def draw_straight_arrow(ax, src, dst, label, color, normal_offset=0.13):
    x1, y1 = POS[src]
    x2, y2 = POS[dst]
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return

    ux, uy = dx / L, dy / L
    r_s = SINK_R if src in sink_nodes else NODE_R
    r_d = SINK_R if dst in sink_nodes else NODE_R

    start = (x1 + ux * r_s, y1 + uy * r_s)
    end   = (x2 - ux * r_d, y2 - uy * r_d)

    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=14,
        linewidth=1.8,
        color=color,
        zorder=2
    ))

    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    nx, ny = -uy, ux

    ax.text(
        mx + normal_offset * nx,
        my + normal_offset * ny,
        label,
        fontsize=7.5,
        ha='center',
        va='center',
        color=color,
        bbox=dict(boxstyle='round,pad=0.18', facecolor=BG, edgecolor='none', alpha=0.88),
        zorder=4
    )


def draw_curved_arrow(ax, src, dst, label, color, rad=0.30, label_pad=0.08):
    x1, y1 = POS[src]
    x2, y2 = POS[dst]

    r_s = SINK_R if src in sink_nodes else NODE_R
    r_d = SINK_R if dst in sink_nodes else NODE_R

    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f'arc3,rad={rad}',
        arrowstyle='-|>',
        mutation_scale=14,
        linewidth=1.8,
        color=color,
        zorder=2,
        shrinkA=r_s * 72,
        shrinkB=r_d * 72
    ))

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return

    nx, ny = -dy / L, dx / L
    bulge = rad * L / 2

    ax.text(
        mx + (bulge + label_pad) * nx,
        my + (bulge + label_pad) * ny,
        label,
        fontsize=7.5,
        ha='center',
        va='center',
        color=color,
        bbox=dict(boxstyle='round,pad=0.18', facecolor=BG, edgecolor='none', alpha=0.88),
        zorder=4
    )


# ------------------------------------------------------------
# Edge flux DataFrame builder
# ------------------------------------------------------------

def build_edge_flux_df(comparison, R_node):
    """
    Returns a DataFrame with one row per edge:
      Comparison, Source, Target, Edge, Enzyme,
      Accession_Human, Accession_Mouse, Edge_FC
    """
    rows = []
    for (s, d, key) in EDGES:
        enzyme    = EDGE_ENZYMES.get(key, '')
        acc_human = ENZYME_ACCESSIONS_HUMAN.get(enzyme, 'N/A')
        acc_mouse = ENZYME_ACCESSIONS_MOUSE.get(enzyme, 'N/A')
        fc        = infer_edge_fc(s, d, R_node)
        rows.append({
            'Comparison':      comparison,
            'Source':          s,
            'Target':          d,
            'Edge':            key,
            'Enzyme':          enzyme,
            'Accession_Human': acc_human,
            'Accession_Mouse': acc_mouse,
            'Edge_FC':         round(fc, 6),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

def draw_comparison(comparison, R_node, sig_nodes, out_path):
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-2.3, 8.1)
    ax.set_ylim(-3.7, 3.7)

    # --- Edges ---
    for (s, d, key) in EDGES:
        fc          = infer_edge_fc(s, d, R_node)
        col         = ratio_color(fc)
        enzyme_name = EDGE_ENZYMES.get(key, '')
        lbl         = f"{enzyme_name}\n{fc:.2f}"

        if key in ('B→D', 'C→D'):
            n_off = 0.20
        elif key in ('B→LPE', 'C→LPC', 'D→LPS'):
            n_off = 0.16
        elif key.endswith('→out'):
            n_off = 0.10
        else:
            n_off = 0.13

        if key == 'D→B':
            draw_curved_arrow(ax, s, d, lbl, col, rad=0.34, label_pad=0.12)
        elif key in ('LPE→B', 'LPC→C', 'LPS→D'):
            draw_curved_arrow(ax, s, d, lbl, col, rad=-0.26, label_pad=0.09)
        elif key == 'B→C':
            draw_curved_arrow(ax, s, d, lbl, col, rad=0.28, label_pad=0.10)
        else:
            draw_straight_arrow(ax, s, d, lbl, color=col, normal_offset=n_off)

    # --- Main nodes ---
    for n in main_nodes:
        x, y     = POS[n]
        edge_col = SIG_RING if n in sig_nodes else NODE_EDGE
        edge_lw  = 2.8 if n in sig_nodes else 1.5

        ax.add_patch(Circle(
            (x, y), NODE_R,
            facecolor=NODE_FACE,
            edgecolor=edge_col,
            linewidth=edge_lw,
            zorder=3
        ))
        ax.text(
            x, y, NODE_LABELS[n],
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=TEXT_DARK, zorder=5,
            linespacing=1.3
        )

    # --- Sink nodes ---
    for n in sink_nodes:
        x, y = POS[n]
        ax.add_patch(Circle(
            (x, y), SINK_R,
            facecolor=SINK_FACE,
            edgecolor=SINK_EDGE,
            linewidth=1.0,
            linestyle='--',
            zorder=3
        ))
        ax.text(
            x, y, NODE_LABELS[n],
            ha='center', va='center',
            fontsize=7, color='#555555',
            zorder=5, linespacing=1.2
        )

    # --- Node FC annotations ---
    for n in ['A', 'B', 'C', 'D', 'LPE', 'LPC', 'LPS']:
        x, y = POS[n]
        fc   = R_node.get(n, 1.0)
        ax.text(
            x, y - NODE_R - 0.13,
            f"{fc:.2f}",
            ha='center', va='top',
            fontsize=7.8,
            color=ratio_color(fc),
            fontweight='bold',
            zorder=5
        )

    # --- Title ---
    ax.set_title(
        f'Lipid pathway — fold changes  ({comparison})\n'
        f'Metric: {METRIC}  |  Node FC = 2^log2FC  |  Edge FC = source node FC\n'
        f'* p < {ALPHA} (gold border)',
        fontsize=10.5, fontweight='bold', color=TEXT_DARK,
        pad=12, loc='left', x=0.01, linespacing=1.5
    )

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], color=GREEN,    lw=2.2, label=f'FC > {1+TOL:.2f}  (increase)'),
        Line2D([0], [0], color=RED,      lw=2.2, label=f'FC < {1-TOL:.2f}  (decrease)'),
        Line2D([0], [0], color=GRAY,     lw=2.2, label=f'FC ≈ 1  (±{int(TOL*100)}%)'),
        Line2D([0], [0], color=SIG_RING, lw=2.2, label=f'p < {ALPHA} (significant)'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        edgecolor='#cccccc'
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved plot: {out_path}")


# ------------------------------------------------------------
# File selection
# ------------------------------------------------------------

def select_input_csv() -> str:
    if not TK_AVAILABLE:
        print(
            "Tkinter is not available. Provide a CSV path as an argument:\n"
            "  python script.py /path/to/paired_ttest_statistics.csv",
            file=sys.stderr
        )
        if len(sys.argv) >= 2 and os.path.isfile(sys.argv[1]):
            return sys.argv[1]
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.update()

    path = filedialog.askopenfilename(
        title="Select the paired_ttest_statistics.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.update()

    if not path:
        try:
            messagebox.showinfo("Canceled", "No file selected. Exiting.")
        except Exception:
            pass
        sys.exit(0)

    return path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    csv_path = select_input_csv()
    out_dir  = os.path.dirname(csv_path)

    print(f"Input CSV      : {csv_path}")
    print(f"Output directory: {out_dir}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {'Plot_Group', 'Metric', LOG2FC_COL, PVAL_COL, 'Comparison'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    keep   = list(ONTOLOGY_TO_NODE.keys())
    df_sub = df[
        df['Plot_Group'].isin(keep) &
        (df['Metric'] == METRIC)
    ].copy()

    if df_sub.empty:
        print("No rows match the selected ONTOLOGY classes and METRIC. Nothing to plot.")
        sys.exit(0)

    df_sub['node'] = df_sub['Plot_Group'].map(ONTOLOGY_TO_NODE)
    df_sub['fc']   = 2.0 ** df_sub[LOG2FC_COL]
    df_sub['sig']  = df_sub[PVAL_COL] < ALPHA

    comparisons = df_sub['Comparison'].unique()
    print(f"Found {len(comparisons)} comparison(s): {list(comparisons)}")

    all_edge_dfs = []

    for comp in comparisons:
        sub = df_sub[df_sub['Comparison'] == comp]

        R_node    = dict(zip(sub['node'], sub['fc']))
        sig_nodes = set(sub.loc[sub['sig'], 'node'])

        print(f"\nComparison: {comp}")
        for _, row in sub.iterrows():
            sign = '+' if row[LOG2FC_COL] >= 0 else ''
            sig  = '  *' if row['sig'] else ''
            print(
                f"  {row['Plot_Group']:20s}  node={row['node']:4s}  "
                f"log2FC={sign}{row[LOG2FC_COL]:.3f}  FC={row['fc']:.3f}  "
                f"p={row[PVAL_COL]:.2e}{sig}"
            )

        # --- Plot ---
        safe_name = str(comp).replace(' ', '_').replace('/', '-')
        out_path  = os.path.join(out_dir, f'lipid_network_{safe_name}.svg')
        draw_comparison(comp, R_node, sig_nodes, out_path)

        # --- Edge flux DataFrame ---
        edge_df = build_edge_flux_df(comp, R_node)
        all_edge_dfs.append(edge_df)

    # Save combined edge flux table
    combined  = pd.concat(all_edge_dfs, ignore_index=True)
    edge_csv  = os.path.join(out_dir, 'edge_flux_table.csv')
    combined.to_csv(edge_csv, index=False)
    print(f"\nEdge flux table saved: {edge_csv}")
    print(combined.to_string(index=False))

    print("\nAll done.")


if __name__ == "__main__":
    main()

