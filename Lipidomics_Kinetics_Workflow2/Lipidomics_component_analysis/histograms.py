# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:13:36 2025

@author: Brigham Young Univ
"""
import os
import matplotlib.pyplot as plt
import re

def _sanitize_filename(name):
    """Replace invalid characters in a filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


import os
import matplotlib.pyplot as plt
import numpy as np

def save_component_histograms(component_sims_dict, identifier, out_dir="component_distributions"):
    """
    Save histograms of component simulation distributions with fixed bin width (0.1).

    Args:
        component_sims_dict (dict[str, np.ndarray]):
            Dictionary mapping component → array of simulated values.
        identifier (str): Genotype or condition name for folder naming.
        out_dir (str): Root folder to save histograms.
    """
    # Fixed x-axis range and bin width
    xlim = (-30, 30)
    bin_width = 0.1
    bins = int((xlim[1] - xlim[0]) / bin_width)

    # Create output folder
    folder = os.path.join(os.path.dirname(__file__), out_dir, identifier)
    os.makedirs(folder, exist_ok=True)

    for comp, sims in component_sims_dict.items():
        plt.figure(figsize=(6, 4))
        plt.hist(sims, bins=bins, range=xlim, alpha=0.8,
                 color='skyblue')
        plt.title(f"Distribution of {comp} (n={len(sims)})")
        plt.xlabel("Estimated n-value")
        plt.ylabel("Frequency")
        plt.xlim(*xlim)
        plt.tight_layout()

        # Sanitize filename
        safe_name = _sanitize_filename(comp)
        filepath = os.path.join(folder, f"{safe_name}.svg")
        plt.savefig(filepath)
        plt.close()



import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fatty_acid_histograms_separate(
    comp_sims_dict_by_ident: dict[str, dict[str, np.ndarray]],
    xlim=(-10, 40),
    bins=None
):
    """
    Generate overlaid histogram plots (one per FA) comparing n-value distributions
    across genotypes, with vertical lines for literature and theoretical values.

    Args:
        comp_sims_dict_by_ident (dict[str, dict[str, np.ndarray]]):
            Dictionary: {genotype → {component → array of simulated n-values}}
        xlim (tuple): x-axis limits
        bins (int or None): number of histogram bins. If None, computed so bin width = 0.1

    Returns:
        figs (dict[str, matplotlib.figure.Figure]): {component → figure}
    """
    # Calculate number of bins if not specified (bin width = 0.1)
    if bins is None:
        bins = int((xlim[1] - xlim[0]) / 0.1)

    # Reference values
    ref_vals = {
        '16:0': {'Lee1994': 17, 'Theoretical': 21, 'C-H sites': 31},
        '18:0': {'Lee1994': 20, 'Theoretical': 24, 'C-H sites': 33},
        '20:4': {'Carson2017': 6, 'Theoretical': 7, 'C-H sites': 31},
        'Glycerol': {'C-H sites': 5, 'Glycolysis Theoretical': 2, 'Gluconeogenisis Theoretical': 3},
        'Serine': {'C-H sites': 3, 'Alamillo2025': 2.6},
        '0:0': {'C-H sites': 0},
        'Inositol': {'C-H sites': 6},
        'Choline': {'C-H sites': 13},
        'Ethanolamine': {'C-H sites': 4}
    }

    ref_colors = {
        'Lee1994': 'magenta',
        'Theoretical': 'dimgray',
        'Carson2017':  'magenta',
        'C-H sites': 'black',
        'Glycolysis Theoretical': 'magenta',
        'Gluconeogenisis Theoretical': 'darkorange',
        'Alamillo2025':  'magenta',
    }

    # Only include FAs that exist in the sims
    components = [fa for fa in ref_vals if any(fa in d for d in comp_sims_dict_by_ident.values())]
    figs = {}

    for comp in components:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        # Plot histograms for each genotype
        for ident, comp_sims in comp_sims_dict_by_ident.items():
            if comp in comp_sims:
                sims = comp_sims[comp]
                ax.hist(sims, bins=bins, density=True, alpha=0.5,
                        label=ident,  linewidth=0.3)

        # Add reference lines (literature & theoretical)
        if comp in ref_vals:
            for label, val in ref_vals[comp].items():
                ax.axvline(val, color=ref_colors.get(label, 'black'),
                           linestyle='--', linewidth=2,
                           label=label)

        # Formatting
        ax.set_title(f"{comp} {r'$n_L$'} Distribution")
        ax.set_xlabel(f"Estimated {r'$n_L$'}")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.legend()
        fig.tight_layout()
        figs[comp] = fig

    return figs
