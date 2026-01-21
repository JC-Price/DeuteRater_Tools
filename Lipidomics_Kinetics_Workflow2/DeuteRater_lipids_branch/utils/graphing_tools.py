# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Bradley Naylor, Christian Andersen, Michael Porter, Kyle Cutler, Chad Quilling, Benjamin Driggs,
    Coleman Nielsen, J.C. Price, and Brigham Young University
All rights reserved.
Redistribution and use in source and binary forms,
with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the
      above copyright notice, this list of conditions
      and the following disclaimer.
    * Redistributions in binary form must reproduce
      the above copyright notice, this list of conditions
      and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors
      may be used to endorse or promote products derived
      from this software without specific prior written
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.pyplot as plt
import numpy as np

import deuterater.settings as settings

import os
os.environ["MPLCONFIGDIR"] = "./mpl_config"

main_line_symbol = 'k-'
# error_line_symbol = 'k--'
data_points_symbol = 'ro'
MAXIMUM_GRAPH_RATE_ERROR = 5
MINIMUM_GRAPH_RATE_ERROR = -2

# the colon is not allowed but seems to make an empty file with a partial name.
# either way the check is here to prevent problems if it is necessary
bad_save_file_characters = ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]





def graph_rate(
    name, id_name, x_values, y_values, rate, asymptote, ci, rate_equation,
    save_folder_name, maximum, asymptote_option, errors=None,
    calc_type=None, full_data=None, title=None, molecule_type=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import os

    if errors is None:
        errors = []


    full_data.to_csv("C:\\Users\\Brigham Young Univ\\Downloads\\HPLC6\\HPLC6\\new_nLs\graphing_test.csv")
    # ----------------------------------------------------
    # 0) Resolve ID column and subset full_data by analyte
    # ----------------------------------------------------
    if full_data is None:
        import pandas as _pd
        full_data = _pd.DataFrame()

    if molecule_type == 'Peptide':
        ID_col = 'Protein ID'
    elif molecule_type == 'Lipid':
        ID_col = 'Lipid Unique Identifier'
    else:
        ID_col = None

    if ID_col is not None and ID_col in full_data.columns:
        full_data = full_data[full_data[ID_col] == id_name].copy()

    # ----------------------------------
    # 1) Basic plot scaffolding
    # ----------------------------------
    if title is None:
        title = name
    plt.title(title)
    plt.xlabel('Time (Days)')
    plt.ylabel('Fraction New')

    # Vertical lines at each time point
    for time in x_values:
        plt.axvline(x=time, linewidth=.5, color="k")

    # ----------------------------------
    # 2) Rate fit line + CI band
    # ----------------------------------
    make_error_lines = True
    fit_line_x = np.arange(0, maximum + maximum / 10, .1)

    if str(asymptote_option).lower() == "variable":
        fit_line_y = rate_equation(fit_line_x, k=rate, a=asymptote)
        try:
            fit_line_y_minus_error = rate_equation(fit_line_x, k=rate - ci, a=asymptote)
            fit_line_y_plus_error  = rate_equation(fit_line_x, k=rate + ci, a=asymptote)
        except Exception:
            make_error_lines = False
    else:
        fit_line_y = rate_equation(fit_line_x, rate)
        try:
            fit_line_y_minus_error = rate_equation(fit_line_x, rate - ci)
            fit_line_y_plus_error  = rate_equation(fit_line_x, rate + ci)
        except Exception:
            make_error_lines = False

    plt.plot(fit_line_x, fit_line_y, 'k-')

    MAXIMUM_GRAPH_RATE_ERROR = 5
    MINIMUM_GRAPH_RATE_ERROR = -2

    if make_error_lines:
        fit_line_y_plus_error[fit_line_y_plus_error > MAXIMUM_GRAPH_RATE_ERROR] = MAXIMUM_GRAPH_RATE_ERROR
        fit_line_y_minus_error[fit_line_y_minus_error < MINIMUM_GRAPH_RATE_ERROR] = MINIMUM_GRAPH_RATE_ERROR
        plt.fill_between(
            fit_line_x, fit_line_y_minus_error, fit_line_y_plus_error,
            color='black', alpha=.15
        )

    # ----------------------------------
    # 3) Plot original datapoints (color by Adduct only)
    # ----------------------------------
    plt.ylim(-0.5, 1.5)

    can_use_full = (
        full_data is not None
        and not full_data.empty
        and "Adduct" in full_data.columns
        and "time" in full_data.columns
        and calc_type is not None
        and calc_type in full_data.columns
    )

    if can_use_full:
        # Automatically assign a color to each unique adduct
        adducts = full_data["Adduct"].astype(str).unique()
        cmap = plt.cm.get_cmap("tab20", len(adducts))
        adduct_to_color = {a: cmap(i) for i, a in enumerate(adducts)}

        seen_adducts = set()

        for adduct, adduct_data in full_data.groupby("Adduct"):
            adduct_str = str(adduct)
            color = adduct_to_color.get(adduct_str, "k")

            x = adduct_data["time"].astype(float).to_numpy()
            y = adduct_data[calc_type].astype(float).to_numpy()

            # optional: show one n_value per adduct if it’s constant
            n_val_str = ""
            if "n_value" in adduct_data.columns:
                n_vals = adduct_data["n_value"].dropna().unique()
                if len(n_vals) == 1:
                    n_val_str = f", n={n_vals[0]:.3g}"

            # only label the first occurrence of each adduct
            label = f"{adduct_str}{n_val_str}" if adduct_str not in seen_adducts else None
            seen_adducts.add(adduct_str)

            plt.scatter(
                x,
                y,
                marker="o",          # same marker for all points
                facecolor=color,
                edgecolor="k",
                label=label,
            )
    else:
        # Very minimal fallback if for some reason full_data isn't usable
        plt.scatter(
            x_values,
            y_values,
            marker="o",
            facecolor="r",
            edgecolor="k",
        )


    # ----------------------------------
    # 4) FN overlay based on calc_type
    # ----------------------------------
    if not full_data.empty and "time" in full_data.columns:

        # Map calc_type → correct FN overlay
        overlay_map = {
            "abund_fn": "BA_FN",
            "nsfn":     "BS_FN",
            "cfn":      "Combined_FN",
        }

        desired_fn = overlay_map.get(calc_type, None)

        fn_col = None
        main_prefix = None

        # Try exact expected column
        if desired_fn and desired_fn in full_data.columns:
            fn_col = desired_fn
            main_prefix = desired_fn.split("_")[0]

        # Fallback if needed
        if fn_col is None:
            for cand in ["Combined_FN", "BA_FN", "BS_FN"]:
                if cand in full_data.columns:
                    fn_col = cand
                    main_prefix = cand.split("_")[0]
                    break

        if fn_col is not None:

            # Decide grouping species
            if molecule_type == "Peptide":
                color_col = 'Adducted_identifier'
            elif molecule_type == "Lipid":
                color_col = 'Adducted_identifier'

            else:
                color_col = None

            if color_col is None:
                full_data["_species_single"] = "default"
                color_col = "_species_single"

            species_vals = full_data[color_col].astype(str).unique()
            cmap = plt.cm.get_cmap("tab20", len(species_vals))
            species_to_color = {spec: cmap(i) for i, spec in enumerate(species_vals)}

            # parameter columns
            nL_col   = next((c for c in [f"{main_prefix}_nL","Combined_nL","BA_nL","BS_nL"] if c in full_data.columns), None)
            rate_col = next((c for c in [f"{main_prefix}_rate","Combined_rate","BA_rate","BS_rate"] if c in full_data.columns), None)
            Asyn_col = next((c for c in [f"{main_prefix}_Asyn","Combined_Asyn","BA_Asyn","BS_Asyn"] if c in full_data.columns), None)

            def _fmt(v):
                try:
                    v = float(v)
                    return format(v, ".3g") if np.isfinite(v) else "nan"
                except:
                    return "nan"

            t_vals = full_data["time"].astype(float).to_numpy()
            y_fn   = full_data[fn_col].astype(float).to_numpy()

            seen = set()
            for spec in species_vals:
                mask = (full_data[color_col].astype(str).values == spec)
                mask &= np.isfinite(t_vals) & np.isfinite(y_fn)

                if not mask.any():
                    continue

                row = full_data[full_data[color_col] == spec].iloc[0]

                if spec not in seen:
                    nL_val = row.get(nL_col, np.nan)
                    rate_val = row.get(rate_col, np.nan)
                    Asyn_val = row.get(Asyn_col, np.nan)

                    label = f"{spec}\n nL={_fmt(nL_val)}, Asyn={_fmt(Asyn_val)}, k={_fmt(rate_val)}"
                    seen.add(spec)
                else:
                    label = None

                # --- NEW MODEL-BASED FN(t) = Asyn - Asyn * exp(-k t) LINE ---
                t_line = np.linspace(0, maximum, 300)
                FN_model_line = Asyn_val - Asyn_val * np.exp(-rate_val * t_line)

                plt.plot(
                    t_line,
                    FN_model_line,
                    linewidth=1.2,
                    color=species_to_color[spec],
                    label=label,
                    alpha=0.9
                )

    # ----------------------------------
    # 5) Error bars
    # ----------------------------------
    if errors:
        plt.errorbar(
            x_values, y_values, yerr=errors,
            elinewidth=1, ecolor='red', linewidth=0
        )

    # ----------------------------------
    # 5.5) Draw legend (ONLY if there are labeled artists)
    # ----------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(prop={"size": 9}, framealpha=0.85)

    # ----------------------------------
    # 6) Save figure
    # ----------------------------------
    font = FontProperties()
    font.set_size("small")

    try:
        filename = os.path.join(save_folder_name, name)
        plt.tight_layout()
        fmt = settings.graph_output_format
        if fmt == "png":
            plt.savefig(filename[:-4] + ".png", format="png")
        elif fmt == "svg":
            plt.savefig(filename[:-4] + ".svg", format="svg")

    except OSError:
        for bad_char in ["/","\\",":","*","?","\"","<",">","|"]:
            name = name.replace(bad_char, "_")

        filename = os.path.join(save_folder_name, name)
        fmt = settings.graph_output_format
        if fmt == "png":
            plt.savefig(filename + ".png", format="png")
        elif fmt == "svg":
            plt.savefig(filename + ".svg", format="svg")

    plt.clf()
