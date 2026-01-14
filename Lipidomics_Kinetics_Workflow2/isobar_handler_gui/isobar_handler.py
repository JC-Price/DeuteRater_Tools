# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Bradley Naylor, Christian Andersen, Michael Porter, Kyle Cutler, Chad Quilling, Benjamin Driggs,
    Coleman Nielsen, Martin Sorensen, J.C. Price, and Brigham Young University
Credit: ChatGPT - helped with organization, debugging, and general flow
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

Created on Mon Jul  7 11:33:34 2025

Patched version:
- Fixed undefined `cf` NameError in protein defaults.
- Added robust fallbacks for missing protein / lipid columns (Identification Charge, Adduct_cf, etc.).
- Guarded metadata attribute pulls in isobarFrame to avoid KeyErrors.
- Corrected np.trapz argument order in sum_gaussian().
- Added noise_std safety guard in process_and_plot_datasets().
- Replaced brittle y_sum lookup with np.interp in S/N filtering.
- Sanitized filenames when saving plots.
- Fixed indentation around Data‑type selector UI block.
- Minor path joins normalized.

Original author: Brigham Young Univ
"""


from __future__ import annotations
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from io import StringIO
import re
import numpy as np
import pickle
from tkinter import ttk
import os
from tkinter import filedialog, messagebox
from pyopenms import MSExperiment, MzMLFile
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import yaml
from collections import defaultdict
import sys
import traceback
import warnings
from numbers import Integral, Real
from pathlib import Path
import shutil
import statistics
import gc; gc.collect()
# --- BEGIN PATCH: plotting helpers & user-selectable output dir ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
# (FigureCanvasAgg used for background saving to avoid GUI backends)
# Helper: directory where saved plots go (user chooses once)
PLOT_DIR: str | None = None

# --- BEGIN change: choose_plot_directory creates temp subfolder ---
# Place this in the same location where your current choose_plot_directory() is defined,
# replacing the old function body entirely.

TEMP_PLOTS_SUBDIR_NAME = "temp plots"   # subfolder name inside the user-chosen folder
TEMP_PLOTS_DIR: str | None = None       # module-level cache


def choose_plot_directory(force: bool = False) -> str:
    """
    Prompt user for a directory to save exported plots. If user cancels,
    fall back to a 'plots' folder next to the script/cwd. Also create and return
    a 'temp plots' subfolder inside that folder. Stores both on module and settings.
    """
    global PLOT_DIR, TEMP_PLOTS_DIR
    if PLOT_DIR is not None and not force:
        # ensure temp dir exists too
        if TEMP_PLOTS_DIR is None:
            temp_dir = os.path.join(PLOT_DIR, TEMP_PLOTS_SUBDIR_NAME)
            os.makedirs(temp_dir, exist_ok=True)
            TEMP_PLOTS_DIR = temp_dir
            try:
                setattr(settings, "_temp_plots_dir", TEMP_PLOTS_DIR)
                setattr(settings, "_plot_dir", PLOT_DIR)
            except Exception:
                pass
        return PLOT_DIR

    # ask user for a folder (may raise if Tk not initialised)
    try:
        d = filedialog.askdirectory(title="Choose folder to save exported plots")
    except Exception:
        d = None

    if not d:
        base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        d = os.path.join(base, "plots")

    os.makedirs(d, exist_ok=True)
    PLOT_DIR = d

    # ensure temp subfolder
    temp_dir = os.path.join(PLOT_DIR, TEMP_PLOTS_SUBDIR_NAME)
    os.makedirs(temp_dir, exist_ok=True)
    TEMP_PLOTS_DIR = temp_dir

    # store on settings so other code can use settings._temp_plots_dir
    try:
        setattr(settings, "_plot_dir", PLOT_DIR)
        setattr(settings, "_temp_plots_dir", TEMP_PLOTS_DIR)
    except Exception:
        pass

    return PLOT_DIR
# --- END change ---


def _save_figure_svg(fig: Figure, filename: str) -> str:
    """
    Save *fig* as an SVG into the chosen plot directory (creates dir).
    Returns the full path saved. This uses Figure.savefig or the Agg canvas
    fallback so it is safe to call from worker threads.
    """
    out_dir = choose_plot_directory()
    name = sanitize_filename(filename)
    if not name.lower().endswith(".svg"):
        name = name + ".svg"
    out_path = os.path.join(out_dir, name)
    try:
        # Prefer fig.savefig (works for Figure objects)
        fig.savefig(out_path, format="svg", bbox_inches="tight")
    except Exception:
        # Fallback via Agg canvas (also safe for background threads)
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(out_path, format="svg", bbox_inches="tight")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
    return out_path
# --- END PATCH ---

def _save_temp_figure_svg(fig: Figure, filename: str) -> str:
    """
    Save *fig* into the 'temp plots' subfolder of the chosen plot directory.

    In frozen/packaged builds, matplotlib's SVG backend may be missing
    (e.g., ModuleNotFoundError: matplotlib.backends.backend_svg). In that case,
    we fall back to saving a PNG in the same temp folder so alignment can continue.

    Returns the output path (SVG when available, otherwise PNG).
    """
    # Ensure the main plot dir & temp subdir exist (do not force re-prompt)
    plot_dir = getattr(settings, "_plot_dir", None) or PLOT_DIR
    if not plot_dir:
        # last resort: create default and its temp subdir
        plot_dir = choose_plot_directory()  # choose_plot_directory will create temp subdir
    temp_dir = getattr(settings, "_temp_plots_dir", None) or os.path.join(plot_dir, TEMP_PLOTS_SUBDIR_NAME)
    os.makedirs(temp_dir, exist_ok=True)

    base = sanitize_filename(filename)

    # Prefer SVG, but be prepared to fall back to PNG if backend_svg is missing
    svg_name = base if base.lower().endswith(".svg") else (base + ".svg")
    svg_path = os.path.join(temp_dir, svg_name)

    # Precompute PNG fallback path
    if base.lower().endswith(".svg"):
        png_name = base[:-4] + ".png"
    else:
        png_name = base + ".png"
    png_path = os.path.join(temp_dir, png_name)

    def _save_png_fallback() -> str:
        """Always-available fallback using Agg (works well in executables)."""
        try:
            fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
        except Exception:
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(png_path, format="png", dpi=200, bbox_inches="tight")
        return png_path

    out_path = svg_path
    try:
        # This path triggers matplotlib to import backend_svg; may be missing in frozen builds.
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    except ModuleNotFoundError as e:
        # Common in packaged apps if backend_svg wasn't bundled
        if "matplotlib.backends.backend_svg" in str(e):
            out_path = _save_png_fallback()
        else:
            raise
    except Exception as e:
        # If the error is still caused by missing backend_svg, fall back
        if "matplotlib.backends.backend_svg" in str(e):
            out_path = _save_png_fallback()
        else:
            # Try Agg SVG once; if that still requires backend_svg, fall back to PNG
            try:
                canvas = FigureCanvasAgg(fig)
                canvas.print_figure(svg_path, format="svg", bbox_inches="tight")
                out_path = svg_path
            except Exception as e2:
                if "matplotlib.backends.backend_svg" in str(e2):
                    out_path = _save_png_fallback()
                else:
                    raise
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass

    return out_path




"""Required and Optional Columns for Data Import
For the program to function correctly, the imported dataframe must contain the following columns:

Alignment ID – Unique identifier for each metabolite entry.
Precursor m/z – Mass-to-charge ratio of the precursor ion.
Metabolite name – Name of the detected metabolite.
Adduct – Type of adduct associated with the metabolite.
Formula – Molecular formula of the metabolite.
Ontology – Classification or group the metabolite belongs to.
Adduct_cf – Charge format of the adduct.
Average Rt(min) – Retention time in minutes, used for peak alignment.
Optional Columns:
   
These columns are not required but allow you to view the original retention times and intensities as described in
the imported .csv (assuming a retention time FWHM of 17 seconds).
Columns ending with "_RT_sec" – Retention time (in seconds) for each file.
Columns ending with "_Abn" – Abundance values for each file."""


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with duplicate headers removed (keeps first copy)."""
    return df.loc[:, ~df.columns.duplicated()]


class SkipEIC(Exception):
    """Raised when an EIC has no usable data after smoothing / masking."""
    pass


# Global dictionary to hold isobar frames
isobar_frames = {}


class Settings:
    """
    ────────────────────────────────────────────────────────────────────────
      Checkpoint-aware parameter organisation
        • extraction   –> everything needed up to “EIC extraction” cache
        • polynomial    –> everything needed up to “polynomial” cache
        • other         –> downstream tweaks (peak picking, plots, etc.)
        • global_flags  –> run-time switches (redo, verbose …)
    ────────────────────────────────────────────────────────────────────────
    """

    # ── constructor ──────────────────────────────────────────────────────
    def __init__(self):
        # ❶  Extraction-time parameters
        self.verbose_exceptions = False
        self.save_intermediates = True
        self.ID_file = {'max_cores': 30}
        self.extraction = {
            # mz / isotope handling
            "ppm_value"             : 20,
            "neutromer_avg_mass"    : 1.0048235,

            # Gaussian synthesis / smoothing (happens before polynomial step)
            "gaussian_step_size"    : 0.01,
            "gaussian_fwhm_fraction": 0.50,

            # retention-time pre-processing
            "smoothing_window_len"  : 401,
            "smoothing_poly_order"  : 5,
            "smoothing_percent_cut" : 10,
            "axis_padding_sec"      : 300,
            "neg_rt_cutoff_sec"     : 240,
        }

        # ❷  Polynomial-alignment parameters
        self.polynomial = {
            "downsampling_factor"   : 25,
            "iterations"            : 400,
            "poly_degree"           : 5,
            "iterations_before_min" : 20,
            "coefficient_range"     : (-50, 50),     # tuple for clarity
            "rmsd_target"           : 8.0,
            "leave_cores_open": 10# early-stop threshold
        }

        # ❸  Downstream (does NOT invalidate caches)
        self.other = {
            # peak picking / QC
            "peak_percent_cutoff"   : 5,
            "peak_absolute_cutoff"  : 0,
            "signal_to_noise_cutoff": 1.5,

            # plot & retention windows
            "rt_peak_filter_sec"    : 30
        }

        # ❹  High-level run-time switches
        self.global_flags = {
            "use_full_df"   : True,
            "force_redo_all": False,
            "standards_only": False,
            "verbose_output": True,
        }

        self.gaussian = {
            "step_size"     : self.extraction["gaussian_step_size"],
            "fwhm_fraction" : self.extraction["gaussian_fwhm_fraction"],
        }
       
        self.retention_time = {
            "peak_filter_seconds": self.other["rt_peak_filter_sec"],
            "axis_padding"       : self.extraction["axis_padding_sec"],
            "negative_rt_cutoff" : self.extraction["neg_rt_cutoff_sec"],
            "smoothing": {
                "window_length"   : self.extraction["smoothing_window_len"],
                "polynomial_order": self.extraction["smoothing_poly_order"],
                "percent_cutoff"  : self.extraction["smoothing_percent_cut"],
            },
        }
       
        self.polynomial_correction = {
            "downsampling_factor"      : self.polynomial["downsampling_factor"],
            "iterations"               : self.polynomial["iterations"],
            "iterations_before_minima" : self.polynomial["iterations_before_min"],
            "coefficient_range"        : self.polynomial["coefficient_range"],
            "testable_values"          : 100,       # default kept
        }

        self.metabolyte_type = 'lipid'
        self.export_method = 'save'
       
    # ── backward-compatibility shim ──────────────────────────────────────
    def __getattr__(self, name: str):
        """
        Allow legacy code (settings.extraction["ppm_value"], settings.gaussian, …)
        to keep working.  Emits a one-time deprecation warning.
        """
        alias_map = {
            # extraction
            "ppm_value"               : ("extraction", "ppm_value"),
            "neutromer_average_mass"  : ("extraction", "neutromer_avg_mass"),
            "gaussian"      : ("gaussian",),
            "retention_time": ("retention_time",),

            # polynomial
            "polynomial_correction"   : ("polynomial",),   # whole block
            "rmsd_cutoff"             : ("polynomial", "rmsd_target"),

            # other
            "peak_picking_percent_cutoff" : ("other", "peak_percent_cutoff"),
            "peak_picking_absolute_cutoff": ("other", "peak_absolute_cutoff"),
            "signal_to_noise_cutoff"      : ("other", "signal_to_noise_cutoff"),

            # flags
            "use_full_df"   : ("global_flags", "use_full_df"),
            "force_redo_all": ("global_flags", "force_redo_all"),
            "standards_only": ("global_flags", "standards_only"),
            "verbose_output": ("global_flags", "verbose_output"),
        }

        if name in alias_map:
            warnings.warn(
                f"[Settings] ‘{name}’ is deprecated; use settings.{'.'.join(alias_map[name])} instead.",
                stacklevel=2,
            )
            # walk the dict once → cache attribute for future hits
            val = self
            for key in alias_map[name]:
                val = val[key] if isinstance(val, dict) else getattr(val, key)
            setattr(self, name, val)
            return val

        raise AttributeError(name)


settings = Settings()  # Global settings object


def set_default_settings():
    global settings
    settings = Settings()
   


def settings_gui():
    global settings                  # the Settings() instance
    win = tk.Toplevel()
    win.title("Settings")

    # ------------------------------------------------------------------
    # 1. build <param-name -> tk.Variable> maps for each block
    # ------------------------------------------------------------------
    widgets = defaultdict(dict)      # {block: {key: tk.Variable}}

    # helper: choose Var class by current value type
    def _var_for(value):
        if isinstance(value, bool):
            return tk.BooleanVar(value=value)
        return tk.StringVar(value=str(value))

    for block_name in ("extraction", "polynomial", "other", "global_flags"):
        blk = getattr(settings, block_name)
        for key, val in (blk.items() if isinstance(blk, dict) else blk.__dict__.items()):
            widgets[block_name][key] = _var_for(val)

    # ------------------------------------------------------------------
    # 2. create Notebook with one tab per block
    # ------------------------------------------------------------------
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    for block_name, var_map in widgets.items():
        frame = ttk.Frame(nb)
        nb.add(frame, text=block_name.replace('_', ' ').title())

        for row, (key, var) in enumerate(var_map.items()):
            ttk.Label(frame, text=key).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, padx=4, pady=2)
            else:
                ent = ttk.Entry(frame, textvariable=var, width=15)
                ent.grid(row=row, column=1, padx=4, pady=2)

    # ------------------------------------------------------------------
    # 3. helpers
    # ------------------------------------------------------------------
    def _convert(value, original):   # str -> original’s type
        if isinstance(original, bool):
            return value.lower() in ("1", "true", "yes")
        if isinstance(original, Integral):
            return int(value)
        if isinstance(original, Real):
            return float(value)
        # accept comma-separated list of numbers
        if isinstance(original, (list, tuple)):
            items = [float(x) if '.' in x else int(x) for x in value.split(',')]
            return type(original)(items)
        return value

    def _apply_changes():
        # Track whether extraction / polynomial knobs changed
        extraction_changed = False
        polynomial_changed = False

        try:
            # update each block in place
            for block_name, var_map in widgets.items():
                blk = getattr(settings, block_name)
                for key, tk_var in var_map.items():
                    new_val = _convert(tk_var.get(), blk[key] if isinstance(blk, dict) else getattr(blk, key))
                    if isinstance(blk, dict):
                        changed = (blk[key] != new_val)
                        blk[key] = new_val
                    else:
                        changed = (getattr(blk, key) != new_val)
                        setattr(blk, key, new_val)

                    # checkpoint invalidation
                    if changed and block_name == "extraction":
                        extraction_changed = True
                    elif changed and block_name == "polynomial":
                        polynomial_changed = True

            # cascade invalidations
            if extraction_changed:
                settings.global_flags["force_redo_all"] = True      # redo both checkpoints
            elif polynomial_changed:
                pass  # polynomial rebuild will be triggered downstream

            messagebox.showinfo("Settings", "Updated successfully.")
            win.destroy()

        except ValueError as e:
            messagebox.showerror("Settings", f"Invalid value: {e}")

    def _import_yaml():
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yml *.yaml")])
        if not path:
            return
        with open(path, 'r') as fh:
            data = yaml.safe_load(fh)
        settings.__dict__.update(data)            # trust the file
        win.destroy()
        settings_gui()                            # reopen with new values

    # ------------------------------------------------------------------
    # 4. buttons
    # ------------------------------------------------------------------
    btn_frame = ttk.Frame(win)
    btn_frame.pack(pady=6)

    ttk.Button(btn_frame, text="Save / Update", command=_apply_changes).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Import YAML",   command=_import_yaml).grid(row=0, column=1, padx=5)

def _load_trace(fr, file_name: str, raw: bool = True):
    """
    Load exactly one (x,y) pair from the cached npz without materializing the whole archive.
    Returns (x, y) or (None, None) if missing.
    """
    path = fr.smooth_raw_path if raw else fr.smooth_path
    key  = _encode_key(file_name)
    with np.load(path, allow_pickle=False) as npz:
        if key not in npz.files:
            return None, None
        arr = npz[key]            # shape (2, N), dtype float32
        return arr[0], arr[1]



def build_global_mtic(num_points=4000):
    """
    Build a global mTIC by summing Pol1-transformed *raw* smoothed EICs
    across all IDs and files, then (later) normalizing for plotting.
    Falls back to pre-polynomial summed_eics if Pol1dict is missing.

    Returns:
      (x_seconds, y_global_sum, avg_df)

    where avg_df has columns:
      File, average_mTIC

    If unavailable:
      (None, None, empty_df)
    """
    global file_specific_info

    empty_df = pd.DataFrame(columns=["File", "average_mTIC"])

    have_polys = any(getattr(fr, "Pol1dict", None) for fr in isobar_frames.values())
    if not have_polys:
        sdict = file_specific_info.get("summed_eics", {})
        data = []
        per_file_avg = {}

        for fname, arr in sdict.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2 and arr.size:
                x = arr[:, 0].astype(float, copy=False)
                y = arr[:, 1].astype(float, copy=False)
                if x.size and np.any(y > 0):
                    data.append((x, y))
                    per_file_avg[fname] = float(np.nanmean(y))

        if not data:
            return None, None, empty_df

        common_x, y_sum, _ = sum_datasets(data, num_points=num_points)

        avg_df = (
            pd.DataFrame([{"File": str(k), "average_mTIC": float(v)} for k, v in per_file_avg.items()])
              .sort_values("File")
              .reset_index(drop=True)
        )
        return (np.asarray(common_x), np.asarray(y_sum), avg_df) if common_x is not None else (None, None, avg_df)

    # Pol1 path: accumulate per-file, then sum across files
    per_file_traces = defaultdict(list)

    for fr in isobar_frames.values():
        if getattr(fr, "hide", False):
            continue
        if not getattr(fr, "Pol1dict", None):
            continue

        raw_dict = _get_smooth_raw(fr)  # {file.mzML: (x_raw, y_raw)}

        for fname, (x_nat, y_nat) in raw_dict.items():
            if fname not in fr.Pol1dict:
                continue

            coeffs, rmsd, lo, hi = fr.Pol1dict[fname]

            rt_corr = apply_legendre_polynomial(
                rescale_to_legendre_space(x_nat, lo, hi), coeffs
            )
            x_corr = correct_retention_time(np.clip(x_nat, lo, hi), rt_corr, lo, hi)

            # raw intensity; sum happens before normalization
            per_file_traces[fname].append((x_corr, y_nat))

        fr.smooth_unnormalized_dict = None

    # Sum within each file on a shared grid + compute per-file average from summed curve
    per_file_sums = []
    per_file_avg = {}

    for fname, parts in per_file_traces.items():
        cx, ysum, _ = sum_datasets(parts, num_points=num_points)
        if cx is None or ysum is None:
            continue

        per_file_sums.append((cx, ysum))
        per_file_avg[fname] = float(np.nanmean(ysum))

    avg_df = (
        pd.DataFrame([{"File": str(k), "average_mTIC": float(v)} for k, v in per_file_avg.items()])
          .sort_values("File")
          .reset_index(drop=True)
    )

    if not per_file_sums:
        return None, None, avg_df

    common_x, global_sum, _ = sum_datasets(per_file_sums, num_points=num_points)
    if common_x is None or global_sum is None:
        return None, None, avg_df

    return np.asarray(common_x), np.asarray(global_sum), avg_df


def export_global_mtic_overlay(folder_path, filename="global_mTIC_overlay.svg"):
    """
    Saves an SVG into folder_path overlaying:
      • Global mTIC (sum of all files' summed_eics)
      • Reference file's summed_eic
    Both normalized to their own maxima (→ %).
    """
    global file_specific_info, settings
    import numpy as np

    x_mtic, y_mtic, avg_df = build_global_mtic()
    if x_mtic is None:
        # Nothing to plot
        return

    # Reference summed EIC
    ref_key = f"{settings.ref_file}.mzML"
    ref_arr = file_specific_info.get('summed_eics', {}).get(ref_key, None)

    # Prepare plotting grid
    if ref_arr is not None and isinstance(ref_arr, np.ndarray) and ref_arr.size and ref_arr.ndim == 2 and ref_arr.shape[1] >= 2:
        ref_x = ref_arr[:, 0].astype(float, copy=False)
        ref_y = ref_arr[:, 1].astype(float, copy=False)

        lo = max(x_mtic.min(), ref_x.min())
        hi = min(x_mtic.max(), ref_x.max())
        if hi <= lo:
            # No overlap; just plot the mTIC alone
            grid = x_mtic.copy()
            mtic_interp = y_mtic.copy()
            ref_interp = None
        else:
            grid_len = max(len(x_mtic), len(ref_x))
            grid = np.linspace(lo, hi, grid_len)
            mtic_interp = np.interp(grid, x_mtic, y_mtic, left=0.0, right=0.0)
            ref_interp  = np.interp(grid, ref_x,  ref_y,  left=0.0, right=0.0)
    else:
        grid = x_mtic.copy()
        mtic_interp = y_mtic.copy()
        ref_interp = None

    # Normalize → %
    mtic_norm = (mtic_interp / (mtic_interp.max() if mtic_interp.max() > 0 else 1.0)) * 100.0
    if ref_interp is not None:
        ref_norm = (ref_interp / (ref_interp.max() if ref_interp.max() > 0 else 1.0)) * 100.0

    # Use a non-interactive Figure so this can run off-main-thread safely
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(grid / 60.0, mtic_norm, linewidth=2, label="Global mTIC (all files)")
    if ref_interp is not None:
        ax.plot(grid / 60.0, ref_norm, linewidth=1.5, linestyle='--',
                label=f"Reference mTIC ({settings.ref_file})")

    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Intensity (% of curve max)")
    ax.set_title("Global mTIC vs. Reference")
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.legend(loc="best", fontsize="small")

    out_name = filename if filename.lower().endswith(".svg") else f"{os.path.splitext(filename)[0]}.svg"
    out_path = os.path.join(folder_path, out_name)
    
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
        
    avg_df.to_csv(os.path.join(folder_path, "Average_mTICs_per_mzML.csv"), index=False)


def list_chemical_formula(formula):
    # Regular expression pattern to match elements and their counts
    formula = formula.replace('[2H]', 'D')
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    if "D" in formula:
        print(formula)
   
    elements_dict = defaultdict(int)
    for (element, count) in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        elements_dict[element] += count

    return dict(elements_dict)



def format_chemical_formula(elements):
    # Convert elements dictionary back to string format
    formula = ""
    for element, count in elements.items():
        if count == 1:
            formula += element
        elif count == 0:
            pass
        else:
            formula += f"{element}{count}"

    return formula



def adduct_cf(adduct, cf):
    try:
        elements = list_chemical_formula(str(cf))
        #positive adducts
        if adduct == 'M+H':
            elements['H'] = elements.get('H', 0) + 1
        elif adduct == 'M+H-[H2O]':
            elements['H'] = elements.get('H', 0) - 1
            elements['O'] = elements.get('O', 0) - 1  
        elif adduct == 'M+NH4':
            elements['N'] = elements.get('N', 0) + 1
            elements['H'] = elements.get('H', 0) + 4
        elif adduct == 'M+Na':
            elements['Na'] = elements.get('Na', 0) + 1
        elif adduct == 'M+Na-[H2O]':
            elements['Na'] = elements.get('Na', 0) + 1
            elements['H']  = elements.get('H', 0) - 1
            elements['O']  = elements.get('O', 0) - 1  
        elif adduct == 'M+NH4-[H2O]':
            elements['N'] = elements.get('N', 0) + 1
            elements['H'] = elements.get('H', 0) + 2
            elements['O'] = elements.get('O', 0) - 1  
        #negative adducts        
        elif adduct == 'M-H':
            elements['H'] = elements.get('H', 0) - 1
        elif adduct == 'M-H-[H2O]':
            elements['H'] = elements.get('H', 0) - 3
            elements['O'] = elements.get('O', 0) - 1  
        elif adduct == '+C2H3O2-':
            elements['H'] = elements.get('H', 0) + 3
            elements['C'] = elements.get('C', 0) + 2
            elements['O'] = elements.get('O', 0) + 2
        elif adduct == '+COOH':
            elements['H'] = elements.get('H', 0) + 1
            elements['C'] = elements.get('C', 0) + 1
            elements['O'] = elements.get('O', 0) + 2
        return format_chemical_formula(elements)
    except Exception:
        return float('NaN')        
           

class isobarFrame:
    def __init__(self, sub_df: pd.DataFrame):
        """
        Build one isobar frame.  If the user said the file is a protein export
        (`settings.metabolyte_type == "protein"`), harmonise Skyline / DIA-NN
        column names to the canonical names the rest of the pipeline expects.
        """

        # ───────────────────────────a─────
        # 1. Harmonise protein ID columns
        # ────────────────────────────────
        if settings.metabolyte_type.lower() == 'lipid':
            sub_df['Alignment ID'] = (
                sub_df['Adducted_Name']
                + "_"
                + sub_df['Average Rt(min)'].round(2).astype(str)
            )
        if settings.metabolyte_type.lower() == "protein":
            # 1) Convert retention‑time seconds → minutes **before** we lose the column
            if "Precursor Retention Time (sec)" in sub_df.columns:
                sub_df["Precursor Retention Time (sec)"] = (
                    pd.to_numeric(sub_df["Precursor Retention Time (sec)"], errors="coerce") / 60.0
                )
            # 2) Rename to canonical headings
            rename_map = {
                "Protein ID": "Metabolite name",          # becomes display label
                "cf"       : "Formula",                   # molecular formula
                "Precursor Retention Time (sec)": "Average Rt(min)",
            }
            sub_df = sub_df.rename(columns={k: v for k, v in rename_map.items() if k in sub_df.columns})
            # 3) Guarantee mandatory columns exist (create sensible defaults)
            if "Alignment ID" not in sub_df.columns:
                # build from fields if we have them; fall back to row index
                if {"Metabolite name", "Precursor m/z"}.issubset(sub_df.columns):
                    sub_df["Alignment ID"] = (
                            sub_df["Metabolite name"].fillna("").astype(str) + "_" +
                            sub_df["Average Rt(min)"].round(4).astype(str) + "_+" +
                            sub_df["Identification Charge"].astype(str)
                        )

                else:
                    sub_df["Alignment ID"] = [f"Prot_{i}" for i in range(len(sub_df))]

            # --- PATCH: guarantee mandatory downstream columns for protein mode ---
            # Skyline/DIA-NN exports vary a lot. We normalise here so downstream lipid-style code works.

            # 1) Identification charge (try common alternates; fall back to +1)
            if "Identification Charge" not in sub_df.columns:
                charge_col = None
                for alt in ("Charge", "Precursor Charge", "z"):
                    if alt in sub_df.columns:
                        charge_col = alt
                        break
                if charge_col is not None:
                    sub_df["Identification Charge"] = (
                        pd.to_numeric(sub_df[charge_col], errors="coerce").fillna(1).astype(int)
                    )
                else:
                    sub_df["Identification Charge"] = 1

            # 2) Formula (proteins rarely have a small-molecule MF; keep if present)
            if "Formula" not in sub_df.columns:
                sub_df["Formula"] = np.nan

            # 3) Adduct (default protonated)
            if "Adduct" not in sub_df.columns:
                sub_df["Adduct"] = "M+H"

            # 4) Adduct_cf: computed row-wise from Formula + Adduct when possible
            if "Adduct_cf" not in sub_df.columns:
                def _mk_acf(row):
                    f = row.get("Formula", np.nan)
                    a = row.get("Adduct", "M+H")
                    return adduct_cf(a, f) if pd.notna(f) else np.nan
                sub_df["Adduct_cf"] = sub_df.apply(_mk_acf, axis=1)

            # 5) Ontology tag so downstream UI shows something sensible
            if "Ontology" not in sub_df.columns:
                sub_df["Ontology"] = "Protein"
            # --- END PATCH --------------------------------------------------------

        # ────────────────────────────────
        # 2. Store dataframe & build helpers
        # ────────────────────────────────
        self.sub_df = sub_df.reset_index(drop=True)
        self.file_dict, self.original_file_dict = self._create_file_dict()

        # ────────────────────────────────
        # 3. Metadata used elsewhere (safe fallbacks)
        # ────────────────────────────────
        sd = self.sub_df  # shorthand
        self.ID       = sd["Alignment ID"].iloc[0] if "Alignment ID" in sd.columns else f"UNK_{id(self)}"
        self.mz       = sd["Precursor m/z"].iloc[0] if "Precursor m/z" in sd.columns else np.nan
        self.charge   = (sd["Identification Charge"].iloc[0] if "Identification Charge" in sd.columns else 1)
        self.name     = sd["Metabolite name"].iloc[0] if "Metabolite name" in sd.columns else self.ID
        self.adduct   = sd["Adduct"].iloc[0] if "Adduct" in sd.columns else "M+H"
        self.formula  = sd["Formula"].iloc[0] if "Formula" in sd.columns else np.nan
        self.ontology = sd["Ontology"].iloc[0] if "Ontology" in sd.columns else ""
        if "Adduct_cf" in sd.columns:
            self.adduct_cf = sd["Adduct_cf"].iloc[0]
        else:
            self.adduct_cf = adduct_cf(self.adduct, self.formula) if pd.notna(self.formula) else np.nan
        self.id_file_av_rt = sd["Average Rt(min)"].iloc[0] if "Average Rt(min)" in sd.columns else np.nan

        # average intensity across any *_Abn columns (safe if none exist)
        abn_cols = sd.filter(like="_Abn")
        self.average_intensity = (
            abn_cols.mean(axis=1, skipna=True).max() if not abn_cols.empty else 0
        )

        # runtime flags
        self.edited = False
        self.hide   = False

    # ---------------------------------------------------------------------- #
    # helper: build {sample: mini-df} for RT / Abn pairs
    # ---------------------------------------------------------------------- #
    def _create_file_dict(self):
        file_dict = {}
        rt_cols = [c for c in self.sub_df.columns if c.endswith("_RT_sec")]
        for rt_col in rt_cols:
            root = rt_col[:-7]                 # strip "_RT_sec"
            abn_col =  f"{root}_Abn".replace('.mzML', "")
            if abn_col in self.sub_df.columns:
                tmp = self.sub_df[[rt_col, abn_col]].copy()
                tmp.columns = ["RT", "Area"]
                tmp["FWHM"] = 1        # placeholder; filled later
                tmp["NF"]   = 1
                file_dict[root] = tmp
        return file_dict, file_dict.copy()

   

def save_variables(temp=False, file_path=None):
    global isobar_frames, remaining_df, file_specific_info, original_df, settings

    # Check if all required variables exist
    required_vars = ["isobar_frames", "remaining_df", "file_specific_info", "original_df", "settings"]
    missing_vars = [var for var in required_vars if var not in globals()]

    if missing_vars:
        response = messagebox.askyesno(
            "Missing Variables",
            "Some required variables are missing. Would you like to import a project instead?"
        )
        if response:
            file_path = filedialog.askopenfilename(
                title="Import Project",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            if file_path:
                with open(file_path, 'rb') as file:
                    isobar_frames, remaining_df, file_specific_info, original_df, settings = pickle.load(file)
                return
        else:
            messagebox.showinfo("New Project", "Please start a new project before saving.")
            return

    if temp:
        # Save to a temporary file in the current working directory (relative path)
        temp_file_path = "temp.pkl"  # same idea as smooth_cache: relative -> cwd
        with open(temp_file_path, "wb") as file:
            pickle.dump((isobar_frames, remaining_df, file_specific_info, original_df, settings), file)

    elif file_path:
        # Save directly to the provided file path
        with open(file_path, 'wb') as file:
            pickle.dump((isobar_frames, remaining_df, file_specific_info, original_df, settings), file)
    else:
        # Open a save file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        if file_path:
            with open(file_path, 'wb') as file:
                pickle.dump((isobar_frames, remaining_df, file_specific_info, original_df, settings), file)


       
       



def load_variables():
    global isobar_frames, remaining_df, file_specific_info, original_df, settings

    # Open a load file dialog
    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
    )

    if not file_path:
        return

    # Load the variables from the file
    with open(file_path, 'rb') as file:
        isobar_frames, remaining_df, file_specific_info, original_df, settings = pickle.load(file)
    settings.verbose_output = True

    # Rebase cache paths to the project's local smooth_cache folder if needed
    import os
    proj_dir = os.path.dirname(file_path)
    local_cache = os.path.join(proj_dir, "smooth_cache")

    for fr in isobar_frames.values():
        for attr in ("smooth_path", "smooth_raw_path"):
            p = getattr(fr, attr, None)
            if not p or not isinstance(p, str):
                continue
            if os.path.exists(p):
                continue  # path is valid
            candidate = os.path.join(local_cache, os.path.basename(p))
            if os.path.exists(candidate):
                setattr(fr, attr, candidate)

    # Activate the UI
    display_buttons()


def calculate_ppm_diff(mz1, mz2):
    return abs(mz1 - mz2) / mz1 * 1e6

def _sync_datatype_from_gui() -> str:
    """
    Read the current data‑type selection from the main GUI combobox and
    push it into settings.metabolyte_type.  Returns the normalised value.

    Called at the start of load_csv() so the user's most recent selection
    always wins—even if they clicked 'Load CSV' before the combobox's event
    handler fired.
    """
    # Make sure Tk has processed any pending widget events.
    try:
        root.update_idletasks()
    except Exception:
        pass  # safe if called before root is fully built in some contexts

    val = datatype_var.get().strip().lower() if datatype_var.get() else "lipid"
    if val not in ("protein", "lipid"):
        val = "lipid"  # conservative fallback

    # Detect change to trigger downstream rebuilds.
    prev = getattr(settings, "metabolyte_type", None)
    settings.metabolyte_type = val
    if prev is not None and prev != val:
        # switching modes invalidates everything
        settings.global_flags["force_redo_all"] = True
        if settings.global_flags.get("verbose_output", False):
            print(f"[settings] Data type changed {prev!r} → {val!r}; forcing full rebuild.")

    return val


def load_csv() -> None:
    """
    Import an ID file (lipid or protein), normalise headings, inject an
    Alignment ID, build the isobar‑frames and refresh the GUI.

    A pristine copy is kept in `original_df` for round‑trip export.
    """
    global original_df, settings

    # ------------------------------------------------------------------
    # 0. figure out which mode the user picked in the GUI
    # ------------------------------------------------------------------
    metabolyte = settings.metabolyte_type = (
        "protein"
        if datatype_combobox.get().strip().lower().startswith("prot")
        else "lipid"
    )

    if settings.global_flags.get("verbose_output", True):
        print(f"[load_csv] using metabolyte_type={metabolyte}")

    ppm_value = settings.extraction.get("ppm_value", 5)

    file_path = filedialog.askopenfilename(
        title="Select ID‑file",
        filetypes=[("CSV files", "*.csv")],
    )
    if not file_path:
        return

    # ------------------------------------------------------------------
    # 1. read CSV & purge empties
    # ------------------------------------------------------------------
    raw = pd.read_csv(
        file_path,
        skip_blank_lines=True,
        keep_default_na=True,
        na_values=["", " "],
        low_memory=False,
    )
    raw = (
        raw.replace(r"^\s*$", np.nan, regex=True)   # delimiter‑only → NaN
           .dropna(how="all")                      # zap all‑NaN rows
           .pipe(_dedup_columns)                   # keep first copy of dup headers
           .reset_index(drop=True)
    )

    original_df = raw.copy()
    df = original_df.copy()

    # ensure precursor m/z is numeric – used in ID synthesis later
    if "Precursor m/z" in df.columns:
        df["Precursor m/z"] = pd.to_numeric(df["Precursor m/z"], errors="coerce")

    # ------------------------------------------------------------------
    # 2A. protein‑specific harmonisation
    # ------------------------------------------------------------------
    if settings.metabolyte_type == "protein":
        # optional Skyline / DIA‑NN extras
        if "Adduct" not in df.columns:
            df["Adduct"] = "M+H"

        if "cf" in df.columns:
            df["Formula"] = df["cf"]

        if "Precursor Retention Time (sec)" in df.columns:
            df["Average Rt(min)"] = (
                pd.to_numeric(df["Precursor Retention Time (sec)"], errors="coerce") / 60.0
            )

        if "Average Rt(min)" not in df.columns:        # may be all‑NaN – okay
            df["Average Rt(min)"] = np.nan

        # try to recover charge column before we build the Alignment ID
        if "Identification Charge" not in df.columns:
            ch_col = next((c for c in ("Charge", "Precursor Charge", "z")
                           if c in df.columns), None)
            if ch_col:
                df["Identification Charge"] = (
                    pd.to_numeric(df[ch_col], errors="coerce")
                      .fillna(1)
                      .astype(int)
                )
            else:
                df["Identification Charge"] = 1

        # build unique peptide key
        prot_col = "Protein ID" if "Protein ID" in df.columns else "Protein Name"
        if prot_col not in df.columns or "Sequence" not in df.columns:
            messagebox.showerror(
                "Missing Columns",
                "Protein files need ‘Protein ID’ (or ‘Protein Name’) "
                "and ‘Sequence’ columns.",
            )
            return

        df["Alignment ID"] = (
            df[prot_col].astype(str).str.strip()
            + "|" + df["Sequence"].astype(str).str.strip()
            + "(+" + df["Identification Charge"].astype(str) + ")-"
            + df["Average Rt(min)"].astype(str)
        )

        # downstream code expects this name column
        df["Metabolite name"] = df["Alignment ID"]

    # ------------------------------------------------------------------
    # 2B. lipid mode – supply any missing essentials
    # ------------------------------------------------------------------
    else:
        # make sure we have a Metabolite name column
        if "Metabolite name" not in df.columns:
            messagebox.showerror(
                "Missing Columns",
                "Lipid files need a ‘Metabolite name’ column.",
            )
            return

        # default charge
        if "Identification Charge" not in df.columns:
            df["Identification Charge"] = 1

        # create a robust Alignment ID if one is missing
        if "Alignment ID" not in df.columns:
            df["Alignment ID"] = (
                df["Metabolite name"].astype(str).str.strip()
                + "_" +
                df["Precursor m/z"].round(4).astype(str)
            )

    # ------------------------------------------------------------------
    # 2C. optional: derive Adduct_cf
    # ------------------------------------------------------------------
    if (
        "Adduct_cf" not in df.columns
        and "Adduct" in df.columns
        and "Formula" in df.columns
    ):
        def _mk_acf(row):
            f = row.get("Formula", np.nan)
            a = row.get("Adduct", "M+H")
            return adduct_cf(a, f) if pd.notna(f) else np.nan

        df["Adduct_cf"] = df.apply(_mk_acf, axis=1)

    # ------------------------------------------------------------------
    # 3. sanity‑check key columns
    # ------------------------------------------------------------------
    required_cols = {"Precursor m/z", "Metabolite name"}
    missing = required_cols - set(df.columns)
    if missing:
        messagebox.showerror(
            "Missing Columns",
            f"CSV is missing required column(s): {', '.join(sorted(missing))}",
        )
        return

    # drop any row still lacking mandatory fields
    df = df.dropna(subset=list(required_cols)).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. build isobar‑frames & refresh GUI
    # ------------------------------------------------------------------
    create_isobar_frames(df, ppm_value)
    display_buttons()


def _make_isoframe(sub_df, metabolyte_type):
    """
    Child processes come up with default settings (lipid).  Force them to
    use the real type chosen in the GUI before we touch sub_df.
    """
    settings.metabolyte_type = metabolyte_type      # sync in the child
    fr = isobarFrame(sub_df.reset_index(drop=True))
    return fr.ID, fr, len(sub_df)

 

# ──────────────────────────────────────────────────────────────────────
# helper for multiprocessing – lives at module level
# ──────────────────────────────────────────────────────────────────────
def _worker_build_isoframe(tmp_path: str,
                           start: int,
                           stop: int,
                           metabolyte_type: str):
    """
    Child‑side helper.
    • Re‑uses the same on‑disk pickle of the *entire* DataFrame.
    • Reads only the slice [start:stop) it needs, builds an isobarFrame,
      and returns a lightweight triple.
    """
    # each process must know which mode (protein / lipid) it is in
    settings.metabolyte_type = metabolyte_type

    # lazy‑load the shared DataFrame (fast because it’s local disk, not pickle over pipe)
    df = pd.read_pickle(tmp_path)
    sub_df = df.iloc[start:stop].reset_index(drop=True)

    fr = isobarFrame(sub_df)
    return fr.ID, fr, len(sub_df)


def create_isobar_frames(df: pd.DataFrame, ppm_value: float):
    """
    Fast grouping → IsoFrame construction.

    ► Writes the *whole* sorted DataFrame once to a temporary pickle and
      passes only (start, stop) integer slices to subprocesses – this avoids
      the huge pickle cost at start‑up.

    ► Falls back to single‑core when the file is small.
    """
    import tempfile, os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    global isobar_frames, remaining_df

    # ── 0. sanity & sort ────────────────────────────────────────────
    isobar_frames = {}
    df = df.sort_values("Precursor m/z").reset_index(drop=True)

    # ── 1. group indices that are within ppm tolerance ──────────────
    visited = np.zeros(len(df), dtype=bool)
    groups: list[tuple[int, int]] = []          # list of (start, stop)

    i = 0
    while i < len(df):
        if visited[i]:
            i += 1
            continue
        j = i + 1
        while j < len(df):
            if calculate_ppm_diff(df.at[i, "Precursor m/z"],
                                  df.at[j, "Precursor m/z"]) < ppm_value:
                visited[j] = True
                j += 1
            else:
                break
        groups.append((i, j))                   # slice [i:j)
        i = j

    # ── 2. decide on multi‑process or single‑core ───────────────────
    use_mp      = len(df) > 500                # threshold
    max_workers = settings.ID_file["max_cores"]
    mtype       = settings.metabolyte_type
    excluded_idx = []

    # write the dataframe once for all workers
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fh:
        tmp_path = fh.name
        df.to_pickle(tmp_path)

    try:
        if use_mp:
            fut_to_slice = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for s, e in groups:
                    fut = exe.submit(_worker_build_isoframe,
                                     tmp_path, s, e, mtype)
                    fut_to_slice[fut] = (s, e)

                for fut in as_completed(fut_to_slice):
                    (s, e) = fut_to_slice[fut]
                    key, fr, size = fut.result()
                    if size:
                        isobar_frames[key] = fr
                        excluded_idx.extend(range(s, e))
        else:  # single‑core (small files)
            for s, e in groups:
                key, fr, size = _worker_build_isoframe(tmp_path, s, e, mtype)
                if size:
                    isobar_frames[key] = fr
                    excluded_idx.extend(range(s, e))
    finally:
        # always clean up the temporary file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── 3. build / clear `remaining_df` exactly as before ───────────
    if settings.global_flags["use_full_df"]:
        remaining_df = pd.DataFrame(columns=df.columns)
    else:
        remaining_df = (
            df.drop(index=excluded_idx)
              .reset_index(drop=True)
              .assign(**{
                  "Marked as having potential isobaric issues": False,
                  "Modified by Lipidomics isobar Adjuster"    : False,
                  "av_RMSD"                                   : None,
              })
        )

    # final ordering for deterministic GUI layout
    isobar_frames = dict(sorted(isobar_frames.items()))



def create_gaussian(A, FWHM, x0):
    # Generate 100 points in 0.01 minute intervals starting at x0 - (FWHM * 3)
    x = np.arange(x0-(FWHM*3), x0+(FWHM*3), settings.extraction["gaussian_step_size"])
   
    # Calculate the amplitude from the area and FWHM
    amplitude = A / (np.sqrt(2 * np.pi) * (FWHM / 2.3548))  # FWHM = 2.3548 * sigma
   
    # Calculate the Gaussian function values
    y = amplitude * np.exp(-4 * np.log(2) * (x - x0)**2 / FWHM**2)
   
    # Numerically integrate the Gaussian using the trapezoidal rule
    integral = np.trapz(y, x)
   
    # Return the list of points and the integral
    return (x, y), integral


def sum_gaussian(points1, points2):
    # Unpack the points and y values for both Gaussians
    x1, y1 = points1
    x2, y2 = points2
   
    # Find the common x range by taking the union of both x1 and x2
    x_common = np.union1d(x1, x2)
   
    # Create new y values for both Gaussians with the common x range
    y1_common = np.interp(x_common, x1, y1)  # Interpolate y1 onto x_common
    y2_common = np.interp(x_common, x2, y2)  # Interpolate y2 onto x_common
   
    # Sum the two Gaussians
    y_sum = y1_common + y2_common
   
    # FIX: correct argument order in trapezoid integration
    integral = np.trapz(y_sum, x_common)
   
    # Return the summed Gaussian points
    return (x_common, y_sum), integral


   

def plot_data(isobar_key, file_name, editable=True):
    selected_frame = isobar_frames[isobar_key].file_dict[file_name]

    # Create a new window for the plot and optional editable dataframe
    plot_window = tk.Toplevel(root)
    plot_window.title(f"{isobar_key} - {file_name}")
    plot_window.geometry("700x500")

    # Create an empty plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Build summed Gaussian trace
    summed_gaussians = None
    for _, row in selected_frame.iterrows():
        A = float(row.get("Area", np.nan))
        FWHM = float(row.get("FWHM", np.nan))
        x0 = float(row.get("RT", np.nan))
        if not np.isfinite(A) or not np.isfinite(x0):
            continue
        if not np.isfinite(FWHM) or FWHM <= 0:
            FWHM = 17.0  # safe default in seconds
        g_points, _ = create_gaussian(A, FWHM, x0)
        if summed_gaussians is None:
            summed_gaussians = g_points
        else:
            summed_gaussians, _ = sum_gaussian(summed_gaussians, g_points)

    # Nothing to plot?
    if summed_gaussians is None:
        ax.text(0.5, 0.5, "No valid data to plot",
                transform=ax.transAxes, ha="center", va="center")
    else:
        # Unpack and format (sec->min, normalize to %)
        x_sec, y = summed_gaussians
        x_min = np.asarray(x_sec, dtype=float) / 60.0
        y = np.asarray(y, dtype=float)
        ymax = np.max(y) if y.size and np.isfinite(np.max(y)) else 0.0
        y_norm = (y / ymax) * 100.0 if ymax > 0 else y

        ax.plot(x_min, y_norm, label="Summed Gaussians", color='black', linewidth=2)
        ax.legend(loc="best", fontsize="small")

    # Titles / labels reflecting units and normalization
    ax.set_title(f"{isobar_key} - {file_name}")
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("Summed intensity (% of file max)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.set_ylim(bottom=0)

    # Display plot in the new window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    if editable:
        # Create a Frame for DataFrame and save button
        frame = tk.Frame(plot_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        text_widget = tk.Text(frame, wrap=tk.WORD, width=80, height=10)
        text_widget.pack(fill="both", expand=True)

        # Insert the dataframe content into the Text widget
        df_text = selected_frame.to_string(index=False)
        text_widget.insert(tk.END, df_text)

        def save_changes():
            updated_data = text_widget.get("1.0", tk.END).strip()
            updated_data = '\n'.join([' '.join(line.split()) for line in updated_data.split('\n')]).replace(' ', ',')
            updated_df = pd.read_csv(StringIO(updated_data))

            # Update the dataframe with the new data
            isobar_frames[isobar_key].file_dict[file_name] = updated_df

            # Recreate the plot with the updated data (same units/normalization)
            ax.clear()
            summed_gaussians = None
            for _, row in updated_df.iterrows():
                A = float(row.get("Area", np.nan))
                FWHM = float(row.get("FWHM", np.nan))
                x0 = float(row.get("RT", np.nan))
                if not np.isfinite(A) or not np.isfinite(x0):
                    continue
                if not np.isfinite(FWHM) or FWHM <= 0:
                    FWHM = 17.0
                g_points, _ = create_gaussian(A, FWHM, x0)
                if summed_gaussians is None:
                    summed_gaussians = g_points
                else:
                    summed_gaussians, _ = sum_gaussian(summed_gaussians, g_points)

            if summed_gaussians is None:
                ax.text(0.5, 0.5, "No valid data to plot",
                        transform=ax.transAxes, ha="center", va="center")
            else:
                x_sec, y = summed_gaussians
                x_min = np.asarray(x_sec, dtype=float) / 60.0
                y = np.asarray(y, dtype=float)
                ymax = np.max(y) if y.size and np.isfinite(np.max(y)) else 0.0
                y_norm = (y / ymax) * 100.0 if ymax > 0 else y
                ax.plot(x_min, y_norm, label="Summed Gaussians", color='black', linewidth=2)
                ax.legend(loc="best", fontsize="small")

            ax.set_title(f"{isobar_key} - {file_name}")
            ax.set_xlabel("Retention time (min)")
            ax.set_ylabel("Summed intensity (% of file max)")
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            ax.set_ylim(bottom=0)
            canvas.draw()

        save_button = tk.Button(frame, text="Save Changes", command=save_changes)
        save_button.pack(side="bottom", pady=10)

    close_button = tk.Button(plot_window, text="Close", command=plot_window.destroy)
    close_button.pack(side="bottom", pady=10)

     
def sort_isobar_frames_by(attribute: str, descending: bool = False):
    """
    Return a new dict whose values (isobarFrame objects) are ordered by
    the chosen scalar attribute.  If the attribute is unexpectedly a
    Series, list, NumPy array, etc. the first element is used; NaNs are
    coerced to an empty string so the key is always comparable.
    """
    valid_attributes = {"name", "adduct", "average_intensity", "ID"}
    if attribute not in valid_attributes:
        raise ValueError(f"{attribute!r} is not a sortable field")

    def _key(frame):
        val = getattr(frame, attribute, "")
        # ───── normalise to a scalar ─────
        if isinstance(val, pd.Series):
            val = val.iloc[0] if not val.empty else ""
        elif isinstance(val, (list, np.ndarray)):
            val = val[0] if len(val) else ""
        if pd.isna(val):
            val = ""
        return val

    return dict(
        sorted(isobar_frames.items(),
               key=lambda item: _key(item[1]),
               reverse=descending)
    )


   
   
       
def plot_all(isobar_key, original=False):
    # Choose which dict to use
    files_dict = (
        isobar_frames[isobar_key].original_file_dict
        if original else
        isobar_frames[isobar_key].file_dict
    )

    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    mode = "Original" if original else "Edited"
    plot_window.title(f"{isobar_key} - All Files ({mode})")
    plot_window.geometry("900x650")

    # Create an empty plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # Axis labels / title
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("RT (min)", fontsize=14)
    ax.set_ylabel("Summed intensity (% of file max)", fontsize=14)
    ax.set_title(f"Individual Retention Times for {isobar_key}", fontsize=16)

    any_plotted = False

    for file_name, selected_frame in files_dict.items():
        # Skip empty or None frames safely
        if selected_frame is None or selected_frame.empty:
            continue

        summed_gaussians = None

        # Sum Gaussians for this file
        for _, row in selected_frame.iterrows():
            # Pull values with guards
            try:
                A = float(row.get("Area", np.nan))
                x0 = float(row.get("RT", np.nan))
                FWHM = row.get("FWHM", np.nan)
                FWHM = float(FWHM) if pd.notna(FWHM) else np.nan
            except Exception:
                continue

            # Basic sanity checks
            if not np.isfinite(A) or not np.isfinite(x0):
                continue

            # FWHM guard: fall back to 17 sec if missing/invalid/non‑positive
            if not np.isfinite(FWHM) or FWHM <= 0:
                FWHM = 17.0

            # Build and sum the Gaussians
            g_points, _ = create_gaussian(A, FWHM, x0)
            if summed_gaussians is None:
                summed_gaussians = g_points
            else:
                summed_gaussians, _ = sum_gaussian(summed_gaussians, g_points)

        # Nothing to plot for this file
        if summed_gaussians is None:
            continue

        x_values, y_values = summed_gaussians
        if not isinstance(x_values, np.ndarray):
            x_values = np.asarray(x_values, dtype=float)
        if not isinstance(y_values, np.ndarray):
            y_values = np.asarray(y_values, dtype=float)

        if x_values.size == 0 or y_values.size == 0:
            continue

        ymax = np.max(y_values)
        if not np.isfinite(ymax) or ymax <= 0:
            continue

        # Normalize to % and convert x to minutes
        y_norm = (y_values / ymax) * 100.0
        x_min = x_values / 60.0

        ax.plot(
            x_min,
            y_norm,
            label=f"{file_name}",
            linestyle='-',
            linewidth=1.8,
            alpha=0.9
        )
        any_plotted = True

    # Decorate & render
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)
    ax.set_ylim(bottom=0)
    if any_plotted:
        ax.legend(loc="best", fontsize="small")
    else:
        ax.text(
            0.5, 0.5, "No valid traces to plot",
            transform=ax.transAxes, ha="center", va="center"
        )

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)
    canvas.draw()




def alphanumeric_sort_key(value):
    # Split the value into numeric and non-numeric components
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', value)]


def export_settings(settings, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "settings.yaml")
   
    try:
        # Convert settings object to a dictionary
        data = settings.__dict__
       
        with open(file_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
       
        messagebox.showinfo("Success", "Settings exported successfully!")
    except Exception as e:
        if settings.verbose_exceptions:
            messagebox.showerror("Error", f"Failed to export settings: {e}")

def _restore_protein_column_syntax(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
   
    if getattr(settings, "metabolyte_type", "metabolite") != "protein":
        return df        
    # ---------- reverse rename map -----------------------------------------
    reverse_map = {}
    if "Metabolite name" in df.columns:
        if "Protein ID" in original_df.columns:          # Skyline DIA‑NN export
            reverse_map["Metabolite name"] = "Protein ID"
        elif "Protein Name" in original_df.columns:      # alternate naming
            reverse_map["Metabolite name"] = "Protein Name"

    if "Formula" in df.columns and "cf" in original_df.columns:
        reverse_map["Formula"] = "cf"

    if "Average Rt(min)" in df.columns and "Precursor Retention Time (sec)" in original_df.columns:
        # convert minutes back to seconds
        df["Precursor Retention Time (sec)"] = (df["Average Rt(min)"] * 60.0).round(2)
        reverse_map["Average Rt(min)"] = "Precursor Retention Time (sec)"

    # optional protein‑specific extras
    if "Adduct"     in df.columns and "Protein Adduct"     in original_df.columns:
        reverse_map["Adduct"]     = "Protein Adduct"
    if "Adduct_cf"  in df.columns and "Protein Adduct_cf"  in original_df.columns:
        reverse_map["Adduct_cf"]  = "Protein Adduct_cf"
    if "Ontology"   in df.columns and "Protein Ontology"   in original_df.columns:
        reverse_map["Ontology"]   = "Protein Ontology"

    # ---------- apply renaming ---------------------------------------------
    df = df.rename(columns=reverse_map)

    # ---------- restore original order -------------------------------------
    left = [c for c in original_df.columns if c in df.columns]
    right = [c for c in df.columns if c not in left]
    return df[left + right]





def save_final_df():
    global isobar_frames, settings, original_df

    import pandas as pd
    import os
    from tkinter import filedialog, messagebox
    print("Please wait while the program outputs the project folder... (will print 'Finished' when completed.")
    settings.verbose_output = True

    # 1) Collect all visible alignment_dfs
    dfs = []
    for fr in isobar_frames.values():
        if getattr(fr, 'hide', False) and not getattr(settings, 'verbose_output', False):
            continue
        dfs.append(fr.alignment_df)

    if not dfs:
        messagebox.showinfo("Info", "No data to save.")
        return

    output_df = pd.concat(dfs, ignore_index=True)


    # 2) Ask user where to save (treat returned path as a folder)
    folder_path = filedialog.asksaveasfilename(
        title="Select Folder and Enter Name",
        defaultextension="",
        filetypes=[("All Files", "*.*")]
    )
    if not folder_path:
        return

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 3) Restore protein syntax if needed
    if getattr(settings, "metabolyte_type", "metabolite") == "protein":
        output_df = _restore_protein_column_syntax(output_df, original_df)
        output_df = output_df.drop(columns=["Metabolite name"], errors="ignore")

    # 4) Write outputs
    output_df.insert(0, 'Alignment ID', output_df.pop('Alignment ID'))
    output_df.to_csv(f'{folder_path}//Final_dataframe.csv', index=False)
    export_settings(settings, folder_path)
    original_df.to_csv(f'{folder_path}//Original_dataframe.csv', index=False)
   

    # Create and save the overlay plot (global mTIC vs reference) in the output folder
    export_global_mtic_overlay(folder_path)
    
   


    # 5) Persist smoothed EIC caches alongside the project and rewrite paths
    cache_out_dir = os.path.join(folder_path, "smooth_cache")
    os.makedirs(cache_out_dir, exist_ok=True)

    for fr in isobar_frames.values():
        for attr in ("smooth_path", "smooth_raw_path"):
            p = getattr(fr, attr, None)
            if p and isinstance(p, str) and os.path.exists(p):
                dest = os.path.join(cache_out_dir, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dest):
                    shutil.copy2(p, dest)
                    setattr(fr, attr, dest)

    # 6) Save the project AFTER paths point to the portable cache
    save_variables(file_path=f'{folder_path}//Project.pkl')

    # 7) Move any temp plots (from the chosen temp subfolder) into the project output folder
    if settings.export_method == 'save':
        # Source temp folder where intermediate SVGs were written
        src_dir_path = getattr(settings, "_temp_plots_dir", None)
        if not src_dir_path:
            # fallback: try to derive from PLOT_DIR (do not prompt)
            plot_dir_guess = getattr(settings, "_plot_dir", None) or PLOT_DIR
            if plot_dir_guess:
                src_dir_path = os.path.join(plot_dir_guess, TEMP_PLOTS_SUBDIR_NAME)
            else:
                src_dir_path = os.path.join(os.path.dirname(__file__), TEMP_PLOTS_SUBDIR_NAME)

        src_dir = Path(src_dir_path)
        dst_dir = Path(folder_path) / TEMP_PLOTS_SUBDIR_NAME
        os.makedirs(dst_dir, exist_ok=True)

        if src_dir.exists():
            # move .svg (and other files if needed)
            for f in src_dir.glob("*"):
                try:
                    shutil.move(str(f), str(dst_dir / f.name))
                except Exception:
                    try:
                        shutil.copy2(str(f), str(dst_dir / f.name))
                    except Exception:
                        pass

            # attempt cleanup of source
            try:
                for item in src_dir.iterdir():
                    if item.is_file():
                        try: item.unlink()
                        except Exception: pass
                    else:
                        try: shutil.rmtree(item)
                        except Exception: pass
                src_dir.rmdir()
            except Exception:
                pass


    print('Finished outputting project folder. Thank you.')

def display_buttons(scroll_to_position=None):
    global isobar_frames, container
    isobar_frames = sort_isobar_frames_by(sortby_attribute, descending=descending)
   
    # ── NEW: short-circuit if the ID file is large ────────────────────
    if isinstance(original_df, pd.DataFrame) and len(isobar_frames) > 500:

        # Clear any existing button grid
        if 'container' in globals() and container.winfo_exists():
            container.destroy()

        # Show a simple message instead of the button matrix
        container = tk.Frame(root)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text=f"ID file loaded of {len(isobar_frames)}, ready for automated alignment",
            font=("Helvetica", 12, "bold"),
            pady=40
        ).pack()

        return  # Skip the rest of the function
   
   
    # Check if the container frame already exists and destroy it
    if 'container' in globals() and container.winfo_exists():
        container.destroy()

    # Create a frame to hold the canvas and scrollbars
    container = tk.Frame(root)
    container.pack(fill="both", expand=True)

    # Create the canvas for scrolling
    canvas = tk.Canvas(container)
    canvas.pack(side="left", fill="both", expand=True)

    # Add vertical and horizontal scrollbars
    v_scroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    v_scroll.pack(side="right", fill="y")

    h_scroll = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
    h_scroll.pack(side="bottom", fill="x")

    # Configure the canvas to use the scrollbars
    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    # Create a frame inside the canvas to hold the buttons
    button_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=button_frame, anchor="nw")

   
    for row_idx, isobar_key in enumerate(isobar_frames.keys()):
        # Skip if the 'hide' attribute is set to True
        if getattr(isobar_frames[isobar_key], 'hide', False):
            continue
       
        # "Reveal" button to open a new window with main buttons
        reveal_button = tk.Button(
            button_frame,
            text=f"Reveal\n{isobar_key}",
            command=lambda k=isobar_key: reveal_main_buttons(k),
            width=15,
            height=5,
        )
        reveal_button.grid(row=row_idx, column=0, padx=2, pady=2)

        # "Reveal Original" button to open a new window with original buttons
        reveal_original_button = tk.Button(
            button_frame,
            text=f"Reveal Original\n{isobar_key}",
            command=lambda k=isobar_key: reveal_original_main_buttons(k),
            width=15,
            height=5,
        )
        reveal_original_button.grid(row=row_idx, column=1, padx=2, pady=2)

        # Extra button for additional functionality
        extra_button = tk.Button(
            button_frame,
            text=f"{isobar_key}\nDisplay all?",
            command=lambda k=isobar_key: on_extra_button_click(k),
            width=15,
            height=5,
        )
        extra_button.grid(row=row_idx, column=2, padx=2, pady=2)

        # New "Display all (Original)" button referencing original_file_dict
        display_original_button = tk.Button(
            button_frame,
            text=f"{isobar_key}\nDisplay all (Original)?",
            command=lambda k=isobar_key: on_display_original_button_click(k),
            width=15,
            height=5,
        )
        display_original_button.grid(row=row_idx, column=3, padx=2, pady=2)

        # Edit button for marking rows as edited or unedited
        edit_text = f"Mark\n{isobar_key}\nAs un-edited?" if isobar_frames[isobar_key].edited else f"Mark\n{isobar_key}\nAs edited?"
        edit_button = tk.Button(
            button_frame,
            text=edit_text,
            command=lambda k=isobar_key: on_edit_button_click(k),
            width=15,
            height=5,
            bg="red" if isobar_frames[isobar_key].edited else "green",
        )
        edit_button.grid(row=row_idx, column=4, padx=2, pady=2)

    button_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    if scroll_to_position:
        scrollable_width = canvas.bbox("all")[2]
        scrollable_height = canvas.bbox("all")[3]
        vertical_units_to_scroll = int(scrollable_height * scroll_to_position[1])
        horizontal_units_to_scroll = int(scrollable_width * scroll_to_position[0])
        canvas.yview_scroll(vertical_units_to_scroll, "units")
        canvas.xview_scroll(horizontal_units_to_scroll, "units")
       
   
    def on_button_click(isobar_key, file_name, button, change = True, editable = True):
        # Call the plot function
        plot_data(isobar_key, file_name, editable)
        # Change the button's background color to green
        if change:
            button.config(bg="green")
   
   
    def reveal_main_buttons(isobar_key):
        new_window = tk.Toplevel(root)
        new_window.title(f"Buttons for {isobar_key}")
   
        # Create buttons for the selected isobar_key
        for file_idx, file_name in enumerate(isobar_frames[isobar_key].file_dict.keys()):
            mz_value = isobar_frames[isobar_key].mz
   
            # Define the button inside the loop
            button = tk.Button(
                new_window,
                text=f"{file_name}\n(mz: {mz_value})",
                width=15,
                height=5,
            )
            button.grid(row=file_idx // 5, column=file_idx % 5, padx=2, pady=2)
   
            # Use a lambda to pass the button instance explicitly
            button.config(
                command=lambda f=file_name, b=button: on_button_click(isobar_key, f, b)
            )


    def reveal_original_main_buttons(isobar_key):
        new_window = tk.Toplevel(root)
        new_window.title(f"Original Buttons for {isobar_key}")
       
        # Create buttons for the selected isobar_key's original files
        for file_idx, file_name in enumerate(isobar_frames[isobar_key].original_file_dict.keys()):
            mz_value = isobar_frames[isobar_key].mz
       
            # Define the button inside the loop
            button = tk.Button(
                new_window,
                text=f"{file_name}\n(mz: {mz_value})",
                width=15,
                height=5,
            )
            button.grid(row=file_idx // 5, column=file_idx % 5, padx=2, pady=2)
       
            # Prevent the button from turning green on click by setting activebackground to None
            button.config(
                activebackground=new_window.cget("bg"),  # Set it to the window's background color or None
                command=lambda f=file_name, b=button: on_button_click(isobar_key, f, b, change=False, editable = False)
            )


    def on_extra_button_click(isobar_key):
        plot_all(isobar_key)

    def on_edit_button_click(isobar_key):
        isobar_frames[isobar_key].edited = not isobar_frames[isobar_key].edited
        current_scroll_position = (canvas.xview()[0], canvas.yview()[0])
        display_buttons(scroll_to_position=current_scroll_position)

    def on_display_original_button_click(isobar_key):
        plot_all(isobar_key, original = True)


def load_mzml_files(reference_file_name):
    global remaining_df
    global original_df
    df = original_df
    df = _dedup_columns(df)

    # Open file dialog to select mzML files
    file_paths = filedialog.askopenfilenames(
    title="Select mzML files",
    filetypes=[("mzML files", "*.mzML")]
)
    # Dictionary to store file paths with filenames as keys
    mzml_files_dict = {}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        mzml_files_dict[file_name] = file_path
        cols_to_add = [f'{file_name.split(".mzML")[0]}_Abn', f'{file_name.split(".mzML")[0]}_RT_sec']
        for col in cols_to_add:
            if col not in original_df.columns:
                original_df[col] = np.nan

    if f'{reference_file_name}.mzML' not in mzml_files_dict:
        print(f"Missing the following required file: {reference_file_name}.mzML. Please include it in your selection.")
        return None, False
   
    create_isobar_frames(df, settings.extraction["ppm_value"])
       
    return mzml_files_dict,True


def calculate_ppm_range(mz_value, ppm_tolerance):
    delta = mz_value * ppm_tolerance / 1e6
    return mz_value - delta, mz_value + delta



# ------------------------------------------------------------------
# helper: build TIC
# ------------------------------------------------------------------
def extract_tic(exp):
    return [
        (spectrum.getRT(), sum(spectrum.get_peaks()[1]))
        for spectrum in exp.getSpectra()
        if spectrum.getMSLevel() == 1 and len(spectrum.get_peaks()[1]) > 0
    ]

# ------------------------------------------------------------------
# helper: ppm windows  (***now takes local isobar_frames***)
# ------------------------------------------------------------------
def initialize_mz_ranges_and_eics(isobar_frames, ppm_tolerance):
    return {
        key: calculate_ppm_range(obj.mz, ppm_tolerance)
        for key, obj in isobar_frames.items()
    }

# ------------------------------------------------------------------
# helper: sum multiple EICs into a single trace
# ------------------------------------------------------------------
def sum_sampled_eics(sampled_eics):
    intensity_dict = defaultdict(list)
    for eic in sampled_eics.values():
        for rt, intensity in eic:
            intensity_dict[rt].append(intensity)
    return sorted((rt, sum(intensities)) for rt, intensities in intensity_dict.items())

# ------------------------------------------------------------------
# helper: pull intensities for one spectrum
# ------------------------------------------------------------------


def process_spectrum(rt, mz_array, intensity_array,
                     mz_ranges, charge_map,
                     eics_dict, step, use_mass_shift=True):
    """
    Uniform scan-stride down-sampling decided once per spectrum (by RT),
    without changing any callers. Works because process_spectrum is called
    multiple times per spectrum with the same RT; we cache the decision
    and reuse it within that spectrum.

    - step <= 1  → keep every MS¹ scan
    - step  > 1  → keep scan if (scan_index % step == 0), where scan_index
                   advances only when RT changes.
    - If RT decreases (new file), the internal state is reset.
    """
    # ---------- internal persistent state (no external changes needed) ----------
    st = getattr(process_spectrum, "_ds_state", None)
    if st is None:
        st = {'last_rt': None, 'scan_index': -1, 'last_keep': True}
        process_spectrum._ds_state = st

    # Detect file boundary or RT rewind → reset state
    if st['last_rt'] is not None and rt < st['last_rt']:
        st['last_rt'] = None
        st['scan_index'] = -1

    # New spectrum (RT changed) → advance scan index and decide keep/skip
    if st['last_rt'] is None or abs(rt - st['last_rt']) > 1e-12:
        st['scan_index'] += 1
        st['last_rt'] = rt
        st['last_keep'] = True if (step is None or step <= 1) else (st['scan_index'] % int(step) == 0)

    # Skip work if this scan is not selected by the stride
    if not st['last_keep']:
        return eics_dict

    # ---------- intensity accumulation (unchanged logic) ----------
    neutron_mass = settings.neutromer_average_mass  # aliased to extraction['neutromer_avg_mass']

    for key, (mz_min, mz_max) in mz_ranges.items():
        z = charge_map.get(key, 1)  # fall back to +1

        if use_mass_shift:
            # per-charge spacing: Δm/z = neutron_mass / z
            shifts = (0, neutron_mass / z, 2 * neutron_mass / z)  # M, M+1, M+2
        else:
            shifts = (0,)  # only the monoisotopic window

        total_intensity = 0.0
        for shift in shifts:
            start_idx = np.searchsorted(mz_array, mz_min + shift, side="left")
            end_idx   = np.searchsorted(mz_array, mz_max + shift, side="right")
            if start_idx < end_idx:
                total_intensity += float(intensity_array[start_idx:end_idx].sum())

        # Append for every key on kept scans (zeros included), preserving alignment
        # (your dicts are already pre-populated, but setdefault keeps this robust)
        eics_dict.setdefault(key, []).append((rt, total_intensity))

    return eics_dict


# ------------------------------------------------------------------
# main routine
# ------------------------------------------------------------------
def extract_eics_from_isobar_frames(file_path, isobar_frames,
                                    ppm_tolerance=settings.extraction["ppm_value"], step=1):

    # ---- load file ------------------------------------------------
    exp = MSExperiment()
    MzMLFile().load(file_path, exp)

    # ---- initialise dicts & m/z windows ---------------------------
    mz_ranges       = initialize_mz_ranges_and_eics(isobar_frames, ppm_tolerance)
    charges         = {key: frame.charge for key, frame in isobar_frames.items()}
    sampled_eics    = {key: [] for key in isobar_frames}
    sampled_eics_M0 = {key: [] for key in isobar_frames}

    if settings.standards_only:
        normalization_eics = {key: [] for key in isobar_frames if "d7" in key}
    else:
        normalization_eics = sampled_eics        # alias → already filled below

    # ---- iterate over spectra ------------------------------------
    for spectrum in tqdm(exp.getSpectra(),
                         desc=f"Extracting EICs from {os.path.basename(file_path)}",
                         unit="spectrum", leave=False):

        if spectrum.getMSLevel() != 1:
            continue

        mz_array, intensity_array = spectrum.get_peaks()
        rt = spectrum.getRT()

        # main traces (with & without mass‑shift)
        process_spectrum(rt, mz_array, intensity_array,
                     mz_ranges, charges,
                     sampled_eics, step)
        process_spectrum(rt, mz_array, intensity_array,
                     mz_ranges, charges,
                     sampled_eics_M0, step, use_mass_shift=False)

        # normalisation trace (only standards when requested)
        if settings.standards_only:
            process_spectrum(rt, mz_array, intensity_array,
                         mz_ranges, charges,
                         normalization_eics, step)
    # ---- build summed trace & TIC --------------------------------
    summed_eics = sum_sampled_eics(normalization_eics)
    tic         = extract_tic(exp)

    # ---- quick plot ----------------------------------------------
    if summed_eics:
       try:
           # unpack
           rt_vals, sum_vals = zip(*summed_eics)

           fig = Figure(figsize=(10, 6))
           ax = fig.add_subplot(111)
           ax.plot([rt / 60.0 for rt in rt_vals], sum_vals,
                   label="Summed_eics", linewidth=1.6)
           ax.set_xlabel("Retention Time (min)")
           ax.set_ylabel("Summed EIC Intensity")
           ax.set_title(f"Summed EIC: {os.path.basename(file_path)}")
           ax.grid(True)
           ax.legend()

           # create a safe filename from the mzML filename and save as SVG
           base_name = os.path.splitext(os.path.basename(file_path))[0]
           _save_figure_svg(fig, f"{base_name}_SummedEIC.svg")
       except Exception as e:
           # only spam console when verbose exceptions intended
           if settings.verbose_exceptions:
               print(f"[extract_eics_from_isobar_frames] failed to save summed_eics plot: {e}")
       finally:
           try:
               plt.close(fig)
           except Exception:
               plt.close()  # best effort
    else:
       if settings.verbose_exceptions:
           print("⚠️  No summed_eics were generated.")

# ------------------------------------------------------------------
# ---- build summed trace & TIC  ----------------------------------
#   … all your existing code up to here …
# -------------------------eeeeee-----------------------------------------

# ── NEW: make every EIC a compact 2-column NumPy array ────────────
    for d in (sampled_eics, sampled_eics_M0):
        for k, pts in d.items():                         # pts is the Python list
            d[k] = np.asarray(pts, dtype=np.float32)     # (N,2) array  <<<<<
    tic         = np.asarray(tic,         dtype=np.float32)   # ← NEW line
    summed_eics = np.asarray(summed_eics, dtype=np.float32)
# ------------------------------------------------------------------
# ---- quick plot + return  -------------------
    # ---- return ---------------------------------------------------
    return sampled_eics, tic, sampled_eics_M0, summed_eics


def remove_duplicates(x, y):
    unique_x, indices = np.unique(x, return_index=True)
    unique_y = y[indices]
    return unique_x, unique_y




def interpolate_and_smooth(
    eic,
    window_length=settings.retention_time['smoothing']['window_length'],
    polyorder=settings.retention_time['smoothing']['polynomial_order']
):
    """Interpolate and smooth the data using Savitzky–Golay filter, with x‑axis extension to negative seconds.
       Fixes interp1d warnings by avoiding duplicate x values (especially at 0 sec) and enforcing uniqueness.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    # Unpack and coerce to float arrays
    x, y = zip(*eic)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove exact duplicates in the original series
    x, y = remove_duplicates(x, y)

    # Drop NaNs
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # Extend x-axis to negative times WITHOUT duplicating 0
    axis_padding = settings.retention_time['axis_padding']
    pad_x = np.linspace(-axis_padding, 0.0, axis_padding, endpoint=False)  # don't include 0
    x_extended = np.concatenate((pad_x, x))
    y_extended = np.concatenate((np.zeros(pad_x.size, dtype=float), y))

    # Ensure strictly increasing x for interp1d (remove duplicates introduced by concatenation)
    x_extended, uniq_idx = np.unique(x_extended, return_index=True)
    y_extended = y_extended[uniq_idx]

    # Build interpolation grid
    x_interp = np.linspace(x_extended.min(), x_extended.max(), max(10 * x_extended.size, 1000))

    # Use cubic when there are enough points, otherwise fall back to linear
    kind = 'cubic' if x_extended.size >= 4 else 'linear'
    interpolator = interp1d(
        x_extended,
        y_extended,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True
    )
    y_interp = interpolator(x_interp)

    # Smooth
    w = _savgol_window(len(x_interp), preferred_w=window_length, polyorder=polyorder)
    if w is None:
        # Very short series: just use the interpolated values (or do a tiny moving average if you prefer)
        y_smooth = y_interp
    else:
        # Poly must still be <= w-1; SciPy requires polyorder < window_length
        poly = min(polyorder, w - 1)
        y_smooth = savgol_filter(y_interp, w, poly, mode='nearest')

    # Clip negatives
    y_smooth = np.maximum(y_smooth, 0.0)

    # Remove the negative RT region
    valid_range = x_interp > settings.retention_time['negative_rt_cutoff']
    x_interp = x_interp[valid_range]
    y_smooth = y_smooth[valid_range]

    # Normalize to % of max (guard against all‑zero)
    max_val = np.max(y_smooth) if np.max(y_smooth) > 0 else 1.0
    y_smooth_norm = (y_smooth / max_val) * 100.0

    # Final safety clipping
    y_smooth_norm[y_smooth_norm < 0] = 0.0
    y_smooth[y_smooth < 0] = 0.0

    return (x_interp, y_smooth_norm), (x_interp, y_smooth)



def rescale_to_legendre_space(x, min_x, max_x):
    # Clip the input array to be within the specified range (inclusive)
    x_clipped = np.clip(x, min_x, max_x)
   
    # Apply the rescaling formula to the clipped values
    return 2 * (x_clipped - min_x) / (max_x - min_x) - 1

def apply_legendre_polynomial(x_scaled, coeffs):
    x_scaled = np.array(x_scaled)  # Ensure x_scaled is a NumPy array
    P = np.polynomial.Legendre(coeffs)
    return P(x_scaled)

def correct_retention_time(original_x, correction, minimum=None, maximum=None):
    # Ensure original_x is a numpy array
    original_x = np.array(original_x)
   
   
    # Create a copy of the original_x to prevent modifying the input directly
    corrected_x = original_x.copy()
   
    # Apply correction only in the specified region
    if minimum is not None and maximum is not None:
        # Find indices within the region of interest
        in_range = (original_x >= minimum) & (original_x <= maximum)
       
        # Apply correction to the selected range
        corrected_x[in_range] += correction
       
    # If no region is provided, apply the correction globally
    elif minimum is None and maximum is None:
        corrected_x += correction
       
    return corrected_x


def calculate_rmsd(ref_x, ref_y, target_x, target_y):
    target_interpolator = interp1d(target_x, target_y, kind='linear', fill_value="extrapolate")
    target_y_interp = target_interpolator(ref_x)
    return np.sqrt(np.mean((ref_y - target_y_interp)**2))



# ── NEW, FAST VERSION ─────────────────────────────────────────────
from scipy.optimize import differential_evolution


def optimize_coefficients(ref_x, ref_y, tgt_x, tgt_y,
                           *,                     # force keyword args
                           degree=5,
                           iterations=400,        # <- _poly_worker passes this
                           coeff_range=(-50, 50)):
    """
    Fast global optimisation of Legendre-polynomial RT warp using
    differential evolution.  Signature is the same one _poly_worker()
    expects, so no other code has to change.
    Returns (coeffs, rmsd, lo_rt, hi_rt).
    """
    # ---------- early exits ----------------------------------------
    if (tgt_x.size == 0 or ref_x.size == 0 or
        np.all(ref_y == 0)   or np.all(tgt_y == 0)):
        raise SkipEIC("empty or zeroed EIC")

    # ---------- normalise & down-sample ----------------------------
    down = settings.polynomial_correction['downsampling_factor']
    ref_y = ref_y / ref_y.max()
    tgt_y = tgt_y / tgt_y.max()
    ref_x, ref_y = ref_x[::down], ref_y[::down]
    tgt_x, tgt_y = tgt_x[::down], tgt_y[::down]

    pad = settings.retention_time['axis_padding']
    nz  = np.flatnonzero(tgt_y)
    if nz.size == 0:
        raise SkipEIC("zeroed EIC")
       
    tgt_x = tgt_x[max(nz[0]-pad,0): min(nz[-1]+pad+1, tgt_y.size)]
    tgt_y = tgt_y[max(nz[0]-pad,0): min(nz[-1]+pad+1, tgt_y.size)]

    lo, hi = tgt_x.min(), tgt_x.max()
    x_scaled = 2*(tgt_x - lo)/(hi - lo) - 1
    V = np.polynomial.legendre.legvander(x_scaled, degree)  # n × (deg+1)

    #ref_interp = np.interp(tgt_x, ref_x, ref_y)

    def rmsd(c):
        corr   = tgt_x + V @ c
        tgt_ip = np.interp(ref_x, corr, tgt_y, left=0, right=0)
        return np.sqrt(np.mean((ref_y - tgt_ip)**2))

    bounds = [coeff_range]*(degree+1)
    res = differential_evolution(
        rmsd, bounds,
        popsize=15, maxiter=iterations, tol=1e-3,
        polish=True, workers=1,          # 1 or None, NOT -1
        updating='deferred'
)
    return res.x, res.fun, lo, hi


   
   
def sum_datasets(data_list, num_points=1000):
    """
    Robustly sum multiple (x, y) datasets on a shared grid.
    - Enforces strictly increasing, finite x for each dataset to avoid interp1d
      divide-by-zero warnings (duplicate x causes zero denominators).
    - Uses only the overlapping x-range across all valid datasets.
    - Returns (common_x, y_sum, interpolated_ys); or (None, None, []) if not enough data.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # Quick exits
    if not data_list:
        return None, None, []

    cleaned = []
    for x, y in data_list:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Drop NaNs and enforce finite values
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            continue

        # Sort by x
        order = np.argsort(x, kind="mergesort")
        x = x[order]
        y = y[order]

        # Enforce strictly increasing x (remove duplicates)
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if x_unique.size < 2:  # still not enough to interpolate
            continue

        cleaned.append((x_unique, y_unique))

    # If nothing valid remains
    if not cleaned:
        return None, None, []

    # Overlapping x-range across all datasets
    try:
        min_x = max(x.min() for x, _ in cleaned)
        max_x = min(x.max() for x, _ in cleaned)
    except ValueError:
        return None, None, []

    # No overlap or degenerate range
    if not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
        return None, None, []

    # Build a common grid
    npts = int(max(2, num_points))
    common_x = np.linspace(min_x, max_x, npts, dtype=float)

    # Interpolate each dataset onto the common grid
    interpolated_ys = []
    for x, y in cleaned:
        # If dataset can't interpolate (shouldn't happen after cleaning), skip safely
        if x.size < 2:
            continue
        f = interp1d(
            x, y,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True
        )
        interpolated_ys.append(f(common_x))

    if not interpolated_ys:
        return None, None, []

    # Sum across datasets, ignoring NaNs
    y_stack = np.vstack(interpolated_ys)
    y_sum = np.nansum(y_stack, axis=0)

    return common_x, y_sum, interpolated_ys
   

def automated_alignment():
    global isobar_frames
    global file_specific_info
    # Ensure the user chooses the plot directory up-front before any EIC extraction
    try:
        # If the plot directory hasn't been chosen, ask now (non-forcing)
        choose_plot_directory()
    except Exception:
        # best-effort fallback — keep going if dialogs are not available
        try:
            # create defaults without prompting
            if PLOT_DIR is None:
                base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
                PLOT_DIR_local = os.path.join(base, "plots")
                os.makedirs(PLOT_DIR_local, exist_ok=True)
                temp_dir_local = os.path.join(PLOT_DIR_local, TEMP_PLOTS_SUBDIR_NAME)
                os.makedirs(temp_dir_local, exist_ok=True)
                try:
                    setattr(settings, "_plot_dir", PLOT_DIR_local)
                    setattr(settings, "_temp_plots_dir", temp_dir_local)
                except Exception:
                    pass
        except Exception:
            pass
    try:
        file_specific_info
           
    except:
         file_specific_info = {}
   
       
   
    try:
        if not isobar_frames:
            return

       
       
        redo_all = False
        '''
        if settings.force_redo_all:
            redo_all = True'''
        # Check if reference file needs to be loaded or skipped
        reference_file_name, load_reference_file= handle_reference_file()
           
        if load_reference_file:
            redo_all = True
        # Check if EICs need to be extracted or skipped
        if redo_all:
            extract_EIC = True
        else:
            extract_EIC = handle_eic_extraction()

        if extract_EIC:
            mzml_dict, success = load_mzml_files(reference_file_name)
            for key in isobar_frames:

                isobar_frames[key].eic_dict = {}
                isobar_frames[key].eic_dict_all = {}
            if not success:
                print("Please try again.")
                return
            extract_eics(mzml_dict)
            smooth_eics()
           
            if settings.save_intermediates:
                save_variables(temp=True)

       
       
        if redo_all:
            Create_Polynomials =True
        else:
            # Check if polynomials need to be created or skipped
            Create_Polynomials = handle_polynomial_creation()
       
        if Create_Polynomials:
            create_polynomials()
            if settings.save_intermediates:
                save_variables(temp=True)

                   
        #Post Correction step
        process_and_plot_datasets()
       
        get_id_file_peaks()
       
       
        get_normalization_constants()

       
        normalize()
   
       
        display_buttons()
       
       
        """
        # ── one-liner that deletes every *.npz file in the cache folder ──
        import pathlib
        for f in pathlib.Path(CACHE_DIR).glob("*.npz"):
            try:
                f.unlink()
            except OSError:
                pass                     # ignore files that vanished meanwhile
        """
       
        print('Automated alignment completed')

    except Exception as e:
        import os, sys, traceback
        from datetime import datetime
    
        # Always write tracebacks to a file (important for --noconsole exe)
        log_path = "alignment_errors.log"  # relative => goes to os.getcwd()
    
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"CWD: {os.getcwd()}\n")
                f.write(f"sys.executable: {sys.executable}\n")
                f.write(f"argv: {sys.argv}\n")
                try:
                    f.write(f"__file__: {__file__}\n")
                except Exception:
                    f.write("__file__: <unavailable>\n")
                f.write(f"Exception: {repr(e)}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception:
            # If logging fails, fall back to printing (might be invisible in exe, but harmless)
            pass
    
        # Keep your existing behavior too (useful in python runs)
        if settings.verbose_exceptions:
            print(f"An error occurred: {e}")
            print("Traceback:")
        traceback.print_exc()

    return


def handle_reference_file():
    """Handle reference file logic."""
    existing_ref_file = get_existing_ref_file()
    load_reference_file = True  # Default assumption
    if existing_ref_file:
        user_response = messagebox.askyesno(
            "Reference File Exists",
            f"A reference file has already been chosen: {existing_ref_file}. Do you want to choose a different reference fil?e\n Note: If yes, this will result in the re-extraction and re-polynomial fitting as well."
        )
        if not user_response:
            load_reference_file = False
    if load_reference_file:
        reference_file_name = select_reference_file()
        set_ref_file_for_isobar_frames(reference_file_name)
    else:
        reference_file_name = existing_ref_file
    return reference_file_name, load_reference_file

def get_existing_ref_file():
    """Return existing reference file name if it exists in global settings."""
    return getattr(settings, 'ref_file', None)


def select_reference_file():
    """Allow user to select an mzML file as reference."""
    reference_mzml_path = filedialog.askopenfilename(
        title="Select reference mzML file", filetypes=[("mzML files", "*.mzML")]
    )
    if reference_mzml_path:
        return os.path.splitext(os.path.basename(reference_mzml_path))[0]
    return None


def select_internal_reference_mass_csv():
    """Allow user to select a CSV file and load it as a pandas DataFrame."""
    reference_csv_path = filedialog.askopenfilename(
        title="Select reference CSV file", filetypes=[("CSV files", "*.csv")]
    )
    if reference_csv_path:
        try:
            # Load the selected CSV file as a pandas DataFrame
            reference_dataframe = pd.read_csv(reference_csv_path)
            return reference_dataframe
        except Exception as e:
            if settings.verbose_exceptions:
                print(f"Error loading CSV file: {e}")
            return None
   # print("No reference CSV file selected. Skipping reference file selection.")
    return None

def set_ref_file_for_isobar_frames(reference_file_name):

    settings.ref_file = reference_file_name

def handle_eic_extraction():
    """Handle EIC extraction logic."""
    eic_populated = check_eic_populated()
    extract_EIC = False
    if eic_populated:
        user_response = messagebox.askyesno(
            "Re-extract EICs?",
            "EICs have already been extracted. Do you want to re-extract them?"
        )
        if user_response:
            extract_EIC = True
    else:

        extract_EIC = True
    return extract_EIC


def check_eic_populated():
    """Return True if any isobar frame already has EICs OR a persisted smooth cache."""
    import os
    return any(
        (hasattr(fr, 'eic_dict') and fr.eic_dict) or
        (hasattr(fr, 'smooth_path') and isinstance(fr.smooth_path, str) and os.path.exists(fr.smooth_path)) or
        (hasattr(fr, 'smooth_raw_path') and isinstance(fr.smooth_raw_path, str) and os.path.exists(fr.smooth_raw_path))
        for fr in isobar_frames.values()
    )



def extract_eics(mzml_dict):
    global file_specific_info

    #file_specific_info['reference_masses']:{}
    """Extract EICs from mzML files."""
    file_specific_info['tics'] = {}
    file_specific_info['summed_eics'] = {}
    file_specific_info['mz_profile'] = {}
    #file_specific_info['reference_masses'] = {}
    for key1 in tqdm(mzml_dict, desc='Extracting EICs from mzml_dict'):
        temp_eic_dict_all, tic, temp_eic_dict, summed_eics = extract_eics_from_isobar_frames(mzml_dict[key1], isobar_frames)
        file_specific_info['tics'][key1] = tic
        file_specific_info['summed_eics'][key1] = summed_eics
        '''
        for i in standard_eics:
            file_specific_info['reference_masses'][key1][i]['eic'] = standard_eics[i]'''
        for key2 in tqdm(isobar_frames, desc=f'Unpacking EICs from {key1}', leave=False):
            isobar_frames[key2].eic_dict[key1] = temp_eic_dict[key2]
            isobar_frames[key2].eic_dict_all[key1] = temp_eic_dict_all[key2]
           
           

def _encode_key(name: str) -> str:
    """Make a .npz-safe key ('.' → '__DOT__')."""
    return name.replace('.', '__DOT__')

def _decode_key(name: str) -> str:
    """Reverse the .npz key mangling."""
    return name.replace('__DOT__', '.')        


# helper lives at module level (define once, re-use everywhere)
import uuid
CACHE_DIR = "smooth_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _dump_smooth_dict(sdict):
    path = os.path.join(CACHE_DIR, f"{uuid.uuid4().hex}.npz")

    np.savez_compressed(
        path,
        **{
            _encode_key(k): np.vstack(
                [np.asarray(v[0], dtype=np.float32),
                 np.asarray(v[1], dtype=np.float32)]
            )
            for k, v in sdict.items()
        }
    )
    return path
# ---------------------------------------------------------------


def smooth_eics():
    """Interpolate / smooth all EICs and cache them on disk."""
    for key1, fr in isobar_frames.items():

        # ---------- build smoothed traces ------------------------
        fr.smooth_eic_dict          = {}
        fr.smooth_unnormalized_dict = {}

        for key2, eic in fr.eic_dict.items():
            (x_norm, y_norm), (x_raw, y_raw) = interpolate_and_smooth(eic)
            fr.smooth_eic_dict[key2]          = (x_norm, y_norm)
            fr.smooth_unnormalized_dict[key2] = (x_raw, y_raw)

        # ---------- DEBUG: print size BEFORE dumping -------------
        """
        for fname, (x, _) in fr.smooth_eic_dict.items():
            print(f"[{key1}] {fname:<25} points: {len(x):6}")"""

        # ---------- write compressed cache -----------------------
        fr.smooth_path = _dump_smooth_dict(fr.smooth_eic_dict)
        fr.smooth_raw_path = _dump_smooth_dict(fr.smooth_unnormalized_dict)  # NEW

        # free normalised data (unnormalised kept in RAM)
        fr.smooth_eic_dict = None
        fr.smooth_unnormalized_dict = None
       
        # NEW: drop heavy pre‑smooth EICs to reclaim RAM
        if hasattr(fr, 'eic_dict'):
            fr.eic_dict.clear()
        if hasattr(fr, 'eic_dict_all'):
            fr.eic_dict_all.clear()
           
       




def handle_polynomial_creation():
    """Handle polynomial creation logic."""
    polynomials_populated = check_polynomials_populated()
    Create_polynomials = False
    if polynomials_populated:
        user_response = messagebox.askyesno(
            "Polynomials Exist",
            "Polynomials (Pol1dict and Pol2dict) have already been created. Do you want to regenerate them?"
        )
        if user_response:
            Create_polynomials = True
    else:
        Create_polynomials = True
       
       
    return Create_polynomials




def check_polynomials_populated():
    """Check if any isobar_frame already has Pol1dict and Pol2dict populated."""
    return any(
        hasattr(isobar_frame, 'Pol1dict') and isobar_frame.Pol1dict
        for isobar_frame in isobar_frames.values()
    )




def _poly_worker(task):
    """
    Runs in a separate process.
    Loads one smoothed-EIC cache file, fits forward & reverse polynomials,
    returns lightweight dicts.
    """
    key, smooth_path, ref_file, degree, iterations = task

    try:
        # ---------- load the (2 × n) arrays on demand ----------
        npz  = np.load(smooth_path, allow_pickle=False)
        sdict = { _decode_key(k): (arr[0], arr[1]) for k, arr in npz.items() }
        ref_x, ref_y = sdict[f"{ref_file}.mzML"]

        Pol1, Pol2 = {}, {}
        for fname, (x, y) in sdict.items():
            # forward warp: ref → target
            c, r, lo, hi = optimize_coefficients(
                            ref_x, ref_y, x, y,
                            degree=degree,
                            iterations=iterations,
                            coeff_range=settings.polynomial['coefficient_range']   # <-- add
                    )
           
            Pol1[fname] = (c, r, lo, hi)

            # reverse warp: target → ref
            c, r, lo, hi = optimize_coefficients(x, y, ref_x, ref_y,
                                                 degree=degree,
                                                 iterations=iterations,coeff_range=settings.polynomial['coefficient_range']   # <-- add
                    )
            Pol2[fname] = (c, r, lo, hi)

        return key, Pol1, Pol2, False            # normal exit

    except SkipEIC:
        return key, None, None, True             # mark as hidden
    except Exception as e:
       
        if settings.verbose_exceptions:
            print(f"[ERROR] Polynomial fit for {key}: {e}")
        return key, None, None, True             # hide on any error



def create_polynomials():
    """Fit forward (Pol1dict) and reverse (Pol2dict) Legendre warps.

    – Smoothed EICs are streamed from disk per worker to keep RAM low
    – Worker count is capped at 60 to avoid WinAPI handle limits
    """
    global isobar_frames, settings

    ref_file   = settings.ref_file
    degree     = settings.polynomial['poly_degree']
    iterations = settings.polynomial['iterations']

    tasks = [
        (key, fr.smooth_path, ref_file, degree, iterations)
        for key, fr in isobar_frames.items()
        if not fr.hide
    ]

    # ---- 60-process hard ceiling avoids “need at most 63 handles” on Windows
    n_jobs = min(max(1, multiprocessing.cpu_count() -  settings.polynomial['leave_cores_open']), 60)

    with Pool(processes=n_jobs) as pool:
        # `.imap` keeps ordering; switch to `.uimap` if you prefer unordered
        results = list(
            tqdm(pool.imap(_poly_worker, tasks),
                 total=len(tasks),
                 desc="Fitting polynomials")
        )

    # ---- merge lightweight results back into the live objects ---------
    for key, P1, P2, hide in results:
        fr = isobar_frames[key]
        if hide:
            fr.hide = True
            fr.reason_for_hiding = "zeroed EIC"
        else:
            fr.Pol1dict = P1
            fr.Pol2dict = P2

   



def calculate_fwhm_from_peaks(x_values, y_values, peak_x_values, tol=1e-8):
    """
    Calculates the full width at half maximum (FWHM) from the measured full width
    at 75% of the peak height, assuming a Gaussian profile.
   
    Parameters:
        x_values (array-like): The x-axis values.
        y_values (array-like): The y-axis values corresponding to x_values.
        peak_x_values (list): List of x-values where peaks are located.
        tol (float): Tolerance for numerical comparisons.
   
    Returns:
        fwhm_results (list): Calculated FWHM values for each peak.
        crossing_points (list): Tuple of (left_idx, right_idx) crossing points at 75% level.
    """
    fwhm_results = []
    crossing_points = []  # For debugging/visualization

    for peak_x in peak_x_values:
        # Find the index of the peak.
        peak_idx = np.searchsorted(x_values, peak_x)
        peak_height = y_values[peak_idx]

        # Set the reference level at 75% of the peak height.
        ref_height = 0.75 * peak_height

        # Find the left crossing point where the signal falls below the 75% level.
        left_idx = peak_idx
        while left_idx > 0 and y_values[left_idx] >= ref_height - tol:
            left_idx -= 1
        if left_idx == peak_idx:
            left_idx = 0  # Edge case: no crossing found

        # Find the right crossing point where the signal falls below the 75% level.
        right_idx = peak_idx
        while right_idx < len(y_values) - 1 and y_values[right_idx] >= ref_height - tol:
            right_idx += 1
        if right_idx == peak_idx:
            right_idx = len(y_values) - 1  # Edge case: no crossing found

        # Compute the full width at 75%
        width75 = x_values[right_idx] - x_values[left_idx]

        # For a Gaussian, the relationship is:
        # FWHM = width75 * sqrt(ln2 / -ln(0.75))
        scale_factor = np.sqrt(np.log(2) / (-np.log(0.75)))
        fwhm = width75 * scale_factor

        fwhm_results.append(fwhm)
        crossing_points.append((left_idx, right_idx))
       
    return fwhm_results


# ── helper: lazy access to cached smoothed EICs ───────────────────
def _get_smooth(isoframe):
    """
    Return a dict {file.mzML: (x,y)} for *this* isobarFrame.

    • If it is already in RAM, return it immediately.  
    • If it was freed after dumping, load the `.npz` on demand and
      memoise it so the next call is free.
    """
    if isoframe.smooth_eic_dict is None:
        npz = np.load(isoframe.smooth_path, allow_pickle=False)
        isoframe.smooth_eic_dict = {
            _decode_key(k): (arr[0], arr[1])          # x, y  (each 1-D)
            for k, arr in npz.items()
        }
    return isoframe.smooth_eic_dict


def _get_smooth_raw(isoframe):
    """Lazy-load the raw-intensity smoothed EICs."""
    if isoframe.smooth_unnormalized_dict is None:
        npz = np.load(isoframe.smooth_raw_path, allow_pickle=False)
        isoframe.smooth_unnormalized_dict = {
            _decode_key(k): (arr[0], arr[1]) for k, arr in npz.items()
        }
    return isoframe.smooth_unnormalized_dict


# ── patched version ───────────────────────────────────────────────
def collect_datasets(key1):
    """Return two lists of (x,y) tuples – normalised & un-normalised."""
    fr = isobar_frames[key1]
    if fr.hide:
        return [], []

    # ---------- load normalised smoothed traces lazily -------------
    sdict = _get_smooth(fr)  
    raw_dict = _get_smooth_raw(fr)                   # <- NEW lazy load

    datasets              = []                 # normalised
    unnormalized_datasets = []                 # raw intensities

    # -------- normalised (already scaled 0–100 %) ------------------
    for fname, (x, y) in sdict.items():
        coeffs, rmsd, lo, hi = fr.Pol1dict[fname]
        rt_corr = apply_legendre_polynomial(
            rescale_to_legendre_space(x, lo, hi), coeffs)
        x_corr  = correct_retention_time(np.clip(x, lo, hi),
                                         rt_corr, lo, hi)
        datasets.append((x_corr, y))

    # -------- un-normalised (still kept in RAM) --------------------
    for fname, (x_nat, y_nat) in raw_dict.items():
        coeffs, rmsd, lo, hi = fr.Pol1dict[fname]
        rt_corr = apply_legendre_polynomial(
            rescale_to_legendre_space(x_nat, lo, hi), coeffs)
        x_corr  = correct_retention_time(np.clip(x_nat, lo, hi),
                                         rt_corr, lo, hi)
        unnormalized_datasets.append((x_corr, y_nat))
       
    fr.smooth_eic_dict          = None
    fr.smooth_unnormalized_dict = None
       

    return datasets, unnormalized_datasets


   
def sanitize_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)


def plot_summed_and_interpolated_datasets(common_x, y_sum, interpolated_ys,
                                          name, rmsd, filtered_peaks,
                                          sn_threshold, ref_Rt, rt_window):
    export_method = settings.export_method

    # Use non-interactive Figure (safe to create anywhere)
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    # normalize
    norm = np.max(y_sum) if np.max(y_sum) > 0 else 1.0
    y_norm = (y_sum / norm) * 100.0
    sn_norm = (sn_threshold / norm) * 100.0

    x_min = common_x / 60.0  # sec→min

    # overlay interpolated per-file traces
    for y_interp in interpolated_ys:
        ax.plot(x_min, (y_interp / norm) * 100.0, linestyle='dashed', alpha=0.6)

    ax.plot(x_min, y_norm, color='black', linewidth=2, label="Summed Intensity")

    if filtered_peaks:
        fp = np.asarray(filtered_peaks, dtype=float)
        fy = np.interp(fp, common_x, y_norm)  # IMPORTANT: use y_norm scale
        ax.scatter(fp/60.0, fy, zorder=5, label="Filtered Peaks")

    ax.axhline(y=sn_norm, linestyle='dashed', linewidth=2, label=f"S/N Cut ({settings.signal_to_noise_cutoff}×)")

    # draw window
    ax.axvline(x=(ref_Rt-rt_window)/60.0, linestyle='dashed', linewidth=1, label="RT window start")
    ax.axvline(x=(ref_Rt+rt_window)/60.0, linestyle='dotted', linewidth=1, label="RT window end")

    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Summed Intensities (%)")
    ax.set_title(f"Summed Peak-Picking: {name} (RMSD: {rmsd:.2f}) RT = {ref_Rt/60}")
    ax.grid(True)
    ax.legend(loc="best", fontsize="small")

    if export_method == 'plot':
        # plotting interactively — show in main thread GUIs only
        plt.show()
    elif export_method == 'save':
        # Save as SVG
        _save_temp_figure_svg(fig, f"{name}_SummedPeakPicking.svg")


def baseline_std(x, y, x_value, n):
    """
    Calculate the standard deviation of the baseline y-values within n units of a given x-value.

    Parameters:
        x (array-like): Array of x-values.
        y (array-like): Array of y-values (intensities).
        x_value (float): The x-value around which to calculate baseline noise.
        n (float): The range around x_value to consider (i.e., x_value ± n).

    Returns:
        float: Standard deviation of the baseline in the selected range.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Find indices where x-values are within n units of x_value
    mask = (x >= (x_value - n)) & (x <= (x_value + n))

    # Extract corresponding y-values
    baseline_y = y[mask]

    # Compute and return standard deviation of the baseline
    return np.std(baseline_y, ddof=1) if baseline_y.size > 1 else np.nan

# Example usage
x_values = np.linspace(0, 100, 1000)  # Simulated x-values
y_values = np.random.normal(10, 2, 1000)  # Simulated baseline noise with mean=10, std=2

std_baseline = baseline_std(x_values, y_values, x_value=50, n=5)
#print("Baseline standard deviation:", std_baseline)

           
def _savgol_window(n: int, preferred_w: int, polyorder: int) -> int | None:
    """
    Return a valid odd window length for Savitzky–Golay, clamped to the data length.
    - Ensures odd window
    - Ensures window > polyorder
    - Ensures window <= n (series length)
    - Returns None if n is too small to smooth
    """
    # If the series is too short, skip smoothing
    if n < 3:
        return None

    # Start from the preferred window, force odd
    w = int(preferred_w)
    if w % 2 == 0:
        w += 1

    # Cap by data length (must be odd)
    max_odd = n if (n % 2 == 1) else (n - 1)
    w = min(w, max_odd)

    # Minimum odd window that is strictly greater than polyorder
    # (If polyorder is odd, the next odd > polyorder is polyorder+2;
    #  if polyorder is even, it's polyorder+1.)
    min_odd_gt_poly = polyorder + 1 if (polyorder % 2 == 0) else polyorder + 2
    w = max(w, min_odd_gt_poly, 3)  # also enforce at least 3

    # If we can’t achieve a window strictly > polyorder, skip smoothing
    if w <= polyorder or w < 3:
        return None

    return w





def process_and_plot_datasets():
    global settings, isobar_frames, file_specific_info

    items = list(isobar_frames.items())  # stable snapshot for tqdm length
    for key1, fr in tqdm(items,
                         total=len(items),
                         desc="Summing & peak picking",
                         leave=True):
        # Gather corrected traces
        datasets, unnormalized_datasets = collect_datasets(key1)

        # Sum normalized (%) datasets on a common grid
        common_x, y_sum, interpolated_ys = sum_datasets(datasets)
        if common_x is None:
            fr.hide = True
            fr.reason_for_hiding = "empty or invalid normalized data"
            fr.av_rmsd = 'NA'
            continue

        # --- RMSD bookkeeping (unchanged) ---
        avg_pol1_rmsd = np.mean([p[1] for p in fr.Pol1dict.values()]) if fr.Pol1dict else 0
        avg_pol2_rmsd = np.mean([p[1] for p in fr.Pol2dict.values()]) if fr.Pol2dict else 0
        av_av_rmsd    = 0.5 * (avg_pol1_rmsd + avg_pol2_rmsd)
        fr.av_rmsd    = av_av_rmsd

        if av_av_rmsd > settings.polynomial["rmsd_target"]:
            fr.hide = True
            fr.reason_for_hiding = "RMSD too high"
            continue
        fr.reason_for_hiding = ""

        # --- peak candidates from summed trace (unchanged logic; S/G guarded) ---
        poly = 4
        w = _savgol_window(len(y_sum), preferred_w=51, polyorder=poly)  # prefer ~51, will clamp
        if w is None:
            smooth_sum = y_sum  # fallback: no smoothing
        else:
            smooth_sum = savgol_filter(y_sum, w, poly)
        peak_indices = find_peaks(smooth_sum)[0]
        peak_heights = y_sum[peak_indices]

        # --- unnormalized absolute filter (FIX: align grids before indexing) ---
        common_x_u, y_sum_unnorm, _ = sum_datasets(unnormalized_datasets)
        if (common_x_u is None) or (y_sum_unnorm is None):
            fr.hide = True
            fr.reason_for_hiding = "empty or invalid un-normalized data"
            continue

        # Interpolate unnormalized sum onto the normalized grid used for peak indices
        y_unnorm_on_common = np.interp(common_x, common_x_u, y_sum_unnorm, left=0.0, right=0.0)

        # Now it's safe to use the same indices
        unnorm_peak_heights = y_unnorm_on_common[peak_indices]

        # Number of files guard for absolute intensity scaling
        try:
            num_files = len(file_specific_info.get('tics', {}))
        except Exception:
            num_files = 1
        if not num_files:
            num_files = 1

        abs_heights = [h / num_files for h in unnorm_peak_heights]

        max_peak_height = peak_heights.max() if peak_heights.size else 0
        pct_cut = settings.other["peak_percent_cutoff"] / 100.0
        abs_cut = settings.peak_picking_absolute_cutoff  # (kept as-is)

        # master list in seconds
        master_peaks = [
            common_x[i]
            for i, h_rel, h_abs in zip(peak_indices, peak_heights, abs_heights)
            if (h_rel >= pct_cut * max_peak_height) and (h_abs > abs_cut)
        ]
        master_peaks = np.asarray(master_peaks, dtype=float)

        # --- per‑ID refinement (unchanged) ---
        fr.reference_peaks = {}
        ID_column = 'Alignment ID'
        RT_column = 'Average Rt(min)'
        rt_secs   = fr.sub_df[RT_column].astype(float).values * 60.0
        ids       = fr.sub_df[ID_column].astype(str).values
        rt_window = settings.retention_time['peak_filter_seconds']
        sn_cut    = settings.signal_to_noise_cutoff

        for this_id, rt_sec in zip(ids, rt_secs):
            cand = master_peaks

            noise_std = baseline_std(common_x, y_sum, rt_sec, rt_window * 3)
            if not np.isfinite(noise_std) or noise_std <= 0:
                noise_std = np.nanstd(y_sum)
                if noise_std <= 0:
                    noise_std = 1.0  # fallback

            sn_masked = [
                p for p in cand
                if (np.interp(p, common_x, y_sum) / noise_std) >= sn_cut
            ]

            win_masked = [p for p in sn_masked if abs(p - rt_sec) <= rt_window]
            win_masked = adjust_peaks(
                win_masked, common_x, y_sum,
                window=int(settings.retention_time['peak_filter_seconds']/2)
            )

            fr.reference_peaks[this_id] = {}
            fr.reference_peaks[this_id]['x_values'] = win_masked
            fr.reference_peaks[this_id]['fwhm'] = calculate_fwhm_from_peaks(
                common_x, y_sum, win_masked
            )

            plot_summed_and_interpolated_datasets(
                common_x,
                y_sum,
                interpolated_ys,
                this_id,          # title
                av_av_rmsd,
                win_masked,
                noise_std * sn_cut,
                rt_sec,
                rt_window
            )

        # --- build alignment_df (unchanged) ---
        alignment_rows = []
        ID_column = 'Alignment ID'
        for _, sub_row in fr.sub_df.iterrows():
            this_id   = str(sub_row[ID_column])
            ref_info  = fr.reference_peaks.get(this_id, {})
            x_vals    = ref_info.get('x_values', [])
            fwhm_vals = ref_info.get('fwhm', [])

            if not x_vals:
                continue

            if not hasattr(fwhm_vals, '__iter__') or isinstance(fwhm_vals, (float, int)):
                fwhm_vals = [fwhm_vals] * len(x_vals)

            suffixes = [''] if len(x_vals) == 1 else [chr(ord('a') + i) for i in range(len(x_vals))]

            for suf, ref_rt, fwhm in zip(suffixes, x_vals, fwhm_vals):
                new_row = sub_row.copy()
                new_row[ID_column]      = this_id + suf if suf else this_id
                new_row['Reference_RT'] = ref_rt
                new_row['fwhm']         = fwhm
                alignment_rows.append(new_row)

        fr.alignment_df = pd.DataFrame(alignment_rows)
        fr.alignment_df = fr.alignment_df.drop_duplicates(subset='Reference_RT')




def process_reference_peaks_and_areas():
    for key1 in file_specific_info['reference_masses']:
        for key2 in file_specific_info['reference_masses'][key1]:

            if 'eic_smooth' in file_specific_info['reference_masses'][key1][key2]:
                eic_x = file_specific_info['reference_masses'][key1][key2]['eic_smooth'][0]
                eic_y = file_specific_info['reference_masses'][key1][key2]['eic_smooth'][1]
            else:

                continue
           
            # Flatten arrays and proceed with the rest of the function
            retention_times = np.array(eic_x).flatten()
            intensities = np.array(eic_y).flatten()
           
            # Check if intensities are valid
            if len(intensities) == 0 or np.all(intensities == 0):
                continue
           
            # Detect peaks
            peak_indices, peak_properties = find_peaks(intensities, height=0)
           
            if len(peak_indices) == 0:
                print(f"No peaks found for {key2}.")

                continue
           
            # Get the highest peak
            highest_peak_index = peak_indices[intensities[peak_indices].argmax()]
            proper_peak_rt = retention_times[highest_peak_index]
           
            # Calculate FWHM
            results_half = peak_widths(intensities, [highest_peak_index], rel_height=settings.gaussian['fwhm_fraction'])
            fwhm = results_half[0][0]  # Full width at half maximum in index space
           
            # Convert FWHM to retention time units
            fwhm_rt = fwhm * (retention_times[1] - retention_times[0])  # Assume uniform spacing
           
            # Store the highest peak and FWHM in the dictionary
            file_specific_info['reference_masses'][key1][key2]['highest_peak_rt'] = proper_peak_rt
            file_specific_info['reference_masses'][key1][key2]['fwhm'] = fwhm_rt
           
            # Now calculate the area of the peak
            fwhm = file_specific_info['reference_masses'][key1][key2]['fwhm']
            location = file_specific_info['reference_masses'][key1][key2]['highest_peak_rt']
            eic = file_specific_info['reference_masses'][key1][key2]['eic']  # x, y pairs

            # Convert the x, y pairs into separate lists for processing
            x_values, y_values = zip(*eic)  # Unpack the pairs into separate x and y lists
           
            # Find the nearest x-value to 'location'
            nearest_index = min(range(len(x_values)), key=lambda i: abs(x_values[i] - location))
           
            # Extract the y-value at the nearest x-value (height)
            height = y_values[nearest_index]
           
            # Calculate the area of the peak (simplified)
            area = height * fwhm  # assuming a basic Gaussian approximation for area
           
            # Store height and area in key2 dictionary
            file_specific_info['reference_masses'][key1][key2]['height'] = height
            file_specific_info['reference_masses'][key1][key2]['area'] = area


def generate_alphabetic_key(index):
    """Generate alphabetic keys like A, B, ..., Z, AA, AB, etc."""
    key = ""
    while index >= 0:
        key = chr(index % 26 + 65) + key
        index = index // 26 - 1
    return key




import numpy as np

def adjust_peaks(x_values, eic_x_values, eic_y_values, window=30):

    # Convert inputs to numpy arrays for ease of computation.
    x_values = np.array(x_values)
    eic_x_values = np.array(eic_x_values)
    eic_y_values = np.array(eic_y_values)
   
    # Ensure that eic_x_values and eic_y_values have the same length.
    if len(eic_x_values) != len(eic_y_values):
        raise ValueError("eic_x_values and eic_y_values must be of the same length.")
   
    adjusted_peaks = []
    half_window = window // 2  # For an odd window, this gives the number of points on either side.
   
    for x in x_values:
        # Find the index in eic_x_values closest to the current x.
        idx = np.argmin(np.abs(eic_x_values - x))
       
        # Define the boundaries of the window around the index.
        start_idx = max(0, idx - half_window)
        end_idx = min(len(eic_x_values), idx + half_window + 1)
       
        # Extract the local window of EIC y-values and corresponding x-values.
        local_y = eic_y_values[start_idx:end_idx]
        local_x = eic_x_values[start_idx:end_idx]
       
        # Find the index of the maximum intensity within this window.
        local_max_idx = np.argmax(local_y)
       
        # The adjusted x_value is the eic_x_value with the maximum y in the window.
        adjusted_peak = local_x[local_max_idx]
        adjusted_peaks.append(adjusted_peak)
   
    return adjusted_peaks


   



   

   
def get_id_file_peaks(show_plots=False):  # default off to save memory
    global isobar_frames
    from matplotlib import pyplot as plt
   
    def _plot_trace(fr, file_name, x_raw, y_raw,
                    snapped_sec, center_sec, halfwin_sec, export_method):
        if x_raw.size == 0 or y_raw.max() == 0:
            return

        # x axis in minutes, y normalized to % of file max
        x_min  = x_raw / 60.0
        y_norm = (y_raw / y_raw.max()) * 100.0

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(x_min, y_norm, label=file_name, lw=1)

        # Draw snap windows (± halfwin around each pre-snap center)
        if center_sec is not None and np.size(center_sec) and halfwin_sec is not None:
            centers_min = np.asarray(center_sec, dtype=float) / 60.0
            half_min = float(halfwin_sec) / 60.0
            lefts  = centers_min - half_min
            rights = centers_min + half_min
            ymin, ymax = ax.get_ylim()
            xs = np.concatenate([lefts, rights])
            ax.vlines(xs, ymin, ymax, linestyles=':', alpha=0.7, label="snap window")

        snapped_sec = np.asarray(snapped_sec, dtype=float)
        ax.scatter(snapped_sec / 60.0,
                   np.interp(snapped_sec, x_raw, y_norm),
                   s=40, zorder=5, label="picked peaks")

        ax.set_xlabel("RT (min)")
        ax.set_ylabel("Intensity (% of file max)")
        ax.set_title(f"{fr.ID}  –  {file_name}")
        ax.grid(True)
        ax.legend(fontsize="small")

        if export_method == "plot":
            plt.show()
        elif export_method == "save":
            _save_temp_figure_svg(fig, f"{fr.ID}_{file_name}.svg")

    items = list(isobar_frames.items())
    for key1, fr in tqdm(items, total=len(items),
                         desc="Back‑warping peaks & filling *_RT_sec/_Abn", leave=True):
        if fr.hide or not hasattr(fr, 'Pol2dict'):
            continue

        try:
            # Common per‑frame vectors
            peaks  = fr.alignment_df['Reference_RT'].to_numpy(float)
            fwhms  = fr.alignment_df['fwhm'].to_numpy(float)
            if peaks.size == 0:
                continue

            for file_name, (coeffs, rmsd, minimum, maximum) in fr.Pol2dict.items():
                # Stream one trace from disk
                x_raw, y_raw = _load_trace(fr, file_name, raw=True)
                if x_raw is None or x_raw.size == 0:
                    continue

                # Reverse warp peaks into file space
                corr = apply_legendre_polynomial(
                    rescale_to_legendre_space(peaks, minimum, maximum), coeffs)
                peaks_trans = correct_retention_time(
                    np.clip(peaks, minimum, maximum), corr)

                # Snap to local maxima
                half_win_sec = float(settings.retention_time['peak_filter_seconds']) / 3.0
                peaks_adj = adjust_peaks(
                    peaks_trans, x_raw, y_raw,
                    window=int(half_win_sec)
                )

                # Fill columns (no copy of entire dict)
                col_rt  = f"{file_name}_RT_sec".replace('.mzML', "")
                col_abn = f"{file_name}_Abn".replace('.mzML', "")
                peaks_adj_arr = np.asarray(peaks_adj, dtype=float)

                # Best effort guard if lengths are off for any reason
                if peaks_adj_arr.size == len(fr.alignment_df):
                    fr.alignment_df[col_rt] = peaks_adj_arr
                else:
                    # Pre-fill with NaNs, then place what we have in order
                    fr.alignment_df[col_rt] = np.nan
                    fr.alignment_df.loc[fr.alignment_df.index[:peaks_adj_arr.size], col_rt] = peaks_adj_arr

                # Use the snapped positions for heights/areas (you already were)
                heights = np.interp(peaks_adj_arr, x_raw, y_raw)
                fr.alignment_df[col_abn] = heights * fwhms * np.sqrt(2*np.pi)

                if show_plots:
                    _plot_trace(
                        fr, file_name, x_raw, y_raw,
                        snapped_sec=np.asarray(peaks_adj),
                        center_sec=np.asarray(peaks_trans),
                        halfwin_sec=half_win_sec,
                        export_method=settings.export_method
                    )

                # Proactively free per‑file buffers
                del x_raw, y_raw, heights, peaks_adj, peaks_trans, corr
            gc.collect()

        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            if settings.verbose_exceptions:
                print(f"[{key1}] error at {tb.filename}:{tb.lineno} – {e}")
            fr.hide = True
            fr.reason_for_hiding = "exception during peak extraction"
           
           
           



def get_normalization_constants():
    global file_specific_info

    tics = file_specific_info.get('summed_eics', {})
    tic_medians = {}
    for key, tic in tqdm(tics.items(),
                         total=len(tics),
                         desc="Computing TIC medians",
                         leave=True):
        tic_medians[key] = (statistics.median(tic[:, 1]) if tic.size else 0)

    global_med = statistics.median(tic_medians.values()) if tic_medians else 0

    norm = {}
    for k, m in tqdm(tic_medians.items(),
                     total=len(tic_medians),
                     desc="Building normalization ratios",
                     leave=False):
        norm[k] = {'other': (m / global_med) if global_med else 0}

    file_specific_info['TIC_normalization_ratio'] = norm




def normalize():
    global isobar_frames, file_specific_info

    ratios = file_specific_info.get('TIC_normalization_ratio', {})

    items = list(isobar_frames.items())
    for key1, fr in tqdm(items,
                         total=len(items),
                         desc="Normalizing peak areas (TIC)",
                         leave=True):
        if fr.hide:
            continue
        if not hasattr(fr, 'Pol2dict'):
            continue
        for key2 in fr.Pol2dict:
            abn_col = f"{key2}_Abn".replace('.mzML', "")
            if abn_col not in fr.alignment_df.columns:
                continue
            norm_factor = ratios.get(key2, {}).get('other', 1) or 1
            fr.alignment_df[abn_col] = fr.alignment_df[abn_col] / norm_factor


# Global variables
sortby_attribute = "name"
descending = False

def set_sortby_attribute(event):
    global sortby_attribute
    sortby_attribute = sortby_combobox.get()


def toggle_descending():
    global descending
    descending = descending_var.get()

def check_metabolyte_type(event=None):
    global settings
    settings.metabolyte_type = datatype_var.get().lower()


def on_close(window_name):
    """Close handler that shuts down the program if any window except 'settings' is closed."""
    if window_name != "settings":
        root.quit()  # Gracefully stop main loop
        root.destroy()  # Destroy main window

if __name__ == "__main__":
    set_default_settings()

    multiprocessing.freeze_support()
    root = tk.Tk()
    root.title("isobar Frame Viewer")
    root.geometry("800x600")

    # Bind close event to terminate the app
    root.protocol("WM_DELETE_WINDOW", lambda: on_close("main"))

    # Frame to hold buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    # Load CSV Button
    load_button = tk.Button(button_frame, text="Load CSV", command=load_csv)
    load_button.pack(side=tk.LEFT, padx=10)

    # Automated Alignment Button
    align_button = tk.Button(button_frame, text="Automated Alignment", command=automated_alignment)
    align_button.pack(side=tk.LEFT, padx=10)

    # Separate window for "Save Remaining Data"
    save_window = tk.Toplevel(root)
    save_window.title("Save Data")
    save_window.geometry("200x100")
    save_window.protocol("WM_DELETE_WINDOW", lambda: on_close("save_data"))

    # Save Button
    save_button = tk.Button(save_window, text="Output Results", command=save_final_df)
    save_button.grid(row=0, column=0, pady=20, sticky="nsew")

    # Configure grid for centering
    save_window.grid_rowconfigure(0, weight=1)
    save_window.grid_columnconfigure(0, weight=1)

    # Option Selector (Dropdown Menu)
    sortby_label = tk.Label(root, text="Sort by:")
    sortby_label.pack()

    sortby_combobox = ttk.Combobox(root, values=["name", "adduct", "average_intensity", "ID"])
    sortby_combobox.current(0)
    sortby_combobox.bind("<<ComboboxSelected>>", set_sortby_attribute)
    sortby_combobox.pack(pady=10)

    # Checkbox for Descending Order
    descending_var = tk.BooleanVar()
    descending_checkbox = tk.Checkbutton(root, text="Descending order?", variable=descending_var, command=toggle_descending)
    descending_checkbox.pack(pady=10)
   
   
    # ──────────────────────────────────────────────────────────────
    # Data‑type selector (Proteomics vs. Lipidomics)
    # ──────────────────────────────────────────────────────────────
    datatype_label = tk.Label(root, text="Data type:")
    datatype_label.pack()
   
    datatype_var = tk.StringVar(value="lipid")        # defaulta
    datatype_combobox = ttk.Combobox(
            root,
            textvariable=datatype_var,
            values=["protein", "lipid"],
            state="readonly")
   
    datatype_combobox.pack(pady=10)
   
    datatype_combobox.bind("<<ComboboxSelected>>", check_metabolyte_type)
    check_metabolyte_type()


    # Menu Bar
    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Save Project", command=save_variables)
    file_menu.add_command(label="Load Project", command=load_variables)
    file_menu.add_command(label="Settings", command=settings_gui)
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menu_bar)

    root.mainloop()  # Start the GUI event loop