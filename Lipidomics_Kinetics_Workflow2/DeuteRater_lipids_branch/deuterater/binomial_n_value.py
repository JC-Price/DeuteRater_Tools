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

import os, re, sys,  math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import minimize, lsq_linear
from scipy.linalg import toeplitz
from scipy.special import gammaln
import multiprocessing as mp
from tqdm import tqdm
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional, Dict
import deuterater.settings as settings
import traceback
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Spacing (multi-moment) controls
MOMENT_WEIGHT_SCHEME= "variance"        # "variance" or "harmonic"
TAIL_WARN_THRESH    = 0.01              # (diagnostic placeholder)

#Cores to run with
CORES = 20


FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")



# ---------------- Isotope data (IUPAC--1997) ----------------
ELEMENT_ISOTOPES = {
    "H": [
        {"mass": 1.00782503223, "abundance": 99.9885},
        {"mass": 2.01410177812, "abundance": 0.0115},
    ],
    "C": [
        {"mass": 12.00000000000, "abundance": 98.93},
        {"mass": 13.00335483507, "abundance": 1.07},
    ],
    "N": [
        {"mass": 14.00307400443, "abundance": 99.632},
        {"mass": 15.00010889888, "abundance": 0.368},
    ],
    "O": [
        {"mass": 15.99491461957, "abundance": 99.757},
        {"mass": 16.99913175650, "abundance": 0.038},
        {"mass": 17.99915961286, "abundance": 0.205},
    ],
    "P": [{"mass": 30.97376199842, "abundance": 100.0}],
    "S": [
        {"mass": 31.9720711744, "abundance": 94.93},
        {"mass": 32.9714589098, "abundance": 0.76},
        {"mass": 33.967867004,  "abundance": 4.29},
        {"mass": 35.96708071,   "abundance": 0.02},
    ],
}

# --------------------------
# 1) Input file functions
# --------------------------
def normalize_abundances_if_missing(df, abund_col="abundances", new_col="normalized_empirical_abundances"):

    if new_col in df.columns:
        print(f"[Info] '{new_col}' already exists — skipping normalization.")
        return df

    def _parse_and_norm(cell):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return None
        if isinstance(cell, (list, tuple, np.ndarray)):
            arr = np.asarray(cell, dtype=float)
        else:
            nums = [float(m.group()) for m in FLOAT_RE.finditer(str(cell))]
            arr = np.asarray(nums, dtype=float)
        if arr.size == 0:
            return None
        s = np.nansum(arr)
        if not np.isfinite(s) or s <= 0:
            return None
        normed = arr / s
        return ",".join(f"{x:.6g}" for x in normed)

    df[new_col] = df[abund_col].apply(_parse_and_norm)
    print(f"[Added] '{new_col}' generated from '{abund_col}'.")
    return df

# ------------------------------------
# 3) Downsample the dataset if testing
# ------------------------------------
def downsample_unique_pairs(df: pd.DataFrame,
                            id_col: str = "Lipid Unique Identifier",
                            formula_col: str = "Adduct_cf",
                            frac: float = 0.10,
                            max_n: int = 100,
                            seed: int = 42) -> pd.DataFrame:
    unique_cols = [id_col, formula_col]
    if not all(c in df.columns for c in unique_cols):
        print(f"[Downsample] Skipped — one or more required columns missing: {unique_cols}")
        return df

    unique_keys = df[unique_cols].drop_duplicates()
    n_total = len(unique_keys)
    if n_total == 0:
        print("[Downsample] Skipped — no unique pairs found.")
        return df

    n_sample = min(max_n, max(1, int(frac * n_total)))
    rng = np.random.default_rng(seed)
    sampled_keys = unique_keys.sample(n=n_sample, random_state=rng.integers(0, 1e9))

    df_down = df.merge(sampled_keys, on=unique_cols, how="inner")
    print(f"[Downsample] Selected {n_sample} of {n_total} unique (Lipid × Adduct_cf) combinations → kept {len(df_down)} rows.")
    return df_down




# -----------------------------------------
# 4) Output directory + metadata accounting
# -----------------------------------------
def write_run_metadata(
    save_dir: str,
    cores: int,
    time_tol: float,
    require_all: bool,
    max_rate_cap: float,
    min_rate_cap: float,
    moment_weight_scheme: str,
    n_boot: int,
    boot_seed: int,
):
    run_info = {
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "CORES": cores,
        "TIME_TOL": time_tol,
        "REQUIRE_ALL": require_all,
        "MAX_RATE_CAP": max_rate_cap,
        "MIN_RATE_CAP": min_rate_cap,
        "MOMENT_WEIGHT_SCHEME": moment_weight_scheme,
        "BOOT_SAMPLES": n_boot,
        "BOOT_SEED": boot_seed,
    }

    meta_path = os.path.join(save_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"[Meta] Wrote run metadata → {meta_path}")



# ------------------------------------------------------------------------
# 5a) Data Preparation-- H-spectrum deconvolution (essential for BA fit)
# ------------------------------------------------------------------------
# Helpers for H-spectrum deconvolution, using IUPAC-based isotopes from
# resources/elements.tsv as the single source of truth for natural abundances.

import csv

from pathlib import Path

# repo_root/deuterater/your_script.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parents[1]

# repo_root/resources/elements.tsv
ELEMENTS_TSV = REPO_ROOT / "resources" / "elements.tsv"



def _load_isotopes(path: Path = ELEMENTS_TSV) -> dict:
    """
    Load isotopic data from resources/elements.tsv into a nested dict:
      ISOTOPES["C"][13]["abundance"] -> 0.0107, etc.
    Only rows with a non-empty isotopic_composition are kept.
    """
    iso = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            comp = (row.get("isotopic_composition") or "").strip()
            if not comp:
                # skip isotopes without a defined natural composition
                continue

            symbol = (row.get("isotope_letter") or "").strip()
            if not symbol:
                continue

            try:
                mass_number = int(row["isotope_suffix"])
                abundance = float(comp)  # already a fraction (0–1), not percent
                atomic_number = int(row["atomic_number"])
            except (KeyError, ValueError):
                continue

            entry = {
                "atomic_number": atomic_number,
                "mass_number": mass_number,
                "mass": float(row["relative_atomic_mass"]),
                "abundance": abundance,
                "standard_atomic_weight": (row.get("standard_atomic_weight") or "").strip() or None,
                "notes": (row.get("notes") or "").strip() or None,
            }
            iso.setdefault(symbol, {})[mass_number] = entry
    return iso


# Global cache of isotopes from the TSV
ISOTOPES = _load_isotopes()


def _iso_prob(symbol: str, mass_number: int, default: float = 0.0) -> float:
    """
    Return natural isotopic composition (as a probability) for a given isotope,
    e.g. _iso_prob("C", 13) -> 0.0107.
    Falls back to 'default' if the isotope is not found.
    """
    try:
        return ISOTOPES[symbol][mass_number]["abundance"]
    except KeyError:
        return default


def _iso_len_by_id(df, ID_COL, spec_col) -> dict:
    out = {}
    for uid, g in df.groupby(ID_COL):
        lens = []
        for s in g[spec_col].dropna().astype(str):
            parts = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            if parts:
                lens.append(len(parts))
        out[uid] = int(np.median(lens)) if lens else 3
    return out


def get_element_counts(formula: str):
    def grab(elem):
        m = re.search(rf"{elem}(\d+)", formula)
        return int(m.group(1)) if m else (1 if re.search(rf"{elem}(?![a-z])", formula) else 0)

    return grab("C"), grab("H"), grab("N"), grab("O"), grab("S")


def _conv_shift(base: np.ndarray, dist: np.ndarray, step: int) -> np.ndarray:
    L = len(base)
    out = np.zeros(L, dtype=float)
    for i in range(L):
        bi = base[i]
        if bi == 0.0:
            continue
        max_j = (L - 1 - i) // step
        out[i : i + step * (max_j + 1) : step] += bi * dist[: max(0, max_j) + 1]
    return out


def build_shift_matrix(dist, L, step=1):
    dist = np.asarray(dist, float)
    K = len(dist)

    C = np.zeros((L, L), dtype=float)

    for i in range(L):
        for k in range(K):
            j = i + k * step
            if j < L:
                C[j, i] += dist[k]

    return C


def _trunc_binom_prefix(n: int, p: float, K: int) -> np.ndarray:
    K = max(0, int(K))
    out = np.zeros(K + 1, dtype=float)
    q = 1.0 - p
    try:
        P0 = float(np.exp(n * np.log(q))) if q > 0 else (0.0 if n > 0 else 1.0)
    except OverflowError:
        P0 = 0.0
    out[0] = P0
    if q != 0.0:
        for i in range(1, K + 1):
            prev = out[i - 1]
            out[i] = prev * (n - (i - 1)) / i * (p / q)
    s = out.sum()
    if not np.isfinite(s) or s <= 0:
        lam = n * p
        out = np.array([math.exp(-lam) * (lam ** i) / math.factorial(i) for i in range(K + 1)], dtype=float)
        s = out.sum()
    out /= (s if s > 0 else 1.0)
    return out


def _kernel_CNOS_trunc(nC, nN, nO, nS, iso_len: int) -> np.ndarray:
    """
    Build the natural-abundance kernel for C/N/O/S based on IUPAC isotopic
    compositions from resources/elements.tsv.

    Heavy isotopes modeled:
      C:  13C (+1)
      N:  15N (+1)
      O:  17O (+1), 18O (+2)
      S:  33S (+1), 34S (+2)
    """
    # Heavy-isotope probabilities from TSV (fractions, not percents)
    pC   = _iso_prob("C", 13)   # 13C
    pN   = _iso_prob("N", 15)   # 15N
    pO17 = _iso_prob("O", 17)   # 17O (+1)
    pO18 = _iso_prob("O", 18)   # 18O (+2)
    pS33 = _iso_prob("S", 33)   # 33S (+1)
    pS34 = _iso_prob("S", 34)   # 34S (+2)

    L = int(iso_len)
    K1 = L - 1
    K2 = (L - 1) // 2

    k = np.zeros(L, dtype=float)
    k[0] = 1.0

    # +1 mass shifts
    if nC > 0 and pC   > 0.0:
        k = _conv_shift(k, _trunc_binom_prefix(nC, pC,   K1), 1)
    if nN > 0 and pN   > 0.0:
        k = _conv_shift(k, _trunc_binom_prefix(nN, pN,   K1), 1)
    if nO > 0 and pO17 > 0.0:
        k = _conv_shift(k, _trunc_binom_prefix(nO, pO17, K1), 1)
    if nS > 0 and pS33 > 0.0:
        k = _conv_shift(k, _trunc_binom_prefix(nS, pS33, K1), 1)

    # +2 mass shifts
    if nO > 0 and pO18 > 0.0 and K2 >= 0:
        k = _conv_shift(k, _trunc_binom_prefix(nO, pO18, K2), 2)
    if nS > 0 and pS34 > 0.0 and K2 >= 0:
        k = _conv_shift(k, _trunc_binom_prefix(nS, pS34, K2), 2)

    s = k.sum()
    k = k / (s if s > 0 else 1.0)
    return toeplitz(k, np.zeros_like(k))[:L, :L]


def _kernel_CNOS(cf: str, iso_len: int) -> np.ndarray:
    nC, nH, nN, nO, nS = get_element_counts(cf)
    return _kernel_CNOS_trunc(nC, nN, nO, nS, iso_len)


def _build_kernel_cache(df, iso_len_map, ID_COL, FORMULA_COL):
    cache = {}
    for uid, g in df.groupby(ID_COL):
        s = g[FORMULA_COL].dropna()
        if s.empty:
            continue
        cf = str(s.iloc[0])
        nC, nH, nN, nO, nS = get_element_counts(cf)
        iso_len = iso_len_map.get(uid, 3)
        cache[uid] = (_kernel_CNOS(cf, iso_len), int(nH))
    return cache


def parse_num_seq(cell, iso_len: int, pad_with=0.0) -> np.ndarray:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        arr = np.asarray([], float)
    elif isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=float)
    else:
        s = str(cell)
        nums = [float(m.group()) for m in FLOAT_RE.finditer(s)]
        arr = np.asarray(nums, dtype=float)

    if arr.size >= iso_len:
        out = arr[:iso_len]
    else:
        out = np.full(iso_len, pad_with, dtype=float)
        if arr.size:
            out[:arr.size] = arr
    return out


def parse_spectrum_cell(cell, iso_len: int):
    arr = parse_num_seq(cell, iso_len, pad_with=0.0)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    s = arr.sum()
    return (arr / s) if s > 0 else None
# End of H-spectrum deconvolution helpers


def deconvolve_hydrogen_spectrum(
    df: pd.DataFrame,
    ID_COL: str,
    spec_col: str,
    FORMULA_COL: str,
    MZ_COL: str = None,
    Z_COL: str = None,
) -> pd.DataFrame:
    """
    Per-row natural-isotope deconvolution to hydrogen-only D_spectrum.
    On any solver failure or bad solution → row is skipped (not included downstream).
    """
    iso_len_map = _iso_len_by_id(df, ID_COL, spec_col)
    kernel_cache = _build_kernel_cache(df, iso_len_map, ID_COL, FORMULA_COL)

    D_spectra, rss_list, ok_list, iso_lens, nHs = [], [], [], [], []
    mz_arrays, z_vals = [], []

    def _append_skip(default_iso_len=3, default_nH=1):
        D_spectra.append(None)
        rss_list.append(np.nan)
        ok_list.append(False)
        iso_lens.append(default_iso_len)
        nHs.append(default_nH)
        mz_arrays.append(None)
        z_vals.append(np.nan)

    for _, row in df.iterrows():
        uid = row.get(ID_COL)
        if uid not in kernel_cache:
            _append_skip()
            continue

        K, nH = kernel_cache[uid]
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1] or not np.all(np.isfinite(K)):
            _append_skip(default_nH=nH)
            continue

        iso_len = int(K.shape[0])
        if iso_len <= 0:
            _append_skip(default_nH=nH)
            continue

        arr = parse_spectrum_cell(row.get(spec_col), iso_len=iso_len)
        if arr is None or (not np.all(np.isfinite(arr))):
            _append_skip(default_iso_len=iso_len, default_nH=nH)
            continue

        try:
            res = lsq_linear(K, arr, bounds=(0.0, 1.0), method="bvls")
        except Exception:
            _append_skip(default_iso_len=iso_len, default_nH=nH)
            continue

        if (not res.success) or (res.x is None) or (not np.all(np.isfinite(res.x))):
            _append_skip(default_iso_len=iso_len, default_nH=nH)
            continue

        D = np.clip(res.x, 0.0, 1.0)
        s = float(D.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            _append_skip(default_iso_len=iso_len, default_nH=nH)
            continue

        D /= s
        D_spectra.append(D)
        rss_list.append(float(np.linalg.norm(K @ D - arr)))
        ok_list.append(True)
        iso_lens.append(iso_len)
        nHs.append(nH)

        if MZ_COL in df.columns:
            mz_arr = parse_num_seq(row.get(MZ_COL, None), iso_len, pad_with=np.nan)
        else:
            mz_arr = np.full(iso_len, np.nan)
        mz_arrays.append(mz_arr)

        z_vals.append(pd.to_numeric(row.get(Z_COL, np.nan), errors="coerce"))

    q = df.copy()
    q["D_spectrum"] = D_spectra
    q["deconv_rss"] = rss_list
    q["deconv_ok"] = ok_list
    q["iso_len"] = iso_lens
    q["nH"] = nHs
    q["mz_array"] = mz_arrays
    q[Z_COL] = z_vals

    before = len(q)
    q = q[q["deconv_ok"] & q["D_spectrum"].notna()].copy()
    print(f"[Step 1] kept {len(q)}/{before} rows with valid D-spectra")
    return q




# -------------------------------------------------------------------------------------------------
# 5b) Data Preparation-- collapsing analyte-specific time-information into a single informative row
# -------------------------------------------------------------------------------------------------

#Helpers for time-collapse
def _cluster_times(ts: np.ndarray, tol: float):
    idx = np.argsort(ts); ts = ts[idx]
    if len(idx)==0: return []
    clusters=[[idx[0]]]
    for j in range(1,len(idx)):
        if ts[j]-ts[j-1] <= tol: clusters[-1].append(idx[j])
        else: clusters.append([idx[j]])
    return clusters

def norm1(x, axis=None, eps=1e-12):
    s = np.sum(x, axis=axis, keepdims=True)
    s = np.where(np.isfinite(s), s, 0.0)
    return np.divide(x, np.where(s>eps, s, 1.0), out=np.zeros_like(x, dtype=float), where=True)
#End of helpers for time-collapse

def _init_flag_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Flag" not in df.columns:
        df["Flag"] = "PASS"
    return df

def _flag_reason(df: pd.DataFrame, mask: pd.Series, reason: str):
    mask = mask.fillna(False)
    if not mask.any():
        return
    cur = df.loc[mask, "Flag"].astype(str)
    df.loc[mask, "Flag"] = np.where(
        cur.str.startswith("PASS"), "FAIL: " + reason, cur + "; " + reason
    )





def collapse_D_by_time(dfD: pd.DataFrame, ID_COL: str, GROUP_COL: str, FORMULA_COL: str,
                       TIME_COL: str, Z_COL: str, tol: float) -> pd.DataFrame:
    """
    Collapse replicate D_spectra by time (within ±tol), preserving enrichment.

    CHANGE REQUESTED:
        signal_noise and abundances are now AVERAGED (like mz_array),
        not preserved as lists.
    """

    if "enrichment" not in dfD.columns:
        raise KeyError(
            "[Step 2] 'enrichment' column is required in dfD but was not found. "
            "Please merge or compute enrichment before running collapse_D_by_time()."
        )

    recs = []
    for (uid, grp, cf), g in dfD.groupby([ID_COL, GROUP_COL, FORMULA_COL]):

        tt = pd.to_numeric(g[TIME_COL], errors="coerce").to_numpy(float)
        ok = np.isfinite(tt)
        if not ok.any():
            continue

        g = g.iloc[np.where(ok)[0]]
        tt = tt[np.where(ok)[0]]

        clusters = _cluster_times(tt, tol)

        for cl in clusters:
            gg = g.iloc[cl]

            # ----- COLLAPSE D_spectrum -----
            Ds = np.vstack([d for d in gg["D_spectrum"].to_numpy()
                            if isinstance(d, np.ndarray)])
            if Ds.size == 0:
                continue
            mean_D = norm1(Ds.mean(axis=0))

            # ----- enrichment (required) -----
            enrich_vals = pd.to_numeric(gg["enrichment"], errors="coerce").to_numpy(float)
            ok_en = np.isfinite(enrich_vals)
            if not ok_en.any():
                print(f"[Step 2] Skipping (ID={uid}, group={grp}, cf={cf}) — no finite enrichment.")
                continue
            mean_enrich = float(enrich_vals[ok_en].mean())

            # ----- collapsed metadata -----
            rep_time = float(np.nanmean(tt[cl]))
            iso_len  = int(np.median(pd.to_numeric(gg["iso_len"], errors="coerce")))
            nH       = int(np.median(pd.to_numeric(gg["nH"], errors="coerce")))

            # ----- COLLAPSE mz_array -----
            mz_stack = np.vstack([
                parse_num_seq(v, iso_len, pad_with=np.nan)
                for v in gg["mz_array"].to_numpy()
            ])
            mean_mz = np.nanmean(mz_stack, axis=0)

            z_med = float(np.nanmedian(pd.to_numeric(gg[Z_COL], errors="coerce")))

            # =========================================================
            # KEY CHANGE: AVERAGE noise and abundances (numeric)
            # =========================================================
            import ast

            # Convert each entry into a numeric vector
            sn_vals = []
            for v in gg["signal_noise"]:
                x = v
                while isinstance(x, (list, tuple)) and len(x) == 1:
                    x = x[0]
                if isinstance(x, str):
                    x = ast.literal_eval(x)
                sn_vals.append(np.array(x, float))
            mean_sn = np.mean(np.vstack(sn_vals), axis=0)

            abn_vals = []
            for v in gg["abundances"]:
                x = v
                while isinstance(x, (list, tuple)) and len(x) == 1:
                    x = x[0]
                if isinstance(x, str):
                    x = ast.literal_eval(x)
                abn_vals.append(np.array(x, float))
            mean_abn = np.mean(np.vstack(abn_vals), axis=0)
            # =========================================================

            # Build output row
            row = [
                uid, grp, cf, rep_time,
                mean_D, mean_enrich,
                iso_len, nH,
                mean_mz, z_med
            ]
            row.append(mean_sn.tolist())
            row.append(mean_abn.tolist())

            recs.append(row)

    # Build column list dynamically
    cols = [
        ID_COL, GROUP_COL, FORMULA_COL, TIME_COL,
        "D_spectrum", "enrichment", "iso_len", "nH",
        "mz_array", Z_COL,
        "signal_noise", "abundances"
    ]

    collapsed = pd.DataFrame(recs, columns=cols)

    # Add TP_count (count of timepoints per ID×group×formula)
    tp_counts = (
        collapsed
        .groupby([ID_COL, GROUP_COL, FORMULA_COL])[TIME_COL]
        .transform("nunique")
        .astype(int)
    )
    collapsed["TP_count"] = tp_counts

    print(f"[Step 2] collapsed → {len(collapsed)} rows (Averaged noise & abundances)")
    return collapsed

# ----------------------------------------------
# 7) Binomial Abundance fit
# ----------------------------------------------
#Joint helpers for Abundance and Spacing Fits
def _parallel_fit(f, args_list, n_cores=None, pool=None, desc="Fitting"):
    """
    Parallel fit helper that can reuse an existing multiprocessing pool.
    If a pool is passed, it uses it directly; otherwise, it creates one.
    """
    results = []
    if pool is not None:
        for out in tqdm(pool.imap_unordered(f, args_list),
                        total=len(args_list), desc=desc):
            results.append(out)
    elif n_cores and n_cores > 1:
        try:
            with mp.Pool(n_cores) as p:
                for out in tqdm(p.imap_unordered(f, args_list),
                                total=len(args_list), desc=desc):
                    results.append(out)
        except Exception as e:
            print("[WARN] Multiprocessing fallback:", e)
            for a in tqdm(args_list, desc="Single-core fallback"):
                results.append(f(a))
    else:
        for a in tqdm(args_list, desc="Single-core"):
            results.append(f(a))
    return results


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\.-]+", "_", str(s))

def fractional_binom_pmf(i, nL, p):
    """
    Stable fractional binomial PMF:
      C(nL, i) * p^i * (1-p)^(nL-i), with non-integer nL allowed.
    Uses log-gamma to avoid over/underflow.
    """
    if not (np.isfinite(nL)) or nL <= 0 or i > nL:
        return 0.0
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    log_c = gammaln(nL + 1.0) - gammaln(i + 1.0) - gammaln(nL - i + 1.0)
    log_pmf = log_c + i * np.log(p) + (nL - i) * np.log1p(-p)
    return float(np.exp(log_pmf))

def _build_model_rows(nL, k, Asyn, times, iso_len, P_samples):
    P_samples = np.asarray(P_samples, dtype=float)
    p = np.clip(P_samples, 1e-12, 1.0 - 1e-12)
    rows = []
    for t, p_t in zip(times, p):
        binom_vec_t = np.array([fractional_binom_pmf(i, nL, p_t) for i in range(iso_len)], dtype=float)
        binom_vec_t /= (binom_vec_t.sum() + 1e-12)
        f_new = Asyn * (1.0 - np.exp(-k * t))
        f_old = 1.0 - f_new
        dist = f_old * np.eye(1, iso_len, 0)[0] + f_new * binom_vec_t
        rows.append(dist)
    sim = np.vstack(rows)
    sim = sim / (sim.sum(axis=1, keepdims=True) + 1e-12)
    return sim



def _predict_centroid_shift(nL, k, A, times, p_t, z):
    deltaH = 1.0063 / z
    p_t = np.asarray(p_t, float)
    return deltaH * nL * A * (1.0 - np.exp(-k * np.asarray(times, float))) * p_t




def plot_fit_vs_time(times, obs, sim=None, analyte_label="", params=None,
                     save_path=None, P_t=None, iso_len=None, tag=None):
    import numpy as np
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    obs   = np.asarray(obs, dtype=float)
    sim   = None if sim is None else np.asarray(sim, dtype=float)

    # --- Build sim if not provided ---
    if sim is None:
        if params is None or P_t is None or iso_len is None:
            raise ValueError("When `sim` is None, provide params=(nL,k,A), P_t, and iso_len.")
        nL, k, A = map(float, params)
        sim = _build_model_rows(nL, k, A, times, int(iso_len), np.asarray(P_t, dtype=float))

    # --- Normalize to show the fitted quantity ---
    if tag == "BS":
        # Use SAME spacing definition as the SPACING FIT:
        # Δ(Mi – M0) for i ≥ 1
        obs_shift = obs[:, 1:] - obs[:, [0]]
        sim_shift = sim[:, 1:] - sim[:, [0]]

        # Baseline-subtract relative to time zero
        obs_shift = obs_shift - obs_shift[0:1, :]
        sim_shift = sim_shift - sim_shift[0:1, :]

        ylabel = "Δ(m/z spacing) from t0"
    else:
        obs_shift = obs
        sim_shift = sim
        ylabel = "Relative isotopomer abundance"

    # --- Plot ---
    iso_len_eff = obs_shift.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, iso_len_eff))

    plt.figure(figsize=(7, 5))
    for i in range(iso_len_eff):
        if tag == "BS":
            m = i + 1
            obs_lbl = f"Obs Δ(M{m}-M{m-1})"
            fit_lbl = f"Fit Δ(M{m}-M{m-1})"
        else:
            obs_lbl = f"Obs M{i}"
            fit_lbl = f"Fit M{i}"

        plt.plot(times, obs_shift[:, i], "o-", color=colors[i], label=obs_lbl)
        plt.plot(times, sim_shift[:, i], "--", color=colors[i], alpha=0.7, label=fit_lbl)

    plt.xlabel("Time (days)")
    plt.ylabel(ylabel)

    if params is not None and len(params) == 3:
        nL, k, A = params
        ttl = f"{analyte_label} [{tag}]\n nL={float(nL):.2f}, k={float(k):.3f}, Asyn={float(A):.3f}"
    else:
        ttl = analyte_label

    plt.title(ttl)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()



def _save_fit_plot(
    uid,
    grp,
    plot_dir,
    prefix,
    times,
    obs,
    pred,
    params
):
    """
    Save spacing-fit plot:
    Observed vs predicted Δm/z spacing over time.
    """

    label = f"{uid} / {grp}"
    out_fn = os.path.join(
        plot_dir,
        f"fit_{safe_filename(label)}.png"
    )

    plot_fit_vs_time(
        times=times,
        obs=obs,
        sim=pred,
        analyte_label=label,
        params=params,
        save_path=out_fn,
        tag=prefix     # should be "BS" for spacing-fit
    )


from scipy.stats import norm
from scipy.optimize import curve_fit

from scipy.optimize import curve_fit


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def plot_3panel_boot_hist(
    boot_samples,
    analyte_label=None,
    save_path=None,
    bins=40,
    n_kept=None,
    prefix="BA",
    df_target=None,
    uid=None,
):
    """
    Plots resampling histograms with fitted Gaussian overlays.

    Returns
    -------
    r2_dict : dict(str -> float)
        e.g. { 'BA_nL_R2': 0.92, 'BA_rate_R2': 0.85, ... }
    good_flag : bool
        True if histograms AND Gaussian fits were OK.
        False → automatic QC_FAIL upstream.
    """

    # If no bootstrap samples: return clean failure
    if boot_samples is None or len(boot_samples) == 0:
        return {
            f"{prefix}_nL_R2": np.nan,
            f"{prefix}_rate_R2": np.nan,
            f"{prefix}_Asyn_R2": np.nan
        }, False

    # Ensure array shape
    try:
        boot = np.asarray(boot_samples, float)
        if boot.ndim != 2 or boot.shape[1] < 3:
            return {
                f"{prefix}_nL_R2": np.nan,
                f"{prefix}_rate_R2": np.nan,
                f"{prefix}_Asyn_R2": np.nan
            }, False
    except Exception:
        return {
            f"{prefix}_nL_R2": np.nan,
            f"{prefix}_rate_R2": np.nan,
            f"{prefix}_Asyn_R2": np.nan
        }, False

    # Prepare figure
    try:
        fig, axes = plt.subplots(3, 1, figsize=(6, 9))
        plt.subplots_adjust(hspace=0.45)
    except Exception:
        return {
            f"{prefix}_nL_R2": np.nan,
            f"{prefix}_rate_R2": np.nan,
            f"{prefix}_Asyn_R2": np.nan
        }, False

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    labels = [
        (f"{prefix}_nL", 0),
        (f"{prefix}_rate", 1),
        (f"{prefix}_Asyn", 2),
    ]

    r2_dict = {}
    all_good = True   # ← This controls QC keep/drop

    for idx, (ax, (lab, col_idx)) in enumerate(zip(axes, labels)):
        try:
            data = boot[:, col_idx]
            data = data[np.isfinite(data)]
            if data.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                r2_dict[lab + "_R2"] = np.nan
                all_good = False
                continue

            # Histogram
            try:
                counts, bins_edges, _ = ax.hist(
                    data, bins=bins, density=True,
                    alpha=0.85, color="steelblue"
                )
            except Exception:
                r2_dict[lab + "_R2"] = np.nan
                all_good = False
                continue

            centers = (bins_edges[:-1] + bins_edges[1:]) / 2

            amp_guess = counts.max()
            mu_guess = np.mean(data)
            sigma_guess = np.std(data, ddof=1)

            # Gaussian fit
            try:
                popt, _ = curve_fit(
                    gaussian,
                    centers,
                    counts,
                    p0=[amp_guess, mu_guess, sigma_guess],
                    maxfev=800
                )
                y_fit = gaussian(centers, *popt)

                ss_res = np.sum((counts - y_fit)**2)
                ss_tot = np.sum((counts - np.mean(counts))**2)
                R2_gauss = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

                r2_dict[lab + "_R2"] = R2_gauss
                ax.plot(centers, y_fit, "r-", lw=1.6)

                if not np.isfinite(R2_gauss):
                    all_good = False

            except Exception:
                # Fit failed
                r2_dict[lab + "_R2"] = np.nan
                all_good = False
                ax.text(0.5, 0.5, "fit failed", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Metadata on plot
            med = float(np.median(data))
            lo, hi = np.percentile(data, [2.5, 97.5])
            kept = n_kept[idx] if (n_kept and idx < len(n_kept)) else len(data)

            ax.axvline(med, color="black", linestyle="--", lw=1)
            ax.axvline(lo, color="gray", linestyle=":", lw=1)
            ax.axvline(hi, color="gray", linestyle=":", lw=1)

            ax.text(
                0.98, 0.95,
                f"med={med:.3g}\n95%CI=[{lo:.3g},{hi:.3g}]\nn={kept}\nR2={R2_gauss:.3f}",
                ha="right", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8)
            )

        except Exception:
            r2_dict[lab + "_R2"] = np.nan
            all_good = False
            continue

    # Save figure (optional)
    try:
        if save_path is not None:
            fig.savefig(save_path, dpi=120)
        plt.close(fig)
    except Exception:
        pass

    return r2_dict, all_good



def _save_boot_plot(uid, grp, cf, boot, boot_dir, prefix):
    label = f"{uid}_{grp}_{cf}"
    out_fn = os.path.join(boot_dir, f"boot_hist_{safe_filename(label)}.png")
    r2_vals =plot_3panel_boot_hist(boot, analyte_label=label, save_path=out_fn, prefix=prefix)
    return r2_vals



def _density_trim(x, bins=40, frac_cut=0.10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return x
    hist, edges = np.histogram(x, bins=bins, density=True)
    if hist.size == 0 or np.all(hist == 0): return x
    cutoff = frac_cut * hist.max()
    mask_bins = hist >= cutoff
    if not mask_bins.any(): return x
    x_min = edges[np.argmax(mask_bins)]
    x_max = edges[::-1][np.argmax(mask_bins[::-1])]
    return x[(x >= x_min) & (x <= x_max)]

def _median_range_trim(x, frac_width=0.25):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return x
    med = np.median(x); rng = np.nanmax(x) - np.nanmin(x)
    if not np.isfinite(rng) or rng == 0: return x
    lo, hi = med - frac_width*rng, med + frac_width*rng
    return x[(x >= lo) & (x <= hi)]




def boundary_penalty(x, L, U):
    mid  = 0.5 * (L + U)
    half = 0.5 * (U - L)
    if half <= 0:
        return 0.0
    return max(0.0, ((x - L)*(U - x)) / (half**2))



def boundary_penalty(x, L, U):
    """
    Vectorized quadratic penalty that is zero at the boundaries (L, U)
    and positive inside. Works for scalars or arrays.
    """
    x = np.asarray(x, float)
    L = np.asarray(L, float)
    U = np.asarray(U, float)

    mid  = 0.5 * (L + U)
    half = 0.5 * (U - L)

    # Avoid division-by-zero
    half = np.maximum(half, 1e-12)

    val = ((x - L) * (U - x)) / (half ** 2)

    return np.maximum(val, 0.0)



import numpy as np
import re

# ---------- CI PARSER ----------

_ci_re = r"\[\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\]"

def parse_ci95_to_arrays(ci_series):
    """Parse CI strings like '[0.382, 0.538]' into lo[], hi[]."""
    s = ci_series.astype(str)
    m = s.str.extract(_ci_re)
    lo = m[0].to_numpy(float)
    hi = m[1].to_numpy(float)
    return lo, hi


# ---------- NEW BOUNDARY PENALTY USING CI ----------

def boundary_penalty_from_ci(theta, ci_lo, ci_hi, Lacc, Uacc, eps=1e-12):
    """
    Fraction of CI margin available before hitting nearest acceptable boundary:
         B = cushion / CI_margin
    Then clamped so B ≤ 1 (no penalty if ≥ one CI-margin away).
    """

    theta = np.asarray(theta, float)
    ci_lo = np.asarray(ci_lo, float)
    ci_hi = np.asarray(ci_hi, float)
    Lacc  = np.asarray(Lacc, float)
    Uacc  = np.asarray(Uacc, float)

    # CI margins around theta
    mL = np.maximum(theta - ci_lo, eps)   # margin to lower CI
    mU = np.maximum(ci_hi - theta, eps)   # margin to upper CI

    # Cushions to acceptable boundaries (negative means outside → clamp to 0)
    dL = theta - Lacc
    dU = Uacc - theta
    cushion = np.maximum(np.minimum(dL, dU), 0.0)

    # Direction to nearest boundary determines which CI side to use
    use_lower = dL <= dU
    margin = np.where(use_lower, mL, mU)

    B = cushion / margin  # fractional cushion
    B = np.maximum(B, 0.0)
    B = np.minimum(B, 1.0)  # <-- clamp to 1

    return np.maximum(B, eps)


# ---------- MAIN FUNCTION ----------




#End of joint helpers for Abundance and Spacing Fits

# ----------------------------------------------
# 7a) Binomial Abundance fit
# ----------------------------------------------
#Abundance-specific helpers



from scipy.stats import norm

def _weighted_rmsd(obs, sim, noise_arr, empiricals):
    """
    obs, sim: (n_time, iso_len)
    noise_arr: (n_time, iso_len) averaged noise
    empiricals: (n_time, iso_len) averaged abundances
    """
    # Convert inputs to float arrays
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    noise_arr = np.asarray(noise_arr, float)
    empiricals = np.asarray(empiricals, float)

    # Safety: guard zero or negative noise values
    safe_noise = np.where(noise_arr <= 0, 1e-12, noise_arr)

    # Compute SNR weights
    w = empiricals / safe_noise     # shape matches obs/sim

    resid = obs - sim

    # Weighted RMSD
    return float(np.sqrt(np.sum(w * resid**2)))



def simple_r2_score(y_true, y_pred):
    """
    Minimal replacement for sklearn.metrics.r2_score
    Works on any 1D numpy arrays (ignores NaNs).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan






def compute_boot_stats(
    boot: np.ndarray,
    trim_bins: int = 40,
    density_frac: float = 0.30,
    median_frac_width: float = 0.25,
):
    """
    Standardized resampling pipeline: trim → SD/CI → Gaussian R².

    NOTE
    ----
    Returns *bootstrap standard deviation* (SD): the spread of the bootstrap
    distribution (this matches your prior behavior, now named correctly).

    Returns
    -------
    boot_trimmed : np.ndarray
        Rectangular (N_kept_max × P) array padded with NaN.
    sd : np.ndarray
        Sample standard deviation per parameter column (ddof=1).
    ci : list[list[float, float]]
        95% CI per column as [lo, hi].
    n_kept_cols : list[int]
        Number of retained bootstrap samples per column after trimming.
    r2_gauss : np.ndarray
        R² of Gaussian PDF fit to trimmed histogram for each column.
    """
    boot = np.asarray(boot, float)
    if boot.ndim != 2 or boot.shape[1] < 3:
        return (
            np.full((1, 3), np.nan),
            np.full(3, np.nan),
            [[np.nan, np.nan]] * 3,
            [0, 0, 0],
            np.full(3, np.nan),
        )

    cols_trim: list[np.ndarray] = []
    n_kept_cols: list[int] = []
    r2_gauss: list[float] = []

    P = boot.shape[1]

    for j in range(P):
        x = boot[:, j]
        x = x[np.isfinite(x)]

        if x.size == 0:
            cols_trim.append(np.array([np.nan], float))
            n_kept_cols.append(0)
            r2_gauss.append(np.nan)
            continue

        # ---- density trimming ----
        hist, edges = np.histogram(x, bins=trim_bins, density=True)
        if hist.size and np.any(hist > 0):
            cutoff = density_frac * hist.max()
            mask_bins = hist >= cutoff
            if mask_bins.any():
                x_min = edges[np.argmax(mask_bins)]
                x_max = edges[::-1][np.argmax(mask_bins[::-1])]
                x = x[(x >= x_min) & (x <= x_max)]

        # ---- median-range trimming ----
        if x.size:
            med = np.median(x)
            rng = np.nanmax(x) - np.nanmin(x)
            if np.isfinite(rng) and rng > 0:
                lo_, hi_ = med - median_frac_width * rng, med + median_frac_width * rng
                x = x[(x >= lo_) & (x <= hi_)]

        # ✅ ALWAYS record trimmed samples
        cols_trim.append(x if x.size else np.array([np.nan], float))
        n_kept_cols.append(int(np.sum(np.isfinite(cols_trim[-1]))))

        # ---- Gaussian R² (optional diagnostic) ----
        if x.size >= 10 and np.nanstd(x) > 0:
            mu = float(np.nanmean(x))
            sd_local = float(np.nanstd(x, ddof=1))
            y_obs, bins = np.histogram(x, bins=trim_bins, density=True)
            centers = (bins[:-1] + bins[1:]) / 2
            y_pred = norm.pdf(centers, mu, sd_local)
            r2_gauss.append(simple_r2_score(y_obs, y_pred))
        else:
            r2_gauss.append(np.nan)

    # ✅ pad to rectangle AFTER looping over all columns
    max_len = max(len(c) for c in cols_trim) if cols_trim else 0
    if max_len:
        boot_trimmed = np.column_stack([
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in cols_trim
        ])
    else:
        boot_trimmed = np.full((1, P), np.nan)

    # Bootstrap SD (ddof=1)
    sd = np.nanstd(boot_trimmed, axis=0, ddof=1)

    # 95% CIs
    lo = np.nanpercentile(boot_trimmed, 2.5, axis=0)
    hi = np.nanpercentile(boot_trimmed, 97.5, axis=0)
    ci = [[float(f"{l:.3g}"), float(f"{h:.3g}")] for l, h in zip(lo, hi)]

    return boot_trimmed, sd, ci, n_kept_cols, np.asarray(r2_gauss, float)


def _r2_abundance(
    obs: np.ndarray,
    sim: np.ndarray,
    w2d: Optional[np.ndarray] = None,
):
    """
    R² in intensity space over all timepoints × isotopomers.
    If w2d (T×L) is provided, computes a weighted R² consistent with the BA loss.
    """
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if w2d is not None:
        w = np.asarray(w2d, float)
        if w.shape != obs.shape:
            # broadcast weights per-row if only per-isotopomer weights were given
            if w.ndim == 1 and w.size == obs.shape[1]:
                w = np.broadcast_to(w.reshape(1, -1), obs.shape)
            else:
                raise ValueError("w2d shape must match obs/sim or be length L.")
        w = np.where(mask, w, 0.0)
        y = np.where(mask, obs, 0.0)
        yhat = np.where(mask, sim, 0.0)
        # weighted mean of y
        ybar = np.sum(w * y) / (np.sum(w) + 1e-12)
        ss_res = float(np.sum(w * (y - yhat)**2))
        ss_tot = float(np.sum(w * (y - ybar)**2) + 1e-12)
    else:
        y = obs[mask]; yhat = sim[mask]
        ybar = float(np.mean(y)) if y.size else 0.0
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - ybar)**2) + 1e-12)
    return float(1.0 - ss_res / ss_tot)





def fit_one_pair_D(
    sub: pd.DataFrame,
    nH_max: int,
    asymptote_mode: str,
    user_Asyn_value: Optional[float] = None,
    n_boot: int = 1000,
    TIME_COL=None,
    BOOT_SEED=None
):


    try:
        # --------------------------
        # Extract core arrays
        # --------------------------
        times = sub[TIME_COL].to_numpy(float)
        obs   = np.vstack(sub["D_spectrum"].to_numpy())
        P_t   = sub["enrichment"].to_numpy(float)

        iso_len = obs.shape[1]

        noises = np.vstack(sub["signal_noise"].apply(lambda x: np.array(x, float)))
        empiricals = np.vstack(sub["abundances"].apply(lambda x: np.array(x, float)))


        # ==================================================
        # 1) Optimization (L-BFGS-B)
        # ==================================================
        if asymptote_mode == "fixed" and user_Asyn_value is not None:

            def loss(x):
                nL, k = x
                if not (0.1 <= nL <= nH_max and 0.01 <= k <= 5.0):
                    return np.inf
                sim = _build_model_rows(nL, k, user_Asyn_value, times, iso_len, P_t)
                return _weighted_rmsd(obs, sim, noises, empiricals)

            x0 = [max(0.5, min(10.0, nH_max / 2)), 0.3]
            bounds = [(0.1, nH_max), (0.01, 5.0)]

            res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
            best = (res.x[0], res.x[1], user_Asyn_value)

        else:
            # Variable Asymptote
            def loss(x):
                nL, k, A = x
                if not (0.1 <= nL <= nH_max and 0.01 <= k <= 5.0 and 0.0 <= A <= 1.0):
                    return np.inf
                sim = _build_model_rows(nL, k, A, times, iso_len, P_t)
                return _weighted_rmsd(obs, sim, noises, empiricals)

            x0 = [max(0.5, min(10.0, nH_max / 2)), 0.3, 0.8]
            bounds = [(0.1, nH_max), (0.01, 5.0), (0.0, 1.0)]

            res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
            best = tuple(res.x.tolist())

        # ==================================================
        # 2) Bootstrap Resampling
        # ==================================================
        rng = np.random.default_rng(BOOT_SEED)
        boot = []
        n_t = len(times)

        for _ in range(n_boot):

            idx = rng.integers(0, n_t, size=n_t)

            tb = times[idx]
            ob = obs[idx]
            Pb = P_t[idx]

            nb = noises[idx]   # <-- correct weight resampling
            eb = empiricals[idx]      # <-- correct weight resampling

            if asymptote_mode == "fixed" and user_Asyn_value is not None:

                def loss_b(x):
                    nL, k = x
                    A = user_Asyn_value
                    sim = _build_model_rows(nL, k, A, tb, iso_len, Pb)
                    return _weighted_rmsd(ob, sim, nb, eb)

                rb = minimize(
                    loss_b,
                    x0=[best[0], best[1]],
                    bounds=[(0.1, nH_max), (0.01, 5.0)],
                    method="L-BFGS-B"
                ).x

                boot.append((rb[0], rb[1], user_Asyn_value))

            else:
                def loss_b(x):
                    nL, k, A = x
                    sim = _build_model_rows(nL, k, A, tb, iso_len, Pb)
                    return _weighted_rmsd(ob, sim, nb, eb)

                rb = minimize(
                    loss_b,
                    x0=[best[0], best[1], best[2]],
                    bounds=[(0.1, nH_max), (0.01, 5.0), (0.0, 1.0)],
                    method="L-BFGS-B"
                ).x

                boot.append((rb[0], rb[1], rb[2]))

        boot = np.asarray(boot, float)

        # ==================================================
        # 3) Trim → SE/CI → Gaussian R²
        # ==================================================

        boot_trimmed, sd, ci, n_kept_cols, r2_gauss = compute_boot_stats(
            boot,
            trim_bins=int(getattr(settings, "binomial_boot_trim_bins", 40)),
            density_frac=float(getattr(settings, "binomial_boot_density_frac", 0.30)),
            median_frac_width=float(getattr(settings, "binomial_boot_median_frac_width", 0.25)),
        )

        # ==================================================
        # 4) BA R² (unweighted intensity-space)
        # ==================================================
        sim_best = _build_model_rows(best[0], best[1], best[2], times, iso_len, P_t)
        BA_R2 = _r2_abundance(obs, sim_best)


    except Exception as e:
        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.max_colwidth', None,
                'display.width', 2000
        ):
            print("\n===== ERROR IN fit_one_pair_D =====")
            print(sub)
            print("\nException:", e)
            print("\nTraceback:")
            traceback.print_exc()
            print("===================================\n")

    # ==================================================
    # 5) Return
    # ==================================================
    return {
        "BA_nL": best[0],
        "BA_rate": best[1],
        "BA_Asyn": best[2],
        "BA_resampling_samples": boot,
        "BA_n_kept_boot": n_kept_cols,
        "BA_nL_SD": sd[0],
        "BA_rate_SD": sd[1],
        "BA_Asyn_SD": sd[2],
        "BA_nL_CI95": ci[0],
        "BA_rate_CI95": ci[1],
        "BA_Asyn_CI95": ci[2],
        "BA_fit_R2": BA_R2,
        "BA_resampling_trimmed": boot_trimmed,
        "signal_noise": sub["signal_noise"].iloc[0]
    }






def _fit_abundance_one(args):
    (ID_COL, GROUP_COL, FORMULA_COL, TIME_COL, BOOT_SEED,
     lipid_uid, sample_group, adduct_cf, sub_df,
     asymptote_mode, user_Asyn_value, n_boot) = args


    # Determine the maximum number of exchangeable hydrogens (per analyte)
    nH_vals = pd.to_numeric(sub_df.get("nH", pd.Series([1])), errors="coerce")
    nH_base = int(max(1, np.nanmax(nH_vals)))

    # Widen BA bound by +10%
    import math
    nH_max = float(math.ceil(nH_base * 1.10))  # use ceil to actually grow small values

    # Perform fit with user-defined Asyn handling
    out = fit_one_pair_D(sub_df, nH_max, asymptote_mode, user_Asyn_value, n_boot=n_boot, TIME_COL=TIME_COL, BOOT_SEED = None)
    if out is None:
        return None

    # Append identifying metadata
    out.update({
        ID_COL: lipid_uid,
        GROUP_COL: sample_group,
        FORMULA_COL: adduct_cf
    })
    return out

#End of abundance-specific helpers

def perform_abundance_fit(
    collapsed: pd.DataFrame,
    ID_COL: str,
    asymptote_mode: str,
    user_Asyn_value: Optional[float] = None,
    n_boot: int = 1000,
    tp_min: int = 3,
    cores = 2,
    save_dir: Optional[str] = None,
    pool=None,
    FORMULA_COL:str = None, TP_MIN: int = None,
        GROUP_COL:str = None, TIME_COL = None, BOOT_SEED = None
) -> pd.DataFrame:
    """Parallel Binomial-Abundance fit (BA_nL, BA_rate, BA_Asyn)."""

    prefix = "BA"

    # -----------------------
    # Pre-filter flagged rows
    # -----------------------
    if "Flag" in collapsed.columns:
        before = len(collapsed)
        collapsed = collapsed.loc[collapsed["Flag"].astype(str).str.startswith("PASS")].copy()
        print(f"[{prefix}] Skipping flagged rows → kept {len(collapsed)}/{before} PASS")
    else:
        print(f"[{prefix}] No 'Flag' column found — fitting all rows.")

    if collapsed.empty:
        print(f"[{prefix}] No unflagged rows available — skipping.")
        return pd.DataFrame()

    # -----------------------
    # Additional molecule filters before fitting
    # -----------------------
    keep_mask = (
        (collapsed["Flag"].astype(str) == "PASS")
        & collapsed["D_spectrum"].notna()
        & (pd.to_numeric(collapsed.get("TP_count", 0), errors="coerce") >= tp_min)
    )

    before_filter = len(collapsed)
    collapsed = collapsed.loc[keep_mask].copy()
    print(f"[{prefix}] Filtered analytes before fitting → kept {len(collapsed)}/{before_filter} "
          f"(Flag=PASS, TP_count≥{TP_MIN}, valid D_spectrum)")

    if collapsed.empty:
        print(f"[{prefix}] No analytes passed pre-fit filters — skipping {prefix} fitting.")
        return pd.DataFrame()

    # -----------------------
    # Prepare output directories
    # -----------------------
    plot_dir = os.path.join(save_dir, f"{prefix}_fit_plots")
    boot_dir = os.path.join(save_dir, f"{prefix}_resampling_plots")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(boot_dir, exist_ok=True)

    # -----------------------
    # Build argument list for parallel fitting
    # -----------------------
    groups = list(collapsed.groupby([ID_COL, GROUP_COL, FORMULA_COL], sort=False))
    args_list = [
        (ID_COL, GROUP_COL, FORMULA_COL, TIME_COL, BOOT_SEED, uid, grp, cf, sub,
         asymptote_mode, user_Asyn_value, n_boot)
        for (uid, grp, cf), sub in groups
    ]

    total = len(args_list)
    print(f"[{prefix}] Fitting {total} analyte-group combinations ({cores} cores)")

    # -----------------------
    # Parallel fitting
    # -----------------------
    results = _parallel_fit(
        _fit_abundance_one,
        args_list,
        n_cores=cores,
        pool=pool,
        desc=f"{prefix} fitting"
    )

    rows = [r for r in results if r is not None]
    df_fit = pd.DataFrame(rows)

    if df_fit.empty:
        print(f"[{prefix}] No successful abundance fits — skipping plots/outputs.")
        return df_fit

    # -----------------------
    # Plotting fits
    # -----------------------
    for (uid, grp, cf), sub in tqdm(
        collapsed.groupby([ID_COL, GROUP_COL, FORMULA_COL]),
        desc=f"{prefix} plotting"
    ):
        row_match = df_fit[
            (df_fit[ID_COL] == uid)
            & (df_fit[GROUP_COL] == grp)
            & (df_fit[FORMULA_COL] == cf)
        ]
        if row_match.empty:
            continue

        nL, k, A = row_match.iloc[0][
            [f"{prefix}_nL", f"{prefix}_rate", f"{prefix}_Asyn"]
        ].astype(float)

        times = pd.to_numeric(sub[TIME_COL], errors="coerce").to_numpy(float)
        obs = np.vstack(sub["D_spectrum"].to_numpy())
        P_s = pd.to_numeric(sub["enrichment"], errors="coerce").to_numpy(float)
        sim = _build_model_rows(nL, k, A, times, obs.shape[1], P_s)
        _save_fit_plot(uid, grp, plot_dir, prefix, times, obs, sim, (nL, k, A))
    df_final = collapsed.merge(df_fit, on=[ID_COL, GROUP_COL, FORMULA_COL], how="inner")
    return df_final



from scipy.optimize import minimize

# -------------------------------------------------------
# 8) Recombining with data from "Combine Extracted Files"
# -------------------------------------------------------
from typing import List

def reorder_with_metadata(df: pd.DataFrame, metadata_cols: List[str]) -> pd.DataFrame:

    """Put metadata columns first, preserve their original order."""
    existing_meta = [c for c in metadata_cols if c in df.columns]
    new_cols = [c for c in df.columns if c not in existing_meta]
    return df[existing_meta + new_cols]

def attach_and_reorder(
    df_core: pd.DataFrame,
    df_meta: pd.DataFrame,
    id_col: str,
    metadata_cols: List[str],
    FORMULA_COL: str = None,
    GROUP_COL: str = None
) -> pd.DataFrame:

    merged = df_core.merge(
        df_meta[[c for c in df_meta.columns
                 if c not in df_core.columns or c in [id_col, GROUP_COL, FORMULA_COL]]],
        on=[id_col, GROUP_COL, FORMULA_COL],
        how="left"
    )
    return reorder_with_metadata(merged, metadata_cols)






def _flag_rate_cap(
    df: pd.DataFrame,
    prefix: str,
    max_rate_cap: Optional[float],
    min_rate_cap: Optional[float],
):
    """
    Flag rows where the fitted rate k is outside allowed bounds.

      - If max_rate_cap is not None: flag k > max_rate_cap
      - If min_rate_cap is not None: flag k < min_rate_cap
    """
    if max_rate_cap is None and min_rate_cap is None:
        return df

    col = f"{prefix}_rate"
    if col not in df.columns:
        return df

    k_vals = pd.to_numeric(df[col], errors="coerce")

    # Upper cap
    if max_rate_cap is not None:
        mask_high = k_vals > float(max_rate_cap)
        _flag_reason(df, mask_high, f"{prefix}: k>{max_rate_cap}")

    # Lower cap
    if min_rate_cap is not None:
        mask_low = k_vals < float(min_rate_cap)
        _flag_reason(df, mask_low, f"{prefix}: k<{min_rate_cap}")

    return df







def annotate_qc_flags(
    df_fit: pd.DataFrame,
    prefix: str,
    minimum_standard_error: float,
    maximum_standard_error: float,
    asymptote_mode: str,
    max_rate_cap: Optional[float],
    min_rate_cap: Optional[float],
) -> pd.DataFrame:
    """
    QC Summary:

      SD QC ON ALL METRICS:
          metrics = ["nL", "rate"] OR ["nL", "rate", "Asyn"]
          FAIL if {prefix}_{m}_SD < minimum_standard_error
          FAIL if {prefix}_{m}_SD > maximum_standard_error

      RATE QC:
          FAIL if {prefix}_rate > max_rate_cap
          FAIL if {prefix}_rate < min_rate_cap
    """

    df = df_fit.copy()

    if "Flag" not in df.columns:
        df["Flag"] = "PASS"

    def _flag_reason_local(mask: pd.Series, reason: str):
        mask = mask.fillna(False)
        if not mask.any():
            return
        cur = df.loc[mask, "Flag"].astype(str)
        df.loc[mask, "Flag"] = np.where(
            cur.str.startswith("PASS"),
            "FAIL: " + reason,
            cur + "; " + reason,
        )

    prefix = prefix.strip("_")

    # ------------------------------------------------------------
    # Determine metrics to QC
    # ------------------------------------------------------------
    if asymptote_mode == "fixed":
        metrics = ["nL", "rate"]
    else:
        metrics = ["nL", "rate", "Asyn"]

    # ------------------------------------------------------------
    # QC: Standard Error bounds for ALL metrics
    # ------------------------------------------------------------
    for m in metrics:
        sd_col = f"{prefix}_{m}_SD"

        if sd_col not in df.columns:
            _flag_reason_local(pd.Series(True, index=df.index),
                               f"{prefix}: missing {sd_col}")
            continue

        sd_vals = pd.to_numeric(df[sd_col], errors="coerce")

        _flag_reason_local(
            sd_vals < minimum_standard_error,
            f"{prefix}: {m}_SD < {minimum_standard_error}"
        )

        _flag_reason_local(
            se_vals > maximum_standard_error,
            f"{prefix}: {m}_SD > {maximum_standard_error}"
        )

    # ------------------------------------------------------------
    # QC: rate caps
    # ------------------------------------------------------------
    rate_col = f"{prefix}_rate"
    if rate_col in df.columns:
        k_vals = pd.to_numeric(df[rate_col], errors="coerce")

        if max_rate_cap is not None:
            _flag_reason_local(k_vals > float(max_rate_cap),
                               f"{prefix}: k>{max_rate_cap}")

        if min_rate_cap is not None:
            _flag_reason_local(k_vals < float(min_rate_cap),
                               f"{prefix}: k<{min_rate_cap}")

    return df




def annotate_qc_flags(
    df_fit: pd.DataFrame,
    prefix: str,
    minimum_standard_error: float,
    maximum_standard_error: float,
    asymptote_mode: str,
    max_rate_cap: Optional[float],
    min_rate_cap: Optional[float],
) -> pd.DataFrame:
    """
    QC Summary (SD-based)
    ---------------------
    Checks bootstrap SD on all fitted metrics:
      - fixed asymptote: nL, rate
      - variable asymptote: nL, rate, Asyn

    FAIL if {prefix}_{metric}_SD < minimum_standard_error
    FAIL if {prefix}_{metric}_SD > maximum_standard_error

    Also applies rate caps (if provided):
      FAIL if {prefix}_rate > max_rate_cap
      FAIL if {prefix}_rate < min_rate_cap
    """
    df = df_fit.copy()
    if "Flag" not in df.columns:
        df["Flag"] = "PASS"

    def _flag_reason_local(mask: pd.Series, reason: str):
        mask = mask.fillna(False)
        if not mask.any():
            return
        cur = df.loc[mask, "Flag"].astype(str)
        df.loc[mask, "Flag"] = np.where(
            cur.str.startswith("PASS"),
            "FAIL: " + reason,
            cur + "; " + reason,
        )

    prefix = prefix.strip("_")

    # Metrics to QC
    if asymptote_mode == "fixed":
        metrics = ["nL", "rate"]
    else:
        metrics = ["nL", "rate", "Asyn"]

    # SD bounds QC
    for m in metrics:
        sd_col = f"{prefix}_{m}_SD"
        if sd_col not in df.columns:
            _flag_reason_local(pd.Series(True, index=df.index), f"{prefix}: missing {sd_col}")
            continue

        sd_vals = pd.to_numeric(df[sd_col], errors="coerce")

        _flag_reason_local(
            sd_vals < minimum_standard_error,
            f"{prefix}: {m}_SD < {minimum_standard_error}"
        )
        _flag_reason_local(
            sd_vals > maximum_standard_error,
            f"{prefix}: {m}_SD > {maximum_standard_error}"
        )

    # Rate caps QC
    rate_col = f"{prefix}_rate"
    if rate_col in df.columns:
        k_vals = pd.to_numeric(df[rate_col], errors="coerce")

        if max_rate_cap is not None:
            _flag_reason_local(k_vals > float(max_rate_cap), f"{prefix}: k>{max_rate_cap}")
        if min_rate_cap is not None:
            _flag_reason_local(k_vals < float(min_rate_cap), f"{prefix}: k<{min_rate_cap}")

    return df





def post_steps(
    df_fit: pd.DataFrame,
    ID_COL: str,
    save_dir: str,
    prefix: str,
    require_all: bool,
    asymptote_mode: str,
    user_Asyn_value,
    max_rate_cap: Optional[float],
    min_rate_cap: Optional[float],
    FORMULA_COL: str = None,
    GROUP_COL: str = None
) -> pd.DataFrame:

    boot_dir = os.path.join(save_dir, f"{prefix}_resampling_plots")
    os.makedirs(boot_dir, exist_ok=True)
    print(f"[{prefix}] Writing resampling histograms → {boot_dir}")

    df_qc = df_fit.copy()
    df_qc = _init_flag_column(df_qc)

    # --- Bootstrap histograms ---
    for idx, row in tqdm(df_qc.iterrows(), total=len(df_qc), desc=f"{prefix} boot hists"):

        uid = row.get(ID_COL)
        grp = row.get(GROUP_COL)
        cf  = row.get(FORMULA_COL)
        boot = row.get(f"{prefix}_resampling_samples", None)

        if boot is None or len(boot) == 0:
            df_qc.at[idx, "Flag"] = "FAIL-no_bootstrap"
            continue

        try:
            save_path = os.path.join(
                boot_dir,
                f"{prefix}_boot_{uid}_{grp}_{cf}.png"
            )
            r2_vals, ok = plot_3panel_boot_hist(
                boot,
                analyte_label=uid,
                save_path=save_path,
                prefix=prefix
            )

            for key, val in r2_vals.items():
                df_qc.at[idx, key] = val

            if not ok:
                df_qc.at[idx, "Flag"] = "FAIL-hist_gaussian"
                continue

        except Exception:
            df_qc.at[idx, "Flag"] = "FAIL-hist_exception"
            continue

    minimum_standard_error = settings.binomial_min_standard_error
    maximum_standard_error = settings.binomial_max_standard_error

    df_qc = annotate_qc_flags(
        df_qc,
        prefix=prefix,
        minimum_standard_error=minimum_standard_error,
        maximum_standard_error=maximum_standard_error,
        asymptote_mode=asymptote_mode,
        max_rate_cap=max_rate_cap,
        min_rate_cap=min_rate_cap,
    )

    # --- Report ---
    n_pass = (df_qc["Flag"] == "PASS").sum()
    print(f"[{prefix}] Post-processing complete — PASS {n_pass}/{len(df_qc)}")

    return df_qc



# ----------------------------------------------
# 9) Merge BA ∩ BS (QC-passed) + conformity plot
# ----------------------------------------------
def _stringify_cell(v):
    """Robust stringifier for scalars, lists, ndarrays, tuples."""
    if isinstance(v, (list, tuple, np.ndarray)):
        # flatten one level and join with commas
        arr = np.asarray(v, dtype=object).ravel()
        # convert inner arrays cleanly (avoid nested brackets)
        parts = []
        for x in arr:
            if isinstance(x, (list, tuple, np.ndarray)):
                parts.append(",".join(str(y) for y in np.asarray(x, dtype=object).ravel()))
            else:
                parts.append(str(x))
        return ",".join(parts)
    return str(v)

def _collapse_series_to_scalar_or_list(s: pd.Series) -> object:
    """
    If all non-null values are equivalent -> return that single value.
    Else -> return a comma-delimited list of unique values (order-preserving).
    """
    # drop NaN-like
    vals = [v for v in s if v is not None and (not isinstance(v, np.ndarray) or v.size > 0)]

    if len(vals) == 0:
        return np.nan
    # normalize to string for equivalence test but keep original first value as scalar
    str_vals = [_stringify_cell(v) for v in vals]
    first = vals[0]
    if all(_stringify_cell(v) == str_vals[0] for v in vals):
        return first  # equivalent across the group → keep scalar
    # non-equivalent → unique, order-preserving, comma-delimited
    seen = set()
    uniq = []
    for sv in str_vals:
        if sv not in seen:
            seen.add(sv); uniq.append(sv)
    return ",".join(uniq)



def collapse_analyte_rows(
    df: pd.DataFrame,
    keys: List[str],
) -> pd.DataFrame:

    """
    One-row-per-analyte collapse:
      - For each non-key column: if values differ, join unique values with commas.
      - If values are equivalent, keep the scalar as-is.
    """
    if df.empty:
        return df

    # Ensure all key columns exist
    missing = [k for k in keys if k not in df.columns]
    if missing:
        print(f"[Collapse] Skipped — missing key columns: {missing}")
        return df

    non_key_cols = [c for c in df.columns if c not in keys]

    def _agg(group: pd.DataFrame) -> pd.Series:
        out = {}
        for c in non_key_cols:
            out[c] = _collapse_series_to_scalar_or_list(group[c])
        # carry keys from the first row
        for k in keys:
            out[k] = group[k].iloc[0]
        return pd.Series(out)

    collapsed = (
        df.groupby(keys, sort=False, as_index=False)
          .apply(_agg)
          .reset_index(drop=True)
    )
    # Put keys first again
    return collapsed[keys + [c for c in collapsed.columns if c not in keys]]






# --------------------------
# 10) Final report
# --------------------------



def compute_final_nvalue_summary(
        merged: pd.DataFrame,
        fraction_mode: str,
) -> pd.DataFrame:
    """
    FINAL n-value assignment based *exclusively* on the user-selected
    fraction_new_calculation mode in the settings file.

    Modes:
    -------
    • "abundance" → use BA_nL
    """

    merged = merged.copy()
    mode = fraction_mode.strip().lower()

    # -------------------------
    # 1) Abundance-only mode
    # -------------------------
    if mode == "abundance":
        if "BA_nL" not in merged.columns:
            raise ValueError("[FinalSummary] BA_nL missing but fraction mode = 'abundance'")
        merged["n_value"] = merged["BA_nL"]
        merged["n_value_SD"] = merged["BA_nL_SD"]
        merged["n_value_dof"] = merged["BA_n_kept_boot"].apply(lambda x: max(int(x[0]) - 1, 1) if isinstance(x, (list, tuple)) and len(x) else np.nan)
        merged['n_val_lower_margin'] = merged['BA_nL_CI95']
        merged['n_val_upper_margin'] = merged['BA_nL_CI95']
        merged["final_method"] = "BA"
        return merged




def generate_fit_FN(
    df: pd.DataFrame,
    time_col: str,
    molecule_type: Optional[str] = None,
    plot_fitted_FN: bool = False,
    save_dir: Optional[str] = None,
    FORMULA_COL=None,
    GROUP_COL: str = None
) -> pd.DataFrame:

    out = df.copy()

    if time_col not in out.columns:
        print(f"[FN] Time column '{time_col}' not found — skipping FN generation.")
        return out

    t = pd.to_numeric(out[time_col], errors="coerce")

    # UPDATED: add BC (Combined joint fit)
    prefix_specs = {
        "BA": {
            "rate_col": "BA_rate",
            "asyn_col": "BA_Asyn",
            "fn_col": "BA_FN",
        },
        "BS": {
            "rate_col": "BS_rate",
            "asyn_col": "BS_Asyn",
            "fn_col": "BS_FN",
        },
        "BC": {
            "rate_col": "BC_rate",
            "asyn_col": "BC_Asyn",
            "fn_col": "BC_FN",
        },
    }

    fn_cols = []

    # Compute FN curves
    for prefix, spec in prefix_specs.items():
        rate_col = spec["rate_col"]
        asyn_col = spec["asyn_col"]
        fn_col = spec["fn_col"]

        if rate_col not in out.columns or asyn_col not in out.columns:
            continue

        rate = pd.to_numeric(out[rate_col], errors="coerce")
        Asyn = pd.to_numeric(out[asyn_col], errors="coerce")

        out[fn_col] = Asyn - Asyn * np.exp(-rate * t)
        fn_cols.append(fn_col)

    if not plot_fitted_FN:
        return out

    # Prefer BC plots over BA/BS
    if "BC_FN" in fn_cols:
        fn_cols = ["BC_FN"]

    # --- plotting code below unchanged ---
    # (your existing plotting logic is kept exactly the same)
    ...
    return out


def _debug_id_cols(df, tag=""):
    seq = "Sequence" in df.columns
    lip = "Lipid Unique Identifier" in df.columns
    print(f"[DEBUG-ID] {tag}: Sequence={seq}, LipidUID={lip}")

def plot_final_nL_panels(final_df, graph_directory):
    """
    Plot BS_nL, BA_nL, and BC_nL vs Observed Mass (or 'mass' fallback),
    showing ONLY Flag == 'PASS', using multi-panel layout if multiple exist.
    """

    # Determine mass column
    mass_col = None
    for candidate in ["Observed Mass", "mass", "Mass", "neutral_mass"]:
        if candidate in final_df.columns:
            mass_col = candidate
            break

    if mass_col is None:
        print("[Plot] ERROR: No mass column found.")
        return

    # Identify which nL columns exist
    plot_targets = []
    for prefix in ["BS", "BA", "BC"]:
        col = f"{prefix}_nL"
        if col in final_df.columns:
            plot_targets.append((prefix, col))

    if not plot_targets:
        print("[Plot] No nL columns (BS_nL / BA_nL / BC_nL) found.")
        return

    # Filter for Flag == PASS (only exact PASS)
    df_pass = final_df[ final_df["Flag"].astype(str).str.startswith("PASS") ].copy()

    if df_pass.empty:
        print("[Plot] WARNING: No PASS rows to plot.")
        return

    # Create figure
    n_panels = len(plot_targets)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]  # make iterable

    for ax, (prefix, col) in zip(axes, plot_targets):
        ax.scatter(df_pass[mass_col], df_pass[col],
                   s=12, c="blue", alpha=0.6)
        ax.set_title(f"{prefix} — {col} vs {mass_col} (PASS only)")
        ax.set_ylabel(f"{col}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(mass_col)

    # Save + show
    os.makedirs(graph_directory, exist_ok=True)
    out_path = os.path.join(graph_directory, "final_nL_multiplot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[Plot] Wrote final nL plot → {out_path}")



def main(dataframe, settings_path, molecule_type, out_path, graph_directory, processors):
    # ==============================================================
    # (0) CONFIG COLLECTION — DeuteRater or Stand-alone testing mode
    # ==============================================================
    REQUIRE_ALL = True
    BOOT_SEED = 0

    FORMULA_COL = 'Adduct_cf'
    TIME_COL = "time"
    GROUP_COL = "sample_group"
    MZ_COL = 'mzs'
    Z_COL = 'z'
    spec_col = "normalized_empirical_abundances"

    print('just loaded up df...')
    _debug_id_cols(dataframe, "ENTER main()")

    # --------------------------
    # Load settings
    # --------------------------
    import deuterater.settings as settings
    try:
        settings.load(settings_path)
        print(f"[Settings] Loaded from {settings_path}")
    except Exception as e:
        import traceback
        print(f"[Settings] Failed to load settings file: {settings_path}")
        traceback.print_exc()
        raise SystemExit("[Settings] Cannot continue without valid settings.")

    # Wire settings
    tp_min = int(getattr(settings, "tp_min"))
    n_boot = int(getattr(settings, "n_boot"))
    time_tol = float(getattr(settings, "time_tol"))
    max_rate_cap = float(getattr(settings, "maximum_allowed_sequence_rate"))
    min_rate_cap = float(getattr(settings, "minimum_allowed_sequence_rate"))
    cores = int(getattr(settings, "n_processors"))

    calc_mode = str(getattr(settings, "nL", "abundance")).lower()
    fit_mode = str(getattr(settings, "fit_mode", "both")).lower()

    # *** NEW: this setting now triggers the JOINT fit ***
    fraction_mode = str(getattr(settings, "fraction_new_calculation", "combined")).strip().lower()

    combined_abundance_vs_spacing_weights = float(
        getattr(settings, "combined_abundance_vs_spacing_weights", 1.0)
    )

    fixed_asymptote_value = float(
        getattr(settings, "fixed_asymptote_value", 1.0)
    )
    asymptote_mode = str(getattr(settings, "asymptote", "variable")).lower()

    # --------------------------
    # Setup molecule identifiers
    # --------------------------
    if molecule_type == 'Peptide':
        dataframe['Adduct'] = dataframe['z'].apply(lambda x: f"[M+{x}H]+{'' if x == 1 else x}H")
        dataframe['Adducted_identifier'] = dataframe['Sequence'] + "_" + dataframe['Adduct']

    if molecule_type == 'Lipid':
        dataframe['Adducted_identifier'] = dataframe['Lipid Unique Identifier'] + "_" + dataframe['Adduct']

    print(f"[Settings] asymptote = {asymptote_mode} (fixed value = {fixed_asymptote_value})")

    # --------------------------
    # Determine which fits to run — *NEW LOGIC*
    # --------------------------
    RUN_PREFIXES = ["BA"]

    print(f"[Settings] fraction_new_calculation = '{fraction_mode}' → Running: {RUN_PREFIXES}")

    # Set asymptote
    if asymptote_mode == "fixed":
        user_Asyn_value = fixed_asymptote_value
        print(f"[Mode] Asyn fixed to {user_Asyn_value}")
    else:
        user_Asyn_value = None
        print("[Mode] Asyn variable (fit freely)")

    # --------------------------
    # 1) Input prep
    # --------------------------
    df = dataframe.copy()
    df = normalize_abundances_if_missing(df)

    print(f"[Main] Loaded dataframe with {len(df)} rows")
    metadata_cols = list(df.columns)

    # --------------------------
    # 2) Identify molecule ID column
    # --------------------------
    mol_type = str(molecule_type).strip().lower()
    if mol_type == "lipid":
        ID_COL = "Lipid Unique Identifier"
    elif mol_type == "peptide":
        ID_COL = "Sequence"
    else:
        messagebox.showerror("Missing ID", "Molecule must be a lipid or peptide.")
        return
    print(f"[ID] Using '{ID_COL}' as unique identifier column.")

    # --------------------------
    # 3) Output directory and metadata
    # --------------------------
    save_dir = os.path.abspath(out_path)
    os.makedirs(save_dir, exist_ok=True)

    write_run_metadata(
        save_dir=save_dir,
        cores=cores,
        time_tol=time_tol,
        require_all=REQUIRE_ALL,
        max_rate_cap=max_rate_cap,
        min_rate_cap=min_rate_cap,
        moment_weight_scheme=MOMENT_WEIGHT_SCHEME,
        n_boot=n_boot,
        boot_seed=BOOT_SEED,
    )

    # --------------------------
    # 4) Deconvolution (D-spectrum)
    # --------------------------
    dfD = deconvolve_hydrogen_spectrum(
        df,
        ID_COL=ID_COL,
        spec_col=spec_col,
        FORMULA_COL=FORMULA_COL,
        MZ_COL=MZ_COL,
        Z_COL=Z_COL
    )
    _debug_id_cols(dfD, "After deconvolution")

    # --------------------------
    # 5) Collapse time
    # --------------------------
    collapsed = collapse_D_by_time(
        dfD,
        ID_COL=ID_COL,
        GROUP_COL=GROUP_COL,
        FORMULA_COL=FORMULA_COL,
        TIME_COL=TIME_COL,
        Z_COL=Z_COL,
        tol=time_tol
    )
    collapsed = _init_flag_column(collapsed)
    _debug_id_cols(collapsed, "After collapse")

    # --------------------------
    # 6) Multiprocessing pool
    # --------------------------
    pool = mp.Pool(cores)

    # --------------------------
    # 7) Setup fitters
    # --------------------------
    FITTERS = {
        "BA": lambda coll: perform_abundance_fit(
            coll,
            ID_COL,
            asymptote_mode=asymptote_mode,
            user_Asyn_value=user_Asyn_value,
            n_boot=n_boot,
            tp_min=tp_min,
            cores=cores,
            save_dir=save_dir,
            pool=pool,
            FORMULA_COL=FORMULA_COL,
            TP_MIN=tp_min,
            GROUP_COL=GROUP_COL,
            TIME_COL=TIME_COL,
            BOOT_SEED=BOOT_SEED,
        )

    }


    results_raw = {}
    results_keep = {}

    # --------------------------
    # 8) Run fits
    # --------------------------
    for prefix in RUN_PREFIXES:
        print(f"[Fit-{prefix}] starting…")
        df_raw = FITTERS[prefix](collapsed)

        if df_raw is None or df_raw.empty:
            print(f"[Fit-{prefix}] no rows — skipping downstream.")
            continue

        df_keep = post_steps(
            df_raw,
            ID_COL,
            save_dir=save_dir,
            prefix=prefix,
            require_all=REQUIRE_ALL,
            asymptote_mode=asymptote_mode,
            user_Asyn_value=user_Asyn_value,
            max_rate_cap=max_rate_cap,
            min_rate_cap=min_rate_cap,
            FORMULA_COL=FORMULA_COL,
            GROUP_COL=GROUP_COL
        )
        df_keep = attach_and_reorder(df_keep, df_meta=df, id_col=ID_COL, metadata_cols=metadata_cols, FORMULA_COL=FORMULA_COL, GROUP_COL=GROUP_COL)

        results_raw[prefix] = df_raw
        results_keep[prefix] = df_keep

    # --------------------------
    # 9) Merge / select final results
    # --------------------------
    if "Combined" in RUN_PREFIXES:
        # Joint fit replaces BA+BS logic entirely
        if "Combined" not in results_keep or results_keep["Combined"].empty:
            messagebox.showerror("No results", "Combined fit returned no valid rows.")
            pool.close(); pool.join(); return

        merged = results_keep["Combined"].copy()
        print("[Merge] Using Combined (joint) fit results only.")

    else:
        # BA / BS normal merge
        df_ba = results_keep.get("BA")
        df_bs = results_keep.get("BS")

        if df_ba is not None and not df_ba.empty and df_bs is not None and not df_bs.empty:
            merged = df_ba.merge(df_bs, on=[ID_COL, GROUP_COL, FORMULA_COL], how="outer", suffixes=("_BA", "_BS"))
        elif df_ba is not None:
            merged = df_ba.copy()
        elif df_bs is not None:
            merged = df_bs.copy()
        else:
            messagebox.showerror("No results", "No BA or BS results available.")
            pool.close(); pool.join(); return

    # --------------------------
    # 10) Final nL summary
    # --------------------------
    print("[Final] Computing final nL summary…")

    # fnc was already parsed from settings earlier:
    # fnc = str(getattr(settings, "fraction_new_calculation", "")).strip().lower()

    final_df = compute_final_nvalue_summary(
        merged=merged,
        fraction_mode=fraction_mode  # <<< THIS IS THE NEW CORRECT WAY
    )

    # Generate FN plots using the chosen mode’s columns
    final_df = generate_fit_FN(
        final_df,
        time_col=TIME_COL,
        molecule_type=molecule_type,
        plot_fitted_FN=True,
        save_dir=save_dir,
        FORMULA_COL=FORMULA_COL,
        GROUP_COL=GROUP_COL
    )

    # --------------------------
    # 11) Write output
    # --------------------------
    out_file = os.path.join(save_dir, "binomial_nL_results.tsv")
    final_df.to_csv(out_file, sep="\t", index=False)
    print(f"[Final] Wrote: {out_file}")

    pool.close()
    pool.join()

    # --------------------------
    # 12) Plot BS_nL / BA_nL / BC_nL panels
    # --------------------------
    plot_final_nL_panels(final_df, save_dir)
    print("\n[Main] Completed successfully.")

    return final_df




def run_binomial_pipeline(
    dataframe,
    settings_path,
    biomolecule_type,
    out_path,
    graphs_location,
    processors,
):
    print("\n" + "=" * 80)
    print("[Binomial] DEBUG: Incoming dataframe columns:")
    print(list(dataframe.columns))
    print("=" * 80 + "\n")

    """
    #remove, alone with other not used TESTING techniques
    dataframe  = downsample_unique_pairs(
            dataframe, id_col='Sequence', formula_col='Adduct_cf', max_n= 60
        )


    dataframe  = downsample_unique_pairs(
            dataframe, id_col='Lipid Unique Identifier', formula_col='Adduct_cf', max_n=60
        )
    """


    # -----------------------------------------------------
    # FIX: out_path is ALWAYS a file.
    # Convert file path → containing folder.
    # -----------------------------------------------------
    out_path = str(out_path)
    output_dir = os.path.dirname(out_path)

    if output_dir == "":
        # Handle cases like: "output.tsv" with no path
        output_dir = os.getcwd()

    print(f"[Binomial] Using output folder: {output_dir}")

    # Ensure folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Pass the folder to main()
    output = main(
        dataframe=dataframe,
        settings_path=settings_path,
        molecule_type=biomolecule_type,
        out_path=output_dir,
        graph_directory=graphs_location,
        processors=processors,
    )
    return output


##NOTE The combined nL, Asyn and rate fit should actually be a minimization reflecting the fit of all 8 lines, as opposed to just 4 per.
