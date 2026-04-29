#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lipid simulation with concentration tracking, isotope pattern tracking,
and matplotlib visualization using pandas.

OPTIMIZED VERSION — key changes vs original:
  1. FFT convolution replaces O(i³) nested Python loops in isotope model
  2. Binomial vectors pre-computed & cached per (nL, p) — not recomputed per call
  3. Component matrices vectorized across all time-points at once
  4. Multiprocessing Pool created ONCE outside the loss function (not per call)
  5. Staged optimization: independent pool fits → per-species fits → joint refinement
     instead of blind 42-parameter L-BFGS-B from scratch
  6. build_model_distribution vectorized over time

Multiple FA types (FA1, FA2, FA3) and HG types (HG1, HG2) are available.
Each lipid synthesis randomly draws one FA for each FA slot and one HG.

Unique lipid species are defined by (FA_slot1, FA_slot2, HG) — up to 12
combinations. FA slot order is sorted so FA1+FA2 == FA2+FA1.

Pool replenishment is balanced so all precursor pools stay stable.
To scale turnover, only change Lipid_syn_r and Lipid_deg_r — everything
else adjusts automatically.
"""

import random as rand
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from math import lgamma
from functools import lru_cache
from itertools import combinations_with_replacement
import matplotlib
matplotlib.use('Qt5Agg')  # Spyder-compatible interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Fitting functions ─────────────────────────────────────────────────────────

def fractional_binom_pmf_raw(i, nL, p):
    """Un-cached scalar computation."""
    if nL <= 0 or i > nL:
        return 0.0
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    log_c = lgamma(nL + 1.0) - lgamma(i + 1.0) - lgamma(nL - i + 1.0)
    return float(np.exp(log_c + i * np.log(p) + (nL - i) * np.log1p(-p)))


# Keep the old name for any code that calls it directly
def fractional_binom_pmf(i, nL, p):
    return fractional_binom_pmf_raw(i, nL, p)


@lru_cache(maxsize=4096)
def _binom_vec_cached(nL_r, p_r, iso_len):
    """
    Cache the full binomial probability vector for a given (nL, p, iso_len).
    nL and p are pre-rounded to avoid float cache misses.
    """
    vec = np.array([fractional_binom_pmf_raw(i, nL_r, p_r) for i in range(iso_len)],
                   dtype=np.float64)
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec  # NOTE: callers must NOT mutate this array


def binom_vec(nL, p, iso_len):
    """Return a normalised binomial PMF vector (read-only cache hit friendly)."""
    return _binom_vec_cached(round(float(nL), 3), round(float(p), 6), int(iso_len))


# ── Vectorized component model ────────────────────────────────────────────────

def build_component_matrix(nL, A, k, times, iso_len, p):
    """
    Returns array of shape (T, iso_len) for one component across all times.

    Vectorized: the binom vector is the same for all t; only the mixing
    fraction f(t) = A*(1 - exp(-k*t)) varies, so we broadcast.
    """
    times = np.asarray(times, dtype=np.float64)
    bv    = binom_vec(nL, p, iso_len).copy()   # safe copy — we won't mutate cached

    baseline    = np.zeros(iso_len, dtype=np.float64)
    baseline[0] = 1.0

    f = A * (1.0 - np.exp(-k * times))          # shape (T,)
    # Outer broadcast: (T,1) * (1,iso_len)
    return (1.0 - f)[:, None] * baseline[None, :] + f[:, None] * bv[None, :]


def build_model_full_fast(FAa, FAb, HGz,
                           FA_params, HG_params,
                           nL_Gly, A_Gly, k_Gly,
                           A_L, k_L,
                           times, iso_len, p):
    """
    Build the full convolved lipid isotope model for all time-points at once.

    Replaces the original O(i³) nested-loop convolution with FFT convolution.
    """
    times = np.asarray(times, dtype=np.float64)
    T     = len(times)

    nL_a, A_a, k_a = FA_params[FAa]
    nL_b, A_b, k_b = FA_params[FAb]
    nL_h, A_h, k_h = HG_params[HGz]

    Ma = build_component_matrix(nL_a,    A_a,    k_a,    times, iso_len, p)
    Mb = build_component_matrix(nL_b,    A_b,    k_b,    times, iso_len, p)
    Mg = build_component_matrix(nL_Gly,  A_Gly,  k_Gly,  times, iso_len, p)
    Mh = build_component_matrix(nL_h,    A_h,    k_h,    times, iso_len, p)

    f_L    = A_L * (1.0 - np.exp(-k_L * times))  # shape (T,)
    base   = np.zeros(iso_len, dtype=np.float64)
    base[0] = 1.0

    results = np.zeros((T, iso_len), dtype=np.float64)
    for ti in range(T):
        conv = fftconvolve(Ma[ti], Mb[ti])[:iso_len]
        conv = fftconvolve(conv,   Mg[ti])[:iso_len]
        conv = fftconvolve(conv,   Mh[ti])[:iso_len]
        s = conv.sum()
        if s > 0:
            conv /= s
        results[ti] = (1.0 - f_L[ti]) * base + f_L[ti] * conv

    return results


def build_model_distribution(nL, k, Asyn, times, iso_len, p_label):
    """
    Vectorized version of the original function.
    Returns shape (T, iso_len).
    """
    times = np.asarray(times, dtype=np.float64)
    p     = float(np.clip(p_label, 1e-12, 1.0 - 1e-12))
    bv    = binom_vec(nL, p, iso_len).copy()

    baseline    = np.zeros(iso_len, dtype=np.float64)
    baseline[0] = 1.0

    f_new = Asyn * (1.0 - np.exp(-k * times))   # (T,)
    sim   = (1.0 - f_new)[:, None] * baseline[None, :] + f_new[:, None] * bv[None, :]

    row_sums = sim.sum(axis=1, keepdims=True)
    sim /= np.where(row_sums > 0, row_sums, 1.0)
    return sim


# ── Single-pool fit (unchanged API, faster internals) ─────────────────────────

def fit_pool(isotope_df, pool_name, p_label, days):
    subset = isotope_df[isotope_df['pool'] == pool_name]
    if subset.empty:
        return None
    pivot = subset.pivot(index='day', columns='label_count', values='fraction').fillna(0)
    if pivot.shape[0] < 3:
        return None

    times   = pivot.index.to_numpy(float)
    obs     = pivot.values[:, :4]
    iso_len = obs.shape[1]
    nH_max  = float(iso_len * 4)

    def loss(x):
        nL, k, A = x
        if not (0.1 <= nL <= nH_max and 0.001 <= k <= 10.0 and 0.0 < A <= 1.0):
            return np.inf
        sim = build_model_distribution(nL, k, A, times, iso_len, p_label)
        return float(np.sum((obs - sim) ** 2))

    best_res = None
    for nL0 in [1.0, iso_len / 2, iso_len]:
        for k0 in [0.05, 0.3, 1.0]:
            res = minimize(loss, x0=[nL0, k0, 0.8],
                           bounds=[(0.1, nH_max), (0.001, 10.0), (0.0, 1.0)],
                           method='L-BFGS-B')
            if best_res is None or res.fun < best_res.fun:
                best_res = res

    nL_fit, k_fit, A_fit = best_res.x
    sim_best = build_model_distribution(nL_fit, k_fit, A_fit, times, iso_len, p_label)
    ss_res = np.sum((obs - sim_best) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    R2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        'pool':    pool_name,
        'nL':      round(nL_fit, 3),
        'k':       round(k_fit,  4),
        'Asyn':    round(A_fit,  3),
        'R2':      round(R2,     4),
        'obs':     obs,
        'sim':     sim_best,
        'times':   times,
        'iso_len': iso_len,
    }


# ── Staged global fit (replaces monolithic multiprocessing version) ───────────

def fit_all_lipids_shared_components(isotope_df, species_keys, p_label):
    """
    Fit all lipid species simultaneously with shared component parameters.

    OPTIMIZATION STRATEGY (replaces original blind 42-param L-BFGS-B):
      Stage 1 — Fit each precursor pool independently (fast 3-param fits).
                 These become warm-start values for shared parameters.
      Stage 2 — Fix shared params; fit per-species (A_L, k_L) independently.
                 Reduces 42-param problem to twelve 2-param problems.
      Stage 3 — Short joint L-BFGS-B refinement starting from Stage 1+2 solution.

    No multiprocessing needed — vectorized NumPy is faster for this size.
    """

    FA_NAMES  = ["FA1", "FA2", "FA3"]
    HG_NAMES  = ["HG1", "HG2"]
    GLY_NAME  = "Glycerol"
    ALL_POOLS = FA_NAMES + [GLY_NAME] + HG_NAMES

    # ── Prepare pivot data ──────────────────────────────────────────────────
    def get_pivot(pool_name):
        sub = isotope_df[isotope_df['pool'] == pool_name]
        if sub.empty:
            return None, None
        piv = sub.pivot(index='day', columns='label_count', values='fraction').fillna(0)
        piv = piv[[c for c in piv.columns if c < 4]]
        if piv.shape[0] < 3:
            return None, None
        return piv.index.to_numpy(float), piv.values

    # Build species data dict
    species_data = {}
    for sp in species_keys:
        times, obs = get_pivot(sp)
        if times is not None:
            species_data[sp] = {"times": times, "obs": obs, "iso_len": obs.shape[1]}
    species_list = list(species_data.keys())
    nS = len(species_list)

    # ── Stage 1: independent pool fits ─────────────────────────────────────
    print("  [Stage 1] Fitting precursor pools independently…")
    pool_fits = {}
    for pool_name in ALL_POOLS:
        times, obs = get_pivot(pool_name)
        if times is None:
            pool_fits[pool_name] = (5.0, 0.5, 0.05)
            continue
        iso_len = obs.shape[1]
        nH_max  = float(iso_len * 4)

        def make_loss(obs_, times_, iso_len_):
            def loss(x):
                nL, A, k = x
                if not (0.1 <= nL <= nH_max and 0.0 < A <= 1.0 and 0.001 <= k <= 10.0):
                    return np.inf
                sim = build_model_distribution(nL, k, A, times_, iso_len_, p_label)
                return float(np.sum((obs_ - sim) ** 2))
            return loss

        best = None
        for nL0 in [2.0, 5.0, 10.0]:
            for k0 in [0.05, 0.3, 1.0]:
                r = minimize(make_loss(obs, times, iso_len),
                             x0=[nL0, 0.5, k0],
                             bounds=[(0.1, nH_max), (0.0, 1.0), (0.001, 10.0)],
                             method='L-BFGS-B')
                if best is None or r.fun < best.fun:
                    best = r
        nL_f, A_f, k_f = best.x
        pool_fits[pool_name] = (float(nL_f), float(A_f), float(k_f))
        print(f"    {pool_name}: nL={nL_f:.3f}  A={A_f:.3f}  k={k_f:.4f}")

    # ── Stage 2: per-species (A_L, k_L) with shared params fixed ──────────
    print("  [Stage 2] Fitting per-species A_L, k_L with fixed shared params…")
    FA_params_s1  = {fa: pool_fits[fa] for fa in FA_NAMES}
    HG_params_s1  = {hg: pool_fits[hg] for hg in HG_NAMES}
    nL_Gly_s1, A_Gly_s1, k_Gly_s1 = pool_fits[GLY_NAME]

    species_AL_kL = {}
    for sp in species_list:
        d      = species_data[sp]
        FAa, FAb, HGz = sp.split("+")
        obs_sp = d["obs"]
        times_sp = d["times"]
        iso_len  = d["iso_len"]

        def make_sp_loss(obs_, times_, iso_len_, FAa_, FAb_, HGz_):
            def loss(x):
                A_L, k_L = x
                if not (0.0 <= A_L <= 1.0 and 0.001 <= k_L <= 10.0):
                    return np.inf
                sim = build_model_full_fast(
                    FAa_, FAb_, HGz_,
                    FA_params_s1, HG_params_s1,
                    nL_Gly_s1, A_Gly_s1, k_Gly_s1,
                    A_L, k_L,
                    times_, iso_len_, p_label
                )
                return float(np.sum((obs_ - sim) ** 2))
            return loss

        best = None
        for A0 in [0.3, 0.7]:
            for k0 in [0.05, 0.5]:
                r = minimize(make_sp_loss(obs_sp, times_sp, iso_len, FAa, FAb, HGz),
                             x0=[A0, k0],
                             bounds=[(0.0, 1.0), (0.001, 10.0)],
                             method='L-BFGS-B')
                if best is None or r.fun < best.fun:
                    best = r
        species_AL_kL[sp] = tuple(best.x)
        print(f"    {sp}: A_L={best.x[0]:.3f}  k_L={best.x[1]:.4f}")

    # ── Stage 3: joint refinement from warm start ─────────────────────────
    print("  [Stage 3] Joint refinement from warm start…")

    def pack_x(pool_fits_, species_AL_kL_):
        x = []
        for fa in FA_NAMES:
            x += list(pool_fits_[fa])          # nL, A, k
        x += list(pool_fits_[GLY_NAME])
        for hg in HG_NAMES:
            x += list(pool_fits_[hg])
        for sp in species_list:
            x += list(species_AL_kL_[sp])      # A_L, k_L
        return np.array(x, dtype=np.float64)

    def unpack_x(x):
        idx = 0
        FA_p = {}
        for fa in FA_NAMES:
            FA_p[fa] = tuple(x[idx:idx+3]); idx += 3
        nL_G, A_G, k_G = x[idx:idx+3]; idx += 3
        HG_p = {}
        for hg in HG_NAMES:
            HG_p[hg] = tuple(x[idx:idx+3]); idx += 3
        sp_params = {}
        for sp in species_list:
            sp_params[sp] = tuple(x[idx:idx+2]); idx += 2
        return FA_p, nL_G, A_G, k_G, HG_p, sp_params

    def joint_loss(x):
        FA_p, nL_G, A_G, k_G, HG_p, sp_params = unpack_x(x)

        # Validate shared params
        for nL, A, k in list(FA_p.values()) + list(HG_p.values()) + [(nL_G, A_G, k_G)]:
            if nL <= 0 or k <= 0 or not (0.0 <= A <= 1.0):
                return np.inf

        total = 0.0
        for sp in species_list:
            A_L, k_L = sp_params[sp]
            if not (0.0 <= A_L <= 1.0 and 0.001 <= k_L <= 10.0):
                return np.inf
            d   = species_data[sp]
            FAa, FAb, HGz = sp.split("+")
            sim = build_model_full_fast(
                FAa, FAb, HGz,
                FA_p, HG_p,
                nL_G, A_G, k_G,
                A_L, k_L,
                d["times"], d["iso_len"], p_label
            )
            total += float(np.sum((d["obs"] - sim) ** 2))
        return total

    x0 = pack_x(pool_fits, species_AL_kL)

    bounds = (
        [(0.1, 30), (0.0, 1.0), (0.001, 10.0)] * len(FA_NAMES) +
        [(0.1, 10), (0.0, 1.0), (0.001, 10.0)] +               # Gly
        [(0.1, 10), (0.0, 1.0), (0.001, 10.0)] * len(HG_NAMES) +
        [(0.0, 1.0), (0.001, 10.0)] * nS
    )

    res = minimize(joint_loss, x0=x0, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6})
    print(f"  [Stage 3] Done. loss={res.fun:.6f}  converged={res.success}")
    return res


# ── Component / Lipid classes ─────────────────────────────────────────────────

class Component():
    def __init__(self, kind, nL, Asyn, L, diet_r, syn_r):
        self.kind   = kind
        self.nL     = nL
        self.Asyn   = Asyn
        self.L      = L
        self.diet_r = diet_r
        self.syn_r  = syn_r


class Lipid():
    def __init__(self, FA1, FA2, glycerol, HG, deg_r, syn_r):
        self.nL      = FA1.nL + FA2.nL + glycerol.nL + HG.nL
        self.deg_r   = deg_r
        self.syn_r   = syn_r
        self.L       = FA1.L + FA2.L + glycerol.L + HG.L
        fa_sorted    = tuple(sorted([FA1.kind, FA2.kind]))
        self.species = f"{fa_sorted[0]}+{fa_sorted[1]}+{HG.kind}"


# ── Helper ────────────────────────────────────────────────────────────────────

def all_species_keys(fa_kinds, hg_kinds):
    keys = []
    for fa_pair in combinations_with_replacement(sorted(fa_kinds), 2):
        for hg in sorted(hg_kinds):
            keys.append(f"{fa_pair[0]}+{fa_pair[1]}+{hg}")
    return keys


# ── Main simulation class ─────────────────────────────────────────────────────

class Cup():
    def __init__(self, Asyn=0.15, verbose=True):
        self.days = 30

        # ── Turnover rate — only change these two values to scale lipid turnover
        self.Lipid_syn_r = 1000  # lipids synthesised per day
        self.Lipid_deg_r = 1000  # lipids degraded per day (keep equal to syn_r)

        # Scale factor so precursor pools replenish enough to supply lipid synthesis
        scale = self.Lipid_syn_r / 100

        # ── Component nL and Asyn ──────────────────────────────────────────────
        self.FA1nL   = 14;  self.FA1Asyn   = Asyn
        self.FA2nL   = 16;  self.FA2Asyn   = Asyn
        self.FA3nL   = 18;  self.FA3Asyn   = Asyn
        self.GlycerolnL = 5; self.GlycerolAsyn = Asyn
        self.HG1nL   = 5;   self.HG1Asyn   = Asyn
        self.HG2nL   = 8;   self.HG2Asyn   = Asyn

        # ── Syn/diet rates — auto-scaled ───────────────────────────────────────
        self.FA1syn_r      = int(100 * scale)
        self.FA1diet_r     = self.FA1syn_r * ((1 - self.FA1Asyn) / self.FA1Asyn)

        self.FA2syn_r      = int(100 * scale)
        self.FA2diet_r     = self.FA2syn_r * ((1 - self.FA2Asyn) / self.FA2Asyn)

        self.FA3syn_r      = int(100 * scale)
        self.FA3diet_r     = self.FA3syn_r * ((1 - self.FA3Asyn) / self.FA3Asyn)

        self.Glycerolsyn_r  = int(100 * scale)
        self.Glyceroldiet_r = self.Glycerolsyn_r * ((1 - self.GlycerolAsyn) / self.GlycerolAsyn)

        self.HG1syn_r      = int(100 * scale)
        self.HG1diet_r     = self.HG1syn_r * ((1 - self.HG1Asyn) / self.HG1Asyn)

        self.HG2syn_r      = int(100 * scale)
        self.HG2diet_r     = self.HG2syn_r * ((1 - self.HG2Asyn) / self.HG2Asyn)

        # ── Template molecules ─────────────────────────────────────────────────
        self.basic_FA1      = Component('FA1',      self.FA1nL,      self.FA1Asyn,      0, self.FA1diet_r,      self.FA1syn_r)
        self.basic_FA2      = Component('FA2',      self.FA2nL,      self.FA2Asyn,      0, self.FA2diet_r,      self.FA2syn_r)
        self.basic_FA3      = Component('FA3',      self.FA3nL,      self.FA3Asyn,      0, self.FA3diet_r,      self.FA3syn_r)
        self.basic_Glycerol = Component('Glycerol', self.GlycerolnL, self.GlycerolAsyn, 0, self.Glyceroldiet_r, self.Glycerolsyn_r)
        self.basic_HG1      = Component('HG1',      self.HG1nL,      self.HG1Asyn,      0, self.HG1diet_r,      self.HG1syn_r)
        self.basic_HG2      = Component('HG2',      self.HG2nL,      self.HG2Asyn,      0, self.HG2diet_r,      self.HG2syn_r)

        # Option lists for random draws
        self.FA_options = [self.basic_FA1, self.basic_FA2, self.basic_FA3]
        self.HG_options = [self.basic_HG1, self.basic_HG2]

        # ── Initial pools (~10 000 each) ───────────────────────────────────────
        self.FA1_pool      = [Component('FA1',      self.FA1nL,      self.FA1Asyn,      0, self.FA1diet_r,      self.FA1syn_r)      for _ in range(10000)]
        self.FA2_pool      = [Component('FA2',      self.FA2nL,      self.FA2Asyn,      0, self.FA2diet_r,      self.FA2syn_r)      for _ in range(10000)]
        self.FA3_pool      = [Component('FA3',      self.FA3nL,      self.FA3Asyn,      0, self.FA3diet_r,      self.FA3syn_r)      for _ in range(10000)]
        self.Glycerol_pool = [Component('Glycerol', self.GlycerolnL, self.GlycerolAsyn, 0, self.Glyceroldiet_r, self.Glycerolsyn_r) for _ in range(10000)]
        self.HG1_pool      = [Component('HG1',      self.HG1nL,      self.HG1Asyn,      0, self.HG1diet_r,      self.HG1syn_r)      for _ in range(10000)]
        self.HG2_pool      = [Component('HG2',      self.HG2nL,      self.HG2Asyn,      0, self.HG2diet_r,      self.HG2syn_r)      for _ in range(10000)]
        self.Lipid_pool = [
            Lipid(
                rand.choice(self.FA_options),
                rand.choice(self.FA_options),
                self.basic_Glycerol,
                rand.choice(self.HG_options),
                self.Lipid_syn_r,
                self.Lipid_deg_r
            )
            for _ in range(10000)
        ]

        self.p = 5  # D2O percent

        self.species_keys = all_species_keys(
            [o.kind for o in self.FA_options],
            [o.kind for o in self.HG_options]
        )

        # ── Data tracking ──────────────────────────────────────────────────────
        self.conc_records    = []
        self.isotope_records = []

        self._record_day(0)
        for day in range(1, self.days + 1):
            self.progress()
            self._record_day(day)

        self.conc_df    = pd.DataFrame(self.conc_records)
        self.isotope_df = pd.DataFrame(self.isotope_records)

        # ── GLOBAL compound lipid fit (staged) ─────────────────────────────────
        print("\n=== Running staged global fit ===")
        res = fit_all_lipids_shared_components(
            self.isotope_df,
            self.species_keys,
            self.p / 100.0
        )

        self.global_fit_result = res
        x = res.x

        FA_NAMES = ["FA1", "FA2", "FA3"]
        HG_NAMES = ["HG1", "HG2"]

        # Unpack shared parameters
        idx  = 0
        fa_p = {}
        for fa in FA_NAMES:
            fa_p[fa] = x[idx:idx+3]; idx += 3
        nL_G, A_G, k_G = x[idx:idx+3]; idx += 3
        hg_p = {}
        for hg in HG_NAMES:
            hg_p[hg] = x[idx:idx+3]; idx += 3

        # ── Constituent parameter table ─────────────────────────────────────
        comp_rows = [{"Component": fa,
                      "nL": round(fa_p[fa][0], 3),
                      "Asyn": round(fa_p[fa][1], 3),
                      "k": round(fa_p[fa][2], 4)}
                     for fa in FA_NAMES]
        comp_rows.append({"Component": "Glycerol",
                          "nL": round(nL_G, 3), "Asyn": round(A_G, 3), "k": round(k_G, 4)})
        for hg in HG_NAMES:
            comp_rows.append({"Component": hg,
                               "nL": round(hg_p[hg][0], 3),
                               "Asyn": round(hg_p[hg][1], 3),
                               "k": round(hg_p[hg][2], 4)})
        self.component_fit_df = pd.DataFrame(comp_rows)

        if verbose:
            print("\n=== Constituent Parameter Estimates (GLOBAL FIT) ===")
            print(self.component_fit_df.to_string(index=False))

        # ── Lipid-level turnover results ────────────────────────────────────
        # species_list order matches what the fit used; rebuild it here
        species_in_data = [
            sp for sp in self.species_keys
            if not self.isotope_df[self.isotope_df['pool'] == sp].empty
        ]
        lipid_rows = []
        for i, sp in enumerate(species_in_data):
            A_L = x[idx + 2*i]
            k_L = x[idx + 2*i + 1]
            lipid_rows.append({"Species": sp,
                                "Asyn_L": round(float(A_L), 3),
                                "k_L":    round(float(k_L), 4)})
        self.lipid_fit_df = pd.DataFrame(lipid_rows)

        if verbose:
            print("\n=== Lipid-Level Kinetic Parameters ===")
            print(self.lipid_fit_df.to_string(index=False))

        # ── Fit pools (independent binomial fits) ─────────────────────────────
        p_label         = self.p / 100.0
        precursor_pools = ['FA1', 'FA2', 'FA3', 'Glycerol', 'HG1', 'HG2']
        fit_pool_names  = precursor_pools + ['Lipid']

        self.fit_results = {}
        fit_rows = []

        for pool_name in fit_pool_names:
            result = fit_pool(self.isotope_df, pool_name, p_label, self.days)
            if result is not None:
                self.fit_results[pool_name] = result
                fit_rows.append({
                    'Pool':     pool_name,
                    'Fit nL':   result['nL'],
                    'Fit k':    result['k'],
                    'Fit Asyn': result['Asyn'],
                    'R2':       result['R2'],
                })

        self.fit_df = pd.DataFrame(fit_rows)

        if verbose:
            print("\n=== Binomial fits ===")
            print(self.fit_df.to_string(index=False))
            self.plot()

    # ── Public accessor ────────────────────────────────────────────────────────

    def lipid_R2(self):
        return self.fit_results.get('Lipid', {}).get('R2', np.nan)

    # ── Recording helpers ──────────────────────────────────────────────────────

    def _record_day(self, day):
        species_counts = Counter(lip.species for lip in self.Lipid_pool)
        row = {
            'day':      day,
            'FA1':      len(self.FA1_pool),
            'FA2':      len(self.FA2_pool),
            'FA3':      len(self.FA3_pool),
            'FA_total': len(self.FA1_pool) + len(self.FA2_pool) + len(self.FA3_pool),
            'Glycerol': len(self.Glycerol_pool),
            'HG1':      len(self.HG1_pool),
            'HG2':      len(self.HG2_pool),
            'HG_total': len(self.HG1_pool) + len(self.HG2_pool),
            'Lipid':    len(self.Lipid_pool),
        }
        for key in self.species_keys:
            row[key] = species_counts.get(key, 0)
        self.conc_records.append(row)

        self._record_isotope(day, 'FA1',      self.FA1_pool)
        self._record_isotope(day, 'FA2',      self.FA2_pool)
        self._record_isotope(day, 'FA3',      self.FA3_pool)
        self._record_isotope(day, 'Glycerol', self.Glycerol_pool)
        self._record_isotope(day, 'HG1',      self.HG1_pool)
        self._record_isotope(day, 'HG2',      self.HG2_pool)
        self._record_isotope(day, 'Lipid',    self.Lipid_pool)

        for key in self.species_keys:
            sp_pool = [lip for lip in self.Lipid_pool if lip.species == key]
            if sp_pool:
                self._record_isotope(day, key, sp_pool)

    def _record_isotope(self, day, pool_name, pool):
        if not pool:
            return
        counts = Counter(mol.L for mol in pool)
        total  = len(pool)
        for label_count, n in sorted(counts.items()):
            self.isotope_records.append({
                'day':         day,
                'pool':        pool_name,
                'label_count': label_count,
                'fraction':    n / total,
            })

    # ── Simulation logic ───────────────────────────────────────────────────────

    def Day(self, Molecule, Pool, Diet_rate, Synth_rate, Lipidc_rate, p):
        """
        Replenish a precursor pool while keeping its size stable.
        Deg_rate = syn + diet - consumed, so net change = consumed in = consumed out.
        """
        Diet_rate   = int(Diet_rate)
        Synth_rate  = int(Synth_rate)
        Lipidc_rate = int(Lipidc_rate)
        Deg_rate    = Diet_rate + Synth_rate - Lipidc_rate

        for _ in range(Synth_rate):
            L = sum(1 for _ in range(Molecule.nL) if rand.randint(0, 100) < p)
            Pool.append(Component(Molecule.kind, Molecule.nL, Molecule.Asyn, L,
                                  Molecule.diet_r, Molecule.syn_r))

        for _ in range(Diet_rate):
            Pool.append(Component(Molecule.kind, Molecule.nL, Molecule.Asyn, 0,
                                  Molecule.diet_r, Molecule.syn_r))

        for _ in range(max(0, Deg_rate)):
            if Pool:
                Pool.pop(rand.randint(0, len(Pool) - 1))

    def _draw_component(self, options_list, pool_lookup):
        template = rand.choice(options_list)
        pool     = pool_lookup[template.kind]
        if not pool:
            for opt in options_list:
                fallback = pool_lookup[opt.kind]
                if fallback:
                    return fallback.pop(rand.randint(0, len(fallback) - 1))
            raise RuntimeError(f"All pools empty for: {[o.kind for o in options_list]}")
        return pool.pop(rand.randint(0, len(pool) - 1))

    def progress(self):
        fa_pools = {'FA1': self.FA1_pool, 'FA2': self.FA2_pool, 'FA3': self.FA3_pool}
        hg_pools = {'HG1': self.HG1_pool, 'HG2': self.HG2_pool}

        # Each lipid consumes 2 FA across 3 pools → 2/3 * syn_r per FA pool
        # Each lipid consumes 1 HG across 2 pools → 1/2 * syn_r per HG pool
        fa_consume = (self.Lipid_syn_r * 2) // 3
        hg_consume = self.Lipid_syn_r // 2

        for _ in range(self.Lipid_syn_r):
            fa1 = self._draw_component(self.FA_options, fa_pools)
            fa2 = self._draw_component(self.FA_options, fa_pools)
            hg  = self._draw_component(self.HG_options, hg_pools)
            gly = self.Glycerol_pool.pop(rand.randint(0, len(self.Glycerol_pool) - 1))
            self.Lipid_pool.append(Lipid(fa1, fa2, gly, hg,
                                         self.Lipid_deg_r, self.Lipid_syn_r))

        for _ in range(self.Lipid_deg_r):
            if self.Lipid_pool:
                self.Lipid_pool.pop(rand.randint(0, len(self.Lipid_pool) - 1))

        self.Day(self.basic_FA1,      self.FA1_pool,      self.FA1diet_r,      self.FA1syn_r,      fa_consume,        self.p)
        self.Day(self.basic_FA2,      self.FA2_pool,      self.FA2diet_r,      self.FA2syn_r,      fa_consume,        self.p)
        self.Day(self.basic_FA3,      self.FA3_pool,      self.FA3diet_r,      self.FA3syn_r,      fa_consume,        self.p)
        self.Day(self.basic_Glycerol, self.Glycerol_pool, self.Glyceroldiet_r, self.Glycerolsyn_r, self.Lipid_syn_r,  self.p)
        self.Day(self.basic_HG1,      self.HG1_pool,      self.HG1diet_r,      self.HG1syn_r,      hg_consume,        self.p)
        self.Day(self.basic_HG2,      self.HG2_pool,      self.HG2diet_r,      self.HG2syn_r,      hg_consume,        self.p)

    # ── Plotting ───────────────────────────────────────────────────────────────

    def plot(self):
        snapshot_days = [0, self.days // 2, self.days]
        bar_alphas    = [0.4, 0.7, 1.0]
        bar_labels    = [f'Day {d}' for d in snapshot_days]
        width         = 0.25
        offsets       = [-width, 0, width]

        precursor_colors = {
            'FA1': '#e63946', 'FA2': '#457b9d', 'FA3': '#f4845f',
            'Glycerol': '#2a9d8f', 'HG1': '#e9c46a', 'HG2': '#f4a261',
        }

        # ── Figure 1: Pool stability + precursor isotopes ─────────────────────
        fig1 = plt.figure(figsize=(20, 16))
        fig1.suptitle('Lipid Simulation — Pool Stability & Precursor Isotopes',
                      fontsize=16, fontweight='bold')
        gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.55, wspace=0.35)

        ax_conc = fig1.add_subplot(gs1[0, :])
        ax_conc.plot(self.conc_df['day'], self.conc_df['FA_total'], label='FA total', color='#e63946', linewidth=2)
        ax_conc.plot(self.conc_df['day'], self.conc_df['HG_total'], label='HG total', color='#e9c46a', linewidth=2)
        ax_conc.plot(self.conc_df['day'], self.conc_df['Glycerol'], label='Glycerol', color='#2a9d8f', linewidth=2)
        ax_conc.plot(self.conc_df['day'], self.conc_df['Lipid'],    label='Lipid',    color='#6a0572', linewidth=2)
        ax_conc.set_xlabel('Day');  ax_conc.set_ylabel('Pool Size (molecules)')
        ax_conc.set_title('Concentration vs Time — Pool Stability Check')
        ax_conc.legend(loc='upper right', framealpha=0.7);  ax_conc.grid(True, alpha=0.3)

        for pos, pool_name in zip([gs1[1,0], gs1[1,1], gs1[1,2], gs1[2,0], gs1[2,1], gs1[2,2]],
                                  ['FA1', 'FA2', 'FA3', 'Glycerol', 'HG1', 'HG2']):
            ax  = fig1.add_subplot(pos)
            col = precursor_colors[pool_name]
            sub = self.isotope_df[self.isotope_df['pool'] == pool_name]
            piv = sub.pivot(index='day', columns='label_count', values='fraction').fillna(0)
            for lc in piv.columns:
                alpha = 0.35 + 0.65 * (lc / (piv.columns.max() or 1))
                ax.plot(piv.index, piv[lc], label=f'M+{lc}', color=col, alpha=alpha, linewidth=1.5)
            ax.set_title(f'{pool_name} Isotope Fractions');  ax.set_xlabel('Day');  ax.set_ylabel('Fraction')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.6);  ax.grid(True, alpha=0.3)

        fig1.savefig(os.path.join(SCRIPT_DIR, 'lipid_simulation_combined.png'), dpi=150, bbox_inches='tight')
        print(f"\nFigure 1 saved: lipid_simulation_combined.png")

        # ── Figure 2: Combined lipid pool ─────────────────────────────────────
        fig2, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle('Combined Lipid Pool — Isotope Distribution', fontsize=14, fontweight='bold')

        lip_sub = self.isotope_df[self.isotope_df['pool'] == 'Lipid']
        lip_piv = lip_sub.pivot(index='day', columns='label_count', values='fraction').fillna(0)
        lip_lc  = lip_piv.columns.tolist()

        for snap, off, alp, lbl in zip(snapshot_days, offsets, bar_alphas, bar_labels):
            if snap in lip_piv.index:
                ax_bar.bar([lc + off for lc in lip_lc], lip_piv.loc[snap, lip_lc],
                           width=width, color='#6a0572', alpha=alp, label=lbl)
        ax_bar.set_title('Distribution Snapshots');  ax_bar.set_xlabel('L');  ax_bar.set_ylabel('Fraction')
        ax_bar.set_xticks(lip_lc);  ax_bar.legend(fontsize=8);  ax_bar.grid(True, axis='y', alpha=0.3)

        for lc in lip_lc:
            alpha = 0.3 + 0.7 * (lc / (max(lip_lc) or 1))
            ax_line.plot(lip_piv.index, lip_piv[lc], label=f'M+{lc}',
                         color='#6a0572', alpha=alpha, linewidth=1.8)
        ax_line.set_title('Isotope Fractions Over Time');  ax_line.set_xlabel('Day');  ax_line.set_ylabel('Fraction')
        ax_line.legend(fontsize=7, loc='upper right', framealpha=0.6);  ax_line.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(os.path.join(SCRIPT_DIR, 'lipid_simulation_combined_lipid.png'), dpi=150, bbox_inches='tight')
        print(f"Figure 2 saved: lipid_simulation_combined_lipid.png")

        # ── Figures 3+: One figure per lipid species ──────────────────────────
        species_present = [k for k in self.species_keys
                           if k in self.isotope_df['pool'].values]
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(species_present), 1)))
        species_colors = {k: cmap[i] for i, k in enumerate(species_present)}

        for sp_key in species_present:
            sp_sub = self.isotope_df[self.isotope_df['pool'] == sp_key]
            if sp_sub.empty:
                continue
            sp_piv = sp_sub.pivot(index='day', columns='label_count', values='fraction').fillna(0)
            sp_lc  = sp_piv.columns.tolist()
            col    = species_colors[sp_key]

            fig_sp, (ax_sb, ax_sl) = plt.subplots(1, 2, figsize=(14, 6))
            fig_sp.suptitle(f'Lipid Species: {sp_key}', fontsize=14, fontweight='bold')

            for snap, off, alp, lbl in zip(snapshot_days, offsets, bar_alphas, bar_labels):
                if snap in sp_piv.index:
                    ax_sb.bar([lc + off for lc in sp_lc], sp_piv.loc[snap, sp_lc],
                              width=width, color=col, alpha=alp, label=lbl)
            ax_sb.set_title('Distribution Snapshots');  ax_sb.set_xlabel('L');  ax_sb.set_ylabel('Fraction')
            ax_sb.set_xticks(sp_lc);  ax_sb.legend(fontsize=8);  ax_sb.grid(True, axis='y', alpha=0.3)

            for lc in sp_lc:
                alpha = 0.3 + 0.7 * (lc / (max(sp_lc) or 1))
                ax_sl.plot(sp_piv.index, sp_piv[lc], label=f'M+{lc}',
                           color=col, alpha=alpha, linewidth=1.8)
            ax_sl.set_title('Isotope Fractions Over Time');  ax_sl.set_xlabel('Day');  ax_sl.set_ylabel('Fraction')
            ax_sl.legend(fontsize=7, loc='upper right', framealpha=0.6);  ax_sl.grid(True, alpha=0.3)

            fig_sp.tight_layout()
            sp_fname = sp_key.replace('+', '_') + '.png'
            fig_sp.savefig(os.path.join(SCRIPT_DIR, sp_fname), dpi=150, bbox_inches='tight')
            print(f"Species figure saved: {sp_fname}")

        # ── Figure: Binomial fits (precursors + combined lipid) ───────────────
        fit_names = [p for p in ['FA1', 'FA2', 'FA3', 'Glycerol', 'HG1', 'HG2', 'Lipid']
                     if p in self.fit_results]
        n_fits = len(fit_names)
        ncols  = 3
        nrows  = (n_fits + ncols) // ncols
        fig_fit, axes_fit = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
        fig_fit.suptitle('Binomial Model Fits — Precursors & Combined Lipid',
                         fontsize=15, fontweight='bold')
        axes_fit = axes_fit.flatten()

        for ax, pool_name in zip(axes_fit[:n_fits], fit_names):
            r         = self.fit_results[pool_name]
            lc_colors = plt.cm.viridis(np.linspace(0, 1, r['iso_len']))
            for i in range(r['iso_len']):
                ax.plot(r['times'], r['obs'][:, i], 'o-', color=lc_colors[i],
                        label=f'Obs M+{i}', markersize=3, linewidth=1.2)
                ax.plot(r['times'], r['sim'][:, i], '--', color=lc_colors[i],
                        alpha=0.75, linewidth=1.5)
            ax.set_title(f"{pool_name}  nL={r['nL']}  k={r['k']}  Asyn={r['Asyn']}  R²={r['R2']}",
                         fontsize=9, fontweight='bold')
            ax.set_xlabel('Day');  ax.set_ylabel('Fraction')
            ax.legend(fontsize=6, loc='upper right', framealpha=0.6, ncol=2)
            ax.grid(True, alpha=0.3)

        ax_tbl = axes_fit[n_fits]
        ax_tbl.axis('off')
        tbl_data = [[row['Pool'], row['Fit nL'], row['Fit k'], row['Fit Asyn'], row['R2']]
                    for _, row in self.fit_df[self.fit_df['Pool'].isin(fit_names)].iterrows()]
        tbl = ax_tbl.table(cellText=tbl_data,
                           colLabels=['Pool', 'Fit nL', 'Fit k', 'Fit Asyn', 'R²'],
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False);  tbl.set_fontsize(8);  tbl.scale(1.2, 2.0)
        ax_tbl.set_title('Fit Summary', fontweight='bold', pad=20)

        for ax in axes_fit[n_fits + 1:]:
            ax.axis('off')

        fig_fit.tight_layout()
        fig_fit.savefig(os.path.join(SCRIPT_DIR, 'lipid_simulation_fits.png'), dpi=150, bbox_inches='tight')
        print(f"Fit figure saved: lipid_simulation_fits.png")

        plt.show()


if __name__ == '__main__':
    Cup()