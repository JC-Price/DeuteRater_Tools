#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lipid simulation with concentration tracking, isotope pattern tracking,
and matplotlib visualization using pandas.

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
from math import lgamma
from itertools import combinations_with_replacement
import matplotlib
matplotlib.use('Qt5Agg')  # Spyder-compatible interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Fitting functions ─────────────────────────────────────────────────────────

def fractional_binom_pmf(i, nL, p):
    if nL <= 0 or i > nL:
        return 0.0
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    log_c = lgamma(nL + 1.0) - lgamma(i + 1.0) - lgamma(nL - i + 1.0)
    return float(np.exp(log_c + i * np.log(p) + (nL - i) * np.log1p(-p)))


def convolved_isotope_pmf(i, nL_FA, nL_Gly, nL_HG, p):
    """
    Explicit convolution:
    i = i_FA(slot1) + i_FA(slot2) + i_Gly + i_HG
    FA slots are identical.
    """
    total = 0.0

    for i_FA1 in range(i + 1):
        for i_FA2 in range(i + 1 - i_FA1):
            for i_Gly in range(i + 1 - i_FA1 - i_FA2):
                i_HG = i - i_FA1 - i_FA2 - i_Gly
                if i_HG < 0:
                    continue

                total += (
                    fractional_binom_pmf(i_FA1, nL_FA,  p) *
                    fractional_binom_pmf(i_FA2, nL_FA,  p) *
                    fractional_binom_pmf(i_Gly, nL_Gly, p) *
                    fractional_binom_pmf(i_HG,  nL_HG,  p)
                )

    return total

def component_isotope_pmf(i_k, nL_k, A_k, k_k, t, p):
    f_k = A_k * (1.0 - np.exp(-k_k * t))
    return (
        (1.0 - f_k) * (1.0 if i_k == 0 else 0.0)
        +
        f_k * fractional_binom_pmf(i_k, nL_k, p)
    )

def prepare_species_data(isotope_df, species_keys):
    data = {}
    for sp in species_keys:
        sub = isotope_df[isotope_df['pool'] == sp]
        if sub.empty:
            continue

        piv = (
            sub
            .pivot(index='day', columns='label_count', values='fraction')
            .fillna(0)
        )
        piv = piv[[i for i in piv.columns if i < 4]]

        if piv.shape[0] >= 3:
            data[sp] = {
                "times": piv.index.to_numpy(float),
                "obs": piv.values,
                "iso_len": piv.shape[1],
            }
    return data


def build_model_distribution(nL, k, Asyn, times, iso_len, p_label):
    p = float(np.clip(p_label, 1e-12, 1.0 - 1e-12))
    rows = []
    for t in times:
        binom = np.array([fractional_binom_pmf(i, nL, p) for i in range(iso_len)])
        s = binom.sum()
        if s > 0:
            binom /= s
        f_new = Asyn * (1.0 - np.exp(-k * t))
        f_old = 1.0 - f_new
        baseline = np.zeros(iso_len)
        baseline[0] = 1.0
        dist = f_old * baseline + f_new * binom
        rows.append(dist)
    sim = np.vstack(rows)
    row_sums = sim.sum(axis=1, keepdims=True)
    sim /= np.where(row_sums > 0, row_sums, 1.0)
    return sim

def build_convolved_model_distribution(
    nL_FA, nL_Gly, nL_HG,
    k, Asyn,
    times,
    iso_len,
    p
):
    rows = []

    for t in times:
        Q = np.array([
            convolved_isotope_pmf(i, nL_FA, nL_Gly, nL_HG, p)
            for i in range(iso_len)
        ])

        if Q.sum() > 0:
            Q /= Q.sum()

        f_new = Asyn * (1.0 - np.exp(-k * t))
        f_old = 1.0 - f_new

        baseline = np.zeros(iso_len)
        baseline[0] = 1.0

        rows.append(f_old * baseline + f_new * Q)

    return np.vstack(rows)

def convolved_isotope_pmf_time(
    i, t,
    FAa, FAb, HGz,
    FA_params, HG_params,
    nL_Gly, A_Gly, k_Gly,
    p
):
    nL_FAa, A_FAa, k_FAa = FA_params[FAa]
    nL_FAb, A_FAb, k_FAb = FA_params[FAb]
    nL_HG,  A_HG,  k_HG  = HG_params[HGz]

    total = 0.0
    for i_a in range(i + 1):
        for i_b in range(i + 1 - i_a):
            for i_g in range(i + 1 - i_a - i_b):
                i_h = i - i_a - i_b - i_g
                if i_h < 0:
                    continue

                total += (
                    component_isotope_pmf(i_a, nL_FAa, A_FAa, k_FAa, t, p) *
                    component_isotope_pmf(i_b, nL_FAb, A_FAb, k_FAb, t, p) *
                    component_isotope_pmf(i_g, nL_Gly, A_Gly, k_Gly, t, p) *
                    component_isotope_pmf(i_h, nL_HG,  A_HG,  k_HG,  t, p)
                )
    return total


def build_convolved_model_distribution_full(
    nL_FA1, A_FA1, k_FA1,
    nL_FA2, A_FA2, k_FA2,
    nL_Gly, A_Gly, k_Gly,
    nL_HG,  A_HG,  k_HG,
    A_L, k_L,
    times, iso_len, p
):
    rows = []

    for t in times:
        f_L = A_L * (1.0 - np.exp(-k_L * t))

        Q = np.array([
            convolved_isotope_pmf_time(
                i, t,
                nL_FA1, A_FA1, k_FA1,
                nL_FA2, A_FA2, k_FA2,
                nL_Gly, A_Gly, k_Gly,
                nL_HG,  A_HG,  k_HG,
                p
            )
            for i in range(iso_len)
        ])

        if Q.sum() > 0:
            Q /= Q.sum()

        baseline = np.zeros(iso_len)
        baseline[0] = 1.0

        rows.append((1.0 - f_L) * baseline + f_L * Q)

    return np.vstack(rows)

def build_model_full(
    FAa, FAb, HGz,
    FA_params, HG_params,
    nL_Gly, A_Gly, k_Gly,
    A_L, k_L,
    times, iso_len, p
):
    rows = []
    for t in times:
        f_L = A_L * (1.0 - np.exp(-k_L * t))

        Q = np.array([
            convolved_isotope_pmf_time(
                i, t,
                FAa, FAb, HGz,
                FA_params, HG_params,
                nL_Gly, A_Gly, k_Gly,
                p
            )
            for i in range(iso_len)
        ])

        if Q.sum() > 0:
            Q /= Q.sum()

        base = np.zeros(iso_len)
        base[0] = 1.0
        rows.append((1 - f_L) * base + f_L * Q)

    return np.vstack(rows)


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

def fit_lipid_species_convolved(isotope_df, species_name, p_label):
    sub = isotope_df[isotope_df['pool'] == species_name]
    if sub.empty:
        return None

    piv = (
        sub
        .pivot(index='day', columns='label_count', values='fraction')
        .fillna(0)
    )
    piv = piv[[i for i in piv.columns if i < 4]]  # M+0..M+3
    if piv.shape[0] < 3:
        return None

    times = piv.index.to_numpy(float)
    obs   = piv.values
    iso_len = obs.shape[1]

    def loss(x):
        (
            nL_FA1, A_FA1, k_FA1,
            nL_FA2, A_FA2, k_FA2,
            nL_Gly, A_Gly, k_Gly,
            nL_HG,  A_HG,  k_HG,
            A_L,    k_L
        ) = x

        # --- constraints ---
        if (
            nL_FA1 <= 0 or nL_FA2 <= 0 or nL_Gly <= 0 or nL_HG <= 0 or
            k_FA1 <= 0 or k_FA2 <= 0 or k_Gly <= 0 or k_HG <= 0 or k_L <= 0 or
            not (0 <= A_FA1 <= 1) or not (0 <= A_FA2 <= 1) or
            not (0 <= A_Gly <= 1) or not (0 <= A_HG <= 1) or
            not (0 <= A_L <= 1)
        ):
            return np.inf

        sim = build_convolved_model_distribution_full(
            nL_FA1, A_FA1, k_FA1,
            nL_FA2, A_FA2, k_FA2,
            nL_Gly, A_Gly, k_Gly,
            nL_HG,  A_HG,  k_HG,
            A_L, k_L,
            times, iso_len, p_label
        )

        return np.sum((obs - sim) ** 2)

    # --- initial guess ---
    x0 = [
        5.0, 0.5, 0.05,   # FA1
        5.0, 0.5, 0.05,   # FA2
        2.0, 0.5, 0.05,   # Gly
        2.0, 0.5, 0.05,   # HG
        0.5, 0.05         # lipid-level
    ]

    # --- bounds ---
    bounds = [
        (0.1, 30.0), (0.0, 1.0), (0.001, 10.0),  # FA1
        (0.1, 30.0), (0.0, 1.0), (0.001, 10.0),  # FA2
        (0.1, 10.0), (0.0, 1.0), (0.001, 10.0),  # Gly
        (0.1, 10.0), (0.0, 1.0), (0.001, 10.0),  # HG
        (0.0, 1.0),  (0.001, 10.0)               # lipid-level
    ]

    res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")

    if not res.success:
        return None

    return {
        "Species": species_name,
        "nL_FA1": res.x[0],
        "A_FA1":  res.x[1],
        "k_FA1":  res.x[2],
        "nL_FA2": res.x[3],
        "A_FA2":  res.x[4],
        "k_FA2":  res.x[5],
        "nL_Gly": res.x[6],
        "A_Gly":  res.x[7],
        "k_Gly":  res.x[8],
        "nL_HG":  res.x[9],
        "A_HG":   res.x[10],
        "k_HG":   res.x[11],
        "A_L":    res.x[12],
        "k_L":    res.x[13],
        "Loss":   res.fun,
    }

def fit_all_lipids_shared_components(isotope_df, species_keys, p_label):

    # --- prepare data ---
    data = {}
    for sp in species_keys:
        sub = isotope_df[isotope_df['pool'] == sp]
        if sub.empty:
            continue
        piv = sub.pivot(index='day', columns='label_count', values='fraction').fillna(0)
        piv = piv[[i for i in piv.columns if i < 4]]
        if piv.shape[0] >= 3:
            data[sp] = {
                "times": piv.index.to_numpy(float),
                "obs": piv.values,
                "iso_len": piv.shape[1],
            }

    species_list = list(data.keys())
    nS = len(species_list)

    # --- parameter vector ---
    # FA1, FA2, FA3 (each: nL, A, k)
    # Gly (nL, A, k)
    # HG1, HG2 (each: nL, A, k)
    # Lipid-level A_L, k_L per species

    def loss(x):
        idx = 0

        FA_params = {}
        for fa in ["FA1", "FA2", "FA3"]:
            FA_params[fa] = tuple(x[idx:idx+3])
            idx += 3

        nL_Gly, A_Gly, k_Gly = x[idx:idx+3]
        idx += 3

        HG_params = {}
        for hg in ["HG1", "HG2"]:
            HG_params[hg] = tuple(x[idx:idx+3])
            idx += 3

        lipid_params = x[idx:]

        # constraints
        for nL,A,k in list(FA_params.values()) + list(HG_params.values()) + [(nL_Gly,A_Gly,k_Gly)]:
            if nL <= 0 or k <= 0 or not (0 <= A <= 1):
                return np.inf

        total = 0.0
        for i, sp in enumerate(species_list):
            A_L = lipid_params[2*i]
            k_L = lipid_params[2*i + 1]
            if k_L <= 0 or not (0 <= A_L <= 1):
                return np.inf

            FAa, FAb, HGz = sp.split("+")
            d = data[sp]

            sim = build_model_full(
                FAa, FAb, HGz,
                FA_params, HG_params,
                nL_Gly, A_Gly, k_Gly,
                A_L, k_L,
                d["times"], d["iso_len"], p_label
            )

            total += np.sum((d["obs"] - sim) ** 2)

        return total

    # --- initial guess ---
    x0 = []
    for _ in range(3):  # FA1,2,3
        x0 += [5.0, 0.5, 0.05]
    x0 += [2.0, 0.5, 0.05]  # Gly
    for _ in range(2):      # HG1, HG2
        x0 += [2.0, 0.5, 0.05]
    for _ in species_list:
        x0 += [0.5, 0.05]  # A_L, k_L

    # --- bounds ---
    bounds = (
        [(0.1,30),(0,1),(0.001,10)] * 3 +
        [(0.1,10),(0,1),(0.001,10)] +
        [(0.1,10),(0,1),(0.001,10)] * 2 +
        [(0,1),(0.001,10)] * nS
    )

    res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
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

def fit_lipid_species(isotope_df, species_name, p_label):
    """
    Fit nL, k, Asyn for a single lipid species using
    all time points and M+0..M+3 only.
    """
    sub = isotope_df[isotope_df['pool'] == species_name]
    if sub.empty:
        return None

    piv = (
        sub
        .pivot(index='day', columns='label_count', values='fraction')
        .fillna(0)
    )

    if piv.shape[0] < 3:
        return None

    # restrict to first 4 isotopologues
    piv = piv[[i for i in piv.columns if i < 4]]
    iso_len = piv.shape[1]

    times = piv.index.to_numpy(float)
    obs   = piv.values

    nL_max = float(iso_len * 6)

    def loss(x):
        nL, k, A = x
        if not (0.1 <= nL <= nL_max and 0.001 <= k <= 10.0 and 0.0 < A <= 1.0):
            return np.inf
        sim = build_model_distribution(nL, k, A, times, iso_len, p_label)
        return np.sum((obs - sim) ** 2)

    best = None
    for nL0 in [0.5, iso_len, 2 * iso_len]:
        for k0 in [0.05, 0.3, 1.0]:
            res = minimize(
                loss,
                x0=[nL0, k0, 0.7],
                bounds=[(0.1, nL_max), (0.001, 10.0), (0.0, 1.0)],
                method="L-BFGS-B",
            )
            if best is None or res.fun < best.fun:
                best = res

    nL, k, A = best.x
    sim_best = build_model_distribution(nL, k, A, times, iso_len, p_label)

    ss_res = np.sum((obs - sim_best) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "species": species_name,
        "nL": round(nL, 3),
        "k": round(k, 4),
        "Asyn": round(A, 3),
        "R2": round(R2, 4),
        "times": times,
        "obs": obs,
        "sim": sim_best,
    }
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

        # ── Initial pools (~1000 each) ─────────────────────────────────────────
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
        
        
        # ── Compound lipid deconvolution ───────────────────────────────────────────
        # ── Compound lipid deconvolution (explicit convolution) ─────────────
        
        # ── GLOBAL compound lipid fit (shared components) ───────────────────
        
        res = fit_all_lipids_shared_components(
            self.isotope_df,
            self.species_keys,
            self.p / 100.0
        )
        
        self.global_fit_result = res
        x = res.x
                
        
        
        FA1 = x[0:3]     # nL, Asyn, k
        FA2 = x[3:6]    
        FA3 = x[6:9]
        Gly = x[9:12]
        HG1 = x[12:15]
        HG2 = x[15:18]

        
        # ── Constituent parameter table (from GLOBAL fit) ───────────────────
        
        self.component_fit_df = pd.DataFrame([
            {"Component": "FA1", "nL": FA1[0], "Asyn": FA1[1], "k": FA1[2]},
            {"Component": "FA2", "nL": FA2[0], "Asyn": FA2[1], "k": FA2[2]},
            {"Component": "FA3", "nL": FA3[0], "Asyn": FA3[1], "k": FA3[2]},
            {"Component": "Glycerol", "nL": Gly[0], "Asyn": Gly[1], "k": Gly[2]},
            {"Component": "HG1", "nL": HG1[0], "Asyn": HG1[1], "k": HG1[2]},
            {"Component": "HG2", "nL": HG2[0], "Asyn": HG2[1], "k": HG2[2]},
        ])
        
        if verbose:
            print("\n=== Constituent Parameter Estimates (GLOBAL FIT) ===")
            print(self.component_fit_df.to_string(index=False))
                

        # ── Lipid-level turnover results ────────────────────────────────────
        
        offset = 18  # first 18 entries are shared components
        lipid_rows = []
        
        for i, sp in enumerate(self.species_keys):
            A_L = x[offset + 2*i]
            k_L = x[offset + 2*i + 1]
            lipid_rows.append({
                "Species": sp,
                "Asyn_L": A_L,
                "k_L": k_L
            })
        
        self.lipid_fit_df = pd.DataFrame(lipid_rows)
        
        if verbose:
            print("\n=== Lipid-Level Kinetic Parameters ===")
            print(self.lipid_fit_df.to_string(index=False))

        # ── Fit pools ─────────────────────────────────────────────────────────
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
        
        

    

    
    

Cup()