"""
bs_spacing.py — Binomial Spacing (m/z centroid) fit

Key ideas implemented here:
- Natural-isotope PMF (CNOS) with correct +2 isotope accounting (18O/34S as 2-step events).
- Hydrogen-label PMF built on the *full* fractional-binomial support (0..floor(nL)),
  then sliced to the observed iso_len **without renormalizing**. This preserves true
  low-i probabilities and prevents "spacing shrink" artifacts.
- Spacing loss is computed from drift-corrected per-timepoint spacings:
    (Mi - M0)(t)
  which cancels common-mode instrument drift in m/z at each timepoint.

This module is designed to be drop-in usable; if your pipeline already defines
some helpers elsewhere, you can ignore the fallback implementations below.
"""

from __future__ import annotations

import os
import re
import math
import numpy as np
import pandas as pd

from typing import Optional, Tuple, Sequence, Callable, Any

try:
    from scipy.optimize import minimize
except Exception as _e:  # pragma: no cover
    minimize = None  # type: ignore

try:
    import multiprocessing as mp
except Exception:  # pragma: no cover
    mp = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from scipy.special import gammaln

# ----------------------------
# Regex helpers / parsing
# ----------------------------
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\.-]+", "_", str(s))


def parse_num_seq(cell, iso_len: int, pad_with=0.0) -> np.ndarray:
    """
    Robustly parse a cell containing a sequence of numbers.
    Accepts list/tuple/ndarray, or strings like "[1.0, 2.0, ...]".
    Pads/truncates to iso_len.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        arr = np.asarray([], float)
    elif isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=float)
    else:
        s = str(cell)
        nums = [float(m.group()) for m in FLOAT_RE.finditer(s)]
        arr = np.asarray(nums, dtype=float)

    if arr.size >= iso_len:
        return arr[:iso_len].astype(float)
    out = np.full(int(iso_len), pad_with, dtype=float)
    if arr.size:
        out[:arr.size] = arr
    return out


# ----------------------------
# Chemistry helpers (fallback)
# ----------------------------
def get_element_counts(formula: str) -> Tuple[int, int, int, int, int]:
    """
    Fallback elemental parser:
      returns (nC, nH, nN, nO, nS) from something like "C16H32O2".
    If your pipeline already provides get_element_counts(), you can override
    by assigning bs_spacing.get_element_counts = your_function.
    """
    if not isinstance(formula, str):
        formula = str(formula)

    # strip obvious decorations; keep only element tokens + counts
    # (this is intentionally conservative)
    formula = formula.replace(" ", "")
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0}
    for elem, num in ELEM_RE.findall(formula):
        if elem in counts:
            counts[elem] += int(num) if num else 1
    return counts["C"], counts["H"], counts["N"], counts["O"], counts["S"]


# ----------------------------
# Fractional binomial PMF (needed for H-labeling)
# ----------------------------
def fractional_binom_pmf(i: int, nL: float, p: float) -> float:
    """
    Stable fractional binomial PMF:
      C(nL, i) * p^i * (1-p)^(nL-i), with non-integer nL allowed.
    Uses log-gamma to avoid over/underflow.

    IMPORTANT: returns 0 if i > nL (so the true support ends at floor(nL)).
    """
    if not (np.isfinite(nL)) or nL <= 0 or i > nL:
        return 0.0
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    log_c = gammaln(nL + 1.0) - gammaln(i + 1.0) - gammaln(nL - i + 1.0)
    log_pmf = log_c + i * np.log(p) + (nL - i) * np.log1p(-p)
    return float(np.exp(log_pmf))


# ----------------------------
# Isotope masses (lazy init)
# ----------------------------
# If your pipeline defines ELEMENT_ISOTOPES, we will use it.
ELEMENT_ISOTOPES = globals().get("ELEMENT_ISOTOPES", None)

_ISO_READY = False
DELTA_H_MASS = None  # type: ignore
_D13C = _D15N = _D17O = _D18O = _D33S = _D34S = None  # type: ignore


def _require_isotopes():
    global _ISO_READY, DELTA_H_MASS, _D13C, _D15N, _D17O, _D18O, _D33S, _D34S, ELEMENT_ISOTOPES
    if _ISO_READY:
        return
    if ELEMENT_ISOTOPES is None:
        raise RuntimeError(
            "ELEMENT_ISOTOPES is not defined. "
            "Either define it before importing this module, or assign "
            "bs_spacing.ELEMENT_ISOTOPES = <your isotopes dict>."
        )

    def _delta(elem: str, i_heavy: int, i_light: int = 0) -> float:
        return float(ELEMENT_ISOTOPES[elem][i_heavy]["mass"] - ELEMENT_ISOTOPES[elem][i_light]["mass"])

    DELTA_H_MASS = _delta("H", 1, 0)
    _D13C = _delta("C", 1, 0)
    _D15N = _delta("N", 1, 0)
    _D17O = _delta("O", 1, 0)
    _D18O = _delta("O", 2, 0)
    _D33S = _delta("S", 1, 0)
    _D34S = _delta("S", 2, 0)

    _ISO_READY = True


# ----------------------------
# Natural isotope PMF + μ_nat
# ----------------------------
def _binom_prefix_no_renorm(n: int, p: float, L: int) -> np.ndarray:
    """
    Binomial(n,p) PMF for i=0..L-1 WITHOUT renormalizing after truncation.
    Any mass at i>=L is intentionally dropped.
    """
    n = int(max(n, 0))
    L = int(L)
    out = np.zeros(L, dtype=float)
    if L <= 0:
        return out
    p = float(np.clip(p, 0.0, 1.0))
    q = 1.0 - p

    out[0] = q ** n
    m = min(n, L - 1)
    if m == 0:
        return out

    # recurrence: pmf(i+1) = pmf(i) * (n-i)/(i+1) * p/q
    if q == 0.0:
        # degenerate at i=n
        if n < L:
            out[:] = 0.0
            out[n] = 1.0
        return out

    ratio = p / q
    for i in range(m):
        out[i + 1] = out[i] * (n - i) / (i + 1) * ratio
    return out


def _nat_pmf_and_mu(cf: str, iso_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Natural-isotope PMF over nominal neutromer shifts (r=0..L-1) and the corresponding
    centroid mass offsets μ_nat[r] (Daltons) for the CNOS-only distribution.

    Implementation details:
      - P1 is the distribution over x = number of +1 events (13C, 15N, 17O, 33S).
      - P2 is the distribution over y = number of +2 events (18O, 34S).
      - Combine to nominal r via r = x + 2y (correct +2 accounting).
      - No renormalization after truncation; for r<L this does not bias the conditional μ.
    """
    _require_isotopes()
    nC, _nH, nN, nO, nS = get_element_counts(cf)
    L = int(iso_len)

    # natural abundances (hard-coded; replace with your isotope table if desired)
    pC   = 0.0107
    pN   = 0.00366
    pO17 = 0.00038
    pO18 = 0.00205
    pS33 = 0.0075
    pS34 = 0.0421

    # +1 bucket: x
    P1 = np.zeros(L, dtype=float)
    P1[0] = 1.0
    if nC > 0: P1 = np.convolve(P1, _binom_prefix_no_renorm(nC, pC,   L))[:L]
    if nN > 0: P1 = np.convolve(P1, _binom_prefix_no_renorm(nN, pN,   L))[:L]
    if nO > 0: P1 = np.convolve(P1, _binom_prefix_no_renorm(nO, pO17, L))[:L]
    if nS > 0: P1 = np.convolve(P1, _binom_prefix_no_renorm(nS, pS33, L))[:L]

    # +2 bucket: y (only need y up to floor((L-1)/2))
    Y = (L - 1) // 2 + 1
    P2 = np.zeros(Y, dtype=float)
    P2[0] = 1.0
    if nO > 0: P2 = np.convolve(P2, _binom_prefix_no_renorm(nO, pO18, Y))[:Y]
    if nS > 0: P2 = np.convolve(P2, _binom_prefix_no_renorm(nS, pS34, Y))[:Y]

    # combine r = x + 2y
    nat = np.zeros(L, dtype=float)
    for r in range(L):
        ymax = min(r // 2, Y - 1)
        s = 0.0
        for y in range(ymax + 1):
            x = r - 2 * y
            s += P1[x] * P2[y]
        nat[r] = s

    # expected exact mass per +1 and +2 event
    w1 = np.array([nC * pC, nN * pN, nO * pO17, nS * pS33], dtype=float)
    m1 = np.array([_D13C, _D15N, _D17O, _D33S], dtype=float)  # type: ignore
    mu1 = float((w1 @ m1) / w1.sum()) if w1.sum() > 0 else 0.0

    w2 = np.array([nO * pO18, nS * pS34], dtype=float)
    m2 = np.array([_D18O, _D34S], dtype=float)  # type: ignore
    mu2 = float((w2 @ m2) / w2.sum()) if w2.sum() > 0 else 0.0

    # μ_nat[r] = E[mu1*x + mu2*y | x + 2y = r]
    mu_nat = np.zeros(L, dtype=float)
    for r in range(L):
        ymax = min(r // 2, Y - 1)
        num = 0.0
        den = 0.0
        for y in range(ymax + 1):
            x = r - 2 * y
            w = P1[x] * P2[y]
            den += w
            num += w * (mu1 * x + mu2 * y)
        mu_nat[r] = num / den if den > 0 else 0.0

    return nat, mu_nat


# ----------------------------
# Hydrogen-label PMF (fixed)
# ----------------------------
def hydrogen_probabilities(nL: float, k: float, A: float, t: float, p_env: float, L_obs: int) -> np.ndarray:
    """
    Hydrogen distribution Pr(i,t) used in BOTH abundance & spacing fits.

    Fix vs the old implementation:
      - build the fractional-binomial PMF on its *true support* i=0..floor(nL),
        normalize there,
      - mix old/new on that full support,
      - then SLICE to L_obs without re-normalizing.

    This preserves true low-i probabilities (what matters for bins r < L_obs) and
    stops the optimizer from faking smaller nL via truncation artifacts.
    """
    L_obs = int(L_obs)
    if L_obs <= 0:
        return np.zeros(0, dtype=float)

    # Fraction new/old
    f_new = float(A) * (1.0 - float(np.exp(-float(k) * float(t))))
    f_new = float(np.clip(f_new, 0.0, 1.0))
    f_old = 1.0 - f_new

    if not (np.isfinite(nL)) or nL <= 0:
        out = np.zeros(L_obs, dtype=float)
        out[0] = 1.0
        return out

    nmax = int(math.floor(float(nL)))
    L_full = max(L_obs, nmax + 1)

    probs = np.zeros(L_full, dtype=float)
    for i in range(L_full):
        probs[i] = fractional_binom_pmf(i, nL, p_env)

    s = probs.sum()
    if s > 0:
        probs /= s

    out_full = np.zeros(L_full, dtype=float)
    out_full[0] = f_old + f_new * probs[0]
    if L_full > 1:
        out_full[1:] = f_new * probs[1:]

    return out_full[:L_obs]


# ----------------------------
# Predict centroids (m/z)
# ----------------------------
_EPS = 1e-30

def _predict_mz_matrix(
    nL: float,
    k: float,
    A: float,
    times: Sequence[float],
    p_t: Sequence[float],
    z: float,
    nat: np.ndarray,
    mu_nat: np.ndarray,
    L: int,
    base_m0: float,
) -> np.ndarray:
    """
    Predict centroid m/z values (T x L) across timepoints for each neutromer bin.

    Uses:
      conv[r] = (nat * H)[r]
      mu_final[r] = E[ nat_mass_offset + H_mass_offset | bin r ]
    Then converts Daltons -> m/z via /|z| and adds base_m0.

    Critical behavior: we do NOT renormalize after truncation.
    """
    _require_isotopes()
    z_abs = max(abs(float(z)) if np.isfinite(z) and z else 1.0, 1.0)
    dH = float(DELTA_H_MASS)  # type: ignore

    times = np.asarray(times, float)
    p_t = np.asarray(p_t, float)

    L = int(L)
    out = np.full((len(times), L), np.nan, float)
    h_idx = np.arange(L, dtype=float)

    for ti, (t, p_env) in enumerate(zip(times, p_t)):
        H = hydrogen_probabilities(nL, k, A, float(t), float(p_env), L)

        # These convolutions are correct for r < L even though H is sliced,
        # as long as H[:L] was NOT renormalized.
        conv = np.convolve(nat, H)[:L]
        mu_nat_term = np.convolve(nat * mu_nat, H)[:L]
        mu_H_term = np.convolve(nat, (h_idx * dH) * H)[:L]

        mu_final = (mu_nat_term + mu_H_term) / (conv + _EPS)
        out[ti, :] = float(base_m0) + (mu_final / z_abs)

    return out


# ----------------------------
# Loss frame helpers (your plot frame)
# ----------------------------
LOSS_SCALE = 1e8

def _delta_offset_features(mz_matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Drift-corrected per-timepoint spacings:

        features_r(t) = M_r(t) - M_0(t)   for r = 1..L-1

    This cancels common-mode instrument drift δ(t) that shifts all centroids
    equally at a given timepoint. (The `times` argument is unused but kept for
    signature compatibility.)

    Returns shape (T, L-1).
    """
    mz = np.asarray(mz_matrix, float)
    if mz.ndim != 2 or mz.shape[1] < 2:
        return np.zeros((mz.shape[0] if mz.ndim == 2 else 0, 0), dtype=float)
    return mz[:, 1:] - mz[:, [0]]


def _scaled_mse_features(obs_mz: np.ndarray, pred_mz: np.ndarray, times: np.ndarray) -> float:
    d = _delta_offset_features(obs_mz, times) - _delta_offset_features(pred_mz, times)
    m = np.isfinite(d)
    if not m.any():
        return float("inf")
    return float(LOSS_SCALE * np.mean(d[m] ** 2))


def _r2_features(obs_mz: np.ndarray, pred_mz: np.ndarray, times: np.ndarray) -> float:
    yo = _delta_offset_features(obs_mz, times).ravel()
    yp = _delta_offset_features(pred_mz, times).ravel()
    mask = np.isfinite(yo) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    y = yo[mask]; yhat = yp[mask]
    ybar = float(np.mean(y))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2) + 1e-12)
    return 1.0 - ss_res / ss_tot


# ----------------------------
# Bootstrap stats (fallback)
# ----------------------------
def compute_boot_stats(boot: np.ndarray, trim_frac: float = 0.1):
    """
    Fallback bootstrap summary:
      - drops non-finite rows
      - returns trimmed mean, SE, 95% CI
    """
    boot = np.asarray(boot, float)
    if boot.ndim != 2 or boot.size == 0:
        p = boot.shape[1] if boot.ndim == 2 else 3
        nan = np.full(p, np.nan)
        return nan, nan, [(np.nan, np.nan)] * p, 0, np.nan

    ok = np.all(np.isfinite(boot), axis=1)
    boot = boot[ok]
    n = boot.shape[0]
    p = boot.shape[1]
    if n == 0:
        nan = np.full(p, np.nan)
        return nan, nan, [(np.nan, np.nan)] * p, 0, np.nan

    trimmed = np.zeros(p, float)
    se = np.zeros(p, float)
    ci = []
    k = int(math.floor(trim_frac * n))
    for j in range(p):
        x = np.sort(boot[:, j])
        xt = x[k:n-k] if (n - 2*k) >= 1 else x
        trimmed[j] = float(np.mean(xt))
        se[j] = float(np.std(xt, ddof=1) / math.sqrt(max(1, xt.size)))
        ci.append((float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))))
    return trimmed, se, ci, int(n), np.nan


# ----------------------------
# Parallel fit helper (fallback)
# ----------------------------
def _parallel_fit(f, args_list, n_cores=None, pool=None, desc="Fitting"):
    results = []
    if pool is not None:
        for out in tqdm(pool.imap_unordered(f, args_list), total=len(args_list), desc=desc):
            results.append(out)
    elif n_cores and n_cores > 1 and mp is not None:
        try:
            with mp.Pool(int(n_cores)) as p:
                for out in tqdm(p.imap_unordered(f, args_list), total=len(args_list), desc=desc):
                    results.append(out)
        except Exception as e:  # pragma: no cover
            print("[WARN] Multiprocessing fallback:", e)
            for a in tqdm(args_list, desc="Single-core fallback"):
                results.append(f(a))
    else:
        for a in tqdm(args_list, desc="Single-core"):
            results.append(f(a))
    return results


# ----------------------------
# Plot helper (fallback)
# ----------------------------
def _save_fit_plot(uid, grp, plot_dir, prefix, times, obs_mz, pred_mz, fit_params):
    """
    BS fit plot helper.

    - Plots adjacent spacings Δ(Mi - M{i-1})
    - Baseline-subtracted at t0
    - Time-sorted (critical for clean curves)
    - Displays nL, k, Asyn, and *external R2 passed from fit*
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # fit_params now contains (nL, k, Asyn, R2)
    nL, k, Asyn, R2 = fit_params

    times = np.asarray(times, float)
    obs_mz = np.asarray(obs_mz, float)
    pred_mz = np.asarray(pred_mz, float)

    if obs_mz.ndim != 2 or pred_mz.ndim != 2:
        return

    T, L = obs_mz.shape
    if L < 2:
        return

    # -------------------------------------------------------
    # Sort by time so curves are plotted correctly
    # -------------------------------------------------------
    order = np.argsort(times)
    times = times[order]
    obs_mz = obs_mz[order, :]
    pred_mz = pred_mz[order, :]

    # -------------------------------------------------------
    # Adjacent spacing vectors Δ(Mi - M{i-1})
    # -------------------------------------------------------
    obs_adj = obs_mz[:, 1:] - obs_mz[:, :-1]
    pred_adj = pred_mz[:, 1:] - pred_mz[:, :-1]

    # Baseline-subtract at earliest timepoint (now row 0)
    obs_adj = obs_adj - obs_adj[0:1, :]
    pred_adj = pred_adj - pred_adj[0:1, :]

    colors = plt.cm.tab10.colors

    plt.figure(figsize=(10, 6))
    for r in range(1, L):
        color = colors[(r - 1) % len(colors)]

        plt.plot(
            times,
            obs_adj[:, r - 1],
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"Obs Δ(M{r}-M{r-1})"
        )

        plt.plot(
            times,
            pred_adj[:, r - 1],
            "--",
            color=color,
            linewidth=2,
            label=f"Fit Δ(M{r}-M{r-1})"
        )

    # -------------------------------------------------------
    # Title with fit parameters AND externally computed R2
    # -------------------------------------------------------
    title = (
        f"{uid} / {grp} [{prefix}]\n"
        f"nL={nL:.2f},  k={k:.3f},  Asyn={Asyn:.3f},  R²={R2:.3f}"
    )
    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Δ(m/z spacing) from t0")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    # Save
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"{uid}_{grp}_{prefix}_adj_spacing.png".replace("/", "_")
    plt.savefig(os.path.join(plot_dir, fname), dpi=200)
    plt.close()


    # -----------------------------
    # Save file
    # -----------------------------
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"{uid}_{grp}_{prefix}_adj_spacing.png".replace("/", "_")
    fpath = os.path.join(plot_dir, fname)
    plt.savefig(fpath, dpi=200)
    plt.close()
    print(f"[BS] Saved adjacent-spacing plot → {fpath}")


# ----------------------------
# Core fitter (single group)
# ----------------------------
def _fit_spacing_one(args):
    """
    Fit (nL, k, Asyn) by matching predicted centroid spacings over time.

    Loss frame (drift-corrected, per timepoint):
        features_r(t) = M_r(t) - M_0(t)   for r = 1..L-1

    This cancels common-mode instrument drift at each timepoint while still
    letting (nL, k, Asyn) control the time evolution through the H-only PMF.
    """
    if minimize is None:
        raise RuntimeError("scipy is required for fitting (scipy.optimize.minimize).")

    (ID_COL, GROUP_COL, FORMULA_COL, TIME_COL, Z_COL, BOOT_SEED,
     lipid_uid, sample_group, adduct_cf, sub_df,
     asymptote_mode, user_Asyn_value, n_boot, save_dir) = args

    # inputs
    times = pd.to_numeric(sub_df[TIME_COL], errors="coerce").to_numpy(float)
    p_t   = pd.to_numeric(sub_df["enrichment"], errors="coerce").to_numpy(float)
    z_med = float(np.nanmedian(pd.to_numeric(sub_df[Z_COL], errors="coerce")))
    L     = int(np.nanmedian(pd.to_numeric(sub_df["iso_len"], errors="coerce")))

    mz_rows = [parse_num_seq(v, L, pad_with=np.nan) for v in sub_df["mz_array"].to_numpy()]
    obs_mz  = np.vstack(mz_rows)

    # sort by time (stable baseline choice)
    order = np.argsort(times)
    times = times[order]
    p_t   = p_t[order]
    obs_mz = obs_mz[order, :]

    base_m0 = float(obs_mz[0, 0]) if (obs_mz.size and np.isfinite(obs_mz[0, 0])) else float(np.nanmedian(obs_mz[:, 0]))
    nat, mu_nat = _nat_pmf_and_mu(str(adduct_cf), L)

    # bounds & inits
    nH_vals = pd.to_numeric(sub_df.get("nH", pd.Series([1])), errors="coerce")
    nH_max  = int(max(1, np.nanmax(nH_vals))) + 10 #allows wondering slightly past nH
    nL0     = max(0.5, min(10.0, nH_max / 2))
    k0, A0  = 0.3, 0.8

    def loss_free(x):
        nL, k, A = x
        if not (0.1 <= nL <= nH_max and 0.01 <= k <= 5.0 and 0.0 <= A <= 1.0):
            return float("inf")
        pred = _predict_mz_matrix(nL, k, A, times, p_t, z_med, nat, mu_nat, L, base_m0)
        return _scaled_mse_features(obs_mz, pred, times)

    if asymptote_mode == "fixed" and user_Asyn_value is not None:
        A_fixed = float(user_Asyn_value)

        def loss_fixed(x):
            nL, k = x
            if not (0.1 <= nL <= nH_max and 0.01 <= k <= 5.0):
                return float("inf")
            pred = _predict_mz_matrix(nL, k, A_fixed, times, p_t, z_med, nat, mu_nat, L, base_m0)
            return _scaled_mse_features(obs_mz, pred, times)

        res = minimize(
            loss_fixed, x0=[nL0, k0],
            bounds=[(0.1, nH_max), (0.01, 5.0)],
            method="L-BFGS-B",
            options={"disp": False, "maxiter": 200, "gtol": 1e-12, "ftol": 1e-12, "eps": 1e-4},
        )
        best = (float(res.x[0]), float(res.x[1]), A_fixed)
    else:
        res = minimize(
            loss_free, x0=[nL0, k0, A0],
            bounds=[(0.1, nH_max), (0.01, 5.0), (0.0, 1.0)],
            method="L-BFGS-B",
            options={"disp": False, "maxiter": 200, "gtol": 1e-12, "ftol": 1e-12, "eps": 1e-4},
        )
        best = (float(res.x[0]), float(res.x[1]), float(res.x[2]))

    # bootstrap
    rng = np.random.default_rng(int(BOOT_SEED) if BOOT_SEED is not None else 0)
    T = len(times)
    boot = []
    for _ in range(int(n_boot)):
        idx = np.empty(T, dtype=int)
        idx[0] = 0
        idx[1:] = rng.integers(0, T, size=T-1)

        tb = times[idx]
        pb = p_t[idx]
        ob = obs_mz[idx, :]

        o2 = np.argsort(tb)
        tb = tb[o2]; pb = pb[o2]; ob = ob[o2, :]

        # base_m0 cancels in the spacing loss, but keep a stable anchor for absolute m/z prediction/plots
        base = base_m0

        if asymptote_mode == "fixed" and user_Asyn_value is not None:
            A_fixed = float(user_Asyn_value)

            def loss_b(x):
                pred = _predict_mz_matrix(float(x[0]), float(x[1]), A_fixed, tb, pb, z_med, nat, mu_nat, L, base)
                return _scaled_mse_features(ob, pred, tb)

            xb = minimize(
                loss_b, x0=[best[0], best[1]],
                bounds=[(0.1, nH_max), (0.01, 5.0)],
                method="L-BFGS-B",
                options={"disp": False, "maxiter": 200, "gtol": 1e-12, "ftol": 1e-12, "eps": 1e-4},
            ).x
            boot.append((float(xb[0]), float(xb[1]), A_fixed))
        else:
            def loss_b(x):
                pred = _predict_mz_matrix(float(x[0]), float(x[1]), float(x[2]), tb, pb, z_med, nat, mu_nat, L, base)
                return _scaled_mse_features(ob, pred, tb)

            xb = minimize(
                loss_b, x0=[best[0], best[1], best[2]],
                bounds=[(0.1, nH_max), (0.01, 5.0), (0.0, 1.0)],
                method="L-BFGS-B",
                options={"disp": False, "maxiter": 200, "gtol": 1e-12, "ftol": 1e-12, "eps": 1e-4},
            ).x
            boot.append((float(xb[0]), float(xb[1]), float(xb[2])))

    boot = np.asarray(boot, float)
    boot_trimmed, se, ci, n_kept_cols, r2_gauss = compute_boot_stats(boot)

    # final prediction + R2 in the same feature frame
    pred_best = _predict_mz_matrix(best[0], best[1], best[2], times, p_t, z_med, nat, mu_nat, L, base_m0)
    R2_rel = _r2_features(obs_mz, pred_best, times)

    # export predicted relative spacing (plot frame)
    pred_feat = _delta_offset_features(pred_best, times)
    df_pred = pd.DataFrame(pred_feat, columns=[f"dM{r+1}" for r in range(pred_feat.shape[1])])
    df_pred.insert(0, "time", times)

    out_path = os.path.join(save_dir, "BS_predicted_spacing_csvs")
    os.makedirs(out_path, exist_ok=True)
    fname = safe_filename(f"{lipid_uid}_{sample_group}_{adduct_cf}_pred_rel_spacing.csv")
    df_pred.to_csv(os.path.join(out_path, fname), index=False)

    return {
        "BS_nL": best[0],
        "BS_rate": best[1],
        "BS_Asyn": best[2],
        "BS_fit_R2": R2_rel,
        "BS_resampling_samples": boot,
        "BS_resampling_trimmed": boot_trimmed,
        "BS_n_kept_boot": n_kept_cols,
        "BS_nL_SE": se[0],
        "BS_rate_SE": se[1],
        "BS_Asyn_SE": se[2],
        "BS_nL_CI95": ci[0],
        "BS_rate_CI95": ci[1],
        "BS_Asyn_CI95": ci[2],
        "nH": nH_max,
        ID_COL: lipid_uid,
        GROUP_COL: sample_group,
        FORMULA_COL: adduct_cf,
        "signal_noise": sub_df["signal_noise"].iloc[0] if "signal_noise" in sub_df.columns else np.nan,
    }


# ----------------------------
# Public entry: perform_spacing_fit
# ----------------------------
def perform_spacing_fit(
    collapsed: pd.DataFrame,
    ID_COL: str,
    save_dir: str,
    asymptote_mode: str,
    user_Asyn_value: Optional[float] = None,
    n_boot: int = 1000,
    tp_min: int = 3,
    cores: int = 2,
    pool=None,
    FORMULA_COL: str = None,
    TP_MIN: int = None,  # kept for backward-compat; not used
    GROUP_COL: str = None,
    TIME_COL: str = None,
    BOOT_SEED: int = None,
    Z_COL: str = None,
) -> pd.DataFrame:
    """
    Spacing (m/z centroid) fit that matches the progression of empirical m/z values over time.
    Mirrors perform_abundance_fit: parallel fit + plots + CSV outputs.

    NOTE: This version is modified to:
      - use the corrected H PMF handling (full support, no slice renorm)
      - compute the spacing loss from absolute centroids (no double baseline subtraction)
    """
    prefix = "BS"

    if collapsed is None or len(collapsed) == 0:
        print(f"[{prefix}] Empty input — skipping.")
        return pd.DataFrame()

    # Pre-filter flagged rows
    if "Flag" in collapsed.columns:
        before = len(collapsed)
        collapsed = collapsed.loc[collapsed["Flag"].astype(str).str.startswith("PASS")].copy()
        print(f"[{prefix}] Skipping flagged rows → kept {len(collapsed)}/{before} PASS*")
    else:
        print(f"[{prefix}] No 'Flag' column found — fitting all rows.")

    if collapsed.empty:
        print(f"[{prefix}] No unflagged rows available — skipping.")
        return pd.DataFrame()

    # stricter inclusion before fitting
    keep_mask = (
        (collapsed.get("Flag", "PASS").astype(str).str.startswith("PASS"))
        & collapsed.get("D_spectrum", pd.Series([np.nan] * len(collapsed))).notna()
        & (pd.to_numeric(collapsed.get("TP_count", 0), errors="coerce") >= int(tp_min))
    )
    before_filter = len(collapsed)
    collapsed = collapsed.loc[keep_mask].copy()
    print(
        f"[{prefix}] Filtered analytes before fitting → kept {len(collapsed)}/{before_filter} "
        f"(Flag=PASS*, TP_count≥{tp_min}, valid D_spectrum)"
    )

    if collapsed.empty:
        print(f"[{prefix}] No analytes passed pre-fit filters — skipping {prefix} fitting.")
        return pd.DataFrame()

    # Output directories
    plot_dir = os.path.join(save_dir, f"{prefix}_fit_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Build argument list
    if FORMULA_COL is None or GROUP_COL is None or TIME_COL is None or Z_COL is None:
        raise ValueError("FORMULA_COL, GROUP_COL, TIME_COL, and Z_COL must be provided.")

    groups = list(collapsed.groupby([ID_COL, GROUP_COL, FORMULA_COL], sort=False))
    args_list = [
        (ID_COL, GROUP_COL, FORMULA_COL, TIME_COL, Z_COL, BOOT_SEED,
         uid, grp, cf, sub,
         asymptote_mode, user_Asyn_value, n_boot, save_dir)
        for (uid, grp, cf), sub in groups
    ]

    total = len(args_list)
    print(f"[{prefix}] Fitting {total} analyte-group combinations ({cores} cores)")

    # Parallel fitting
    results = _parallel_fit(
        _fit_spacing_one,
        args_list,
        n_cores=cores,
        pool=pool,
        desc=f"{prefix} fitting"
    )

    rows = [r for r in results if r is not None]
    df_fit = pd.DataFrame(rows)
    if df_fit.empty:
        print(f"[{prefix}] No successful spacing fits — skipping plots/outputs.")
        return df_fit

    # Plotting (absolute centroid curves)
    for (uid, grp, cf), sub in tqdm(
        collapsed.groupby([ID_COL, GROUP_COL, FORMULA_COL], sort=False),
        desc=f"{prefix} plotting"
    ):
        row_match = df_fit[
            (df_fit[ID_COL] == uid)
            & (df_fit[GROUP_COL] == grp)
            & (df_fit[FORMULA_COL] == cf)
        ]
        if row_match.empty:
            continue

        nL, k, A = row_match.iloc[0][[f"{prefix}_nL", f"{prefix}_rate", f"{prefix}_Asyn"]].astype(float)

        # rebuild sorted arrays for plot consistency
        times = pd.to_numeric(sub[TIME_COL], errors="coerce").to_numpy(float)
        order = np.argsort(times)
        times = times[order]
        p_t = pd.to_numeric(sub["enrichment"], errors="coerce").to_numpy(float)[order]



        L = int(np.nanmedian(pd.to_numeric(sub["iso_len"], errors="coerce")))
        z_med = float(np.nanmedian(pd.to_numeric(sub[Z_COL], errors="coerce")))
        obs = np.vstack([parse_num_seq(v, L, pad_with=np.nan) for v in sub["mz_array"].to_numpy()])[order]
        base0 = float(obs[0, 0]) if (obs.size and np.isfinite(obs[0, 0])) else float(np.nanmedian(obs[:, 0]))

        nat, mu_nat = _nat_pmf_and_mu(str(cf), L)
        pred = _predict_mz_matrix(nL, k, A, times, p_t, z_med, nat, mu_nat, L, base0)

        _save_fit_plot(uid, grp, plot_dir, prefix, times, obs, pred, (nL, k, A))

    return df_fit


__all__ = [
    "perform_spacing_fit",
    "_fit_spacing_one",
    "_predict_mz_matrix",
    "_nat_pmf_and_mu",
    "hydrogen_probabilities",
    "_delta_offset_features",
    "_scaled_mse_features",
    "_r2_features",
    "parse_num_seq",
    "fractional_binom_pmf",
    "get_element_counts",
]
