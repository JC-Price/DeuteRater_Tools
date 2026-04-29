"""
prep.py
-------
This module performs the post hoc quantitative analysis for DeuteRater-derived data.

After initial extraction, each lipid’s abundance, turnover rate (k), and asymptote (A)
are analyzed through a combination of robust regression, parametric error propagation,
and classical hypothesis testing.

Abundances
Raw DeuteRater abundances are linearly aligned between experiments in log₂ space to correct
for systematic bias. The fitted line (ŷ = a·x + b) is exponentiated back to natural scale so
that all abundances undergo an equivalent transformation. Residual normality can be verified
in automatically generated diagnostic plots. The corrected values are then used for Welch’s
two-sample t-tests with Benjamini–Hochberg false discovery rate (FDR) correction, forming
the basis for volcano plots of abundance changes.

Asymptote comparison
Each lipid’s incorporation curve is modeled as  
  *y(t) = A – A·exp(–k·t)*,  
where *A* represents the plateau abundance reached at infinite labeling time.  
Following nonlinear regression in DeuteRater, each fit provides the asymptote value (A),
its standard error (SEₐ), and its degrees of freedom (νₐ) derived from the covariance matrix
of the least-squares fit. Between-group differences in A are tested using a Welch-style
t-statistic,  

  *t = (Aₑₓₚ – A꜀ₜₗ) / √(SEₑₓₚ² + SE꜀ₜₗ²)*,  

with effective degrees of freedom calculated via the Welch–Satterthwaite equation,  

  *ν_eff = (SEₑₓₚ² + SE꜀ₜₗ²)² / (SEₑₓₚ⁴/νₑₓₚ + SE꜀ₜₗ⁴/ν꜀ₜₗ)*.  

Two-sided p-values are derived from the Student’s t distribution with ν_eff degrees of freedom,
yielding parametric significance estimates for asymptote comparisons.

Turnover rate comparison
The same parametric approach is applied to the rate constant *k*, which governs the exponential
rise toward equilibrium. For each lipid, DeuteRater provides *k*, its SEₖ, and νₖ. These are
combined using the identical Welch framework above to obtain p-values for *k_exp / k_ctl*.
This method preserves the model-based uncertainty from curve fitting while avoiding
resampling-related stochastic variance.

nL comparison
For each lipid, replicate lists of fitted nL values (the number of incorporable deuterons)
are compared between groups using a replicate-aware Welch’s t-test. Fold-changes are computed
as mean(*nL_exp*) / mean(*nL_ctl*) and reported in both linear and log₂ form.

Flux derivation
Total flux combines abundance and turnover rate terms:  
  *FC_flux = FC_rate × FC_abn*,  
and its p-value is obtained via Fisher’s method for combining independent probabilities.
Using the asymptote-derived synthesis fraction (*A_syn = A*), synthesis and dietary fluxes are
further separated as:  
  *FC_synth_flux = FC_flux × (A_exp / A_ctl)*,  
  *FC_diet_flux = FC_flux × ((1 – A_exp) / (1 – A_ctl))*.  
These capture relative changes in endogenously synthesized versus serum-derived lipid influx.

Together, these metrics define the complete post hoc statistical framework for DeuteRater
outputs, integrating abundance, turnover, asymptote, n-value, and flux significance
in a unified, model-aware analysis.
"""




import os
import re
import ast
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.optimize import curve_fit
from scipy import stats
import warnings
from sklearn.linear_model import HuberRegressor
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import numpy, pandas, scipy, statsmodels
import traceback
import sys
import platform
from scipy.stats import ttest_ind
from scipy.stats import t
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import linregress
from numpy.linalg import lstsq
from scipy.optimize import least_squares
from scipy.stats import norm



# ============================================================================
# Stats helpers
# ============================================================================
def _apply_transform_list(df, cols, slope, intercept, suffix="_corrected_to_ctl"):
    if not cols:
        return df
    for c in cols:
        if c in df.columns:
            try:
                df[c + suffix] = slope * pd.to_numeric(df[c], errors="coerce") + intercept
            except Exception:
                df[c + suffix] = np.nan
    return df





TRUTHY = {"true","t","1","yes","y"}

def _truthy_mask(s):
    return s.astype(str).str.strip().str.lower().isin(TRUTHY)

def _mad(z):
    med = np.median(z)
    return 1.4826 * np.median(np.abs(z - med))  # robust sigma

def fit_abundance_line(df, x_col, y_col, standards_col="Standards",
                          fit_intercept=True, loss="huber", f_scale="auto",
                          min_points=3):
    """
    Robust line y ≈ a + b x on standards using SciPy least_squares with robust loss.
    Returns (a, b, result) where result includes diagnostics.
    """
    m = _truthy_mask(df[standards_col])
    x = df.loc[m, x_col].astype(float).to_numpy()
    y = df.loc[m, y_col].astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < min_points:
        raise ValueError("Not enough standards for robust regression.")

    # Initial guess via OLS
    if fit_intercept:
        X = np.c_[np.ones_like(x), x]
        beta0, *_ = lstsq(X, y, rcond=None)  # [a0, b0]
    else:
        b0 = float(np.dot(x, y) / max(np.dot(x, x), 1e-18))
        beta0 = np.array([0.0, b0])

    def residuals(beta):
        a, b = beta
        return y - (a + b * x)

    # Robust scale for loss threshold
    r0 = residuals(beta0)
    scale = _mad(r0)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(1e-6, np.std(r0))

    res = least_squares(
        fun=residuals,
        x0=beta0,
        loss=loss,          # 'huber', 'soft_l1', or 'cauchy'
        f_scale=scale if f_scale == "auto" else float(f_scale),
        method="trf",
        max_nfev=1000,
    )
    a, b = map(float, res.x)
    return a, b, {"n": int(x.size), "scale": float(scale), "success": bool(res.success), "cost": float(res.cost)}





def apply_global_scale(df, cols_to_scale, factor):
    for c in cols_to_scale:
        df[c] = pd.to_numeric(df[c], errors="coerce") / factor
    return df



def parse_series_any(cell):
    """
    Robustly parse n-value cell contents that may look like:
      [29.0 31.0 34.5]  or  [29.0, 31.0, 34.5]  or already be a list/array.
    Returns a numpy array of floats.
    """
    if isinstance(cell, (list, np.ndarray)):
        return np.array(cell, dtype=float)

    if isinstance(cell, str):
        s = cell.strip("[] \n\r\t")
        # Split on either spaces or commas
        parts = re.split(r"[\s,]+", s)
        try:
            return np.array([float(x) for x in parts if x], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    # Fallback for None, NaN, or unexpected types
    return np.array([], dtype=float)



def _lipid_replicate_table(df, exp_cols, ctl_cols,
                           lipid_col="Lipid Unique Identifier",
                           adduct_col="Alignment ID"):
    """
    Build replicate table *without collapsing adducts*.

    Each Alignment ID (adduct) remains distinct.
    Returns long table:
        Lipid Unique Identifier | Alignment ID | sample_id | group | log2_abundance_adduct
    """
    rows = []
    tiny = 1e-9
    for cols, grp in ((exp_cols, 1), (ctl_cols, 0)):
        if not cols:
            continue
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            for idx, val in s.items():
                if pd.notna(val):
                    rows.append({
                        "Lipid Unique Identifier": df.at[idx, lipid_col],
                        "Alignment ID": df.at[idx, adduct_col],
                        "sample_id": c,
                        "group": grp,
                        "log2_abundance_adduct": np.log2(float(val) + tiny),
                    })
    return pd.DataFrame(rows)


import numpy as np
from numpy.linalg import pinv, matrix_rank
from scipy.stats import chi2

def _parse_pcov(cell):
    """
    Parse a covariance matrix from a variety of serialized forms:
    - string like '[[v11,v12],[v21,v22]]' or 'v11,v12;v21,v22'
    - flat list 'v11,v12,v21,v22' (will reshape to square if possible)
    - Python list/ndarray
    Returns: np.ndarray (2D) or None if cannot parse.
    """
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return None
    try:
        # Already an array/list of lists?
        if isinstance(cell, (np.ndarray, list, tuple)):
            arr = np.asarray(cell, dtype=float)
        else:
            s = str(cell).strip()
            # Try to eval if it looks like Python list syntax
            if s.startswith('[') and s.endswith(']'):
                arr = np.asarray(ast.literal_eval(s), dtype=float)
            else:
                # Fallback: split common separators
                s = s.replace(';', '\n')
                rows = [r for r in s.splitlines() if r.strip()]
                rows = [ [float(x) for x in re.split(r'[,\s]+', r.strip()) if x] for r in rows ]
                arr = np.asarray(rows, dtype=float)
        # If it's flat, try to make it square
        if arr.ndim == 1:
            L = arr.size
            n = int(round(L ** 0.5))
            if n * n == L:
                arr = arr.reshape(n, n)
            else:
                return None
        if arr.ndim != 2:
            return None
        # sanity
        if not np.all(np.isfinite(arr)):
            return None
        return arr
    except Exception:
        return None

def wald_test(theta, V, R=None, r=None, *, rcond=1e-12):
    """
    Canonical Wald test for linear restrictions:

        H0: R theta = r

    Wald statistic:
        W = (R theta - r)^T [R V R^T]^(-1) (R theta - r)

    Under H0, asymptotically: W ~ chi2_df where df = rank(R).  (Canonical Wald)  [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

    Parameters
    ----------
    theta : array-like, shape (p,)
        Parameter estimate vector. [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    V : array-like, shape (p,p)
        Covariance matrix of theta (pcov-style). [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[3](https://stackoverflow.com/questions/69046347/scipy-minimize-scipy-curve-fit-lmfit)
    R : array-like, shape (q,p), optional
        Restriction matrix. If None, defaults to identity (tests theta == r). [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    r : array-like, shape (q,), optional
        Restriction target vector. If None, defaults to zeros. [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    rcond : float
        Regularization parameter for pseudoinverse stability. [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

    Returns
    -------
    dict with keys: W, df, p, q, ok
    """
    theta = np.asarray(theta, float).reshape(-1)
    V = np.asarray(V, float)

    p = theta.size
    if V.shape != (p, p) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(theta)):
        return {"W": np.nan, "df": np.nan, "p": np.nan, "q": np.nan, "ok": False}

    # Default: test theta == 0 (or == r if provided) using R=I
    if R is None:
        R = np.eye(p, dtype=float)
    R = np.asarray(R, float)

    q = R.shape[0]
    if R.shape[1] != p or not np.all(np.isfinite(R)):
        return {"W": np.nan, "df": np.nan, "p": np.nan, "q": q, "ok": False}

    if r is None:
        r = np.zeros(q, dtype=float)
    r = np.asarray(r, float).reshape(-1)
    if r.size != q or not np.all(np.isfinite(r)):
        return {"W": np.nan, "df": np.nan, "p": np.nan, "q": q, "ok": False}

    diff = (R @ theta) - r
    S = R @ V @ R.T

    # Degrees of freedom is the rank of the restriction matrix (canonical choice). [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    df = int(matrix_rank(R))

    if df <= 0:
        return {"W": 0.0, "df": 0, "p": 1.0, "q": q, "ok": True}

    # Use pseudoinverse for robustness when S is near-singular. [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    Sinv = pinv(S, rcond=rcond)

    W = float(diff.T @ Sinv @ diff)
    pval = float(1.0 - chi2.cdf(W, df))

    return {"W": W, "df": df, "p": pval, "q": q, "ok": True}



def wald_diff(theta_exp, V_exp, theta_ctl, V_ctl, R=None, r=None, *, rcond=1e-12):
    """
    Wald test for differences between two independent parameter estimates:

        H0: R (theta_exp - theta_ctl) = r

    Uses:
        d = theta_exp - theta_ctl
        V = V_exp + V_ctl   (independence assumption)

    Then calls canonical wald_test(d, V, R, r). [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    """
    theta_exp = np.asarray(theta_exp, float).reshape(-1)
    theta_ctl = np.asarray(theta_ctl, float).reshape(-1)
    if theta_exp.size != theta_ctl.size:
        return {"W": np.nan, "df": np.nan, "p": np.nan, "q": np.nan, "ok": False}

    V_exp = np.asarray(V_exp, float)
    V_ctl = np.asarray(V_ctl, float)
    p = theta_exp.size
    if V_exp.shape != (p, p) or V_ctl.shape != (p, p):
        return {"W": np.nan, "df": np.nan, "p": np.nan, "q": np.nan, "ok": False}

    d = theta_exp - theta_ctl
    V = V_exp + V_ctl

    return wald_test(d, V, R=R, r=r, rcond=rcond)



def _rate_t_p_from_stats(row, exp: str, ctl: str):
    """
    Compute a Welch t-test for abundance rates using summary statistics.

    Expects each row to contain:
        Abundance rate_{ID}
        Abundance std_error_{ID}
        Abundance num_measurements_{ID}

    For both experiment (`exp`) and control (`ctl`).

    Notes
    -----
    - Converts standard error (SE) to standard deviation (SD) via * sqrt(n).
    - Returns NaN values if inputs are missing or insufficient.

    Parameters
    ----------
    row : pd.Series
        A single row containing the required summary-stat columns.
    exp, ctl : str
        Experiment and control identifiers.

    Returns
    -------
    tuple(float, float)
        (t_statistic, p_value) for Welch’s t-test.
    """
    req = [
        f'Abundance rate_{exp}',  f'Abundance SE_K_{exp}',  f'Abundance num_measurements_{exp}',
        f'Abundance rate_{ctl}',  f'Abundance SE_K_{ctl}',  f'Abundance num_measurements_{ctl}',
    ]
    if not all(c in row.index for c in req):
        return (np.nan, np.nan)

    m1 = row[f'Abundance rate_{exp}']
    se1 = row[f'Abundance SE_K_{exp}']
    n1 = row[f'Abundance num_measurements_{exp}']

    m2 = row[f'Abundance rate_{ctl}']
    se2 = row[f'Abundance SE_K_{ctl}']
    n2 = row[f'Abundance num_measurements_{ctl}']

    if any(pd.isna(v) for v in [m1, se1, n1, m2, se2, n2]) or (n1 <= 1) or (n2 <= 1):
        return (np.nan, np.nan)

    # convert SE -> SD
    sd1 = se1 * np.sqrt(n1)
    sd2 = se2 * np.sqrt(n2)

    try:
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=m1, std1=sd1, nobs1=n1,
            mean2=m2, std2=sd2, nobs2=n2,
            equal_var=False
        )
        return (float(t_stat), float(p_val))
    except Exception:
        return (np.nan, np.nan)


def fisher_method(p_values: Iterable[float]) -> float:
    """
    Combine multiple p-values using Fisher’s method.

    Invalid values (NaN, <=0, >1) are ignored.

    Parameters
    ----------
    p_values : iterable of float
        A collection of p-values.

    Returns
    -------
    float
        Combined p-value, or NaN if no valid inputs.
    """
    p = np.asarray(list(p_values), dtype=float)
    p = p[np.isfinite(p)]
    p = (p[(p > 0) & (p <= 1)])
    if p.size == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(p))
    return 1.0 - chi2.cdf(stat, 2 * p.size)



def _normalize_col(s: str) -> str:
    """Normalize column name by lowercasing and stripping non-alphanumerics."""
    s = s.lower()
    return ''.join(ch for ch in s if ch.isalnum())




def _find_series_col(columns: Iterable[str], base_tokens: List[str], ident: str) -> Optional[str]:
    """
    Heuristically locate a time/trace column by tokens and identifier.

    Example:
    --------
    base_tokens = ["rate", "graph", "time", "points", "x"]
    ident       = "A2"
    Will match columns like:
        "rate_graph_time_points_x_A2"
        "rate graph time points x A2"
        "rateGraphTimePointsXA2"

    Parameters
    ----------
    columns : iterable of str
        DataFrame columns.
    base_tokens : list of str
        Expected word stems in the name.
    ident : str
        Experimental/control identifier.

    Returns
    -------
    str or None
        Matching column name, or None if not found.
    """
    ident_norm = _normalize_col(str(ident))
    pattern = r'[_\s]*'.join(map(re.escape, base_tokens)) + r'[_\s]*' + re.escape(str(ident))
    rx = re.compile(pattern, flags=re.IGNORECASE)

    # Pass 1: regex direct match
    for c in columns:
        if rx.search(c):
            return c

    # Pass 2: normalized containment
    want = ''.join(base_tokens)
    want_norm = _normalize_col(want)
    for c in columns:
        cn = _normalize_col(c)
        if want_norm in cn and ident_norm in cn:
            return c

    # Pass 3: fallback exact token join
    candidates = ['_'.join(base_tokens) + f'_{ident}', ' '.join(base_tokens) + f' {ident}']
    for cand in candidates:
        for c in columns:
            if _normalize_col(c) == _normalize_col(cand):
                return c

    return None


def _get_series_from_row(row: pd.Series, ident: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Extract paired (time, y) arrays from a row for a given identifier.

    Uses fuzzy matching via `_find_series_col` to resolve column names.

    Parameters
    ----------
    row : pd.Series
        A single DataFrame row.
    ident : str
        Identifier (e.g., "A2", "E3").

    Returns
    -------
    tuple(list[float] or None, list[float] or None)
        Time points and corresponding values.
    """
    cols = row.index

    # Time candidates
    time_base = ["rate", "graph", "time", "points", "x"]
    time_exact = f'rate_graph_time_points_x_{ident}'
    time_col = time_exact if time_exact in cols else _find_series_col(cols, time_base, ident)

    # Y candidates
    y_base = ["normed", "isotope", "data", "y"]
    y_exact = f'normed_isotope_data_y_{ident}'
    y_col = y_exact if y_exact in cols else _find_series_col(cols, y_base, ident)

    t = parse_series_any(row.get(time_col, None)) if time_col else None
    y = parse_series_any(row.get(y_col, None)) if y_col else None
    return t, y



def covaware_flux_p(
    log_fc,                 # array-like Δ = log(F_exp/F_ctl)
    var_log_k_exp,
    var_log_k_ctl,
    var_log_A_exp,
    var_log_A_ctl,
    cov_logAk_exp,
    cov_logAk_ctl,
    var_log_abn_exp,
    var_log_abn_ctl,
):
    """
    Computes a covariance-aware Wald p-value for flux.
    Returns p (float or NaN).
    Falls back to NaN if any component is missing.

    All variances must be per-row scalars.
    """
    import numpy as np
    from scipy.stats import norm

    # Require all inputs
    arrs = [log_fc, var_log_k_exp, var_log_k_ctl,
            var_log_A_exp, var_log_A_ctl,
            cov_logAk_exp, cov_logAk_ctl,
            var_log_abn_exp, var_log_abn_ctl]
    if any([a is None for a in arrs]):
        return np.nan

    try:
        # Total variance for (log flux difference)
        var_total = (
            var_log_k_exp + var_log_k_ctl +
            var_log_A_exp + var_log_A_ctl +
            2 * (cov_logAk_exp + cov_logAk_ctl) +
            var_log_abn_exp + var_log_abn_ctl
        )

        if not np.isfinite(var_total) or var_total <= 0:
            return np.nan

        z = log_fc / np.sqrt(var_total)
        return float(2 * norm.sf(abs(z)))
    except Exception:
        return np.nan


# ============================================================================
# Model + bootstrap
# ============================================================================

def model_function(t, A, k):
    """Simple exponential decay model: y = A * exp(-k t)."""
    return A- (A * np.exp(-k * t))


def perform_bootstrap_test(
    time_E, y_E, time_C, y_C, num_bootstraps: int = 1000,
    seed: int = 0, param_index: int = 0
) -> Dict[str, float]:
    """
    Compare fitted parameters between experiment and control via bootstrap.

    Model: y = A - A * exp(-k t)
    param_index:
        0 → Compare A (asymptote)
        1 → Compare k (rate constant)
    """
    # Coerce arrays
    try:
        tE = np.asarray(time_E, dtype=float); yE = np.asarray(y_E, dtype=float)
        tC = np.asarray(time_C, dtype=float); yC = np.asarray(y_C, dtype=float)
    except Exception:
        return {"observed_fold_change": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    # Sanity checks
    if tE.size < 2 or tC.size < 2:
        return {"observed_fold_change": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}
    if not (np.isfinite(tE).all() and np.isfinite(tC).all()
            and np.isfinite(yE).any() and np.isfinite(yC).any()):
        return {"observed_fold_change": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    # Sort by time
    idxE = np.argsort(tE); tE, yE = tE[idxE], yE[idxE]
    idxC = np.argsort(tC); tC, yC = tC[idxC], yC[idxC]

    # Fit initial models
    A0E = max(1e-9, float(np.nanmax(yE)))
    A0C = max(1e-9, float(np.nanmax(yC)))
    try:
        poptE, _ = curve_fit(model_function, tE, yE, p0=(A0E, 0.1),
                             bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=4000)
        poptC, _ = curve_fit(model_function, tC, yC, p0=(A0C, 0.1),
                             bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=4000)
        obs = float(poptE[param_index] / poptC[param_index])
    except Exception:
        return {"observed_fold_change": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    nE, nC = tE.size, tC.size
    boots = []
    for _ in range(num_bootstraps):
        try:
            jE = rng.choice(nE, nE, replace=True)
            jC = rng.choice(nC, nC, replace=True)
            pE, _ = curve_fit(model_function, tE[jE], yE[jE], p0=(A0E, 0.1),
                              bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=4000)
            pC, _ = curve_fit(model_function, tC[jC], yC[jC], p0=(A0C, 0.1),
                              bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=4000)
            boots.append(pE[param_index] / pC[param_index])
        except Exception:
            continue

    if len(boots) == 0:
        return {"observed_fold_change": obs, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return {"observed_fold_change": obs, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}

    ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5])

    # Two-sided + add-one smoothing ⇒ never exactly 0
    B = boots.size
    p_hi = (np.sum(boots >= obs) + 1) / (B + 1)
    p_lo = (np.sum(boots <= obs) + 1) / (B + 1)
    p_value = float(min(1.0, 2 * min(p_hi, p_lo)))

    return {"observed_fold_change": obs, "p_value": p_value,
            "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}



# ───────────────────────────────
# Robust linear bias correction
# ───────────────────────────────
def linear_bias_correct(df, exp_col, ctl_col, out_col=None, robust=True, return_model=False):
    """
    Robust (Huber) or OLS bias correction with safe fallbacks.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    exp_col : str
        The name of the experimental column (dependent variable).
    ctl_col : str
        The name of the control column (independent variable).
    out_col : str, optional
        The output column for the corrected experimental values.
    robust : bool
        Whether to use robust regression (Huber/TheilSen) or fallback to OLS.
    return_model : bool
        If True, also returns the (slope, intercept) tuple used for transformation.

    Returns
    -------
    df : pd.DataFrame
        Same as input with the correction column added.
    (optional) (slope, intercept) : tuple of floats
        The coefficients used in the linear transformation (exp ~ slope * ctl + intercept).
    """
    if out_col is None:
        out_col = f"{exp_col}_linCorr"
    
    if exp_col not in df.columns or ctl_col not in df.columns:
        df[out_col] = df.get(exp_col, np.nan)
        return (df, (np.nan, np.nan)) if return_model else df

    tmp = df[[exp_col, ctl_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.shape[0] < 3:
        df[out_col] = df.get(exp_col, np.nan)
        return (df, (np.nan, np.nan)) if return_model else df

    X = tmp[ctl_col].values.reshape(-1, 1)
    y = tmp[exp_col].values

    # Initialize
    model = None
    intercept = slope = np.nan

    if robust:
        try:
            model = HuberRegressor(epsilon=1.35, max_iter=2000, tol=1e-6).fit(X, y)
            intercept = float(model.intercept_)
            slope = float(model.coef_[0])
        except Exception as e:
            warnings.warn(f"HuberRegressor failed {e}.")

    else:
        slope, intercept = np.polyfit(tmp[ctl_col].values, tmp[exp_col].values, deg=1)

    # Correction: map experimental values onto control frame
    ctl_vals = df[ctl_col].replace([np.inf, -np.inf], np.nan).values
    pred_exp_from_ctl = intercept + slope * ctl_vals
    df[out_col] = df[exp_col] - (pred_exp_from_ctl - ctl_vals)

    if return_model:
        return df, (slope, intercept)
    else:
        return df




def classify_metabolites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify metabolites into broad lipid categories.

    Adds boolean flags for:
      • Ethers
      • Glycerophospholipids (including ether and lyso variants)
      • Glycerolipids (MG/DG/TG + glycerophospholipids)
      • Sphingolipids
      • Lysos
      • Internal standards (d7)
      • Neutral lipids (TG, DG, MG, CE)
      • Ionic lipids (everything not neutral)
      • Kenedy pathways lipids

    Parameters
    ----------
    df : pd.DataFrame
        Lipidomics data with at least 'Ontology' column.

    Returns
    -------
    pd.DataFrame
        Copy of df with new boolean classification columns.
    """
    df = df.copy()
    if 'Ontology' in df.columns:

        # Ether lipids
        df['Ethers'] = (
            df['Ontology'].astype(str).str.contains('ether', case=False, na=False) |
            df.get('Lipid Unique Identifier', '').astype(str).str.contains(r'\b[OP]-', case=False, na=False) |
            df.get('Lipid Name', '').astype(str).str.contains(r'\b[OP]-', case=False, na=False)
        )

        phospho_tokens = ['PA','PC','PI','PE','PS','PG','CL']
        lyso_tokens    = ['LPA','LPC','LPI','LPE','LPS','LPG']

        def is_glycerophospho(x: str) -> bool:
            for tok in phospho_tokens + lyso_tokens:
                if x.startswith(tok) or x.startswith(f"{tok} O-") or x.startswith(f"{tok} P-"):
                    return True
            return False

        df['glycerophospholipids'] = df['Ontology'].astype(str).apply(is_glycerophospho)

        glycero_tokens = ['MG','DG','TG']
        df['glycerolipids'] = df['Ontology'].astype(str).apply(
            lambda x: (x in glycero_tokens) or is_glycerophospho(x)
        )

        df['sphingolipids'] = df['Ontology'].astype(str).apply(
            lambda x: (('Cer' in x) or ('SM' in x))
                      and ('ether' not in x.lower())
                      and not is_glycerophospho(x)
        )

        df['lysos'] = df['Ontology'].astype(str).apply(
            lambda x: any(x.startswith(tok) for tok in lyso_tokens)
        )

        df['Standards'] = df.get('Lipid Unique Identifier','').astype(str).str.contains('d7', na=False)
        
        df['Kennedy_lipids'] = df['Ontology'].astype(str).apply(
            lambda x: (('PC' in x) or ('PE' in x)) or ('PS' in x))


        # ------------------------------------------------------------------
        # NEW: Neutral vs. Ionic Lipids
        # Neutral: TG, DG, MG, CE
        # Ionic: everything else
        # ------------------------------------------------------------------
        """
        neutral_roots = ['TG', 'DG', 'MG', 'CE']

        def is_neutral(x: str) -> bool:
            if not isinstance(x, str):
                return False
            x = x.strip()
            return any(x.startswith(tok) for tok in neutral_roots)

        df['neutral_lipids'] = df['Ontology'].astype(str).apply(is_neutral)
        df['ionic_lipids']   = ~df['neutral_lipids']"""

    return df



def classify_metabolites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification rules (per user spec):
    1) The 'Ontology' string itself is the only ontology/class label (no collapsing).
    2) Higher-order families are assigned ONLY for the curated set of headgroup tokens:
       {PC, PE, PS, PI, PG, PA}, by searching within the Ontology string.
       - Example: 'LPE' and 'Cer-PE' are NOT PE class, but ARE in the ethanolamine family.
    3) Keep other higher-order groupings such as lysos, ethers, glycerophospholipids,
       glycerolipids, sphingolipids, neutral vs ionic, and Kennedy pathway flags.

    Returns a copy of df with added columns:
      - ontology_class
      - family_choline, family_ethanolamine, family_serine, family_inositol,
        family_glycerol (PG/PA merged here), family_CL (optional if you want CL family)
      - is_lyso
      - Ethers
      - glycerophospholipids, glycerolipids
      - sphingolipids
      - Standards
      - neutral_lipids, ionic_lipids
      - Kennedy_lipids
    """
    df = df.copy()

    # ------------------------------------------------------------------------
    # 1) Literal ontology class (no interpretation/collapsing)
    # ------------------------------------------------------------------------
    oncol = "Ontology"
    if oncol not in df.columns:
        df[oncol] = ""
    df["ontology_class"] = df[oncol].astype(str).fillna("")

    # Uppercased working view to simplify contains checks
    ont_upper = df["ontology_class"].str.upper()

    # ------------------------------------------------------------------------
    # 2) Higher-order families: ONLY from curated tokens {PC, PE, PS, PI, PG, PA}
    #    We search within the Ontology string (case-insensitive) to assign families.
    # ------------------------------------------------------------------------
    CORE = ["PC", "PE", "PS", "PI", "PG", "PA"]
    family_map = {
        "PC": "family_choline",
        "PE": "family_ethanolamine",
        "PS": "family_serine",
        "PI": "family_inositol",
        # Group PG and PA into a single "glycerol" family (per your earlier usage).
        "PG": "family_glycerol",
        "PA": "family_glycerol",
    }

    # Initialize all family columns to False
    for fam in set(family_map.values()):
        df[fam] = False

    # Assign families purely based on whether the curated token appears in the Ontology
    for token, fam in family_map.items():
        df[fam] = df[fam] | ont_upper.str.contains(token, na=False)

    # Optional: Cardiolipins as their own "family_CL" (not in curated six, but often useful)
    df["family_CL"] = ont_upper.str.startswith("CL", na=False)

    # ------------------------------------------------------------------------
    # 3) Lyso lipids (kept as a higher-order grouping)
    #    Any Ontology that starts with L + one of the curated headgroups.
    # ------------------------------------------------------------------------
    lyso_prefixes = tuple("L" + t for t in CORE)  # LPA, LPC, LPE, LPI, LPS, LPG
    df["is_lyso"] = ont_upper.str.startswith(lyso_prefixes, na=False)

    # ------------------------------------------------------------------------
    # 4) Ethers (preserved)
    #    Detect vinyl-ether/plasmalogen markers ("P-"), alkyl-ether ("O-"), or 'ether' literal.
    #    Check in Ontology, Lipid Name, and Lipid Unique Identifier (if present).
    # ------------------------------------------------------------------------
    ethers_ont = ont_upper.str.contains("ETHER", na=False) | \
                 df["ontology_class"].str.contains(r"\bO-|^O-", regex=True, na=False) | \
                 df["ontology_class"].str.contains(r"\bP-|^P-", regex=True, na=False)
    ethers_luid = df.get("Lipid Unique Identifier", pd.Series("", index=df.index)).astype(str).str.contains(r"\bO-|^O-", regex=True, na=False)
    ethers_lname = df.get("Lipid Name", pd.Series("", index=df.index)).astype(str).str.contains(r"\bO-|^O-", regex=True, na=False)
    df["Ethers"] = ethers_ont | ethers_luid | ethers_lname

    # ------------------------------------------------------------------------
    # 5) Glycerophospholipids (preserved)
    #    Strict prefix-based match to classic classes including lyso forms.
    # ------------------------------------------------------------------------
    classical_heads = CORE + ["CL"]
    classical_lyso = ["L" + t for t in CORE]  # LPA/LPC/LPE/LPI/LPS/LPG
    df["glycerophospholipids"] = (
        ont_upper.str.startswith(tuple(classical_heads), na=False) |
        ont_upper.str.startswith(tuple(classical_lyso), na=False)
    )

    # ------------------------------------------------------------------------
    # 6) Glycerolipids (preserved): MG/DG/TG plus glycerophospholipids
    # ------------------------------------------------------------------------
    df["glycerolipids"] = (
        ont_upper.str.startswith(("MG", "DG", "TG"), na=False) |
        df["glycerophospholipids"]
    )

    # ------------------------------------------------------------------------
    # 7) Sphingolipids (preserved): Cer, SM families
    #    Keep simple and structural: prefix match Cer or SM.
    # ------------------------------------------------------------------------
    df["sphingolipids"] = ont_upper.str.startswith(("CER", "SM"), na=False)

    # ------------------------------------------------------------------------
    # 8) Internal standards (preserved)
    # ------------------------------------------------------------------------
    df["Standards"] = (
        df.get("Lipid Unique Identifier", pd.Series("", index=df.index))
          .astype(str)
          .str.contains("d7", case=False, na=False)
    )

    # ------------------------------------------------------------------------
    # 9) Neutral vs Ionic (preserved)
    # ------------------------------------------------------------------------
    df["neutral_lipids"] = ont_upper.str.startswith(("TG", "DG", "MG", "CE"), na=False)
    df["ionic_lipids"] = ~df["neutral_lipids"]

    # ------------------------------------------------------------------------
    # 10) Kennedy pathway flag (preserved): PC, PE, PS anywhere in the ontology string
    # ------------------------------------------------------------------------
    df["Kennedy_lipids"] = (
        ont_upper.str.contains("PC", na=False) |
        ont_upper.str.contains("PE", na=False) |
        ont_upper.str.contains("PS", na=False)
    )

    return df

def classify_metabolites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification rules:
      1) 'Ontology' itself is the only class label (no collapsing or remapping).
      2) Higher-order families are assigned ONLY by searching the Ontology string
         for tokens in {PC, PE, PS, PI, PG, PA} (case-insensitive).
         - Example: 'PE'   -> family_ethanolamine = True
         -          'LPE'  -> family_ethanolamine = True (but class stays 'LPE')
         -          'Cer-PE' -> family_ethanolamine = True (class stays 'Cer-PE')
      3) Preserve other higher-order flags: lyso, ethers, glycerophospholipids,
         glycerolipids, sphingolipids, neutral/ionic, Kennedy pathway, Standards.

    Adds columns:
      ontology_class
      family_choline, family_ethanolamine, family_serine, family_inositol,
      family_glycerol, (optional) family_CL
      is_lyso, Ethers, glycerophospholipids, glycerolipids, sphingolipids,
      Standards, neutral_lipids, ionic_lipids, Kennedy_lipids
    """
    df = df.copy()

    # 1) Literal ontology class
    oncol = "Ontology"
    if oncol not in df.columns:
        df[oncol] = ""
    df["ontology_class"] = df[oncol].astype(str).fillna("")
    ont_upper = df["ontology_class"].str.upper()

    # 2) Higher-order families from curated tokens
    CORE = ["PC", "PE", "PS", "PI", "PG", "PA"]
    family_map = {
        "PC": "family_choline",
        "PE": "family_ethanolamine",
        "PS": "family_serine",
        "PI": "family_inositol",
        "PG": "family_glycerol",   # group PG + PA under glycerol family
        "PA": "family_glycerol",
    }

    # Initialize family columns
    for fam in set(family_map.values()):
        df[fam] = False

    # Assign families if the curated token appears anywhere in the ontology string
    for token, fam in family_map.items():
        df[fam] = df[fam] | ont_upper.str.contains(token, na=False)

    # Optional convenience: cardiolipins as their own family
    df["family_CL"] = ont_upper.str.startswith("CL", na=False)

    # 3) Lyso (kept): L + core headgroup prefix (LPA/LPC/LPE/LPI/LPS/LPG)
    lyso_prefixes = tuple("L" + t for t in CORE)
    df["is_lyso"] = ont_upper.str.startswith(lyso_prefixes, na=False)

    # 4) Ethers (kept): detect O-/P-/‘ether’ across typical fields
    ethers_ont  = ont_upper.str.contains("ETHER", na=False) \
                  | df["ontology_class"].str.contains(r"\bO-|^O-", regex=True, na=False) \
                  | df["ontology_class"].str.contains(r"\bP-|^P-", regex=True, na=False)
    ethers_luid = df.get("Lipid Unique Identifier", pd.Series("", index=df.index)).astype(str)\
                    .str.contains(r"\b[OP]-", regex=True, na=False)
    ethers_lname = df.get("Lipid Name", pd.Series("", index=df.index)).astype(str)\
                     .str.contains(r"\b[OP]-", regex=True, na=False)
    df["Ethers"] = ethers_ont | ethers_luid | ethers_lname

    # 5) Glycerophospholipids (kept): classic classes incl. lyso (prefix-only)
    classical_heads = CORE + ["CL"]
    classical_lyso  = ["L" + t for t in CORE]
    df["glycerophospholipids"] = (
        ont_upper.str.startswith(tuple(classical_heads), na=False)
        | ont_upper.str.startswith(tuple(classical_lyso), na=False)
    )

    # 6) Glycerolipids (kept): MG/DG/TG plus glycerophospholipids
    df["glycerolipids"] = (
        ont_upper.str.startswith(("MG", "DG", "TG"), na=False) | df["glycerophospholipids"]
    )

    # 7) Sphingolipids (kept): Cer, SM
    df["sphingolipids"] = ont_upper.str.startswith(("CER", "SM"), na=False)

    # 8) Internal standards (kept)
    df["Standards"] = (
        df.get("Lipid Unique Identifier", pd.Series("", index=df.index))
          .astype(str)
          .str.contains("d7", case=False, na=False)
    )

    # 9) Neutral vs ionic (kept)
    df["neutral_lipids"] = ont_upper.str.startswith(("TG", "DG", "MG", "CE"), na=False)
    df["ionic_lipids"] = ~df["neutral_lipids"]

    # 10) Kennedy pathway (kept): PC / PE / PS anywhere in the ontology string
    df["Kennedy_lipids"] = (
        ont_upper.str.contains("PC", na=False)
        | ont_upper.str.contains("PE", na=False)
        | ont_upper.str.contains("PS", na=False)
    )

    return df



def classify_metabolites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification rules (per spec):
      1) 'Ontology' itself is the only class label (no collapsing/remapping).
      2) Higher-order families are assigned ONLY by searching the Ontology string
         (case-insensitive) for tokens in {PC, PE, PS, PI, PG, PA}.
         - Example: 'PE'     -> higher_order_ethanolamines = True
                   'LPE'    -> higher_order_ethanolamines = True (class remains 'LPE')
                   'Cer-PE' -> higher_order_ethanolamines = True (class remains 'Cer-PE')
      3) Preserve other higher-order flags: lyso, ethers, glycerophospholipids,
         glycerolipids, sphingolipids, neutral/ionic, Kennedy pathway, Standards.
    """
    df = df.copy()

    # 1) Literal ontology class (no interpretation/collapsing)
    oncol = "Ontology"
    if oncol not in df.columns:
        df[oncol] = ""
    df["ontology_class"] = df[oncol].astype(str).fillna("")
    ont_upper = df["ontology_class"].str.upper()

    # 2) Higher-order families from curated tokens
    CORE = ["PC", "PE", "PS", "PI", "PG", "PA"]
    # Map curated tokens to requested higher-order column names
    family_map = {
        "PC": "cholines",
        "PE": "ethanolamines",
        "PS": "serines",
        "PI": "inositols",
        # Group PG and PA under a single glycerol higher-order (as discussed)
        "PG": "glycerols",
        "PA": "phosphatidics",
    }

    # Initialize all higher-order columns as False
    for fam in set(family_map.values()):
        df[fam] = False

    # Assign higher-order families if curated token appears anywhere in the ontology string
    for token, fam in family_map.items():
        df[fam] = df[fam] | ont_upper.str.contains(token, na=False)

    # 3) Lyso (kept): L + core headgroup prefix (LPA/LPC/LPE/LPI/LPS/LPG)
    lyso_prefixes = tuple("L" + t for t in CORE)  # ('LPC','LPE','LPI','LPS','LPG','LPA')
    df["is_lyso"] = ont_upper.str.startswith(lyso_prefixes, na=False)

    # 4) Ethers (kept): detect O-/P-/‘ether’ across typical fields
    ethers_ont  = ont_upper.str.contains("ETHER", na=False) \
                  | df["ontology_class"].str.contains(r"\bO-|^O-", regex=True, na=False) \
                  | df["ontology_class"].str.contains(r"\bP-|^P-", regex=True, na=False)
    ethers_luid = df.get("Lipid Unique Identifier", pd.Series("", index=df.index)).astype(str)\
                    .str.contains(r"\b[OP]-", regex=True, na=False)
    ethers_lname = df.get("Lipid Name", pd.Series("", index=df.index)).astype(str)\
                     .str.contains(r"\b[OP]-", regex=True, na=False)
    df["Ethers"] = ethers_ont | ethers_luid | ethers_lname

    # 5) Glycerophospholipids (kept): classic classes incl. lyso (prefix-only)
    classical_heads = CORE + ["CL"]
    classical_lyso  = ["L" + t for t in CORE]
    df["glycerophospholipids"] = (
        ont_upper.str.startswith(tuple(classical_heads), na=False)
        | ont_upper.str.startswith(tuple(classical_lyso), na=False)
    )

    # 6) Glycerolipids (kept): MG/DG/TG plus glycerophospholipids
    df["glycerolipids"] = (
        ont_upper.str.startswith(("MG", "DG", "TG"), na=False) |
        df["glycerophospholipids"]
    )

    # 7) Sphingolipids (kept): Cer, SM
    df["sphingolipids"] = ont_upper.str.startswith(("CER", "SM"), na=False)

    # 8) Internal standards (kept)
    df["Standards"] = (
        df.get("Lipid Unique Identifier", pd.Series("", index=df.index))
          .astype(str)
          .str.contains("d7", case=False, na=False)
    )

    # 9) Neutral vs ionic (kept)
    df["neutral_lipids"] = ont_upper.str.startswith(("TG", "DG", "MG", "CE"), na=False)
    df["ionic_lipids"] = ~df["neutral_lipids"]

    # 10) Kennedy pathway (kept): PC / PE / PS anywhere in the ontology string
    df["Kennedy_lipids"] = (
        ont_upper.str.contains("PC", na=False) |
        ont_upper.str.contains("PE", na=False) |
        ont_upper.str.contains("PS", na=False)
    )

    # --- Optional convenience: exact class flags (EXACT match to ontology) ---
    ont_clean = ont_upper.str.strip()
    for tok in CORE:
        df[f"class_{tok}"] = (ont_clean == tok)          # e.g., class_PE only when Ontology == "PE"
        df[f"class_L{tok}"] = (ont_clean == f"L{tok}")   # e.g., class_LPE only when Ontology == "LPE"

    return df



def get_FA_carbons(name: str) -> Optional[int]:
    """Extract total carbon count from lipid name (e.g., '18:1')."""
    if not isinstance(name, str): return None
    idx = name.find(':')
    if idx == -1: return None
    nums = re.findall(r'\d+', name[:idx])
    return int(nums[-1]) if nums else None


def get_desaturations(name: str) -> Optional[int]:
    """Extract number of double bonds from lipid name (after ':')."""
    if not isinstance(name, str): return None
    m = re.search(r':\s*(\d+)', name)
    return int(m.group(1)) if m else None


def get_fatty_acid_chains(ontology: str) -> Optional[int]:
    """Map ontology class to expected number of fatty acid chains."""
    mapping = {
        'CL': 4, 'TG': 3, 'DG': 2, 'MG': 1, 'PC': 2, 'PE': 2, 'PI': 2, 'PS': 2, 'PG': 2,
        'LPC': 1, 'LPE': 1, 'LPI': 1, 'LPS': 1, 'LPG': 1, 'LPA': 1,
        'SM': 2, 'CE': 1, 'CAR': 1, 'DMPE': 2
    }
    if not isinstance(ontology, str): return None
    for k, v in mapping.items():
        if ontology.startswith(k):
            return v
    return None


# ============================================================================
# Replicate column finders
# ============================================================================

def _plain_abn_rep_cols(df: pd.DataFrame, ident: str) -> list[str]:
    """
    Identify replicate columns for non-DR abundances of an identifier.

    Includes columns containing 'abn' or 'abundance'.
    Excludes derived/statistical fields (e.g., 'dr', 'median', 'std').
    """
    ident_l = str(ident).lower()
    deny = ('dr', 'median', 'lincorr', 'fc', 'log2', 'rate', 'asymptote',
            'std', 'stderr', 'SE', 'num', 'count',
            'p_value', 'p-', 't_', 'tstat')
    cols = []
    for c in df.columns:
        cl = c.lower()
        if (ident_l in cl) and (('abn' in cl) or ('abundance' in cl)):
            if not any(tok in cl for tok in deny):
                cols.append(c)
    return cols


def _dr_abn_rep_cols(df: pd.DataFrame, ident: str) -> list[str]:
    """
    Identify replicate columns for DR-corrected abundances of an identifier.

    Requires 'dr' + ('abn' or 'abundance').
    Excludes derived/statistical fields.
    """
    ident_l = str(ident).lower()
    deny = ('median', 'lincorr', 'fc', 'log2', 'rate', 'asymptote',
            'std', 'stderr', 'SE', 'num', 'count',
            'p_value', 'p-', 't_', 'tstat')
    cols = []
    for c in df.columns:
        cl = c.lower()
        if (ident_l in cl) and ('dr' in cl) and (('abn' in cl) or ('abundance' in cl)):
            if not any(tok in cl for tok in deny):
                cols.append(c)
    return cols



import numpy as np
import pandas as pd
from typing import Optional, Dict, Iterable

# ---------------------------
# Robust parsers & utilities
# ---------------------------
def _to_array(cell) -> np.ndarray:
    """Parse list/ndarray/CSV-string into a float numpy array (NaNs dropped)."""
    if cell is None:
        return np.array([], float)
    if isinstance(cell, np.ndarray):
        x = cell.astype(float, copy=False)
    elif isinstance(cell, (list, tuple)):
        x = np.asarray(cell, float)
    else:
        # string-like with commas or spaces
        s = str(cell).strip().strip("[]")
        if not s:
            return np.array([], float)
        parts = [p for p in s.replace("\n", " ").split(",") if p.strip()]
        if len(parts) == 1:  # maybe space-delimited
            parts = s.split()
        try:
            x = np.asarray([float(p) for p in parts], float)
        except Exception:
            x = np.array([], float)
    x = x[np.isfinite(x)]
    return x

def _center(x: np.ndarray, how: str = "median") -> float:
    if x.size == 0:
        return np.nan
    return float(np.nanmedian(x) if how == "median" else np.nanmean(x))

def _sd(x: np.ndarray) -> float:
    if x.size <= 1:
        return np.nan
    return float(np.nanstd(x, ddof=1))

# -------------------------------------------

import re

def split_by_polarity(cols, *, raise_on_conflict=True):
    """
    Split a list of column names into NEG and POS sets using robust matching:
    - Supports synonyms: neg|negative, pos|positive (any capitalization).
    - Uses word-ish boundaries to avoid false positives (e.g., 'deposit' won't match 'pos').
    - If a column matches both, raises by default (or skips if raise_on_conflict=False).
    - Columns with neither tag are excluded (returned in 'other').
    """
    # You can add/adjust patterns here to reflect your naming scheme.
    # The patterns include:
    #   - full words (\\bnegative\\b, \\bpositive\\b), and
    #   - symbol forms near word edges (e.g., 'ESI-', 'neg_', '_pos')
    NEG_PATTERNS = [
        r"\bneg\b", r"\bnegative\b",
        r"(?:^|[^A-Za-z0-9])neg(?:[^A-Za-z0-9]|$)",   # ' neg_' or '_neg' or '[neg]' etc.
        r"(?:^|[^A-Za-z0-9])-(?:[^A-Za-z0-9]|$)",     # patterns with '-' sign (e.g., 'ESI-')
    ]
    POS_PATTERNS = [
        r"\bpos\b", r"\bpositive\b",
        r"(?:^|[^A-Za-z0-9])pos(?:[^A-Za-z0-9]|$)",   # ' pos_' or '_pos'
        r"(?:^|[^A-Za-z0-9])\+(?:[^A-Za-z0-9]|$)",    # patterns with '+' sign (e.g., 'ESI+')
    ]

    # Compile a single regex per polarity for performance
    re_neg = re.compile("|".join(NEG_PATTERNS), flags=re.IGNORECASE)
    re_pos = re.compile("|".join(POS_PATTERNS), flags=re.IGNORECASE)

    neg, pos, other = [], [], []

    for c in cols:
        if not isinstance(c, str):
            other.append(c)
            continue
        has_neg = bool(re_neg.search(c))
        has_pos = bool(re_pos.search(c))

        if has_neg and has_pos:
            msg = (f"Column '{c}' matched BOTH negative and positive polarity patterns. "
                   "Check your naming or adjust the patterns.")
            if raise_on_conflict:
                raise ValueError(msg)
            else:
                # If you prefer to skip ambiguous ones:
                other.append(c)
                continue

        if has_neg:
            neg.append(c)
        elif has_pos:
            pos.append(c)
        else:
            other.append(c)

    return {"neg": neg, "pos": pos, "other": other}

# ============================================================================
# Experiment container
# ============================================================================

@dataclass
class Experiment:
    """
    Container for a paired experiment vs. control dataset.

    Reads CSV(s), standardizes columns, computes derived metrics,
    applies bias correction, and attaches statistical results.

    Attributes
    ----------
    file_paths : list of str
        Paths to input CSVs.
    pair : (str, str)
        (experiment_id, control_id).
    all_ids : iterable of str
        Set/list of all identifiers from user-provided pairs.
    number, total : int
        GUI bookkeeping (dataset number and total datasets).
    df : pd.DataFrame
        Fully processed data.
    experimental_identifier, control_identifier : str
        Extracted IDs from `pair`.
    file_name : str
        Combined filename label.
    ionization_mode : str
        Inferred mode ('Positive', 'Negative', 'Mixed', 'Unknown').
    """
    file_paths: List[str]
    pair: Tuple[str, str]
    all_ids: Iterable[str]
    number: int = 1
    total: int = 1
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalization_df: Optional[pd.DataFrame] = None
    normalize_by_standards: bool = False
    baseline: str = None
    experimental_identifier: str = field(init=False)
    control_identifier: str = field(init=False)
    file_name: str = field(init=False)
    ionization_mode: str = field(default="Unknown"),
    perform_mtic: bool = False, 
    norm_method: str = "none"

    def __post_init__(self):
        self.experimental_identifier, self.control_identifier = self.pair
        self.file_name = '_'.join(os.path.basename(x) for x in self.file_paths)
        self.process_csv()

    def _safe_to_numeric(self, cols: List[str]):
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

    def process_csv(self):
        """
        Full workflow with robust regression, QC diagnostics, and fold change computation.
        """
        try:
            # ==================================================================
            # 1. Load and clean
            # ==================================================================
            dfs = [pd.read_csv(p) for p in self.file_paths]
            combined = pd.concat(dfs, ignore_index=True)

            # Clean sentinel strings from DeuteRater
            combined = combined.replace(to_replace=r".*no valid time points.*", value=np.nan, regex=True)

            # ==================================================================
            # 2. Metabolite classification and fatty-acid features
            # ==================================================================
            combined = classify_metabolites(combined)

            combined["FAs"] = combined["Ontology"].apply(get_fatty_acid_chains)
            combined["FA_carbons"] = (
                combined["Lipid Name"].apply(get_FA_carbons)
                if "Lipid Name" in combined.columns else np.nan
            )

            combined["Average_FA_length"] = (
                pd.to_numeric(combined["FA_carbons"], errors="coerce")
                / pd.to_numeric(combined["FAs"], errors="coerce")
            ).astype(str)

            combined["Desaturations"] = (
                combined["Lipid Name"].apply(get_desaturations)
                if "Lipid Name" in combined.columns else np.nan
            )
            combined["Desaturations_per_FA"] = (
                pd.to_numeric(combined["Desaturations"], errors="coerce")
                / pd.to_numeric(combined["FAs"], errors="coerce")
            ).astype(str)

            combined["contains_odd_chain"] = (
                pd.to_numeric(combined["FA_carbons"], errors="coerce") % 2 == 1
            )

            exp, ctl = self.experimental_identifier, self.control_identifier


            
            # ============================================================
            # SIMPLE ABUNDANCE ANALYSIS
            # (independent mTIC + standards normalization)
            # ============================================================
            
            combined = combined.copy()
            
            # Baseline is optional unless standards normalization is enabled
            use_baseline = self.baseline is not None and str(self.baseline).strip() != ""
            baseline_required = bool(self.normalize_by_standards)
            
            # ------------------------------------------------------------
            # 1. Raw abundance columns (*_Abn)
            # ------------------------------------------------------------
            abn_cols = [c for c in combined.columns if c.endswith("_Abn")]
            if not abn_cols:
                raise ValueError("No raw abundance (_Abn) columns found.")
            
            exp_cols = [c for c in abn_cols if c.startswith(exp)]
            ctl_cols = [c for c in abn_cols if c.startswith(ctl)]
            
            if not exp_cols or not ctl_cols:
                raise ValueError("Could not identify experiment/control abundance columns.")
            
            # Baseline columns (used ONLY for standards normalization)
            base_cols = []
            if use_baseline:
                base = str(self.baseline).strip()
                base_cols = [c for c in abn_cols if c.startswith(base)]
            
                if not base_cols:
                    if baseline_required:
                        raise ValueError(
                            "Baseline required for standards normalization, "
                            "but no baseline *_Abn columns found."
                        )
                    else:
                        warnings.warn(
                            f"Baseline='{base}' was set but no *_Abn columns found. "
                            "Baseline will be ignored because standards normalization is off."
                        )
                        use_baseline = False
            
            # ------------------------------------------------------------
            # 2. Normalization selector (mtic / quantile / standards / none)
            # ------------------------------------------------------------
            norm_method = str(getattr(self, "norm_method", "none") or "none").lower()
            mtic_exclude_standards = True
            
            # ----------------------------
            # Simple polarity detector
            # ----------------------------
            def detect_polarity(c: str):
                c = c.lower()
                if "neg" in c:
                    return "neg"
                if "pos" in c:
                    return "pos"
                return None
            
            def split_simple_polarity(cols):
                out = {"neg": [], "pos": []}
                for c in cols:
                    pol = detect_polarity(c)
                    if pol in out:
                        out[pol].append(c)
                return out
            
            # ----------------------------
            # mTIC helpers
            # ----------------------------
            def compute_tic(cols, exclude_mask=None):
                tic = {}
                for c in cols:
                    v = pd.to_numeric(combined[c], errors="coerce")
                    if exclude_mask is not None:
                        v = v.mask(exclude_mask)
                    tic[c] = float(np.nansum(v))
                return tic
            
            def mtic_scale(cols, ref_med, tic_map):
                out = []
                for c in cols:
                    t = tic_map[c]
                    if not (np.isfinite(t) and t > 0):
                        raise ValueError(f"[mTIC] Invalid TIC for {c}")
                    coeff = ref_med / t
                    new = c + "_norm"
                    combined[new] = pd.to_numeric(combined[c], errors="coerce") * coeff
                    out.append(new)
                return out
            
            # ----------------------------
            # Quantile normalization helper
            # ----------------------------
            def quantile_normalize_matrix(df_log2):
                M = df_log2.to_numpy(float)
                order = np.argsort(M, axis=0)
                sorted_vals = np.take_along_axis(M, order, axis=0)
                rank_means = np.nanmean(sorted_vals, axis=1)
            
                out = np.full_like(M, np.nan)
                for j in range(M.shape[1]):
                    col = sorted_vals[:, j]
                    mask = np.isfinite(col)
                    k = mask.sum()
                    if k == 0:
                        continue
                    rows = order[mask, j]
                    out[rows, j] = rank_means[mask][:k]
            
                return pd.DataFrame(out, columns=df_log2.columns, index=df_log2.index)
            
            # Collect run columns
            all_run_cols = exp_cols + ctl_cols + (base_cols if use_baseline else [])
            
            exp_cols_used = []
            ctl_cols_used = []
            base_cols_used = [] if use_baseline else None
            
            abundance_repr = "Raw abundances (no global normalization)"
            
            # ============================================================
            # BRANCH 1 — mTIC
            # ============================================================
            if norm_method == "mtic":
                print("[prep] Applying mTIC normalization...")
            
                exclude_mask = (
                    combined["Standards"] == True
                    if ("Standards" in combined.columns and mtic_exclude_standards)
                    else None
                )
            
                tic_map = compute_tic(all_run_cols, exclude_mask)
            
                pol_groups = split_simple_polarity(all_run_cols)
            
                # Per‑polarity reference medians
                ref_med = {}
                for pol, cols in pol_groups.items():
                    vals = [tic_map[c] for c in cols if tic_map[c] > 0]
                    if vals:
                        ref_med[pol] = float(np.median(vals))
            
                # Scale all groups
                for pol, cols in pol_groups.items():
                    if pol not in ref_med:
                        continue
                    rm = ref_med[pol]
            
                    exp_in = [c for c in exp_cols if c in cols]
                    ctl_in = [c for c in ctl_cols if c in cols]
                    base_in = [c for c in base_cols if use_baseline and c in cols]
            
                    if exp_in:
                        exp_cols_used += mtic_scale(exp_in, rm, tic_map)
                    if ctl_in:
                        ctl_cols_used += mtic_scale(ctl_in, rm, tic_map)
                    if use_baseline and base_in:
                        base_cols_used += mtic_scale(base_in, rm, tic_map)
            
                abundance_repr = "mTIC-normalized"
            
            # ============================================================
            # BRANCH 2 — QUANTILE
            # ============================================================
            elif norm_method == "quantile":
                print("[prep] Applying quantile normalization...")
            
                tiny = np.finfo(float).tiny
                pol_groups = split_simple_polarity(all_run_cols)
            
                for pol, cols in pol_groups.items():
                    if not cols:
                        continue
            
                    M = combined[cols].apply(pd.to_numeric, errors="coerce")
                    M_log2 = np.log2(M.clip(lower=tiny))
                    Q = quantile_normalize_matrix(M_log2)
                    Qlin = np.power(2.0, Q)
            
                    for c in cols:
                        nc = c + "_qnorm"
                        combined[nc] = Qlin[c]
            
                    exp_cols_used += [c + "_qnorm" for c in cols if c in exp_cols]
                    ctl_cols_used += [c + "_qnorm" for c in cols if c in ctl_cols]
                    if use_baseline:
                        base_cols_used += [c + "_qnorm" for c in cols if c in base_cols]
            
                abundance_repr = "Quantile-normalized"
            
            # ============================================================
            # BRANCH 3 — STANDARDS NORMALIZATION
            # ============================================================
            elif norm_method == "standards":
                print("[prep] Applying standards normalization...")
            
                if not use_baseline:
                    raise ValueError("Standards normalization requires a baseline group.")
            
                tiny = np.finfo(float).tiny
                eps = 1e-12
            
                exp_cols_used = exp_cols.copy()
                ctl_cols_used = ctl_cols.copy()
                base_cols_used = base_cols.copy()
            
                pol_groups = split_simple_polarity(all_run_cols)
            
                # Standards mask
                std_mask = combined["Standards"].astype(str).str.lower().isin(
                    ["true", "t", "1", "yes", "y"]
                )
            
                # --- Exclude all lyso species (LPA/LPC/LPE/LPI/LPS/LPG) and DG from standards regression ---
                # Prefer the boolean flag produced by classify_metabolites if available
                if 'is_lyso' in combined.columns:
                    lyso_mask = combined['is_lyso'].astype(bool)
                    # build an uppercase Ontology view for DG test
                    ont_upper = combined.get('Ontology', pd.Series('', index=combined.index)).astype(str).str.upper().str.strip()
                else:
                    ont_upper = combined.get('Ontology', pd.Series('', index=combined.index)).astype(str).str.upper().str.strip()
                    lyso_mask = ont_upper.str.startswith(('LPA', 'LPC', 'LPE', 'LPI', 'LPS', 'LPG'))
            
                dg_mask = ont_upper.str.startswith('DG')
            
                # Keep only standards that are NOT lyso and NOT DG
                std_mask = std_mask & ~(lyso_mask | dg_mask)
            
                for pol, cols in pol_groups.items():
            
                    exp_in = [c for c in exp_cols if c in cols]
                    ctl_in = [c for c in ctl_cols if c in cols]
                    base_in = [c for c in base_cols if c in cols]
            
                    if not base_in:
                        continue
            
                    base_mean = combined[base_in].mean(axis=1)
                    combined[f"__base_{pol}"] = base_mean
            
                    # EXP
                    if exp_in:
                        exp_mean = combined[exp_in].mean(axis=1)
                        combined[f"__exp_{pol}"] = exp_mean
            
                        x = np.log2(
                            pd.to_numeric(combined.loc[std_mask, f"__base_{pol}"], errors="coerce")
                            .clip(lower=tiny)
                        )
                        y = np.log2(
                            pd.to_numeric(combined.loc[std_mask, f"__exp_{pol}"], errors="coerce")
                            .clip(lower=tiny)
                        )
            
                        a, b, _ = fit_abundance_line(
                            pd.DataFrame({"x": x, "y": y, "Standards": True}),
                            x_col="x",
                            y_col="y",
                        )
            
                        if not np.isfinite(b) or abs(b) < eps:
                            raise ValueError(f"Slope degenerate (EXP, {pol})")
            
                        for c in exp_in:
                            v = np.log2(pd.to_numeric(combined[c], errors="coerce").clip(lower=tiny))
                            combined[c] = np.power(2.0, (v - a) / b)
            
                    # CTL
                    if ctl_in:
                        ctl_mean = combined[ctl_in].mean(axis=1)
                        combined[f"__ctl_{pol}"] = ctl_mean
            
                        x = np.log2(
                            pd.to_numeric(combined.loc[std_mask, f"__base_{pol}"], errors="coerce")
                            .clip(lower=tiny)
                        )
                        y = np.log2(
                            pd.to_numeric(combined.loc[std_mask, f"__ctl_{pol}"], errors="coerce")
                            .clip(lower=tiny)
                        )
            
                        a, b, _ = fit_abundance_line(
                            pd.DataFrame({"x": x, "y": y, "Standards": True}),
                            x_col="x",
                            y_col="y",
                        )
            
                        if not np.isfinite(b) or abs(b) < eps:
                            raise ValueError(f"Slope degenerate (CTL, {pol})")
            
                        for c in ctl_in:
                            v = np.log2(pd.to_numeric(combined[c], errors="coerce").clip(lower=tiny))
                            combined[c] = np.power(2.0, (v - a) / b)
            
                    # cleanup
                    for tmp in (
                        f"__base_{pol}",
                        f"__exp_{pol}",
                        f"__ctl_{pol}",
                    ):
                        if tmp in combined.columns:
                            del combined[tmp]
            
                abundance_repr = "Standards-normalized"
            
            # ============================================================
            # BRANCH 4 — NONE
            # ============================================================
            else:
                exp_cols_used = exp_cols.copy()
                ctl_cols_used = ctl_cols.copy()
                if use_baseline:
                    base_cols_used = base_cols.copy()
                abundance_repr = "Raw abundances (no global normalization)"
            
            print(f"[prep] Using {abundance_repr}.")
            
            # ------------------------------------------------------------
            # 3. Guard: ensure valid columns
            # ------------------------------------------------------------
            if not exp_cols_used or not ctl_cols_used:
                raise RuntimeError("[prep] No abundance columns available for median computation.")
            if use_baseline and base_cols_used is not None and len(base_cols_used) == 0:
                raise RuntimeError("[prep] Baseline requested but no baseline columns available.")
            
            print(f"[prep] Using {abundance_repr} abundances.")
            
            # ------------------------------------------------------------
            # 4. Compute medians and means (initial pass)
            #     (For 'standards' the columns above were already transformed in place.)
            # ------------------------------------------------------------
            exp_median = f"abundance_median_{exp}"
            ctl_median = f"abundance_median_{ctl}"
            
            combined[exp_median] = combined[exp_cols_used].median(axis=1, skipna=True)
            combined[ctl_median] = combined[ctl_cols_used].median(axis=1, skipna=True)
            
            if use_baseline:
                baseline_median = f"abundance_median_{self.baseline}"
                combined[baseline_median] = combined[base_cols_used].median(axis=1, skipna=True)
            
            exp_mean = f"abundance_mean_{exp}"
            ctl_mean = f"abundance_mean_{ctl}"
            
            combined[exp_mean] = combined[exp_cols_used].mean(axis=1, skipna=True)
            combined[ctl_mean] = combined[ctl_cols_used].mean(axis=1, skipna=True)
            
            if use_baseline:
                baseline_mean = f"{self.baseline}_abundance_mean"
                combined[baseline_mean] = combined[base_cols_used].mean(axis=1, skipna=True)
                        
            # ------------------------------------------------------------
            # 6. Log2-transformed summaries — ALWAYS create these
            #     (works for raw, mTIC, and/or standards-normalized data)
            # ------------------------------------------------------------
            tiny = np.finfo(float).tiny  # safe floor for log2
            
            def _log2_clip(series):
                return np.log2(pd.to_numeric(series, errors="coerce").clip(lower=tiny))
            
            exp_mean_log2    = f"abundance_mean_log2_{exp}"
            ctl_mean_log2    = f"abundance_mean_log2_{ctl}"
            exp_median_log2  = f"abundance_median_log2_{exp}"
            ctl_median_log2  = f"abundance_median_log2_{ctl}"
            
            combined[exp_mean_log2]    = _log2_clip(combined[exp_mean])
            combined[ctl_mean_log2]    = _log2_clip(combined[ctl_mean])
            combined[exp_median_log2]  = _log2_clip(combined[exp_median])
            combined[ctl_median_log2]  = _log2_clip(combined[ctl_median])
            
            if use_baseline:
                baseline_mean_log2    = f"abundance_mean_log2_{self.baseline}"
                baseline_median_log2  = f"abundance_median_log2_{self.baseline}"
                combined[baseline_mean_log2]    = _log2_clip(combined[baseline_mean])
                combined[baseline_median_log2]  = _log2_clip(combined[baseline_median])
            
            # (Optional) If you also want per-sample log2 columns, uncomment:
            # for c in (exp_cols_used + ctl_cols_used + (base_cols_used or [])):
            #     combined[c + "_log2"] = _log2_clip(combined[c])
                        
            # ------------------------------------------------------------
            # 4. Welch t-test per lipid  (run test in log2 space)
            # ------------------------------------------------------------
            pvals = []
            
            tiny = np.finfo(float).tiny  # to avoid log2(0)
            
            for _, row in combined.iterrows():
                # Pull replicates, coerce to numeric, clip at tiny, then log2-transform
                exp_vals = pd.to_numeric(row[exp_cols_used], errors="coerce")
                exp_vals = np.log2(exp_vals.clip(lower=tiny)).dropna().values
            
                ctl_vals = pd.to_numeric(row[ctl_cols_used], errors="coerce")
                ctl_vals = np.log2(ctl_vals.clip(lower=tiny)).dropna().values
            
                if len(exp_vals) >= 2 and len(ctl_vals) >= 2:
                    _, p = ttest_ind(exp_vals, ctl_vals, equal_var=False)
                else:
                    p = np.nan
            
                pvals.append(p)
            
            combined["abn_p_value"] = pvals
            
            # ------------------------------------------------------------
            # 5. Benjamini–Hochberg FDR
            # ------------------------------------------------------------
            combined["abn_p_adj_BH"] = np.nan
            mask = combined["abn_p_value"].notna()
            
            if mask.any():
                combined.loc[mask, "abn_p_adj_BH"] = multipletests(
                    combined.loc[mask, "abn_p_value"],
                    method="fdr_bh"
                )[1]
            

            # ------------------------------------------------------------
            # 6. Fold-change & volcano metrics (MEANS ONLY)
            # ------------------------------------------------------------
            tiny = np.finfo(float).tiny
            
            # Canonical abundance fold-change: mean(EXP) / mean(CTL)
            combined["FC_abn"] = (
                combined[exp_mean] / combined[ctl_mean].replace(0, tiny)
            )
            
            combined["log2_abn_FC"] = np.log2(
                combined["FC_abn"].replace(0, tiny)
            )
            
            # P-value display metrics (from existing abundance tests)
            combined["-log10abn_P"] = -np.log10(
                pd.to_numeric(combined["abn_p_value"], errors="coerce").replace(0, tiny)
            )
            
            combined["-log10abnBH"] = -np.log10(
                pd.to_numeric(combined["abn_p_adj_BH"], errors="coerce").replace(0, tiny)
            )

    

            # ==================================================================
            # Standards diagnostic (simple abundance-based)
            # ==================================================================
            try:
                if "Standards" in combined.columns:
                    Standards_bool = (
                        combined["Standards"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["true", "t", "1", "yes", "y"])
                    )
                else:
                    Standards_bool = pd.Series(False, index=combined.index)
            
                std_df = combined.loc[Standards_bool].copy()
                print(f"[standards] rows flagged Standards==True: {int(Standards_bool.sum())}")
            
                if not std_df.empty:
                    # Use medians already computed (raw or normalized)
                    if exp_median not in std_df.columns or ctl_median not in std_df.columns:
                        raise ValueError("Abundance medians not found for Standards diagnostic.")
            
                    std_df = std_df.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=[exp_median, ctl_median]
                    )
            
                    if not std_df.empty:
                        log2fc = np.log2(std_df[exp_median] / std_df[ctl_median])
            
                        plt.figure(figsize=(6, 4))
                        plt.hist(log2fc, bins=30, alpha=0.7, edgecolor="black")
                        plt.axvline(0, color="k", linestyle="--", lw=1)
            
                        plt.title("Standards Abundance Stability")
                        plt.xlabel("log2 Fold Change (Experiment / Control)")
                        plt.ylabel("Count")
            
                        output_dir = os.path.dirname(self.file_paths[0])
                        exp_name = self.experimental_identifier
                        ctl_name = self.control_identifier
                        exp_label = f"{exp_name}_vs_{ctl_name}"
            
                        save_path = os.path.join(
                            output_dir,
                            f"{exp_label}_standards_abundance_log2FC.png"
                        )
            
                        plt.tight_layout()
                        plt.savefig(save_path, dpi=300)
                        plt.close()
            
                        print(f"[standards] saved plot to {save_path}")
            
            except Exception as e:
                warnings.warn(f"Failed to create Standards abundance diagnostic: {e}")

            #Standards errors
            
            # --- Standard error (SE) of the mean for abundances (linear + log2) ---
            
            tiny = np.finfo(float).tiny  # for safe log2
            
            def _row_se_of_mean(row, cols):
                """
                Returns:
                  n, se_lin, se_log2
                for the replicate columns in `cols`.
                """
                vals_lin = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
                vals_lin = vals_lin[np.isfinite(vals_lin)]
                n = vals_lin.size
                if n >= 2:
                    sd_lin = float(np.nanstd(vals_lin, ddof=1))
                    se_lin = float(sd_lin / np.sqrt(n))
                    vals_log2 = np.log2(np.clip(vals_lin, tiny, None))
                    sd_log2 = float(np.nanstd(vals_log2, ddof=1))
                    se_log2 = float(sd_log2 / np.sqrt(n))
                else:
                    se_lin = np.nan
                    se_log2 = np.nan
                return pd.Series([n, se_lin, se_log2], index=["n", "se_lin", "se_log2"])
            
            # EXP
            exp_se = combined.apply(lambda r: _row_se_of_mean(r, exp_cols_used), axis=1)
            combined[f"abundance_num_measurements_{exp}"] = exp_se["n"].astype("Int64")
            combined[f"abundance_se_{exp}"]              = exp_se["se_lin"]      # linear units
            combined[f"abundance_se_log2_{exp}"]         = exp_se["se_log2"]     # log2 units (recommended)
            
            # CTL
            ctl_se = combined.apply(lambda r: _row_se_of_mean(r, ctl_cols_used), axis=1)
            combined[f"abundance_num_measurements_{ctl}"] = ctl_se["n"].astype("Int64")
            combined[f"abundance_se_{ctl}"]               = ctl_se["se_lin"]     # linear units
            combined[f"abundance_se_log2_{ctl}"]          = ctl_se["se_log2"]    # log2 units (recommended)
            
                            

            # ==================================================================
            # 8. Provenance
            # ==================================================================
            try:
                print("[session]")
                print("  seed: 0")
                print("  python:", sys.version.split()[0], "|", platform.platform())
                print(
                    "  numpy:", np.__version__,
                    "pandas:", pd.__version__,
                    "scipy:", scipy.__version__,
                    "statsmodels:", statsmodels.__version__,
                )
            except Exception:
                pass

            # ==================================================================
            # 9. Continue downstream analysis (rates, asymptote, flux, etc.)
            # ==================================================================
            self.df = combined


   



    
            # ------------------------------------------------------------------
            # n-value metrics (CANONICAL WALD using BA_pcov)
            # ------------------------------------------------------------------
            
            tiny = np.finfo(float).tiny
            
            # --- column resolvers (in case naming differs slightly) ---
            def _get_first_existing(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None
            
            # nL point estimates (you may already have these as BA_nL_{exp/ctl} or similar)
            col_nL_E = _get_first_existing(combined, [f"BA_nL_{exp}", f"nL_{exp}", f"n_value_{exp}", f"BA_nL"])
            col_nL_C = _get_first_existing(combined, [f"BA_nL_{ctl}", f"nL_{ctl}", f"n_value_{ctl}", f"BA_nL"])
            
            # pcov columns (you said you'll use BA_pcov; often stored per-group as BA_pcov_{ID})
            col_pcov_E = _get_first_existing(combined, [f"BA_pcov_{exp}", f"BA_pcov_{exp}".lower(), "BA_pcov"])
            col_pcov_C = _get_first_existing(combined, [f"BA_pcov_{ctl}", f"BA_pcov_{ctl}".lower(), "BA_pcov"])
            
            # If you truly have a wide table with both groups in one row, you should have BA_pcov_{exp} and BA_pcov_{ctl}.
            # If you only have a single BA_pcov column, you cannot compare exp vs ctl in a single row unless you also have
            # separate rows by group. In that case, you should pair rows first (like you did for nL bootstraps).
            
            # Coerce estimates
            nL_E = pd.to_numeric(combined.get(col_nL_E), errors="coerce") if col_nL_E else pd.Series(np.nan, index=combined.index)
            nL_C = pd.to_numeric(combined.get(col_nL_C), errors="coerce") if col_nL_C else pd.Series(np.nan, index=combined.index)
            
            # Prepare outputs (legacy columns)
            for c in [
                "n_val_p_value",
                "-log10n_val_p",
                "n_val_FC",
                "log2_n_val_FC",
                "n_val_diff_mean",
                "n_val_diff_CI95_lo",
                "n_val_diff_CI95_hi",
                "n_val_t_obs",
                "n_val_fraction_difference",
                "n_val_center_Experiment",
                "n_val_center_Control",
            ]:
                if c not in combined.columns:
                    combined[c] = np.nan
            
            # Compute per-row Wald from pcov
            z_crit = 1.96  # normal 95% CI for diff (optional)
            
            for i in range(len(combined)):
                # Parse pcov matrices
                V_E = _parse_pcov(combined.at[i, col_pcov_E]) if col_pcov_E else None
                V_C = _parse_pcov(combined.at[i, col_pcov_C]) if col_pcov_C else None
            
                # Need point estimates
                thE = float(nL_E.iloc[i]) if np.isfinite(nL_E.iloc[i]) else np.nan
                thC = float(nL_C.iloc[i]) if np.isfinite(nL_C.iloc[i]) else np.nan
                if not (np.isfinite(thE) and np.isfinite(thC)):
                    continue
            
                # Extract var(nL) from BA_pcov: order is [nL, rate, Asyn] => var(nL)=pcov[0,0] [4](https://byu-my.sharepoint.com/personal/cniels21_byu_edu/Documents/Microsoft%20Copilot%20Chat%20Files/binomial_n_value.py)
                varE = np.nan
                varC = np.nan
            
                if V_E is not None and np.asarray(V_E).ndim == 2 and np.asarray(V_E).shape[0] >= 1:
                    varE = float(np.asarray(V_E, float)[0, 0])
            
                if V_C is not None and np.asarray(V_C).ndim == 2 and np.asarray(V_C).shape[0] >= 1:
                    varC = float(np.asarray(V_C, float)[0, 0])
            
                # If pcov missing, you *could* fall back to BA_nL_SE^2 if you have it.
                if (not np.isfinite(varE)) and (f"BA_nL_SE_{exp}" in combined.columns):
                    se = pd.to_numeric(combined.at[i, f"BA_nL_SE_{exp}"], errors="coerce")
                    varE = float(se * se) if np.isfinite(se) else np.nan
            
                if (not np.isfinite(varC)) and (f"BA_nL_SE_{ctl}" in combined.columns):
                    se = pd.to_numeric(combined.at[i, f"BA_nL_SE_{ctl}"], errors="coerce")
                    varC = float(se * se) if np.isfinite(se) else np.nan
            
                # Wald variance of difference
                varD = varE + varC
                if not np.isfinite(varD) or varD <= 0:
                    continue
            
                diff = thE - thC
                z = diff / np.sqrt(varD)
                # Canonical 1-df Wald: W=z^2 ~ chi2_1; two-sided p equals normal two-sided p. [1](https://stackoverflow.com/questions/42288740/estimate-of-inverse-hessian-using-scipy-minimization)[2](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
                p = float(2 * norm.sf(abs(z)))
            
                # Write legacy outputs
                combined.at[i, "n_val_p_value"] = p
                combined.at[i, "-log10n_val_p"] = -np.log10(max(p, tiny))
                combined.at[i, "n_val_diff_mean"] = diff
                combined.at[i, "n_val_t_obs"] = z  # keep name; it's a Wald z-stat now
            
                # CI for difference (optional)
                lo = diff - z_crit * np.sqrt(varD)
                hi = diff + z_crit * np.sqrt(varD)
                combined.at[i, "n_val_diff_CI95_lo"] = lo
                combined.at[i, "n_val_diff_CI95_hi"] = hi
            
                # FC + log2 FC (same conventions as before)
                fc = thE / thC if (np.isfinite(thC) and thC != 0) else np.nan
                combined.at[i, "n_val_FC"] = fc
                combined.at[i, "log2_n_val_FC"] = np.log2(fc) if (np.isfinite(fc) and fc > 0) else np.nan
            
                # Extras for volcano/QC
                combined.at[i, "n_val_center_Experiment"] = thE
                combined.at[i, "n_val_center_Control"] = thC
                combined.at[i, "n_val_fraction_difference"] = ((thE - thC) / thC) if (np.isfinite(thC) and thC != 0) else np.nan
    
            # ------------------------------------------------------------------
            # CANONICAL WALD TESTS using exponential_decay_pcov_{exp/ctl}
            #   - Individual Wald tests: A and k (each chi2 df=1)
            #   - Joint Wald test: (A,k) together (chi2 df=2)
            # ------------------------------------------------------------------
            
            # Point estimates (same as before)
            ke = pd.to_numeric(combined.get(f'Abundance rate_{exp}'), errors='coerce')
            kc = pd.to_numeric(combined.get(f'Abundance rate_{ctl}'), errors='coerce')
            
            Ae = pd.to_numeric(combined.get(f'Abundance asymptote_{exp}'), errors='coerce')
            Ac = pd.to_numeric(combined.get(f'Abundance asymptote_{ctl}'), errors='coerce')
            
            # pcov matrices (must be available for canonical joint Wald)
            pcovE_s = combined.get(f'exponential_decay_pcov_{exp}')
            pcovC_s = combined.get(f'exponential_decay_pcov_{ctl}')
            
            # Optional: keep these as a fallback if a pcov is missing on some rows
            seK_E = pd.to_numeric(combined.get(f'Abundance SE_K_{exp}'), errors='coerce')
            seK_C = pd.to_numeric(combined.get(f'Abundance SE_K_{ctl}'), errors='coerce')
            seA_E = pd.to_numeric(combined.get(f'Abundance SE_A_{exp}'), errors='coerce')
            seA_C = pd.to_numeric(combined.get(f'Abundance SE_A_{ctl}'), errors='coerce')
            
            n = len(combined)
            
            # Output buffers
            rate_W = np.full(n, np.nan, float)
            rate_p = np.full(n, np.nan, float)
            rate_z = np.full(n, np.nan, float)   # signed z (so you keep a "t-like" direction)
            
            as_W   = np.full(n, np.nan, float)
            as_p   = np.full(n, np.nan, float)
            as_z   = np.full(n, np.nan, float)   # signed z
            
            Ak_joint_W  = np.full(n, np.nan, float)
            Ak_joint_df = np.full(n, np.nan, float)
            Ak_joint_p  = np.full(n, np.nan, float)
            
            # Restriction matrices for theta = [A, k]
            R_A = np.array([[1.0, 0.0]], float)  # test A_diff = 0
            R_k = np.array([[0.0, 1.0]], float)  # test k_diff = 0
            
            for i in range(n):
                # --- parse pcov (you said you already added helper(s)) ---
                VE = _parse_pcov(pcovE_s.iloc[i]) if isinstance(pcovE_s, pd.Series) else None
                VC = _parse_pcov(pcovC_s.iloc[i]) if isinstance(pcovC_s, pd.Series) else None
            
                # Build 2x2 covariance blocks for [A,k]
                # If a row lacks pcov, fall back to diagonal from SEs (still OK for 1D tests,
                # but joint test will be a diagonal approximation).
                if VE is None or np.asarray(VE).ndim != 2 or np.asarray(VE).shape[0] < 2:
                    varA = float(seA_E.iloc[i]**2) if np.isfinite(seA_E.iloc[i]) else np.nan
                    vark = float(seK_E.iloc[i]**2) if np.isfinite(seK_E.iloc[i]) else np.nan
                    VE2 = np.array([[varA, 0.0],
                                    [0.0, vark]], float)
                else:
                    VE2 = np.asarray(VE, float)[:2, :2]
            
                if VC is None or np.asarray(VC).ndim != 2 or np.asarray(VC).shape[0] < 2:
                    varA = float(seA_C.iloc[i]**2) if np.isfinite(seA_C.iloc[i]) else np.nan
                    vark = float(seK_C.iloc[i]**2) if np.isfinite(seK_C.iloc[i]) else np.nan
                    VC2 = np.array([[varA, 0.0],
                                    [0.0, vark]], float)
                else:
                    VC2 = np.asarray(VC, float)[:2, :2]
            
                # Guard: need finite covariance blocks and finite point estimates
                thE = np.array([Ae.iloc[i], ke.iloc[i]], float)  # theta = [A, k]
                thC = np.array([Ac.iloc[i], kc.iloc[i]], float)
            
                if not (np.all(np.isfinite(thE)) and np.all(np.isfinite(thC))):
                    continue
                if not (np.all(np.isfinite(VE2)) and np.all(np.isfinite(VC2))):
                    continue
            
                # ==========================================================
                # 1) RATE: canonical Wald test for H0: k_exp - k_ctl = 0
                # ==========================================================
                res_k = wald_diff(thE, VE2, thC, VC2, R=R_k, r=np.array([0.0]))
                if res_k.get("ok", False) and np.isfinite(res_k["W"]) and res_k["df"] == 1:
                    rate_W[i] = res_k["W"]
                    rate_p[i] = res_k["p"]
                    dk = float(ke.iloc[i] - kc.iloc[i])
                    rate_z[i] = np.sign(dk) * np.sqrt(res_k["W"])  # signed sqrt(W)
            
                # ==========================================================
                # 2) ASYMPTOTE: canonical Wald test for H0: A_exp - A_ctl = 0
                # ==========================================================
                res_A = wald_diff(thE, VE2, thC, VC2, R=R_A, r=np.array([0.0]))
                if res_A.get("ok", False) and np.isfinite(res_A["W"]) and res_A["df"] == 1:
                    as_W[i] = res_A["W"]
                    as_p[i] = res_A["p"]
                    dA = float(Ae.iloc[i] - Ac.iloc[i])
                    as_z[i] = np.sign(dA) * np.sqrt(res_A["W"])
            
                # ==========================================================
                # 3) JOINT (A,k): canonical Wald omnibus test
                #    H0: [A_diff, k_diff] = [0,0]
                # ==========================================================
                res_joint = wald_diff(thE, VE2, thC, VC2, R=np.eye(2), r=np.zeros(2))
                if res_joint.get("ok", False) and np.isfinite(res_joint["W"]):
                    Ak_joint_W[i]  = res_joint["W"]
                    Ak_joint_df[i] = res_joint["df"]  # should be 2 for identity restriction
                    Ak_joint_p[i]  = res_joint["p"]
            
            
            # -------------------------------
            # Store RATE results (Wald)
            # -------------------------------
            combined['FC_rate'] = np.where((kc > 0) & np.isfinite(kc), ke / kc, np.nan)
            combined['rate_difference'] = ke - kc
            combined['log2_rate_FC'] = np.log2(combined['FC_rate'].replace(0, np.nan))
            
            combined['rate_W'] = rate_W
            combined['rate_p'] = rate_p
            combined['rate_t'] = rate_z          # keep legacy name; now signed Wald z
            combined['rate_dof'] = 1.0           # Wald df for single restriction
            combined['-log10rate_P'] = -np.log10(pd.to_numeric(combined['rate_p'], errors="coerce").replace(0, np.nan))
            
            
            # -------------------------------
            # Store ASYMPTOTE results (Wald)
            # -------------------------------
            combined['FC_asymptote'] = np.where((Ac > 0) & np.isfinite(Ac), Ae / Ac, np.nan)
            combined['asymptote_difference'] = Ae - Ac
            combined['log2_asymptote_FC'] = np.log2(combined['FC_asymptote'].replace(0, np.nan))
            
            combined['asymptote_W'] = as_W
            combined['p_asymptote'] = as_p
            combined['asymptote_t'] = as_z       # keep legacy name; now signed Wald z
            combined['asymptote_dof'] = 1.0      # Wald df for single restriction
            combined['-log10_asymptote_p'] = -np.log10(pd.to_numeric(combined['p_asymptote'], errors="coerce").replace(0, np.nan))
            
            
            # -------------------------------
            # Store JOINT (A,k) omnibus Wald
            # -------------------------------
            combined['Ak_joint_W']  = Ak_joint_W
            combined['Ak_joint_df'] = Ak_joint_df
            combined['Ak_joint_p']  = Ak_joint_p
            combined['-log10Ak_joint_p'] = -np.log10(pd.to_numeric(combined['Ak_joint_p'], errors="coerce").replace(0, np.nan))

            # ------------------------------------------------------------------
            # Flux metrics (abundance-based, bias-free)
            # ------------------------------------------------------------------
            
            # Helper
            def _num(x):
                return pd.to_numeric(x, errors="coerce")
            
            tiny = np.finfo(float).tiny
            
            # ------------------------------------------------------------------
            # 0. Safeguard prerequisite columns
            # ------------------------------------------------------------------
            required_cols = [
                f"abundance_median_{exp}",
                f"abundance_median_{ctl}",
                f"Abundance rate_{exp}",
                f"Abundance rate_{ctl}",
            ]

            
            for col in required_cols:
                if col not in combined.columns:
                    raise ValueError(f"Missing required column for flux calculation: {col}")
            
            Ae = _num(combined[f"abundance_median_{exp}"])
            Ac = _num(combined[f"abundance_median_{ctl}"]).replace(0, tiny)
            
            ke = _num(combined[f"Abundance rate_{exp}"])
            kc = _num(combined[f"Abundance rate_{ctl}"])
            
                        
            # ------------------------------------------------------------------
            # 1. Total flux (k × abundance)
            # ------------------------------------------------------------------
            combined[f"Flux_{exp}"] = ke * Ae
            combined[f"Flux_{ctl}"] = kc * Ac
            
            combined["FC_flux"] = combined[f"Flux_{exp}"] / combined[f"Flux_{ctl}"].replace(0, tiny)
            combined["log2_flux_FC"] = np.log2(combined["FC_flux"].replace(0, tiny))
            
            # ------------------------------------------------------------------
            # 2. Flux p-values — combine rate and abundance (Fisher)
            # ------------------------------------------------------------------
            if {"rate_p", "abn_p_value"} <= set(combined.columns):
                combined["p_flux"] = combined[["rate_p", "abn_p_value"]].apply(
                    lambda s: fisher_method(s.values), axis=1
                )
            else:
                combined["p_flux"] = np.nan
            
            combined["-log10flux_p"] = -np.log10(combined["p_flux"].replace(0, tiny))
            

               
            # ------------------------------------------------------------------
            # Significance flags
            # ------------------------------------------------------------------
            # Thresholds: |log2 FC| > 1 and –log10 P > 1.3
            combined['abn_FC_significant'] = (
                pd.to_numeric(combined.get('log2_abn_FC'), errors='coerce').abs() > 1
            )
            combined['abn_statistically_significant'] = (
                pd.to_numeric(combined.get('-log10abn_P'), errors='coerce') > 1.3
            )
            combined['abn_Overall_significant'] = (
                combined['abn_FC_significant'] & combined['abn_statistically_significant']
            )
   
            combined['rate_FC_significant'] = (
                pd.to_numeric(combined.get('log2_rate_FC'), errors='coerce').abs() > 1
            )
            combined['rate_statistically_significant'] = (
                pd.to_numeric(combined.get('-log10rate_P'), errors='coerce') > 1.3
            )
            combined['rate_Overall_significant'] = (
                combined['rate_FC_significant'] & combined['rate_statistically_significant']
            )
   
            # ------------------------------------------------------------------
            # Save final DataFrame
            # ------------------------------------------------------------------
            self.df = combined
            
            if self.normalization_df is not None and not self.normalization_df.empty:

                print('normalization dataframe successfully loaded')

        except Exception as e:
           error_details = traceback.format_exc()
           print(f"Error processing files: {self.file_name}: {e}\n{error_details}")
