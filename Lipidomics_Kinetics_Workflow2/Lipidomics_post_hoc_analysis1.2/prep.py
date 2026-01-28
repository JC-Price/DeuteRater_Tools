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



from __future__ import annotations
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






def standards_median_ratio_factor(df, num_med, den_med, standards_col="Standards", tiny=1e-9):
    std_mask = (
        df[standards_col].astype(str).str.strip().str.lower()
        .isin(["true", "t", "1", "yes", "y"])
    )
    m = std_mask & (df[num_med] > 0) & (df[den_med] > 0)
    if m.sum() < 3:
        raise ValueError("Not enough standards to compute scaling factor.")

    log2_ratio = np.log2(df.loc[m, num_med] + tiny) - np.log2(df.loc[m, den_med] + tiny)
    factor = 2 ** np.nanmean(log2_ratio)
    return float(factor)


def standards_median_ratio_factor(df, num_med, den_med, standards_col="Standards", tiny=1e-9):
    std_mask = (
        df[standards_col].astype(str).str.strip().str.lower()
        .isin(["true", "t", "1", "yes", "y"])
    )
    m = std_mask & (df[num_med] > 0) & (df[den_med] > 0)
    if m.sum() < 3:
        raise ValueError("Not enough standards to compute scaling factor.")

    ratio = (df.loc[m, num_med].astype(float) + tiny) / (df.loc[m, den_med].astype(float) + tiny)
    factor = np.nanmedian(ratio)
    return float(factor)



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
# Core MacKinnon-style p-value from bootstraps
# -------------------------------------------
def mackinnon_p_from_boot(
    boot_exp: np.ndarray,
    boot_ctl: np.ndarray,
    boot_null: np.ndarray,
    *,
    center: str = "median",
    B: Optional[int] = None,
    B2: Optional[int] = None,
    seed: Optional[int] = 0,
    denom_mode: str = "observed",  # "observed" or "null"
) -> Dict[str, float]:
    """
    MacKinnon-style bootstrap test using existing bootstrap draws.

    Parameters
    ----------
    boot_exp, boot_ctl : ndarray
        Bootstrap draws of the parameter from Experiment and Control fits.
        (These are your 'bootstrap_{param}_{experiment}' and 'bootstrap_{param}_{control}'.)
    boot_null : ndarray
        Bootstrap draws from the pooled/combined "null" fit ('null_bootstrap_{param}').
    center : {"median","mean"}
        Center estimator for point estimate (defaults to median; robust).
    B : int or None
        Number of null-reference draws. Default uses min(max(999,len(boot_null)), 4999).
    B2 : int or None
        Double bootstrap inner replications per first-level draw (e.g., 199). None disables.
    seed : int or None
        RNG seed for reproducibility.
    denom_mode : {"observed","null"}
        - "observed": denom = sqrt(Var_exp + Var_ctl) estimated from boot_exp/boot_ctl.
        - "null":     denom = sqrt(2) * sd(boot_null). Use if you want symmetric null variance.

    Returns
    -------
    dict with keys:
      t_obs, p_boot, p_double, theta_E, theta_C, diff_mean,
      diff_ci_lo, diff_ci_hi, fc, log2_fc
    """
    rng = np.random.default_rng(seed)

    # Sanity + finite cleaning
    xe = np.asarray(boot_exp, float); xe = xe[np.isfinite(xe)]
    xc = np.asarray(boot_ctl, float); xc = xc[np.isfinite(xc)]
    xn = np.asarray(boot_null, float); xn = xn[np.isfinite(xn)]

    if xe.size < 2 or xc.size < 2 or xn.size < 2:
        return dict(
            t_obs=np.nan, p_boot=np.nan, p_double=np.nan,
            theta_E=np.nan, theta_C=np.nan, diff_mean=np.nan,
            diff_ci_lo=np.nan, diff_ci_hi=np.nan,
            fc=np.nan, log2_fc=np.nan
        )

    theta_E = _center(xe, center)
    theta_C = _center(xc, center)
    se_E = _sd(xe)
    se_C = _sd(xc)
    if denom_mode == "null":
        se = np.sqrt(2.0) * _sd(xn)
    else:
        se = np.sqrt(se_E**2 + se_C**2)

    if not np.isfinite(se) or se <= 0:
        return dict(
            t_obs=np.nan, p_boot=np.nan, p_double=np.nan,
            theta_E=theta_E, theta_C=theta_C,
            diff_mean=(theta_E - theta_C),
            diff_ci_lo=np.nan, diff_ci_hi=np.nan,
            fc=(theta_E / theta_C if theta_C != 0 and np.isfinite(theta_C) else np.nan),
            log2_fc=(np.log2(theta_E/theta_C) if theta_C not in (0, np.nan) and np.isfinite(theta_E/theta_C) else np.nan)
        )

    t_obs = (theta_E - theta_C) / se

    # Reference T* under null: draw two independent null thetas each time
    if B is None:
        B = int(min(max(999, xn.size), 4999))
    idx1 = rng.integers(0, xn.size, size=B)
    idx2 = rng.integers(0, xn.size, size=B)
    theta1 = xn[idx1]; theta2 = xn[idx2]

    if denom_mode == "null":
        se_ref = se * np.ones(B, float)  # constant denom
    else:
        # Keep observed denom; this preserves asymmetric group variance (recommended)
        se_ref = se * np.ones(B, float)

    t_ref = (theta1 - theta2) / se_ref
    t_ref = t_ref[np.isfinite(t_ref)]
    if t_ref.size == 0:
        p_boot = np.nan
    else:
        # Two-sided p with add-one smoothing
        B_eff = t_ref.size
        p_hi = (np.sum(t_ref >= t_obs) + 1) / (B_eff + 1)
        p_lo = (np.sum(t_ref <= t_obs) + 1) / (B_eff + 1)
        p_boot = float(min(1.0, 2 * min(p_hi, p_lo)))

    # Optional double bootstrap (MacKinnon) — refine single-bootstrap p-value
    p_double = np.nan
    if (B2 is not None) and (B2 > 0) and (t_ref.size > 0):
        # For each first-level draw tb = t_ref[b], build its own inner null (again from xn).
        # Compute the fraction of inner p-values <= p_boot.
        inner_ps = []
        for _tb in t_ref[: min(200, t_ref.size)]:  # cap for speed
            ii1 = rng.integers(0, xn.size, size=B2)
            ii2 = rng.integers(0, xn.size, size=B2)
            th1 = xn[ii1]; th2 = xn[ii2]
            t2 = (th1 - th2) / se  # same denom choice as above
            t2 = t2[np.isfinite(t2)]
            if t2.size == 0:
                continue
            B2_eff = t2.size
            p2_hi = (np.sum(t2 >= _tb) + 1) / (B2_eff + 1)
            p2_lo = (np.sum(t2 <= _tb) + 1) / (B2_eff + 1)
            inner_ps.append(min(1.0, 2 * min(p2_hi, p2_lo)))
        if inner_ps:
            inner_ps = np.asarray(inner_ps, float)
            p_double = float((np.sum(inner_ps <= p_boot) + 1) / (inner_ps.size + 1))

    # Percentile CI for the difference using cross-draws from exp & ctl bootstraps
    # (ancillary; MacKinnon focuses on p-values)
    Bdiff = int(min(4999, max(xe.size, xc.size)))
    di = xe[np.random.default_rng(seed).integers(0, xe.size, size=Bdiff)] \
         - xc[np.random.default_rng(seed).integers(0, xc.size, size=Bdiff)]
    di = di[np.isfinite(di)]
    if di.size:
        lo, hi = np.percentile(di, [2.5, 97.5])
    else:
        lo = hi = np.nan

    fc = theta_E / theta_C if (theta_C not in (0, np.nan) and np.isfinite(theta_E) and np.isfinite(theta_C)) else np.nan
    l2 = (np.log2(fc) if (fc not in (0, np.nan) and np.isfinite(fc)) else np.nan)

    return dict(
        t_obs=float(t_obs),
        p_boot=float(p_boot) if np.isfinite(p_boot) else np.nan,
        p_double=float(p_double) if np.isfinite(p_double) else np.nan,
        theta_E=float(theta_E), theta_C=float(theta_C),
        diff_mean=float(theta_E - theta_C),
        diff_ci_lo=float(lo), diff_ci_hi=float(hi),
        fc=float(fc) if np.isfinite(fc) else np.nan,
        log2_fc=float(l2) if np.isfinite(l2) else np.nan
    )

# ------------------------------------------------------------
# DataFrame-level runner for nL / rate / Asyn using your cols
# ------------------------------------------------------------



def run_mackinnon_tests_from_df(
    df: pd.DataFrame,
    experiment: str,
    control: str,
    params: Iterable[str] = ("nL",),
    *,
    center: str = "median",
    B: Optional[int] = None,
    B2: Optional[int] = None,
    seed: int = 0,
    denom_mode: str = "observed",
    print_every: int = 250,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply MacKinnon-style tests per row for selected parameters using
    bootstrap columns:
        bootstrap_{param}_{experiment},
        bootstrap_{param}_{control},
        null_bootstrap_{param}

    For nL, results are written DIRECTLY using legacy column names:
        n_val_p_value
        -log10n_val_p
        n_val_FC
        log2_n_val_FC
        n_val_diff_mean
        n_val_diff_CI95_lo
        n_val_diff_CI95_hi
        n_val_t_obs
    """
    import time
    out = df.copy()
    rng_base = np.random.default_rng(seed)

    n_rows = len(out)
    print(f"[run_mackinnon] Starting MacKinnon nL test on {n_rows} rows")

    start_all = time.time()

    for param in params:
        if param.lower() != "nl":
            continue  # this function is explicitly nL-only for legacy compatibility

        col_exp = f"bootstrap_{param}_{experiment}"
        col_ctl = f"bootstrap_{param}_{control}"
        col_null = f"null_bootstrap_{param}"

        print(f"\n[run_mackinnon:nL] Columns:")
        print(f"  EXP : {col_exp}")
        print(f"  CTL : {col_ctl}")
        print(f"  NULL: {col_null}")

        legacy_cols = [
            "n_val_p_value",
            "-log10n_val_p",
            "n_val_FC",
            "log2_n_val_FC",
            "n_val_diff_mean",
            "n_val_diff_CI95_lo",
            "n_val_diff_CI95_hi",
            "n_val_t_obs",
        ]
        for c in legacy_cols:
            if c not in out.columns:
                out[c] = np.nan

        tested = 0
        skipped = 0
        t0 = time.time()

        for i in range(n_rows):
            if print_every and i % print_every == 0:
                if i == 0:
                    print(f"[run_mackinnon:nL] 0/{n_rows} …")
                else:
                    elapsed = time.time() - t0
                    print(f"[run_mackinnon:nL] {i}/{n_rows} "
                          f"({i/n_rows:.0%}) elapsed={elapsed:.1f}s")

            xe = _to_array(out.at[i, col_exp]) if col_exp in out.columns else np.array([], float)
            xc = _to_array(out.at[i, col_ctl]) if col_ctl in out.columns else np.array([], float)
            xn = _to_array(out.at[i, col_null]) if col_null in out.columns else np.array([], float)

            if xe.size >= 2 and xc.size >= 2 and xn.size >= 2:
                res = mackinnon_p_from_boot(
                    xe, xc, xn,
                    center=center,
                    B=B,
                    B2=B2,
                    seed=int(rng_base.integers(0, 2**31 - 1)),
                    denom_mode=denom_mode,
                )

                # ✅ WRITE LEGACY COLUMNS DIRECTLY
                out.at[i, "n_val_p_value"] = res["p_boot"]
                out.at[i, "-log10n_val_p"] = (
                    -np.log10(res["p_boot"]) if res["p_boot"] > 0 else np.nan
                )
                out.at[i, "n_val_FC"] = res["fc"]
                out.at[i, "log2_n_val_FC"] = (
                    np.log2(res["fc"]) if res["fc"] > 0 else np.nan
                )
                out.at[i, "n_val_diff_mean"] = res["diff_mean"]
                out.at[i, "n_val_diff_CI95_lo"] = res["diff_ci_lo"]
                out.at[i, "n_val_diff_CI95_hi"] = res["diff_ci_hi"]
                out.at[i, "n_val_t_obs"] = res["t_obs"]

                tested += 1
            else:
                skipped += 1
                if verbose:
                    reasons = []
                    if xe.size < 2: reasons.append(f"EXP n={xe.size}")
                    if xc.size < 2: reasons.append(f"CTL n={xc.size}")
                    if xn.size < 2: reasons.append(f"NULL n={xn.size}")
                    print(f"[run_mackinnon:nL] skip row {i} — {', '.join(reasons)}")

        elapsed = time.time() - t0
        print(f"[run_mackinnon:nL] complete — tested={tested}, skipped={skipped}, "
              f"elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - start_all
    print(f"[run_mackinnon] Finished in {total_elapsed:.1f}s")

    return out



def run_mackinnon_tests_from_df(
    df: pd.DataFrame,
    experiment: str,
    control: str,
    params: Iterable[str] = ("nL",),
    *,
    center: str = "median",
    B: Optional[int] = None,
    B2: Optional[int] = None,
    seed: int = 0,
    denom_mode: str = "observed",
    print_every: int = 250,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply MacKinnon-style tests per row for selected parameters using
    bootstrap columns:
        bootstrap_{param}_{experiment},
        bootstrap_{param}_{control},
        null_bootstrap_{param}

    For nL, results are written DIRECTLY using legacy column names:
        n_val_p_value
        -log10n_val_p
        n_val_FC
        log2_n_val_FC
        n_val_diff_mean
        n_val_diff_CI95_lo
        n_val_diff_CI95_hi
        n_val_t_obs

    NEW (for nL volcano & QC):
        n_val_fraction_difference      # (theta_E - theta_C) / theta_C
        n_val_center_Experiment        # theta_E
        n_val_center_Control           # theta_C
    """
    import time
    out = df.copy()
    rng_base = np.random.default_rng(seed)

    n_rows = len(out)
    print(f"[run_mackinnon] Starting MacKinnon nL test on {n_rows} rows")

    start_all = time.time()

    for param in params:
        if param.lower() != "nl":
            # this function is explicitly nL-only for legacy compatibility
            continue

        col_exp = f"bootstrap_{param}_{experiment}"
        col_ctl = f"bootstrap_{param}_{control}"
        col_null = f"null_bootstrap_{param}"

        print(f"\n[run_mackinnon:nL] Columns:")
        print(f"  EXP : {col_exp}")
        print(f"  CTL : {col_ctl}")
        print(f"  NULL: {col_null}")

        # ---- create legacy output columns upfront (NaN) ----
        legacy_cols = [
            "n_val_p_value",
            "-log10n_val_p",
            "n_val_FC",
            "log2_n_val_FC",
            "n_val_diff_mean",
            "n_val_diff_CI95_lo",
            "n_val_diff_CI95_hi",
            "n_val_t_obs",
        ]
        for c in legacy_cols:
            if c not in out.columns:
                out[c] = np.nan

        # ---- NEW: columns used by the nL volcano (and helpful QC) ----
        new_cols = [
            "n_val_fraction_difference",
            "n_val_center_Experiment",
            "n_val_center_Control",
        ]
        for c in new_cols:
            if c not in out.columns:
                out[c] = np.nan

        tested = 0
        skipped = 0
        t0 = time.time()

        for i in range(n_rows):
            if print_every and i % print_every == 0:
                if i == 0:
                    print(f"[run_mackinnon:nL] 0/{n_rows} …")
                else:
                    elapsed = time.time() - t0
                    print(
                        f"[run_mackinnon:nL] {i}/{n_rows} "
                        f"({i/n_rows:.0%}) elapsed={elapsed:.1f}s"
                    )

            xe = _to_array(out.at[i, col_exp]) if col_exp in out.columns else np.array([], float)
            xc = _to_array(out.at[i, col_ctl]) if col_ctl in out.columns else np.array([], float)
            xn = _to_array(out.at[i, col_null]) if col_null in out.columns else np.array([], float)

            if xe.size >= 2 and xc.size >= 2 and xn.size >= 2:
                res = mackinnon_p_from_boot(
                    xe, xc, xn,
                    center=center,
                    B=B,
                    B2=B2,
                    seed=int(rng_base.integers(0, 2**31 - 1)),
                    denom_mode=denom_mode,
                )

                # ✅ WRITE LEGACY COLUMNS DIRECTLY
                out.at[i, "n_val_p_value"] = res["p_boot"]
                out.at[i, "-log10n_val_p"] = (
                    -np.log10(res["p_boot"]) if res["p_boot"] > 0 else np.nan
                )
                out.at[i, "n_val_FC"] = res["fc"]
                out.at[i, "log2_n_val_FC"] = (
                    np.log2(res["fc"]) if res["fc"] > 0 else np.nan
                )
                out.at[i, "n_val_diff_mean"] = res["diff_mean"]
                out.at[i, "n_val_diff_CI95_lo"] = res["diff_ci_lo"]
                out.at[i, "n_val_diff_CI95_hi"] = res["diff_ci_hi"]
                out.at[i, "n_val_t_obs"] = res["t_obs"]

                # --- NEW: store centers and fraction difference for volcano X-axis ---
                theta_E = res.get("theta_E", np.nan)
                theta_C = res.get("theta_C", np.nan)

                out.at[i, "n_val_center_Experiment"] = theta_E
                out.at[i, "n_val_center_Control"] = theta_C

                if np.isfinite(theta_E) and np.isfinite(theta_C) and theta_C != 0:
                    out.at[i, "n_val_fraction_difference"] = (theta_E - theta_C) / theta_C
                else:
                    out.at[i, "n_val_fraction_difference"] = np.nan

                tested += 1
            else:
                skipped += 1
                if verbose:
                    reasons = []
                    if xe.size < 2: reasons.append(f"EXP n={xe.size}")
                    if xc.size < 2: reasons.append(f"CTL n={xc.size}")
                    if xn.size < 2: reasons.append(f"NULL n={xn.size}")
                    print(f"[run_mackinnon:nL] skip row {i} — {', '.join(reasons)}")

        elapsed = time.time() - t0
        print(
            f"[run_mackinnon:nL] complete — tested={tested}, "
            f"skipped={skipped}, elapsed={elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_all
    print(f"[run_mackinnon] Finished in {total_elapsed:.1f}s")

    return out




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
    ionization_mode: str = field(default="Unknown")

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
            
            # baseline is optional unless standards normalization is enabled
            use_baseline = self.baseline is not None and str(self.baseline).strip() != ""
            baseline_required = bool(self.normalize_by_standards)
            
            # ------------------------------------------------------------
            # 1. Identify raw abundance columns (*_Abn)
            # ------------------------------------------------------------
            abn_cols = [c for c in combined.columns if c.endswith("_Abn")]
            if not abn_cols:
                raise ValueError("No raw abundance (_Abn) columns found.")
            
            exp_cols = [c for c in abn_cols if c.startswith(exp)]
            ctl_cols = [c for c in abn_cols if c.startswith(ctl)]
            
            if not exp_cols or not ctl_cols:
                raise ValueError("Could not identify experiment/control abundance columns.")
            
            # --- baseline columns are ONLY required for standards normalization ---
            base_cols = []  # always define to avoid UnboundLocal issues later
            if use_baseline:
                base = str(self.baseline).strip()
                base_cols = [c for c in abn_cols if c.startswith(base)]
            
                if not base_cols:
                    if baseline_required:
                        raise ValueError(
                            "Baseline specified/required for standards normalization, "
                            "but no baseline abundance (_Abn) columns found."
                        )
                    else:
                        warnings.warn(
                            f"Baseline='{base}' was set, but no baseline _Abn columns were found. "
                            "Baseline will be ignored because standards normalization is off."
                        )
                        use_baseline = False  # IMPORTANT: disables downstream baseline usage

            # ------------------------------------------------------------
            # 2. Optional mTIC normalization (creates *_norm columns)
            # ------------------------------------------------------------
            if self.normalization_df is not None:
                norm_df = self.normalization_df.copy()
            
                if "File" not in norm_df.columns or "normalization coefficient" not in norm_df.columns:
                    raise ValueError(
                        "Normalization dataframe must contain "
                        "'File' and 'normalization coefficient' columns."
                    )
            
                coeff_map = dict(
                    zip(
                        norm_df["File"].astype(str),
                        norm_df["normalization coefficient"].astype(float)
                    )
                )
            
                def _apply_mtic(cols):
                    out = []
                    for col in cols:
                        run_id = col.replace("_Abn", "") + ".mzML"
                        if run_id not in coeff_map:
                            raise ValueError(f"No normalization coefficient found for {run_id}")
                        new_col = col + "_norm"
                        combined[new_col] = (
                            pd.to_numeric(combined[col], errors="coerce")
                            * coeff_map[run_id]
                        )
                        out.append(new_col)
                    return out
            
                exp_cols_used = _apply_mtic(exp_cols)
                ctl_cols_used = _apply_mtic(ctl_cols)
                base_cols_used = _apply_mtic(base_cols) if use_baseline else None
            
                abundance_repr = "mTIC-normalized"
            else:
                exp_cols_used = exp_cols
                ctl_cols_used = ctl_cols
                base_cols_used = base_cols if use_baseline else None
                abundance_repr = "raw"
            
            # ------------------------------------------------------------
            # 3. Guard: ensure valid columns
            # ------------------------------------------------------------
            if not exp_cols_used or not ctl_cols_used:
                raise RuntimeError("[prep] No abundance columns available for median computation.")
            
            if use_baseline and not base_cols_used:
                raise RuntimeError("[prep] Baseline requested but no baseline columns available.")
            
            print(f"[prep] Using {abundance_repr} abundances.")
            

            # ------------------------------------------------------------
            # 4. Compute medians (always exactly once)
            # ------------------------------------------------------------
            exp_median = f"{exp}_abundance_median"
            ctl_median = f"{ctl}_abundance_median"
            
            combined[exp_median] = combined[exp_cols_used].median(axis=1, skipna=True)
            combined[ctl_median] = combined[ctl_cols_used].median(axis=1, skipna=True)
            
            if use_baseline:
                baseline_median = f"{self.baseline}_abundance_median"
                combined[baseline_median] = combined[base_cols_used].median(axis=1, skipna=True)



            exp_mean = f"{exp}_abundance_mean"
            ctl_mean = f"{ctl}_abundance_mean"
            
            combined[exp_mean] = combined[exp_cols_used].mean(axis=1, skipna=True)
            combined[ctl_mean] = combined[ctl_cols_used].mean(axis=1, skipna=True)
            
            if use_baseline:
                baseline_mean = f"{self.baseline}_abundance_mean"
                combined[baseline_mean] = combined[base_cols_used].mean(axis=1, skipna=True)
            
            # ------------------------------------------------------------
            # 5. Optional standards-based global scaling
            # ------------------------------------------------------------
            if self.normalize_by_standards:
                if not use_baseline:
                    raise ValueError(
                        "Standards normalization requires a baseline group. "
                        "Set Experiment.baseline to enable EXP→BASE and CTL→BASE scaling."
                    )
            
                # EXP → BASE
                factor_exp = standards_median_ratio_factor(
                    combined,
                    num_med=exp_median,
                    den_med=baseline_median,
                )
                combined = apply_global_scale(combined, exp_cols_used, factor_exp)
            
                # CTL → BASE
                factor_ctl = standards_median_ratio_factor(
                    combined,
                    num_med=ctl_median,
                    den_med=baseline_median,
                )
                combined = apply_global_scale(combined, ctl_cols_used, factor_ctl)
            

                # Recompute medians on SAME columns (already present)
                combined[exp_median] = combined[exp_cols_used].median(axis=1, skipna=True)
                combined[ctl_median] = combined[ctl_cols_used].median(axis=1, skipna=True)
                combined[baseline_median] = combined[base_cols_used].median(axis=1, skipna=True)
                
                #  Recompute MEANS as well (needed for mean-based FCs)
                combined[exp_mean] = combined[exp_cols_used].mean(axis=1, skipna=True)
                combined[ctl_mean] = combined[ctl_cols_used].mean(axis=1, skipna=True)
                combined[baseline_mean] = combined[base_cols_used].mean(axis=1, skipna=True)

            
                print(
                    "[prep] Standards normalization applied: "
                    f"EXP→BASE ({factor_exp:.3f}), "
                    f"CTL→BASE ({factor_ctl:.3f})"
                )

            
            # ------------------------------------------------------------
            # 4. Welch t-test per lipid
            # ------------------------------------------------------------
            pvals = []
            
            for _, row in combined.iterrows():
                exp_vals = pd.to_numeric(row[exp_cols_used], errors="coerce").dropna().values
                ctl_vals = pd.to_numeric(row[ctl_cols_used], errors="coerce").dropna().values
            

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
            # Rate metrics (parametric Welch-style t-test using SE and imported dof)
            # ------------------------------------------------------------------
            ke = pd.to_numeric(combined.get(f'Abundance rate_{exp}'), errors='coerce')
            kc = pd.to_numeric(combined.get(f'Abundance rate_{ctl}'), errors='coerce')
            seE = pd.to_numeric(combined.get(f'Abundance SE_K_{exp}'), errors='coerce')
            seC = pd.to_numeric(combined.get(f'Abundance SE_K_{ctl}'), errors='coerce')
            dofE = pd.to_numeric(combined.get(f'Abundance dof_{exp}'), errors='coerce')
            dofC = pd.to_numeric(combined.get(f'Abundance dof_{ctl}'), errors='coerce')
            
            # Fold change and difference
            combined['FC_rate'] = np.where((kc > 0) & np.isfinite(kc), ke / kc, np.nan)
            combined['rate_difference'] = ke - kc
            combined['log2_rate_FC'] = np.log2(combined['FC_rate'].replace(0, np.nan))
            
            # Welch-style denominator
            denom = np.sqrt(seE**2 + seC**2)
            t_stat = (ke - kc) / denom
            
            # Effective Welch–Satterthwaite degrees of freedom (based on imported dof)
            dof = ((seE**2 + seC**2)**2) / (
                (seE**4 / dofE.replace(0, np.nan)) + (seC**4 / dofC.replace(0, np.nan))
            )
            dof = dof.replace([np.inf, -np.inf], np.nan)
            
            # Two-sided p-values
            rate_p = 2 * t.sf(np.abs(t_stat), dof)
            rate_p = np.clip(rate_p, 0, 1)
            
            # Store results
            combined['rate_t'] = t_stat
            combined['rate_dof'] = dof
            combined['rate_p'] = rate_p
            combined['-log10rate_P'] = -np.log10(combined['rate_p'].replace(0, np.nan))
            


            # ------------------------------------------------------------------
            # n-value metrics (replicate-aware Welch's t-test)
            # ------------------------------------------------------------------

            # Run for nL only (replace your current Welch t-test block)


            # --- MacKinnon nL test: write legacy columns directly ---
            combined = run_mackinnon_tests_from_df(
                df=combined,
                experiment=exp,      # e.g., "A2"
                control=ctl,         # e.g., "E3"
                params=("nL",),      # nL only; function is nL-focused for legacy output
                center="median",     # or "mean" if you prefer
                B=None,              # None -> use function's internal default in mackinnon_p_from_boot
                B2=None,             # set to 199 if you want double-bootstrap refinement
                seed=0,
                denom_mode="observed",
                print_every=250,     # heartbeat in console
                verbose=False        # True prints per-row skip reasons
            )



            # ------------------------------------------------------------------
            # Asymptote metrics (parametric Welch-style t-test using SE and imported dof)
            # ------------------------------------------------------------------
            Ae = pd.to_numeric(combined.get(f'Abundance asymptote_{exp}'), errors='coerce')
            Ac = pd.to_numeric(combined.get(f'Abundance asymptote_{ctl}'), errors='coerce')
            seA_e = pd.to_numeric(combined.get(f'Abundance SE_A_{exp}'), errors='coerce')
            seA_c = pd.to_numeric(combined.get(f'Abundance SE_A_{ctl}'), errors='coerce')
            dofA_e = pd.to_numeric(combined.get(f'Abundance dof_{exp}'), errors='coerce')
            dofA_c = pd.to_numeric(combined.get(f'Abundance dof_{ctl}'), errors='coerce')
            
            # Fold change and difference
            combined['FC_asymptote'] = np.where((Ac > 0) & np.isfinite(Ac), Ae / Ac, np.nan)
            combined['asymptote_difference'] = Ae - Ac
            combined['log2_asymptote_FC'] = np.log2(combined['FC_asymptote'].replace(0, np.nan))
            
            # Welch-style t statistic
            denom_A = np.sqrt(seA_e**2 + seA_c**2)
            t_as = (Ae - Ac) / denom_A
            
            # Effective Welch–Satterthwaite degrees of freedom
            dof_A = ((seA_e**2 + seA_c**2)**2) / (
                (seA_e**4 / dofA_e.replace(0, np.nan)) + (seA_c**4 / dofA_c.replace(0, np.nan))
            )
            dof_A = dof_A.replace([np.inf, -np.inf], np.nan)
            
            # Two-sided p-values
            p_as = 2 * t.sf(np.abs(t_as), dof_A)
            p_as = np.clip(p_as, 0, 1)
            
            # Store results
            combined['asymptote_t'] = t_as
            combined['asymptote_dof'] = dof_A
            combined['p_asymptote'] = p_as
            combined['-log10_asymptote_p'] = -np.log10(combined['p_asymptote'].replace(0, np.nan))

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
                f"{exp}_abundance_median",
                f"{ctl}_abundance_median",
                f"Abundance rate_{exp}",
                f"Abundance rate_{ctl}",
            ]

            
            for col in required_cols:
                if col not in combined.columns:
                    raise ValueError(f"Missing required column for flux calculation: {col}")
            
            Ae = _num(combined[f"{exp}_abundance_median"])
            Ac = _num(combined[f"{ctl}_abundance_median"]).replace(0, tiny)
            
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
            # 2. Flux p-values (inherit abundance p-values)
            # ------------------------------------------------------------------
            if "abn_p_value" in combined.columns:
                combined["p_flux"] = _num(combined["abn_p_value"])
                combined["-log10flux_p"] = -np.log10(combined["p_flux"].replace(0, tiny))
            else:
                combined["p_flux"] = np.nan
                combined["-log10flux_p"] = np.nan
            
            # ------------------------------------------------------------------
            # 3. Partition flux into synthesis & dietary components
            # ------------------------------------------------------------------
            Aexp = _num(combined.get(f"Abundance asymptote_{exp}"))
            Actl = _num(combined.get(f"Abundance asymptote_{ctl}"))
            
            if Aexp is None or Actl is None:
                warnings.warn("Asymptote columns missing — synthesis/diet flux skipped.")
            else:
                # Absolute fluxes
                combined[f"synth_flux_{exp}"] = combined[f"Flux_{exp}"] * Aexp
                combined[f"synth_flux_{ctl}"] = combined[f"Flux_{ctl}"] * Actl
                combined[f"diet_flux_{exp}"]  = combined[f"Flux_{exp}"] * (1 - Aexp)
                combined[f"diet_flux_{ctl}"]  = combined[f"Flux_{ctl}"] * (1 - Actl)
            
                # Fold-changes
                FC_Asyn = np.where(Actl > 0, Aexp / Actl, np.nan)
                FC_Adiet = np.where((1 - Actl) > 0, (1 - Aexp) / (1 - Actl), np.nan)
            
                combined["FC_synth_flux"] = combined["FC_flux"] * FC_Asyn
                combined["FC_diet_flux"]  = combined["FC_flux"] * FC_Adiet
            
                combined["log2_synth_flux_FC"] = np.log2(combined["FC_synth_flux"].replace(0, tiny))
                combined["log2_diet_flux_FC"]  = np.log2(combined["FC_diet_flux"].replace(0, tiny))
            
                # p-values: inherit total-flux p-values
                combined["p_synth_flux"] = combined["p_flux"]
                combined["p_diet_flux"]  = combined["p_flux"]
            
                combined["-log10synth_flux_p"] = -np.log10(combined["p_synth_flux"].replace(0, tiny))
                combined["-log10diet_flux_p"]  = -np.log10(combined["p_diet_flux"].replace(0, tiny))
            
               
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
