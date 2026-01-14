# ================================================================
# SAMPLING MODULE (1. sampling)
#
# This module handles all **Monte Carlo sampling** tasks.
# It is responsible for generating random draws from asymmetric
# confidence intervals and solving the linear system for each draw.
#
# Functions:
#   _sample_asymmetric_normal()
#       - Takes a reported center value and its lower/upper 95% CI bounds.
#       - Converts CI bounds to asymmetric Gaussian standard deviations.
#       - Draws samples from a "split-normal" distribution:
#           left tail uses sigma_lo, right tail uses sigma_hi.
#       - Used to mimic uncertainty in lipid abundance estimates.
#
#   _mc_fit_identifier()
#       - For a single genotype ("identifier"), solves the linear system:
#             A * x ≈ b
#         where:
#           A = design matrix (built in 3. design_matrix)
#           b = sampled lipid values (from _sample_asymmetric_normal)
#           x = estimated component n-values
#       - Repeats this solve many times (num_simulations),
#         producing a Monte Carlo distribution of component values.
#
# Step-by-step in _mc_fit_identifier():
#   (A) If no data, return zeros of shape (num_simulations × n_components).
#   (B) Build design matrix A using (3. design_matrix._build_design_matrix).
#   (C) Extract the reported center, lower, and upper CI values for each lipid.
#   (D) For each Monte Carlo iteration:
#       (1) Sample lipid values b from split-normal distributions.
#       (2) Clip b to ensure no negative lipid values.
#       (3) Solve non-negative least squares (lsq_linear) for x.
#   (E) Return all simulations as a 2D array.
# ================================================================

import numpy as np
from scipy.optimize import lsq_linear
from design_matrix import _build_design_matrix  # (3. design_matrix)

# ------------------------------------------------------------
# (1a) Split-normal sampling
# ------------------------------------------------------------
def _sample_asymmetric_normal(center: float, lo: float, hi: float, size: int = 1) -> np.ndarray:
    """
    Sample from an asymmetric Gaussian (split-normal) distribution.
    - center = reported point estimate (mean or median)
    - lo     = lower 95% CI bound
    - hi     = upper 95% CI bound
    - size   = number of samples
    """
    # Convert 95% CI bounds into one-sided standard deviations
    sigma_lo = max((center - lo) / 1.96, 1e-12)
    sigma_hi = max((hi - center) / 1.96, 1e-12)

    draws = []
    for _ in range(size):
        if np.random.rand() < 0.5:
            # Left side → subtract a Gaussian draw with sigma_lo
            draws.append(center - abs(np.random.normal(0, sigma_lo)))
        else:
            # Right side → add a Gaussian draw with sigma_hi
            draws.append(center + abs(np.random.normal(0, sigma_hi)))
    return np.array(draws)


# ------------------------------------------------------------
# (1b) Monte Carlo simulation for a single identifier
# ------------------------------------------------------------
def _mc_fit_identifier(
    sub_df,
    comp_to_idx: dict[str, int],
    num_simulations: int = 100,
) -> np.ndarray:
    """
    Full Monte Carlo for ONE identifier.
    Returns a sims matrix of shape (num_simulations, n_components).
    Supports asymmetric confidence intervals.
    """
    if sub_df.empty:
        return np.zeros((num_simulations, len(comp_to_idx)))

    # (B) Build design matrix
    A = _build_design_matrix(sub_df, comp_to_idx)

    # (C) Extract central values and CI bounds
    b_center = sub_df['value'].to_numpy(float)
    ci_lo = sub_df['ci_lower'].to_numpy(float)
    ci_hi = sub_df['ci_upper'].to_numpy(float)

    sims = np.zeros((num_simulations, A.shape[1]), dtype=float)

    # (D) Repeat Monte Carlo draws
    for s in range(num_simulations):
        # (D1) Sample each lipid value from its asymmetric Gaussian
        b = np.array([
            _sample_asymmetric_normal(c, lo, hi)[0]
            for c, lo, hi in zip(b_center, ci_lo, ci_hi)
        ])
        # (D2) Clip negatives
        b = np.clip(b, 0, None)

        # (D3) Solve least-squares with non-negativity constraint
        #sims[s] = lsq_linear(A, b, bounds=(0, np.inf)).x
        
        sims[s] = lsq_linear(A, b).x

    # (E) Return all Monte Carlo results
    return sims
