"""This code was authored by Coleman Nielsen, with support from ChatGPT"""
# ================================================================
# STATISTICAL TESTS MODULE (2. stats_tests)
#
# This module provides functions for **statistical hypothesis testing**
# in the lipid component pipeline.
#
# Functions:
#   adjusted_ttest()
#       - Performs a paired t-test with an adjustment for correlation.
#       - Standard paired t-tests assume independence, but here the
#         component estimates are correlated (they come from the same
#         linear system). This adjustment accounts for that.
#
# Step-by-step in adjusted_ttest():
#   (A) Extract paired arrays x and y from the DataFrame.
#   (B) Compute the difference vector d = y - x.
#   (C) If fewer than 2 pairs, return NaNs (not enough data).
#   (D) Compute the mean (d_mean) and standard deviation (d_sd) of d.
#   (E) Estimate correlation between x and y (rho).
#   (F) Clip rho to [0, 1] to avoid negative variance estimates.
#   (G) Compute adjusted standard error:
#           se_adj = (SD / sqrt(n)) * sqrt(1 + (n-1)*rho)
#   (H) Compute adjusted t-statistic:
#           t_adj = d_mean / se_adj
#   (I) Degrees of freedom = n - 1
#   (J) Compute two-tailed p-value from the t-distribution.
#   (K) Return (mean difference, t-statistic, p-value, correlation).
# ================================================================

import numpy as np
from scipy.stats import t as t_dist

# ------------------------------------------------------------
# (2a) Correlation-adjusted paired t-test
# ------------------------------------------------------------
def adjusted_ttest(df_all, x_col, y_col):
    """
    Paired t-test with correlation-adjusted standard error (Option 1).
    """
    # (A) Extract columns as numpy arrays
    x = df_all[x_col].to_numpy(float)
    y = df_all[y_col].to_numpy(float)

    # (B) Differences
    d = y - x
    n = len(d)

    # (C) Not enough data
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    # (D) Mean and unadjusted SD of differences
    d_mean = np.mean(d)
    d_sd = np.std(d, ddof=1)

    # (E) Estimate correlation between x and y
    rho_matrix = np.corrcoef([x, y])
    rho = rho_matrix[0, 1]

    # (F) Clip rho to [0, 1] (prevents negative variance estimates)
    rho = np.clip(rho, 0, 1)

    # (G) Adjusted standard error
    se_adj = d_sd / np.sqrt(n) * np.sqrt(1 + (n - 1) * rho)

    # (H) Adjusted t-statistic
    t_adj = d_mean / se_adj

    # (I) Degrees of freedom
    df = n - 1

    # (J) Two-tailed p-value
    p = 2 * t_dist.sf(np.abs(t_adj), df)

    # (K) Return results
    return d_mean, t_adj, p, rho
