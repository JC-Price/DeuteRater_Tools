"""This code was authored by Coleman Nielsen, with support from ChatGPT"""
# ================================================================
# DESIGN MATRIX MODULE (3. design_matrix)
#
# This module builds the **design matrix** used in the linear system:
#
#       A * x ≈ b
#
#   where:
#     - A = design matrix (lipids × components)
#     - x = vector of component n-values (to be estimated)
#     - b = observed lipid measurements (sampled in 1. sampling)
#
# Functions:
#   _build_design_matrix()
#       - Input: sub_df (lipid rows), comp_to_idx (mapping of components → index).
#       - Output: A 2D numpy array of shape (n_lipids, n_components).
#       - Each entry A[i, j] = number of times component j appears in lipid i.
#
# Step-by-step in _build_design_matrix():
#   (A) Initialize a zero matrix of shape (n_rows × n_components).
#   (B) Loop over each lipid row in the DataFrame.
#   (C) For that lipid, count all its components (using collections.Counter).
#   (D) For each component, find its column index in comp_to_idx.
#   (E) Assign the count into the matrix at A[row, col].
#   (F) Return the completed design matrix.
# ================================================================

import numpy as np
from collections import Counter
import pandas as pd

# ------------------------------------------------------------
# (3a) Build design matrix
# ------------------------------------------------------------
def _build_design_matrix(sub_df: pd.DataFrame, comp_to_idx: dict[str, int]) -> np.ndarray:
    """
    Construct the design matrix A for linear system solving.
    - Rows = lipids in sub_df
    - Cols = components in comp_to_idx
    - A[i, j] = count of component j in lipid i
    """
    # (A) Initialize matrix with zeros
    A = np.zeros((len(sub_df), len(comp_to_idx)), dtype=float)

    # (B) Loop over lipids (rows)
    for i, comps in enumerate(sub_df['Components']):
        # (C) Count how many times each component occurs
        for comp, cnt in Counter(comps).items():
            # (D) Find column index
            j = comp_to_idx.get(comp)
            if j is not None:
                # (E) Fill matrix with counts
                A[i, j] = cnt

    # (F) Return design matrix
    return A




def save_design_matrix(sub_df: pd.DataFrame, comp_to_idx: dict[str, int], out_path: str):
    """
    Build and save the design matrix for a given sub_df → CSV.

    Args:
        sub_df (pd.DataFrame): Lipid-level dataframe containing 'Components'.
        comp_to_idx (dict): Mapping of component → column index.
        out_path (str): File path for CSV output.
    """
    A = _build_design_matrix(sub_df, comp_to_idx)
    # Label rows by Alignment ID, cols by component
    df_out = pd.DataFrame(A, index=sub_df["Alignment ID"], columns=comp_to_idx.keys())
    df_out.to_csv(out_path, index=True)
    print(f"[design_matrix] Saved design matrix with shape {A.shape} to {out_path}")

