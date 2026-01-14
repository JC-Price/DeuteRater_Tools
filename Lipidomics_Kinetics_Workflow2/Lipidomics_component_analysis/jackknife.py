"""This code was authored by Coleman Nielsen, with support from ChatGPT"""
# ================================================================
# jackknife MODULE (5. jackknife)
#
# This module estimates **component n-values** using jackknife +
# Monte Carlo simulation. It provides robust confidence intervals
# by repeatedly resampling lipids and propagating uncertainty.
#
# Functions:
#   jackknife_component_n_values_from_tidy()
#       - Input: tidy dataframe (with CIs), list of identifiers.
#       - Output: 
#           * out  → aggregated component-level results
#           * sdf0 → lipid-level dataframe with parsed Components
#           * sims_by_ident (optional) → full MC simulation dictionary
# ================================================================

import numpy as np
import pandas as pd
from parsing import _components_from_row         # (4. parsing)
from sampling import _mc_fit_identifier          # (1. sampling)
from design_matrix import save_design_matrix
from histograms import save_component_histograms


# ------------------------------------------------------------
# (5a) jackknife estimator
# ------------------------------------------------------------
def jackknife_component_n_values_from_tidy(
    df: pd.DataFrame,
    identifiers: list[str],
    num_jackknifes: int = 500,
    drop_fraction: float = 0.05,
    num_simulations: int = 1,
    seed: int | None = 0,
    restrict_classes=('PE','PC','PI','PS','PG','PA','DG','TG',
                      'LPE','LPC','LPA','LPG','LPI','LPS'),
    exclude_ether: bool = False,
    exclude_components: list[str] = None,
    verbose: bool = True,
    return_sims: bool = False  # <-- NEW PARAMETER
):
    """
    Jackknife component n-value estimates by randomly dropping a fraction
    of lipids each time and running Monte Carlo simulations.

    Returns:
        out  (pd.DataFrame) → aggregated component-level summary
        sdf0 (pd.DataFrame) → lipid-level dataframe with Components
        sims_by_ident (dict) → optional simulation distributions if return_sims=True
    """
    
    print('made it to jacknife')
    # (A) Seed RNG
    if seed is not None:
        np.random.seed(seed)
        
    print(len(df))
    
    print(df.head())
    #Somehow i lost my n_values before this step
    # (B) Filter to n_value metric
    sdf0 = df[df['metric'].str.lower() == 'n_value'].copy()
    
    
    print(len(sdf0))

    # (C) Ensure numeric columns
    for c in ('value', 'ci_lower', 'ci_upper'):
        sdf0[c] = pd.to_numeric(sdf0[c], errors='coerce')

    # (D) Parse components
    sdf0['Components'] = [
        _components_from_row(ont, aid,
                             restrict_classes=restrict_classes,
                             exclude_ether=exclude_ether)
        for ont, aid in zip(sdf0['Ontology'], sdf0['Alignment ID'])
    ]
    sdf0 = sdf0[sdf0['Components'].notna()].copy()
    
    if exclude_components:
        sdf0 = sdf0[~sdf0['Components'].apply(
            lambda comps: any(c in exclude_components for c in comps)
        )].copy()

    # (E) Handle empty case
    if sdf0.empty:
        return (
            pd.DataFrame(columns=['Component'] + [
                f'n_value_{ident}_median' for ident in identifiers] +
                [f'n_value_{ident}_lower_quantile' for ident in identifiers] +
                [f'n_value_{ident}_upper_quantile' for ident in identifiers]),
            sdf0,
            {} if return_sims else None
        )

    # (F) Initialize result collector
    sims_by_ident = {ident: {} for ident in identifiers}

    # (G) jackknife loop
    for b in range(num_jackknifes):
        # (G1) Drop fraction of lipids
        lipid_ids = sdf0['Alignment ID'].unique()
        drop_n = max(1, int(len(lipid_ids) * drop_fraction))
        drop_lipids = np.random.choice(lipid_ids, size=drop_n, replace=False)
        sdf = sdf0[~sdf0['Alignment ID'].isin(drop_lipids)].copy()

        # (G2) Collect unique components
        components = sorted({c for comps in sdf['Components'] for c in comps})
        print(components)
        comp_to_idx = {c: j for j, c in enumerate(components)}

        # (G3) Simulate for each identifier
        for ident in identifiers:
            sub = sdf[sdf['genotype'] == ident].copy()
            sims = _mc_fit_identifier(sub, comp_to_idx, num_simulations=num_simulations)
            if b == 0:
                save_design_matrix(
                    sub, comp_to_idx,
                    f"design_matrix_{ident}.csv"
                )

            # (G4) Store results
            for j, c in enumerate(components):
                sims_by_ident[ident].setdefault(c, []).extend(sims[:, j])

        # (G5) Progress report
        if verbose and (b + 1) % 10 == 0:
            print(f"[jackknife] Completed {b+1}/{num_jackknifes}")

    # (H) Collapse sims into summary
    all_components = sorted({c for comps in sdf0['Components'] for c in comps})
    out = pd.DataFrame({'Component': all_components})
    #print(len(out))

    # Save histograms (just use first identifier’s collected data)
    for ident in identifiers:
        component_sims_dict = {
            comp: np.array(sims_by_ident[ident][comp])
            for comp in sims_by_ident[ident]
        }
        save_component_histograms(component_sims_dict, identifier=ident)

    # Compute percentiles per identifier
    for ident in identifiers:
        medians, lowers, uppers = [], [], []
        for c in all_components:
            vals = np.array(sims_by_ident[ident].get(c, []))
            if vals.size > 0:
                medians.append(np.median(vals))
                lowers.append(np.percentile(vals, 2.5))
                uppers.append(np.percentile(vals, 97.5))
            else:
                medians.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)

        out[f'n_value_{ident}_median'] = medians
        out[f'n_value_{ident}_lower_quantile'] = lowers
        out[f'n_value_{ident}_upper_quantile'] = uppers

    # (I) Return everything
    if return_sims:
        return out, sdf0, sims_by_ident
    return out, sdf0
