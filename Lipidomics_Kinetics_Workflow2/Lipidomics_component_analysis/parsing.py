# ================================================================
"""This code was authored by Coleman Nielsen, with support from ChatGPT"""

import re

# ------------------------------------------------------------
# (4a) Regex pattern for fatty acid tokens
# Matches things like "16:0" or "18:1" (with optional O- prefix).
# ------------------------------------------------------------
_FA_RE = re.compile(r'(?:O-)?(\d{1,2}:\d{1,2})')


# ------------------------------------------------------------
# (4b) Extract fatty acid tokens from lipid IDs
# ------------------------------------------------------------

def _extract_fa_tokens(ID: str, ontology: str):
    """
    Extract fatty acid tokens from a lipid ID + ontology string.
    Handles ether prefixes, lysos, DG padding, and QC printout.
    """
    alignment_id = ID  # ✅ initialize first so it always exists


    
    # Helper: include Ether-prefixed variants of class names
    def with_ether(names):
        return set(names) | {f"Ether{x}" for x in names}

    # (A) Define lipid class groups
    main_two_fa = with_ether({'PC','PE','PI','PS','PG','PA'})
    lyso_two_fa = with_ether({'LPC','LPI','LPE','LPS','LPA','LPG'})
    tri_fa = with_ether({'DG'})   # DG is padded to 3; TG already has 3 in data
    require_species = main_two_fa | with_ether({'DG'})

    last_suffix = None
    ID = re.sub(r'\(M[^)]*\)', '', ID)
    
    if "_" in ID:
        suffix = ID.rsplit("_", 1)[1].strip()
        if suffix != "":
            last_suffix = suffix
            # remove the suffix part from alignment_id
            alignment_id = ID.rsplit("_", 1)[0].strip()

    # (C) Require species info for some classes
    if ontology in require_species and "|" not in alignment_id:
        return None

    # (D) Keep part after "|" if present
    if "|" in alignment_id:
        alignment_id = alignment_id.split("|", 1)[1]

    # (E) Normalize prefixes
    alignment_id = alignment_id.replace("P-", "").replace("O-", "").strip()

    # (F) Extract FA tokens
    fa_tokens = _FA_RE.findall(alignment_id)

    # (G) Pad with 0:0 if necessary
    if ontology in tri_fa:
        target = 3
    elif ontology in lyso_two_fa:
        target = 2
    else:
        target = None

    if target is not None and len(fa_tokens) < target:
        pads_needed = target - len(fa_tokens)
        alignment_id = alignment_id + (
            "_" + "_".join(["0:0"] * pads_needed) if pads_needed > 0 else ""
        )
        fa_tokens = _FA_RE.findall(alignment_id)

    # (H) QC printout
    alignment_id_qc = alignment_id.replace("/", "_")
    print(f'{ID}  --> {alignment_id_qc}')

    # (I) Return FA tokens
    if not fa_tokens:
        return None
    return fa_tokens


# ------------------------------------------------------------
# (4c) Build components for a row (FA + structural groups)
# ------------------------------------------------------------
def _components_from_row(
    ontology: str,
    alignment_id: str,
    restrict_classes=('PE','PC','PI','PS','PG','PA',
                      'DG','TG','LPE','LPC','LPA',
                      'LPG','LPI','LPS'),
    exclude_ether=True
) -> list[str] | None:
    """
    Build component list (FAs + structural pieces) for a row.
    """
    # (A) Restrict classes / skip ethers if requested
    if restrict_classes and ontology not in restrict_classes and ontology not in [f'Ether{x}' for x in restrict_classes]:
        return None
    if exclude_ether and str(ontology).startswith('Ether'):
        return None

    # (B) Extract FA tokens
    fa = _extract_fa_tokens(alignment_id, ontology)
    if not fa:
        return None

    # (C) Start with FA tokens
    comps = list(fa)

    # (D) Add structural components
    if ontology in ('PC','LPC','PE','LPE','DG','TG',
                    'EtherPC','EtherLPC','EtherPE','EtherLPE','EtherDG','EtherTG'):
        comps.append('Glycerol')
    if ontology in ('PC','LPC','EtherPC','EtherLPC'):
        comps.append('Choline')
    if ontology in ('PE','LPE','EtherPE','EtherLPE'):
        comps.append('Ethanolamine')
    if ontology in ('PS','LPS','EtherPS','EtherLPS'):
        comps.append('Serine')
    if ontology in ('PI','LPI','EtherPI','EtherLPI'):
        comps.append('Inositol')
    if ontology in ('PG','LPG','EtherPG','EtherLPG'):
        comps += ['Glycerol','Glycerol']  # PG has two glycerol groups

    # (E) Return component list
    return comps
