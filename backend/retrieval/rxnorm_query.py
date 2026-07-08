from backend.pipeline.rxnorm import find_rxcui_approx_candidates, find_rxcui_exact


def resolve_query_drug(name: str) -> dict:
    """Resolve a user-provided drug name to an RXCUI for retrieval filtering.

    Returns:
        {
            "rxcui": str | None,       # resolved id, or None if not confident
            "match_type": str,         # "exact" | "approx" | "ambiguous" | "unresolved"
            "candidates": list[dict],  # populated only when match_type == "ambiguous",
                                        # each item {"name": str, "rxcui": str}
        }

    "ambiguous" means the name is a plausible typo of more than one real drug
    (e.g. "metfromin" matches both "merbromin" and "metformin") — retrieval
    should not filter on either guess. "unresolved" means no candidate
    resolved to a real rxcui at all.
    """
    rxcui = find_rxcui_exact(name)
    if rxcui is not None:
        return {"rxcui": rxcui, "match_type": "exact", "candidates": []}

    candidates = find_rxcui_approx_candidates(name)
    if len(candidates) == 1:
        return {"rxcui": candidates[0]["rxcui"], "match_type": "approx", "candidates": []}
    if len(candidates) > 1:
        return {"rxcui": None, "match_type": "ambiguous", "candidates": candidates}
    return {"rxcui": None, "match_type": "unresolved", "candidates": []}
