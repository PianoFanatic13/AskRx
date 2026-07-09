import json
from pathlib import Path

from backend.pipeline.rxnorm import find_rxcui_approx_candidates, find_rxcui_exact

_DEFAULT_CACHE_PATH = Path("data/rxnorm_cache.json")


def _load_rxnorm_cache(path: Path) -> dict:
    """Return the ingestion-time RxNorm cache, or {} if it doesn't exist.

    Reimplemented here (rather than importing pipeline.load_rxnorm_cache)
    since backend.pipeline.pipeline transitively imports chunker.py, which
    loads transformers.AutoTokenizer at module level — too heavy to pull into
    a lightweight query-time lookup for a 4-line file read.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_query_drug(name: str, *, cache_path: Path = _DEFAULT_CACHE_PATH) -> dict:
    """Resolve a user-provided drug name to an RXCUI for retrieval filtering.

    Returns:
        {
            "rxcui": str | None,       # resolved id, or None if not confident
            "match_type": str,         # "exact" | "approx" | "ambiguous" | "unresolved" | "cached"
            "candidates": list[dict],  # populated only when match_type == "ambiguous",
                                        # each item {"name": str, "rxcui": str}
        }

    "ambiguous" means the name is a plausible typo of more than one real drug
    (e.g. "metfromin" matches both "merbromin" and "metformin") — retrieval
    should not filter on either guess. "unresolved" means no candidate
    resolved to a real rxcui at all.

    Checks the ingestion-time RxNorm cache (built during Phase 1) before
    hitting the live API, so a drug already resolved once doesn't need a
    fresh network call. Cache keys aren't normalized on disk (mixed casing
    straight from label XML), so the lookup is case-insensitive. A cached
    None (a known ingestion-time failure) isn't trusted as a final answer —
    that logic can't distinguish ambiguous from unresolved, so it falls
    through to a live lookup for the richer signal instead.
    """
    cache = _load_rxnorm_cache(cache_path)
    lowered = {k.lower(): v for k, v in cache.items()}
    if name.lower() in lowered:
        cached_rxcui = lowered[name.lower()]
        if cached_rxcui is not None:
            return {"rxcui": cached_rxcui, "match_type": "cached", "candidates": []}

    rxcui = find_rxcui_exact(name)
    if rxcui is not None:
        return {"rxcui": rxcui, "match_type": "exact", "candidates": []}

    candidates = find_rxcui_approx_candidates(name)
    if len(candidates) == 1:
        return {"rxcui": candidates[0]["rxcui"], "match_type": "approx", "candidates": []}
    if len(candidates) > 1:
        return {"rxcui": None, "match_type": "ambiguous", "candidates": candidates}
    return {"rxcui": None, "match_type": "unresolved", "candidates": []}
