from typing import Optional

from backend.retrieval.dense import dense_search
from backend.retrieval.rxnorm_query import resolve_query_drug
from backend.retrieval.text_search import text_search

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

_RRF_K = 60


def reciprocal_rank_fusion(keyword_ids: list, dense_ids: list, *, k: int = _RRF_K) -> dict:
    """Fuse two ranked id lists into RRF scores: {id: score}, unsorted.

    Each input is an ordered sequence of ids, most relevant first (rank 1). An
    id appearing in both lists accumulates score from each. k dampens the
    impact of rank position so a rank-1 vs. rank-2 gap doesn't dominate a
    rank-40 vs. rank-41 gap the same way raw scores would.
    """
    scores: dict = {}
    for ranked_ids in (keyword_ids, dense_ids):
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(
    query_text: str,
    *,
    query_embedding: Optional[list[float]] = None,
    rxcui: Optional[str] = None,
    retriever_top_k: int = 30,
    top_k: int = 10,
    dsn: str = _DEFAULT_DSN,
) -> list[dict]:
    """Fuse keyword (FTS) and dense (pgvector) search via Reciprocal Rank Fusion.

    Runs both retrievers sequentially against the same query/filter, fuses
    their ranked id lists, and returns the top_k chunks with full metadata
    plus an rrf_score field.
    """
    keyword_results = text_search(query_text, rxcui=rxcui, top_k=retriever_top_k, dsn=dsn)
    dense_results = dense_search(
        query_text, query_embedding=query_embedding, rxcui=rxcui, top_k=retriever_top_k, dsn=dsn
    )

    keyword_ids = [r["id"] for r in keyword_results]
    dense_ids = [r["id"] for r in dense_results]
    fused_scores = reciprocal_rank_fusion(keyword_ids, dense_ids)

    chunks_by_id: dict = {}
    for r in dense_results:
        chunks_by_id[r["id"]] = dict(r)
    for r in keyword_results:
        chunks_by_id.setdefault(r["id"], {}).update(r)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
    return [{**chunks_by_id[cid], "rrf_score": fused_scores[cid]} for cid in ranked_ids]


def resolve_and_search(
    query_text: str,
    drug_name: Optional[str] = None,
    *,
    query_embedding: Optional[list[float]] = None,
    retriever_top_k: int = 30,
    top_k: int = 10,
    dsn: str = _DEFAULT_DSN,
) -> dict:
    """Resolve drug_name to an rxcui (if given) and run a filtered hybrid_search.

    Returns:
        {
            "results": list[dict],        # same shape hybrid_search returns
            "filter_applied": bool,        # False if unfiltered, for any reason
            "match_type": str | None,      # resolve_query_drug's match_type, "not_indexed", or None
            "candidates": list[dict],      # populated only when match_type == "ambiguous"
            "resolution_note": str | None, # explains an unfiltered fallback, else None
        }

    A resolution failure (ambiguous or unresolved) falls back to an
    unfiltered hybrid_search rather than returning no results — the coverage
    gap is surfaced via filter_applied/resolution_note instead of being
    silently swallowed.

    A resolved rxcui can still be absent from the indexed corpus (e.g. RxNorm
    resolves a brand name to a brand-level concept, while chunks are stored
    under the ingredient's rxcui; or a combination product's own label was
    dropped during ingestion dedup). Filtering on such an rxcui would silently
    match nothing, so the filtered search's own result is used as the check —
    a resolved rxcui that's actually absent from the corpus returns zero
    results (dense_search's ANN component never applies a relevance
    threshold, so a real rxcui with any chunks at all is guaranteed at least
    one match), which is treated the same as an outright unresolved name and
    falls back to an unfiltered search.
    """
    rxcui = None
    match_type = None
    candidates: list = []

    if drug_name is not None:
        resolution = resolve_query_drug(drug_name)
        rxcui = resolution["rxcui"]
        match_type = resolution["match_type"]
        candidates = resolution["candidates"]

    results = []
    if rxcui is not None:
        results = hybrid_search(
            query_text, query_embedding=query_embedding, rxcui=rxcui,
            retriever_top_k=retriever_top_k, top_k=top_k, dsn=dsn,
        )
        if not results:
            rxcui = None
            match_type = "not_indexed"
            candidates = []

    if rxcui is None:
        results = hybrid_search(
            query_text, query_embedding=query_embedding, rxcui=None,
            retriever_top_k=retriever_top_k, top_k=top_k, dsn=dsn,
        )

    resolution_note = None
    if match_type == "ambiguous":
        names = ", ".join(c["name"] for c in candidates)
        resolution_note = f"'{drug_name}' is ambiguous (could be: {names}); searched without a drug filter"
    elif match_type == "unresolved":
        resolution_note = f"could not resolve '{drug_name}' to a known drug; searched without a drug filter"
    elif match_type == "not_indexed":
        resolution_note = (
            f"'{drug_name}' resolved to a real drug, but it isn't represented "
            "in the indexed corpus; searched without a drug filter"
        )

    return {
        "results": results,
        "filter_applied": rxcui is not None,
        "match_type": match_type,
        "candidates": candidates,
        "resolution_note": resolution_note,
    }
