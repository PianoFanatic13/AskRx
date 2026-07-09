from typing import Optional

from backend.retrieval.dense import dense_search
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
