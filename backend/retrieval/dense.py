from typing import Optional

import psycopg
import torch
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"
_MODEL_ID = "BAAI/bge-large-en-v1.5"

_COLUMNS = """
    id, setid, drug_name, rxcui, loinc_code, section_title_path, section_type,
    chunk_text, token_count
"""

# ORDER BY uses the raw <=> expression (not the derived "similarity" alias) so
# Postgres recognizes it against the chunks_embedding_idx IVFFlat index and
# runs an approximate nearest-neighbor search instead of a full sort.
_QUERY = """
    SELECT {columns},
           1 - (embedding <=> %(query_vector)s::vector) AS similarity
    FROM chunks
    {rxcui_filter}
    ORDER BY embedding <=> %(query_vector)s::vector
    LIMIT %(top_k)s
"""

_model: Optional[SentenceTransformer] = None


def _get_model(model_id: str = _MODEL_ID) -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(model_id, device=device)
    return _model


def dense_search(
    query_text: str,
    *,
    query_embedding: Optional[list[float]] = None,
    rxcui: Optional[str] = None,
    top_k: int = 30,
    dsn: str = _DEFAULT_DSN,
) -> list[dict]:
    """Semantic search over chunk embeddings via pgvector cosine similarity.

    Embeds query_text with BGE-large (normalize_embeddings=True, matching
    ingestion) unless query_embedding is supplied directly — tests and other
    callers that already have a vector can bypass loading the model entirely.
    """
    if query_embedding is None:
        model = _get_model()
        query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()

    rxcui_filter = "WHERE rxcui = %(rxcui)s" if rxcui is not None else ""
    sql = _QUERY.format(columns=_COLUMNS, rxcui_filter=rxcui_filter)

    params = {"query_vector": query_embedding, "top_k": top_k}
    if rxcui is not None:
        params["rxcui"] = rxcui

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
