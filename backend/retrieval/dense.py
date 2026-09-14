import logging
import os
import time
from typing import Optional

import psycopg
import requests
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

load_dotenv()

log = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"
_MODEL_ID = "BAAI/bge-large-en-v1.5"
_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{_MODEL_ID}/pipeline/feature-extraction"

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

def _embed_query(text: str) -> list[float]:
    """Embed a single query string via HF Inference Providers (hf-inference, BGE-large).

    normalize=True matches ingestion's normalize_embeddings=True. Retries on
    503 (model cold-starting) and transient network errors, matching the
    backoff pattern in pipeline/rxnorm.py.
    """
    token = os.environ["HF_KEY"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text, "normalize": True}

    retries = 3
    for attempt in range(retries):
        try:
            resp = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning("HF embedding request failed (%s), retrying in %ds", e, wait)
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (429, 500, 502, 503, 504):
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("HF embedding HTTP %d, retrying in %ds", e.response.status_code, wait)
                time.sleep(wait)
            else:
                raise


def dense_search(
    query_text: str,
    *,
    query_embedding: Optional[list[float]] = None,
    rxcui: Optional[str] = None,
    top_k: int = 30,
    dsn: str = _DEFAULT_DSN,
) -> list[dict]:
    """Semantic search over chunk embeddings via pgvector cosine similarity.

    Embeds query_text via the HF-hosted BGE-large model (normalize=True,
    matching ingestion) unless query_embedding is supplied directly — tests
    and other callers that already have a vector can bypass the API call
    entirely.
    """
    if query_embedding is None:
        query_embedding = _embed_query(query_text)

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
