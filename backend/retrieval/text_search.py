from typing import Optional

import psycopg
from psycopg.rows import dict_row

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

_COLUMNS = """
    id, setid, drug_name, rxcui, loinc_code, section_title_path, section_type,
    chunk_text, token_count
"""

_QUERY = """
    SELECT {columns},
           ts_rank_cd(to_tsvector('english', chunk_text), query) AS rank
    FROM chunks, websearch_to_tsquery('english', %(query_text)s) query
    WHERE to_tsvector('english', chunk_text) @@ query
    {rxcui_filter}
    ORDER BY rank DESC
    LIMIT %(top_k)s
"""


def text_search(
    query_text: str,
    *,
    rxcui: Optional[str] = None,
    top_k: int = 30,
    dsn: str = _DEFAULT_DSN,
) -> list[dict]:
    """Keyword search over chunk_text via Postgres FTS, ranked by ts_rank_cd.

    Recomputes to_tsvector('english', chunk_text) to match the expression
    index chunks_fts_idx, so Postgres can use it instead of scanning.
    """
    rxcui_filter = "AND rxcui = %(rxcui)s" if rxcui is not None else ""
    sql = _QUERY.format(columns=_COLUMNS, rxcui_filter=rxcui_filter)

    params = {"query_text": query_text, "top_k": top_k}
    if rxcui is not None:
        params["rxcui"] = rxcui

    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
