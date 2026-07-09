import uuid

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

# Marks every inserted row's setid so teardown can delete exactly these rows
# and nothing else — this fixture must never touch real ingested data.
_TEST_PREFIX = "TEST-"

_INSERT = """
    INSERT INTO chunks (setid, drug_name, rxcui, loinc_code, section_type, chunk_text, token_count)
    VALUES (%(setid)s, %(drug_name)s, %(rxcui)s, %(loinc_code)s, %(section_type)s, %(chunk_text)s, %(token_count)s)
    RETURNING id
"""


@pytest.fixture
def seeded_chunks():
    """Insert a small, known set of fake chunks for DB-dependent retrieval tests.

    Every row's text includes "zqlorafenib" — a made-up word guaranteed not to
    appear anywhere in the real ingested corpus. Tests that need to reason about
    an exact result set (not just "my rows are somewhere in the top_k") search on
    that marker so they're isolated from the ~434k real rows already in the dev
    database, instead of racing real chunks for ranking position.

    Rows (index reference for tests):
      0: drugavir/1001 — "headache" mentioned twice (dense match)
      1: drugavir/1001 — "headache" mentioned once (sparse match)
      2: drugbex/2002  — identical text to row 0, different rxcui (filter test)
      3: drugcin/3003  — "side effects" plural (stemming test)
      4: drugdol/4004  — no "headache"/"side effect" (no-match control)
    """
    run_id = uuid.uuid4().hex[:8]

    def sid(n):
        return f"{_TEST_PREFIX}{run_id}-{n}"

    rows = [
        dict(
            setid=sid(1), drug_name="drugavir", rxcui="1001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib headache is a common adverse reaction. "
                        "Zqlorafenib headache may also occur with dizziness.",
            token_count=16,
        ),
        dict(
            setid=sid(2), drug_name="drugavir", rxcui="1001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Patients using zqlorafenib may report headache in rare cases.",
            token_count=10,
        ),
        dict(
            setid=sid(3), drug_name="drugbex", rxcui="2002", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib headache is a common adverse reaction. "
                        "Zqlorafenib headache may also occur with dizziness.",
            token_count=16,
        ),
        dict(
            setid=sid(4), drug_name="drugcin", rxcui="3003", loinc_code="34084-4",
            section_type="standard",
            chunk_text="Common zqlorafenib side effects include nausea and fatigue.",
            token_count=9,
        ),
        dict(
            setid=sid(5), drug_name="drugdol", rxcui="4004", loinc_code="34090-1",
            section_type="standard",
            chunk_text="Store zqlorafenib away from light and moisture at room temperature.",
            token_count=11,
        ),
    ]

    with psycopg.connect(DSN) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            for row in rows:
                cur.execute(_INSERT, row)
                row["id"] = cur.fetchone()["id"]
        conn.commit()

    try:
        yield rows
    finally:
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE setid LIKE %s",
                    (f"{_TEST_PREFIX}{run_id}-%",),
                )
            conn.commit()


_EMBED_DIM = 1024

_INSERT_WITH_EMBEDDING = """
    INSERT INTO chunks (setid, drug_name, rxcui, loinc_code, section_type, chunk_text, token_count, embedding)
    VALUES (%(setid)s, %(drug_name)s, %(rxcui)s, %(loinc_code)s, %(section_type)s, %(chunk_text)s, %(token_count)s, %(embedding)s)
    RETURNING id
"""


def _unit_vector(active_dims: dict) -> list:
    """Build a unit-length 1024-dim vector with the given {dim_index: weight} set.

    Hand-picked sparse vectors give exact, predictable cosine similarity between
    fixture rows without needing a real embedding model in tests.
    """
    vec = [0.0] * _EMBED_DIM
    for idx, weight in active_dims.items():
        vec[idx] = weight
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


# QUERY_VECTOR points purely along dim 0. The fixture rows are placed at known
# angles from it so expected similarity ordering is exact, not just plausible.
QUERY_VECTOR = _unit_vector({0: 1.0})
VEC_CLOSE = _unit_vector({0: 0.95, 1: 0.05})       # near-identical direction to query
VEC_FAR = _unit_vector({0: 0.5, 2: 0.5})           # same general direction, weaker
VEC_ORTHOGONAL = _unit_vector({500: 1.0})          # unrelated, cosine similarity ~0
VEC_OPPOSITE = _unit_vector({0: -1.0})             # opposite direction, cosine similarity -1


@pytest.fixture
def seeded_dense_chunks():
    """Insert fake chunks with hand-picked embeddings for dense-search tests.

    Mirrors seeded_chunks' row shape (dense vs. sparse match, a duplicate under
    a different rxcui for filter tests, a control row) but keyed on cosine
    distance from QUERY_VECTOR instead of keyword overlap.

    Rows (index reference for tests):
      0: drugavir/1001 — VEC_CLOSE  (most similar to QUERY_VECTOR)
      1: drugavir/1001 — VEC_FAR    (same direction, less similar)
      2: drugbex/2002  — VEC_CLOSE  (identical embedding to row 0, different rxcui)
      3: drugcin/3003  — VEC_ORTHOGONAL (~0 similarity — irrelevant, not "no match")
      4: drugdol/4004  — VEC_OPPOSITE   (negative similarity — least similar)
    """
    run_id = uuid.uuid4().hex[:8]

    def sid(n):
        return f"{_TEST_PREFIX}{run_id}-{n}"

    rows = [
        dict(
            setid=sid(1), drug_name="drugavir", rxcui="1001", loinc_code="34071-1",
            section_type="standard", chunk_text="zqlorafenib fixture row 0",
            token_count=4, embedding=VEC_CLOSE,
        ),
        dict(
            setid=sid(2), drug_name="drugavir", rxcui="1001", loinc_code="34071-1",
            section_type="standard", chunk_text="zqlorafenib fixture row 1",
            token_count=4, embedding=VEC_FAR,
        ),
        dict(
            setid=sid(3), drug_name="drugbex", rxcui="2002", loinc_code="34071-1",
            section_type="standard", chunk_text="zqlorafenib fixture row 2",
            token_count=4, embedding=VEC_CLOSE,
        ),
        dict(
            setid=sid(4), drug_name="drugcin", rxcui="3003", loinc_code="34084-4",
            section_type="standard", chunk_text="zqlorafenib fixture row 3",
            token_count=4, embedding=VEC_ORTHOGONAL,
        ),
        dict(
            setid=sid(5), drug_name="drugdol", rxcui="4004", loinc_code="34090-1",
            section_type="standard", chunk_text="zqlorafenib fixture row 4",
            token_count=4, embedding=VEC_OPPOSITE,
        ),
    ]

    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            for row in rows:
                cur.execute(_INSERT_WITH_EMBEDDING, row)
                row["id"] = cur.fetchone()["id"]
        conn.commit()

    try:
        yield rows
    finally:
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE setid LIKE %s",
                    (f"{_TEST_PREFIX}{run_id}-%",),
                )
            conn.commit()


@pytest.fixture
def seeded_section_chunks():
    """Insert fake chunks for get_section's exact rxcui+loinc_code fetch tests.

    Rows (index reference for tests):
      0, 1, 2: drugsec/9001, loinc_code=34071-1 — the target section, 3 chunks
               inserted in order, so their ids ascend in document order
      3: drugsec/9002, loinc_code=34071-1 — other-rxcui control (filter test)
      4: drugsec/9001, loinc_code=34090-1 — other-section, same drug control
         (confirms loinc_code filter isn't ignored)
    """
    run_id = uuid.uuid4().hex[:8]

    def sid(n):
        return f"{_TEST_PREFIX}{run_id}-{n}"

    rows = [
        dict(
            setid=sid(1), drug_name="drugsec", rxcui="9001", loinc_code="34071-1",
            section_type="standard", chunk_text="Zqlorafenib section chunk 1 of 3.",
            token_count=6,
        ),
        dict(
            setid=sid(2), drug_name="drugsec", rxcui="9001", loinc_code="34071-1",
            section_type="standard", chunk_text="Zqlorafenib section chunk 2 of 3.",
            token_count=6,
        ),
        dict(
            setid=sid(3), drug_name="drugsec", rxcui="9001", loinc_code="34071-1",
            section_type="standard", chunk_text="Zqlorafenib section chunk 3 of 3.",
            token_count=6,
        ),
        dict(
            setid=sid(4), drug_name="drugsex", rxcui="9002", loinc_code="34071-1",
            section_type="standard", chunk_text="Zqlorafenib other-drug control chunk.",
            token_count=5,
        ),
        dict(
            setid=sid(5), drug_name="drugsec", rxcui="9001", loinc_code="34090-1",
            section_type="standard", chunk_text="Zqlorafenib other-section control chunk.",
            token_count=5,
        ),
    ]

    with psycopg.connect(DSN) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            for row in rows:
                cur.execute(_INSERT, row)
                row["id"] = cur.fetchone()["id"]
        conn.commit()

    try:
        yield rows
    finally:
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE setid LIKE %s",
                    (f"{_TEST_PREFIX}{run_id}-%",),
                )
            conn.commit()


@pytest.fixture
def seeded_hybrid_chunks():
    """Insert fake chunks with keyword rank and dense rank deliberately crossed.

    Neither seeded_chunks (real text, no embedding) nor seeded_dense_chunks
    (embeddings, generic text) is enough on its own to prove hybrid_search
    actually fuses two retrievers instead of just echoing one. Rows here are
    scoped to rxcui=7001 (except row 4, the filter-test row) so isolation from
    both the real corpus and the other synthetic fixtures doesn't depend only
    on the "zqlorafenib" marker.

    Rows (index reference for tests), keyword/dense rank both scoped to rxcui=7001:
      0: "headache" x3, VEC_ORTHOGONAL      — keyword rank 1, dense rank 3
      1: no "headache" (absent from FTS),
         VEC_CLOSE                          — keyword: absent, dense rank 1
      2: "headache" x1, VEC_FAR             — keyword rank 2, dense rank 2
      3: no "headache" (absent from FTS),
         VEC_OPPOSITE                       — keyword: absent, dense rank 4 (worst)
      4: same text as row 0, VEC_ORTHOGONAL,
         rxcui=8002                         — filter-test row (other drug)
    """
    run_id = uuid.uuid4().hex[:8]

    def sid(n):
        return f"{_TEST_PREFIX}{run_id}-{n}"

    rows = [
        dict(
            setid=sid(1), drug_name="drugfus", rxcui="7001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib headache is common. Zqlorafenib headache occurs "
                        "frequently. Zqlorafenib headache is reported often.",
            token_count=16, embedding=VEC_ORTHOGONAL,
        ),
        dict(
            setid=sid(2), drug_name="drugfus", rxcui="7001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib is stored at room temperature away from moisture.",
            token_count=10, embedding=VEC_CLOSE,
        ),
        dict(
            setid=sid(3), drug_name="drugfus", rxcui="7001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib headache may occur.",
            token_count=6, embedding=VEC_FAR,
        ),
        dict(
            setid=sid(4), drug_name="drugfus", rxcui="7001", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib packaging includes a child-resistant cap.",
            token_count=8, embedding=VEC_OPPOSITE,
        ),
        dict(
            setid=sid(5), drug_name="drugfux", rxcui="8002", loinc_code="34071-1",
            section_type="standard",
            chunk_text="Zqlorafenib headache is common. Zqlorafenib headache occurs "
                        "frequently. Zqlorafenib headache is reported often.",
            token_count=16, embedding=VEC_ORTHOGONAL,
        ),
    ]

    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            for row in rows:
                cur.execute(_INSERT_WITH_EMBEDDING, row)
                row["id"] = cur.fetchone()["id"]
        conn.commit()

    try:
        yield rows
    finally:
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE setid LIKE %s",
                    (f"{_TEST_PREFIX}{run_id}-%",),
                )
            conn.commit()
