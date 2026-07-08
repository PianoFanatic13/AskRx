import uuid

import psycopg
import pytest
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
