import psycopg
import pytest

import backend.retrieval.dense as dense_module
from backend.retrieval.dense import dense_search
from tests.conftest import DSN, QUERY_VECTOR, _unit_vector

pytestmark = pytest.mark.db


def _ids(results):
    return {r["id"] for r in results}


class TestBasicMatching:
    def test_ranking_follows_cosine_similarity(self, seeded_dense_chunks):
        # Rows are at known angles from QUERY_VECTOR (see conftest docstring),
        # so the expected order is exact, not just plausible.
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=10)
        ranked_ids = [r["id"] for r in results]
        close_id = seeded_dense_chunks[0]["id"]  # VEC_CLOSE
        far_id = seeded_dense_chunks[1]["id"]    # VEC_FAR
        assert ranked_ids.index(close_id) < ranked_ids.index(far_id)

    def test_similarity_values_match_expected_cosine(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=10)
        by_id = {r["id"]: r["similarity"] for r in results}
        close_id = seeded_dense_chunks[0]["id"]
        far_id = seeded_dense_chunks[1]["id"]
        # VEC_CLOSE = {0: 0.95, 1: 0.05} normalized; cosine similarity to a pure
        # dim-0 unit vector is 0.95 / ||VEC_CLOSE|| ~= 0.9986.
        assert by_id[close_id] == pytest.approx(0.9986, abs=1e-3)
        # VEC_FAR = {0: 0.5, 2: 0.5} normalized; same ratio, 0.5 / ||VEC_FAR||.
        assert by_id[far_id] == pytest.approx(0.7071, abs=1e-3)

    def test_orthogonal_vector_has_near_zero_similarity(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="3003", top_k=10)
        assert results[0]["similarity"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vector_has_negative_similarity(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="4004", top_k=10)
        assert results[0]["similarity"] == pytest.approx(-1.0, abs=1e-6)


class TestNoThresholdSemantics:
    # Unlike text_search, ANN search always returns the nearest available rows —
    # there's no relevance threshold at this layer (see PHASE2_PLAN.md's
    # "Confidence threshold" decision). An irrelevant query still returns
    # results; only an rxcui filter that matches zero rows returns [].
    def test_unfiltered_search_returns_results_even_for_distant_query(self, seeded_dense_chunks):
        far_query = _unit_vector({999: 1.0})  # unrelated to every fixture row
        results = dense_search("unused", query_embedding=far_query, top_k=5)
        assert len(results) > 0

    def test_filter_with_no_matching_rxcui_returns_empty(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="9999", top_k=10)
        assert results == []


class TestRxcuiFilter:
    def test_filter_restricts_to_rxcui(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=10)
        assert all(r["rxcui"] == "1001" for r in results)
        other_drug_id = seeded_dense_chunks[2]["id"]  # rxcui 2002, identical embedding
        assert other_drug_id not in _ids(results)

    def test_unfiltered_searches_across_drugs(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, top_k=10)
        rxcuis = {r["rxcui"] for r in results}
        assert {"1001", "2002"} <= rxcuis


class TestTopK:
    def test_top_k_limits_to_best_ranked(self, seeded_dense_chunks):
        full = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=10)
        limited = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=1)
        assert len(limited) == 1
        assert limited[0]["id"] == full[0]["id"]


class TestQueryEmbeddingBypass:
    def test_supplying_query_embedding_skips_model_load(self, seeded_dense_chunks, monkeypatch):
        # If dense_search touched the real model here, this would fail (or hang
        # downloading BGE-large) since _get_model is poisoned to raise.
        import backend.retrieval.dense as dense_module

        def _fail(*args, **kwargs):
            raise AssertionError("model should not be loaded when query_embedding is supplied")

        monkeypatch.setattr(dense_module, "_get_model", _fail)
        results = dense_search("irrelevant text", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=5)
        assert len(results) > 0


class TestRobustness:
    def test_wrong_dimension_embedding_raises(self, seeded_dense_chunks):
        with pytest.raises(psycopg.Error):
            dense_search("unused", query_embedding=[0.1, 0.2, 0.3], rxcui="1001", top_k=5)

    def test_rxcui_with_sql_metacharacters_is_treated_as_literal(self, seeded_dense_chunks):
        # Parameters are bound, never string-interpolated, so this is inert.
        results = dense_search(
            "unused", query_embedding=QUERY_VECTOR, rxcui="'; DROP TABLE chunks; --", top_k=5
        )
        assert results == []
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('chunks')")
                assert cur.fetchone()[0] == "chunks"


class TestResultShape:
    def test_rows_have_expected_keys(self, seeded_dense_chunks):
        results = dense_search("unused", query_embedding=QUERY_VECTOR, rxcui="1001", top_k=5)
        expected = {
            "id", "setid", "drug_name", "rxcui", "loinc_code",
            "section_title_path", "section_type", "chunk_text",
            "token_count", "similarity",
        }
        assert expected <= set(results[0].keys())


class TestIndexUsage:
    def test_query_uses_raw_cosine_distance_expression_not_alias(self):
        # Regression guard: ORDER BY must reference the raw <=> expression (not
        # a "similarity" alias) or Postgres can't recognize it against
        # chunks_embedding_idx and falls back to a sequential scan. This can't
        # be checked via a live EXPLAIN against a small seeded fixture table —
        # Postgres's planner correctly prefers a sequential scan over the
        # index on a handful of rows regardless of query shape, since a seq
        # scan really is cheaper at that size (confirmed: this assertion only
        # holds against the real ~434k-row corpus, verified manually via
        # EXPLAIN ANALYZE during Block 2 — see PHASE2_PLAN.md). So this checks
        # the query text itself, which is what actually matters and is
        # environment-independent.
        assert "ORDER BY embedding <=>" in dense_module._QUERY
        assert "ORDER BY similarity" not in dense_module._QUERY
