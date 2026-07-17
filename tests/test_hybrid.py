import pytest

import backend.retrieval.hybrid as hybrid_module
from backend.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion, resolve_and_search
from tests.conftest import QUERY_VECTOR


def _ids(results):
    return {r["id"] for r in results}


class TestReciprocalRankFusion:
    def test_single_list_contribution(self):
        scores = reciprocal_rank_fusion([10, 20], [])
        assert scores == {10: pytest.approx(1 / 61), 20: pytest.approx(1 / 62)}

    def test_both_list_summation(self):
        # id 5: keyword rank 1, dense rank 2 -> sum of both contributions.
        scores = reciprocal_rank_fusion([5, 6], [7, 5])
        assert scores[5] == pytest.approx(1 / 61 + 1 / 62)

    def test_appearing_in_both_lists_can_outrank_single_list_rank_one(self):
        # id 99 is rank 2 in both lists; id 10/20 are rank 1 in only one list.
        # 2 * 1/(60+2) ~= 0.03226 > 1/(60+1) ~= 0.01639
        scores = reciprocal_rank_fusion([10, 99], [20, 99])
        assert scores[99] > scores[10]
        assert scores[99] > scores[20]

    def test_earlier_rank_scores_higher(self):
        scores = reciprocal_rank_fusion([1, 2, 3], [])
        assert scores[1] > scores[2] > scores[3]

    def test_both_lists_empty_returns_empty_dict(self):
        assert reciprocal_rank_fusion([], []) == {}

    def test_one_list_empty_scores_from_other_only(self):
        scores = reciprocal_rank_fusion([1, 2], [])
        assert scores == {1: pytest.approx(1 / 61), 2: pytest.approx(1 / 62)}

    def test_custom_k_changes_score_but_not_order(self):
        default_scores = reciprocal_rank_fusion([1, 2], [])
        custom_scores = reciprocal_rank_fusion([1, 2], [], k=1)
        assert custom_scores[1] != default_scores[1]
        assert custom_scores[1] > custom_scores[2]


@pytest.mark.db
class TestHybridSearchDB:
    def test_fusion_ordering_end_to_end(self, seeded_hybrid_chunks):
        # Rows 0 and 2 are found by both retrievers; rows 1 and 3 by only one
        # (or rank worst) — real fused scores should reflect that, not just
        # mirror a single retriever's order.
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=10
        )
        scores = {r["id"]: r["rrf_score"] for r in results}
        row0, row1, row2, row3 = (seeded_hybrid_chunks[i]["id"] for i in range(4))

        assert scores[row0] > scores[row1]
        assert scores[row2] > scores[row1]
        assert scores[row0] > scores[row3]
        assert scores[row2] > scores[row3]

    def test_chunk_found_by_both_retrievers_has_merged_metadata(self, seeded_hybrid_chunks):
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=10
        )
        by_id = {r["id"]: r for r in results}
        row0_id = seeded_hybrid_chunks[0]["id"]
        assert "rank" in by_id[row0_id]
        assert "similarity" in by_id[row0_id]

    def test_chunk_found_by_only_one_retriever_lacks_the_other_key(self, seeded_hybrid_chunks):
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=10
        )
        by_id = {r["id"]: r for r in results}
        row1_id = seeded_hybrid_chunks[1]["id"]  # dense-only (no "headache" in text)
        assert "similarity" in by_id[row1_id]
        assert "rank" not in by_id[row1_id]

    def test_rxcui_filter_excludes_other_drug(self, seeded_hybrid_chunks):
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=10
        )
        other_drug_id = seeded_hybrid_chunks[4]["id"]  # rxcui=8002
        assert other_drug_id not in _ids(results)

    def test_top_k_limits_final_results(self, seeded_hybrid_chunks):
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=1
        )
        assert len(results) == 1

    def test_query_embedding_bypass_avoids_model_load(self, seeded_hybrid_chunks, monkeypatch):
        import backend.retrieval.dense as dense_module

        def _fail(*args, **kwargs):
            raise AssertionError("model should not be loaded when query_embedding is supplied")

        monkeypatch.setattr(dense_module, "_get_model", _fail)
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=5
        )
        assert len(results) > 0

    def test_rrf_score_present_on_results(self, seeded_hybrid_chunks):
        results = hybrid_search(
            "zqlorafenib headache", query_embedding=QUERY_VECTOR, rxcui="7001", top_k=10
        )
        assert all("rrf_score" in r for r in results)


class TestResolveAndSearch:
    def test_no_drug_name_skips_resolution(self, monkeypatch):
        def _fail(*args, **kwargs):
            raise AssertionError("should not resolve when drug_name is None")

        monkeypatch.setattr(hybrid_module, "resolve_query_drug", _fail)
        monkeypatch.setattr(hybrid_module, "hybrid_search", lambda *a, **k: ["chunk"])

        result = resolve_and_search("side effects", None)

        assert result == {
            "results": ["chunk"], "filter_applied": False,
            "match_type": None, "candidates": [], "resolution_note": None,
        }

    def test_resolved_name_filters_and_reports_applied(self, monkeypatch):
        calls = []

        def fake_hybrid_search(query_text, *, rxcui=None, **kwargs):
            calls.append(rxcui)
            return ["chunk"]

        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": "6809", "match_type": "exact", "candidates": []},
        )
        monkeypatch.setattr(hybrid_module, "hybrid_search", fake_hybrid_search)

        result = resolve_and_search("side effects", "metformin")

        # Exactly one search — the resolved rxcui had results, so no second
        # (unfiltered fallback) search should have run.
        assert calls == ["6809"]
        assert result["filter_applied"] is True
        assert result["match_type"] == "exact"
        assert result["resolution_note"] is None
        assert result["results"] == ["chunk"]

    def test_resolved_but_not_indexed_falls_back_unfiltered(self, monkeypatch):
        calls = []

        def fake_hybrid_search(query_text, *, rxcui=None, **kwargs):
            calls.append(rxcui)
            # The resolved rxcui has zero chunks in the corpus; the unfiltered
            # fallback call (rxcui=None) does find results.
            return [] if rxcui is not None else ["chunk"]

        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": "202433", "match_type": "exact", "candidates": []},
        )
        monkeypatch.setattr(hybrid_module, "hybrid_search", fake_hybrid_search)

        result = resolve_and_search("dosage", "Tylenol")

        assert calls == ["202433", None]
        assert result["filter_applied"] is False
        assert result["match_type"] == "not_indexed"
        assert result["candidates"] == []
        assert "Tylenol" in result["resolution_note"]
        assert result["results"] == ["chunk"]

    def test_ambiguous_and_unresolved_search_only_once(self, monkeypatch):
        calls = []

        def fake_hybrid_search(query_text, *, rxcui=None, **kwargs):
            calls.append(rxcui)
            return ["chunk"]

        monkeypatch.setattr(hybrid_module, "hybrid_search", fake_hybrid_search)
        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": None, "match_type": "unresolved", "candidates": []},
        )

        result = resolve_and_search("side effects", "zqlorafenibxyz123")

        # rxcui was already None from resolution — never filtered, so there's
        # nothing to fall back from and only one search should run.
        assert calls == [None]
        assert result["match_type"] == "unresolved"

    def test_ambiguous_name_falls_back_unfiltered(self, monkeypatch):
        calls = {}
        candidates = [{"name": "merbromin", "rxcui": "1001"}, {"name": "metformin", "rxcui": "6809"}]

        def fake_hybrid_search(query_text, *, rxcui=None, **kwargs):
            calls["rxcui"] = rxcui
            return ["chunk"]

        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": None, "match_type": "ambiguous", "candidates": candidates},
        )
        monkeypatch.setattr(hybrid_module, "hybrid_search", fake_hybrid_search)

        result = resolve_and_search("side effects", "metfromin")

        assert calls["rxcui"] is None
        assert result["filter_applied"] is False
        assert result["match_type"] == "ambiguous"
        assert result["candidates"] == candidates
        assert "merbromin" in result["resolution_note"]
        assert "metformin" in result["resolution_note"]

    def test_unresolved_name_falls_back_unfiltered(self, monkeypatch):
        calls = {}

        def fake_hybrid_search(query_text, *, rxcui=None, **kwargs):
            calls["rxcui"] = rxcui
            return ["chunk"]

        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": None, "match_type": "unresolved", "candidates": []},
        )
        monkeypatch.setattr(hybrid_module, "hybrid_search", fake_hybrid_search)

        result = resolve_and_search("side effects", "zqlorafenibxyz123")

        assert calls["rxcui"] is None
        assert result["filter_applied"] is False
        assert result["match_type"] == "unresolved"
        assert result["candidates"] == []
        assert result["resolution_note"] is not None

    def test_results_are_passed_through_unchanged(self, monkeypatch):
        sentinel = [{"id": 1, "chunk_text": "x"}]
        monkeypatch.setattr(
            hybrid_module, "resolve_query_drug",
            lambda name: {"rxcui": "6809", "match_type": "exact", "candidates": []},
        )
        monkeypatch.setattr(hybrid_module, "hybrid_search", lambda *a, **k: sentinel)

        result = resolve_and_search("side effects", "metformin")

        assert result["results"] is sentinel
