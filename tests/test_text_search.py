import pytest

from backend.retrieval.text_search import get_section, text_search

pytestmark = pytest.mark.db


def _ids(results):
    return {r["id"] for r in results}


class TestBasicMatching:
    def test_matching_chunks_returned_unrelated_excluded(self, seeded_chunks):
        # "zqlorafenib" anchors the query to only our synthetic rows, so this
        # is an exact check, not a race against the real ~434k-row corpus.
        results = text_search("zqlorafenib headache")
        result_ids = _ids(results)

        expected_ids = {seeded_chunks[i]["id"] for i in (0, 1, 2)}  # contain "headache"
        excluded_ids = {seeded_chunks[i]["id"] for i in (3, 4)}     # no "headache"

        assert expected_ids <= result_ids
        assert excluded_ids.isdisjoint(result_ids)

    def test_ranking_favors_denser_match(self, seeded_chunks):
        # Scoped to one rxcui so the two chunks with different match density
        # aren't tied against row 2's identical-text duplicate under another drug.
        results = text_search("zqlorafenib headache", rxcui="1001")
        dense_id = seeded_chunks[0]["id"]   # "headache" x2
        sparse_id = seeded_chunks[1]["id"]  # "headache" x1
        ranked_ids = [r["id"] for r in results]
        assert ranked_ids.index(dense_id) < ranked_ids.index(sparse_id)

    def test_stemming_matches_plural(self, seeded_chunks):
        results = text_search("zqlorafenib side effect")
        assert _ids(results) == {seeded_chunks[3]["id"]}  # chunk says "side effects"

    def test_no_match_returns_empty(self, seeded_chunks):
        assert text_search("xylophone quantum bicycle") == []

    def test_misspelled_term_does_not_fuzzy_match(self, seeded_chunks):
        # FTS only stems/normalizes — no edit-distance tolerance. "heedache" is
        # a near-miss of "headache" (present in rows 0-2), but it must behave
        # exactly like unrelated gibberish: zero results, not a partial match.
        assert text_search("zqlorafenib heedache") == []


class TestRxcuiFilter:
    def test_filter_restricts_to_rxcui(self, seeded_chunks):
        results = text_search("zqlorafenib headache", rxcui="1001")
        assert all(r["rxcui"] == "1001" for r in results)
        other_drug_id = seeded_chunks[2]["id"]  # rxcui 2002, identical text
        assert other_drug_id not in _ids(results)

    def test_filter_with_no_matches_returns_empty(self, seeded_chunks):
        assert text_search("zqlorafenib headache", rxcui="9999") == []

    def test_unfiltered_searches_across_drugs(self, seeded_chunks):
        results = text_search("zqlorafenib headache")
        rxcuis = {r["rxcui"] for r in results}
        assert {"1001", "2002"} <= rxcuis


class TestTopK:
    def test_top_k_limits_to_best_ranked(self, seeded_chunks):
        full = text_search("headache", rxcui="1001", top_k=10)
        limited = text_search("headache", rxcui="1001", top_k=1)
        assert len(limited) == 1
        assert limited[0]["id"] == full[0]["id"]


class TestRobustness:
    def test_empty_query_returns_empty(self, seeded_chunks):
        assert text_search("") == []

    def test_punctuation_heavy_query_does_not_error(self, seeded_chunks):
        text_search('5-fluorouracil (topical) -- "side effects"?')

    def test_sql_metacharacters_are_treated_as_literal_text(self, seeded_chunks):
        # Must not error, and must not affect the table — parameters are bound,
        # never string-interpolated, so this is inert search text.
        text_search("'; DROP TABLE chunks; --")
        remaining = text_search("headache")
        assert len(remaining) > 0


class TestResultShape:
    def test_rows_have_expected_keys(self, seeded_chunks):
        results = text_search("headache")
        expected = {
            "id", "setid", "drug_name", "rxcui", "loinc_code",
            "section_title_path", "section_type", "chunk_text",
            "token_count", "rank",
        }
        assert expected <= set(results[0].keys())


class TestGetSection:
    def test_returns_section_chunks_in_document_order(self, seeded_section_chunks):
        results = get_section("9001", "34071-1")
        result_ids = [r["id"] for r in results]
        expected_ids = [seeded_section_chunks[i]["id"] for i in (0, 1, 2)]
        assert result_ids == expected_ids

    def test_other_rxcui_excluded(self, seeded_section_chunks):
        results = get_section("9001", "34071-1")
        other_drug_id = seeded_section_chunks[3]["id"]  # rxcui=9002
        assert other_drug_id not in {r["id"] for r in results}

    def test_other_section_same_drug_excluded(self, seeded_section_chunks):
        results = get_section("9001", "34071-1")
        other_section_id = seeded_section_chunks[4]["id"]  # loinc_code=34090-1
        assert other_section_id not in {r["id"] for r in results}

    def test_no_match_returns_empty(self, seeded_section_chunks):
        assert get_section("0000", "00000-0") == []

    def test_rows_have_expected_keys(self, seeded_section_chunks):
        results = get_section("9001", "34071-1")
        expected = {
            "id", "setid", "drug_name", "rxcui", "loinc_code",
            "section_title_path", "section_type", "chunk_text", "token_count",
        }
        assert expected <= set(results[0].keys())
