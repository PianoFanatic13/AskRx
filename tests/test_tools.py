from unittest.mock import call, patch

from backend.agent.tools import resolve_drug_name, retrieve_drug_info, retrieve_interactions

_MODULE = "backend.agent.tools"

_RAW_CHUNK = {
    "id": 194448,
    "setid": "abc-123",
    "drug_name": "Metformin",
    "rxcui": "6809",
    "loinc_code": "34084-4",
    "section_title_path": ["6 ADVERSE REACTIONS"],
    "section_type": "standard",
    "chunk_text": "Common side effects include...",
    "token_count": 42,
    "rrf_score": 0.031,
}
_TRIMMED_CHUNK = {
    "setid": "abc-123",
    "drug_name": "Metformin",
    "loinc_code": "34084-4",
    "section_title_path": ["6 ADVERSE REACTIONS"],
    "section_type": "standard",
    "chunk_text": "Common side effects include...",
}


def test_resolve_drug_name_passes_through():
    fake_result = {"rxcui": "6809", "match_type": "exact", "candidates": []}
    with patch(f"{_MODULE}.resolve_query_drug", return_value=fake_result) as mock:
        result = resolve_drug_name.invoke({"name": "metformin"})
    assert result == fake_result
    mock.assert_called_once_with("metformin")


class TestRetrieveDrugInfo:
    def test_resolved_returns_trimmed_results(self):
        resolution = {"rxcui": "6809", "match_type": "exact", "candidates": []}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.hybrid_search", return_value=[_RAW_CHUNK]) as mock_search,
        ):
            result = retrieve_drug_info.invoke({"query_text": "side effects", "drug_name": "metformin"})
        assert result == {"results": [_TRIMMED_CHUNK], "match_type": "exact", "candidates": []}
        mock_search.assert_called_once_with("side effects", rxcui="6809")

    def test_ambiguous_short_circuits_without_searching(self):
        candidates = [{"name": "merbromin", "rxcui": "1001"}, {"name": "metformin", "rxcui": "6809"}]
        resolution = {"rxcui": None, "match_type": "ambiguous", "candidates": candidates}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.hybrid_search") as mock_search,
        ):
            result = retrieve_drug_info.invoke({"query_text": "dosage", "drug_name": "metfromin"})
        assert result == {"results": [], "match_type": "ambiguous", "candidates": candidates}
        mock_search.assert_not_called()

    def test_unresolved_searches_unfiltered(self):
        resolution = {"rxcui": None, "match_type": "unresolved", "candidates": []}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.hybrid_search", return_value=[_RAW_CHUNK]) as mock_search,
        ):
            result = retrieve_drug_info.invoke({"query_text": "headache", "drug_name": "not a drug"})
        assert result == {"results": [_TRIMMED_CHUNK], "match_type": "unresolved", "candidates": []}
        mock_search.assert_called_once_with("headache", rxcui=None)

    def test_resolved_but_empty_retries_unfiltered(self):
        resolution = {"rxcui": "202433", "match_type": "exact", "candidates": []}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.hybrid_search", side_effect=[[], [_RAW_CHUNK]]) as mock_search,
        ):
            result = retrieve_drug_info.invoke({"query_text": "dosage", "drug_name": "Tylenol"})
        assert result == {"results": [_TRIMMED_CHUNK], "match_type": "not_indexed", "candidates": []}
        assert mock_search.call_args_list == [
            call("dosage", rxcui="202433"),
            call("dosage", rxcui=None),
        ]


class TestRetrieveInteractions:
    def test_resolved_returns_trimmed_results(self):
        resolution = {"rxcui": "6809", "match_type": "exact", "candidates": []}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.get_section", return_value=[_RAW_CHUNK]) as mock_section,
        ):
            result = retrieve_interactions.invoke({"drug_name": "metformin"})
        assert result == {"results": [_TRIMMED_CHUNK], "match_type": "exact", "candidates": []}
        mock_section.assert_called_once_with("6809", "34073-7")

    def test_ambiguous_short_circuits_without_fetching(self):
        candidates = [{"name": "merbromin", "rxcui": "1001"}, {"name": "metformin", "rxcui": "6809"}]
        resolution = {"rxcui": None, "match_type": "ambiguous", "candidates": candidates}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.get_section") as mock_section,
        ):
            result = retrieve_interactions.invoke({"drug_name": "metfromin"})
        assert result == {"results": [], "match_type": "ambiguous", "candidates": candidates}
        mock_section.assert_not_called()

    def test_unresolved_short_circuits_without_fetching(self):
        resolution = {"rxcui": None, "match_type": "unresolved", "candidates": []}
        with (
            patch(f"{_MODULE}.resolve_query_drug", return_value=resolution),
            patch(f"{_MODULE}.get_section") as mock_section,
        ):
            result = retrieve_interactions.invoke({"drug_name": "not a drug"})
        assert result == {"results": [], "match_type": "unresolved", "candidates": []}
        mock_section.assert_not_called()
