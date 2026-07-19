from unittest.mock import patch

from backend.agent.tools import resolve_drug_name, retrieve_drug_info, retrieve_interactions

_MODULE = "backend.agent.tools"


def test_resolve_drug_name_passes_through():
    fake_result = {"rxcui": "6809", "match_type": "exact", "candidates": []}
    with patch(f"{_MODULE}.resolve_query_drug", return_value=fake_result) as mock:
        result = resolve_drug_name.invoke({"name": "metformin"})
    assert result == fake_result
    mock.assert_called_once_with("metformin")


def test_retrieve_drug_info_passes_through():
    fake_result = [{"id": 1, "chunk_text": "..."}]
    with patch(f"{_MODULE}.hybrid_search", return_value=fake_result) as mock:
        result = retrieve_drug_info.invoke({"query_text": "side effects", "rxcui": "6809"})
    assert result == fake_result
    mock.assert_called_once_with("side effects", rxcui="6809")


def test_retrieve_interactions_passes_through():
    fake_result = [{"id": 2, "chunk_text": "..."}]
    with patch(f"{_MODULE}.get_section", return_value=fake_result) as mock:
        result = retrieve_interactions.invoke({"rxcui": "6809"})
    assert result == fake_result
    mock.assert_called_once_with("6809", "34073-7")
