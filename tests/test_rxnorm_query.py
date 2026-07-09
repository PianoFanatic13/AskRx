import json
from unittest.mock import patch

from backend.retrieval.rxnorm_query import resolve_query_drug

_MODULE = "backend.retrieval.rxnorm_query"


def _write_cache(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


class TestExactAndApproxResolution:
    def test_exact_match_short_circuits_approx(self, tmp_path):
        with (
            patch(f"{_MODULE}.find_rxcui_exact", return_value="6809") as mock_exact,
            patch(f"{_MODULE}.find_rxcui_approx_candidates") as mock_approx,
        ):
            result = resolve_query_drug("metformin", cache_path=tmp_path / "missing.json")

        assert result == {"rxcui": "6809", "match_type": "exact", "candidates": []}
        mock_exact.assert_called_once_with("metformin")
        mock_approx.assert_not_called()

    def test_exact_fails_single_approx_candidate(self, tmp_path):
        with (
            patch(f"{_MODULE}.find_rxcui_exact", return_value=None),
            patch(
                f"{_MODULE}.find_rxcui_approx_candidates",
                return_value=[{"name": "lisinopril", "rxcui": "29046"}],
            ),
        ):
            result = resolve_query_drug("lisinipril", cache_path=tmp_path / "missing.json")

        assert result == {"rxcui": "29046", "match_type": "approx", "candidates": []}

    def test_exact_fails_multiple_approx_candidates_is_ambiguous(self, tmp_path):
        candidates = [
            {"name": "merbromin", "rxcui": "1001"},
            {"name": "metformin", "rxcui": "6809"},
        ]
        with (
            patch(f"{_MODULE}.find_rxcui_exact", return_value=None),
            patch(f"{_MODULE}.find_rxcui_approx_candidates", return_value=candidates),
        ):
            result = resolve_query_drug("metfromin", cache_path=tmp_path / "missing.json")

        assert result == {"rxcui": None, "match_type": "ambiguous", "candidates": candidates}

    def test_exact_and_approx_both_fail_is_unresolved(self, tmp_path):
        with (
            patch(f"{_MODULE}.find_rxcui_exact", return_value=None),
            patch(f"{_MODULE}.find_rxcui_approx_candidates", return_value=[]),
        ):
            result = resolve_query_drug("zqlorafenibxyz123", cache_path=tmp_path / "missing.json")

        assert result == {"rxcui": None, "match_type": "unresolved", "candidates": []}


class TestCache:
    def test_cache_hit_skips_network_entirely(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        _write_cache(cache_path, {"metformin": "6809"})

        with (
            patch(f"{_MODULE}.find_rxcui_exact") as mock_exact,
            patch(f"{_MODULE}.find_rxcui_approx_candidates") as mock_approx,
        ):
            result = resolve_query_drug("metformin", cache_path=cache_path)

        assert result == {"rxcui": "6809", "match_type": "cached", "candidates": []}
        mock_exact.assert_not_called()
        mock_approx.assert_not_called()

    def test_cached_none_falls_through_to_live_lookup(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        _write_cache(cache_path, {"someherb": None})

        with (
            patch(f"{_MODULE}.find_rxcui_exact", return_value="5555") as mock_exact,
            patch(f"{_MODULE}.find_rxcui_approx_candidates") as mock_approx,
        ):
            result = resolve_query_drug("someherb", cache_path=cache_path)

        assert result == {"rxcui": "5555", "match_type": "exact", "candidates": []}
        mock_exact.assert_called_once_with("someherb")
        mock_approx.assert_not_called()

    def test_cache_miss_falls_through_to_live_lookup(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        _write_cache(cache_path, {"ibuprofen": "5640"})

        with patch(f"{_MODULE}.find_rxcui_exact", return_value="6809") as mock_exact:
            result = resolve_query_drug("metformin", cache_path=cache_path)

        assert result == {"rxcui": "6809", "match_type": "exact", "candidates": []}
        mock_exact.assert_called_once_with("metformin")

    def test_cache_match_is_case_insensitive(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        _write_cache(cache_path, {"Metformin": "6809"})

        with (
            patch(f"{_MODULE}.find_rxcui_exact") as mock_exact,
            patch(f"{_MODULE}.find_rxcui_approx_candidates") as mock_approx,
        ):
            result = resolve_query_drug("metformin", cache_path=cache_path)

        assert result == {"rxcui": "6809", "match_type": "cached", "candidates": []}
        mock_exact.assert_not_called()
        mock_approx.assert_not_called()

    def test_missing_cache_file_falls_through_without_error(self, tmp_path):
        with patch(f"{_MODULE}.find_rxcui_exact", return_value="6809") as mock_exact:
            result = resolve_query_drug("metformin", cache_path=tmp_path / "does_not_exist.json")

        assert result == {"rxcui": "6809", "match_type": "exact", "candidates": []}
        mock_exact.assert_called_once_with("metformin")
