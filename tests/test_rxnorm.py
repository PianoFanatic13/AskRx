"""
Integration tests for the RxNorm client — hit the real API.
Run with: pytest -m integration
Skipped in normal test runs to avoid network dependency.
"""
import re
import pytest
from backend.pipeline.rxnorm import resolve_rxcui, set_failure_log

RXCUI_RE = re.compile(r"^\d+$")


@pytest.mark.integration
class TestResolveRxcui:
    def test_common_prescription_drug(self):
        # metformin is unambiguous; any valid RXCUI is a digit string
        result = resolve_rxcui("metformin")
        assert result is not None
        assert RXCUI_RE.match(result)

    def test_otc_generic(self):
        result = resolve_rxcui("ibuprofen")
        assert result is not None
        assert RXCUI_RE.match(result)

    def test_otc_brand_name(self):
        # Tylenol is a brand name — should resolve via exact or approx
        result = resolve_rxcui("Tylenol")
        assert result is not None
        assert RXCUI_RE.match(result)

    def test_supplement(self):
        # Cholecalciferol (vitamin D3) — in RxNorm but coverage can be inconsistent
        # We just assert it doesn't crash and returns a string or None
        result = resolve_rxcui("cholecalciferol")
        assert result is None or RXCUI_RE.match(result)

    def test_combination_drug(self):
        # Combination drug ingredient string as it appears in SPL labels
        result = resolve_rxcui("amoxicillin")
        assert result is not None
        result2 = resolve_rxcui("clavulanate potassium")
        assert result2 is not None

    def test_unknown_string_returns_none(self):
        result = resolve_rxcui("zyxwvutsrqponmlkjihgfedcba")
        assert result is None

    def test_failure_written_to_log(self, tmp_path):
        log_file = tmp_path / "rxnorm_failures.tsv"
        set_failure_log(log_file)
        resolve_rxcui("zyxwvutsrqponmlkjihgfedcba")
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        ts, name = lines[0].split("\t")
        assert name == "zyxwvutsrqponmlkjihgfedcba"
        assert ts.endswith("Z")
        # Reset so other tests aren't affected
        set_failure_log(None)
