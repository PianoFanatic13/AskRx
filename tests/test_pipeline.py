import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from backend.pipeline.pipeline import (
    load_rxnorm_cache,
    run_pipeline,
    save_rxnorm_cache,
)

_MODULE = "backend.pipeline.pipeline"


# --- Cache helpers ---

def test_load_rxnorm_cache_absent_file(tmp_path):
    assert load_rxnorm_cache(tmp_path / "missing.json") == {}


def test_load_rxnorm_cache_reads_existing(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"metformin": "6809", "unknown": None}), encoding='utf-8')

    result = load_rxnorm_cache(cache_path)

    assert result == {"metformin": "6809", "unknown": None}


def test_save_rxnorm_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    original = {"metformin": "6809", "zyxwv": None}

    save_rxnorm_cache(cache_path, original)

    assert load_rxnorm_cache(cache_path) == original


def test_save_rxnorm_cache_atomic(tmp_path):
    cache_path = tmp_path / "cache.json"

    save_rxnorm_cache(cache_path, {"x": "1"})

    # .tmp sibling must be gone after a successful save
    assert not cache_path.with_suffix('.tmp').exists()
    assert cache_path.exists()


# --- Helpers for pipeline behaviour tests ---

def _fake_header(
    setid: str,
    ingredients: list[str],
    *,
    drug_name: str = "DRUG",
    rxcui: str | None = None,
    mkt: str = "C73594",
    eff: str = "20200101",
    form: str = "TABLET",
    route: str = "ORAL",
) -> dict:
    return {
        "setid": setid,
        "drug_name": drug_name,
        "active_ingredients": ingredients,
        "marketing_category": mkt,
        "effective_time": eff,
        "dosage_form": form,
        "route": route,
        "doc_type": None,
    }


def _fake_section() -> dict:
    return {
        "loinc_code": "34067-9",
        "loinc_source": "direct",
        "section_title": "INDICATIONS AND USAGE",
        "section_title_path": ["INDICATIONS AND USAGE"],
        "section_type": "standard",
        "text": "Used to treat hypertension.",
        "depth": 0,
    }


def _fake_chunk(setid: str) -> dict:
    return {
        "setid": setid,
        "drug_name": "DRUG",
        "rxcui": "6809",
        "loinc_code": "34067-9",
        "loinc_source": "direct",
        "section_title_path": ["INDICATIONS AND USAGE"],
        "section_type": "standard",
        "chunk_text": "Used to treat hypertension.",
        "token_count": 5,
        "merged": False,
        "merged_title_paths": None,
    }


def _write_xml(labels_dir: Path, setid: str) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / f"{setid}.xml"
    path.write_text("<root/>", encoding='utf-8')
    return path


# --- Pipeline behaviour ---

def test_single_ingredient_label_gets_resolved_rxcui(tmp_path):
    setid = "aaaa"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["metformin"])) as mock_header,
        patch(f"{_MODULE}.resolve_rxcui", return_value="6809"),
        patch(f"{_MODULE}.select_canonical", return_value=[]) as mock_canonical,
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )
        records_passed = mock_canonical.call_args[0][0]

    assert len(records_passed) == 1
    assert records_passed[0]["rxcui"] == "6809"


def test_multi_ingredient_label_gets_null_rxcui(tmp_path):
    setid = "bbbb"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    captured = []

    def capture_canonical(records):
        captured.extend(records)
        return []

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["amoxicillin", "clavulanate potassium"])),
        patch(f"{_MODULE}.resolve_rxcui", side_effect=["111", "222"]),
        patch(f"{_MODULE}.select_canonical", side_effect=capture_canonical),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    assert len(captured) == 1
    assert captured[0]["rxcui"] is None


def test_all_ingredients_resolved_for_combo_drug(tmp_path):
    setid = "cccc"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    mock_resolve = MagicMock(return_value="999")

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["amoxicillin", "clavulanate potassium"])),
        patch(f"{_MODULE}.resolve_rxcui", mock_resolve),
        patch(f"{_MODULE}.select_canonical", return_value=[]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    assert mock_resolve.call_count == 2
    called_names = {c.args[0] for c in mock_resolve.call_args_list}
    assert called_names == {"amoxicillin", "clavulanate potassium"}


def test_cached_ingredient_not_re_resolved(tmp_path):
    setid = "dddd"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    save_rxnorm_cache(cache_path, {"metformin": "6809"})
    output_path = tmp_path / "out.jsonl"

    mock_resolve = MagicMock()

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["metformin"])),
        patch(f"{_MODULE}.resolve_rxcui", mock_resolve),
        patch(f"{_MODULE}.select_canonical", return_value=[]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    mock_resolve.assert_not_called()


def test_cache_written_after_pass1(tmp_path):
    setid = "eeee"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["lisinopril"])),
        patch(f"{_MODULE}.resolve_rxcui", return_value="29046"),
        patch(f"{_MODULE}.select_canonical", return_value=[]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    cache = load_rxnorm_cache(cache_path)
    assert cache.get("lisinopril") == "29046"


def test_pass2_writes_chunks_to_jsonl(tmp_path):
    setid = "ffff"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["metformin"])),
        patch(f"{_MODULE}.resolve_rxcui", return_value="6809"),
        patch(f"{_MODULE}.select_canonical", return_value=[setid]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label", return_value={"header": _fake_header(setid, ["metformin"]), "sections": [_fake_section()]}),
        patch(f"{_MODULE}.chunk_section", return_value=[_fake_chunk(setid), _fake_chunk(setid)]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    lines = output_path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2


def test_chunks_are_valid_json_objects(tmp_path):
    setid = "gggg"
    _write_xml(tmp_path / "labels", setid)
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header(setid, ["warfarin"])),
        patch(f"{_MODULE}.resolve_rxcui", return_value="11289"),
        patch(f"{_MODULE}.select_canonical", return_value=[setid]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label", return_value={"header": _fake_header(setid, ["warfarin"]), "sections": [_fake_section()]}),
        patch(f"{_MODULE}.chunk_section", return_value=[_fake_chunk(setid)]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    for line in output_path.read_text(encoding='utf-8').strip().splitlines():
        obj = json.loads(line)
        assert "setid" in obj
        assert "chunk_text" in obj


def test_each_zip_part_extracted_once(tmp_path):
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    zip_paths = [tmp_path / f"part{i}.zip" for i in range(1, 4)]
    for p in zip_paths:
        p.write_bytes(b"")

    with (
        patch(f"{_MODULE}.extract_xmls", return_value=[]) as mock_extract,
        patch(f"{_MODULE}.select_canonical", return_value=[]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
    ):
        run_pipeline(
            zip_paths=zip_paths,
            labels_dir=labels_dir,
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    # One call per part, no effective limit applied to extraction
    assert mock_extract.call_count == 3
    for c in mock_extract.call_args_list:
        assert c.kwargs.get('limit') is None


def test_missing_canonical_xml_skipped(tmp_path):
    # "yyyy" exists on disk; select_canonical returns "xxxx" (no file) — pass2 skips it
    _write_xml(tmp_path / "labels", "yyyy")
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    with (
        patch(f"{_MODULE}.extract_header", return_value=_fake_header("yyyy", ["metformin"])),
        patch(f"{_MODULE}.resolve_rxcui", return_value="6809"),
        patch(f"{_MODULE}.select_canonical", return_value=["xxxx"]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label") as mock_parse,
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    mock_parse.assert_not_called()
    assert output_path.read_text(encoding='utf-8').strip() == ""


def test_header_parse_failure_does_not_abort(tmp_path):
    _write_xml(tmp_path / "labels", "good")
    _write_xml(tmp_path / "labels", "bad")
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.jsonl"

    good_header = _fake_header("good", ["metformin"])
    call_count = {"n": 0}

    def header_side_effect(path):
        call_count["n"] += 1
        if "bad" in path:
            raise ValueError("corrupt XML")
        return good_header

    with (
        patch(f"{_MODULE}.extract_header", side_effect=header_side_effect),
        patch(f"{_MODULE}.resolve_rxcui", return_value="6809"),
        patch(f"{_MODULE}.select_canonical", return_value=[]),
        patch(f"{_MODULE}.select_canonical_no_rxcui", return_value=[]),
        patch(f"{_MODULE}.parse_label"),
        patch(f"{_MODULE}.chunk_section", return_value=[]),
    ):
        run_pipeline(
            labels_dir=tmp_path / "labels",
            output_path=output_path,
            rxnorm_cache_path=cache_path,
        )

    assert call_count["n"] == 2
