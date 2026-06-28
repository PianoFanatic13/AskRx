from backend.pipeline.dedup import select_canonical, select_canonical_no_rxcui

NDA  = "C73594"
BLA  = "C73585"
ANDA = "C73584"
OTC  = "C200263"


def _rec(setid, rxcui, form, route, mkt, eff, ingredients=None):
    return {
        "setid": setid,
        "rxcui": rxcui,
        "dosage_form": form,
        "route": route,
        "marketing_category": mkt,
        "effective_time": eff,
        "active_ingredients": ingredients or [],
    }


# --- 5a: grouping ---

def test_ir_and_er_are_separate_groups():
    # Same RXCUI, different dosage form → two separate canonical labels
    records = [
        _rec("IR", "6809", "TABLET", "ORAL", NDA, "20200101"),
        _rec("ER", "6809", "TABLET, EXTENDED RELEASE", "ORAL", NDA, "20200101"),
    ]
    result = select_canonical(records)
    assert set(result) == {"IR", "ER"}


def test_different_routes_are_separate_groups():
    records = [
        _rec("ORAL", "5640", "TABLET", "ORAL", NDA, "20200101"),
        _rec("IV",   "5640", "INJECTION", "INTRAVENOUS", NDA, "20200101"),
    ]
    result = select_canonical(records)
    assert set(result) == {"ORAL", "IV"}


def test_records_without_rxcui_excluded_from_select_canonical():
    records = [
        _rec("A", None,   "TABLET", "ORAL", NDA, "20230101"),
        _rec("B", "6809", "TABLET", "ORAL", NDA, "20230101"),
    ]
    result = select_canonical(records)
    assert result == ["B"]


# --- 5b: NDA/BLA keep, ANDA drop ---

def test_anda_dropped_when_nda_present():
    records = [
        _rec("NDA1",  "6809", "TABLET", "ORAL", NDA,  "20200101"),
        _rec("ANDA1", "6809", "TABLET", "ORAL", ANDA, "20230101"),
        _rec("ANDA2", "6809", "TABLET", "ORAL", ANDA, "20220101"),
    ]
    result = select_canonical(records)
    assert result == ["NDA1"]


def test_bla_kept_and_anda_dropped():
    records = [
        _rec("BLA1",  "100", "INJECTION", "SUBCUTANEOUS", BLA,  "20210101"),
        _rec("ANDA1", "100", "INJECTION", "SUBCUTANEOUS", ANDA, "20230101"),
    ]
    result = select_canonical(records)
    assert result == ["BLA1"]


def test_anda_only_group_kept_as_fallback():
    # No NDA/BLA in DailyMed for this group — keep most recent ANDA
    records = [
        _rec("ANDA_OLD", "9999", "TABLET", "ORAL", ANDA, "20190101"),
        _rec("ANDA_NEW", "9999", "TABLET", "ORAL", ANDA, "20230101"),
    ]
    result = select_canonical(records)
    assert result == ["ANDA_NEW"]


# --- 5c: effectiveTime tiebreaker ---

def test_latest_nda_wins():
    records = [
        _rec("OLD", "6809", "TABLET", "ORAL", NDA, "20180101"),
        _rec("MID", "6809", "TABLET", "ORAL", NDA, "20210601"),
        _rec("NEW", "6809", "TABLET", "ORAL", NDA, "20230101"),
    ]
    result = select_canonical(records)
    assert result == ["NEW"]


def test_missing_effective_time_loses_tiebreaker():
    records = [
        _rec("NO_DATE", "6809", "TABLET", "ORAL", NDA, None),
        _rec("DATED",   "6809", "TABLET", "ORAL", NDA, "20200101"),
    ]
    result = select_canonical(records)
    assert result == ["DATED"]


# --- 5d: OTC monograph recency ---

def test_otc_monograph_recency_tiebreaker():
    records = [
        _rec("OTC_OLD", "5640", "TABLET", "ORAL", OTC, "20200101"),
        _rec("OTC_NEW", "5640", "TABLET", "ORAL", OTC, "20231201"),
    ]
    result = select_canonical(records)
    assert result == ["OTC_NEW"]


def test_otc_monograph_not_mixed_with_anda():
    # OTC monograph and ANDA in the same group — OTC wins
    records = [
        _rec("OTC",  "5640", "TABLET", "ORAL", OTC,  "20200101"),
        _rec("ANDA", "5640", "TABLET", "ORAL", ANDA, "20231201"),
    ]
    result = select_canonical(records)
    assert result == ["OTC"]


# --- 5e: unknown code ---

def test_unknown_code_skipped(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="backend.pipeline.dedup"):
        records = [_rec("REPACK", "6809", "TABLET", "ORAL", "C73606", "20230101")]
        result = select_canonical(records)
    assert result == []
    assert "C73606" in caplog.text


def test_unknown_code_does_not_block_known_codes_in_same_group(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="backend.pipeline.dedup"):
        records = [
            _rec("REPACK", "6809", "TABLET", "ORAL", "C73606", "20231201"),
            _rec("NDA1",   "6809", "TABLET", "ORAL", NDA,      "20200101"),
        ]
        result = select_canonical(records)
    assert result == ["NDA1"]
    assert "C73606" in caplog.text


# --- 5f: RxNorm failure fallback ---

def test_fallback_groups_by_ingredient_string():
    # Same ingredient + form + route → one group
    records = [
        _rec("A", None, "TABLET", "ORAL", NDA, "20200101", ["metformin hydrochloride"]),
        _rec("B", None, "TABLET", "ORAL", ANDA, "20230101", ["METFORMIN HYDROCHLORIDE"]),
    ]
    result = select_canonical_no_rxcui(records)
    assert result == ["A"]


def test_fallback_different_forms_separate_groups():
    records = [
        _rec("IR", None, "TABLET", "ORAL", NDA, "20200101", ["metformin hydrochloride"]),
        _rec("ER", None, "TABLET, EXTENDED RELEASE", "ORAL", NDA, "20200101", ["metformin hydrochloride"]),
    ]
    result = select_canonical_no_rxcui(records)
    assert set(result) == {"IR", "ER"}


def test_fallback_ingredient_order_does_not_matter():
    # Combination drug: ingredient order in the XML may vary
    records = [
        _rec("A", None, "TABLET", "ORAL", NDA, "20200101", ["amoxicillin", "clavulanate potassium"]),
        _rec("B", None, "TABLET", "ORAL", ANDA, "20220101", ["clavulanate potassium", "amoxicillin"]),
    ]
    result = select_canonical_no_rxcui(records)
    assert result == ["A"]


def test_fallback_excludes_records_with_rxcui():
    # select_canonical_no_rxcui should only touch rxcui=None records
    records = [
        _rec("HAS_RXCUI", "6809", "TABLET", "ORAL", NDA, "20230101", ["metformin"]),
        _rec("NO_RXCUI",  None,   "TABLET", "ORAL", NDA, "20200101", ["metformin"]),
    ]
    result = select_canonical_no_rxcui(records)
    assert result == ["NO_RXCUI"]
