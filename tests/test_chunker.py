from unittest.mock import patch
from backend.pipeline.chunker import chunk_section_text, chunk_section, TOKEN_CEIL, TOKEN_FLOOR

_PATCH = "backend.pipeline.chunker.count_tokens"


def _wc(text: str) -> int:
    return len(text.split())


def _words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _section(text: str, path: list[str] | None = None) -> dict:
    return {
        "text": text,
        "section_title_path": path or ["Warnings"],
        "loinc_code": "34071-1",
        "loinc_source": "direct",
        "section_type": "standard",
    }


# --- ceiling ---

@patch(_PATCH, side_effect=_wc)
def test_short_paragraph_passes_through(mock_ct):
    text = _words(200)
    assert chunk_section_text(text) == [text]


@patch(_PATCH, side_effect=_wc)
def test_ceiling_splits_long_paragraph(mock_ct):
    # Sentences must start uppercase so the sentence boundary regex fires
    sentences = ["Sentence " + _words(99, f"s{i}") + "." for i in range(5)]
    text = " ".join(sentences)
    chunks = chunk_section_text(text)
    assert len(chunks) > 1
    assert all(_wc(c) <= TOKEN_CEIL for c in chunks)


@patch(_PATCH, side_effect=_wc)
def test_no_chunk_exceeds_ceiling(mock_ct):
    paras = [" ".join(_words(100, f"p{i}s{j}") + "." for j in range(3)) for i in range(3)]
    chunks = chunk_section_text("\n\n".join(paras))
    assert all(_wc(c) <= TOKEN_CEIL for c in chunks)


@patch(_PATCH, side_effect=_wc)
def test_ceiling_preserves_paragraph_that_fits(mock_ct):
    short = _words(50)
    long_sentences = [_words(100, f"s{i}") + "." for i in range(5)]
    text = short + "\n\n" + " ".join(long_sentences)
    chunks = chunk_section_text(text)
    assert chunks[0] == short


@patch(_PATCH, side_effect=_wc)
def test_ceiling_word_split_fallback_no_sentence_boundaries(mock_ct):
    # All lowercase words with no .!? — sentence splitter finds nothing,
    # word-split fallback must enforce the ceiling
    text = _words(TOKEN_CEIL + 100)  # 550 lowercase words, no sentence breaks
    chunks = chunk_section_text(text)
    assert len(chunks) > 1
    assert all(_wc(c) <= TOKEN_CEIL for c in chunks)


# --- floor ---

@patch(_PATCH, side_effect=_wc)
def test_floor_merges_into_prev_sibling(mock_ct):
    long = _words(200)
    short = _words(TOKEN_FLOOR - 10)
    chunks = chunk_section_text(f"{long}\n\n{short}")
    assert len(chunks) == 1
    assert _wc(chunks[0]) == 200 + (TOKEN_FLOOR - 10)


@patch(_PATCH, side_effect=_wc)
def test_floor_merges_into_next_when_prev_full(mock_ct):
    big = _words(420)
    short = _words(TOKEN_FLOOR - 10)
    small = _words(100)
    chunks = chunk_section_text(f"{big}\n\n{short}\n\n{small}")
    assert len(chunks) == 2
    assert _wc(chunks[0]) == 420
    assert _wc(chunks[1]) == (TOKEN_FLOOR - 10) + 100


@patch(_PATCH, side_effect=_wc)
def test_floor_leaves_fragment_when_both_siblings_full(mock_ct):
    big1 = _words(420)
    short = _words(TOKEN_FLOOR - 10)
    big2 = _words(420)
    chunks = chunk_section_text(f"{big1}\n\n{short}\n\n{big2}")
    assert len(chunks) == 3


@patch(_PATCH, side_effect=_wc)
def test_floor_cascades_multiple_short_fragments(mock_ct):
    short1 = _words(TOKEN_FLOOR - 10)
    short2 = _words(TOKEN_FLOOR - 10)
    short3 = _words(TOKEN_FLOOR - 10)
    chunks = chunk_section_text(f"{short1}\n\n{short2}\n\n{short3}")
    assert len(chunks) == 1
    assert _wc(chunks[0]) == 3 * (TOKEN_FLOOR - 10)


# --- metadata (6e / 6f) ---

@patch(_PATCH, side_effect=_wc)
def test_unmerged_chunk_has_correct_metadata(mock_ct):
    section = _section(_words(200), path=["Adverse Reactions"])
    chunks = chunk_section(
        section, setid="abc", drug_name="metformin", rxcui="6809",
        dosage_form="TABLET", route="ORAL",
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c["setid"] == "abc"
    assert c["drug_name"] == "metformin"
    assert c["rxcui"] == "6809"
    assert c["dosage_form"] == "TABLET"
    assert c["route"] == "ORAL"
    assert c["loinc_code"] == "34071-1"
    assert c["loinc_source"] == "direct"
    assert c["section_title_path"] == ["Adverse Reactions"]
    assert c["section_type"] == "standard"
    assert c["merged"] is False
    assert c["merged_title_paths"] is None
    assert c["token_count"] == 200


@patch(_PATCH, side_effect=_wc)
def test_merged_chunk_sets_flag_and_paths(mock_ct):
    long = _words(200)
    short = _words(TOKEN_FLOOR - 10)
    section = _section(f"{long}\n\n{short}", path=["Dosage"])
    chunks = chunk_section(section, setid="s1", drug_name="ibuprofen", rxcui="5640")
    assert len(chunks) == 1
    c = chunks[0]
    assert c["merged"] is True
    assert c["merged_title_paths"] == [["Dosage"], ["Dosage"]]


@patch(_PATCH, side_effect=_wc)
def test_empty_section_returns_no_chunks(mock_ct):
    chunks = chunk_section(_section(""), setid="s1", drug_name="drug", rxcui=None)
    assert chunks == []


@patch(_PATCH, side_effect=_wc)
def test_token_count_matches_chunk_text(mock_ct):
    section = _section(_words(200))
    chunks = chunk_section(section, setid="s1", drug_name="drug", rxcui="1")
    for c in chunks:
        assert c["token_count"] == _wc(c["chunk_text"])


@patch(_PATCH, side_effect=_wc)
def test_rxcui_none_propagates(mock_ct):
    section = _section(_words(100))
    chunks = chunk_section(section, setid="s1", drug_name="supplement", rxcui=None)
    assert chunks[0]["rxcui"] is None


@patch(_PATCH, side_effect=_wc)
def test_dosage_form_and_route_default_to_none(mock_ct):
    section = _section(_words(100))
    chunks = chunk_section(section, setid="s1", drug_name="drug", rxcui="1")
    assert chunks[0]["dosage_form"] is None
    assert chunks[0]["route"] is None
