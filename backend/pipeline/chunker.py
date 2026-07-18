import re
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_MODEL = "BAAI/bge-large-en-v1.5"
_tokenizer: PreTrainedTokenizerBase | None = None

TOKEN_CEIL = 450
TOKEN_FLOOR = 50


def _get_tokenizer() -> PreTrainedTokenizerBase:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return the number of content tokens in text (no special tokens)."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]


_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]


def _word_split(text: str) -> list[str]:
    """Recursively halve text at a word boundary until all pieces are within TOKEN_CEIL.

    Used as a last resort when no sentence boundary is available to split on.
    """
    if count_tokens(text) <= TOKEN_CEIL:
        return [text]
    words = text.split()
    if len(words) <= 1:
        return [text]  # single token-like word; can't split further
    mid = len(words) // 2
    return _word_split(" ".join(words[:mid])) + _word_split(" ".join(words[mid:]))


def _enforce_ceiling(fragments: list[str]) -> list[str]:
    """Split any fragment exceeding TOKEN_CEIL on sentence boundaries.

    Falls back to word-midpoint splitting when no sentence boundary is found.
    """
    result = []
    for frag in fragments:
        if count_tokens(frag) <= TOKEN_CEIL:
            result.append(frag)
            continue
        sentences = _split_sentences(frag)
        bucket: list[str] = []
        for sent in sentences:
            candidate = " ".join(bucket + [sent])
            if bucket and count_tokens(candidate) > TOKEN_CEIL:
                result.extend(_word_split(" ".join(bucket)))
                bucket = [sent]
            else:
                bucket.append(sent)
        if bucket:
            remainder = " ".join(bucket)
            result.extend(_word_split(remainder))
    return result


def _enforce_floor(fragments: list[str]) -> list[str]:
    """Merge fragments below TOKEN_FLOOR into neighbors (prev sibling, or next sibling).

    Fragments that cannot be merged without exceeding TOKEN_CEIL are left as-is;
    cross-section parent merging is handled at the pipeline level.
    """
    if not fragments:
        return []
    result = list(fragments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            if count_tokens(result[i]) < TOKEN_FLOOR:
                merged = False
                if i > 0:
                    combined = result[i - 1] + " " + result[i]
                    if count_tokens(combined) <= TOKEN_CEIL:
                        result[i - 1] = combined
                        result.pop(i)
                        changed = True
                        merged = True
                if not merged and i < len(result) - 1:
                    combined = result[i] + " " + result[i + 1]
                    if count_tokens(combined) <= TOKEN_CEIL:
                        result[i + 1] = combined
                        result.pop(i)
                        changed = True
                        merged = True
                if not merged:
                    i += 1
            else:
                i += 1
    return result


def chunk_section_text(text: str) -> list[str]:
    """Split a section's text into token-bounded chunk strings."""
    paragraphs = _split_paragraphs(text)
    bounded = _enforce_ceiling(paragraphs)
    return _enforce_floor(bounded)


# --- 6e / 6f: metadata-aware chunking ---

@dataclass
class _Frag:
    text: str
    title_paths: list[list[str]]  # one entry per original paragraph that contributed


def _enforce_floor_tracked(frags: list[_Frag]) -> list[_Frag]:
    """Floor enforcement that accumulates source title paths across merges."""
    if not frags:
        return []
    result = list(frags)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            if count_tokens(result[i].text) < TOKEN_FLOOR:
                merged = False
                if i > 0:
                    combined = result[i - 1].text + " " + result[i].text
                    if count_tokens(combined) <= TOKEN_CEIL:
                        result[i - 1] = _Frag(combined, result[i - 1].title_paths + result[i].title_paths)
                        result.pop(i)
                        changed = True
                        merged = True
                if not merged and i < len(result) - 1:
                    combined = result[i].text + " " + result[i + 1].text
                    if count_tokens(combined) <= TOKEN_CEIL:
                        result[i + 1] = _Frag(combined, result[i].title_paths + result[i + 1].title_paths)
                        result.pop(i)
                        changed = True
                        merged = True
                if not merged:
                    i += 1
            else:
                i += 1
    return result


def chunk_section(
    section: dict,
    setid: str,
    drug_name: str,
    rxcui: str | None,
    dosage_form: str | None = None,
    route: str | None = None,
) -> list[dict]:
    """Chunk a parsed section dict and attach full provenance metadata."""
    text = section.get("text", "")
    title_path = section["section_title_path"]

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    bounded = _enforce_ceiling(paragraphs)
    frags = [_Frag(t, [title_path]) for t in bounded]
    final_frags = _enforce_floor_tracked(frags)

    chunks = []
    for frag in final_frags:
        is_merged = len(frag.title_paths) > 1
        chunks.append({
            "setid": setid,
            "drug_name": drug_name,
            "rxcui": rxcui,
            "dosage_form": dosage_form,
            "route": route,
            "loinc_code": section["loinc_code"],
            "loinc_source": section["loinc_source"],
            "section_title_path": title_path,
            "section_type": section["section_type"],
            "chunk_text": frag.text,
            "token_count": count_tokens(frag.text),
            "merged": is_merged,
            "merged_title_paths": frag.title_paths if is_merged else None,
        })
    return chunks
