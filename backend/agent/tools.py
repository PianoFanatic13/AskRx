from langchain_core.tools import tool

from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rxnorm_query import resolve_query_drug
from backend.retrieval.text_search import get_section

_DRUG_INTERACTIONS_LOINC = "34073-7"

_KEPT_CHUNK_FIELDS = {"chunk_text", "setid", "loinc_code", "section_title_path", "drug_name", "section_type"}


def _trim_chunks(chunks: list[dict]) -> list[dict]:
    """Drop fields the LLM doesn't need to answer or cite correctly (id, rxcui, token_count, ranking scores)."""
    return [{k: v for k, v in chunk.items() if k in _KEPT_CHUNK_FIELDS} for chunk in chunks]


@tool
def resolve_drug_name(name: str) -> dict:
    """Resolve a drug name (brand or generic, possibly misspelled) to its RxNorm identifier (RXCUI).

    Only needed when you want to check a name before deciding what to do with
    it (e.g. sorting out several drugs in one query) — retrieve_drug_info and
    retrieve_interactions already resolve names internally. match_type is
    "exact"/"approx"/"cached" when resolved, "ambiguous" when multiple drugs
    could match (ask the user which one), or "unresolved" when it's not a
    recognized drug.
    """
    return resolve_query_drug(name)


@tool
def retrieve_drug_info(query_text: str, drug_name: str) -> dict:
    """Retrieve relevant sections from FDA drug label(s) to answer a question about a drug.

    Covers indications, warnings, dosing, adverse reactions, etc. Resolves
    drug_name internally, so you don't need to call resolve_drug_name first.

    Returns {"results": [...], "match_type": str, "candidates": [...]}. If
    match_type is "ambiguous", results is empty and candidates lists the
    possible drugs — ask the user which one instead of guessing. Otherwise
    each result includes setid, loinc_code, and section_title_path — cite
    these when answering.
    """
    resolution = resolve_query_drug(drug_name)
    if resolution["match_type"] == "ambiguous":
        return {"results": [], "match_type": "ambiguous", "candidates": resolution["candidates"]}

    rxcui = resolution["rxcui"]
    results = hybrid_search(query_text, rxcui=rxcui)
    match_type = resolution["match_type"]
    if rxcui is not None and not results:
        # Resolved to a real drug, but it isn't represented in the indexed corpus.
        results = hybrid_search(query_text, rxcui=None)
        match_type = "not_indexed"

    return {"results": _trim_chunks(results), "match_type": match_type, "candidates": []}


@tool
def retrieve_interactions(drug_name: str) -> dict:
    """Fetch the exact Drug Interactions section from a specific drug's label.

    Resolves drug_name internally, so you don't need to call resolve_drug_name
    first. Returns {"results": [...], "match_type": str, "candidates": [...]}.
    If match_type is "ambiguous", ask the user which drug they meant instead
    of guessing. If "unresolved", the drug couldn't be identified at all, so
    no interactions lookup was possible. An empty results list otherwise means
    the label doesn't mention interactions — report this as "the label
    doesn't discuss this," never as "these drugs don't interact."
    """
    resolution = resolve_query_drug(drug_name)
    if resolution["match_type"] == "ambiguous":
        return {"results": [], "match_type": "ambiguous", "candidates": resolution["candidates"]}
    if resolution["rxcui"] is None:
        return {"results": [], "match_type": "unresolved", "candidates": []}

    results = get_section(resolution["rxcui"], _DRUG_INTERACTIONS_LOINC)
    return {"results": _trim_chunks(results), "match_type": resolution["match_type"], "candidates": []}
