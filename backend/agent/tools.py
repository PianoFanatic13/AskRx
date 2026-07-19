from typing import Optional

from langchain_core.tools import tool

from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rxnorm_query import resolve_query_drug
from backend.retrieval.text_search import get_section

_DRUG_INTERACTIONS_LOINC = "34073-7"


@tool
def resolve_drug_name(name: str) -> dict:
    """Resolve a drug name (brand or generic, possibly misspelled) to its RxNorm identifier (RXCUI).

    Call this first for any drug named in the query, before retrieve_drug_info
    or retrieve_interactions, so those calls can be filtered to the right drug.
    match_type is "exact"/"approx"/"cached" when resolved, "ambiguous" when
    multiple drugs could match (ask the user which one before proceeding), or
    "unresolved" when it's not a recognized drug (proceed without a filter).
    """
    return resolve_query_drug(name)


@tool
def retrieve_drug_info(query_text: str, rxcui: Optional[str] = None) -> list[dict]:
    """Retrieve relevant sections from FDA drug label(s) to answer a question about a drug.

    Covers indications, warnings, dosing, adverse reactions, etc. Pass the
    rxcui from resolve_drug_name when available to filter to that specific
    drug; omit it for an unfiltered search. Each result includes setid,
    loinc_code, and section_title_path — cite these when answering.
    """
    return hybrid_search(query_text, rxcui=rxcui)


@tool
def retrieve_interactions(rxcui: str) -> list[dict]:
    """Fetch the exact Drug Interactions section from a specific drug's label.

    Requires a resolved rxcui (call resolve_drug_name first). An empty result
    means the label doesn't mention interactions — report this as "the label
    doesn't discuss this," never as "these drugs don't interact."
    """
    return get_section(rxcui, _DRUG_INTERACTIONS_LOINC)
