import re
import logging
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# NCI Thesaurus codes as they appear in SPL <approval> elements
NDA_CODES = {
    "C73594",  # NDA
    "C73585",  # BLA
}
ANDA_CODES = {"C73584"}
OTC_MONOGRAPH_CODES = {"C200263"}  # OTC Monograph Drug

_KNOWN_CODES = NDA_CODES | ANDA_CODES | OTC_MONOGRAPH_CODES


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r'[^A-Z0-9\s]', ' ', s.upper())
    return re.sub(r'\s+', ' ', s).strip()


def _group_key(record: dict) -> tuple:
    return (
        record["rxcui"],
        _norm(record.get("dosage_form")),
        _norm(record.get("route")),
    )


def _ingredient_key(record: dict) -> tuple:
    ings = record.get("active_ingredients") or []
    return (
        tuple(sorted(_norm(i) for i in ings)),
        _norm(record.get("dosage_form")),
        _norm(record.get("route")),
    )


def _latest(records: list[dict]) -> dict:
    # effective_time is "YYYYMMDD"; empty string sorts earlier than any real date
    return max(records, key=lambda r: r.get("effective_time") or "")


def _pick_from_group(records: list[dict]) -> Optional[str]:
    """Return the canonical SETID for one dedup group, or None to skip the group."""
    unknown = [r for r in records if r.get("marketing_category") not in _KNOWN_CODES]
    for r in unknown:
        log.warning(
            "Unknown marketing_category %r on setid %s — skipping record",
            r.get("marketing_category"),
            r.get("setid"),
        )

    known = [r for r in records if r.get("marketing_category") in _KNOWN_CODES]
    if not known:
        return None

    # NDA/BLA is the authoritative label; drop ANDAs when one is present
    nda_bla = [r for r in known if r["marketing_category"] in NDA_CODES]
    if nda_bla:
        return _latest(nda_bla)["setid"]

    # OTC monograph: no NDA/ANDA hierarchy, keep most recently updated
    otc = [r for r in known if r["marketing_category"] in OTC_MONOGRAPH_CODES]
    if otc:
        return _latest(otc)["setid"]

    # ANDA-only group: no reference label in DailyMed, keep most recent
    anda = [r for r in known if r["marketing_category"] in ANDA_CODES]
    if anda:
        return _latest(anda)["setid"]

    return None


def select_canonical(records: list[dict]) -> list[str]:
    """Return canonical SETIDs from records that have a resolved RXCUI.

    Groups by (rxcui, dosage_form, route). Records with rxcui=None are
    excluded — handle those with select_canonical_no_rxcui.

    Each record must have: setid, rxcui, dosage_form, route,
    marketing_category, effective_time.
    """
    with_rxcui = [r for r in records if r.get("rxcui")]

    groups: dict[tuple, list] = defaultdict(list)
    for r in with_rxcui:
        groups[_group_key(r)].append(r)

    canonical = []
    for group in groups.values():
        setid = _pick_from_group(group)
        if setid:
            canonical.append(setid)
    return canonical


def select_canonical_no_rxcui(records: list[dict]) -> list[str]:
    """Return canonical SETIDs for records where RxNorm lookup failed.

    Groups by (normalized active ingredients, dosage_form, route) and
    applies the same NDA/BLA > OTC > ANDA priority logic.
    """
    without_rxcui = [r for r in records if not r.get("rxcui")]

    groups: dict[tuple, list] = defaultdict(list)
    for r in without_rxcui:
        groups[_ingredient_key(r)].append(r)

    canonical = []
    for group in groups.values():
        setid = _pick_from_group(group)
        if setid:
            canonical.append(setid)
    return canonical
