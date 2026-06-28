import re
from lxml import etree
from typing import Optional

NS = "urn:hl7-org:v3"
NCI_CS = "2.16.840.1.113883.3.26.1.1"
LOINC_CS = "2.16.840.1.113883.6.1"
PRODUCT_DATA_LOINC = "48780-1"
UNCLASSIFIED_LOINC = "42229-5"
ACTIVE_CLASS_CODES = {"ACTIB", "ACTIM"}

# Normalized section title -> LOINC code, used when the XML element has no direct code
TITLE_LOINC_MAP = {
    "BOXED WARNING":                                         "34066-1",
    "INDICATIONS AND USAGE":                                 "34067-9",
    "DOSAGE AND ADMINISTRATION":                             "34068-7",
    "DOSAGE FORMS AND STRENGTHS":                            "43678-2",
    "CONTRAINDICATIONS":                                     "34070-3",
    "WARNINGS":                                              "34071-1",
    "WARNINGS AND PRECAUTIONS":                              "43685-7",
    "PRECAUTIONS":                                           "42232-9",
    "ADVERSE REACTIONS":                                     "34084-4",
    "DRUG INTERACTIONS":                                     "34073-7",
    "USE IN SPECIFIC POPULATIONS":                           "43684-0",
    "PREGNANCY":                                             "42228-7",
    "LACTATION":                                             "77290-5",
    "NURSING MOTHERS":                                       "34080-2",
    "LABOR AND DELIVERY":                                    "34079-4",
    "PEDIATRIC USE":                                         "34081-0",
    "GERIATRIC USE":                                         "34082-8",
    "FEMALES AND MALES OF REPRODUCTIVE POTENTIAL":           "77291-3",
    "OVERDOSAGE":                                            "34088-5",
    "DESCRIPTION":                                           "34089-3",
    "CLINICAL PHARMACOLOGY":                                 "34090-1",
    "MECHANISM OF ACTION":                                   "43679-0",
    "PHARMACOKINETICS":                                      "43682-4",
    "PHARMACODYNAMICS":                                      "43681-6",
    "NONCLINICAL TOXICOLOGY":                                "43680-8",
    "CARCINOGENESIS MUTAGENESIS AND IMPAIRMENT OF FERTILITY": "34083-6",
    "CLINICAL STUDIES":                                      "34092-7",
    "REFERENCES":                                            "34093-5",
    "HOW SUPPLIED":                                          "34069-5",
    "STORAGE AND HANDLING":                                  "44425-7",
    "INFORMATION FOR PATIENTS":                              "34076-0",
    "PATIENT COUNSELING INFORMATION":                        "34076-0",
    "RENAL IMPAIRMENT":                                      "88828-9",
    "HEPATIC IMPAIRMENT":                                    "88829-7",
    "IMMUNOGENICITY":                                        "88830-5",
    "CLINICAL TRIALS EXPERIENCE":                            "90374-0",
    "POSTMARKETING EXPERIENCE":                              "90375-7",
    "INSTRUCTIONS FOR USE":                                  "59845-8",
    "DRUG ABUSE AND DEPENDENCE":                             "42227-9",
    "LABORATORY TESTS":                                      "34075-2",
    "RECENT MAJOR CHANGES":                                  "43683-2",
}

# Sections excluded from the index — metadata-only or low patient value
DROP_LOINC = {
    "48780-1",  # SPL product data elements
    "34089-3",  # Description
    "43680-8",  # Nonclinical Toxicology
    "34083-6",  # Carcinogenesis & Mutagenesis
    "34069-5",  # How Supplied
    "51945-4",  # Package Label / Display Panel
    "34093-5",  # References
    "43683-2",  # Recent Major Changes
}

DROP_TITLE_PATTERNS = {"PRINCIPAL DISPLAY PANEL", "DISPLAY PANEL"}

# LOINC codes that identify each section type
MEDICATION_GUIDE_LOINC   = {"42231-1"}
PPI_LOINC                = {"42230-3"}
OTC_DOC_TYPES            = {"34390-5"}
PATIENT_COUNSELING_LOINC = {"34076-0", "59845-8"}
CLINICAL_PHARMA_LOINC    = {"34090-1", "43679-0", "43682-4", "43681-6", "34092-7"}


def _t(el) -> Optional[str]:
    if el is None:
        return None
    text = (el.text or "").strip()
    return text if text else None


def _normalize_title(title: str) -> str:
    title = re.sub(r'^\d+(\.\d+)*\s+', '', title)
    title = title.upper()
    title = re.sub(r'[^A-Z0-9\s]', ' ', title)
    return re.sub(r'\s+', ' ', title).strip()


def _infer_loinc(title: str) -> Optional[str]:
    if not title:
        return None
    return TITLE_LOINC_MAP.get(_normalize_title(title))


def _resolve_loinc(
    code_el,
    title: str,
    parent_loinc: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    # Priority: direct code → title inference → inherit from parent → None
    # 42229-5 (unclassified) is treated as absent so inheritance still applies
    if code_el is not None:
        cs = code_el.get("codeSystem", "")
        code = code_el.get("code", "")
        if cs == LOINC_CS and code and code != UNCLASSIFIED_LOINC:
            return code, "direct"

    inferred = _infer_loinc(title)
    if inferred:
        return inferred, "title_inferred"

    if parent_loinc is not None:
        return parent_loinc, "inherited"

    return None, None


def _should_drop(loinc_code: Optional[str], title: str) -> bool:
    if loinc_code in DROP_LOINC:
        return True
    return _normalize_title(title) in DROP_TITLE_PATTERNS


def _get_section_type(loinc_code: Optional[str], doc_type: Optional[str]) -> str:
    if loinc_code in MEDICATION_GUIDE_LOINC:
        return "medication_guide"
    if loinc_code in PPI_LOINC:
        return "ppi"
    if doc_type in OTC_DOC_TYPES:
        return "otc_drug_facts"
    if loinc_code in PATIENT_COUNSELING_LOINC:
        return "patient_counseling"
    if loinc_code in CLINICAL_PHARMA_LOINC:
        return "clinical_pharmacology"
    return "standard"


def _extract_text(text_el) -> str:
    if text_el is None:
        return ""
    return re.sub(r'\s+', ' ', " ".join(text_el.itertext())).strip()


def _walk(
    parent_el,
    doc_type: Optional[str],
    parent_loinc: Optional[str],
    parent_title_path: list[str],
    depth: int,
    results: list,
) -> None:
    for component in parent_el:
        if etree.QName(component.tag).localname != "component":
            continue
        section = component.find(f"{{{NS}}}section")
        if section is None:
            continue

        code_el  = section.find(f"{{{NS}}}code")
        title_el = section.find(f"{{{NS}}}title")
        text_el  = section.find(f"{{{NS}}}text")

        raw_title = (title_el.text or "").strip() if title_el is not None else ""

        loinc_code, loinc_source = _resolve_loinc(code_el, raw_title, parent_loinc)

        # continue skips the recursive call below, so children are dropped too
        if _should_drop(loinc_code, raw_title):
            continue

        section_type = _get_section_type(loinc_code, doc_type)
        title_path = parent_title_path + ([raw_title] if raw_title else [])
        text = _extract_text(text_el)

        if text:
            results.append({
                "loinc_code":         loinc_code,
                "loinc_source":       loinc_source,
                "section_title":      raw_title,
                "section_title_path": title_path,
                "section_type":       section_type,
                "text":               text,
                "depth":              depth,
            })

        _walk(section, doc_type, loinc_code, title_path, depth + 1, results)


def walk_sections(structured_body, doc_type: Optional[str]) -> list[dict]:
    """Walk a structuredBody element and return a flat list of section dicts, drop-filtered and type-tagged."""
    results = []
    _walk(structured_body, doc_type, None, [], 0, results)
    return results


def extract_header(xml_path: str) -> dict:
    """Parse label-level metadata from an SPL XML file. Missing fields return None; never raises."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    _sid = root.find(f"{{{NS}}}setId")
    setid = _sid.get("root") if _sid is not None else None

    _eff = root.find(f"{{{NS}}}effectiveTime")
    effective_time = _eff.get("value") if _eff is not None else None

    doc_el = root.find(f"{{{NS}}}code")
    doc_type = doc_el.get("code") if doc_el is not None else None

    marketing_category = None
    drug_name = None
    active_ingredients = []
    dosage_form = None
    route = None

    product_section = None
    for section in root.iter(f"{{{NS}}}section"):
        code_el = section.find(f"{{{NS}}}code")
        if code_el is not None and code_el.get("code") == PRODUCT_DATA_LOINC:
            product_section = section
            break

    if product_section is not None:
        for approval_el in product_section.iter(f"{{{NS}}}approval"):
            code_el = approval_el.find(f"{{{NS}}}code")
            if code_el is not None and code_el.get("codeSystem") == NCI_CS:
                marketing_category = code_el.get("code")
                break

        for mp in product_section.iter(f"{{{NS}}}manufacturedProduct"):
            if drug_name is None:
                name_el = mp.find(f"{{{NS}}}name")
                if name_el is None:
                    name_el = mp.find(f".//{{{NS}}}name")
                drug_name = _t(name_el)

            if dosage_form is None:
                fc = mp.find(f".//{{{NS}}}formCode")
                if fc is not None:
                    dosage_form = fc.get("displayName")

            if route is None:
                rc = mp.find(f".//{{{NS}}}routeCode")
                if rc is not None:
                    route = rc.get("displayName")

        for ing in product_section.iter(f"{{{NS}}}ingredient"):
            if ing.get("classCode") in ACTIVE_CLASS_CODES:
                sub = ing.find(f".//{{{NS}}}ingredientSubstance")
                if sub is not None:
                    name_el = sub.find(f"{{{NS}}}name")
                    name = _t(name_el)
                    if name and name not in active_ingredients:
                        active_ingredients.append(name)

    return {
        "setid":               setid,
        "effective_time":      effective_time,
        "doc_type":            doc_type,
        "marketing_category":  marketing_category,
        "drug_name":           drug_name,
        "active_ingredients":  active_ingredients,
        "dosage_form":         dosage_form,
        "route":               route,
    }


def parse_label(xml_path: str) -> dict:
    """Parse a complete SPL XML label. Returns {"header": dict, "sections": list[dict]}."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    header = extract_header(xml_path)

    structured_body = None
    for comp in root:
        if etree.QName(comp.tag).localname == "component":
            sb = comp.find(f"{{{NS}}}structuredBody")
            if sb is not None:
                structured_body = sb
                break

    sections = walk_sections(structured_body, header["doc_type"]) if structured_body is not None else []

    return {"header": header, "sections": sections}
