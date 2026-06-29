import argparse
import json
import logging
from pathlib import Path

from backend.pipeline.chunker import chunk_section
from backend.pipeline.dedup import select_canonical, select_canonical_no_rxcui
from backend.pipeline.downloader import extract_xmls
from backend.pipeline.parser import extract_header, parse_label
from backend.pipeline.rxnorm import resolve_rxcui, set_failure_log

log = logging.getLogger(__name__)

_CACHE_SAVE_INTERVAL = 500


def load_rxnorm_cache(path: Path) -> dict[str, str | None]:
    """Return existing cache from JSON, or {} if the file does not exist."""
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_rxnorm_cache(path: Path, cache: dict[str, str | None]) -> None:
    """Atomically write cache via a .tmp sibling then rename."""
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _resolve_label_rxcui(
    ingredients: list[str],
    cache: dict[str, str | None],
) -> str | None:
    """Resolve every ingredient against the cache, querying RxNorm for misses.

    Returns None for multi-ingredient labels — they fall through to
    select_canonical_no_rxcui, which groups by ingredient tuple instead.
    """
    for ing in ingredients:
        if ing not in cache:
            cache[ing] = resolve_rxcui(ing)
    if len(ingredients) != 1:
        return None
    return cache[ingredients[0]]


def _pass1(
    xml_paths: list[Path],
    cache: dict[str, str | None],
    cache_path: Path,
) -> tuple[list[dict], dict[str, str | None], dict[str, Path]]:
    all_records: list[dict] = []
    setid_to_rxcui: dict[str, str | None] = {}
    # Maps setid -> path of the file with the latest effective_time for that setid.
    # DailyMed filenames are document IDs, not setids, so we must track this explicitly.
    setid_to_path: dict[str, Path] = {}
    setid_best_eff: dict[str, str] = {}
    new_resolutions = 0
    prev_saved_at = 0
    total = len(xml_paths)

    for i, xml_path in enumerate(xml_paths, 1):
        try:
            header = extract_header(str(xml_path))
        except Exception as exc:
            log.warning("Header extraction failed for %s: %s", xml_path.name, exc)
            continue

        cache_before = len(cache)
        rxcui = _resolve_label_rxcui(header['active_ingredients'], cache)
        new_resolutions += len(cache) - cache_before

        record = {**header, 'rxcui': rxcui}
        all_records.append(record)
        setid = header['setid']
        setid_to_rxcui[setid] = rxcui

        # Keep the file whose effective_time is latest (matches dedup's canonical pick).
        eff = header.get('effective_time') or ''
        if setid not in setid_to_path or eff > setid_best_eff.get(setid, ''):
            setid_to_path[setid] = xml_path
            setid_best_eff[setid] = eff

        if new_resolutions - prev_saved_at >= _CACHE_SAVE_INTERVAL:
            save_rxnorm_cache(cache_path, cache)
            prev_saved_at = new_resolutions

        if i % 1000 == 0:
            log.info("Pass 1: %d/%d labels processed", i, total)

    save_rxnorm_cache(cache_path, cache)
    log.info(
        "Pass 1 complete: %d records, %d new RxNorm resolutions (%d total cached)",
        len(all_records), new_resolutions, len(cache),
    )
    return all_records, setid_to_rxcui, setid_to_path


def _pass2(
    canonical_setids: list[str],
    setid_to_path: dict[str, Path],
    setid_to_rxcui: dict[str, str | None],
    output_path: Path,
) -> int:
    chunks_written = 0
    total = len(canonical_setids)

    with output_path.open('w', encoding='utf-8') as out:
        for i, setid in enumerate(canonical_setids, 1):
            xml_path = setid_to_path.get(setid)
            if xml_path is None:
                log.warning("Canonical setid has no file mapping: %s", setid)
                continue
            try:
                label = parse_label(str(xml_path))
            except Exception as exc:
                log.warning("parse_label failed for %s: %s", setid, exc)
                continue

            drug_name = label['header'].get('drug_name') or setid
            rxcui = setid_to_rxcui.get(setid)

            for section in label['sections']:
                for chunk in chunk_section(section, setid, drug_name, rxcui):
                    out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    chunks_written += 1

            if i % 500 == 0:
                log.info(
                    "Pass 2: %d/%d canonical labels done, %d chunks written",
                    i, total, chunks_written,
                )

    log.info("Pass 2 complete: %d chunks written to %s", chunks_written, output_path)
    return chunks_written


def run_pipeline(
    labels_dir: Path,
    output_path: Path,
    *,
    zip_paths: list[Path] | None = None,
    rxnorm_cache_path: Path,
    rxnorm_failure_log: Path | None = None,
    limit: int | None = None,
) -> None:
    """Run the full ingestion pipeline.

    If zip_paths is provided, extracts XMLs from each DailyMed part ZIP into
    labels_dir first. If omitted, labels_dir must already be populated.
    """
    if rxnorm_failure_log is not None:
        set_failure_log(rxnorm_failure_log)

    cache = load_rxnorm_cache(rxnorm_cache_path)
    log.info("RxNorm cache loaded: %d entries", len(cache))

    if zip_paths:
        total_extracted = 0
        remaining = limit
        for zip_path in zip_paths:
            if remaining is not None and remaining <= 0:
                break
            log.info("Extracting %s -> %s", zip_path.name, labels_dir)
            count = sum(1 for _ in extract_xmls(zip_path, labels_dir, limit=remaining))
            total_extracted += count
            log.info("  %d XMLs extracted", count)
            if remaining is not None:
                remaining -= count
        log.info("Extraction complete: %d total XMLs", total_extracted)
        xml_paths = sorted(labels_dir.glob("*.xml"))
        if limit is not None:
            xml_paths = xml_paths[:limit]
    else:
        xml_paths = sorted(labels_dir.glob("*.xml"))
        if limit is not None:
            xml_paths = xml_paths[:limit]
        log.info("Using %d pre-extracted XMLs from %s", len(xml_paths), labels_dir)

    all_records, setid_to_rxcui, setid_to_path = _pass1(xml_paths, cache, rxnorm_cache_path)

    canonical_with = select_canonical(all_records)
    canonical_without = select_canonical_no_rxcui(all_records)
    canonical_setids = list(dict.fromkeys(canonical_with + canonical_without))
    log.info(
        "Dedup: %d canonical labels (%d rxcui path, %d ingredient-string path) from %d total",
        len(canonical_setids), len(canonical_with), len(canonical_without), len(all_records),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_written = _pass2(canonical_setids, setid_to_path, setid_to_rxcui, output_path)

    log.info(
        "Pipeline complete: %d total labels -> %d canonical -> %d chunks written",
        len(all_records), len(canonical_setids), chunks_written,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AskRx ingestion pipeline: extract -> dedup -> chunk -> JSONL."
    )
    parser.add_argument(
        '--zip', default=None, nargs='+', type=Path,
        help='One or more part ZIPs (e.g. --zip part1.zip part2.zip ...)',
    )
    parser.add_argument(
        '--labels-dir', default=Path('data/labels'), type=Path,
        help='Directory for extracted XMLs (default: data/labels)',
    )
    parser.add_argument(
        '--output', default=Path('data/chunks/chunks.jsonl'), type=Path,
        help='JSONL output path (default: data/chunks/chunks.jsonl)',
    )
    parser.add_argument(
        '--rxnorm-cache', default=Path('data/rxnorm_cache.json'), type=Path,
        help='Ingredient-to-RXCUI cache path (default: data/rxnorm_cache.json)',
    )
    parser.add_argument(
        '--rxnorm-failure-log', default=None, type=Path,
        help='TSV file for failed RxNorm lookups (optional)',
    )
    parser.add_argument(
        '--limit', default=None, type=int,
        help='Process only the first N labels (use for dry runs)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    run_pipeline(
        zip_paths=args.zip,
        labels_dir=args.labels_dir,
        output_path=args.output,
        rxnorm_cache_path=args.rxnorm_cache,
        rxnorm_failure_log=args.rxnorm_failure_log,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
