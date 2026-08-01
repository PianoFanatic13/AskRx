import argparse
import logging
from pathlib import Path

import psycopg

from backend.pipeline.parser import extract_header
from backend.pipeline.pipeline import load_rxnorm_cache, save_rxnorm_cache
from backend.pipeline.rxnorm import resolve_rxcui, set_failure_log

log = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"
_DEFAULT_CACHE_PATH = Path("data/rxnorm_cache.json")
_COMMIT_INTERVAL = 1
_LOG_INTERVAL = 500


def backfill(
    labels_dir: Path,
    dsn: str,
    cache_path: Path,
    offset: int = 0,
    limit: int | None = None,
) -> None:
    """Resolve rxcui for single-ingredient labels that were never resolved.

    Interleaves the XML scan with per-match resolution and per-row commits
    (rather than collect-everything-then-write-once) so a killed/interrupted
    run still leaves committed progress behind - re-running picks up only
    the setids still NULL, since target_setids is re-queried fresh each run.

    offset/limit slice the sorted XML file list, so a full backfill can be
    driven as several short, bounded calls instead of one long-running one.
    """
    cache = load_rxnorm_cache(cache_path)
    log.info("RxNorm cache loaded: %d entries", len(cache))

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT setid FROM chunks WHERE rxcui IS NULL")
            target_setids = {row[0] for row in cur.fetchall()}
        log.info("%d distinct setids with NULL rxcui", len(target_setids))

        best_eff: dict[str, str] = {}
        found_setids: set[str] = set()
        combination_skipped = 0
        backfilled = 0
        still_unresolved: list[str] = []
        since_commit = 0

        all_xml_paths = sorted(labels_dir.glob("*.xml"))
        xml_paths = all_xml_paths[offset : offset + limit if limit else None]
        log.info(
            "Scanning slice [%d:%d] of %d total XMLs",
            offset, offset + len(xml_paths), len(all_xml_paths),
        )
        cur = conn.cursor()
        for i, xml_path in enumerate(xml_paths, 1):
            if not target_setids:
                break

            try:
                header = extract_header(str(xml_path))
            except Exception as exc:
                log.warning("Header extraction failed for %s: %s", xml_path.name, exc)
                continue

            setid = header.get("setid")
            if setid not in target_setids:
                continue

            # Same latest-effective_time tie-break as pipeline.py's _pass1 -
            # multiple XML files can share a setid.
            eff = header.get("effective_time") or ""
            if setid in found_setids and eff <= best_eff.get(setid, ""):
                continue
            best_eff[setid] = eff
            found_setids.add(setid)

            ingredients = header.get("active_ingredients", [])
            if len(ingredients) != 1:
                # Combination drug - correctly has no single rxcui by design
                # (see pipeline.py's _resolve_label_rxcui). Not a candidate.
                combination_skipped += 1
                target_setids.discard(setid)
                continue

            ingredient = ingredients[0]
            if ingredient not in cache:
                cache[ingredient] = resolve_rxcui(ingredient)
            rxcui = cache[ingredient]

            if rxcui:
                cur.execute("UPDATE chunks SET rxcui = %s WHERE setid = %s", (rxcui, setid))
                backfilled += 1
                since_commit += 1
            else:
                still_unresolved.append(ingredient)

            target_setids.discard(setid)

            if since_commit >= _COMMIT_INTERVAL:
                conn.commit()
                save_rxnorm_cache(cache_path, cache)
                since_commit = 0

            if i % _LOG_INTERVAL == 0:
                log.info(
                    "Scanned %d/%d XMLs, %d setids remaining, %d backfilled so far",
                    i, len(xml_paths), len(target_setids), backfilled,
                )

        conn.commit()
        save_rxnorm_cache(cache_path, cache)
        cur.close()

        # Only meaningful for a full (unsliced) scan - with slicing, most
        # remaining setids simply haven't been scanned yet in this slice.
        if limit is None and offset == 0 and target_setids:
            log.warning(
                "%d setids had no matching XML found: %s",
                len(target_setids), sorted(target_setids)[:5],
            )

        log.info(
            "Slice complete: %d combination drugs skipped (by design), "
            "%d single-ingredient labels backfilled, %d still unresolved, %d not found in this slice",
            combination_skipped, backfilled, len(still_unresolved), len(target_setids),
        )
        if still_unresolved:
            log.warning("Still unresolved ingredients: %s", sorted(set(still_unresolved))[:20])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chunks.rxcui for single-ingredient labels that were never "
        "resolved (no re-embedding). Combination-drug NULLs are left untouched by design. "
        "Safe to interrupt and re-run, and safe to run in slices via --offset/--limit - "
        "progress commits incrementally and re-queries remaining NULLs fresh each run."
    )
    parser.add_argument("--labels-dir", default=Path("data/labels"), type=Path)
    parser.add_argument(
        "--dsn",
        default=_DEFAULT_DSN,
        help=f"PostgreSQL DSN (default: {_DEFAULT_DSN})",
    )
    parser.add_argument("--cache-path", default=_DEFAULT_CACHE_PATH, type=Path)
    parser.add_argument("--failure-log", default=Path("data/rxnorm_failures.tsv"), type=Path)
    parser.add_argument("--offset", type=int, default=0, help="Start index into the sorted XML file list")
    parser.add_argument("--limit", type=int, default=None, help="Number of XML files to scan in this run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_failure_log(args.failure_log)
    backfill(args.labels_dir, args.dsn, args.cache_path, offset=args.offset, limit=args.limit)


if __name__ == "__main__":
    main()
