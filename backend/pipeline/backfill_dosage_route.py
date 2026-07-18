import argparse
import logging
from pathlib import Path

import psycopg

from backend.pipeline.parser import extract_header

log = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"


def _collect_dosage_route(
    labels_dir: Path, target_setids: set[str]
) -> dict[str, tuple[str | None, str | None]]:
    """Map each target setid to (dosage_form, route) from its latest-effective_time XML.

    Mirrors pipeline.py's _pass1 tie-break (multiple XML files can share a
    setid; the one with the latest effective_time wins), without the RxNorm
    resolution _pass1 also does, which this backfill has no use for.
    """
    best_eff: dict[str, str] = {}
    result: dict[str, tuple[str | None, str | None]] = {}
    xml_paths = sorted(labels_dir.glob("*.xml"))
    for i, xml_path in enumerate(xml_paths, 1):
        try:
            header = extract_header(str(xml_path))
        except Exception as exc:
            log.warning("Header extraction failed for %s: %s", xml_path.name, exc)
            continue
        setid = header.get("setid")
        if setid not in target_setids:
            continue
        eff = header.get("effective_time") or ""
        if setid not in best_eff or eff > best_eff[setid]:
            best_eff[setid] = eff
            result[setid] = (header.get("dosage_form"), header.get("route"))
        if i % 20000 == 0:
            log.info("Scanned %d/%d XMLs", i, len(xml_paths))
    return result


def backfill(labels_dir: Path, dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT setid FROM chunks")
            target_setids = {row[0] for row in cur.fetchall()}
        log.info("%d distinct setids in chunks", len(target_setids))

        mapping = _collect_dosage_route(labels_dir, target_setids)
        log.info("Resolved dosage_form/route for %d/%d setids", len(mapping), len(target_setids))
        missing = target_setids - mapping.keys()
        if missing:
            log.warning("%d setids had no matching XML found: %s", len(missing), sorted(missing)[:5])

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET dosage_form = %s, route = %s WHERE setid = %s",
                [(form, route, setid) for setid, (form, route) in mapping.items()],
            )
        conn.commit()
        log.info("Backfill complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chunks.dosage_form/route from source XML headers (no re-embedding)."
    )
    parser.add_argument("--labels-dir", default=Path("data/labels"), type=Path)
    parser.add_argument(
        "--dsn",
        default=_DEFAULT_DSN,
        help=f"PostgreSQL DSN (default: {_DEFAULT_DSN})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    backfill(args.labels_dir, args.dsn)


if __name__ == "__main__":
    main()
