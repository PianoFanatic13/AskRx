import io
import logging
import zipfile
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.svg'}


def _extract_xml_from_inner(inner: zipfile.ZipFile, output_dir: Path) -> Path | None:
    for name in inner.namelist():
        suffix = Path(name).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            continue
        if suffix == '.xml':
            dest = output_dir / Path(name).name
            dest.write_bytes(inner.read(name))
            return dest
    return None


def extract_xmls(
    zip_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[Path]:
    """Yield Paths of extracted XMLs from a DailyMed ZIP-of-ZIPs.

    Each inner ZIP named {setid}.zip contains one SPL XML file plus images.
    Images are skipped. Existing destination files are silently overwritten.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(zip_path, 'r') as outer:
        for member in outer.infolist():
            if limit is not None and count >= limit:
                break
            if not member.filename.endswith('.zip'):
                log.debug("Skipping non-ZIP member: %s", member.filename)
                continue
            inner_bytes = outer.read(member.filename)
            with zipfile.ZipFile(io.BytesIO(inner_bytes), 'r') as inner:
                path = _extract_xml_from_inner(inner, output_dir)
            if path is None:
                log.warning("No XML found in inner ZIP: %s", member.filename)
                continue
            count += 1
            yield path

    log.info("Extracted %d XMLs to %s", count, output_dir)
