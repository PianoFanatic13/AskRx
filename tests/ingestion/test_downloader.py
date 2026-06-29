import io
import zipfile

import pytest

from backend.pipeline.downloader import extract_xmls


def _make_inner_zip(setid: str, xml_content: str = "<document/>", include_images: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f"{setid}.xml", xml_content)
        if include_images:
            zf.writestr(f"{setid}.jpg", b"\xff\xd8\xff")
            zf.writestr(f"{setid}.png", b"\x89PNG")
    return buf.getvalue()


def _make_outer_zip(setids: list[str], extra_members: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for setid in setids:
            zf.writestr(f"{setid}.zip", _make_inner_zip(setid))
        if extra_members:
            for name, data in extra_members.items():
                zf.writestr(name, data)
    return buf.getvalue()


def test_extract_xmls_writes_xml_files(tmp_path):
    outer = _make_outer_zip(["aaaa", "bbbb", "cccc"])
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "labels"

    paths = list(extract_xmls(zip_path, out_dir))

    assert len(paths) == 3
    assert all(p.exists() for p in paths)
    assert {p.stem for p in paths} == {"aaaa", "bbbb", "cccc"}


def test_extract_xmls_skips_image_files(tmp_path):
    outer = _make_outer_zip(["aaaa"], )
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "labels"

    list(extract_xmls(zip_path, out_dir))

    suffixes = {p.suffix.lower() for p in out_dir.iterdir()}
    assert suffixes == {'.xml'}


def test_extract_xmls_limit_respected(tmp_path):
    outer = _make_outer_zip(["a", "b", "c", "d", "e"])
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "labels"

    paths = list(extract_xmls(zip_path, out_dir, limit=3))

    assert len(paths) == 3
    assert len(list(out_dir.glob("*.xml"))) == 3


def test_extract_xmls_skips_non_zip_members(tmp_path):
    outer = _make_outer_zip(["aaaa"], extra_members={"README.txt": b"hello"})
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "labels"

    paths = list(extract_xmls(zip_path, out_dir))

    assert len(paths) == 1
    assert paths[0].name == "aaaa.xml"


def test_extract_xmls_creates_output_dir(tmp_path):
    outer = _make_outer_zip(["aaaa"])
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "new" / "nested" / "labels"

    assert not out_dir.exists()
    list(extract_xmls(zip_path, out_dir))
    assert out_dir.exists()


def test_extract_xmls_idempotent(tmp_path):
    outer = _make_outer_zip(["aaaa"])
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer)
    out_dir = tmp_path / "labels"

    list(extract_xmls(zip_path, out_dir))
    list(extract_xmls(zip_path, out_dir))  # should not raise

    assert len(list(out_dir.glob("*.xml"))) == 1


def test_extract_xmls_yielded_paths_match_content(tmp_path):
    xml_content = '<document xmlns="urn:hl7-org:v3"/>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as outer:
        outer.writestr("aaaa.zip", _make_inner_zip("aaaa", xml_content=xml_content, include_images=False))
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(buf.getvalue())
    out_dir = tmp_path / "labels"

    paths = list(extract_xmls(zip_path, out_dir))

    assert len(paths) == 1
    assert paths[0].read_text(encoding='utf-8') == xml_content


def test_extract_xmls_inner_zip_with_no_xml_skipped(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as inner:
        inner.writestr("image.png", b"\x89PNG")
    images_only = buf.getvalue()

    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, 'w') as outer:
        outer.writestr("aaaa.zip", images_only)
    zip_path = tmp_path / "outer.zip"
    zip_path.write_bytes(outer_buf.getvalue())
    out_dir = tmp_path / "labels"

    paths = list(extract_xmls(zip_path, out_dir))

    assert paths == []
    assert not any(out_dir.glob("*.xml"))
