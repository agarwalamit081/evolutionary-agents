"""Tests for the document_parser builtin: HTML input (D1) + pymupdf rich PDF (D3).

HTML parses via the shared ``web_scraper`` trafilatura→markdownify chain; the
opt-in ``extract_figures`` path renders tables/figures via pymupdf (``fitz``)
and degrades to pypdf text-only when fitz is absent. Plus regression coverage
for path traversal, unsupported types, and plain text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.tools.builtin import document_parser as dp

# Distinctive article phrase that appears ONLY in the body, never in nav/footer.
_ARTICLE_PHRASE = "quicksort-partition-isolation-fixture-7Q"
# Boilerplate that appears ONLY in the page nav, never in the article body.
_NAV_PHRASE = "menu-signin-boilerplate-9Z"


def _html_fixture() -> str:
    """A well-formed article so trafilatura's main-content extractor engages."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><title>Sample Article</title></head>
<body>
<nav><a href="/">Home</a><a href="/login">{_NAV_PHRASE}</a></nav>
<article>
<h1>Sample Article</h1>
<p>This article describes the {_ARTICLE_PHRASE} procedure in detail.</p>
<p>The first paragraph establishes the motivation for the work and the context
in which the experiment was performed. It spans several sentences so that the
main-content extractor recognizes genuine prose rather than a stray fragment.</p>
<p>A second paragraph continues the discussion with supporting evidence and a
brief note on the methodology, including the observed latency and throughput
numbers that frame the contribution relative to the prior baseline.</p>
</article>
<footer><a href="/about">About</a> copyright notice</footer>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# D1 — HTML → markdown
# --------------------------------------------------------------------------- #


async def test_html_is_parsed_to_markdown(tmp_path: Path) -> None:
    pytest.importorskip("trafilatura")
    (tmp_path / "doc.html").write_text(_html_fixture(), encoding="utf-8")

    out = await dp.document_parser(file_path="doc.html", sandbox_root=str(tmp_path))

    assert not out.startswith("ERROR"), out
    # The article body survives extraction (robust: holds whether trafilatura or
    # the markdownify fallback produced the markdown).
    assert _ARTICLE_PHRASE in out


async def test_html_extension_htm_also_parsed(tmp_path: Path) -> None:
    pytest.importorskip("trafilatura")
    (tmp_path / "doc.htm").write_text(_html_fixture(), encoding="utf-8")

    out = await dp.document_parser(file_path="doc.htm", sandbox_root=str(tmp_path))

    assert not out.startswith("ERROR"), out
    assert _ARTICLE_PHRASE in out


# --------------------------------------------------------------------------- #
# D3 — pymupdf rich PDF extraction (tables + figures), opt-in
# --------------------------------------------------------------------------- #


def test_rows_to_markdown_renders_and_escapes() -> None:
    md = dp._rows_to_markdown([["name", "value"], ["alpha", "1|2"], ["beta", None]])

    # Header + separator + two data rows.
    assert md.count("\n") == 3
    assert md.startswith("| name | value |")
    # Literal pipe within a cell is escaped so it never breaks the grid.
    assert "| alpha | 1\\|2 |" in md
    # A None cell renders as empty, not the string "None".
    assert "None" not in md
    assert md.endswith("| beta |  |")


def _make_text_pdf(path: Path) -> None:
    """Create a text-only PDF fixture with fitz (no embedded images)."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Turing agent document parser test fixture.")
    page.insert_text((72, 100), "Second line of the fixture content body.")
    doc.save(str(path))
    doc.close()


def test_pdf_rich_extracts_text_without_figures(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    pdf = tmp_path / "doc.pdf"
    _make_text_pdf(pdf)
    figures = tmp_path / "figures"

    out = dp._extract_pdf_rich(pdf, figures)

    assert "fixture content" in out
    # No embedded images → nothing rendered, no figures section emitted.
    assert "Extracted figures" not in out
    assert not figures.exists()


def test_pdf_rich_falls_back_when_fitz_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build the fixture with the REAL fitz first, then hide fitz from the importer.
    pdf = tmp_path / "doc.pdf"
    _make_text_pdf(pdf)  # importorskips here if fitz is absent
    monkeypatch.setitem(sys.modules, "fitz", None)

    out = dp._extract_pdf_rich(pdf, tmp_path / "figures")

    # ImportError → graceful degrade to the pypdf text-only path.
    assert "fixture content" in out
    assert "ERROR" not in out


async def test_pdf_default_path_is_pypdf_text_only(tmp_path: Path) -> None:
    """extract_figures defaults to False → the pypdf text path, not pymupdf."""
    pdf = tmp_path / "doc.pdf"
    _make_text_pdf(pdf)

    out = await dp.document_parser(file_path="doc.pdf", sandbox_root=str(tmp_path))

    assert "fixture content" in out
    # pymupdf per-page headers are NOT added on the default (text-only) path.
    assert "### Page" not in out


# --------------------------------------------------------------------------- #
# Regression — path guard, unsupported types, plain text
# --------------------------------------------------------------------------- #


async def test_path_traversal_blocked(tmp_path: Path) -> None:
    out = await dp.document_parser(file_path="../secret.txt", sandbox_root=str(tmp_path))
    assert out.startswith("ERROR: Path traversal blocked")


async def test_missing_file_error(tmp_path: Path) -> None:
    out = await dp.document_parser(file_path="nope.pdf", sandbox_root=str(tmp_path))
    assert out.startswith("ERROR: File not found")


async def test_unsupported_extension_error(tmp_path: Path) -> None:
    (tmp_path / "f.bin").write_bytes(b"\x00\x01")
    out = await dp.document_parser(file_path="f.bin", sandbox_root=str(tmp_path))
    assert out.startswith("ERROR: Unsupported file type")
    assert "html" in out  # html is now in the supported list


async def test_plain_text_passthrough(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello turing agent", encoding="utf-8")
    out = await dp.document_parser(file_path="notes.txt", sandbox_root=str(tmp_path))
    assert out == "hello turing agent"
