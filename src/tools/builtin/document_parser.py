"""Tool extracting text from non-plain-text documents (PDF/DOCX/XLSX/CSV).

A dedicated parser is faster, safer, and cheaper than wrapping ``pypdf``/
``python-docx``/``openpyxl`` in a ``code_executor`` script every time. Sandbox
path guard reused from the ``file_reader`` pattern.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    lines: list[str] = []
    for ws in wb.worksheets:
        lines.append(f"### Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = [("" if c is None else str(c)) for c in row]
            if any(cell.strip() for cell in cells):
                lines.append("\t".join(cells))
    wb.close()
    return "\n".join(lines)


def _extract_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        return "\n".join(",".join(row) for row in reader)


def _extract(target: Path, ext: str, max_chars: int) -> str:
    """Dispatch by extension (sync; run off the event loop)."""
    try:
        if ext == ".pdf":
            text = _extract_pdf(target)
        elif ext == ".docx":
            text = _extract_docx(target)
        elif ext == ".xlsx":
            text = _extract_xlsx(target)
        elif ext == ".csv":
            text = _extract_csv(target)
        elif ext in (".txt", ".md", ".text", ".markdown"):
            text = target.read_text(encoding="utf-8", errors="replace")
        else:
            return f"ERROR: Unsupported file type '{ext}'. Supported: pdf, docx, xlsx, csv, txt, md."
    except Exception as exc:
        return f"ERROR: Failed to parse {target.name}: {exc}"

    text = text.strip()
    if not text:
        return f"ERROR: No extractable text in {target.name}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return text


async def document_parser(
    file_path: str,
    max_chars: int = 8000,
    sandbox_root: Optional[str] = None,
) -> str:
    """Extract text from a PDF, Word, Excel, CSV, or plain-text document.

    Args:
        file_path: Relative path to the document within the sandbox root.
        max_chars: Maximum characters of extracted text to return (default 8000).
        sandbox_root: Root directory for sandboxing (prevents path traversal).
            Defaults to ``settings.agent.workspace_root``.

    Returns:
        Extracted text, or an ``ERROR:`` string. Path traversal is blocked.
    """
    if sandbox_root is None:
        sandbox_root = get_settings().agent.workspace_root
    root = Path(sandbox_root).resolve()
    target = (root / file_path).resolve()

    if not target.is_relative_to(root):
        return f"ERROR: Path traversal blocked: {file_path}"
    if not target.exists():
        return f"ERROR: File not found: {file_path}"
    if not target.is_file():
        return f"ERROR: Not a file: {file_path}"

    # Cap input size at 20 MB to bound parsing cost.
    if target.stat().st_size > 20_000_000:
        return f"ERROR: File too large ({target.stat().st_size} bytes, max 20 MB)"

    logger.info(f"document_parser: {file_path}")
    ext = target.suffix.lower()
    return await asyncio.to_thread(_extract, target, ext, max_chars)


TOOL_DEFINITION = {
    "name": "document_parser",
    "handler": document_parser,
    "description": (
        "Extract text from a non-plain-text document: PDF, Word (.docx), "
        "Excel (.xlsx), CSV, or plain text/markdown. Faster and safer than "
        "writing a parsing script via code_executor. Path traversal is blocked."
    ),
    # Deterministic per (path, max_chars) — safe to cache within/across runs.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the document within the project directory.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of extracted text (default: 8000).",
                "default": 8000,
            },
        },
        "required": ["file_path"],
    },
}
