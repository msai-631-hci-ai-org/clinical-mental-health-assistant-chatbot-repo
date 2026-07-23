"""Load and chunk trusted local knowledge-base documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class SourceDocument:
    """Text extracted from one source unit, such as a PDF page."""

    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A retrieval-ready piece of a source document."""

    text: str
    source: str
    page: int | None
    chunk_id: int


def discover_source_files(data_dir: Path) -> list[Path]:
    """Return supported knowledge files in deterministic order."""

    if not data_dir.exists():
        return []
    return sorted(
        (
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.lower().startswith("readme")
        ),
        key=lambda path: path.as_posix().lower(),
    )


def source_fingerprint(data_dir: Path) -> str:
    """Hash file paths and bytes so stale indexes can be detected."""

    files = discover_source_files(data_dir)
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(data_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_documents(data_dir: Path) -> list[SourceDocument]:
    """Extract text from Markdown, text, and PDF knowledge files."""

    documents: list[SourceDocument] = []
    for path in discover_source_files(data_dir):
        source = path.relative_to(data_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path, source))
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(SourceDocument(text=text, source=source))
    return documents


def _load_pdf(path: Path, source: str) -> list[SourceDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf. Install requirements.txt first."
        ) from exc

    pages: list[SourceDocument] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(SourceDocument(text=text, source=source, page=page_number))
    return pages


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Split text near natural boundaries with deterministic overlap."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = _normalize_text(text)
    cursor = 0
    while cursor < len(normalized):
        end = min(cursor + chunk_size, len(normalized))
        if end < len(normalized):
            floor = cursor + chunk_size // 2
            boundaries = (
                normalized.rfind("\n\n", floor, end),
                normalized.rfind(". ", floor, end),
                normalized.rfind(" ", floor, end),
            )
            boundary = max(boundaries)
            if boundary >= floor:
                end = boundary + (1 if normalized[boundary] == " " else 0)

        chunk = normalized[cursor:end].strip()
        if chunk:
            yield chunk
        if end >= len(normalized):
            break
        next_cursor = max(end - overlap, cursor + 1)
        if (
            next_cursor > 0
            and next_cursor < len(normalized)
            and not normalized[next_cursor - 1].isspace()
            and not normalized[next_cursor].isspace()
        ):
            next_space = normalized.find(
                " ",
                next_cursor,
                min(next_cursor + 40, len(normalized)),
            )
            if next_space != -1:
                next_cursor = next_space + 1
        cursor = next_cursor


def chunk_documents(
    documents: Iterable[SourceDocument],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Convert extracted documents into numbered retrieval chunks."""

    chunks: list[TextChunk] = []
    for document in documents:
        for text in split_text(document.text, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    text=text,
                    source=document.source,
                    page=document.page,
                    chunk_id=len(chunks),
                )
            )
    return chunks
