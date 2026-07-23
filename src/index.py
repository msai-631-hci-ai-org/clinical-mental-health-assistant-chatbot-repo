"""Safe local FAISS persistence and semantic retrieval."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import Settings
from src.documents import (
    TextChunk,
    chunk_documents,
    load_documents,
    source_fingerprint,
)

INDEX_FILENAME = "clinical.index"
RECORDS_FILENAME = "records.json"
MANIFEST_FILENAME = "manifest.json"
INDEX_FORMAT_VERSION = 2


class NoKnowledgeBaseError(RuntimeError):
    """Raised when no ingestible source documents are present."""


@dataclass(frozen=True)
class SearchResult:
    """One retrieved knowledge chunk and its cosine similarity."""

    text: str
    source: str
    page: int | None
    score: float


class LocalVectorIndex:
    """Build and query a FAISS index without unsafe pickle deserialization."""

    def __init__(self, settings: Settings, embedding_model: Any | None = None):
        self.settings = settings
        self._embedding_model = embedding_model
        self._index: Any | None = None
        self._records: list[TextChunk] = []
        self._lock = threading.Lock()

    @property
    def record_count(self) -> int:
        return len(self._records)

    def ensure_current(self, force: bool = False) -> int:
        """Load a matching index or rebuild it when sources/config change."""

        if self._index is not None and not force:
            return self.record_count
        with self._lock:
            if self._index is not None and not force:
                return self.record_count
            fingerprint = source_fingerprint(self.settings.data_dir)
            if not fingerprint:
                raise NoKnowledgeBaseError(
                    "No PDF, Markdown, or text knowledge files were found in "
                    f"{self.settings.data_dir}."
                )

            if not force and self._can_load(fingerprint):
                self._load()
            else:
                self._build(fingerprint)
            return self.record_count

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the most semantically similar chunks."""

        if self._index is None:
            self.ensure_current()

        import numpy as np

        query_vector = self._model().encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores, indexes = self._index.search(
            np.asarray(query_vector, dtype="float32"),
            min(top_k, len(self._records)),
        )

        results: list[SearchResult] = []
        for score, position in zip(scores[0], indexes[0]):
            if position < 0:
                continue
            record = self._records[int(position)]
            results.append(
                SearchResult(
                    text=record.text,
                    source=record.source,
                    page=record.page,
                    score=float(score),
                )
            )
        return results

    def _model(self) -> Any:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is not installed. "
                    "Install requirements.txt first."
                ) from exc

            device = (
                f"cuda:{self.settings.device}" if self.settings.device >= 0 else "cpu"
            )
            self._embedding_model = SentenceTransformer(
                self.settings.embedding_model,
                device=device,
                revision=self.settings.embedding_revision,
            )
        return self._embedding_model

    def _can_load(self, fingerprint: str) -> bool:
        paths = self._paths()
        if not all(path.is_file() for path in paths.values()):
            return False
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return manifest == self._manifest(fingerprint)

    def _build(self, fingerprint: str) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "FAISS and NumPy are required. Install requirements.txt first."
            ) from exc

        documents = load_documents(self.settings.data_dir)
        records = chunk_documents(
            documents,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        if not records:
            raise NoKnowledgeBaseError(
                "The knowledge files contained no extractable text."
            )

        vectors = self._model().encode(
            [record.text for record in records],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        paths = self._paths()
        temporary_index = paths["index"].with_suffix(".index.tmp")
        faiss.write_index(index, str(temporary_index))
        os.replace(temporary_index, paths["index"])
        self._write_json_atomic(
            paths["records"],
            [asdict(record) for record in records],
        )
        self._write_json_atomic(paths["manifest"], self._manifest(fingerprint))

        self._index = index
        self._records = records

    def _load(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is required. Install requirements.txt first."
            ) from exc

        paths = self._paths()
        index = faiss.read_index(str(paths["index"]))
        raw_records = json.loads(paths["records"].read_text(encoding="utf-8"))
        records = [TextChunk(**record) for record in raw_records]
        if index.ntotal != len(records):
            raise RuntimeError("The local vector index and metadata do not match.")
        self._index = index
        self._records = records

    def _manifest(self, fingerprint: str) -> dict[str, Any]:
        return {
            "format_version": INDEX_FORMAT_VERSION,
            "source_fingerprint": fingerprint,
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.settings.embedding_revision,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }

    def _paths(self) -> dict[str, Path]:
        return {
            "index": self.settings.index_dir / INDEX_FILENAME,
            "records": self.settings.index_dir / RECORDS_FILENAME,
            "manifest": self.settings.index_dir / MANIFEST_FILENAME,
        }

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
