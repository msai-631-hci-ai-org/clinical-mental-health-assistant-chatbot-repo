"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _read_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings with laptop-friendly defaults."""

    data_dir: Path
    index_dir: Path
    embedding_model: str
    embedding_revision: str
    generation_model: str
    generation_revision: str
    device: int
    top_k: int
    min_relevance_score: float
    chunk_size: int
    chunk_overlap: int
    max_history_messages: int

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables and an optional `.env`."""

        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env", override=False)
        except ImportError:
            # `.env` support is convenient, not required for programmatic use.
            pass

        chunk_size = _read_int("CMHA_CHUNK_SIZE", 700, 200)
        chunk_overlap = _read_int("CMHA_CHUNK_OVERLAP", 100, 0)
        if chunk_overlap >= chunk_size:
            raise ValueError("CMHA_CHUNK_OVERLAP must be smaller than CMHA_CHUNK_SIZE")

        return cls(
            data_dir=_resolve_path(os.getenv("CMHA_DATA_DIR", ""), "data"),
            index_dir=_resolve_path(os.getenv("CMHA_INDEX_DIR", ""), "vector_db"),
            embedding_model=os.getenv(
                "CMHA_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_revision=os.getenv(
                "CMHA_EMBEDDING_REVISION",
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            ),
            generation_model=os.getenv(
                "CMHA_GENERATION_MODEL",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ),
            generation_revision=os.getenv(
                "CMHA_GENERATION_REVISION",
                "7ae557604adf67be50417f59c2c2f167def9a775",
            ),
            device=int(os.getenv("CMHA_DEVICE", "-1")),
            top_k=_read_int("CMHA_TOP_K", 3, 1),
            min_relevance_score=_read_float(
                "CMHA_MIN_RELEVANCE_SCORE",
                0.25,
                -1.0,
                1.0,
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_history_messages=_read_int(
                "CMHA_MAX_HISTORY_MESSAGES",
                4,
                0,
            ),
        )
