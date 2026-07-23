"""Command-line entry point for building the local knowledge index."""

from __future__ import annotations

import argparse

from src.config import Settings
from src.index import LocalVectorIndex, NoKnowledgeBaseError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build C-MHA's local FAISS knowledge index.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the source fingerprint has not changed",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    index = LocalVectorIndex(settings)
    try:
        count = index.ensure_current(force=args.force)
    except NoKnowledgeBaseError as exc:
        parser.error(str(exc))
    print(f"Indexed {count} chunks in {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
