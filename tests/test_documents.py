from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.documents import (
    SourceDocument,
    chunk_documents,
    discover_source_files,
    source_fingerprint,
    split_text,
)


class DocumentTests(unittest.TestCase):
    def test_discovery_skips_readme_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.txt").write_text("trusted", encoding="utf-8")
            (root / "README.md").write_text("instructions", encoding="utf-8")
            (root / "image.png").write_bytes(b"not-a-document")

            self.assertEqual(discover_source_files(root), [root / "guide.txt"])

    def test_fingerprint_changes_with_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "guide.txt"
            source.write_text("version one", encoding="utf-8")
            first = source_fingerprint(root)
            source.write_text("version two", encoding="utf-8")

            self.assertNotEqual(first, source_fingerprint(root))

    def test_split_text_preserves_overlap(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = list(split_text(text, chunk_size=30, overlap=8))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertTrue(all(not chunk.startswith("entence") for chunk in chunks))

    def test_chunk_metadata_is_preserved(self) -> None:
        documents = [SourceDocument("A " * 100, "manual.pdf", page=7)]
        chunks = chunk_documents(documents, chunk_size=40, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source == "manual.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.page == 7 for chunk in chunks))
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            list(range(len(chunks))),
        )


if __name__ == "__main__":
    unittest.main()
