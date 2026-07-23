from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import Settings
from src.index import SearchResult
from src.rag import LOW_CONFIDENCE_RESPONSE, RAGAssistant


def settings_for(directory: str) -> Settings:
    return Settings(
        data_dir=Path(directory) / "data",
        index_dir=Path(directory) / "index",
        embedding_model="fake-embedding",
        embedding_revision="fake-embedding-revision",
        generation_model="fake-generator",
        generation_revision="fake-generation-revision",
        device=-1,
        top_k=3,
        min_relevance_score=0.25,
        chunk_size=700,
        chunk_overlap=100,
        max_history_messages=4,
    )


class FakeIndex:
    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.ensure_calls = 0
        self.search_calls = 0

    def ensure_current(self, force: bool = False) -> int:
        self.ensure_calls += 1
        return len(self.results)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.search_calls += 1
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def generate(self, question: str, context: str, history: str) -> str:
        self.calls.append(
            {"question": question, "context": context, "history": history}
        )
        return "Grounded response."


class RAGTests(unittest.TestCase):
    def test_grounded_answer_lists_deduplicated_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(
                [
                    SearchResult("Relevant text.", "guide.pdf", 2, 0.81),
                    SearchResult("More text.", "guide.pdf", 2, 0.63),
                ]
            )
            generator = FakeGenerator()
            assistant = RAGAssistant(
                settings_for(directory),
                vector_index=index,  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
            )

            answer = assistant.answer(
                "What does the guide say?",
                [{"role": "user", "content": "Earlier question"}],
            )

            self.assertEqual(answer.text, "Grounded response.")
            self.assertEqual(answer.sources, ("guide.pdf (page 2)",))
            self.assertEqual(len(generator.calls), 1)
            self.assertIn("Relevant text.", generator.calls[0]["context"])

    def test_low_relevance_does_not_call_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex([SearchResult("Unrelated.", "guide.txt", None, 0.1)])
            generator = FakeGenerator()
            assistant = RAGAssistant(
                settings_for(directory),
                vector_index=index,  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
            )

            answer = assistant.answer("Out-of-scope question")

            self.assertEqual(answer.text, LOW_CONFIDENCE_RESPONSE)
            self.assertEqual(generator.calls, [])

    def test_crisis_route_bypasses_index_and_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex([])
            generator = FakeGenerator()
            assistant = RAGAssistant(
                settings_for(directory),
                vector_index=index,  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
            )

            answer = assistant.answer("I am suicidal")

            self.assertEqual(answer.safety_category, "self_harm")
            self.assertEqual(index.ensure_calls, 0)
            self.assertEqual(index.search_calls, 0)
            self.assertEqual(generator.calls, [])

    def test_violence_route_bypasses_index_and_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex([])
            generator = FakeGenerator()
            assistant = RAGAssistant(
                settings_for(directory),
                vector_index=index,  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
            )

            answer = assistant.answer("I want to hurt someone")

            self.assertEqual(answer.safety_category, "harm_others")
            self.assertEqual(index.ensure_calls, 0)
            self.assertEqual(index.search_calls, 0)
            self.assertEqual(generator.calls, [])

    def test_zero_history_setting_excludes_all_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(directory)
            settings = Settings(
                **{
                    **settings.__dict__,
                    "max_history_messages": 0,
                }
            )
            index = FakeIndex([SearchResult("Relevant.", "guide.txt", None, 0.8)])
            generator = FakeGenerator()
            assistant = RAGAssistant(
                settings,
                vector_index=index,  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
            )

            assistant.answer(
                "Question",
                [{"role": "user", "content": "Private prior message"}],
            )

            self.assertEqual(generator.calls[0]["history"], "")


if __name__ == "__main__":
    unittest.main()
