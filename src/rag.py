"""Safety-first retrieval-augmented response pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.generator import LocalGenerator
from src.index import LocalVectorIndex, NoKnowledgeBaseError, SearchResult
from src.safety import route_for_safety

NO_KNOWLEDGE_BASE_RESPONSE = """
The local knowledge base is not ready, so I can’t give a source-grounded
answer. Add trusted PDF, Markdown, or text files to `data/`, then run
`python -m src.ingest`.
""".strip()

LOW_CONFIDENCE_RESPONSE = """
I couldn’t find sufficiently relevant information in the local knowledge base
to answer that safely. Try rephrasing the question or consult a qualified
health professional for guidance specific to you.
""".strip()


@dataclass(frozen=True)
class Answer:
    """A response plus transparent retrieval metadata."""

    text: str
    sources: tuple[str, ...] = ()
    safety_category: str | None = None

    def to_markdown(self) -> str:
        if not self.sources:
            return self.text
        source_lines = "\n".join(f"- `{source}`" for source in self.sources)
        return f"{self.text}\n\n---\n**Local sources retrieved**\n{source_lines}"


class RAGAssistant:
    """Coordinate deterministic safety, retrieval, and local generation."""

    def __init__(
        self,
        settings: Settings,
        vector_index: LocalVectorIndex | None = None,
        generator: LocalGenerator | None = None,
    ):
        self.settings = settings
        self.vector_index = vector_index or LocalVectorIndex(settings)
        self.generator = generator or LocalGenerator(settings)

    def answer(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> Answer:
        clean_question = " ".join(question.split()).strip()
        safety_route = route_for_safety(clean_question)
        if safety_route:
            return Answer(
                text=safety_route.response,
                safety_category=safety_route.category,
            )

        try:
            self.vector_index.ensure_current()
        except NoKnowledgeBaseError:
            return Answer(text=NO_KNOWLEDGE_BASE_RESPONSE)

        results = self.vector_index.search(clean_question, self.settings.top_k)
        relevant = [
            result
            for result in results
            if result.score >= self.settings.min_relevance_score
        ]
        if not relevant:
            return Answer(text=LOW_CONFIDENCE_RESPONSE)

        context = self._format_context(relevant)
        recent_history = self._format_history(history or ())
        generated = self.generator.generate(
            question=clean_question,
            context=context,
            history=recent_history,
        )
        if not generated:
            return Answer(text=LOW_CONFIDENCE_RESPONSE)

        return Answer(
            text=generated,
            sources=self._source_labels(relevant),
        )

    @staticmethod
    def _format_context(results: Sequence[SearchResult]) -> str:
        sections: list[str] = []
        # Keep retrieved context bounded for responsive laptop-scale models and
        # leave room for the instructions, question, and recent history.
        character_budget = 1_200
        for number, result in enumerate(results, start=1):
            label = result.source
            if result.page is not None:
                label += f", page {result.page}"
            section = f"[{number}] Source: {label}\n{result.text}"
            remaining = character_budget - sum(len(item) for item in sections)
            if remaining <= 0:
                break
            sections.append(section[:remaining])
        return "\n\n".join(sections)

    def _format_history(self, history: Sequence[dict[str, Any]]) -> str:
        if self.settings.max_history_messages == 0:
            return ""
        lines: list[str] = []
        for message in history[-self.settings.max_history_messages :]:
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            compact = " ".join(content.split())[:500]
            lines.append(f"{role}: {compact}")
        return "\n".join(lines)

    @staticmethod
    def _source_labels(results: Sequence[SearchResult]) -> tuple[str, ...]:
        labels: list[str] = []
        for result in results:
            label = result.source
            if result.page is not None:
                label += f" (page {result.page})"
            if label not in labels:
                labels.append(label)
        return tuple(labels)
