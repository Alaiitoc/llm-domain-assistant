"""Unit tests for rag.cli — the ask flow with stubbed retrieve/generate."""

from __future__ import annotations

import httpx
import pytest

import rag.cli as cli
from ingestion.chunk import Chunk
from rag.config import QUESTION_MAX_CHARS, LLMConfig, RagConfig
from rag.generator import Answer
from rag.retriever import Hit

CFG = RagConfig(
    collection="test",
    qdrant_url="http://localhost:6333",
    embedding_model="mistral-embed",
    top_k=5,
    llm=LLMConfig(base_url="http://localhost:11434/v1", model="mistral:7b"),
)


def _hit(n: int) -> Hit:
    chunk = Chunk(id=f"id{n}", text=f"chunk {n}", source_url=f"https://example.com/{n}")
    return Hit(chunk=chunk, score=0.9 - n * 0.1)


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the two pipeline stages; ask() itself is what's under test."""
    monkeypatch.setattr(cli, "retrieve", lambda question, cfg, top_k=None: [_hit(1)])
    monkeypatch.setattr(
        cli, "generate", lambda question, chunks, llm: Answer(text="stub answer", sources=["s"])
    )


class TestAsk:
    def test_prints_answer_sources_and_timing(
        self, stubbed: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.ask("any question?", CFG)
        out = capsys.readouterr().out
        assert code == 0
        assert "stub answer" in out
        assert "source: s" in out
        assert "answered in" in out
        assert "retrieval " in out and "generation " in out


class TestFmtDuration:
    def test_seconds_under_a_minute(self) -> None:
        assert cli._fmt_duration(1.234) == "1.2s"

    def test_minutes_above_a_minute(self) -> None:
        assert cli._fmt_duration(723) == "12m 03s"


class TestCliBoundary:
    def test_overlong_question_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = cli.ask("q" * (QUESTION_MAX_CHARS + 1), CFG)
        captured = capsys.readouterr()
        assert code == 1
        assert "question too long" in captured.err

    def test_generation_failure_is_one_line_not_a_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(cli, "retrieve", lambda question, cfg, top_k=None: [_hit(1)])
        monkeypatch.setattr(
            cli,
            "generate",
            lambda question, chunks, llm: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")),
        )
        code = cli.ask("q?", CFG)
        captured = capsys.readouterr()
        assert code == 1
        assert "generation failed" in captured.err
        assert "Ollama" in captured.err

    def test_missing_key_runtime_error_is_friendly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def boom(question: str, cfg: object, top_k: int | None = None) -> list[Hit]:
            raise RuntimeError("MISTRAL_API_KEY is not set")

        monkeypatch.setattr(cli, "retrieve", boom)
        code = cli.ask("q?", CFG)
        captured = capsys.readouterr()
        assert code == 1
        assert "MISTRAL_API_KEY" in captured.err
