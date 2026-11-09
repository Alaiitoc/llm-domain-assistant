"""Unit tests for rag.generator — stubbed httpx, no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rag.generator as generator_module
from ingestion.chunk import Chunk
from rag.config import LLMConfig
from rag.generator import build_context, build_messages, cited_sources, generate
from rag.prompts import ANSWER_SYSTEM_PROMPT

LLM = LLMConfig(base_url="http://ollama:11434/v1", model="mistral:7b")


def _chunk(n: int) -> Chunk:
    return Chunk(id=f"id{n}", text=f"chunk {n}", source_url=f"https://example.com/{n}")


class TestBuildContext:
    def test_chunks_numbered_in_retrieval_order(self) -> None:
        context = build_context([_chunk(1), _chunk(2)])
        assert context == "[1] chunk 1\n\n[2] chunk 2"


class TestBuildMessages:
    def test_roles_and_question_present(self) -> None:
        messages = build_messages("What is a module?", [_chunk(1)])
        assert messages[0]["role"] == "system"
        assert "Terraform" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "What is a module?" in messages[1]["content"]

    def test_regression_prompt_is_for_terraform(self) -> None:
        """v1 claimed to be a Mistral products assistant on a Terraform corpus."""
        assert "Terraform" in ANSWER_SYSTEM_PROMPT
        assert "Mistral AI products" not in ANSWER_SYSTEM_PROMPT


class TestCitedSources:
    def test_cited_numbers_map_to_chunk_urls(self) -> None:
        chunks = [_chunk(1), _chunk(2), _chunk(3)]
        assert cited_sources("see [2] and also [1].", chunks) == [
            "https://example.com/1",
            "https://example.com/2",
        ]

    def test_duplicates_and_out_of_range_ignored(self) -> None:
        chunks = [_chunk(1), _chunk(2)]
        assert cited_sources("[1] [1] [9]", chunks) == ["https://example.com/1"]

    def test_no_citations_falls_back_to_all_sources(self) -> None:
        chunks = [_chunk(1), _chunk(2)]
        assert cited_sources("no citations here", chunks) == [
            "https://example.com/1",
            "https://example.com/2",
        ]


class TestGenerate:
    def test_uses_llm_config_and_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, object]] = []

        def fake_post(url: str, json: dict, timeout: object) -> SimpleNamespace:
            calls.append({"url": url, "json": json, "timeout": timeout})
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "answer citing [1]"}}]},
            )

        monkeypatch.setattr(generator_module.httpx, "post", fake_post)
        answer = generate("question?", [_chunk(1), _chunk(2)], LLM)

        assert answer.text == "answer citing [1]"
        assert answer.sources == ["https://example.com/1"]
        assert answer.prompt_version  # provenance rides along
        assert calls[0]["url"] == "http://ollama:11434/v1/chat/completions"
        payload = calls[0]["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "mistral:7b"
        assert payload["temperature"] == 0  # reproducible answers for evals
        assert payload["max_tokens"] == 512  # bounded answer, from config

    def test_empty_retrieval_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No httpx call may happen with nothing retrieved.
        def boom(url: str, json: dict, timeout: float) -> SimpleNamespace:
            raise AssertionError("httpx must not be called with no chunks")

        monkeypatch.setattr(generator_module.httpx, "post", boom)
        answer = generate("question?", [], LLM)
        assert answer.sources == []
        assert "No relevant documentation" in answer.text
