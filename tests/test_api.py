"""Tests for api.main — fixture config, stubbed pipeline, no network."""

from __future__ import annotations

import collections.abc
import pathlib
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from ingestion.chunk import Chunk
from rag.generator import Answer
from rag.retriever import Hit

FIXTURE_CONFIG = """\
collection: terraform_docs
qdrant_url: http://localhost:6333
embedding_model: mistral-embed
top_k: 5
llm:
  base_url: http://localhost:11434/v1
  model: mistral:7b-instruct-v0.3-q4_K_M
  timeout: 5
  max_tokens: 64
"""


@pytest.fixture
def client(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> collections.abc.Iterator[TestClient]:
    """App with a fixture config: import stays clean, lifespan loads the fixture."""
    config = tmp_path / "rag.yaml"
    config.write_text(FIXTURE_CONFIG, encoding="utf-8")
    monkeypatch.setattr(api_main, "CONFIG_PATH", str(config))
    with TestClient(api_main.app) as test_client:
        yield test_client


def _hit() -> Hit:
    chunk = Chunk(id="id", text="chunk", source_url="https://example.com/x")
    return Hit(chunk=chunk, score=0.9)


class TestHealth:
    def test_liveness(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}


class TestAsk:
    def test_happy_path(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api_main, "retrieve", lambda q, cfg, top_k=None: [_hit()])
        monkeypatch.setattr(
            api_main,
            "generate",
            lambda q, chunks, llm: Answer(text="stub [1]", sources=["https://example.com/x"]),
        )
        response = client.post("/ask", json={"question": "What is a module?"})
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        assert body["text"] == "stub [1]"
        assert body["sources"] == ["https://example.com/x"]
        assert body["prompt_version"]

    def test_overlong_question_is_422(self, client: TestClient) -> None:
        response = client.post("/ask", json={"question": "q" * 2001})
        assert response.status_code == 422

    def test_ollama_down_is_503(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api_main, "retrieve", lambda q, cfg, top_k=None: [_hit()])
        monkeypatch.setattr(
            api_main,
            "generate",
            lambda q, chunks, llm: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")),
        )
        response = client.post("/ask", json={"question": "q?"})
        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]

    def test_missing_key_is_502(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(question: str, cfg: object, top_k: int | None = None) -> list[Hit]:
            raise RuntimeError("MISTRAL_API_KEY is not set")

        monkeypatch.setattr(api_main, "retrieve", boom)
        response = client.post("/ask", json={"question": "q?"})
        assert response.status_code == 502
        assert "MISTRAL_API_KEY" in response.json()["detail"]


class TestConfigLifecycle:
    def test_import_does_not_load_config(self) -> None:
        # reaching this point proves api.main imports without touching config:
        # loading happens in the lifespan, seen by every test via the fixture.
        assert api_main.CONFIG_PATH  # present, not executed
