"""Unit tests for ingestion.embed — pure logic only, no network."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from mistralai.client.errors.sdkerror import SDKError

import ingestion.embed as embed
from ingestion.chunk import Chunk
from ingestion.embed import build_points, embed_texts


def _rate_limit_error() -> SDKError:
    # A real httpx.Response: the SDK error constructor reads .text, .headers, ...
    response = httpx.Response(429, json={"message": "rate limited"})
    return SDKError("429 Too Many Requests", raw_response=response)


def _server_error() -> SDKError:
    response = httpx.Response(500, json={"message": "boom"})
    return SDKError("500 Something broke", raw_response=response)


class StubMistral:
    """Minimal stand-in for the Mistral client's embeddings endpoint."""

    def __init__(self, errors: list[SDKError] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.errors = list(errors or [])

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class _Embeddings:
    def __init__(self, owner: StubMistral) -> None:
        self._owner = owner

    def create(self, model: str, inputs: list[str]) -> SimpleNamespace:
        self._owner.calls.append(list(inputs))
        if self._owner.errors:
            raise self._owner.errors.pop(0)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in inputs])


def _stub(errors: list[SDKError] | None = None) -> StubMistral:
    stub = StubMistral(errors)
    stub.embeddings = _Embeddings(stub)  # type: ignore[attr-defined]
    return stub


class TestEmbedTexts:
    def test_batches_and_preserves_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(embed.time, "sleep", lambda _: None)
        stub = _stub()
        vectors = embed_texts(["a", "b", "c", "d"], stub, batch_size=2)  # type: ignore[arg-type]
        assert stub.calls == [["a", "b"], ["c", "d"]]
        assert len(vectors) == 4
        assert all(v == [0.1, 0.2, 0.3] for v in vectors)

    def test_retries_rate_limits_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(embed.time, "sleep", sleeps.append)
        stub = _stub(errors=[_rate_limit_error(), _rate_limit_error()])
        vectors = embed_texts(["chunk"], stub)  # type: ignore[arg-type]
        assert len(vectors) == 1
        assert sleeps == [1, 2]  # exponential backoff: 2^0, 2^1

    def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(embed.time, "sleep", lambda _: None)
        stub = _stub(errors=[_rate_limit_error()] * 10)
        with pytest.raises(SDKError):
            embed_texts(["chunk"], stub)  # type: ignore[arg-type]

    def test_does_not_retry_server_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[int] = []
        monkeypatch.setattr(embed.time, "sleep", sleeps.append)
        stub = _stub(errors=[_server_error()])
        with pytest.raises(SDKError):
            embed_texts(["chunk"], stub)  # type: ignore[arg-type]
        assert sleeps == []  # failed immediately, no backoff


def _chunk(cid: str = "abc", **meta: str) -> Chunk:
    return Chunk(id=cid, text="hello", source_url="https://example.com/x", metadata=meta)


class TestBuildPoints:
    def test_payload_merges_metadata_with_reserved_keys(self) -> None:
        chunk = _chunk(metadata_text="user text", section="language")
        points = build_points([chunk], [[0.1, 0.2]], "mistral-embed")
        assert len(points) == 1
        payload = points[0].payload
        assert payload is not None
        assert payload["text"] == "hello"  # reserved key wins over metadata
        assert payload["source_url"] == "https://example.com/x"
        assert payload["embedding_model"] == "mistral-embed"
        assert payload["section"] == "language"  # metadata passthrough

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            build_points([_chunk()], [], "mistral-embed")
