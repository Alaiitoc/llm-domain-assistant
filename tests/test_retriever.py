"""Unit tests for rag.retriever — stubbed connections, no network, no Qdrant."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rag.retriever as retriever_module
from rag.config import LLMConfig, RagConfig
from rag.retriever import retrieve

CFG = RagConfig(
    collection="test",
    qdrant_url="http://qdrant:6333",
    embedding_model="mistral-embed",
    top_k=5,
    llm=LLMConfig(base_url="http://ollama:11434/v1", model="mistral:7b"),
)


def _point(point_id: str, text: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id, payload={"text": text, "source_url": f"url/{point_id}"}, score=score
    )


class StubQdrant:
    """Records the query_points call; returns fixed points."""

    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.calls: list[dict[str, object]] = []

    def query_points(
        self, collection_name: str, query: list[float], limit: int, with_payload: bool
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "collection": collection_name,
                "limit": limit,
                "with_payload": with_payload,
                "query_len": len(query),
            }
        )
        return SimpleNamespace(points=self.points[:limit])


class StubEmbed:
    """Records the model each embedding call was made with."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, inputs: list[str]) -> SimpleNamespace:
        self.models.append(model)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * 1024) for _ in inputs])


@pytest.fixture
def stub_connections(monkeypatch: pytest.MonkeyPatch) -> tuple[StubEmbed, StubQdrant]:
    """Swap the client factories for stubs; embed_texts runs against a fake endpoint."""
    embed = StubEmbed()
    qdrant = StubQdrant(
        [_point("a", "first", 0.9), _point("b", "second", 0.8), _point("c", "third", 0.7)]
    )
    monkeypatch.setattr(retriever_module, "mistral_client", lambda: embed)
    monkeypatch.setattr(retriever_module, "qdrant_client", lambda url: qdrant)
    return embed, qdrant


class TestRetrieve:
    def test_returns_hits_in_rank_order(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        hits = retrieve("query", CFG)
        assert [h.chunk.source_url for h in hits] == ["url/a", "url/b", "url/c"]
        assert [h.score for h in hits] == [0.9, 0.8, 0.7]

    def test_top_k_override_respects_flag_and_limit(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        _, qdrant = stub_connections
        retrieve("query", CFG, top_k=1)
        assert qdrant.calls[0]["limit"] == 1  # override wins over cfg.top_k

    def test_top_k_zero_is_rejected_loudly(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        # Explicit 0 must not silently mean "config wins"; it is a caller bug.
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            retrieve("query", CFG, top_k=0)

    def test_config_top_k_used_when_no_override(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        _, qdrant = stub_connections
        retrieve("query", CFG)  # CFG.top_k == 5
        assert qdrant.calls[0]["limit"] == 5

    def test_query_embedded_with_configured_model(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        embed, qdrant = stub_connections
        assert qdrant.calls == []  # nothing before retrieve runs
        retrieve("query", CFG)
        assert qdrant.calls[0]["collection"] == "test"
        assert qdrant.calls[0]["query_len"] == 1024  # vector dims flow through
        # The comparability contract: the query is embedded with the configured
        # model — not just any model, not just any dimensionality.
        assert embed.models == [CFG.embedding_model]

    def test_chunks_rebuilt_from_payload(
        self, stub_connections: tuple[StubEmbed, StubQdrant]
    ) -> None:
        hits = retrieve("query", CFG)
        assert hits[0].chunk.text == "first"
        assert hits[0].chunk.id == "a"
