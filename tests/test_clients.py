"""Unit tests for clients — construction only, no network."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from clients import mistral_client, qdrant_client


class TestMistralClient:
    def test_missing_api_key_is_a_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            mistral_client()


class TestQdrantClient:
    def test_builds_client_for_url(self) -> None:
        client = qdrant_client("http://localhost:6333")
        assert isinstance(client, QdrantClient)
