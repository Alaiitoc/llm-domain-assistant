"""Unit tests for qdrant_store — the payload schema contract."""

from __future__ import annotations

from ingestion.chunk import Chunk
from qdrant_store import RESERVED_KEYS, from_payload, to_payload


def _chunk() -> Chunk:
    return Chunk(
        id="some-uuid",
        text="hello",
        source_url="https://example.com/x",
        metadata={"section": "language", "version": "v1.16.x"},
    )


class TestPayloadRoundTrip:
    """to_payload -> from_payload rebuilds the chunk: the schema contract."""

    def test_round_trip_rebuilds_chunk(self) -> None:
        chunk = _chunk()
        rebuilt, score = from_payload(chunk.id, to_payload(chunk, "mistral-embed"), 0.87)
        assert rebuilt == chunk
        assert score == 0.87

    def test_reserved_keys_win_over_metadata_on_write(self) -> None:
        fat_chunk = Chunk(
            id="x",
            text="real text",
            source_url="https://example.com/real",
            metadata={"text": "user text", "source_url": "user url", "note": "kept"},
        )
        payload = to_payload(fat_chunk, "mistral-embed")
        assert payload["text"] == "real text"
        assert payload["source_url"] == "https://example.com/real"
        assert payload["embedding_model"] == "mistral-embed"
        assert payload["note"] == "kept"  # non-reserved metadata survives

    def test_reserved_keys_listed_matches_schema(self) -> None:
        # If a reserved key is added to the schema, both halves must know.
        assert set(RESERVED_KEYS) == {"text", "source_url", "embedding_model"}

    def test_embedding_model_dropped_on_read(self) -> None:
        chunk = _chunk()
        rebuilt, score = from_payload(chunk.id, to_payload(chunk, "mistral-embed"), 0.5)
        assert "embedding_model" not in rebuilt.metadata  # index provenance, not chunk metadata
        assert score == 0.5

    def test_from_payload_tolerates_missing_payload(self) -> None:
        rebuilt, score = from_payload("id", None)
        assert rebuilt.text == ""
        assert rebuilt.source_url == ""
        assert rebuilt.metadata == {}
        assert score == 0.0
