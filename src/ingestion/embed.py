"""Embed chunks and upsert them into the Qdrant collection with metadata."""

from __future__ import annotations

from ingestion.chunk import Chunk


def embed_and_upsert(chunks: list[Chunk], collection: str) -> int:
    """Embed ``chunks`` with the configured model and upsert to Qdrant; returns points written."""
    raise NotImplementedError
