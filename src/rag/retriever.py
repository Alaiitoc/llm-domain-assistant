"""Top-k retriever over the Qdrant collection."""

from __future__ import annotations

from ingestion.chunk import Chunk


def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    """Embed ``query`` and return the ``top_k`` most similar chunks from Qdrant."""
    raise NotImplementedError
