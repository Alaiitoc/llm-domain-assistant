"""Qdrant wire format: the payload schema and collection lifecycle.

``to_payload`` and ``from_payload`` are mirror halves of the point payload
schema and must stay symmetric - the round-trip test pins that contract.
Connection construction lives in clients.py; embedding and search semantics
stay in their own modules.

``embedding_model`` is deliberately asymmetrical: written on every point
(index-level provenance), dropped on read (it describes the vector, not the
chunk).
"""

from __future__ import annotations

from collections.abc import Mapping

from qdrant_client import QdrantClient, models

from ingestion.chunk import Chunk

# Set by the store on write; chunk metadata never overwrites these.
RESERVED_KEYS = ("text", "source_url", "embedding_model")


def to_payload(chunk: Chunk, embedding_model: str) -> dict[str, str]:
    """Merge chunk metadata with the reserved keys; reserved keys win."""
    return {
        **chunk.metadata,
        "text": chunk.text,
        "source_url": chunk.source_url,
        "embedding_model": embedding_model,
    }


def from_payload(
    point_id: str, payload: Mapping[str, str] | None, score: float = 0.0
) -> tuple[Chunk, float]:
    """Rebuild a ``(Chunk, score)`` pair from a stored payload.

    Inverse of :func:`to_payload`: reserved keys go back to their Chunk
    fields (``embedding_model`` is dropped - index provenance, not chunk
    metadata), everything else is metadata again.
    """
    data = dict(payload or {})
    chunk = Chunk(
        id=point_id,
        text=data.pop("text", ""),
        source_url=data.pop("source_url", ""),
        metadata={str(k): str(v) for k, v in data.items() if k != "embedding_model"},
    )
    return chunk, score


def ensure_collection(
    qdrant: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = True,
) -> None:
    """Recreate (default) or create the collection; carries the idempotency decision."""
    exists = qdrant.collection_exists(collection)
    if recreate and exists:
        qdrant.delete_collection(collection)
        exists = False
    if not exists:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
