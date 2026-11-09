"""Embed chunks with mistral-embed and upsert them into Qdrant.

Connections come from clients.py, the payload schema from qdrant_store.py;
this module keeps the embedding semantics: batching, retry, the
embed-then-recreate ordering (a failed API call must never destroy the
existing index), and the ingestion entry point.
"""

from __future__ import annotations

import time

from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from qdrant_client import models

from clients import mistral_client, qdrant_client
from ingestion.chunk import Chunk
from qdrant_store import ensure_collection, to_payload


def _is_rate_limited(err: Exception) -> bool:
    status = getattr(getattr(err, "raw_response", None), "status_code", None)
    return status == 429 or "429" in str(err)


def embed_texts(
    texts: list[str],
    client: Mistral,
    model: str = "mistral-embed",
    batch_size: int = 64,
    max_retries: int = 4,
) -> list[list[float]]:
    """Embed ``texts`` in batches, retrying rate limits with backoff.

    ``client`` is injected (a Mistral client from clients.py) so tests can
    pass a stub; no network here.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, inputs=batch)
                for item in response.data:
                    if item.embedding is None:  # SDK types it Optional
                        raise RuntimeError("Mistral returned a null embedding")
                    vectors.append(item.embedding)
                break
            except SDKError as err:
                if not _is_rate_limited(err) or attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding count mismatch: {len(vectors)} != {len(texts)}")
    return vectors


def build_points(
    chunks: list[Chunk],
    vectors: list[list[float]],
    embedding_model: str,
) -> list[models.PointStruct]:
    """Zip chunks and vectors into Qdrant points via the store payload schema."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks/vectors mismatch: {len(chunks)} != {len(vectors)}")
    return [
        models.PointStruct(id=chunk.id, vector=vector, payload=to_payload(chunk, embedding_model))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def embed_and_upsert(
    chunks: list[Chunk],
    collection: str,
    *,
    qdrant_url: str = "http://localhost:6333",
    embedding_model: str = "mistral-embed",
    embedding_dim: int = 1024,
    batch_size: int = 64,
    recreate_collection: bool = True,
) -> int:
    """Embed ``chunks`` with the configured model and upsert to Qdrant; returns points written."""
    if not chunks:
        return 0
    vectors = embed_texts([c.text for c in chunks], mistral_client(), embedding_model, batch_size)
    points = build_points(chunks, vectors, embedding_model)

    qdrant = qdrant_client(qdrant_url)
    ensure_collection(qdrant, collection, embedding_dim, recreate_collection)
    upsert_batch = 256
    for start in range(0, len(points), upsert_batch):
        qdrant.upsert(collection_name=collection, points=points[start : start + upsert_batch])
    return len(points)
