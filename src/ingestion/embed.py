"""Embed chunks with mistral-embed and upsert them into Qdrant.

Order of operations is deliberate: embedding happens BEFORE any collection
recreation, so a failed API call never destroys the existing index. Payloads
are flat str-only maps (a Qdrant constraint) and carry the embedding model
name, because an API-backed model can be updated silently and every point
should record what produced its vector.
"""

from __future__ import annotations

import os
import time

from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from qdrant_client import QdrantClient, models

from ingestion.chunk import Chunk


def mistral_client() -> Mistral:
    """Build the Mistral client; the key lives in the env, never in config."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Put it in the environment"
            " (e.g. source .env); config files never hold secrets."
        )
    return Mistral(api_key=api_key)


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

    ``client`` is injected so tests can pass a stub; no network here.
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
    """Zip chunks and vectors into Qdrant points with provenance payloads."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks/vectors mismatch: {len(chunks)} != {len(vectors)}")
    return [
        models.PointStruct(
            id=chunk.id,
            vector=vector,
            payload={
                **chunk.metadata,
                "text": chunk.text,
                "source_url": chunk.source_url,
                "embedding_model": embedding_model,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


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

    qdrant = QdrantClient(url=qdrant_url)
    ensure_collection(qdrant, collection, embedding_dim, recreate_collection)
    upsert_batch = 256
    for start in range(0, len(points), upsert_batch):
        qdrant.upsert(collection_name=collection, points=points[start : start + upsert_batch])
    return len(points)
