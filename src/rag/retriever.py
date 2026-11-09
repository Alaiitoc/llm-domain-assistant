"""Top-k retriever: embed a query, search Qdrant, rebuild ranked chunks.

The query path is: question -> mistral-embed vector -> Qdrant query_points ->
``Hit`` list. Two contracts worth stating because downstream code depends on
them: hits keep Qdrant's rank order (it is the MRR input), and the query is
embedded with the model named in the config, which must match the corpus
(see configs/rag.yaml).
"""

from __future__ import annotations

from pydantic import BaseModel

from clients import mistral_client, qdrant_client
from ingestion.chunk import Chunk
from ingestion.embed import embed_texts
from qdrant_store import from_payload
from rag.config import RagConfig


class Hit(BaseModel):
    """One retrieved chunk with its similarity score, in ranked order.

    Score is Qdrant's cosine similarity (higher = more similar, 1.0 = identical),
    not a distance.
    """

    chunk: Chunk
    score: float


def retrieve(query: str, cfg: RagConfig, top_k: int | None = None) -> list[Hit]:
    """Embed ``query`` and return the top_k most similar chunks, ranked.

    ``top_k`` overrides ``cfg.top_k`` per call (eval wants recall@10 without
    forging a config); ``None`` means config wins.
    """
    limit = top_k or cfg.top_k

    # Same embed path as ingestion: symmetry is the whole comparability story.
    vector = embed_texts([query], mistral_client(), cfg.embedding_model)[0]

    response = qdrant_client(cfg.qdrant_url).query_points(
        collection_name=cfg.collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    # response.points arrives rank-ordered; never re-sort downstream (MRR).
    return [
        Hit(chunk=chunk, score=score)
        for chunk, score in (
            # SDK types point.id as int | str | UUID; our ids are UUID strings.
            from_payload(str(point.id), point.payload, point.score)
            for point in response.points
        )
    ]
