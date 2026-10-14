"""Generate an answer from retrieved chunks via the configured LLM endpoint."""

from __future__ import annotations

from pydantic import BaseModel

from ingestion.chunk import Chunk


class Answer(BaseModel):
    """A grounded answer with the chunk ids it cites."""

    text: str
    sources: list[str] = []


def generate(question: str, chunks: list[Chunk]) -> Answer:
    """Fill the answer prompt with ``chunks`` and call the LLM endpoint (configs/rag.yaml)."""
    raise NotImplementedError
