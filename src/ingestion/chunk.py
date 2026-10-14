"""Split cleaned documents into overlapping chunks with source provenance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of documentation, carried into prompts and evals."""

    id: str
    text: str
    source_url: str
    metadata: dict[str, str] = Field(default_factory=dict)


def chunk(
    text: str,
    source_url: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks, stable ids derived from ``source_url``."""
    raise NotImplementedError
