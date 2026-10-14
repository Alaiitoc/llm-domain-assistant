"""Golden eval set: held-out Q/A pairs written from the corpus before any training."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    """One eval question with its reference answer and relevant chunk ids."""

    question: str
    expected_answer: str
    relevant_chunk_ids: set[str] = Field(default_factory=set)


def load_golden_set(path: str) -> list[GoldenExample]:
    """Load the golden jsonl; the set must be disjoint from the training data."""
    raise NotImplementedError
