"""Golden eval set: held-out Q/A pairs written from the corpus before any training.

Labels are deliberately **page-level** (``relevant_sources`` holds docs URLs, not
chunk ids): chunk ids are derived (uuid5 of url#index) and shift on every
chunker tweak, while page URLs survive re-ingests and chunker experiments.
Retrieval metrics therefore operate on the page URLs of retrieved chunks.

``answerable: False`` marks corpus-gap questions — the answer is intentionally
not in the docs — so the harness can score "admits it" separately from
retrieval misses.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel, model_validator


class GoldenExample(BaseModel):
    """One eval question with its reference answer and relevant docs pages."""

    question: str
    expected_answer: str
    relevant_sources: list[str] = []
    answerable: bool = True

    @model_validator(mode="after")
    def _labels_match_answerability(self) -> GoldenExample:
        """Answerable questions must cite pages; corpus-gap ones must not."""
        if self.answerable and not self.relevant_sources:
            raise ValueError(
                f"answerable question needs at least one relevant page: {self.question!r}"
            )
        if not self.answerable and self.relevant_sources:
            raise ValueError(
                f"unanswerable question must have no relevant pages: {self.question!r}"
            )
        return self


def load_golden_set(path: str) -> list[GoldenExample]:
    """Load the golden jsonl; the set must be disjoint from the training data."""
    examples: list[GoldenExample] = []
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            examples.append(GoldenExample.model_validate(raw))
        except ValueError as err:
            raise ValueError(f"{path}:{number}: invalid golden example: {err}") from err
    if not examples:
        raise ValueError(f"golden set is empty: {path}")
    return examples
