"""Retrieval metrics: recall@k and MRR over chunk ids."""

from __future__ import annotations


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant chunk ids present in the top ``k`` of ``retrieved``."""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    """Mean reciprocal rank of the first relevant chunk in ``retrieved``."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0
