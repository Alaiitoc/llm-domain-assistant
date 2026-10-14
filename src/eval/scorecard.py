"""Scorecard: aggregate metrics and compare them against the gates in configs/eval.yaml."""

from __future__ import annotations

from pydantic import BaseModel

FIELDS = ("recall_at_k", "mrr", "faithfulness", "relevance")


class Scorecard(BaseModel):
    """The metrics printed as the scorecard and gated in CI."""

    recall_at_k: float
    mrr: float
    faithfulness: float
    relevance: float


def check_gates(scorecard: Scorecard, gates: Scorecard) -> list[str]:
    """Return a human-readable violation for every metric below its gate."""
    return [
        f"{field}={getattr(scorecard, field):.3f} is below gate {getattr(gates, field):.3f}"
        for field in FIELDS
        if getattr(scorecard, field) < getattr(gates, field)
    ]
