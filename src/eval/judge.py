"""LLM-as-judge: score faithfulness and relevance with the versioned prompt in configs/."""

from __future__ import annotations

from pydantic import BaseModel


class JudgeScores(BaseModel):
    """Judge outputs, each on a 0-1 scale."""

    faithfulness: float
    relevance: float


def judge_answer(question: str, answer: str, context: str) -> JudgeScores:
    """Score ``answer`` against ``context`` with the judge model and prompt from configs/."""
    raise NotImplementedError
