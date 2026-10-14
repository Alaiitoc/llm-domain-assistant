"""Unit tests for the implemented eval building blocks."""

from __future__ import annotations

from eval.metrics import mrr, recall_at_k
from eval.scorecard import Scorecard, check_gates


class TestRecallAtK:
    def test_full_recall(self) -> None:
        assert recall_at_k({"a", "b"}, ["a", "x", "b", "y"], 5) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k({"a", "b"}, ["a", "x", "y"], 5) == 0.5

    def test_k_truncates(self) -> None:
        assert recall_at_k({"b"}, ["x", "b"], 1) == 0.0
        assert recall_at_k({"b"}, ["x", "b"], 2) == 1.0

    def test_no_relevant_ids(self) -> None:
        assert recall_at_k(set(), ["a"], 5) == 0.0


class TestMrr:
    def test_first_hit(self) -> None:
        assert mrr({"a"}, ["a", "b"]) == 1.0

    def test_second_hit(self) -> None:
        assert mrr({"b"}, ["a", "b"]) == 0.5

    def test_no_hit(self) -> None:
        assert mrr({"z"}, ["a", "b"]) == 0.0


class TestCheckGates:
    def test_all_pass(self) -> None:
        scorecard = Scorecard(recall_at_k=0.8, mrr=0.7, faithfulness=0.9, relevance=0.9)
        gates = Scorecard(recall_at_k=0.6, mrr=0.5, faithfulness=0.7, relevance=0.7)
        assert check_gates(scorecard, gates) == []

    def test_violations_reported(self) -> None:
        scorecard = Scorecard(recall_at_k=0.55, mrr=0.7, faithfulness=0.9, relevance=0.9)
        gates = Scorecard(recall_at_k=0.6, mrr=0.5, faithfulness=0.7, relevance=0.7)
        violations = check_gates(scorecard, gates)
        assert len(violations) == 1
        assert violations[0].startswith("recall_at_k=0.550 is below gate 0.600")
