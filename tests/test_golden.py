"""Unit tests for eval.golden — schema contract and loader; no corpus needed."""

from __future__ import annotations

import json

import pytest

from eval.golden import GoldenExample, load_golden_set


class TestGoldenExample:
    def test_answerable_question_needs_a_page(self) -> None:
        with pytest.raises(ValueError, match="at least one relevant page"):
            GoldenExample.model_validate(
                {"question": "q?", "expected_answer": "a", "answerable": True}
            )

    def test_unanswerable_question_must_have_no_pages(self) -> None:
        with pytest.raises(ValueError, match="must have no relevant pages"):
            GoldenExample.model_validate(
                {
                    "question": "q?",
                    "expected_answer": "not in docs",
                    "relevant_sources": ["https://developer.hashicorp.com/terraform/x"],
                    "answerable": False,
                }
            )

    def test_valid_examples_pass(self) -> None:
        answerable = GoldenExample.model_validate(
            {"question": "q?", "expected_answer": "a", "relevant_sources": ["u"]}
        )
        gap = GoldenExample.model_validate(
            {"question": "q?", "expected_answer": "not in docs", "answerable": False}
        )
        assert answerable.answerable and gap.relevant_sources == []


class TestLoadGoldenSet:
    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "golden.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return str(path)

    def test_round_trip(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            [
                {
                    "question": "q?",
                    "expected_answer": "a",
                    "relevant_sources": ["u"],
                },
                {"question": "gap?", "expected_answer": "no", "answerable": False},
            ],
        )
        examples = load_golden_set(path)
        assert len(examples) == 2
        assert examples[0].relevant_sources == ["u"]
        assert examples[1].answerable is False

    def test_blank_lines_are_tolerated(self, tmp_path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "\n\n"
            + json.dumps({"question": "q", "expected_answer": "a", "relevant_sources": ["u"]})
            + "\n\n",
            encoding="utf-8",
        )
        assert len(load_golden_set(str(path))) == 1

    def test_invalid_line_names_its_line_number(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            [
                {"question": "ok?", "expected_answer": "a", "relevant_sources": ["u"]},
                {"question": "bad?", "expected_answer": "a", "answerable": True},
            ],
        )
        with pytest.raises(ValueError, match=r"golden.jsonl:2"):
            load_golden_set(path)

    def test_empty_set_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_golden_set(str(path))
