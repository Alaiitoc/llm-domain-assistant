"""Integration test: the golden set's page labels must resolve to real corpus files.

Labels are page-level docs URLs (the inverse of ingestion.pipeline.source_url_for).
The corpus and data/golden.jsonl are both gitignored-or-absent on CI, so this
skips with a clear reason wherever either is missing.
"""

from __future__ import annotations

import pathlib

import pytest

from eval.golden import GoldenExample, load_golden_set

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data/raw/terraform-docs-repo/content/terraform/v1.16.x/docs"
GOLDEN_PATH = REPO_ROOT / "data/golden.jsonl"
BASE_URL = "https://developer.hashicorp.com/terraform"


@pytest.fixture(scope="module")
def golden() -> list[GoldenExample]:
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"golden set not present: {GOLDEN_PATH} (write it before gating on it)")
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"docs corpus not present: {CORPUS_DIR}")
    return load_golden_set(str(GOLDEN_PATH))


def url_to_corpus_candidates(url: str) -> list[pathlib.Path]:
    """Inverse of source_url_for: a page URL may resolve to page.mdx or page/index.mdx."""
    assert url.startswith(f"{BASE_URL}/"), url
    parts = url[len(f"{BASE_URL}/") :].split("/")
    return [
        CORPUS_DIR.joinpath(*parts).with_suffix(".mdx"),
        CORPUS_DIR.joinpath(*parts, "index.mdx"),
    ]


class TestGoldenLabels:
    def test_size_is_exactly_fifty(self, golden: list[GoldenExample]) -> None:
        assert len(golden) == 50

    def test_unanswerable_mix_is_present(self, golden: list[GoldenExample]) -> None:
        mix = sum(1 for example in golden if not example.answerable)
        assert 1 <= mix <= 5  # corpus-gap questions, a small deliberate slice

    def test_every_labelled_page_exists_in_the_corpus(self, golden: list[GoldenExample]) -> None:
        for example in golden:
            for url in example.relevant_sources:
                candidates = url_to_corpus_candidates(url)
                assert any(c.is_file() for c in candidates), (url, example.question)

    def test_questions_are_unique(self, golden: list[GoldenExample]) -> None:
        questions = [example.question for example in golden]
        assert len(set(questions)) == len(questions)

    def test_overlapping_labels_share_a_page_not_a_label(self, golden: list[GoldenExample]) -> None:
        # Two questions about the same page are fine; identical label sets on
        # many questions usually mean one lazy labeling spree.
        from collections import Counter

        label_sets = Counter(tuple(sorted(e.relevant_sources)) for e in golden)
        assert label_sets.most_common(1)[0][1] <= 3
