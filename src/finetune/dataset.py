"""Dataset builder: synthetic Q/A pairs generated from the corpus, deduplicated and filtered."""

from __future__ import annotations


def build_dataset(corpus_path: str, output_path: str) -> int:
    """Build the training jsonl at ``output_path``; returns the number of kept pairs.

    Pairs overlapping the golden eval set (src/eval/golden.py) must be filtered out
    to avoid train/eval leakage.
    """
    raise NotImplementedError
