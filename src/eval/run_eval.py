"""Eval harness entry point (``make eval``): retrieval metrics + LLM-as-judge, CI-gated."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Run the harness over the golden set; in ``--gate`` mode, exit non-zero on violations."""
    parser = argparse.ArgumentParser(description="Run the eval harness and print the scorecard")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="CI mode: exit 1 if any metric falls below its gate in configs/eval.yaml",
    )
    parser.parse_args(argv)
    raise NotImplementedError("Eval harness not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
