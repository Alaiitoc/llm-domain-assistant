"""CLI for ``make ask``: retrieve chunks and generate a cited answer."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Handle the ``ask`` subcommand: print the answer and its sources."""
    parser = argparse.ArgumentParser(prog="rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Ask the assistant a question")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    # Wire rag.retriever.retrieve + rag.generator.generate here.
    print(f"Not implemented yet: retriever/generator are stubs (question: {args.question})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
