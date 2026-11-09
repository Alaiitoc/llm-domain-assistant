"""CLI for ``make ask``: retrieve chunks and generate a cited answer.

This is the config boundary: yaml is loaded HERE and passed downward, so
``retrieve`` and ``generate`` stay pure functions of their arguments. It is
also the error boundary: expected operational failures (Ollama down, key
missing, embedding API erroring) print one line and exit 1; anything else is
a bug and keeps its traceback.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx
from mistralai.client.errors.sdkerror import SDKError

from rag.config import QUESTION_MAX_CHARS, RagConfig, load_rag_config
from rag.generator import Answer, generate
from rag.retriever import Hit, retrieve


def _fmt_duration(seconds: float) -> str:
    """Humanize durations: <60s keeps one decimal, longer becomes Xm YYs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes}m {remaining:02d}s"


def _print_hits(hits: list[Hit]) -> None:
    """Show what was retrieved: rank, score, source, and a text snippet."""
    if not hits:
        print("no chunks retrieved")
        return
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text[:80].replace("\n", " ")
        print(f"[{rank}] score={hit.score:.3f}  {hit.chunk.source_url}")
        print(f"    {snippet}")


def ask(question: str, cfg: RagConfig, top_k: int | None = None) -> int:
    """Retrieve, then generate; returns the exit code."""
    if len(question) > QUESTION_MAX_CHARS:
        print(
            f"question too long: {len(question)} chars (max {QUESTION_MAX_CHARS})",
            file=sys.stderr,
        )
        return 1

    try:
        started = time.perf_counter()
        hits = retrieve(question, cfg, top_k=top_k)
        retrieved_at = time.perf_counter()
        _print_hits(hits)

        answer: Answer = generate(question, [hit.chunk for hit in hits], cfg.llm)
        finished = time.perf_counter()
    except httpx.HTTPError as exc:
        print(
            f"generation failed: {exc} — is Ollama up on {cfg.llm.base_url}?",
            file=sys.stderr,
        )
        return 1
    except SDKError as exc:
        print(f"embedding API error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n{answer.text}\n")
    for source in answer.sources:
        print(f"source: {source}")
    print(
        f"\nanswered in {_fmt_duration(finished - started)} "
        f"(retrieval {_fmt_duration(retrieved_at - started)}, "
        f"generation {_fmt_duration(finished - retrieved_at)})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Handle the ``ask`` subcommand: config -> retrieve -> generate -> print."""
    parser = argparse.ArgumentParser(prog="rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Ask the assistant a question")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=None, help="override cfg.top_k")
    ask_parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args(argv)

    cfg = load_rag_config(args.config)
    return ask(args.question, cfg, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
