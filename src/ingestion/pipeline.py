"""CLI entry point for the full ingestion pipeline (``make ingest``)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Run scrape -> clean -> chunk -> embed -> upsert for the configured sources."""
    parser = argparse.ArgumentParser(description="Build the Qdrant vector index from docs")
    parser.add_argument("--config", default="configs/ingestion.yaml")
    parser.parse_args(argv)
    raise NotImplementedError("Ingestion pipeline not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
