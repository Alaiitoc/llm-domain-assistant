"""CLI entry point for the full ingestion pipeline (``make ingest``).

Walks the configured sources, maps each file to its live docs URL, then runs
clean -> chunk -> embed -> upsert. Embedding happens once for the whole
corpus (single collection recreate), not per file.

``--dry-run`` runs clean+chunk and prints counts without any API calls or
Qdrant writes - the cheap way to sanity-check wiring and chunk statistics.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import yaml
from pydantic import BaseModel

from ingestion.chunk import Chunk, chunk
from ingestion.clean import clean
from ingestion.embed import embed_and_upsert

BASE_URL = "https://developer.hashicorp.com/terraform"


class IngestionConfig(BaseModel):
    """Typed view of configs/ingestion.yaml; wrong keys fail at load time."""

    sources: list[str]
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model: str = "mistral-embed"
    embedding_dim: int = 1024
    collection: str = "terraform_docs"
    qdrant_url: str = "http://localhost:6333"


def load_config(path: str) -> IngestionConfig:
    """Load the ingestion YAML into the typed config; missing keys fail fast."""
    with open(path, encoding="utf-8") as f:
        return IngestionConfig.model_validate(yaml.safe_load(f))


def source_url_for(path: pathlib.Path, docs_root: pathlib.Path) -> str:
    """Map a corpus path to the live docs URL it renders to.

    ``docs/language/index.mdx`` -> .../terraform/language
    ``docs/language/syntax/configuration.mdx`` -> .../terraform/language/syntax/configuration
    """
    relative = path.relative_to(docs_root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    return f"{BASE_URL}/{'/'.join(parts)}"


def discover_files(sources: list[str]) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Return (file, docs_root) pairs from the configured sources, sorted.

    ``partials/`` directories are skipped: they are include fragments rendered
    inside other pages, not standalone documents with their own URL.
    """
    found: list[tuple[pathlib.Path, pathlib.Path]] = []
    for source in sources:
        root = pathlib.Path(source)
        if not root.is_dir():
            raise SystemExit(f"source directory not found: {source}")
        for path in sorted(root.rglob("*.mdx")):
            if "partials" in path.relative_to(root).parts[:-1]:
                continue
            found.append((path, root))
    return found


def build_chunks(
    files: list[tuple[pathlib.Path, pathlib.Path]], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Clean and chunk every file, carrying frontmatter + provenance metadata."""
    all_chunks: list[Chunk] = []
    for path, docs_root in files:
        doc = clean(str(path))
        metadata = {
            **doc.metadata,
            "version": path.relative_to(docs_root).parts[0],
        }
        all_chunks.extend(
            chunk(doc.text, source_url_for(path, docs_root), chunk_size, overlap, metadata)
        )
    return all_chunks


def main(argv: list[str] | None = None) -> int:
    """Run clean -> chunk -> embed -> upsert for the configured sources."""
    parser = argparse.ArgumentParser(description="Build the Qdrant vector index from docs")
    parser.add_argument("--config", default="configs/ingestion.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N files")
    parser.add_argument(
        "--dry-run", action="store_true", help="clean + chunk only; no embedding, no Qdrant"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    started = time.time()

    files = discover_files(config.sources)
    if args.limit is not None:
        files = files[: args.limit]
    print(f"discovered {len(files)} files in {len(config.sources)} source(s)")

    chunks = build_chunks(files, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise SystemExit("no chunks produced; check the sources config")
    chars = sum(len(c.text) for c in chunks)
    longest = max(len(c.text) for c in chunks)
    print(
        f"chunked into {len(chunks)} chunks "
        f"({chars / max(len(chunks), 1):.0f} chars avg, {longest} max, {chars} total)"
    )

    if args.dry_run:
        print("dry run: skipping embedding and Qdrant writes")
        return 0

    written = embed_and_upsert(
        chunks,
        config.collection,
        qdrant_url=config.qdrant_url,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
    )
    print(f"upserted {written} points into '{config.collection}' in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
