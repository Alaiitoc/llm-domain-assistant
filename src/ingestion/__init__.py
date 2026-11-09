"""Ingestion pipeline: clean -> chunk -> embed -> upsert to Qdrant.

The corpus is local MDX from hashicorp/web-unified-docs, so there is no scrape
step: files are read from disk, cleaned and chunked directly."""
