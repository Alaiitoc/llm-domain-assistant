"""Fetch raw documentation pages from the configured sources."""

from __future__ import annotations

from pydantic import BaseModel


class RawDocument(BaseModel):
    """A fetched documentation page, before cleaning."""

    url: str
    html: str
    title: str = ""


def scrape(sources: list[str]) -> list[RawDocument]:
    """Fetch every source URL and return the raw documents."""
    raise NotImplementedError
