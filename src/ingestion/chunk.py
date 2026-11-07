"""Split cleaned documents into overlapping chunks with source provenance.

Strategy (option 2, agreed): paragraph-boundary packing where fenced code
blocks are atomic. IDs are uuid5(source_url + chunk index) so a re-ingest of
unchanged content overwrites the same points; corpus freshness is guaranteed
by recreating the collection (see configs/ingestion.yaml), never by
point-level bookkeeping.
"""

from __future__ import annotations

import re
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of documentation, carried into prompts and evals."""

    id: str
    text: str
    source_url: str
    metadata: dict[str, str] = Field(default_factory=dict)


# A fence toggle is a full line of backticks with an optional info string
# (```hcl opens, bare ``` closes). Slang inside a fence that happens to match
# is an accepted simplification: real HCL examples never contain a bare
# ```-only line.
_FENCE_LINE = re.compile(r"^```\w*\s*$")
_HEADING = re.compile(r"^#{1,6}\s")


def split_blocks(text: str) -> list[str]:
    """Split into paragraphs and atomic fenced code blocks.

    A fence line joins the block it opens; everything until the closing fence
    line (or EOF, for unbalanced input) stays a single block.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if in_fence:
            current.append(line)
            if _FENCE_LINE.fullmatch(line):
                in_fence = False
                blocks.append("\n".join(current))
                current = []
        elif _FENCE_LINE.fullmatch(line):
            if current:
                blocks.append("\n".join(current))
                current = []
            in_fence = True
            current = [line]
        elif not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:  # trailing paragraph, or an unclosed fence at EOF
        blocks.append("\n".join(current))
    return [b.strip() for b in blocks if b.strip()]


def pack_blocks(blocks: list[str], max_size: int) -> list[str]:
    """Greedily pack blocks into chunks, never slicing a fence.

    A fence longer than max_size gets its own oversized chunk; a prose block
    larger than max_size raises, because at that size clean() upstream broke,
    not the chunker. Headings pack forward (never end a chunk alone).
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        is_fence = block.startswith("```")
        if not is_fence and len(block) > max_size:
            raise ValueError(f"prose block of {len(block)} chars exceeds max_size={max_size}")
        if is_fence and len(block) > max_size:
            if current:
                chunks.append(current)
                current = []
            chunks.append([block])
            continue
        if current:
            # Keep a block under a heading glued to that heading.
            prev_is_heading = bool(_HEADING.match(current[-1]))
            joined = len("\n\n".join(current)) + 2 + len(block)
            if not prev_is_heading and joined > max_size:
                chunks.append(current)
                current = []
        current.append(block)
    if current:
        chunks.append(current)
    return ["\n\n".join(c).strip() for c in chunks]


def _overlap_tail(prev: str, overlap: int) -> str:
    """Return the tail of prev to prepend as context, up to ~2*overlap chars.

    The tail is prev's last *block* - via split_blocks, because blank lines
    inside code make naive ``\\n\\n`` splitting mistake mid-code for a
    paragraph - and never a fence. If the last block is a fence the block
    before it is used; if there is none, there is no overlap.
    """
    blocks = [b for b in split_blocks(prev) if b.strip()]
    if not blocks:
        return ""
    tail = blocks[-1]
    if tail.startswith("```"):
        tail = blocks[-2] if len(blocks) > 1 else ""
    if not tail or len(tail) <= overlap:
        return tail
    window = tail[len(tail) - 2 * overlap :]
    boundary = re.search(r"[.!?](?:['\"])?\s", window)
    return window[boundary.end() :] if boundary else window


def apply_overlap(chunk_texts: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to every chunk but the first."""
    out: list[str] = []
    for i, text in enumerate(chunk_texts):
        if i == 0 or overlap <= 0:
            out.append(text)
            continue
        tail = _overlap_tail(chunk_texts[i - 1], overlap)
        out.append(f"{tail}\n\n{text}" if tail else text)
    return out


def chunk(
    text: str,
    source_url: str,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks, stable ids derived from ``source_url``."""
    if not text.strip():
        return []
    packed = pack_blocks(split_blocks(text), chunk_size)
    overlapped = apply_overlap(packed, overlap)
    base_meta = metadata or {}
    return [
        Chunk(
            id=str(uuid5(NAMESPACE_URL, f"{source_url}#chunk-{i}")),
            text=chunk_text,
            source_url=source_url,
            metadata=base_meta,
        )
        for i, chunk_text in enumerate(overlapped)
    ]
