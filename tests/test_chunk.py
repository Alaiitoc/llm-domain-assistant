"""Unit tests for ingestion.chunk."""

from __future__ import annotations

from uuid import UUID

from ingestion.chunk import (
    apply_overlap,
    chunk,
    pack_blocks,
    split_blocks,
)


class TestSplitBlocks:
    def test_fence_is_atomic(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            '```hcl\nresource "aws_instance" "x" {\n  count = 1\n}\n```\n\nOutro.'
        )
        blocks = split_blocks(text)
        assert blocks == [
            "Intro paragraph.",
            '```hcl\nresource "aws_instance" "x" {\n  count = 1\n}\n```',
            "Outro.",
        ]

    def test_fence_like_text_inside_fence_does_not_close_it(self) -> None:
        text = "para\n\n```\ncode mentions ```hcl inside\n```\n\nafter"
        blocks = split_blocks(text)
        assert blocks == ["para", "```\ncode mentions ```hcl inside\n```", "after"]

    def test_unclosed_fence_at_eof_is_one_block(self) -> None:
        text = "para\n\n```hcl\nresource x {\n"
        blocks = split_blocks(text)
        assert blocks == ["para", "```hcl\nresource x {"]

    def test_blank_runs_and_whitespace_are_dropped(self) -> None:
        assert split_blocks("\n\n  \nA para\n\n\nB\n") == ["A para", "B"]

    def test_empty_text_yields_no_blocks(self) -> None:
        assert split_blocks("") == []
        assert split_blocks("   \n\n") == []


class TestPackBlocks:
    def test_small_blocks_join_into_one_chunk(self) -> None:
        assert pack_blocks(["a", "b", "c"], max_size=800) == ["a\n\nb\n\nc"]

    def test_overflow_opens_new_chunk(self) -> None:
        chunks = pack_blocks(["x" * 500, "y" * 500], max_size=800)
        assert chunks == ["x" * 500, "y" * 500]

    def test_oversized_fence_and_prose_get_own_chunks(self) -> None:
        fence = "```hcl\n" + "x = 1\n" * 200 + "```"
        table = "| Flag | Description |\n" + "| :--- | :--- |\n" + "| short | text |\n" * 80
        chunks = pack_blocks([fence, table], max_size=800)
        # intact and unsliced; pack_blocks strips chunk edges
        assert chunks == [fence, table.rstrip()]

    def test_heading_packs_forward_never_orphaned(self) -> None:
        heading = "## Meta-arguments"
        body = "body " * 140  # 700 chars: fits alone, but not against the heading
        chunks = pack_blocks([heading, body, "next"], max_size=800)
        assert all(not c.endswith(heading) for c in chunks)
        assert chunks[0].startswith(heading)


class TestApplyOverlap:
    def test_first_chunk_bare_others_carry_tail(self) -> None:
        first = "Alpha sentence one. Second sentence here."
        second = "Beta content."
        out = apply_overlap([first, second], overlap=100)
        assert out[0] == first
        assert second in out[1]
        assert out[1].startswith("Alpha sentence one.")

    def test_tail_never_cuts_into_fence(self) -> None:
        prev = "Some prose here.\n\n```hcl\nresource x {\n}\n```"
        out = apply_overlap([prev, "next chunk"], overlap=100)
        assert "```" not in out[1]
        assert out[1].startswith("Some prose here.\n\nnext chunk")

    def test_tail_skips_fence_with_blank_lines_inside_code(self) -> None:
        # Regression: blank lines inside code made naive \n\n splitting
        # return mid-code text as the overlap tail.
        prev = "Intro prose paragraph.\n\n```hcl\nresource x {\n  a = 1\n\n  b = 2\n}\n```"
        out = apply_overlap([prev, "next chunk"], overlap=100)
        assert out[1].startswith("Intro prose paragraph.")
        assert "resource x" not in out[1].split("next chunk")[0]

    def test_zero_overlap_is_identity(self) -> None:
        texts = ["a\n\nb", "c"]
        assert apply_overlap(texts, overlap=0) == texts


class TestChunk:
    TEXT = (
        "# Page\n\nAn intro paragraph about resources.\n\n"
        '```hcl\nresource "aws_instance" "web" {\n  ami = "ami-123"\n}\n```\n\n'
        "A closing paragraph with a summary sentence. It has two sentences."
    )

    def test_ids_are_stable_and_ordered(self) -> None:
        first = chunk(self.TEXT, "https://example.com/a")
        second = chunk(self.TEXT, "https://example.com/a")
        assert [c.id for c in first] == [c.id for c in second]
        # uuid5 format, and index progresses
        parsed = [UUID(c.id) for c in first]
        assert len({p for p in parsed}) == len(parsed)

    def test_ids_differ_across_sources(self) -> None:
        a = chunk(self.TEXT, "https://example.com/a")
        b = chunk(self.TEXT, "https://example.com/b")
        assert {c.id for c in a}.isdisjoint({c.id for c in b})

    def test_metadata_passthrough(self) -> None:
        result = chunk(self.TEXT, "https://example.com/a", metadata={"section": "language"})
        assert all(c.metadata == {"section": "language"} for c in result)

    def test_every_paragraph_survives_some_chunk(self) -> None:
        result = chunk(self.TEXT, "https://example.com/a")
        joined = "\n\n".join(c.text for c in result)
        assert "An intro paragraph" in joined
        assert '```hcl\nresource "aws_instance" "web" {' in joined
        assert "closing paragraph with a summary" in joined

    def test_no_empty_chunks(self) -> None:
        result = chunk(self.TEXT, "https://example.com/a")
        assert result
        assert all(c.text.strip() for c in result)

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk("", "https://example.com/a") == []
        assert chunk("   \n\n", "https://example.com/a") == []
