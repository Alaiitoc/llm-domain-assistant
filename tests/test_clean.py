"""Unit tests for ingestion.clean."""

from __future__ import annotations

import pathlib

import pytest

from ingestion.clean import clean


class TestCleanFrontmatter:
    def test_parses_frontmatter_into_metadata(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "intro.mdx"
        path.write_text(
            "---\ntitle: Internals\nsection: Backend\n---\n\n# Internals\n\nBody text.\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert doc.metadata == {"title": "Internals", "section": "Backend"}
        assert "Body text." in doc.text
        assert doc.source_file == str(path)

    def test_coerces_yaml_values_to_strings(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "typed.mdx"
        path.write_text(
            "---\nweight: 42\npublished: true\n---\n\nBody.\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert doc.metadata == {"weight": "42", "published": "True"}

    def test_parses_folded_scalar_description(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "folded.mdx"
        path.write_text(
            "---\ndescription: >-\n  Key constructs of the native Terraform language syntax,\n"
            "  including identifiers and comments.\n---\n\n# Syntax\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert doc.metadata["description"] == (
            "Key constructs of the native Terraform language syntax, "
            "including identifiers and comments."
        )
        assert ">-" not in doc.metadata["description"]

    def test_empty_frontmatter_yields_empty_metadata(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "empty.mdx"
        path.write_text("---\n---\n\n# Body only\n", encoding="utf-8")
        doc = clean(str(path))
        assert doc.metadata == {}
        assert "# Body only" in doc.text

    def test_body_separators_are_preserved(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "body.mdx"
        path.write_text("---\ntitle: T\n---\na --- b\n", encoding="utf-8")
        doc = clean(str(path))
        assert doc.text == "a --- b"


class TestCleanNoFrontmatter:
    def test_no_frontmatter_yields_empty_metadata(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "plain.mdx"
        path.write_text("# Just a heading\n\nNo frontmatter here.\n", encoding="utf-8")
        doc = clean(str(path))
        assert doc.metadata == {}
        # python-frontmatter strips the trailing newline; documented on purpose.
        assert doc.text == "# Just a heading\n\nNo frontmatter here."


class TestStripMdxComponents:
    def test_tab_heading_becomes_bold_marker(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "tabs.mdx"
        path.write_text(
            '---\ntitle: T\n---\n\n<Tabs>\n\n<Tab heading="AWS">\n\n'
            "aws_instance\n\n</Tab>\n\n</Tabs>\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert "**AWS**" in doc.text
        assert "aws_instance" in doc.text
        assert "<Tab" not in doc.text
        assert "<Tabs>" not in doc.text

    def test_callout_content_kept_with_marker(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "note.mdx"
        path.write_text(
            "---\ntitle: T\n---\n\n<Note>\n\n"
            "Ephemeral variables are available in v1.10.\n\n</Note>\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert "**Note:**" in doc.text
        assert "Ephemeral variables are available in v1.10." in doc.text
        assert "</Note>" not in doc.text

    def test_hcl_placeholders_survive(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "placeholders.mdx"
        path.write_text(
            "---\ntitle: T\n---\n\nA `<META>` block accepts `<NAME> = <VALUE>`.\n"
            '```hcl\n(Block_type) "<BLOCK_TYPE>" {}\n```\n',
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert "<BLOCK_TYPE>" in doc.text
        assert "<NAME> = <VALUE>" in doc.text

    def test_embed_and_table_dropped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "drops.mdx"
        path.write_text(
            '---\ntitle: T\n---\n\nIntro text.\n\n<VideoEmbed url="https://youtube.com/x"/>\n\n'
            "<ProviderTable />\n",
            encoding="utf-8",
        )
        doc = clean(str(path))
        assert "Intro text." in doc.text
        assert "<VideoEmbed" not in doc.text
        assert "<ProviderTable" not in doc.text


class TestCleanValidation:
    def test_rejects_non_mdx_path(self) -> None:
        with pytest.raises(ValueError, match=r"Expected \.mdx file"):
            clean("docs/intro.html")
