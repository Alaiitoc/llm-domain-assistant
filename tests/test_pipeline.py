"""Unit tests for ingestion.pipeline — pure logic, no network, no Qdrant."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from ingestion.pipeline import discover_files, load_config, source_url_for


class TestSourceUrl:
    DOCS_ROOT = pathlib.Path("/corpus/content/terraform/v1.16.x/docs")

    def test_nested_page_maps_to_docs_url(self) -> None:
        path = self.DOCS_ROOT / "language/meta-arguments/for_each.mdx"
        assert (
            source_url_for(path, self.DOCS_ROOT)
            == "https://developer.hashicorp.com/terraform/language/meta-arguments/for_each"
        )

    def test_index_maps_to_parent_section(self) -> None:
        path = self.DOCS_ROOT / "language/index.mdx"
        assert source_url_for(path, self.DOCS_ROOT) == (
            "https://developer.hashicorp.com/terraform/language"
        )

    def test_top_level_index_maps_to_base_url(self) -> None:
        path = self.DOCS_ROOT / "intro/index.mdx"
        assert source_url_for(path, self.DOCS_ROOT) == (
            "https://developer.hashicorp.com/terraform/intro"
        )


class TestDiscoverFiles:
    def test_finds_mdx_sorted_and_skips_partials(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "docs"
        (root / "language").mkdir(parents=True)
        (root / "partials").mkdir()
        (root / "language" / "b.mdx").write_text("# B")
        (root / "language" / "a.mdx").write_text("# A")
        (root / "partials" / "fragment.mdx").write_text("# frag")
        files = discover_files([str(root)])
        assert [p.name for p, _ in files] == ["a.mdx", "b.mdx"]

    def test_missing_source_dir_is_fatal(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(SystemExit, match="not found"):
            discover_files([str(tmp_path / "nope")])


class TestLoadConfig:
    def test_missing_required_keys_fail_fast(self, tmp_path: pathlib.Path) -> None:
        config = tmp_path / "cfg.yaml"
        config.write_text("chunk_size: 800\n")  # sources missing
        with pytest.raises(ValidationError, match="sources"):
            load_config(str(config))

    def test_full_config_loads_typed(self, tmp_path: pathlib.Path) -> None:
        config = tmp_path / "cfg.yaml"
        config.write_text(
            "sources: [data/x]\nchunk_size: 800\nchunk_overlap: 100\n"
            "embedding_model: mistral-embed\nembedding_dim: 1024\n"
            "collection: terraform_docs\nqdrant_url: http://localhost:6333\n"
        )
        loaded = load_config(str(config))
        assert loaded.collection == "terraform_docs"
        assert loaded.chunk_size == 800
        assert isinstance(loaded.sources, list)
