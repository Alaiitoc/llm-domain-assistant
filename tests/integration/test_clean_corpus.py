"""Integration tests: run ingestion.clean over the real Terraform docs corpus.

The corpus lives under data/raw/ and is gitignored, so these tests skip with a
clear reason on any machine (like CI) that has not fetched the corpus.
Run selectively with: pytest -m integration
Skip locally with:            pytest -m "not integration"
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ingestion.clean import CleanedDocument, clean

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data/raw/terraform-docs-repo/content/terraform/v1.16.x/docs"


@pytest.fixture(scope="module")
def mdx_files() -> list[pathlib.Path]:
    if not CORPUS_DIR.is_dir():
        pytest.skip(f"docs corpus not present: {CORPUS_DIR}")
    return sorted(CORPUS_DIR.rglob("*.mdx"))


class TestCleanCorpus:
    def test_corpus_is_nonempty(self, mdx_files: list[pathlib.Path]) -> None:
        assert len(mdx_files) > 100

    def test_every_file_cleans_without_error(self, mdx_files: list[pathlib.Path]) -> None:
        for path in mdx_files:
            doc = clean(str(path))
            assert isinstance(doc, CleanedDocument)
            assert isinstance(doc.text, str) and doc.text.strip(), path
            assert isinstance(doc.metadata, dict)
            assert all(
                isinstance(k, str) and isinstance(v, str) for k, v in doc.metadata.items()
            ), path

    def test_known_file_strips_frontmatter(self, mdx_files: list[pathlib.Path]) -> None:
        path = CORPUS_DIR / "language/syntax/configuration.mdx"
        doc = clean(str(path))
        assert doc.metadata["page_title"] == "Syntax - Configuration Language"
        assert "page_title:" not in doc.text
        assert "# Configuration Syntax" in doc.text

    def test_folded_scalar_description_is_stringified(self) -> None:
        path = CORPUS_DIR / "language/syntax/configuration.mdx"
        doc = clean(str(path))
        assert doc.metadata["description"].startswith("Key constructs of the native")

    _COMPONENT_TAG = re.compile(
        r"</?(?:Tabs?|Note|Warning|Tip|EnterpriseAlert|Highlight|VideoEmbed|ProviderTable)\b"
    )

    def test_no_mdx_components_remain(self, mdx_files: list[pathlib.Path]) -> None:
        for path in mdx_files:
            doc = clean(str(path))
            assert not self._COMPONENT_TAG.search(doc.text), path
