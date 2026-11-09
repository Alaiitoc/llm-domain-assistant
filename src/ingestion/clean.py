"""Parse MDX docs into clean text plus frontmatter metadata.

Frontmatter is YAML and parsed with python-frontmatter (which wraps PyYAML):
hand-rolled line parsing cannot handle folded scalars (``>-``) and breaks on
empty frontmatter. The body is stripped of the docs-site MDX components via an
allowlist, so HCL placeholders like ``<BLOCK_TYPE>`` in code examples survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import frontmatter


@dataclass
class CleanedDocument:
    """A cleaned document, ready for chunking and embedding."""

    text: str
    source_file: str
    metadata: dict[str, str] = field(default_factory=dict)


# Allowlisted docs-site components. Anything else that looks tag-like (e.g.
# HCL's <METADATA_BLOCK> placeholders) is content and must be left alone.
_TAB_OPEN = re.compile(r'<Tab\s+(?:heading|title)="(?P<heading>[^"]*)"\s*>')
_TAB_MISC = re.compile(r"</?Tabs?\s*>")
_CALLOUT_OPEN = re.compile(r"<(?P<kind>Note|Warning|Tip|EnterpriseAlert)\b[^>]*>")
_CALLOUT_CLOSE = re.compile(r"</(?:Note|Warning|Tip|EnterpriseAlert)>")
_HIGHLIGHT = re.compile(r"</?Highlight[^>]*>")
# CodeBlockConfig wraps fences with rendering config; the filename attribute is
# the only meaningful bit (it names the file the example lives in).
_CODEBLOCK_OPEN = re.compile(r'<CodeBlockConfig\s+filename="(?P<filename>[^"]*)"\s*/?>')
_CODEBLOCK_MISC = re.compile(r"<CodeBlockConfig\b[^>]*/?>|</CodeBlockConfig>")
# Self-contained components with no readable content.
_DROP = re.compile(r"<(?:VideoEmbed|ProviderTable)\b[^>]*/>|<p>\s*</p>")


def strip_mdx_components(text: str) -> str:
    """Replace allowlisted MDX components with plain markers or nothing."""
    text = _TAB_OPEN.sub(lambda m: f"\n**{m.group('heading')}**\n", text)
    text = _TAB_MISC.sub("", text)  # unwraps <Tabs>/<Tab> variants
    text = _CALLOUT_OPEN.sub(lambda m: f"\n**{m.group('kind')}:**\n", text)
    text = _CALLOUT_CLOSE.sub("\n", text)
    text = _HIGHLIGHT.sub("", text)
    text = _CODEBLOCK_OPEN.sub(lambda m: f"\n**File: {m.group('filename')}**\n", text)
    text = _CODEBLOCK_MISC.sub("", text)
    text = _DROP.sub("", text)
    return text


def clean(path: str) -> CleanedDocument:
    if not path.endswith(".mdx"):
        raise ValueError(f"Expected .mdx file, got {path}")

    post = frontmatter.load(path)
    # PyYAML types values (bools, datetimes, ints); payloads must be strings.
    metadata = {str(k): str(v) for k, v in post.metadata.items()}
    return CleanedDocument(
        text=strip_mdx_components(post.content),
        source_file=path,
        metadata=metadata,
    )
