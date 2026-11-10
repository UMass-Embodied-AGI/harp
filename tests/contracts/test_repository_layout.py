"""Navigation and source ownership must survive a clean source checkout."""

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _missing_local_links(document, repo=REPO):
    """Check explicit file links, not code examples, anchors or remote URLs."""
    missing = []
    for match in MARKDOWN_LINK.finditer(document.read_text(encoding="utf-8")):
        target = urlsplit(match.group(1).strip("<>"))
        if target.scheme or target.netloc or not target.path:
            continue
        path = (document.parent / unquote(target.path)).resolve()
        # Repository-relative links should not accidentally resolve in a
        # developer's parent workspace and appear distributable.
        if not path.is_relative_to(repo.resolve()) or not path.exists():
            missing.append(match.group(1))
    return missing


def test_active_documentation_file_links_resolve():
    documents = [REPO / "README.md", REPO / "CONTRIBUTING.md",
                 REPO / "scripts/README.md", REPO / "config/README.md",
                 REPO / "tests/README.md"]
    documents.extend(sorted((REPO / "docs").rglob("*.md")))
    missing = [f"{document.relative_to(REPO)}: {target}"
               for document in documents
               for target in _missing_local_links(document)]
    assert not missing, "broken documentation file links:\n" + "\n".join(missing)


def test_link_check_distinguishes_anchors_urls_missing_and_escaping_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "with space.md"
    target.write_text("# Example\n", encoding="utf-8")
    index = docs / "README.md"
    index.write_text(
        "[local](with%20space.md#section)\n[anchor](#section)\n"
        "[remote](https://example.invalid/not-local.md)\n"
        "[missing](absent.md)\n[outside](../../outside.md)\n",
        encoding="utf-8",
    )
    assert _missing_local_links(index, repo=tmp_path) == [
        "absent.md", "../../outside.md",
    ]
