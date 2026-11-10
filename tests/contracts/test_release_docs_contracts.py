"""Static contracts for user-facing release documentation."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ACTIVE_DOCS = tuple(
    str(path.relative_to(REPO))
    for path in [REPO / "README.md", REPO / "CONTRIBUTING.md", *sorted((REPO / "docs").rglob("*.md"))]
)
LOCAL_ONLY = {"scripts/env.local.sh"}
CHECKED_IN_PATH = re.compile(
    r"\b(?:scripts|envs|tests|config|docs)/[A-Za-z0-9_./-]+"
)


def test_active_release_docs_reference_existing_checked_in_paths() -> None:
    missing: list[str] = []
    for doc_name in ACTIVE_DOCS:
        text = (REPO / doc_name).read_text(encoding="utf-8")
        for match in CHECKED_IN_PATH.finditer(text):
            # Placeholders and globs (`scripts/finetune_<model>.sh`, `*.yml`)
            # and the machine-local, gitignored env override are not paths.
            if text[match.end():match.end() + 1] in "*$<{":
                continue
            path_name = match.group(0).rstrip(".")
            if path_name in LOCAL_ONLY or (REPO / path_name).exists():
                continue
            missing.append(f"{doc_name}: {path_name}")
    assert not missing, "documented paths must be checked in:\n" + "\n".join(missing)


def test_general_readme_routes_specialized_guides_conditionally() -> None:
    text = (REPO / "docs/general/README.md").read_text(encoding="utf-8")
    assert "Required reading:" in text
    assert "Read when relevant:" in text
