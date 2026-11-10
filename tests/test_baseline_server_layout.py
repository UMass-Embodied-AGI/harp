"""Check standalone model-server paths without importing any model stack."""

import ast
from pathlib import Path
import re

import pytest


REPO = Path(__file__).resolve().parents[1]
SERVERS = REPO / "baseline" / "servers"


@pytest.mark.parametrize("server", sorted(path.name for path in SERVERS.glob("*_server.py")))
def test_server_examples_use_current_portable_paths(server):
    source = (SERVERS / server).read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(source))
    assert "/scratch4/" not in doc
    assert re.search(r"baseline/[a-z_]+_server\.py", doc) is None
    assert f"baseline/servers/{server}" in doc
