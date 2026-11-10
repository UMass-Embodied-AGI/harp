"""Guard against shipping a tracked refactor with missing local-only files."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "compat" / "release_files_v1.txt"
RUNTIME_ROOTS = {"envs", "scripts", "eval_sets"}
BOWL_MANIFEST = (
    "envs/task_bases/stack_bowls_collision/bowl_002_model1/package_manifest.json"
)


def _tracked_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, text=True,
        stdout=subprocess.PIPE, check=True,
    )
    return set(filter(None, result.stdout.split("\0")))


def _inventory_issues(repo: Path, paths: list[str], tracked: set[str]) -> list[str]:
    issues = []
    for relative in paths:
        if not (repo / relative).is_file():
            issues.append(f"missing required release file: {relative}")
        if relative not in tracked:
            issues.append(f"untracked required release file: {relative}")
    return issues


def _release_paths() -> list[str]:
    paths = [
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    return paths


def test_release_v1_foundation_is_present_and_tracked() -> None:
    issues = _inventory_issues(REPO, _release_paths(), _tracked_paths(REPO))
    assert not issues, "\n".join(issues)


def _module_paths(repo: Path, parts: tuple[str, ...], tracked: set[str]) -> set[str]:
    """Resolve project modules without importing the simulator or other code.

    Namespace packages have no required initializer. Ordinary package
    initializers do, including every ancestor of an imported submodule.
    """
    if not parts or parts[0] not in RUNTIME_ROOTS:
        return set()

    def exists(relative: str) -> bool:
        return relative in tracked or (repo / relative).is_file()

    paths = set()
    for end in range(1, len(parts) + 1):
        initializer = "/".join((*parts[:end], "__init__.py"))
        if exists(initializer):
            paths.add(initializer)
    initializer = "/".join((*parts, "__init__.py"))
    module = "/".join(parts) + ".py"
    if not exists(initializer) and exists(module):
        paths.add(module)
    return paths


def _project_import_paths(
    repo: Path, relative: str, tracked: set[str],
) -> set[str]:
    tree = ast.parse((repo / relative).read_text(encoding="utf-8"), filename=relative)
    package = Path(relative).parent.parts
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                dependencies.update(_module_paths(repo, tuple(name.name.split(".")), tracked))
        elif isinstance(node, ast.ImportFrom):
            module = tuple(node.module.split(".")) if node.module else ()
            if node.level:
                if node.level > len(package):
                    continue
                module = package[:len(package) - node.level + 1] + module
            dependencies.update(_module_paths(repo, module, tracked))
            # A from-import may name a symbol or a submodule. Only a real local
            # module adds a dependency; symbols and wildcard imports add none.
            for name in node.names:
                if name.name != "*":
                    dependencies.update(_module_paths(repo, (*module, name.name), tracked))
    return dependencies


def _runtime_import_issues(repo: Path) -> list[str]:
    tracked = _tracked_paths(repo)
    issues = []
    for relative in sorted(tracked):
        if not relative.endswith(".py") or Path(relative).parts[0] not in RUNTIME_ROOTS:
            continue
        if not (repo / relative).is_file():
            issues.append(
                f"missing tracked runtime source: {relative}; "
                "restore it or record the intended deletion in Git"
            )
            continue
        for dependency in sorted(_project_import_paths(repo, relative, tracked)):
            if dependency not in tracked:
                issues.append(f"{relative}: imports untracked project module {dependency}")
    return issues


def test_release_runtime_imports_do_not_depend_on_untracked_python() -> None:
    """Conservatively catch local-only modules imported by shipped code."""
    issues = _runtime_import_issues(REPO)
    assert not issues, "release runtime inventory is incomplete:\n" + "\n".join(issues)


def _collision_package_paths(repo: Path, manifest_name: str) -> list[str]:
    manifest_path = repo / manifest_name
    assert manifest_path.is_file(), f"missing collision package manifest: {manifest_name}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hull_count = manifest["n_hulls"]
    assert type(hull_count) is int and hull_count > 0
    expected = {"bowl.urdf", *(f"hull_{index:03d}.obj" for index in range(hull_count))}
    # Exact curated names also reject paths escaping the package directory.
    assert set(manifest["sha256"]) == expected, "incomplete collision package manifest"
    package = Path(manifest_name).parent
    return sorted([manifest_name, *((package / name).as_posix() for name in expected)])


def test_release_bowl_collision_package_is_present_and_tracked() -> None:
    paths = _collision_package_paths(REPO, BOWL_MANIFEST)
    assert len(paths) == 202, "the released bowl package has 200 hulls, a URDF, and a manifest"
    issues = _inventory_issues(REPO, paths, _tracked_paths(REPO))
    assert not issues, "\n".join(issues)


@pytest.fixture
def tiny_repo(tmp_path: Path):
    """Stage only fixture files; never modify the real repository's index."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)

    def write(relative: str, source: str = "", *, tracked: bool = False) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        if tracked:
            subprocess.run(["git", "add", "--", relative], cwd=tmp_path, check=True)
        return path

    return tmp_path, write


@pytest.mark.parametrize("source", [
    "import envs.family.helper\n",
    "import envs.family.helper as local_helper\n",
    "from envs.family.helper import (\n    value,\n)\n",
    "from envs.family import helper\n",
    "from .helper import value\n",
    "from . import (\n    helper,\n)\n",
    "from ..family import helper\n",
    "def lazy():\n    from .helper import value\n",
])
def test_inventory_detects_untracked_import_forms(tiny_repo, source: str) -> None:
    repo, write = tiny_repo
    write("envs/family/entry.py", source, tracked=True)
    write("envs/family/helper.py", "value = 1\n")
    assert _runtime_import_issues(repo) == [
        "envs/family/entry.py: imports untracked project module envs/family/helper.py"
    ]


def test_inventory_requires_package_initializers_and_supports_namespaces(tiny_repo) -> None:
    repo, write = tiny_repo
    write("scripts/entry.py", "from envs.family import helper\n", tracked=True)
    write("envs/__init__.py", tracked=True)
    write("envs/family/__init__.py")
    write("envs/family/helper/__init__.py", "value = 1\n", tracked=True)
    assert _runtime_import_issues(repo) == [
        "scripts/entry.py: imports untracked project module envs/family/__init__.py"
    ]
    write("envs/family/__init__.py", tracked=True)
    assert _runtime_import_issues(repo) == []


def test_inventory_ignores_symbols_external_imports_and_commented_imports(tiny_repo) -> None:
    repo, write = tiny_repo
    write("scripts/entry.py", (
        "from envs.helper import value\n"
        "from envs.helper import *\n"
        "from numpy import array\n"
        "# from envs import scratch\n"
        "description = 'import envs.scratch'\n"
    ), tracked=True)
    write("envs/helper.py", "value = 1\n", tracked=True)
    write("envs/scratch.py")
    assert _runtime_import_issues(repo) == []


def test_inventory_does_not_hide_ignored_dependencies(tiny_repo) -> None:
    repo, write = tiny_repo
    write(".gitignore", "helper.py\n", tracked=True)
    write("scripts/entry.py", "from scripts.helper import value\n", tracked=True)
    write("scripts/helper.py", "value = 1\n")
    assert _runtime_import_issues(repo) == [
        "scripts/entry.py: imports untracked project module scripts/helper.py"
    ]


def test_inventory_reports_missing_tracked_sources_without_hiding_other_issues(tiny_repo) -> None:
    repo, write = tiny_repo
    write("envs/deleted.py", tracked=True).unlink()
    write("scripts/entry.py", "import scripts.helper\n", tracked=True)
    write("scripts/helper.py")
    assert _runtime_import_issues(repo) == [
        "missing tracked runtime source: envs/deleted.py; "
        "restore it or record the intended deletion in Git",
        "scripts/entry.py: imports untracked project module scripts/helper.py",
    ]


def test_collision_inventory_requires_every_curated_asset(tiny_repo) -> None:
    repo, write = tiny_repo
    manifest_name = "envs/bowls/package_manifest.json"
    manifest = {"n_hulls": 2, "sha256": {
        "bowl.urdf": "unused", "hull_000.obj": "unused", "hull_001.obj": "unused",
    }}
    write(manifest_name, json.dumps(manifest), tracked=True)
    write("envs/bowls/bowl.urdf", tracked=True)
    write("envs/bowls/hull_000.obj", tracked=True)
    write(".gitignore", "hull_001.obj\n", tracked=True)
    last_hull = write("envs/bowls/hull_001.obj")
    paths = _collision_package_paths(repo, manifest_name)
    assert len(paths) == 4
    assert _inventory_issues(repo, paths, _tracked_paths(repo)) == [
        "untracked required release file: envs/bowls/hull_001.obj"
    ]
    last_hull.unlink()
    assert _inventory_issues(repo, paths, _tracked_paths(repo)) == [
        "missing required release file: envs/bowls/hull_001.obj",
        "untracked required release file: envs/bowls/hull_001.obj",
    ]


def test_collision_inventory_rejects_incomplete_or_escaping_manifest(tiny_repo) -> None:
    repo, write = tiny_repo
    manifest_name = "envs/bowls/package_manifest.json"
    for names in ({"bowl.urdf", "../hull_000.obj"}, {"bowl.urdf"}):
        write(manifest_name, json.dumps({
            "n_hulls": 1, "sha256": dict.fromkeys(names, "unused"),
        }))
        with pytest.raises(AssertionError, match="incomplete collision package manifest"):
            _collision_package_paths(repo, manifest_name)
