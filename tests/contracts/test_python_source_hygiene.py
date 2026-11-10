"""Static source-quality contracts for release Python modules."""

from __future__ import annotations

import ast
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "overload")
        or (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "overload"
        )
        for decorator in node.decorator_list
    )


def _python_sources(
    repo_root: Path = REPO_ROOT, *,
    roots: tuple[str, ...] = ("envs", "scripts", "baseline/servers"),
) -> list[tuple[str, Path]]:
    """Include current tracked and nonignored new runtime sources."""
    sources = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", *roots],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.split("\0")
    return [
        (relative, repo_root / relative)
        for relative in sorted(set(sources))
        if relative.endswith(".py") and (repo_root / relative).is_file()
    ]


def test_python_source_inventory_matches_current_worktree(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    git("init", "--quiet")
    for directory in ("envs", "scripts", "docs"):
        (tmp_path / directory).mkdir()
    tracked = ("envs/tracked.py", "envs/deleted.py", "scripts/ignored_tracked.py")
    for relative in tracked:
        (tmp_path / relative).write_text("", encoding="utf-8")
    git("add", "--", *tracked)
    (tmp_path / "envs/deleted.py").unlink()
    (tmp_path / ".gitignore").write_text("ignored*\n", encoding="utf-8")
    for relative in ("scripts/new.py", "envs/name\nwith space.py",
                     "envs/ignored.py", "scripts/ignored.py",
                     "scripts/nonpython.txt", "docs/outside.py"):
        (tmp_path / relative).write_text("", encoding="utf-8")

    expected = sorted(("envs/tracked.py", "scripts/ignored_tracked.py",
                       "scripts/new.py", "envs/name\nwith space.py"))
    assert _python_sources(tmp_path) == [
        (relative, tmp_path / relative) for relative in expected
    ]


def test_modules_do_not_silently_redefine_symbols() -> None:
    duplicates: list[str] = []
    for relative, path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        symbols: dict[str, list[int]] = defaultdict(list)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols[node.name].append(node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_overload(node):
                    symbols[node.name].append(node.lineno)
        for name, lines in symbols.items():
            if len(lines) > 1:
                duplicates.append(f"{relative}:{name} lines={lines}")

    assert duplicates == [], "duplicate module symbols:\n" + "\n".join(duplicates)


def test_classes_do_not_silently_redefine_methods() -> None:
    """A later class-body definition otherwise erases the earlier method."""

    duplicates: list[str] = []
    for relative_text, path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            methods: dict[str, list[int]] = defaultdict(list)
            for node in class_node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _is_overload(node):
                        methods[node.name].append(node.lineno)
            for name, lines in methods.items():
                if len(lines) > 1:
                    duplicates.append(
                        f"{relative_text}:{class_node.name}.{name} lines={lines}"
                    )

    assert duplicates == [], "duplicate class methods:\n" + "\n".join(duplicates)


_RUN_LOCAL_ROOTS = {"job_logs", "data", ".tmp", "outputs", "runs"}


def _run_local_test_dependencies(source: str) -> list[str]:
    """Find literal artifact paths feeding Python source loaders or imports.

    Follow simple assignment aliases so both inline paths and module-level
    ``PATH = REPO / ...`` declarations are covered. This is a source contract,
    not execution of arbitrary test code or an attempt to resolve dynamic paths.
    """
    tree = ast.parse(source)
    assignments: dict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id].append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id].append(node.value)

    def strings(node: ast.AST, seen: frozenset[str] = frozenset()) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name):
            if node.id in seen:
                return []
            return [text for value in assignments[node.id]
                    for text in strings(value, seen | {node.id})]
        return [text for child in ast.iter_child_nodes(node) for text in strings(child, seen)]

    def is_artifact_path(value: str) -> bool:
        return bool(set(value.replace("\\", "/").split("/")) & _RUN_LOCAL_ROOTS)

    issues = []
    loaders = {
        "spec_from_file_location": (1, "location"),
        "SourceFileLoader": (1, "path"),
        "run_path": (0, "path_name"),
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([name.name for name in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""])
            for module in modules:
                if module.split(".")[0] in _RUN_LOCAL_ROOTS:
                    issues.append(f"line {node.lineno}: imports run-local module {module}")
        elif isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else "")
            if name not in loaders:
                continue
            index, keyword = loaders[name]
            location = (node.args[index] if len(node.args) > index else
                        next((arg.value for arg in node.keywords if arg.arg == keyword), None))
            if location is not None:
                for value in sorted(set(strings(location))):
                    if is_artifact_path(value):
                        issues.append(f"line {node.lineno}: {name} loads run-local source {value}")
    return issues


def test_tests_do_not_load_python_from_ignored_run_artifacts() -> None:
    issues = [f"{relative}: {issue}"
              for relative, path in _python_sources(roots=("tests",))
              for issue in _run_local_test_dependencies(path.read_text(encoding="utf-8"))]
    assert not issues, "test source dependencies must be distributable:\n" + "\n".join(issues)


@pytest.mark.parametrize("source", [
    'spec_from_file_location("observer", "job_logs/observer.py")',
    'path = ROOT / "data" / "observer.py"\n'
    'spec = importlib.util.spec_from_file_location("observer", path)',
    'path: Path = ROOT / "job_logs" / "observer.py"\nalias = path\n'
    'spec_from_file_location("observer", location=alias)',
    'runpy.run_path(path_name="/repo/.tmp/observer.py")',
    'SourceFileLoader("observer", "runs/observer.py")',
    'from job_logs.observer import install',
    'import data.observer',
])
def test_run_local_dependency_guard_rejects_hidden_python(source: str) -> None:
    assert _run_local_test_dependencies(source)


@pytest.mark.parametrize("source", [
    'from scripts.diagnostics import dump_contact_observer',
    'path = ROOT / "envs/contact.py"\nspec_from_file_location("reader", path)',
    'trace = ROOT / "job_logs"\ntrace.mkdir()\n'
    'path = ROOT / "tests/fixtures/helper.py"\nrunpy.run_path(path)',
    'def load(path):\n    return spec_from_file_location("dynamic", path)',
    'path = alias\nalias = path\nspec_from_file_location("dynamic", path)',
])
def test_run_local_dependency_guard_allows_source_and_unresolved_paths(source: str) -> None:
    assert _run_local_test_dependencies(source) == []
