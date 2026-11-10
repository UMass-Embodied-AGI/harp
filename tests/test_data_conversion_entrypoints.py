"""Legacy conversion commands remain aliases, not copies with split globals."""

import ast
import importlib
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[1]
CONVERTERS = [
    ("build_genesis_hr_bench_rlds", "rlds"),
    ("convert_genesis_hr_bench_to_lerobot", "lerobot"),
]


@pytest.fixture(params=CONVERTERS, ids=[pair[1] for pair in CONVERTERS])
def converter_alias(request, monkeypatch):
    legacy, canonical = request.param
    scripts = importlib.import_module("scripts")
    package = importlib.import_module("scripts.data_conversion")
    implementation = ModuleType(f"scripts.data_conversion.{canonical}")
    implementation.configuration = {"value": "original"}
    implementation.calls = []

    def main():
        implementation.calls.append(list(sys.argv))
        return implementation.configuration

    implementation.main = main
    implementation._private_helper = main
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    monkeypatch.setattr(package, canonical, implementation, raising=False)
    # Register the original state even when absent so a newly imported alias
    # is removed again after the test, not leaked into the rest of the suite.
    monkeypatch.setitem(sys.modules, f"scripts.{legacy}", None)
    del sys.modules[f"scripts.{legacy}"]
    monkeypatch.setattr(scripts, legacy, None, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return legacy, implementation


def test_legacy_imports_share_helpers_and_mutable_globals(converter_alias):
    legacy, implementation = converter_alias
    old_module = importlib.import_module(f"scripts.{legacy}")
    new_module = importlib.import_module(implementation.__name__)
    assert old_module is new_module is implementation
    assert old_module._private_helper is new_module.main
    old_module.configuration = {"value": "patched through old import"}
    assert new_module.main() == old_module.configuration
    new_module.configuration = {"value": "patched through new import"}
    assert old_module.main() == new_module.configuration


def test_legacy_cli_dispatches_without_changing_arguments(converter_alias, monkeypatch,
                                                         tmp_path):
    legacy, implementation = converter_alias
    command = REPO / "scripts" / f"{legacy}.py"
    argv = [str(command), "--task", "pour_water", "--include-failures"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(command), run_name="__main__")
    assert implementation.calls == [argv]
    assert Path(sys.path[0]) == REPO


@pytest.mark.parametrize("converter", ["rlds"])
def test_moved_converter_defaults_remain_anchored_to_the_checkout(converter, monkeypatch,
                                                                tmp_path):
    path = REPO / "scripts" / "data_conversion" / f"{converter}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPO_ROOT"
                for target in node.targets)
    )
    namespace = {"Path": Path, "__file__": str(path)}
    monkeypatch.chdir(tmp_path)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["REPO_ROOT"] == REPO


def test_converter_package_does_not_eagerly_load_optional_stacks():
    path = REPO / "scripts" / "data_conversion" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
