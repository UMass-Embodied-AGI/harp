"""The portable quality command must run both gates and propagate failures."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "release candidate"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/check_quality.sh"
    shutil.copy2(REPO / "scripts/check_quality.sh", script)
    fake_python = root / "test interpreter"
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "python:%s\\n" "$*"\n'
        'case "$2" in\n'
        'ruff) exit "${LINT_RESULT:-0}";;\n'
        'pytest) exit "${TEST_RESULT:-0}";;\n'
        '*) exit 99;;\n'
        'esac\n',
    )
    fake_python.chmod(0o755)
    (root / "scripts/env.sh").write_text('export MAWM_PY="$TEST_INTERPRETER"\n')
    env = dict(os.environ, TEST_INTERPRETER=str(fake_python),
               LINT_RESULT="0", TEST_RESULT="0")
    env.pop("RUFF_BIN", None)
    return script, env


@pytest.mark.parametrize("lint_result,test_result,expected", [
    (0, 0, 0), (1, 0, 1), (0, 1, 1), (2, 5, 1),
])
def test_both_checks_run_and_failures_propagate(launcher, lint_result, test_result, expected):
    script, env = launcher
    env.update(LINT_RESULT=str(lint_result), TEST_RESULT=str(test_result))
    result = subprocess.run(["bash", str(script)], cwd="/tmp", env=env,
                            text=True, capture_output=True)
    assert result.returncode == expected
    assert "python:-m ruff check envs scripts tests baseline/servers" in result.stdout
    assert "python:-m pytest -q tests" in result.stdout
    assert ("Source quality checks passed" in result.stdout) == (expected == 0)


@pytest.mark.parametrize("lint_result", [0, 1])
def test_explicit_linter_executable_is_honored(launcher, lint_result):
    script, env = launcher
    linter = script.parent / "test linter"
    linter.write_text('#!/usr/bin/env bash\nprintf "ruff:%s\\n" "$*"\n'
                      'exit "${LINT_RESULT:-0}"\n')
    linter.chmod(0o755)
    env["RUFF_BIN"] = str(linter)
    env["LINT_RESULT"] = str(lint_result)
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert result.returncode == lint_result
    assert "ruff:check envs scripts tests baseline/servers" in result.stdout
    assert "python:-m ruff" not in result.stdout
    assert "python:-m pytest -q tests" in result.stdout


@pytest.mark.parametrize("args,code", [(["--help"], 0), (["--skip-tests"], 2)])
def test_help_and_invalid_arguments_do_not_start_checks(launcher, args, code):
    script, env = launcher
    result = subprocess.run(["bash", str(script), *args], env=env,
                            text=True, capture_output=True)
    assert result.returncode == code
    assert "Usage:" in result.stdout
    assert "python:" not in result.stdout
