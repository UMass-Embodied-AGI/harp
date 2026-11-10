"""Eval/collect entry points refuse physics- or planner-bypassing switches."""
from pathlib import Path

import pytest

from envs.debug_switches import BYPASS_SWITCHES, active_bypass_switches, refuse_debug_switches

REPO = Path(__file__).resolve().parents[1]


def test_no_switch_runs():
    refuse_debug_switches("x", environ={"MPLIB_PLAN_DIAG": "1", "DEBUG_TELEPORT": "0"})


@pytest.mark.parametrize("name", BYPASS_SWITCHES)
def test_each_bypass_switch_is_refused(name):
    with pytest.raises(SystemExit, match=name):
        refuse_debug_switches("x", environ={name: "1"})


def test_explicit_allow_only_warns(capsys):
    refuse_debug_switches("x", environ={"USE_SETQPOS": "1", "HARP_ALLOW_DEBUG_SWITCHES": "1"})
    assert "not benchmark results" in capsys.readouterr().err
    assert active_bypass_switches({"USE_SETQPOS": " 1 "}) == ["USE_SETQPOS"]


@pytest.mark.parametrize("path", [
    "scripts/eval.py",
    "scripts/eval_vla.py",
    "scripts/eval_expert_set.py",
    "scripts/collect.py",
    "scripts/vla_data/collect_vla.py",
])
def test_entry_points_call_the_guard(path):
    source = (REPO / path).read_text(encoding="utf-8")
    assert f'refuse_debug_switches("{path}")' in source
