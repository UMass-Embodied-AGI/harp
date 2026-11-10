from types import SimpleNamespace

import numpy as np

from envs.base_task_parts import grasp_select
from envs.grasp import GraspPose
from envs.planning.base import PlanResult
from envs.utils import Pose


def _fake_task(monkeypatch):
    grasps = [
        GraspPose(name, [0, 0, 0], [0, 1, 0, 0], source="manual")
        for name in ("first", "second")
    ]
    monkeypatch.setattr(grasp_select, "load_grasp_poses", lambda *_a, **_k: grasps)
    executions, planned = [], []

    def plan(grasp, *_args, **_kwargs):
        planned.append(grasp.name)
        marker = 1 if grasp.name == "first" else 2
        result = PlanResult(True, np.full((2, 7), marker))
        return result, result, Pose([0, 0, 1]), Pose([0, 0, 1.12])

    task = SimpleNamespace(
        config={}, TABLE_TOP_Z=0.765,
        robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(origin_pose=Pose([0, -0.5, 0]))),
        _plan_one_grasp=plan,
        execute_plan=lambda plan, tag: executions.append((plan, tag)),
    )
    return task, planned, executions


def test_default_helper_still_executes_first_ranked_grasp(monkeypatch):
    task, planned, executions = _fake_task(monkeypatch)
    result = grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "object", Pose([0, 0, 1]), "right",
    )
    assert result[2].name == "first"
    assert planned == ["first"]
    assert len(executions) == 2


def test_rejected_continuation_executes_nothing_before_next_candidate(monkeypatch):
    task, planned, executions = _fake_task(monkeypatch)

    def check(plans):
        assert executions == []
        return bool(plans[1].position[-1, 0] == 2)

    result = grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "object", Pose([0, 0, 1]), "right", post_plan_check=check,
    )
    assert result[2].name == "second"
    assert planned == ["first", "second"]
    assert len(executions) == 2


def test_allowlist_filters_before_planning_without_forcing_a_pose(monkeypatch):
    task, planned, executions = _fake_task(monkeypatch)
    result = grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "object", Pose([0, 0, 1]), "right", grasp_names=("second",),
    )
    assert result[2].name == "second"
    assert planned == ["second"]
    assert len(executions) == 2


def test_all_continuations_rejected_leave_arm_untouched(monkeypatch):
    task, planned, executions = _fake_task(monkeypatch)
    result = grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "object", Pose([0, 0, 1]), "right", post_plan_check=lambda _: False,
    )
    assert result is None
    assert planned == ["first", "second"]
    assert executions == []
