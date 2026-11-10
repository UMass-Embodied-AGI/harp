from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases import block_grasp
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.utils import Pose


def _task(planner):
    arm = SimpleNamespace(
        planner=planner, tcp_offset=Pose([0, 0, 0]),
        get_arm_qpos=lambda: np.zeros(7),
    )
    return SimpleNamespace(
        robot=SimpleNamespace(get_arm=lambda _tag: arm),
        TABLE_TOP_Z=0.765,
        _R_DOWN=BlocksRankingRGB._R_DOWN,
        execute_plan=lambda *_args: pytest.fail("preflight executed an arm motion"),
    )


def test_no_execution_when_every_final_descent_fails(monkeypatch):
    calls = []

    def approach(qpos, target, **_kwargs):
        calls.append("approach")
        return PlanResult(True, np.ones((2, 7)))

    def descent(qpos, target, **_kwargs):
        calls.append("descent")
        # The pre-grasp leg works; the final approach to the cube does not.
        return PlanResult(False) if target[2] < 0.85 else PlanResult(True, np.ones((2, 7)))

    planner = SimpleNamespace(plan_path=approach, plan_screw_path=descent)
    monkeypatch.setattr(block_grasp, "refresh_block_obstacles", lambda *_args: None)
    monkeypatch.setattr(block_grasp, "endpoint_matches", lambda _p, plan, _t: plan.success)
    assert block_grasp.plan_block_grasp(_task(planner), [0, 0, 0.795], 0.03, 0.965, "right") is None
    assert calls.count("approach") == 8
    assert calls.count("descent") == 16


def test_symmetric_yaw_is_selected_before_any_execution(monkeypatch):
    attempts = []

    def approach(qpos, target, **_kwargs):
        attempts.append(np.asarray(target))
        return PlanResult(True, np.ones((2, 7)))

    def descent(qpos, target, **_kwargs):
        if len(attempts) <= 2 and target[2] < 0.85:
            return PlanResult(False)
        return PlanResult(True, np.ones((2, 7)))

    planner = SimpleNamespace(plan_path=approach, plan_screw_path=descent)
    monkeypatch.setattr(block_grasp, "refresh_block_obstacles", lambda *_args: None)
    monkeypatch.setattr(block_grasp, "endpoint_matches", lambda _p, plan, _t: plan.success)
    result = block_grasp.plan_block_grasp(_task(planner), [0, 0, 0.795], 0.03, 0.965, "right")
    assert result is not None
    assert np.isclose(result.yaw, np.pi / 2)
    assert len(result.legs) == 3


def test_displaced_cube_does_not_trigger_replan_or_execution():
    plan = block_grasp.BlockGraspPlan(0.0, [(np.array([0, 0, 1, 1, 0, 0, 0]), object())])
    task = SimpleNamespace(execute_plan=lambda *_args: pytest.fail("moved cube executed"))
    with pytest.raises(RuntimeError, match="block moved after preflight"):
        block_grasp.execute_grasp_leg(task, plan, [0.02, 0, 1, 1, 0, 0, 0], "right")
    assert len(plan.legs) == 1


@pytest.mark.parametrize("collider", [None, SimpleNamespace(
    current_capsules=lambda: [], current_boxes=lambda: [],
)])
def test_absent_or_empty_avatar_clears_stale_obstacles(collider):
    updates = []
    task = _task(SimpleNamespace(update_obstacles=lambda pts, **kw: updates.append(pts)))
    task.avatar_collider = collider
    block_grasp.refresh_block_obstacles(task, "right")
    assert len(updates) == 1
    assert updates[0].shape == (0, 3)


def test_rejected_grasp_stays_an_episode_failure_not_an_error(monkeypatch):
    task = BlocksRankingRGB({"setting": "sim"})
    steps = []
    opens = []

    def rejected(*_args):
        raise RuntimeError("block moved after preflight")

    monkeypatch.setattr(task, "_pick_and_place_block", rejected)
    monkeypatch.setattr(task, "open_gripper", lambda tag: opens.append(tag))
    monkeypatch.setattr(task, "step_sim", lambda: steps.append(1))
    assert task._place_block_with_recovery("red") is False
    assert opens == ["right"]
    assert len(steps) == task.FAILURE_RECOVERY_STEPS


def test_pre_close_rejection_retries_once_after_retreat(monkeypatch):
    task = BlocksRankingRGB({"setting": "sim"})
    calls = []

    def pick(key, arm_tag="right"):
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("block moved after preflight")
        return True

    monkeypatch.setattr(task, "_pick_and_place_block", pick)
    monkeypatch.setattr(task, "_block_retry_allowed", lambda arm_tag, key=None: True)
    monkeypatch.setattr(task, "_retry_block_after_retreat",
                        lambda key, arm_tag: bool(task._pick_and_place_block(key, arm_tag)))
    monkeypatch.setattr(task, "open_gripper", lambda tag: None)
    monkeypatch.setattr(task, "step_sim", lambda: None)
    assert task._place_block_with_recovery("red") is True
    assert calls == ["red", "red"]
    task.config["block_pre_close_retries"] = 0
    monkeypatch.undo()
    assert task._block_retry_allowed("right") is False
