"""A finer correction retry preserves straight-path and endpoint rejection."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases import blocks_ranking_rgb as module
from envs.task_bases.block_grasp import BlockGraspPlan
from envs.utils import Pose


@pytest.mark.parametrize("statuses,qualities,accepted", [
    ([True], [True], True),
    ([False, True], [False, True], True),
    ([False, False], [False, False], False),
    ([False, True], [False, False], False),
])
def test_only_checked_straight_correction_executes(monkeypatch, statuses, qualities, accepted):
    calls, executed, checked, refreshed = [], [], [], []
    plans = [PlanResult(success, np.zeros((2, 7))) for success in statuses]
    qpos = np.arange(7.)
    def screw(q, target, **kwargs):
        calls.append((q.copy(), target.copy(), dict(kwargs)))
        return plans[len(calls) - 1]
    planner = NS(plan_screw_path=screw)
    arm = NS(planner=planner, gripper_val=1., tcp_offset=Pose([0., 0., .1034]),
             get_arm_qpos=lambda: qpos.copy())
    task = NS(_active_block_grasp_plan=BlockGraspPlan(0., []),
              robot=NS(get_arm=lambda tag: arm),
              _top_down_tcp=lambda target, yaw=0.: Pose(target, [0., 1., 0., 0.]),
              execute_plan=lambda plan, tag: executed.append((plan, tag)))
    def endpoint(planner_arg, plan, target):
        assert planner_arg is planner and plan is plans[len(checked)]
        checked.append(plan)
        return qualities[len(checked) - 1]
    monkeypatch.setattr(module, "endpoint_matches", endpoint)
    monkeypatch.setattr(module, "refresh_block_obstacles", lambda task, tag: refreshed.append(tag))
    if accepted:
        assert module.BlocksRankingRGB._move_screw(task, [.1, .2, .8], "right") is True
        assert executed == [(plans[-1], "right")]
    else:
        with pytest.raises(RuntimeError, match="pre-close correction has no straight path"):
            module.BlocksRankingRGB._move_screw(task, [.1, .2, .8], "right")
        assert not executed
    assert refreshed == ["right"] and len(calls) == len(statuses) == len(checked)
    assert calls[0][2] == {"log": False}
    if len(calls) == 2:
        assert calls[1][2] == {"log": False, "qpos_step": .03}
        np.testing.assert_array_equal(calls[0][0], calls[1][0])
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
    np.testing.assert_array_equal(qpos, np.arange(7.))


def test_queued_grasp_leg_does_not_enter_correction_retry(monkeypatch):
    called = []
    arm = NS(gripper_val=1., tcp_offset=Pose([0., 0., .1034]))
    preflight = BlockGraspPlan(0., [(None, None)])
    task = NS(_active_block_grasp_plan=preflight, robot=NS(get_arm=lambda tag: arm),
              _top_down_tcp=lambda target, yaw=0.: Pose(target, [0., 1., 0., 0.]))
    monkeypatch.setattr(module, "execute_grasp_leg", lambda *args: called.append(args))
    monkeypatch.setattr(module, "refresh_block_obstacles", lambda *args: pytest.fail("queued leg refreshed correction"))
    assert module.BlocksRankingRGB._move_screw(task, [.1, .2, .8], "right") is True
    assert len(called) == 1 and called[0][1] is preflight
