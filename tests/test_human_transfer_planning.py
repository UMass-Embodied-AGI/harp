"""Handover regressions: safe planning and reset-time layout translation."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.human_transfer_planning import (
    endpoint_reaches_goal, refresh_avatar_obstacles, move_seeded,
)
from envs.utils import Pose


def test_fk_rejects_relaxed_endpoint_without_teleporting_simulator():
    class Pin:
        def compute_forward_kinematics(self, qpos):
            self.qpos = qpos

        def get_link_pose(self, index):
            return Pose([0.03, 0.0, 0.0])

    backend = SimpleNamespace(
        pad_move_group_qpos=lambda q: q,
        pinocchio_model=Pin(), link_name_2_idx={"hand": 0}, move_group="hand",
        robot=SimpleNamespace(get_base_pose=lambda: Pose()),
    )
    planner = SimpleNamespace(_mplib=backend, _to_mplib_pose=Pose.from_pose7)
    plan = PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7)))
    assert not endpoint_reaches_goal(planner, plan, [0, 0, 0, 1, 0, 0, 0])


@pytest.mark.parametrize("collider", [None, SimpleNamespace(current_capsules=lambda: [], current_boxes=lambda: [])])
def test_empty_avatar_snapshot_clears_previous_point_cloud(collider):
    clouds = []
    planner = SimpleNamespace(update_obstacles=lambda pts, **kw: clouds.append(pts))
    task = SimpleNamespace(
        avatar_collider=collider,
        config={}, robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(planner=planner)),
    )
    refresh_avatar_obstacles(task, "right")
    assert len(clouds) == 1
    assert clouds[0].shape == (0, 3)


@pytest.mark.parametrize("reason", ["ik_failed", "branch_flip", "rrt_failed"])
def test_every_seeded_fallback_rejects_bad_fk_before_execution(monkeypatch, reason):
    calls = []
    target = np.zeros(7) if reason != "branch_flip" else np.full(7, 2.0)
    ik = PlanResult(reason != "ik_failed", np.stack([np.zeros(7), target]), np.zeros((2, 7)))
    planner = SimpleNamespace(
        update_obstacles=lambda *a, **kw: None,
        solve_ik=lambda *a, **kw: ik,
        _pad_qpos=lambda q: q,
        _mplib=SimpleNamespace(plan_qpos=lambda **kw: {"status": "Failure"}),
    )
    arm = SimpleNamespace(planner=planner, get_arm_qpos=lambda: np.zeros(7))
    task = SimpleNamespace(
        avatar_collider=None, robot=SimpleNamespace(get_arm=lambda _: arm),
        move_to_pose=lambda *a: PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7))),
        execute_plan=lambda *a: calls.append("executed"), plan_success=True,
    )
    monkeypatch.setattr("envs.task_bases.human_transfer_planning.endpoint_reaches_goal", lambda *a: False)
    assert move_seeded(task, [0, 0, 0, 1, 0, 0, 0], "right") is None
    assert not task.plan_success
    assert calls == []


@pytest.mark.parametrize("valid_fk,delta,accepted", [(True, 0.01, True), (False, 0.01, False), (True, 1.0, False)])
def test_deliver_tiny_lift_uses_mplib_and_rejects_bad_endpoints(monkeypatch, valid_fk, delta, accepted):
    from envs.tasks.deliver_to_human import DeliverToHuman

    calls = []
    result = PlanResult(True, np.full((2, 7), delta), np.zeros((2, 7)))
    planner = SimpleNamespace(solve_ik=lambda *a, **kw: calls.append("mplib") or result)
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), n_arm=7,
                          get_arm_qpos=lambda: np.zeros(7), _cached_target=np.zeros(9))
    task = DeliverToHuman.__new__(DeliverToHuman)
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    task._refresh_planner_obstacles = lambda _: calls.append("obstacles")
    task.plan_success = True
    monkeypatch.setattr("envs.tasks.deliver_to_human.endpoint_reaches_goal", lambda *a: valid_fk)
    actual = task._solve_ik(Pose(), "right")
    assert (actual is not None) is accepted
    # A rejected lift waypoint is skipped (re-solved first), never latched:
    # the legs after the lift re-plan and verify their arrival.
    assert task.plan_success is True
    n_solves = 1 if accepted else DeliverToHuman.LIFT_IK_ATTEMPTS
    assert calls == ["obstacles"] + ["mplib"] * n_solves
    assert getattr(task, "_lift_ik_skips", 0) == (0 if accepted else 1)


def test_seeded_fallback_rejects_post_execution_pd_arrival_error(monkeypatch):
    calls = []
    planner = SimpleNamespace(
        update_obstacles=lambda *a, **kw: None,
        solve_ik=lambda *a, **kw: PlanResult(False, np.zeros((0, 7)), np.zeros((0, 7))),
    )
    arm = SimpleNamespace(
        planner=planner, get_arm_qpos=lambda: np.zeros(7),
        ee_link=SimpleNamespace(get_pos=lambda: np.array([0.126, 0, 0])),
    )
    task = SimpleNamespace(
        avatar_collider=None, robot=SimpleNamespace(get_arm=lambda _: arm),
        move_to_pose=lambda *a: PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7))),
        execute_plan=lambda *a: calls.append("executed"), plan_success=True,
        _reseed_cached_arm_target=lambda a: calls.append("reseeded"),
    )
    monkeypatch.setattr("envs.task_bases.human_transfer_planning.endpoint_reaches_goal", lambda *a: True)
    assert move_seeded(task, [0, 0, 0, 1, 0, 0, 0], "right") is None
    assert not task.plan_success
    assert calls == ["executed", "reseeded"]
