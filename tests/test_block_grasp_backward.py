"""Task-local reverse preflight is bounded, default-off and simulator-free."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases import block_grasp as bg
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.utils import Pose


POINTS = (np.array([0., 0., .965]), np.array([0., 0., .945]), np.array([0., 0., .789]))
HOME = np.array([0., -.3, 1., 0., 1., 0., 0.])


def fixture():
    last = {"q": HOME.copy()}
    calls = {"ik": [], "screw": [], "rrt": [], "collision": []}
    backend = SimpleNamespace(
        joint_limits=np.tile([-10., 10.], (7, 1)),
        move_group_joint_indices=np.arange(7), move_group="hand", link_name_2_idx={"hand": 0},
        robot=SimpleNamespace(get_base_pose=Pose),
        pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q: last.update(q=q.copy()),
            get_link_pose=lambda _: Pose.from_pose7(last["q"]),
        ),
        planning_world=SimpleNamespace(
            set_qpos_all=lambda q: calls["collision"].append(q.copy()),
            is_state_colliding=lambda: False,
        ),
    )

    def ik(goal, qpos, **kwargs):
        calls["ik"].append(kwargs)
        return "Success", [goal.to_pose7()]

    def screw(qpos, target, **kwargs):
        calls["screw"].append((qpos.copy(), target.copy(), kwargs))
        position = np.linspace(qpos, target, 3)
        return PlanResult(True, position, np.tile(np.arange(7.), (3, 1)))

    def rrt(goals, qpos, **kwargs):
        calls["rrt"].append((goals, qpos.copy(), kwargs))
        position = np.linspace(qpos, goals[0], 3)
        return {"status": "Success", "position": position, "velocity": np.ones_like(position)}

    backend.IK = ik
    backend.plan_qpos = rrt
    planner = SimpleNamespace(_mplib=backend, _pad_qpos=lambda q: q.copy(),
                              _to_mplib_pose=Pose.from_pose7, plan_screw_path=screw)
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), get_arm_qpos=lambda: HOME.copy(),
                          entity=SimpleNamespace(set_qpos=lambda *_: pytest.fail("simulator qpos mutated")))
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: arm),
                           config={}, TABLE_TOP_Z=.765, _R_DOWN=BlocksRankingRGB._R_DOWN,
                           execute_plan=lambda *_: pytest.fail("preflight executed a trajectory"))
    return task, arm, calls


def test_backward_caches_exact_joint_transit_and_reversed_target_queue():
    task, arm, calls = fixture()
    result = bg._plan_block_grasp_backward(task, HOME, POINTS, "right")
    assert result is not None and result.yaw == 0
    assert len(calls["ik"]) == 1 and len(calls["screw"]) == 2 and len(calls["rrt"]) == 1
    assert calls["ik"][0]["n_init_qpos"] == 20
    assert calls["rrt"][0][2]["fix_joint_limits"] is False
    for leg, target in zip(result.legs, POINTS):
        np.testing.assert_allclose(leg[0][:3], target)
    np.testing.assert_allclose(result.legs[0][1].position[-1], result.legs[1][1].position[0])
    np.testing.assert_allclose(result.legs[1][1].position[-1], result.legs[2][1].position[0])
    np.testing.assert_allclose(result.legs[1][1].velocity, -np.tile(np.arange(7.), (3, 1)))
    executed = []
    task.execute_plan = lambda plan, tag: executed.append(plan)
    targets = [leg[0].copy() for leg in result.legs]
    for target in targets:
        bg.execute_grasp_leg(task, result, target, "right")
    assert len(executed) == 3 and result.legs == []


def test_reverse_velocity_is_negated_and_input_arrays_are_preserved():
    source = PlanResult(True, np.arange(21.).reshape(3, 7), np.arange(21., 42.).reshape(3, 7))
    reversed_plan = bg._reverse_block_plan(source)
    np.testing.assert_array_equal(reversed_plan.position, source.position[::-1])
    np.testing.assert_array_equal(reversed_plan.velocity, -source.velocity[::-1])
    assert not np.shares_memory(reversed_plan.position, source.position)
    assert not np.shares_memory(reversed_plan.velocity, source.velocity)


@pytest.mark.parametrize("invalid", [np.full(7, np.nan), np.zeros(6), np.full(7, 11.)])
def test_invalid_ik_candidates_fail_closed_without_screw_or_execution(invalid):
    task, arm, calls = fixture()
    arm.planner._mplib.IK = lambda *_a, **_k: ("Success", [invalid])
    assert bg._plan_block_grasp_backward(task, HOME, POINTS, "right") is None
    assert calls["screw"] == [] and calls["rrt"] == []


def test_unsupported_backend_fails_closed():
    task, arm, _ = fixture()
    arm.planner = SimpleNamespace()
    assert bg._plan_block_grasp_backward(task, HOME, POINTS, "right") is None


def test_backward_ik_and_solution_search_are_bounded():
    task, arm, calls = fixture()

    def ik(goal, qpos, **kwargs):
        calls["ik"].append(kwargs)
        return "Success", [goal.to_pose7() for _ in range(20)]

    arm.planner._mplib.IK = ik
    arm.planner.plan_screw_path = lambda *_a, **_k: calls["screw"].append(1) or PlanResult(False)
    assert bg._plan_block_grasp_backward(task, HOME, POINTS, "right") is None
    assert len(calls["ik"]) == 4 and all(call["n_init_qpos"] == 20 for call in calls["ik"])
    assert len(calls["screw"]) == 4 * 8 and calls["rrt"] == []


@pytest.mark.parametrize("failure", ["screw", "endpoint", "collision", "joint_limit", "lateral", "rotation", "below", "nan", "velocity", "start", "rrt", "rrt_joint_arrival"])
def test_invalid_continuation_never_returns_executable_queue(failure):
    task, arm, calls = fixture()
    original_screw = arm.planner.plan_screw_path
    original_rrt = arm.planner._mplib.plan_qpos

    def screw(qpos, target, **kwargs):
        plan = original_screw(qpos, target, **kwargs)
        if failure == "screw":
            return PlanResult(False)
        if failure == "endpoint":
            plan.position[-1, 0] += .02
        elif failure == "joint_limit":
            plan.position[1, 0] = 11.
        elif failure == "lateral":
            plan.position[1, 0] += .011
        elif failure == "rotation":
            plan.position[1, 3:7] = [1, 0, 0, 0]
        elif failure == "below":
            plan.position[1, 2] = min(qpos[2], target[2]) - .011
        elif failure == "nan":
            plan.position[1, 0] = np.nan
        elif failure == "velocity":
            plan.velocity[1, 0] = np.nan
        elif failure == "start":
            plan.position[0, 0] += .0001
        return plan

    def rrt(goals, qpos, **kwargs):
        result = original_rrt(goals, qpos, **kwargs)
        if failure == "rrt":
            return {"status": "Fail"}
        result["position"][-1, 0] += .001  # FK-valid but wrong exact joint arrival.
        return result

    arm.planner.plan_screw_path = screw
    if failure == "collision":
        arm.planner._mplib.planning_world.is_state_colliding = lambda: True
    if failure.startswith("rrt"):
        arm.planner._mplib.plan_qpos = rrt
    assert bg._plan_block_grasp_backward(task, HOME, POINTS, "right") is None


@pytest.mark.parametrize("enabled,first_success", [(False, False), (True, True), (True, False)])
def test_original_attempt_order_and_default_behavior_are_preserved(monkeypatch, enabled, first_success):
    task, arm, _ = fixture()
    task.config["block_grasp_backward_preflight"] = enabled
    approaches, refreshes, fallback = [], [], []
    arm.planner.plan_path = lambda *_a, **_k: approaches.append(1) or PlanResult(first_success, np.ones((2, 7)))
    arm.planner.plan_screw_path = lambda *_a, **_k: PlanResult(True, np.ones((2, 7)))
    monkeypatch.setattr(bg, "endpoint_matches", lambda _p, plan, _t: plan.success)
    monkeypatch.setattr(bg, "refresh_block_obstacles", lambda *_a: refreshes.append(1))
    recovery = bg.BlockGraspPlan(0., [])

    def backward(*_args):
        assert len(approaches) == 8
        fallback.append(1)
        return recovery

    monkeypatch.setattr(bg, "_plan_block_grasp_backward", backward)
    result = bg.plan_block_grasp(task, [0, 0, .795], .03, .965, "right")
    assert refreshes == [1]
    assert len(approaches) == (1 if first_success else 8)
    assert bool(fallback) is (enabled and not first_success)
    assert (result is recovery) is (enabled and not first_success)
