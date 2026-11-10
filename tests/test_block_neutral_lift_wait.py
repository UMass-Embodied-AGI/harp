"""Bounded physical waits must retain lift rejection and finger commands."""

from types import MethodType

import numpy as np
import pytest

from envs.robot.franka_robot import FrankaArm
from envs.tasks.rank_blocks_by_color_copresence import RankBlocksByColorCopresence
from envs.tasks.rank_blocks_by_size_copresence import RankBlocksBySizeCopresence
from envs.utils import Pose
from test_block_mplib_motion import TARGET, motion as _motion_fixture, names


# Register the imported fixture under the name requested by waiting.
motion = _motion_fixture


@pytest.fixture
def waiting(motion):
    task, arm, state, events = motion
    task.BLOCK_CAN_LIFT_WAIT_RECOVERY = True
    task._neutral_avatar_started = True
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl"}
    task.avatar_collided = False
    for name in ("set_arm_joints", "_get_target", "_build_qpos_with_arm"):
        setattr(arm, name, MethodType(getattr(FrankaArm, name), arm))
    return motion


def install_branch(arm, *, count=None):
    original = arm.planner.solve_ik
    calls = []

    def solve(seed, target, **kwargs):
        result = original(seed, target, **kwargs)
        calls.append(seed.copy())
        if count is None or len(calls) <= count:
            result.position[-1, 0] = seed[0] + .301
        return result

    arm.planner.solve_ik = solve
    return calls


def lift(task):
    return task._solve_ik(Pose.from_pose7(TARGET), "right")


def test_neutral_classes_opt_in():
    assert RankBlocksByColorCopresence.BLOCK_CAN_LIFT_WAIT_RECOVERY
    assert RankBlocksBySizeCopresence.BLOCK_CAN_LIFT_WAIT_RECOVERY


def test_accepted_lift_keeps_original_query_order(waiting):
    task, arm, _state, events = waiting
    saved = arm._cached_target.copy()
    np.testing.assert_allclose(lift(task), np.full(7, .01))
    assert names(events) == ["cloud", "ik", "fk"]
    np.testing.assert_array_equal(saved, arm._cached_target)


@pytest.mark.parametrize("scope", ["default", "disabled", "unstarted", "sponge", "missing_job"])
def test_wait_is_limited_to_opted_in_started_can_work(waiting, scope):
    task, arm, _state, events = waiting
    if scope == "default":
        del task.BLOCK_CAN_LIFT_WAIT_RECOVERY
    elif scope == "disabled":
        task.BLOCK_CAN_LIFT_WAIT_RECOVERY = False
    elif scope == "unstarted":
        task._neutral_avatar_started = False
    elif scope == "sponge":
        task._neutral_avatar_job = {"kind": "sponge"}
    else:
        task._neutral_avatar_job = None
    calls = install_branch(arm)
    with pytest.raises(RuntimeError, match="branch jump"):
        lift(task)
    assert len(calls) == 1
    assert not {"pd", "step"}.intersection(names(events))
    assert not task.plan_success


def test_wait_uses_real_pd_api_preserves_fingers_and_refreshes_obstacles(waiting):
    task, arm, _state, events = waiting
    arm._cached_target[:7] = .02
    calls = install_branch(arm, count=1)
    goal = lift(task)
    assert len(calls) == 2
    np.testing.assert_allclose(calls, [np.full(7, .02), np.zeros(7)])
    np.testing.assert_allclose(goal, np.full(7, .01))
    assert names(events) == ["cloud", "ik", "fk", "pd"] + ["step"] * 30 + ["cloud", "ik", "fk"]
    np.testing.assert_array_equal(events[3][1], np.r_[np.zeros(7), .017, .019])
    assert arm.gripper_val == .5
    assert task.plan_success


def test_wait_budget_is_finite(waiting):
    task, arm, _state, events = waiting
    calls = install_branch(arm)
    with pytest.raises(RuntimeError, match="branch jump"):
        lift(task)
    assert len(calls) == 9
    assert names(events).count("step") == 240
    assert names(events).count("pd") == 8
    assert names(events).count("cloud") == 9
    assert not task.plan_success


@pytest.mark.parametrize("fault", ["failed", "position_fk", "rotation_fk", "shape", "nan", "velocity"])
def test_invalid_initial_plan_still_fails_without_wait(waiting, fault):
    task, arm, state, events = waiting
    original = arm.planner.solve_ik

    def bad(seed, target, **kwargs):
        result = original(seed, target, **kwargs)
        if fault == "failed":
            result.success = False
        elif fault == "position_fk":
            state["fk_offset"][0] = .0101
        elif fault == "rotation_fk":
            state["fk_quat"] = [np.cos(.026), np.sin(.026), 0, 0]
        elif fault == "shape":
            result.position = np.zeros((2, 6))
        elif fault == "nan":
            result.position[-1, 0] = np.nan
        else:
            result.velocity[-1, 0] = np.inf
        return result

    arm.planner.solve_ik = bad
    with pytest.raises(RuntimeError, match="IK/FK rejected"):
        lift(task)
    assert names(events).count("ik") == 1
    assert not {"pd", "step"}.intersection(names(events))
    assert not task.plan_success


@pytest.mark.parametrize("already_collided", [False, True])
def test_collision_aborts_wait_and_retains_failure(waiting, already_collided):
    task, arm, _state, events = waiting
    task.avatar_collided = already_collided
    install_branch(arm)

    def step():
        events.append(("step",))
        task.avatar_collided = True

    task.step_sim = step
    with pytest.raises(RuntimeError, match="avatar collision"):
        lift(task)
    assert names(events).count("step") == (0 if already_collided else 1)
    assert names(events).count("ik") == 1
    assert task.avatar_collided and not task.plan_success


def test_recovery_does_not_clear_prior_plan_failure(waiting):
    task, arm, _state, _events = waiting
    task.plan_success = False
    install_branch(arm, count=1)
    lift(task)
    assert not task.plan_success


def test_reseeding_cannot_bypass_initial_continuity_bound(waiting):
    task, arm, _state, events = waiting
    arm.get_arm_qpos = lambda: np.full(7, .20)
    original = arm.planner.solve_ik

    def solve(seed, target, **kwargs):
        result = original(seed, target, **kwargs)
        result.position[-1] = .35
        return result

    arm.planner.solve_ik = solve
    with pytest.raises(RuntimeError, match="branch jump"):
        lift(task)
    assert names(events).count("ik") == 9
    assert not task.plan_success


def test_retry_still_rejects_failed_ik(waiting):
    task, arm, _state, events = waiting
    calls = install_branch(arm, count=1)
    original = arm.planner.solve_ik

    def solve(seed, target, **kwargs):
        result = original(seed, target, **kwargs)
        if len(calls) == 2:
            result.success = False
        return result

    arm.planner.solve_ik = solve
    with pytest.raises(RuntimeError, match="IK/FK rejected"):
        lift(task)
    assert len(calls) == 2
    assert names(events).count("step") == 30
    assert not task.plan_success
