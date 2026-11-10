"""Block-only MPLib motion guards preserve acquisition, PD timing and grip."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.manipulation import TopDownPickPlaceMixin
from envs.planning.base import PlanResult
from envs.task_bases import block_grasp as bg
from envs.task_bases import blocks_ranking_rgb as core
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.task_bases.blocks_ranking_size import BlocksRankingSize
from envs.utils import Pose


TARGET = np.array([.1, -.2, .9, 1., 0., 0., 0.])


@pytest.fixture
def motion():
    events = []
    state = {"target": TARGET.copy(), "fk_offset": np.zeros(3),
             "measured": TARGET[:3].copy(), "fk_quat": None}

    def forbidden(*_args, **_kwargs):
        pytest.fail("Genesis IK or simulator pose mutation is forbidden")

    def fk_pose(_index):
        q = state["target"][3:] if state["fk_quat"] is None else state["fk_quat"]
        return Pose(state["target"][:3] + state["fk_offset"], q)

    def solve(seed, target, **kwargs):
        events.append(("ik", seed.copy(), target.copy(), kwargs))
        state["target"] = target.copy()
        return PlanResult(True, np.stack([seed, seed + .01]))

    def rrt(**kwargs):
        events.append(("rrt", kwargs))
        return {"status": "Success", "position": np.stack([
            kwargs["current_qpos"], kwargs["goal_qposes"][0],
        ])}

    def fallback(current, target, **kwargs):
        events.append(("fallback", current.copy(), target.copy(), kwargs))
        state["target"] = target.copy()
        return PlanResult(True, np.stack([current, current + .02]))

    backend = SimpleNamespace(
        plan_qpos=rrt, robot=SimpleNamespace(get_base_pose=Pose),
        pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q: events.append(("fk", q.copy())),
            get_link_pose=fk_pose,
        ),
        move_group="hand", link_name_2_idx={"hand": 0},
    )
    planner = SimpleNamespace(
        _mplib=backend, solve_ik=solve, plan_path=fallback,
        _pad_qpos=lambda q: q.copy(), _to_mplib_pose=Pose.from_pose7,
        update_obstacles=lambda points, **kwargs: events.append(("cloud", points.copy())),
    )
    arm = SimpleNamespace(
        planner=planner, n_arm=7, get_arm_qpos=lambda: np.zeros(7),
        tcp_offset=Pose(), _cached_target=np.r_[np.zeros(7), .017, .019],
        arm_joints=[SimpleNamespace(dof_idx_local=i) for i in range(7)],
        gripper_val=.5,
        ee_link=SimpleNamespace(get_pos=lambda: state["measured"].copy()),
        entity=SimpleNamespace(
            inverse_kinematics=forbidden, set_qpos=forbidden,
            get_qpos=lambda: np.r_[np.zeros(7), .034, .035],
            control_dofs_position=lambda q: events.append(("pd", q.copy())),
        ),
    )
    task = BlocksRankingRGB.__new__(BlocksRankingRGB)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.avatar_collider = None
    task.config = {}
    task.plan_success = True
    task._active_block_grasp_plan = None
    task.execute_plan = lambda plan, tag: events.append(("execute", plan, tag))
    task.step_sim = lambda: events.append(("step",))
    return task, arm, state, events


def names(events):
    return [event[0] for event in events]


def test_tiny_lift_uses_cached_arm_seed_and_no_simulator_mutation(motion):
    task, arm, _state, events = motion
    arm._cached_target[:7] = .12
    saved = arm._cached_target.copy()
    goal = task._solve_ik(Pose.from_pose7(TARGET), "right")
    np.testing.assert_allclose(goal, np.full(7, .13))
    np.testing.assert_array_equal(events[1][1], saved[:7])
    np.testing.assert_array_equal(arm._cached_target, saved)
    assert names(events) == ["cloud", "ik", "fk"]
    assert task.plan_success


@pytest.mark.parametrize("fault", ["failed", "position_fk", "rotation_fk", "branch", "shape", "nan", "velocity"])
def test_tiny_lift_rejects_before_any_pd_or_execution(motion, fault):
    task, arm, state, events = motion
    original = arm.planner.solve_ik

    def bad(seed, target, **kwargs):
        plan = original(seed, target, **kwargs)
        if fault == "failed":
            plan.success = False
        elif fault == "position_fk":
            state["fk_offset"][0] = .0101
        elif fault == "rotation_fk":
            state["fk_quat"] = [np.cos(.026), np.sin(.026), 0, 0]
        elif fault == "branch":
            plan.position[-1, 0] = seed[0] + .301
        elif fault == "shape":
            plan.position = np.zeros((2, 6))
        elif fault == "nan":
            plan.position[0, 0] = np.nan
        else:
            plan.velocity[0, 0] = np.inf
        return plan

    arm.planner.solve_ik = bad
    with pytest.raises(RuntimeError, match="block MPLib tiny lift rejected"):
        task._solve_ik(Pose.from_pose7(TARGET), "right")
    assert not task.plan_success
    assert not set(names(events)) & {"pd", "execute", "step", "fallback"}


def test_rejected_cartesian_waypoint_aborts_into_existing_block_recovery(motion):
    task, arm, _state, events = motion
    arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)
    task.open_gripper = lambda tag: events.append(("open", tag))
    task._pick_and_place_block = lambda *_a: task._move_cartesian(
        np.array([0., 0., .8]), np.array([0., 0., .82]), "right", n_steps=3, sim_per_step=2,
    )
    assert task._place_block_with_recovery("red") is False
    assert names(events).count("cloud") == 1
    assert names(events).count("open") == 1
    assert names(events).count("step") == task.FAILURE_RECOVERY_STEPS
    assert "pd" not in names(events)


def test_tiny_lift_retains_inherited_pd_timing_and_commanded_finger_targets(motion):
    task, arm, _state, events = motion
    task._move_cartesian(np.array([0., 0., .8]), np.array([0., 0., .82]),
                         "right", n_steps=3, sim_per_step=2)
    assert names(events).count("ik") == 3
    assert names(events).count("cloud") == 3
    assert names(events).count("pd") == 3
    assert names(events).count("step") == 6
    for event in events:
        if event[0] == "pd":
            np.testing.assert_array_equal(event[1][7:], [.017, .019])
    np.testing.assert_array_equal(arm._cached_target[7:], [.017, .019])
    assert "execute" not in names(events)


def test_seeded_motion_uses_checked_mplib_joint_rrt_and_preserves_grip(motion):
    task, arm, _state, events = motion
    plan = task._move_seeded(TARGET, "right")
    assert plan is not None and task.plan_success
    assert names(events) == ["cloud", "ik", "fk", "rrt", "fk", "fk", "execute"]
    np.testing.assert_array_equal(arm._cached_target[7:], [.017, .019])
    assert "pd" not in names(events) and "step" not in names(events)


@pytest.mark.parametrize("reason", ["ik_failed", "branch", "rrt_failed", "rrt_exception", "rrt_fk"])
def test_seeded_fallback_is_mplib_and_fk_checked_before_execution(motion, reason):
    task, arm, state, events = motion
    original = arm.planner.solve_ik

    def solve(seed, target, **kwargs):
        plan = original(seed, target, **kwargs)
        if reason == "ik_failed":
            plan.success = False
        elif reason == "branch":
            plan.position[-1] = seed + 2.
        return plan

    arm.planner.solve_ik = solve
    rrt = arm.planner._mplib.plan_qpos

    def rejected_rrt(**kwargs):
        result = rrt(**kwargs)
        if reason == "rrt_failed":
            return {"status": "Failure"}
        if reason == "rrt_exception":
            raise ValueError("no RRT")
        state["fk_offset"][0] = .02
        return result

    if reason.startswith("rrt"):
        arm.planner._mplib.plan_qpos = rejected_rrt
    fallback = arm.planner.plan_path

    def checked_fallback(*args, **kwargs):
        state["fk_offset"][:] = 0
        return fallback(*args, **kwargs)

    arm.planner.plan_path = checked_fallback
    assert task._move_seeded(TARGET, "right") is not None
    assert names(events).count("fallback") == 1
    assert names(events).count("execute") == 1
    assert names(events)[-2:] == ["fk", "execute"]


@pytest.mark.parametrize("fault", ["position", "rotation", "nan", "shape", "velocity", "failed"])
def test_bad_seeded_fallback_never_executes(motion, fault):
    task, arm, state, events = motion
    arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)
    fallback = arm.planner.plan_path

    def bad(*args, **kwargs):
        plan = fallback(*args, **kwargs)
        if fault == "position":
            state["fk_offset"][0] = .02
        elif fault == "rotation":
            state["fk_quat"] = [0., 1., 0., 0.]
        elif fault == "nan":
            plan.position[0, 0] = np.nan
        elif fault == "shape":
            plan.position = np.zeros(7)
        elif fault == "velocity":
            plan.velocity = np.zeros((1, 7))
        else:
            plan.success = False
        return plan

    arm.planner.plan_path = bad
    assert task._move_seeded(TARGET, "right") is None
    assert not task.plan_success and "execute" not in names(events)


@pytest.mark.parametrize("route", ["rrt", "fallback"])
@pytest.mark.parametrize("measured", [TARGET[:3] + [.051, 0, 0], [np.nan, 0, 0], [0, 0]])
def test_all_seeded_routes_reject_bad_physical_arrival(motion, route, measured):
    task, arm, state, events = motion
    state["measured"] = np.asarray(measured)
    if route == "fallback":
        arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)
    assert task._move_seeded(TARGET, "right") is None
    assert not task.plan_success and names(events).count("execute") == 1


@pytest.mark.parametrize("fault", ["target_nan", "target_shape", "quaternion", "current_nan", "current_shape"])
def test_bad_seeded_inputs_never_reach_planner(motion, fault):
    task, arm, _state, events = motion
    target = TARGET.copy()
    if fault == "target_nan":
        target[0] = np.nan
    elif fault == "target_shape":
        target = target[None, :]
    elif fault == "quaternion":
        target[3:] = 0
    elif fault == "current_nan":
        arm.get_arm_qpos = lambda: np.full(7, np.nan)
    else:
        arm.get_arm_qpos = lambda: np.zeros(6)
    assert task._move_seeded(target, "right") is None
    assert not task.plan_success and events == []


@pytest.mark.parametrize("bad_seed", [np.zeros(6), np.zeros((1, 9)), np.r_[np.zeros(7), np.nan, 0]])
def test_bad_tiny_lift_seed_aborts_before_planning(motion, bad_seed):
    task, arm, _state, events = motion
    arm._cached_target = bad_seed
    with pytest.raises(RuntimeError):
        task._solve_ik(Pose.from_pose7(TARGET), "right")
    assert not task.plan_success and events == []


def test_existing_preflight_queue_bypasses_new_seeded_motion(motion, monkeypatch):
    task, _arm, _state, events = motion
    plan = PlanResult(True, np.zeros((2, 7)))
    task._active_block_grasp_plan = bg.BlockGraspPlan(0., [(TARGET.copy(), plan)])
    monkeypatch.setattr(core, "move_block_seeded", lambda *_a: pytest.fail("queued path replanned"))
    assert task._move_seeded(TARGET, "right") is plan
    assert names(events) == ["execute"]
    assert task._active_block_grasp_plan.legs == []


def test_failed_active_pickup_screw_fallback_aborts_before_next_stage(motion):
    task, arm, _state, events = motion
    task._active_block_grasp_plan = bg.BlockGraspPlan(0., [], closed=True)
    arm.planner._mplib.plan_screw = lambda **_kwargs: {"status": "Failure"}
    arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)
    arm.planner.plan_path = lambda *_a, **_k: PlanResult(False)
    task.open_gripper = lambda tag: events.append(("open", tag))

    def pickup(*_args):
        task._move_screw([0., 0., 1.], "right")
        events.append(("next_stage",))
        return True

    task._pick_and_place_block = pickup
    assert task._place_block_with_recovery("red") is False
    assert "next_stage" not in names(events) and "execute" not in names(events)
    assert names(events).count("open") == 1
    assert names(events).count("step") == task.FAILURE_RECOVERY_STEPS
    assert not task.plan_success


@pytest.mark.parametrize("active", [False, True])
def test_post_close_screw_refreshes_block_cloud_before_existing_delegation(motion, monkeypatch, active):
    task, _arm, _state, events = motion
    if active:
        task._active_block_grasp_plan = bg.BlockGraspPlan(0., [], closed=True)
    monkeypatch.setattr(TopDownPickPlaceMixin, "_move_screw",
                        lambda *_a, **_k: events.append(("screw",)) or True)
    assert task._move_screw([0, 0, 1], "right") is True
    assert names(events) == ["cloud", "screw"]


def test_tiny_and_seeded_motions_use_neutral_avatar_prop_refresh(motion):
    task, _arm, _state, events = motion
    task._refresh_avatar_planner_obstacles = lambda tag: events.append(("neutral_cloud", tag))
    task._solve_ik(Pose.from_pose7(TARGET), "right")
    task._move_seeded(TARGET, "right")
    assert names(events).count("neutral_cloud") == 2
    assert "cloud" not in names(events)


def test_size_family_inherits_mplib_overrides_without_enabling_backward_recovery():
    assert BlocksRankingSize._solve_ik is BlocksRankingRGB._solve_ik
    assert BlocksRankingSize._move_seeded is BlocksRankingRGB._move_seeded
    assert BlocksRankingRGB.BLOCK_GRASP_PREFLIGHT is True
    assert BlocksRankingRGB({"setting": "sim"}).config.get("block_grasp_backward_preflight", False) is False
