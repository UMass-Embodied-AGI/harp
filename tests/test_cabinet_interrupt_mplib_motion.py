"""Cabinet-native MPLib correction: control, rejection, and dispatch contracts."""

from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.manipulation import PickSpec, PlaceSpec, TopDownPickPlaceMixin
from envs.planning.base import PlanResult
from envs.robot.franka_robot import FrankaArm
from envs.task_bases import put_object_cabinet as core
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.task_bases.put_object_cabinet import CabinetTaskCore
from envs.tasks.put_object_in_cabinet_contribution import PutObjectInCabinetContribution
from envs.tasks.put_object_in_cabinet_interruption import (
    PutObjectInCabinetInterruption, _CabinetMotionRejected,
)
from envs.tasks.put_object_in_cabinet_copresence import PutObjectInCabinetCopresence
from envs.utils import Pose


LINK_TARGET = np.array([0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0])


def _names(events):
    return [event[0] for event in events]


@pytest.fixture
def motion():
    events = []
    state = {
        "target": LINK_TARGET.copy(), "fk_offset": np.zeros(3),
        "fk_quat": None, "measured": None,
        "qpos": np.r_[np.zeros(7), 0.034, 0.035],
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail("Genesis IK, simulator teleport, and pose fallbacks are forbidden")

    def control(qpos):
        events.append(("pd", qpos.copy()))
        state["qpos"][:7] = qpos[:7]

    arm = FrankaArm(SimpleNamespace(
        inverse_kinematics=forbidden, set_qpos=forbidden,
        get_qpos=lambda: state["qpos"].copy(), control_dofs_position=control,
    ), Pose([0.05, -0.35, 0.4]))
    arm.arm_joints = [SimpleNamespace(dof_idx_local=i) for i in range(7)]
    arm.finger_joints = [SimpleNamespace(dof_idx_local=i) for i in (7, 8)]
    arm._cached_target = np.r_[np.zeros(7), 0.020, 0.020]
    arm.gripper_val = 0.5

    def measured():
        if state["measured"] is not None:
            return state["measured"]
        return (Pose.from_pose7(state["target"].copy()) * arm.tcp_offset).to_pose7()

    arm.get_ee_pose = measured

    def fk_pose(_index):
        quaternion = state["target"][3:] if state["fk_quat"] is None else state["fk_quat"]
        world = Pose(state["target"][:3] + state["fk_offset"], np.array(quaternion))
        return arm.origin_pose.inv() * world

    def solve(seed, target, **kwargs):
        events.append(("ik", seed.copy(), target.copy(), kwargs))
        state["target"] = target.copy()
        return PlanResult(True, np.stack([seed, seed + 0.01]))

    def rrt(**kwargs):
        events.append(("rrt", kwargs))
        return {"status": "Success", "position": np.stack([
            kwargs["current_qpos"], kwargs["goal_qposes"][0],
        ])}

    backend = SimpleNamespace(
        plan_qpos=rrt, move_group_link_id=0,
        pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q: events.append(("fk", q.copy())),
            get_link_pose=fk_pose,
        ),
    )
    arm.planner = SimpleNamespace(
        _mplib=backend, solve_ik=solve, plan_path=forbidden,
        _pad_qpos=lambda q: q.copy(), _to_mplib_pose=Pose.from_pose7,
        update_obstacles=lambda points, **kw: events.append(("points", points.copy(), kw)),
    )
    task = PutObjectInCabinetInterruption.__new__(PutObjectInCabinetInterruption)
    task.robot = SimpleNamespace(
        get_arm=lambda _tag: arm,
        set_arm_joints=lambda q, _tag: arm.set_arm_joints(q),
        set_gripper=lambda value, _tag: arm.set_gripper(value),
        teleport_arm_joints=forbidden,
    )
    task.config = {}
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    task.TABLE_TOP_Z = 0.765  # __new__ skips BaseTask's normal table initialization.
    task.plan_success = True
    task._refresh_avatar_planner_obstacles = lambda tag: events.append(("cloud", tag))
    task.step_sim = lambda: events.append(("step",))

    def execute(plan, tag):
        events.append(("execute", plan, tag))
        state["qpos"][:7] = plan.position[-1]

    task.execute_plan = execute
    return task, arm, state, events


def test_seeded_reuses_native_driver_with_exact_link_tcp_conversion(motion):
    task, arm, _state, events = motion
    arm.tcp_offset = Pose([0.01, -0.02, 0.1034], [np.cos(0.1), 0, np.sin(0.1), 0])
    target = LINK_TARGET.copy()
    original = target.copy()
    assert task._move_seeded(target, "right") is True
    np.testing.assert_array_equal(target, original)
    np.testing.assert_allclose(events[1][2], original, atol=1e-12)
    assert events[1][3] == {"num_waypoints": 2, "log": False}
    assert _names(events) == ["cloud", "ik", "rrt", "fk", "execute"]
    rrt = events[2][1]
    assert rrt["time_step"] == task.scene.sim.dt
    assert rrt["planning_time"] == 10 and rrt["rrt_range"] == 0.15
    np.testing.assert_array_equal(arm._cached_target[7:], [0.020, 0.020])
    np.testing.assert_array_equal(arm._cached_target[:7], np.full(7, 0.01))
    assert task.plan_success


def test_native_execution_holds_commanded_fingers_without_added_settle(motion, monkeypatch):
    task, arm, _state, events = motion
    monkeypatch.delenv("DEBUG_TELEPORT", raising=False)
    monkeypatch.delenv("USE_SETQPOS", raising=False)
    task.execute_plan = BaseTask.execute_plan.__get__(task)
    task._apply_avatar_retreat_if_needed = lambda _tags: False
    assert task._move_seeded(LINK_TARGET, "right") is True
    assert _names(events).count("step") == 2
    assert _names(events).count("pd") == 4  # arm and gripper targets each step
    for event in events:
        if event[0] == "pd":
            np.testing.assert_array_equal(event[1][7:], [0.020, 0.020])
    np.testing.assert_array_equal(arm._cached_target[7:], [0.020, 0.020])


@pytest.mark.parametrize("fault", [
    "ik_none", "ik_failed", "ik_exception", "ik_empty", "ik_shape", "ik_nan",
    "ik_ragged", "ik_velocity", "branch", "rrt_failed", "rrt_exception",
    "rrt_empty", "rrt_shape", "rrt_nan", "rrt_velocity", "rrt_ragged",
    "fk_position", "fk_rotation", "fk_nan_position", "fk_nan_quaternion", "fk_shape",
    "fk_none", "fk_missing_fields",
])
def test_native_rejections_do_not_drive_or_fall_back(motion, fault):
    task, arm, state, events = motion
    solve = arm.planner.solve_ik
    rrt = arm.planner._mplib.plan_qpos

    def bad_solve(*args, **kwargs):
        plan = solve(*args, **kwargs)
        if fault == "ik_none":
            return None
        if fault == "ik_failed":
            plan.success = False
        elif fault == "ik_exception":
            raise RuntimeError("backend rejected IK")
        elif fault == "ik_empty":
            plan.position = np.empty((0, 7))
        elif fault == "ik_shape":
            plan.position = np.zeros((2, 6))
        elif fault == "ik_nan":
            plan.position[0, 0] = np.nan
        elif fault == "ik_ragged":
            plan.position = [[0] * 7, [0] * 6]
        elif fault == "ik_velocity":
            plan.velocity[0, 0] = np.inf
        elif fault == "branch":
            plan.position[-1, 0] = 3.501
        return plan

    def bad_rrt(**kwargs):
        result = rrt(**kwargs)
        if fault == "rrt_failed":
            return {"status": "Collision"}
        if fault == "rrt_exception":
            raise RuntimeError("backend rejected RRT")
        if fault == "rrt_empty":
            result["position"] = np.empty((0, 7))
        elif fault == "rrt_shape":
            result["position"] = np.zeros(7)
        elif fault == "rrt_nan":
            result["position"][0, 0] = np.nan
        elif fault == "rrt_velocity":
            result["velocity"] = np.zeros((1, 7))
        elif fault == "rrt_ragged":
            result["position"] = [[0] * 7, [0] * 6]
        return result

    arm.planner.solve_ik = bad_solve
    arm.planner._mplib.plan_qpos = bad_rrt
    if fault == "fk_position":
        state["fk_offset"][0] = 0.0301
    elif fault == "fk_rotation":
        state["fk_quat"] = [np.cos(np.deg2rad(5.01)), 0, 0, np.sin(np.deg2rad(5.01))]
    elif fault.startswith("fk_nan") or fault == "fk_shape":
        endpoint = SimpleNamespace(p=np.zeros(3), q=np.array([1., 0, 0, 0]))
        if fault == "fk_nan_position":
            endpoint.p[0] = np.nan
        elif fault == "fk_nan_quaternion":
            endpoint.q[0] = np.nan
        else:
            endpoint.p = np.zeros(2)
        task._drawer_plan_endpoint_tcp = lambda *_args: endpoint
    elif fault in ("fk_none", "fk_missing_fields"):
        task._drawer_plan_endpoint_tcp = lambda *_args: None if fault == "fk_none" else object()
    assert task._move_seeded(LINK_TARGET, "right") is None
    assert not task.plan_success
    assert not set(_names(events)) & {"execute", "pd", "step"}
    if fault.startswith("ik_") or fault == "branch":
        assert "rrt" not in _names(events)


@pytest.mark.parametrize("fault", [
    "target_nan", "target_shape", "target_nonunit", "target_zero_quat", "target_string",
    "current_nan", "current_shape", "current_string",
])
def test_bad_seeded_inputs_never_reach_planning(motion, fault):
    task, arm, _state, events = motion
    target = LINK_TARGET.copy()
    if fault == "target_nan":
        target[0] = np.nan
    elif fault == "target_shape":
        target = target[None, :]
    elif fault == "target_nonunit":
        target[3] = 2
    elif fault == "target_zero_quat":
        target[3:] = 0
    elif fault == "target_string":
        target = ["bad"] * 7
    elif fault == "current_nan":
        arm.get_arm_qpos = lambda: np.full(7, np.nan)
    elif fault == "current_shape":
        arm.get_arm_qpos = lambda: np.zeros(6)
    else:
        arm.get_arm_qpos = lambda: ["bad"] * 7
    assert task._move_seeded(target, "right") is None
    assert events == [] and not task.plan_success


@pytest.mark.parametrize("measured", [
    [0.151, -0.20, 1.0034, 1, 0, 0, 0],
    [0.10, -0.20, 1.0544, 1, 0, 0, 0],
    [np.nan, 0, 0, 1, 0, 0, 0], [0, 0, 0, np.nan, 0, 0, 0],
    [0, 0, 0], ["bad"] * 7, [0.10, -0.20, 1.0034, 0, 0, 0, 0],
    [0.10, -0.20, 1.0034, 2, 0, 0, 0],
])
def test_native_rejects_bad_measured_arrival_after_only_one_execution(motion, measured):
    task, _arm, state, events = motion
    state["measured"] = measured
    assert task._move_seeded(LINK_TARGET, "right") is None
    assert _names(events).count("execute") == 1
    assert "step" not in _names(events) and not task.plan_success


def test_drawer_rejects_unbounded_ik_branch_before_execution(motion):
    task, arm, _state, events = motion
    solve = arm.planner.solve_ik

    def distant(*args, **kwargs):
        plan = solve(*args, **kwargs)
        plan.position[-1, 0] = 4.0
        return plan

    arm.planner.solve_ik = distant
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    assert not CabinetTaskCore._drive_to_tcp(task, tcp.p, "right", n_settle=2, tcp_pose=tcp)
    assert not set(_names(events)) & {"rrt", "execute", "pd", "step"}


def test_drawer_uses_physical_clock_and_preserves_endpoint_settle(motion):
    task, arm, _state, events = motion
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    assert task._drive_to_tcp(tcp.p, "right", n_settle=2, tcp_pose=tcp)
    assert _names(events).count("execute") == 1
    assert _names(events).count("step") == 2
    assert _names(events).count("pd") == 4
    assert events[2][1]["time_step"] == .002
    assert task._cabinet_plan_is_physical_clock is False


@pytest.mark.parametrize("fault", ["start_jump", "middle_jump", "end_branch", "fk", "arrival", "orientation"])
def test_drawer_rejects_discontinuous_routes_and_bad_arrival(motion, fault):
    task, arm, state, events = motion
    rrt = arm.planner._mplib.plan_qpos

    def bad_rrt(**kwargs):
        result = rrt(**kwargs)
        if fault in ("start_jump", "middle_jump", "end_branch"):
            pos = np.vstack([np.zeros(7), np.zeros(7), np.full(7, .01)])
            if fault == "start_jump":
                pos[0, 0] = .301
            elif fault == "middle_jump":
                pos[1, 0] = .301
            else:
                pos = np.linspace(np.zeros(7), np.full(7, 1.4), 20)
            result["position"] = pos
        return result

    arm.planner._mplib.plan_qpos = bad_rrt
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    if fault == "fk":
        state["fk_offset"][0] = .0151
    elif fault == "arrival":
        state["measured"] = tcp.to_pose7()
        state["measured"][0] += .0151
    elif fault == "orientation":
        state["measured"] = tcp.to_pose7()
        state["measured"][3:] = [np.cos(.1), 0, 0, np.sin(.1)]
    assert not task._drive_to_tcp(tcp.p, "right", n_settle=0, tol=.04, tcp_pose=tcp)
    assert _names(events).count("execute") == int(fault in ("arrival", "orientation"))


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf, .005])
def test_drawer_rejects_bad_physical_clock_before_planning(motion, dt):
    task, arm, _state, events = motion
    task.scene.sim.dt = dt
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    assert not task._drive_to_tcp(tcp.p, "right", tcp_pose=tcp)
    assert events == []


@pytest.mark.parametrize("task_type,active", [
    (PutObjectInCabinetInterruption, "_cabinet_pick_active"),
    (PutObjectInCabinetContribution, "_ordinary_pick_active"),
])
@pytest.mark.parametrize("raises", [False, True])
def test_drawer_clock_is_not_expanded_again_inside_payload(motion, monkeypatch, task_type, active, raises):
    original, arm, _state, events = motion
    task = task_type.__new__(task_type)
    task.__dict__.update(original.__dict__)
    del task.execute_plan
    setattr(task, active, True)
    captured = []

    def execute(self, plan, tag, gripper_result=None):
        assert self._cabinet_plan_is_physical_clock
        captured.append(plan.position.copy())
        if raises:
            raise RuntimeError("execution interrupted")

    monkeypatch.setattr(BaseTask, "execute_plan", execute)
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    if raises:
        with pytest.raises(RuntimeError, match="execution interrupted"):
            task._drive_to_tcp(tcp.p, "right", n_settle=0, tcp_pose=tcp)
    else:
        assert task._drive_to_tcp(tcp.p, "right", n_settle=0, tcp_pose=tcp)
    assert captured[0].shape == (2, 7)
    assert task._cabinet_plan_is_physical_clock is False
    assert next(event[1]["time_step"] for event in events if event[0] == "rrt") == .002


@pytest.mark.parametrize("bound", [0, -1, np.nan, np.inf])
def test_invalid_optional_branch_bound_fails_before_planning(motion, bound):
    task, arm, _state, events = motion
    tcp = Pose.from_pose7(LINK_TARGET.copy()) * arm.tcp_offset
    assert not task._drive_to_tcp(tcp.p, "right", tcp_pose=tcp, max_joint_delta=bound)
    assert events == []


def test_tiny_lift_uses_cached_arm_seed_and_pure_cabinet_fk(motion):
    task, arm, _state, events = motion
    arm._cached_target[:7] = 0.12
    original = arm._cached_target.copy()
    tcp = task._top_down_tcp([0.1, -0.2, 0.9], yaw=0.37)
    result = task._solve_ik(tcp, "right")
    np.testing.assert_allclose(result, np.full(7, 0.13))
    np.testing.assert_array_equal(events[1][1], original[:7])
    np.testing.assert_array_equal(arm._cached_target, original)
    assert _names(events) == ["cloud", "ik", "fk"]
    assert task.plan_success


@pytest.mark.parametrize("fault", [
    "none", "failed", "exception", "empty", "shape", "ragged", "nan", "velocity",
    "branch", "fk_position", "fk_rotation", "fk_nan", "fk_shape", "fk_none", "fk_missing_fields",
])
def test_tiny_lift_rejects_before_pd_and_does_not_skip_active_waypoint(motion, fault):
    task, arm, state, events = motion
    task._cabinet_pick_active = True
    solve = arm.planner.solve_ik

    def bad(*args, **kwargs):
        plan = solve(*args, **kwargs)
        if fault == "none":
            return None
        if fault == "failed":
            plan.success = False
        elif fault == "exception":
            raise RuntimeError("backend rejected tiny IK")
        elif fault == "empty":
            plan.position = np.empty((0, 7))
        elif fault == "shape":
            plan.position = np.zeros((2, 6))
        elif fault == "ragged":
            plan.position = [[0] * 7, [0] * 6]
        elif fault == "nan":
            plan.position[0, 0] = np.nan
        elif fault == "velocity":
            plan.velocity[0, 0] = np.inf
        elif fault == "branch":
            plan.position[-1, 0] = args[0][0] + 0.301
        return plan

    arm.planner.solve_ik = bad
    if fault == "fk_position":
        state["fk_offset"][0] = 0.0101
    elif fault == "fk_rotation":
        state["fk_quat"] = [np.cos(0.0251), 0, 0, np.sin(0.0251)]
    elif fault in ("fk_nan", "fk_shape"):
        endpoint = SimpleNamespace(p=np.zeros(3), q=np.array([1., 0, 0, 0]))
        if fault == "fk_nan":
            endpoint.q[0] = np.nan
        else:
            endpoint.p = np.zeros(2)
        task._drawer_plan_endpoint_tcp = lambda *_args: endpoint
    elif fault in ("fk_none", "fk_missing_fields"):
        task._drawer_plan_endpoint_tcp = lambda *_args: None if fault == "fk_none" else object()
    with pytest.raises(_CabinetMotionRejected):
        task._solve_ik(Pose.from_pose7(LINK_TARGET.copy()), "right")
    assert not task.plan_success
    assert not set(_names(events)) & {"rrt", "execute", "pd", "step"}


@pytest.mark.parametrize("seed", [np.zeros(6), np.zeros((1, 9)), ["bad"] * 9, np.r_[np.zeros(7), np.nan, 0]])
def test_bad_tiny_seed_fails_before_planning(motion, seed):
    task, arm, _state, events = motion
    arm._cached_target = seed
    assert task._solve_ik(Pose.from_pose7(LINK_TARGET.copy()), "right") is None
    assert not task.plan_success and events == []


@pytest.mark.parametrize("position,quaternion", [
    ([np.nan, 0, 0], [1, 0, 0, 0]), ([0, 0], [1, 0, 0, 0]),
    ([0, 0, 0], [0, 0, 0, 0]), ([0, 0, 0], [2, 0, 0, 0]),
])
def test_bad_tiny_pose_fails_before_planning(motion, position, quaternion):
    task, _arm, _state, events = motion
    assert task._solve_ik(SimpleNamespace(p=position, q=quaternion), "right") is None
    assert not task.plan_success and events == []


@pytest.mark.parametrize("pose", [None, object(), SimpleNamespace(p=np.zeros(3))])
def test_missing_tiny_pose_fields_are_clean_rejections(motion, pose):
    task, _arm, _state, events = motion
    assert task._solve_ik(pose, "right") is None
    assert not task.plan_success and events == []


def test_seeded_motion_rejects_measured_orientation_mismatch(motion):
    task, _arm, state, _events = motion
    state["measured"] = [0.10, -0.20, 1.0034, 0, 1, 0, 0]
    assert task._move_seeded(LINK_TARGET, "right") is None
    assert not task.plan_success


@pytest.mark.parametrize("raw_pose", [
    None, object(), SimpleNamespace(p=np.full(3, np.nan), q=np.array([1., 0, 0, 0])),
    SimpleNamespace(p=np.zeros(3), q=np.array([np.nan, 0, 0, 0])),
    SimpleNamespace(p=np.zeros(2), q=np.array([1., 0, 0, 0])),
])
def test_raw_mplib_fk_output_is_validated_before_pose_composition(motion, raw_pose):
    task, arm, _state, events = motion
    arm.planner._mplib.pinocchio_model.get_link_pose = lambda _index: raw_pose
    assert task._move_seeded(LINK_TARGET, "right") is None
    assert not task.plan_success and "execute" not in _names(events)


@pytest.mark.parametrize("method", ["seeded", "tiny"])
def test_invalid_tcp_offset_is_rejected_before_transform_or_planning(motion, method):
    task, arm, _state, events = motion
    arm.tcp_offset = SimpleNamespace(p=np.zeros(3), q=np.full(4, np.nan))
    if method == "seeded":
        assert task._move_seeded(LINK_TARGET, "right") is None
    else:
        assert task._solve_ik(Pose.from_pose7(LINK_TARGET.copy()), "right") is None
    assert not task.plan_success and events == []


def test_tiny_lift_without_cached_target_uses_measured_arm_joints(motion):
    task, arm, state, events = motion
    arm._cached_target = None
    state["qpos"][:7] = 0.1
    np.testing.assert_allclose(task._solve_ik(Pose.from_pose7(LINK_TARGET.copy()), "right"), np.full(7, 0.11))
    np.testing.assert_array_equal(events[1][1], np.full(7, 0.1))


@pytest.mark.parametrize("yaw", [0.0, 0.4])
def test_shared_tiny_loop_keeps_30_by_30_timing_yaw_and_commanded_fingers(motion, yaw):
    task, arm, _state, events = motion
    arm._cached_target[7:] = [0.017, 0.019]
    start, end = np.array([0.1, -0.2, 0.8]), np.array([0.1, -0.2, 0.82])
    TopDownPickPlaceMixin._move_cartesian(
        task, start, end, "right", n_steps=30, sim_per_step=30, tcp_yaw=yaw,
    )
    assert _names(events).count("ik") == _names(events).count("cloud") == 30
    assert _names(events).count("pd") == 30 and _names(events).count("step") == 900
    assert "execute" not in _names(events)
    expected_q = task._top_down_tcp(start, yaw=yaw).q
    for event in events:
        if event[0] == "ik":
            assert abs(float(np.dot(event[2][3:], expected_q))) == pytest.approx(1.0)
        elif event[0] == "pd":
            np.testing.assert_array_equal(event[1][7:], [0.017, 0.019])
    np.testing.assert_array_equal(arm._cached_target[7:], [0.017, 0.019])


def test_rotated_acquisition_shared_screw_failure_stops_before_close(motion):
    task, arm, _state, events = motion
    solve = arm.planner.solve_ik
    calls = []

    def first_only(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs) if len(calls) == 1 else PlanResult(False)

    arm.planner.solve_ik = first_only
    arm.planner._mplib.plan_screw = lambda **_kwargs: {"status": "Failure"}
    task._move_screw = lambda *_a, **_k: pytest.fail("rotated pick must use its existing explicit shared dispatch")
    task.open_gripper = lambda tag: events.append(("open", tag))
    task.set_gripper = lambda *_args: pytest.fail("failed pre-grasp must not close")
    pick = PickSpec(get_center=lambda: np.array([0.1, -0.2, 0.8]), radius=0.025,
                    get_grasp_yaw=lambda: 0.4)
    place = PlaceSpec(pos=np.array([0.2, -0.5, 1.0]), transport_z=1.2)
    assert task.pick_and_place(pick, place, "right") is False
    assert len(calls) == 2 and _names(events).count("execute") == 1
    assert not task.plan_success and task._cabinet_pick_active is False


def test_rejected_tiny_loop_aborts_instead_of_skipping_to_next_stage(motion, monkeypatch):
    task, arm, _state, events = motion
    arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)

    def pickup(self, *_args):
        TopDownPickPlaceMixin._move_cartesian(
            self, np.array([0., 0., .8]), np.array([0., 0., .82]), "right",
            n_steps=30, sim_per_step=30, tcp_yaw=0.2,
        )
        events.append(("next_stage",))
        return True

    monkeypatch.setattr(TopDownPickPlaceMixin, "pick_and_place", pickup)
    assert task.pick_and_place(None, None, "right") is False
    assert _names(events) == ["cloud"]
    assert not task.plan_success and task._cabinet_pick_active is False


@pytest.mark.parametrize("previous_active", [False, True])
@pytest.mark.parametrize("outcome", [True, False, "runtime_error", "value_error"])
def test_pick_boundary_preserves_result_and_exposes_unexpected_exceptions(
    motion, monkeypatch, outcome, previous_active,
):
    task, _arm, _state, _events = motion
    task._cabinet_pick_active = previous_active

    def pickup(self, pick, place, tag):
        assert self._cabinet_pick_active
        assert (pick, place, tag) == ("pick", "place", "right")
        if outcome == "runtime_error":
            raise RuntimeError("unrelated implementation bug")
        if outcome == "value_error":
            raise ValueError("unrelated implementation bug")
        return outcome

    monkeypatch.setattr(TopDownPickPlaceMixin, "pick_and_place", pickup)
    if outcome in ("runtime_error", "value_error"):
        exception_type = RuntimeError if outcome == "runtime_error" else ValueError
        with pytest.raises(exception_type, match="unrelated implementation bug"):
            task.pick_and_place("pick", "place", "right")
    else:
        assert task.pick_and_place("pick", "place", "right") is outcome
    assert task._cabinet_pick_active is previous_active


@pytest.mark.parametrize("stage", ["ik", "rrt", "execute"])
def test_unexpected_programming_errors_are_not_motion_rejections(motion, stage):
    task, arm, _state, _events = motion

    def broken(*_args, **_kwargs):
        raise TypeError("programming defect")

    if stage == "ik":
        arm.planner.solve_ik = broken
    elif stage == "rrt":
        arm.planner._mplib.plan_qpos = broken
    else:
        task.execute_plan = broken
    with pytest.raises(TypeError, match="programming defect"):
        task._move_seeded(LINK_TARGET, "right")


def test_intent_and_retreat_keep_none_contract_outside_pick(motion):
    task, arm, _state, events = motion
    arm.planner.solve_ik = lambda *_a, **_k: PlanResult(False)
    assert task._move_seeded(LINK_TARGET, "right") is None
    task._move_screw = lambda *_a, **_k: None
    assert task._retreat_safe("right") is None
    assert _names(events) == ["cloud", "cloud"]
    assert not task.plan_success and not getattr(task, "_cabinet_pick_active", False)


@pytest.mark.parametrize("object_id,model,robot,forward", [
    ("038_milk-box", 0, "franka", 0.20),
    ("038_milk-box", 1, "franka", 0.25),
    ("038_milk-box", 0, "piper", 0.25),
    ("031_jam-jar", 0, "franka", 0.25),
    ("035_apple", 1, "franka", 0.25),
])
def test_inspection_park_clearance_changes_only_diagnosed_milk_case(
    motion, object_id, model, robot, forward,
):
    task, arm, _state, _events = motion
    task._object_id, task._object_model_id = object_id, model
    task.config["robot_type"] = robot
    targets = []
    task._move_screw = lambda target, tag: targets.append((target.copy(), tag)) or True
    task._retreat_safe("right")
    assert len(targets) == 1
    assert targets[0][1] == "right"
    np.testing.assert_allclose(targets[0][0], [
        arm.origin_pose.p[0], arm.origin_pose.p[1] + forward, task.TABLE_TOP_Z + 0.30,
    ])


def test_release_hook_withdraws_from_measured_pose_with_open_fingers(motion):
    task, arm, state, events = motion
    actual = Pose([0.2, -0.5, 1.065], [0, 1, 0, 0])
    state["measured"] = actual.to_pose7()
    arm.gripper_val = 1.0
    moves = []
    task._payload_move_pose = lambda target, tag, label: moves.append((target, tag, label)) or True
    place = PlaceSpec(pos=[0.21, -0.49, 1.0], support_z=0.99, transport_z=1.19)
    task._after_pick_place_release(None, place, "right", [0.21, -0.49, 1.0])
    assert len(moves) == 1
    target, tag, label = moves[0]
    np.testing.assert_allclose(target.p, [0.2, -0.5, 1.19])
    np.testing.assert_allclose(target.q, actual.q)
    assert (tag, label) == ("right", "release retreat")
    assert arm.gripper_val == 1.0 and not events


@pytest.mark.parametrize("failure", ["motion", "nan_height", "lower_height", "measured"])
def test_release_withdrawal_rejection_propagates_through_active_pick(motion, failure):
    task, _arm, state, _events = motion
    task._cabinet_pick_active = True
    state["measured"] = np.array([0.2, -0.5, 1.06, 0, 1, 0, 0])
    height = {"nan_height": np.nan, "lower_height": 1.0}.get(failure, 1.19)
    if failure == "measured":
        state["measured"][0] = np.nan
    task._payload_move_pose = lambda *_a, **_k: False
    place = PlaceSpec(pos=[0.2, -0.5, 1.0], support_z=0.99, transport_z=height)
    with pytest.raises(_CabinetMotionRejected, match="release retreat"):
        task._after_pick_place_release(None, place, "right", [0.2, -0.5, 1.0])
    assert not task.plan_success


def test_exact_cabinet_capsule_cloud_is_preserved_for_both_adapters(motion, monkeypatch):
    task, _arm, _state, events = motion
    capsules = [("r_palm",), ("r_forearm",)]
    sampled = []

    def sample(actual, *, inflate_factor):
        sampled.append((actual, inflate_factor))
        return np.array([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(core, "sample_capsule_planner_points", sample)
    task.avatar_collider = SimpleNamespace(current_capsules=lambda: capsules)
    task.config = {"collision_ignore_capsules": ["r_palm"]}
    task._refresh_avatar_planner_obstacles = CabinetTaskCore._refresh_avatar_planner_obstacles.__get__(task)
    assert task._move_seeded(LINK_TARGET, "right") is True
    assert task._solve_ik(Pose.from_pose7(LINK_TARGET.copy()), "right") is not None
    assert sampled == [(capsules, 1.8), (capsules, 1.8)]
    points_events = [event for event in events if event[0] == "points"]
    assert len(points_events) == 2
    assert all(event[2] == {"resolution": 0.04} for event in points_events)


def test_ordinary_native_dispatch_does_not_change_interrupt_or_abstract_core():
    from envs.task_bases.cabinet_ordinary_motion import CabinetOrdinaryMotionMixin
    assert CabinetTaskCore._solve_ik is TopDownPickPlaceMixin._solve_ik
    assert CabinetTaskCore.pick_and_place is TopDownPickPlaceMixin.pick_and_place
    assert CabinetTaskCore._move_seeded is TopDownPickPlaceMixin._move_seeded
    for task_class in (PutObjectInCabinetContribution, PutObjectInCabinetCopresence):
        assert task_class._solve_ik is CabinetOrdinaryMotionMixin._solve_ik
    assert PutObjectInCabinetContribution.pick_and_place is CabinetOrdinaryMotionMixin.pick_and_place
    neutral = PutObjectInCabinetCopresence.__new__(PutObjectInCabinetCopresence)
    assert super(PutObjectInCabinetCopresence, neutral).pick_and_place.__func__ is CabinetOrdinaryMotionMixin.pick_and_place
    assert PutObjectInCabinetContribution._move_seeded is CabinetOrdinaryMotionMixin._move_seeded
    assert PutObjectInCabinetCopresence._move_seeded is NeutralAvatarTableWorkMixin._move_seeded
    assert CabinetOrdinaryMotionMixin not in PutObjectInCabinetInterruption.__mro__


def _milk_screw_fixture(motion):
    task, arm, state, events = motion

    def screw(seed, target, *, qpos_step):
        # 0.02 first; the finer steps only re-solve a rejected coarse leg.
        assert qpos_step in (0.02, 0.01, 0.005)
        events.append(("screw", seed.copy(), target.copy()))
        state["target"] = target.copy()
        return PlanResult(True, np.stack([seed, seed + 0.01]))

    arm.planner.plan_screw_path = screw
    target = Pose.from_pose7(LINK_TARGET) * arm.tcp_offset
    return task, arm, state, events, target


def test_milk_post_grasp_screw_uses_native_fk_and_measured_arrival(motion):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    assert task._payload_move_pose(target, "right", "lift") is True
    names = _names(events)
    assert names[:4] == ["cloud", "screw", "fk", "execute"]
    assert names.count("step") == 60
    assert "ik" not in names and "rrt" not in names
    np.testing.assert_array_equal(arm._cached_target[7:], [0.020, 0.020])
    np.testing.assert_allclose(events[1][2], LINK_TARGET, atol=1e-12)


@pytest.mark.parametrize("fault", [
    "none", "failed", "empty", "shape", "nan", "velocity", "branch",
    "path_jump", "tiny_jump", "fk_position", "fk_quaternion", "fk_nan",
    "not_native", "exception",
])
def test_milk_invalid_post_grasp_plan_never_drives(motion, fault):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    original = arm.planner.plan_screw_path

    def bad_screw(seed, goal, **kwargs):
        if fault == "exception":
            raise RuntimeError("no feasible screw")
        plan = original(seed, goal, **kwargs)
        if fault == "none":
            return None
        if fault == "failed": plan.success = False
        if fault == "empty": plan.position = np.empty((0, 7))
        if fault == "shape": plan.position = np.zeros((2, 6))
        if fault == "nan": plan.position[-1, 0] = np.nan
        if fault == "velocity": plan.velocity = np.zeros((3, 7))
        if fault == "branch": plan.position[-1, 0] = 3.6
        if fault == "path_jump": plan.position = np.stack([seed, seed + 0.4, seed + 0.01])
        # A dense tiny lift may turn a joint up to 0.6 rad; 0.61 exceeds it.
        if fault == "tiny_jump": plan.position = np.linspace(seed, seed + 0.61, 14)
        if fault in ("path_jump", "tiny_jump"):
            plan.velocity = np.zeros_like(plan.position)
        return plan

    arm.planner.plan_screw_path = bad_screw
    if fault == "fk_position": state["fk_offset"] = np.array([0.03, 0, 0])
    if fault == "fk_quaternion": state["fk_quat"] = np.array([0, 1, 0, 0])
    if fault == "fk_nan": state["fk_offset"] = np.array([np.nan, 0, 0])
    if fault == "not_native": del arm.planner._mplib
    assert not task._payload_move_pose(target, "right", "diagnostic", tiny=fault == "tiny_jump")
    assert not task.plan_success
    assert "execute" not in _names(events)
    assert "pd" not in _names(events)


@pytest.mark.parametrize("measured", [
    np.array([0.9, 0.9, 0.9, 1, 0, 0, 0]),
    np.array([0.1, -0.2, 1.0034, 0, 1, 0, 0]),
    np.full(7, np.nan), np.zeros(6),
])
def test_milk_tracking_failure_is_reported_after_execution(motion, measured):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    state["measured"] = measured
    assert not task._payload_move_pose(target, "right", "carry")
    assert _names(events).count("execute") == 1
    assert not task.plan_success


@pytest.mark.parametrize('fault', [None, 'failed', 'nan', 'jump', 'fixture', 'fk', 'arrival'])
def test_apple_tiny_finer_retry_preserves_native_validation(motion, fault):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    task._object_id, task._object_model_id = '035_apple', 1
    task.config = {'robot_type': 'franka'}
    calls = []

    def screw(seed, goal, *, qpos_step):
        calls.append((seed.copy(), goal.copy(), qpos_step))
        state['target'] = goal.copy()
        if qpos_step == .02:
            return PlanResult(False)
        assert qpos_step in (.01, .005)
        pos = np.stack([seed, seed + .01])
        if fault == 'nan':
            pos[-1, 0] = np.nan
        if fault == 'jump':
            pos = np.linspace(seed, seed + .61, 14)
        return PlanResult(fault != 'failed', pos)

    arm.planner.plan_screw_path = screw
    if fault == 'fixture':
        task._body_path_clear = lambda *_: False
    if fault == 'fk':
        state['fk_offset'] = np.array([.02, 0, 0])
    if fault == 'arrival':
        state['measured'] = target.to_pose7().copy()
        state['measured'][0] += .02
    assert bool(task._payload_move_pose(target, 'right', 'tiny lift', tiny=True)) == (fault is None)
    # A finer step that yields a continuous path ends the retry; otherwise
    # the finest step is tried too. Every call re-solves the same leg.
    rejected_plan = fault in ('failed', 'nan', 'jump')
    assert [c[2] for c in calls] == ([.02, .01, .005] if rejected_plan else [.02, .01])
    for call in calls[1:]:
        np.testing.assert_array_equal(calls[0][0], call[0])
        np.testing.assert_array_equal(calls[0][1], call[1])
    assert _names(events).count('execute') == int(fault in (None, 'arrival'))
    assert task.plan_success == (fault is None)


@pytest.mark.parametrize('object_id,model,robot,tiny', [
    ('035_apple', 1, 'franka', False), ('035_apple', 0, 'franka', True),
    ('035_apple', 1, 'other', True), ('048_stapler', 0, 'franka', True),
    ('077_rubiks-cube', 0, 'franka', True), ('seal', 0, 'franka', True),
])
def test_finer_screw_retry_applies_to_every_payload_leg(motion, object_id, model, robot, tiny):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task.config = {'robot_type': robot}
    steps = []
    def screw(*_args, qpos_step):
        steps.append(qpos_step)
        return PlanResult(False)
    arm.planner.plan_screw_path = screw
    assert not task._payload_move_pose(target, 'right', 'diagnostic', tiny=tiny)
    assert steps == [.02, .01, .005]
    assert 'execute' not in _names(events)


def _milk_route_fixture(motion):
    task, arm, state, events = motion
    task.config = {"robot_type": "franka"}
    task._body_acquisition_obstacles = lambda _: nullcontext()
    task._object_id, task._object_model_id = "038_milk-box", 0
    task._object_scale = 1.0
    task._drawer_rim_z = 1.09
    task._object_relative_grasp_yaw = lambda: 0.0
    state["measured"] = np.array([0.1, -0.2, 0.9, 1, 0, 0, 0], dtype=float)
    state["center"] = state["measured"][:3] + np.array([0.02, 0.0, -0.02])
    state["held"] = False
    task.object = SimpleNamespace(entity=SimpleNamespace(
        get_pos=lambda: np.array([0.1, -0.2, 0.765]),
        get_quat=lambda: np.array([1., 0, 0, 0]),
    ))

    def opened(tag):
        events.append(("open", tag))
        if state["held"]:
            state["center"][2] -= .20  # free fall in this sequence-only fixture
        state["held"] = False

    def closed(value, tag):
        events.append(("close", value, tag)); state["held"] = True

    task.open_gripper, task.set_gripper = opened, closed
    pick = PickSpec(
        get_center=lambda: state["center"].copy(), radius=0.0327,
        label="038_milk-box", close_value=0.55,
        get_half_height=lambda: 0.04, max_grasp_offset=0.10,
        require_lift=True, transport_retention_tolerance=0.08,
        after_close=lambda: events.append(("after_close",)),
    )
    place = PlaceSpec(pos=np.array([0.3, -0.5, 1.04]), label="drawer",
                      transport_z=1.2, support_z=1.0, support_clearance=0.005)
    return task, arm, state, events, pick, place


def test_milk_uses_public_grasp_entry_with_live_pose_and_scale(motion, monkeypatch):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_scale = 0.7  # Verify forwarding, not a hardcoded annotation scale.
    captured = []
    task.select_and_execute_grasp = lambda **kw: captured.append(kw)
    monkeypatch.setattr(TopDownPickPlaceMixin, "pick_and_place", lambda *_: pytest.fail("legacy milk grasp"))
    assert task.pick_and_place(pick, place, "right") is False
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["object_name"] == "038_milk-box" and kwargs["model_id"] == 0
    assert kwargs["object_scale"] == 0.7 and kwargs["max_candidates"] == 1
    assert kwargs["screw_qpos_step"] == 0.02 and callable(kwargs["post_plan_check"])
    assert kwargs["table_z"] == task.TABLE_TOP_Z
    np.testing.assert_array_equal(kwargs["object_pose"].p, [0.1, -0.2, 0.765])
    assert "close" not in _names(events)
    assert task._cabinet_pick_active is False


def test_milk_side_candidates_clear_native_palm_from_carton_and_table(motion):
    from itertools import product

    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    captured = []
    task.select_and_execute_grasp = lambda **kw: captured.append(kw)
    assert task.pick_and_place(pick, place, "right") is False
    candidates = captured[0]["candidate_grasps"]
    assert len(candidates) == 1
    object_pose = Pose([0, 0, 0.765], [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    native_bounds = np.array([
        [-0.0316359177, -0.1039897203, -0.0259248018],
        [0.0316157825, 0.1004258767, 0.0659621507],
    ])
    corners = np.array(list(product(*zip(native_bounds[0], native_bounds[1]))))
    for candidate in candidates:
        assert candidate.scale_frame == "mesh_unit"
        tcp = candidate.to_world(object_pose, object_scale=1.0)
        rotation = tcp.to_matrix()[:3, :3]
        np.testing.assert_allclose(rotation[:, 2], [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(np.abs(rotation[:, 1]), [0, 1, 0], atol=1e-12)
        palm = (corners - [0, 0, 0.1034]) @ rotation.T + tcp.p
        # Center on the measured mass axis with positive native palm clearance.
        # Physical trials must review the narrower4.2mm gap.
        assert abs(tcp.p[0]) < 1e-8
        assert palm[:, 0].min() > 0.033221 + 0.004
        assert palm[:, 2].min() > 0.765 + 0.018
        assert tcp.p[2] == pytest.approx(0.815)


@pytest.mark.parametrize("yaw", [0, -np.pi / 3])
@pytest.mark.parametrize("scale", [.05, .07])
def test_jar_public_body_grasp_preserves_world_geometry_under_yaw_and_scale(motion, yaw, scale):
    from itertools import product

    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_scale = "031_jam-jar", scale
    rotation = Pose([0, 0, 0], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]) * Pose(
        [0, 0, 0], [np.sqrt(.5), np.sqrt(.5), 0, 0],
    )
    pose = Pose([.1, -.2, .763], rotation.q)
    task.object.entity.get_pos = lambda: pose.p
    task.object.entity.get_quat = lambda: pose.q
    task.object.entity.get_AABB = lambda: np.array([[.065, -.235, .763], [.135, -.165, .858]])
    captured = []
    task.select_and_execute_grasp = lambda **kw: captured.append(kw)
    assert task.pick_and_place(pick, place, "right") is False
    assert len(captured) == 1 and "close" not in _names(events)
    kwargs = captured[0]
    assert kwargs['object_name'] == '031_jam-jar' and kwargs['object_scale'] == scale
    assert kwargs['table_z'] == task.TABLE_TOP_Z and kwargs['screw_qpos_step'] == .02
    candidate, fallback = kwargs['candidate_grasps']
    assert kwargs['max_candidates'] == 2
    assert candidate.name == 'cabinet_jar_topdown'
    assert fallback.name == 'cabinet_jar_topdown_orthogonal'
    np.testing.assert_allclose(fallback.to_world(pose, scale).p, [.1, -.2, .838])
    assert candidate.scale_frame == 'mesh_unit'
    tcp = candidate.to_world(pose, scale)
    np.testing.assert_allclose(tcp.p, [.1, -.2, .838])
    np.testing.assert_allclose(candidate.pre_grasp_world(pose, scale).p, [.1, -.2, .958])
    rotation = tcp.to_matrix()[:3, :3]
    np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.abs(rotation[:, 1]), [0, 1, 0], atol=1e-12)
    alternate = fallback.to_world(pose, scale).to_matrix()[:3, :3]
    np.testing.assert_allclose(alternate[:, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.abs(alternate[:, 1]), [1, 0, 0], atol=1e-12)
    bounds = np.array([[-.0316359177, -.1039897203, -.0259248018], [.0316157825, .1004258767, .0659621507]])
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    palm = (corners - [0, 0, .1034]) @ rotation.T + tcp.p
    assert palm[:, 2].min() > .858 + .017
    assert palm[:, 2].min() > .765 + .110


@pytest.mark.parametrize("fault", ['scale_zero', 'scale_nan', 'aabb_nan', 'aabb_inverted'])
def test_invalid_jar_body_geometry_stops_before_public_planning(motion, fault):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_scale = '031_jam-jar', .05
    aabb = np.array([[.065, -.235, .763], [.135, -.165, .858]])
    if fault == 'scale_zero': task._object_scale = 0
    if fault == 'scale_nan': task._object_scale = np.nan
    if fault == 'aabb_nan': aabb[0, 0] = np.nan
    if fault == 'aabb_inverted': aabb = aabb[::-1]
    task.object.entity.get_AABB = lambda: aabb
    task.select_and_execute_grasp = lambda **kw: pytest.fail('invalid body geometry planned')
    assert task.pick_and_place(pick, place, 'right') is False
    assert task.plan_success is False and 'close' not in _names(events)


@pytest.mark.parametrize("fault", [None, "pre_branch", "grasp_branch", "path_jump", "nan", "empty", "failed", "fk"])
def test_public_grasp_preflight_preserves_cabinet_motion_gates(motion, fault):
    task, arm, state, events = motion
    seed = arm.get_arm_qpos()
    pre = PlanResult(True, np.linspace(seed, seed + 0.01, 3))
    grasp = PlanResult(True, np.linspace(seed + 0.01, seed + 0.02, 3))
    link = Pose.from_pose7(LINK_TARGET)
    if fault == "pre_branch": pre.position[-1, 0] = 3.6
    if fault == "grasp_branch": grasp.position[-1, 0] = 3.7
    if fault == "path_jump": grasp.position[1, 0] = 0.5
    if fault == "nan": pre.position[-1, 0] = np.nan
    if fault == "empty": grasp.position = np.empty((0, 7))
    if fault == "failed": pre.success = False
    if fault == "fk": state["fk_offset"] = np.array([0.02, 0, 0])
    assert task._body_grasp_plan_valid((pre, grasp, link, link), "right") is (fault is None)
    assert "execute" not in _names(events) and "pd" not in _names(events)


@pytest.mark.parametrize("slip_stage", [None, "lift", "turn", "carry"])
def test_milk_rigid_rotation_retains_but_slip_stops_before_release(motion, slip_stage):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    grasp_tcp = Pose.from_pose7(state["measured"])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name="manual_000"))
    local_offset = state["center"] - grasp_tcp.p
    stages = []

    def move(target, tag, label, **kw):
        stages.append(label)
        state["measured"] = target.to_pose7()
        if state["held"]:
            state["center"] = target.p + target.to_matrix()[:3, :3] @ local_offset
            if label == slip_stage:
                state["center"][0] += 0.12
        return True

    task._payload_move_pose = move
    assert task.pick_and_place(pick, place, "right") is (slip_stage is None)
    assert [e[1] for e in events if e[0] == "close"] == [0.55]
    assert _names(events).count("after_close") == 1
    if slip_stage is None:
        assert stages == ["tiny lift", "lift", "turn", "carry", "descent", "release retreat"]
        assert _names(events).count("open") == 2
        assert not state["held"]
        assert state["center"][2] + pick.get_half_height() < state["measured"][2] - .015
    else:
        assert stages[-1] == slip_stage
        assert _names(events).count("open") == 1
        assert "descent" not in stages and "release retreat" not in stages


def _body_fixture_world(motion):
    task, arm, state, events = motion
    objects, entries = {}, {}
    world = SimpleNamespace(
        has_object=lambda name: name in objects,
        add_object=lambda name, obj: objects.__setitem__(name, obj),
        remove_object=lambda name: objects.pop(name),
    )

    def remove_entries(name):
        for pair in list(entries):
            if name in pair:
                del entries[pair]

    acm = SimpleNamespace(
        has_entry=lambda name: any(name in pair for pair in entries),
        set_entry=lambda a, b, allowed: entries.__setitem__((a, b), allowed),
        remove_entry=remove_entries,
    )
    arm.planner._mplib.planning_world = world
    arm.planner._mplib.acm = acm
    task.table = SimpleNamespace(get_AABB=lambda: np.array([[-.5, -.8, .715], [.5, .1, .765]]))
    task.cabinet = SimpleNamespace(entity=SimpleNamespace(
        get_AABB=lambda: np.array([[.21, -.75, .765], [.39, -.43, 1.089]])))
    return task, arm, state, events, objects, entries


@pytest.mark.parametrize("failure", [False, True])
def test_acquisition_fixture_scene_preserves_avatar_and_cleans_mount_pair(motion, failure):
    task, arm, state, events, objects, entries = _body_fixture_world(motion)
    avatar = object()
    objects['avatar_owned'] = avatar
    entries[('some_existing_link', 'some_existing_object')] = False
    try:
        with task._body_acquisition_obstacles('right'):
            assert set(objects) == {'avatar_owned', 'cabinet_body_acquisition_table',
                                    'cabinet_body_acquisition_fixture'}
            assert entries == {('some_existing_link', 'some_existing_object'): False,
                               ('panda_link0', 'cabinet_body_acquisition_table'): True}
            task._refresh_avatar_planner_obstacles('right')
            assert objects['avatar_owned'] is avatar
            if failure:
                raise _CabinetMotionRejected('planned acquisition failed')
            task._clear_body_acquisition_obstacles()  # Before drawer entry.
            assert set(objects) == {'avatar_owned'}
    except _CabinetMotionRejected:
        assert failure
    assert objects == {'avatar_owned': avatar}
    assert entries == {('some_existing_link', 'some_existing_object'): False}
    assert task._body_fixture_scene is None


@pytest.mark.parametrize("fault", ['nan', 'inverted', 'existing_object', 'existing_pair'])
def test_invalid_acquisition_fixture_never_changes_existing_world(motion, fault):
    task, arm, state, events, objects, entries = _body_fixture_world(motion)
    if fault == 'nan':
        task.table.get_AABB = lambda: np.full((2, 3), np.nan)
    elif fault == 'inverted':
        task.table.get_AABB = lambda: np.array([[1., 1, 1], [0., 0, 0]])
    elif fault == 'existing_object':
        objects['cabinet_body_acquisition_fixture'] = object()
    else:
        entries[('panda_link1', 'cabinet_body_acquisition_table')] = False
    original_objects, original_entries = objects.copy(), entries.copy()
    with pytest.raises(_CabinetMotionRejected):
        with task._body_acquisition_obstacles('right'):
            pytest.fail('invalid acquisition scene entered')
    assert objects == original_objects and entries == original_entries
    assert getattr(task, '_body_fixture_scene', None) is None


def test_body_collision_gate_checks_every_state_with_open_finger_envelope(motion):
    task, arm, state, events, objects, entries = _body_fixture_world(motion)
    queried = []
    arm.planner._pad_qpos = lambda q: np.r_[q, 0., 0.]
    def collision(q):
        queried.append(q.copy())
        return ['moving wrist/table'] if q[0] > .015 else []
    arm.planner._mplib.check_for_env_collision = collision
    path = np.array([[0.] * 7, [.01] * 7, [.02] * 7, [.03] * 7])
    original = path.copy()
    with task._body_acquisition_obstacles('right'):
        assert not task._body_path_clear(path, 'left')
        assert not task._body_path_clear(path, 'right')
    assert len(queried) == 3
    np.testing.assert_array_equal(np.asarray(queried)[:, 7:], np.full((3, 2), .04))
    np.testing.assert_array_equal(path, original)


def test_body_fixture_collision_rejects_transfer_before_pd_execution(motion):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    task._body_path_clear = lambda *_: False
    task._cabinet_pick_active = True
    with pytest.raises(_CabinetMotionRejected, match='fixture collision'):
        task._payload_move_pose(target, 'right', 'tiny lift', tiny=True)
    assert 'execute' not in _names(events)


@pytest.mark.parametrize('support_release', [1.02, 1.04, 1.10])
def test_body_release_height_clears_native_palm_and_preserves_support_minimum(motion, support_release):
    task, arm, state, events = motion
    task._drawer_rim_z = 1.08835721
    height = task._body_release_height(support_release, 1.18835721)
    palm_bottom = height + .1034 - .0659621507
    assert height >= support_release
    assert palm_bottom >= task._drawer_rim_z + .008
    assert height == pytest.approx(max(support_release, 1.05935721))
    assert events == []


@pytest.mark.parametrize('support,safe,rim', [(np.nan, 1.2, 1.09), (1.03, np.inf, 1.09),
                                             (1.03, 1.2, np.nan), (1.03, 1.04, 1.09),
                                             (1.3, 1.2, 1.09)])
def test_invalid_or_unreachable_body_release_height_is_rejected(motion, support, safe, rim):
    task, arm, state, events = motion
    task._drawer_rim_z = rim
    with pytest.raises(_CabinetMotionRejected):
        task._body_release_height(support, safe)
    assert events == []


def test_payload_clock_preserves_native_duration_positions_and_gripper():
    from envs.tasks.put_object_in_cabinet_interruption import _cabinet_plan_on_sim_clock
    pos = np.arange(21, dtype=float).reshape(3, 7) / 100
    vel = np.full_like(pos, .2)
    plan = PlanResult(True, pos.copy(), vel.copy())
    grip = np.array([1., .7, .5])
    timed, output_grip = _cabinet_plan_on_sim_clock(plan, .002, grip)
    assert len(timed.position) * .002 == len(pos) * .004
    np.testing.assert_array_equal(timed.position[::2], pos)
    np.testing.assert_array_equal(timed.position[1::2], pos)
    np.testing.assert_array_equal(timed.velocity, np.repeat(vel, 2, axis=0))
    np.testing.assert_array_equal(output_grip, [1., 1., .7, .7, .5, .5])
    np.testing.assert_array_equal(plan.position, pos)
    np.testing.assert_array_equal(grip, [1., .7, .5])
    same, none = _cabinet_plan_on_sim_clock(plan, .004)
    assert same is plan and none is None


@pytest.mark.parametrize('dt', [0., -.002, np.nan, np.inf, .003, .008])
def test_payload_clock_rejects_invalid_or_unsupported_dt(dt):
    from envs.tasks.put_object_in_cabinet_interruption import _cabinet_plan_on_sim_clock
    with pytest.raises(_CabinetMotionRejected):
        _cabinet_plan_on_sim_clock(PlanResult(True, np.zeros((2, 7))), dt)


@pytest.mark.parametrize('gripper', [[1.], [1., np.nan]])
def test_payload_clock_rejects_unsynchronized_gripper(gripper):
    from envs.tasks.put_object_in_cabinet_interruption import _cabinet_plan_on_sim_clock
    with pytest.raises(_CabinetMotionRejected):
        _cabinet_plan_on_sim_clock(PlanResult(True, np.zeros((2, 7))), .002, gripper)


def test_payload_clock_is_scoped_to_pick_and_preserves_native_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(BaseTask, 'execute_plan',
        lambda self, plan, arm_tag, gripper_result=None: calls.append((plan, gripper_result)))
    task = PutObjectInCabinetInterruption.__new__(PutObjectInCabinetInterruption)
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    arm = SimpleNamespace(planner=SimpleNamespace(_mplib=object()))
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    plan = PlanResult(True, np.zeros((2, 7)))
    task.execute_plan(plan, 'right')
    assert calls[-1][0] is plan  # Drawer and other phases keep existing timing.
    task._cabinet_pick_active = True
    task.execute_plan(plan, 'right')
    assert len(calls[-1][0].position) == 4
    arm.planner = object()
    with pytest.raises(_CabinetMotionRejected, match='native MPLib'):
        task.execute_plan(plan, 'right')
    assert task.plan_success is False
    assert len(calls) == 2


@pytest.mark.parametrize('yaw', [0., -.7])
@pytest.mark.parametrize('scale', [.7, .9])
def test_public_apple_candidates_preserve_world_clearance_and_scale(motion, yaw, scale):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id, task._object_scale = '035_apple', 1, scale
    task._object_relative_grasp_yaw = lambda: yaw
    bounds = np.array([[.075, -.225, .763], [.125, -.175, .817]])
    task.object.entity.get_AABB = lambda: bounds.copy()
    captured = []
    task.select_and_execute_grasp = lambda **kw: captured.append(kw)
    assert task.pick_and_place(pick, place, 'right') is False
    kwargs = captured[0]
    assert kwargs['object_name'] == '035_apple' and kwargs['model_id'] == 1
    assert kwargs['object_scale'] == scale and kwargs['max_candidates'] == 4
    assert kwargs['table_z'] == task.TABLE_TOP_Z
    frames = []
    for grasp in kwargs['candidate_grasps']:
        world = grasp.to_world(kwargs['object_pose'], scale)
        np.testing.assert_allclose(world.p, [.1, -.2, .788])
        np.testing.assert_allclose(world.to_matrix()[:3, 2], [0., 0., -1.], atol=1e-12)
        assert world.p[2] + .037 - bounds[1,2] == pytest.approx(.008)
        assert grasp.pre_grasp_world(kwargs['object_pose'], scale).p[2] == pytest.approx(.908)
        frames.append(world.to_matrix()[:3, 1])
    expected = task._top_down_tcp([.1, -.2, .788], yaw=yaw-np.pi/4)
    np.testing.assert_allclose(frames[0], expected.to_matrix()[:3, 1], atol=1e-12)
    assert np.dot(frames[0], frames[1]) == pytest.approx(-1., abs=1e-12)
    assert np.dot(frames[2], frames[3]) == pytest.approx(0., abs=1e-12)
    assert np.dot(frames[0], frames[2]) == pytest.approx(np.sqrt(.5), abs=1e-12)
    assert 'close' not in _names(events)


@pytest.mark.parametrize('selected_yaw', [-np.pi/4, 3*np.pi/4, np.pi/2])
def test_apple_keeps_selected_downward_orientation_through_retained_transfer(motion, selected_yaw):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id, task._object_scale = '035_apple', 1, .7
    task.object.entity.get_AABB = lambda: np.array([[.075, -.225, .763], [.125, -.175, .817]])
    selected_tcp = task._top_down_tcp([.1, -.2, .788], yaw=selected_yaw)
    state['measured'] = selected_tcp.to_pose7()
    local_offset = np.array([0., 0., .01])
    state['center'] = selected_tcp.p + selected_tcp.to_matrix()[:3,:3] @ local_offset
    grasp_link = selected_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='orthogonal'))
    stages = []
    def move(target, tag, label, **kw):
        stages.append(label)
        np.testing.assert_allclose(target.q, selected_tcp.q, atol=1e-12)
        state['measured'] = target.to_pose7()
        if state['held']:
            state['center'] = target.p + target.to_matrix()[:3,:3] @ local_offset
        return True
    task._payload_move_pose = move
    assert task.pick_and_place(pick, place, 'right') is True
    assert stages == ['tiny lift', 'lift', 'carry', 'descent', 'release retreat']
    assert _names(events).count('open') == 2
    assert not state['held']


@pytest.mark.parametrize('object_id,model,scale,depth', [
    ('048_stapler', 0, .05, .020), ('086_woodenblock', 1, .7, .029),
])
def test_solid_public_candidates_preserve_closing_axis_and_actual_scale(motion, object_id, model, scale, depth):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id, task._object_scale = object_id, model, scale
    task._object_relative_grasp_yaw = lambda: -.4
    bounds = np.array([[.075, -.225, .763], [.125, -.175, .837]])
    task.object.entity.get_AABB = lambda: bounds.copy()
    captured = []
    task.select_and_execute_grasp = lambda **kw: captured.append(kw)
    assert task.pick_and_place(pick, place, 'right') is False
    kwargs = captured[0]
    assert kwargs['object_name'] == object_id and kwargs['model_id'] == model
    assert kwargs['object_scale'] == scale
    frames = [g.to_world(kwargs['object_pose'], scale) for g in kwargs['candidate_grasps']]
    assert len(frames) == (4 if object_id == '086_woodenblock' else 2)
    for frame in frames:
        np.testing.assert_allclose(frame.p, [.1, -.2, .837 - depth])
        np.testing.assert_allclose(frame.to_matrix()[:3, 2], [0, 0, -1], atol=1e-12)
    assert np.dot(frames[0].to_matrix()[:3, 1], frames[1].to_matrix()[:3, 1]) == pytest.approx(-1.)
    if object_id == '086_woodenblock':
        assert [g.name for g in kwargs['candidate_grasps'][:2]] == [
            'cabinet_wood_palm_clear', 'cabinet_wood_palm_clear_equivalent']
        assert np.dot(frames[0].to_matrix()[:3, 1], frames[2].to_matrix()[:3, 1]) == pytest.approx(0.,abs=1e-12)
        assert np.dot(frames[2].to_matrix()[:3, 1], frames[3].to_matrix()[:3, 1]) == pytest.approx(-1.)
    assert 'close' not in _names(events)


def test_wood_quarter_turn_is_scoped_to_model1(motion):
    task, _, _, _, _, _ = _milk_route_fixture(motion)
    task._object_id, task._object_model_id, task._object_scale = '086_woodenblock', 0, .7
    task._object_relative_grasp_yaw = lambda: -.4
    task.object.entity.get_AABB = lambda: np.array([[.075,-.225,.763],[.125,-.175,.837]])
    assert [g.name for g in task._body_grasp_candidates(Pose())] == [
        'cabinet_wood_palm_clear','cabinet_wood_palm_clear_equivalent']


def test_wood_quarter_turn_preserves_actual_captured_pad_sections(motion):
    import itertools
    import trimesh
    from envs.utils import ASSETS_PATH
    task, arm, _, _, _, _ = _milk_route_fixture(motion)
    task._object_id, task._object_model_id, task._object_scale = '086_woodenblock', 1, .7
    # Original19 measured geometry, before any pickup motion; this tests the
    # real asset's available purchase rather than assuming a square AABB.
    pose = Pose.from_pose7([-.14214040338993073,-.2799657881259918,.7617783546447754,
        .7068841005030707,.7065872097614088,-.02277362579212683,-.023037912792433893])
    original = Pose.from_pose7([1.8373876742742e-05,.0585886287223858,1.3199636938307107e-05,
        .7069541044166486,.7072594020731824,.00012308107677905245,.00013144804126197175])
    rot = (pose * original).to_matrix()[:3,:3]
    task._object_relative_grasp_yaw = lambda: float(np.arctan2(rot[1,0],rot[0,0]))
    bounds = np.array([[-.15979070961475372,-.297742635011673,.7617877721786499],
                       [-.12443386763334274,-.26217561960220337,.8317903876304626]])
    task.object.entity.get_AABB = lambda: bounds.copy()
    mesh = trimesh.load(ASSETS_PATH/'objects/086_woodenblock/visual/base1.glb',force='mesh',process=False)
    world = mesh.vertices*.7@pose.to_matrix()[:3,:3].T+pose.p
    frames = [g.to_world(pose,.7) for g in task._body_grasp_candidates(pose)]
    assert len(frames) == 4
    widths=[]
    for frame in frames:
        np.testing.assert_allclose(frame.p,[*bounds.mean(0)[:2],bounds[1,2]-.029],atol=1e-12)
        local=mesh.copy();local.vertices=(world-frame.p)@frame.to_matrix()[:3,:3]
        row=[]
        for x,z in itertools.product((-.0055,0.,.0055),(-.0085,0.,.0085)):
            hits,_,_=local.ray.intersects_location(np.array([[x,-.12,z]]),np.array([[0.,1.,0.]]),multiple_hits=True)
            assert len(hits)==2
            low,high=sorted(hits[:,1])
            assert low < -.014 and high > .014
            row.append(high-low)
        assert min(row)>.032 and max(row)<.036
        widths.append(row)
    assert abs(widths[0][4]-widths[2][4])<.001


def test_native_body_transfer_profile_restores_original_limits_after_failure(motion):
    task, arm, _, _ = motion
    task.config = {'robot_type': 'franka'}
    task._cabinet_pick_active = True
    task._object_id = '035_apple'
    backend = arm.planner._mplib
    velocity, acceleration = np.ones(7), np.ones(7)
    backend.joint_vel_limits, backend.joint_acc_limits = velocity, acceleration
    with pytest.raises(RuntimeError, match='planned failure'):
        with task._body_transfer_profile(arm):
            np.testing.assert_array_equal(backend.joint_vel_limits, np.full(7, 2.))
            np.testing.assert_array_equal(backend.joint_acc_limits, np.full(7, 4.))
            raise RuntimeError('planned failure')
    assert backend.joint_vel_limits is velocity
    assert backend.joint_acc_limits is acceleration


@pytest.mark.parametrize('reason', ['seal', 'tiny', 'inactive', 'custom'])
def test_body_transfer_profile_preserves_other_motion_limits(motion, reason):
    task, arm, _, _ = motion
    task.config = {'robot_type': 'franka'}
    task._cabinet_pick_active = reason != 'inactive'
    task._object_id = '100_seal' if reason == 'seal' else '038_milk-box'
    backend = arm.planner._mplib
    velocity = np.full(7, .5 if reason == 'custom' else 1.)
    acceleration = np.ones(7)
    backend.joint_vel_limits, backend.joint_acc_limits = velocity, acceleration
    with task._body_transfer_profile(arm, tiny=reason == 'tiny'):
        assert backend.joint_vel_limits is velocity
        assert backend.joint_acc_limits is acceleration


def test_body_transfer_profile_rejects_nonfinite_limits(motion):
    task, arm, _, _ = motion
    task.config = {'robot_type': 'franka'}
    task._cabinet_pick_active = True
    task._object_id = '038_milk-box'
    arm.planner._mplib.joint_vel_limits = np.full(7, np.nan)
    arm.planner._mplib.joint_acc_limits = np.ones(7)
    with pytest.raises(_CabinetMotionRejected, match='limits'):
        with task._body_transfer_profile(arm):
            pytest.fail('invalid profile reached planning')


@pytest.mark.parametrize('object_id,expected_yaw', [('031_jam-jar', 0.), ('038_milk-box', -1.)])
def test_body_turn_does_not_follow_a_round_jars_postgrasp_yaw(motion, object_id, expected_yaw):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id = object_id
    task._object_relative_grasp_yaw = lambda: -1.
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='side')]
    grasp_tcp = Pose.from_pose7(state['measured'])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='side'))
    local_offset = state['center'] - grasp_tcp.p
    turns = []
    def move(target, tag, label, **kw):
        if label == 'turn':
            turns.append(target)
        state['measured'] = target.to_pose7()
        if state['held']:
            state['center'] = target.p + target.to_matrix()[:3,:3] @ local_offset
        return True
    task._payload_move_pose = move
    assert task.pick_and_place(pick, place, 'right') is True
    assert len(turns) == 1
    expected = task._top_down_tcp(turns[0].p, yaw=expected_yaw)
    assert abs(np.dot(turns[0].q, expected.q)) == pytest.approx(1.)


@pytest.mark.parametrize('radius,expected', [(.014, .25), (.025, .50), (.04, .70)])
def test_public_solid_pick_preserves_shared_radius_default_when_close_is_unset(motion, radius, expected):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id = '048_stapler'
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='stapler')]
    pick.close_value, pick.radius = None, radius
    grasp_tcp = Pose.from_pose7(state['measured'])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='stapler'))
    # Stop at the first post-close motion: the regression is None-to-default
    # handling after successful PUBLIC acquisition, before any transport.
    task._payload_move_pose = lambda *a, **kw: False
    assert task.pick_and_place(pick, place, 'right') is False
    assert [e[1] for e in events if e[0] == 'close'] == [pytest.approx(expected)]


@pytest.mark.parametrize('failure', [None, 'release turn', 'descent'])
@pytest.mark.parametrize('object_id,model', [
    ('048_stapler', 0), ('086_woodenblock', 1),
])
def test_solid_release_requires_retained_lower_descent(motion, failure, object_id, model):
    if object_id == '048_stapler' and failure == 'release turn':
        return  # Stapler's measured branch admits direct descent.
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='body')]
    grasp_tcp = Pose.from_pose7(state['measured'])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='side'))
    local_offset = state['center'] - grasp_tcp.p
    moves, openings = [], []
    def move(target, tag, label, **kw):
        moves.append((label, target, state['held']))
        if label == failure:
            return False
        state['measured'] = target.to_pose7()
        if state['held']:
            state['center'] = target.p + target.to_matrix()[:3,:3] @ local_offset
        return True
    def opened(tag):
        openings.append((state['held'], state['measured'].copy()))
        if state['held']:
            state['center'][2] -= .20
        state['held'] = False
    task._payload_move_pose, task.open_gripper = move, opened
    assert task.pick_and_place(pick, place, 'right') is (failure is None)
    if failure is None:
        assert len(openings) == 2 and openings[-1][0]
        assert openings[-1][1][2] < place.transport_z
        assert openings[-1][1][2] >= task._drawer_rim_z + .008 - .037
        assert moves[-2][0] == 'descent' and moves[-2][2]
        label, retreat, held = moves[-1]
        assert label == 'release retreat' and not held
        np.testing.assert_allclose(retreat.p[:2], openings[-1][1][:2])
        assert retreat.p[2] == place.transport_z
        turns = [target for label, target, _ in moves if label == 'release turn']
        assert len(turns) == (1 if object_id == '086_woodenblock' else 0)
        if turns:
            expected = task._top_down_tcp(turns[0].p, yaw=np.pi / 2)
            assert abs(np.dot(turns[0].q, expected.q)) == pytest.approx(1.)
            assert turns[0].p[2] == place.transport_z
    else:
        assert len(openings) == 1  # Failed held motion never opens over drawer.
        assert moves[-1][0] == failure and state['held']


@pytest.mark.parametrize('object_id,model,close,expected', [
    ('035_apple',1,.50,.675), ('035_apple',1,.55,.55), ('035_apple',1,.40,.40),
    ('035_apple',0,.50,.50), ('100_seal',0,.50,.50), ('038_milk-box',0,.55,.55),
])
def test_apple_additional_jaw_travel_preserves_other_close_settings(object_id,model,close,expected):
    task=PutObjectInCabinetInterruption.__new__(PutObjectInCabinetInterruption)
    task._object_id,task._object_model_id=object_id,model
    assert task._body_close_target(SimpleNamespace(close_value=close,radius=.02))==pytest.approx(expected)


@pytest.mark.parametrize('object_id,model', [('035_apple',1),('031_jam-jar',0),('038_milk-box',0)])
def test_round_body_releases_after_rim_clear_descent_not_high_drop(motion,object_id,model):
    task,arm,state,events,pick,place=_milk_route_fixture(motion)
    task._object_id,task._object_model_id=object_id,model
    task._body_grasp_candidates=lambda _:[SimpleNamespace(name='apple')]
    grasp_tcp=Pose.from_pose7(state['measured'])
    grasp_link=grasp_tcp*arm.tcp_offset.inv()
    task.select_and_execute_grasp=lambda **_:(grasp_link,grasp_link,SimpleNamespace(name='apple'))
    offset=state['center']-grasp_tcp.p
    sequence=[]
    def move(target,tag,label,**kw):
        sequence.append(label)
        state['measured']=target.to_pose7()
        if state['held']:
            state['center']=target.p+target.to_matrix()[:3,:3]@offset
        return True
    def opened(_tag):
        if state['held']:
            sequence.append('open')
            assert state['measured'][2]<place.transport_z
            assert state['measured'][2]>=task._drawer_rim_z+.008-.037
            state['center'][2]-=.10
        state['held']=False
    task._payload_move_pose,task.open_gripper=move,opened
    assert task.pick_and_place(pick,place,'right')
    assert sequence.index('descent')<sequence.index('open')<sequence.index('release retreat')


@pytest.mark.parametrize('fault', [None, 'already_aligned', 'object_shift', 'object_rotation',
                                  'nonconvergent', 'correction_bound', 'large_error',
                                  'angle', 'bad_object', 'plan_reject'])
def test_body_open_alignment_corrects_bias_or_stops_before_close(motion, fault):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    target = Pose.from_pose7(state['measured'].copy())
    bias = np.array([.003, -.003, -.003])
    if fault == 'already_aligned': bias[:] = 0
    if fault == 'large_error': bias = np.array([.009, 0., 0.])
    if fault == 'correction_bound': bias = np.array([.007, 0., 0.])
    state['measured'][:3] += bias
    if fault == 'angle': state['measured'][3:] = [np.cos(.06), np.sin(.06), 0., 0.]
    object_pose = {'p':np.array([.1, -.2, .765]), 'q':np.array([1., 0., 0., 0.])}
    if fault == 'bad_object': object_pose['p'][0] = np.nan
    task.object.entity.get_pos = lambda:object_pose['p'].copy()
    task.object.entity.get_quat = lambda:object_pose['q'].copy()
    moves = []
    def move(command, tag, label, **kwargs):
        assert not state['held']
        assert kwargs == {'tiny':True, 'reseed':False}
        moves.append(command)
        if fault == 'plan_reject': return False
        if fault not in ('nonconvergent', 'correction_bound'):
            state['measured'] = command.to_pose7()
            state['measured'][:3] += bias
        if fault == 'object_shift': object_pose['p'][0] += .004
        if fault == 'object_rotation': object_pose['q'] = np.array([np.cos(.04), np.sin(.04), 0., 0.])
        return True
    task._payload_move_pose = move
    assert bool(task._refine_open_body_grasp(target, 'right')) == (fault in (None, 'already_aligned'))
    assert 'close' not in _names(events) and len(moves) <= 2
    if fault is None:
        assert len(moves) == 1
        np.testing.assert_allclose(moves[0].p, target.p-bias)
        np.testing.assert_allclose(state['measured'][:3], target.p)
    if fault in ('already_aligned', 'large_error', 'angle', 'bad_object'):
        assert not moves


@pytest.mark.parametrize('reseed', [True, False])
def test_payload_alignment_can_preserve_checked_planned_arm_target(motion, reseed):
    task, arm, state, events, target = _milk_screw_fixture(motion)
    calls = []
    task._reseed_cached_arm_target = lambda a:calls.append(a)
    assert task._payload_move_pose(target, 'right', 'open grasp alignment', tiny=True, reseed=reseed)
    assert calls == ([arm] if reseed else [])
    assert 'execute' in _names(events)


@pytest.mark.parametrize('object_id,model', [('086_woodenblock',1),('035_apple',1),
                                           ('086_woodenblock',0),('035_apple',0),('048_stapler',0)])
@pytest.mark.parametrize('robot', ['franka', 'other'])
def test_body_alignment_failure_prevents_close_and_is_scoped(motion, object_id, model, robot):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task.config['robot_type'] = robot
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='body')]
    target = Pose.from_pose7(state['measured'])
    link = target * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw:(link,link,SimpleNamespace(name='body'))
    calls = []
    task._refine_open_body_grasp = lambda *a:calls.append(a) or False
    task._payload_move_pose = lambda *a,**kw:False
    assert task._pick_and_place_body(pick, place, 'right') is False
    aligned = robot == 'franka' and (object_id,model) in (('086_woodenblock',1),('035_apple',1))
    assert bool(calls) == aligned
    assert ('close' in _names(events)) == (not aligned)


def _stapler_descent_fixture(motion, fault=None):
    task, arm, state, events = motion
    task._object_id, task._object_model_id = '048_stapler', 0
    task.config = {'robot_type': 'franka'}
    target = Pose([.33, -.56, 1.06])
    state['measured'] = target.to_pose7()
    arm.planner._pad_qpos = lambda q: np.r_[q, 0., 0.]
    arm.planner.plan_screw_path = lambda *_a, **_kw: PlanResult(False)

    def solve(seed, goal, **kwargs):
        events.append(('ik', seed.copy(), goal.copy()))
        if fault == 'ik_exception':
            raise RuntimeError('native IK failed')
        if fault == 'ik_none':
            return None
        pos = np.stack([seed, seed + .10])
        if fault == 'ik_branch':
            pos[-1, 0] = 3.6
        if fault == 'ik_nan':
            pos[-1, 0] = np.nan
        return PlanResult(fault != 'ik_failed', pos)

    def route(**kwargs):
        events.append(('rrt', kwargs))
        if fault == 'rrt_exception':
            raise RuntimeError('native RRT failed')
        if fault == 'rrt_none':
            return None
        pos = np.linspace(np.zeros(7), np.full(7, .10), 6)
        if fault == 'path_nan':
            pos[2, 0] = np.nan
        if fault == 'path_jump':
            pos[2, 0] = .5
        if fault == 'path_branch':
            pos = np.linspace(np.zeros(7), np.full(7, 1.4), 20)
        vel = np.zeros_like(pos)
        if fault == 'velocity':
            vel = np.zeros((1, 7))
        return {'status': 'Failed' if fault == 'rrt_failed' else 'Success',
                'position': pos, 'velocity': vel}

    def fk(tag, q):
        fraction = float(q[0] / .1)
        p = target.p + [0, 0, .12 * (1-fraction)]
        quat = target.q.copy()
        middle = .35 < fraction < .85
        if middle and fault == 'xy':
            p[0] += .011
        if middle and fault == 'orientation':
            quat = np.array([np.cos(.03), np.sin(.03), 0, 0])
        if middle and fault == 'upward':
            p[2] = 1.19
        if middle and fault == 'overshoot':
            p[2] = 1.05
        if middle and fault == 'fk_nan':
            p[0] = np.nan
        if fraction > .99 and fault == 'endpoint':
            p[2] += .016
        return Pose(p, quat)

    def collide(q, kind):
        np.testing.assert_array_equal(q[7:], [.04, .04])
        events.append((kind, q.copy()))
        return [object()] if fault == kind and .035 < q[0] < .085 else []

    arm.planner.solve_ik, arm.planner._mplib.plan_qpos = solve, route
    arm.planner._mplib.check_for_self_collision = lambda q: collide(q, 'self_collision')
    arm.planner._mplib.check_for_env_collision = lambda q: collide(q, 'env_collision')
    task._drawer_plan_endpoint_tcp = fk
    if fault == 'arrival':
        state['measured'][0] += .020
    return task, arm, state, events, target


@pytest.mark.parametrize('fault', [None, 'ik_none', 'ik_failed', 'ik_nan', 'ik_branch',
    'ik_exception', 'rrt_none', 'rrt_failed', 'rrt_exception', 'path_nan',
    'path_jump', 'path_branch', 'velocity', 'xy', 'orientation', 'upward',
    'overshoot', 'fk_nan', 'endpoint', 'self_collision', 'env_collision', 'arrival'])
def test_stapler_descent_native_route_checks_entire_path_before_execution(motion, fault):
    task, arm, state, events, target = _stapler_descent_fixture(motion, fault)
    assert bool(task._payload_move_pose(target, 'right', 'descent')) == (fault is None)
    assert _names(events).count('execute') == int(fault in (None, 'arrival'))
    assert task.plan_success == (fault is None)
    if fault is None:
        native = next(event[1] for event in events if event[0] == 'rrt')
        assert native['time_step'] == .004
        assert native['planning_time'] == 10 and native['rrt_range'] == .15
        assert _names(events).count('self_collision') == 7
        assert _names(events).count('env_collision') == 7
        assert _names(events).count('step') == 60
        np.testing.assert_array_equal(arm._cached_target[7:], [.020, .020])


@pytest.mark.parametrize('object_id,model,robot,label,tiny,fallback', [
    ('048_stapler', 0, 'franka', 'carry', False, False),
    ('048_stapler', 0, 'franka', 'release retreat', False, True),   # V45: ascending leg uses the route
    ('048_stapler', 0, 'franka', 'descent', True, False),
    ('048_stapler', 1, 'franka', 'descent', False, True),
    ('048_stapler', 0, 'other', 'descent', False, False),
    ('035_apple', 1, 'franka', 'descent', False, True),
    ('086_woodenblock', 1, 'franka', 'descent', False, True),
    ('031_jam-jar', 0, 'franka', 'descent', False, True),
    ('100_seal', 0, 'franka', 'descent', False, False),
    ('100_seal', 0, 'franka', 'release retreat', False, True),   # V47: retreat is object independent
    ('073_rubikscube', 1, 'franka', 'release retreat', False, True),
    ('073_rubikscube', 1, 'other', 'release retreat', False, False),
])
def test_body_descent_fallback_is_scoped(motion, object_id, model, robot, label, tiny, fallback):
    """V41: the checked joint-space descent route applies to every body payload's
    non-tiny descent on Franka (probe 64595352), never to other legs, robots,
    tiny motions or the mixin-path seal."""
    task, arm, state, events, target = _stapler_descent_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task.config = {'robot_type': robot}
    assert bool(task._payload_move_pose(target, 'right', label, tiny=tiny)) == fallback
    assert ('ik' in _names(events)) == fallback and ('execute' in _names(events)) == fallback


@pytest.mark.parametrize('invalid', [False, True])
def test_stapler_successful_screw_does_not_trigger_route_retry(motion, invalid):
    task, arm, state, events, target = _stapler_descent_fixture(motion)
    pos = np.linspace(np.zeros(7), np.full(7, .1), 6)
    if invalid:
        pos[2, 0] = np.nan
    arm.planner.plan_screw_path = lambda *_a, **_kw: PlanResult(True, pos)
    assert bool(task._payload_move_pose(target, 'right', 'descent')) == (not invalid)
    assert 'ik' not in _names(events)
    assert _names(events).count('execute') == int(not invalid)


@pytest.mark.parametrize('object_id,model', [('048_stapler', 0), ('086_woodenblock', 1), ('031_jam-jar', 0)])
@pytest.mark.parametrize('second', ['ok', 'reject', 'turn_lost'])
def test_descent_planning_rejection_retries_once_from_mirrored_branch(motion, object_id, model, second):
    """V37: stapler16/wood19 carried into a wrist branch whose descent had no IK.
    A pre-execution descent rejection triggers exactly one 180 degree yaw turn
    at transport height and one more descent; a second rejection, or lost
    retention during the turn, fails without opening over the drawer."""
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='body')]
    grasp_tcp = Pose.from_pose7(state['measured'])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='side'))
    local_offset = state['center'] - grasp_tcp.p
    moves, openings, descents = [], [], []
    def move(target, tag, label, **kw):
        moves.append((label, target, state['held']))
        if label == 'descent':
            descents.append(target)
            if len(descents) == 1 or second == 'reject':
                task.plan_success = False
                raise _CabinetMotionRejected('payload descent: screw unavailable')
        state['measured'] = target.to_pose7()
        if state['held']:
            state['center'] = target.p + target.to_matrix()[:3, :3] @ local_offset
            if label == 'branch turn' and second == 'turn_lost':
                state['center'] = state['center'] + np.array([.10, 0., 0.])  # fixture tolerance is .08
        return True
    def opened(tag):
        openings.append((state['held'], state['measured'].copy()))
        if state['held']:
            state['center'][2] -= .20
        state['held'] = False
    task._payload_move_pose, task.open_gripper = move, opened
    result = task.pick_and_place(pick, place, 'right')
    labels = [label for label, _, _ in moves]
    assert labels.count('branch turn') == 1
    turn = next(target for label, target, _ in moves if label == 'branch turn')
    before = descents[0]
    yaw = np.arctan2(before.to_matrix()[1, 0], before.to_matrix()[0, 0])
    expected = task._top_down_tcp(turn.p, yaw=yaw + np.pi)
    assert abs(np.dot(turn.q, expected.q)) == pytest.approx(1.)
    assert turn.p[2] == place.transport_z
    if second == 'ok':
        assert result and len(descents) == 2
        assert len(openings) == 2 and openings[-1][0]
        assert openings[-1][1][2] >= task._drawer_rim_z + .008 - .037
        assert labels[-1] == 'release retreat'
    else:
        assert result is False
        assert len(openings) == 1  # never opens over the drawer while a leg is rejected
        assert len(descents) == (2 if second == 'reject' else 1)


def test_descent_arrival_failure_is_not_retried(motion):
    """Only pre-execution rejections are retried; a measured-arrival failure
    after an executed descent returns False without a branch turn."""
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id = '048_stapler', 0
    task._body_grasp_candidates = lambda _: [SimpleNamespace(name='body')]
    grasp_tcp = Pose.from_pose7(state['measured'])
    grasp_link = grasp_tcp * arm.tcp_offset.inv()
    task.select_and_execute_grasp = lambda **kw: (grasp_link, grasp_link, SimpleNamespace(name='side'))
    local_offset = state['center'] - grasp_tcp.p
    moves = []
    def move(target, tag, label, **kw):
        moves.append(label)
        if label == 'descent':
            return False
        state['measured'] = target.to_pose7()
        if state['held']:
            state['center'] = target.p + target.to_matrix()[:3, :3] @ local_offset
        return True
    task._payload_move_pose = move
    task.open_gripper = lambda tag: state.__setitem__('held', False)
    assert task.pick_and_place(pick, place, 'right') is False
    assert 'branch turn' not in moves and moves.count('descent') == 1


@pytest.mark.parametrize('object_id,model,stall_gap_m,moves', [
    ('035_apple', 1, .060, False),   # stall at 30 mm per jaw (orthogonal axis)
    ('035_apple', 1, .057, False),   # stall at 28.5 mm per jaw (45-degree axis)
    ('035_apple', 1, .066, True),    # apple moves at first touch before any stall
    ('048_stapler', 0, .020, False), # non-apple: single close-target command
])
def test_apple_close_is_staged_and_latches_at_the_measured_stall(motion, object_id, model, stall_gap_m, moves):
    task, arm, state, events, pick, place = _milk_route_fixture(motion)
    task._object_id, task._object_model_id = object_id, model
    task.config = {'robot_type': 'franka'}
    calls = []
    def set_gripper(value, tag):
        calls.append(float(value))
        per_jaw = max(float(value) * .04, stall_gap_m / 2)
        state['qpos'][7:9] = [per_jaw, per_jaw]
        if moves and float(value) * .08 < .066:
            state['center'] = state['center'] + np.array([.003, 0., 0.])
    task.set_gripper = set_gripper
    state['qpos'][7:9] = [.04, .04]
    hold = task._close_body_payload(pick, 'right')
    if object_id != '035_apple':
        assert calls == [pytest.approx(task._body_close_target(pick))] and hold == pytest.approx(task._body_close_target(pick))
        return
    assert calls[-1] == pytest.approx(hold)
    assert all(.30 <= c <= 1.0 for c in calls) and all(b < a for a, b in zip(calls[:-2], calls[1:-1]))
    expected_gap = max(stall_gap_m, calls[-2] * .08) if not moves else state['qpos'][7] * 2
    assert hold == pytest.approx(np.clip((expected_gap - .003) / .08, .30, 1.0), abs=1e-6)
    assert hold * .08 < expected_gap  # holds a real bite past the measured stall


@pytest.mark.parametrize('fault', [None, 'upward_overshoot', 'backtrack'])
def test_release_retreat_uses_the_checked_vertical_route_when_the_screw_fails(motion, fault):
    """V45: V44 seed 2 placed the Rubik's cube and opened, then the vertical
    release-retreat screw failed and the arm stayed over the drawer. The same
    checked joint-space route now serves ascending legs: monotone upward,
    10 mm lateral, .05 rad, no overshoot past the target."""
    task, arm, state, events, target = _stapler_descent_fixture(motion)
    start = Pose(target.p - np.array([0., 0., .12]), target.q)
    state['measured'] = start.to_pose7()
    def fk(tag, q):
        fraction = float(q[0] / .1)
        p = start.p + [0, 0, .12 * fraction]
        middle = .35 < fraction < .85
        if middle and fault == 'upward_overshoot':
            p[2] = target.p[2] + .02
        if middle and fault == 'backtrack':
            p[2] = start.p[2] - .01
        return Pose(p, target.q.copy())
    task._drawer_plan_endpoint_tcp = fk
    if fault is None:
        state['measured'] = target.to_pose7()  # measured arrival after execution
        task._payload_arrived_pose = lambda tgt, tag, label, tiny=False: tgt
    ok = bool(task._payload_move_pose(target, 'right', 'release retreat'))
    assert ok == (fault is None)
    assert 'ik' in _names(events)
    assert _names(events).count('execute') == int(fault is None)


def test_carry_leg_never_uses_the_vertical_route(motion):
    task, arm, state, events, target = _stapler_descent_fixture(motion)
    assert not task._payload_move_pose(target, 'right', 'carry')
    assert 'ik' not in _names(events) and 'execute' not in _names(events)
