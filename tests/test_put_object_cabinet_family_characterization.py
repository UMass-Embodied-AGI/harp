"""Scene-free release characterization for the cabinet task family.

These tests freeze public behavior, not ownership by any particular legacy
mixin.  They are intended to stay valid while the duplicated cabinet modules
move behind a shared family core and interaction controllers.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.avatar.inspect_motion_mixin import InspectMotionMixin
from envs.interaction.eval_mode import CabinetAssistController
from envs.interaction.inspect_motion import CabinetInterruptController
from envs.task_bases.put_object_cabinet import CabinetTaskCore, PutObjectCabinet
from envs.tasks.put_object_in_cabinet_contribution import PutObjectInCabinetContribution
from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
from envs.tasks.put_object_in_cabinet_copresence import PutObjectInCabinetCopresence
from envs.utils import ASSETS_PATH, Pose
from scripts.eval_expert_set import policy_play_mode


CORE_CLASSES = (
    PutObjectCabinet,
    PutObjectInCabinetContribution,
    PutObjectInCabinetInterruption,
)
ALL_CLASSES = (*CORE_CLASSES, PutObjectInCabinetCopresence)


@pytest.mark.parametrize("robot", ["franka", "other"])
@pytest.mark.parametrize("rejected_leg", [None, 0, 1])
def test_robot_drawer_handle_clearance_preserves_cached_geometry_and_arrival_gates(robot, rejected_leg):
    task = PutObjectCabinet({"robot_type": robot})
    task._drawer_handle_world0 = np.array([.35, -.53, 1.075])
    original = task._drawer_handle_world0.copy()
    task.robot = SimpleNamespace(get_arm=lambda _: SimpleNamespace())
    events = []
    task.open_gripper = lambda tag: events.append(("open", tag))
    motions = []

    def move(position, tag, **kwargs):
        motions.append((position.copy(), kwargs["tcp_pose"], kwargs["tol"]))
        events.append(("move", tag))
        return len(motions)-1 != rejected_leg

    class ReachedClose(Exception):
        pass

    def close(value, tag):
        events.append(("close", value, tag))
        raise ReachedClose

    task._drive_to_tcp, task.set_gripper = move, close
    task.step_sim = lambda: pytest.fail("failed approach must not advance into drawer pulling")
    if rejected_leg is None:
        with pytest.raises(ReachedClose):
            task._open_drawer_physics("right")
        assert events[-1] == ("close", 0.0 if robot == "franka" else .08, "right")
    else:
        assert task._open_drawer_physics("right") is False
        assert all(event[0] != "close" for event in events)
        assert len(motions) == rejected_leg+1
    np.testing.assert_array_equal(task._drawer_handle_world0, original)
    expected = original + [0., .010 if robot == "franka" else 0., 0.]
    np.testing.assert_allclose(motions[0][0], expected+[0.,task._HANDLE_PRE_DIST,0.])
    if len(motions) == 2:
        np.testing.assert_allclose(motions[1][0], expected)
    for position, pose, tolerance in motions:
        np.testing.assert_allclose(pose.p, position)
        np.testing.assert_allclose(pose.q, task._handle_tcp(position).q)
        assert tolerance == .025


@pytest.mark.parametrize("parent_result", [False, True])
def test_neutral_can_pick_waits_for_live_clip_without_clearing_contacts(monkeypatch, parent_result):
    task = PutObjectInCabinetCopresence({})
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl", "motion": "cans"}
    task._neutral_avatar_started = True
    task.plan_success = True
    task.avatar_collided = False
    motion = SimpleNamespace(data=[None] * 5, at_frame=2)
    task.avatar = SimpleNamespace(motion_modules={"cans": motion}, spare=lambda: motion.at_frame >= 5)
    events = []

    def step():
        events.append("physical_step")
        motion.at_frame += 1
        task.avatar_collided = True

    def parent(self, pick, place, arm_tag):
        assert self.avatar.spare()
        assert (pick, place, arm_tag) == ("pick", "place", "right")
        events.append("pickup")
        return parent_result

    task.step_sim = step
    monkeypatch.setattr(PutObjectCabinet, "pick_and_place", parent)
    assert task.pick_and_place("pick", "place", "right") is parent_result
    assert events == ["physical_step"] * 3 + ["pickup"]
    assert task.avatar_collided


@pytest.mark.parametrize("scope", ["other_job", "loop", "no_avatar", "not_started", "already_done"])
def test_neutral_can_wait_does_not_delay_other_work(monkeypatch, scope):
    task = PutObjectInCabinetCopresence({})
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl", "motion": "cans"}
    task._neutral_avatar_started = scope != "not_started"
    task.avatar = None if scope == "no_avatar" else SimpleNamespace(spare=lambda: scope == "already_done")
    if scope == "other_job":
        task._neutral_avatar_job["kind"] = "sponge"
    if scope == "loop":
        task._neutral_avatar_job["loop"] = True
    task.step_sim = lambda: pytest.fail("unrelated work must not wait")
    calls = []
    monkeypatch.setattr(PutObjectCabinet, "pick_and_place", lambda *args: calls.append(args) or True)
    assert task.pick_and_place("pick", "place", "right")
    assert len(calls) == 1


@pytest.mark.parametrize("count,frame", [(0, 0), (2, -1), (2, 2), (2, 1.5), (20001, 0), (None, 0)])
def test_neutral_invalid_can_clip_rejects_before_wait_or_pick(monkeypatch, count, frame):
    task = PutObjectInCabinetCopresence({})
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl", "motion": "cans"}
    task._neutral_avatar_started = True
    task.plan_success = True
    motion = SimpleNamespace(data=None if count is None else [None] * count, at_frame=frame)
    task.avatar = SimpleNamespace(motion_modules={"cans": motion}, spare=lambda: False)
    task.step_sim = lambda: pytest.fail("invalid clip must not step")
    monkeypatch.setattr(PutObjectCabinet, "pick_and_place", lambda *args: pytest.fail("invalid clip must not pick"))
    assert not task.pick_and_place("pick", "place", "right")
    assert not task.plan_success


def test_neutral_stalled_can_clip_has_a_finite_wait_and_rejects_pickup(monkeypatch):
    task = PutObjectInCabinetCopresence({})
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl", "motion": "cans"}
    task._neutral_avatar_started = True
    task.plan_success = True
    motion = SimpleNamespace(data=[None] * 2, at_frame=0)
    task.avatar = SimpleNamespace(motion_modules={"cans": motion}, spare=lambda: False)
    steps = []
    task.step_sim = lambda: steps.append(True)
    monkeypatch.setattr(PutObjectCabinet, "pick_and_place", lambda *args: pytest.fail("stalled clip must not pick"))
    assert not task.pick_and_place("pick", "place", "right")
    assert len(steps) == 102
    assert not task.plan_success


@pytest.mark.parametrize("yaw", [-0.94, 0.0, 0.7, 1.5])
def test_upright_circular_jar_avoids_unnecessary_wrist_rotation(yaw):
    task = PutObjectInCabinetInterruption({})
    task._object_id, task._object_model_id = "031_jam-jar", 0
    reference = Pose([0, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    rotated = Pose([0, 0, 0], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]) * reference
    task._object_grasp_reference_quat = reference.q.copy()
    task.object = SimpleNamespace(entity=SimpleNamespace(get_quat=lambda: rotated.q.copy()))
    assert task._object_relative_grasp_yaw() == 0.0


@pytest.mark.parametrize("object_id,model,robot,tilt", [
    ("031_jam-jar", 0, "franka", 0.3),
    ("031_jam-jar", 1, "franka", 0.0),
    ("031_jam-jar", 0, "piper", 0.0),
    ("086_woodenblock", 1, "franka", 0.0),
    ("038_milk-box", 0, "franka", 0.0),
])
def test_jar_yaw_scope_preserves_other_orientation_policies(object_id, model, robot, tilt):
    task = PutObjectInCabinetInterruption({"robot_type": robot})
    task._object_id, task._object_model_id = object_id, model
    reference = Pose([0, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    turn = Pose([0, 0, 0], [np.cos(-0.94 / 2), 0, 0, np.sin(-0.94 / 2)])
    tip = Pose([0, 0, 0], [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0])
    current = turn * tip * reference
    task._object_grasp_reference_quat = reference.q.copy()
    task.object = SimpleNamespace(entity=SimpleNamespace(get_quat=lambda: current.q.copy()))
    expected = task.relative_parallel_jaw_yaw(current.q, reference.q)
    assert abs(expected) > 0.5
    assert task._object_relative_grasp_yaw() == pytest.approx(expected)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 6, np.pi / 4])
def test_model1_block_closes_on_live_width_after_human_rotation(yaw):
    task = PutObjectInCabinetInterruption({})
    task._object_id = "086_woodenblock"
    task._object_model_id = 1
    task._object_topdown_pick_options = {"close_value": 0.52, "grasp_z_offset": 0.012}
    width = 0.035
    projected = width * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
    task.object = SimpleNamespace(entity=SimpleNamespace(
        get_AABB=lambda: np.array([[0, 0, 0.75], [projected, projected, 0.82]]),
    ))
    task._object_relative_grasp_yaw = lambda: yaw

    options = task._payload_pick_options()

    assert options["close_value"] * 0.08 == pytest.approx(width - 0.008)
    assert options["grasp_z_offset"] == 0.012
    assert task._object_topdown_pick_options["close_value"] == 0.52


def test_payload_close_preserves_catalog_when_grip_already_contacts():
    task = PutObjectInCabinetContribution({})
    task._object_id = "086_woodenblock"
    task._object_model_id = 1
    task._object_topdown_pick_options = {"close_value": 0.3}
    task.object = SimpleNamespace(entity=SimpleNamespace(
        get_AABB=lambda: np.array([[0, 0, 0.75], [0.035, 0.035, 0.82]]),
    ))
    assert task._payload_pick_options() == {"close_value": 0.3}


def _milk_pick_task(task_cls=PutObjectInCabinetInterruption, robot_type="franka"):
    task = task_cls({"robot_type": robot_type})
    task._object_id = "038_milk-box"
    task._object_model_id = 0
    task._object_radius = 0.03272618
    task._object_topdown_pick_options = {
        "close_value": 0.55, "grasp_z_offset": 0.0,
        "approach_xy_offset": (0.0, 0.0), "preopen_steps": 0,
    }
    # Original seed11 immediately before the open-gripper approach.
    task.object = SimpleNamespace(entity=SimpleNamespace(get_AABB=lambda: np.array([
        [-0.13059209, -0.33146173, 0.76327717],
        [-0.06363247, -0.26759624, 0.88330030],
    ])))
    return task


def test_interrupt_milk_keeps_catalog_after_failed_roof_clearance_trial():
    task = _milk_pick_task()
    before = dict(task._object_topdown_pick_options)
    assert task._payload_pick_options() == before
    assert task._object_topdown_pick_options == before


def test_interrupt_apple_does_not_lower_an_already_clear_grasp():
    task = _milk_pick_task()
    task._object_id, task._object_model_id = "035_apple", 1
    task._object_radius = 0.02578036
    task.object.entity.get_AABB = lambda: np.array([[0, 0, 0.7627], [0.054, 0.054, 0.8159]])
    task._object_topdown_pick_options["grasp_z_offset"] = 0.02
    assert task._payload_pick_options() == task._object_topdown_pick_options


def test_interrupt_apple_clearance_covers_measured_approach_tracking_error():
    task = _milk_pick_task()
    task._object_id, task._object_model_id = "035_apple", 1
    task._object_radius = 0.02578036
    bottom, roof = 0.76269114, 0.81587166
    task.object.entity.get_AABB = lambda: np.array([[0, 0, bottom], [0.054, 0.054, roof]])
    task._object_topdown_pick_options["close_value"] = 0.5
    before = dict(task._object_topdown_pick_options)
    options = task._payload_pick_options()
    nominal_tcp = (bottom + roof) / 2 - 0.3 * task._object_radius
    measured_tcp = 0.77699622
    palm_above_tcp = 0.1034 - 0.06596215069293976
    assert measured_tcp + palm_above_tcp < roof
    assert nominal_tcp + options["grasp_z_offset"] + palm_above_tcp >= roof + 0.008
    assert measured_tcp + options["grasp_z_offset"] + palm_above_tcp > roof + 0.003
    assert 0 < options["grasp_z_offset"] < 0.01
    assert {k: v for k, v in options.items() if k != "grasp_z_offset"} == {
        k: v for k, v in before.items() if k != "grasp_z_offset"
    }
    assert task._object_topdown_pick_options == before


def test_interrupt_jar_does_not_reapply_failed_topdown_height_hint():
    task = _milk_pick_task()
    task._object_id, task._object_model_id = "031_jam-jar", 0
    task._object_radius = 0.048272
    bottom, roof = 0.762412548, 0.857646823
    task.object.entity.get_AABB = lambda: np.array([[0, 0, bottom], [0.07, 0.07, roof]])
    before = dict(task._object_topdown_pick_options)
    options = task._payload_pick_options()
    assert options["close_value"] == before["close_value"] == 0.55
    assert options == before
    assert task._object_topdown_pick_options == before


@pytest.mark.parametrize("task_cls,robot_type,object_id,model_id", [
    (PutObjectInCabinetContribution, "franka", "038_milk-box", 0),
    (PutObjectInCabinetCopresence, "franka", "038_milk-box", 0),
    (PutObjectInCabinetInterruption, "piper", "038_milk-box", 0),
    (PutObjectInCabinetInterruption, "franka", "038_milk-box", 1),
    (PutObjectInCabinetInterruption, "franka", "035_apple", 0),
    (PutObjectInCabinetContribution, "franka", "035_apple", 1),
    (PutObjectInCabinetCopresence, "franka", "035_apple", 1),
    (PutObjectInCabinetInterruption, "piper", "035_apple", 1),
    (PutObjectInCabinetInterruption, "franka", "100_seal", 1),
    (PutObjectInCabinetInterruption, "franka", "031_jam-jar", 1),
    (PutObjectInCabinetContribution, "franka", "031_jam-jar", 0),
    (PutObjectInCabinetCopresence, "franka", "031_jam-jar", 0),
    (PutObjectInCabinetInterruption, "piper", "031_jam-jar", 0),
])
def test_roof_clearance_is_scoped_to_diagnosed_interrupt_franka_payloads(
    task_cls, robot_type, object_id, model_id,
):
    task = _milk_pick_task(task_cls, robot_type)
    task._object_id, task._object_model_id = object_id, model_id
    assert task._payload_pick_options() == task._object_topdown_pick_options


@pytest.mark.parametrize("failure", ["ik", "plan", "empty_plan", "endpoint"])
def test_drawer_waypoint_never_drives_after_planning_failure(failure):
    task = PutObjectCabinet({})
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    motion_events = []
    obstacles = []
    target = np.array([0.3, -0.5, 0.9])
    trajectory = np.array([[0.0] * 7, [0.1] * 7])
    plan_result = {"status": "Success", "position": trajectory}
    if failure == "plan":
        plan_result = {"status": "Collision"}
    elif failure == "empty_plan":
        plan_result["position"] = np.empty((0, 7))
    planner = SimpleNamespace(
        solve_ik=lambda *_args, **_kwargs: SimpleNamespace(
            success=failure != "ik", position=trajectory,
        ),
        _pad_qpos=lambda qpos: qpos,
        _mplib=SimpleNamespace(plan_qpos=lambda **_kwargs: plan_result),
    )
    arm = SimpleNamespace(
        planner=planner, tcp_offset=Pose(), n_arm=7, gripper_val=0.5,
        get_arm_qpos=lambda: np.zeros(7),
        get_ee_pose=lambda: np.r_[target, [1, 0, 0, 0]],
    )
    task.robot = SimpleNamespace(
        get_arm=lambda _tag: arm,
        set_arm_joints=lambda *_args: motion_events.append("arm"),
        set_gripper=lambda *_args: motion_events.append("gripper"),
    )
    task.step_sim = lambda: motion_events.append("step")
    task.execute_plan = lambda *_args: motion_events.append("execute")
    task._refresh_avatar_planner_obstacles = lambda tag: obstacles.append(tag)
    task._drawer_plan_endpoint_tcp = lambda *_args: task._top_down_tcp(
        target + np.array([0.1 if failure == "endpoint" else 0.0, 0, 0]),
    )

    assert task._drive_to_tcp(target, "right", n_settle=3) is False
    assert obstacles == ["right"]
    assert motion_events == []


def test_drawer_waypoint_holds_only_the_verified_plan_endpoint():
    task = PutObjectCabinet({})
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    target = np.array([0.3, -0.5, 0.9])
    endpoint_qpos = np.arange(7, dtype=float) / 20
    trajectory = np.vstack([np.zeros(7), endpoint_qpos])
    events = []
    planner = SimpleNamespace(
        solve_ik=lambda *_args, **_kwargs: SimpleNamespace(success=True, position=trajectory),
        _pad_qpos=lambda qpos: qpos,
        _mplib=SimpleNamespace(plan_qpos=lambda **_kwargs: {
            "status": "Success", "position": trajectory,
        }),
    )
    arm = SimpleNamespace(
        planner=planner, tcp_offset=Pose(), n_arm=7, gripper_val=0.5,
        get_arm_qpos=lambda: np.zeros(7),
        get_ee_pose=lambda: task._top_down_tcp(target).to_pose7(),
    )
    held = []
    task.robot = SimpleNamespace(
        get_arm=lambda _tag: arm,
        set_arm_joints=lambda qpos, _tag: held.append(qpos.copy()),
        set_gripper=lambda *_args: None,
    )
    task.step_sim = lambda: events.append("step")
    task.execute_plan = lambda *_args: events.append("execute")
    task._refresh_avatar_planner_obstacles = lambda _tag: events.append("obstacles")
    task._drawer_plan_endpoint_tcp = lambda *_args: task._top_down_tcp(target)

    assert task._drive_to_tcp(target, "right", n_settle=2) is True
    assert events == ["obstacles", "execute", "step", "step"]
    assert len(held) == 2
    np.testing.assert_array_equal(held[0], endpoint_qpos)
    np.testing.assert_array_equal(held[1], endpoint_qpos)


@pytest.mark.parametrize("failure", [None, "collision", "exception", "empty",
    "endpoint", "lateral", "rotation", "invalid_fk", "invalid_start_fk", "joint_jump", "arrival"])
def test_straight_drawer_pull_rejects_detours_without_rrt_or_unplanned_driving(failure):
    task = PutObjectCabinet({"robot_type": "franka"})
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=1/300))
    start = np.array([.3, -.5, .9])
    target = start + [0., .02, 0.]
    goal = task._handle_tcp(target)
    trajectory = np.tile(np.array([0., .05, .1])[:, None], (1, 7))
    if failure == "joint_jump":
        trajectory[-1] = 2.
    events, calls = [], []

    def forbidden(*args, **kwargs):
        pytest.fail("a clamped drawer pull must not fall back to IK/RRT")

    def screw(**kwargs):
        calls.append(kwargs)
        if failure == "exception":
            raise RuntimeError("collision")
        return {"status": "Collision" if failure == "collision" else "Success",
                "position": np.empty((0, 7)) if failure == "empty" else trajectory}

    planner = SimpleNamespace(solve_ik=forbidden, _pad_qpos=lambda q: q,
        _to_mplib_pose=lambda p: p,
        _mplib=SimpleNamespace(plan_screw=screw, plan_qpos=forbidden))
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), n_arm=7,
        gripper_val=0., get_arm_qpos=lambda: np.zeros(7),
        get_ee_pose=lambda: np.r_[target+[.05 if failure == "arrival" else 0.,0.,0.],goal.q])
    task.robot = SimpleNamespace(get_arm=lambda _: arm,
        set_arm_joints=lambda *_: events.append("hold"), set_gripper=lambda *_: None)
    task.step_sim = lambda: events.append("step")
    task.execute_plan = lambda *_: events.append("execute")
    task._refresh_avatar_planner_obstacles = lambda _: events.append("avatar")

    def fk(_tag, q):
        if failure == "invalid_start_fk" and np.isclose(q[0], 0.):
            return None
        position = start + [0., .2*q[0], 0.]
        pose = task._handle_tcp(position)
        if np.isclose(q[0], .05):
            if failure == "invalid_fk":
                return None
            if failure == "lateral":
                pose = task._handle_tcp(position + [.02, 0., 0.])
            if failure == "rotation":
                pose = Pose(position, [1., 0., 0., 0.])
        if np.isclose(q[0], .1) and failure == "endpoint":
            pose = task._handle_tcp(position + [.03, 0., 0.])
        return pose

    task._drawer_plan_endpoint_tcp = fk
    assert task._drive_to_tcp(target, "right", n_settle=1,
        tcp_pose=goal, straight_pull=True) is (failure is None)
    # A failed (but not raising) straight pull is re-integrated at the
    # remaining screw steps; the first call keeps the original step.
    steps = task.DRAWER_PULL_SCREW_QPOS_STEPS
    expected = (0 if failure == "invalid_start_fk"
                else len(steps) if failure == "collision" else 1)
    assert len(calls) == expected
    assert [c["qpos_step"] for c in calls] == list(steps[:expected])
    if calls:
        assert calls[0]["time_step"] == task.scene.sim.dt
        assert calls[0]["wrt_world"] is True
    assert events == (["avatar", "execute", "hold", "step"]
                      if failure in (None, "arrival") else ["avatar"])


@pytest.mark.parametrize("nominal_rotation_error", [0., .08])
def test_native_drawer_translation_reaches_captured_final_waypoint_without_relaxing_guard(
    nominal_rotation_error,
):
    """V7 seed1's 2.029mm screw endpoint miss must not become an accepted miss."""
    from transforms3d.quaternions import axangle2quat, qmult
    from envs.robot.franka_robot import FrankaArm
    from envs.planning.mplib_planner import MplibPlanner

    task = PutObjectCabinet({"robot_type": "franka"})
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    origin = Pose([0., -.65, .75], [2**-.5, 0., 0., 2**-.5])
    arm = FrankaArm(None, origin)
    folder = ASSETS_PATH / "embodiments/franka-panda"
    arm.planner = MplibPlanner(str(folder/"panda.urdf"), str(folder/"panda.srdf"),
                              "panda_hand", origin.to_pose7(), n_arm=7)
    seed = np.array([.24261419475078583, .41137585043907166, -.8949066996574402,
                     -2.1178038120269775, -.6514959931373596, 1.1191349029541016,
                     1.6558592319488525])
    arm.get_arm_qpos = lambda: seed.copy()
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    goal = task._handle_tcp(np.array([.31984348595142365, -.4396015876531601,
                                     1.0753572101593019]))
    if nominal_rotation_error:
        goal = Pose(goal.p, qmult(goal.q, axangle2quat([0., 0., 1.], nominal_rotation_error)))
    measured = [task._drawer_plan_endpoint_tcp("right", seed)]
    arm.get_ee_pose = lambda: measured[0].to_pose7()
    executed, refreshed = [], []
    task._refresh_avatar_planner_obstacles = lambda tag: refreshed.append(tag)
    task.step_sim = lambda: pytest.fail("native regression does not execute physics")

    def execute(plan, tag):
        assert task._cabinet_plan_is_physical_clock
        executed.append(plan)
        measured[0] = task._drawer_plan_endpoint_tcp(tag, plan.position[-1])

    task.execute_plan = execute
    assert task._drive_to_tcp(goal.p, "right", n_settle=0, tcp_pose=goal,
                              straight_pull=True) is (nominal_rotation_error == 0.)
    assert refreshed == ["right"]
    assert not getattr(task, "_cabinet_plan_is_physical_clock", False)
    if nominal_rotation_error:
        assert executed == [], "preserving measured orientation must not loosen nominal alignment"
    else:
        assert len(executed) == 1
        assert np.linalg.norm(measured[0].p-goal.p) < .0001
        path = np.vstack([seed, executed[0].position])
        assert task._drawer_pull_path_is_straight("right", path, goal)


@pytest.mark.parametrize("robot", ["franka", "other"])
def test_straight_drawer_motion_is_scoped_to_postclose_franka_pull(monkeypatch, robot):
    from envs.task_bases import put_object_cabinet as module
    task = PutObjectCabinet({"robot_type": robot})
    task.robot = SimpleNamespace(get_arm=lambda _: SimpleNamespace())
    task._drawer_handle_world0 = np.array([.35, -.53, 1.075])
    task._drawer_open_qpos_target = .1
    task._drawer_dof_idx = 2
    task.cabinet = SimpleNamespace(entity=SimpleNamespace(get_qpos=lambda: np.zeros(3)))
    task.open_gripper = lambda _: None
    task.set_gripper = lambda *_: None
    task.step_sim = lambda: None
    monkeypatch.setattr(module, "control_dof_force_toward_position_compat",
                        lambda *_: (True, 0., .1, 2.))
    motions = []

    class AtPull(Exception):
        pass

    def move(*_args, **kwargs):
        motions.append(kwargs)
        if len(motions) == 3:
            raise AtPull
        return True

    task._drive_to_tcp = move
    with pytest.raises(AtPull):
        task._open_drawer_physics("right")
    assert all(not m.get("straight_pull", False) for m in motions[:2])
    assert motions[2]["straight_pull"] is (robot == "franka")


@pytest.mark.parametrize("failure", [None, "plan", "force", "settle_force"])
def test_drawer_rail_assistance_tracks_hand_waypoints_and_stops_on_rejection(monkeypatch, failure):
    from envs.task_bases import put_object_cabinet as module
    task = PutObjectCabinet({"robot_type": "franka"})
    task.robot = SimpleNamespace(get_arm=lambda _: SimpleNamespace())
    task._drawer_handle_world0 = np.array([.35, -.53, 1.075])
    task._drawer_open_qpos_target = .16
    task._drawer_dof_idx = 2
    drawer = np.zeros(3)
    task.cabinet = SimpleNamespace(entity=SimpleNamespace(get_qpos=lambda: drawer.copy()))
    phase = ["approach"]
    forces, cleared, motions, steps, held = [], [], [], [], []
    task.open_gripper = lambda _: None
    task.set_gripper = lambda *_: None
    task.step_sim = lambda: steps.append(phase[0])
    task._hold_drawer_open = lambda: held.append(task._drawer_held_qpos) or True

    def force(_entity, _index, target, **kwargs):
        assert kwargs == {}, "preserve existing force gains and limits"
        forces.append((phase[0], target))
        ok = not (failure == "force" and len(forces) == 1
                  or failure == "settle_force" and phase[0] == "settle")
        return ok, drawer[2], 0., .4

    def move(position, _tag, **kwargs):
        if not kwargs.get("straight_pull", False):
            return True
        phase[0] = "pull"
        motions.append(position.copy())
        assert forces[-1][1] == pytest.approx(position[1] - (task._drawer_handle_world0[1]+.010))
        if failure == "plan":
            return False
        drawer[2] = forces[-1][1]
        # Simulate a legal early finish within the unchanged opening tolerance.
        if len(motions) == 7:
            drawer[2] = task._drawer_open_qpos_target-.005
            phase[0] = "settle"
        return True

    task._drive_to_tcp = move
    monkeypatch.setattr(module, "control_dof_force_toward_position_compat", force)
    monkeypatch.setattr(module, "control_dof_force_compat",
                        lambda _entity, _index, value: cleared.append(value) or True)
    assert task._open_drawer_physics("right") is (failure is None)
    assert cleared == [0.]
    if failure in ("plan", "force"):
        assert len(forces) == 1
        assert len(steps) == 120, "no powered simulation steps after rejection"
        assert held == []
    else:
        np.testing.assert_allclose([value for _, value in forces[:7]], np.arange(1,8)*.02)
        assert all(value == pytest.approx(.14) for _, value in forces[7:])
        if failure == "settle_force":
            assert len(forces) == 8 and len(steps) == 120 and held == []
        else:
            assert len(forces) == 67 and held == [pytest.approx(.155)]


@pytest.mark.parametrize(
    ("task_cls", "module", "instruction", "uses_avatar"),
    (
        (
            PutObjectCabinet,
            "envs.task_bases.put_object_cabinet",
            "place the object inside the open cabinet drawer",
            False,
        ),
        (
            PutObjectInCabinetContribution,
            "envs.tasks.put_object_in_cabinet_contribution",
            "place the object inside the human-opened cabinet drawer",
            True,
        ),
        (
            PutObjectInCabinetInterruption,
            "envs.tasks.put_object_in_cabinet_interruption",
            "place the object inside the cabinet drawer after the human inspects it",
            True,
        ),
        (
            PutObjectInCabinetCopresence,
            "envs.tasks.put_object_in_cabinet_copresence",
            "put the table object in the cabinet drawer while the human works nearby on the same table",
            True,
        ),
    ),
)
def test_public_identity_and_family_surface(task_cls, module, instruction, uses_avatar):
    assert task_cls.__module__ == module
    assert task_cls.__name__.startswith(("PutObjectCabinet", "PutObjectInCabinet"))
    assert issubclass(task_cls, BaseTask)
    assert task_cls.INSTRUCTION == instruction
    if task_cls in (PutObjectInCabinetInterruption, PutObjectInCabinetContribution):
        assert task_cls.OBJECT_SET == "cabinet_interrupt_pick_pool"
    else:
        assert task_cls.OBJECT_SET == "tabletop_pick_pool"
    assert bool(task_cls.use_avatar) is uses_avatar

    signature = inspect.signature(task_cls)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method in ("load_actors", "play_once", "check_success", "evaluate"):
        assert callable(getattr(task_cls, method))


def test_family_core_and_interaction_axes_are_separate():
    assert issubclass(PutObjectCabinet, CabinetTaskCore)
    assert issubclass(PutObjectInCabinetContribution, CabinetTaskCore)
    assert issubclass(PutObjectInCabinetInterruption, CabinetTaskCore)
    assert not issubclass(PutObjectInCabinetContribution, PutObjectCabinet)
    assert not issubclass(PutObjectInCabinetInterruption, PutObjectCabinet)
    assert InspectMotionMixin not in PutObjectInCabinetInterruption.__mro__

    assist = PutObjectInCabinetContribution({})
    interrupt = PutObjectInCabinetInterruption({})
    assert isinstance(assist.interaction_controller, CabinetAssistController)
    assert isinstance(
        interrupt.interaction_controller, CabinetInterruptController,
    )
    assert assist.interaction_controller.handles_policy_ticks is True
    assert interrupt.interaction_controller.handles_policy_ticks is True
    assert assist.interaction_controller.handles_scripted_rollout_ticks is False
    # The expert harness ticks the interrupt at policy cadence
    # through advance_interaction_policy_step (play_blind_once no longer
    # self-drives a 5-sim-step tick).
    assert (
        interrupt.interaction_controller.handles_scripted_rollout_ticks is True
    )


def test_neutral_variant_keeps_base_robot_behavior_and_workspace_contract():
    assert issubclass(PutObjectInCabinetCopresence, PutObjectCabinet)
    spec = PutObjectInCabinetCopresence.NEUTRAL_TASK_SPEC
    assert spec.work_xy == (0.42, 0.12)
    assert spec.mouse_work_xy == (0.42, 0.16)
    assert spec.table_xy_bounds == (-0.50, 0.50, -0.80, 0.10)
    assert spec.work_candidates == (
        (0.42, 0.12),
        (0.42, 0.20),
        (0.34, 0.16),
        (0.50, 0.12),
    )


@pytest.mark.parametrize("task_cls", ALL_CLASSES)
def test_constructor_does_not_mutate_input_and_keeps_robot_defaults(task_cls):
    config = {"renderer": "rasterizer", "object_name": "048_stapler"}
    original = dict(config)
    task = task_cls(config)

    assert config == original
    assert task.config["robot_type"] == "franka"
    assert task.config["robot_single_arm"] is True
    assert task.config["object_name"] == "048_stapler"
    assert task._object_id == "048_stapler"
    assert task._object_model_id == 0
    assert task._object_is_primitive is False
    np.testing.assert_array_equal(task._cabinet_pos, [0.30, -0.65, 0.927])


def test_variant_specific_configuration_defaults_and_overrides():
    assist = PutObjectInCabinetContribution({})
    assert assist.config["track_avatar_collision"] is False
    assert assist.EVAL_TRIGGER_STEP_MIN == 20
    assert assist.EVAL_TRIGGER_STEP_MAX == 60

    interrupt = PutObjectInCabinetInterruption({
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 9,
    })
    # The trigger-step bounds only size the burned legacy draw;
    # the per-task glide constant moved to the shared interrupt contract.
    assert interrupt.EVAL_TRIGGER_STEP_MIN == 7
    assert interrupt.EVAL_TRIGGER_STEP_MAX == 9
    assert not hasattr(interrupt, "EVAL_GLIDE_POLICY_STEPS")
    assert interrupt.INTERRUPT_COMMIT_RADIUS_M == 0.22
    assert interrupt.INTERRUPT_DRAWER_OPEN_QPOS == 0.050
    assert interrupt.INSPECT_MOTION_NAME == "Inspect2"
    assert interrupt.INSPECT_ATTACH_FRAME == 31
    assert interrupt.INSPECT_DETACH_FRAME == 84
    assert interrupt.INSPECT_RETURN_TO_IDLE_FRAMES == 30


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
@pytest.mark.parametrize("seed", (0, 7, 42))
def test_seeded_cabinet_sampling_and_global_rng_tail(task_cls, seed):
    task = task_cls({})

    np.random.seed(seed)
    expected = np.array([
        np.random.uniform(0.24, 0.36),
        np.random.uniform(-0.67, -0.61),
        0.927,
    ])
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual = task._sample_cabinet_pos()
    actual_tail = np.random.random_sample(8)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_tail, expected_tail)


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
@pytest.mark.parametrize("seed", (1, 19))
def test_seeded_object_layout_and_global_rng_tail(task_cls, seed):
    task = task_cls({})
    extents = np.array([0.06, 0.04, 0.08], dtype=np.float64)
    cabinet_pos = np.array([0.30, -0.65, 0.927], dtype=np.float64)

    np.random.seed(seed)
    # Spawn band mirrored left of center, clear of the
    # cabinet's drawer-open approach corridor.
    expected = (
        float(np.random.uniform(-0.14, -0.02)),
        float(np.random.uniform(-0.34, -0.20)),
    )
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual = task._sample_object_xy(extents, cabinet_pos)
    actual_tail = np.random.random_sample(8)

    assert actual == expected
    np.testing.assert_array_equal(actual_tail, expected_tail)


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_forced_catalog_object_is_stable_and_consumes_no_rng(task_cls):
    task = task_cls({"object_name": "048_stapler"})
    np.random.seed(113)
    expected_tail = np.random.random_sample(8)
    np.random.seed(113)
    spec = task._sample_object_spec()
    actual_tail = np.random.random_sample(8)

    assert spec["object_id"] == "048_stapler"
    assert int(spec.get("model_id", 0)) == 0
    assert "quat" in spec
    np.testing.assert_array_equal(actual_tail, expected_tail)


@pytest.mark.parametrize(
    ("task_name", "task_cls", "expert_mode"),
    (
        ("put_object_cabinet", PutObjectCabinet, "play_once"),
        ("put_object_in_cabinet_contribution", PutObjectInCabinetContribution, "play_blind_once"),
        ("put_object_in_cabinet_interruption", PutObjectInCabinetInterruption, "play_blind_once"),
        ("put_object_in_cabinet_copresence", PutObjectInCabinetCopresence, "play_once"),
    ),
)
def test_expert_routing_surface(task_name, task_cls, expert_mode):
    assert policy_play_mode(task_name, task_cls, "expert") == (expert_mode, None)
    # Interrupt tasks run the full-state expert in eval mode under
    # the interrupt contract (interrupt_expert_attempt + yield).
    full_state = (
        "interrupt_full_state" if task_name.endswith("_interruption") else "play_once"
    )
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        full_state,
        None,
    )


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_full_state_rollout_settles_before_missing_scene_short_circuit(task_cls):
    task = task_cls({})
    task.cabinet = None
    task.object = None
    steps = []
    task.step_sim = lambda: steps.append("step")
    task.suppress_recording = nullcontext

    assert task.play_once() is False
    assert steps == ["step"] * 80


@pytest.mark.parametrize("task_cls", (PutObjectInCabinetContribution, PutObjectInCabinetInterruption))
def test_blind_rollout_settles_before_missing_scene_short_circuit(task_cls):
    task = task_cls({"eval_mode": False})
    task.cabinet = None
    task.object = None
    steps = []
    original_step = lambda: steps.append("step")
    task.step_sim = original_step

    assert task.play_blind_once() is False
    assert steps == ["step"] * 80
    assert task.step_sim is original_step


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_missing_scene_metric_and_success_surface(task_cls):
    task = task_cls({})
    task.cabinet = None
    task.object = None
    task._drawer_link = None

    metrics = task._object_in_drawer_metrics()
    assert metrics == {
        "object_in_drawer": False,
        "object_in_drawer_error": "missing drawer link or object",
    }
    assert task.check_success() is False


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_success_reads_only_the_cabinet_metric(task_cls):
    task = task_cls({})
    task._object_in_drawer_metrics = lambda: {"object_in_drawer": True}
    assert task.check_success() is True
    task._object_in_drawer_metrics = lambda: {
        "object_in_drawer": True,
        "object_in_drawer_error": "bad geometry",
    }
    assert task.check_success() is False


def test_assist_policy_hook_precedes_robot_action_and_preserves_arguments(monkeypatch):
    task = PutObjectInCabinetContribution({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("avatar",))

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "observation"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = object()
    task.advance_interaction_policy_step()
    assert task.take_action(action, action_type="qpos_abs") == "observation"
    assert events == [("avatar",), ("robot", action, "qpos_abs")]


def test_interrupt_policy_hook_precedes_robot_action_and_preserves_arguments(monkeypatch):
    task = PutObjectInCabinetInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("avatar",))

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "observation"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = object()
    task.advance_interaction_policy_step()
    assert task.take_action(action, action_type="qpos_abs") == "observation"
    assert events == [("avatar",), ("robot", action, "qpos_abs")]


@pytest.mark.parametrize(("eval_mode", "expected_hook"), ((False, "calibrate"), (True, "eval")))
def test_interrupt_reset_routes_avatar_setup_after_base_reset(
    eval_mode, expected_hook, monkeypatch
):
    task = PutObjectInCabinetInterruption({"eval_mode": eval_mode})
    task.cabinet = None
    task._drawer_link = None
    task.avatar = SimpleNamespace(frame_ratio=None)
    events = []

    monkeypatch.setattr(BaseTask, "reset", lambda _self, seed=0: events.append(("base", seed)) or "obs")
    task._resolve_inspect_motion = lambda: events.append(("inspect",))
    task._calibrate_avatar_for_inspect = lambda: events.append(("calibrate",))
    task._init_eval_interrupt_state = lambda: events.append(("eval",))

    observation = task.reset(seed=17)
    task.interaction_controller.after_base_reset_observation(
        task, observation=observation,
    )
    assert observation == "obs"
    assert events == [("base", 17), ("inspect",), (expected_hook,)]
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW


class _MetricEntity:
    def __init__(self, *, aabb=None, qpos=None, reads=None, label="entity"):
        self._aabb = None if aabb is None else np.asarray(aabb, dtype=np.float64)
        self._qpos = None if qpos is None else np.asarray(qpos, dtype=np.float64)
        self._reads = reads if reads is not None else {}
        self._label = label

    def _read(self, suffix):
        key = f"{self._label}_{suffix}"
        self._reads[key] = self._reads.get(key, 0) + 1

    def get_AABB(self):
        self._read("aabb")
        if self._aabb is None:
            raise AssertionError(f"{self._label} has no AABB")
        return self._aabb.copy()

    def get_qpos(self):
        self._read("qpos")
        if self._qpos is None:
            raise AssertionError(f"{self._label} has no qpos")
        return self._qpos.copy()


class _MetricDrawer:
    def __init__(self, *, aabb, pos, reads):
        self._entity = _MetricEntity(
            aabb=aabb,
            reads=reads,
            label="drawer",
        )
        self._pos = np.asarray(pos, dtype=np.float64)
        self._reads = reads

    def get_AABB(self):
        return self._entity.get_AABB()

    def get_pos(self):
        self._reads["drawer_pos"] = self._reads.get("drawer_pos", 0) + 1
        return self._pos.copy()


class _MetricObject:
    def __init__(self, center, *, aabb=None, reads=None):
        self.center = np.asarray(center, dtype=np.float64)
        self.reads = reads if reads is not None else {}
        if aabb is None:
            aabb = np.array([
                self.center - np.array([0.03, 0.04, 0.05]),
                self.center + np.array([0.03, 0.04, 0.05]),
            ])
        self.entity = _MetricEntity(
            aabb=aabb,
            reads=self.reads,
            label="object",
        )

    def get_pose(self):
        self.reads["primitive_pose"] = self.reads.get("primitive_pose", 0) + 1
        return SimpleNamespace(p=self.center.copy())


def _configured_metric_task(task_cls, center=(0.40, 0.50, 0.30), *, primitive=True):
    reads = {}
    task = task_cls({})
    task._object_id = "unit_payload"
    task._object_model_id = 3
    task._object_is_primitive = bool(primitive)
    task._object_grasp_z_offset = 0.027
    task._cabinet_pos = np.array([0.10, 0.0, 0.90], dtype=np.float64)
    task._drawer_floor_z = 0.20
    task._drawer_rim_z = 0.60
    task._drawer_dof_idx = 1
    task._drawer_link = _MetricDrawer(
        aabb=((0.0, 0.0, 0.10), (1.0, 1.0, 0.70)),
        pos=(0.30, 0.40, 0.25),
        reads=reads,
    )
    task.cabinet = SimpleNamespace(
        entity=_MetricEntity(
            qpos=(0.0, 0.123, 0.0),
            reads=reads,
            label="cabinet",
        )
    )
    task.object = _MetricObject(center, reads=reads)
    task.robot = SimpleNamespace(
        right_arm=SimpleNamespace(
            get_ee_pose=lambda: reads.__setitem__(
                "ee_pose", reads.get("ee_pose", 0) + 1
            ) or np.array([0.25, 0.35, 0.45, 1.0, 0.1, 0.2, 0.3]),
            gripper_val=0.37,
        )
    )
    if not primitive:
        def mesh_center(actor, object_name, model_id):
            reads["mesh_center"] = reads.get("mesh_center", 0) + 1
            assert actor is task.object
            assert object_name == "unit_payload"
            assert model_id == 3
            return task.object.center.copy()

        task._get_object_world_center = mesh_center
    return task, task.object, reads


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_exact_scoring_profile_values_key_order_and_single_live_read(task_cls):
    task, _object, reads = _configured_metric_task(task_cls)
    metrics = task._object_in_drawer_metrics()

    expected = {
        "object_in_drawer": True,
        "object_in_drawer_xy": True,
        "object_in_drawer_aabb_xy": True,
        "object_in_drawer_exposed": True,
        "object_in_drawer_z": True,
        "object_name": "unit_payload",
        "object_model_id": 3,
    }
    if task_cls is PutObjectInCabinetContribution:
        expected["object_grasp_z_offset"] = 0.027
    expected.update({
        "object_center": [0.40, 0.50, 0.30],
        "drawer_center": [0.30, 0.40, 0.25],
        "drawer_aabb_min": [0.0, 0.0, 0.10],
        "drawer_aabb_max": [1.0, 1.0, 0.70],
        "drawer_floor_z": 0.20,
        "drawer_rim_z": 0.60,
        "drawer_qpos": 0.123,
        "cabinet_pos": [0.10, 0.0, 0.90],
        "cabinet_front_y": 0.094,
    })
    if task_cls is PutObjectCabinet:
        expected.update({
            "ee_pose": [0.25, 0.35, 0.45, 1.0, 0.1, 0.2, 0.3],
            "ee_in_drawer_aabb": True,
            "ee_to_drawer_center": float(np.linalg.norm(
                np.array([0.25, 0.35, 0.45])
                - np.array([0.30, 0.40, 0.25])
            )),
            "gripper_open_value": 0.37,
        })

    assert metrics == expected
    assert list(metrics) == list(expected)
    assert reads["primitive_pose"] == 1
    assert reads["drawer_aabb"] == 1
    assert reads["drawer_pos"] == 1
    assert reads["cabinet_qpos"] == 1
    assert reads.get("ee_pose", 0) == (1 if task_cls is PutObjectCabinet else 0)


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_mesh_scoring_routes_through_corrected_world_center_once(task_cls):
    task, object_actor, reads = _configured_metric_task(
        task_cls,
        primitive=False,
    )
    object_actor.get_pose = lambda: pytest.fail("mesh scoring used primitive pose")

    metrics = task._object_in_drawer_metrics()
    assert metrics["object_center"] == [0.40, 0.50, 0.30]
    assert reads["mesh_center"] == 1


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
@pytest.mark.parametrize(
    ("boundary", "expected"),
    (
        (
            "aabb_wall_margin",
            {
                "object_in_drawer": False,
                "object_in_drawer_xy": False,
                "object_in_drawer_aabb_xy": False,
                "object_in_drawer_exposed": True,
                "object_in_drawer_z": True,
            },
        ),
        (
            "cabinet_front",
            {
                "object_in_drawer": False,
                "object_in_drawer_xy": False,
                "object_in_drawer_aabb_xy": True,
                "object_in_drawer_exposed": False,
                "object_in_drawer_z": True,
            },
        ),
        (
            "drawer_floor_lower_tolerance",
            {
                "object_in_drawer": False,
                "object_in_drawer_xy": True,
                "object_in_drawer_aabb_xy": True,
                "object_in_drawer_exposed": True,
                "object_in_drawer_z": False,
            },
        ),
        (
            "drawer_rim_upper_tolerance",
            {
                "object_in_drawer": False,
                "object_in_drawer_xy": True,
                "object_in_drawer_aabb_xy": True,
                "object_in_drawer_exposed": True,
                "object_in_drawer_z": False,
            },
        ),
    ),
)
def test_scoring_boundaries_are_strict(task_cls, boundary, expected):
    task, object_actor, _reads = _configured_metric_task(task_cls)
    if boundary == "aabb_wall_margin":
        object_actor.center[0] = 0.0 + 0.005
    elif boundary == "cabinet_front":
        object_actor.center[1] = float(task._cabinet_pos[1]) + 0.094
    elif boundary == "drawer_floor_lower_tolerance":
        object_actor.center[2] = task._drawer_floor_z - 0.05
    elif boundary == "drawer_rim_upper_tolerance":
        object_actor.center[2] = task._drawer_rim_z + 0.02
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(boundary)

    metrics = task._object_in_drawer_metrics()
    assert {name: metrics[name] for name in expected} == expected


def test_base_scoring_lazily_initializes_drawer_once_before_live_diagnostics():
    task, _object, reads = _configured_metric_task(PutObjectCabinet)
    resolved_drawer = task._drawer_link
    task._drawer_link = None
    events = []

    def pick_drawer():
        events.append("pick_drawer")
        task._drawer_link = resolved_drawer

    task._pick_drawer = pick_drawer
    metrics = task._object_in_drawer_metrics()
    assert events == ["pick_drawer"]
    assert metrics["object_in_drawer"] is True
    assert metrics["ee_pose"] == [0.25, 0.35, 0.45, 1.0, 0.1, 0.2, 0.3]
    assert metrics["ee_in_drawer_aabb"] is True
    assert reads["ee_pose"] == 1


def test_base_scoring_reports_exact_lazy_drawer_initialization_error():
    task = PutObjectCabinet({})
    task.cabinet = object()
    task.object = object()
    task._drawer_link = None

    def fail_drawer_init():
        raise RuntimeError("broken rail")

    task._pick_drawer = fail_drawer_init
    assert task._object_in_drawer_metrics() == {
        "object_in_drawer": False,
        "object_in_drawer_error": "drawer init failed: broken rail",
    }


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
def test_evaluate_metric_order_and_legacy_or_deduplicated_read_count(task_cls):
    task = task_cls({"eval_mode": task_cls is PutObjectInCabinetContribution})
    task.plan_success = True
    task.avatar_collision_summary = lambda: {"enabled": False}
    task.safe_distance_summary = lambda: {"enabled": False}
    task.avatar_safety_summary = lambda: {"enabled": False}
    task.avatar_safety_intervention_summary = lambda: {"enabled": False}
    reads = []

    def metrics():
        reads.append(len(reads) + 1)
        return {
            "object_in_drawer": True,
            "metric_read": reads[-1],
        }

    task._object_in_drawer_metrics = metrics
    result = task.evaluate()

    # BaseTask.check_success performs the first read; the family evaluation
    # performs the second. The current assist/interrupt wrappers perform one
    # redundant third update, while the intended profile/core target removes
    # that duplicate. Both graphs have the same externally ordered result.
    if task_cls is PutObjectCabinet:
        assert len(reads) == 2
    else:
        assert len(reads) in (2, 3)
    assert result["metric_read"] == len(reads)

    expected_keys = [
        "plan_success",
        "success",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        # BaseTask safety-metrics block.
        "safety",
        "object_in_drawer",
        "metric_read",
    ]
    if task_cls is PutObjectInCabinetContribution:
        expected_keys.extend([
            "eval_mode",
            "eval_trigger_step",
            "eval_policy_step_count",
            "eval_avatar_started",
            "eval_avatar_failed",
            "eval_drawer_sync_active",
            "eval_avatar_body_return_active",
        ])
    assert list(result) == expected_keys


@pytest.mark.parametrize("task_cls", CORE_CLASSES)
@pytest.mark.parametrize(
    ("held_qpos", "expected_target"),
    ((None, 0.051), (0.083, 0.083)),
)
@pytest.mark.parametrize("compat_success", (False, True))
def test_drawer_hold_target_pd_arguments_and_success_only_proxy_sync(
    task_cls,
    held_qpos,
    expected_target,
    compat_success,
    monkeypatch,
):
    task = task_cls({})
    entity = object()
    drawer = object()
    proxy = object()
    task.cabinet = SimpleNamespace(entity=entity)
    task._drawer_dof_idx = 4
    task._drawer_open_qpos_target = 0.051
    task._drawer_held_qpos = held_qpos
    task._drawer_link = drawer
    task._drawer_floor_proxy = proxy
    task._drawer_floor_z = 0.72
    events = []

    def hold_compat(actual_entity, dof_index, target, **kwargs):
        events.append((
            "hold",
            actual_entity,
            dof_index,
            target,
            kwargs,
        ))
        return compat_success

    method_globals = task._hold_drawer_open.__func__.__globals__
    monkeypatch.setitem(
        method_globals,
        "hold_dof_position_compat",
        hold_compat,
    )
    task.sync_support_proxy = lambda actual_proxy, actual_drawer, floor_z: (
        events.append(("sync", actual_proxy, actual_drawer, floor_z))
    )

    assert task._hold_drawer_open() is compat_success
    assert events[0] == (
        "hold",
        entity,
        4,
        expected_target,
        {
            "kp": task._DRAWER_HOLD_KP,
            "kv": task._DRAWER_HOLD_KV,
            "force_limit": task._DRAWER_HOLD_FORCE,
        },
    )
    support_z = 0.7234822
    expected_sync = [("sync", proxy, drawer, support_z)] if compat_success else []
    assert events[1:] == expected_sync


def test_interrupt_proxy_sync_guard_requires_complete_drawer_state():
    task = PutObjectInCabinetInterruption({})
    proxy = object()
    drawer = object()
    task.sync_support_proxy = lambda *args: pytest.fail(
        f"incomplete interrupt state synchronized: {args!r}"
    )

    for missing_name in (
        "_drawer_floor_proxy",
        "_drawer_link",
        "_drawer_floor_z",
    ):
        task._drawer_floor_proxy = proxy
        task._drawer_link = drawer
        task._drawer_floor_z = 0.72
        setattr(task, missing_name, None)
        task._sync_drawer_floor_proxy()

    events = []
    task._drawer_floor_proxy = proxy
    task._drawer_link = drawer
    task._drawer_floor_z = 0.72
    task.sync_support_proxy = lambda *args: events.append(args)
    task._sync_drawer_floor_proxy()
    assert len(events) == 1 and events[0][:2] == (proxy, drawer)
    assert events[0][2] == pytest.approx(0.7234822)


@pytest.mark.parametrize("scale", [.2, .3])
@pytest.mark.parametrize("task_cls", [PutObjectCabinet, PutObjectInCabinetContribution,
                                    PutObjectInCabinetCopresence, PutObjectInCabinetInterruption])
def test_family_physical_support_matches_actual_drawer_visual_floor(scale, task_cls):
    import xml.etree.ElementTree as ET

    asset = ASSETS_PATH / "objects/036_cabinet/46653"
    drawer = ET.parse(asset / "mobility.urdf").getroot().find("link[@name='link_3']")
    heights, floor_heights = [], []
    for visual in drawer.findall("visual"):
        mesh = visual.find("geometry/mesh")
        path = asset / mesh.attrib["filename"]
        # The authored cabinet's fixed joint maps mesh Y to world Z.
        y = [float(line.split()[2]) for line in path.read_text().splitlines() if line.startswith("v ")]
        heights.extend(y)
        if visual.attrib['name'].startswith('drawer_bottom'):
            floor_heights.extend(y)
    cabinet_z = .927
    task = task_cls({})
    task._CABINET_SCALE = scale
    task._drawer_floor_z = cabinet_z + min(heights) * scale + .010
    grading_reference = task._drawer_floor_z
    assert task._drawer_support_surface_z() == pytest.approx(cabinet_z + max(floor_heights) * scale)
    assert task._drawer_floor_z == grading_reference


def test_assist_pick_spec_preserves_catalog_grasp_offset_and_drop_placement():
    task = PutObjectInCabinetContribution({})
    task.cabinet = object()
    task.avatar = object()
    task._drawer_link = _MetricDrawer(
        aabb=((0.20, -0.50, 0.70), (0.60, -0.10, 0.90)),
        pos=(0.40, -0.30, 0.78),
        reads={},
    )
    task._drawer_handle_world0 = np.array([0.40, -0.25, 0.85])
    task._drawer_open_qpos_target = 0.08
    task._drawer_floor_z = 0.72
    task._drawer_rim_z = 0.88
    task._object_id = "offset_payload"
    task._object_model_id = 0
    task._object_is_primitive = True
    task._object_radius = 0.03
    task._object_topdown_pick_options = task.topdown_pick_options({
        "grasp_z_offset": 0.027,
    })
    task._object_grasp_z_offset = 0.027
    task.object = _MetricObject(
        center=(0.13, -0.31, 0.79),
        aabb=((0.10, -0.35, 0.75), (0.16, -0.27, 0.83)),
    )

    step_count = []
    captured = []
    task.suppress_recording = nullcontext
    task.step_sim = lambda: step_count.append(1)
    task._pick_drawer = lambda: None
    task._boost_finger_pd = lambda _arm: None
    task._soften_object_gripper_pd = lambda _arm: None
    task._start_open_drawer_with_avatar = lambda: True
    task.pick_and_place = lambda pick, place, arm: captured.append(
        (pick, place, arm)
    ) or True

    assert task.play_once() is True
    assert len(step_count) == 140
    assert len(captured) == 1
    pick, place, arm_tag = captured[0]
    assert arm_tag == "right"
    assert pick.label == "offset_payload"
    assert pick.radius == 0.03
    assert pick.grasp_z_offset == 0.027
    assert pick.approach_xy_offset == (0.0, 0.0)
    assert pick.preopen_steps == 0
    assert pick.require_lift is True
    assert pick.transport_retention_tolerance == 0.08
    np.testing.assert_allclose(pick.get_center(), (0.13, -0.31, 0.79))
    assert pick.get_half_height() == pytest.approx(0.04)

    # final_front=-0.17; inset=depth/2 + clearance - held allowance
    # = 0.04 + 0.05 - 0.005 = 0.085, so the center is -0.255.
    np.testing.assert_allclose(place.pos, (0.40, -0.255, 0.7584822))
    assert place.label == "drawer"
    assert place.transport_z == pytest.approx(0.98)
    assert place.support_z == pytest.approx(0.7234822)
    assert place.support_clearance == 0.005


def test_family_strategy_switches_default_to_ledger_and_only_interrupt_overrides():
    """family_v48 composition: the core's three switches default to the ledger
    behaviour (catalog friction + 4.0 proxy, y-extent opening, Franka
    translation pull) and only the interrupt variant overrides them."""
    from envs.task_bases.put_object_cabinet import CabinetTaskCore
    from envs.tasks.put_object_in_cabinet_contribution import PutObjectInCabinetContribution
    from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
    from envs.tasks.put_object_in_cabinet_copresence import PutObjectInCabinetCopresence
    for cls in (CabinetTaskCore, PutObjectCabinet, PutObjectInCabinetContribution, PutObjectInCabinetCopresence):
        assert cls.CABINET_PAIR_FRICTION is None
        assert cls.DRAWER_OPENING_MAX_EXTENT is False
        assert cls.DRAWER_TRANSLATION_PULL is True
    assert PutObjectInCabinetInterruption.CABINET_PAIR_FRICTION == 1.0
    assert PutObjectInCabinetInterruption.DRAWER_OPENING_MAX_EXTENT is True
    assert PutObjectInCabinetInterruption.DRAWER_TRANSLATION_PULL is False
    # the switch is robot-aware: only a Franka with the flag pulls by translation
    assert PutObjectCabinet({"robot_type": "franka"})._drawer_translation_pull() is True
    assert PutObjectCabinet({"robot_type": "piper"})._drawer_translation_pull() is False
    assert PutObjectInCabinetInterruption({"robot_type": "franka"})._drawer_translation_pull() is False
    assert PutObjectInCabinetContribution({"robot_type": "franka"})._drawer_translation_pull() is True
    assert PutObjectInCabinetCopresence({"robot_type": "franka"})._drawer_translation_pull() is True


def test_pair_friction_switch_builds_payload_and_proxy_only_when_set():
    """The interrupt's pair friction 1.0 reaches both the payload load and the
    drawer-floor proxy; None keeps the catalog value and the 4.0 proxy."""
    import inspect
    from envs.task_bases.put_object_cabinet import CabinetTaskCore
    source = inspect.getsource(CabinetTaskCore.load_actors)
    assert source.count("self.CABINET_PAIR_FRICTION is None") == 2
    assert source.count("float(self.CABINET_PAIR_FRICTION)") == 2
    assert 'float(spec.get("friction", 4.0))' in source
    assert "friction=(4.0 if self.CABINET_PAIR_FRICTION is None" in source


def test_opening_switch_selects_largest_horizontal_extent_only_for_interrupt():
    """V44 sizing (largest horizontal extent) is applied only where the switch
    is set; the default keeps the ledger's spawn y-extent sizing."""
    import inspect
    from envs.task_bases.put_object_cabinet import CabinetTaskCore
    source = inspect.getsource(CabinetTaskCore._pick_drawer)
    assert "payload_extents = object_aabb[1, :2] - object_aabb[0, :2]" in source
    assert "float(np.max(payload_extents))" in source
    assert "if self.DRAWER_OPENING_MAX_EXTENT" in source
    assert "else float(payload_extents[1])" in source


def test_translation_pull_switch_gates_every_franka_specific_drawer_step():
    """All four Franka-specific translation-pull sites read the switch, and no
    raw robot_type check remains in the drawer opening path."""
    import inspect
    from envs.task_bases.put_object_cabinet import CabinetTaskCore
    source = inspect.getsource(CabinetTaskCore._open_drawer_physics)
    assert source.count("self._drawer_translation_pull()") == 4
    assert 'robot_type") == "franka"' not in source
