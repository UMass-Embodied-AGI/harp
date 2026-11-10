"""One geometry-directed oil egress; no simulator teleport or score changes."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.planning.base import PlanResult
from envs.task_bases import oil_recovery as oil
from envs.tasks.oil_bottle_recovery import OilBottleRecovery
from envs.utils import Pose
from test_oil_bottle_recovery_planning import fixture as planning_fixture, run


def geometry_fixture(points=None, start=(0., 0., 0.), end=(0., 0., 1.)):
    if points is None:
        points = [[.1, .0, .4], [.3, .2, .5]]
    task = NS(avatar_collider=NS(current_capsules=lambda: [("r_forearm", start, end, .05)]))
    arm = NS(entity=NS(links=[NS(name="link6", _geoms=[NS(get_verts=lambda: np.array(points))])]))
    return task, arm


def test_egress_uses_nearest_collision_vertex_not_tcp_and_preserves_height():
    task, arm = geometry_fixture()
    arm.get_ee_pose = lambda: [-10., 0., 0., 1., 0., 0., 0.]
    offset, report = oil.oil_lift_clearance_offset(task, arm)
    assert offset == pytest.approx([.06, 0., 0.])
    assert report["link6_nearest_vertex"] == pytest.approx([.1, 0., .4])
    assert report["initial_proxy_gap_m"] == pytest.approx(.05)


def test_egress_rotates_and_translates_with_measured_scene_geometry():
    task, arm = geometry_fixture([[.3, -.3, 1.4], [.1, -.1, 1.5]],
                                 start=(.3, -.4, 1.), end=(.3, -.4, 2.))
    offset, _ = oil.oil_lift_clearance_offset(task, arm)
    assert offset == pytest.approx([0., .06, 0.])


@pytest.mark.parametrize("points", [[[0., 0., .5]], [[np.nan, 0., .5]], []])
def test_ambiguous_or_invalid_geometry_is_not_given_a_guessed_direction(points):
    task, arm = geometry_fixture(points)
    with pytest.raises(ValueError):
        oil.oil_lift_clearance_offset(task, arm)


def path_fixture(*, colliding=False, endpoint=None):
    names = ["panda_finger_joint2"] + [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1"]
    measured = np.array([.02, *([.1] * 7), .03])
    saved = np.array([.011, *([.2] * 7), .012])
    state = [saved.copy()]
    world_state, fk_state = [saved[1:8].copy()], [saved.copy()]
    observations = []
    robot = NS(get_user_joint_names=lambda: names,
               get_qpos=lambda: state[0].copy(),
               set_qpos=lambda q, *_: state.__setitem__(0, np.asarray(q).copy()),
               get_base_pose=lambda: Pose([.5, -.4, .2], [np.sqrt(.5), 0., 0., np.sqrt(.5)]))
    def check():
        observations.append(state[0].copy())
        return [object()] if colliding else []
    def set_world(q):
        world_state[0] = np.asarray(q).copy()
        state[0][1:8] = q
    native = NS(robot=robot, move_group_joint_indices=np.arange(1, 8),
                move_group="panda_hand", link_name_2_idx={"panda_hand": 0},
                joint_limits=np.tile([-3., 3.], (9, 1)),
                planning_world=NS(get_object_names=lambda: ["scene_pcd", "existing_fixture"],
                                  set_qpos_all=set_world, check_collision=check),
                pinocchio_model=NS(compute_forward_kinematics=lambda q: fk_state.__setitem__(0, np.asarray(q).copy()),
                                   get_link_pose=lambda _: endpoint or Pose([.1, 0., .7])),
                test_world_state=world_state, test_fk_state=fk_state)
    arm = NS(planner=NS(_mplib=native))
    result = PlanResult(True, np.array([[.1]*7, [.11]*7]))
    fingers = {"cached": np.array([.025, .015]), "scalar": np.array([.0016, .0016])}
    target = (robot.get_base_pose() * Pose([.1, 0., .7])).to_pose7()
    return arm, result, measured, fingers, target, saved, observations


def test_path_checks_every_waypoint_with_actual_and_both_commanded_finger_states():
    arm, result, measured, fingers, target, saved, observations = path_fixture()
    report = oil.check_oil_lift_path(arm, result, measured, fingers, target)
    assert report["endpoint_error"]["position_m"] < 1e-9
    assert len(observations) == 6
    assert observations[0] == pytest.approx(measured)
    assert observations[2][[8, 0]] == pytest.approx(fingers["cached"])
    assert observations[4][[8, 0]] == pytest.approx(fingers["scalar"])
    assert arm.planner._mplib.robot.get_qpos() == pytest.approx(saved)
    assert arm.planner._mplib.test_world_state[0] == pytest.approx(saved[1:8])
    assert arm.planner._mplib.test_fk_state[0] == pytest.approx(saved)


@pytest.mark.parametrize("fault", ["repaired_start", "collision", "wrong_fk", "long_path", "no_cloud"])
def test_rejected_native_path_is_never_repaired_and_restores_native_state(fault):
    arm, result, measured, fingers, target, saved, _ = path_fixture(colliding=fault == "collision")
    if fault == "repaired_start": result.position[0, 0] += .002
    if fault == "wrong_fk": target[0] += .01
    if fault == "long_path": result.position = np.full((801, 7), .1); result.velocity = np.zeros_like(result.position)
    if fault == "no_cloud": arm.planner._mplib.planning_world.get_object_names = lambda: ["existing_fixture"]
    with pytest.raises(ValueError):
        oil.check_oil_lift_path(arm, result, measured, fingers, target)
    assert arm.planner._mplib.robot.get_qpos() == pytest.approx(saved)


@pytest.mark.parametrize("fault", ["finger_clobber", "query_exception"])
def test_native_world_clobber_and_query_exceptions_restore_all_native_states(fault):
    arm, result, measured, fingers, target, saved, _ = path_fixture()
    native = arm.planner._mplib
    if fault == "finger_clobber":
        original = native.planning_world.set_qpos_all
        def clobber(q):
            original(q)
            changed = native.robot.get_qpos()
            changed[0] = 0.
            native.robot.set_qpos(changed, True)
        native.planning_world.set_qpos_all = clobber
    else:
        def fail():
            raise RuntimeError("native query failed")
        native.planning_world.check_collision = fail
    with pytest.raises((ValueError, RuntimeError)):
        oil.check_oil_lift_path(arm, result, measured, fingers, target)
    assert native.robot.get_qpos() == pytest.approx(saved)
    assert native.test_world_state[0] == pytest.approx(saved[1:8])
    assert native.test_fk_state[0] == pytest.approx(saved)


def test_native_lift_keeps_tcp_offset_full_fingers_and_current_obstacles(monkeypatch):
    task, arm = geometry_fixture()
    arm.tcp_offset = Pose([0., 0., .08], [np.sqrt(.5), 0., np.sqrt(.5), 0.])
    arm.get_ee_pose = lambda: [.3, -.2, .8, 1., 0., 0., 0.]
    full = np.array([.02, *([.1]*7), .03])
    calls = []
    raw = {"status": "Success", "position": np.array([[.1]*7, [.11]*7]), "velocity": np.zeros((2, 7))}
    def screw(**kwargs):
        calls.append(kwargs)
        return raw
    arm.planner = NS(_mplib=NS(plan_screw=screw), _to_mplib_pose=Pose.from_pose7)
    monkeypatch.setattr(oil, "_require_mode", lambda *a: calls.append("mode"))
    monkeypatch.setattr(oil, "refresh_obstacles", lambda *a: calls.append("refresh_all_obstacles"))
    monkeypatch.setattr(oil, "joint_state", lambda *a: (full.copy(), {}))
    monkeypatch.setattr(oil, "check_oil_lift_path", lambda *a: {"checked": True})
    result, target = oil.plan_oil_clearance_lift(task, arm, "right", .12)
    assert result.success and calls[:2] == ["mode", "refresh_all_obstacles"]
    assert calls[2]["current_qpos"] == pytest.approx(full)
    assert calls[2]["qpos_step"] == .01 and calls[2]["wrt_world"] is True
    tcp = target * arm.tcp_offset
    assert tcp.p == pytest.approx([.36, -.2, .92])
    assert abs(np.dot(tcp.q, [1., 0., 0., 0.])) == pytest.approx(1.)


def test_candidate_carry_rejects_latched_contact_without_planning_or_waiting():
    task, arm, _, calls = planning_fixture(Pose([.2, 0., .3]))
    task.config = {"oil_clearance_lift": True}
    task.avatar_collided = True
    task.plan_success = True
    result, _ = run(task, arm)
    assert result is None and not task.plan_success and not calls


def test_candidate_carry_rejects_native_repaired_start_without_fk_or_execution(monkeypatch):
    task, arm, plan, calls = planning_fixture(Pose([.2, 0., .3]))
    task.config = {"oil_clearance_lift": True}
    task.plan_success = True
    plan.position[0, 0] = .002
    arm.planner._mplib.move_group_joint_indices = np.arange(7)
    def checked(*unused):
        calls.append("refresh")
        return np.zeros(9)
    monkeypatch.setattr(oil, "checked_oil_carry_start", checked)
    result, _ = run(task, arm)
    assert result is None and not task.plan_success and calls == ["refresh"]


def test_carry_start_refreshes_fail_closed_and_checks_measured_fingers(monkeypatch):
    arm, _, full, fingers, target, saved, observed = path_fixture()
    arm.get_ee_pose = lambda: target
    arm.tcp_offset = Pose()
    calls = []
    monkeypatch.setattr(oil, "_require_mode", lambda *a: calls.append("mode"))
    monkeypatch.setattr(oil, "refresh_obstacles", lambda *a: calls.append("refresh"))
    monkeypatch.setattr(oil, "joint_state", lambda *a: (full, fingers))
    assert oil.checked_oil_carry_start(NS(), arm) == pytest.approx(full)
    assert calls == ["mode", "refresh"] and observed[0] == pytest.approx(full)
    assert arm.planner._mplib.robot.get_qpos() == pytest.approx(saved)
    def fail(*args):
        raise RuntimeError("cloud update failed")
    monkeypatch.setattr(oil, "refresh_obstacles", fail)
    with pytest.raises(RuntimeError, match="cloud update failed"):
        oil.checked_oil_carry_start(NS(), arm)


def test_carry_endpoint_fk_retains_shuffled_native_order_and_measured_fingers(monkeypatch):
    task, arm, plan, calls = planning_fixture(Pose([.2, 0., .3]))
    task.config = {"oil_clearance_lift": True}
    native = arm.planner._mplib
    native.move_group_joint_indices = np.arange(1, 8)
    full = np.array([.02, *([0.]*7), .03])
    plan.position[-1] = .01
    observed = []
    native.pinocchio_model.compute_forward_kinematics = lambda q: observed.append(q.copy())
    native.pad_move_group_qpos = lambda q: pytest.fail("candidate must not assume arm-first ordering")
    monkeypatch.setattr(oil, "checked_oil_carry_start", lambda *a: full.copy())
    assert run(task, arm)[0] is plan
    assert observed[0] == pytest.approx([.02, *([.01]*7), .03])


@pytest.mark.parametrize("flag", [False, "false", "true", 1, None])
def test_clearance_candidate_does_not_enable_from_truthy_nonliteral_flags(flag):
    task, arm, plan, _ = planning_fixture(Pose([.2, 0., .3]))
    task.config = {"oil_clearance_lift": flag}
    assert run(task, arm)[0] is plan


@pytest.mark.parametrize("supplied, expected", [
    (None, True), ({}, True), ({"oil_clearance_lift": True}, True),
    ({"oil_clearance_lift": False}, False), ({"oil_clearance_lift": None}, None),
    ({"oil_clearance_lift": "true"}, "true"),
    ({"oil_clearance_lift": "false"}, "false"), ({"oil_clearance_lift": 1}, 1),
])
def test_constructor_enables_clearance_by_default_and_preserves_explicit_overrides(
        monkeypatch, supplied, expected):
    original = None if supplied is None else supplied.copy()
    received = []
    monkeypatch.setattr(BaseTask, "__init__", lambda self, cfg: received.append(cfg))
    resolved = []
    monkeypatch.setattr(OilBottleRecovery, "resolve_target_object", lambda self: resolved.append(self))
    task = OilBottleRecovery(supplied)
    assert len(received) == 1 and resolved == [task]
    assert received[0]["oil_clearance_lift"] is expected
    assert supplied == original
    if supplied is not None:
        assert received[0] is not supplied


def test_clearance_methods_keep_literal_opt_out_and_settle_experiment_stays_removed():
    source = (Path(__file__).parents[1] / "envs/tasks/oil_bottle_recovery.py").read_text()
    assert 'cfg.setdefault("oil_clearance_lift", True)' in source
    assert 'self.config.get("oil_clearance_lift", False)' in source
    assert "oil_measured_lift_settle" not in source
