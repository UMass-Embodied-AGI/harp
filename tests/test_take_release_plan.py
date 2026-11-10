"""Lightweight native-API stubs; no Genesis scene, planning or physics."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "take_release_plan_under_test",
    Path(__file__).resolve().parents[1] / "envs/task_bases/take_release_plan.py",
)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


class Pose:
    def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
        self.p, self.q = np.asarray(p), np.asarray(q)

    def inv(self):
        return Pose(-self.p, self.q)

    def __mul__(self, other):
        return Pose(self.p + other.p, other.q)


class Object:
    def __init__(self, name, pose=None, shapes=None, shape_poses=None):
        self.name, self.pose = name, pose or Pose()
        self.shapes = shapes if shapes is not None else [object()]


class Backend:
    pass


@pytest.fixture
def case(monkeypatch):
    monkeypatch.delenv("DEBUG_BOUNDED_JOINT_PATHS", raising=False)
    native = ModuleType("mplib")
    native.Planner = Backend
    pymp = ModuleType("mplib.pymp")
    pymp.Pose = Pose
    collision = ModuleType("mplib.pymp.collision_detection")
    fcl = ModuleType("mplib.pymp.collision_detection.fcl")
    fcl.FCLObject = Object
    fcl.Box = lambda *size: size
    fcl.CollisionObject = lambda shape, pose: SimpleNamespace(shape=shape, pose=pose)
    for module in (native, pymp, collision, fcl):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    state = SimpleNamespace(world_checks=[], distance_checks=[], collision=lambda q: False,
                            gap=lambda name, fixture, q: .02, fk_pose=None, chain=[])
    objects = [Object(name) for name in sorted(validator._COLLISION_LINKS)]
    robot = SimpleNamespace(qpos=np.r_[np.full(7, .73), .01, .01])
    state.saved = robot.qpos.copy()
    robot.get_qpos = lambda: robot.qpos.copy()
    robot.get_name = lambda: "panda"
    names = validator._ARM_NAMES + validator._FINGER_NAMES
    robot.get_user_joint_names = lambda: names
    robot.get_base_pose = lambda: Pose()

    def set_qpos(qpos, full):
        assert full is True
        robot.qpos = np.asarray(qpos).copy()

    robot.set_qpos = set_qpos
    robot.get_fcl_model = lambda: SimpleNamespace(
        get_collision_objects=lambda: objects,
        get_collision_link_names=lambda: [obj.name for obj in objects],
    )
    pin = SimpleNamespace(qpos=state.saved.copy())
    pin.compute_forward_kinematics = lambda q: setattr(pin, "qpos", np.asarray(q).copy())
    pin.get_chain_joint_name = lambda name: state.chain
    pin.get_link_pose = lambda index: state.fk_pose or Pose([pin.qpos[0], 0., 0.])
    world = SimpleNamespace(get_planned_articulations=lambda: [robot])

    def world_set(qpos):
        robot.qpos[:7] = qpos

    def colliding():
        state.world_checks.append(robot.qpos.copy())
        return state.collision(robot.qpos.copy())

    def distance(obj, fixture):
        state.distance_checks.append((obj.name, fixture.name, robot.qpos.copy()))
        return SimpleNamespace(min_distance=state.gap(obj.name, fixture.name, robot.qpos.copy()))

    world.set_qpos_all = world_set
    world.is_state_colliding = colliding
    fcl.distance = distance
    backend = Backend()
    backend.robot, backend.pinocchio_model, backend.planning_world = robot, pin, world
    backend.joint_limits = np.array([[-4., 4.]] * 7 + [[0., .04]] * 2)
    backend.move_group_joint_indices = np.arange(7)
    backend.move_group = "panda_hand"
    backend.link_name_2_idx = {"panda_hand": 8}
    indices = np.array([3, 1, 8, 5, 0, 10, 6, 11, 4])
    sim = np.zeros(12)
    sim[indices[-2:]] = [.019, .020]
    cached = sim.copy()
    cached[indices[-2:]] = .014
    planner = SimpleNamespace(_mplib=backend, n_arm=7, _to_mplib_pose=lambda q: Pose(q[:3], q[3:]),
                              _arm_joint_index={name.removeprefix("panda_"): int(index)
                                                for name, index in zip(names, indices)})
    entity = SimpleNamespace(get_qpos=lambda: sim.copy())  # Deliberately no simulator setters.
    arm = SimpleNamespace(planner=planner, n_arm=7, entity=entity,
                          get_arm_qpos=lambda: sim[indices[:7]].copy(),
                          _cached_target=cached, gripper_val=.35, gripper_open=.04, gripper_closed=0.)
    plan = SimpleNamespace(success=True, position=np.zeros((3, 7)), velocity=np.zeros((3, 7)))
    plan.position[:, 0] = [0., .05, .10]
    target = np.array([.1, 0., 0., 1., 0., 0., 0.])
    bounds = np.array([[-.5, -.5, .70], [.5, .5, .765]])
    state.arm, state.backend, state.plan = arm, backend, plan
    state.target, state.bounds, state.objects = target, bounds, objects
    state.sim, state.cached, state.indices = sim, cached, indices
    state.run = lambda: validator.validate_release_plan(
        arm, plan, target, table_bounds=bounds, plate_bounds=bounds + [0., 0., .03])
    return state


def assert_restored(case):
    np.testing.assert_array_equal(case.backend.robot.qpos, case.saved)
    np.testing.assert_array_equal(case.backend.pinocchio_model.qpos, case.saved)


def test_valid_path_maps_measured_commanded_fingers_and_restores(case):
    sim, cached = case.sim.copy(), case.cached.copy()
    report = case.run()
    json.dumps(report, allow_nan=False)
    assert report["continuous_collision_checked"] is False
    assert report["table_exempt_link"] == "panda_link0"
    assert report["min_fixture_distance_m"] == {"table": .02, "plate": .02}
    assert report["opening_checked_states"] > 0
    checks = case.world_checks
    assert any(np.allclose(q[7:], [.019, .02]) for q in checks)
    assert any(np.allclose(q[7:], [.014, .014]) for q in checks)
    assert any(np.allclose(q[7:], [.04, .04]) for q in checks)
    assert all(not (name == "panda_link0" and fixture.endswith("table"))
               for name, fixture, _ in case.distance_checks)
    assert any(name == "panda_link0" and fixture.endswith("plate")
               for name, fixture, _ in case.distance_checks)
    np.testing.assert_array_equal(case.sim, sim)
    np.testing.assert_array_equal(case.cached, cached)
    assert_restored(case)


def test_endpoint_diagnostics_keep_world_frame_and_signed_bias(case):
    base = np.array([.3, -.2, .75])
    bias = np.array([.001, -.002, -.003])
    case.backend.robot.get_base_pose = lambda: Pose(base)
    case.target[:3] += base
    case.fk_pose = Pose(np.array([.1, 0., 0.]) + bias)
    report = case.run()
    np.testing.assert_allclose(report["endpoint_world_link_pose7"][:3],
                               case.target[:3] + bias, atol=1e-12)
    np.testing.assert_allclose(report["endpoint_world_link_error_xyz_m"], bias, atol=1e-12)
    assert report["endpoint_position_error_m"] == pytest.approx(np.linalg.norm(bias))
    assert_restored(case)


@pytest.mark.parametrize("field", ["position", "velocity", "target", "sim", "cached"])
def test_nonfinite_inputs_rejected(case, field):
    value = getattr(case.plan, field) if hasattr(case.plan, field) else getattr(case, field)
    value.flat[-1] = np.nan
    with pytest.raises(ValueError):
        case.run()
    assert not case.world_checks
    assert_restored(case)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.001, .001999])
def test_fixture_gap_rejection_restores_even_on_first_sample(case, value):
    case.gap = lambda name, fixture, q: value
    with pytest.raises(ValueError, match="clearance"):
        case.run()
    assert len(case.world_checks) == 1
    assert_restored(case)


@pytest.mark.parametrize("point", [0., .05, .10])
def test_world_collision_checked_at_initial_intermediate_and_final(case, point):
    case.collision = lambda q: np.isclose(q[0], point)
    with pytest.raises(ValueError, match="world collision"):
        case.run()
    assert_restored(case)


def test_collision_at_intermediate_opening_rejected(case):
    case.collision = lambda q: .029 < q[7] < .032
    with pytest.raises(ValueError, match="opening_measured"):
        case.run()
    assert_restored(case)


def test_commanded_fingers_are_not_replaced_with_zero_padding(case):
    case.collision = lambda q: q[7] < .015
    with pytest.raises(ValueError, match="cached_command"):
        case.run()
    assert_restored(case)


@pytest.mark.parametrize("pose", [Pose([np.nan, 0., 0.]), Pose([.12, 0., 0.]),
                                   Pose([.1, 0., 0.], [np.cos(.1), np.sin(.1), 0., 0.]),
                                   Pose([.1, 0., 0.], [2., 0., 0., 0.])])
def test_bad_fk_rejected_and_restored(case, pose):
    case.fk_pose = pose
    with pytest.raises(ValueError, match="FK"):
        case.run()
    assert_restored(case)


@pytest.mark.parametrize("path", [[0., .31], [0., .25, 0., .25, 0., .25, 0., .25, 0., .25],
                                   list(np.linspace(0., 1.8, 10))])
def test_adjacent_cumulative_and_excursion_bounds(case, path):
    case.plan.position = np.zeros((len(path), 7))
    case.plan.position[:, 0] = path
    case.plan.velocity = np.zeros_like(case.plan.position)
    with pytest.raises(ValueError, match="branch"):
        case.run()
    assert not case.world_checks


def test_current_start_mismatch_rejected(case):
    case.plan.position[0, 0] = .001
    with pytest.raises(ValueError, match="measured arm"):
        case.run()


@pytest.mark.parametrize("which", ["arm", "finger"])
def test_full_limits_checked(case, which):
    case.backend.joint_limits[0 if which == "arm" else 7, 1] = .01
    with pytest.raises(ValueError, match="joint limit"):
        case.run()
    assert_restored(case)


def test_fixed_base_exception_requires_empty_chain(case):
    case.chain = ["panda_joint1"]
    with pytest.raises(ValueError, match="fixed root"):
        case.run()
    assert not case.world_checks


def test_no_moving_link_exemption(case):
    case.gap = lambda name, fixture, q: 0. if name == "panda_link1" else .02
    with pytest.raises(ValueError, match="panda_link1"):
        case.run()
    assert_restored(case)


def test_missing_collision_link_fails_closed(case):
    case.objects.pop()
    with pytest.raises(ValueError, match="collision geometry"):
        case.run()


def test_unavailable_distance_fails_closed_and_restores(case):
    def unavailable(*args):
        raise RuntimeError("FCL unavailable")
    case.gap = unavailable
    with pytest.raises(ValueError, match="FCL unavailable"):
        case.run()
    assert_restored(case)


def test_world_must_preserve_measured_fingers(case):
    original = case.backend.planning_world.set_qpos_all

    def overwrites_fingers(q):
        original(q)
        case.backend.robot.qpos[7:] = 0.

    case.backend.planning_world.set_qpos_all = overwrites_fingers
    with pytest.raises(ValueError, match="finger configuration"):
        case.run()
    assert not case.world_checks
    assert_restored(case)


def test_debug_backend_forbidden(case, monkeypatch):
    monkeypatch.setenv("DEBUG_BOUNDED_JOINT_PATHS", "1")
    with pytest.raises(ValueError, match="native"):
        case.run()


@pytest.mark.parametrize("mutation", ["shape", "mapping", "bounds", "success", "aperture"])
def test_malformed_contracts_fail_closed(case, mutation):
    if mutation == "shape":
        case.plan.position = np.zeros((2, 8))
    elif mutation == "mapping":
        case.arm.planner._arm_joint_index["finger_joint2"] = 100
    elif mutation == "bounds":
        case.bounds[1] = case.bounds[0]
    elif mutation == "success":
        case.plan.success = False
    else:
        case.arm.gripper_open = float("nan")
    with pytest.raises(ValueError):
        case.run()
    assert not case.world_checks


def test_native_mplib_fixture_opening_and_restoration_without_genesis_scene():
    """Exercise the installed native FCL API, not just the test doubles above."""
    import mplib
    from mplib.pymp import Pose as NativePose

    assets = Path(__file__).resolve().parents[1] / "assets/embodiments/franka-panda"
    backend = mplib.Planner(urdf=assets / "panda.urdf", srdf=assets / "panda.srdf",
                            move_group="panda_hand", use_convex=False)
    backend.set_base_pose(NativePose(p=[.4, -.2, .75], q=[np.sqrt(.5), 0., 0., np.sqrt(.5)]))
    names = list(backend.robot.get_user_joint_names())
    current = np.array([0., -.785, 0., -2.356, 0., 1.571, .785])
    sim = np.zeros(9)
    for name, value in zip(validator._ARM_NAMES, current):
        sim[names.index(name)] = value
    sim[[names.index(name) for name in validator._FINGER_NAMES]] = [.019, .020]
    cached = sim.copy()
    cached[[names.index(name) for name in validator._FINGER_NAMES]] = .014
    saved = sim.copy()
    saved[names.index("panda_joint1")] += .05
    backend.robot.set_qpos(saved, True)
    backend.pinocchio_model.compute_forward_kinematics(saved)
    fcl_model = backend.robot.get_fcl_model()
    hand_before = next(obj for obj in fcl_model.get_collision_objects()
                       if obj.name == "panda_hand").pose.p.copy()
    position = np.stack([current, current + [.01, 0., 0., 0., 0., 0., 0.]])
    endpoint = sim.copy()
    endpoint[np.asarray(backend.move_group_joint_indices)] = position[-1]
    backend.robot.set_qpos(endpoint, True)
    backend.planning_world.set_qpos_all(endpoint[np.asarray(backend.move_group_joint_indices)])
    np.testing.assert_array_equal(backend.robot.get_qpos(), endpoint)
    hand_after = next(obj for obj in fcl_model.get_collision_objects()
                      if obj.name == "panda_hand").pose.p.copy()
    assert np.linalg.norm(hand_after - hand_before) > 1e-5
    backend.pinocchio_model.compute_forward_kinematics(endpoint)
    goal = backend.robot.get_base_pose() * backend.pinocchio_model.get_link_pose(
        backend.link_name_2_idx["panda_hand"])
    target = np.r_[goal.p, goal.q]
    backend.robot.set_qpos(saved, True)
    backend.planning_world.set_qpos_all(saved[np.asarray(backend.move_group_joint_indices)])
    backend.pinocchio_model.compute_forward_kinematics(saved)
    wrapper = SimpleNamespace(
        _mplib=backend, n_arm=7,
        _arm_joint_index={name.removeprefix("panda_"): i for i, name in enumerate(names)},
        _to_mplib_pose=lambda q: NativePose(p=q[:3], q=q[3:]),
    )
    arm = SimpleNamespace(n_arm=7, planner=wrapper, get_arm_qpos=lambda: current.copy(),
                          entity=SimpleNamespace(get_qpos=lambda: sim.copy()),
                          _cached_target=cached, gripper_val=.35, gripper_open=.04, gripper_closed=0.)
    plan = SimpleNamespace(success=True, position=position, velocity=np.zeros_like(position))
    report = validator.validate_release_plan(
        arm, plan, target,
        table_bounds=np.array([[-1., -1., -3.], [1., 1., -2.9]]),
        plate_bounds=np.array([[-.1, -.1, -2.5], [.1, .1, -2.4]]),
    )
    assert report["opening_checked_states"] >= 10
    assert report["finger_states"] == ["measured", "cached_command", "gripper_command"]
    assert report["endpoint_position_error_m"] < 1e-6
    np.testing.assert_allclose(report["endpoint_world_link_pose7"], target, atol=1e-12)
    np.testing.assert_allclose(report["endpoint_world_link_error_xyz_m"], 0., atol=1e-12)
    assert all(value > .002 for value in report["min_fixture_distance_m"].values())
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(backend.robot.get_qpos(), saved)
