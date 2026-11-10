"""Pure numeric/native-API stubs for oil's opt-in geometry planning context."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1]/"envs/task_bases/oil_recovery_plan.py"
_SPEC = importlib.util.spec_from_file_location("oil_recovery_plan_tested", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)


class Pose:
    def __init__(self, matrix=None, *, p=None, q=None):
        value = (helper.pose7(matrix) if matrix is not None else
                 np.r_[np.zeros(3) if p is None else p, [1., 0., 0., 0.] if q is None else q])
        self.p, self.q = value[:3].copy(), value[3:].copy()


class Object:
    def __init__(self, name, pose=None, shapes=None, shape_poses=None):
        self.name, self.pose = name, pose or Pose()
        self.shapes = [object()] if shapes is None else shapes


class Backend:
    pass


@pytest.fixture
def case(monkeypatch):
    for name in ("DEBUG_TELEPORT", "USE_SETQPOS", "DEBUG_BOUNDED_JOINT_PATHS"):
        monkeypatch.delenv(name, raising=False)
    native, pymp = ModuleType("mplib"), ModuleType("mplib.pymp")
    collision, fcl = ModuleType("mplib.pymp.collision_detection"), ModuleType("mplib.pymp.collision_detection.fcl")
    native.Planner, pymp.Pose = Backend, Pose
    fcl.Box, fcl.CollisionObject, fcl.FCLObject = lambda *size: size, lambda *args: args, Object
    for module in (native, pymp, collision, fcl): monkeypatch.setitem(sys.modules, module.__name__, module)
    state = SimpleNamespace(collision=lambda q: False, gap=lambda a, b, q: .2, checks=[], gaps=[], chain=[],
                            acm="original ACM", base=Pose(), fk=None, names=helper._ARM+helper._FINGERS,
                            articulation_names=["panda"])
    objects = [Object(name) for name in sorted(helper._LINKS)]
    state.objects = objects
    robot = SimpleNamespace(qpos=np.r_[np.repeat(.7, 7), .01, .02])
    state.saved = robot.qpos.copy()
    robot.set_qpos = lambda q, full: setattr(robot, "qpos", np.asarray(q).copy())
    robot.get_qpos = lambda: robot.qpos.copy()
    robot.get_user_joint_names = lambda: state.names
    robot.get_base_pose = lambda: state.base
    robot.get_name = lambda: "panda"
    robot.get_fcl_model = lambda: SimpleNamespace(get_collision_objects=lambda: state.objects,
        get_collision_link_names=lambda: [obj.name for obj in state.objects])
    world_objects = {"scene_pcd": Object("scene_pcd"), "unrelated_wall": Object("unrelated_wall")}
    state.world_objects, state.original_objects = world_objects, world_objects.copy()
    def add(obj): world_objects[obj.name] = obj
    def remove(name):
        world_objects.pop(name)
        return True
    def world_set(q): robot.qpos[:7] = q
    def colliding():
        state.checks.append(robot.qpos.copy())
        return [SimpleNamespace(object_name1="panda", object_name2="scene_pcd", link_name1="panda_hand",
                                link_name2="")] if state.collision(robot.qpos) else []
    def distance(first, second):
        state.gaps.append((first.name, second.name, robot.qpos.copy()))
        return SimpleNamespace(min_distance=state.gap(first.name, second.name, robot.qpos.copy()))
    fcl.distance = distance
    world = SimpleNamespace(get_object_names=lambda: list(world_objects), get_object=lambda name: world_objects[name],
        add_object=add, remove_object=remove, get_allowed_collision_matrix=lambda: state.acm,
        get_planned_articulations=lambda: [robot], get_articulation_names=lambda: state.articulation_names,
        set_qpos_all=world_set, check_collision=colliding)
    pin = SimpleNamespace(qpos=state.saved.copy())
    pin.compute_forward_kinematics = lambda q: setattr(pin, "qpos", np.asarray(q).copy())
    pin.get_link_pose = lambda index: state.fk or Pose(p=[pin.qpos[0], 0., .9])
    pin.get_chain_joint_name = lambda name: state.chain
    backend = Backend()
    backend.robot, backend.planning_world, backend.pinocchio_model = robot, world, pin
    backend.move_group, backend.move_group_joint_indices = "panda_hand", np.arange(7)
    backend.link_name_2_idx = {"panda_hand": 9}
    backend.joint_limits = np.array([[-4., 4.]]*7+[[0., .04]]*2)
    measured = np.r_[np.zeros(7), .019, .025]
    state.kwargs = dict(measured_qpos=measured, finger_targets={"cached": [.012, .014], "scalar": [.016, .016]},
        gripper_open=.04, world_base_pose7=[0., 0., 0., 1., 0., 0., 0.],
        hand_tcp_pose7=[0., 0., .1034, 1., 0., 0., 0.], tcp_payload_pose7=[.02, -.01, .03, 1., 0., 0., 0.],
        payload_bounds_local=[[-.03, 0., -.02], [.025, .137, .02]],
        fixture_aabbs={name: [[-.2, -.2, .7], [.2, .2, .8]] for name in
                      ("table", "cabinet", "cooktop", "vinegar", "soy_sauce", "actual_cooktop_body")})
    state.backend = backend
    state.new = lambda **extra: helper.OilRecoveryPlan(backend, **dict(state.kwargs, **extra))
    return state


def restored(case):
    np.testing.assert_array_equal(case.backend.robot.get_qpos(), case.saved)
    np.testing.assert_array_equal(case.backend.pinocchio_model.qpos, case.saved)
    assert case.world_objects == case.original_objects and case.acm == "original ACM"


def test_measured_fullstate_commands_opening_and_restore(case):
    with case.new() as context:
        state = context.check_state(opening=True)
        np.testing.assert_allclose(state["bottle_pose7"][:3], [.02, -.01, 1.0334])
        assert context.diagnostics()["continuous_collision_checked"] is False
    for q in ([.019, .025], [.012, .014], [.016, .016], [.04, .04]):
        assert any(np.allclose(checked[7:], q) for checked in case.checks)
    assert all(not (a == "panda_link0" and b == "oil_recovery_table") for a, b, _ in case.gaps)
    assert any(a == "panda_link0" and b == "oil_recovery_cabinet" for a, b, _ in case.gaps)
    assert any(a == "oil_recovery_held_payload" and b == "scene_pcd" for a, b, _ in case.gaps)
    assert any(a == "oil_recovery_held_payload" and b == "unrelated_wall" for a, b, _ in case.gaps)
    assert not any(a == "oil_recovery_held_payload" and b in helper._TOUCH for a, b, _ in case.gaps)
    restored(case)


@pytest.mark.parametrize("obstacle", ["actual_cooktop_body", "vinegar", "table"])
def test_extra_actual_stove_and_every_fixture_protect_payload(case, obstacle):
    case.gap = lambda a, b, q: -.01 if a == "oil_recovery_held_payload" and b == "oil_recovery_"+obstacle else .2
    with pytest.raises(ValueError, match="payload/"+obstacle):
        with case.new() as context: context.check_state()
    restored(case)


def test_only_fixed_root_exempt_from_table(case):
    case.gap = lambda a, b, q: -.01 if a == "panda_link1" and b == "oil_recovery_table" else .2
    with pytest.raises(ValueError, match="panda_link1/table"):
        with case.new() as context: context.check_state()
    restored(case)


def test_commanded_and_intermediate_opening_collisions_rejected(case):
    case.collision = lambda q: .031 < q[7] < .034
    with pytest.raises(ValueError, match="opening_measured"):
        with case.new() as context: context.check_state(opening=True)
    restored(case)


def test_finger_command_collision_rejected(case):
    case.collision = lambda q: q[7] < .015
    with pytest.raises(ValueError, match="cached"):
        with case.new() as context: context.check_state()
    restored(case)


def test_initial_collision_never_calls_ik(case):
    case.collision = lambda q: True
    case.backend.IK = lambda *args, **kwargs: pytest.fail("IK after invalid start")
    with pytest.raises(ValueError, match="world/self"):
        with case.new() as context: context.select_branch([.36, .08], .771)
    restored(case)


@pytest.mark.parametrize("names", [["panda", "unplanned_obstacle"], [], ["other_robot"]])
def test_unplanned_or_mismatched_articulation_rejected_before_mutation(case, names):
    case.articulation_names = names
    with pytest.raises(ValueError, match="unmodeled articulated obstacle"):
        case.new()
    assert not case.checks and not case.gaps
    restored(case)


def test_base_tcp_and_held_transform_composed_once(case):
    base = [.4, -.2, .1, np.sqrt(.5), 0., 0., np.sqrt(.5)]
    case.base = Pose(p=base[:3], q=base[3:])
    case.kwargs["world_base_pose7"] = base
    with case.new() as context:
        state = context.check_state()
        np.testing.assert_allclose(state["bottle_pose7"][:3], [.41, -.18, 1.1334], atol=1e-12)
        bottle = helper.pose_matrix(state["bottle_pose7"])
        np.testing.assert_allclose(context.link_target(bottle), helper.pose_matrix(state["link_pose7"]), atol=1e-12)
    restored(case)


def test_bounds_center_kept(case):
    with case.new() as context:
        box = context._box("offset", np.array([[1., 2., 3.], [3., 4., 5.]]))
        np.testing.assert_allclose(box.pose.p, [2., 3., 4.])


@pytest.mark.parametrize("change", ["nan", "commands", "fixture", "cloud", "mapping", "links", "root", "base"])
def test_missing_geometry_or_inconsistent_inputs_fail_closed(case, change):
    if change == "nan": case.kwargs["measured_qpos"][0] = np.nan
    if change == "commands": case.kwargs["finger_targets"].pop("scalar")
    if change == "fixture": case.kwargs["fixture_aabbs"].pop("vinegar")
    if change == "cloud": case.world_objects.pop("scene_pcd")
    if change == "mapping": case.backend.move_group_joint_indices = np.arange(1, 8)
    if change == "links": case.objects.pop()
    if change == "root": case.chain = ["panda_joint1"]
    if change == "base": case.kwargs["world_base_pose7"][0] = 1.
    with pytest.raises(ValueError): case.new()
    np.testing.assert_array_equal(case.backend.robot.get_qpos(), case.saved)


@pytest.mark.parametrize("flag", ["DEBUG_TELEPORT", "USE_SETQPOS", "DEBUG_BOUNDED_JOINT_PATHS"])
def test_debug_bypasses_forbidden(case, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="bypass"): case.new()


def test_name_conflict_does_not_remove_existing_object(case):
    case.world_objects["oil_recovery_vinegar"] = Object("oil_recovery_vinegar")
    original = case.world_objects.copy()
    with pytest.raises(ValueError, match="name conflict"):
        with case.new(): pass
    assert case.world_objects == original
    np.testing.assert_array_equal(case.backend.robot.get_qpos(), case.saved)


def test_native_exception_restores_context(case):
    case.gap = lambda *args: (_ for _ in ()).throw(RuntimeError("native FCL unavailable"))
    with pytest.raises(RuntimeError, match="native FCL"):
        with case.new() as context: context.check_state()
    restored(case)


def test_released_payload_not_carried_but_stays_protected(case):
    with pytest.raises(ValueError, match="released_bottle"): case.new(held=False)
    case.kwargs["fixture_aabbs"]["released_bottle"] = [[.3, .05, .765], [.4, .15, .91]]
    with case.new(held=False) as context: context.check_state()
    assert not any(a == "oil_recovery_held_payload" for a, _, _ in case.gaps)
    assert any(b == "oil_recovery_released_bottle" for _, b, _ in case.gaps)
    restored(case)


def sample_plan(end=.1):
    return dict(status="Success", position=np.array([np.zeros(7), np.repeat(end, 7)]), velocity=np.zeros((2, 7)))


def test_full_path_densifies_and_catches_midpoint_collision(case):
    case.collision = lambda q: np.isclose(q[0], .05)
    with pytest.raises(ValueError, match="world/self"):
        with case.new() as context: context.check_path(sample_plan(), np.zeros(7))
    restored(case)


@pytest.mark.parametrize("change", ["start", "goal", "jump", "velocity", "nan"])
def test_path_contracts(change):
    plan = sample_plan()
    goal = None
    if change == "start": plan["position"][0, 0] = .01
    if change == "goal": goal = np.repeat(.11, 7)
    if change == "jump": plan["position"][-1, 0] = .11
    if change == "velocity": plan["velocity"] = np.zeros((2, 6))
    if change == "nan": plan["position"][-1, 0] = np.nan
    with pytest.raises(ValueError): helper.dense_path(plan, np.zeros(7), goal)


def test_existing_planresult_accepted():
    native = sample_plan()
    result = SimpleNamespace(success=True, position=native["position"], velocity=native["velocity"])
    samples, report = helper.dense_path(result, np.zeros(7))
    assert len(samples) == 11 and report["continuous_collision_checked"] is False


def test_bad_fk_target_rejected(case):
    with pytest.raises(ValueError, match="FK target error"):
        with case.new() as context: context.check_state(target_link=[0., 0., .91, 1., 0., 0., 0.])
    restored(case)


def test_upright_yaw_and_height():
    target = helper.upright_target([.36, .08], .771)
    np.testing.assert_allclose(target[:3, :3] @ [0., 1., 0.], [0., 0., 1.])
    np.testing.assert_allclose(target[:3, 3], [.36, .08, .771])
    np.testing.assert_allclose(target[:3, :3] @ [1., 0., 0.], [-1., 0., 0.])


def test_branch_selection_checks_opening_and_whole_corridor(case, monkeypatch):
    checked, ik_calls = [], []
    with case.new() as context:
        def ik(target, start, trials, closest):
            ik_calls.append((trials, closest))
            if not closest:
                return [context.full_qpos(np.repeat(.1, 7)), context.full_qpos(np.repeat(.2, 7))]
            return context.full_qpos(np.asarray(start)+.005)
        def check(arm=None, **kwargs):
            checked.append((arm, kwargs))
            if arm is not None and np.allclose(arm, .1): raise ValueError("first endpoint blocked")
            return {}
        monkeypatch.setattr(context, "_ik", ik)
        monkeypatch.setattr(context, "check_state", check)
        branch = context.select_branch([.36, .08], .771)
    assert ik_calls[0] == (32, False) and ik_calls[1:] == [(1, True)]*13
    assert len(branch["rejected_branches"]) == 1
    np.testing.assert_allclose(branch["low_qpos"], .2)
    np.testing.assert_allclose(branch["hover_qpos"], .265)
    assert checked[0][0] is None and sum(row[1].get("opening", False) for row in checked) == 2
    assert branch["hover_pose7"][2]-branch["low_pose7"][2] == pytest.approx(.13)
    restored(case)


def test_context_cannot_be_reused_after_arrival(case):
    context = case.new()
    with context: context.check_state()
    with pytest.raises(ValueError, match="fresh context"):
        with context: pass
    restored(case)


def test_hover_planning_preserves_exact_branch_and_measured_start(case, monkeypatch):
    calls = []
    case.backend.plan_qpos = lambda **kwargs: calls.append(kwargs) or sample_plan()
    with case.new() as context:
        low, hover, low_q, hover_q = np.eye(4), np.eye(4), np.zeros(7), np.repeat(.1, 7)
        monkeypatch.setattr(context, "_branch", lambda branch: (low, hover, low_q, hover_q))
        monkeypatch.setattr(context, "check_state", lambda *args, **kwargs: {})
        monkeypatch.setattr(context, "check_path", lambda *args, **kwargs: {})
        context.plan_to_hover({})
    assert len(calls) == 1 and calls[0]["fix_joint_limits"] is False
    np.testing.assert_array_equal(calls[0]["current_qpos"], case.kwargs["measured_qpos"])
    np.testing.assert_array_equal(calls[0]["goal_qposes"][0][:7], hover_q)
    np.testing.assert_array_equal(calls[0]["goal_qposes"][0][7:], [.019, .025])
    restored(case)


def test_descent_uses_fresh_measured_arrival_not_nominal_hover(case, monkeypatch):
    measured = case.kwargs["measured_qpos"].copy()
    measured[:7] = .02
    low_q = np.repeat(.04, 7)
    calls = []
    result = dict(status="Success", position=np.stack([measured[:7], low_q]), velocity=np.zeros((2, 7)))
    case.backend.plan_screw = lambda **kwargs: calls.append(kwargs) or result
    with case.new(measured_qpos=measured) as context:
        monkeypatch.setattr(context, "_branch", lambda branch: (np.eye(4), np.eye(4), low_q, np.repeat(.03, 7)))
        monkeypatch.setattr(context, "check_state", lambda *args, **kwargs: {})
        monkeypatch.setattr(context, "check_path", lambda *args, **kwargs: {})
        context.plan_descent({})
    np.testing.assert_array_equal(calls[0]["current_qpos"], measured)
    assert calls[0]["qpos_step"] == .01 and calls[0]["wrt_world"] is True
    restored(case)


def test_retreat_native_screw_requires_empty_context(case, monkeypatch):
    with case.new() as context:
        with pytest.raises(ValueError, match="held=False"): context.plan_retreat(np.eye(4))
    case.kwargs["fixture_aabbs"]["released_bottle"] = [[.3, .05, .765], [.4, .15, .91]]
    calls = []
    case.backend.plan_screw = lambda **kwargs: calls.append(kwargs) or sample_plan()
    with case.new(held=False) as context:
        monkeypatch.setattr(context, "check_state", lambda *args, **kwargs: {})
        monkeypatch.setattr(context, "check_path", lambda *args, **kwargs: {})
        context.plan_retreat(np.eye(4))
    assert len(calls) == 1 and calls[0]["qpos_step"] == .01
    restored(case)


def test_no_live_second_planner_rng_attachment_or_simulator_calls():
    tree = ast.parse(_PATH.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attrs = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert not attrs.intersection({"Planner", "set_global_seed", "attach_object", "attach_box", "step_sim",
                                  "inverse_kinematics", "remove_point_cloud", "update_point_cloud", "set_entry"})
    assert {"IK", "plan_qpos", "plan_screw"} <= attrs
