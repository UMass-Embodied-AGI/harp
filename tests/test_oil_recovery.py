"""Runtime contracts; native geometry and physical pilots are separate gates."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases import oil_recovery as oil


def test_active_frying_clip_honors_runtime_rate_changes_without_restarting():
    frames = []
    motion = NS(at_frame=27)
    def advance(skip_avatar_animation=False):
        assert not skip_avatar_animation
        motion.at_frame += 1
        frames.append(motion.at_frame)
    motion.step = advance
    avatar = NS(frame_ratio=4)
    clock = oil.OilFryingMotionClock(motion, avatar, 4)
    for _ in range(3):
        clock.step()
    assert frames == [28, 29, 30]
    avatar.frame_ratio = 16
    for _ in range(12):
        clock.step()
    assert frames == [28, 29, 30, 31, 32, 33]
    avatar.frame_ratio = 4
    for _ in range(3):
        clock.step()
    assert frames == list(range(28, 37))
    assert clock.at_frame == 36


@pytest.mark.parametrize("ratio", [0, -4, np.inf, np.nan])
def test_frying_clock_rejects_rates_that_would_freeze_or_reverse_motion(ratio):
    motion = NS(step=lambda: pytest.fail("invalid rate advanced clip"))
    clock = oil.OilFryingMotionClock(motion, NS(frame_ratio=ratio), 4)
    with pytest.raises(ValueError, match="finite, nonaccelerated"):
        clock.step()


def entity(low=(0., 0., 1.), high=(.1, .1, 1.1)):
    return NS(get_AABB=lambda: np.array([low, high]), get_pos=lambda: np.array(low),
              get_quat=lambda: np.array([1., 0., 0., 0.]))


@pytest.fixture
def task():
    return NS(table=entity(), _COOKTOP_X=(-.78, -.32), _COOKTOP_Y=(.05, .51),
              _COOKTOP_Z=(.74, .80), kitchen_cooktop={"body": entity(), "burners": [entity()]},
              kitchen_backdrop={"wall": entity(), "cabinet": {"top": entity()}, "floor": object()},
              kitchen_items={"vinegar": entity(), "soy_sauce": entity()}, oil_entity=entity(),
              plan_success=True, _avatar_retreat_enabled=lambda: False,
              avatar=NS(frame_ratio=16), TABLE_TOP_Z=.765)


def test_actual_cooktop_and_all_backdrop_preserved(task):
    task.kitchen_items["other_clutter"] = entity()
    result = oil.fixture_bounds(task)
    assert {"cooktop", "actual_cooktop_body", "actual_cooktop_burners_0",
            "actual_backdrop_wall", "actual_backdrop_cabinet_top", "vinegar",
            "soy_sauce", "other_clutter", "table", "cabinet"} == set(result)
    assert not np.array_equal(result["cooktop"], result["actual_cooktop_body"])


@pytest.mark.parametrize("missing", ["body", "burners", "wall", "cabinet", "vinegar", "soy_sauce"])
def test_missing_fixture_rejected(task, missing):
    for mapping in (task.kitchen_cooktop, task.kitchen_backdrop, task.kitchen_items):
        mapping.pop(missing, None)
    with pytest.raises(ValueError):
        oil.fixture_bounds(task)


def test_payload_never_added_as_neighbor(task):
    task.kitchen_items["oil"] = task.oil_entity
    with pytest.raises(ValueError, match="ambiguous"):
        oil.fixture_bounds(task)


@pytest.mark.parametrize("bad", [None, [], [[np.nan, 0, 0]], [[0, 0]]])
def test_missing_avatar_cloud_cannot_be_dropped(task, bad):
    calls = []
    task._avatar_obstacle_points = lambda: bad
    task._cabinet_obstacle_points = task._cooktop_obstacle_points = lambda: np.ones((2, 3))
    arm = NS(planner=NS(update_obstacles=lambda *a, **k: calls.append(a)))
    with pytest.raises((ValueError, TypeError)):
        oil.refresh_obstacles(task, arm)
    assert not calls


def test_cloud_failure_propagates(task):
    task._avatar_obstacle_points = task._cabinet_obstacle_points = task._cooktop_obstacle_points = lambda: np.ones((2, 3))
    task._AVATAR_OBSTACLE_RES = .02
    def fail(*args, **kwargs):
        raise RuntimeError("native cloud failure")
    with pytest.raises(RuntimeError, match="cloud failure"):
        oil.refresh_obstacles(task, NS(planner=NS(update_obstacles=fail)))


def test_full_joint_order_and_finger_sources():
    names = ["panda_finger_joint2"] + [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1"]
    index = {f"joint{i}": i-1 for i in range(1, 8)} | {"finger_joint1": 7, "finger_joint2": 8}
    arm = NS(planner=NS(_mplib=NS(robot=NS(get_user_joint_names=lambda: names)), _arm_joint_index=index),
             entity=NS(get_qpos=lambda: np.arange(9)/100), _cached_target=np.arange(9)/200,
             gripper_closed=0., gripper_open=.04, gripper_val=.04)
    full, fingers = oil.joint_state(arm)
    assert full == pytest.approx([.08, 0., .01, .02, .03, .04, .05, .06, .07])
    assert fingers["cached"] == pytest.approx([.035, .04])
    assert fingers["scalar"] == pytest.approx([.0016, .0016])


@pytest.mark.parametrize("problem", ["prior", "avatar", "retreat", "checker", "attached", "debug"])
def test_unsupported_execution_rejected(task, problem):
    arm = NS(planner=NS(_bounded_paths_debug_enabled=lambda: problem == "debug"))
    if problem == "prior": task.plan_success = False
    if problem == "avatar": task.avatar_collided = True
    if problem == "retreat": task._avatar_retreat_enabled = lambda: True
    if problem == "checker": task.avatar_retreat_checker = object()
    if problem == "attached": task._gripper_attached = {"right": object()}
    with pytest.raises(ValueError):
        oil._require_mode(task, arm)


def test_rejection_restores_avatar_and_never_opens(task, monkeypatch):
    def reject(*args):
        raise ValueError("blocked before carry")
    monkeypatch.setattr(oil, "_Placement", reject)
    assert not oil.place_rotated_oil(task, "right", 4)
    assert not task.plan_success and task.avatar.frame_ratio == 4
    assert task._oil_rotated_diagnostics["opened"] is False


def test_timed_pd_and_quiet_arrival(task):
    operation = oil._Placement.__new__(oil._Placement)
    operation.task, operation.tag, operation.grip = task, "right", .04
    operation.phase = "hover"
    commands, grips, steps = [], [], []
    identity = np.array([0., 0., 1., 1., 0., 0., 0.])
    operation.arm = NS(set_arm_joints=lambda q: commands.append(q.copy()),
                       tcp_offset=NS(to_pose7=lambda: [0, 0, 0, 1, 0, 0, 0]))
    task.robot = NS(set_gripper=lambda v, _: grips.append(v))
    operation.step = lambda **kw: steps.append(kw)
    operation.observe = lambda **kw: (oil.pose_matrix(identity), oil._pose(task.oil_entity))
    operation.emit = lambda *a, **kw: None
    result = {"position": np.ones((3, 7))}
    operation.execute(result, identity, held=True)
    assert len(steps) == 3*2 + 50  # Five quiet20ms samples, after timed replay.
    assert grips == [.04]*56 and len(commands) == 56


@pytest.fixture
def observed_operation(task):
    operation = oil._Placement.__new__(oil._Placement)
    operation.task, operation.tag, operation.minimum_bottom = task, "right", float("inf")
    fingers = [entity() for _ in range(2)]
    for index, finger in enumerate(fingers):
        finger.name = "finger" + str(index)
    operation.arm = NS(entity=NS(links=fingers), get_ee_pose=lambda: [0, 0, 1, 1, 0, 0, 0],
                       planner=NS(_bounded_paths_debug_enabled=lambda: False))
    operation.held = np.eye(4)
    operation.clutter = {name: oil._pose(item) for name, item in task.kitchen_items.items()}
    return operation


def test_retention_in_tcp_frame_allows_intended_rotation(observed_operation):
    operation = observed_operation
    operation.arm.get_ee_pose = lambda: [0, 0, 1, 0, 0, 0, 1]
    operation.task.oil_entity.get_quat = lambda: [0, 0, 0, 1]
    operation.observe(held=True)


def test_slip_rejected_without_recaching(observed_operation):
    operation = observed_operation
    operation.task.oil_entity.get_pos = lambda: [0, 0, .99]
    original = operation.held.copy()
    with pytest.raises(ValueError, match="slipped"):
        operation.observe(held=True)
    np.testing.assert_array_equal(operation.held, original)


def test_neighbor_motion_is_not_a_nominal_placement(observed_operation):
    operation = observed_operation
    operation.task.kitchen_items["vinegar"].get_pos = lambda: [.003, 0, 1]
    with pytest.raises(ValueError, match="neighbor moved"):
        operation.observe(held=True)


def test_fingers_remain_clear_during_opening(observed_operation):
    operation = observed_operation
    operation.arm.entity.links[0].get_AABB = lambda: np.array([[0, 0, .765], [.1, .1, .8]])
    with pytest.raises(ValueError, match="fingers"):
        operation.observe(held=False)


def test_released_bottle_impact_is_not_ignored(observed_operation):
    operation = observed_operation
    operation.task.oil_entity.get_AABB = lambda: np.array([[0, 0, .762], [.1, .1, .8]])
    with pytest.raises(ValueError, match="penetrated"):
        operation.observe(held=False)


@pytest.mark.parametrize("mismatch", [None, "position", "rotation"])
def test_fresh_context_checks_physical_native_frame_and_exits(observed_operation, monkeypatch, mismatch):
    operation = observed_operation
    operation.payload = np.array([[0., 0., 0.], [.05, .05, .15]])
    operation.arm.gripper_open = .04
    operation.arm.tcp_offset = NS(to_pose7=lambda: [0, 0, 0, 1, 0, 0, 0])
    operation.arm.planner._mplib = NS(robot=NS(get_base_pose=lambda: NS(p=np.zeros(3), q=[1, 0, 0, 0])))
    events = []
    monkeypatch.setattr(oil, "refresh_obstacles", lambda *a: events.append("refresh"))
    monkeypatch.setattr(oil, "joint_state", lambda a: (np.zeros(9), {"cached": [0, 0], "scalar": [0, 0]}))
    class Planner:
        def __init__(self, native, **kwargs):
            assert kwargs["held"] is False and "released_bottle" in kwargs["fixture_aabbs"]
        def __enter__(self):
            events.append("enter")
            return self
        def __exit__(self, *args):
            events.append("exit")
        def check_state(self):
            pose = [0., 0., 1., 1., 0., 0., 0.]
            if mismatch == "position": pose[0] = .003
            if mismatch == "rotation": pose[3], pose[6] = np.cos(.01), np.sin(.01)
            return {"tcp_pose7": pose}
    monkeypatch.setattr(oil, "OilRecoveryPlan", Planner)
    if mismatch is None:
        with operation.context(held=False): events.append("planned")
        assert events == ["refresh", "enter", "planned", "exit"]
    else:
        with pytest.raises(ValueError, match="frame disagreement"):
            with operation.context(held=False): pytest.fail("bad frame accepted")
        assert events == ["refresh", "enter", "exit"]


@pytest.mark.parametrize("problem", [None, "high", "penetrated", "moving", "spinning", "finger_touch", "hand_touch"])
def test_completion_requires_quiet_table_support_and_hand_separation(task, problem):
    task.oil_entity = entity((0, 0, .765), (.05, .05, .90))
    task.oil_entity.get_vel = lambda: [0., 0., .05 if problem == "moving" else 0.]
    task.oil_entity.get_ang = lambda: [.5 if problem == "spinning" else 0., 0., 0.]
    fingers = [entity((.1, .1, 1), (.2, .2, 1.1)) for _ in range(2)]
    for index, finger in enumerate(fingers): finger.name = "finger"+str(index)
    arm = NS(entity=NS(links=fingers), ee_link=entity((.1, .1, 1), (.2, .2, 1.1)))
    if problem == "high": task.oil_entity.get_AABB = lambda: np.array([[0, 0, .865], [.05, .05, 1.]])
    if problem == "penetrated": task.oil_entity.get_AABB = lambda: np.array([[0, 0, .762], [.05, .05, .90]])
    if problem == "finger_touch": fingers[0].get_AABB = task.oil_entity.get_AABB
    if problem == "hand_touch": arm.ee_link.get_AABB = task.oil_entity.get_AABB
    assert oil.released_support(task, arm)["supported_and_quiet"] is (problem is None)


def test_no_simulator_teleport_attachment_or_fallback_calls():
    tree = ast.parse(Path(oil.__file__).read_text())
    forbidden = {"set_qpos", "set_pos", "set_quat", "teleport_arm_joints", "attach_to_gripper",
                 "inverse_kinematics", "solve_ik_cartesian", "execute_plan", "move_and_execute"}
    assert not [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden]


def test_task_hook_default_off_after_grasp_cache():
    source = (Path(oil.__file__).parents[1]/"tasks/oil_bottle_recovery.py").read_text()
    assert 'self.config.get("oil_rotated_placement", False)' in source
    assert source.index("self._T_ee_bottle = ee_pose.inv() * bottle_now") < source.index(
        'self.config.get("oil_rotated_placement", False)') < source.index("upright_bottle_hover =")
