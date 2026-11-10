"""Interrupt contract for categorize, burger/fries and cabinet.

Scene-free: the robot, avatar and objects are small fakes.  Covers the
progress trigger wiring, eligibility filters, phase gates, the glide (no
teleport), clip attach target / idle return, and the no-override rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.base_task import BaseTask
from envs.interaction.interrupt_progress import (
    DEFAULT_IDLE_RETURN_FRAMES,
    interrupt_metrics,
    new_interrupt_episode,
)
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.place_burger_and_fries_interruption import PlaceBurgerAndFriesInterruption
from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
from envs.utils import Pose

TABLE = 0.765


class _Entity:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos.copy()


class _Actor:
    def __init__(self, pos):
        self.entity = _Entity(pos)

    def get_pose(self):
        return SimpleNamespace(p=self.entity.pos.copy())


class _Arm:
    def __init__(self, tcp, gripper=1.0):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


class _AvatarRobot:
    def __init__(self, root):
        self.global_trans = np.asarray(root, dtype=float)

    def get_palm_center(self, hand_id):
        return self.global_trans + np.array([0.0, 0.0, 0.9])


class _Avatar:
    def __init__(self, root):
        self.robot = _AvatarRobot(root)
        self.roots = [self.robot.global_trans.copy()]
        self.motion_modules = {}
        self.busy = False
        self.frame_ratio = 10.0

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.roots.append(self.robot.global_trans.copy())

    def spare(self):
        return not self.busy


def _wire(task, tcp, avatar_root=(0.0, 0.7, -0.18)):
    task.TABLE_TOP_Z = TABLE
    arm = _Arm(tcp)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.avatar = _Avatar(avatar_root)
    task._eval_palm_default = np.array([0.0, 0.3, 0.75])
    plays = []

    def play(**kwargs):
        plays.append(kwargs)
        task.avatar.busy = True

    task._play_inspect_motion = play
    return arm, plays


def _max_root_jump(avatar):
    roots = np.asarray(avatar.roots)
    return float(np.max(np.linalg.norm(np.diff(roots, axis=0), axis=1)))


# ----------------------------------------------------------------------
# categorize_interrupt
# ----------------------------------------------------------------------


def _categorize_task():
    task = CategorizeObjectsInterruption({"eval_mode": True})
    task.basket_poses = [Pose((-0.36, -0.30, 0.80)), Pose((0.36, -0.30, 0.80))]
    task._sort_targets = [
        (_Actor((-0.20, 0.18, TABLE + 0.03)), "a", 0, 0),
        (_Actor((0.20, 0.18, TABLE + 0.03)), "b", 0, 1),
    ]
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task._interrupted_idx = 0  # pre-chosen at reset; NOT the eval target
    return task


def test_categorize_never_fires_for_a_still_robot():
    task = _categorize_task()
    arm, plays = _wire(task, tcp=(0.20, 0.0, TABLE + 0.25))  # ~0.3 m from b
    np.random.seed(0)
    task._eval_at_reset()
    task._arm_eval_interrupt()
    for step in range(1, 301):
        task._eval_at_step(step)
    m = interrupt_metrics(task)
    assert m["fired"] is False and m["not_fired_reason"] == "robot_not_moved"
    assert plays == [] and len(task.avatar.roots) == 1


def test_categorize_targets_the_approached_object_and_glides():
    task = _categorize_task()
    arm, plays = _wire(task, tcp=(0.0, -0.20, TABLE + 0.45))
    np.random.seed(0)
    task._eval_at_reset()
    task._arm_eval_interrupt()
    for step in range(1, 25):
        task._eval_at_step(step)
    assert interrupt_metrics(task)["fired"] is False  # robot has not moved
    arm.tcp = np.array([0.20, 0.10, TABLE + 0.18])  # committing to object 1
    task._eval_at_step(25)
    m = interrupt_metrics(task)
    assert m["fired"] and m["trigger_step"] == 25 and m["trigger_target"] == 1
    assert task._interrupted_idx == 1 and task._eval_inspect_fired
    glide = task._interrupt.human.glide_steps
    for step in range(26, 26 + glide):
        task._eval_at_step(step)
    assert len(plays) == 1
    call = plays[0]
    assert call["attach_obj"] is task._sort_targets[1][0].entity
    assert call["return_to_idle_after"] == DEFAULT_IDLE_RETURN_FRAMES
    assert _max_root_jump(task.avatar) <= 0.10  # glide, no teleport
    # Clip done -> glide back to the pre-trigger root.
    task.avatar.busy = False
    start = task.avatar.roots[0]
    for step in range(100, 100 + 2 + task._interrupt.human.glide_out_steps):
        task._eval_at_step(step)
    assert task._interrupt.human.done
    np.testing.assert_allclose(task.avatar.robot.global_trans, start)


def test_categorize_skips_sorted_and_held_objects():
    task = _categorize_task()
    arm, _plays = _wire(task, tcp=(0.20, 0.18, TABLE + 0.05), avatar_root=(0, 0.7, 0))
    # object 0 sorted into its basket
    task._sort_targets[0][0].entity.pos = np.array([-0.36, -0.30, 0.80])
    arm.gripper_val = 0.1  # closed on object 1
    assert task._interrupt_candidates() == {}
    arm.gripper_val = 1.0
    assert list(task._interrupt_candidates()) == [1]


def test_categorize_expert_attempt_is_idempotent(monkeypatch):
    task = _categorize_task()
    task._boost_arm_pd = lambda: None
    task._open_basket_lids = lambda: None
    task._move_to_safe = lambda _arm: None
    task.step_sim = lambda: None
    sorted_now = []
    task._sort_target_once = lambda idx, arm_tag="right": sorted_now.append(idx) or True
    task._sort_targets[1][0].entity.pos = np.array([0.36, -0.30, 0.80])  # b sorted
    assert task.interrupt_expert_attempt() is True
    assert sorted_now == [0]


# ----------------------------------------------------------------------
# place_burger_fries_interrupt
# ----------------------------------------------------------------------


def _burger_task(monkeypatch):
    task = PlaceBurgerAndFriesInterruption({"eval_mode": True})
    burger = _Actor((0.10, 0.0, TABLE + 0.04))
    fries = _Actor((0.0, -0.08, TABLE + 0.05))
    task.food_actors = [
        (burger, SimpleNamespace(asset_id="burger", model_id=0)),
        (fries, SimpleNamespace(asset_id="fries", model_id=0)),
    ]
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task._food_on_tray = lambda actor: False
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.3, 0.75])
    task.AVATAR_HAND_ID = 1
    return task


def test_burger_no_calibration_teleport_at_trigger(monkeypatch):
    task = _burger_task(monkeypatch)
    arm, plays = _wire(task, tcp=(0.0, -0.45, TABLE + 0.40))
    calibrations = []
    task._calibrate_avatar_for_inspect = lambda: calibrations.append(1)
    task.interaction_controller.init_eval_state(task)
    # Hover over the burger as the demos do (xy ~0, ~0.29 m above it).
    arm.tcp = np.array([0.10, 0.0, TABLE + 0.04 + 0.29])
    for _ in range(20):
        task._eval_step_avatar()
    m = interrupt_metrics(task)
    assert m["fired"] and m["trigger_target"] == 0
    assert m["trigger_step"] == PlaceBurgerAndFriesInterruption.INTERRUPT_MIN_POLICY_STEPS
    for _ in range(task._interrupt.human.glide_steps):
        task._eval_step_avatar()
    assert calibrations == [] and len(plays) == 1
    assert plays[0]["attach_obj"] is task.food_actors[0][0].entity
    assert _max_root_jump(task.avatar) <= 0.10


def test_burger_trigger_needs_a_hover_dwell_not_just_motion(monkeypatch):
    # With the moved-gate alone the trigger would fire on any
    # robot motion (the reset TCP is within the 0.32 m radius of the food).
    task = _burger_task(monkeypatch)
    arm, plays = _wire(task, tcp=(0.0, -0.45, TABLE + 0.40))
    task.interaction_controller.init_eval_state(task)
    arm.tcp = np.array([0.10, -0.05, TABLE + 0.20])    # moved, near, no hover
    for _ in range(20):
        task._eval_step_avatar()
    m = interrupt_metrics(task)
    assert m["fired"] is False and m["not_fired_reason"] == "no_hover_dwell"
    hover = np.array([0.10, 0.0, TABLE + 0.04 + 0.29])
    arm.tcp = hover
    task._eval_step_avatar()
    task._eval_step_avatar()
    arm.tcp = np.array([0.10, -0.05, TABLE + 0.20])    # 2-step hover, then away
    task._eval_step_avatar()
    assert interrupt_metrics(task)["fired"] is False
    arm.tcp = hover
    for _ in range(3):
        task._eval_step_avatar()
    m = interrupt_metrics(task)
    assert m["fired"] and m["trigger_target"] == 0 and m["trigger_step"] == 26
    assert len(plays) == 0  # glide first, clip after


def test_burger_robot_carrying_food_blocks_trigger(monkeypatch):
    task = _burger_task(monkeypatch)
    arm, _plays = _wire(task, tcp=(0.0, -0.45, TABLE + 0.40))
    task.interaction_controller.init_eval_state(task)
    # Robot carries the fries (lifted, closed gripper) right past the burger.
    task.food_actors[1][0].entity.pos = np.array([0.10, -0.05, TABLE + 0.15])
    arm.tcp = np.array([0.10, -0.05, TABLE + 0.17])
    arm.gripper_val = 0.2
    for _ in range(30):
        task._eval_step_avatar()
    m = interrupt_metrics(task)
    assert m["fired"] is False and m["not_fired_reason"] == "robot_holding_object"


def test_burger_take_action_never_substitutes_the_policy_action(monkeypatch):
    task = PlaceBurgerAndFriesInterruption({"eval_mode": True})
    task.avatar = object()
    seen = []
    task._eval_step_avatar = lambda: seen.append("tick")
    monkeypatch.setattr(
        BaseTask, "take_action",
        lambda _self, action, action_type="qpos": seen.append((action, action_type)),
    )
    action = np.arange(8.0)
    task.take_action(action, action_type="ee")
    assert seen[0] == "tick" and seen[1][0] is action and seen[1][1] == "ee"


# ----------------------------------------------------------------------
# put_object_cabinet_interrupt
# ----------------------------------------------------------------------


def _cabinet_task():
    task = PutObjectInCabinetInterruption({"eval_mode": True})
    task.object = _Actor((0.0, 0.0, TABLE + 0.04))
    task._object_id = "obj"
    task._object_model_id = 0
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task._drawer_link = None
    task._drawer_dof_idx = 0
    task._drawer_q = 0.0
    task.cabinet = SimpleNamespace(entity=SimpleNamespace(
        get_qpos=lambda: np.array([task._drawer_q]),
    ))
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.3, 0.75])
    return task


def test_cabinet_trigger_waits_for_the_drawer_and_burns_the_legacy_draw():
    task = _cabinet_task()
    arm, plays = _wire(task, tcp=(0.30, -0.45, TABLE + 0.40))
    np.random.seed(3)
    expected = int(np.random.randint(100, 201))
    tail = np.random.rand()
    np.random.seed(3)
    task._init_eval_interrupt_state()
    assert task._eval_trigger_step == expected and np.random.rand() == tail
    arm.tcp = np.array([0.0, -0.05, TABLE + 0.20])  # hovering over the object
    for _ in range(30):
        task._eval_step_avatar()
    m = interrupt_metrics(task)
    assert m["fired"] is False and m["not_fired_reason"] == "drawer_not_open"
    task._drawer_q = 0.08
    task._eval_step_avatar()
    assert interrupt_metrics(task)["fired"] is True
    for _ in range(task._interrupt.human.glide_steps):
        task._eval_step_avatar()
    assert len(plays) == 1
    assert plays[0]["return_to_idle_after"] == DEFAULT_IDLE_RETURN_FRAMES
    assert _max_root_jump(task.avatar) <= 0.10


def test_cabinet_held_object_is_not_a_candidate():
    task = _cabinet_task()
    arm, _ = _wire(task, tcp=(0.0, 0.0, TABLE + 0.08))
    arm.gripper_val = 0.0
    assert task.interaction_controller.eligible_objects(task) == {}
    arm.gripper_val = 1.0
    assert list(task.interaction_controller.eligible_objects(task)) == [0]


def test_cabinet_blind_rollout_does_not_self_tick():
    task = PutObjectInCabinetInterruption({"eval_mode": True})
    task.cabinet = None
    task.object = None
    ticks = []
    task._eval_step_avatar = lambda: ticks.append(1)
    task.step_sim = lambda: None
    assert task.play_blind_once() is False
    assert ticks == []


def test_metrics_report_not_fired_reason_for_all_three():
    for task in (_categorize_task(), _cabinet_task()):
        new_interrupt_episode(task)
        assert interrupt_metrics(task)["fired"] is False


def test_cabinet_expert_attempt_is_idempotent_and_hovers_only_before_firing():
    task = _cabinet_task()
    events = []
    task._interrupt_expert_started = True
    task._drawer_ready = lambda: True
    task._hold_drawer_open = lambda: events.append("hold") or True
    task._open_drawer_for_rollout = lambda _arm: events.append("open") or True
    in_drawer = {"v": False}
    task._object_in_drawer_metrics = lambda: {"object_in_drawer": in_drawer["v"]}
    task._interrupt_expert_hover = lambda _arm: events.append("hover")
    task._pick_object_into_drawer = lambda _arm: events.append("pick") or True
    task.step_sim = lambda: None
    episode = new_interrupt_episode(task)
    assert task.interrupt_expert_attempt() is True
    assert events == ["hold", "hover", "pick"]
    events.clear()
    episode.trigger.fired = True  # resumed after the human: no second hover
    assert task.interrupt_expert_attempt() is True
    assert events == ["hold", "pick"]
    events.clear()
    in_drawer["v"] = True
    task._drawer_ready = lambda: False
    assert task.interrupt_expert_attempt() is True
    assert events == ["open"]
