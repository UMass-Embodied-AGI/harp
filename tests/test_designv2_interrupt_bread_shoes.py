"""Interrupt contract for place_bread_in_basket / place_dual_shoes.

Scene-free: fakes stand in for the robot, avatar and objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.tasks.place_bread_in_basket_interruption import PlaceBreadInBasketInterruption
from envs.tasks.place_pair_of_shoes_interruption import PlacePairOfShoesInterruption


class _Entity:
    def __init__(self, pos, half=0.05):
        self.pos = np.asarray(pos, dtype=float)
        self.half = half

    def get_pos(self):
        return self.pos.copy()

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.array([self.pos - self.half, self.pos + self.half])


class _Arm:
    def __init__(self, ee):
        self.ee = ee
        self.gripper_val = 1.0

    def get_ee_pose(self):
        return list(self.ee) + [1.0, 0.0, 0.0, 0.0]


class _AvatarRobot:
    def __init__(self, pos):
        self.global_trans = np.asarray(pos, dtype=float).copy()

    def get_palm_center(self, hand_id):
        return self.global_trans + np.array([0.0, 0.0, 0.5])


class _Avatar:
    def __init__(self, pos):
        self.robot = _AvatarRobot(pos)
        self.resets = []
        self.plays = []
        self.motion_modules = {}
        self.busy = False

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.resets.append(self.robot.global_trans.copy())

    def play_animation(self, name, **kwargs):
        self.plays.append((name, kwargs))
        self.busy = True

    def spare(self):
        return not self.busy


def _shoes_task(ee):
    task = PlacePairOfShoesInterruption({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    task.shoebox = SimpleNamespace(entity=_Entity([-0.10, -0.25, 0.76]))
    task.shoes = [
        SimpleNamespace(entity=_Entity([0.05, -0.10, 0.80])),
        SimpleNamespace(entity=_Entity([-0.06, -0.12, 0.80])),
    ]
    arm = _Arm(ee)
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.avatar = _Avatar(task.avatar_init_pos)
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.5, 0.9])
    return task, arm


def _bread_task(ee):
    task = PlaceBreadInBasketInterruption({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    task._basket_rim_z = 0.82
    task.basket = SimpleNamespace(entity=_Entity([-0.2, -0.3, 0.75]))
    task.breads = [
        SimpleNamespace(entity=_Entity([0.05, -0.10, 0.79])),
        SimpleNamespace(entity=_Entity([0.15, -0.05, 0.79])),
    ]
    task._bread_center_mesh = np.zeros(3)
    task._bread_scale = 1.0
    arm = _Arm(ee)
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.avatar = _Avatar(task.avatar_init_pos)
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.5, 0.9])
    task._play_inspect_motion = lambda **kw: task.avatar.play_animation("Inspect", **kw)
    return task, arm


@pytest.mark.parametrize("factory", (_shoes_task, _bread_task))
def test_hold_still_robot_near_objects_never_triggers(factory):
    # Reset TCP within 0.12 m (xy) of a target: the retired force_dist
    # trigger fired at step 1 here.  A robot that never moves never fires.
    task, _arm = factory([0.05, -0.05, 0.92])
    task._init_eval_interrupt_state()
    for _ in range(300):
        task._eval_step_avatar()
    assert task._eval_interrupt_fired is False
    m = task.evaluate.__func__  # evaluate surface exists
    metrics = task.interaction_controller.interaction_metrics(task)
    assert metrics["interrupt"]["fired"] is False
    assert metrics["interrupt"]["not_fired_reason"] == "robot_not_moved"
    assert task.avatar.resets == [] and task.avatar.plays == []
    assert callable(m)


@pytest.mark.parametrize("factory", (_shoes_task, _bread_task))
def test_progress_trigger_glides_then_plays_clip_and_returns_to_idle(factory):
    ee = [0.40, -0.55, 1.15]
    task, arm = factory(ee)
    task._init_eval_interrupt_state()
    for _ in range(20):
        task._eval_step_avatar()
    assert task._eval_interrupt_fired is False
    arm.ee = [0.05, -0.10, 0.97]  # hovering over object 0
    task._eval_step_avatar()
    assert task._eval_interrupt_fired is True
    assert task._interrupt.trigger.trigger_target == 0
    glide = task._interrupt.human.glide_steps
    assert glide >= 30  # >= 1.5 s of visible approach at 20 Hz
    for _ in range(glide):
        task._eval_step_avatar()
    assert len(task.avatar.plays) == 1
    _name, kwargs = task.avatar.plays[0]
    assert kwargs["return_to_idle_after"] == task._interrupt.human.idle_return_frames
    resets = np.asarray(task.avatar.resets)
    start = np.asarray(task.avatar_init_pos, dtype=float)
    total = np.linalg.norm(resets[-1] - start)
    assert np.max(np.linalg.norm(np.diff(np.vstack([start, resets]), axis=0), axis=1)) \
        <= total / glide + 1e-9
    # Clip (and its idle blend) ends -> glide back to the pre-trigger root.
    task.avatar.busy = False
    for _ in range(task._interrupt.human.glide_out_steps + 1):
        task._eval_step_avatar()
    assert task._interrupt.human.done
    np.testing.assert_allclose(task.avatar.robot.global_trans, start)


def test_shoes_robot_holding_a_shoe_blocks_trigger():
    ee = [0.40, -0.55, 1.15]
    task, arm = _shoes_task(ee)
    task._init_eval_interrupt_state()
    for _ in range(20):
        task._eval_step_avatar()
    # Carrying shoe 1 (lifted, gripper closed) past shoe 0.
    task.shoes[1].entity.pos = np.array([0.05, -0.05, 0.95])
    arm.ee = [0.05, -0.05, 0.97]
    arm.gripper_val = 0.2
    task._eval_step_avatar()
    assert task._eval_interrupt_fired is False
    assert task._interrupt.trigger.block_reason == "robot_holding_object"


@pytest.mark.parametrize("task_cls", (PlacePairOfShoesInterruption, PlaceBreadInBasketInterruption))
def test_take_action_applies_policy_action_unmodified(task_cls, monkeypatch):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    ticks = []
    task._eval_step_avatar = lambda: ticks.append(1)
    seen = {}

    def base_action(_self, action, action_type="qpos"):
        seen["action"], seen["type"] = action, action_type
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = np.arange(8.0)
    assert task.take_action(action, action_type="ee") == "obs"
    assert ticks == [1] and seen["action"] is action and seen["type"] == "ee"


def test_retired_constants_are_gone():
    for cls in (PlacePairOfShoesInterruption, PlaceBreadInBasketInterruption):
        for name in (
            "EVAL_FORCE_TRIGGER_DIST_XY", "PACE_TRIGGER_PROGRESS",
            "PACE_TRIGGER_DIST_XY", "EVAL_ROBOT_RETREAT_POLICY_STEPS",
            "EVAL_ROBOT_RETURN_POLICY_STEPS", "EVAL_GLIDE_POLICY_STEPS",
        ):
            assert not hasattr(cls, name), (cls.__name__, name)
    assert PlaceBreadInBasketInterruption.INTERRUPT_COMMIT_RADIUS_M == 0.22
    assert PlacePairOfShoesInterruption.INTERRUPT_COMMIT_RADIUS_M == 0.24
    for cls in (PlacePairOfShoesInterruption, PlaceBreadInBasketInterruption):
        assert callable(cls.interrupt_expert_attempt)
        assert callable(cls.interrupt_expert_retreat)
