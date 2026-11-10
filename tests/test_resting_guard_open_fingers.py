"""Review clip 08: open gripper fingers left inside an object still hold it.

``resting_guard.robot_holds`` used to count the gripper only when closed (or
kinematically attached), so a "successful" 3-bowl stack ended with the open
fingers still inside the top bowl.  Now a finger inside, or within 1 cm of,
the object's live AABB counts as holding it while the TCP is within the
radius; a released and retracted gripper does not.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import resting_guard
from envs.task_bases.resting_guard import (
    DEFAULT_FINGER_AABB_MARGIN_M,
    fingers_in_object,
    held_by_anyone,
    robot_holds,
)
from envs.task_bases.stack_bowls_three import StackBowlsThree
from envs.tasks.stack_three_bowls_contribution import StackThreeBowlsContribution

TABLE = 0.765
# Franka: finger-link origins sit ~4.5 cm above the TCP (fingertip centre).
FINGER_ROOT_ABOVE_TCP = 0.045


class _Ent:
    def __init__(self, pos, half):
        self.pos = np.asarray(pos, dtype=float)
        self.half = np.broadcast_to(np.asarray(half, dtype=float), (3,)).copy()

    def get_pos(self):
        return self.pos.copy()

    def get_AABB(self):
        return np.stack([self.pos - self.half, self.pos + self.half])


class _Link:
    def __init__(self, name, arm, side):
        self.name = name
        self._arm = arm
        self._side = side

    def get_pos(self):
        # Vertical top-down gripper: roots above the TCP, opened along x.
        half_open = 0.04 * self._arm.gripper_val
        return self._arm.tcp + np.array([self._side * half_open, 0.0, FINGER_ROOT_ABOVE_TCP])


class _Arm:
    def __init__(self):
        self.tcp = np.array([0.0, -0.6, 1.2])
        self.gripper_val = 1.0
        self.entity = SimpleNamespace(links=[
            SimpleNamespace(name="panda_link7", get_pos=lambda: self.tcp + [0, 0, 0.2]),
            _Link("panda_leftfinger", self, +1.0),
            _Link("panda_rightfinger", self, -1.0),
        ])

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0.0, 0.0, 0.0]])


def _task(task=None):
    task = task if task is not None else SimpleNamespace()
    arm = _Arm()
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm, left_arm=None, right_arm=arm)
    task.avatar = SimpleNamespace(robot=SimpleNamespace(
        attached_object=[None, None], two_hand_attached_object=None), spare=lambda: True)
    return task, arm


# A bowl: 12 cm wide, 5 cm tall, resting on the table.
BOWL_CENTER = np.array([0.0, 0.0, TABLE + 0.025])
BOWL_HALF = np.array([0.06, 0.06, 0.025])


def test_open_fingers_inside_a_bowl_count_as_holding_it():
    task, arm = _task()
    bowl = _Ent(BOWL_CENTER, BOWL_HALF)
    # Open gripper, fingertips 1 cm above the bowl floor near the rim.
    arm.gripper_val = 1.0
    arm.tcp = BOWL_CENTER + np.array([0.0, 0.0, -0.015])
    assert fingers_in_object(task, bowl) is True
    assert robot_holds(task, bowl, BOWL_CENTER, radius_m=0.09) is True
    assert held_by_anyone(task, bowl, BOWL_CENTER, radius_m=0.09) is True


def test_open_fingers_within_the_margin_above_the_rim_still_hold():
    task, arm = _task()
    bowl = _Ent(BOWL_CENTER, BOWL_HALF)
    rim = BOWL_CENTER[2] + BOWL_HALF[2]
    arm.tcp = np.array([0.0, 0.0, rim + 0.5 * DEFAULT_FINGER_AABB_MARGIN_M])
    assert robot_holds(task, bowl, BOWL_CENTER, radius_m=0.09) is True
    arm.tcp = np.array([0.0, 0.0, rim + 2.0 * DEFAULT_FINGER_AABB_MARGIN_M])
    assert robot_holds(task, bowl, BOWL_CENTER, radius_m=0.09) is False


def test_released_and_retracted_gripper_does_not_hold():
    task, arm = _task()
    bowl = _Ent(BOWL_CENTER, BOWL_HALF)
    arm.gripper_val = 1.0
    arm.tcp = BOWL_CENTER + np.array([0.0, 0.0, 0.12])     # lifted clear
    assert fingers_in_object(task, bowl) is False
    assert robot_holds(task, bowl, BOWL_CENTER, radius_m=0.09) is False
    arm.tcp = np.array([0.0, -0.6, 1.2])                   # parked
    assert held_by_anyone(task, bowl, BOWL_CENTER, radius_m=0.09) is False


def test_fingers_inside_but_tcp_outside_the_radius_do_not_hold():
    task, arm = _task()
    big = _Ent([0.0, 0.0, TABLE + 0.10], [0.30, 0.30, 0.10])
    arm.tcp = np.array([0.25, 0.0, TABLE + 0.10])
    assert fingers_in_object(task, big) is True
    assert robot_holds(task, big, np.array([0.0, 0.0, TABLE + 0.10]), radius_m=0.07) is False


def test_closed_gripper_rule_is_unchanged():
    task, arm = _task()
    cube = _Ent([0.0, 0.0, TABLE + 0.02], 0.02)
    arm.gripper_val = 0.2
    arm.tcp = np.array([0.0, 0.0, TABLE + 0.08])            # closed, fingers clear
    assert robot_holds(task, cube, cube.get_pos()) is True
    arm.gripper_val = 1.0
    assert robot_holds(task, cube, cube.get_pos()) is False


def test_without_finger_links_the_check_is_skipped(monkeypatch):
    task, arm = _task()
    arm.entity = SimpleNamespace(links=[])
    bowl = _Ent(BOWL_CENTER, BOWL_HALF)
    arm.tcp = BOWL_CENTER.copy()
    assert resting_guard.finger_points(task) is None
    assert robot_holds(task, bowl, BOWL_CENTER, radius_m=0.09) is False


@pytest.mark.parametrize("task_cls", [StackBowlsThree, StackThreeBowlsContribution])
def test_stack_success_waits_for_the_fingers_to_leave_the_top_bowl(task_cls):
    """Clip 08 shape: three bowls stacked, gripper open with its fingers in
    the top bowl -> not a success; lifted clear -> success."""
    task = task_cls({})
    task._num_bowls = 3
    task._stacked_indices = [1, 2]
    task._bowl_height = 0.05
    bowls = [SimpleNamespace(entity=_Ent([0.0, 0.0, TABLE + 0.025 + 0.015 * i], BOWL_HALF))
             for i in range(3)]
    task.bowls = bowls
    task._bowl_xy_half_bounds = lambda: np.array([0.06, 0.06])
    task, arm = _task(task)
    ok = StackBowlsThree.check_success.__get__(task)
    top = bowls[2].entity.get_pos()
    arm.gripper_val = 1.0
    arm.tcp = top + np.array([0.0, 0.0, -0.01])
    assert ok() is False
    arm.tcp = top + np.array([0.0, 0.0, 0.15])
    assert ok() is True
