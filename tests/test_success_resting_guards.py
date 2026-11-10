"""Resting-success regressions: a placed object counts only while it rests at the
goal, and success must hold for N policy steps (envs/success_hold.py).

For every covered family (blocks rgb/size, categorize, dual shoes,
stack_bowls_three, oil_bottle_recovery, take_from_human_safety): an object
held by the human, held in a closed gripper, floating above the goal, or
carried through it mid-transport must not score success; the same object
resting there (gripper open) does.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.success_hold import DEFAULT_SUCCESS_HOLD_STEPS, success_hold_steps
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.task_bases.blocks_ranking_size import BlocksRankingSize
from envs.task_bases.place_dual_shoes import PlaceDualShoes
from envs.task_bases.stack_bowls_three import StackBowlsThree
from envs.tasks.rank_blocks_by_color_contribution import RankBlocksByColorContribution
from envs.tasks.rank_blocks_by_color_interruption import RankBlocksByColorInterruption
from envs.tasks.rank_blocks_by_size_copresence import RankBlocksBySizeCopresence
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence
from envs.tasks.oil_bottle_recovery import OilBottleRecovery, _upright_q
from envs.tasks.place_pair_of_shoes_contribution import PlacePairOfShoesContribution
from envs.tasks.place_pair_of_shoes_copresence import PlacePairOfShoesCopresence
from envs.tasks.stack_three_bowls_interruption import StackThreeBowlsInterruption
from envs.tasks.stack_three_bowls_copresence import StackThreeBowlsCopresence
from envs.tasks.take_from_human_safety import TakeFromHumanSafety
from envs.utils import Pose

TABLE = 0.765


class _Ent:
    """Rigid body stand-in: position, identity quat, AABB of half size ``half``."""

    def __init__(self, pos, half=0.02, quat=(1.0, 0.0, 0.0, 0.0)):
        self.pos = np.asarray(pos, dtype=float)
        self.half = np.broadcast_to(np.asarray(half, dtype=float), (3,)).copy()
        self.quat = np.asarray(quat, dtype=float)

    def get_pos(self):
        return self.pos.copy()

    def set_pos(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_quat(self):
        return self.quat.copy()

    def get_AABB(self):
        return np.stack([self.pos - self.half, self.pos + self.half])


class _Arm:
    def __init__(self, tcp=(0.0, -0.6, 1.2), gripper=1.0):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0.0, 0.0, 0.0]])


class _Holders:
    """Avatar rig + robot arm attached to ``task``; toggles each holder."""

    def __init__(self, task):
        self.task = task
        self.rig = SimpleNamespace(attached_object=[None, None], two_hand_attached_object=None)
        task.avatar = SimpleNamespace(robot=self.rig, spare=lambda: True)
        self.arm = _Arm()
        task.robot = SimpleNamespace(get_arm=lambda _tag: self.arm, left_arm=None,
                                     right_arm=self.arm)

    def human_holds(self, entity):
        self.rig.attached_object[1] = (entity, np.zeros(3), np.eye(3))

    def release_all(self, park=(0.0, -0.6, 1.2)):
        self.rig.attached_object[1] = None
        self.rig.two_hand_attached_object = None
        self.arm.tcp = np.asarray(park, dtype=float)
        self.arm.gripper_val = 1.0

    def gripper_closed_at(self, pos):
        self.arm.tcp = np.asarray(pos, dtype=float)
        self.arm.gripper_val = 0.2

    def gripper_open_at(self, pos):
        self.arm.tcp = np.asarray(pos, dtype=float)
        self.arm.gripper_val = 1.0


def _assert_guard(task, holders, entity, rest_pos, float_pos, carry_pos, ok_fn):
    """The shared regression ladder for one placed object."""
    entity.set_pos(rest_pos)
    holders.release_all()
    assert ok_fn() is True, "resting object must count"

    holders.human_holds(entity)
    assert ok_fn() is False, "held by the human"
    holders.release_all()

    holders.gripper_closed_at(rest_pos)
    assert ok_fn() is False, "still in a closed gripper"
    holders.gripper_open_at(rest_pos)
    assert ok_fn() is True, "open gripper at the object is fine"
    holders.release_all()

    entity.set_pos(float_pos)
    assert ok_fn() is False, "floating above the goal"

    entity.set_pos(carry_pos)
    holders.gripper_closed_at(carry_pos)
    assert ok_fn() is False, "passing through mid-carry"
    holders.release_all()
    entity.set_pos(rest_pos)
    assert ok_fn() is True


# ----------------------------------------------------------------------
# Blocks (rgb, size)
# ----------------------------------------------------------------------


def _blocks_task(task_cls, keys, halves):
    task = task_cls({})
    task.TABLE_TOP_Z = TABLE
    task.plan_success = True
    task.avatar_collided = False
    task.avatar_collision_checker = None
    task.blocks = {}
    task.target_slots = {}
    for i, key in enumerate(keys):
        x = -0.16 + 0.16 * i
        task.target_slots[key] = np.array([x, -0.08, TABLE])
        task.blocks[key] = _Ent([x, -0.08, TABLE + halves[key]], half=halves[key])
    return task


@pytest.mark.parametrize("task_cls", [BlocksRankingRGB, RankBlocksByColorContribution,
                                      RankBlocksByColorInterruption])
def test_blocks_rgb_block_must_rest_in_its_slot(task_cls):
    keys = ("red", "green", "blue")
    half = BlocksRankingRGB.CUBE_HALF
    task = _blocks_task(task_cls, keys, {k: half for k in keys})
    holders = _Holders(task)
    cube = task.blocks["blue"]
    x, y = task.target_slots["blue"][:2]
    _assert_guard(
        task, holders, cube,
        rest_pos=[x, y, TABLE + half],
        float_pos=[x, y, TABLE + half + 0.05],
        carry_pos=[x, y, TABLE + 0.20],
        ok_fn=BlocksRankingRGB.check_success.__get__(task),
    )
    if task_cls is BlocksRankingRGB:
        assert task.evaluate()["block_resting_ok"] == {k: True for k in keys}


@pytest.mark.parametrize("task_cls", [BlocksRankingSize, RankBlocksBySizeCopresence])
def test_blocks_size_block_must_rest_in_its_slot(task_cls):
    keys = ("large", "medium", "small")
    halves = {k: float(BlocksRankingSize.SIZE_HALVES[k]) for k in keys}
    task = _blocks_task(task_cls, keys, halves)
    holders = _Holders(task)
    cube = task.blocks["small"]
    x, y = task.target_slots["small"][:2]
    h = halves["small"]
    _assert_guard(
        task, holders, cube,
        rest_pos=[x, y, TABLE + h],
        float_pos=[x, y, TABLE + h + 0.05],
        carry_pos=[x, y, TABLE + 0.20],
        ok_fn=BlocksRankingSize.check_success.__get__(task),
    )


# ----------------------------------------------------------------------
# Categorize (cooperative, interrupt, neutral)
# ----------------------------------------------------------------------


def _categorize_task(task_cls):
    task = task_cls({})
    task.plan_success = True
    task.avatar_collision_checker = None
    task.avatar_collided = False
    basket_z = TABLE + task._BASKET_HALF_Z          # basket pose = box centre
    task.basket_poses = [Pose((0.0, 0.15, basket_z)), Pose((0.25, 0.15, basket_z))]
    obj = SimpleNamespace(entity=_Ent((0.0, 0.15, TABLE + 0.04), half=0.03))
    task._sort_targets = [(obj, "robot", 1, 0)]
    task._avatar_sort_targets = []
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task._actor_footprint_aabb = lambda actor, _n, _m=0: (
        actor.entity.get_pos()[:2] - 0.01, actor.entity.get_pos()[:2] + 0.01)
    return task, obj


@pytest.mark.parametrize("task_cls", [CategorizeObjectsContribution, CategorizeObjectsInterruption,
                                      CategorizeObjectsCopresence])
def test_categorize_object_must_rest_in_its_basket(task_cls):
    task, obj = _categorize_task(task_cls)
    holders = _Holders(task)
    rim = TABLE + 2 * task._BASKET_HALF_Z
    # The scripted drop height is basket centre + 0.18 m.
    _assert_guard(
        task, holders, obj.entity,
        rest_pos=[0.0, 0.15, TABLE + 0.04],
        float_pos=[0.0, 0.15, rim + 0.10],
        carry_pos=[0.0, 0.15, TABLE + task._BASKET_HALF_Z + 0.18],
        ok_fn=task.check_success,
    )


def test_categorize_cooperative_avatar_side_object_must_rest_too():
    task, _obj = _categorize_task(CategorizeObjectsContribution)
    avatar_obj = SimpleNamespace(entity=_Ent((0.25, 0.15, TABLE + 0.04), half=0.03))
    task._avatar_sort_targets = [(avatar_obj, "avatar", 2, 1)]
    holders = _Holders(task)
    assert task.check_success() is True
    holders.human_holds(avatar_obj.entity)
    assert task.check_success() is False
    row = task._basket_metrics()["basket_objects"][1]
    assert row["side"] == "avatar" and row["resting_success"] is False


# ----------------------------------------------------------------------
# Dual shoes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", [PlaceDualShoes, PlacePairOfShoesContribution,
                                      PlacePairOfShoesCopresence])
def test_shoe_must_rest_in_the_box(task_cls):
    task = task_cls({})
    task.TABLE_TOP_Z = TABLE
    task.plan_success = True
    task.avatar_collided = False
    task.avatar_collision_checker = None
    task._box_rim_z = TABLE + 0.10
    task.shoebox = SimpleNamespace(entity=_Ent([0.3, 0.0, TABLE]))
    half = np.array([0.05, 0.12, 0.05])
    shoes = [SimpleNamespace(entity=_Ent([0.3, -0.05, TABLE + 0.05], half=half)),
             SimpleNamespace(entity=_Ent([0.3, 0.05, TABLE + 0.05], half=half))]
    task.shoes = shoes
    holders = _Holders(task)
    ok = DualShoesCheck(task)
    _assert_guard(
        task, holders, shoes[1].entity,
        rest_pos=[0.3, 0.05, TABLE + 0.05],
        float_pos=[0.3, 0.05, TABLE + 0.10 + 0.04 + 0.05 + 0.03],
        carry_pos=[0.3, 0.05, TABLE + 0.25],
        ok_fn=ok,
    )
    # A collar grasp holds the shoe ~8 cm from its bbox centre.
    holders.gripper_closed_at([0.3, 0.13, TABLE + 0.08])
    assert ok() is False
    holders.release_all()
    assert task._shoe_in_box_metrics()["shoe_b_resting"] is True


def DualShoesCheck(task):
    from envs.task_bases.place_dual_shoes import DualShoesTaskCore

    return DualShoesTaskCore.check_success.__get__(task)


# ----------------------------------------------------------------------
# Stack bowls three
# ----------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", [StackBowlsThree, StackThreeBowlsInterruption,
                                      StackThreeBowlsCopresence])
def test_stacked_bowl_must_rest_on_the_stack(task_cls):
    task = task_cls({})
    task._num_bowls = 3
    task._stacked_indices = [1, 2]
    task._bowl_height = 0.05
    bowls = [SimpleNamespace(entity=_Ent([0.0, 0.0, TABLE + 0.015 * i], half=0.06))
             for i in range(3)]
    task.bowls = bowls
    task._bowl_xy_half_bounds = lambda: np.array([0.06, 0.06])
    holders = _Holders(task)
    ok = StackBowlsThree.check_success.__get__(task)
    # The rim pinch sits ~one rim radius (6-8 cm) from the bowl's centre.
    _assert_guard(
        task, holders, bowls[2].entity,
        rest_pos=[0.0, 0.0, TABLE + 0.03],
        float_pos=[0.0, 0.0, TABLE + 0.25],
        carry_pos=[0.0, 0.0, TABLE + 0.08],
        ok_fn=ok,
    )
    holders.gripper_closed_at([0.07, 0.0, TABLE + 0.03 + 0.025])
    assert ok() is False


# ----------------------------------------------------------------------
# Oil bottle recovery
# ----------------------------------------------------------------------


def test_oil_bottle_must_rest_upright_in_the_zone():
    task = OilBottleRecovery({})
    task.TABLE_TOP_Z = TABLE
    task.plan_success = True
    task.avatar_collided = False
    task._safe_target_xy = np.array([0.36, 0.08])
    task._oil_upright_origin_z_on_table = TABLE + 0.005
    task.oil_entity = _Ent([0.36, 0.08, TABLE + 0.005], half=[0.04, 0.04, 0.12],
                           quat=_upright_q())
    task._get_entity_pose = lambda ent, default: Pose(ent.get_pos(), ent.get_quat())
    holders = _Holders(task)
    rest = [0.37, 0.07, TABLE + 0.005 - 0.006]   # measured expert end state
    _assert_guard(
        task, holders, task.oil_entity,
        rest_pos=rest,
        float_pos=[0.37, 0.07, TABLE + 0.05],
        carry_pos=[0.37, 0.07, TABLE + 0.13],
        ok_fn=task.check_success,
    )
    assert task._recovery_metrics()["target_resting"] is True


def test_oil_take_action_no_longer_latches_success(monkeypatch):
    from envs.base_task import BaseTask

    task = OilBottleRecovery({"eval_mode": True})
    task.avatar = None
    monkeypatch.setattr(BaseTask, "take_action", lambda self, a, action_type="qpos": "obs")
    task.check_success = lambda: True
    assert task.take_action(np.zeros(8)) == "obs"
    assert task.eval_success is False


# ----------------------------------------------------------------------
# Take from human (safety): knife in the box
# ----------------------------------------------------------------------


def _knife_task():
    task = TakeFromHumanSafety({})
    task.plan_success = True
    task.TABLE_TOP_Z = TABLE
    task.KNIFE_BOX_XY = (0.50, 0.25)
    task._handle_collider = _Ent((0.54, 0.24, TABLE + 0.01 + 0.035))
    return task


def test_knife_must_rest_in_the_box():
    task = _knife_task()
    holders = _Holders(task)
    bottom = TABLE + 0.01
    _assert_guard(
        task, holders, task._handle_collider,
        rest_pos=[0.54, 0.24, bottom + 0.035],
        float_pos=[0.54, 0.24, bottom + 0.10],
        carry_pos=[0.54, 0.24, bottom + 0.17],   # scripted drop hover
        ok_fn=task.check_success,
    )


def test_knife_box_footprint_is_per_axis():
    task = _knife_task()
    bottom = TABLE + 0.01
    task._handle_collider.set_pos([0.50 + 0.14, 0.25, bottom + 0.03])
    assert task.check_success() is True          # long axis: half 0.15
    task._handle_collider.set_pos([0.50, 0.25 + 0.10, bottom + 0.03])
    assert task.check_success() is False         # short axis: half 0.08
    task._handle_collider.set_pos([0.50, 0.25 + 0.07, bottom + 0.03])
    assert task.check_success() is True


# ----------------------------------------------------------------------
# The N-step rule
# ----------------------------------------------------------------------


def test_success_hold_steps_default_and_override():
    assert DEFAULT_SUCCESS_HOLD_STEPS == 10
    assert success_hold_steps({}) == 10
    assert success_hold_steps(None) == 10
    assert success_hold_steps({"eval_success_hold_steps": 3}) == 3
    assert success_hold_steps({"eval_success_hold_steps": 0}) == 1


class _HoldTask:
    """take_action advances a scripted success trace."""

    def __init__(self, trace):
        self.trace = list(trace)
        self.actions = []
        arm = SimpleNamespace(get_arm_qpos=lambda: np.zeros(7), gripper_val=1.0)
        self.robot = SimpleNamespace(left_arm=None, get_arm=lambda _t: arm)
        self.i = -1

    def take_action(self, action, action_type="qpos"):
        assert action_type == "qpos_abs"
        self.actions.append(np.asarray(action))
        self.i += 1

    def evaluate(self):
        return {"success": bool(self.trace[min(self.i, len(self.trace) - 1)])}


def test_expert_post_play_hold():
    from scripts.eval_expert_set import post_play_success_hold

    held = post_play_success_hold(_HoldTask([True] * 12), {})
    assert held["success_hold_ok"] is True and held["success_hold_steps"] == 10
    assert len(held["success_hold_trace"]) == 10
    # An object rolling out of the goal on the 6th held step.
    rolled = post_play_success_hold(_HoldTask([True] * 5 + [False] * 10), {})
    assert rolled["success_hold_ok"] is False and rolled["success_hold_streak"] == 5
    # Measurement mode holds longer; the verdict still uses the first N.
    long = post_play_success_hold(
        _HoldTask([True] * 20 + [False] * 10), {"eval_success_hold_measure_steps": 30})
    assert long["success_hold_ok"] is True
    assert long["success_hold_streak"] == 20 and len(long["success_hold_trace"]) == 30
    task = _HoldTask([True] * 3)
    post_play_success_hold(task, {"eval_success_hold_steps": 3})
    np.testing.assert_allclose(task.actions[0], np.r_[np.zeros(7), 1.0])


# ----------------------------------------------------------------------
# Neutral minor fixes
# ----------------------------------------------------------------------


def test_dump_bin_neutral_knocked_prop_fails_the_episode(monkeypatch):
    from envs.task_bases.dump_bin import DumpBin
    from envs.tasks.dump_bin_copresence import DumpBinCopresence

    monkeypatch.setattr(DumpBin, "check_success", lambda _self: True)
    task = DumpBinCopresence({})
    task.avatar_collision_checker = None
    task.avatar_collided = False
    task._neutral_prop_disturbed = None
    assert task.check_success() is True
    task._neutral_prop_disturbed = {"prop": "can", "tilt_deg": 80.0}
    assert task.check_success() is False


def test_categorize_neutral_reports_the_neutral_controller_metrics(monkeypatch):
    from envs.base_task import BaseTask

    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": True})
    task, _obj = _categorize_task(CategorizeObjectsCopresence)
    task.avatar_collision_log = [{"pairs": [("hand", "neutral_obj:can")]}]
    task._neutral_eval_avatar_failed = True
    task.neutral_layout_metrics = lambda: {}
    metrics = task.evaluate()
    assert metrics["neutral_object_collided"] is True
    assert metrics["neutral_eval_avatar_failed"] is True
    assert metrics["neutral_robot_only_grading"] is True
