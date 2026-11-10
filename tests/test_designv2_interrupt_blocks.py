"""Interrupt contract for the block-ranking interrupt tasks.

Scene-free: fakes stand in for the robot, the avatar and the blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.tasks._block_ranking_interrupt_facade import BlockRankingInterruptFacade
from envs.tasks.rank_blocks_by_color_interruption import RankBlocksByColorInterruption
from envs.tasks.rank_blocks_by_size_interruption import RankBlocksBySizeInterruption


TASKS = (RankBlocksByColorInterruption, RankBlocksBySizeInterruption)


class _Block:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos.copy()


class _Arm:
    def __init__(self, tcp, gripper=1.0):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


class _Robot:
    left_arm = None

    def __init__(self, arm):
        self.arm = arm

    def get_arm(self, tag):
        return self.arm


class _AvatarRobot:
    def __init__(self):
        self.global_trans = np.array([0.0, 0.55, -0.38])

    def get_palm_center(self, hand_id):
        return self.global_trans + np.array([0.0, 0.0, 2.0])


class _Avatar:
    def __init__(self):
        self.robot = _AvatarRobot()
        self.frame_ratio = None
        self.motion_modules = {}
        self.resets = []

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.resets.append(self.robot.global_trans.copy())

    def spare(self):
        return True


def _task(task_cls, tcp=(0.0, -0.35, 1.10), gripper=1.0):
    task = task_cls({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    task.robot = _Robot(_Arm(tcp, gripper))
    task.avatar = _Avatar()
    task._resolve_inspect_motion = lambda: None
    task._interaction_policy_step = 0
    keys = task.block_keys()
    task.blocks = {
        keys[0]: _Block([0.12, 0.02, 0.80]),
        keys[1]: _Block([-0.12, 0.05, 0.80]),
        keys[2]: _Block([0.00, -0.08, 0.80]),
    }
    task.target_slots = {
        keys[0]: np.array([0.9, 0.9, 0.8]),
        keys[1]: np.array([-0.9, 0.9, 0.8]),
        keys[2]: np.array([0.00, -0.08, 0.80]),  # already in its slot
    }
    task.block_colors = dict(zip(keys, ("red", "green", "blue")))
    task.interaction_controller.dryrun_capture_palm = lambda _t: np.zeros(3)
    task.interaction_controller.after_base_reset_observation(task, observation=None)
    return task


@pytest.mark.parametrize("task_cls", TASKS)
def test_placed_and_grasped_blocks_are_not_candidates(task_cls):
    task = _task(task_cls)
    keys = task.block_keys()
    controller = task.interaction_controller
    assert set(controller.eligible_blocks(task)) == {keys[0], keys[1]}
    # Gripper closed around block 0 at the table: held, so excluded.
    task.robot.arm.tcp = np.array([0.12, 0.02, 0.82])
    task.robot.arm.gripper_val = 0.2
    assert set(controller.eligible_blocks(task)) == {keys[1]}


@pytest.mark.parametrize("task_cls", TASKS)
def test_no_trigger_while_robot_carries_a_block(task_cls):
    task = _task(task_cls)
    keys = task.block_keys()
    # The hover/dwell gate acts on the reset-chosen block.
    task.interaction_controller.interrupted_key = keys[1]
    dwell = task.interaction_controller.hover_dwell_params(task)
    for _ in range(25):
        task.advance_interaction_policy_step()
    # Carry block 0 (lifted, gripper closed) right above block 1.
    task.blocks[keys[0]].pos = np.array([-0.12, 0.05, 0.95])
    task.robot.arm.tcp = np.array([-0.12, 0.05, 0.97])
    task.robot.arm.gripper_val = 0.1
    task.advance_interaction_policy_step()
    assert not task._eval_interrupt_fired
    assert task._interrupt.trigger.block_reason == "robot_holding_object"
    # Released and approaching block 1 with an open gripper: fires.
    task.blocks[keys[0]].pos = np.array([0.30, 0.10, 0.80])
    task.robot.arm.gripper_val = 1.0
    for _ in range(dwell["steps"] if dwell else 1):
        task.advance_interaction_policy_step()
    assert task._eval_interrupt_fired
    assert task._interrupt.trigger.trigger_target == keys[1]


def _hover(task, key, dz=0.16, dxy=(0.0, 0.0)):
    pos = task.blocks[key].pos
    task.robot.arm.tcp = np.array([pos[0] + dxy[0], pos[1] + dxy[1], pos[2] + dz])


@pytest.mark.parametrize("task_cls", TASKS)
def test_trigger_needs_a_hover_dwell_over_the_reset_chosen_block(task_cls):
    task = _task(task_cls)
    keys = task.block_keys()
    controller = task.interaction_controller
    controller.interrupted_key = keys[1]
    dwell = controller.hover_dwell_params(task)
    assert dwell == {"steps": 3, "xy_m": 0.06, "dz_min_m": 0.08, "dz_max_m": 0.20}
    for _ in range(25):
        task.advance_interaction_policy_step()
    # Hovering over another block (picking it first, as the demos do) never
    # fires, however long.
    _hover(task, keys[0])
    for _ in range(20):
        task.advance_interaction_policy_step()
    assert not task._eval_interrupt_fired
    # Passing near the chosen block (10 cm off in xy, or descending to grasp
    # beside it) does not count as hovering.
    _hover(task, keys[1], dxy=(0.10, 0.0))
    for _ in range(10):
        task.advance_interaction_policy_step()
    _hover(task, keys[1], dz=0.05)
    for _ in range(10):
        task.advance_interaction_policy_step()
    assert not task._eval_interrupt_fired
    assert task._interrupt.trigger.block_reason == "no_hover_dwell"
    # A two-step pass over it does not fire either; the run resets.
    _hover(task, keys[1])
    task.advance_interaction_policy_step()
    task.advance_interaction_policy_step()
    _hover(task, keys[0])
    task.advance_interaction_policy_step()
    assert not task._eval_interrupt_fired
    # Three consecutive hover steps fire on the chosen block.
    _hover(task, keys[1], dxy=(0.03, -0.02))
    for i in range(3):
        task.advance_interaction_policy_step()
        assert bool(task._eval_interrupt_fired) is (i == 2)
    m = task._interrupt.metrics()
    assert m["trigger_target"] == keys[1]
    assert m["hover_dwell"]["steps"] == 3 and m["hover_dwell_max_run"] == 3


@pytest.mark.parametrize("task_cls", TASKS)
def test_dwell_fires_on_the_demonstrated_hover_not_on_earlier_pick_passes(task_cls):
    # Geometry from the replayed demos (rgb 6, size 6; interrupt_check demo
    # probe): before the interrupt the robot picks 0-2 other blocks and passes
    # the chosen one at best 7 cm off in xy at 6 cm height (13 cm at 5 cm
    # typical); the intent hover sits 0.8-3.4 cm off in xy, 15.4-16.6 cm
    # above the block centre, for 11-16 policy steps.
    task = _task(task_cls)
    keys = task.block_keys()
    task.interaction_controller.interrupted_key = keys[1]
    for _ in range(25):
        task.advance_interaction_policy_step()
    for dxy, dz in (((0.07, 0.0), 0.063), ((0.13, 0.0), 0.053), ((0.0, 0.10), 0.12)):
        _hover(task, keys[1], dz=dz, dxy=dxy)
        for _ in range(15):
            task.advance_interaction_policy_step()
        assert not task._eval_interrupt_fired
    _hover(task, keys[1], dz=0.166, dxy=(0.034, 0.0))
    for _ in range(3):
        task.advance_interaction_policy_step()
    assert task._eval_interrupt_fired
    assert task._interrupt.trigger.trigger_target == keys[1]


def test_both_block_tasks_share_the_dwell_thresholds():
    for cls in TASKS:
        task = _task(cls)
        params = task.interaction_controller.eval_episode.trigger.params
        assert params.trigger_progress == 0.25 and params.commit_radius_m == 0.20
        assert task._interrupt.metrics()["hover_dwell"] == {
            "steps": 3, "xy_m": 0.06, "dz_min_m": 0.08, "dz_max_m": 0.20,
        }


@pytest.mark.parametrize("task_cls", TASKS)
def test_interrupt_tasks_do_not_override_policy_actions(task_cls):
    # The policy keeps control: no interrupt class substitutes or skips actions.
    for cls in (task_cls, BlockRankingInterruptFacade):
        for name in ("take_action", "eval_skip_policy_action",
                     "consume_eval_policy_reset_requested"):
            assert name not in vars(cls), (cls.__name__, name)
    assert task_cls.interrupt_expert_attempt is BlockRankingInterruptFacade.interrupt_expert_attempt


@pytest.mark.parametrize("task_cls", TASKS)
def test_full_state_expert_attempt_is_idempotent(task_cls):
    task = _task(task_cls)
    keys = task.block_keys()
    events = []
    task._prepare_block_ranking_script = lambda arm: events.append("prepare")
    task.FINAL_SETTLE_STEPS = 1
    task.step_sim = lambda: events.append("sim")

    def place(key, arm):
        events.append(("place", key))
        task.blocks[key].pos = np.asarray(task.target_slots[key], dtype=float).copy()
        return True

    task._place_block_with_recovery = place
    assert task.interrupt_expert_attempt() is True
    assert events == ["prepare", ("place", keys[0]), ("place", keys[1]), "sim"]
    events.clear()
    # Resume after a yield: nothing left to do, no second prepare.
    assert task.interrupt_expert_attempt() is True
    assert events == ["sim"]


@pytest.mark.parametrize("task_cls", TASKS)
def test_full_state_expert_retreat_opens_then_retreats(task_cls):
    task = _task(task_cls)
    events = []
    task.set_gripper = lambda val, arm, n: events.append(("gripper", val, n))
    task.interaction_controller.retreat_safe = lambda _t, arm: events.append(("retreat", arm))
    task.interrupt_expert_retreat()
    assert events == [("gripper", 1.0, 20), ("retreat", "right")]
