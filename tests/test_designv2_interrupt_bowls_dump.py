"""Interrupt contract for stack_bowls_three_interrupt and
dump_bin_interrupt (scene-free fakes)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from envs.tasks.dump_bin_interruption import DumpBinInterruption
from envs.tasks.stack_three_bowls_interruption import StackThreeBowlsInterruption


class _Entity:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos.copy()


class _Arm:
    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = 1.0

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0, 0, 0]])


class _AvatarRobot:
    def __init__(self, root):
        self.global_trans = np.asarray(root, dtype=float)
        self.reaching = False

    def get_palm_center(self, _hand):
        off = [0.0, -0.4, 1.18] if self.reaching else [0.0, 0.0, 0.5]
        return self.global_trans + np.asarray(off)


class _Avatar:
    def __init__(self, root):
        self.robot = _AvatarRobot(root)
        self.resets = []
        self.busy = False
        self.motion_modules = {}
        self.played = []

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.resets.append(self.robot.global_trans.copy())

    def spare(self):
        return not self.busy


def _dump_task(cubes, tcp):
    task = DumpBinInterruption({"eval_mode": True, "hard": True})
    task.big_bin_pose = SimpleNamespace(p=np.array([0.5, -0.3, 0.0]))
    task._big_bin_rim_z = 0.24
    task._big_bin_xy_half = np.array([0.14, 0.20])
    task.objects = [_Entity(p) for p in cubes]
    task.num_objects = len(cubes)
    task._dump_object_center = lambda c: c.get_pos()
    task.TABLE_TOP_Z = 0.765
    arm = _Arm(tcp)
    task.robot = SimpleNamespace(get_arm=lambda _t: arm)
    task.avatar = _Avatar(task.avatar_init_pos)
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.15, 1.0])

    def play(**kw):
        task.avatar.played.append(kw)
        task.avatar.busy = True
        task.avatar.robot.reaching = True

    task._play_inspect_motion = play
    task._init_eval_interrupt_state()
    return task, arm


def test_dump_cube_under_home_gripper_never_triggers_for_still_robot():
    # reset_collisions.md: cubes within 0.11 m (xy) of the home TCP fired
    # force_dist at step 1.  Under the contract a still robot never triggers.
    home = np.array([0.02, 0.03, 0.95])
    task, _arm = _dump_task([[0.0, 0.0, 0.79], [0.12, -0.08, 0.79]], home)
    for _ in range(300):
        task._eval_step_avatar()
    m = task.interaction_controller.interaction_metrics(task)
    assert m["interrupt"]["fired"] is False
    assert m["interrupt"]["not_fired_reason"] == "robot_not_moved"
    assert task.avatar.resets == [] and task.avatar.played == []


def test_dump_trigger_targets_approached_cube_and_glides_with_warning():
    task, arm = _dump_task(
        [[-0.12, 0.0, 0.79], [0.12, -0.08, 0.79]], [0.3, -0.45, 1.05],
    )
    home = task.avatar.robot.global_trans.copy()
    for _ in range(20):
        task._eval_step_avatar()
    arm.tcp = np.array([0.12, -0.08, 0.95])  # hover above cube 1
    task._eval_step_avatar()
    assert task._eval_interrupt_fired and task._interrupted_idx == 1
    glide = task._interrupt.human.glide_steps
    assert glide >= 30
    for _ in range(glide):
        task._eval_step_avatar()
    assert len(task.avatar.played) == 1
    kw = task.avatar.played[0]
    assert kw["attach_obj"] is task.objects[1]
    assert kw["return_to_idle_after"] == task._interrupt.human.idle_return_frames
    jumps = np.linalg.norm(
        np.diff(np.vstack([home, task.avatar.resets]), axis=0), axis=1,
    )
    assert jumps.max() < 0.05  # no teleport
    task._eval_step_avatar()  # first clip tick: palm at the cube
    h = task._interrupt.human.metrics()
    assert h["hand_reach_step"] - h["trigger_step"] >= 30  # >= 1.5 s warning
    task.avatar.busy = False
    task.avatar.robot.reaching = False
    for _ in range(1 + task._interrupt.human.glide_out_steps):
        task._eval_step_avatar()
    assert task._interrupt.human.done
    np.testing.assert_allclose(task.avatar.robot.global_trans, home)


def test_dump_holding_gate_blocks_trigger_while_carrying():
    task, arm = _dump_task([[-0.12, 0.0, 0.79], [0.12, -0.08, 0.86]], [0.3, -0.45, 1.05])
    for _ in range(20):
        task._eval_step_avatar()
    # Cube 1 lifted in the closed gripper, carried right over cube 0.
    arm.gripper_val = 0.0
    arm.tcp = np.array([-0.10, 0.0, 0.92])
    task.objects[1].pos = np.array([-0.10, 0.0, 0.88])
    task._eval_step_avatar()
    assert task._eval_interrupt_fired is False
    assert task._interrupt.trigger.block_reason == "robot_holding_object"


def test_dump_expert_attempt_is_idempotent_over_binned_cubes():
    task = DumpBinInterruption({"eval_mode": True, "hard": True})
    task.big_bin_pose = SimpleNamespace(p=np.array([0.5, -0.3, 0.0]))
    task._big_bin_rim_z = 0.24
    task._big_bin_xy_half = np.array([0.14, 0.20])
    task.objects = [_Entity([0.5, -0.3, 0.1]), _Entity([0.0, 0.0, 0.79])]
    task._dump_object_center = lambda c: c.get_pos()
    task._interrupt_expert_started = True
    picks = []
    task._pick_and_drop_cube = lambda c, arm: picks.append(task.objects.index(c)) or True
    task._ensure_final_gripper_z = lambda arm: None
    assert task.interrupt_expert_attempt() is True
    assert picks == [1]


def test_bowls_expert_attempt_skips_stacked_bowls():
    task = StackThreeBowlsInterruption({"hard": True})
    task.INITIAL_SETTLE_STEPS = 0
    task.FINAL_SETTLE_STEPS = 0
    task._bowl_height = 0.05
    task._bowl_scale = 0.5
    task._bowl_extents = np.array([0.3, 0.1, 0.3])
    task.bowls = [
        SimpleNamespace(entity=_Entity([0.0, 0.0, 0.77])),
        SimpleNamespace(entity=_Entity([0.01, 0.0, 0.80])),  # already stacked
        SimpleNamespace(entity=_Entity([0.3, 0.1, 0.77])),
    ]
    task._stacked_indices = [1, 2]

    @contextmanager
    def suppress():
        yield

    task.suppress_recording = suppress
    task.step_sim = lambda: None
    stacks = []
    task._pick_and_stack_one = lambda idx, target: stacks.append((idx, target)) or True
    assert task.interrupt_expert_attempt() is True
    assert [i for i, _ in stacks] == [2]
    assert stacks[0][1][2] == 0.80 + 0.05  # on top of the stacked bowl
