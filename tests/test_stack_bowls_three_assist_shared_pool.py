"""Scene-free contracts for the stack_bowls_three_assist shared bowl pool (v2).

Both candidate bowls sit in one shared band.  The human picks one uniformly
at random and goes for it no matter what; the blind robot commits at t=0 to
the candidate nearest its gripper and never reads the human's choice.  No
conflict handling on either side.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import envs.tasks.stack_three_bowls_contribution as assist_module
from envs.interaction.stack_bowls_three import StackBowlsThreeAssistController
from envs.task_bases.stack_bowls_motion import _BowlMotionRejected
from envs.task_bases.stack_bowls_three import _MIN_SEP_M
from envs.tasks.stack_three_bowls_contribution import StackThreeBowlsContribution

TABLE_Z = 0.765


class _Entity:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self.pos.copy()


class _Actor:
    def __init__(self, pos):
        self.entity = _Entity(pos)


class _Arm:
    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, dtype=np.float64)

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0.0, 0.0, 0.0]])


class _Avatar:
    def __init__(self):
        self.calls = []

    def pick_and_place(self, **kwargs):
        self.calls.append(kwargs)

    def spare(self):
        return True


def _task(positions=StackThreeBowlsContribution.SHARED_FIXED_XY, *, tcp=(0.0, -0.19, 0.842)):
    task = StackThreeBowlsContribution({"hard": len(positions) > 3})
    task.TABLE_TOP_Z = TABLE_Z
    task._bowl_height = 0.05
    task._bowl_scale = 1.0
    task._bowl_extents = np.array([0.114, 0.05, 0.114])
    task.bowls = [_Actor([x, y, TABLE_Z + 0.001]) for x, y in positions]
    task._num_bowls = len(positions)
    StackBowlsThreeAssistController.allocate(task)
    arm = _Arm(tcp)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm, arm=arm)
    task.avatar = _Avatar()
    task.avatar_collided = False
    task.avatar_collision_log = []
    return task


def _patch_loading(monkeypatch):
    def load(_scene, pose, _object_id, **_kwargs):
        return _Actor(pose.p)

    monkeypatch.setattr(assist_module, "load_object", load)
    monkeypatch.setattr(assist_module, "load_fixed_collision_bowl", load)
    monkeypatch.setattr(assist_module, "_read_scale", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(assist_module, "_read_extents",
                        lambda *_a, **_k: np.array([0.114, 0.05, 0.114]))


# ----------------------------------------------------------------------
# Layout + random human choice
# ----------------------------------------------------------------------


@pytest.mark.parametrize("hard", (False, True))
def test_human_choice_is_uniform_over_shared_candidates(hard, monkeypatch):
    _patch_loading(monkeypatch)
    counts = {}
    for seed in range(400):
        task = StackThreeBowlsContribution({"hard": hard})
        task.scene = object()
        task.TABLE_TOP_Z = TABLE_Z
        np.random.seed(seed)
        task.load_actors()
        human = task._human_target
        assert human in task._stacked_indices
        assert task._shared_pool == {"human_target": human, "robot_target": None}
        counts[human] = counts.get(human, 0) + 1
    n = len(counts)
    assert n == (3 if hard else 2)
    for c in counts.values():
        assert abs(c / 400 - 1.0 / n) < 0.08


def _check_layout(task, pts, band_x, band_y):
    bx, by = pts[0]
    assert task.BASE_X[0] <= bx <= task.BASE_X[1]
    assert task.BASE_Y[0] <= by <= task.BASE_Y[1]
    for x, y in pts[1:]:
        assert band_x[0] <= x <= band_x[1]
        assert band_y[0] <= y <= band_y[1]
        assert np.hypot(x - task.ROBOT_MOUNT_XY[0], y - task.ROBOT_MOUNT_XY[1]) <= task.ROBOT_REACH_M + 1e-9
        if len(pts) == 3:
            assert np.hypot(x - task.ROBOT_HOME_TCP_XY[0],
                            y - task.ROBOT_HOME_TCP_XY[1]) >= task.CANDIDATE_HOME_CLEARANCE_M
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= _MIN_SEP_M - 1e-9


def test_easy_default_layout_puts_both_candidates_in_the_shared_band():
    task = StackThreeBowlsContribution({})
    pts = task._sample_assist_xy(np.random.RandomState(0))
    assert len(pts) == 3
    _check_layout(task, pts, task.SHARED_X, task.SHARED_Y)
    assert pts[1][1] == pts[2][1] and pts[1][0] == -pts[2][0]


def test_easy_random_layout_keeps_candidates_reachable_and_clear_of_home_hand():
    task = StackThreeBowlsContribution({})
    task.simplified_mode_enabled = lambda: False
    fixed = [tuple(p) for p in task.SHARED_FIXED_XY]
    fallbacks = 0
    for seed in range(50):
        pts = task._sample_assist_xy(np.random.RandomState(seed))
        if [tuple(p) for p in pts] == fixed:
            fallbacks += 1
            continue
        _check_layout(task, pts, task.SHARED_X, task.SHARED_Y)
    assert fallbacks <= 5


def test_hard_layout_samples_three_candidates_in_the_shared_band():
    task = StackThreeBowlsContribution({"hard": True})
    fixed = [tuple(p) for p in task.HARD_FALLBACK_XY]
    _check_layout(task, fixed, task.HARD_SHARED_X, (-0.22, -0.08))
    fallbacks = 0
    for seed in range(50):
        pts = task._sample_assist_xy(np.random.RandomState(seed))
        assert len(pts) == 4
        if [tuple(p) for p in pts] == fixed:
            fallbacks += 1
            continue
        _check_layout(task, pts, task.HARD_SHARED_X, task.HARD_SHARED_Y)
    assert fallbacks <= 5


# ----------------------------------------------------------------------
# Human: goes for its bowl no matter what
# ----------------------------------------------------------------------


def test_human_goes_for_its_bowl_even_if_the_robot_holds_it():
    task = _task()
    task._human_target = 1
    task.robot.arm.tcp = task.bowls[1].entity.pos + [0.03, 0.0, 0.05]
    task.bowls[1].entity.pos[2] += 0.10            # robot already lifting it
    assert task._start_avatar_assist() is True
    assert task.avatar.calls[0]["attach_obj"] is task.bowls[1].entity
    assert task._human_target == 1


def test_eval_reset_starts_the_human_at_step_zero_without_delay_opt_in():
    task = _task()
    task.config["eval_mode"] = True
    task._human_target = 2
    task._eval_at_reset()
    assert task._eval_assist_trigger_step == 0 and task._eval_assist_fired is True
    assert task.avatar.calls[0]["attach_obj"] is task.bowls[2].entity


def test_controller_has_no_conflict_handling_hooks():
    for name in ("before_sim_step", "free_for_human", "robot_claims", "human_claims",
                 "_retarget", "_withdraw", "_give_up"):
        assert not hasattr(StackBowlsThreeAssistController, name)
    for name in ("_robot_watch_tick", "_retreat_after_yield", "_park_robot",
                 "_wait_for_stack_clear", "_wait_for_hand_clear"):
        assert not hasattr(StackThreeBowlsContribution, name)
    assert "step_sim" not in StackThreeBowlsContribution.__dict__
    assert "set_gripper" not in StackThreeBowlsContribution.__dict__


# ----------------------------------------------------------------------
# Experts
# ----------------------------------------------------------------------


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _rollout_task(**kw):
    task = _task(**kw)
    task.INITIAL_SETTLE_STEPS = 0
    task.FINAL_SETTLE_STEPS = 0
    task.suppress_recording = lambda: _NullCtx()
    task.step_sim = lambda: None
    calls = []
    task._pick_and_stack_one = lambda idx, pos: calls.append(idx) or True
    return task, calls


def test_blind_commits_to_the_nearest_bowl_regardless_of_the_human():
    for human in (1, 2):
        task, calls = _rollout_task(tcp=(0.05, -0.30, 1.0))
        task._human_target = human
        task._shared_pool["human_target"] = human
        assert task.play_blind_once() is True
        assert calls == [1]
        pool = task._assist_controller().shared_pool_metrics(task)
        assert pool == {"human_target": human, "robot_target": 1, "same_target": human == 1}


def test_blind_does_not_retry_or_take_leftovers():
    task, calls = _rollout_task(tcp=(-0.05, -0.30, 1.0))
    task._pick_and_stack_one = lambda idx, pos: calls.append(idx) or False
    assert task.play_blind_once() is False
    assert calls == [2]


def test_full_state_takes_the_other_bowl():
    for human, robot in ((1, 2), (2, 1)):
        task, calls = _rollout_task()
        task._human_target = human
        task._shared_pool["human_target"] = human
        assert task.play_once() is True
        assert calls == [robot]
        assert task.avatar.calls[0]["attach_obj"] is task.bowls[human].entity
        assert task._shared_pool["robot_target"] == robot


def test_left_candidate_carry_falls_back_to_segments_without_waiting(monkeypatch):
    task = _task()
    labels = []

    def move(self, target, arm_tag, label, tiny=False):
        labels.append((label, tuple(np.round(target.p, 3))))
        if len(labels) == 1:
            self.plan_success = False
            raise _BowlMotionRejected("carry: path, collision or FK rejected")
        return "ok"

    monkeypatch.setattr(assist_module.StackBowlsThree, "_bowl_move", move)
    task.plan_success = True
    target = assist_module.Pose([0.0, -0.1, 1.0], [1.0, 0.0, 0.0, 0.0])
    task.robot.arm.tcp = np.array([-0.2, -0.2, 1.0])
    assert task._bowl_move(target, "right", "carry") == "ok"
    assert [l for l, _ in labels] == ["carry"] * 3
    assert labels[1][1] == (-0.1, -0.15, 1.0)
    assert task.plan_success is True
