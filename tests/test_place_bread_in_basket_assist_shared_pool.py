"""Scene-free contracts for the simple shared-pool (50-50) bread assist.

* the human goes for the loaf drawn at reset, unconditionally (no checks,
  no retarget), from one stance chosen at reset;
* the blind expert picks one loaf by a human-independent rule at t=0 and
  commits, never reading the human's choice;
* the full-state expert takes the loaf the human did not take;
* success needs no robot-avatar collision; ``shared_pool`` metrics.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from envs.tasks.place_bread_in_basket_contribution import PlaceBreadInBasketContribution

TABLE_Z = 0.75
REST_Z = 0.77  # mesh origin of a resting loaf (centre offset is zero below)


class _Entity:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


class _Actor:
    def __init__(self, pos):
        self.entity = _Entity(pos)


class _Arm:
    def __init__(self):
        self.gripper_val = 1.0


class _Robot:
    def __init__(self, tcp=(0.0, -0.5, 1.2)):
        self.tcp = np.asarray(tcp, dtype=np.float64)
        self.arm = _Arm()

    def get_ee_pose(self, arm_tag):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]

    def get_arm(self, arm_tag):
        return self.arm


class _Motion:
    def __init__(self, attach_obj, hand_id=1):
        self.frames = list(range(10))
        self.at_frame = 0
        self.attach_frame = 5
        self.detach_frame = 8
        self._attach_obj = attach_obj
        self._hand_id = hand_id
        self.stepped = 0

    def step(self, skip_avatar_animation=False):
        self.stepped += 1
        self.at_frame += 1


class _AvatarRobot:
    action_state = "pick_and_place"
    skin = None  # shoulder lookup falls back to "reachable"


class _Avatar:
    def __init__(self, spare=True):
        self.calls = []
        self.motion = None
        self.motion_modules = {}
        self.robot = _AvatarRobot()
        self._spare = spare

    def pick_and_place(self, **kwargs):
        self.calls.append(kwargs)
        hand = kwargs.get("hand_id")
        self.motion = _Motion(kwargs["attach_obj"], 1 if hand is None else hand)
        self.motion_modules["pick_and_place"] = self.motion
        return self.motion

    def spare(self):
        return self._spare


def _task(n=2, **config):
    task = PlaceBreadInBasketContribution({"hard": n == 3, **config})
    task.TABLE_TOP_Z = TABLE_Z
    task.basket = _Actor([0.0, 0.0, TABLE_Z])
    xs = [0.25, -0.25, 0.40][:n]
    task.breads = [_Actor([x, 0.0, REST_Z]) for x in xs]
    task._bread_center_mesh = np.zeros(3)
    task._bread_scale = 1.0
    task._basket_rim_z = 0.82
    task._refresh_basket_geom = lambda: None
    task.robot = _Robot()
    task.avatar = _Avatar()
    task.INITIAL_SETTLE_STEPS = 1
    task.AVATAR_FINISH_SETTLE_STEPS = 1
    task._avatar_in_planner = lambda arm: _null()
    return task


@contextmanager
def _null():
    yield


def _plan(task, idx):
    controller = task.interaction_controller
    controller.avatar_bread_idx = idx
    controller.robot_bread_indices = tuple(
        i for i in range(len(task.breads)) if i != idx
    )
    controller._publish_allocation(task)
    return controller


def _move_into_basket(task, idx):
    task.breads[idx].entity.set_pos([0.0, 0.0, 0.80])


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_defaults_require_collision_free_success_and_no_start_delay():
    task = PlaceBreadInBasketContribution({})
    assert task.config["track_avatar_collision"] is True
    assert task.config["success_require_no_avatar_collision"] is True
    assert task.AVATAR_START_DELAY_STEPS == 0
    assert task._num_breads == 2


def test_human_goes_for_the_reset_choice_unconditionally():
    task = _task()
    controller = _plan(task, 1)
    # Even with the robot gripper closed on that loaf, the human goes for it.
    task.robot.tcp = task.breads[1].entity.get_pos()
    task.robot.arm.gripper_val = 0.3
    task._start_avatar_assist(add_start_delay=False)
    assert controller.human_target == 1
    assert len(task.avatar.calls) == 1
    assert task.avatar.calls[0]["attach_obj"] is task.breads[1].entity
    motion = task.avatar.motion
    motion.step()
    assert motion.stepped == 1  # no guard wrapping the motion
    assert len(task.avatar.calls) == 1


def test_eval_reset_fires_the_human_at_reset_by_default():
    task = _task(eval_mode=True)
    _plan(task, 0)
    task.interaction_controller.eval_reset(task)
    assert task._eval_assist_fired is True
    assert len(task.avatar.calls) == 1


def _forbid_allocation_reads(task, monkeypatch):
    controller = task.interaction_controller

    def boom(*_a, **_k):
        raise AssertionError("blind expert read the allocation")

    monkeypatch.setattr(controller, "scripted_robot_indices", boom)
    monkeypatch.setattr(controller, "_adopt_allocation", boom)


@pytest.mark.parametrize("human", [0, 1])
def test_blind_expert_commits_to_nearest_gripper_loaf(monkeypatch, human):
    task = _task()
    _plan(task, human)
    task.robot.tcp = np.array([0.20, 0.0, 1.0])  # nearest loaf: index 0
    task.step_sim = lambda: None
    picks = []

    def pick(bread, arm, label):
        picks.append(task.breads.index(bread))
        return False  # e.g. the human took it: honest failure, no retry

    task._pick_and_place_bread = pick
    _forbid_allocation_reads(task, monkeypatch)
    assert task.play_blind_once() is False
    assert picks == [0]
    metrics = task.interaction_controller.shared_pool_metrics()
    assert metrics["robot_target"] == 0


def test_full_state_takes_the_other_loaf_after_the_human_schedule():
    task = _task()
    _plan(task, 1)
    sims = {"n": 0}
    task.step_sim = lambda: sims.__setitem__("n", sims["n"] + 1)
    task.avatar._spare = False
    order = []

    def pick(bread, arm, label):
        order.append((sims["n"], task.breads.index(bread)))
        return True

    task._pick_and_place_bread = pick
    task._AVATAR_SPARE_MAX_STEPS = 3
    task.AVATAR_FINISH_SETTLE_STEPS = 0
    assert task.play_once() is True
    # Privileged timing: starts after the human's 10 scheduled frames.
    assert order == [(11, 0)]
    metrics = task.interaction_controller.shared_pool_metrics()
    assert metrics == {"human_target": 1, "robot_target": 0, "same_target": False}


def test_shared_pool_metrics_and_collision_gate(monkeypatch):
    from envs.base_task import BaseTask

    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"success": False})
    task = _task()
    controller = _plan(task, 0)
    task.plan_success = True
    for b in task.breads:
        b.entity.set_pos([0.0, 0.0, 0.80])
    controller.human_target = 0
    controller.note_robot_target(0)
    controller.note_robot_target(1)  # only the first counts
    task.avatar_collision_checker = object()
    task.avatar_collided = False
    metrics = task.evaluate()
    assert metrics["shared_pool"] == {
        "human_target": 0, "robot_target": 0, "same_target": True,
    }
    assert metrics["success"] is True
    task.avatar_collided = True
    assert task.evaluate()["success"] is False
    controller.on_episode_seeded(task, 1)
    assert controller.shared_pool_metrics()["robot_target"] is None


class _Skin:
    def __init__(self, robot):
        self.robot = robot

    def get_global_translation(self, bone):
        lateral = 0.17 if bone == "LeftArm" else -0.17
        return [self.robot.global_trans + np.array([lateral, 0.0, 1.38])]


class _StanceRobot(_AvatarRobot):
    def __init__(self):
        self.global_trans = np.array([0.0, 0.85, -0.18])
        self.global_rot = np.eye(3)
        self.skin = _Skin(self)


class _StanceAvatar(_Avatar):
    def __init__(self):
        super().__init__()
        self.robot = _StanceRobot()
        self.resets = []

    def reset(self, pos, rot):
        self.resets.append((np.asarray(pos).copy(), np.asarray(rot).copy()))
        self.robot.global_trans = np.asarray(pos, dtype=np.float64)



def test_pool_stance_is_set_at_reset_and_reused():
    task = _task()
    controller = _plan(task, 0)
    task.avatar = _StanceAvatar()
    controller.on_base_reset_ready(task)
    assert controller.pool_hand in (0, 1)
    assert len(task.avatar.resets) == 1
    body = task.avatar.resets[0][0]
    assert body[1] == pytest.approx(controller.pool_stance_y)
    lateral = 0.17 if controller.pool_hand == 0 else -0.17
    shoulder = body + np.array([lateral, 0.0, 1.38])
    human = controller.avatar_bread_idx
    targets = [task.breads[human].entity.get_pos(), controller._place_pos(task)]
    worst = max(np.linalg.norm(t - shoulder) for t in targets)
    for dx in (-0.05, 0.05):
        other = shoulder + np.array([dx, 0.0, 0.0])
        assert worst <= max(np.linalg.norm(t - other) for t in targets) + 1e-9
    task._start_avatar_assist(add_start_delay=False)
    call = task.avatar.calls[0]
    assert call["natural"] is False and call["hand_id"] == controller.pool_hand
    assert len(task.avatar.resets) == 1  # no re-stance at start


def test_layouts_put_basket_between_loaves_beside_the_robot():
    from envs.tasks.place_bread_in_basket_contribution import (
        _POOL_LOAF_Y_MAX, _ROBOT_BASE_XY, _ROBOT_HOME_HAND_XY,
        _ROBOT_HOME_KEEPOUT_M, _ROBOT_REACH_ANNULUS,
    )

    fallbacks = 0
    for seed in range(30):
        task = PlaceBreadInBasketContribution({})
        controller = task.interaction_controller
        positions = controller.plan_positions(task, np.random.RandomState(seed))
        assert len(positions) == 3
        basket, left, right = positions
        fallbacks += positions == list(controller.fallback_positions[:3])
        assert left[0] + 0.15 - 1e-9 <= basket[0] <= right[0] - 0.15 + 1e-9
        for x, y in positions:
            r = float(np.hypot(x - _ROBOT_BASE_XY[0], y - _ROBOT_BASE_XY[1]))
            assert _ROBOT_REACH_ANNULUS[0] - 1e-9 <= r <= _ROBOT_REACH_ANNULUS[1] + 1e-9
            assert float(np.hypot(x - _ROBOT_HOME_HAND_XY[0], y - _ROBOT_HOME_HAND_XY[1])) >= (
                _ROBOT_HOME_KEEPOUT_M - 1e-9
            )
        assert all(y <= _POOL_LOAF_Y_MAX + 1e-9 for _x, y in (left, right))
        assert controller._one_stance_reaches(positions)
    assert fallbacks <= 1


def test_fallback_layout_satisfies_every_pool_constraint():
    from envs.tasks.place_bread_in_basket_contribution import (
        _ASSIST_FALLBACK_POSITIONS,
    )

    task = PlaceBreadInBasketContribution({"hard": True})
    controller = task.interaction_controller
    assert controller._separated(list(_ASSIST_FALLBACK_POSITIONS[:3]))




def test_reset_draws_the_human_loaf_uniformly():
    # A fair per-episode coin from the controller's private
    # stream (episode seed + stream name), not the post-build global stream
    # (which is biased toward loaf 0).
    chosen = []
    for seed in range(400):
        np.random.seed(seed)
        expected_tail = (np.random.randint(2), np.random.rand())[1]
        np.random.seed(seed)
        task = _task()
        task._episode_seed = seed
        task._episode_rngs = {}
        controller = _plan(task, 1)
        controller.on_base_reset_ready(task)
        chosen.append(controller.avatar_bread_idx)
        assert task._avatar_bread_idx == chosen[-1]
        assert task._robot_bread_indices == [1 - chosen[-1]]
        # The legacy global draw is still consumed exactly once.
        assert np.random.rand() == expected_tail
    assert 170 <= chosen.count(0) <= 230


def test_above_basket_move_replans_within_bound(monkeypatch):
    from envs.base_task import BaseTask

    task = _task()
    sims = {"n": 0}
    task.step_sim = lambda: sims.__setitem__("n", sims["n"] + 1)
    outcomes = iter([None, None, "plan"])

    def fake_move(self, pose, arm_tag):
        result = next(outcomes)
        if result is None:
            self.plan_success = False
        return result

    monkeypatch.setattr(BaseTask, "move_to_pose", fake_move)
    task.plan_success = True
    with task._basket_replan():
        assert task.move_to_pose([0.0, 0.0, 1.0, 1, 0, 0, 0], "right") == "plan"
    assert sims["n"] == 2 * task._BASKET_REPLAN_EVERY
    # Not over the basket: no re-plan.
    outcomes = iter([None])
    task.plan_success = True
    with task._basket_replan():
        assert task.move_to_pose([0.3, 0.0, 1.0, 1, 0, 0, 0], "right") is None
