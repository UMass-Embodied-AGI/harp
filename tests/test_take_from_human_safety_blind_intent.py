"""take_from_human_safety blind ("no intent") expert: no waits, live knife target."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import envs.avatar.motions.pick_place_motion as ppm
from envs.base_task import BaseTask
from envs.utils import Pose
from envs.tasks.take_from_human_safety import TakeFromHumanSafety
from scripts.eval_expert_set import (
    BLIND_INTENT_TASKS,
    configure_parent_rollout_eval,
    run_with_eval_step_driver,
)

TASK = "take_from_human_safety"


class _Body:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.pos = np.asarray(pos, dtype=float)
        self.quat = np.asarray(quat, dtype=float)

    def get_pos(self):
        return self.pos.copy()

    def get_quat(self):
        return self.quat.copy()

    def set_pos(self, p):
        self.pos = np.asarray(p, dtype=float)

    def set_quat(self, q):
        self.quat = np.asarray(q, dtype=float)


class _AvatarRobot:
    def __init__(self, events):
        self.events = events

    def get_palm_center(self, hand_id):
        return np.array([-0.4, -0.5, 0.9])

    def detach_object(self, hand_id):
        self.events.append(("detach", hand_id))

    def attach_object_to_hand(self, hand_id, obj):
        self.events.append(("attach", hand_id))


class _Avatar:
    """Each started clip lasts ``left`` avatar steps."""

    def __init__(self, events, chop_left):
        self.left = chop_left
        self.robot = _AvatarRobot(events)
        self.motion_modules = {}

    def step(self):
        self.left = max(0, self.left - 1)

    def spare(self):
        return self.left == 0


class _FakeReach:
    def __init__(self, name, robot):
        self.name = name

    def start(self, **kw):
        _FakeReach.avatar.left = int(kw["approach_frames"])


def _fake_task(monkeypatch, *, chop_left, trigger, reach_frames=6):
    events = []
    task = TakeFromHumanSafety.__new__(TakeFromHumanSafety)
    task.config = {"eval_mode": True, "eval_avatar_pre_policy": False,
                   "eval_force_policy_ticks": True,
                   "eval_parent_rollout_tick_interval": 5}
    task.robot = None  # release gate inactive in this scene-free fake
    task.avatar = _Avatar(events, chop_left)
    task._handle_collider = _Body([-0.5, -0.3, 0.8])
    task._knife_visual = _Body([-0.5, -0.25, 0.8])
    task._knife_scale = 0.15
    task._lock_collider_quat = True
    task._locked_collider_quat = np.array([1.0, 0.0, 0.0, 0.0])
    task._knife_local_offset = np.zeros(3)
    task._knife_local_R = np.eye(3)
    task._handover_jitter = np.zeros(3)
    task._compute_handover_xyz = lambda: np.array([-0.1, -0.6, 1.0])
    task._interaction_sim_step = 0
    task._interaction_policy_step = 0
    task._interaction_controller = None
    task._eval_policy_step_count = 0
    task._eval_handover_done = False
    task._eval_handover_triggered_at = None
    task._eval_trigger_step = trigger
    task._blind_active = True
    task._blind_reach = None
    task._blind_handover_complete = False
    task.HANDOVER_APPROACH_FRAMES = reach_frames

    def base_step(self):
        self._interaction_sim_step += 1
        self.avatar.step()

    monkeypatch.setattr(BaseTask, "step_sim", base_step)
    _FakeReach.avatar = task.avatar
    monkeypatch.setattr(ppm, "PickPlaceMotion", _FakeReach)
    return task, events


def test_harness_configures_blind_rollout():
    assert TASK in BLIND_INTENT_TASKS
    cfg = {"action_substeps": 25}
    configure_parent_rollout_eval(cfg, TASK, "play_blind_once")
    assert cfg["eval_avatar_pre_policy"] is False
    assert cfg["eval_force_policy_ticks"] is True
    # One human tick per VLA policy step (action_substeps sim steps).
    assert cfg["eval_parent_rollout_tick_interval"] == 25


def test_blind_rollout_never_drives_or_waits_on_the_human():
    src = inspect.getsource(TakeFromHumanSafety.play_blind_once)
    src += inspect.getsource(TakeFromHumanSafety._blind_pick_handle_and_drop)
    for banned in ("_extend_arm_to_handover", "_attach_knife_to_avatar",
                   "play_animation", "spare()", "_eval_trigger_step",
                   "PRELOADED_KNIFE_SETTLE_STEPS", "SETTLE_AFTER_ATTACH_STEPS",
                   "_drive_avatar_until_spare", "while ", "for _ in"):
        assert banned not in src, banned
    # Knife target is re-read from the live collider at every plan.
    assert src.count("_slab_body_grip_xy()") == 3


def test_handover_reach_is_non_blocking_under_policy_driver(monkeypatch):
    # Chop ends after 12 sim steps; trigger at policy step 1 (5 sim steps).
    task, events = _fake_task(monkeypatch, chop_left=12, trigger=1)
    robot_steps = []

    def robot_script():
        for _ in range(60):
            task.step_sim()
            robot_steps.append(task._interaction_sim_step)
        return True

    assert run_with_eval_step_driver(task, robot_script, task.config)
    # Every robot step ran (the robot was never paused by the human).
    assert robot_steps == list(range(1, 61))
    # Chop ends at sim 12; first policy tick after that is policy step 3 (sim 15).
    assert task._eval_handover_triggered_at == 3
    assert task._blind_human_started_sim_step == 15
    # Reach (6 frames) then finalize on the following step.
    assert task._blind_handover_complete is True
    assert task._blind_human_ready_sim_step == 21
    assert events == [("detach", 1), ("attach", 1)]
    assert task._lock_collider_quat is True
    np.testing.assert_allclose(
        task._handle_collider.get_pos(),
        np.array([-0.4, -0.5, 0.9]) + np.array([1.0, 0.0, 0.0]) * 0.10,
    )


def test_trigger_after_chop_end_delays_handover(monkeypatch):
    task, _events = _fake_task(monkeypatch, chop_left=3, trigger=7)
    run_with_eval_step_driver(
        task, lambda: [task.step_sim() for _ in range(40)] and True, task.config)
    assert task._eval_handover_triggered_at == 7
    assert task._blind_human_started_sim_step == 35


def test_blind_metrics_only_for_blind_rollout():
    task = TakeFromHumanSafety({"eval_mode": True})
    task._blind_robot_interaction_sim_step = 900
    task._blind_human_ready_before_robot = False
    task._eval_trigger_step = 30
    m = task._blind_intent_metrics()
    for key in ("trigger_step", "human_started_sim_step",
                "robot_interaction_sim_step", "human_ready_before_robot"):
        assert key in m
    assert m["trigger_step"] == 30
    assert m["robot_interaction_sim_step"] == 900
    assert m["human_ready_before_robot"] is False


def test_play_blind_once_without_avatar_fails():
    task = TakeFromHumanSafety({"eval_mode": True})
    task.avatar = None
    task._step_with_sync = lambda n=1: None
    assert task.play_blind_once() is False


@pytest.mark.parametrize("ready", [True, False])
def test_blind_robot_never_detaches_the_knife_itself(ready):
    events = []
    task = TakeFromHumanSafety({"eval_mode": True})
    task.avatar = SimpleNamespace(
        robot=_AvatarRobot(events),
        play_return_to_idle=lambda frames: events.append(("idle", frames)),
    )
    task._handle_collider = _Body([0.1, -0.4, 1.0])
    task._blind_handover_complete = ready
    task._lock_collider_quat = True
    task._locked_collider_quat = np.array([1.0, 0.0, 0.0, 0.0])
    task.TABLE_TOP_Z = 0.765
    task._interaction_sim_step = 123
    arm = SimpleNamespace(tcp_offset=Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
                          get_ee_pose=lambda: np.array([0.1, -0.4, 1.0, 1, 0, 0, 0]))
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.open_gripper = lambda tag: None
    task.move_and_execute = lambda pose, tag: object()
    task.set_gripper = lambda *a, **k: None
    task._step_with_sync = lambda n=1: None
    task._move_cartesian = lambda *a, **k: None  # knife never rises
    task._eval_knife_released = False
    assert task._blind_pick_handle_and_drop("right") is False
    assert task._blind_fail_stage == "lift"
    assert task._blind_robot_interaction_sim_step == 123
    assert task._blind_human_ready_before_robot is ready
    # Only the eval grip gate (_eval_maybe_release_knife) lets go.
    assert events == [] and task._lock_collider_quat is True
    assert task._blind_human_released is False
