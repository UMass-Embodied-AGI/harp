"""Scene-free contracts for the stamp_documents blind ("no intent") expert."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from envs.interaction.eval_mode import IntentEvalModeController
from envs.tasks.feed_documents import FeedDocuments
from scripts.eval_expert_set import BLIND_INTENT_TASKS, policy_play_mode


class _Doc:
    def __init__(self, xy, z=0.8):
        self.pos = np.array([xy[0], xy[1], z], dtype=float)

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


def _bare_task(config=None, docs=None, tcp=(0.0, -0.5, 1.0)):
    task = object.__new__(FeedDocuments)
    task.config = dict(config or {})
    task.avatar = object()
    task._stamp_xy = np.array([0.10, 0.10])
    task._doc_half = (0.05, 0.05, 0.0125)
    task.TABLE_TOP_Z = 0.78
    task.document_actors = list(docs or [])
    task._stamp_mark_attached = [None] * len(task.document_actors)
    arm = SimpleNamespace(get_ee_pose=lambda: np.array([*tcp, 1, 0, 0, 0], dtype=float))
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task._eval_cycle = None
    task._eval_cycles_log = []
    return task


BLIND_CFG = {"eval_mode": True, "eval_avatar_pre_policy": False}
LEGACY_CFG = {"eval_mode": True, "stamp_eval_human": "robot_triggered"}


def test_controller_and_harness_dispatch():
    task = object.__new__(FeedDocuments)
    assert isinstance(task.create_interaction_controller(), IntentEvalModeController)
    assert "feed_documents" in BLIND_INTENT_TASKS
    assert policy_play_mode("feed_documents", FeedDocuments, "expert") == (
        "play_blind_once", None)
    assert policy_play_mode(
        "feed_documents", FeedDocuments, "expert_full_state") == (
        "play_full_eval_once", None)


def test_trigger_window_is_private_per_seed_and_inert_for_legacy_human():
    np.random.seed(0)
    legacy = _bare_task(LEGACY_CFG)
    state = np.random.get_state()
    legacy._eval_at_reset()
    assert legacy._eval_trigger_step is None

    lo, hi = FeedDocuments.EVAL_TRIGGER_STEP_MIN, FeedDocuments.EVAL_TRIGGER_STEP_MAX
    # Doc 0 is never stamped before the robot reaches standby in the demos.
    assert lo >= 26
    seen = set()
    for seed in range(40):
        for cfg in ({"eval_mode": True}, BLIND_CFG):  # VLA default == expert
            t = _bare_task(cfg)
            t._episode_seed = seed
            t._eval_at_reset()
            assert lo <= t._eval_trigger_step <= hi
            seen.add(t._eval_trigger_step)
        again = _bare_task({"eval_mode": True})
        again._episode_seed = seed
        again._eval_at_reset()
        assert again._eval_trigger_step == t._eval_trigger_step
    assert len(seen) > 10
    # The draw uses a private stream: the global RNG is untouched.
    assert np.array_equal(np.random.get_state()[1], state[1])

    t = _bare_task({**BLIND_CFG, "eval_trigger_step_min": 7, "eval_trigger_step_max": 7})
    t._eval_at_reset()
    assert t._eval_trigger_step == 7


def test_first_stamp_starts_at_trigger_not_before():
    t = _bare_task(BLIND_CFG)
    t._eval_trigger_step = 10
    started = []
    t._eval_start_stamp_cycle = lambda step: started.append(step)
    for step in range(1, 10):
        t._eval_at_step(step)
    assert started == []
    t._eval_at_step(10)
    assert started == [10]


def test_hooks_inert_for_the_legacy_robot_triggered_human():
    t = _bare_task(LEGACY_CFG)
    t._eval_trigger_step = 0
    t._eval_start_stamp_cycle = lambda step: (_ for _ in ()).throw(AssertionError)
    t._eval_at_step(50)


def test_second_stamp_needs_fresh_doc_in_zone_and_clear_robot():
    near_doc = _Doc((0.11, 0.09))
    far_doc = _Doc((-0.20, 0.10))
    t = _bare_task(BLIND_CFG, docs=[far_doc, near_doc], tcp=(0.10, 0.05, 0.85))
    t._eval_trigger_step = 0
    t._eval_cycles_log = [{"doc_idx": 0}]
    t._stamp_mark_attached = ["stamped", None]
    started = []
    t._eval_start_stamp_cycle = lambda step: started.append(step)
    assert t._eval_doc_in_zone() == 1
    t._eval_robot_clear_steps = 0
    t._eval_at_step(200)          # robot hand still at the zone
    assert started == []
    # Standby (0.20 m behind the zone): the human waits until the hand has
    # been clear for EVAL_ROBOT_CLEAR_STEPS policy steps.
    arm = SimpleNamespace(get_ee_pose=lambda: np.array([0.1, -0.1, 1.0, 1, 0, 0, 0]))
    t.robot = SimpleNamespace(get_arm=lambda tag: arm)
    need = FeedDocuments.EVAL_ROBOT_CLEAR_STEPS
    for k in range(need - 1):
        t._eval_at_step(201 + k)
    assert started == []
    t._eval_at_step(300)
    assert started == [300]
    # No document in the zone -> no stamp.
    near_doc.pos[:2] = (0.30, -0.20)
    started.clear()
    t._eval_at_step(302)
    assert started == []
    assert t._eval_doc_in_zone() is None


def test_second_stamp_eases_back_from_the_retreat_pose_without_blocking():
    near_doc = _Doc((0.11, 0.09))
    t = _bare_task(BLIND_CFG, docs=[_Doc((-0.20, 0.10)), near_doc],
                   tcp=(0.10, -0.10, 1.0))
    t._eval_trigger_step = 0
    t._eval_cycles_log = [{"doc_idx": 0}]
    t._stamp_mark_attached = ["stamped", None]
    t._eval_robot_clear_steps = FeedDocuments.EVAL_ROBOT_CLEAR_STEPS
    t._eval_avatar_at_retreat = True
    spare = [False]
    t.avatar = SimpleNamespace(spare=lambda: spare[0])
    eased, started = [], []
    t._eval_start_ease_to_stamp_start = lambda: eased.append(1) or True
    t._eval_start_stamp_cycle = lambda step: started.append(step)
    t._eval_at_step(400)
    assert eased == [1] and started == []
    t._eval_at_step(401)          # still easing
    assert started == []
    spare[0] = True
    t._eval_at_step(402)
    assert started == [402]


def test_blind_rollout_never_waits_on_the_human():
    src = inspect.getsource(FeedDocuments.play_blind_once)
    for banned in ("spare(", "_play_one_stamp_cycle", "_calibrate_motion_key_frames",
                   "_stamp_xy", "_eval_trigger_step", "motion_modules"):
        assert banned not in src, banned
    # The full-state expert keeps its stamp-then-push sequencing.
    full = inspect.getsource(FeedDocuments.play_once)
    assert full.count("self._play_one_stamp_cycle(") == 2
    assert "_blind" not in full and "_eval_" not in full


def test_step_sim_keeps_core_interaction_hook():
    assert "_notify_interaction_sim_step" in FeedDocuments.step_sim.__code__.co_names
