"""Random-start eval human on the four intent tasks that measure intent.

stamp_documents, take_from_human_safety, deliver_to_human_easy and
oil_bottle_recovery: in eval mode (the VLA policy path and both eval
experts) the human acts at a per-seed random policy step during the policy,
drawn from a private RNG stream; nothing happens in the pre-policy warmup.
The full-state expert runs in eval mode against the same human and waits
for the cue; the blind expert does not.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.interaction.cue_wait import await_cue
from envs.tasks.deliver_to_human import DeliverToHuman
from envs.tasks.oil_bottle_recovery import OilBottleRecovery
from envs.tasks.feed_documents import FeedDocuments
from envs.tasks.take_from_human_safety import TakeFromHumanSafety
from scripts.eval_expert_set import (
    FULL_EVAL_INTENT_TASKS,
    build_config,
    configure_parent_rollout_eval,
    policy_play_mode,
)

CLASSES = {
    "feed_documents": FeedDocuments,
    "take_from_human_safety": TakeFromHumanSafety,
    "deliver_to_human": DeliverToHuman,
    "oil_bottle_recovery": OilBottleRecovery,
}


def test_full_eval_tasks_are_exactly_the_four_intent_measuring_tasks():
    assert set(FULL_EVAL_INTENT_TASKS) == set(CLASSES)


@pytest.mark.parametrize("task_name", sorted(CLASSES))
def test_full_state_expert_faces_the_eval_human(task_name):
    cls = CLASSES[task_name]
    assert policy_play_mode(task_name, cls, "expert_full_state") == (
        "play_full_eval_once", None)
    row = {"task": task_name, "category": "cueing",
           "config_overrides": {"eval_mode": True}}
    cfg = build_config(row, "expert_full_state")
    assert cfg["eval_mode"] is True
    configure_parent_rollout_eval(cfg, task_name, "play_full_eval_once")
    assert cfg["eval_force_policy_ticks"] is True


@pytest.mark.parametrize("cls", [TakeFromHumanSafety, DeliverToHuman,
                                 OilBottleRecovery])
def test_no_pre_policy_cue_by_default(cls):
    assert cls.EVAL_AVATAR_PRE_POLICY is False
    task = cls({"eval_mode": True})
    task.avatar = object()
    task.get_obs = lambda: "obs"
    # The warmup returns before touching the (bare) avatar.
    assert task.eval_pre_policy_warmup() == "obs"


def test_stamp_warmup_does_nothing_for_the_random_start_human():
    task = FeedDocuments({"eval_mode": True})
    task.avatar = object()
    task.document_actors = [object(), object()]
    task._attach_stamp_to_palm = lambda: pytest.fail("seal handled at reset")
    task._play_one_stamp_cycle = lambda **kw: pytest.fail("stamped in warmup")
    task.get_obs = lambda: "obs"
    assert task.eval_pre_policy_warmup() == "obs"


def test_oil_window_is_private_and_seeded():
    lo, hi = OilBottleRecovery.EVAL_TRIGGER_STEP_MIN, OilBottleRecovery.EVAL_TRIGGER_STEP_MAX
    np.random.seed(0)
    state = np.random.get_state()
    seen = set()
    for seed in range(30):
        draws = []
        for _ in range(2):
            task = OilBottleRecovery({"eval_mode": True})
            task._episode_seed = seed
            task._attach_items_to_hands = lambda: None
            task.avatar = SimpleNamespace(frame_ratio=1.0)
            task._init_eval_recovery_state()
            draws.append(task._eval_trigger_step)
        assert draws[0] == draws[1] and lo <= draws[0] <= hi
        seen.add(draws[0])
    assert len(seen) > 10
    assert np.array_equal(np.random.get_state()[1], state[1])


def test_safety_policy_path_reach_is_non_blocking():
    task = TakeFromHumanSafety({"eval_mode": True})
    task._blind_active = False          # the VLA path
    task._eval_handover_done = False
    task._eval_trigger_step = 30
    task.avatar = SimpleNamespace(spare=lambda: True)
    started = []
    task._blind_start_handover_reach = lambda: started.append(1)
    task._eval_at_step(29)
    assert started == []
    task._eval_at_step(30)
    assert started == [1] and task._eval_handover_done is True
    assert task._eval_handover_triggered_at == 30


def test_await_cue_is_a_noop_for_the_blind_expert_and_bounded_otherwise():
    task = SimpleNamespace(config={}, steps=0)
    task.step_sim = lambda: setattr(task, "steps", task.steps + 1)
    assert await_cue(task, lambda: False, "x") is None and task.steps == 0
    task._full_eval_active = True
    assert await_cue(task, lambda: task.steps >= 7, "x") == 7
    assert await_cue(task, lambda: False, "y", max_sim_steps=5) == 5
    assert task._full_eval_cue_waits == {
        "x": {"sim_steps": 7, "ready": True},
        "y": {"sim_steps": 5, "ready": False},
    }
