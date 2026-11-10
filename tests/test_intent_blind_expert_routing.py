"""The unprivileged ("no intent") expert on intent tasks must not be the
full-state rollout.

If an intent row ran ``play_once`` with the avatar merely hidden from
planning, the "blind" expert would wait for the human's cue exactly like the
full-state expert and the full-vs-blind gap would vanish silently.  These
checks pin the routing so that cannot happen unnoticed.
"""

import pytest

from envs.tasks import resolve_task_class
from envs.tasks.registry import TASK_DESCRIPTORS, TaskRegistration, TaskVariant
from scripts.eval_expert_set import (
    BLIND_INTENT_TASKS,
    build_config,
    configure_parent_rollout_eval,
    hides_avatar_from_planning,
    policy_play_mode,
)


def _intent_names():
    return sorted(
        d.name for d in TASK_DESCRIPTORS
        if d.variant is TaskVariant.INTENT
        and d.registration is TaskRegistration.CANONICAL
    )


# The seven intent tasks of the benchmark.  (The
# registry also files the factory scenarios under the intent variant; they
# are not part of the benchmark and have no blind expert.)
BENCHMARK_INTENT_TASKS = {
    "deliver_to_human", "oil_bottle_recovery", "open_microwave",
    "pour_water", "feed_documents", "take_from_human_easy",
    "take_from_human_safety",
}


def test_every_benchmark_intent_task_has_a_blind_expert():
    assert set(BLIND_INTENT_TASKS) == BENCHMARK_INTENT_TASKS
    assert BENCHMARK_INTENT_TASKS <= set(_intent_names())


@pytest.mark.parametrize("task_name", sorted(BLIND_INTENT_TASKS))
def test_blind_and_full_state_experts_take_different_rollouts(task_name):
    _, cls = resolve_task_class(task_name, no_human=False)
    assert callable(getattr(cls, "play_blind_once", None))
    blind_mode, _ = policy_play_mode(task_name, cls, "expert")
    full_mode, _ = policy_play_mode(task_name, cls, "expert_full_state")
    assert blind_mode == "play_blind_once"
    assert full_mode != blind_mode


@pytest.mark.parametrize("task_name", sorted(BLIND_INTENT_TASKS))
def test_blind_intent_expert_plans_on_the_avatar_and_runs_in_eval_mode(task_name):
    row = {"task": task_name, "category": "cueing", "config_overrides": {}}
    assert not hides_avatar_from_planning(row, "expert")
    cfg = build_config(row, "expert")
    configure_parent_rollout_eval(cfg, task_name, "play_blind_once")
    assert cfg["eval_force_policy_ticks"] is True
