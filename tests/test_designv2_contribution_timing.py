"""In every contribution task the eval human shows intent first.

The human starts at reset (policy step 0) in eval mode, before the robot can
commit, with no reactive behaviour.  ``eval_assist_trigger_delayed`` remains
an explicit opt-in for ablations only.
"""

from types import SimpleNamespace

import numpy as np

from envs.interaction.place_food_in_skillet import PlaceFoodInSkilletAssistController
from scripts.eval_expert_set import DELAYED_ASSIST_TRIGGER_TASKS


def _skillet_task(**config):
    calls = []
    task = SimpleNamespace(
        config=dict(config),
        EVAL_TRIGGER_STEP_MIN=20,
        EVAL_TRIGGER_STEP_MAX=60,
        AVATAR_START_DELAY_RANGE=(40, 180),
        food_actors=[("burger_actor", "burger_spec"), ("bread_actor", "bread_spec")],
    )
    task._choose_avatar_food = lambda: (0, [1])
    task._start_avatar_food_place = lambda actor, spec: calls.append((actor, spec))
    return task, calls


def test_skillet_assist_eval_human_starts_at_reset_and_keeps_the_rng_stream():
    np.random.seed(5)
    legacy_trigger = np.random.randint(20, 61)
    after_burn = np.random.rand()

    task, calls = _skillet_task(eval_mode=True)
    np.random.seed(5)
    PlaceFoodInSkilletAssistController().eval_reset(task)
    assert calls == [("burger_actor", "burger_spec")]
    assert task._eval_assist_trigger_step == 0
    assert task._eval_assist_fired is True
    assert legacy_trigger != 0
    # The legacy trigger draw is burned, so later global draws are unchanged.
    assert np.random.rand() == after_burn


def test_skillet_assist_delayed_opt_in_keeps_the_legacy_window():
    task, calls = _skillet_task(eval_mode=True, eval_assist_trigger_delayed=True)
    np.random.seed(5)
    expected = np.random.randint(20, 61)
    np.random.seed(5)
    controller = PlaceFoodInSkilletAssistController()
    controller.eval_reset(task)
    assert calls == [] and task._eval_assist_trigger_step == expected
    controller.eval_step(task, expected - 1)
    assert calls == []
    controller.eval_step(task, expected)
    assert calls == [("burger_actor", "burger_spec")]


def test_skillet_assist_eval_start_switch():
    controller = PlaceFoodInSkilletAssistController()
    eval_task, _ = _skillet_task(eval_mode=True)
    train_task, _ = _skillet_task(eval_mode=False)
    delayed_task, _ = _skillet_task(eval_mode=True, eval_assist_trigger_delayed=True)
    assert controller.eval_starts_at_reset(eval_task) is True
    assert controller.eval_starts_at_reset(train_task) is False
    assert controller.eval_starts_at_reset(delayed_task) is False


def test_blind_expert_harness_delays_no_contribution_human():
    assert DELAYED_ASSIST_TRIGGER_TASKS == frozenset()
