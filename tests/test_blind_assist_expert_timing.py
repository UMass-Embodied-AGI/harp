"""Blind assist experts face a human that starts at an unknown step."""

from types import SimpleNamespace

import numpy as np

from envs.interaction.categorize import CategorizeCooperativeController
from envs.interaction.eval_mode import delayed_assist_trigger_step
from envs.interaction.place_burger_fries import PlaceBurgerFriesAssistController
from envs.interaction.stack_bowls_three import StackBowlsThreeAssistController
from envs.tasks import resolve_task_class
from scripts.eval_expert_set import (
    DELAYED_ASSIST_TRIGGER_TASKS,
    configure_parent_rollout_eval,
    policy_play_mode,
    run_with_eval_step_driver,
)


def _task(**config):
    return SimpleNamespace(config=config, EVAL_TRIGGER_STEP_MIN=20, EVAL_TRIGGER_STEP_MAX=60)


def test_default_trigger_is_unchanged_and_consumes_no_rng():
    np.random.seed(3)
    expected = np.random.rand()
    np.random.seed(3)
    assert delayed_assist_trigger_step(_task()) is None
    assert np.random.rand() == expected


def test_delayed_trigger_draws_from_the_task_window():
    np.random.seed(0)
    steps = {delayed_assist_trigger_step(_task(eval_assist_trigger_delayed=True)) for _ in range(200)}
    assert min(steps) >= 20 and max(steps) <= 60 and len(steps) > 10
    pinned = _task(eval_assist_trigger_delayed=True, eval_trigger_step_min=7, eval_trigger_step_max=7)
    assert delayed_assist_trigger_step(pinned) == 7


def test_delayed_assist_controllers_arm_instead_of_firing():
    for controller, start in (
        (PlaceBurgerFriesAssistController(), "_start_avatar_burger_assist"),
        (StackBowlsThreeAssistController(region_x=(0.0, 1.0), region_y=(0.0, 1.0),
                                         min_separation=0.1), "_start_avatar_assist"),
    ):
        calls = []
        task = _task(eval_assist_trigger_delayed=True, eval_trigger_step_min=30,
                     eval_trigger_step_max=30)
        setattr(task, start, lambda *a, **k: calls.append(1) or True)
        controller.eval_reset(task)
        assert calls == [] and task._eval_assist_fired is False
        controller.eval_step(task, 29)
        assert calls == []
        controller.eval_step(task, 30)
        assert calls == [1] and task._eval_assist_fired is True


def test_delayed_categorize_cooperative_waits_for_trigger():
    calls = []
    task = _task(eval_assist_trigger_delayed=True, eval_trigger_step_min=12,
                 eval_trigger_step_max=12)
    task._avatar_sort_targets = [("a", "obj", 0, 0)]
    task._eval_fire_next_avatar_motion = lambda: calls.append(1)
    CategorizeCooperativeController().eval_reset(task)
    assert calls == [] and task._eval_trigger_step == 12
    CategorizeCooperativeController().eval_step(task, 12)
    assert calls == [1]


def test_harness_opts_blind_assist_rows_into_delay_and_ticks():
    for name in DELAYED_ASSIST_TRIGGER_TASKS:
        cfg = {}
        configure_parent_rollout_eval(cfg, name, "play_blind_once")
        assert cfg == {"eval_assist_trigger_delayed": True, "eval_force_policy_ticks": True}
    # Interrupt rows get no harness trigger overrides; the
    # progress trigger belongs to the task (same as for the policies).
    for name in ("categorize_objects_interruption", "dump_bin_interruption", "stack_three_bowls_interruption"):
        for mode in ("play_blind_once", "parent_play_once"):
            cfg = {}
            configure_parent_rollout_eval(cfg, name, mode)
            assert cfg == {}


def test_bowls_and_bread_assist_have_their_own_blind_rollout():
    for name in ("stack_three_bowls_contribution", "place_bread_in_basket_contribution"):
        mode, parent = policy_play_mode(name, resolve_task_class(name)[1], "expert")
        assert (mode, parent) == ("play_blind_once", None)


def test_forced_ticks_advance_controllers_that_skip_scripted_rollouts():
    ticks = []
    controller = SimpleNamespace(handles_policy_ticks=True, handles_scripted_rollout_ticks=False)
    task = SimpleNamespace(
        config={"eval_mode": True}, avatar=object(), interaction_controller=controller,
        advance_interaction_policy_step=lambda: ticks.append(1),
    )
    task.step_sim = lambda: None

    # The driver ticks at policy cadence (action_substeps sim
    # steps, default 25) instead of every 5 sim steps.
    def rollout():
        for _ in range(50):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(task, rollout, {})
    assert ticks == []
    assert run_with_eval_step_driver(task, rollout, {"eval_force_policy_ticks": True})
    assert ticks == [1, 1]
    assert run_with_eval_step_driver(
        task, rollout, {"eval_force_policy_ticks": True, "action_substeps": 10},
    )
    assert ticks == [1] * 7
