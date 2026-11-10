"""Freeze the expert-eval parent-rollout tick dispatch after phase-3 step 7.

``scripts/eval_expert_set.py`` used to probe every task for private
``_eval_step_avatar`` / ``_eval_at_step`` hooks.  It now calls
``InteractionController.on_scripted_rollout_tick``.  For every registered task
the new dispatch must advance exactly what the old chain advanced.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from envs.interaction.base import InteractionController
from envs.tasks import TASK_MAP, resolve_task_class
from scripts.eval_expert_set import run_with_eval_step_driver


def _task_names():
    return sorted(TASK_MAP)


@pytest.mark.parametrize("task_name", _task_names())
def test_scripted_rollout_tick_matches_legacy_private_hook_chain(task_name):
    _, task_cls = resolve_task_class(task_name)
    try:
        task = task_cls({"eval_mode": True})
    except Exception as exc:  # pragma: no cover - constructor needs a scene
        pytest.skip(f"{task_name} cannot be built without a scene: {exc}")
    controller = task.interaction_controller
    if controller.handles_policy_ticks:
        return
    overridden = (
        type(controller).on_scripted_rollout_tick
        is not InteractionController.on_scripted_rollout_tick
    )
    legacy_hook = hasattr(task, "_eval_step_avatar")
    # The legacy ``_eval_at_step`` branch was only reachable for tasks whose
    # controller does not own policy ticks; no registered task is in that
    # position, so the migration did not need to preserve it.
    assert not (
        not legacy_hook and hasattr(task, "_eval_at_step")
    ), f"{task_name}: legacy _eval_at_step driver branch would have fired"
    assert overridden == legacy_hook, (
        f"{task_name}: controller override={overridden} but task "
        f"_eval_step_avatar={legacy_hook}"
    )


def test_base_hook_is_a_no_op_with_task_only_signature():
    params = tuple(
        inspect.signature(InteractionController.on_scripted_rollout_tick)
        .parameters
    )
    assert params == ("self", "task")
    assert InteractionController().on_scripted_rollout_tick(object()) is None


def test_driver_never_probes_private_task_hooks():
    class Controller(InteractionController):
        handles_policy_ticks = False

        def __init__(self):
            self.ticks = 0

        def on_scripted_rollout_tick(self, task):
            self.ticks += 1

    class Task:
        def __init__(self):
            self.interaction_controller = Controller()
            self.config = {"eval_mode": True}
            self.avatar = object()
            self.sim = 0

        def step_sim(self):
            self.sim += 1

        def _eval_step_avatar(self):
            pytest.fail("driver reached a private task hook")

        def _eval_at_step(self, step_idx):
            pytest.fail("driver reached a private task hook")

    task = Task()

    def rollout():
        for _ in range(7):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(
        task, rollout, {"eval_parent_rollout_tick_interval": 3}
    )
    assert task.sim == 7
    assert task.interaction_controller.ticks == 2
    assert np.isfinite(task.sim)


def test_default_tick_interval_is_the_policy_cadence():
    # Experts tick the eval-mode human at the VLA rate.
    from scripts.eval_expert_set import rollout_tick_interval

    assert rollout_tick_interval({}) == 25
    assert rollout_tick_interval({"action_substeps": 50}) == 50
    assert rollout_tick_interval({"action_substeps": 25,
                                  "eval_parent_rollout_tick_interval": 3}) == 3
