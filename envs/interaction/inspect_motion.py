"""Composable inspect-motion lifecycle for migrated interrupt families.

The family task classes still own their release-v1 reset, policy-action,
scripted-play, evaluation, and compatibility state.  These controllers only
replace the legacy ``InspectMotionMixin.reset`` boundary: resolve the selected
inspect motion after the base reset observation has already been captured.
"""

from __future__ import annotations

from typing import Any

from ..avatar.inspect_motion_mixin import InspectMotionRuntime
from .base import InterruptInteractionController


class InspectMotionInterruptController(InterruptInteractionController):
    """Resolve inspect metadata without taking ownership of task state."""

    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def __init__(self) -> None:
        self.inspect_runtime = InspectMotionRuntime()

    def resolve_inspect_motion(self, task) -> None:
        self.inspect_runtime.resolve(
            task,
            prepare_entry=task._prepare_inspect_motion_entry,
        )

    def on_scripted_rollout_tick(self, task) -> None:
        # Families whose eval machine is driven from the task's take_action
        # wrapper expose it as ``_eval_step_avatar``; advance it here so the
        # expert-eval runner never probes the task for private hooks.
        hook = getattr(task, "_eval_step_avatar", None)
        if callable(hook):
            hook()

    def after_base_reset_observation(
        self,
        task,
        *,
        observation: Any,
    ) -> None:
        # Legacy InspectMotionMixin performed this immediately after
        # BaseTask.reset returned.  This hook runs at the same observable
        # boundary: the initial observation is already captured and unchanged.
        # Dispatch through the task just as the mixin did so a downstream
        # compatibility override of _resolve_inspect_motion remains effective.
        task._resolve_inspect_motion()


class CategorizeInterruptController(InspectMotionInterruptController):
    """Preserve categorize interrupt reset and policy hook ordering.

    Scripted experts are ticked through
    ``BaseTask.advance_interaction_policy_step`` at policy cadence, so this
    controller opts into scripted-rollout ticks.
    """

    rng_namespace = "categorize.interrupt.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = True

    def after_base_reset_observation(
        self,
        task,
        *,
        observation: Any,
    ) -> None:
        active = (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        )
        if active:
            task._eval_policy_step_count = 0
            task._eval_at_reset()
        super().after_base_reset_observation(task, observation=observation)
        task._categorize_interrupt_after_reset()

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if (
            getattr(task, "avatar", None) is None
            or not bool(task.config.get("eval_mode", False))
        ):
            return
        task._eval_policy_step_count = int(
            getattr(task, "_eval_policy_step_count", 0)
        ) + 1
        task._eval_at_step(task._eval_policy_step_count)

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = getattr(task, "_interrupt", None)
        if episode is not None and getattr(task, "avatar", None) is not None:
            episode.human.on_sim_step(task)

# Family controllers that grew out of this module keep their historical
# import path here.  The lookups are lazy (PEP 562) so importing a family
# module through another seam cannot form an import cycle.
_FAMILY_EXPORTS = {
    "PlaceBurgerFriesInterruptController": ".place_burger_fries",
    "SkilletInterruptController": ".place_food_in_skillet",
    "DualShoesInterruptController": ".place_dual_shoes",
    "CabinetInterruptController": ".put_object_cabinet",
}


def __getattr__(name: str):
    module = _FAMILY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __package__), name)


def __dir__():
    return sorted(set(globals()) | set(_FAMILY_EXPORTS))
