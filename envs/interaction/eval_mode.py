"""Composable lifecycle drivers for incrementally migrated eval-mode tasks.

The legacy :class:`EvalModeAvatarMixin` coupled reset/action ordering to the
task MRO.  These controllers preserve that ordering through the BaseTask
interaction hooks while task-specific behavior remains on the owning task.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from .base import (
    AssistInteractionController,
    InteractionController,
    IntentInteractionController,
)
from .neutral_table_work import NeutralTableWorkController


class TaskDrivenEvalTickController(InteractionController):
    """Adapter for standalone intent tasks that own their eval machine.

    ``OpenMicrowave`` and ``OilBottleRecovery`` advance their eval-mode human
    behavior from their own ``take_action`` wrapper (``_eval_step_avatar``)
    and never migrated to a family controller.  Expert evaluation drives
    their scripted parent rollout and needs the same tick at policy cadence,
    so the controller forwards the scripted-rollout hook to the task.  It
    holds no state and consumes no RNG.
    """

    rng_namespace = "task_driven_eval_tick.v1"
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def on_scripted_rollout_tick(self, task) -> None:
        task._eval_step_avatar()


def delayed_assist_trigger_step(task) -> int | None:
    """Policy step at which an eval-mode assist human starts, or ``None``.

    By default assist controllers start the human's contribution at reset.
    The ``eval_assist_trigger_delayed`` config opts into drawing the start
    from the task's ``EVAL_TRIGGER_STEP_MIN/MAX`` window (overridable via
    ``eval_trigger_step_min/max``), so a rollout cannot rely on the human
    acting at a fixed time.  The draw consumes RNG only when opted in, which
    keeps default seeded streams unchanged.
    """
    if not bool(task.config.get("eval_assist_trigger_delayed", False)):
        return None
    lo = int(task.config.get(
        "eval_trigger_step_min", getattr(task, "EVAL_TRIGGER_STEP_MIN", 20),
    ))
    hi = int(task.config.get(
        "eval_trigger_step_max", getattr(task, "EVAL_TRIGGER_STEP_MAX", 60),
    ))
    return int(np.random.randint(lo, max(lo, hi) + 1))


class AssistEvalModeController(AssistInteractionController):
    """Drive the existing assist reset/policy hooks through composition.

    The controller is intentionally stateless: release-v1 scene replay already
    persists the task-facing ``_eval_*`` compatibility fields, and keeping
    those fields authoritative avoids introducing a second state owner during
    incremental family migrations.
    """

    rng_namespace = "assist_eval_mode.v1"
    handles_policy_ticks = True
    # Parent-fallback assists queue their single-shot human motion at reset,
    # making later hooks latched no-ops. The delayed skillet blind expert
    # self-drives its hook at the chosen trigger. The legacy generic parent
    # driver never advanced the task's published policy counter, so opting out
    # here preserves that metric and cadence exactly.
    handles_scripted_rollout_ticks = False

    @staticmethod
    def _active(task) -> bool:
        return (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        )

    @staticmethod
    def _hook(task, name: str):
        hook = getattr(task, name, None)
        if not callable(hook):
            raise TypeError(
                f"{type(task).__name__} uses AssistEvalModeController but "
                f"does not define callable {name}()"
            )
        return hook

    def after_base_reset_observation(
        self,
        task,
        *,
        observation: Any,
    ) -> None:
        if not self._active(task):
            return
        task._eval_policy_step_count = 0
        self._hook(task, "_eval_at_reset")()

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if not self._active(task):
            return
        # Count active eval calls rather than copying the framework's absolute
        # counter. This exactly matches EvalModeAvatarMixin if eval/avatar state
        # is toggled during an episode.
        task._eval_policy_step_count = int(
            getattr(task, "_eval_policy_step_count", 0)
        ) + 1
        self._hook(task, "_eval_at_step")(task._eval_policy_step_count)


class IntentEvalModeController(IntentInteractionController):
    """Preserve the legacy single-shot intent-task eval lifecycle.

    Intent tasks queue their opening avatar behavior only after the initial
    reset observation has been captured. Policy calls then advance a 1-based
    compatibility counter before ``BaseTask`` can short-circuit on success or
    apply the robot action. The legacy mixin supplied a default no-op
    ``_eval_at_step``; composed tasks may omit that hook with identical effect.
    """

    rng_namespace = "intent_eval_mode.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    @staticmethod
    def _active(task) -> bool:
        return (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        )

    def after_base_reset_observation(
        self,
        task,
        *,
        observation: Any,
    ) -> None:
        if not self._active(task):
            return
        task._eval_policy_step_count = 0
        hook = getattr(task, "_eval_at_reset", None)
        if not callable(hook):
            raise TypeError(
                f"{type(task).__name__} uses IntentEvalModeController but "
                "does not define callable _eval_at_reset()"
            )
        hook()

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if not self._active(task):
            return
        task._eval_policy_step_count = int(
            getattr(task, "_eval_policy_step_count", 0)
        ) + 1
        hook = getattr(task, "_eval_at_step", None)
        if hook is None:
            return
        if not callable(hook):
            raise TypeError(
                f"{type(task).__name__} uses IntentEvalModeController but "
                "defines non-callable _eval_at_step"
            )
        hook(task._eval_policy_step_count)


class CategorizeNeutralController(NeutralTableWorkController):
    """Preserve release-v1 seeded layouts without running Inspect behavior.

    The former neutral MRO passed through the interrupt scenario. Its reset
    consumed an interrupt-trigger draw and, when enabled, an inspect-pool draw
    before the neutral mixin sampled its own trigger. The values were unrelated
    to neutral behavior, but the draw positions affect reproducible episodes.
    Burn those draws without publishing interrupt state or touching the avatar.
    """

    rng_namespace = "categorize.neutral.v1"
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def after_base_reset_observation(
        self,
        task,
        *,
        observation: Any,
    ) -> None:
        if (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        ):
            lo = int(task.config.get(
                "eval_trigger_step_min", task.EVAL_TRIGGER_STEP_MIN,
            ))
            hi = int(task.config.get(
                "eval_trigger_step_max", task.EVAL_TRIGGER_STEP_MAX,
            ))
            if hi < lo:
                hi = lo
            np.random.randint(lo, hi + 1)

        pool = STANDARD_INSPECT_POOL
        forced_name = task.config.get("inspect_motion_name")
        if (
            not forced_name
            and task.config.get("randomize_inspect", False)
            and len(pool) > 1
        ):
            np.random.randint(len(pool))

# Family controllers that grew out of this module keep their historical
# import path here via lazy (PEP 562) lookups; see inspect_motion.py.
_FAMILY_EXPORTS = {
    "DualShoesAssistController": ".place_dual_shoes",
    "CabinetAssistController": ".put_object_cabinet",
}


def __getattr__(name: str):
    module = _FAMILY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __package__), name)


def __dir__():
    return sorted(set(globals()) | set(_FAMILY_EXPORTS))
