"""Stable lifecycle contract for task interaction modes.

This module deliberately has no Genesis imports. Controllers are ordinary
Python objects, which keeps their state machines independently testable and
prevents the task registry from initializing the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ..base_task import BaseTask


class InteractionMode(str, Enum):
    """Semantic relationship between the human and robot objective."""

    BASE = "base"
    ASSIST = "assist"
    INTERRUPT = "interrupt"
    NEUTRAL = "neutral"
    INTENT = "intent"
    LEGACY = "legacy"


class PolicyDisposition(str, Enum):
    """Controller request made immediately before a policy action."""

    PROCEED = "proceed"
    YIELD = "yield"
    RETREAT = "retreat"
    RESUME = "resume"


@dataclass(frozen=True)
class PolicyDirective:
    """Public interaction decision consumed by policy/evaluation runners.

    Controllers return semantic directives rather than exposing private
    ``_eval_*`` fields. During the first migration phase, ``PROCEED`` is
    recorded and every other value fails closed; runners gain the remaining
    behavior as individual task families move onto the controller framework.
    """

    disposition: PolicyDisposition = PolicyDisposition.PROCEED
    reason: str = ""
    hold_steps: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            disposition = PolicyDisposition(self.disposition)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported policy disposition: {self.disposition!r}"
            ) from exc
        if not isinstance(self.reason, str):
            raise TypeError("PolicyDirective reason must be a string")
        if isinstance(self.hold_steps, bool) or not isinstance(
            self.hold_steps, Integral
        ):
            raise TypeError("PolicyDirective hold_steps must be an integer")
        if self.hold_steps < 0:
            raise ValueError("hold_steps cannot be negative")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "hold_steps", int(self.hold_steps))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/pickle-friendly versioned representation."""

        return {
            "schema_version": "genesis_hr_bench.policy_directive.v1",
            "disposition": self.disposition.value,
            "reason": self.reason,
            "hold_steps": self.hold_steps,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyDirective":
        if not isinstance(value, Mapping):
            raise TypeError("PolicyDirective payload must be a mapping")
        version = value.get("schema_version")
        if version != "genesis_hr_bench.policy_directive.v1":
            raise ValueError(f"Unsupported PolicyDirective schema: {version!r}")
        required = {
            "schema_version",
            "disposition",
            "reason",
            "hold_steps",
            "metadata",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                "Malformed PolicyDirective payload; missing fields: "
                + ", ".join(missing)
            )
        extra = sorted(set(value) - required)
        if extra:
            raise ValueError(
                "Malformed PolicyDirective payload; unknown fields: "
                + ", ".join(extra)
            )
        return cls(
            disposition=value["disposition"],
            reason=value["reason"],
            hold_steps=value["hold_steps"],
            metadata=value["metadata"],
        )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("PolicyDirective metadata keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("PolicyDirective metadata floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "PolicyDirective metadata must contain only JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("PolicyDirective metadata must be a mapping")
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class InteractionController:
    """Phase-0 composition seam with conservative no-op defaults.

    Hook names explicitly describe the ``BaseTask`` core boundary. Legacy MRO
    wrappers can still perform work before or after these calls. New
    controllers use :meth:`episode_rng`; a migration may retain a legacy RNG
    stream only when preserving a frozen seeded-scene contract and must cover
    that exception with parity tests.
    """

    mode = InteractionMode.BASE
    rng_namespace: str | None = None
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def episode_rng(self, task: "BaseTask", stream: str):
        """Return a deterministic stream isolated from family layout RNG."""

        namespace = self.rng_namespace
        if not isinstance(namespace, str) or not namespace.strip():
            raise RuntimeError(
                f"{type(self).__name__} must define a stable rng_namespace "
                "before sampling interaction randomness"
            )
        return task.episode_rng(f"interaction:{namespace}:{stream}")

    def on_episode_seeded(self, task: "BaseTask", seed: int) -> None:
        """Clear controller-only state after RNGs are seeded.

        Scene/avatar/robot attributes can still refer to the prior episode at
        this boundary and must not be inspected.
        """

    def augment_scene_before_build(self, task: "BaseTask") -> None:
        """Add interaction-specific entities before ``scene.build()``."""

    def on_avatar_created_before_build(self, task: "BaseTask") -> None:
        """Configure the created, not-yet-posed avatar before scene build."""

    def on_base_reset_ready(self, task: "BaseTask") -> None:
        """Observe the ready BaseTask core before outer MRO reset wrappers."""

    def after_base_reset_observation(
        self,
        task: "BaseTask",
        *,
        observation: Any,
    ) -> None:
        """Run after the BaseTask core's initial observation is captured.

        This is the composition equivalent of legacy mixin work performed
        after ``super().reset()`` returns.  The observation has already been
        produced, so starting an avatar animation here cannot silently change
        the release's initial-observation contract.
        """

    def on_policy_tick(self, task: "BaseTask", *, step_index: int) -> None:
        """Advance controller state at one policy-step boundary.

        Controllers opt in by setting :attr:`handles_policy_ticks`.  Scripted
        rollout adapters can then use BaseTask's public driver instead of
        probing task-private ``_eval_*`` methods.
        """

    def before_base_policy_action(
        self,
        task: "BaseTask",
        *,
        step_index: int,
        action: Any,
        action_type: str,
    ) -> PolicyDirective:
        """Return the interaction decision for the next policy action."""
        return PolicyDirective()

    def after_base_policy_action(
        self,
        task: "BaseTask",
        *,
        step_index: int,
        directive: PolicyDirective,
    ) -> None:
        """Observe completion of one policy action."""

    def on_base_sim_step(self, task: "BaseTask", *, step_index: int) -> None:
        """Advance interaction-only state once per BaseTask simulation tick."""

    def on_scripted_rollout_tick(self, task: "BaseTask") -> None:
        """Advance eval-mode interaction state during a scripted parent rollout.

        Expert evaluation drives a family's *parent* ``play_once`` (which calls
        ``step_sim`` directly, never ``take_action``) and still needs the
        eval-mode human behavior to advance at policy-step cadence.  The
        runner calls this hook every N sim steps for controllers that do not
        own policy ticks (``handles_policy_ticks`` is ``False``); controllers
        that own policy ticks are advanced through
        ``BaseTask.advance_interaction_policy_step`` when they also set
        ``handles_scripted_rollout_ticks``.  The default is a no-op.
        """

    def interaction_metrics(self, task: "BaseTask") -> Mapping[str, Any]:
        """Return additive public evaluation metrics for this interaction."""

        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible controller state for future replay support."""

        return {"schema_version": "genesis_hr_bench.interaction_state.v1"}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        """Restore controller state, rejecting unknown future schemas."""

        if not isinstance(value, Mapping):
            raise TypeError("interaction state must be a mapping")
        required = {"schema_version"}
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError(
                "Malformed interaction state; missing fields: "
                + ", ".join(missing)
            )
        if extra:
            raise ValueError(
                "Malformed interaction state; unknown fields: "
                + ", ".join(extra)
            )
        version = value.get("schema_version")
        if version != "genesis_hr_bench.interaction_state.v1":
            raise ValueError(f"Unsupported interaction state schema: {version!r}")

    def restore_task_state(
        self,
        task: "BaseTask",
        value: Mapping[str, Any],
    ) -> None:
        """Restore controller state and synchronize any task-facing surface."""

        self.load_state_dict(value)


class AssistInteractionController(InteractionController):
    """Base class for cooperative controllers that help the robot objective."""

    mode = InteractionMode.ASSIST


class InterruptInteractionController(InteractionController):
    """Base class for controllers that ask the robot to yield or retreat."""

    mode = InteractionMode.INTERRUPT


class NeutralInteractionController(InteractionController):
    """Base class for unrelated human work performed near the robot."""

    mode = InteractionMode.NEUTRAL


class IntentInteractionController(InteractionController):
    """Base class for human-intention and work-cell interaction controllers."""

    mode = InteractionMode.INTENT


class LegacyInteractionController(InteractionController):
    """Compatibility adapter while behavior remains in legacy task classes."""

    mode = InteractionMode.LEGACY
    rng_namespace = "legacy.v1"
