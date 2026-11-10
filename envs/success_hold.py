"""One success rule for every evaluator: the task's success predicate must
hold for ``N`` consecutive policy steps.

Policies (``scripts/eval.py``, used by ``eval_vla.py``) stop early once the
predicate has held for ``N`` steps; at the horizon the trailing streak must
also be ``N`` long.  Scripted experts (``scripts/eval_expert_set.py``) are
judged at the end of their rollout and then hold still for ``N`` policy
steps; the predicate must stay true on every one of them.

N = 10 policy steps (0.5 s at 20 Hz).  A held or carried object that briefly
satisfies a placement check no longer ends the episode as a success, and an
object that settles at the goal keeps satisfying it (audit X1,
data/vla_train37_eval/task_audit/AUDIT.md).  ``eval_success_hold_steps`` in
the episode config (CLI ``--success-hold-steps``) overrides it.
"""

from __future__ import annotations

from typing import Any, Mapping

DEFAULT_SUCCESS_HOLD_STEPS = 10
CONFIG_KEY = "eval_success_hold_steps"


def success_hold_steps(config: Mapping[str, Any] | None) -> int:
    """Consecutive policy steps the success predicate must hold (>= 1)."""
    value = (config or {}).get(CONFIG_KEY)
    if value is None:
        value = DEFAULT_SUCCESS_HOLD_STEPS
    return max(1, int(value))
