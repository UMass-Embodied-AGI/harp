"""Full-state expert under the eval-mode human: wait for the human's cue.

The intent tasks' full-state expert in eval mode runs the same robot script
as the blind expert (``play_blind_once``) but, at the point where the
scripted ``play_once`` waits for the human, it waits until the task reports
the human's cue (privileged state).  The blind expert calls the same hook,
which is a no-op unless ``task._full_eval_active`` is set.
"""
from __future__ import annotations

from typing import Callable

# Bounded wait: the longest eval trigger window plus one human clip, in sim
# steps, with margin.  Hitting the cap is an honest failure downstream.
DEFAULT_CUE_MAX_SIM_STEPS = 12000


def await_cue(task, ready: Callable[[], bool], label: str,
              max_sim_steps: int | None = None) -> int | None:
    """Step the sim until ``ready()``; return the steps waited (None = no-op)."""
    if not bool(getattr(task, "_full_eval_active", False)):
        return None
    cap = int(max_sim_steps if max_sim_steps is not None else task.config.get(
        "full_eval_cue_max_sim_steps", DEFAULT_CUE_MAX_SIM_STEPS))
    waited = 0
    while waited < cap and not ready():
        task.step_sim()
        waited += 1
    log = dict(getattr(task, "_full_eval_cue_waits", None) or {})
    log[label] = {"sim_steps": waited, "ready": bool(ready())}
    task._full_eval_cue_waits = log
    return waited
