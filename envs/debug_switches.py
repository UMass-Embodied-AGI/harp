"""Refuse benchmark runs while a physics- or planner-bypassing debug switch is set.

These environment variables exist for interactive debugging of scripted
experts. Each one replaces part of the simulated rollout, so results produced
with them set are not benchmark results:

- ``DEBUG_TELEPORT=1``: the planner returns a single IK waypoint and plans are
  executed by teleporting the arm instead of PD control.
- ``USE_SETQPOS=1``: planned trajectories are executed with ``set_qpos``
  (kinematic joint writes) instead of PD control.
- ``DEBUG_BOUNDED_JOINT_PATHS=1``: an experimental bounded-joint planning
  branch replaces the default mplib path.

``MPLIB_PLAN_DIAG=1`` is not refused: it only prints planner diagnostics
(why a screw plan or RRT start fails, which seeded-IK helper ran) and does not
change any plan.

The eval and collection entry points call :func:`refuse_debug_switches` before
building a task. Developers who need a switch with one of those entry points
can set ``HARP_ALLOW_DEBUG_SWITCHES=1``; a warning is printed and the outputs
must not be reported as benchmark results.
"""

from __future__ import annotations

import os
import sys

BYPASS_SWITCHES = ("DEBUG_TELEPORT", "USE_SETQPOS", "DEBUG_BOUNDED_JOINT_PATHS")
ALLOW_ENV = "HARP_ALLOW_DEBUG_SWITCHES"


def active_bypass_switches(environ=None) -> list[str]:
    """Names of the bypassing switches that are enabled (value ``1``)."""
    environ = os.environ if environ is None else environ
    return [name for name in BYPASS_SWITCHES if str(environ.get(name, "")).strip() == "1"]


def refuse_debug_switches(entry_point: str, environ=None) -> None:
    """Exit with a clear error when a bypassing debug switch is enabled."""
    environ = os.environ if environ is None else environ
    active = active_bypass_switches(environ)
    if not active:
        return
    names = ", ".join(f"{name}=1" for name in active)
    if str(environ.get(ALLOW_ENV, "")).strip() == "1":
        print(
            f"WARNING: {entry_point}: debug switch(es) {names} bypass physics or "
            "planning; these outputs are not benchmark results.",
            file=sys.stderr,
            flush=True,
        )
        return
    raise SystemExit(
        f"{entry_point}: refusing to run with {names} set. These debug switches "
        "bypass physics or planning, so rollouts would not be benchmark rollouts. "
        f"Unset them (or set {ALLOW_ENV}=1 for local debugging only)."
    )
