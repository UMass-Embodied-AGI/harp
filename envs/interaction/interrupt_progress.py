"""Shared interrupt contract for the ``*_interrupt`` tasks (Interruption).

The contract has four parts, shared by every interrupt family:

1. :class:`ProgressTrigger` decides *when* the human interrupts.  It fires
   only on robot progress toward a task object: at least
   ``min_policy_steps`` policy steps since reset, the TCP has moved at least
   ``min_moved_m`` from its reset pose, and some eligible object is both
   ``progress >= trigger_progress`` (3-D TCP distance relative to the
   distance at reset, floored at ``start_dist_floor_m``) and within the
   task's ``commit_radius_m``.  Families add phase gates (pan on the cooktop,
   drawer open, robot not carrying an object).  There is no time-based
   fallback: if the robot never approaches an object, no interruption
   happens, and the metrics say why.
2. :class:`InterruptHumanSequence` drives the human after the trigger: a
   root glide from where the human stands to the reach-calibrated pose
   (never a teleport), the family's clip (one- or two-handed), the clip's
   blend back to the idle pose, and a glide back to the pre-trigger root.
   The human is not reactive: no retargeting, no pausing for the robot.  The
   clip never attaches an object the robot is grasping or that moved more
   than ``attach_max_drift_m`` since the trigger.
3. The policy keeps control: no family may substitute, skip or blend robot
   actions.  The task's ``take_action`` only ticks the machine.
4. Scripted experts are driven at policy cadence by
   :class:`ScriptedRolloutDriver`.  The full-state expert yields on the
   privileged fired flag (:class:`InterruptYield` unwinds the scripted
   motion, the task retreats with mplib, waits for the human to finish and
   resumes from the objects' live poses); the no-intent expert never yields.

This module has no Genesis imports so the trigger logic is unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping

import numpy as np


TRIGGER_SCHEMA = "genesis_hr_bench.interrupt_progress.v1"

# Contract defaults (policy steps are 20 Hz: action_substeps=25 at dt=0.002).
DEFAULT_MIN_POLICY_STEPS = 20
DEFAULT_MIN_MOVED_M = 0.05
DEFAULT_TRIGGER_PROGRESS = 0.5
DEFAULT_COMMIT_RADIUS_M = 0.22
DEFAULT_START_DIST_FLOOR_M = 0.30
# Glide-in duration.  30 policy steps = 1.5 s of visible approach before the
# reach clip even starts, so the >= 1.5 s warning holds for every clip.
DEFAULT_GLIDE_POLICY_STEPS = 30
DEFAULT_GLIDE_OUT_POLICY_STEPS = 20
# Sim frames for the clip-end blend back to the idle pose (0.5 s at 500 Hz).
DEFAULT_IDLE_RETURN_FRAMES = 250
DEFAULT_ATTACH_MAX_DRIFT_M = 0.03
WARNING_REACH_RADIUS_M = 0.10
# Gripper command below which the robot is treated as closed/holding.
DEFAULT_HELD_GRIPPER_BELOW = 0.6
DEFAULT_HELD_TCP_RADIUS_M = 0.07
# Expert yield: sim-step cap while waiting for the human to finish.
DEFAULT_EXPERT_WAIT_MAX_SIM_STEPS = 12000
DEFAULT_EXPERT_POST_HUMAN_SETTLE_STEPS = 60


def _vec3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size < 3:
        raise ValueError(f"expected a 3-vector, got shape {arr.shape}")
    return arr[:3].copy()


def _json_key(key: Hashable) -> Any:
    if isinstance(key, (np.integer,)):
        return int(key)
    if isinstance(key, (int, str)):
        return key
    return str(key)


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# ----------------------------------------------------------------------
# Trigger
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressTriggerParams:
    """Thresholds for :class:`ProgressTrigger`.

    Per-task values come from class attributes (``INTERRUPT_COMMIT_RADIUS_M``
    and friends) and can be overridden per run with the lower-case config
    keys listed in :data:`_CONFIG_KEYS`.
    """

    min_policy_steps: int = DEFAULT_MIN_POLICY_STEPS
    min_moved_m: float = DEFAULT_MIN_MOVED_M
    trigger_progress: float = DEFAULT_TRIGGER_PROGRESS
    commit_radius_m: float = DEFAULT_COMMIT_RADIUS_M
    start_dist_floor_m: float = DEFAULT_START_DIST_FLOOR_M

    def __post_init__(self) -> None:
        if int(self.min_policy_steps) < 0:
            raise ValueError("min_policy_steps must be >= 0")
        for name in ("min_moved_m", "commit_radius_m", "start_dist_floor_m"):
            if not float(getattr(self, name)) >= 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= float(self.trigger_progress) <= 1.0:
            raise ValueError("trigger_progress must lie in [0, 1]")

    @classmethod
    def for_task(cls, task, **defaults) -> "ProgressTriggerParams":
        values = {}
        for fld, (attr, key) in _CONFIG_KEYS.items():
            value = defaults.get(fld, getattr(cls, fld))
            value = getattr(task, attr, value)
            config = getattr(task, "config", None) or {}
            if key in config and config[key] is not None:
                value = config[key]
            values[fld] = int(value) if fld == "min_policy_steps" else float(value)
        return cls(**values)


# field -> (task class attribute, config override key)
_CONFIG_KEYS = {
    "min_policy_steps": ("INTERRUPT_MIN_POLICY_STEPS", "interrupt_min_policy_steps"),
    "min_moved_m": ("INTERRUPT_MIN_MOVED_M", "interrupt_min_moved_m"),
    "trigger_progress": ("INTERRUPT_TRIGGER_PROGRESS", "interrupt_trigger_progress"),
    "commit_radius_m": ("INTERRUPT_COMMIT_RADIUS_M", "interrupt_commit_radius_m"),
    "start_dist_floor_m": ("INTERRUPT_START_DIST_FLOOR_M", "interrupt_start_dist_floor_m"),
}


class ProgressTrigger:
    """Fire once when the robot commits to an eligible object.

    ``arm`` records the TCP pose and per-object distances at reset.  Each
    :meth:`update` receives the *eligible* candidates only (families filter
    out held, placed, stacked or out-of-reach objects) and returns the key of
    the object the robot is going for when the trigger fires, else ``None``.
    """

    def __init__(self, params: ProgressTriggerParams | None = None) -> None:
        self.params = params or ProgressTriggerParams()
        self.clear()

    # -- state -----------------------------------------------------------

    def clear(self) -> None:
        self.armed = False
        self.reset_tcp: np.ndarray | None = None
        self.initial_dist: dict[Hashable, float] = {}
        self.best_progress: dict[Hashable, float] = {}
        self.min_dist: dict[Hashable, float] = {}
        self.fired = False
        self.trigger_step: int | None = None
        self.trigger_target: Hashable | None = None
        self.trigger_progress: float | None = None
        self.trigger_dist_m: float | None = None
        self.trigger_moved_m: float | None = None
        self.policy_steps = 0
        self.max_moved_m = 0.0
        self.block_reason = "not_armed"

    def arm(self, tcp_pos, candidates: Mapping[Hashable, Any]) -> None:
        self.clear()
        self.reset_tcp = _vec3(tcp_pos)
        floor = float(self.params.start_dist_floor_m)
        for key, pos in candidates.items():
            d = float(np.linalg.norm(self.reset_tcp - _vec3(pos)))
            self.initial_dist[key] = max(d, floor)
        self.armed = True
        self.block_reason = "min_policy_steps"

    # -- update ----------------------------------------------------------

    def progress_to(self, key: Hashable, dist: float) -> float:
        d0 = self.initial_dist.get(key)
        if d0 is None:
            # First sighting after reset (e.g. an object that entered the
            # eligible band later): its current distance is the baseline.
            d0 = max(float(dist), float(self.params.start_dist_floor_m))
            self.initial_dist[key] = d0
        return float(np.clip(1.0 - float(dist) / max(d0, 1e-6), 0.0, 1.0))

    def update(
        self,
        step: int,
        tcp_pos,
        candidates: Mapping[Hashable, Any],
        *,
        gate_ok: bool = True,
        gate_reason: str = "phase_gate",
    ) -> Hashable | None:
        if self.fired or not self.armed:
            return None
        p = self.params
        step = int(step)
        self.policy_steps = step
        tcp = _vec3(tcp_pos)
        moved = float(np.linalg.norm(tcp - self.reset_tcp))
        self.max_moved_m = max(self.max_moved_m, moved)

        qualifying = []
        for key, pos in candidates.items():
            d = float(np.linalg.norm(tcp - _vec3(pos)))
            prog = self.progress_to(key, d)
            self.best_progress[key] = max(self.best_progress.get(key, 0.0), prog)
            self.min_dist[key] = min(self.min_dist.get(key, float("inf")), d)
            if prog >= float(p.trigger_progress) and d <= float(p.commit_radius_m):
                qualifying.append((d, prog, key))

        if step < int(p.min_policy_steps):
            self.block_reason = "min_policy_steps"
            return None
        if moved < float(p.min_moved_m):
            self.block_reason = "robot_not_moved"
            return None
        if not gate_ok:
            self.block_reason = str(gate_reason)
            return None
        if not candidates:
            self.block_reason = "no_eligible_object"
            return None
        if not qualifying:
            self.block_reason = "no_progress_within_commit_radius"
            return None

        qualifying.sort(key=lambda item: item[0])
        d, prog, key = qualifying[0]
        self.fired = True
        self.trigger_step = step
        self.trigger_target = key
        self.trigger_progress = float(prog)
        self.trigger_dist_m = float(d)
        self.trigger_moved_m = moved
        self.block_reason = ""
        return key

    # -- reporting -------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        best = max(self.best_progress.values()) if self.best_progress else 0.0
        min_d = min(self.min_dist.values()) if self.min_dist else None
        return {
            "schema_version": TRIGGER_SCHEMA,
            "fired": bool(self.fired),
            "trigger_step": self.trigger_step,
            "trigger_reason": "progress" if self.fired else None,
            "not_fired_reason": None if self.fired else self.block_reason,
            "trigger_target": (
                None if self.trigger_target is None else _json_key(self.trigger_target)
            ),
            "trigger_progress": self.trigger_progress,
            "trigger_dist_m": self.trigger_dist_m,
            "trigger_tcp_moved_m": self.trigger_moved_m,
            "best_progress": float(best),
            "best_progress_by_target": {
                str(_json_key(k)): float(v) for k, v in self.best_progress.items()
            },
            "min_dist_m": _finite_or_none(min_d),
            "max_tcp_moved_m": float(self.max_moved_m),
            "policy_steps": int(self.policy_steps),
            "params": asdict(self.params),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "armed": bool(self.armed),
            "reset_tcp": None if self.reset_tcp is None else self.reset_tcp.tolist(),
            "initial_dist": [[_json_key(k), float(v)] for k, v in self.initial_dist.items()],
            "best_progress": [[_json_key(k), float(v)] for k, v in self.best_progress.items()],
            "min_dist": [[_json_key(k), float(v)] for k, v in self.min_dist.items()],
            "fired": bool(self.fired),
            "trigger_step": self.trigger_step,
            "trigger_target": (
                None if self.trigger_target is None else _json_key(self.trigger_target)
            ),
            "trigger_progress": self.trigger_progress,
            "trigger_dist_m": self.trigger_dist_m,
            "trigger_moved_m": self.trigger_moved_m,
            "policy_steps": int(self.policy_steps),
            "max_moved_m": float(self.max_moved_m),
            "block_reason": str(self.block_reason),
            "params": asdict(self.params),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = set(self.state_dict())
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                f"Malformed progress-trigger state; missing={missing} extra={extra}"
            )
        self.params = ProgressTriggerParams(**dict(value["params"]))
        self.armed = bool(value["armed"])
        self.reset_tcp = None if value["reset_tcp"] is None else _vec3(value["reset_tcp"])
        self.initial_dist = {k: float(v) for k, v in value["initial_dist"]}
        self.best_progress = {k: float(v) for k, v in value["best_progress"]}
        self.min_dist = {k: float(v) for k, v in value["min_dist"]}
        self.fired = bool(value["fired"])
        self.trigger_step = value["trigger_step"]
        self.trigger_target = value["trigger_target"]
        self.trigger_progress = value["trigger_progress"]
        self.trigger_dist_m = value["trigger_dist_m"]
        self.trigger_moved_m = value["trigger_moved_m"]
        self.policy_steps = int(value["policy_steps"])
        self.max_moved_m = float(value["max_moved_m"])
        self.block_reason = str(value["block_reason"])


# ----------------------------------------------------------------------
# Robot-state helpers shared by family eligibility filters
# ----------------------------------------------------------------------


def robot_tcp(task, arm_tag: str = "right") -> np.ndarray:
    return _vec3(task.robot.get_arm(arm_tag).get_ee_pose()[:3])


def robot_gripper_closed(
    task,
    *,
    arm_tag: str = "right",
    closed_below: float = DEFAULT_HELD_GRIPPER_BELOW,
) -> bool:
    try:
        val = float(task.robot.get_arm(arm_tag).gripper_val)
    except Exception:
        return False
    return val < float(closed_below)


def robot_grasping(
    task,
    obj_pos,
    *,
    radius_m: float = DEFAULT_HELD_TCP_RADIUS_M,
    closed_below: float = DEFAULT_HELD_GRIPPER_BELOW,
    arm_tag: str = "right",
) -> bool:
    """True when the gripper is closed with the TCP within ``radius_m`` of
    ``obj_pos``.  Conservative on purpose: it errs toward "held", which only
    suppresses a trigger or an avatar attach."""
    if not robot_gripper_closed(task, arm_tag=arm_tag, closed_below=closed_below):
        return False
    try:
        tcp = robot_tcp(task, arm_tag)
    except Exception:
        return False
    return float(np.linalg.norm(tcp - _vec3(obj_pos))) < float(radius_m)


# ----------------------------------------------------------------------
# Human sequence
# ----------------------------------------------------------------------


@dataclass
class _AttachGuard:
    motion_name: str
    object_pos: Callable[[], Any]
    grasped: Callable[[], bool]
    reference_pos: np.ndarray
    max_drift_m: float


class InterruptHumanSequence:
    """Glide in -> clip -> idle-pose blend -> glide back -> done.

    Ticked once per policy step by the family controller.  ``start_clip`` is
    a family callback that starts the reach clip (passing
    :attr:`idle_return_frames` as ``return_to_idle_after``) and returns
    ``True`` on success.  The sequence never moves the robot.
    """

    PHASES = ("idle", "glide_in", "clip", "glide_out", "done")

    def __init__(
        self,
        *,
        glide_steps: int = DEFAULT_GLIDE_POLICY_STEPS,
        glide_out_steps: int = DEFAULT_GLIDE_OUT_POLICY_STEPS,
        idle_return_frames: int = DEFAULT_IDLE_RETURN_FRAMES,
        hand_ids: Iterable[int] = (1,),
    ) -> None:
        self.glide_steps = max(1, int(glide_steps))
        self.glide_out_steps = max(1, int(glide_out_steps))
        self.idle_return_frames = max(1, int(idle_return_frames))
        self.hand_ids = tuple(int(h) for h in hand_ids)
        self.phase = "idle"
        self._k = 0
        self._start_root: np.ndarray | None = None
        self._target_root: np.ndarray | None = None
        self._home_root: np.ndarray | None = None
        self._rot: np.ndarray | None = None
        self._start_clip: Callable[[], bool] | None = None
        self._guard: _AttachGuard | None = None
        self.target_key: Any = None
        self.target_pos: np.ndarray | None = None
        self.trigger_step: int | None = None
        self.clip_start_step: int | None = None
        self.clip_end_step: int | None = None
        self.done_step: int | None = None
        self.reach_step: int | None = None
        self.min_hand_target_dist_m: float | None = None
        self.clip_started = False
        self.attach_cancelled_reason: str | None = None
        self.glide_dist_m = 0.0
        # Diagnostics (metrics only, not serialized): robot TCP-object
        # distance / gripper one policy step after the trigger, when the
        # glide ends (clip start), at hand reach and at the attach check.
        self.diag: dict[str, Any] = {}
        self._live_pos: Callable[[], Any] | None = None
        self._near_steps = 0
        self._min_live_tcp_dist: float | None = None

    # -- lifecycle -------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.phase in ("glide_in", "clip", "glide_out")

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def start(
        self,
        *,
        step: int,
        start_root,
        target_root,
        rot,
        target_key: Any,
        target_pos,
        start_clip: Callable[[], bool],
    ) -> None:
        if self.phase != "idle":
            raise RuntimeError(f"interrupt sequence already started ({self.phase})")
        self._start_root = _vec3(start_root)
        self._home_root = self._start_root.copy()
        self._target_root = _vec3(target_root)
        self._rot = np.asarray(rot, dtype=np.float64).copy()
        self._start_clip = start_clip
        self.target_key = target_key
        self.target_pos = _vec3(target_pos)
        self.trigger_step = int(step)
        self.glide_dist_m = float(np.linalg.norm(self._target_root - self._start_root))
        self._k = 0
        self.phase = "glide_in"

    def set_attach_guard(
        self,
        *,
        motion_name: str,
        object_pos: Callable[[], Any],
        grasped: Callable[[], bool],
        max_drift_m: float = DEFAULT_ATTACH_MAX_DRIFT_M,
    ) -> None:
        """Cancel the clip's pending attach if, one sim frame before the
        attach frame, the object is grasped by the robot or has drifted from
        its trigger pose."""
        ref = self.target_pos if self.target_pos is not None else _vec3(object_pos())
        self._live_pos = object_pos
        self._guard = _AttachGuard(
            motion_name=str(motion_name),
            object_pos=object_pos,
            grasped=grasped,
            reference_pos=ref.copy(),
            max_drift_m=float(max_drift_m),
        )

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, 1.0))
        return (1.0 - t) * a + t * b

    def _set_root(self, task, root: np.ndarray) -> None:
        task.avatar.reset(root.copy(), self._rot.copy())

    def tick(self, task, step: int) -> None:
        """Advance one policy step."""
        step = int(step)
        if self.phase == "glide_in":
            self._k += 1
            t = self._k / float(self.glide_steps)
            self._set_root(task, self._lerp(self._start_root, self._target_root, t))
            if self._k >= self.glide_steps:
                self.phase = "clip"
                self.clip_start_step = step
                ok = False
                try:
                    ok = bool(self._start_clip()) if self._start_clip else False
                except Exception as exc:  # pragma: no cover - defensive
                    if getattr(task, "config", {}).get("debug_eval_avatar", False):
                        print(f"[interrupt] clip start failed: {exc}", flush=True)
                    ok = False
                self.clip_started = ok
                if not ok:
                    self._begin_glide_out(task, step)
        elif self.phase == "clip":
            if task.avatar.spare():
                self._begin_glide_out(task, step)
        elif self.phase == "glide_out":
            self._k += 1
            t = self._k / float(self.glide_out_steps)
            self._set_root(task, self._lerp(self._start_root, self._home_root, t))
            if self._k >= self.glide_out_steps:
                self.phase = "done"
                self.done_step = step
        if self.phase in ("glide_in", "clip") and self._k == 1 and "after_trigger" not in self.diag:
            self._snap(task, "after_trigger", step)
        if self.phase == "clip" and "clip_start" not in self.diag:
            self._snap(task, "clip_start", step)
        reach_before = self.reach_step
        self._observe_hands(task, step)
        if reach_before is None and self.reach_step is not None:
            self._snap(task, "hand_reach", step)
        self._track_near(task)

    def _robot_state(self, task) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            arm = task.robot.get_arm("right")
            tcp = _vec3(arm.get_ee_pose()[:3])
            out["gripper_cmd"] = float(arm.gripper_val)
        except Exception:
            return out
        try:
            q = arm.entity.get_qpos()
            q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
            q = np.asarray(q, dtype=np.float64).ravel()
            idx = list(getattr(arm, "_finger_dof_indices", []) or [])
            if idx:
                out["finger_width_m"] = float(sum(q[i] for i in idx))
        except Exception:
            pass
        pos = None
        try:
            pos = _vec3(self._live_pos()) if self._live_pos is not None else self.target_pos
        except Exception:
            pos = self.target_pos
        if pos is not None:
            out["tcp_obj_dist_m"] = float(np.linalg.norm(tcp - pos))
            out["tcp_obj_xy_m"] = float(np.linalg.norm(tcp[:2] - pos[:2]))
            out["tcp_above_obj_m"] = float(tcp[2] - pos[2])
            if self.target_pos is not None:
                out["obj_drift_m"] = float(np.linalg.norm(pos - self.target_pos))
        return out

    def _snap(self, task, label: str, step: int) -> None:
        rec = self._robot_state(task)
        rec["step"] = int(step)
        self.diag[label] = rec

    def _track_near(self, task) -> None:
        if self.phase not in ("glide_in", "clip"):
            return
        rec = self._robot_state(task)
        d = rec.get("tcp_obj_dist_m")
        if d is None:
            return
        if self._min_live_tcp_dist is None or d < self._min_live_tcp_dist:
            self._min_live_tcp_dist = d
        if d < 0.10:
            self._near_steps += 1

    def _begin_glide_out(self, task, step: int) -> None:
        self.clip_end_step = step
        self.phase = "glide_out"
        self._k = 0
        self._start_root = _vec3(task.avatar.robot.global_trans)
        self._guard = None

    def _observe_hands(self, task, step: int) -> None:
        if self.target_pos is None or self.phase in ("idle", "done"):
            return
        try:
            palms = [
                _vec3(task.avatar.robot.get_palm_center(h)) for h in self.hand_ids
            ]
        except Exception:
            return
        d = min(float(np.linalg.norm(p - self.target_pos)) for p in palms)
        if self.min_hand_target_dist_m is None or d < self.min_hand_target_dist_m:
            self.min_hand_target_dist_m = d
        if self.reach_step is None and d <= WARNING_REACH_RADIUS_M:
            self.reach_step = step

    def on_sim_step(self, task) -> None:
        """Per-sim-step attach guard (cheap unless the attach is imminent)."""
        guard = self._guard
        if guard is None or self.phase != "clip":
            return
        motion = task.avatar.motion_modules.get(guard.motion_name)
        if motion is None or getattr(motion, "_attach_obj", None) is None:
            return
        attach_frame = int(getattr(motion, "_attach_frame", 0))
        if int(getattr(motion, "at_frame", 0)) + 1 < attach_frame:
            return
        self.diag["attach"] = dict(self._robot_state(task), sim_frame=int(getattr(motion, "at_frame", 0)))
        reason = None
        try:
            if guard.grasped():
                reason = "robot_grasping"
            else:
                drift = float(np.linalg.norm(_vec3(guard.object_pos()) - guard.reference_pos))
                if drift > guard.max_drift_m:
                    reason = "object_moved"
        except Exception:
            reason = "object_state_unavailable"
        if reason is not None:
            motion._attach_obj = None
            # Nothing was attached, so nothing to detach either.
            motion._detach_frame = None
            self.attach_cancelled_reason = reason
        self._guard = None

    # -- reporting -------------------------------------------------------

    def metrics(self, *, policy_dt_s: float = 0.05) -> dict[str, Any]:
        warning_s = None
        if self.reach_step is not None and self.trigger_step is not None:
            warning_s = (self.reach_step - self.trigger_step) * float(policy_dt_s)
        return {
            "phase": self.phase,
            "target": None if self.target_key is None else _json_key(self.target_key),
            "trigger_step": self.trigger_step,
            "clip_started": bool(self.clip_started),
            "clip_start_step": self.clip_start_step,
            "clip_end_step": self.clip_end_step,
            "returned_to_idle_step": self.done_step,
            "hand_reach_step": self.reach_step,
            "warning_s": warning_s,
            "min_hand_target_dist_m": self.min_hand_target_dist_m,
            "attach_cancelled_reason": self.attach_cancelled_reason,
            "glide_dist_m": float(self.glide_dist_m),
            "diag": dict(self.diag),
            "diag_min_live_tcp_obj_dist_m": self._min_live_tcp_dist,
            "diag_near10cm_policy_steps": int(self._near_steps),
        }

    def state_dict(self) -> dict[str, Any]:
        """JSON state for replay; family callbacks are not serialized and a
        restored sequence in ``glide_in`` needs the family to re-provide them
        through :meth:`rebind`."""

        def arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).tolist()

        return {
            "glide_steps": self.glide_steps,
            "glide_out_steps": self.glide_out_steps,
            "idle_return_frames": self.idle_return_frames,
            "hand_ids": list(self.hand_ids),
            "phase": self.phase,
            "k": self._k,
            "start_root": arr(self._start_root),
            "target_root": arr(self._target_root),
            "home_root": arr(self._home_root),
            "rot": arr(self._rot),
            "target_key": None if self.target_key is None else _json_key(self.target_key),
            "target_pos": arr(self.target_pos),
            "trigger_step": self.trigger_step,
            "clip_start_step": self.clip_start_step,
            "clip_end_step": self.clip_end_step,
            "done_step": self.done_step,
            "reach_step": self.reach_step,
            "min_hand_target_dist_m": self.min_hand_target_dist_m,
            "clip_started": bool(self.clip_started),
            "attach_cancelled_reason": self.attach_cancelled_reason,
            "glide_dist_m": float(self.glide_dist_m),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = set(self.state_dict())
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            raise ValueError(
                f"Malformed interrupt-sequence state; missing={missing} extra={extra}"
            )

        def arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64)

        self.glide_steps = int(value["glide_steps"])
        self.glide_out_steps = int(value["glide_out_steps"])
        self.idle_return_frames = int(value["idle_return_frames"])
        self.hand_ids = tuple(int(h) for h in value["hand_ids"])
        if value["phase"] not in self.PHASES:
            raise ValueError(f"Unknown interrupt-sequence phase {value['phase']!r}")
        self.phase = value["phase"]
        self._k = int(value["k"])
        self._start_root = arr(value["start_root"])
        self._target_root = arr(value["target_root"])
        self._home_root = arr(value["home_root"])
        self._rot = arr(value["rot"])
        self.target_key = value["target_key"]
        self.target_pos = arr(value["target_pos"])
        self.trigger_step = value["trigger_step"]
        self.clip_start_step = value["clip_start_step"]
        self.clip_end_step = value["clip_end_step"]
        self.done_step = value["done_step"]
        self.reach_step = value["reach_step"]
        self.min_hand_target_dist_m = value["min_hand_target_dist_m"]
        self.clip_started = bool(value["clip_started"])
        self.attach_cancelled_reason = value["attach_cancelled_reason"]
        self.glide_dist_m = float(value["glide_dist_m"])

    def rebind(self, start_clip: Callable[[], bool] | None) -> None:
        self._start_clip = start_clip


DEFAULT_GLIDE_MIN_S = 1.5


def glide_steps_for_task(task, default: int = DEFAULT_GLIDE_POLICY_STEPS) -> int:
    """Glide length in policy steps.  An explicit config/class override wins;
    otherwise the glide lasts at least ``DEFAULT_GLIDE_MIN_S`` seconds at the
    task's policy period (30 steps at 20 Hz, 60 at 40 Hz for dt=0.001
    tasks)."""
    config = getattr(task, "config", None) or {}
    value = config.get("interrupt_glide_policy_steps")
    if value is None:
        value = getattr(task, "INTERRUPT_GLIDE_POLICY_STEPS", None)
    if value is not None:
        return max(1, int(value))
    min_s = float(getattr(task, "INTERRUPT_GLIDE_MIN_S", DEFAULT_GLIDE_MIN_S))
    dt = policy_dt_s(task)
    steps = int(np.ceil(min_s / dt - 1e-9)) if dt > 0 else int(default)
    return max(1, steps)


def new_human_sequence(task, *, hand_ids: Iterable[int] = (1,)) -> InterruptHumanSequence:
    config = getattr(task, "config", None) or {}
    return InterruptHumanSequence(
        glide_steps=glide_steps_for_task(task),
        glide_out_steps=int(config.get(
            "interrupt_glide_out_policy_steps",
            getattr(task, "INTERRUPT_GLIDE_OUT_POLICY_STEPS", DEFAULT_GLIDE_OUT_POLICY_STEPS),
        )),
        idle_return_frames=int(config.get(
            "interrupt_idle_return_frames",
            getattr(task, "INTERRUPT_IDLE_RETURN_FRAMES", DEFAULT_IDLE_RETURN_FRAMES),
        )),
        hand_ids=hand_ids,
    )


def burn_legacy_trigger_draw(task) -> int:
    """Consume the global-RNG draw the retired random-step trigger made.

    The scene and inspect-pool draws that follow keep their release-v1
    values per seed.  The bounds are the same ones the legacy draw used.
    """
    config = getattr(task, "config", None) or {}
    lo = int(config.get("eval_trigger_step_min", getattr(task, "EVAL_TRIGGER_STEP_MIN", 0)))
    hi = int(config.get("eval_trigger_step_max", getattr(task, "EVAL_TRIGGER_STEP_MAX", lo)))
    hi = max(lo, hi)
    return int(np.random.randint(lo, hi + 1))


# ----------------------------------------------------------------------
# Episode-level glue (stored on the task as ``task._interrupt``)
# ----------------------------------------------------------------------


@dataclass
class InterruptEpisode:
    """Per-episode interrupt state shared by every family.

    ``policy_steps`` counts eval ticks (one per policy step, or one per
    ``action_substeps`` sim steps for a scripted expert).
    """

    trigger: ProgressTrigger
    human: InterruptHumanSequence
    policy_steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return bool(self.trigger.fired)

    @property
    def human_active(self) -> bool:
        return bool(self.human.active)

    def metrics(self, *, policy_dt_s: float = 0.05) -> dict[str, Any]:
        out = self.trigger.metrics()
        out["human"] = self.human.metrics(policy_dt_s=policy_dt_s)
        out["eval_policy_steps"] = int(self.policy_steps)
        out.update(self.extra)
        return out


def new_interrupt_episode(task, *, hand_ids: Iterable[int] = (1,)) -> InterruptEpisode:
    episode = InterruptEpisode(
        trigger=ProgressTrigger(ProgressTriggerParams.for_task(task)),
        human=new_human_sequence(task, hand_ids=hand_ids),
    )
    task._interrupt = episode
    return episode


def interrupt_episode(task) -> InterruptEpisode | None:
    episode = getattr(task, "_interrupt", None)
    return episode if isinstance(episode, InterruptEpisode) else None


def interrupt_fired(task) -> bool:
    episode = interrupt_episode(task)
    return bool(episode is not None and episode.fired)


def interrupt_human_active(task) -> bool:
    episode = interrupt_episode(task)
    return bool(episode is not None and episode.human_active)


def policy_dt_s(task) -> float:
    config = getattr(task, "config", None) or {}
    substeps = int(config.get("action_substeps", 25) or 25)
    try:
        dt = float(task._sim_dt())
    except Exception:
        dt = 0.002
    return substeps * dt


def interrupt_metrics(task) -> dict[str, Any]:
    """Uniform ``interrupt`` metrics block for ``evaluate()``.

    Present for every interrupt task in eval mode; ``fired`` is False with a
    ``not_fired_reason`` when the robot never committed to an object.
    """
    episode = interrupt_episode(task)
    if episode is None:
        return {
            "schema_version": TRIGGER_SCHEMA,
            "fired": False,
            "trigger_step": None,
            "trigger_reason": None,
            "not_fired_reason": "interrupt_contract_inactive",
        }
    return episode.metrics(policy_dt_s=policy_dt_s(task))


def legacy_pace_block(task) -> dict[str, Any]:
    """Release-v1 ``pace_interrupt`` key surface, filled from the new trigger.

    Downstream aggregation scripts read ``pace_interrupt``; ``trigger_mode``
    is always ``"progress"`` now and distances are 3-D.
    """
    m = interrupt_metrics(task)
    return {
        "enabled": True,
        "trigger_mode": "progress",
        "fired": bool(m.get("fired", False)),
        "trigger_step": m.get("trigger_step"),
        "trigger_reason": m.get("trigger_reason"),
        "trigger_progress": m.get("trigger_progress"),
        "trigger_dist_m": m.get("trigger_dist_m"),
        "best_progress": float(m.get("best_progress", 0.0) or 0.0),
        "target_idx": m.get("trigger_target"),
        "not_fired_reason": m.get("not_fired_reason"),
    }


# ----------------------------------------------------------------------
# Scripted-expert support
# ----------------------------------------------------------------------


class InterruptYield(BaseException):
    """Unwinds a scripted expert motion when the human interrupt fires.

    Subclasses ``BaseException`` so the ``except Exception`` blocks inside
    planners and grasp helpers cannot swallow it.
    """


class ScriptedRolloutDriver:
    """Tick a task's eval-mode interaction at policy cadence during a
    scripted rollout, optionally raising :class:`InterruptYield` once when
    the interrupt fires (full-state expert only)."""

    def __init__(self, task, *, tick_interval: int, force_policy_ticks: bool = False):
        self.task = task
        self.tick_interval = max(1, int(tick_interval))
        self.force_policy_ticks = bool(force_policy_ticks)
        self.yield_armed = False
        self.yield_raised = False
        self.ticks = 0
        self._sim = 0
        self._orig = None

    def tick(self) -> None:
        task = self.task
        if not bool(getattr(task, "config", {}).get("eval_mode", False)):
            return
        if getattr(task, "avatar", None) is None:
            return
        self.ticks += 1
        controller = task.interaction_controller
        if controller.handles_policy_ticks:
            if controller.handles_scripted_rollout_ticks or self.force_policy_ticks:
                task.advance_interaction_policy_step()
            return
        controller.on_scripted_rollout_tick(task)

    def _driven_step_sim(self) -> None:
        self._orig()
        self._sim += 1
        if self._sim % self.tick_interval != 0:
            return
        self.tick()
        if self.yield_armed and not self.yield_raised and interrupt_fired(self.task):
            self.yield_raised = True
            self.yield_armed = False
            raise InterruptYield()

    def __enter__(self) -> "ScriptedRolloutDriver":
        self._orig = self.task.step_sim
        self.task.step_sim = self._driven_step_sim
        return self

    def __exit__(self, *exc) -> None:
        self.task.step_sim = self._orig
        return None


def run_full_state_interrupt_expert(task, driver: ScriptedRolloutDriver) -> bool:
    """Privileged expert under the interrupt contract.

    ``task.interrupt_expert_attempt()`` must be an idempotent scripted
    rollout that re-reads object poses and skips finished work.
    ``task.interrupt_expert_retreat()`` moves the arm clear with mplib (the
    avatar is in the planner obstacle cloud).  After the yield the expert
    waits for the human to return to idle, then resumes.
    """
    driver.yield_armed = True
    try:
        return bool(task.interrupt_expert_attempt())
    except InterruptYield:
        pass
    driver.yield_armed = False
    episode = interrupt_episode(task)
    if episode is not None:
        episode.extra["expert_yielded_at_tick"] = int(driver.ticks)
    try:
        task.interrupt_expert_retreat()
    except InterruptYield:  # pragma: no cover - yield raises once only
        pass
    config = getattr(task, "config", None) or {}
    cap = int(config.get("interrupt_expert_wait_max_sim_steps", DEFAULT_EXPERT_WAIT_MAX_SIM_STEPS))
    n = 0
    while interrupt_human_active(task) and n < cap:
        task.step_sim()
        n += 1
    for _ in range(int(getattr(task, "INTERRUPT_EXPERT_POST_HUMAN_SETTLE_STEPS",
                               DEFAULT_EXPERT_POST_HUMAN_SETTLE_STEPS))):
        task.step_sim()
    if episode is not None:
        episode.extra["expert_wait_sim_steps"] = int(n)
    return bool(task.interrupt_expert_attempt())
