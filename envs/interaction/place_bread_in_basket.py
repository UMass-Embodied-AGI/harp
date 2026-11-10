"""Composable human interaction for the place-bread-in-basket family.

The robot objective (pick two loaves, place them in the basket) stays on the
task classes.  These controllers own the *human* side of each variant: which
loaf the avatar takes and when its motion starts (assist), or which pick is
interrupted, where the avatar must stand so its palm meets the loaf, and the
shared eval-mode interrupt contract (progress trigger -> glide -> inspect
-> idle return; envs/interaction/interrupt_progress.py) for the interrupt.

Both controllers keep the release-v1 task-facing attributes (``_avatar_bread_idx``,
``_interrupted_idx``, ``_eval_*``, ``_pace_*`` ...) published on the task so
scene-init replay, evaluation metrics, and existing tooling keep reading the
same surface.  Layout sampling reproduces the legacy draw order exactly.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import transforms3d as t3d

from ..utils import to_numpy
from .eval_mode import AssistEvalModeController, delayed_assist_trigger_step
from .inspect_motion import InspectMotionInterruptController
from .interrupt_progress import (
    InterruptEpisode,
    burn_legacy_trigger_draw,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    robot_tcp,
)


def _validated_state_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} cannot be below {minimum}")
    return result


def _optional_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).ravel()
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector.copy()


def bread_world_center(task, bread) -> np.ndarray:
    """Live bbox centre of a loaf in world frame.

    ``entity.get_pos()`` returns the mesh origin; the bbox centre sits at
    ``mesh_origin + R(q) @ (center_offset_mesh * scale)``.
    """

    origin = to_numpy(bread.entity.get_pos()).ravel()[:3]
    q = to_numpy(bread.entity.get_quat()).ravel()[:4]
    rotation = t3d.quaternions.quat2mat(q)
    offset = rotation @ (task._bread_center_mesh * task._bread_scale)
    return origin + offset


# ----------------------------------------------------------------------
# Assist
# ----------------------------------------------------------------------


class PlaceBreadInBasketAssistController(AssistEvalModeController):
    """Human side of the shared-pool (50-50) bread assist.

    Two loaves and the basket sit in one shared region that both agents can
    reach.  The human's loaf is drawn uniformly at reset from the episode
    stream (the layout RNG keeps its legacy index draw so seeded layouts are
    unchanged).  The human stands at one stance chosen at reset and, once
    triggered (at reset in every mode unless ``eval_assist_trigger_delayed``
    is set), goes for that loaf no matter what: no free/claimed checks, no
    retarget, no waiting.  Only the full-state expert reads the human's
    choice (``scripted_robot_indices``); the blind expert reads none of it.

    The generic eval-mode reset/policy cadence is inherited from
    :class:`AssistEvalModeController` and dispatches through the task-facing
    ``_eval_at_reset`` / ``_eval_at_step`` hooks, which the task delegates
    back here.
    """

    rng_namespace = "place_bread_in_basket.assist.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False
    STATE_SCHEMA = "genesis_hr_bench.place_bread_in_basket_assist_state.v1"

    def __init__(
        self,
        *,
        region_x: tuple[float, float],
        region_y: tuple[float, float],
        min_separation: float,
        sample_attempts: int,
        fallback_positions: Sequence[tuple[float, float]],
        start_delay_steps: int,
        robot_base_xy: tuple[float, float] | None = None,
        robot_reach_annulus: tuple[float, float] | None = None,
        max_spread: float | None = None,
        keepout_xy: tuple[float, float] | None = None,
        keepout_radius: float = 0.0,
        pool_stance_y: float | None = None,
        human_reach: Mapping[str, float] | None = None,
        loaf_y_max: float | None = None,
        pool_layout: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.region_x = (float(region_x[0]), float(region_x[1]))
        self.region_y = (float(region_y[0]), float(region_y[1]))
        self.min_separation = float(min_separation)
        self.sample_attempts = int(sample_attempts)
        if self.sample_attempts < 1:
            raise ValueError("sample_attempts must be positive")
        self.fallback_positions = tuple(
            (float(x), float(y)) for x, y in fallback_positions
        )
        self.start_delay_steps = int(start_delay_steps)
        if self.start_delay_steps < 0:
            raise ValueError("start_delay_steps cannot be negative")
        # Optional robot top-down reach band: every sampled object's xy must
        # lie within [min, max] of the robot base (rejection constraint only;
        # the RNG draw count/order is unchanged).
        self.robot_base_xy = (
            None if robot_base_xy is None
            else (float(robot_base_xy[0]), float(robot_base_xy[1]))
        )
        self.robot_reach_annulus = (
            None if robot_reach_annulus is None
            else (float(robot_reach_annulus[0]), float(robot_reach_annulus[1]))
        )
        # Optional: every pair at most ``max_spread`` apart (one human stance
        # reaches the whole pool) and every object at least
        # ``keepout_radius`` from ``keepout_xy`` (the robot's parked hand).
        self.max_spread = None if max_spread is None else float(max_spread)
        self.keepout_xy = (
            None if keepout_xy is None
            else (float(keepout_xy[0]), float(keepout_xy[1]))
        )
        self.keepout_radius = float(keepout_radius)
        # Shared-pool human: one stance at body y ``pool_stance_y`` facing the
        # table.  ``human_reach`` (keys: shoulder_z, radius, basket_z, loaf_z)
        # makes layout acceptance require that ONE shoulder position at that
        # body y reaches the basket drop and every loaf (nominal geometry,
        # margin below the 0.68 m FABRIK limit).
        self.pool_stance_y = None if pool_stance_y is None else float(pool_stance_y)
        # Loaves (not the basket) stay at y <= loaf_y_max, away from the
        # human's body/arms in the robot's planner cloud.
        self.loaf_y_max = None if loaf_y_max is None else float(loaf_y_max)
        # Structured shared-pool layout (same 2-draws-per-object order as the
        # legacy sampler): basket at (basket_x, basket_y); loaf k at
        # basket_x -/+ offset (alternating sides), y in loaf_y.  The basket
        # sits between the loaves so each agent's reach/carry stays on its
        # own side.
        self.pool_layout = None if pool_layout is None else {
            k: (float(v[0]), float(v[1])) for k, v in dict(pool_layout).items()
        }
        self.human_reach = None if human_reach is None else {
            k: float(v) for k, v in dict(human_reach).items()
        }
        self.avatar_bread_idx: int | None = None
        self.robot_bread_indices: tuple[int, ...] = ()
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = False
        self.eval_assist_failed = False
        self.policy_step_count = 0
        self._reset_shared_pool()

    def _reset_shared_pool(self) -> None:
        # Shared-pool bookkeeping (runtime only; not part of the replay state).
        self.human_target: int | None = None
        self.robot_target: int | None = None
        self.pool_hand: int | None = None

    # -- lifecycle -------------------------------------------------------

    def on_episode_seeded(self, task, seed: int) -> None:
        self.avatar_bread_idx = None
        self.robot_bread_indices = ()
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = False
        self.eval_assist_failed = False
        self.policy_step_count = 0
        self._reset_shared_pool()

    def on_base_reset_ready(self, task) -> None:
        """Adopt release-v1 allocation attributes after scene-init replay,
        then stand the human at its one shared-pool stance."""

        self._adopt_allocation(task)
        breads = getattr(task, "breads", None)
        if breads and self.pool_stance_y is not None:
            # Shared pool: the human's loaf is a fair coin per episode from
            # the controller's private child stream (episode seed + stream
            # name), independent of the layout and of every global draw.
            # Drawing it from the global stream after scene build is biased; over
            # 78 evaluation seeds it gave loaf 0 51 times
            # (p ~ 0.01).  The global draw is still consumed so the global
            # stream after reset is unchanged (the layout RNG's legacy index
            # draw is likewise kept only so seeded layouts are unchanged).
            np.random.randint(len(breads))
            rng = self.episode_rng(task, "human_target")
            human = int(rng.integers(len(breads)))
            self.avatar_bread_idx = human
            self.robot_bread_indices = tuple(
                i for i in range(len(breads)) if i != human
            )
        self._publish_allocation(task)
        self.place_human_at_pool_stance(task)

    def place_human_at_pool_stance(self, task) -> bool:
        """Stand the avatar where one stance and hand reach the human's loaf
        and the basket, so the human never re-stances mid-episode.

        Body y is fixed at ``pool_stance_y`` (torso clear of the table edge),
        facing the table as at init; body x and hand minimise the farthest
        shoulder-to-target distance (live shoulder offsets of this avatar).
        Done at reset, before the first observation (no visible teleport).
        """

        self.pool_hand = None
        avatar = getattr(task, "avatar", None)
        if (avatar is None or not getattr(task, "breads", None)
                or self.pool_stance_y is None):
            return False
        try:
            task._refresh_basket_geom()
            robot = avatar.robot
            base = np.asarray(robot.global_trans, dtype=np.float64).copy()
            rot = np.asarray(robot.global_rot, dtype=np.float64).copy()
            offsets = {}
            for hand, bone in ((0, "LeftArm"), (1, "RightArm")):
                world = np.asarray(
                    robot.skin.get_global_translation(bone)[0], dtype=np.float64,
                ).ravel()[:3]
                offsets[hand] = rot.T @ (world - base)
            # Stance for the human's own loaf and the basket (chosen at reset).
            human = self.avatar_bread_idx
            loaves = (
                task.breads if human is None else [task.breads[int(human)]]
            )
            targets = [task._bread_world_center(b) for b in loaves]
            targets.append(self._place_pos(task))
            xs = [float(t[0]) for t in targets]
            best = None
            for x in np.arange(min(xs) - 0.45, max(xs) + 0.4501, 0.01):
                body = base.copy()
                body[0], body[1] = float(x), self.pool_stance_y
                for hand, off in offsets.items():
                    shoulder = body + rot @ off
                    worst = max(float(np.linalg.norm(t - shoulder)) for t in targets)
                    if best is None or worst < best[0] - 1e-9:
                        best = (worst, body.copy(), hand)
            worst, body, hand = best
            avatar.reset(body, rot)
            collider = getattr(task, "avatar_collider", None)
            if collider is not None:
                collider.update()
        except Exception as exc:  # keep reset alive; fall back to per-pick stance
            print(f"[bread_assist] pool stance failed: {exc}", flush=True)
            return False
        self.pool_hand = int(hand)
        print(f"[bread_assist] pool stance body={np.round(body, 3).tolist()} "
              f"hand={hand} farthest_target={worst:.3f}m", flush=True)
        return True

    def on_policy_tick(self, task, *, step_index: int) -> None:
        super().on_policy_tick(task, step_index=step_index)
        if self._active(task):
            self.policy_step_count = int(
                getattr(task, "_eval_policy_step_count", self.policy_step_count)
            )

    # -- allocation / layout --------------------------------------------

    def plan_positions(self, task, rng) -> list[tuple[float, float]]:
        """Return ``[basket, bread_0, ...]`` xy tuples and allocate loaves.

        Verbatim legacy draw order: a loaf index first (kept so seeded
        layouts are unchanged; it is only the provisional allocation until
        the human actually chooses at start time), then up to
        ``sample_attempts`` rejection-sampled layouts from the one shared
        region, then a deterministic spread.
        """

        n_breads = max(2, int(task._num_breads))
        avatar_idx = int(rng.randint(0, n_breads))
        self.avatar_bread_idx = avatar_idx
        self.robot_bread_indices = tuple(
            i for i in range(n_breads) if i != avatar_idx
        )
        self._publish_allocation(task)

        for _outer in range(self.sample_attempts):
            if self.pool_layout is not None:
                layout = self.pool_layout
                basket = (
                    float(rng.uniform(*layout["basket_x"])),
                    float(rng.uniform(*layout["basket_y"])),
                )
                breads = []
                for k in range(n_breads):
                    offset = float(rng.uniform(*layout["offset"]))
                    y = float(rng.uniform(*layout["loaf_y"]))
                    side = -1.0 if k % 2 == 0 else 1.0
                    breads.append((basket[0] + side * offset, y))
                positions = [basket] + breads
                if self._separated(positions):
                    return positions
                continue
            basket = (
                float(rng.uniform(*self.region_x)),
                float(rng.uniform(*self.region_y)),
            )
            breads = [
                (
                    float(rng.uniform(*self.region_x)),
                    float(rng.uniform(*self.region_y)),
                )
                for _ in range(n_breads)
            ]
            positions = [basket] + breads
            if self._separated(positions):
                return positions
        return list(self.fallback_positions[: 1 + n_breads])

    def _separated(self, positions) -> bool:
        if self.robot_base_xy is not None and self.robot_reach_annulus is not None:
            lo, hi = self.robot_reach_annulus
            bx, by = self.robot_base_xy
            for x, y in positions:
                r = float(np.hypot(x - bx, y - by))
                if r < lo or r > hi:
                    return False
        if not self._one_stance_reaches(positions):
            return False
        if self.loaf_y_max is not None and any(
            y > self.loaf_y_max for _x, y in positions[1:]
        ):
            return False
        if self.keepout_xy is not None:
            kx, ky = self.keepout_xy
            for x, y in positions:
                if float(np.hypot(x - kx, y - ky)) < self.keepout_radius:
                    return False
        limit = self.min_separation ** 2
        spread = None if self.max_spread is None else self.max_spread ** 2
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d2 = dx * dx + dy * dy
                if d2 < limit or (spread is not None and d2 > spread):
                    return False
        return True

    def _one_stance_reaches(self, positions) -> bool:
        """For EACH loaf, some shoulder x (at the stance's y/z) reaches both
        that loaf and the basket drop (the human only ever carries its own
        loaf, from a stance chosen at reset for it)."""
        if self.human_reach is None or self.pool_stance_y is None:
            return True
        return all(
            self._reaches([positions[0], loaf]) for loaf in positions[1:]
        )

    def _reaches(self, positions) -> bool:
        reach = self.human_reach
        lo, hi = -np.inf, np.inf
        for k, (x, y) in enumerate(positions):
            z = reach["basket_z"] if k == 0 else reach["loaf_z"]
            r2 = (reach["radius"] ** 2 - (self.pool_stance_y - y) ** 2
                  - (reach["shoulder_z"] - z) ** 2)
            if r2 < 0.0:
                return False
            half = float(np.sqrt(r2))
            lo, hi = max(lo, x - half), min(hi, x + half)
        return lo <= hi

    def _adopt_allocation(self, task) -> None:
        """Task-facing attributes remain authoritative for the allocation.

        Scene-init replay, tests, and legacy tooling write them directly.
        """

        avatar_idx = getattr(task, "_avatar_bread_idx", None)
        if avatar_idx is not None:
            try:
                self.avatar_bread_idx = int(avatar_idx)
            except (TypeError, ValueError):
                pass
        robot = getattr(task, "_robot_bread_indices", None)
        if robot is None:
            single = getattr(task, "_robot_bread_idx", None)
            robot = None if single is None else [single]
        if robot is not None:
            try:
                self.robot_bread_indices = tuple(int(i) for i in robot)
            except (TypeError, ValueError):
                pass

    def _publish_allocation(self, task) -> None:
        if self.avatar_bread_idx is None:
            return
        task._avatar_bread_idx = int(self.avatar_bread_idx)
        task._robot_bread_indices = [int(i) for i in self.robot_bread_indices]
        if self.robot_bread_indices:
            task._robot_bread_idx = int(self.robot_bread_indices[0])

    def scripted_robot_indices(self, task) -> list[int]:
        """Full-state expert's loaves (privileged): every loaf except the
        human's choice, closest-to-basket first."""

        self._adopt_allocation(task)
        n = len(task.breads)
        human = self.human_target if self.human_target is not None else self.avatar_bread_idx
        if human is not None:
            indices = [i for i in range(n) if i != int(human)]
        else:
            indices = [int(i) for i in self.robot_bread_indices]
        basket_xy = to_numpy(task.basket.entity.get_pos()).ravel()[:2]
        return sorted(
            indices,
            key=lambda i: float(np.linalg.norm(
                to_numpy(task.breads[i].entity.get_pos()).ravel()[:2]
                - basket_xy,
            )),
        )

    # -- shared pool metrics ---------------------------------------------

    def note_robot_target(self, idx: int) -> None:
        if self.robot_target is None:
            self.robot_target = int(idx)

    def shared_pool_metrics(self) -> dict[str, Any]:
        same = (
            self.human_target is not None
            and self.robot_target is not None
            and int(self.human_target) == int(self.robot_target)
        )
        return {
            "human_target": self.human_target,
            "robot_target": self.robot_target,
            "same_target": bool(same),
        }

    # -- avatar motion ---------------------------------------------------

    def _place_pos(self, task) -> np.ndarray:
        basket_pos = to_numpy(task.basket.entity.get_pos()).ravel()[:3]
        return np.array([
            basket_pos[0],
            basket_pos[1],
            task._basket_rim_z + task.AVATAR_DROP_ABOVE_RIM,
        ])

    def start_avatar_assist(self, task, *, add_start_delay: bool) -> None:
        """Human start: go for the loaf drawn at reset, unconditionally."""

        # Measure the basket rim from the live AABB before the avatar aims
        # its drop at the rim.
        task._refresh_basket_geom()
        self._adopt_allocation(task)
        if self.avatar_bread_idx is None:
            raise RuntimeError("assist episode has not been planned")
        target = int(self.avatar_bread_idx)
        self.human_target = target
        avatar_bread = task.breads[target]
        common = dict(
            pick_pos=task._bread_world_center(avatar_bread),
            place_pos=self._place_pos(task),
            attach_obj=avatar_bread.entity,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
        )
        if self.pool_hand is not None:
            # Already standing at the shared-pool stance (set at reset).
            task.avatar.pick_and_place(
                hand_id=self.pool_hand, natural=False, **common,
            )
        else:
            task.avatar.pick_and_place(
                hand_id=None,
                natural=True,
                body_margin=task.AVATAR_NATURAL_BODY_MARGIN,
                yaw_limit_deg=task.AVATAR_NATURAL_YAW_LIMIT_DEG,
                body_y_bounds=task.AVATAR_NATURAL_BODY_Y_BOUNDS,
                settle_steps=task.AVATAR_NATURAL_SETTLE_STEPS,
                **common,
            )
        task.avatar_collided = False
        task.avatar_collision_log = []
        if add_start_delay:
            self.delay_avatar_motion(task, self.start_delay_steps)

    @staticmethod
    def delay_avatar_motion(task, delay_steps: int) -> None:
        """Hold the scheduled pick-and-place at its first frame for N steps.

        Prepends copies of ``frames[0]`` (the FABRIK solution at the live
        palm position, i.e. the rest pose) and shifts the attach/detach
        keyframes so the attach still fires when the hand reaches the loaf.
        """

        if delay_steps <= 0:
            return
        motion = task.avatar.motion_modules.get("pick_and_place")
        if motion is None or not motion.frames:
            return
        hold = motion.frames[0]
        motion.frames = [hold] * int(delay_steps) + motion.frames
        motion.attach_frame += int(delay_steps)
        motion.detach_frame += int(delay_steps)

    def wait_for_scripted_completion(self, task) -> None:
        if task.avatar is None:
            return
        while not task.avatar.spare():
            task.step_sim()
        for _ in range(int(task.AVATAR_FINISH_SETTLE_STEPS)):
            task.step_sim()

    # -- eval-mode state machine ----------------------------------------

    def eval_reset(self, task) -> None:
        """Legacy ``_eval_at_reset``: fire the assist immediately, or arm the
        delayed trigger when ``eval_assist_trigger_delayed`` is set."""

        delayed = delayed_assist_trigger_step(task)
        self.eval_assist_failed = False
        if delayed is not None:
            self.eval_assist_trigger_step = delayed
            self.eval_assist_fired = False
            self._publish_eval_state(task)
            return
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = True
        try:
            task._start_avatar_assist(add_start_delay=False)
        except Exception as exc:
            self.eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(
                    f"[bread_assist/eval] avatar assist failed: {exc}",
                    flush=True,
                )
        self._publish_eval_state(task)

    def eval_step(self, task, step_idx: int) -> None:
        """Legacy ``_eval_at_step``: fire once the trigger step is reached."""

        self.policy_step_count = int(step_idx)
        if self.eval_assist_fired or self.eval_assist_failed:
            self._publish_eval_state(task)
            return
        # Legacy read the trigger from the task attribute, defaulting to
        # EVAL_TRIGGER_STEP_MIN when eval reset never armed it.
        trigger = getattr(task, "_eval_assist_trigger_step", None)
        if trigger is None:
            trigger = task.EVAL_TRIGGER_STEP_MIN
        self.eval_assist_trigger_step = int(trigger)
        if int(step_idx) < int(self.eval_assist_trigger_step):
            self._publish_eval_state(task)
            return
        self.eval_assist_fired = True
        try:
            task._start_avatar_assist(add_start_delay=False)
        except Exception as exc:
            self.eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(
                    f"[bread_assist/eval] avatar assist failed: {exc}",
                    flush=True,
                )
        self._publish_eval_state(task)

    def _publish_eval_state(self, task) -> None:
        self._publish_allocation(task)
        task._eval_assist_trigger_step = int(self.eval_assist_trigger_step)
        task._eval_assist_fired = bool(self.eval_assist_fired)
        task._eval_assist_failed = bool(self.eval_assist_failed)

    # -- metrics / state -------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        """Release-v1 allocation and eval metric keys (task attrs are read
        so replayed or externally written state is reported faithfully)."""

        return {
            "assist_avatar_bread_idx": int(getattr(task, "_avatar_bread_idx", -1)),
            "assist_robot_bread_idx": int(getattr(task, "_robot_bread_idx", -1)),
            "assist_robot_bread_indices": [
                int(i) for i in getattr(task, "_robot_bread_indices", [])
            ],
            "eval_mode": bool(task.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                task, "_eval_assist_trigger_step", -1
            )),
            "eval_assist_fired": bool(getattr(task, "_eval_assist_fired", False)),
            "eval_assist_failed": bool(getattr(task, "_eval_assist_failed", False)),
            "eval_policy_step_count": int(getattr(
                task, "_eval_policy_step_count", 0
            )),
            "shared_pool": self.shared_pool_metrics(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA,
            "avatar_bread_idx": self.avatar_bread_idx,
            "robot_bread_indices": [int(i) for i in self.robot_bread_indices],
            "eval_assist_trigger_step": int(self.eval_assist_trigger_step),
            "eval_assist_fired": bool(self.eval_assist_fired),
            "eval_assist_failed": bool(self.eval_assist_failed),
            "policy_step_count": int(self.policy_step_count),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("bread assist state must be a mapping")
        required = {
            "schema_version",
            "avatar_bread_idx",
            "robot_bread_indices",
            "eval_assist_trigger_step",
            "eval_assist_fired",
            "eval_assist_failed",
            "policy_step_count",
        }
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError("malformed assist state; missing: " + ", ".join(missing))
        if extra:
            raise ValueError("malformed assist state; unknown: " + ", ".join(extra))
        if value["schema_version"] != self.STATE_SCHEMA:
            raise ValueError(
                f"unsupported bread assist state schema: {value['schema_version']!r}"
            )
        avatar_idx = value["avatar_bread_idx"]
        if avatar_idx is not None:
            avatar_idx = _validated_state_int(avatar_idx, name="avatar_bread_idx")
        robot_raw = value["robot_bread_indices"]
        if not isinstance(robot_raw, (list, tuple)):
            raise TypeError("robot_bread_indices must be a list or tuple")
        robot = tuple(
            _validated_state_int(i, name="robot_bread_indices") for i in robot_raw
        )
        if avatar_idx is not None and avatar_idx in robot:
            raise ValueError("robot_bread_indices must exclude avatar_bread_idx")
        if avatar_idx is None and robot:
            raise ValueError("robot_bread_indices require an avatar_bread_idx")
        trigger = _validated_state_int(
            value["eval_assist_trigger_step"], name="eval_assist_trigger_step"
        )
        policy_step_count = _validated_state_int(
            value["policy_step_count"], name="policy_step_count"
        )
        for name in ("eval_assist_fired", "eval_assist_failed"):
            if not isinstance(value[name], bool):
                raise TypeError(f"{name} must be a bool")
        self.avatar_bread_idx = avatar_idx
        self.robot_bread_indices = robot
        self.eval_assist_trigger_step = trigger
        self.eval_assist_fired = value["eval_assist_fired"]
        self.eval_assist_failed = value["eval_assist_failed"]
        self.policy_step_count = policy_step_count

    def restore_task_state(self, task, value: Mapping[str, Any]) -> None:
        self.load_state_dict(value)
        task._interaction_policy_step = int(self.policy_step_count)
        task._eval_policy_step_count = int(self.policy_step_count)
        self._publish_eval_state(task)


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class PlaceBreadInBasketInterruptController(InspectMotionInterruptController):
    """Human inspects the loaf the robot is reaching for, then returns to
    idle.  The robot is never overridden; yielding is the policy's job.

    Owns the interrupted-pick choice, the palm dry-run/calibration that
    stands the avatar so its palm meets the loaf at the attach frame, and
    the eval-mode interrupt (shared progress trigger + human
    sequence).  The task keeps the robot choreography and the
    ``take_action`` tick that drives :meth:`advance_eval_step` before the
    unmodified base action, which is why ``handles_policy_ticks`` stays
    ``False``.
    """

    rng_namespace = "place_bread_in_basket.interrupt.inspect.v1"
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False
    STATE_SCHEMA = "genesis_hr_bench.place_bread_in_basket_interrupt_state.v2"

    def __init__(
        self,
        *,
        region_x: tuple[float, float] | None = None,
        region_y: tuple[float, float] | None = None,
        min_separation: float | None = None,
        fallback_margin: float | None = None,
        sample_attempts: int = 80,
    ) -> None:
        super().__init__()
        self.region_x = None if region_x is None else (
            float(region_x[0]), float(region_x[1])
        )
        self.region_y = None if region_y is None else (
            float(region_y[0]), float(region_y[1])
        )
        self.min_separation = None if min_separation is None else float(
            min_separation
        )
        self.fallback_margin = None if fallback_margin is None else float(
            fallback_margin
        )
        self.sample_attempts = int(sample_attempts)
        if self.sample_attempts < 1:
            raise ValueError("sample_attempts must be positive")
        self.interrupted_pick_ordinal: int | None = None
        self.interrupted_idx: int | None = None
        self.eval_palm_default: np.ndarray | None = None
        self.eval_trigger_step: int | None = None
        self.eval_interrupt_fired = False
        self.eval_target_idx: int | None = None
        self.policy_step_count = 0
        self.episode: InterruptEpisode | None = None

    # -- lifecycle -------------------------------------------------------

    def on_episode_seeded(self, task, seed: int) -> None:
        self.interrupted_pick_ordinal = None
        self.interrupted_idx = None
        self._clear_eval_state()

    def _clear_eval_state(self) -> None:
        self.eval_palm_default = None
        self.eval_trigger_step = None
        self.eval_interrupt_fired = False
        self.eval_target_idx = None
        self.policy_step_count = 0
        self.episode = None

    def on_base_reset_ready(self, task) -> None:
        self._adopt_episode_state(task)
        self._publish_episode_state(task)

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        # Inspect resolution keeps its release-v1 position: after the initial
        # observation is captured, before any avatar calibration.
        super().after_base_reset_observation(task, observation=observation)
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        if bool(task.config.get("eval_mode", False)):
            # Leave the avatar idle; take_action's tick fires the interrupt
            # once the robot makes progress toward a loaf.
            task._init_eval_interrupt_state()
        else:
            # Scripted mode: pre-shift the avatar so its palm meets the
            # interrupted loaf exactly at the (scaled) attach frame.
            task._calibrate_avatar_for_inspect()

    # -- episode planning ------------------------------------------------

    def choose_interrupt_pick_ordinal(self, task, rng) -> int:
        n_breads = max(1, int(getattr(task, "_num_breads", 1)))
        if "interrupt_pick_ordinal" in task.config:
            return int(task.config["interrupt_pick_ordinal"]) % n_breads
        if task.config.get("random_interrupt_target", True):
            return int(rng.randint(0, n_breads))
        return 0

    @staticmethod
    def scripted_pick_order_from_positions(basket_xy, bread_xys) -> list[int]:
        basket = np.asarray(basket_xy, dtype=np.float64)
        bread_xys = [np.asarray(xy, dtype=np.float64) for xy in bread_xys]
        return sorted(
            range(len(bread_xys)),
            key=lambda i: float(np.linalg.norm(bread_xys[i] - basket)),
        )

    def scripted_pick_order(self, task) -> list[int]:
        basket_xy = to_numpy(task.basket.entity.get_pos()).ravel()[:2]
        return sorted(
            range(len(task.breads)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(task.breads[i].entity.get_pos()).ravel()[:2]
                - basket_xy,
            )),
        )

    def plan_positions(self, task, rng) -> list[tuple[float, float]]:
        """Return ``[basket, bread_0, ...]`` and choose the interrupted loaf.

        Both loaves share the avatar-reachable band so the layout does not
        reveal which one the human will inspect.  Verbatim legacy draw order.
        """

        if self.region_x is None or self.region_y is None:
            raise RuntimeError("interrupt controller was created without a layout region")
        interrupted_pick_ordinal = task._choose_interrupt_pick_ordinal(rng)
        self.interrupted_pick_ordinal = int(interrupted_pick_ordinal)
        self.interrupted_idx = None
        self._publish_episode_state(task)

        n_breads = max(1, int(getattr(task, "_num_breads", 1)))
        limit = float(self.min_separation) ** 2
        for _outer in range(self.sample_attempts):
            basket = (
                float(rng.uniform(*self.region_x)),
                float(rng.uniform(*self.region_y)),
            )
            bread_xys = [
                (
                    float(rng.uniform(task.INT_X_LO, task.INT_X_HI)),
                    float(rng.uniform(task.INT_Y_LO, task.INT_Y_HI)),
                )
                for _ in range(n_breads)
            ]
            positions = [basket] + bread_xys
            sep_ok = True
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    if dx * dx + dy * dy < limit:
                        sep_ok = False
                        break
                if not sep_ok:
                    break
            if sep_ok:
                pick_order = task._scripted_pick_order_from_positions(
                    basket, bread_xys,
                )
                self.interrupted_idx = int(pick_order[interrupted_pick_ordinal])
                self._publish_episode_state(task)
                return positions
        # Fall-through: deterministic spread.
        margin = float(self.fallback_margin)
        basket = (
            float(self.region_x[0] + margin),
            float(self.region_y[0] + margin),
        )
        bread_xys = [(
            float(task.INT_X_LO + margin),
            float(task.INT_Y_LO + margin),
        )]
        if n_breads > 1:
            bread_xys.append((
                float(task.INT_X_HI - margin),
                float(task.INT_Y_HI - margin),
            ))
        positions = [basket] + bread_xys
        pick_order = task._scripted_pick_order_from_positions(
            positions[0], bread_xys,
        )
        self.interrupted_idx = int(pick_order[interrupted_pick_ordinal])
        self._publish_episode_state(task)
        return positions

    def _adopt_episode_state(self, task) -> None:
        ordinal = getattr(task, "_interrupted_pick_ordinal", None)
        if ordinal is not None:
            try:
                self.interrupted_pick_ordinal = int(ordinal)
            except (TypeError, ValueError):
                pass
        idx = getattr(task, "_interrupted_idx", None)
        if idx is not None:
            try:
                self.interrupted_idx = int(idx)
            except (TypeError, ValueError):
                pass

    def _publish_episode_state(self, task) -> None:
        if self.interrupted_pick_ordinal is not None:
            task._interrupted_pick_ordinal = int(self.interrupted_pick_ordinal)
        if self.interrupted_idx is not None:
            task._interrupted_idx = int(self.interrupted_idx)

    # -- avatar calibration ---------------------------------------------

    def dryrun_capture_palm(self, task) -> np.ndarray | None:
        """Play the Inspect motion at the avatar's home pose up to the attach
        frame, capture the palm position, then reset the avatar home.

        Cached per task class: the palm trajectory depends only on the motion
        name + frame_ratio + hand + avatar init pose, not on the layout, so
        every reset after the first reuses it (~3 min/seed saved).
        """

        cache_key = (
            task.INSPECT_MOTION_NAME,
            float(task.AVATAR_MOTION_SLOW),
            task.AVATAR_HAND_ID,
            tuple(np.asarray(task.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(task), "_palm_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        task.avatar.play_animation(task.INSPECT_MOTION_NAME)
        palm_default = None
        step = 0
        attach_step = task._attach_frame_scaled
        while step <= attach_step and not task.avatar.spare():
            task.step_sim()
            if step == attach_step:
                palm_default = np.asarray(
                    task.avatar.robot.get_palm_center(task.AVATAR_HAND_ID),
                    dtype=np.float64,
                ).copy()
                break
            step += 1
        task.avatar.reset(
            np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(int(task.DRYRUN_RESET_SETTLE_STEPS)):
            task.step_sim()
        task.avatar_collided = False
        task.avatar_collision_log = []
        if palm_default is not None:
            cache = getattr(type(task), "_palm_default_cache", None)
            if cache is None:
                cache = {}
                type(task)._palm_default_cache = cache
            cache[cache_key] = palm_default.copy()
        return palm_default

    def calibrate_avatar(self, task) -> None:
        """Dry-run + shift the avatar so its palm meets the interrupted loaf."""

        self._adopt_episode_state(task)
        if self.interrupted_idx is None:
            raise RuntimeError("interrupt episode has not been planned")
        bread = task.breads[self.interrupted_idx]
        target_palm = to_numpy(bread.entity.get_pos()).ravel()[:3].astype(float)

        palm_default = task._dryrun_capture_palm()
        if palm_default is None:
            return
        shift = target_palm - palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task.avatar.reset(
            shifted_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(int(task.CALIBRATION_SETTLE_STEPS)):
            task.step_sim()

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        """Capture the palm once, burn the retired trigger draw, and arm the
        shared progress trigger from the reset TCP pose."""

        self._clear_eval_state()
        self.eval_palm_default = task._dryrun_capture_palm()
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        self.episode = episode
        if self.eval_palm_default is None:
            self.eval_interrupt_fired = True
            episode.extra["calibration_failed"] = True
        else:
            # The release-v1 random-step trigger drew here; keep the draw so
            # the global RNG tail per seed is unchanged.
            self.eval_trigger_step = burn_legacy_trigger_draw(task)
            self.eval_interrupt_fired = False
        episode.trigger.arm(robot_tcp(task), self.eligible_breads(task))
        if self.eval_palm_default is None:
            episode.trigger.armed = False
            episode.trigger.block_reason = "avatar_calibration_failed"
        self._publish_eval_state(task)

    def advance_eval_step(self, task) -> None:
        """Per policy step: tick an active human sequence, or decide whether
        the robot has committed to a loaf."""

        episode = self.episode
        if episode is None:
            return
        episode.policy_steps += 1
        self.policy_step_count = episode.policy_steps
        step = episode.policy_steps
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            self._publish_eval_state(task)
            return
        if episode.fired or not episode.trigger.armed:
            self._publish_eval_state(task)
            return
        gate_ok, gate_reason = True, "phase_gate"
        if self.robot_holding_any(task):
            gate_ok, gate_reason = False, "robot_holding_object"
        key = episode.trigger.update(
            step, robot_tcp(task), self.eligible_breads(task),
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is not None:
            self.eval_interrupt_fired = True
            self.start_human(task, int(key), step)
        self._publish_eval_state(task)

    def _glide_target_root(self, task, idx: int) -> np.ndarray:
        # Same palm relation as the scripted calibration (mesh-origin pose).
        bread = task.breads[idx]
        anchor = to_numpy(bread.entity.get_pos()).ravel()[:3].astype(float)
        shift = anchor - self.eval_palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        return np.asarray(task.avatar_init_pos, dtype=np.float64) + shift

    def _make_start_clip(self, task, idx: int):
        episode = self.episode
        bread = task.breads[idx]

        def start_clip() -> bool:
            # The glide already placed the avatar; never let the inspect
            # runtime snap it to a stale motion root.
            task._inspect_motion_root_pos = None
            task._play_inspect_motion(
                attach_obj=bread.entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=lambda: bread_world_center(task, bread),
                grasped=lambda: robot_grasping(
                    task, bread_world_center(task, bread),
                ),
            )
            return True

        return start_clip

    def start_human(self, task, idx: int, step: int) -> None:
        self.eval_target_idx = int(idx)
        self.interrupted_idx = int(idx)
        self.episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=self._glide_target_root(task, idx),
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=bread_world_center(task, task.breads[idx]),
            start_clip=self._make_start_clip(task, idx),
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        if self.episode is not None and getattr(task, "avatar", None) is not None:
            self.episode.human.on_sim_step(task)

    def eligible_breads(self, task) -> dict:
        """Loaves on the table: not in the basket, not held by the robot."""

        basket_xy = to_numpy(task.basket.entity.get_pos()).ravel()[:2]
        out = {}
        for i, bread in enumerate(task.breads):
            p = to_numpy(bread.entity.get_pos()).ravel()[:3]
            in_basket = (
                np.linalg.norm(p[:2] - basket_xy) < task.SUCCESS_DIST_XY_M
                and p[2] < task._basket_rim_z + task.EVAL_BASKET_Z_MARGIN
            )
            if in_basket:
                continue
            if (
                p[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or p[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            center = bread_world_center(task, bread)
            if robot_grasping(task, center):
                continue
            out[i] = center
        return out

    @staticmethod
    def robot_holding_any(task) -> bool:
        for bread in task.breads:
            center = bread_world_center(task, bread)
            if center[2] > task.TABLE_TOP_Z + 0.04 and robot_grasping(task, center):
                return True
        return False

    def closest_table_bread_candidate(self, task):
        """Index and 3-D TCP distance of the closest eligible loaf (legacy
        helper surface; the trigger itself uses :meth:`eligible_breads`)."""

        tcp = robot_tcp(task)
        best_idx, best_d = None, float("inf")
        for i, center in self.eligible_breads(task).items():
            d = float(np.linalg.norm(center - tcp))
            if d < best_d:
                best_idx, best_d = i, d
        return best_idx, best_d

    def _publish_eval_state(self, task) -> None:
        self._publish_episode_state(task)
        task._eval_palm_default = (
            None if self.eval_palm_default is None else self.eval_palm_default.copy()
        )
        if self.eval_trigger_step is not None:
            task._eval_trigger_step = int(self.eval_trigger_step)
        task._eval_interrupt_fired = bool(self.eval_interrupt_fired)
        breads = getattr(task, "breads", None) or []
        task._eval_target_bread = (
            None
            if self.eval_target_idx is None or self.eval_target_idx >= len(breads)
            else breads[self.eval_target_idx]
        )
        task._policy_step_count = int(self.policy_step_count)
        if self.episode is not None:
            task._interrupt = self.episode

    # -- metrics / state -------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        """Uniform ``interrupt`` block plus the release-v1 ``pace_interrupt``
        key surface (eval mode only)."""

        if not bool(task.config.get("eval_mode", False)):
            return {}
        block = legacy_pace_block(task)
        block["policy_steps"] = int(getattr(task, "_policy_step_count", 0))
        return {
            "interrupt": interrupt_metrics(task),
            "pace_interrupt": block,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize controller bookkeeping (not simulator or avatar physics)."""

        def vector(value):
            return None if value is None else [float(v) for v in np.asarray(value).ravel()]

        episode = self.episode
        return {
            "schema_version": self.STATE_SCHEMA,
            "interrupted_pick_ordinal": self.interrupted_pick_ordinal,
            "interrupted_idx": self.interrupted_idx,
            "eval_palm_default": vector(self.eval_palm_default),
            "eval_trigger_step": self.eval_trigger_step,
            "eval_interrupt_fired": bool(self.eval_interrupt_fired),
            "eval_target_idx": self.eval_target_idx,
            "policy_step_count": int(self.policy_step_count),
            "trigger": None if episode is None else episode.trigger.state_dict(),
            "human": None if episode is None else episode.human.state_dict(),
        }

    _STATE_FIELDS = (
        "schema_version",
        "interrupted_pick_ordinal",
        "interrupted_idx",
        "eval_palm_default",
        "eval_trigger_step",
        "eval_interrupt_fired",
        "eval_target_idx",
        "policy_step_count",
        "trigger",
        "human",
    )

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("bread interrupt state must be a mapping")
        required = set(self._STATE_FIELDS)
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError("malformed interrupt state; missing: " + ", ".join(missing))
        if extra:
            raise ValueError("malformed interrupt state; unknown: " + ", ".join(extra))
        if value["schema_version"] != self.STATE_SCHEMA:
            raise ValueError(
                "unsupported bread interrupt state schema: "
                f"{value['schema_version']!r}"
            )

        def optional_int(name, minimum=0):
            raw = value[name]
            return None if raw is None else _validated_state_int(
                raw, name=name, minimum=minimum
            )

        if not isinstance(value["eval_interrupt_fired"], bool):
            raise TypeError("eval_interrupt_fired must be a bool")
        palm = value["eval_palm_default"]
        if palm is not None:
            palm = _optional_vector(palm)
            if palm is None:
                raise ValueError("eval_palm_default must be a finite 3-vector")
        if (value["trigger"] is None) != (value["human"] is None):
            raise ValueError("trigger and human state must be both present or both absent")
        episode = None
        if value["trigger"] is not None:
            from .interrupt_progress import InterruptHumanSequence, ProgressTrigger

            trigger = ProgressTrigger()
            trigger.load_state_dict(value["trigger"])
            human = InterruptHumanSequence()
            human.load_state_dict(value["human"])
            episode = InterruptEpisode(trigger=trigger, human=human)

        self.interrupted_pick_ordinal = optional_int("interrupted_pick_ordinal")
        self.interrupted_idx = optional_int("interrupted_idx")
        self.eval_palm_default = palm
        self.eval_trigger_step = optional_int("eval_trigger_step")
        self.eval_interrupt_fired = value["eval_interrupt_fired"]
        self.eval_target_idx = optional_int("eval_target_idx")
        self.policy_step_count = _validated_state_int(
            value["policy_step_count"], name="policy_step_count"
        )
        if episode is not None:
            episode.policy_steps = self.policy_step_count
        self.episode = episode

    def restore_task_state(self, task, value: Mapping[str, Any]) -> None:
        self.load_state_dict(value)
        # Clip callbacks are not serialized: rebind the pending clip start for
        # a sequence restored mid-glide.
        if (
            self.episode is not None
            and self.episode.human.phase == "glide_in"
            and self.eval_target_idx is not None
            and self.eval_target_idx < len(getattr(task, "breads", None) or [])
        ):
            self.episode.human.rebind(self._make_start_clip(task, self.eval_target_idx))
        self._publish_eval_state(task)
