"""Composable human interaction for the stack-bowls-three family.

The robot objective (rim-grasp each non-base bowl and nest it on the base)
stays on the task classes.  These controllers own the human side:

* assist — the shared candidate-bowl layout, the human's uniformly random
  bowl choice, the avatar's natural pick-and-place onto the stack, and the
  eval-mode trigger (fires at reset unless the delayed opt-in is set);
* interrupt — the seed-tied interrupted-bowl / stack-layer choice, the
  two-hand ``Bowl_clip`` dry-run calibration, the reset-order hook, and the
  shared eval-mode interrupt contract (shared progress trigger -> avatar
  glide -> two-hand bowl motion -> idle return; ``interrupt_progress``).  The
  policy keeps control of the robot throughout.

Release-v1 task-facing attributes remain the single state owner; draw
orders are reproduced verbatim.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..utils import to_numpy
from .base import InterruptInteractionController
from .eval_mode import AssistEvalModeController, delayed_assist_trigger_step
from .interrupt_progress import (
    burn_legacy_trigger_draw,
    interrupt_episode,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    robot_tcp,
)


# ----------------------------------------------------------------------
# Assist
# ----------------------------------------------------------------------


class StackBowlsThreeAssistController(AssistEvalModeController):
    """Human and robot share one pool of candidate bowls (simple 50-50).

    Every non-base bowl is a shared candidate in one band both agents reach.
    The human picks ONE candidate uniformly at random (seeded layout RNG) and
    goes for it at reset, no matter what the robot does: no free/claimed
    checks, no retarget, no waiting.  The robot side lives on the task class.
    """

    rng_namespace = "stack_bowls_three.assist.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    def __init__(
        self,
        *,
        region_x: tuple[float, float],
        region_y: tuple[float, float],
        min_separation: float,
    ) -> None:
        self.region_x = (float(region_x[0]), float(region_x[1]))
        self.region_y = (float(region_y[0]), float(region_y[1]))
        self.min_separation = float(min_separation)

    # -- layout + per-episode state --------------------------------------

    @staticmethod
    def shared_region(task) -> tuple[tuple[float, float], tuple[float, float]]:
        """Band holding every candidate bowl: the overlap of the human's
        natural reach and the robot's comfortable rim-grasp reach."""
        if int(getattr(task, "_num_bowls", 3)) > 3:
            return tuple(task.HARD_SHARED_X), tuple(task.HARD_SHARED_Y)
        return tuple(task.SHARED_X), tuple(task.SHARED_Y)

    @staticmethod
    def candidate_ok(task, x: float, y: float) -> bool:
        """Within the robot's comfortable rim-grasp reach and clear of the
        robot hand's home pose (a human reach there hits the idle arm)."""
        mx, my = task.ROBOT_MOUNT_XY
        hx, hy = task.ROBOT_HOME_TCP_XY
        hard = int(getattr(task, "_num_bowls", 3)) > 3
        clearance = (task.HARD_CANDIDATE_HOME_CLEARANCE_M if hard
                     else task.CANDIDATE_HOME_CLEARANCE_M)
        return (float(np.hypot(x - mx, y - my)) <= task.ROBOT_REACH_M
                and float(np.hypot(x - hx, y - hy)) >= clearance)

    def sample_assist_xy(self, task, rng) -> list[tuple[float, float]]:
        """Base (the goal) in its drop band; every candidate bowl in one
        shared band (no human side / robot side split)."""
        n_bowls = int(getattr(task, "_num_bowls", 3))
        if task.simplified_mode_enabled():
            return [tuple(p) for p in task.SHARED_FIXED_XY[:n_bowls]]
        (x_lo, x_hi), (y_lo, y_hi) = self.shared_region(task)
        (bx_lo, bx_hi), (by_lo, by_hi) = tuple(task.BASE_X), tuple(task.BASE_Y)
        limit = self.min_separation ** 2
        for _ in range(task.ASSIST_SAMPLE_ATTEMPTS):
            pts = []
            ok = True
            for i in range(n_bowls):
                if i == 0:
                    x = float(rng.uniform(bx_lo, bx_hi))
                    y = float(rng.uniform(by_lo, by_hi))
                else:
                    x = float(rng.uniform(x_lo, x_hi))
                    y = float(rng.uniform(y_lo, y_hi))
                    if not self.candidate_ok(task, x, y):
                        ok = False
                        break
                for px, py in pts:
                    if (x - px) ** 2 + (y - py) ** 2 < limit:
                        ok = False
                        break
                if not ok:
                    break
                pts.append((x, y))
            if ok and len(pts) == n_bowls:
                return pts
        fallback = task.HARD_FALLBACK_XY if n_bowls > 3 else task.SHARED_FIXED_XY
        return [tuple(p) for p in fallback[:n_bowls]]

    @staticmethod
    def allocate(task) -> None:
        """Bowl 0 = base; every other bowl is a shared candidate."""
        task._stacked_indices = list(range(1, task._num_bowls))
        task._shared_candidates = list(task._stacked_indices)
        task._human_target = task._stacked_indices[-1]
        task._avatar_bowl_idx = task._human_target
        task._shared_pool = {"human_target": task._human_target,
                             "robot_target": None}

    @staticmethod
    def draw_human_choice(task, rng) -> None:
        """The human's uniformly random pick, drawn from the task's layout
        child RNG (only this task's layout stream)."""
        idx = task._stacked_indices[int(rng.randint(len(task._stacked_indices)))]
        task._human_target = idx
        task._avatar_bowl_idx = idx
        task._shared_pool["human_target"] = idx

    # -- avatar motion ---------------------------------------------------

    @staticmethod
    def start_avatar_assist(task) -> bool:
        if task.avatar is None or any(b is None for b in task.bowls):
            return False
        idx = task._human_target
        bowl = task.bowls[idx]
        # Pick: palm just above the rim (mesh origin = bowl bottom).
        origin = to_numpy(bowl.entity.get_pos()).ravel()[:3]
        pick_pos = origin + np.array([0.0, 0.0, task.AVATAR_PICK_LIFT])
        # Drop: above the base bowl; gravity nests the bowl.
        base_xyz = task._bowl_xyz(0)
        drop_z = (task.TABLE_TOP_Z + task.AVATAR_DROP_BOWLS_ABOVE * task._bowl_height
                  + task.AVATAR_DROP_CLEARANCE_M)
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=np.array([
                base_xyz[0] + float(task.AVATAR_DROP_XY_OFFSET[0]),
                base_xyz[1] + float(task.AVATAR_DROP_XY_OFFSET[1]),
                drop_z,
            ]),
            attach_obj=bowl.entity,
            hand_id=0,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=task.AVATAR_BODY_MARGIN,
            body_x_bounds=task.AVATAR_BODY_X_BOUNDS,
            body_y_bounds=(task.AVATAR_MIN_BODY_Y, task.AVATAR_MAX_BODY_Y),
            yaw_limit_deg=task.AVATAR_YAW_LIMIT_DEG,
        )
        task.avatar_collided = False
        task.avatar_collision_log = []
        return True

    # -- eval-mode hooks -------------------------------------------------

    def eval_reset(self, task) -> None:
        delayed = delayed_assist_trigger_step(task)
        task._eval_assist_failed = False
        if delayed is not None:
            task._eval_assist_trigger_step = delayed
            task._eval_assist_fired = False
            return
        task._eval_assist_trigger_step = 0
        task._eval_assist_fired = True
        if not task._start_avatar_assist():
            task._eval_assist_failed = True

    def eval_step(self, task, step_idx: int) -> None:
        if (getattr(task, "_eval_assist_fired", False)
                or getattr(task, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(
            task, "_eval_assist_trigger_step", task.EVAL_TRIGGER_STEP_MIN,
        )):
            return
        task._eval_assist_fired = True
        if not task._start_avatar_assist():
            task._eval_assist_failed = True

    @staticmethod
    def shared_pool_metrics(task) -> dict:
        pool = dict(getattr(task, "_shared_pool", None) or {})
        human, robot = pool.get("human_target"), pool.get("robot_target")
        pool["same_target"] = bool(human is not None and robot is not None and human == robot)
        return pool

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        human = getattr(task, "_human_target", None)
        return {
            "eval_mode": bool(task.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                task, "_eval_assist_trigger_step", -1,
            )),
            "eval_assist_fired": bool(getattr(task, "_eval_assist_fired", False)),
            "eval_assist_failed": bool(getattr(task, "_eval_assist_failed", False)),
            "eval_policy_step_count": int(getattr(
                task, "_eval_policy_step_count", 0,
            )),
            "assist_avatar_bowl_idx": -1 if human is None else int(human),
            "shared_pool": self.shared_pool_metrics(task),
        }


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class StackBowlsThreeInterruptController(InterruptInteractionController):
    """Human picks up and inspects the bowl the robot is reaching for with
    a two-hand motion, then returns to idle.  The robot is never overridden.

    ``handles_policy_ticks`` stays ``False``: the task's ``take_action``
    ticks this machine before applying the unmodified policy action.
    """

    rng_namespace = "stack_bowls_three.interrupt.v1"
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def __init__(
        self,
        *,
        sample_attempts: int,
        fallback_xy: Sequence[tuple[float, float]],
        base_x: tuple[float, float],
        base_y: tuple[float, float],
        lift_x: tuple[float, float],
        lift_y: tuple[float, float],
        reach_x: tuple[float, float],
        reach_y: tuple[float, float],
        min_separation: float,
    ) -> None:
        self.sample_attempts = int(sample_attempts)
        self.fallback_xy = [tuple(map(float, xy)) for xy in fallback_xy]
        self.base_x = tuple(map(float, base_x))
        self.base_y = tuple(map(float, base_y))
        self.lift_x = tuple(map(float, lift_x))
        self.lift_y = tuple(map(float, lift_y))
        self.reach_x = tuple(map(float, reach_x))
        self.reach_y = tuple(map(float, reach_y))
        self.min_separation = float(min_separation)

    # -- lifecycle -------------------------------------------------------

    def on_episode_seeded(self, task, seed: int) -> None:
        # The interrupted bowl / stack layer are tied to the episode seed
        # (not the global stream after robot/camera setup) so seed sweeps
        # cover both interruption timings reproducibly.
        task._episode_seed = int(seed)

    def on_scripted_rollout_tick(self, task) -> None:
        task._eval_step_avatar()

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        # Reset placement is unchanged: the idle human stands calibrated to
        # the seed-chosen bowl (a reset-time pose, not a teleport mid-episode).
        task._calibrate_avatar_for_bowl_motion()
        task._eval_policy_step_count = 0
        task._eval_interrupt_fired = False
        task._eval_interrupt_failed = False
        task._eval_trigger_step = None
        if bool(task.config.get("eval_mode", False)):
            # The release-v1 random-step trigger drew here; keep the draw so
            # the global RNG tail per seed is unchanged.
            task._eval_trigger_step = burn_legacy_trigger_draw(task)
            self.init_eval_state(task)

    def init_eval_state(self, task) -> None:
        task._eval_bowl_frame_default = task._dryrun_capture_bowl_frame()
        episode = new_interrupt_episode(task, hand_ids=(0, 1))
        episode.trigger.arm(robot_tcp(task), self.eligible_bowls(task))
        if task._eval_bowl_frame_default is None:
            task._eval_interrupt_fired = True
            task._eval_interrupt_failed = True
            episode.extra["calibration_failed"] = True
            episode.trigger.armed = False
            episode.trigger.block_reason = "avatar_calibration_failed"

    # -- episode planning ------------------------------------------------

    @staticmethod
    def choose_interrupt_plan(task) -> None:
        """Interrupted bowl in {1, 2} and stack layer in {2, 3}, drawn from a
        seed-tied stream; the simplified layout fixes both."""
        seed = int(getattr(task, "_episode_seed", 0))
        rng_top = np.random.RandomState(seed + task.INTERRUPT_SEED_OFFSET)
        if task.simplified_mode_enabled():
            task._interrupted_idx = 1
            task._interrupt_stack_layer = 2
        else:
            task._interrupted_idx = int(rng_top.choice([1, 2]))
            task._interrupt_stack_layer = int(rng_top.choice([2, 3]))

    def sample_bowl_xy(self, rng, n_bowls: int) -> list[tuple[float, float]]:
        """Family layout with both non-base bowls in the avatar-reach band."""
        limit = self.min_separation ** 2
        for _ in range(self.sample_attempts):
            pts = []
            ok = True
            for slot in range(n_bowls):
                if slot == 0:
                    x = float(rng.uniform(*self.base_x))
                    y = float(rng.uniform(*self.base_y))
                else:
                    x = float(rng.uniform(*self.lift_x))
                    y = float(rng.uniform(*self.lift_y))
                for px, py in pts:
                    if (x - px) ** 2 + (y - py) ** 2 < limit:
                        ok = False
                        break
                if not ok:
                    break
                pts.append((x, y))
            if ok and len(pts) == n_bowls:
                return pts
        fallback = list(self.fallback_xy)
        fallback[1] = (self.reach_x[0] + 0.05, self.reach_y[1] - 0.03)
        fallback[2] = (self.reach_x[1] - 0.05, self.reach_y[0] + 0.03)
        return fallback[:n_bowls]

    # -- two-hand bowl frame calibration ----------------------------------

    @staticmethod
    def attach_frame_scaled(task) -> int:
        return int(round(task.BOWL_ATTACH_FRAME * task.AVATAR_MOTION_SLOW))

    @staticmethod
    def detach_frame_scaled(task) -> int:
        return int(round(task.BOWL_DETACH_FRAME * task.AVATAR_MOTION_SLOW))

    @staticmethod
    def bowl_frame_center(task) -> np.ndarray:
        center, _rot = task.avatar.robot._get_two_hand_frame(
            task.BOWL_TWO_HAND_OFFSET
        )
        return np.asarray(center, dtype=np.float64).copy()

    def dryrun_capture_bowl_frame(self, task):
        """Play Bowl_clip at the home pose, capture the two-hand frame at the
        (scaled) attach frame, restore home.  Cached per task class."""
        cache_key = (
            task.BOWL_MOTION_NAME,
            float(task.AVATAR_MOTION_SLOW),
            tuple(np.asarray(task.BOWL_TWO_HAND_OFFSET, dtype=np.float64).ravel()),
            tuple(np.asarray(task.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(task), "_bowl_frame_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        task.avatar.play_bowl_motion(snap_to_hands=False)
        bowl_frame_default = None
        step = 0
        attach_step = task._attach_frame_scaled
        while step <= attach_step and not task.avatar.spare():
            task.step_sim()
            if step == attach_step:
                bowl_frame_default = task._bowl_frame_center()
                break
            step += 1
        task.avatar.reset(
            np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(task.DRYRUN_RESET_SETTLE_STEPS):
            task.step_sim()
        task.avatar_collided = False
        task.avatar_collision_log = []
        if bowl_frame_default is not None:
            cache = getattr(type(task), "_bowl_frame_default_cache", None)
            if cache is None:
                cache = {}
                type(task)._bowl_frame_default_cache = cache
            cache[cache_key] = bowl_frame_default.copy()
        return bowl_frame_default

    def calibrate_avatar(self, task) -> None:
        """Shift the avatar so the two-hand frame meets the interrupted bowl."""
        bowl = task.bowls[task._interrupted_idx]
        target_frame = to_numpy(bowl.entity.get_pos()).ravel()[:3].astype(float)

        bowl_frame_default = task._dryrun_capture_bowl_frame()
        if bowl_frame_default is None:
            return

        shift = target_frame - bowl_frame_default
        shifted_pos = (
            np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        )
        task.avatar.reset(
            shifted_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(task.CALIBRATION_SETTLE_STEPS):
            task.step_sim()

    @staticmethod
    def wait_until_avatar_attach(task) -> None:
        motion = task.avatar.motion_modules.get(task.BOWL_MOTION_NAME)
        if motion is None:
            return
        while (
            not task.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < task._attach_frame_scaled
        ):
            task.step_sim()

    # -- eval-mode interrupt contract ------------------------

    @staticmethod
    def closest_interrupt_bowl_to_ee(task):
        """Legacy helper (xy-closest non-base bowl); not used by the trigger."""
        if not getattr(task, "bowls", None):
            return None, float("inf")
        ee = np.asarray(task.robot.get_arm("right").get_ee_pose()[:3], dtype=float)
        best_i = None
        best_d = float("inf")
        for i in (1, 2):
            if i >= len(task.bowls):
                continue
            p = task._bowl_xyz(i)
            if p[2] < task.TABLE_TOP_Z - 0.02:
                continue
            d = float(np.linalg.norm(ee[:2] - p[:2]))
            if d < best_d:
                best_i = i
                best_d = d
        return best_i, best_d

    @staticmethod
    def bowl_grasp_center(task, idx: int) -> np.ndarray:
        center = np.asarray(task._bowl_xyz(idx), dtype=float).copy()
        center[2] += 0.5 * float(task._bowl_height)
        return center

    @staticmethod
    def bowl_stacked(task, idx: int) -> bool:
        base = np.asarray(task._bowl_xyz(0), dtype=float)
        p = np.asarray(task._bowl_xyz(idx), dtype=float)
        half = task._bowl_xy_half_bounds()
        return bool(
            not np.any(np.abs(p[:2] - base[:2]) > half)
            and float(p[2]) - float(base[2]) > task.SUCCESS_DZ_MIN
        )

    def bowl_held(self, task, idx: int) -> bool:
        return robot_grasping(
            task, self.bowl_grasp_center(task, idx),
            radius_m=float(task.INTERRUPT_HELD_TCP_RADIUS_M),
        )

    def eligible_bowls(self, task) -> dict:
        """Non-base bowls on the table: not stacked, not held."""
        out = {}
        for i in (1, 2):
            if i >= len(getattr(task, "bowls", []) or []) or task.bowls[i] is None:
                continue
            p = np.asarray(task._bowl_xyz(i), dtype=float)
            if p[2] < task.TABLE_TOP_Z - 0.02:
                continue
            if self.bowl_stacked(task, i) or self.bowl_held(task, i):
                continue
            out[i] = self.bowl_grasp_center(task, i)
        return out

    def robot_holding_any(self, task) -> bool:
        for i in range(len(task.bowls)):
            p = np.asarray(task._bowl_xyz(i), dtype=float)
            if p[2] > task.TABLE_TOP_Z + 0.04 and self.bowl_held(task, i):
                return True
        return False

    def advance_eval_step(self, task) -> None:
        episode = interrupt_episode(task)
        if episode is None:
            return
        episode.policy_steps += 1
        task._eval_policy_step_count = episode.policy_steps
        step = episode.policy_steps
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            return
        if episode.fired or not episode.trigger.armed:
            return
        candidates = self.eligible_bowls(task)
        gate_ok, gate_reason = True, "phase_gate"
        if self.robot_holding_any(task):
            gate_ok, gate_reason = False, "robot_holding_object"
        key = episode.trigger.update(
            step, robot_tcp(task), candidates,
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is None:
            return
        task._eval_interrupt_fired = True
        self.start_human(task, int(key), step)

    def start_human(self, task, idx: int, step: int) -> None:
        """Glide (no teleport) to the root whose two-hand frame meets bowl
        ``idx`` at the attach frame, then play ``Bowl_clip``."""
        episode = interrupt_episode(task)
        bowl = task.bowls[idx]
        bowl_pos = to_numpy(bowl.entity.get_pos()).ravel()[:3].astype(float)
        shift = bowl_pos - np.asarray(task._eval_bowl_frame_default, dtype=np.float64)
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task._interrupted_idx = int(idx)

        def start_clip() -> bool:
            try:
                task.avatar.play_bowl_motion(
                    attach_obj=bowl.entity,
                    attach_frame=task._attach_frame_scaled,
                    detach_frame=task._detach_frame_scaled,
                    return_to_idle_after=episode.human.idle_return_frames,
                    snap_to_hands=False,
                )
            except Exception as exc:
                task._eval_interrupt_failed = True
                if task.config.get("debug_eval_avatar", False):
                    print(f"[stack_bowls_interrupt/eval] clip failed: {exc}", flush=True)
                return False
            episode.human.set_attach_guard(
                motion_name=task.BOWL_MOTION_NAME,
                object_pos=lambda: self.bowl_grasp_center(task, idx),
                grasped=lambda: self.bowl_held(task, idx),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=self.bowl_grasp_center(task, idx),
            start_clip=start_clip,
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = interrupt_episode(task)
        if episode is not None and task.avatar is not None:
            episode.human.on_sim_step(task)

    @staticmethod
    def eval_skip_policy_action(task) -> bool:
        """The policy is never skipped (shim for scripts/eval.py)."""
        return False

    @staticmethod
    def consume_eval_policy_reset_requested(task) -> bool:
        """No env-requested policy reset (shim for scripts/eval.py)."""
        return False

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        out = {
            "eval_mode": bool(task.config.get("eval_mode", False)),
            "eval_policy_step_count": int(getattr(
                task, "_eval_policy_step_count", 0,
            )),
            "eval_interrupt_fired": bool(getattr(
                task, "_eval_interrupt_fired", False,
            )),
            "eval_interrupt_failed": bool(getattr(
                task, "_eval_interrupt_failed", False,
            )),
            "interrupted_bowl_idx": int(getattr(task, "_interrupted_idx", -1)),
        }
        if out["eval_mode"]:
            out["interrupt"] = interrupt_metrics(task)
            out["pace_interrupt"] = legacy_pace_block(task)
        return out
