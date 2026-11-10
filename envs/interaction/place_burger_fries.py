"""Composable human interaction for the burger-and-fries family.

The robot objective (pick burgers/fries, place them on the tray) stays on
the task classes.  These controllers own the human side:

* assist — two SHARED hamburgers both agents can reach; at its start the
  human takes one of them uniformly at random (child RNG) and carries it
  onto the tray with the natural pick-and-place motion, no matter what the
  robot does; plus the eval trigger and the ``shared_pool`` metrics;
* interrupt — the inspect interruption: which loose food the human takes,
  palm dry-run calibration, the reset-order hook that resolves the inspect
  clip, calibrates, then (eval mode) arms the shared progress trigger;
  on trigger the avatar glides to the food the robot is reaching for,
  inspects it and returns to idle.  The robot is never overridden.

Release-v1 task-facing attributes (``_eval_*``, ``_interrupted_food_idx``)
remain the single state owner for this family; the controllers operate on
them directly and reproduce the legacy draw order.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..utils import to_numpy
from .eval_mode import AssistEvalModeController, delayed_assist_trigger_step
from .inspect_motion import InspectMotionInterruptController
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


class PlaceBurgerFriesAssistController(AssistEvalModeController):
    """Human takes one of the shared hamburgers (50-50) onto the tray."""

    rng_namespace = "place_burger_fries.assist.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    # -- avatar pose helpers -------------------------------------------

    @staticmethod
    def avatar_init_pos_lifted(task) -> np.ndarray:
        p = np.array(task.avatar_init_pos, dtype=float).copy()
        p[2] += task.AVATAR_LIFT_Z
        return p

    @staticmethod
    def avatar_rot_with_yaw(task) -> np.ndarray:
        yaw = np.deg2rad(task.AVATAR_YAW_DEG)
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64
        )
        return Rz @ np.array(task.AVATAR_BASE_ROT, dtype=np.float64)

    @staticmethod
    def avatar_palm_center(task, hand_id: int) -> np.ndarray:
        side = "Left" if hand_id == 0 else "Right"
        pts = []
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            bone = f"{side}Hand{finger}1"
            pos = np.asarray(
                task.avatar.robot.skin.get_global_translation(bone)[0]
            ).ravel()[:3]
            pts.append(pos)
        return np.mean(pts, axis=0)

    @staticmethod
    def avatar_wrist(task, hand_id: int) -> np.ndarray:
        side = "Left" if hand_id == 0 else "Right"
        return np.asarray(
            task.avatar.robot.skin.get_global_translation(f"{side}Hand")[0]
        ).ravel()[:3]

    def avatar_dry_run(self, task):
        init_pos = task._avatar_init_pos_lifted()
        init_rot = task._avatar_rot_with_yaw()
        task.avatar.reset(global_trans=init_pos, global_rot=init_rot)
        task.avatar.play_animation(task.ASSIST_MOTION_NAME)

        palm_at_attach = palm_at_detach = wrist_at_attach = None
        step = 0
        while not task.avatar.spare() and step < 1000:
            task.scene.step()
            task.avatar.step()
            if step == task.ATTACH_FRAME:
                palm_at_attach = task._avatar_palm_center(task.AVATAR_HAND_ID)
                wrist_at_attach = task._avatar_wrist(task.AVATAR_HAND_ID)
            if step == task.DETACH_FRAME:
                palm_at_detach = task._avatar_palm_center(task.AVATAR_HAND_ID)
            step += 1
        if palm_at_detach is None:
            palm_at_detach = task._avatar_palm_center(task.AVATAR_HAND_ID)
        if palm_at_attach is None:
            raise RuntimeError(
                f"dry-run never reached attach frame {task.ATTACH_FRAME} "
                f"(loop exited at step {step})"
            )
        return palm_at_attach, palm_at_detach, wrist_at_attach

    # -- shared pool (simple 50-50) ---------------------------------------
    #
    # Two shared hamburgers.  At its start the human takes ONE of them,
    # uniformly at random, and goes for it no matter what the robot does:
    # no free/claimed checks, no retarget, no idle, no waiting.

    @staticmethod
    def shared_pool(task) -> dict:
        pool = getattr(task, "_shared_pool", None)
        if pool is None:
            pool = new_shared_pool_metrics()
            task._shared_pool = pool
        return pool

    @staticmethod
    def food_pos(task, idx: int) -> np.ndarray:
        actor, _spec = task.food_actors[idx]
        return to_numpy(actor.get_pose().p).ravel()[:3].astype(float)

    def human_target(self, task) -> int:
        """The human's uniformly random pick among the shared burgers, drawn
        once per episode from the controller's child RNG (the global layout
        stream is untouched)."""
        idx = getattr(task, "_human_target_idx", None)
        if idx is None:
            pool = list(task._shared_pool_indices())
            rng = self.episode_rng(task, "human_target")
            idx = int(pool[int(rng.integers(len(pool)))])
            task._human_target_idx = idx
            self.shared_pool(task)["human_target"] = idx
        return int(idx)

    # -- geometry + motion ---------------------------------------------

    @staticmethod
    def assist_geometry(task, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Palm pick/place points for shared burger ``idx``; records the
        drop xy.  Pick is the burger's live xy; place is beside the robot's
        drop slot on the live tray."""
        tray_xy = np.array(task._live_tray_center_xy(), dtype=float)
        drop_target_xy = tray_xy + np.array(task.AVATAR_TRAY_OFFSET, dtype=float)
        task._avatar_burger_drop_xy = tuple(drop_target_xy.tolist())
        pos = PlaceBurgerFriesAssistController.food_pos(task, idx)
        pick_pos = np.array([
            pos[0], pos[1],
            task.TABLE_TOP_Z + task.AVATAR_PICKUP_Z_CLEARANCE + task.AVATAR_PICK_LIFT,
        ], dtype=float)
        place_pos = np.array([
            drop_target_xy[0],
            drop_target_xy[1],
            task.TABLE_TOP_Z + task.AVATAR_PLACE_LIFT,
        ], dtype=float)
        return pick_pos, place_pos

    def start_avatar_burger_assist(self, task):
        """Human start: take the randomly chosen shared burger onto the tray
        with the natural pick-and-place motion (non-blocking)."""
        idx = self.human_target(task)
        pick_pos, place_pos = self.assist_geometry(task, idx)
        actor, _spec = task.food_actors[idx]
        task.avatar_burger = actor
        task.avatar.reset(
            global_trans=task._avatar_init_pos_lifted(),
            global_rot=task.AVATAR_BASE_ROT,
        )
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=actor,
            hand_id=None,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=task.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=task.AVATAR_NATURAL_Y_BOUNDS,
            yaw_limit_deg=task.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=task.AVATAR_NATURAL_SETTLE_STEPS,
        )
        task.avatar_collided = False
        task.avatar_collision_log = []
        print(f"[burger_assist] human start target={idx}", flush=True)
        return pick_pos, place_pos

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
        try:
            task._start_avatar_burger_assist()
        except Exception as exc:
            task._eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(f"[burger_fries_assist/eval] avatar assist failed: {exc}", flush=True)

    def eval_step(self, task, step_idx: int) -> None:
        if (getattr(task, "_eval_assist_fired", False)
                or getattr(task, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(
            task, "_eval_assist_trigger_step", task.EVAL_TRIGGER_STEP_MIN,
        )):
            return
        task._eval_assist_fired = True
        try:
            task._start_avatar_burger_assist()
        except Exception as exc:
            task._eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(f"[burger_fries_assist/eval] avatar assist failed: {exc}", flush=True)

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        pool = dict(getattr(task, "_shared_pool", None) or new_shared_pool_metrics())
        human = pool.get("human_target")
        robot = pool.get("robot_target")
        pool["same_target"] = (
            None if human is None or robot is None else bool(human == robot)
        )
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
            "shared_pool": pool,
        }


def new_shared_pool_metrics() -> dict:
    return {"human_target": None, "robot_target": None, "same_target": None}


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class PlaceBurgerFriesInterruptController(InspectMotionInterruptController):
    """Human inspects the loose food the robot commits to, then returns to
    idle; yielding is the policy's job."""

    rng_namespace = "place_burger_fries.interrupt.v1"
    # The task owns its pre-action eval tick so it still runs before BaseTask's
    # early-success check and robot command.  Scripted experts are ticked by
    # the expert harness through on_scripted_rollout_tick -> _eval_step_avatar.
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        # Inspect resolution first (after the initial observation), then the
        # legacy follow-up: calibrate, clear eval state, draw the trigger.
        super().after_base_reset_observation(task, observation=observation)
        if task.avatar is not None:
            task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
            task._calibrate_avatar_for_inspect()
        task._eval_policy_step_count = 0
        task._eval_interrupt_fired = False
        task._eval_interrupt_failed = False
        task._eval_trigger_step = None
        if task.avatar is not None and bool(task.config.get("eval_mode", False)):
            # the retired random-step draw is burned (same bounds)
            # so seeded RNG tails are unchanged; the progress trigger decides.
            task._eval_legacy_trigger_draw = burn_legacy_trigger_draw(task)
            self.init_eval_state(task)

    # -- episode planning ------------------------------------------------

    @staticmethod
    def choose_interrupted_food(task) -> int:
        """Uniform choice among LOOSE foods (pre-placed tray items are never
        the target); drawn at the ``load_actors`` boundary like the legacy
        class so seeded streams are unchanged."""
        preplaced = getattr(task, "_preplaced_food_idxs", set())
        candidates = [
            i for i in range(len(task.food_spawn_xys)) if i not in preplaced
        ]
        if not candidates:
            candidates = list(range(len(task.food_spawn_xys)))
        return int(np.random.choice(candidates))

    # -- avatar calibration ---------------------------------------------

    def dryrun_capture_palm(self, task):
        cache_key = (
            task.INSPECT_MOTION_NAME,
            float(task.AVATAR_MOTION_SLOW),
            task.AVATAR_HAND_ID,
            tuple(np.asarray(task.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(task), "_palm_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        task.avatar.reset(
            np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        task.avatar.play_animation(task.INSPECT_MOTION_NAME)
        palm_default = None
        attach_step = task._attach_frame_scaled
        step = 0
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
        for _ in range(task.DRYRUN_RESET_SETTLE_STEPS):
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
        actor, _ = task._interrupted_pair()
        target_palm = to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)

        palm_default = task._dryrun_capture_palm()
        if palm_default is None:
            task.avatar.reset(
                np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
                np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
            )
            for _ in range(task.CALIBRATION_RESET_SETTLE_STEPS):
                task.step_sim()
            return

        shift = target_palm - palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift

        task.avatar.reset(
            shifted_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(task.CALIBRATION_SETTLE_STEPS):
            task.step_sim()

        task.avatar_collided = False
        task.avatar_collision_log = []

    def inspect_palm_at_frame(self, task, frame_idx: int):
        """Measure the inspect-pose palm without playing a visible motion."""
        if task.avatar is None:
            return None
        if task.INSPECT_MOTION_NAME not in task.avatar.motion_modules:
            task.avatar.play_animation(task.INSPECT_MOTION_NAME)
        motion = task.avatar.motion_modules.get(task.INSPECT_MOTION_NAME)
        if motion is None or not getattr(motion, "data", None):
            return None
        idx = int(np.clip(int(frame_idx), 0, len(motion.data) - 1))
        task.avatar.robot.pose = motion.data[idx]
        task.avatar.robot.node_trans = motion.node_data[idx]
        task.avatar.robot.global_mat = motion.global_mat
        task.avatar.robot.global_mat_inv = motion.global_mat_inv
        task.avatar.robot.update()
        palm = np.asarray(
            task.avatar.robot.get_palm_center(task.AVATAR_HAND_ID),
            dtype=np.float64,
        ).copy()
        task.avatar.reset(
            np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        return palm

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        # Cached after the reset calibration: no extra sim steps.
        task._eval_palm_default = task._dryrun_capture_palm()
        task._eval_hover_runs = {}
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        dwell = self.hover_dwell_params(task)
        if dwell is not None:
            episode.extra["hover_dwell"] = dict(dwell)
        if task._eval_palm_default is None:
            task._eval_interrupt_failed = True
            episode.extra["calibration_failed"] = True
            episode.trigger.block_reason = "avatar_calibration_failed"
            return
        episode.trigger.arm(robot_tcp(task), self.eligible_foods(task))

    @staticmethod
    def food_center(task, actor, spec) -> np.ndarray:
        return np.asarray(
            task._get_object_world_center(actor, spec.asset_id, spec.model_id),
            dtype=float,
        )

    def eligible_foods(self, task) -> dict:
        """Loose foods on the table: not on the tray, not held."""
        out = {}
        for i, (actor, spec) in enumerate(getattr(task, "food_actors", ()) or ()):
            p = self.food_center(task, actor, spec)
            if p[2] < task.TABLE_TOP_Z - 0.05 or p[2] > task.TABLE_TOP_Z + 0.30:
                continue
            if task._food_on_tray(actor):
                continue
            if robot_grasping(task, p):
                continue
            out[i] = p
        return out

    def robot_holding_any(self, task) -> bool:
        for actor, spec in getattr(task, "food_actors", ()) or ():
            p = self.food_center(task, actor, spec)
            if p[2] > task.TABLE_TOP_Z + 0.04 and robot_grasping(task, p):
                return True
        return False

    @staticmethod
    def closest_food_to_ee(task):
        """Legacy helper (xy distance to the closest loose food)."""
        if not getattr(task, "food_actors", None):
            return None, float("inf")
        ee = np.asarray(task.robot.get_arm("right").get_ee_pose()[:3], dtype=float)
        best_i = None
        best_d = float("inf")
        for i, (actor, _spec) in enumerate(task.food_actors):
            p = to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)
            if p[2] < task.TABLE_TOP_Z - 0.02:
                continue
            if task._food_on_tray(actor):
                continue
            d = float(np.linalg.norm(ee[:2] - p[:2]))
            if d < best_d:
                best_i = i
                best_d = d
        return best_i, best_d

    def advance_eval_step(self, task) -> None:
        task._eval_policy_step_count = int(getattr(
            task, "_eval_policy_step_count", 0,
        )) + 1
        episode = interrupt_episode(task)
        if episode is None:
            return
        step = task._eval_policy_step_count
        episode.policy_steps = step
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            return
        if episode.fired or not episode.trigger.armed:
            return
        holding = self.robot_holding_any(task)
        candidates = self.eligible_foods(task)
        gate_ok, gate_reason = not holding, "robot_holding_object"
        dwell = self.hover_dwell_params(task)
        if dwell is not None:
            # Hover/dwell gate: only a food the open gripper has hovered over
            # for ``steps`` consecutive policy steps, as the demos do before
            # the human starts (not merely "the robot left its reset pose").
            runs = getattr(task, "_eval_hover_runs", None)
            if runs is None:
                runs = task._eval_hover_runs = {}
            tcp = robot_tcp(task)
            for key_i in list(runs) + [k for k in candidates if k not in runs]:
                pos = candidates.get(key_i)
                hovering = (not holding) and pos is not None and self._hovering_over(
                    tcp, pos, dwell)
                runs[key_i] = runs.get(key_i, 0) + 1 if hovering else 0
            episode.extra["hover_dwell_max_run"] = max(
                int(episode.extra.get("hover_dwell_max_run", 0)),
                max(runs.values(), default=0),
            )
            candidates = {
                k: v for k, v in candidates.items() if runs.get(k, 0) >= int(dwell["steps"])
            }
            if gate_ok and not candidates:
                gate_ok, gate_reason = False, "no_hover_dwell"
        key = episode.trigger.update(
            step, robot_tcp(task), candidates,
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is None:
            return
        task._eval_interrupt_fired = True
        task._eval_trigger_step = int(step)
        self.start_human(task, int(key), step)

    @staticmethod
    def hover_dwell_params(task) -> dict | None:
        """Hover/dwell gate parameters (``INTERRUPT_HOVER_DWELL_STEPS`` and
        friends, config ``interrupt_hover_*``), or None when disabled."""
        config = getattr(task, "config", None) or {}
        steps = config.get(
            "interrupt_hover_dwell_steps",
            getattr(task, "INTERRUPT_HOVER_DWELL_STEPS", None),
        )
        if steps is None or int(steps) <= 0:
            return None
        xy = config.get("interrupt_hover_xy_m", getattr(task, "INTERRUPT_HOVER_XY_M", 0.06))
        dz = config.get(
            "interrupt_hover_dz_range_m",
            getattr(task, "INTERRUPT_HOVER_DZ_RANGE_M", (0.20, 0.35)),
        )
        return {"steps": int(steps), "xy_m": float(xy),
                "dz_min_m": float(dz[0]), "dz_max_m": float(dz[1])}

    @staticmethod
    def _hovering_over(tcp, food_pos, dwell: Mapping[str, Any]) -> bool:
        tcp = np.asarray(tcp, dtype=float)
        pos = np.asarray(food_pos, dtype=float)
        dz = float(tcp[2] - pos[2])
        return bool(
            float(np.linalg.norm(tcp[:2] - pos[:2])) <= float(dwell["xy_m"])
            and float(dwell["dz_min_m"]) <= dz <= float(dwell["dz_max_m"])
        )

    def start_human(self, task, idx: int, step: int) -> None:
        """Glide (no teleport) from where the human stands to the reach pose
        for food ``idx``, then Inspect it and return to idle."""
        episode = interrupt_episode(task)
        actor, _spec = task.food_actors[idx]
        task._interrupted_food_idx = int(idx)

        def food_pos():
            return to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)

        target_pos = food_pos()
        shift = target_pos - np.asarray(task._eval_palm_default, dtype=np.float64)
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift

        def start_clip() -> bool:
            task._inspect_motion_root_pos = None
            task._play_inspect_motion(
                attach_obj=actor.entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=food_pos,
                grasped=lambda: robot_grasping(task, food_pos()),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=target_pos,
            start_clip=start_clip,
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = interrupt_episode(task)
        if episode is not None and getattr(task, "avatar", None) is not None:
            episode.human.on_sim_step(task)

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        trigger_step = getattr(task, "_eval_trigger_step", -1)
        if trigger_step is None:
            trigger_step = -1
        return {
            "eval_mode": bool(task.config.get("eval_mode", False)),
            "eval_trigger_step": int(trigger_step),
            "eval_policy_step_count": int(getattr(
                task, "_eval_policy_step_count", 0,
            )),
            "eval_interrupt_fired": bool(getattr(
                task, "_eval_interrupt_fired", False,
            )),
            "eval_interrupt_failed": bool(getattr(
                task, "_eval_interrupt_failed", False,
            )),
            "interrupted_food_idx": int(getattr(
                task, "_interrupted_food_idx", -1,
            )),
            **({
                "interrupt": interrupt_metrics(task),
                "pace_interrupt": legacy_pace_block(task),
            } if bool(task.config.get("eval_mode", False)) else {}),
        }
