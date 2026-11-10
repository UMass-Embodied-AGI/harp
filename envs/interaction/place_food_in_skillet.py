"""Composable human interaction for the food-in-skillet family.

The robot objective (place the skillet on the cooktop, drop the foods in)
stays on the task classes.  These controllers own the human side:

* assist — which food the avatar takes, its natural pick-and-place into the
  pan with a random start delay, and the eval-mode trigger draw;
* interrupt — which food the human inspects, palm dry-run calibration, the
  reset-order hook, and the eval-mode interrupt contract: the shared progress
  trigger (gated on the pan being on the cooktop), then avatar glide ->
  inspect -> idle return.  The policy keeps control throughout.

Release-v1 task-facing attributes remain the single state owner; draw
orders are reproduced verbatim.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .eval_mode import AssistEvalModeController
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


class PlaceFoodInSkilletAssistController(AssistEvalModeController):
    """Avatar places one food in the skillet while the robot places the rest."""

    rng_namespace = "place_food_in_skillet.assist.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    def __init__(self) -> None:
        self._human_food_draw: int | None = None
        self._step_back: dict | None = None

    def on_episode_seeded(self, task, seed: int) -> None:
        self._human_food_draw = None
        self._step_back = None

    # -- step back after the pick-and-place ---------------
    #
    # The natural stance clamps the body to y >= 0.42; without a step back
    # it stays there after the drop, standing over the robot's far food
    # (y 0.26-0.38), and the robot's descent met the idle human's torso on
    # 7/30 CPU seeds.  Once its pick-and-place clip
    # has ended, the human glides its root straight back (+y, away from
    # the table) to the episode's start-stance depth, keeping its x, yaw and
    # end pose.  Same human in every mode (policies, both experts).

    def start_step_back(self, task) -> None:
        motion = task.avatar.motion_modules.get("pick_and_place")
        self._step_back = {"motion": motion, "phase": "wait", "k": 0}

    def step_back_active(self, task) -> bool:
        sb = self._step_back
        return sb is not None and sb["phase"] in ("wait", "glide")

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        sb = self._step_back
        avatar = getattr(task, "avatar", None)
        if sb is None or avatar is None or sb["phase"] == "done":
            return
        if sb["phase"] == "wait":
            if avatar.motion_modules.get("pick_and_place") is not sb["motion"]:
                sb["phase"] = "done"   # another clip replaced the motion
                return
            if not avatar.spare():
                return
            start = np.asarray(avatar.robot.global_trans, dtype=np.float64).copy()
            target = start.copy()
            target[1] = max(start[1], float(np.asarray(task.avatar_init_pos, dtype=float)[1]))
            if target[1] - start[1] < 1e-3:
                sb["phase"] = "done"
                return
            sb.update(phase="glide", start=start, target=target, k=0)
            task._human_step_back = {"start": start.round(4).tolist(),
                                     "target": target.round(4).tolist(),
                                     "begin_sim_step": int(getattr(task, "SIM_STEP_IDX", -1))}
        n = max(1, int(task.HUMAN_STEP_BACK_SIM_STEPS))
        sb["k"] += 1
        t = min(1.0, sb["k"] / float(n))
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        robot = avatar.robot
        robot.global_trans = sb["start"] + t * (sb["target"] - sb["start"])
        robot.update()
        if sb["k"] >= n:
            sb["phase"] = "done"

    def human_food_draw(self, task, n: int) -> int:
        """Fair draw in [0, n) for the human's food, taken once per episode
        from the controller's private stream (episode seed + stream name)."""
        if self._human_food_draw is None:
            rng = self.episode_rng(task, "human_food")
            self._human_food_draw = int(rng.integers(n))
        return self._human_food_draw % n

    def choose_avatar_food(self, task, *, pan_id: str, avatar_only_ids: tuple[str, ...]):
        """Allocate one avatar-owned food by a fair private coin over every
        food (the robot gets the rest).  Publishes the release-v1 task
        attributes.

        Shared pool (assist 50/50): the human used to always take the
        non-bread item (burger/apple), so the food type told the robot its
        food without any human cue.  The legacy global draw is still
        consumed, so the global stream after allocation is unchanged."""
        candidates = [
            i for i, (_, spec) in enumerate(task.food_actors)
            if spec.asset_id != pan_id
        ]
        if not candidates:
            return -1, []
        avatar_only_candidates = [
            i for i in candidates
            if task.food_actors[i][1].asset_id in avatar_only_ids
        ]
        np.random.choice(avatar_only_candidates or candidates)  # legacy draw, burned
        avatar_idx = int(candidates[self.human_food_draw(task, len(candidates))])
        robot_indices = [i for i in range(len(task.food_actors)) if i != avatar_idx]
        task._avatar_food_idx = avatar_idx
        task._robot_food_indices = robot_indices
        return avatar_idx, robot_indices

    @staticmethod
    def delay_avatar_motion(task, delay_steps: int) -> None:
        if delay_steps <= 0 or task.avatar is None:
            return
        motion = task.avatar.motion_modules.get("pick_and_place")
        if motion is None or not getattr(motion, "frames", None):
            return
        hold = motion.frames[0]
        motion.frames = [hold] * int(delay_steps) + motion.frames
        motion.attach_frame += int(delay_steps)
        motion.detach_frame += int(delay_steps)

    def start_avatar_food_place(self, task, actor, spec) -> None:
        bowl = task._pan_bowl_center()
        pick_pos = task._food_center(actor, spec)
        place_pos = np.array([
            bowl[0],
            bowl[1],
            max(task._pan_rim_z() + task.AVATAR_DROP_ABOVE_RIM, task.TABLE_TOP_Z + 0.18),
        ])
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=actor.entity,
            hand_id=None,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=task.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=task.AVATAR_NATURAL_BODY_Y_BOUNDS,
            yaw_limit_deg=task.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=task.AVATAR_NATURAL_SETTLE_STEPS,
        )
        task.avatar_collided = False
        task.avatar_collision_log = []
        if getattr(task, "HUMAN_STEP_BACK_SIM_STEPS", 0):
            self.start_step_back(task)
        delay = int(np.random.randint(
            int(task.AVATAR_START_DELAY_RANGE[0]),
            int(task.AVATAR_START_DELAY_RANGE[1]) + 1,
        ))
        # The 40-180 frame (1.6-7.2 policy step) start delay is the training
        # demos' own timing and is kept in eval mode too: the human still
        # starts well before the robot's first grasp (~step 15).  Starting
        # at frame 0 put the human's hand into the replayed expert's reach
        # (policy-path smoke seed 9000000, contact at step 15).
        task._delay_avatar_motion_start(delay)

    @staticmethod
    def eval_starts_at_reset(task) -> bool:
        """Eval-mode human starts at reset unless a harness opts into the
        legacy delayed trigger (``eval_assist_trigger_delayed``)."""
        config = getattr(task, "config", {}) or {}
        return (bool(config.get("eval_mode", False))
                and not bool(config.get("eval_assist_trigger_delayed", False)))

    def eval_reset(self, task) -> None:
        # The release-v1 trigger window U[EVAL_TRIGGER_STEP_MIN, MAX] (20-60
        # policy steps, before the 40-180 frame start delay) let the robot
        # commit long before the human moved.  The draw is kept (as a burn
        # when the human starts at reset) so the global RNG stream is
        # unchanged; the start delay then matches the training demos.
        drawn = int(np.random.randint(
            task.EVAL_TRIGGER_STEP_MIN,
            task.EVAL_TRIGGER_STEP_MAX + 1,
        ))
        task._eval_assist_fired = False
        task._eval_assist_failed = False
        if not self.eval_starts_at_reset(task):
            task._eval_assist_trigger_step = drawn
            return
        task._eval_assist_trigger_step = 0
        self.eval_step(task, 0)

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
            avatar_idx, _robot_indices = task._choose_avatar_food()
            if avatar_idx < 0:
                task._eval_assist_failed = True
                return
            actor, spec = task.food_actors[avatar_idx]
            task._start_avatar_food_place(actor, spec)
        except Exception as exc:
            task._eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(f"[food_skillet_assist/eval] avatar assist failed: {exc}", flush=True)

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        avatar_idx = getattr(task, "_avatar_food_idx", None)
        avatar_idx = -1 if avatar_idx is None else int(avatar_idx)
        blind_order = getattr(task, "_blind_robot_food_order", [])
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
            "assist_avatar_food_idx": avatar_idx,
            "blind_robot_food_order": [int(i) for i in blind_order],
            "blind_robot_food_attempts": getattr(
                task, "_blind_robot_food_attempts", []
            ),
            "blind_robot_avatar_same_first": (
                bool(blind_order[0] == avatar_idx) if blind_order else False
            ),
        }


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class SkilletInterruptController(InspectMotionInterruptController):
    """Human inspects the food the robot is reaching for, then returns to
    idle.  The robot is never overridden; yielding is the policy's job."""

    rng_namespace = "place_food_in_skillet.interrupt.v1"
    # The task-local action wrapper ticks the interrupt before applying the
    # (unmodified) policy action.
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        super().after_base_reset_observation(task, observation=observation)
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        task._select_interrupted_food()
        if bool(task.config.get("eval_mode", False)):
            task._init_eval_interrupt_state()
        else:
            task._calibrate_avatar_for_inspect()

    # -- episode planning ------------------------------------------------

    @staticmethod
    def select_interrupted_food(task) -> None:
        if not task.food_actors:
            task._interrupted_idx = 0
            return

        requested = task.config.get("interrupt_food", "burger")
        requested = None if requested is None else str(requested).lower()
        if requested == "random" or task.config.get("randomize_interrupt_food", False):
            # Never interrupt a food that already sits in the skillet
            # (pre-placed).
            candidates = [
                idx for idx, (actor, spec) in enumerate(task.food_actors)
                if not task._food_in_skillet(actor, spec)
            ] or list(range(len(task.food_actors)))
            task._interrupted_idx = int(np.random.choice(candidates))
        else:
            task._interrupted_idx = 0
            for idx, (_actor, spec) in enumerate(task.food_actors):
                if spec.name == requested:
                    task._interrupted_idx = idx
                    break

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
        for _ in range(20):
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
        actor, spec = task.food_actors[task._interrupted_idx]
        target_palm = task._actor_center(actor, spec.asset_id, spec.model_id)

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
        for _ in range(30):
            task.step_sim()

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        """Capture the palm reference, burn the retired trigger draw, and arm
        the shared progress trigger from the reset TCP pose."""
        task._eval_palm_default = task._dryrun_capture_palm()
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        if task._eval_palm_default is None:
            # Calibration failed: no human interruption is possible.
            task._eval_interrupt_fired = True
            episode.extra["calibration_failed"] = True
        else:
            # The release-v1 random-step trigger drew here; keep the draw so
            # the global RNG tail per seed is unchanged.
            task._eval_trigger_step = burn_legacy_trigger_draw(task)
            task._eval_interrupt_fired = False
        task._policy_step_count = 0
        task._eval_target_food = None
        episode.trigger.arm(robot_tcp(task), self.eligible_foods(task))
        if task._eval_palm_default is None:
            episode.trigger.block_reason = "avatar_calibration_failed"
            episode.trigger.armed = False

    def advance_eval_step(self, task) -> None:
        episode = interrupt_episode(task)
        if episode is None:
            return
        episode.policy_steps += 1
        task._policy_step_count = episode.policy_steps
        step = episode.policy_steps
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            return
        if episode.fired or not episode.trigger.armed:
            return
        candidates = self.eligible_foods(task)
        pan_gate = bool(task.config.get(
            "pace_require_pan_ready", task.PACE_REQUIRE_PAN_READY,
        ))
        gate_ok, gate_reason = True, "phase_gate"
        if pan_gate and not task._pace_pan_ready():
            gate_ok, gate_reason = False, "pan_not_on_cooktop"
        elif self.robot_holding_any(task):
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
        episode = interrupt_episode(task)
        actor, spec = task.food_actors[idx]
        food_pos = task._actor_center(actor, spec.asset_id, spec.model_id)
        shift = food_pos - task._eval_palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task._interrupted_idx = int(idx)
        task._eval_target_food = (actor, spec)

        def start_clip() -> bool:
            task._play_inspect_motion(
                attach_obj=actor.entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=lambda: task._actor_center(actor, spec.asset_id, spec.model_id),
                grasped=lambda: robot_grasping(
                    task, task._actor_center(actor, spec.asset_id, spec.model_id),
                ),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=food_pos,
            start_clip=start_clip,
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = interrupt_episode(task)
        if episode is not None and task.avatar is not None:
            episode.human.on_sim_step(task)

    def eligible_foods(self, task) -> dict:
        """Loose foods on the table: not in the skillet, not held."""
        bowl_xy = task._pan_bowl_center()[:2]
        out = {}
        for idx, (actor, spec) in enumerate(task.food_actors):
            p = task._actor_center(actor, spec.asset_id, spec.model_id)
            in_skillet = (
                np.linalg.norm(p[:2] - bowl_xy) < task.FOOD_SUCCESS_XY
                and p[2] > task.TABLE_TOP_Z + 0.02
            )
            if in_skillet:
                continue
            if p[2] < task.TABLE_TOP_Z - 0.05 or p[2] > task.TABLE_TOP_Z + 0.30:
                continue
            if robot_grasping(task, p):
                continue
            out[idx] = p
        return out

    @staticmethod
    def robot_holding_any(task) -> bool:
        for actor, spec in task.food_actors:
            p = task._actor_center(actor, spec.asset_id, spec.model_id)
            if p[2] > task.TABLE_TOP_Z + 0.04 and robot_grasping(task, p):
                return True
        return False

    @staticmethod
    def pace_pan_ready(task) -> bool:
        bowl = task._pan_bowl_center()
        dxy = float(np.linalg.norm(bowl[:2] - task._cooktop_xy))
        threshold = float(task.config.get(
            "pace_pan_ready_xy", task.PACE_PAN_READY_XY,
        ))
        return bool(dxy <= threshold and bowl[2] > task.TABLE_TOP_Z)

    @staticmethod
    def closest_table_food_candidate(task):
        arm = task.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        bowl_xy = task._pan_bowl_center()[:2]

        best_idx = None
        best_d = float("inf")
        for idx, (actor, spec) in enumerate(task.food_actors):
            p = task._actor_center(actor, spec.asset_id, spec.model_id)
            in_skillet = (
                np.linalg.norm(p[:2] - bowl_xy) < task.FOOD_SUCCESS_XY
                and p[2] > task.TABLE_TOP_Z + 0.02
            )
            if in_skillet:
                continue
            if p[2] < task.TABLE_TOP_Z - 0.05 or p[2] > task.TABLE_TOP_Z + 0.30:
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_idx = idx
                best_d = d
        return best_idx, best_d

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        if not bool(task.config.get("eval_mode", False)):
            return {}
        block = legacy_pace_block(task)
        block["pan_ready"] = task._pace_pan_ready()
        return {
            "interrupt": interrupt_metrics(task),
            "pace_interrupt": block,
        }
