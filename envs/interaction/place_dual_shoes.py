"""Composable human interaction for the place-dual-shoes family.

The robot objective (top-down shoe pick, drop into the shoebox slot) stays
on the task classes.  These controllers own the human side:

* assist — the side-based shoe allocation inside the layout sampler, the
  blind-expert order draw, the avatar's natural pick-and-place (slowed by
  frame repetition) and the eval-mode start;
* interrupt — the interrupted-shoe draw inside the layout sampler, palm
  dry-run calibration, the reset-order hook, and the shared eval-mode
  interrupt contract (envs/interaction/interrupt_progress.py): progress
  trigger -> avatar glide -> inspect -> idle return.  The policy keeps
  control of the robot throughout.

Release-v1 task-facing attributes remain the single state owner; draw
orders are reproduced verbatim.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..utils import to_numpy
from .base import AssistInteractionController
from .inspect_motion import InspectMotionInterruptController
from .interrupt_progress import (
    burn_legacy_trigger_draw,
    interrupt_episode,
    interrupt_human_active,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    robot_tcp,
)


# ----------------------------------------------------------------------
# Assist
# ----------------------------------------------------------------------


class DualShoesAssistController(AssistInteractionController):
    """Avatar and robot each place one shoe in the shoebox, in parallel."""

    rng_namespace = "place_dual_shoes.assist.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    def __init__(self) -> None:
        self._human_shoe_coin: int | None = None

    # -- lifecycle -------------------------------------------------------

    def on_episode_seeded(self, task, seed: int) -> None:
        self._human_shoe_coin = None

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        task._policy_step_count = 0
        task._eval_avatar_started = False
        task._eval_avatar_failed = False
        task._blind_robot_order = task._sample_blind_robot_order()
        task._blind_robot_first_idx = (
            int(task._blind_robot_order[0]) if task._blind_robot_order else None
        )
        if task.avatar is not None and bool(task.config.get("eval_mode", False)):
            task._eval_trigger_step = 0
            task._eval_avatar_started = True
            if not task._start_avatar_assist():
                task._eval_avatar_failed = True
        else:
            task._eval_trigger_step = None

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        ):
            task._eval_step_avatar()

    @staticmethod
    def advance_eval_step(task) -> None:
        task._policy_step_count += 1
        if task._eval_avatar_started or task._eval_avatar_failed:
            return
        if task._eval_trigger_step is None:
            return
        if task._policy_step_count < task._eval_trigger_step:
            return
        task._eval_avatar_started = True
        if task._start_avatar_assist():
            pass
        else:
            task._eval_avatar_failed = True

    # -- layout / allocation --------------------------------------------

    @staticmethod
    def sample_layout_positions(
        task, rng, *, region_x, box_region_y, shoe_region_y, min_separation,
    ):
        """Box from the shared central band, shoes from the overlap band,
        rejecting shoe samples too close to the box or each other."""
        for _outer in range(64):
            box = (
                float(rng.uniform(*region_x)),
                float(rng.uniform(*box_region_y)),
            )
            placed = [box]
            ok = True
            for _ in range(2):
                got = False
                for _ in range(64):
                    cand = (
                        float(rng.uniform(*region_x)),
                        float(rng.uniform(*shoe_region_y)),
                    )
                    if all(
                        (cand[0] - px) ** 2 + (cand[1] - py) ** 2
                        >= min_separation ** 2
                        for px, py in placed
                    ):
                        placed.append(cand)
                        got = True
                        break
                if not got:
                    ok = False
                    break
            if ok:
                task._assign_shoes(placed)
                return placed
        placed = [
            (0.0, float(np.mean(box_region_y))),
            (region_x[0] + 0.05, float(np.mean(shoe_region_y))),
            (region_x[1] - 0.05, float(np.mean(shoe_region_y))),
        ]
        task._assign_shoes(placed)
        return placed

    def human_shoe_coin(self, task) -> int:
        """Fair coin (0/1) for the human's shoe, drawn once per episode from
        the controller's private stream (episode seed + stream name): it is
        independent of the layout and never consumes the global RNG."""
        if self._human_shoe_coin is None:
            rng = self.episode_rng(task, "human_shoe")
            self._human_shoe_coin = int(rng.integers(2))
        return self._human_shoe_coin

    def assign_shoes(self, task, placed) -> None:
        """Shared pool (assist 50/50): the human takes either shoe by a fair
        private coin and the robot the other.  It used to take the +x shoe
        every episode, so the shoe sides told the robot its shoe without any
        human cue (and the blind expert's random pick was a straw man).
        ``placed`` is unused: the coin does not look at the layout."""
        del placed
        avatar_idx = self.human_shoe_coin(task)
        task._avatar_shoe_idx = int(avatar_idx)
        task._robot_shoe_idx = 1 - int(avatar_idx)

    @staticmethod
    def sample_blind_robot_order(task) -> list[int]:
        n = len(getattr(task, "shoes", []) or [])
        if n <= 0:
            return []
        # Draw from the task stream after layout sampling (reproducible, but
        # not a second seeded channel that can encode the avatar's shoe).
        order = np.random.permutation(n)
        return [int(i) for i in order]

    # -- avatar motion ---------------------------------------------------

    @staticmethod
    def slow_avatar_pick_place(task, repeat: int) -> None:
        if repeat <= 1:
            return
        mm = task.avatar.motion_modules.get("pick_and_place")
        if mm is None or not mm.frames:
            return
        repeated = []
        for frame in mm.frames:
            repeated.extend([frame] * int(repeat))
        mm.frames = repeated
        mm.attach_frame = int(mm.attach_frame) * int(repeat)
        mm.detach_frame = int(mm.detach_frame) * int(repeat)

    def start_avatar_assist(self, task) -> bool:
        if not task.shoes or task.shoebox is None or task.avatar is None:
            return False

        avatar_idx = int(getattr(task, "_avatar_shoe_idx", 0))
        avatar_shoe = task.shoes[avatar_idx]
        avatar_slot = 0

        pick_pos = task._shoe_world_center(avatar_shoe)
        slot_xy = task._box_slot_xy(avatar_slot, task.AVATAR_SLOT_OFFSET_M)
        place_pos = np.array([
            slot_xy[0],
            slot_xy[1],
            task._box_rim_z + task.AVATAR_DROP_ABOVE_RIM,
        ])
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=avatar_shoe.entity,
            hand_id=None,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=task.AVATAR_NATURAL_BODY_MARGIN,
            yaw_limit_deg=task.AVATAR_NATURAL_YAW_LIMIT_DEG,
            body_y_bounds=task.AVATAR_NATURAL_BODY_Y_BOUNDS,
            settle_steps=task.AVATAR_NATURAL_SETTLE_STEPS,
            refine_iters=task.AVATAR_REFINE_ITERS,
        )
        task._slow_avatar_pick_place(task.AVATAR_FRAME_REPEAT)
        task.avatar_collided = False
        task.avatar_collision_log = []
        return True

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        avatar_idx = int(getattr(task, "_avatar_shoe_idx", -1))
        robot_idx = int(getattr(task, "_robot_shoe_idx", -1))
        blind_order = list(getattr(task, "_blind_robot_order", []) or [])
        blind_first = getattr(task, "_blind_robot_first_idx", None)
        metrics = {
            "assist_avatar_shoe_idx": avatar_idx,
            "assist_robot_shoe_idx": robot_idx,
            "blind_robot_order": blind_order,
            "blind_robot_first_idx": (
                int(blind_first) if blind_first is not None else -1
            ),
            "blind_same_first_as_avatar": (
                bool(int(blind_first) == avatar_idx)
                if blind_first is not None and avatar_idx >= 0
                else False
            ),
        }
        if bool(task.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": task._eval_trigger_step,
                "eval_policy_step_count": int(task._policy_step_count),
                "eval_avatar_started": bool(task._eval_avatar_started),
                "eval_avatar_failed": bool(task._eval_avatar_failed),
            })
        return metrics


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class DualShoesInterruptController(InspectMotionInterruptController):
    """Human inspects the shoe the robot is reaching for, then returns to
    idle.  The robot is never overridden; yielding is the policy's job."""

    rng_namespace = "place_dual_shoes.interrupt.v1"
    # The task's take_action ticks the interrupt, then applies the policy
    # action unmodified.
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        super().after_base_reset_observation(task, observation=observation)
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        if bool(task.config.get("eval_mode", False)):
            task._init_eval_interrupt_state()
        else:
            task._calibrate_avatar_for_inspect()

    # -- layout ----------------------------------------------------------

    @staticmethod
    def sample_layout_positions(task, rng):
        """Both shoes from the avatar-reachable band, the box from its own
        robot-side region; the interrupted shoe is drawn first."""
        n_shoes = max(1, int(getattr(task, "_num_shoes", 1)))
        task._interrupted_idx = int(rng.randint(0, n_shoes))
        for _ in range(task.LAYOUT_SAMPLE_ATTEMPTS):
            box = (
                float(rng.uniform(task.BOX_X_LO, task.BOX_X_HI)),
                float(rng.uniform(task.BOX_Y_LO, task.BOX_Y_HI)),
            )
            shoes = [
                (
                    float(rng.uniform(task.SHOE_REACH_X_LO, task.SHOE_REACH_X_HI)),
                    float(rng.uniform(task.SHOE_REACH_Y_LO, task.SHOE_REACH_Y_HI)),
                )
                for _ in range(n_shoes)
            ]
            positions = [box] + shoes
            if task._layout_clear(positions):
                return positions

        fallback = [(-0.18, -0.34), (0.16, 0.02), (-0.04, 0.04)]
        return fallback[: 1 + n_shoes]

    @staticmethod
    def layout_clear(task, positions) -> bool:
        box = positions[0]
        shoes = positions[1:]
        for shoe in shoes:
            dx = box[0] - shoe[0]
            dy = box[1] - shoe[1]
            if dx * dx + dy * dy < task.BOX_SHOE_MIN_SEP_M ** 2:
                return False
        for i in range(len(shoes)):
            for j in range(i + 1, len(shoes)):
                dx = shoes[i][0] - shoes[j][0]
                dy = shoes[i][1] - shoes[j][1]
                if dx * dx + dy * dy < task.SHOE_MIN_SEP_M ** 2:
                    return False
        return True

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
        shoe = task.shoes[task._interrupted_idx]
        target_palm = np.asarray(task._shoe_pos_quat(shoe)[0], dtype=float)
        palm_default = task._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        shifted_pos = (
            np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        )
        task.avatar.reset(
            shifted_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(task.CALIBRATION_SETTLE_STEPS):
            task.step_sim()

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        """Capture the palm reference, burn the retired trigger draw, and arm
        the shared progress trigger from the reset TCP pose."""
        task._eval_palm_default = task._dryrun_capture_palm()
        # The release-v1 random-step trigger drew here in both branches; keep
        # the draw so the global RNG tail per seed is unchanged.
        task._eval_trigger_step = burn_legacy_trigger_draw(task)
        task._eval_interrupt_fired = task._eval_palm_default is None
        task._eval_target_shoe = None
        task._policy_step_count = 0
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        episode.trigger.arm(robot_tcp(task), self.eligible_shoes(task))
        if task._eval_palm_default is None:
            episode.extra["calibration_failed"] = True
            episode.trigger.armed = False
            episode.trigger.block_reason = "avatar_calibration_failed"

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
        gate_ok, gate_reason = True, "phase_gate"
        if self.robot_holding_any(task):
            gate_ok, gate_reason = False, "robot_holding_object"
        key = episode.trigger.update(
            step, robot_tcp(task), self.eligible_shoes(task),
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is None:
            return
        task._eval_interrupt_fired = True
        self.start_human(task, int(key), step)

    @staticmethod
    def shoe_center(task, shoe) -> np.ndarray:
        """Live grasp-centre proxy: the shoe's collision-AABB centre."""
        try:
            return np.asarray(task._shoe_world_aabb(shoe)[2], dtype=float)
        except Exception:
            return np.asarray(task._shoe_pos_quat(shoe)[0], dtype=float)

    def start_human(self, task, idx: int, step: int) -> None:
        episode = interrupt_episode(task)
        shoe = task.shoes[idx]
        # Same palm relation as the scripted calibration (base-link pose).
        anchor = np.asarray(task._shoe_pos_quat(shoe)[0], dtype=float)
        shift = anchor - task._eval_palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task._interrupted_idx = int(idx)
        task._eval_target_shoe = shoe
        center = self.shoe_center(task, shoe)

        def start_clip() -> bool:
            task.avatar.play_animation(
                task.INSPECT_MOTION_NAME,
                attach_obj=shoe.entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=lambda: self.shoe_center(task, shoe),
                grasped=lambda: robot_grasping(
                    task, self.shoe_center(task, shoe),
                    radius_m=task.HELD_TCP_RADIUS_M,
                ),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=center,
            start_clip=start_clip,
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = interrupt_episode(task)
        if episode is not None and getattr(task, "avatar", None) is not None:
            episode.human.on_sim_step(task)

    def eligible_shoes(self, task) -> dict:
        """Shoes on the table: not in the box, not held by the robot."""
        out = {}
        box_xy = to_numpy(task.shoebox.entity.get_pos()).ravel()[:2]
        for i, shoe in enumerate(task.shoes):
            p = np.asarray(task._shoe_pos_quat(shoe)[0])
            in_box = (
                np.linalg.norm(p[:2] - box_xy) < task.SUCCESS_DIST_XY_M
                and p[2] > task.TABLE_TOP_Z + task.SUCCESS_Z_FLOOR_MARGIN_M
            )
            if in_box:
                continue
            if (
                p[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or p[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            center = self.shoe_center(task, shoe)
            if robot_grasping(task, center, radius_m=task.HELD_TCP_RADIUS_M):
                continue
            out[i] = center
        return out

    def robot_holding_any(self, task) -> bool:
        for shoe in task.shoes:
            center = self.shoe_center(task, shoe)
            if (
                center[2] > task.TABLE_TOP_Z + task.HELD_LIFT_M
                and robot_grasping(task, center, radius_m=task.HELD_TCP_RADIUS_M)
            ):
                return True
        return False

    @staticmethod
    def eval_skip_policy_action(task) -> bool:
        """The env never takes the robot away from the policy."""
        return False

    @staticmethod
    def consume_eval_policy_reset_requested(task) -> bool:
        """The env never resets the policy mid-episode."""
        return False

    @staticmethod
    def closest_table_shoe_to_ee(task):
        arm = task.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        box_xy = to_numpy(task.shoebox.entity.get_pos()).ravel()[:2]

        best_idx = None
        best_d = float("inf")
        for i, shoe in enumerate(task.shoes):
            p = np.asarray(task._shoe_pos_quat(shoe)[0])
            in_box = (
                np.linalg.norm(p[:2] - box_xy) < task.SUCCESS_DIST_XY_M
                and p[2] > task.TABLE_TOP_Z + task.SUCCESS_Z_FLOOR_MARGIN_M
            )
            if in_box:
                continue
            if (
                p[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or p[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx, best_d

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        if not bool(task.config.get("eval_mode", False)):
            return {}
        return {
            "eval_mode": True,
            "eval_policy_step_count": int(getattr(task, "_policy_step_count", 0)),
            "eval_interrupt_fired": bool(getattr(task, "_eval_interrupt_fired", False)),
            "eval_human_active": interrupt_human_active(task),
            "interrupt": interrupt_metrics(task),
            "pace_interrupt": legacy_pace_block(task),
        }
