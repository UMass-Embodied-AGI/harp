"""Composable human interaction for the put-object-in-cabinet family.

The robot objective (pick the object, place it in the open drawer) stays on
the task classes.  These controllers own the human side:

* assist — the avatar drawer pull (body placement from the palm contact
  point, the pull motion), the post-pull idle/body return, and the
  eval-mode start;
* interrupt — palm dry-run calibration, the reset-order hook, and the
  shared eval-mode contract: progress trigger gated on the drawer being
  open -> avatar glide -> inspect -> idle return.

The assist task keeps its ``step_sim`` override, which mirrors the avatar's
pull progress onto the drawer joint every physics step; that is a scene
coupling, not human behavior.  Release-v1 task-facing attributes remain the
single state owner.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..utils import to_numpy

from .base import AssistInteractionController
from .eval_mode import delayed_assist_trigger_step
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


class CabinetAssistController(AssistInteractionController):
    """Avatar opens the drawer while the robot picks the object."""

    rng_namespace = "put_object_cabinet.assist.v1"
    handles_policy_ticks = True
    # The cabinet blind expert already advances its legacy state explicitly.
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        if task.cabinet is not None and task._drawer_link is None:
            task._pick_drawer()
        task._avatar_drawer_motion = None
        task._avatar_drawer_sync_active = False
        task._avatar_idle_return_started = False
        task._avatar_body_return_active = False
        task._policy_step_count = 0
        task._eval_avatar_started = False
        task._eval_avatar_failed = False
        delayed = (
            delayed_assist_trigger_step(task)
            if bool(task.config.get("eval_mode", False)) else None
        )
        if delayed is not None:
            task._eval_trigger_step = delayed
        elif bool(task.config.get("eval_mode", False)):
            task._eval_trigger_step = 0
            task._eval_avatar_started = True
            if not task._start_open_drawer_with_avatar():
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
        ok = task._start_open_drawer_with_avatar()
        if not ok:
            task._eval_avatar_failed = True

    # -- body return after the pull --------------------------------------

    @staticmethod
    def start_avatar_body_return(task) -> None:
        if task.avatar is None:
            return
        task._avatar_body_return_active = True
        task._avatar_body_return_from = np.asarray(
            task.avatar.robot.global_trans, dtype=np.float64,
        ).copy()
        task._avatar_body_return_to = np.asarray(
            task.avatar_init_pos, dtype=np.float64,
        ).copy()
        task._avatar_body_return_frame = 0
        task._avatar_body_return_frames = max(1, int(task.AVATAR_RETURN_IDLE_FRAMES))

    @staticmethod
    def step_avatar_body_return(task) -> None:
        """Smoothstep the avatar root back home; called before the avatar
        update inside the task's ``step_sim``."""
        if not task._avatar_body_return_active or task.avatar is None:
            return
        task._avatar_body_return_frame += 1
        t = min(1.0, task._avatar_body_return_frame / float(task._avatar_body_return_frames))
        s = t * t * (3.0 - 2.0 * t)
        task.avatar.robot.global_trans = (
            (1.0 - s) * task._avatar_body_return_from
            + s * task._avatar_body_return_to
        )
        task.avatar.robot.global_rot = np.asarray(task.avatar_init_rot, dtype=np.float64).copy()
        if t >= 1.0:
            task.avatar.robot.global_trans = task._avatar_body_return_to.copy()
            task._avatar_body_return_active = False

    # -- drawer pull -------------------------------------------------------

    @staticmethod
    def start_open_drawer_with_avatar(task) -> bool:
        """Start the avatar drawer pull; drawer qpos is synced by step_sim."""
        if task.avatar is None:
            return False

        handle_closed = np.asarray(task._drawer_handle_world0, dtype=float).copy()
        handle_open = handle_closed + task._DRAWER_PULL_DIR * float(
            task._drawer_open_qpos_target
        )

        # Start from a stable side pose, then shift the body so the handle
        # is within the arm's reachable workspace.
        base_pos = np.asarray(task.avatar_init_pos, dtype=np.float64).copy()
        base_rot = np.asarray(task.avatar_init_rot, dtype=np.float64).copy()
        task.avatar.reset(base_pos, base_rot)
        for _ in range(20):
            task.step_sim()
        palm0 = np.asarray(
            task.avatar.robot.get_palm_center(task.AVATAR_HAND_ID),
            dtype=np.float64,
        ).ravel()[:3]

        def _palm_left_bound_offset(hand_R):
            # Hand frame columns are [thumb-pinky, wrist-fingers, palm normal];
            # the contact point is the finger-side palm edge.
            side = -np.asarray(hand_R, dtype=np.float64)[:, 1]
            n = float(np.linalg.norm(side))
            if n < 1e-8:
                side = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                side = side / n
            return side * float(task.AVATAR_HANDLE_SIDE_OFFSET)

        _, hand_R0 = task.avatar.robot._get_hand_frame(task.AVATAR_HAND_ID)
        contact_offset0 = _palm_left_bound_offset(hand_R0)
        contact0 = palm0 + contact_offset0
        shift = handle_closed - contact0
        shift[2] = float(np.clip(shift[2], -0.05, 0.14))
        start_pos = base_pos + shift
        start_pos[0] = float(np.clip(start_pos[0], *task.AVATAR_BODY_X_BOUNDS))
        start_pos[1] = float(np.clip(start_pos[1], *task.AVATAR_BODY_Y_BOUNDS))
        task.avatar.reset(start_pos, base_rot)
        task.avatar_collided = False
        task.avatar_collision_log = []
        task._avatar_collision_tick = 0

        _, hand_R = task.avatar.robot._get_hand_frame(task.AVATAR_HAND_ID)
        contact_offset = _palm_left_bound_offset(hand_R)
        palm_start = handle_closed - contact_offset
        palm_end = handle_open - contact_offset

        motion = task.avatar.pull_drawer(
            handle_start=palm_start,
            handle_end=palm_end,
            hand_id=task.AVATAR_HAND_ID,
            approach_frames=task.AVATAR_OPEN_APPROACH_FRAMES,
            pull_frames=task.AVATAR_OPEN_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_OPEN_RETRACT_FRAMES,
            retract_offset=(0.16, 0.02, 0.0),
            refine_iters=1,
        )
        if motion is None or not getattr(motion, "frames", None):
            return False

        task._avatar_drawer_motion = motion
        task._avatar_drawer_sync_active = True
        task._avatar_idle_return_started = False
        return True

    @staticmethod
    def wait_for_avatar_drawer_open(task) -> bool:
        while not task.avatar.spare():
            task.step_sim()

        task._set_drawer_qpos(task._drawer_open_qpos_target)
        task._drawer_held_qpos = task._drawer_open_qpos_target
        if not task._hold_drawer_open():
            return False
        for _ in range(40):
            task.step_sim()
        return True

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        if not bool(task.config.get("eval_mode", False)):
            return {}
        return {
            "eval_mode": True,
            "eval_trigger_step": task._eval_trigger_step,
            "eval_policy_step_count": int(task._policy_step_count),
            "eval_avatar_started": bool(task._eval_avatar_started),
            "eval_avatar_failed": bool(task._eval_avatar_failed),
            "eval_drawer_sync_active": bool(task._avatar_drawer_sync_active),
            "eval_avatar_body_return_active": bool(task._avatar_body_return_active),
        }


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class CabinetInterruptController(InspectMotionInterruptController):
    """Human inspects the object mid-task; the robot waits, then resumes."""

    rng_namespace = "put_object_cabinet.interrupt.v1"
    handles_policy_ticks = True
    # Scripted experts (both) are ticked by the expert harness at
    # policy cadence through advance_interaction_policy_step.
    handles_scripted_rollout_ticks = True

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        # InspectMotionMixin historically resolved the chosen clip before the
        # public cabinet reset wrapper initialized calibration/eval state.
        super().after_base_reset_observation(task, observation=observation)
        if task.cabinet is not None and task._drawer_link is None:
            task._pick_drawer()
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        if bool(task.config.get("eval_mode", False)):
            task._init_eval_interrupt_state()
        else:
            task._calibrate_avatar_for_inspect()

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if (
            getattr(task, "avatar", None) is not None
            and bool(task.config.get("eval_mode", False))
        ):
            task._eval_step_avatar()

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
        if task.object is None:
            return
        target_palm = task._get_object_world_center(
            task.object, task._object_id, task._object_model_id,
        ).astype(float)

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
        task._eval_palm_default = task._dryrun_capture_palm()
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        task._policy_step_count = 0
        task._eval_target_object = None
        if task._eval_palm_default is None:
            task._eval_interrupt_fired = True
            episode.extra["calibration_failed"] = True
            episode.trigger.block_reason = "avatar_calibration_failed"
            return
        # The retired random-step trigger drew here; burn it (same bounds).
        task._eval_trigger_step = burn_legacy_trigger_draw(task)
        task._eval_interrupt_fired = False
        episode.trigger.arm(robot_tcp(task), self.eligible_objects(task))

    @staticmethod
    def object_center(task) -> np.ndarray:
        return np.asarray(task._get_object_world_center(
            task.object, task._object_id, task._object_model_id,
        ), dtype=float)

    @staticmethod
    def drawer_qpos(task):
        if (getattr(task, "cabinet", None) is None
                or getattr(task, "_drawer_dof_idx", None) is None):
            return None
        try:
            return float(to_numpy(task.cabinet.entity.get_qpos()).ravel()[
                task._drawer_dof_idx])
        except Exception:
            return None

    def drawer_open(self, task) -> bool:
        q = self.drawer_qpos(task)
        threshold = float(task.config.get(
            "interrupt_drawer_open_qpos", task.INTERRUPT_DRAWER_OPEN_QPOS,
        ))
        return q is not None and q >= threshold

    def eligible_objects(self, task) -> dict:
        """The single object while it is on the table, not in the drawer
        and not held by the robot."""
        if getattr(task, "object", None) is None:
            return {}
        p = self.object_center(task)
        if not task._is_object_on_table(p):
            return {}
        if robot_grasping(task, p, radius_m=task.INTERRUPT_HELD_TCP_RADIUS_M):
            return {}
        return {0: p}

    def advance_eval_step(self, task) -> None:
        task._policy_step_count = int(getattr(task, "_policy_step_count", 0)) + 1
        episode = interrupt_episode(task)
        if episode is None:
            return
        step = task._policy_step_count
        episode.policy_steps = step
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            return
        if episode.fired or not episode.trigger.armed:
            return
        candidates = self.eligible_objects(task)
        gate_ok, gate_reason = True, "phase_gate"
        if not self.drawer_open(task):
            gate_ok, gate_reason = False, "drawer_not_open"
        elif (getattr(task, "object", None) is not None
              and robot_grasping(task, self.object_center(task),
                                 radius_m=task.INTERRUPT_HELD_TCP_RADIUS_M)):
            gate_ok, gate_reason = False, "robot_holding_object"
        key = episode.trigger.update(
            step, robot_tcp(task), candidates,
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is None:
            return
        task._eval_interrupt_fired = True
        self.start_human(task, step)

    def start_human(self, task, step: int) -> None:
        episode = interrupt_episode(task)
        target = task.object
        task._eval_target_object = target
        obj_pos = self.object_center(task)
        shift = obj_pos - np.asarray(task._eval_palm_default, dtype=np.float64)
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift

        def start_clip() -> bool:
            task._inspect_motion_root_pos = None
            task._play_inspect_motion(
                attach_obj=target.entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=lambda: self.object_center(task),
                grasped=lambda: robot_grasping(
                    task, self.object_center(task),
                    radius_m=task.INTERRUPT_HELD_TCP_RADIUS_M,
                ),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=0,
            target_pos=obj_pos,
            start_clip=start_clip,
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = interrupt_episode(task)
        if episode is not None and getattr(task, "avatar", None) is not None:
            episode.human.on_sim_step(task)

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        if not bool(task.config.get("eval_mode", False)):
            return {}
        block = legacy_pace_block(task)
        block["drawer_open"] = self.drawer_open(task)
        return {
            "interrupt": interrupt_metrics(task),
            "pace_interrupt": block,
        }

    @staticmethod
    def is_object_on_table(task, obj_pos: np.ndarray) -> bool:
        if obj_pos[2] < task.TABLE_TOP_Z - 0.05:
            return False
        if obj_pos[2] > task.TABLE_TOP_Z + 0.35:
            return False
        if task._drawer_link is not None:
            metrics = task._object_in_drawer_metrics()
            if bool(metrics.get("object_in_drawer", False)):
                return False
        return True
