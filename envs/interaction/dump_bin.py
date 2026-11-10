"""Composable human interaction for the dump-bin family.

The robot objective (pick each table object, drop it in the bin) stays on the
task classes.  These controllers own the human side:

* assist — the ``Trash_clip`` throw: dry-run palm capture, the constraint
  solve that places the avatar so its drop lands in the bin and its pick
  meets the extra cube, the scripted throw start, and the eval-mode trigger;
* interrupt — the inspect interruption: interrupted-cube/ordinal draws and
  the reachable-band layout, palm dry-run calibration, and the shared
  eval-mode interrupt contract (shared progress trigger -> avatar glide ->
  inspect -> idle return; see ``interrupt_progress``).  The policy keeps
  control of the robot throughout.

During this migration slice the release-v1 task-facing attributes
(``_eval_*``, ``_pace_*``, ``_interrupted_idx`` ...) remain the single state
owner: scene-init replay, evaluation metrics, ``check_success`` and
``scripts/eval.py`` all read them, so the controllers operate on those fields
directly rather than mirroring them.  Draw orders are reproduced verbatim.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

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


class DumpBinAssistController(AssistEvalModeController):
    """Avatar tosses an extra cube into the bin while the robot dumps its own.

    Geometry pipeline ("bin region -> drop pose -> avatar pose -> object
    pose" constraint chain):

    1. Dry-run the throw at the default avatar pose; read the palm at the
       attach and detach clip frames.
    2. Choose a drop target inside the bin (bin centre backed off toward the
       avatar); the avatar must shift by ``drop_target_xy - palm_at_detach.xy``.
    3. Apply the same shift to ``palm_at_attach`` -> the cube spawn pose.
    4. Reset the avatar at the shifted pose and replay the motion with the
       cube attached between the two frames.
    """

    rng_namespace = "dump_bin.assist.v1"
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
        """Average of the 4 finger knuckles (Index1/Middle1/Ring1/Pinky1)
        — same recipe as ``envs.avatar.robot.get_palm_center``."""
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

    def scripted_motion_frames(self, task) -> tuple[int, int]:
        """Attach/detach clip frames resampled by the throw frame ratio."""
        return (
            int(task.ATTACH_FRAME * task.AVATAR_FRAME_RATIO),
            int(task.DETACH_FRAME * task.AVATAR_FRAME_RATIO),
        )

    # -- dry run + constraint solve -------------------------------------

    def avatar_dry_run(self, task):
        """Step the throw at the default pose; return
        ``(palm_at_attach_world, palm_at_detach_world, wrist_at_attach_world)``."""
        init_pos = task._avatar_init_pos_lifted()
        init_rot = task._avatar_rot_with_yaw()
        task.avatar.reset(global_trans=init_pos, global_rot=init_rot)
        task.avatar.play_animation(task.ASSIST_MOTION_NAME)

        palm_at_attach = palm_at_detach = wrist_at_attach = None
        step = 0
        while not task.avatar.spare() and step < task.AVATAR_DRY_RUN_MAX_STEPS:
            task.scene.step()           # raw step — skip step_sim() so we
            task.avatar.step()          # don't write to the video recorder
            if step == task.ATTACH_FRAME:
                palm_at_attach = task._avatar_palm_center(task.AVATAR_HAND_ID)
                wrist_at_attach = task._avatar_wrist(task.AVATAR_HAND_ID)
            if step == task.DETACH_FRAME:
                palm_at_detach = task._avatar_palm_center(task.AVATAR_HAND_ID)
            step += 1
        # If avatar.spare() goes True on the same step DETACH_FRAME would
        # have fired, the loop body never runs — sample once more.
        if palm_at_detach is None:
            palm_at_detach = task._avatar_palm_center(task.AVATAR_HAND_ID)
        if palm_at_attach is None:
            raise RuntimeError(
                f"dry-run never reached attach frame {task.ATTACH_FRAME} "
                f"(loop exited at step {step})"
            )
        return palm_at_attach, palm_at_detach, wrist_at_attach

    def prepare_assist_geometry(self, task) -> np.ndarray:
        palm_at_attach, palm_at_detach, _ = task._avatar_dry_run()

        # Constraint solve: drop above bin xy -> avatar shift -> pickup.
        bin_xy = np.array(task.big_bin_pose.p[:2], dtype=float)
        avatar_init_xy = np.array(task.avatar_init_pos[:2], dtype=float)
        toward_avatar = avatar_init_xy - bin_xy
        norm = np.linalg.norm(toward_avatar)
        if norm > 1e-9:
            toward_avatar = toward_avatar / norm
        drop_target_xy = (bin_xy + toward_avatar * task.AVATAR_BACK_OFFSET
                          + np.asarray(getattr(task, "AVATAR_DROP_XY_SHIFT", (0.0, 0.0)),
                                       dtype=float))
        shift_xy = drop_target_xy - palm_at_detach[:2]
        T_new = task._avatar_init_pos_lifted().copy()
        T_new[:2] += shift_xy
        pickup_world = palm_at_attach.copy()
        pickup_world[:2] += shift_xy
        # Force pickup z to table_top + cube_half so the cube can collide
        # stably instead of spawning inside the table.
        cube_rest_z = (
            task.TABLE_TOP_Z + task.AVATAR_CUBE_HALF + task.AVATAR_CUBE_TABLE_CLEARANCE
        )
        pickup_world[2] = cube_rest_z

        # Rigid-body shifts of T_new; the cube stays at pickup_world and the
        # captured hand-frame offset absorbs the body shift.
        franka_xy = np.array(task._FRANKA_BASE_XY, dtype=float)
        toward_arm_xy = franka_xy - T_new[:2]
        ta_norm = np.linalg.norm(toward_arm_xy)
        if ta_norm > 1e-9:
            toward_arm_xy = toward_arm_xy / ta_norm
        T_new[:2] += toward_arm_xy * task.AVATAR_TOWARD_ARM_SHIFT
        T_new[2] += task.AVATAR_UP_SHIFT
        away_from_bin = avatar_init_xy - bin_xy
        afb_norm = np.linalg.norm(away_from_bin)
        if afb_norm > 1e-9:
            away_from_bin = away_from_bin / afb_norm
        T_new[:2] += away_from_bin * task.AVATAR_AWAY_FROM_BIN_SHIFT
        left_xy = np.array(
            [-toward_arm_xy[1], toward_arm_xy[0]], dtype=float
        )
        T_new[:2] += left_xy * task.AVATAR_LEFT_SHIFT

        # ---- Position avatar cube on the table at pickup.xy, reset avatar ----
        task.avatar_cube.set_pos(pickup_world.astype(float))
        task.avatar_cube.set_quat(task.IDENTITY_QUAT)
        try:
            task.avatar_cube.set_dofs_velocity(np.zeros(6, dtype=np.float64))
        except Exception:
            pass
        task.avatar.reset(
            global_trans=T_new,
            global_rot=task._avatar_rot_with_yaw(),
        )
        # The dry-run / default-pose reset can populate collision logs
        # before the real episode geometry is installed.  Start grading
        # only after the avatar is shifted to its solved assist pose.
        task.avatar_collided = False
        task.avatar_collision_log = []
        task._avatar_collision_tick = 0

        for _ in range(task.AVATAR_POST_DRY_RUN_SETTLE_STEPS):
            task.step_sim()
        return pickup_world

    # -- motion start --------------------------------------------------

    def start_scripted_throw(self, task) -> None:
        """Kick off the throw with auto attach/detach (scripted expert)."""
        attach_resampled, detach_resampled = self.scripted_motion_frames(task)
        task.avatar.play_animation(
            task.ASSIST_MOTION_NAME,
            attach_obj=task.avatar_cube,
            hand_id=task.AVATAR_HAND_ID,
            attach_frame=attach_resampled,
            detach_frame=detach_resampled,
            return_to_idle_after=task.IDLE_RETURN_FRAMES,
        )

    def start_eval_assist(self, task) -> None:
        task._prepare_avatar_assist_geometry()
        task.avatar.frame_ratio = task.AVATAR_FRAME_RATIO
        self.start_scripted_throw(task)

    # -- eval-mode hooks (legacy _eval_at_reset / _eval_at_step) --------

    def eval_reset(self, task) -> None:
        task._eval_assist_trigger_step = 0
        task._eval_assist_fired = True
        task._eval_assist_failed = False
        try:
            task._start_eval_avatar_assist()
        except Exception as exc:
            task._eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(f"[dump_bin_assist/eval] avatar assist failed: {exc}", flush=True)

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
            task._start_eval_avatar_assist()
        except Exception as exc:
            task._eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(f"[dump_bin_assist/eval] avatar assist failed: {exc}", flush=True)

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
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
            "blind_robot_object_idx": int(getattr(
                task, "_blind_robot_object_idx", -1,
            )),
        }


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


class DumpBinInterruptController(InspectMotionInterruptController):
    """Human inspects the cube the robot is reaching for, then returns to
    idle.  The robot is never overridden; yielding is the policy's job.

    ``handles_policy_ticks`` stays ``False``: the task's ``take_action``
    wrapper ticks this machine before applying the unmodified policy action.
    """

    rng_namespace = "dump_bin.interrupt.inspect.v1"
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        # Inspect resolution keeps its release-v1 position: after the initial
        # observation is captured, before any avatar calibration.
        super().after_base_reset_observation(task, observation=observation)
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        if bool(task.config.get("eval_mode", False)):
            task._init_eval_interrupt_state()
        else:
            task._calibrate_avatar_for_inspect()
        # The hidden-avatar refresh of the returned observation stays in the
        # task's reset() wrapper: a lifecycle hook cannot change the value
        # BaseTask.reset returns.

    # -- episode planning ------------------------------------------------

    def sample_interrupt_pick_ordinal(self, task) -> int:
        """0-based robot pick ordinal at which the avatar interrupts.

        Config ``interrupt_pick_ordinal`` may force a 1-based ordinal for
        review videos; otherwise sample uniformly across the pick ordinals.
        """
        forced = task.config.get("interrupt_pick_ordinal")
        if forced is not None:
            ordinal = int(forced) - 1
            if not 0 <= ordinal < task.num_objects:
                raise ValueError(
                    "interrupt_pick_ordinal must be in "
                    f"[1, {task.num_objects}], got {forced!r}"
                )
            return ordinal
        return int(np.random.randint(0, task.num_objects))

    def sample_cube_xys(self, task, *, base_sampler: Callable[[], Any]):
        """Pick the interrupted cube, then spawn every cube in the same
        avatar-reachable band with min-spacing rejection (legacy draw order).
        ``base_sampler`` is the family core's layout for the base-layout
        ablation."""
        if bool(task.config.get("interrupt_use_base_cube_layout", False)):
            task._interrupted_idx = 0
            task._interrupt_pick_ordinal = 0
            return base_sampler()
        task._interrupted_idx = int(np.random.randint(0, task.num_objects))
        task._interrupt_pick_ordinal = task._sample_interrupt_pick_ordinal()
        SX_LO, SX_HI = task.AVATAR_REACH_SPAWN_X_RANGE
        SY_LO, SY_HI = task.AVATAR_REACH_SPAWN_Y_RANGE
        min_spacing_sq = task.MIN_OBJECT_SPACING ** 2
        for _ in range(task.INTERRUPT_LAYOUT_SAMPLE_ATTEMPTS):
            xs = np.empty(task.num_objects)
            ys = np.empty(task.num_objects)
            for i in range(task.num_objects):
                xs[i] = np.random.uniform(SX_LO, SX_HI)
                ys[i] = np.random.uniform(SY_LO, SY_HI)
            ok = True
            for i in range(task.num_objects):
                for j in range(i + 1, task.num_objects):
                    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < min_spacing_sq:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return xs, ys
        raise RuntimeError("[dump_bin_interrupt] could not sample non-overlapping layout")

    # -- avatar calibration ---------------------------------------------

    def dryrun_capture_palm(self, task):
        """Play the Inspect motion at the avatar's home pose, capture the palm
        at the (scaled) attach frame, then reset the avatar home.  Returns
        ``palm_default`` or None if the attach frame was never reached."""
        # BaseTask.reset() creates the VLA recorder before this dry-run; VLA
        # capture is ticked from step_sim immediately, so suppress it or the
        # saved episode contains this hidden Inspect replay.
        saved_vla_recorder = task.vla_recorder
        saved_record_stride = task._record_stride
        task.vla_recorder = None
        task._record_stride = None
        try:
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
            for _ in range(task.CALIBRATION_SETTLE_STEPS):
                task.step_sim()
        finally:
            task.vla_recorder = saved_vla_recorder
            task._record_stride = saved_record_stride
        task.avatar_collided = False
        task.avatar_collision_log = []
        return palm_default

    def calibrate_avatar(self, task) -> None:
        """Scripted flow: dry-run + shift so the palm lands on the
        interrupted cube at the attach frame."""
        cube = task.objects[task._interrupted_idx]
        target_palm = task._dump_object_center(cube).astype(float)

        palm_default = task._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        idle_pos = task._set_inspect_motion_root_pos(shifted_pos)

        task.avatar.reset(
            idle_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(task.POST_CALIBRATION_SETTLE_STEPS):
            task.step_sim()

    # -- hidden avatar / yield switches ---------------------------------

    @staticmethod
    def eval_hide_avatar_until_trigger_enabled(task) -> bool:
        return (
            bool(task.config.get("eval_mode", False))
            and bool(task.config.get("eval_hide_avatar_until_trigger", False))
            and task.avatar is not None
        )

    @staticmethod
    def eval_hidden_avatar_pos(task) -> np.ndarray:
        pos = task.config.get("eval_hidden_avatar_pos")
        if pos is None:
            return task.EVAL_HIDDEN_AVATAR_POS.copy()
        arr = np.asarray(pos, dtype=np.float64).ravel()
        if arr.size != 3:
            raise ValueError("eval_hidden_avatar_pos must be a 3-vector")
        return arr.copy()

    def hide_eval_avatar_until_trigger(self, task) -> None:
        task.avatar.reset(
            task._eval_hidden_avatar_pos(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        """Capture ``palm_default`` once, burn the retired trigger draw and
        arm the shared progress trigger.  The avatar stays where it stands
        until the trigger fires."""
        task._eval_palm_default = task._dryrun_capture_palm()
        # The release-v1 random-step trigger drew here in both branches; keep
        # the draw so the global RNG tail per seed is unchanged.
        task._eval_trigger_step = burn_legacy_trigger_draw(task)
        # Eval plays the clip at the glided-to root; never at a stored
        # scripted-flow motion root (that would teleport the avatar).
        task._inspect_motion_root_pos = None
        task._policy_step_count = 0
        task._eval_target_cube = None
        episode = new_interrupt_episode(task, hand_ids=(int(task.AVATAR_HAND_ID),))
        episode.trigger.arm(robot_tcp(task), self.eligible_cubes(task))
        if task._eval_palm_default is None:
            task._eval_interrupt_fired = True
            episode.extra["calibration_failed"] = True
            episode.trigger.armed = False
            episode.trigger.block_reason = "avatar_calibration_failed"
        else:
            task._eval_interrupt_fired = False

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
        candidates = self.eligible_cubes(task)
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
        episode = interrupt_episode(task)
        cube = task.objects[idx]
        cube_pos = task._dump_object_center(cube).astype(float)
        shift = cube_pos - task._eval_palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task._interrupted_idx = int(idx)
        task._eval_target_cube = cube
        entity = getattr(cube, "entity", cube)

        def cube_center():
            return task._dump_object_center(cube).astype(float)

        def start_clip() -> bool:
            task._play_inspect_motion(
                attach_obj=entity,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=task._attach_frame_scaled,
                detach_frame=task._detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=cube_center,
                grasped=lambda: robot_grasping(task, cube_center()),
            )
            return True

        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=cube_pos,
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

    def _cube_in_bin(self, task, p) -> bool:
        big_x, big_y = task.big_bin_pose.p[:2]
        x_half, y_half = task._big_bin_xy_half
        return (abs(p[0] - big_x) < x_half and abs(p[1] - big_y) < y_half
                and p[2] < task._big_bin_rim_z + task.SUCCESS_RIM_MARGIN)

    def eligible_cubes(self, task) -> dict:
        """Cubes on the table inside the avatar-reachable band, not in the
        bin and not held by the robot."""
        INT_X_LO, INT_X_HI = task.INTERRUPT_SPAWN_X_RANGE
        INT_Y_LO, INT_Y_HI = task.INTERRUPT_SPAWN_Y_RANGE
        out = {}
        for i, cube in enumerate(task.objects):
            p = task._dump_object_center(cube).astype(float)
            if self._cube_in_bin(task, p):
                continue
            if (
                p[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or p[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            if (
                not bool(task.config.get("interrupt_use_base_cube_layout", False))
                and not (INT_X_LO <= p[0] <= INT_X_HI and INT_Y_LO <= p[1] <= INT_Y_HI)
            ):
                continue
            if robot_grasping(task, p):
                continue
            out[i] = p
        return out

    def robot_holding_any(self, task) -> bool:
        for cube in task.objects:
            p = task._dump_object_center(cube).astype(float)
            if p[2] > task.TABLE_TOP_Z + 0.04 and robot_grasping(task, p):
                return True
        return False

    # -- candidates ------------------------------------------------------

    def closest_table_cube_candidate(self, task):
        """Index and xy distance of the cube closest to the right-arm EE that
        is still on the table and inside the avatar-reachable band; the band
        keeps the avatar on the +Y side instead of sliding into the table."""
        arm = task.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        big_x, big_y = task.big_bin_pose.p[:2]
        x_half, y_half = task._big_bin_xy_half
        rim_z = task._big_bin_rim_z

        best_idx = None
        best_d = float("inf")
        INT_X_LO, INT_X_HI = task.INTERRUPT_SPAWN_X_RANGE
        INT_Y_LO, INT_Y_HI = task.INTERRUPT_SPAWN_Y_RANGE
        for i, cube in enumerate(task.objects):
            p = task._dump_object_center(cube)
            in_bin = (abs(p[0] - big_x) < x_half and
                      abs(p[1] - big_y) < y_half and
                      p[2] < rim_z + task.SUCCESS_RIM_MARGIN)
            if in_bin:
                continue
            if (
                p[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN or
                p[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            if (
                not bool(task.config.get("interrupt_use_base_cube_layout", False))
                and not (INT_X_LO <= p[0] <= INT_X_HI and
                         INT_Y_LO <= p[1] <= INT_Y_HI)
            ):
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx, best_d

    # -- metrics ---------------------------------------------------------

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        out: dict[str, Any] = {}
        if hasattr(task, "_interrupted_idx"):
            out["interrupted_object_index"] = int(task._interrupted_idx)
        if hasattr(task, "_interrupt_pick_ordinal"):
            out["interrupt_pick_ordinal"] = int(task._interrupt_pick_ordinal) + 1
        if hasattr(task, "_interrupt_pick_order"):
            out["robot_pick_order"] = [int(i) for i in task._interrupt_pick_order]
        if bool(task.config.get("eval_mode", False)):
            out["eval_mode"] = True
            out["eval_policy_step_count"] = int(getattr(
                task, "_policy_step_count", 0,
            ))
            out["eval_interrupt_fired"] = bool(getattr(
                task, "_eval_interrupt_fired", False,
            ))
            out["interrupt"] = interrupt_metrics(task)
            out["pace_interrupt"] = legacy_pace_block(task)
        return out
