"""Dump bin (Interruption, ``dump_bin_interruption``): same scene as ``dump_bin`` but with a
human avatar that picks up one of the cubes mid-task.

Flow per episode:
  1. Robot picks zero or more non-interrupted cubes, depending on the
     randomly sampled interrupt pick ordinal.
  2. Robot approaches cube[INTERRUPTED_IDX] (above-pose, no commit) when
     that ordinal is reached.
  3. Avatar starts the ``Inspect`` motion (categorize_objects_interruption's
     verbatim animation) with ``attach_obj`` = the interrupted cube.
     The cube is attached to the avatar's hand at frame ATTACH and
     detached at DETACH — the avatar lifts it, holds it, and sets it
     back on the table.
  4. Robot retreats to a safe pose (toward the robot base, away from
     the avatar's lean cone).
  5. While the avatar holds the interrupted cube, the robot picks any
     remaining OTHER cubes and drops them in the bin — all pure physics.
  6. Robot waits for ``avatar.spare()``.
  7. Robot reads cube[INTERRUPTED_IDX]'s LATEST pose (after the avatar
     puts it back down) and picks it via the same pure-physics primitive.

Geometry pipeline (mirrors dump_bin_contribution's "dry run → shift" idiom):
  * Pre-bake: dry-run the Inspect motion at ``avatar_init_pos`` and
    measure the avatar's right-palm position at ATTACH frame.
  * Constraint solve: avatar shift = cube[INTERRUPTED].pos -
    palm_at_attach.  After shifting, the avatar palm meets the cube
    exactly at the attach frame so the attach is geometrically correct.
  * Every cube starts inside the same avatar-reachable band near the
    +Y table edge, so the interrupted target is not revealed by placement.

Eval: success = ALL robot-side cubes inside the bin.

Robot motion is pure physics (no kinematic teleport/attach).  Avatar
attaches the cube to its hand via ``play_animation(attach_obj=...)``,
which is the sanctioned avatar mechanism.

Eval mode (``envs/interaction/interrupt_progress.py``): the human
inspects the cube the robot is approaching once the shared progress trigger
fires, glides in, inspects, returns to idle.  The policy action is never
replaced.  The full-state expert yields via ``interrupt_expert_attempt`` /
``interrupt_expert_retreat``; ``play_blind_once`` is the no-intent expert.

The human side (interrupted-cube choice, calibration, the eval-mode
interrupt contract) is owned by the composed
:class:`DumpBinInterruptController`; this class keeps the robot choreography
and thin ``_``-prefixed delegators.
"""

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..interaction.dump_bin import DumpBinInterruptController
from ..task_bases.dump_bin import DumpBin
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class DumpBinInterruption(InspectMotionInterruptFacade, DumpBin):
    INSTRUCTION = (
        "place each cube into the trash can; the human will inspect "
        "one cube midway through - wait, then continue."
    )

    use_avatar = True
    HARD_NUM_OBJECTS = 3

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # AVATAR_BASE_ROT maps mesh-+X (avatar forward) to world-(-Y).
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # categorize_interrupt's Inspect motion + frame markers.  The same
    # 10× slowdown reads as a deliberate inspection in video.
    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34       # original frame
    INSPECT_DETACH_FRAME  = 187      # original frame
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW    = 10.0     # avatar.frame_ratio
    AVATAR_HAND_ID        = 1        # 1 = right hand
    AVATAR_EXTRA_Z        = 0.04     # palm-above-cube margin so attach
                                     # doesn't sink into the cube
    EVAL_HIDDEN_AVATAR_POS = np.array([0.0, 4.0, -0.18], dtype=np.float64)
    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.8

    # Shared avatar-reachable spawn band near the +Y table edge.  All cubes
    # are sampled here so the robot cannot infer the interrupt target from
    # which object is uniquely close enough for the avatar to inspect.
    AVATAR_REACH_SPAWN_X_RANGE = (-0.18, 0.18)
    AVATAR_REACH_SPAWN_Y_RANGE = (-0.12, 0.06)
    INTERRUPT_SPAWN_X_RANGE = AVATAR_REACH_SPAWN_X_RANGE
    INTERRUPT_SPAWN_Y_RANGE = AVATAR_REACH_SPAWN_Y_RANGE
    INTERRUPT_LAYOUT_SAMPLE_ATTEMPTS = 80

    # Threshold for success — override parent's ⌈N/2⌉ to require ALL
    # cubes in bin (avatar puts its cube back; robot must pick that too).
    SUCCESS_THRESHOLD = None         # None ⇒ num_objects (all in)

    # ------------------------------------------------------------------
    # Eval-mode interrupt contract (config["eval_mode"]=True); the scripted
    # collection flow (collect.py / play_once) ignores these.
    # ------------------------------------------------------------------
    # Bounds of the retired random-step trigger.  Only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 300
    # Progress trigger (3-D TCP to cube centre), fitted to the
    # demonstrated intent hover (transport height, 0.14-0.15 m from the cube,
    # progress 0.49-0.53 -- right at the generic 0.5, so a straight hover
    # could miss it).  0.40 fires on the approach/hover on all 5 probed demos.
    INTERRUPT_COMMIT_RADIUS_M = 0.22
    INTERRUPT_TRIGGER_PROGRESS = 0.40
    INTERRUPT_MIN_POLICY_STEPS = 20
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30
    EVAL_AVATAR_BIN_CLEARANCE = 0.20
    CALIBRATION_SETTLE_STEPS = 20
    POST_CALIBRATION_SETTLE_STEPS = 30
    POST_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    POST_AVATAR_RETURN_SETTLE_STEPS = 30
    # None ⇒ the avatar freezes on the inspect motion's last frame (pose +
    # position) instead of easing back to the neutral idle pose.
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    RETREAT_BACK_DISTANCE = 0.15
    RETREAT_UP_DISTANCE = 0.15

    def create_interaction_controller(self) -> DumpBinInterruptController:
        return DumpBinInterruptController()

    def _interrupt_controller(self) -> DumpBinInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, DumpBinInterruptController):
            raise TypeError("task does not use dump-bin interrupt behavior")
        return controller

    # ------------------------------------------------------------------
    # Scene setup — the controller picks the interrupted cube / ordinal and
    # samples the avatar-reachable layout in the legacy draw order.
    # ------------------------------------------------------------------

    def _sample_cube_xys(self):
        return self._interrupt_controller().sample_cube_xys(
            self, base_sampler=lambda: super(DumpBinInterruption, self)._sample_cube_xys(),
        )

    def _sample_interrupt_pick_ordinal(self) -> int:
        return self._interrupt_controller().sample_interrupt_pick_ordinal(self)

    # ------------------------------------------------------------------
    # Reset — calibration / eval arming run in the controller's
    # after_base_reset_observation hook (inspect resolved after the initial
    # observation, then eval state armed or the avatar pre-shifted).  Only
    # the hidden-avatar observation refresh stays here: a lifecycle hook
    # cannot change the observation BaseTask.reset returns.
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self._eval_hide_avatar_until_trigger_enabled():
            self._hide_eval_avatar_until_trigger()
            if not self.config.get("skip_reset_obs", False):
                obs = self.get_obs()
        return obs

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    # ------------------------------------------------------------------
    # Eval-mode interrupt state machine (config["eval_mode"]=True) — owned
    # by the controller; ``take_action`` below is the application boundary.
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        self._interrupt_controller().init_eval_state(self)

    def _eval_hide_avatar_until_trigger_enabled(self) -> bool:
        return self._interrupt_controller().eval_hide_avatar_until_trigger_enabled(self)

    def _eval_hidden_avatar_pos(self) -> np.ndarray:
        return self._interrupt_controller().eval_hidden_avatar_pos(self)

    def _hide_eval_avatar_until_trigger(self) -> None:
        self._interrupt_controller().hide_eval_avatar_until_trigger(self)

    def take_action(self, action, action_type: str = "qpos"):
        """Tick the eval-mode interrupt, then apply the policy action
        unmodified (no env-forced robot motion)."""
        if bool(self.config.get("eval_mode", False)) and self.avatar is not None:
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def play_blind_once(self) -> bool:
        """No-intent scripted baseline.

        The robot follows the ordinary dump-bin script without reading the
        interrupt or avatar state; it re-reads each cube's live pose before
        picking it, so it re-acquires a cube the human moved.  The expert
        harness ticks the eval-mode human at policy cadence
        (``ScriptedRolloutDriver``); this method does not tick it itself.
        """
        from ..task_bases.dump_bin_motion import DumpMotionRejected
        try:
            return DumpBin.play_once(self)
        except DumpMotionRejected:
            return False

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def eval_skip_policy_action(self) -> bool:
        return self._interrupt_controller().eval_skip_policy_action(self)

    def consume_eval_policy_reset_requested(self) -> bool:
        return self._interrupt_controller().consume_eval_policy_reset_requested(self)

    def _find_closest_table_cube(self):
        best_idx, _ = self._closest_table_cube_candidate()
        return best_idx

    def _closest_table_cube_candidate(self):
        return self._interrupt_controller().closest_table_cube_candidate(self)

    # ------------------------------------------------------------------
    # Robot retreat
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        """Pull the EE toward the robot base (−Y from cube zone) and lift
        — clears the avatar's forward-lean cone before the avatar's
        attach frame fires.
        """
        from ..grasp import tcp_to_link_pose

        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        current_tcp = np.array(arm.get_ee_pose()[:3], dtype=float)
        to_base_xy = arm_base[:2] - current_tcp[:2]
        norm = float(np.linalg.norm(to_base_xy))
        if norm < 1e-6:
            to_base_xy = np.array([0.0, -1.0], dtype=float)
        else:
            to_base_xy = to_base_xy / norm
        retreat_pos = current_tcp.copy()
        retreat_pos[:2] += to_base_xy * self.RETREAT_BACK_DISTANCE
        retreat_pos[2] += self.RETREAT_UP_DISTANCE
        retreat_link = tcp_to_link_pose(self._top_down_tcp(retreat_pos), arm.tcp_offset)
        self._move_seeded(retreat_link.to_pose7(), arm_tag)

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        if self.avatar_collider is None:
            return None
        return sample_capsule_planner_points(
            self.avatar_collider.current_capsules(),
            inflate_factor=self._AVATAR_INFLATE_FACTOR,
        )

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        pts = self._avatar_obstacle_points()
        if pts is None or pts.size == 0:
            return
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception as e:
            print(f"[dump_bin_interrupt] obstacle update failed: {e}")

    def _move_seeded(self, link_pose7, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_seeded(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=0.0):
        # The override exists only to refresh the avatar obstacle cloud before
        # the base plans. It must therefore accept and forward the base's full
        # signature: dump_bin_grasp calls this with tcp_yaw for every oriented
        # top-down pick, and dropping the argument raised TypeError mid-episode.
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag, tcp_yaw)

    def _return_avatar_home_after_inspect(self):
        # The avatar now freezes on the inspect motion's last frame (pose +
        # position) instead of sliding back to avatar_init_pos.  Hold in place
        # for the same settle window so the released cube comes to rest while
        # the human stands where the motion ended.
        if self.avatar is None:
            return
        for _ in range(max(1, int(self.POST_AVATAR_RETURN_SETTLE_STEPS))):
            self.step_sim()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        from ..task_bases.dump_bin_motion import DumpMotionRejected
        try:
            return self._play_dump_interrupt()
        except DumpMotionRejected:
            return False

    def _play_dump_interrupt(self) -> bool:
        from ..grasp import tcp_to_link_pose

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # Settle table objects.
        for _ in range(self.TABLE_SETTLE_STEPS):
            self.step_sim()

        if self.avatar is None:
            return super().play_once()

        interrupted_idx = self._interrupted_idx
        interrupt_ordinal = int(getattr(self, "_interrupt_pick_ordinal", 0))
        int_cube = self.objects[interrupted_idx]

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE

        # Build a normal pick sequence with the interrupted cube placed at
        # the sampled ordinal.  This randomizes whether the human interrupts
        # the robot's 1st, 2nd, or 3rd attempted pick while preserving the
        # avatar-reachable interrupted cube identity chosen at reset.
        other_indices = [i for i in range(self.num_objects) if i != interrupted_idx]
        pick_order = (
            other_indices[:interrupt_ordinal] +
            [interrupted_idx] +
            other_indices[interrupt_ordinal:]
        )
        self._interrupt_pick_order = list(pick_order)

        # ---- Phase 1: pick cubes before the interrupt ordinal ----
        for idx in pick_order[:interrupt_ordinal]:
            ok = self._pick_and_drop_cube(self.objects[idx], arm_tag)
            if not ok:
                return False

        # ---- Phase 2: robot shows intent on the interrupted cube ----
        int_pos = self._dump_object_center(int_cube).astype(float)
        self.open_gripper(arm_tag)
        above_pos = np.array([int_pos[0], int_pos[1], transport_z])
        above_link = tcp_to_link_pose(self._top_down_tcp(above_pos), tcp_offset)
        self._move_seeded(above_link.to_pose7(), arm_tag)

        # ---- Phase 3: avatar plays Inspect with attach ----
        # Non-blocking — frames will tick during subsequent step_sim()s.
        self._play_inspect_motion(
            attach_obj=getattr(int_cube, "entity", int_cube),
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 4: robot retreats so the avatar can grab the cube ----
        self._retreat_safe(arm_tag)
        # Keep stepping until the avatar reaches the attach frame so the
        # cube is visibly held before the robot moves on.
        self._wait_until_avatar_attach()
        for _ in range(self.POST_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 5: wait for avatar to release the interrupted cube ----
        # Keep the robot retreated until the human has finished the inspection.
        # Earlier versions tried to pick remaining cubes while the avatar held
        # the interrupted one; that made the expert full-state path plan through
        # the avatar on seeds where the remaining cube was still near the human.
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        self._return_avatar_home_after_inspect()
        # Give the released cube a moment to settle on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 6: pick remaining non-interrupted cubes ----
        for idx in pick_order[interrupt_ordinal + 1:]:
            cube = self.objects[idx]
            ok = self._pick_and_drop_cube(cube, arm_tag)
            if not ok:
                return False

        # ---- Phase 7: pick the released cube using its LATEST pose ----
        # _pick_and_drop_cube reads the live object position inside, so it always
        # uses the current world position (avatar may have moved it).
        ok = self._pick_and_drop_cube(int_cube, arm_tag)
        if not ok:
            return False

        self._ensure_final_gripper_z(arm_tag)
        return True

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def _cube_in_bin(self, cube) -> bool:
        return self._interrupt_controller()._cube_in_bin(
            self, self._dump_object_center(cube).astype(float),
        )

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: drop every cube that is not yet in
        the bin, reading each cube's live pose."""
        from ..task_bases.dump_bin_motion import DumpMotionRejected

        arm_tag = "right"
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            self._boost_finger_pd(arm_tag)
            with self.suppress_recording():
                for _ in range(self.TABLE_SETTLE_STEPS):
                    self.step_sim()
        try:
            for cube in self.objects:
                if self._cube_in_bin(cube):
                    continue
                if not self._pick_and_drop_cube(cube, arm_tag):
                    return False
            self._ensure_final_gripper_z(arm_tag)
        except DumpMotionRejected:
            return False
        return True

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")

    def evaluate(self) -> dict:
        out = super().evaluate()
        out.update(self.interaction_controller.interaction_metrics(self))
        return out

    # ------------------------------------------------------------------
    # Eval — require ALL cubes in the bin
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if self.avatar_collision_summary().get("any_collision"):
            return False
        # No robot-yield phase gates; a cube the human still holds
        # is simply not in the bin.

        big_x, big_y = self.big_bin_pose.p[:2]
        rim_z = self._big_bin_rim_z
        x_half, y_half = self._big_bin_xy_half

        threshold = self.SUCCESS_THRESHOLD or self.num_objects

        in_count = 0
        for i, cube in enumerate(self.objects):
            p = self._dump_object_center(cube)
            in_xy = abs(p[0] - big_x) < x_half and abs(p[1] - big_y) < y_half
            below_r = p[2] < rim_z + self.SUCCESS_RIM_MARGIN
            above_f = p[2] > self.SUCCESS_FLOOR_Z_MIN
            inside = in_xy and below_r and above_f
            if inside:
                in_count += 1

        return in_count >= threshold
