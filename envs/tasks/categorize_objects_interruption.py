"""Categorize objects (Interruption): the human inspects an object the robot is about to grasp."""

import numpy as np

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..interaction.inspect_motion import CategorizeInterruptController
from ..interaction.interrupt_progress import (
    DEFAULT_GLIDE_POLICY_STEPS,
    burn_legacy_trigger_draw,
    policy_dt_s,
    interrupt_episode,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    robot_tcp,
)
from ..task_bases.categorize import CategorizeTaskCore, PickSpec, PlaceSpec
from ..task_bases.categorize_support.motion import stop_on_motion_rejection
from ..utils import to_numpy
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class CategorizeObjectsInterruption(InspectMotionInterruptFacade, CategorizeTaskCore):
    """Robot categorizes objects, yielding while the human inspects one."""

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Training-pilot defaults, scoped to this task. Explicit false keeps
        # the original behavior available for comparisons and diagnostics.
        cfg.setdefault("categorize_payload_friction_v1", True)
        cfg.setdefault("categorize_split_screw_v1", True)
        super().__init__(cfg)

    use_avatar = True
    _CATEGORIZE_COLUMN_LAYOUT = True
    _CATEGORIZE_ROBOT_Y = -0.5
    _CATEGORIZE_FINGER_KV = 200.0
    _CATEGORIZE_OPEN_LIDS = False

    INSTRUCTION = "sort the objects into the matching baskets"

    # Re-enable the mplib obstacle-inflation that the parent disables —
    # row 2 cate-inte agent validated this end-to-end (10/10 PASS) on
    # cate-inte's column-aligned transits.
    _USE_OBSTACLE_INFLATION = True

    # The interrupt layout is already column-aligned, and the preferred
    # standard-helper candidates validated 5/5 on seeds 1-5.
    PREFERRED_GRASP_NAMES = {
        "035_apple": (
            "manual_center_p0_yaw0",
            "manual_center_p0_yaw2",
            "manual_underhand_p0_yaw0",
            "manual_underhand_p0_yaw2",
        ),
        "086_woodenblock": (
            "manual_topdown_yaw45_center",
            "manual_topdown_yaw45_higher",
        ),
        "100_seal": (
            "manual_topdown_X_center_m2",
            "manual_topdown_Y_center_m2",
            "manual_grasp_015_yaw90_deeper_upper",
        ),
    }

    # Table extended in −Y so the robot base (at y=-0.5) sits on the tabletop
    # instead of floating behind it.  Front (+Y, avatar-side) edge unchanged at
    # y=+0.35 so the avatar still has room to lean in during Inspect.
    #   back edge = dy − half_y = -0.15 − 0.50 = -0.65  (robot at y=-0.5 sits 0.15 m inboard)
    #   front edge = dy + half_y = -0.15 + 0.50 = +0.35
    table_offset = np.array([0.0, -0.15])
    TABLE_HALF_SIZE_Y = 0.50
    TABLE_LEGS_HALF_Y = 0.45

    # Avatar root sits ~0.18m below the mesh feet in cooperative. The final
    # per-episode root pose is shifted by
    # `_calibrate_avatar_for_inspect` so the Inspect palm reaches the selected
    # object.
    avatar_init_pos = np.array([0.0, 0.70, -0.18])

    # Extra Z lift applied to the avatar on top of the palm-meets-object shift
    # computed by `_calibrate_avatar_for_inspect`.  Keeps palm just above the
    # resting object at attach instead of sunk into the table.
    AVATAR_EXTRA_Z = 0.06

    # Legacy fallback only.  By default the interrupt target is sampled from
    # the scripted robot pick order so the human interrupts whichever ordinal
    # pick the robot is about to reach for.
    INTERRUPTED_IDX = 1

    # Put robot-side loose objects closer to the avatar-side table edge than
    # the original y=0.10 layout.  Calibration shifts the avatar root so palm
    # meets this object at attach; moving the target band toward +Y keeps the
    # avatar body out of the tabletop while preserving robot reach.
    INTERRUPT_OBJECT_Y = 0.18

    # Motion slowdown factor: resample Mixamo frames to play 10x slower so
    # the Inspect motion reads as a deliberate pause in the final video.
    AVATAR_MOTION_SLOW = 10.0

    # Inspect motion timing (after clipping frames 13..225 of original).
    # Original F47 -> clipped index 34 (attach), F200 -> clipped 187 (detach).
    # These are in the ORIGINAL (frame_ratio=1.0) frame space; they get
    # scaled by AVATAR_MOTION_SLOW at runtime.
    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL

    # Retreat is empty-handed and only needs to clear the avatar's workspace,
    # so it can use a faster replay than object transport/grasp motions.
    RETREAT_REPLAY_SUBSTEPS = 1

    # VLA primary view: near top-down over the lifted 1.60 x 1.00 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, -0.145, 2.85],
        "forward": [0.0, -0.005, -1.35],
    }]

    # Side view of table and avatar inspecting object.
    recording_camera_pos = [1.7, 0.05, 1.85]
    recording_camera_lookat = [0.0, 0.0, 1.55]
    # Top-down secondary, nudged toward the robot-arm side (-Y).
    side_camera_pos = [0.0, -0.25, 2.7]
    side_camera_lookat = [0.0, -0.25, 1.30]

    # Unified with the cooperative base: objects sampled per-episode from the
    # shared pool (NUM_CATEGORIES distinct draws).  Inherits OBJECT_SET,
    # NUM_CATEGORIES, and the default CATEGORIES from CategorizeObjectsContribution.

    def _scripted_pick_order(self):
        """Object indices in the order the robot will attempt them."""
        if hasattr(self, "_sort_targets"):
            n_targets = len(self._sort_targets)
        else:
            n_targets = len(self.CATEGORIES)
        return sorted(range(n_targets), reverse=True)
    def _calibrate_avatar_for_inspect(self, interrupted_idx):
        """Dry-run the Inspect motion with the default avatar pose, capture the
        palm position at the attach frame, then reset the avatar shifted so
        its palm lands exactly on the resting interrupted object."""
        int_actor = self._sort_targets[interrupted_idx][0]
        obj_pos = to_numpy(int_actor.entity.get_pos()).ravel()[:3]
        target_palm = np.array(obj_pos, dtype=np.float64)

        # This calibration happens inside reset().  Normal MP4 capture starts
        # after reset, but VLA capture is ticked directly from step_sim once
        # BaseTask.reset creates the recorder, so suppress recording here.
        saved_vla_recorder = self.vla_recorder
        saved_record_stride = self._record_stride
        self.vla_recorder = None
        self._record_stride = None
        try:
            self.avatar.play_animation(self.INSPECT_MOTION_NAME)
            palm_default = None
            step = 0
            attach_frame_scaled = self._inspect_attach_frame_scaled
            while not self.avatar.spare():
                self.step_sim()
                if step == attach_frame_scaled:
                    palm_default = np.asarray(
                        self.avatar.robot.get_palm_center(1), dtype=np.float64
                    ).copy()
                step += 1
        finally:
            self.vla_recorder = saved_vla_recorder
            self._record_stride = saved_record_stride

        self._eval_palm_default = (
            None if palm_default is None else np.asarray(palm_default, dtype=np.float64).copy()
        )
        if palm_default is None:
            print("[categorize_interrupt] WARNING: could not capture palm at attach frame")
            self.avatar.reset(
                np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
            for _ in range(20):
                self.step_sim()
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(30):
            self.step_sim()
        print(f"[categorize_interrupt] shifted avatar by {shift} -> new pos {shifted_pos} "
              f"(object rests at {obj_pos})")

    def _retreat_from_object(self, arm_tag):
        """Retreat away from the avatar first (−Y), then up — a straight
        lift overlapped with the avatar's forward-lean during Inspect.

        Uses a screw-motion from current EE position to a safe pose in front
        of the robot base so the arm pulls toward the robot (−Y) while lifting.
        """
        arm = self.robot.get_arm(arm_tag)
        safe_z = self.TABLE_TOP_Z + 0.45
        arm_base = np.array(arm.origin_pose.p)

        # Target: front-of-robot-base in (x=arm_base_x), y=arm_base_y+0.30
        # ≈ −0.20 world, z=safe_z.  The screw path gives a single diagonal
        # motion that clears the avatar's approach cone.
        retreat_pos = np.array([arm_base[0], arm_base[1] + 0.30, safe_z])
        old_replay_substeps = self._REPLAY_SUBSTEPS
        try:
            self._REPLAY_SUBSTEPS = self.RETREAT_REPLAY_SUBSTEPS
            self._move_screw(retreat_pos, arm_tag)
        finally:
            self._REPLAY_SUBSTEPS = old_replay_substeps
        self._move_to_safe(arm_tag)
    # Eval-mode interrupt (progress-trigger contract, envs/interaction/
    # interrupt_progress.py).  The human reaches for whichever unsorted object
    # the robot is approaching once it commits to it (progress trigger); there
    # is no time-based fallback and no pre-chosen eval target.  The avatar
    # stands calibrated for ``_interrupted_idx`` at reset (as in training) and
    # glides from there to the approached object.
    #
    # Bounds of the retired random-step trigger; only its RNG draw remains,
    # burned at reset so seeded scenes keep their release-v1 stream (and the
    # Co-presence sibling's draw cadence stays meaningful).
    EVAL_TRIGGER_STEP_MIN = 80
    EVAL_TRIGGER_STEP_MAX = 180
    # Fitted to the demonstrated intent hover (training data hr_v2_nyx37,
    # demo-replay probe in interrupt_check): the TCP waits 0.218 m above the
    # object centre with progress 0.33-0.38.  The generic 0.22 m / 0.5 never
    # fired there (ACT/Diffusion hovered until timeout); 0.26 / 0.25 fires
    # 17-24 steps before the demo human starts on all 6 probed demos.
    INTERRUPT_COMMIT_RADIUS_M = 0.26
    INTERRUPT_TRIGGER_PROGRESS = 0.25
    INTERRUPT_MIN_POLICY_STEPS = 20
    # An object within this xy distance of its basket centre counts as sorted.
    INTERRUPT_SORTED_XY_M = 0.10

    # Minimum human glide duration in sim seconds.  This task runs at
    # dt=0.001 while eval rows use action_substeps=25, i.e. 25 ms policy
    # steps, so the shared 30-step default would be only 0.75 s.
    INTERRUPT_GLIDE_MIN_S = 1.5

    def _eval_at_reset(self):
        self._eval_inspect_fired = False
        self._eval_interrupt_fired = False
        self._eval_legacy_trigger_draw = burn_legacy_trigger_draw(self)
        if "interrupt_glide_policy_steps" not in self.config:
            dt = policy_dt_s(self)
            self.INTERRUPT_GLIDE_POLICY_STEPS = max(
                DEFAULT_GLIDE_POLICY_STEPS,
                int(np.ceil(float(self.INTERRUPT_GLIDE_MIN_S) / max(dt, 1e-6) - 1e-9)),
            )
        new_interrupt_episode(self, hand_ids=(1,))

    def _arm_eval_interrupt(self) -> None:
        episode = interrupt_episode(self)
        if episode is None:
            return
        if getattr(self, "_eval_palm_default", None) is None:
            episode.trigger.block_reason = "avatar_calibration_failed"
            episode.extra["calibration_failed"] = True
            return
        episode.trigger.arm(robot_tcp(self), self._interrupt_candidates())

    def _object_sorted(self, idx: int) -> bool:
        actor, name, model_id, basket_idx = self._sort_targets[idx]
        center = self._get_object_world_center(actor, name, model_id)
        basket = np.asarray(self.basket_poses[basket_idx].p, dtype=float)
        return bool(
            float(np.linalg.norm(np.asarray(center)[:2] - basket[:2]))
            <= float(self.INTERRUPT_SORTED_XY_M)
        )

    def _interrupt_candidates(self) -> dict:
        """Unsorted objects on the table that the robot is not holding."""
        out = {}
        for idx, (actor, name, model_id, _basket) in enumerate(self._sort_targets):
            if self._object_sorted(idx):
                continue
            center = np.asarray(
                self._get_object_world_center(actor, name, model_id), dtype=float,
            )
            if (center[2] < self.TABLE_TOP_Z - 0.05
                    or center[2] > self.TABLE_TOP_Z + 0.30):
                continue
            if robot_grasping(self, center):
                continue
            out[idx] = center
        return out

    def _robot_holding_any(self) -> bool:
        for actor, name, model_id, _basket in self._sort_targets:
            center = np.asarray(
                self._get_object_world_center(actor, name, model_id), dtype=float,
            )
            if center[2] > self.TABLE_TOP_Z + 0.04 and robot_grasping(self, center):
                return True
        return False

    def _eval_at_step(self, step_idx: int):
        episode = interrupt_episode(self)
        if episode is None:
            return
        step_idx = int(step_idx)
        episode.policy_steps = step_idx
        if episode.human.phase != "idle":
            episode.human.tick(self, step_idx)
            return
        if episode.fired or not episode.trigger.armed:
            return
        holding = self._robot_holding_any()
        key = episode.trigger.update(
            step_idx,
            robot_tcp(self),
            self._interrupt_candidates(),
            gate_ok=not holding,
            gate_reason="robot_holding_object",
        )
        if key is None:
            return
        self._eval_inspect_fired = True
        self._eval_interrupt_fired = True
        self._start_interrupt_human(int(key), step_idx)

    def _start_interrupt_human(self, idx: int, step_idx: int) -> None:
        episode = interrupt_episode(self)
        actor, name, _model_id, _basket = self._sort_targets[idx]
        obj_pos = to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)
        shift = obj_pos - np.asarray(self._eval_palm_default, dtype=np.float64)
        shift[2] += self.AVATAR_EXTRA_Z
        target_root = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        self._interrupted_idx = int(idx)
        print(f"[categorize_interrupt] eval: human goes for {name} "
              f"(idx={idx}) at policy step {step_idx}", flush=True)

        def object_pos():
            return to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)

        def start_clip() -> bool:
            # Never snap to a stored playback root: the glide already put the
            # avatar where the clip must start.
            self._inspect_motion_root_pos = None
            self._play_inspect_motion(
                attach_obj=actor.entity,
                hand_id=1,
                attach_frame=self._inspect_attach_frame_scaled,
                detach_frame=self._inspect_detach_frame_scaled,
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=self.INSPECT_MOTION_NAME,
                object_pos=object_pos,
                grasped=lambda: robot_grasping(self, object_pos()),
            )
            return True

        episode.human.start(
            step=step_idx,
            start_root=np.asarray(self.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(self.avatar_init_rot, dtype=np.float64),
            target_key=int(idx),
            target_pos=obj_pos,
            start_clip=start_clip,
        )

    def _basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for actor, obj_name, model_id, basket_idx in self._sort_targets:
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
            rel_lo = lo_xy - basket_pos[:2]
            rel_hi = hi_xy - basket_pos[:2]
            center_ok = bool(dist_xy <= center_threshold)
            footprint_ok = bool(
                np.all(rel_lo >= -footprint_half)
                and np.all(rel_hi <= footprint_half)
            )
            resting_ok = self._object_resting_in_basket(
                actor, obj_center, basket_pos,
            )
            ok = bool(center_ok and footprint_ok and resting_ok)
            all_correct = all_correct and ok
            per_object.append({
                "object": obj_name,
                "model_id": int(model_id),
                "basket_idx": int(basket_idx),
                "object_pos": obj_center.tolist(),
                "basket_pos": basket_pos.tolist(),
                "dist_xy": dist_xy,
                "rel_aabb_lo_xy": rel_lo.tolist(),
                "rel_aabb_hi_xy": rel_hi.tolist(),
                "center_success": center_ok,
                "footprint_success": footprint_ok,
                "resting_success": resting_ok,
                "success": ok,
            })
        return {
            "basket_center_threshold": center_threshold,
            "basket_footprint_half_xy": float(footprint_half),
            "basket_all_correct": bool(all_correct),
            "basket_objects": per_object,
        }

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._basket_metrics())
        eval_mode = bool(self.config.get("eval_mode", False))
        if eval_mode:
            # These fields historically came from cooperative.evaluate before
            # the interrupt-specific fields were appended. Preserve values and
            # insertion order for release-v1 result artifacts.
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": interrupt_metrics(self).get("trigger_step"),
                "eval_policy_step_count": int(getattr(
                    self, "_eval_policy_step_count", 0,
                )),
                "eval_avatar_started": bool(getattr(
                    self, "_eval_avatar_started", False,
                )),
                "eval_avatar_target_idx": int(getattr(
                    self, "_eval_avatar_target_idx", 0,
                )),
                "eval_avatar_order": [
                    self._avatar_sort_targets[i][1]
                    for i in getattr(self, "_eval_avatar_pick_order", [])
                ],
            })
        metrics.update({
            "interrupted_idx": int(getattr(self, "_interrupted_idx", -1)),
            "interrupted_pick_ordinal": int(getattr(
                self, "_interrupted_pick_ordinal", -1,
            )),
            "interrupted_object": (
                self._sort_targets[self._interrupted_idx][1]
                if hasattr(self, "_sort_targets")
                and hasattr(self, "_interrupted_idx")
                and 0 <= self._interrupted_idx < len(self._sort_targets)
                else None
            ),
        })
        if eval_mode:
            # eval_trigger_step is the progress-trigger step (None
            # when the robot never committed to an object).
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": interrupt_metrics(self).get("trigger_step"),
                "eval_policy_step_count": int(getattr(
                    self, "_eval_policy_step_count", 0,
                )),
                "eval_inspect_fired": bool(getattr(
                    self, "_eval_inspect_fired", False,
                )),
                "interrupt": interrupt_metrics(self),
                "pace_interrupt": legacy_pace_block(self),
            })
        return metrics

    @stop_on_motion_rejection
    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()

        interrupted_idx = self._interrupted_idx
        int_actor, int_name, int_model_id, int_basket = self._sort_targets[interrupted_idx]
        interrupted_pick_ordinal = int(getattr(self, "_interrupted_pick_ordinal", -1))
        pick_order = self._scripted_pick_order()
        print(
            f"[categorize_interrupt] interrupted_pick_ordinal="
            f"{interrupted_pick_ordinal}, interrupted_idx={interrupted_idx} "
            f"({int_name}), pick_order={pick_order}"
        )

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        safe_z = self.TABLE_TOP_Z + 0.45

        def pick_place_target(target_idx: int) -> bool:
            actor, obj_name, model_id, basket_idx = self._sort_targets[target_idx]
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            if not self.pick_and_place(pick, place, arm_tag):
                return False
            self._move_to_safe(arm_tag)
            return True

        # Phase 1: complete scripted picks before the interrupted ordinal. The
        # avatar should interrupt the object right before the robot reaches for
        # that pick, not always before the first pick.
        for ordinal, target_idx in enumerate(pick_order):
            if target_idx == interrupted_idx:
                break
            actor, obj_name, _, _ = self._sort_targets[target_idx]
            print(
                f"[categorize_interrupt] pre-interrupt pick ordinal={ordinal} "
                f"idx={target_idx} ({obj_name})"
            )
            if not pick_place_target(target_idx):
                return False

        # Phase 2: robot approaches the interrupted object (shows intent).
        self._move_to_safe(arm_tag)
        obj_center = self._get_object_world_center(int_actor, int_name, int_model_id)
        above_obj = self._intent_above_position(obj_center)
        above_link = tcp_to_link_pose(self._top_down_tcp(above_obj), tcp_offset)
        self.open_gripper(arm_tag)
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # Yield from the checked approach pose. The former extra100mm descent
        # put the fingers in the incoming inspection hand's path while the
        # diagonal retreat was still accelerating. Acquisition follows only
        # after the human finishes inspecting.

        # Phase 3: avatar plays Inspect, attaching/detaching the interrupted object.
        # `return_to_idle_after=None` freezes the avatar on the Inspect motion's
        # last frame (pose + position) while the robot finishes sorting.
        if self.avatar is not None:
            self._play_inspect_motion(
                attach_obj=int_actor.entity,
                hand_id=1,
                attach_frame=self._inspect_attach_frame_scaled,
                detach_frame=self._inspect_detach_frame_scaled,
                return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
            )

        # Phase 4: robot detects conflict and retreats to safe home.
        print(f"[categorize_interrupt] RETREATING — human is inspecting {int_name}")
        self._retreat_from_object(arm_tag)

        # Short wait while the human inspects — robot stays at safe.
        for _ in range(200):
            self.step_sim()

        # Phase 5: sort later non-interrupted objects after checked retreat.
        interrupted_order_pos = pick_order.index(interrupted_idx)
        for ordinal, target_idx in enumerate(pick_order[interrupted_order_pos + 1:],
                                             start=interrupted_order_pos + 1):
            actor, obj_name, _, _ = self._sort_targets[target_idx]
            print(
                f"[categorize_interrupt] post-interrupt pick ordinal={ordinal} "
                f"idx={target_idx} ({obj_name})"
            )
            if not pick_place_target(target_idx):
                return False

        # Phase 6: make sure the avatar has finished its Inspect + idle return
        # before the robot comes back for the interrupted object.  The idle
        # return is triggered automatically inside `play_animation`, so by the
        # time the robot is done sorting the non-interrupted objects it should
        # already be well underway.
        if self.avatar is not None:
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(100):
            self.step_sim()

        # Phase 7: pick the interrupted object now that the human is done.
        if not pick_place_target(interrupted_idx):
            return False
        for _ in range(100):
            self.step_sim()

        return True

    @staticmethod
    def _intent_above_position(obj_center):
        """Show intent within the arm workspace, above the later120mm hover.

        At the far column, the old table+450mm top-down target has no native
        IK solution. This target keeps100mm of straight descent before the
        inspection hover; normal scene collision/FK/arrival gates still apply.
        """
        center = np.asarray(obj_center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError('invalid interrupted object center')
        return center + np.array([0., 0., .22])

    def _sort_target_once(self, target_idx: int, arm_tag: str = "right") -> bool:
        actor, obj_name, model_id, basket_idx = self._sort_targets[target_idx]
        pick = PickSpec(
            get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
            radius=self._get_object_cross_section_radius(obj_name, model_id),
            label=obj_name,
            close_target_m=self._get_grasp_close_target(obj_name, model_id),
            actor=actor,
        )
        basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
        drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
        place = PlaceSpec(
            pos=basket_pos,
            label=f"basket {basket_idx}",
            basket_idx=basket_idx,
            drop_from_z=float(drop_from_z),
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        self._move_to_safe(arm_tag)
        return bool(ok and self._object_sorted(target_idx))

    @stop_on_motion_rejection
    def play_blind_once(self) -> bool:
        """No-intent expert: sort every object in the scripted order without
        reading interrupt or avatar state, and without yielding.

        Eval-mode ticks come from the expert harness
        (``ScriptedRolloutDriver`` at ``action_substeps`` cadence).
        """
        self._boost_arm_pd()
        self._open_basket_lids()
        self._move_to_safe("right")
        any_success = False
        for target_idx in self._scripted_pick_order():
            ok = self._sort_target_once(target_idx)
            any_success = any_success or ok
        for _ in range(200):
            self.step_sim()
        return any_success

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # ------------------------------------------------------------------

    @stop_on_motion_rejection
    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: sort every object not yet in its
        basket, reading live poses each time."""
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            self._boost_arm_pd()
            self._open_basket_lids()
            self._move_to_safe("right")
        ok_all = True
        for target_idx in self._scripted_pick_order():
            if self._object_sorted(target_idx):
                continue
            ok_all = self._sort_target_once(target_idx) and ok_all
        for _ in range(200):
            self.step_sim()
        return ok_all

    @stop_on_motion_rejection
    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_from_object("right")

    def create_interaction_controller(self) -> CategorizeInterruptController:
        return CategorizeInterruptController()

    def _categorize_interrupt_after_reset(self) -> None:
        if self.avatar is not None:
            self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
            self._calibrate_avatar_for_inspect(self._interrupted_idx)
        self.avatar_collided = False
        self.avatar_collision_log = []
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._arm_eval_interrupt()
