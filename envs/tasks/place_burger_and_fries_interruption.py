"""Place burger and fries (Interruption, ``place_burger_and_fries_interruption``): same scene as
``place_burger_fries`` but with a human avatar that picks up one of the
food items mid-task.

Flow per episode:
  1. Robot approaches the interrupted item (above-pose, no commit).
  2. Avatar plays the ``Inspect`` motion with ``attach_obj`` = the
     interrupted item.  Avatar lifts, holds, and sets it back on the table.
  3. Robot retreats so it doesn't collide with the avatar.
  4. While the avatar holds the interrupted item, the robot picks the
     OTHER food items (in food_actors order, skipping the interrupted
     index) and places them on the tray.
  5. Robot waits for ``avatar.spare()``.
  6. Robot reads the interrupted item's LATEST pose (after the avatar
     puts it back down) and picks it.

All food items spawn in the same avatar-reachable band so placement does not
reveal which one the human will inspect.

The human side (interrupted-item choice, calibration, the reset-order hook,
the eval-mode trigger) is owned by the composed
:class:`PlaceBurgerFriesInterruptController`; this class keeps the robot
choreography, the ``take_action`` tick, and thin ``_``-prefixed delegators.
"""

from __future__ import annotations

import numpy as np

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..interaction.place_burger_fries import PlaceBurgerFriesInterruptController
from ..utils import to_numpy
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class PlaceBurgerAndFriesInterruption(
    InspectMotionInterruptFacade,
    PlaceBurgerFries,
):
    INSTRUCTION = "place the hamburger and the french fries on the tray; if the human inspects one item midway through, wait and then continue"

    use_avatar = True

    # Shorten only the human-facing edge by 0.13 m; keep the robot edge fixed.
    TABLE_HALF_SIZE = (0.60, 0.51)
    TABLE_CENTER_XY = (0.0, -0.29)

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.05, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    AVATAR_CLEARANCE = 0.20
    AVATAR_MIN_Y = 0.10
    CALIBRATION_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    PRE_INTERRUPT_SETTLE_STEPS = 80
    # None ⇒ avatar freezes on the inspect motion's last frame (pose + position).
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_FAILED_PICK_SETTLE_STEPS = 40
    POST_RELEASE_SETTLE_STEPS = 120
    # Bounds of the retired random-step trigger; only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 100
    # Progress-trigger interrupt contract (envs/interaction/interrupt_progress.py),
    # fitted to the demonstrated intent hover: the demos hover the TCP at
    # TABLE_TOP_Z + 0.32 m, 0.29-0.30 m (3-D) from the food -- no closer than
    # the reset pose (~0.30 m), so progress there is only 0.01-0.03.  The
    # generic 0.22 m / 0.5 fired only on the later pick descent (human pulled
    # food from a closing gripper).  Radius 0.32 m with no progress demand,
    # gated on the TCP having left its reset pose by 0.10 m, fires 8-12
    # steps before the demo human starts on every probed demo.
    #
    # Hover/dwell gate (the 0.10 m moved-gate alone makes this a
    # "robot moved" trigger, firing 9-13 steps before the demo human): the
    # open-gripper TCP must also hover over the food, xy <= 6 cm and 20-35 cm
    # above its centre, for 3 consecutive policy steps.  On the 3 probed
    # demos (interrupt_check/demo_probe) the hover is 0.5-1 cm in xy at
    # 0.29 m; the gate fires at policy steps 29/29/31 against demo human
    # starts 33/33/37 (4-6 steps early; the moved-gate alone: 22-25).
    INTERRUPT_COMMIT_RADIUS_M = 0.32
    INTERRUPT_TRIGGER_PROGRESS = 0.0
    INTERRUPT_MIN_POLICY_STEPS = 10
    INTERRUPT_MIN_MOVED_M = 0.10
    INTERRUPT_HOVER_DWELL_STEPS = 3
    INTERRUPT_HOVER_XY_M = 0.06
    INTERRUPT_HOVER_DZ_RANGE_M = (0.20, 0.35)
    DRYRUN_RESET_SETTLE_STEPS = 20

    # Lift higher than parent (0.22) so the long horizontal transit (which
    # passes near previously-placed items on the tray when the avatar is in
    # scene and routes the arm through atypical configs) doesn't brush them.
    TRANSPORT_Z_ABOVE_TABLE = 0.32

    # Shared food spawn band near the +Y side of the table.  Every food item
    # uses this band so the robot cannot infer the interrupt target from
    # which item is uniquely close enough for the avatar to inspect.
    FOOD_REACH_X_RANGE = (-0.05, 0.20)
    FOOD_REACH_Y_RANGE = (-0.10, 0.06)
    INTERRUPTED_REACHABLE_Y = FOOD_REACH_Y_RANGE[0]

    def __init__(self, config: dict = None):
        super().__init__(config)
        # Index into self.food_actors of the item the avatar picks up.
        self._interrupted_food_idx: int = 0

    def create_interaction_controller(
        self,
    ) -> PlaceBurgerFriesInterruptController:
        return PlaceBurgerFriesInterruptController()

    def _interrupt_controller(self) -> PlaceBurgerFriesInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, PlaceBurgerFriesInterruptController):
            raise TypeError("task does not use burger/fries interrupt behavior")
        return controller

    def _sample_spawn_xy(self, xy_range, occupied, max_tries: int = None):
        return super()._sample_spawn_xy(
            (self.FOOD_REACH_X_RANGE, self.FOOD_REACH_Y_RANGE),
            occupied,
            max_tries=max_tries,
        )

    def load_actors(self):
        super().load_actors()

        # All spawned LOOSE foods are in the shared avatar-reachable band;
        # choose uniformly so placement does not leak which item will be
        # inspected.  Pre-placed foods (already on the tray)
        # are never the interrupt target — the human inspects an item the
        # robot is about to pick.
        self._interrupted_food_idx = (
            self._interrupt_controller().choose_interrupted_food(self)
        )

    def _interrupted_pair(self):
        return self.food_actors[self._interrupted_food_idx]

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def _closest_food_to_ee(self):
        return self._interrupt_controller().closest_food_to_ee(self)

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _inspect_palm_at_frame(self, frame_idx: int):
        return self._interrupt_controller().inspect_palm_at_frame(self, frame_idx)

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_z = self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE
        retreat_pos = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            retreat_z,
        ])
        self._move_screw(retreat_pos, arm_tag)

    def play_once(self) -> bool:
        from ..grasp import tcp_to_link_pose

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        for _ in range(self.PRE_INTERRUPT_SETTLE_STEPS):
            self.step_sim()

        if self.avatar is None:
            return super().play_once()

        int_actor, int_spec = self._interrupted_pair()
        int_pos = to_numpy(int_actor.entity.get_pos()).ravel()[:3].astype(float)

        n_total = len(self.food_actors)
        slots = self._drop_slots(n_total)
        tray_x, tray_y = self._tray_xy
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE

        # ---- Phase 1: robot shows intent on the interrupted item ----
        self.open_gripper(arm_tag)
        above_pos = np.array([int_pos[0], int_pos[1], transport_z])
        above_link = tcp_to_link_pose(self._top_down_tcp(above_pos), tcp_offset)
        self._move_seeded(above_link.to_pose7(), arm_tag)

        # ---- Phase 2: avatar plays Inspect with attach ----
        self._play_inspect_motion(
            attach_obj=int_actor.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 3: robot retreats so the avatar can grab ----
        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 4: pick the OTHER items (pure physics) ----
        self.plan_success = True

        for i, (actor, spec) in enumerate(self.food_actors):
            if i == self._interrupted_food_idx:
                continue
            if self._food_on_tray(actor):
                continue  # pre-placed on the tray
            dx, dy = slots[i]
            try:
                ok = self._pick_and_drop(
                    actor, spec.asset_id, arm_tag,
                    close_value=spec.close_value,
                    below_center=spec.below_center,
                    drop_xy=(tray_x + dx, tray_y + dy),
                    tcp_yaw_deg=spec.tcp_yaw_deg,
                    transit_steps=spec.transit_steps,
                    transit_sim=spec.transit_sim,
                    align_to_object_yaw=spec.align_to_object_yaw,
                )
            except Exception:
                ok = False
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.POST_FAILED_PICK_SETTLE_STEPS):
                    self.step_sim()

        # ---- Phase 5: wait for avatar to release ----
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 6: pick the released item using its LATEST pose ----
        dx, dy = slots[self._interrupted_food_idx]
        try:
            ok = self._pick_and_drop(
                int_actor, int_spec.asset_id, arm_tag,
                close_value=int_spec.close_value,
                below_center=int_spec.below_center,
                drop_xy=(tray_x + dx, tray_y + dy),
                tcp_yaw_deg=int_spec.tcp_yaw_deg,
                transit_steps=int_spec.transit_steps,
                transit_sim=int_spec.transit_sim,
                align_to_object_yaw=int_spec.align_to_object_yaw,
            )
        except Exception:
            ok = False
        if not ok:
            self.open_gripper(arm_tag)
            for _ in range(self.POST_FAILED_PICK_SETTLE_STEPS):
                self.step_sim()

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: pick every loose food (not on the
        tray) from its live pose."""
        arm_tag = "right"
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            self._boost_finger_pd(arm_tag)
            for _ in range(self.PRE_INTERRUPT_SETTLE_STEPS):
                self.step_sim()
        self.plan_success = True
        slots = self._drop_slots(len(self.food_actors))
        tray_x, tray_y = self._tray_xy
        ok_all = True
        for i, (actor, spec) in enumerate(self.food_actors):
            if self._food_on_tray(actor):
                continue
            dx, dy = slots[i]
            ok = self._pick_and_drop(
                actor, spec.asset_id, arm_tag,
                close_value=spec.close_value,
                below_center=spec.below_center,
                drop_xy=(tray_x + dx, tray_y + dy),
                tcp_yaw_deg=spec.tcp_yaw_deg,
                transit_steps=spec.transit_steps,
                transit_sim=spec.transit_sim,
                align_to_object_yaw=spec.align_to_object_yaw,
            )
            if not ok:
                ok_all = False
                self.open_gripper(arm_tag)
                for _ in range(self.POST_FAILED_PICK_SETTLE_STEPS):
                    self.step_sim()
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_all

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
