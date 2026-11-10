"""Place bread in basket (Interruption, ``place_bread_in_basket_interruption``): same scene as
``place_bread_in_basket`` but a human avatar inspects one of the two
breads mid-task.

Flow per episode:
  1. Randomly choose whether the interrupt happens on the robot's 1st
     or 2nd bread pick.
  2. Robot completes any earlier bread picks.
  3. Robot approaches above the *interrupted* bread (above-pose, no commit).
  4. Avatar starts the ``Inspect`` motion (categorize_objects_interruption's
     verbatim animation) with ``attach_obj`` = the interrupted bread.
     The bread is attached to the avatar's right hand at frame ATTACH
     and detached at DETACH — the avatar lifts it, holds it, and sets it
     back on the table.
  5. Robot retreats to a safe pose (toward the robot base, away from
     the avatar's lean cone).
  6. Robot handles any later non-interrupted bread picks when present.
  7. Robot waits for ``avatar.spare()``.
  8. Robot reads the released bread's LATEST pose (after the avatar
     puts it back) and places it via the same primitive.

Geometry pipeline (mirrors ``dump_bin_interruption``):
  * Pre-bake: dry-run the Inspect motion at ``avatar_init_pos`` and
    measure the avatar's right-palm position at ATTACH frame.
  * Constraint solve: avatar shift = bread.pos − palm_at_attach.
    After shifting, the avatar palm meets the bread exactly at attach.
  * Both breads spawn in the same avatar-reachable subregion so
    placement does not reveal which one the human will inspect.

Eval: success = both breads in basket (parent's ``check_success``).

Eval mode (``config["eval_mode"]``) follows the progress-trigger interrupt contract
(``envs/interaction/interrupt_progress.py``): the human interrupts when the
robot makes progress toward a loaf, glides in, inspects that loaf, and
returns to idle.  The policy keeps control of the robot throughout.

Robot motion is pure physics (no kinematic teleport/attach).  Avatar
attaches the bread to its hand via ``play_animation(attach_obj=...)``,
the sanctioned avatar mechanism.
"""

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..interaction.place_bread_in_basket import (
    PlaceBreadInBasketInterruptController,
)
from ..utils import Pose, to_numpy
from ..task_bases.place_bread_in_basket import (
    PlaceBreadInBasket, _MIN_SEP_M, _REGION_X, _REGION_Y,
    _SAMPLE_FALLBACK_MARGIN,
)
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


_INTERRUPT_SAMPLE_ATTEMPTS = 80


class PlaceBreadInBasketInterruption(InspectMotionInterruptFacade, PlaceBreadInBasket):
    """Robot picks two pieces of bread; a human avatar inspects one of
    them mid-task — robot must wait, then continue."""

    INSTRUCTION = "put both breads into the basket; if the human inspects one bread midway through, wait and then continue"
    use_avatar = True

    def __init__(self, config: dict = None):
        # Avatar collision tracking is opt-in per BaseTask; force it on
        # so `check_success` can fail any episode where the robot bumps
        # the avatar.  Caller can still pass `track_avatar_collision`
        # explicitly to override.
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)

    def create_interaction_controller(
        self,
    ) -> PlaceBreadInBasketInterruptController:
        return PlaceBreadInBasketInterruptController(
            region_x=_REGION_X,
            region_y=_REGION_Y,
            min_separation=_MIN_SEP_M,
            fallback_margin=_SAMPLE_FALLBACK_MARGIN,
            sample_attempts=_INTERRUPT_SAMPLE_ATTEMPTS,
        )

    def _interrupt_controller(self) -> PlaceBreadInBasketInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, PlaceBreadInBasketInterruptController):
            raise TypeError("task does not use bread-basket interrupt behavior")
        return controller

    # ------------------------------------------------------------------
    # Avatar configuration (mirrors dump_bin_interrupt)
    # ------------------------------------------------------------------

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Maps mesh-+X (avatar forward) to world-(-Y).
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # categorize_interrupt's Inspect motion + frame markers.  10× slow-
    # down reads as a deliberate inspection in video.
    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34       # original frame index
    INSPECT_DETACH_FRAME  = 187      # original frame index
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW    = 10.0     # avatar.frame_ratio
    AVATAR_HAND_ID        = 1        # 1 = right hand
    AVATAR_EXTRA_Z        = 0.04     # palm-above-bread margin so attach
                                     # doesn't sink into the bread
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.18
    # Scripted collection (play_once) only: None freezes the avatar on the
    # inspect clip's last frame.  Eval mode always returns the human to idle.
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120

    # ------------------------------------------------------------------
    # Shared bread spawn region near the +Y table edge.  Both breads are
    # sampled here so the robot cannot infer the interrupt target from
    # which bread is uniquely close enough for the avatar to inspect.
    # ------------------------------------------------------------------
    INT_X_LO, INT_X_HI = -0.10, 0.20
    INT_Y_LO, INT_Y_HI = -0.20, 0.00

    # ------------------------------------------------------------------
    # Eval-mode interrupt (config["eval_mode"]=True), progress-trigger contract.
    # EVAL_TRIGGER_STEP_MIN/MAX are the bounds of the retired random-step
    # trigger; only its RNG draw remains (burned at reset so seeded scenes
    # keep their release-v1 stream).
    # ------------------------------------------------------------------
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30
    EVAL_BASKET_Z_MARGIN = 0.02
    # Fitted to the demonstrated intent hover (TABLE_TOP_Z + INTENT_Z_ABOVE_
    # TABLE, 0.153-0.156 m from the bread, progress 0.48-0.49 < 0.5): 0.40
    # fires on the hover on all probed demos.
    INTERRUPT_COMMIT_RADIUS_M = 0.22
    INTERRUPT_TRIGGER_PROGRESS = 0.40
    INTERRUPT_MIN_POLICY_STEPS = 20

    # ------------------------------------------------------------------
    # Sampling — the controller chooses the interrupted pick ordinal and
    # draws both breads from the same avatar-reachable band (legacy order).
    # Bread ``self.breads[i]`` corresponds to ``positions[i + 1]``.
    # ------------------------------------------------------------------

    def _scripted_pick_order_from_positions(self, basket_xy, bread_xys):
        return PlaceBreadInBasketInterruptController.scripted_pick_order_from_positions(
            basket_xy, bread_xys,
        )

    def _choose_interrupt_pick_ordinal(self, rng) -> int:
        return self._interrupt_controller().choose_interrupt_pick_ordinal(self, rng)

    def _sample_positions(self, rng):
        return self._interrupt_controller().plan_positions(self, rng)

    # ------------------------------------------------------------------
    # Avatar calibration — owned by the controller.  Reset ordering lives in
    # the controller's after_base_reset_observation hook: inspect motion is
    # resolved after the initial observation, then eval-mode state is armed
    # or the avatar is pre-shifted for the scripted expert.
    # ------------------------------------------------------------------

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    # ------------------------------------------------------------------
    # Robot retreat (clear the avatar's lean cone so the avatar can
    # reach in).
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        # Top-down TCP rotation, same convention as parent.
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _scripted_pick_order(self):
        return self._interrupt_controller().scripted_pick_order(self)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def take_action(self, action, action_type: str = "qpos"):
        """Eval mode ticks the interrupt, then applies the policy action
        unmodified through the standard ``BaseTask.take_action``."""
        if (self.avatar is not None and
                bool(self.config.get("eval_mode", False))):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.breads or self.basket is None:
            return False
        self._refresh_basket_geom()
        arm_tag = "right"

        if self.avatar is None:
            return super().play_once()

        pick_order = self._scripted_pick_order()
        interrupted_pick_ordinal = int(getattr(
            self, "_interrupted_pick_ordinal", 0,
        )) % len(pick_order)
        interrupted_idx = int(pick_order[interrupted_pick_ordinal])
        self._interrupted_idx = interrupted_idx
        int_bread = self.breads[interrupted_idx]
        int_label = chr(ord("A") + interrupted_idx)

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        intent_z = self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE

        # ---- Phase 1: complete picks before the interrupted ordinal ---
        any_success = False
        for idx in pick_order[:interrupted_pick_ordinal]:
            label = chr(ord("A") + idx)
            try:
                ok = self._pick_and_place_bread(
                    self.breads[idx], arm_tag, label,
                )
            except Exception:
                ok = False
            if not ok:
                return False
            any_success = True

        # ---- Phase 2: robot shows intent on the interrupted bread ----
        self.open_gripper(arm_tag)
        int_pos = to_numpy(int_bread.entity.get_pos()).ravel()[:3]
        above_p = np.array([int_pos[0], int_pos[1], intent_z])
        above_link = tcp_to_link_pose(
            Pose(above_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # ---- Phase 3: avatar plays Inspect with attach ----
        self._play_inspect_motion(
            attach_obj=int_bread.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 4: robot retreats so the avatar can grab the bread --
        self._retreat_safe(arm_tag)
        # Step until the avatar has reached the attach frame so the
        # bread is visibly held before any later robot pick starts.
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 5: complete later non-interrupted picks -----------
        for idx in pick_order[interrupted_pick_ordinal + 1:]:
            label = chr(ord("A") + idx)
            try:
                ok = self._pick_and_place_bread(
                    self.breads[idx], arm_tag, label,
                )
            except Exception:
                ok = False
            if not ok:
                return False
            any_success = True

        # ---- Phase 6: wait for avatar to release the interrupted bread --
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        # Give the released bread a moment to settle on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 7: pick + place the released bread (latest pose) ----
        try:
            ok = self._pick_and_place_bread(int_bread, arm_tag, int_label)
        except Exception:
            ok = False
        any_success = any_success or ok

        return any_success

    # ------------------------------------------------------------------
    # Eval-mode interrupt — owned by the controller.
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        self._interrupt_controller().init_eval_state(self)

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def _find_closest_table_bread(self):
        best_idx, _ = self._closest_table_bread_candidate()
        return best_idx

    def _closest_table_bread_candidate(self):
        return self._interrupt_controller().closest_table_bread_candidate(self)

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def _bread_in_basket(self, bread) -> bool:
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        p = to_numpy(bread.entity.get_pos()).ravel()[:3]
        return bool(
            np.linalg.norm(p[:2] - basket_xy) <= self.SUCCESS_DIST_XY_M
            and p[2] > self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
        )

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: place every loaf not yet in the
        basket, closest-to-basket first, from its live pose."""
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            with self.suppress_recording():
                for _ in range(self.INITIAL_SETTLE_STEPS):
                    self.step_sim()
        if not self.breads or self.basket is None:
            return False
        self._refresh_basket_geom()
        arm_tag = "right"
        ok_all = True
        for idx in self._scripted_pick_order():
            bread = self.breads[idx]
            if self._bread_in_basket(bread):
                continue
            try:
                ok = self._pick_and_place_bread(bread, arm_tag, chr(ord("A") + idx))
            except Exception:
                ok = False
            ok_all = ok_all and bool(ok)
        return ok_all

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")

    # ------------------------------------------------------------------
    # Success metric — both breads in basket AND no avatar collision
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Inherits parent's "both breads inside basket" + adds a hard
        fail on avatar collision.  Avatar collision tracking is forced
        on in __init__ so this metric is meaningful by default."""
        if self.avatar_collided:
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        out = super().evaluate()
        out.update(self.interaction_controller.interaction_metrics(self))
        return out
