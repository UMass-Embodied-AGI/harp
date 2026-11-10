"""Place pair of shoes (Interruption, ``place_pair_of_shoes_interruption``).

Same receiver and pure-physics shoe primitive as ``place_dual_shoes``, but a
human avatar inspects one shoe midway through.  The robot approaches the
interrupted shoe, yields while the avatar picks it up and puts it back, places
the other shoe, then resumes from the released shoe's live pose.

Eval mode follows the progress-trigger interrupt contract
(``envs/interaction/interrupt_progress.py``): the human interrupts when the
robot makes progress toward a shoe, glides in, inspects it, and returns to
idle.  The policy keeps control throughout.
"""

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..utils import Pose, to_numpy
from ..interaction.interrupt_progress import interrupt_human_active
from ..interaction.place_dual_shoes import DualShoesInterruptController
from ..task_bases.place_dual_shoes import DualShoesTaskCore
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class PlacePairOfShoesInterruption(
    InspectMotionInterruptFacade,
    DualShoesTaskCore,
):
    """Robot puts two shoes in a shoebox while the human inspects one shoe.

    The human side (interrupted-shoe draw, calibration, reset-order hook,
    the eval-mode progress trigger -> glide -> inspect -> idle machine) is
    owned by the composed :class:`DualShoesInterruptController`; this class
    keeps the robot choreography, the ``take_action`` tick, the eval-mode
    full-state expert, and thin ``_`` delegators.
    """

    INSTRUCTION = (
        "put the shoes in the shoebox; the human will inspect one shoe "
        "midway through - wait, then continue"
    )
    use_avatar = True

    # Avatar at +Y side of the table, facing -Y toward the table.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    # Inspect's calibrated ready stance for shoes is behind and lower than
    # the generic bread/dump interrupt home pose.  Starting closer to that
    # stance avoids a large visible glide before the avatar begins motion.
    avatar_init_pos = np.array([0.0, 0.82, -0.34])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    # Shared shoe spawn region near the +Y table edge.  Both shoes are
    # sampled here so the robot cannot infer the interrupt target from
    # which shoe is uniquely close enough for the avatar to inspect.
    BOX_X_LO, BOX_X_HI = -0.18, 0.10
    BOX_Y_LO, BOX_Y_HI = -0.34, -0.12
    # Shoe spawn band must be reachable by BOTH the avatar's Inspect clip
    # (shoulder ~y=+0.82) AND the Franka top-down pick (base y=-0.65, reach
    # ~0.62 m).  The old band (-0.08,+0.06) sat at 0.57-0.71 m from the robot
    # base, so most shoes were beyond the top-down reach and every final
    # robot pick pre-RRT-failed (diag sweep 61293938: interrupt s2/s3/s4 all
    # reach fails -> 0/5).  Pull the band -Y into robot reach; the Inspect
    # clip leans far enough to still reach it (validate on re-sweep).  Narrow
    # x so the far corners stay inside the ~0.62 m reach circle.
    SHOE_REACH_X_LO, SHOE_REACH_X_HI = -0.08, 0.08
    SHOE_REACH_Y_LO, SHOE_REACH_Y_HI = -0.18, -0.06
    INT_X_LO, INT_X_HI = SHOE_REACH_X_LO, SHOE_REACH_X_HI
    INT_Y_LO, INT_Y_HI = SHOE_REACH_Y_LO, SHOE_REACH_Y_HI
    OTHER_Y_LO, OTHER_Y_HI = SHOE_REACH_Y_LO, SHOE_REACH_Y_HI
    SHOE_MIN_SEP_M = 0.18
    BOX_SHOE_MIN_SEP_M = 0.32

    # Bounds of the retired random-step trigger.  Only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    # Progress-trigger interrupt contract: shoes are long, so the commit radius is
    # the wide end of the 0.20-0.24 m band.
    INTERRUPT_COMMIT_RADIUS_M = 0.24
    INTERRUPT_TRIGGER_PROGRESS = 0.5
    # Robot-holds-shoe proxy (closed gripper with TCP near the AABB centre).
    HELD_TCP_RADIUS_M = 0.12
    HELD_LIFT_M = 0.04
    LAYOUT_SAMPLE_ATTEMPTS = 80
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.20
    INITIAL_SETTLE_STEPS = 60
    # Scripted collection (play_once) only: None freezes the avatar on the
    # inspect clip's last frame.  Eval mode always returns the human to idle.
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_AVATAR_RELEASE_SETTLE_STEPS = 150
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.35

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)

    def create_interaction_controller(self):
        return DualShoesInterruptController()

    def _interrupt_controller(self) -> DualShoesInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, DualShoesInterruptController):
            raise TypeError("task does not use dual-shoes interrupt behavior")
        return controller

    # ------------------------------------------------------------------
    # Sampling: both shoes use the same avatar-reachable region; the box keeps
    # its separate robot-side region.
    # ------------------------------------------------------------------

    def _sample_layout_positions(self, rng):
        return self._interrupt_controller().sample_layout_positions(self, rng)

    def _layout_clear(self, positions) -> bool:
        return self._interrupt_controller().layout_clear(self, positions)

    # ------------------------------------------------------------------
    # Reset-time avatar calibration — delegated; ordering lives in the
    # controller's after_base_reset_observation hook.
    # ------------------------------------------------------------------

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    # ------------------------------------------------------------------
    # Robot choreography.
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.asarray(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        q_tcp_down = t3d.quaternions.mat2quat(np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        ))
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _approach_interrupted_shoe(self, shoe, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        q_tcp_down = t3d.quaternions.mat2quat(np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        ))
        shoe_pos = np.asarray(self._shoe_pos_quat(shoe)[0])
        above_p = np.array([
            shoe_pos[0],
            shoe_pos[1],
            self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE,
        ])
        above_link = tcp_to_link_pose(Pose(above_p, q_tcp_down), tcp_offset)
        return self.move_and_execute(above_link.to_pose7(), arm_tag) is not None

    def take_action(self, action, action_type: str = "qpos"):
        # Tick the interrupt, then apply the policy action unmodified.
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False
        if self.avatar is None:
            return super().play_once()

        arm_tag = "right"
        interrupted_idx = int(self._interrupted_idx)
        int_shoe = self.shoes[interrupted_idx]
        int_label = chr(ord("A") + interrupted_idx)

        self.open_gripper(arm_tag)
        self._approach_interrupted_shoe(int_shoe, arm_tag)

        self._play_inspect_motion(
            attach_obj=int_shoe.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        any_success = False
        for other_idx, other_shoe in enumerate(self.shoes):
            if other_idx == interrupted_idx:
                continue
            other_label = chr(ord("A") + other_idx)
            try:
                ok = self._pick_and_place_shoe(
                    other_shoe, arm_tag, other_label, slot_idx=0,
                )
            except Exception:
                ok = False
            any_success = any_success or ok

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(self.POST_AVATAR_RELEASE_SETTLE_STEPS):
            self.step_sim()

        try:
            ok = self._pick_and_place_shoe(
                int_shoe, arm_tag, int_label,
                slot_idx=1 if len(self.shoes) > 1 else 0,
            )
        except Exception:
            ok = False
        any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Eval-mode interrupt state machine — delegated to the controller.
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        self._interrupt_controller().init_eval_state(self)

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def eval_skip_policy_action(self) -> bool:
        return self._interrupt_controller().eval_skip_policy_action(self)

    def consume_eval_policy_reset_requested(self) -> bool:
        return self._interrupt_controller().consume_eval_policy_reset_requested(self)

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def _shoe_in_box(self, shoe) -> bool:
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        p = np.asarray(self._shoe_pos_quat(shoe)[0])
        return bool(
            np.linalg.norm(p[:2] - box_xy) <= self.SUCCESS_DIST_XY_M
            and p[2] > self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        )

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: place every shoe that is not yet in
        the box, closest-to-box first, from its live pose."""
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            with self.suppress_recording():
                for _ in range(self.INITIAL_SETTLE_STEPS):
                    self.step_sim()
        if not self.shoes or self.shoebox is None:
            return False
        arm_tag = "right"
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        placed = [i for i, sh in enumerate(self.shoes) if self._shoe_in_box(sh)]
        todo = sorted(
            (i for i in range(len(self.shoes)) if i not in placed),
            key=lambda i: float(np.linalg.norm(
                np.asarray(self._shoe_pos_quat(self.shoes[i])[0])[:2] - box_xy,
            )),
        )
        ok_all = True
        for slot, idx in enumerate(todo, start=len(placed)):
            label = chr(ord("A") + idx)
            try:
                ok = self._pick_and_place_shoe(
                    self.shoes[idx], arm_tag, label,
                    slot_idx=min(slot, max(0, len(self.shoes) - 1)),
                )
            except Exception:
                ok = False
            ok_all = ok_all and bool(ok)
        return ok_all

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")

    # ------------------------------------------------------------------
    # Evaluation metrics.
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        # A shoe still in the human's hand (interrupt sequence in flight) is
        # not placed.
        if bool(self.config.get("eval_mode", False)) and interrupt_human_active(self):
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._shoe_in_box_metrics())
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
