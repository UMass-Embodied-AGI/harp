"""Feed documents (Cueing, ``feed_documents``): the avatar stamps two documents
while the arm feeds them.

Robot: physics-based push only (no grasp/attach).  Flow:
  avatar stamps doc 1 -> arm clears it -> arm feeds doc 2 -> avatar stamps doc 2.

Avatar: plays `stamp_crop` once per doc with the seal attached to the
right palm.  Ink pad sits at the dip XY measured by a dry-run calibration
in `reset()`.
"""

import copy
import os
import time
from pathlib import Path
import numpy as np
import transforms3d as t3d

from ..avatar.utils import ActionStatus, AvatarState
from ..base_task import BaseTask, TABLE_HEIGHT
from ..grasp import tcp_to_link_pose
from ..interaction import IntentEvalModeController
from ..interaction.cue_wait import await_cue
from ..robot.base import _get_dof_idx
from ..utils import Pose, create_primitive, load_mesh, load_object, to_numpy


_TOP_DOWN_R = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])
_TOP_DOWN_Q = t3d.quaternions.mat2quat(_TOP_DOWN_R)
_TOP_DOWN_R_Y_PUSH = t3d.euler.euler2mat(0.0, 0.0, np.pi / 2.0, axes="sxyz") @ _TOP_DOWN_R
_TOP_DOWN_Q_Y_PUSH = t3d.quaternions.mat2quat(_TOP_DOWN_R_Y_PUSH)
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])

# Doc shape (10×10×2.5 cm) and feeder slots, all proven reachable by the
# right arm's screw planner.
_DOC_HALF = (0.05, 0.05, 0.0125)
_DOC_FEEDER_XY = [(0.12, -0.15), (0.22, -0.15), (0.32, -0.15)]

# Per-seed jitter limits.  X column shift is shared across docs (preserves
# push geometry); Y is per-doc.
# Doc size ±12% keeps half_z above stamp-mark thickness and half_x/y above
# stamp radius (0.0266 m).  Avatar pos+yaw are small enough that the
# calibrated stamp_xy stays in right-arm reach.
_FEEDER_X_JITTER = 0.02
_FEEDER_Y_JITTER = 0.015
_DOC_COUNT = 2
_DOC_SIZE_SCALE_RANGE = (0.88, 1.12)
_AVATAR_POS_JITTER_XY = 0.03
_AVATAR_YAW_JITTER_DEG = 5.0

_DOC_COLOR_POOL = [
    (0.95, 0.92, 0.85),   # cream
    (0.96, 0.96, 0.92),   # off-white
    (0.88, 0.86, 0.80),   # warm beige
    (0.92, 0.90, 0.88),   # linen
    (0.94, 0.88, 0.78),   # manila
]

_STAMP_MOTION = "stamp_crop"     # one dip-stamp cycle, replayed per doc
_STAMP_HAND_ID = 1
_AVATAR_FRAME_RATIO = 12.5       # ~5× slower than default so dips read clearly

# 100_seal mesh: body length 9.6 cm along local +Y, origin at the −Y end.
# Setting pos to palm − body_length·ẑ puts the +Y handle at the palm.
_STAMP_BODY_LEN = 0.0964

# Stamp base disc: x/z half-extent 0.5313 · scale(0.05) = 0.0266 m.
_STAMP_MARK_RADIUS = 0.0266
_STAMP_MARK_HALF_H = 0.00025
_PUSH_GRIPPER_VAL = 0.35
_FIRST_DOC_PUSH_EDGE_FRACTION = 0.8

_INKPAD_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", "inkpad", "inkpad.obj",
)


def doc_covers_point(doc_pos, doc_quat, point_xy, doc_half, margin=0.0) -> bool:
    """True if ``point_xy`` lies within the doc's footprint grown by ``margin``.

    Same test ``_attach_stamp_mark`` uses for an on-doc stamp, evaluated in
    the doc frame so a yawed doc is handled.
    """
    doc_pos = np.asarray(doc_pos, dtype=np.float64).ravel()[:3]
    R = t3d.quaternions.quat2mat(np.asarray(doc_quat, dtype=np.float64).ravel()[:4])
    world = np.array([float(point_xy[0]), float(point_xy[1]), doc_pos[2]])
    local = R.T @ (world - doc_pos)
    return bool(abs(local[0]) <= float(doc_half[0]) + margin
                and abs(local[1]) <= float(doc_half[1]) + margin)


def second_stamp_ready(doc1_in_zone: bool, doc0_in_zone: bool,
                       robot_clear_steps: int, required_clear_steps: int,
                       avatar_spare: bool) -> bool:
    """Eval trigger for the human's doc-1 stamp (see ``_eval_stamp_step``)."""
    return bool(doc1_in_zone and not doc0_in_zone and avatar_spare
                and int(robot_clear_steps) >= int(required_clear_steps))


class FeedDocuments(BaseTask):
    """Arm pushes docs into a stamp zone; avatar dips & stamps each in turn."""

    INSTRUCTION = "push each document into the stamp zone so the human can stamp it"
    use_avatar = True
    # Avatar shifted +0.30 X / +0.11 Z so the motion's stamp/dip positions
    # (avatar-local x ≈ −0.17 for right hand) map into right-arm reach.
    avatar_init_pos = np.array([0.30, 0.89, -0.18 + 0.11])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Capture 1-in-6 keeps the 30 fps video at sim dt=0.002 (12 ms/frame).
    VIDEO_STRIDE = 6
    recording_camera_pos = [1.1, 0.0, 2.05]
    recording_camera_lookat = [0.0, 0.3, 0.85]

    # Wider table for the Franka side while preserving the avatar-side
    # workspace used by the stamp and ink-pad calibration.
    TABLE_HALF_SIZE = (0.60, 0.65)
    TABLE_CENTER_XY = (0.0, -0.25)
    TABLE_THICKNESS_VALUE = 0.05
    TABLE_LEG_RADIUS = 0.025

    # Avatar arm capsule pushed to the planner so screw plans route around it.
    _ARM_CAPSULE_RADIUS = 0.06
    _ARM_CAPSULE_N_ALONG = 14
    _ARM_CAPSULE_N_AROUND = 6

    # Filled by `_calibrate_motion_key_frames` and `load_actors`.
    _stamp_xy = np.array([0.10, 0.10])
    _ink_xy = np.array([-0.10, 0.30])
    _doc_half = _DOC_HALF

    # Avatar control flags.
    _stamp_ready = False     # set True after `_attach_stamp_to_palm`
    _freeze_avatar = False   # if True, avatar.update() instead of avatar.step()

    # Eval-mode human (``stamp_eval_human``):
    #   "random_start" (default): the human stamps doc 0 at a per-seed random
    #       policy step (25 sim steps) from [EVAL_TRIGGER_STEP_MIN, MAX],
    #       whatever the robot is doing, without freezing the robot; the
    #       policy must see the stamp finish before it clears doc 0.  Doc 1 is
    #       stamped once it covers the stamp point and the robot TCP has been
    #       EVAL_ROBOT_CLEAR_RADIUS (xy) clear for EVAL_ROBOT_CLEAR_STEPS
    #       policy steps.  Between stamps the hand rests at the retreat pose.
    #   "robot_triggered": doc 0 is stamped once the robot parks at standby,
    #       with blocking cycles that freeze the robot.  It gives the policy
    #       no cue to read: a robot that never watches the human succeeds
    #       every time.
    # Window: in the training demos the robot reaches standby ~26 policy
    # steps after the first recorded frame and the human then stamps doc 0
    # (one cycle ~77 steps).  Starting the window at 26 keeps the robot's
    # start state and early frames identical to the demos; the 100-step
    # spread makes "always wait for the latest stamp" cost ~100 steps
    # (demo median 467, p95 490, max_steps 735).
    EVAL_HUMAN = "random_start"
    EVAL_TRIGGER_STEP_MIN = 26
    EVAL_TRIGGER_STEP_MAX = 126
    _EVAL_RNG_SALT = 0x57A4D0C
    _EVAL_HUMAN_MAX_CYCLES = _DOC_COUNT
    # Blind / full-state eval experts: bounded tail after the robot's last
    # push, long enough for the clearance wait, the ease and one stamp cycle.
    _BLIND_SETTLE_MAX_SIM_STEPS = 4000
    # Blind / full-state eval experts: corrective pushes of the fed document.
    _FEED_REPUSH_MAX = 2

    _eval_cycle = None       # in-progress non-blocking stamp cycle (dict)
    _eval_sim_step = None    # sim steps since eval-human episode start
    _blind_active = False    # play_blind_once in progress

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

    def create_interaction_controller(self) -> IntentEvalModeController:
        # Hooks are no-ops unless ``eval_avatar_pre_policy`` is False, so the
        # warmup-driven VLA eval path is unchanged.
        return IntentEvalModeController()

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Y span [-0.90, +0.40]: enough support for the default Franka base
        # at y=-0.65 without shortening the avatar-side stamping area.
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = self.TABLE_THICKNESS_VALUE
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self.table = create_primitive(
            self.scene,
            "box",
            Pose(p=[self.TABLE_CENTER_XY[0], self.TABLE_CENTER_XY[1], table_height]),
            size={"half_size": (
                self.TABLE_HALF_SIZE[0],
                self.TABLE_HALF_SIZE[1],
                self.TABLE_THICKNESS / 2,
            )},
            color=(0.80, 0.75, 0.65),
            is_static=True,
        )
        self._remember_table_bounds(
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        leg_x = self.TABLE_HALF_SIZE[0] - 0.05
        leg_y = self.TABLE_HALF_SIZE[1] - 0.05
        cx, cy = self.TABLE_CENTER_XY
        for x, y in (
            (cx - leg_x, cy - leg_y),
            (cx + leg_x, cy - leg_y),
            (cx - leg_x, cy + leg_y),
            (cx + leg_x, cy + leg_y),
        ):
            create_primitive(
                self.scene,
                "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": self.TABLE_LEG_RADIUS, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )

    # ------------------------------------------------------------------
    # Sim step
    # ------------------------------------------------------------------
    def step_sim(self):
        self.scene.step()
        self.SIM_STEP_IDX += 1
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            if self._freeze_avatar:
                self.avatar.robot.update()
            else:
                self.avatar.step()
        self._rigid_vertical_stamp()
        self._update_stamp_marks()
        if self._eval_sim_step is not None:
            self._eval_sim_step += 1
        if self._eval_cycle is not None:
            self._eval_stamp_tick()
        self._notify_interaction_sim_step()
        self._tick_avatar_checks()
        if self.vla_recorder is not None:
            self.vla_recorder.tick(self)
        self.capture_frame()
        if self._record_stride is not None:
            if self._record_tick % self._record_stride == 0:
                self.record_frame()
            self._record_tick += 1

    def _update_stamp_marks(self):
        """Sync each attached red ink-mark pose to its host doc."""
        if not getattr(self, "stamp_marks", None):
            return
        for i, mark in enumerate(self.stamp_marks):
            doc = self._stamp_mark_attached[i]
            offset = self._stamp_mark_local_offset[i]
            if doc is None or offset is None:
                continue
            doc_pos = to_numpy(doc.get_pos()).ravel()
            doc_quat = to_numpy(doc.get_quat()).ravel()
            R = t3d.quaternions.quat2mat(doc_quat)
            mark.set_pos((doc_pos + R @ offset).astype(np.float64))
            mark.set_quat(doc_quat.astype(np.float64))

    def _attach_stamp_mark(self, doc_idx: int, doc, mark_xy,
                           doc_pos_override=None, doc_quat_override=None):
        """Snap mark `doc_idx` at the stamp's world XY on the doc's top face,
        store the local offset for tracking, and record whether the stamp
        landed inside the doc footprint (consumed by `check_success`).

        `doc_pos_override` / `doc_quat_override`: the doc's pose snapshotted at
        the contact instant.  The rigid seal drags the doc as the palm retracts,
        so a live `get_pos()` read here sees the doc already pulled away from
        where the stamp actually pressed; the offset must be computed against
        the doc pose at the same instant the seal XY was sampled.
        """
        if doc_pos_override is not None:
            doc_pos = np.asarray(doc_pos_override, dtype=np.float64).ravel()[:3]
        else:
            doc_pos = to_numpy(doc.get_pos()).ravel()
        if doc_quat_override is not None:
            doc_quat = np.asarray(doc_quat_override, dtype=np.float64).ravel()[:4]
        else:
            doc_quat = to_numpy(doc.get_quat()).ravel()
        top_z = doc_pos[2] + self._doc_half[2] + _STAMP_MARK_HALF_H + 0.0005
        world_pos = np.array([float(mark_xy[0]), float(mark_xy[1]), top_z])
        R = t3d.quaternions.quat2mat(doc_quat)
        local_offset = R.T @ (world_pos - doc_pos)
        self._stamp_mark_attached[doc_idx] = doc
        self._stamp_mark_local_offset[doc_idx] = local_offset
        self.stamp_marks[doc_idx].set_pos(world_pos.astype(np.float64))
        self.stamp_marks[doc_idx].set_quat(doc_quat.astype(np.float64))
        # On-doc: count a mark when the stamp disc visibly overlaps the
        # document.  The avatar stamps near the paper edge, so requiring the
        # disc centre to be strictly inside rejects valid-looking stamps.
        margin = 2.0 * _STAMP_MARK_RADIUS
        on_doc = bool(
            abs(local_offset[0]) <= self._doc_half[0] + margin
            and abs(local_offset[1]) <= self._doc_half[1] + margin
        )
        self._stamp_within_doc[doc_idx] = on_doc

    def _rigid_vertical_stamp(self):
        """Hold the stamp base pointing straight down at palm_z − body_len.

        The Mixamo motion's hand pose isn't perfectly palm-down at the
        stamp frame, so a rigid attach would leave the stamp tilted.  We
        override pos+quat every sim step instead.
        """
        if not self._stamp_ready or self.avatar is None:
            return
        palm = np.asarray(
            self.avatar.robot.get_palm_center(_STAMP_HAND_ID),
            dtype=np.float64,
        ).ravel()
        origin_pos = palm - np.array([0.0, 0.0, _STAMP_BODY_LEN], dtype=np.float64)
        # Clamp the stamp face so it hovers just above the document top instead
        # of penetrating it.  The seal is a kinematic prop driven by set_pos,
        # so when its face dips below the doc top the contact solver shoves the
        # (lighter, movable) document — under Genesis 1.0.0 the descending seal
        # bulldozed the doc ~0.23 m forward before it could stamp, so the stamp
        # never landed on the paper.  A small hover removes the contact while
        # still reading as a stamp; the red mark is placed in code, not by
        # physical contact.
        min_face_z = self.TABLE_TOP_Z + float(self._doc_half[2]) + 0.002
        if origin_pos[2] < min_face_z:
            origin_pos[2] = min_face_z
        # Stamp local +Y → world +Z (body points up from base to handle).
        R_stamp = np.array([
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ])
        q = t3d.quaternions.mat2quat(R_stamp)
        self.stamp_actor.entity.set_pos(origin_pos)
        self.stamp_actor.entity.set_quat(np.asarray(q, dtype=np.float64))

    # ------------------------------------------------------------------
    # Per-seed scene
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        # Per-episode eval state.  The warmup latch must not persist on the
        # instance, or episodes 2..N of one eval process skip the doc-0
        # stamp entirely.
        self._eval_avatar_warmup_done = False
        self._eval_doc0_stamp_fired = False
        self._eval_doc0_stamp_step = None
        self._eval_doc0_standby_steps = 0
        self._eval_doc1_stamp_fired = False
        self._eval_doc1_stamp_step = None
        self._eval_robot_clear_steps = 0
        self._eval_stamp_policy_steps = 0
        self._full_eval_cue_waits = None
        obs = super().reset(seed=seed)
        self._prepare_episode_start()
        return obs

    def _prepare_episode_start(self):
        recorder = self.vla_recorder
        record_stride = self._record_stride
        try:
            self.vla_recorder = None
            self._record_stride = None
            self._calibrate_motion_key_frames()
            self._place_documents_for_stamp_sequence()
            for _ in range(20):
                self.step_sim()
        finally:
            self.vla_recorder = recorder
            self._record_stride = record_stride
            self._clear_episode_buffers()
        if self._eval_human_enabled():
            # Seal in palm, avatar idle at stamp frame 0 until the trigger.
            self._stamp_ready = True
            self._freeze_avatar = True
            self._rigid_vertical_stamp()
            self._eval_sim_step = 0

    def eval_pre_policy_warmup(self):
        """Run the first visible avatar stamp before policy control starts."""
        if self.avatar is None or not self.document_actors:
            return None
        if getattr(self, "_eval_avatar_warmup_done", False):
            return None
        self._eval_avatar_warmup_done = True
        if self._eval_human_mode() == "random_start":
            # No pre-policy behaviour: the seal was seated in
            # _prepare_episode_start and the human acts during the policy.
            return self.get_obs()
        self._attach_stamp_to_palm()
        self._freeze_avatar = True
        if str(self.config.get("stamp_eval_doc0_trigger",
                               self.EVAL_DOC0_TRIGGER)) != "warmup":
            # "policy" (default): the demonstrations start with the robot at
            # home and doc 0 UNstamped; the robot first moves to its standby
            # pose and only then does the human stamp doc 0.  Stamping during
            # the warmup started every eval episode from a frame (robot home,
            # doc 0 already stamped) that never occurs in the demos.  The
            # doc-0 cycle now fires from ``_eval_stamp_step``.
            self._eval_doc0_stamp_fired = False
            return self.get_obs()
        self._play_one_stamp_cycle(doc_idx=0, doc=self.document_actors[0])
        self._eval_doc0_stamp_fired = True
        self._eval_doc0_stamp_step = 0
        self._freeze_avatar = True
        # Do not leave the hand hovering over the stamp zone the robot has to
        # push doc 1 into: ease back to the rest pose while the policy acts.
        self._eval_return_avatar_to_rest()
        return self.get_obs()

    # Doc-0 stamp on the policy path.  ``play_once`` moves the robot to
    # ``_standby_xyz`` (behind the stamp zone, 0.22 m above the table) and
    # the human stamps doc 0 once it is there.  The eval human stamps when
    # the policy's TCP has been within EVAL_DOC0_STANDBY_RADIUS of that
    # standby point for EVAL_DOC0_STANDBY_STEPS consecutive policy steps,
    # or -- fallback for a policy that parks elsewhere -- after
    # EVAL_DOC0_FALLBACK_STEPS policy steps while the TCP is clear of the
    # stamp arc.
    EVAL_DOC0_TRIGGER = "policy"
    EVAL_DOC0_STANDBY_RADIUS = 0.08
    EVAL_DOC0_STANDBY_STEPS = 3
    EVAL_DOC0_FALLBACK_STEPS = 120

    def _standby_xyz(self) -> np.ndarray:
        return np.array([float(self._stamp_xy[0]),
                         float(self._stamp_xy[1]) - 0.20,
                         float(self.TABLE_TOP_Z) + 0.22])

    def _eval_maybe_stamp_doc0(self) -> bool:
        """Fire the doc-0 stamp once the robot is at standby.  Returns True
        while doc 0 is still pending (the doc-1 logic waits for it)."""
        if getattr(self, "_eval_doc0_stamp_fired", True):
            return False
        arm = self.robot.get_arm("right")
        tcp = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        radius = float(self.config.get(
            "stamp_eval_doc0_standby_radius", self.EVAL_DOC0_STANDBY_RADIUS))
        if float(np.linalg.norm(tcp - self._standby_xyz())) <= radius:
            self._eval_doc0_standby_steps = int(
                getattr(self, "_eval_doc0_standby_steps", 0)) + 1
        else:
            self._eval_doc0_standby_steps = 0
        at_standby = self._eval_doc0_standby_steps >= int(self.config.get(
            "stamp_eval_doc0_standby_steps", self.EVAL_DOC0_STANDBY_STEPS))
        clear_r = float(self.config.get(
            "stamp_eval_robot_clear_radius", self.EVAL_ROBOT_CLEAR_RADIUS))
        clear = float(np.linalg.norm(tcp[:2] - np.asarray(self._stamp_xy)[:2])) >= clear_r
        fallback = (int(self._eval_stamp_policy_steps) >= int(self.config.get(
            "stamp_eval_doc0_fallback_steps", self.EVAL_DOC0_FALLBACK_STEPS))
            and clear)
        if not (at_standby or fallback) or not self.avatar.spare():
            return True
        self._eval_doc0_stamp_fired = True
        self._eval_doc0_stamp_step = int(self._eval_stamp_policy_steps)
        self._freeze_avatar = True
        self._play_one_stamp_cycle(doc_idx=0, doc=self.document_actors[0])
        self._freeze_avatar = True
        self._eval_return_avatar_to_rest()
        return False

    # ------------------------------------------------------------------
    # Eval-mode human reaction (policy path).  ``play_once`` keeps its own
    # blocking choreography and never calls these.
    # ------------------------------------------------------------------
    # Robot TCP must stay this far (xy) from the stamp point for
    # EVAL_ROBOT_CLEAR_STEPS consecutive policy steps before the human stamps
    # doc 1.  The scripted standby is 0.20 m away.  0.16: at
    # 0.12-0.15 the stamp arc can clip a policy's wrist (link7 vs
    # r_middle_prox, 1.8 mm, during the doc-1 stamp), which vetoes an
    # otherwise complete episode.
    EVAL_ROBOT_CLEAR_RADIUS = 0.16
    EVAL_ROBOT_CLEAR_STEPS = 10
    # When doc 1 counts as delivered.
    #   "demo" (default): doc-1 centre within EVAL_DOC_TRIGGER_RADIUS of the
    #       stamp point, then the stamp cycle pins it at the stamp point —
    #       exactly the scripted expert / demonstration contract.  The
    #       expert's midline push stops ~0.11-0.12 m short in y and
    #       ``play_once`` relies on that pin, so a policy imitating the demos
    #       ends in the same place.
    #   "strict": the stamp point must already lie on doc 1 (footprint grown
    #       by EVAL_DOC_IN_ZONE_MARGIN) and the doc is pinned where it is.
    EVAL_DOC_CONTRACT = "demo"
    EVAL_DOC_TRIGGER_RADIUS = 0.15
    EVAL_DOC_IN_ZONE_MARGIN = _STAMP_MARK_RADIUS
    EVAL_REST_TRANSITION_FRAMES = 60
    # Where the eval human rests between stamps (``stamp_eval_rest_pose``):
    #   "retreat" (default): EVAL_RETREAT_FRACTION of the way from the stamp
    #       clip's frame 0 to idle.  Frame 0 holds the stamp at the edge of the
    #       zone; the robot's doc-0 clear pass came within ~1 cm of those
    #       fingers and VLAs touched them (4/40 train37 stamp episodes).
    #       At 0.5 the palm sits ~(-0.15, 0.34, 0.96) m, stamp held up over
    #       the near table edge, ~0.28 m from the expert's clear/push
    #       corridor (0.01 / 0.11 m at frame 0).
    #   "frame0" (previous default), "idle", "hold" (stay put).
    # Before the doc-1 stamp the hand eases back to frame 0 (blocking, like
    # the stamp cycle itself) so the clip never starts with a jump.
    EVAL_REST_POSE = "retreat"
    EVAL_RETREAT_FRACTION = 0.5

    def _eval_rest_mode(self) -> str:
        return str(self.config.get("stamp_eval_rest_pose", self.EVAL_REST_POSE))

    def _eval_ease_to_stamp_start(self) -> None:
        """Blend from the rest pose back to frame 0 of the stamp clip,
        blocking (the robot holds its last PD target), before a stamp."""
        rest = getattr(self, "_stamp_rest_pose", None)
        if self.avatar is None or rest is None or self._eval_rest_mode() not in ("retreat", "idle"):
            return
        frames = int(self.config.get(
            "stamp_eval_rest_frames", self.EVAL_REST_TRANSITION_FRAMES))
        self._freeze_avatar = False
        self.avatar.play_transition(
            target_pose=rest["pose"],
            target_node=rest["node_trans"],
            target_mat=rest["global_mat"],
            target_mat_inv=rest["global_mat_inv"],
            frames=frames,
        )
        guard = 0
        while not self.avatar.spare() and guard < frames + 10:
            self.step_sim()
            guard += 1
        self._freeze_avatar = True

    def _eval_return_avatar_to_rest(self):
        """Blend the avatar to its between-stamps rest pose, non-blocking
        (see EVAL_REST_POSE)."""
        if self.avatar is None:
            return
        mode = self._eval_rest_mode()
        frames = int(self.config.get(
            "stamp_eval_rest_frames", self.EVAL_REST_TRANSITION_FRAMES))
        rest = getattr(self, "_stamp_rest_pose", None)
        if mode == "retreat":
            rest = getattr(self, "_stamp_retreat_pose", None) or rest
        if mode == "hold":
            return
        self._freeze_avatar = False
        if mode == "idle" or rest is None:
            self.avatar.play_return_to_idle(frames=frames)
        else:
            self.avatar.play_transition(
                target_pose=rest["pose"],
                target_node=rest["node_trans"],
                target_mat=rest["global_mat"],
                target_mat_inv=rest["global_mat_inv"],
                frames=frames,
            )

    def _robot_tcp_xy(self) -> np.ndarray:
        arm = self.robot.get_arm("right")
        return np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:2]

    def _doc_in_stamp_zone(self, doc, margin: float) -> bool:
        return doc_covers_point(
            to_numpy(doc.get_pos()).ravel()[:3],
            to_numpy(doc.get_quat()).ravel()[:4],
            self._stamp_xy, self._doc_half, margin,
        )

    def _doc1_delivered(self, doc1, contract: str) -> bool:
        if contract == "strict":
            margin = float(self.config.get(
                "stamp_eval_doc_margin", self.EVAL_DOC_IN_ZONE_MARGIN))
            return self._doc_in_stamp_zone(doc1, margin)
        radius = float(self.config.get(
            "stamp_eval_doc_trigger_radius", self.EVAL_DOC_TRIGGER_RADIUS))
        xy = to_numpy(doc1.get_pos()).ravel()[:2]
        return float(np.linalg.norm(xy - np.asarray(self._stamp_xy)[:2])) <= radius

    def _doc0_blocks_zone(self, doc0, contract: str) -> bool:
        """Doc 0 would take the stamp, or (demo contract) overlap doc 1
        once doc 1 is pinned at the stamp point."""
        margin = float(self.config.get(
            "stamp_eval_doc_margin", self.EVAL_DOC_IN_ZONE_MARGIN))
        if self._doc_in_stamp_zone(doc0, margin):
            return True
        if contract == "strict":
            return False
        d = np.abs(to_numpy(doc0.get_pos()).ravel()[:2] - np.asarray(self._stamp_xy)[:2])
        return bool(d[0] < 2.0 * self._doc_half[0] and d[1] < 2.0 * self._doc_half[1])

    def _eval_stamp_step(self):
        """Per policy step: stamp doc 1 once the robot has delivered it."""
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not getattr(self, "_eval_avatar_warmup_done", False)
            or getattr(self, "_eval_doc1_stamp_fired", False)
            or len(getattr(self, "document_actors", []) or []) != _DOC_COUNT
        ):
            return
        self._eval_stamp_policy_steps = int(
            getattr(self, "_eval_stamp_policy_steps", 0)) + 1
        if self._eval_maybe_stamp_doc0():
            return
        clear_r = float(self.config.get(
            "stamp_eval_robot_clear_radius", self.EVAL_ROBOT_CLEAR_RADIUS))
        tcp_xy = self._robot_tcp_xy()
        if float(np.linalg.norm(tcp_xy - np.asarray(self._stamp_xy)[:2])) >= clear_r:
            self._eval_robot_clear_steps = int(
                getattr(self, "_eval_robot_clear_steps", 0)) + 1
        else:
            self._eval_robot_clear_steps = 0
        contract = str(self.config.get(
            "stamp_eval_doc_contract", self.EVAL_DOC_CONTRACT))
        doc0, doc1 = self.document_actors
        required = int(self.config.get(
            "stamp_eval_robot_clear_steps", self.EVAL_ROBOT_CLEAR_STEPS))
        if not second_stamp_ready(
            self._doc1_delivered(doc1, contract),
            self._doc0_blocks_zone(doc0, contract),
            self._eval_robot_clear_steps,
            required,
            self.avatar.spare(),
        ):
            return
        self._eval_doc1_stamp_fired = True
        self._eval_doc1_stamp_step = int(self._eval_stamp_policy_steps)
        # Blocking, like the doc-0 warmup cycle; the robot holds its last
        # PD target meanwhile.
        self._eval_ease_to_stamp_start()
        self._play_one_stamp_cycle(
            doc_idx=1, doc=doc1, pin_at_stamp_xy=(contract != "strict"))
        self._eval_return_avatar_to_rest()

    def take_action(self, action, action_type: str = "qpos"):
        if self._eval_human_mode() == "robot_triggered":
            self._eval_stamp_step()
        return super().take_action(action, action_type=action_type)

    def _place_documents_for_stamp_sequence(self):
        if not self.document_actors:
            return

        half_z = self._doc_half[2]
        stamp_doc_pos = np.array(
            [float(self._stamp_xy[0]), float(self._stamp_xy[1]),
             self.TABLE_TOP_Z + half_z + 0.002],
            dtype=np.float64,
        )
        self.document_actors[0].set_pos(stamp_doc_pos)

        doc1_xy = None
        if len(self.document_actors) > 1:
            xmin = self.TABLE_CENTER_XY[0] - self.TABLE_HALF_SIZE[0] + 0.12
            xmax = self.TABLE_CENTER_XY[0] + self.TABLE_HALF_SIZE[0] - 0.12
            ymin = self.TABLE_CENTER_XY[1] - self.TABLE_HALF_SIZE[1] + 0.12
            ymax = self.TABLE_CENTER_XY[1] + self.TABLE_HALF_SIZE[1] - 0.12
            doc1_xy = np.array([
                np.clip(float(self._stamp_xy[0]) + 0.24, xmin, xmax),
                np.clip(float(self._stamp_xy[1]) - 0.24, ymin, ymax),
            ], dtype=np.float64)
            doc1_pos = np.array(
                [doc1_xy[0], doc1_xy[1], self.TABLE_TOP_Z + half_z + 0.002],
                dtype=np.float64,
            )
            self.document_actors[1].set_pos(doc1_pos)

        if hasattr(self, "_feeder_xy") and self._feeder_xy:
            self._feeder_xy[0] = (float(self._stamp_xy[0]),
                                  float(self._stamp_xy[1]))
            if len(self._feeder_xy) > 1 and doc1_xy is not None:
                self._feeder_xy[1] = (float(doc1_xy[0]), float(doc1_xy[1]))

    def _clear_episode_buffers(self):
        if self.vla_recorder is not None:
            self.vla_recorder._captures.clear()
            self.vla_recorder._tick = 0
            self.vla_recorder._t0 = time.time()
        self.traj_data = []
        self.FRAME_IDX = 0
        # The collision log restarts here, so its sim-step clock does too.
        self.SIM_STEP_IDX = 0
        self.avatar_collided = False
        self.avatar_collision_log = []
        self.avatar_safety_log = []
        self.avatar_safety_intervention_log = []
        self._avatar_collision_tick = 0

    def load_actors(self):
        """Spawn docs (count, size, colour, feeder XY all per-seed), the
        seal, the ink pad, and one red mark cylinder per doc.

        Avatar pose (XY + yaw) is jittered before `_init_avatar` runs;
        the ink pad position then follows automatically because
        `_calibrate_motion_key_frames` derives it from the world palm
        trajectory, which moves with the avatar.
        """
        table_top = self.TABLE_TOP_Z

        # Avatar pose jitter — write instance attrs so reset() picks them up.
        base_pos = np.asarray(FeedDocuments.avatar_init_pos, dtype=np.float64)
        base_rot = np.asarray(FeedDocuments.avatar_init_rot, dtype=np.float64)
        avatar_dx = float(np.random.uniform(-_AVATAR_POS_JITTER_XY, _AVATAR_POS_JITTER_XY))
        avatar_dy = float(np.random.uniform(-_AVATAR_POS_JITTER_XY, _AVATAR_POS_JITTER_XY))
        avatar_dyaw = float(np.deg2rad(np.random.uniform(
            -_AVATAR_YAW_JITTER_DEG, _AVATAR_YAW_JITTER_DEG)))
        c, s = np.cos(avatar_dyaw), np.sin(avatar_dyaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.avatar_init_pos = base_pos + np.array([avatar_dx, avatar_dy, 0.0])
        self.avatar_init_rot = Rz @ base_rot

        # Doc count + size + feeder positions.
        n_docs = _DOC_COUNT
        size_scale = float(np.random.uniform(*_DOC_SIZE_SCALE_RANGE))
        self._doc_half = tuple(float(h) * size_scale for h in _DOC_HALF)
        col_dx = float(np.random.uniform(-_FEEDER_X_JITTER, _FEEDER_X_JITTER))
        self._feeder_xy = []
        for (x, y) in _DOC_FEEDER_XY[:n_docs]:
            dy = float(np.random.uniform(-_FEEDER_Y_JITTER, _FEEDER_Y_JITTER))
            self._feeder_xy.append((float(x) + col_dx, float(y) + dy))

        # Documents.
        self.document_actors = []
        z = table_top + self._doc_half[2] + 0.002
        for (x, y) in self._feeder_xy:
            color = _DOC_COLOR_POOL[int(np.random.randint(len(_DOC_COLOR_POOL)))]
            self.document_actors.append(create_primitive(
                self.scene, "box",
                Pose([x, y, z]),
                size={"half_size": self._doc_half},
                color=color,
                is_static=False,
            ))

        # Stamp (seated in palm during reset()) and ink pad (moved in
        # reset() once calibration locates the dip XY).
        self.stamp_actor = load_object(
            self.scene,
            Pose([0.30, 0.60, table_top + 0.30], _IDENTITY_Q),
            "100_seal", model_id=0, convex=True, is_static=False,
        )
        if Path(_INKPAD_OBJ).is_file():
            try:
                self.ink_entity = load_mesh(
                    self.scene, _INKPAD_OBJ,
                    Pose([self._ink_xy[0], self._ink_xy[1], table_top], _IDENTITY_Q),
                    scale=1.0, is_static=True,
                )
            except FileNotFoundError:
                self.ink_entity = None
        else:
            self.ink_entity = None
        if self.ink_entity is None:
            self.ink_entity = create_primitive(
                self.scene, "box",
                Pose([self._ink_xy[0], self._ink_xy[1], table_top + 0.005], _IDENTITY_Q),
                size={"half_size": (0.045, 0.030, 0.005)},
                color=(0.05, 0.04, 0.04),
                is_static=True,
            )

        # Red mark cylinders, one per doc.  Spawned under the table with
        # collision off; snapped to the doc's top face by `_attach_stamp_mark`
        # the moment that doc's stamp action fires.
        self.stamp_marks = []
        for i in range(len(self._feeder_xy)):
            self.stamp_marks.append(create_primitive(
                self.scene, "cylinder",
                Pose([0.0, 0.0, -0.5 - 0.01 * i]),
                size={"radius": _STAMP_MARK_RADIUS,
                      "half_length": _STAMP_MARK_HALF_H},
                color=(0.85, 0.08, 0.08),
                is_static=True, collision=False,
            ))
        n = len(self._feeder_xy)
        self._stamp_mark_attached: list = [None] * n
        self._stamp_mark_local_offset: list = [None] * n
        self._stamp_within_doc: list = [False] * n

    # ------------------------------------------------------------------
    # Calibration: dry-run the stamp motion to find dip / stamp positions
    # ------------------------------------------------------------------
    def _calibrate_motion_key_frames(self):
        """Replay `stamp_crop` once on the avatar without rendering, record
        the right-hand trajectory, and pick the dip and stamp frames.

        Dip = local-min hand z while hand y is large (closer to body).
        Stamp = local-min hand z while hand y is small (extended forward).
        """
        if self.avatar is None:
            return

        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        trajectory = []
        self._freeze_avatar = False
        while not self.avatar.spare():
            self.step_sim()
            palm = to_numpy(
                self.avatar.robot.get_palm_center(_STAMP_HAND_ID)
            ).ravel()
            trajectory.append(palm.copy())
        trajectory = np.asarray(trajectory)

        y_med = np.median(trajectory[:, 1])
        fwd_mask = trajectory[:, 1] < y_med
        back_mask = ~fwd_mask
        stamp_idx = np.argmin(np.where(fwd_mask, trajectory[:, 2], np.inf))
        dip_idx = np.argmin(np.where(back_mask, trajectory[:, 2], np.inf))

        self._stamp_xy = trajectory[stamp_idx, :2].copy()
        self._ink_xy = trajectory[dip_idx, :2].copy()
        self._stamp_cycle_sim_steps = int(len(trajectory))
        # trajectory[0] is sampled after PlayAnimationMotion advances from
        # frame 0 to frame 1, so convert the sample index back to at_frame.
        self._stamp_motion_frame = int(stamp_idx) + 1

        ink_new = np.array([self._ink_xy[0], self._ink_xy[1], self.TABLE_TOP_Z])
        self.ink_entity.set_pos(ink_new.astype(np.float64))

        # Reset to idle so calibration's intra-motion translation undoes,
        # then seed avatar to frame 0 so the play_once() transition reads
        # as a continuation rather than a jump from idle.
        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(20):
            self.step_sim()
        self._seed_avatar_to_stamp_frame0()

    def _seed_avatar_to_stamp_frame0(self):
        """Pose avatar at frame 0 of `_STAMP_MOTION` without advancing it."""
        if self.avatar is None:
            return
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        mod = self.avatar.motion_modules[_STAMP_MOTION]
        self.avatar.robot.pose = mod.data[0]
        self.avatar.robot.node_trans = mod.node_data[0]
        self.avatar.robot.global_mat = mod.global_mat
        self.avatar.robot.global_mat_inv = mod.global_mat_inv
        self.avatar.robot.action_state = AvatarState.NO_ACTION
        self.avatar.robot.action_status = ActionStatus.SUCCEED
        self.avatar.motion_modules.pop(_STAMP_MOTION, None)
        self.avatar.robot.update()
        # Rest-pose snapshot the eval path blends back to after each stamp.
        robot = self.avatar.robot
        try:
            self._stamp_rest_pose = {
                "pose": copy.deepcopy(robot.pose),
                "node_trans": copy.deepcopy(robot.node_trans),
                "global_mat": copy.deepcopy(robot.global_mat),
                "global_mat_inv": copy.deepcopy(robot.global_mat_inv),
            }
        except Exception:
            self._stamp_rest_pose = None
        self._stamp_retreat_pose = self._snapshot_retreat_pose(float(self.config.get(
            "stamp_eval_retreat_fraction", self.EVAL_RETREAT_FRACTION)))
        self._update_arm_obstacles("right")

    def _snapshot_retreat_pose(self, fraction: float):
        """Pose ``fraction`` of the way from the stamp clip's frame 0 back to
        the avatar's idle pose (same slerp blend as ``play_return_to_idle``).

        The eval human rests here between stamps: the stamp is held in front
        of the body, still in view, but the hand is pulled back out of the
        robot's doc-0-clear / doc-1-push corridor.  The avatar is left at
        frame 0, as before.
        """
        rest = getattr(self, "_stamp_rest_pose", None)
        if self.avatar is None or rest is None:
            return None
        robot = self.avatar.robot
        frames = 100
        k = int(round(min(max(float(fraction), 0.0), 1.0) * frames))
        snap = None

        def put_rest():
            robot.pose = copy.deepcopy(rest["pose"])
            robot.node_trans = copy.deepcopy(rest["node_trans"])
            robot.global_mat = copy.deepcopy(rest["global_mat"])
            robot.global_mat_inv = copy.deepcopy(rest["global_mat_inv"])
            robot.action_state = AvatarState.NO_ACTION
            robot.action_status = ActionStatus.SUCCEED
            robot.update()

        try:
            put_rest()
            if k > 0:
                self.avatar.play_return_to_idle(frames=frames)
                motion = self.avatar.motion_modules["transition"]
                motion.at_frame = k - 1
                motion.step()
                robot.update()
            snap = {
                "pose": copy.deepcopy(robot.pose),
                "node_trans": copy.deepcopy(robot.node_trans),
                "global_mat": copy.deepcopy(robot.global_mat),
                "global_mat_inv": copy.deepcopy(robot.global_mat_inv),
            }
        except Exception:
            snap = None
        finally:
            self.avatar.motion_modules.pop("transition", None)
            put_rest()
        return snap

    # ------------------------------------------------------------------
    # Avatar arm capsule for the planner
    # ------------------------------------------------------------------
    def _update_arm_obstacles(self, arm_tag: str = "right"):
        """Push a point-cloud capsule covering the avatar's right
        elbow→pinky into the mplib planner so screw/RRT plans route around
        the avatar arm instead of through it."""
        if self.avatar is None:
            return
        try:
            skin = self.avatar.robot.skin
            elbow = to_numpy(skin.get_global_translation("RightForeArm")[0]).ravel()[:3]
            pinky = to_numpy(skin.get_global_translation("RightHandPinky1")[0]).ravel()[:3]
        except Exception:
            return

        axis = pinky - elbow
        length = float(np.linalg.norm(axis))
        if length < 1e-4:
            return
        axis = axis / length
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, ref)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(axis, u)

        angles = np.linspace(0.0, 2.0 * np.pi, self._ARM_CAPSULE_N_AROUND, endpoint=False)
        pts = []
        for t in np.linspace(0.0, 1.0, self._ARM_CAPSULE_N_ALONG):
            center = elbow + t * (pinky - elbow)
            for a in angles:
                pts.append(center + self._ARM_CAPSULE_RADIUS * (np.cos(a) * u + np.sin(a) * v))
        pts = np.asarray(pts, dtype=np.float64)

        try:
            arm = self.robot.get_arm(arm_tag)
            arm.planner.update_obstacles(pts, resolution=0.02)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Avatar play
    # ------------------------------------------------------------------
    def _attach_stamp_to_palm(self):
        """Mark the stamp 'ready' so `_rigid_vertical_stamp` overrides its
        pose every sim step.  We don't use avatar.attach_object_to_hand:
        the Mixamo hand pose isn't perfectly palm-down at the stamp frame,
        so a rigid attach would leave the stamp tilted."""
        if self.avatar is None:
            return
        self.step_sim()
        self._stamp_ready = True
        self._rigid_vertical_stamp()

    def _play_one_stamp_cycle(self, doc_idx: int | None = None, doc=None,
                              pin_at_stamp_xy: bool = True):
        """Play one `stamp_crop` cycle (dip → stamp).  When doc/doc_idx
        are given, the red mark snaps onto the doc at the local-min palm-z
        within the stamp half of the trajectory."""
        if self.avatar is None:
            return
        self._freeze_avatar = False
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        motion = self.avatar.motion_modules[_STAMP_MOTION]

        stamp_half_y_cutoff = 0.5 * (float(self._stamp_xy[1]) + float(self._ink_xy[1]))
        stamp_is_fwd = float(self._stamp_xy[1]) < float(self._ink_xy[1])

        # Pin the document for the duration of the stamp.  The avatar's
        # descending hand/seal slides the (light, freshly-settled) doc ~0.23 m
        # forward under Genesis 1.0.0 — it flees the stamp zone before the seal
        # lands, so the imprint never falls on the paper.  A real document being
        # stamped does NOT slide, so we hold it at the pose the robot left it at.
        # (The robot's deliberate push happens earlier with the avatar frozen;
        # this only freezes the doc while the human stamps it.)
        pin_pos = None
        pin_quat = None
        if doc is not None:
            cur = to_numpy(doc.get_pos()).ravel().copy()
            # Pin the XY at the calibrated stamp position (where the seal lands)
            # rather than the possibly-already-drifted current XY, keeping the
            # current resting Z.  Both documents are stamped at stamp_xy (doc 0
            # is placed there; doc 1 is pushed there by the robot first).
            pin_pos = np.array(
                [float(self._stamp_xy[0]), float(self._stamp_xy[1]), float(cur[2])],
                dtype=np.float64,
            ) if pin_at_stamp_xy else cur
            pin_quat = to_numpy(doc.get_quat()).ravel().copy()

        # Track the deepest seal press while the hand is over the stamp zone.
        # The earlier "first local min that rises again" heuristic missed
        # (motion ended before the rise, or `near_doc` failed) and fell back to
        # the stale calibrated stamp_xy. Snapshot the seal XY AND the doc pose
        # together at the global minimum press instead; with the doc pinned the
        # seal is provably over the paper there.
        min_base_z = np.inf
        best_palm_xy = None
        best_doc_pos = None
        best_doc_quat = None
        stamped = False

        while not self.avatar.spare():
            self.step_sim()
            if doc is not None and pin_pos is not None:
                doc.set_pos(pin_pos.astype(np.float64))
                doc.set_quat(pin_quat.astype(np.float64))
            if doc is None or doc_idx is None:
                continue
            palm = to_numpy(
                self.avatar.robot.get_palm_center(_STAMP_HAND_ID)
            ).ravel()
            base_z = float(palm[2]) - _STAMP_BODY_LEN
            in_stamp_half = (palm[1] < stamp_half_y_cutoff) if stamp_is_fwd \
                else (palm[1] > stamp_half_y_cutoff)
            if in_stamp_half and base_z < min_base_z:
                min_base_z = base_z
                best_palm_xy = palm[:2].copy()
                best_doc_pos = to_numpy(doc.get_pos()).ravel().copy()
                best_doc_quat = to_numpy(doc.get_quat()).ravel().copy()
            if (not stamped and in_stamp_half
                    and motion.at_frame >= self._stamp_motion_frame):
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=palm[:2],
                    doc_pos_override=to_numpy(doc.get_pos()).ravel().copy(),
                    doc_quat_override=to_numpy(doc.get_quat()).ravel().copy())
                stamped = True

        if doc is not None and doc_idx is not None and not stamped:
            if best_palm_xy is not None:
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=best_palm_xy,
                    doc_pos_override=best_doc_pos,
                    doc_quat_override=best_doc_quat)
            else:
                # Hand never entered the stamp half (shouldn't happen) — fall
                # back to the calibrated stamp XY against the live doc pose.
                self._attach_stamp_mark(doc_idx, doc, mark_xy=self._stamp_xy)

        self._freeze_avatar = True
        self._update_arm_obstacles("right")

    # ------------------------------------------------------------------
    # Arm push (screw IK with Cartesian fallback — no RRT elbow flips)
    # ------------------------------------------------------------------
    def _tcp(self, xyz, push_axis: int | None = None) -> Pose:
        q = _TOP_DOWN_Q_Y_PUSH if push_axis == 1 else _TOP_DOWN_Q
        return Pose(np.array(xyz, dtype=np.float64), q)

    def _solve_ik_top_down(self, xyz, arm_tag, push_axis: int | None = None):
        arm = self.robot.get_arm(arm_tag)
        link = tcp_to_link_pose(self._tcp(xyz, push_axis=push_axis), arm.tcp_offset)
        pose7 = link.to_pose7()
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        try:
            cur = to_numpy(arm.entity.get_qpos()).ravel()
            sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=cur,
            )
        except Exception:
            return None
        if sol is None:
            return None
        return to_numpy(sol).ravel()[:arm.n_arm]

    def _move_cartesian(self, start_xyz, end_xyz, arm_tag,
                        push_axis: int | None = None,
                        n_steps=30, sim_per_step=15):
        """Linear Cartesian interpolation, IK per waypoint, PD-controlled."""
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            arm_qpos = self._solve_ik_top_down(
                start * (1 - t) + end * t, arm_tag, push_axis=push_axis)
            if arm_qpos is None:
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is None:
                    continue
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = qpos_full.copy()
        return True

    def _move_screw(self, target_xyz, arm_tag, push_axis: int | None = None):
        """Straight-line MPLib screw plan; falls back to manual Cartesian
        interp on screw failure."""
        arm = self.robot.get_arm(arm_tag)
        if self._blind_active:
            # Blind expert: plan around the avatar arm as observed right now.
            self._update_arm_obstacles(arm_tag)
        link = tcp_to_link_pose(self._tcp(target_xyz, push_axis=push_axis), arm.tcp_offset)
        try:
            result = arm.planner.plan_screw_path(arm.get_arm_qpos(), link.to_pose7())
            if getattr(result, "success", False):
                self.execute_plan(result, arm_tag)
                return True
        except Exception:
            pass
        cur_ee = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._move_cartesian(
            cur_ee, np.asarray(target_xyz, dtype=float), arm_tag, push_axis=push_axis)
        return True

    def _hold_arm(self, target_xyz, arm_tag, n_steps: int = 60,
                  push_axis: int | None = None):
        """Hold the arm with PD target at IK(target_xyz).  Holding the
        commanded TCP (not measured qpos) lets PD pull back if the screw
        trajectory overshot the stop."""
        arm = self.robot.get_arm(arm_tag)
        arm_qpos = self._solve_ik_top_down(
            np.asarray(target_xyz, dtype=float), arm_tag, push_axis=push_axis)
        if arm_qpos is None:
            arm_qpos = to_numpy(arm.get_arm_qpos()).ravel()
        hold_gripper = arm.gripper_val
        for _ in range(n_steps):
            self.robot.set_arm_joints(arm_qpos, arm_tag)
            self.robot.set_gripper(hold_gripper, arm_tag)
            self.step_sim()

    def _push_doc(self, doc, target_xy, arm_tag,
                  y_only: bool = False, precise: bool = True) -> bool:
        """Two centerline axis pushes: one X push, then one Y push."""
        if not precise:
            tgt = np.asarray(target_xy, dtype=np.float64)
            src = to_numpy(doc.get_pos()).ravel()[:2]
            if abs(tgt[0] - src[0]) > 0.01:
                if not self._push_axis_midline(doc, axis=0,
                                               target_val=float(tgt[0]),
                                               arm_tag=arm_tag):
                    return False
            if y_only:
                return True
            src = to_numpy(doc.get_pos()).ravel()[:2]
            if abs(tgt[1] - src[1]) > 0.01:
                if not self._push_axis_midline(doc, axis=1,
                                               target_val=float(tgt[1]),
                                               arm_tag=arm_tag):
                    return False
            return True

        max_segment = 0.10 if precise else 1.0
        max_segments = 4 if precise else 1
        tgt = np.asarray(target_xy, dtype=np.float64)
        src = to_numpy(doc.get_pos()).ravel()[:2]
        if abs(tgt[0] - src[0]) > 0.01:
            if not self._push_axis(doc, axis=0, target_val=float(tgt[0]),
                                   arm_tag=arm_tag, max_segment=max_segment,
                                   max_segments=max_segments):
                return False
        if y_only:
            return True
        src = to_numpy(doc.get_pos()).ravel()[:2]
        if abs(tgt[1] - src[1]) > 0.01:
            if not self._push_axis(doc, axis=1, target_val=float(tgt[1]),
                                   arm_tag=arm_tag, max_segment=max_segment,
                                   max_segments=max_segments):
                return False
        return True

    def _push_axis_midline(self, doc, axis: int, target_val: float,
                           arm_tag: str,
                           line_offset: float | None = None) -> bool:
        """Single Piper-style push along one axis.

        The commanded TCP is the fingertip center, not the true side-contact
        face.  For thin documents the stable behavior is to use a partially
        closed gripper and stop the TCP behind the requested document center
        so the paper, not the TCP, lands at the target.  `line_offset` shifts
        the contact line along the perpendicular axis for obstacle clearance.
        """
        p = to_numpy(doc.get_pos()).ravel()
        src_xy = p[:2].copy()
        remaining = target_val - src_xy[axis]
        if abs(remaining) < 0.01:
            return True

        perp_axis = 1 - int(axis)
        line_xy = src_xy.copy()
        if line_offset is not None:
            line_xy[perp_axis] = src_xy[perp_axis] + float(line_offset)

        direction = 1.0 if remaining > 0 else -1.0
        push_dir = np.zeros(2, dtype=np.float64)
        push_dir[axis] = direction

        doc_half_along = float(self._doc_half[axis])
        behind_gap = 0.085
        precontact_gap = 0.018
        contact_offset = doc_half_along + 0.010
        push_z = self.TABLE_TOP_Z + float(self._doc_half[2]) + 0.010
        safe_z = self.TABLE_TOP_Z + 0.18

        behind_xy = line_xy - push_dir * (doc_half_along + behind_gap)
        precontact_xy = line_xy - push_dir * (doc_half_along + precontact_gap)
        stop_xy = line_xy.copy()
        stop_xy[axis] = float(target_val) - direction * contact_offset
        retract_xy = stop_xy - push_dir * 0.070

        for xyz in (
            (behind_xy[0], behind_xy[1], safe_z),
            (behind_xy[0], behind_xy[1], push_z),
            (precontact_xy[0], precontact_xy[1], push_z),
            (stop_xy[0], stop_xy[1], push_z),
        ):
            if not self._move_screw(list(xyz), arm_tag, push_axis=axis):
                return False
        self._hold_arm(
            [stop_xy[0], stop_xy[1], push_z], arm_tag, n_steps=45, push_axis=axis)
        if not self._move_screw(
                [retract_xy[0], retract_xy[1], safe_z], arm_tag, push_axis=axis):
            return False
        for _ in range(30):
            self.step_sim()
        return True

    def _clear_first_doc_push(self, doc, target_x: float, arm_tag: str) -> bool:
        """Clear the stamped document with one x-axis push.

        The avatar hand is intentionally left where the stamp motion ends.
        To avoid clipping that hand, contact the document on the y edge
        farther from the current palm instead of pushing through its center.
        """
        doc_xy = to_numpy(doc.get_pos()).ravel()[:2]
        edge_offset = _FIRST_DOC_PUSH_EDGE_FRACTION * float(self._doc_half[1])
        line_offset = -edge_offset
        if self.avatar is not None:
            try:
                palm_xy = np.asarray(
                    self.avatar.robot.get_palm_center(_STAMP_HAND_ID),
                    dtype=np.float64,
                ).ravel()[:2]
                sign = -1.0 if palm_xy[1] >= doc_xy[1] else 1.0
                line_offset = sign * edge_offset
            except Exception:
                pass
        return self._push_axis_midline(
            doc, axis=0, target_val=target_x, arm_tag=arm_tag,
            line_offset=line_offset,
        )

    def _push_axis(self, doc, axis: int, target_val: float, arm_tag,
                   max_segment: float = 0.10,
                   max_segments: int = 4,
                   min_progress: float = 0.01) -> bool:
        """Axis-aligned push, segmented with perp-axis re-alignment.

        Long pushes split into ≤max_segment chunks; between segments we
        re-fetch the doc pose and re-align the TCP perp-axis with the doc's
        current center, so transverse drift can't accumulate.

        Contact: TCP is at the gripper's pushing face (no stamp_radius
        offset), so it stops at `seg_end − direction · doc_half_along` and
        the doc's edge lands at seg_end.
        """
        doc_half_along = self._doc_half[axis]
        push_z = self.TABLE_TOP_Z + 0.002
        safe_z = self.TABLE_TOP_Z + 0.18

        prev_src_axis = None
        for _ in range(max_segments):
            p = to_numpy(doc.get_pos()).ravel()
            src_xy = p[:2].copy()
            remaining = target_val - src_xy[axis]
            if abs(remaining) < 0.005:
                return True
            # Bail if the previous segment didn't move the doc — would
            # otherwise loop forever under repeated screw failures.
            if prev_src_axis is not None:
                if abs(src_xy[axis] - prev_src_axis) < min_progress:
                    return True
            prev_src_axis = src_xy[axis]

            direction = 1.0 if remaining > 0 else -1.0
            push_dir = np.zeros(2)
            push_dir[axis] = direction
            seg_len = min(abs(remaining), max_segment)
            seg_end = src_xy[axis] + direction * seg_len
            behind_xy = src_xy - push_dir * (doc_half_along + 0.06)
            stop_xy = src_xy.copy()
            stop_xy[axis] = seg_end - direction * doc_half_along
            retract_xy = stop_xy - push_dir * 0.05

            for xyz in (
                (behind_xy[0], behind_xy[1], safe_z),
                (behind_xy[0], behind_xy[1], push_z),
                (stop_xy[0],   stop_xy[1],   push_z),
            ):
                if not self._move_screw(list(xyz), arm_tag):
                    return False
            self._hold_arm([stop_xy[0], stop_xy[1], push_z], arm_tag, n_steps=60)
            if not self._move_screw([retract_xy[0], retract_xy[1], safe_z], arm_tag):
                return False
        return True

    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        self._attach_stamp_to_palm()
        arm_tag = "right"
        self.set_gripper(_PUSH_GRIPPER_VAL, arm_tag)

        # Standby: behind-doc in −Y, up high — clear of the avatar's stamp
        # arc so the avatar has an unobstructed approach to the doc.
        standby_xyz = [self._stamp_xy[0],
                       float(self._stamp_xy[1]) - 0.20,
                       self.TABLE_TOP_Z + 0.22]

        # Avatar frozen by default; `_play_one_stamp_cycle` un/refreezes.
        self._freeze_avatar = True

        if len(self.document_actors) != _DOC_COUNT:
            return False

        first_doc, second_doc = self.document_actors

        # Doc 0 was placed at stamp_xy during reset(); start clear of the
        # avatar and let the avatar stamp it before the arm moves it away.
        self._move_screw(standby_xyz, arm_tag)
        self._play_one_stamp_cycle(doc_idx=0, doc=first_doc)

        src = to_numpy(first_doc.get_pos()).ravel()[:2]
        clear_x = max(
            self.TABLE_CENTER_XY[0] - self.TABLE_HALF_SIZE[0] + 0.12,
            float(src[0]) - 0.18,
        )
        if not self._clear_first_doc_push(first_doc, clear_x, arm_tag):
            return False

        if not self._push_doc(second_doc, self._stamp_xy, arm_tag, precise=False):
            return False
        self._move_screw(standby_xyz, arm_tag)
        self._play_one_stamp_cycle(doc_idx=1, doc=second_doc)

        self._freeze_avatar = False
        return True

    # ------------------------------------------------------------------
    # Eval-mode human (no pre-policy warmup): random-start, non-blocking
    # ------------------------------------------------------------------
    def _eval_human_mode(self) -> str:
        return str(self.config.get("stamp_eval_human", self.EVAL_HUMAN))

    def _eval_human_enabled(self) -> bool:
        """The random-start human (VLA policy path and both eval experts)."""
        return (
            self.avatar is not None
            and bool(self.config.get("eval_mode", False))
            and self._eval_human_mode() == "random_start"
        )

    def _eval_at_reset(self):
        """Draw the first-stamp trigger.  Runs inside ``BaseTask.reset`` (before
        stamp calibration); the seal is seated in ``_prepare_episode_start``."""
        self._eval_trigger_step = None
        self._eval_cycle = None
        self._eval_cycles_log = []
        self._eval_sim_step = None
        self._eval_easing = False
        self._eval_avatar_at_retreat = False
        self._eval_robot_clear_steps = 0
        if not self._eval_human_enabled():
            return
        lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
        hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
        # Private per-seed stream: the draw never shifts the global RNG.
        rng = np.random.default_rng(
            [int(getattr(self, "_episode_seed", 0)), self._EVAL_RNG_SALT])
        self._eval_trigger_step = int(rng.integers(lo, max(lo, hi) + 1))

    def _eval_at_step(self, step: int):
        if not self._eval_human_enabled() or self._eval_trigger_step is None:
            return
        if self._eval_cycle is not None:
            return
        if getattr(self, "_eval_easing", False):
            # Non-blocking ease from the retreat pose back to stamp frame 0;
            # the stamp starts once it has finished.
            if not self.avatar.spare():
                return
            self._eval_easing = False
            self._eval_start_stamp_cycle(int(step))
            return
        n_done = len(self._eval_cycles_log)
        if n_done >= self._EVAL_HUMAN_MAX_CYCLES:
            return
        if n_done == 0:
            if int(step) >= self._eval_trigger_step:
                self._eval_start_stamp_cycle(int(step))
            return
        # Later stamps: start once a fresh document covers the stamp zone and
        # the robot hand has stayed out of the stamp arc.
        try:
            tcp_xy = self._robot_tcp_xy()
        except Exception:
            return
        clear_r = float(self.config.get(
            "stamp_eval_robot_clear_radius", self.EVAL_ROBOT_CLEAR_RADIUS))
        if float(np.linalg.norm(tcp_xy - np.asarray(self._stamp_xy)[:2])) >= clear_r:
            self._eval_robot_clear_steps = int(
                getattr(self, "_eval_robot_clear_steps", 0)) + 1
        else:
            self._eval_robot_clear_steps = 0
        required = int(self.config.get(
            "stamp_eval_robot_clear_steps", self.EVAL_ROBOT_CLEAR_STEPS))
        if self._eval_robot_clear_steps < required or self._eval_doc_in_zone() is None:
            return
        if getattr(self, "_eval_avatar_at_retreat", False):
            self._eval_avatar_at_retreat = False
            self._eval_easing = self._eval_start_ease_to_stamp_start()
            if self._eval_easing:
                return
        self._eval_start_stamp_cycle(int(step))

    def _eval_start_ease_to_stamp_start(self) -> bool:
        """Non-blocking twin of ``_eval_ease_to_stamp_start``."""
        rest = getattr(self, "_stamp_rest_pose", None)
        if self.avatar is None or rest is None:
            return False
        self._freeze_avatar = False
        self.avatar.play_transition(
            target_pose=rest["pose"],
            target_node=rest["node_trans"],
            target_mat=rest["global_mat"],
            target_mat_inv=rest["global_mat_inv"],
            frames=int(self.config.get(
                "stamp_eval_rest_frames", self.EVAL_REST_TRANSITION_FRAMES)),
        )
        return True

    def _eval_doc_in_zone(self):
        """Index of an unstamped document whose footprint covers stamp_xy."""
        for i, doc in enumerate(self.document_actors):
            if self._stamp_mark_attached[i] is not None:
                continue
            pos = to_numpy(doc.get_pos()).ravel()
            R = t3d.quaternions.quat2mat(to_numpy(doc.get_quat()).ravel())
            local = R.T @ (np.array([float(self._stamp_xy[0]), float(self._stamp_xy[1]),
                                     float(pos[2])]) - pos)
            if (abs(local[0]) <= self._doc_half[0]
                    and abs(local[1]) <= self._doc_half[1]):
                return i
        return None

    def _eval_start_stamp_cycle(self, policy_step: int):
        """Start one `stamp_crop` cycle; `_eval_stamp_tick` advances it per
        sim step with the same pin / press-tracking / mark rules as
        `_play_one_stamp_cycle`.  The doc pinned and stamped is the unstamped
        one covering the stamp zone when the cycle starts (None = air stamp)."""
        doc_idx = self._eval_doc_in_zone()
        doc = self.document_actors[doc_idx] if doc_idx is not None else None
        self._freeze_avatar = False
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        cyc = dict(
            motion=self.avatar.motion_modules[_STAMP_MOTION],
            doc_idx=doc_idx, doc=doc,
            pin_pos=None, pin_quat=None,
            min_base_z=np.inf, best_palm_xy=None,
            best_doc_pos=None, best_doc_quat=None, stamped=False,
            press_sim=None,
            policy_step=int(policy_step),
            start_sim=self._eval_sim_step,
        )
        if doc is not None:
            cyc["pin_pos"] = to_numpy(doc.get_pos()).ravel().copy()
            cyc["pin_quat"] = to_numpy(doc.get_quat()).ravel().copy()
        self._eval_cycle = cyc

    def _eval_stamp_tick(self):
        cyc = self._eval_cycle
        doc, doc_idx = cyc["doc"], cyc["doc_idx"]
        if doc is not None:
            doc.set_pos(cyc["pin_pos"].astype(np.float64))
            doc.set_quat(cyc["pin_quat"].astype(np.float64))
            palm = to_numpy(self.avatar.robot.get_palm_center(_STAMP_HAND_ID)).ravel()
            stamp_half_y_cutoff = 0.5 * (float(self._stamp_xy[1]) + float(self._ink_xy[1]))
            stamp_is_fwd = float(self._stamp_xy[1]) < float(self._ink_xy[1])
            base_z = float(palm[2]) - _STAMP_BODY_LEN
            in_stamp_half = (palm[1] < stamp_half_y_cutoff) if stamp_is_fwd \
                else (palm[1] > stamp_half_y_cutoff)
            if in_stamp_half and base_z < cyc["min_base_z"]:
                cyc["min_base_z"] = base_z
                cyc["best_palm_xy"] = palm[:2].copy()
                cyc["best_doc_pos"] = to_numpy(doc.get_pos()).ravel().copy()
                cyc["best_doc_quat"] = to_numpy(doc.get_quat()).ravel().copy()
            if (not cyc["stamped"] and in_stamp_half
                    and cyc["motion"].at_frame >= self._stamp_motion_frame):
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=palm[:2],
                    doc_pos_override=to_numpy(doc.get_pos()).ravel().copy(),
                    doc_quat_override=to_numpy(doc.get_quat()).ravel().copy())
                cyc["stamped"] = True
                cyc["press_sim"] = self._eval_sim_step
        if not self.avatar.spare():
            return
        if doc is not None and not cyc["stamped"]:
            if cyc["best_palm_xy"] is not None:
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=cyc["best_palm_xy"],
                    doc_pos_override=cyc["best_doc_pos"],
                    doc_quat_override=cyc["best_doc_quat"])
            else:
                self._attach_stamp_mark(doc_idx, doc, mark_xy=self._stamp_xy)
        # Rest between stamps (EVAL_REST_POSE): pull the hand back out of the
        # robot's clear / push corridor, non-blocking.
        if self._eval_rest_mode() in ("retreat", "idle"):
            self._eval_return_avatar_to_rest()
            self._eval_avatar_at_retreat = True
        else:
            self._freeze_avatar = True
        self._eval_cycles_log.append(dict(
            doc_idx=doc_idx,
            policy_step=cyc["policy_step"],
            start_sim_step=cyc["start_sim"],
            press_sim_step=cyc["press_sim"],
            end_sim_step=self._eval_sim_step,
            on_doc=(bool(self._stamp_within_doc[doc_idx]) if doc_idx is not None else False),
        ))
        self._eval_cycle = None

    # ------------------------------------------------------------------
    # Blind expert: same pushes as play_once on the robot's own clock
    # ------------------------------------------------------------------
    def play_blind_once(self) -> bool:
        """Clear the document in the stamp zone and feed the other one, never
        waiting on the human.  Targets are the documents' observed poses."""
        arm_tag = "right"
        self._blind_active = True
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        try:
            if len(self.document_actors) != _DOC_COUNT:
                return False
            self.set_gripper(_PUSH_GRIPPER_VAL, arm_tag)
            first_doc, second_doc = self.document_actors
            # The stamp zone is wherever the document already lying in it is.
            zone_xy = to_numpy(first_doc.get_pos()).ravel()[:2].copy()
            standby_xyz = [float(zone_xy[0]), float(zone_xy[1]) - 0.20,
                           self.TABLE_TOP_Z + 0.22]
            self._move_screw(standby_xyz, arm_tag)
            # Full-state eval expert only (no-op for the blind expert): wait
            # until the human has stamped doc 0 and is idle.
            self._await_cue("doc0_stamped", self._eval_doc0_stamped)

            self._blind_robot_interaction_sim_step = self._eval_sim_step
            log = getattr(self, "_eval_cycles_log", [])
            self._blind_human_ready_before_robot = bool(
                log and log[0].get("doc_idx") == 0)

            src = to_numpy(first_doc.get_pos()).ravel()[:2]
            clear_x = max(
                self.TABLE_CENTER_XY[0] - self.TABLE_HALF_SIZE[0] + 0.12,
                float(src[0]) - 0.18,
            )
            if not self._clear_first_doc_push(first_doc, clear_x, arm_tag):
                return False
            if not self._push_doc(second_doc, zone_xy, arm_tag, precise=False):
                return False
            # The single midline push can leave the document short of the
            # zone (13-24 cm short on some layouts, so the human never stamps
            # it).  Re-push from
            # the document's observed pose until its footprint covers the
            # stamp point, a bounded number of times.
            for _ in range(self._FEED_REPUSH_MAX):
                if doc_covers_point(
                        to_numpy(second_doc.get_pos()).ravel()[:3],
                        to_numpy(second_doc.get_quat()).ravel()[:4],
                        zone_xy, self._doc_half):
                    break
                print("[stamp] fed document misses the stamp point; re-push", flush=True)
                if not self._push_doc(second_doc, zone_xy, arm_tag, precise=False):
                    break
            self._move_screw(standby_xyz, arm_tag)

            # Bounded tail: hold standby until both docs carry a stamp and the
            # human is idle, or the cap runs out.
            for _ in range(self._BLIND_SETTLE_MAX_SIM_STEPS):
                if self._eval_cycle is None and all(
                        d is not None for d in self._stamp_mark_attached):
                    break
                self.step_sim()
            return True
        finally:
            self._blind_active = False

    def play_full_eval_once(self) -> bool:
        """Full-state expert under the eval-mode human: play_blind_once's
        robot script, waiting for doc 0's stamp before clearing it."""
        self._full_eval_active = True
        try:
            return self.play_blind_once()
        finally:
            self._full_eval_active = False

    def _await_cue(self, label, ready):
        return await_cue(self, ready, label)

    def _eval_doc0_stamped(self) -> bool:
        """Doc 0 carries its stamp and the human is not mid-stamp."""
        return bool(self._stamp_mark_attached[0] is not None
                    and self._eval_cycle is None
                    and not getattr(self, "_eval_easing", False))

    def evaluate(self) -> dict:
        out = super().evaluate()
        if bool(self.config.get("eval_mode", False)):
            if getattr(self, "_full_eval_cue_waits", None) is not None:
                out["full_eval_cue_waits"] = self._full_eval_cue_waits
            out["eval_doc0_stamp_step"] = getattr(self, "_eval_doc0_stamp_step", None)
            out["eval_doc1_stamp_fired"] = bool(getattr(self, "_eval_doc1_stamp_fired", False))
            out["eval_doc1_stamp_step"] = getattr(self, "_eval_doc1_stamp_step", None)
            out["stamp_within_doc"] = [
                bool(v) for v in (getattr(self, "_stamp_within_doc", None) or [])
            ]
            log = list(getattr(self, "_eval_cycles_log", []) or [])
            first = log[0] if log else None
            cyc = self._eval_cycle
            started = first["start_sim_step"] if first else (
                cyc["start_sim"] if cyc is not None else None)
            out["blind_intent"] = {
                "trigger_step": getattr(self, "_eval_trigger_step", None),
                "human_started_sim_step": started,
                "human_first_stamp_done_sim_step": first["end_sim_step"] if first else None,
                "human_first_press_sim_step": first["press_sim_step"] if first else None,
                "robot_interaction_sim_step": getattr(
                    self, "_blind_robot_interaction_sim_step", None),
                "human_ready_before_robot": getattr(
                    self, "_blind_human_ready_before_robot", None),
                "stamp_cycle_sim_steps": getattr(self, "_stamp_cycle_sim_steps", None),
                "stamp_cycles": log,
                "stamp_within_doc": [bool(v) for v in self._stamp_within_doc],
                "episode_sim_steps": self._eval_sim_step,
            }
        return out

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        # Success = every doc has its stamp mark inside the doc edge.  We no
        # longer require docs to have moved from the feeder XY: the last
        # stamped doc intentionally stays at the stamp pose (the rollout
        # ends as soon as it gets stamped).
        within = getattr(self, "_stamp_within_doc", None)
        if len(within or []) != _DOC_COUNT or not all(within):
            return False
        return True
