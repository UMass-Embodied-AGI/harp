"""Shared neutral-avatar table-work behavior.

Neutral variants keep the robot's original task objects untouched by the
avatar.  The avatar performs a separate tabletop pick/place using props
created by this mixin, while robot planning can include the live avatar
capsule point cloud as an mplib obstacle.
"""

from __future__ import annotations


import numpy as np
import transforms3d as t3d

from ..avatar.collision_checker import sample_capsule_planner_points
from ..avatar.utils import ActionStatus
from ..interaction.neutral_table_work import NeutralTableWorkController
from ..utils import (
    ASSETS_PATH,
    Pose,
    load_mesh,
    to_numpy,
)
from .neutral_avatar_clip_transforms import (
    install_loop_subclip,
)
# Redundant aliases mark deliberate re-exports: this module has always been
# the public surface for the neutral spec constants (contract-tested).
from .neutral_avatar_spec import (
    ROOT_PATH as ROOT_PATH,
    NeutralTaskSpec as NeutralTaskSpec,
    _NEUTRAL_MOTION_META as _NEUTRAL_MOTION_META,
    _NEUTRAL_MOTION_PKL as _NEUTRAL_MOTION_PKL,
    _resolve_repo_path as _resolve_repo_path,
)
from .neutral_avatar_clips import _NeutralAvatarClipsMixin
from .neutral_avatar_props import _NeutralAvatarPropsMixin
from .neutral_avatar_layout import _NeutralAvatarLayoutMixin
from .neutral_avatar_contested import _NeutralAvatarContestedMixin
from .neutral_human_aware import _NeutralHumanAwareMixin


class NeutralAvatarTableWorkMixin(
    _NeutralHumanAwareMixin,
    _NeutralAvatarClipsMixin,
    _NeutralAvatarPropsMixin,
    _NeutralAvatarLayoutMixin,
    _NeutralAvatarContestedMixin,
):
    """Add an unrelated avatar tabletop job to a robot task."""

    use_avatar = True
    NEUTRAL_TASK_SPEC: NeutralTaskSpec | None = None
    NEUTRAL_MOTION_FOOTPRINTS: dict[str, dict] = {}
    NEUTRAL_AVATAR_MOTION_CHOICES = (
        "wipe_table_2_reverse",
        "wipe_table_2",
        "put_objects_in_bowl",
        "take_objects_from_bowl",
        "use_mouse",
        "open_bottle",
        "writing",
        "using_a_fax",
        "Typing",
        "sitting_clap",
    )
    # Short handles AND the historical raw clip names (KIT/GRAB stage-ii) both
    # resolve to the current clean motion keys, so older configs / saved runs
    # keep working after the rename.
    NEUTRAL_AVATAR_MOTION_ALIASES = {
        # wipe_table_2_reverse replaces the retired wipe_table_1 clip.
        "wipe_table_1": "wipe_table_2_reverse",
        "wiping_table": "wipe_table_2_reverse",
        "wipe_table": "wipe_table_2_reverse",
        "sponge": "wipe_table_2_reverse",
        "01_KIT_wipe_table_wiping_the_table01_stageii": "wipe_table_2_reverse",
        "wipe_table_reverse": "wipe_table_2_reverse",
        "wipe_table_2_reverse": "wipe_table_2_reverse",
        # wipe_table_2
        "03_KIT_wipe_table_wiping_the_table05_stageii": "wipe_table_2",
        # put_objects_in_bowl
        "putting_cans_into_plate": "put_objects_in_bowl",
        "put_cans_into_plate": "put_objects_in_bowl",
        "cans_plate": "put_objects_in_bowl",
        "can_plate": "put_objects_in_bowl",
        "09_KIT_put_objects_in_bowl_put_objects_in_mixing_bowl_02_stageii": "put_objects_in_bowl",
        # take_objects_from_bowl (reverse of put_objects_in_bowl: lift cans out)
        "reverse_cans": "take_objects_from_bowl",
        "take_cans_from_plate": "take_objects_from_bowl",
        "unload_bowl": "take_objects_from_bowl",
        # use_mouse
        "mouse": "use_mouse",
        "using_mouse": "use_mouse",
        "14_GRAB_use_mouse_mouse_use_1_stageii": "use_mouse",
        # open_bottle
        "bottle": "open_bottle",
        "open_water_bottle": "open_bottle",
        "18_GRAB_open_water_bottle_waterbottle_open_1_stageii": "open_bottle",
        # writing
        "write": "writing",
        "writing_on_table": "writing",
        # using_a_fax
        "laptop": "using_a_fax",
        "fax": "using_a_fax",
        "using_laptop": "using_a_fax",
        # Typing (seated laptop+mouse)
        "typing": "Typing",
        "type": "Typing",
        "typing_on_laptop": "Typing",
        # sitting_clap (seated clapping, chair only)
        "clap": "sitting_clap",
        "clapping": "sitting_clap",
        "sitting_clapping": "sitting_clap",
    }

    # Motion-metadata object handles are intentionally presentation-oriented
    # (``bottle``, ``mouse``, ``object_1``), while robot targets use catalog
    # asset IDs (``001_bottle``, ``047_mouse``, ...).  Resolve the fixed props
    # here so the neutral-motion sampler can compare actual object identity.
    NEUTRAL_PROP_ASSET_IDS = {
        "bottle": "001_bottle",
        "mouse": "047_mouse",
        "phone": "077_phone",
        "pen": "058_markpen",
        "notebook": "092_notebook",
        "sponge": "cc0_sponge_3",
        "plate": "003_plate",
        "laptop": "015_laptop",
        "chair": "sapien-chair-179",
    }

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.08, 0.58, -0.18], dtype=np.float64)
    avatar_init_rot = AVATAR_BASE_ROT

    NEUTRAL_PROP_HALF = 0.018
    NEUTRAL_SPONGE_HALF_THICKNESS = 0.015
    NEUTRAL_PROP_PICK_XY = np.array([0.34, 0.10], dtype=np.float64)
    NEUTRAL_PROP_PLACE_XY = np.array([0.46, -0.08], dtype=np.float64)
    NEUTRAL_AVATAR_NATURAL_BODY_MARGIN = 0.34
    NEUTRAL_AVATAR_NATURAL_BODY_Y_BOUNDS = (0.28, 0.66)
    NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP = 0.62
    NEUTRAL_AVATAR_TURN_RAD = 0.40
    NEUTRAL_AVATAR_TURN_STEP_RAD = 0.002
    NEUTRAL_AVATAR_MOTION_SLOW = 25.0
    # In eval mode the human starts within the first policy
    # steps, matching training collection (neutral_avatar_start_delay_steps=0).
    # A later start (e.g. U[10, 60]) lets short policy rollouts finish before
    # the human moves.  Rows can still override neutral_eval_trigger_step_min/max.
    NEUTRAL_EVAL_TRIGGER_STEP_MIN = 0
    NEUTRAL_EVAL_TRIGGER_STEP_MAX = 5
    NEUTRAL_AVATAR_HOLD_OFFSET = 0.05
    NEUTRAL_AVATAR_BODY_Z_OFFSET = 0.0
    NEUTRAL_PUT_OBJECTS_AVATAR_TABLE_Z_OFFSET = 0.0
    NEUTRAL_PUT_OBJECTS_BODY_EDGE_GAP = 0.48
    NEUTRAL_AVATAR_BOTTLE_HOLD_OFFSET = -0.04
    NEUTRAL_AVATAR_BOTTLE_BODY_Z_OFFSET = 0.0
    NEUTRAL_AVATAR_MOUSE_BODY_Z_OFFSET = 0.04
    NEUTRAL_MOUSE_CENTER_Z_OFFSET = 0.012
    # For the table-facing bottle clip, avatar-forward is world -Y and
    # avatar-right is world -X.
    NEUTRAL_AVATAR_BOTTLE_EXTRA_BODY_OFFSET = np.zeros(3, dtype=np.float64)
    NEUTRAL_AVATAR_BOTTLE_HAND_SWITCH_FRAMES = (44, 162)
    NEUTRAL_AVATAR_MOUSE_IDLE_Y_OFFSET = 0.0
    NEUTRAL_MOUSE_CLIP_TRIM = (8, 120)
    NEUTRAL_MOUSE_ATTACHED_FORWARD_LOOPS = 3
    NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS = (0, 1, 2, 3, 5, 6)
    NEUTRAL_PUT_OBJECTS_CAN_REST_Z_OFFSET = 0.002
    # The put-/take-objects motion's two handled props default to soda cans
    # (071_can, whose mesh origin is at the base, so resting z only needs a
    # small tabletop clearance).
    # Both the asset and the resting offset are config-overridable so a motion
    # can swap in e.g. apples (035_apple, mesh origin at the base → small offset
    # so the fruit rests just above the tabletop). Override via
    # `neutral_put_objects_object` / `neutral_put_objects_object_rest_z_offset`.
    NEUTRAL_PUT_OBJECTS_OBJECT = "071_can"
    NEUTRAL_PUT_OBJECTS_OBJECT_REST_Z_OFFSET = None  # None → can base clearance
    NEUTRAL_CAN_CLEARANCE_MARGIN = 0.015
    NEUTRAL_CAN_GRIPPER_CLEARANCE_RADIUS = 0.035
    NEUTRAL_CAN_UPRIGHT_DOT_MIN = 0.85
    NEUTRAL_AVATAR_TRANSITION_BASE_FRAMES = 40
    NEUTRAL_WORK_XY = np.array([0.34, 0.10], dtype=np.float64)
    NEUTRAL_MOUSE_WORK_XY = np.array([0.34, 0.22], dtype=np.float64)
    NEUTRAL_SPONGE_WORK_XY = None
    NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY = np.array([0.0, 0.20], dtype=np.float64)
    NEUTRAL_TABLE_XY_BOUNDS = (-0.50, 0.50, -0.80, 0.18)
    NEUTRAL_CLEARANCE_MARGIN = 0.015
    NEUTRAL_PROP_RADII = {
        "sponge": 0.070,
        "mouse": 0.050,
        "bottle": 0.040,
        "bottle_1": 0.055,
        "bottle_2": 0.055,
        "bottle_3": 0.055,
        "object_1": 0.040,
        "object_2": 0.040,
    }
    NEUTRAL_BOWL_HALF_XY = np.array([0.12, 0.08], dtype=np.float64)
    NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN = 0.06
    NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN = 0.06
    NEUTRAL_PUT_OBJECTS_TABLE_MARGIN = 0.15
    NEUTRAL_SPONGE_TABLE_MARGIN = 0.040
    NEUTRAL_BOTTLE_TABLE_MARGIN = 0.100
    NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP = 0.160
    NEUTRAL_USE_REGION_PLACEMENT = True
    NEUTRAL_ARM_REGION_BONES = (
        ("LeftArm", 0.045),
        ("LeftForeArm", 0.040),
        ("LeftHand", 0.035),
        ("LeftHandMiddle1", 0.025),
        ("RightArm", 0.045),
        ("RightForeArm", 0.040),
        ("RightHand", 0.035),
        ("RightHandMiddle1", 0.025),
    )
    NEUTRAL_ROBOT_REGION_RADIUS = 0.12
    NEUTRAL_ROBOT_BASE_REGION_RADIUS = 0.20

    # Seated-writing neutral clip (`writing`): the avatar sits on a chair just
    # off the table edge and writes in a notebook with a pen in the right
    # hand.  The avatar–pen–book–chair relative poses and the avatar–table
    # edge gap are all fixed; only the anchor (book center) slides along the
    # chosen table edge.  Reference values come from an offline writing-pose
    # probe with the avatar reset at the origin facing +X (motion frame).
    NEUTRAL_WRITING_BOOK_LOCAL_XY = (0.5213, -0.0375)  # book center, avatar frame
    NEUTRAL_WRITING_BOOK_EDGE_INSET = 0.22       # book center -> table edge
    NEUTRAL_WRITING_TABLE_TOP_REF = 0.7413       # hand-fit tabletop at body z=0
    NEUTRAL_WRITING_SEAT_HEIGHT_REF = 0.46       # chair seat height at body z=0
    NEUTRAL_WRITING_BOOK_SCALE = 0.18            # 092_notebook -> ~0.34 x 0.22 m
    NEUTRAL_WRITING_BOOK_HALF_LONG = 0.171       # long edge, across the writer
    NEUTRAL_WRITING_BOOK_HALF_DEPTH = 0.113      # depth, along the facing dir
    NEUTRAL_WRITING_PEN_SCALE = 0.1              # 058_markpen -> 0.19 m long
    NEUTRAL_WRITING_PEN_FWD = 0.01               # pen tip from palm, finger dir
    NEUTRAL_WRITING_PEN_TILT_DEG = 18.0          # lean back toward the writer
    NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z = 0.832048  # sapien-chair-179 raw geom
    NEUTRAL_WRITING_CHAIR_RAW_SEAT_H = 0.786
    NEUTRAL_WRITING_BOOK_CLEARANCE = 0.02        # extra ring around the book
    NEUTRAL_WRITING_TABLE_EDGE_MARGIN = 0.03     # book AABB -> lateral edges
    NEUTRAL_WRITING_CANDIDATE_STEP = 0.05        # lateral sweep step along edge
    # Over-table body envelope of the seated writer (avatar frame, facing +X):
    # both hands rest over the book and the head leans in past the table
    # edge.  Checked at selection time, and used to keep the anchor as far
    # from the robot's workspace as possible (the live capsule obstacles only
    # protect mplib plans, not screw/IK moves).
    NEUTRAL_WRITING_BODY_POINTS = (
        ("body:right_hand", (0.5213, -0.0375), 0.09),
        ("body:left_hand", (0.616, 0.108), 0.09),
        ("body:head", (0.32, 0.0), 0.14),
    )
    NEUTRAL_WRITING_ROBOT_REGION_PREFIXES = ("robot_base:", "gripper:")
    # Extra standoff between the body envelope and everything the gripper
    # visits (task objects, slots, targets): static disks don't capture the
    # arm's sweep, so contact-distance alone is not enough.
    NEUTRAL_WRITING_BODY_SWEEP_MARGIN = 0.08

    # Standing-laptop neutral clip (`using_a_fax`): the avatar stands just off
    # the table edge typing on an open SAPIEN laptop (015_laptop/9960).  The
    # avatar–laptop relative pose and the avatar–table-edge gap are fixed;
    # only the anchor (laptop origin) slides along the chosen table edge.
    # Reference values come from the offline laptop/fax calibration probe
    # (laptop scale 0.28, avatar yaw 30°, back offset 0.10, left offset 0.20).
    NEUTRAL_LAPTOP_LOCAL_XY = (0.79575, -0.054349)  # laptop origin, avatar frame
    NEUTRAL_LAPTOP_YAW_LOCAL_DEG = -13.6678      # laptop yaw - avatar yaw
    NEUTRAL_LAPTOP_EDGE_INSET = 0.20964          # laptop origin -> table edge
    NEUTRAL_LAPTOP_TABLE_TOP_REF = 0.937215      # implied tabletop at body z=0
    NEUTRAL_LAPTOP_SCALE = 0.28                  # 015_laptop/9960 -> ~0.32 m wide
    NEUTRAL_LAPTOP_RAW_BOTTOM_Z = 0.223          # pre-scale shell bottom below origin
    NEUTRAL_LAPTOP_RAW_X_BOUNDS = (-0.713, 0.297)  # pre-scale: front edge .. lid lean
    NEUTRAL_LAPTOP_RAW_Y_BOUNDS = (-0.582, 0.571)
    NEUTRAL_LAPTOP_LID_QPOS = 0.0                # open lid; held against gravity
    NEUTRAL_LAPTOP_CLEARANCE = 0.02              # extra ring around the laptop
    NEUTRAL_LAPTOP_TABLE_EDGE_MARGIN = 0.03      # laptop AABB -> lateral edges
    NEUTRAL_LAPTOP_CANDIDATE_STEP = 0.05         # lateral sweep step along edge
    NEUTRAL_LAPTOP_CLIP_TRIM = (122, 414)        # stationary typing window
    NEUTRAL_LAPTOP_PINGPONG_LOOPS = 3
    # Minimum gap (m) the laptop footprint must keep from every task region
    # for a placement to be accepted (a non-overlap floor).  The selector
    # then picks the MAX-clearance placement among those that pass, so the
    # laptop ends up as far from the robot's working area as the table allows
    # — dump_bin yields ~9 cm, the blocks tasks ~4-5 cm.
    NEUTRAL_LAPTOP_TASK_CLEARANCE = 0.01
    # Sub-centimetre grazes (e.g. the forearm skimming the open screen tip)
    # are not a meaningful "touch"; require this much penetration (m) into a
    # neutral object before the collision checker flags it.
    NEUTRAL_OBJECT_COLLISION_TOLERANCE = 0.012

    # Seated-typing neutral clip (`Typing`): the avatar sits on a chair just
    # off the table edge typing on an open SAPIEN laptop (015_laptop/9960)
    # with a mouse (047_mouse) on its right.  Combines the writing chair logic
    # with the laptop-on-table logic.  The avatar-laptop-mouse relative poses
    # and the avatar-table-edge gap are fixed; only the anchor (laptop origin)
    # slides along the chosen table edge.  Reference values come from an
    # offline typing-pose probe with the avatar reset at the
    # origin (identity rotation) — so the avatar-local frame is the world
    # frame and the probed world poses ARE the local-frame constants.
    NEUTRAL_TYPING_LAPTOP_LOCAL_XY = (0.476, 0.057)   # laptop origin, avatar frame
    NEUTRAL_TYPING_LAPTOP_YAW_LOCAL_DEG = 6.07        # laptop yaw - avatar yaw
    NEUTRAL_TYPING_MOUSE_LOCAL_XY = (0.546, -0.229)   # mouse center, avatar frame
    NEUTRAL_TYPING_MOUSE_YAW_LOCAL_DEG = -83.93       # mouse yaw - avatar yaw
    NEUTRAL_TYPING_MOUSE_Z_OFFSET = 0.012             # mouse origin above tabletop
    NEUTRAL_TYPING_EDGE_INSET = 0.21                  # laptop origin -> table edge
    NEUTRAL_TYPING_TABLE_TOP_REF = 0.6976             # implied tabletop at body z=0
    NEUTRAL_TYPING_SEAT_HEIGHT_REF = 0.46             # chair seat height at body z=0
    NEUTRAL_TYPING_SCALE = 0.28                       # 015_laptop/9960 -> ~0.32 m wide
    NEUTRAL_TYPING_MOUSE_SCALE_MODEL_ID = 0           # 047_mouse model_data0 (scale 0.5)
    NEUTRAL_TYPING_LID_QPOS = 0.0                     # open lid; held against gravity
    NEUTRAL_TYPING_CLEARANCE = 0.02                   # extra ring around the footprint
    NEUTRAL_TYPING_TABLE_EDGE_MARGIN = 0.03           # footprint AABB -> lateral edges
    NEUTRAL_TYPING_CANDIDATE_STEP = 0.05              # lateral sweep step along edge
    NEUTRAL_TYPING_CLIP_TRIM = (15, 480)              # drop T-pose blend-in/out frames
    NEUTRAL_TYPING_TASK_CLEARANCE = 0.01              # min footprint gap to task regions
    NEUTRAL_TYPING_MOUSE_HALF = 0.05                  # mouse footprint half-extent (m)

    # Seated-clapping neutral clip (`sitting_clap`): the avatar sits on the
    # same SAPIEN chair as the writing/typing clips, alongside the table with
    # no held/tabletop prop.  The edge anchor is the nearest table edge point
    # in the avatar frame; only the anchor slides along a clear table side.
    NEUTRAL_CLAP_TABLE_DISTANCE = 0.34
    NEUTRAL_CLAP_EDGE_ANCHOR_LOCAL_XY = (NEUTRAL_CLAP_TABLE_DISTANCE, 0.0)
    NEUTRAL_CLAP_TABLE_TOP_REF = 0.74
    NEUTRAL_CLAP_SEAT_HEIGHT_REF = 0.46
    NEUTRAL_CLAP_BODY_Z_OFFSET = -0.075
    NEUTRAL_CLAP_CHAIR_Z_OFFSET = -0.10
    NEUTRAL_CLAP_CLIP_TRIM = (14, 142)
    NEUTRAL_CLAP_EDGE_INSET = 0.02
    NEUTRAL_CLAP_TABLE_EDGE_MARGIN = 0.03
    NEUTRAL_CLAP_CANDIDATE_STEP = 0.05
    NEUTRAL_CLAP_TASK_CLEARANCE = 0.01
    NEUTRAL_CLAP_BODY_SWEEP_MARGIN = 0.08
    # Avatar-local points around the clapping hands/head, relative to the
    # table-edge anchor, used as the on-table clearance footprint.
    NEUTRAL_CLAP_LOCAL_REGION_POINTS = (
        ("body:hands", (0.22, 0.0), 0.16),
        ("body:left_hand", (0.22, 0.12), 0.10),
        ("body:right_hand", (0.22, -0.12), 0.10),
        ("body:head", (-0.04, 0.0), 0.14),
    )

    # Loop the neutral motion back-and-forth (forward -> reverse -> forward
    # ...) for the whole episode instead of playing once and freezing.  A
    # forward+reverse ping-pong clip is built at start time (seamless
    # turnarounds at both ends) and rewound on completion; events fire on
    # the forward half each pass.  Default on for every neutral motion/task.
    NEUTRAL_AVATAR_LOOP_MOTION = True

    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.6
    NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE = True
    # The neutral objects (the human's laptop / book / chair / cans / etc.) are
    # also off-limits to the robot: their world AABBs are pushed into the
    # planner obstacle cloud so plans route around them, and into the analytic
    # collision checker so a touch trips the avatar-collision success gate.
    NEUTRAL_OBJECT_AS_OBSTACLE = True
    NEUTRAL_OBJECT_OBSTACLE_INFLATE = 0.01  # metres added around each object AABB

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("NEUTRAL_TASK_SPEC")
        if spec is None:
            return
        if isinstance(spec, dict):
            spec = NeutralTaskSpec(**spec)
            cls.NEUTRAL_TASK_SPEC = spec
        if not isinstance(spec, NeutralTaskSpec):
            raise TypeError(
                f"{cls.__name__}.NEUTRAL_TASK_SPEC must be NeutralTaskSpec or dict"
            )
        spec.apply_to_class(cls)

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        avatar_cfg = dict(cfg.get("avatar") or {})
        motion_slow = float(cfg.get(
            "neutral_avatar_motion_slow",
            self.NEUTRAL_AVATAR_MOTION_SLOW,
        ))
        avatar_cfg.setdefault(
            "generated_motion_data",
            _resolve_repo_path(cfg.get("neutral_avatar_motion_pkl", _NEUTRAL_MOTION_PKL)),
        )
        avatar_cfg.setdefault("frame_ratio", motion_slow)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)

    def _neutral_avatar_motion_slow(self) -> float:
        return max(
            1.0,
            float(self.config.get(
                "neutral_avatar_motion_slow",
                self.NEUTRAL_AVATAR_MOTION_SLOW,
            )),
        )

    def _neutral_avatar_forced_motion_requested(self) -> bool:
        return bool(str((self.config or {}).get("neutral_avatar_motion", "")).strip())

    def _apply_neutral_camera_overrides(self) -> None:
        """Let a neutral config reposition the recording/side review cameras.

        BaseTask builds the scene cameras from ``self.recording_camera_*`` /
        ``self.side_camera_*`` (class attrs, default None). A neutral motion
        whose avatar would block the default angle can set these via config —
        e.g. the seated take-objects clip. No-op unless the keys are present, so
        every other task keeps its default cameras.
        """
        for cfg_key, attr in (
            ("neutral_recording_camera_pos", "recording_camera_pos"),
            ("neutral_recording_camera_lookat", "recording_camera_lookat"),
            ("neutral_side_camera_pos", "side_camera_pos"),
            ("neutral_side_camera_lookat", "side_camera_lookat"),
        ):
            val = (self.config or {}).get(cfg_key)
            if val is not None:
                setattr(self, attr, list(val))

    def load_actors(self):
        # Keep the historical RNG order: sample the initial neutral job before
        # the parent task samples/spawns its robot targets.  Once those targets
        # are known, reject or replace a semantically duplicate neutral job.
        self._resolve_neutral_avatar_job()
        super().load_actors()
        self._ensure_neutral_avatar_job_is_target_distinct()
        self._load_neutral_avatar_props()
        self._load_neutral_avatar_region_debug_tiles()

    def create_interaction_controller(self):
        return NeutralTableWorkController()

    def _neutral_controller(self) -> NeutralTableWorkController:
        controller = self.interaction_controller
        if not isinstance(controller, NeutralTableWorkController):
            raise TypeError("task does not use neutral table-work behavior")
        return controller

    def reset(self, seed: int = 0):
        # Per-episode state clearing runs in the controller's on_episode_seeded
        # hook; the post-reset start layout and eval trigger draw run here so
        # every base hook (e.g. the categorize draw-burning) precedes them.
        self._neutral_contested_state = None
        obs = super().reset(seed=seed)
        return self._neutral_controller().after_task_reset(self, obs)

    def take_action(self, action, action_type: str = "qpos"):
        self._neutral_controller().advance_eval_step(self)
        return super().take_action(action, action_type=action_type)

    def _load_neutral_avatar_props(self):
        if self._neutral_avatar_job is None:
            self._resolve_neutral_avatar_job()
        job = self._neutral_avatar_job
        if job["kind"] == "writing":
            self._load_neutral_writing_props(job)
            return
        if job["kind"] == "laptop":
            self._load_neutral_laptop_props(job)
            return
        if job["kind"] == "typing":
            self._load_neutral_typing_props(job)
            return
        if job["kind"] == "sitting_clap":
            self._load_neutral_sitting_clap_props(job)
            return
        work_xy = self._neutral_avatar_default_work_xy(job)
        self.neutral_avatar_props = {}

        for obj_idx, obj_name in enumerate(job["objects"]):
            prop_xy = self._neutral_avatar_prebuild_prop_xy(job, work_xy, obj_name, obj_idx)
            pos = np.array(
                [prop_xy[0], prop_xy[1], self._neutral_avatar_prop_center_z(obj_name)],
                dtype=float,
            )
            self.neutral_avatar_props[obj_name] = self._create_neutral_avatar_prop(
                obj_name, pos
            )
        if job["kind"] == "put_objects_in_bowl" and bool(self.config.get(
            "neutral_put_objects_plate",
            self._neutral_put_objects_clip_defaults().get("use_plate", True),
        )):
            self.neutral_avatar_plate = load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "003_plate" / "visual" / "base0.glb",
                Pose(
                    [float(work_xy[0]), float(work_xy[1]), self.TABLE_TOP_Z + 0.020],
                    self._neutral_avatar_prop_quat("plate"),
                ),
                scale=0.025,
                convex=True,
                is_static=False,
                collision=True,
                friction=3.0,
                density=250.0,
            )
            self.neutral_avatar_bowl = self.neutral_avatar_plate

        first_obj = job["objects"][0]
        self.neutral_avatar_prop = self.neutral_avatar_props[first_obj]
        self.neutral_avatar_target = np.array(
            [
                self._neutral_avatar_default_work_xy(job)[0],
                self._neutral_avatar_default_work_xy(job)[1],
                self.TABLE_TOP_Z,
            ],
            dtype=float,
        )

    def _patch_neutral_avatar_step_sim(self, job: dict, event_frame_offset: int = 0) -> None:
        slow = self._neutral_avatar_motion_slow()
        num_frames = max(1, int(job.get("clip", {}).get("num_frames", 1)))
        scaled_len = max(1, int(num_frames * slow))
        scaled_to_clip = np.round(
            np.linspace(0, num_frames - 1, scaled_len)
        ).astype(int)
        scaled_events = []
        for event in job["events"]:
            e = dict(event)
            clip_frame = int(np.clip(int(e["clip_frame"]), 0, num_frames - 1))
            matches = np.flatnonzero(scaled_to_clip >= clip_frame)
            e["scaled_frame"] = (
                int(matches[0]) if matches.size else scaled_len - 1
            ) + int(event_frame_offset)
            scaled_events.append(e)

        state = {
            "handled": set(),
            "pending_animation_start": False,
            "surface_props": {},
            "resting_surface_props": {},
            "attached_objects": set(),
            "final_pose_restored": False,
            "final_reset_done": False,
        }
        original_step_sim = self.step_sim
        self._neutral_original_step_sim = original_step_sim
        self._neutral_avatar_step_state = state

        def patched_step_sim():
            original_step_sim()
            self._neutral_laptop_hold_lid()
            if "idle_body_pos" in state:
                idle_body = np.asarray(state["idle_body_pos"], dtype=np.float64)
                motion_body = np.asarray(state["motion_body_pos"], dtype=np.float64)
                if (
                    not state.get("pending_animation_start")
                    and not state.get("return_started")
                    and state.get("return_step") is None
                    and self.avatar.robot.action_state == "transition"
                    and len(state["handled"]) == len(scaled_events)
                ):
                    state["return_started"] = True
                    state["return_step"] = 0
                if state.get("pending_animation_start"):
                    total = max(1, int(state.get("ease_total", 1)))
                    step = min(total, int(state.get("ease_step", 0)) + 1)
                    state["ease_step"] = step
                    t = float(step) / float(total)
                    self.avatar.robot.global_trans = (1.0 - t) * idle_body + t * motion_body
                    self.avatar.robot.update()
                elif state.get("return_started"):
                    total = max(1, int(state.get("return_total", 1)))
                    step = min(total, int(state.get("return_step", 0)) + 1)
                    state["return_step"] = step
                    t = float(step) / float(total)
                    self.avatar.robot.global_trans = (1.0 - t) * motion_body + t * idle_body
                    self.avatar.robot.update()
                    if step >= total:
                        state["return_started"] = False
            if state.get("pending_animation_start") and self.avatar.spare():
                state["pending_animation_start"] = False
                self.avatar.robot.global_trans = np.asarray(
                    state.get("motion_body_pos", self.avatar.robot.global_trans),
                    dtype=np.float64,
                )
                self.avatar.robot.update()
                self.avatar.play_animation(job["motion"])
                self._neutral_avatar_apply_motion_frame(job["motion"], 0)
            motion = self.avatar.motion_modules.get(job["motion"])
            at_frame = int(getattr(motion, "at_frame", -1)) if motion is not None else -1
            for idx, event in enumerate(scaled_events):
                if idx in state["handled"] or at_frame < int(event["scaled_frame"]):
                    continue
                hand_id = int(event["hand_id"])
                obj_name = str(event["object"])
                obj = self.neutral_avatar_props[obj_name]
                if event["type"] in ("attach", "pick"):
                    self._neutral_assert_can_upright_before_pick(obj_name, obj)
                    if obj_name in ("sponge", "mouse"):
                        state["resting_surface_props"].pop(obj_name, None)
                        state["surface_props"][obj_name] = (hand_id, event)
                        self._neutral_avatar_snap_prop_to_event_pose(
                            job, obj_name, hand_id, event
                        )
                    else:
                        # Objects that start in the hand (e.g. the phone, init=
                        # "hand") must be seated into the palm BEFORE attaching,
                        # otherwise attach_object_to_hand captures the prop's
                        # table load-pose and it floats away from the hand.
                        if obj_name == "phone" or str(event.get("init")) == "hand":
                            self._neutral_avatar_snap_prop_to_event_pose(
                                job, obj_name, hand_id, event
                            )
                        self.avatar.robot.attach_object_to_hand(hand_id, obj)
                        self.avatar.robot.update()
                    state["attached_objects"].add(obj_name)
                    self._neutral_debug(
                        f"[neutral_avatar] {event['type']} {event['object']} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                elif event["type"] in ("detach", "put"):
                    if obj_name in state["surface_props"]:
                        surface_hand_id, surface_event = state["surface_props"][obj_name]
                        hold_event = (
                            event
                            if "hold_offset" in event or "hold_offset_local" in event
                            else surface_event
                        )
                        pos = self._neutral_avatar_surface_prop_pos(
                            job, obj_name, int(surface_hand_id), hold_event
                        )
                        self._neutral_avatar_snap_prop_to_event_pose(
                            job, obj_name, int(surface_hand_id), hold_event
                        )
                        state["resting_surface_props"][obj_name] = pos[:2].copy()
                        state["surface_props"].pop(obj_name, None)
                    else:
                        self.avatar.robot.detach_object(hand_id)
                        # Rest the released prop just above the tabletop at its
                        # XY: bottles always; put-/take-objects props only in the
                        # reversed (take-out) clip, where the prop is set down on
                        # the bare table (forward put drops into the plate and
                        # must NOT be snapped to table height).
                        rest_on_table = (
                            obj_name == "bottle"
                            or str(obj_name).startswith("bottle_")
                            or (
                                job.get("kind") == "put_objects_in_bowl"
                                and bool((job.get("clip") or {}).get("reverse"))
                                and str(obj_name).startswith("object_")
                            )
                        )
                        if rest_on_table:
                            pos = to_numpy(obj.get_pos()).ravel()[:3].astype(np.float64)
                            pos[2] = self._neutral_avatar_prop_center_z(obj_name)
                            obj.set_pos(pos.astype(float))
                            obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                    self._neutral_zero_entity_velocity(obj)
                    self._neutral_debug(
                        f"[neutral_avatar] {event['type']} {event['object']} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                elif event["type"] == "switch":
                    from_hand_id = int(event.get("from_hand_id", 1 - hand_id))
                    self.avatar.robot.detach_object(from_hand_id)
                    self.avatar.robot.detach_object(hand_id)
                    self._neutral_zero_entity_velocity(obj)
                    self.avatar.robot.attach_object_to_hand(hand_id, obj)
                    self.avatar.robot.update()
                    self._neutral_debug(
                        f"[neutral_avatar] switch {event['object']} "
                        f"hand {from_hand_id}->{hand_id} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                state["handled"].add(idx)
            # Pin put-/take-objects props at their resting spawn pose until the
            # avatar picks them up. Near-spherical props (apples) would roll off
            # the grab spot during the long approach otherwise. Gated to non-can
            # assets so the stable-can forward/reverse motions are untouched.
            if (
                job.get("kind") == "put_objects_in_bowl"
                and self._neutral_put_objects_asset() != "071_can"
            ):
                spawn = getattr(self, "_neutral_put_objects_spawn_pose", None) or {}
                for obj_name, (pin_pos, pin_quat) in spawn.items():
                    if obj_name in state["attached_objects"]:
                        continue
                    obj = self.neutral_avatar_props.get(obj_name)
                    if obj is None:
                        continue
                    obj.set_pos(np.asarray(pin_pos, dtype=float))
                    obj.set_quat(np.asarray(pin_quat, dtype=float))
                    self._neutral_zero_entity_velocity(obj)
            for obj_name, (hand_id, event) in list(state["surface_props"].items()):
                obj = self.neutral_avatar_props[obj_name]
                pos = self._neutral_avatar_surface_prop_pos(
                    job, obj_name, int(hand_id), event
                )
                obj.set_pos(pos.astype(float))
                obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                self._neutral_zero_entity_velocity(obj)
            for obj_name, xy in list(state["resting_surface_props"].items()):
                obj = self.neutral_avatar_props[obj_name]
                obj.set_pos(np.array(
                    [
                        float(xy[0]),
                        float(xy[1]),
                        self._neutral_avatar_prop_center_z(obj_name),
                    ],
                    dtype=float,
                ))
                obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                self._neutral_zero_entity_velocity(obj)
            if (
                getattr(self, "_neutral_avatar_started", False)
                and not state["surface_props"]
                and self.avatar.spare()
            ):
                if (
                    bool(job.get("loop"))
                    and not state.get("return_started")
                    and not state.get("pending_animation_start")
                ):
                    # Rewind for another back-and-forth pass; re-arm events
                    # (the manually-attached writing pen stays attached).
                    loop_motion = self.avatar.motion_modules.get(job["motion"])
                    if loop_motion is not None and getattr(loop_motion, "data", None):
                        loop_motion.at_frame = 0
                        self.avatar.robot.action_state = job["motion"]
                        self.avatar.robot.action_status = ActionStatus.ONGOING
                        state["handled"] = set()
                        state["resting_surface_props"] = {}
                        self._neutral_avatar_apply_motion_frame(job["motion"], 0)
                elif not state.get("final_pose_restored", False):
                    self._neutral_avatar_apply_motion_frame(job["motion"], 0)
                    state["final_pose_restored"] = True

        self.step_sim = patched_step_sim

    def _install_neutral_loop_subclip(self, job: dict) -> None:
        """Wrap ``job["motion"]`` into a forward+reverse ping-pong clip.

        The avatar plays the clip forward then backward, and the patched
        step-sim rewinds it on completion, so the neutral motion runs
        back-and-forth for the whole episode instead of freezing after one
        pass.  Events keep their indices (all in the forward half) and
        re-fire each pass when the rewind clears the handled set.  Applied
        centrally after kind-specific subclip installation, so it composes
        with the mouse/sponge/laptop subclips and the writing/cans clips.
        """
        install_loop_subclip(self, job)

    def _delay_neutral_avatar_motion_start(self, motion_name: str, delay_steps: int) -> int:
        if delay_steps <= 0 or self.avatar is None:
            return 0
        motion = self.avatar.motion_modules.get(motion_name)
        data = getattr(motion, "data", None) if motion is not None else None
        node_data = getattr(motion, "node_data", None) if motion is not None else None
        if motion is None or data is None or len(data) == 0:
            return 0
        delay_steps = int(delay_steps)
        motion.data = [data[0]] * delay_steps + list(data)
        if node_data is not None and len(node_data) > 0:
            motion.node_data = [node_data[0]] * delay_steps + list(node_data)
        return delay_steps

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        if self.avatar_collider is None:
            return None
        return sample_capsule_planner_points(
            self.avatar_collider.current_capsules(),
            inflate_factor=self._AVATAR_INFLATE_FACTOR,
        )

    def _neutral_object_entities(self) -> list[tuple[str, object]]:
        """All neutral-object scene entities (laptop / book / chair / cans …).

        These belong to the human, never to the robot task, so the robot must
        avoid them.  Deduped by entity identity; ``Actor`` wrappers are
        unwrapped to their ``entity``.
        """
        out: list[tuple[str, object]] = []
        seen: set[int] = set()

        def add(name: str, obj) -> None:
            if obj is None:
                return
            ent = getattr(obj, "entity", obj)
            if ent is None or id(ent) in seen:
                return
            seen.add(id(ent))
            out.append((str(name), ent))

        for name, obj in (getattr(self, "neutral_avatar_props", {}) or {}).items():
            add(name, obj)
        add("book", getattr(self, "neutral_avatar_book", None))
        add("chair", getattr(self, "neutral_avatar_chair", None))
        add("plate", getattr(self, "neutral_avatar_plate", None))
        return out

    def _neutral_object_obb(self, ent) -> tuple | None:
        """Tight oriented bounding box of an entity in its current pose.

        Returns ``(center_world, axes, half_extents)`` where ``axes`` columns
        are the object's local frame in world coords.  An OBB (not an AABB)
        matters here because the neutral objects are yawed on the table — the
        laptop sits at ~16–90° — so an axis-aligned box of a rotated laptop
        over-covers a large empty wedge and the robot forearm clips that empty
        corner (false contacts of ~mm depth).  Cached per entity per reset
        (the laptop/book/chair are static and the laptop carries ~55 collision
        meshes, so re-collecting verts every check would be wasteful).
        """
        cache = getattr(self, "_neutral_obb_cache", None)
        if cache is None:
            cache = {}
            self._neutral_obb_cache = cache
        key = id(ent)
        if key in cache:
            return cache[key]
        verts = []
        for link in (getattr(ent, "links", None) or []):
            for geom in (getattr(link, "_geoms", None) or []):
                try:
                    v = to_numpy(geom.get_verts()).reshape(-1, 3)
                except Exception:
                    continue
                if v.size:
                    verts.append(v)
        if not verts:
            cache[key] = None
            return None
        V = np.concatenate(verts, axis=0)
        try:
            R = t3d.quaternions.quat2mat(to_numpy(ent.get_quat()).ravel()[:4])
        except Exception:
            R = np.eye(3, dtype=np.float64)
        local = V @ R  # world -> local coords (R columns are the local axes)
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        half = 0.5 * (hi - lo)
        if not np.all(np.isfinite(half)) or np.any(half <= 1e-5):
            cache[key] = None
            return None
        center_world = R @ (0.5 * (lo + hi))
        obb = (
            center_world.astype(np.float64),
            R.astype(np.float64),
            half.astype(np.float64),
        )
        cache[key] = obb
        return obb

    def _neutral_object_aabb_boxes(self, shrink: float = 0.0) -> list[tuple]:
        """Neutral objects as tight oriented analytic-checker boxes.

        Returns ``(name, center, axes, half_extents)``.  Names are prefixed
        ``neutral_obj:`` so the collision log distinguishes object hits from
        avatar-body hits.  ``shrink`` (m) is subtracted from each half-extent
        — used by the collision detector as a grazing tolerance so a sub-cm
        brush of the thin screen tip is not flagged as a touch.
        """
        boxes = []
        for name, ent in self._neutral_object_entities():
            obb = self._neutral_object_obb(ent)
            if obb is None:
                continue
            center, axes, half = obb
            if shrink:
                half = np.maximum(np.asarray(half, dtype=np.float64) - float(shrink), 1e-3)
            boxes.append((f"neutral_obj:{name}", center, axes, half))
        return boxes

    def _neutral_object_obstacle_points(self) -> np.ndarray | None:
        """Surface-shell point samples on the inflated neutral-object OBBs."""
        res = float(self._AVATAR_OBSTACLE_RES)
        inflate = float(getattr(self, "NEUTRAL_OBJECT_OBSTACLE_INFLATE", 0.01))
        parts = []
        for _name, center, axes, half in self._neutral_object_aabb_boxes():
            h = np.asarray(half, dtype=np.float64) + inflate
            ns = [max(2, int(2.0 * h[i] / res) + 1) for i in range(3)]
            gx, gy, gz = np.meshgrid(
                np.linspace(-h[0], h[0], ns[0]),
                np.linspace(-h[1], h[1], ns[1]),
                np.linspace(-h[2], h[2], ns[2]),
                indexing="ij",
            )
            local = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            # Keep the surface shell only (≥1 coord within res/2 of the face).
            on_shell = (np.abs(np.abs(local) - h).min(axis=1) < res * 0.5)
            if on_shell.any():
                local = local[on_shell]
            world = np.asarray(center, dtype=np.float64) + local @ np.asarray(
                axes, dtype=np.float64
            ).T
            parts.append(world)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _neutral_object_obstacles_enabled(self) -> bool:
        return bool(self.config.get(
            "neutral_object_as_obstacle",
            getattr(self, "NEUTRAL_OBJECT_AS_OBSTACLE", True),
        ))

    def _install_neutral_object_collision_detection(self) -> None:
        """Feed neutral-object AABBs into the analytic collision checker.

        The checker already tests robot collision-geom sample points against
        ``avatar_collider.current_boxes()``; we wrap that method so the
        neutral objects' world AABBs are appended.  A robot↔object overlap
        then trips the same ``avatar_collided`` flag / collision log that
        gates neutral-task success.
        """
        if not self._neutral_object_obstacles_enabled():
            return
        collider = getattr(self, "avatar_collider", None)
        if collider is None or getattr(collider, "_neutral_obj_boxes_wrapped", False):
            return
        orig_current_boxes = collider.current_boxes
        task = self
        tol = float(self.config.get(
            "neutral_object_collision_tolerance",
            getattr(self, "NEUTRAL_OBJECT_COLLISION_TOLERANCE", 0.0),
        ))

        def current_boxes_with_objects():
            boxes = list(orig_current_boxes())
            try:
                boxes.extend(task._neutral_object_aabb_boxes(shrink=tol))
            except Exception:
                pass
            return boxes

        collider.current_boxes = current_boxes_with_objects
        collider._neutral_obj_boxes_wrapped = True

    # Plan around the neutral objects as exact boxes
    # (their checker OBB grown by NEUTRAL_OBJECT_OBSTACLE_INFLATE) instead of
    # 4 cm octree voxels of the shell, which added up to ~2 cm more and closed
    # the gap between a laptop/bottle and the task's slots.  Still a strict
    # superset of the boxes the collision checker tests.  Opt-in per task
    # (blocks, burger and categorize neutral): other task experts
    # replace the planner cloud with their own and never planned around props.
    NEUTRAL_OBJECT_PLANNER_BOXES = False

    def _neutral_object_planner_boxes_enabled(self) -> bool:
        return bool(self.config.get(
            "neutral_object_planner_boxes",
            getattr(self, "NEUTRAL_OBJECT_PLANNER_BOXES", False),
        ))

    def _sync_neutral_object_planner_boxes(self, arm_tag: str) -> bool:
        """Replace this arm's neutral-object boxes in the MPLib world."""
        try:
            from mplib.pymp import Pose as MplibPose
            from mplib.pymp.collision_detection.fcl import Box, CollisionObject
            planner = self.robot.get_arm(arm_tag).planner
            world = planner._mplib.planning_world
        except Exception:
            return False
        names = getattr(self, "_neutral_planner_box_names", None)
        if names is None:
            names = {}
            self._neutral_planner_box_names = names
        for name in names.get(arm_tag, []):
            try:
                world.remove_object(name)
            except Exception:
                pass
        added = []
        inflate = float(getattr(self, "NEUTRAL_OBJECT_OBSTACLE_INFLATE", 0.01))
        try:
            import transforms3d as _t3d
            for i, (label, center, axes, half) in enumerate(self._neutral_object_aabb_boxes()):
                h = np.asarray(half, dtype=np.float64) + inflate
                q = _t3d.quaternions.mat2quat(np.asarray(axes, dtype=np.float64))
                # World frame, like the obstacle cloud and block neighbors.
                pose = MplibPose(p=np.asarray(center, dtype=np.float64), q=q)
                name = f"neutral_obj_box_{arm_tag}_{i}"
                world.add_object(name, CollisionObject(Box(*(2.0 * h)), pose))
                added.append(name)
        except Exception as e:
            self._neutral_debug(f"[neutral_avatar] planner boxes failed: {e}")
            for name in added:
                try:
                    world.remove_object(name)
                except Exception:
                    pass
            names[arm_tag] = []
            return False
        names[arm_tag] = added
        return True

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        parts = []
        if bool(getattr(self, "NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE", True)):
            avatar_pts = self._avatar_obstacle_points()
            if avatar_pts is not None and avatar_pts.size:
                parts.append(avatar_pts)
        if self._neutral_object_obstacles_enabled():
            boxed = (self._neutral_object_planner_boxes_enabled()
                     and self._sync_neutral_object_planner_boxes(arm_tag))
            obj_pts = None if boxed else self._neutral_object_obstacle_points()
            if obj_pts is not None and obj_pts.size:
                parts.append(obj_pts)
        if not parts:
            return
        pts = np.concatenate(parts, axis=0)
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception as e:
            self._neutral_debug(f"[neutral_avatar] obstacle update failed: {e}")

    def move_to_pose(self, pose, arm_tag: str):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super().move_to_pose(pose, arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_seeded(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag, *args, **kwargs):
        # Preserve each task's defaults when optional arguments are omitted.
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag, *args, **kwargs)

    def neutral_before_robot_rollout(self) -> None:
        """Hook for neutral tasks that need setup after avatar start."""
        return None

    def neutral_after_robot_rollout(self, rollout_ok: bool) -> None:
        """Hook for neutral tasks that need cleanup after the parent rollout."""
        return None

    def neutral_parent_success(self) -> bool:
        """Parent task success gate used by the default neutral success check."""
        return bool(super().check_success())

    def _neutral_object_collided(self) -> bool:
        return self._neutral_controller().neutral_object_collided(self)

    def neutral_avatar_success_gate(self) -> bool:
        """Shared avatar-side success gate (see the controller)."""
        return self._neutral_controller().avatar_success_gate(self)

    def play_once(self) -> bool:
        self._neutral_controller().start_scripted_table_work(self)
        self.neutral_before_robot_rollout()
        ok = bool(super().play_once())
        self.neutral_after_robot_rollout(ok)
        return ok

    def check_success(self) -> bool:
        parent_ok = self.neutral_parent_success()
        no_avatar_collision = self.neutral_avatar_success_gate()
        self._neutral_debug(
            f"[neutral_avatar] parent_ok={parent_ok}, "
            f"no_avatar_collision={no_avatar_collision}"
        )
        return bool(parent_ok and no_avatar_collision)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        # Contested-layout flag + chosen anchor, so results split.
        metrics.update(self.neutral_layout_metrics())
        metrics.update(self.neutral_aware_metrics())
        return metrics
