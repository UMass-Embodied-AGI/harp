"""Open microwave for human: avatar approaches, arm grasps handle and opens door."""

import json
import numpy as np
import transforms3d as t3d
import genesis as gs
import genesis.utils.geom as gu
import torch

from ..avatar.utils import ActionStatus, AvatarState
from ..base_task import BaseTask
from ..interaction.eval_mode import TaskDrivenEvalTickController
from ..genesis_compat import (
    prepare_articulated_urdf_for_renderer,
    set_dofs_kp_kv_compat,
)
from ..utils import Pose, Actor, ArticulationActor, load_mesh, to_numpy, ASSETS_PATH
from ..grasp import GraspPose, tcp_to_link_pose


class SceneLayoutError(RuntimeError):
    """Reset produced a scene that is invalid before any policy action."""


def reset_clearance_verdict(min_distance, warn_below, raise_below) -> str:
    """Classify the reset-time robot-avatar clearance: ok / warn / raise."""
    d = float(min_distance)
    if d < float(raise_below):
        return "raise"
    if d < float(warn_below):
        return "warn"
    return "ok"


# Pre-computed handle geometry for PartNet 7221 microwave (in link_0 frame, unscaled)
_MW_HANDLE_CENTROID_LINK0_UNSCALED = np.array([1.2537, -0.5379, 0.0534])
_MW_HANDLE_BAR_AXIS_LINK0 = np.array([0.0, 1.0, 0.0])  # bar runs along Y in link_0
_MW_BODY_BBOX_MIN_Z_UNSCALED = -0.5041


class OpenMicrowave(BaseTask):
    """Avatar plays microwave motion; arm grasps handle and opens door."""

    INSTRUCTION = "open the microwave door for the human carrying a plate"
    use_avatar = True
    table_offset = np.array([0.3, -0.4])
    avatar_init_pos = np.array([0.36, 0.10, 0.0])
    _BASE_AVATAR_INIT_POS = np.array([0.36, 0.10, 0.0])
    recording_camera_pos = [-0.25, -0.58, 1.16]
    recording_camera_lookat = [0.50, -0.22, 0.84]

    _MW_PARTNET_SUBDIR = "7221"
    _MW_SCALE = 0.225
    _PRE_GRASP_DIST = 0.12
    _PRE_GRASP_OUT_DIR_WORLD = np.array([-1.0, 0.0, 0.0])
    _GRASP_APPROACH_DIR_WORLD = np.array([1.0, 0.0, 0.0])
    _PULL_STEP_DIST = 0.025
    _PULL_DELTA_ANGLE = 0.12
    # 0.78: the guarded pull stops at the wrist's arc limit, leaving some
    # episodes at door 0.78-0.85 rad, which is visually wide open; a higher
    # gate would score those as failures.  The separate load-path clearance
    # gate still rejects any truly insufficient opening.
    _SUCCESS_OPEN_ANGLE = 0.78
    # 0.85: stop pulling as soon as the door clears the success gate with
    # margin — every extra pull step delays the retreat while the avatar
    # walks the plate in (late retreats cause plate collisions).
    _PRE_RETREAT_OPEN_ANGLE = 0.85
    _CONTACT_FRICTION = 50.0
    _DOOR_HOLD_KP = 3500.0
    _DOOR_HOLD_KV = 250.0
    _DOOR_CLOSED_QPOS = 0.0
    _MW_XY = (0.50, -0.22)
    _MW_RANDOMIZE_X_RANGE = (-0.15, 0.15)
    _MW_RANDOMIZE_Y_RANGE = (-0.15, 0.15)
    _MW_RANDOMIZE_YAW_RANGE_DEG = (-30.0, 30.0)
    _MW_SEEDED_VARIANTS = (
        (0.00, 0.00, 0.0),
        (0.08, 0.02, 10.0),
        (0.06, 0.00, 0.0),
    )
    _AVATAR_FRAME_RATIO = 5.0
    _PLATE_TWO_HAND_OFFSET = np.array([0.0, 0.0, 0.05])
    _PLATE_TASK_SCALE = 1.5
    # z -0.08: the avatar carries the plate in at door height; retreating
    # UP puts the arm in its path.  Duck under the
    # plate plane while clearing sideways.
    _ARM_SIDE_CLEAR_OFFSET = np.array([-0.24, 0.0, -0.08])
    _ARM_BACK_RETREAT_OFFSET = np.array([-0.05, -0.30, 0.04])
    _ARM_BASE_RETREAT_DIST = 0.35
    _ARM_BASE_RETREAT_Z_OFFSET = 0.04
    _ARM_RETREAT_IK_STEPS = 100
    _ARM_RETREAT_IK_SIM_PER_STEP = 4
    _PLATE_COLLISION_MARGIN = 0.04
    _PLATE_HAND_MAX_DIST = 0.08
    _LOAD_PATH_RADIUS = 0.18
    _LOAD_PATH_Z_MARGIN = 0.16
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 80
    # Eval-mode waiting pose.  The rest pose hung the right forearm against
    # the Franka home wrist (link7) on some avatar/variant combinations, so the
    # waiting human holds the plate at frame 0 of the clip it will play (the
    # pose the scripted expert also starts from).
    EVAL_WAIT_POSE = "clip_frame0"
    # Reset-time robot-avatar clearance gate (eval mode, collision tracking
    # on): warn below RESET_MIN_CLEARANCE, raise SceneLayoutError below
    # RESET_RAISE_CLEARANCE (an overlap that would veto every policy).
    RESET_MIN_CLEARANCE = 0.05
    RESET_RAISE_CLEARANCE = 0.0

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("robot_kwargs", {})
        cfg["robot_kwargs"].setdefault("pos", [0.35, -0.65, 0.75])
        # Anchor for the per-episode base shift: the mount follows the
        # microwave placement delta the same way the avatar does, so every
        # seeded variant sees the same relative geometry.  With a fixed base,
        # failures split by variant — missed handles and mid-arc stalls were
        # reach marginality, not grasp errors.
        self._robot_base_anchor = [float(v) for v in cfg["robot_kwargs"]["pos"]]
        cfg.setdefault("microwave_randomize_xy", True)
        cfg.setdefault("microwave_seeded_variants", True)
        cfg.setdefault("avatar", {})
        cfg["avatar"].setdefault("frame_ratio", self._AVATAR_FRAME_RATIO)
        super().__init__(cfg)
        self._eval_trigger_step = None
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._policy_step_count = 0

    def reset(self, seed: int = 0):
        self._episode_seed = int(seed)
        if (self.config.get("microwave_randomize_xy", True)
                and self.config.get("microwave_seeded_variants", True)):
            dx, dy, _yaw = self._MW_SEEDED_VARIANTS[
                int(seed) % len(self._MW_SEEDED_VARIANTS)
            ]
            anchor = getattr(
                self, "_robot_base_anchor", [0.35, -0.65, 0.75],
            )
            self.config.setdefault("robot_kwargs", {})["pos"] = [
                anchor[0] + float(dx), anchor[1] + float(dy), anchor[2],
            ]
        obs = super().reset(seed)
        self._reset_microwave_door_closed()
        self._policy_step_count = 0
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._reset_avatar_clearance = None
        self._eval_human_started_sim_step = None
        self._eval_human_finished_sim_step = None
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        self._blind_sim_step0 = int(getattr(self, "_interaction_sim_step", 0))
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self.avatar.frame_ratio = self._AVATAR_FRAME_RATIO
            if str(self.config.get(
                    "microwave_eval_wait_pose", self.EVAL_WAIT_POSE)) == "clip_frame0":
                self._seed_avatar_to_clip_frame0("microwave")
            self._attach_plate_to_avatar()
            self._check_reset_clearance()
            lo = int(self.config.get(
                "eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get(
                "eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                hi = lo
            self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        else:
            self._eval_trigger_step = None
        return obs

    def _seed_avatar_to_clip_frame0(self, motion_name: str):
        """Pose the avatar at frame 0 of ``motion_name`` without playing it.

        ``_eval_step_avatar`` later starts the same clip from frame 0, so the
        waiting pose flows straight into the approach.
        """
        avatar = self.avatar
        avatar.play_animation(motion_name)
        mod = avatar.motion_modules.get(motion_name)
        if mod is None:
            return
        robot = avatar.robot
        robot.pose = mod.data[0]
        robot.node_trans = mod.node_data[0]
        robot.global_mat = mod.global_mat
        robot.global_mat_inv = mod.global_mat_inv
        robot.action_state = AvatarState.NO_ACTION
        robot.action_status = ActionStatus.SUCCEED
        avatar.motion_modules.pop(motion_name, None)
        robot.update()
        if self.avatar_collider is not None:
            self.avatar_collider.update()

    def _check_reset_clearance(self):
        """Measure robot-avatar clearance before the first policy action."""
        checker = self.avatar_collision_checker
        if checker is None:
            return None
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        result = checker.check()
        d = float(result.get("min_distance", float("inf")))
        self._reset_avatar_clearance = {
            "min_distance": d,
            "closest_pair": result.get("closest_pair"),
        }
        warn_below = float(self.config.get(
            "microwave_reset_min_clearance", self.RESET_MIN_CLEARANCE))
        raise_below = float(self.config.get(
            "microwave_reset_raise_clearance", self.RESET_RAISE_CLEARANCE))
        verdict = reset_clearance_verdict(d, warn_below, raise_below)
        if verdict != "ok":
            msg = (
                f"[open_microwave] reset robot-avatar clearance {d:.4f} m "
                f"(pair={result.get('closest_pair')}, seed={self._episode_seed}) "
                f"below {warn_below if verdict == 'warn' else raise_below:.3f} m"
            )
            print(msg, flush=True)
            if verdict == "raise":
                raise SceneLayoutError(msg)
        return d

    def _reset_microwave_door_closed(self):
        """PartNet 7221 loads with a nonzero hinge qpos; start every rollout closed."""
        self._door_hold_active = False
        self._door_hold_target = self._DOOR_CLOSED_QPOS
        closed = self._microwave_qpos_with_door(self._DOOR_CLOSED_QPOS)
        try:
            self.microwave.entity.set_qpos(closed)
        except Exception:
            pass
        try:
            self.microwave.entity.set_dofs_position(closed)
        except Exception:
            pass
        try:
            self.microwave.entity.set_dofs_velocity(np.zeros_like(closed))
        except Exception:
            pass

    def _microwave_qpos_with_door(self, angle: float, entity=None) -> np.ndarray:
        """Return a full microwave qpos vector with only the door hinge set."""
        if entity is None:
            entity = self.microwave.entity
        try:
            qpos = to_numpy(entity.get_qpos()).ravel().astype(np.float64)
        except Exception:
            n_qs = int(getattr(entity, "n_qs", 1) or 1)
            qpos = np.zeros((n_qs,), dtype=np.float64)
        if qpos.size == 0:
            qpos = np.zeros((1,), dtype=np.float64)
        qpos[0] = float(angle)
        if qpos.size > 1:
            qpos[1:] = 0.0
        return qpos

    def _resolve_microwave_urdf(self, subdir_name=None):
        """Find the microwave URDF path, trying preferred subdir first."""
        mw_dir = ASSETS_PATH / "objects" / "044_microwave"
        preferred_subdir = subdir_name or self._MW_PARTNET_SUBDIR
        preferred = mw_dir / preferred_subdir / "mobility.urdf"
        if preferred.exists():
            return preferred, preferred_subdir

        for sub in sorted(mw_dir.iterdir()):
            if sub.is_dir():
                urdf = sub / "mobility.urdf"
                if urdf.exists():
                    return urdf, sub.name
        raise FileNotFoundError(f"No microwave URDF found in {mw_dir}")

    def _load_avatar_plate(self, table_top):
        """Load a task-local scaled plate without changing shared model metadata."""
        model_dir = ASSETS_PATH / "objects" / "003_plate"
        mesh_path = model_dir / "visual" / "base0.glb"
        with open(model_dir / "model_data0.json") as f:
            config = json.load(f)
        base_scale = np.asarray(config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if base_scale.size == 1:
            base_scale = np.repeat(base_scale.item(), 3)
        scaled = (base_scale * self._PLATE_TASK_SCALE).tolist()
        plate_config = dict(config)
        plate_config["scale"] = scaled
        entity = load_mesh(
            self.scene,
            mesh_path,
            Pose([0.35, 0.18, table_top + 0.20], [0.707, 0.707, 0.0, 0.0]),
            scale=scaled,
            is_static=False,
            convex=True,
            friction=5.0,
        )
        return Actor(entity, plate_config, "003_plate")

    def create_interaction_controller(self) -> TaskDrivenEvalTickController:
        return TaskDrivenEvalTickController()

    def take_action(self, action, action_type: str = "qpos"):
        """Policy/testbed step: apply robot action and run eval avatar logic."""
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        obs = super().take_action(action, action_type=action_type)
        if bool(self.config.get("eval_mode", False)):
            self.eval_success = bool(
                self._door_open_success()
                and self._load_path_success()
            )
        return obs

    def load_actors(self):
        """Add PartNet microwave to the scene."""
        # Anchor the microwave's measured lowest mesh point directly on the
        # tabletop; any clearance (e.g. +2 cm) leaves the fixed appliance
        # visibly floating in side views.
        table_top = self.TABLE_TOP_Z
        urdf_path, self._mw_subdir = self._resolve_microwave_urdf()
        scale = self._MW_SCALE

        z_root = table_top - _MW_BODY_BBOX_MIN_Z_UNSCALED * scale
        # Keep the microwave near the validated reachable pose, with seeded
        # variants used for reproducible evaluation videos.  The non-seeded
        # mode samples the full 0.3 m x 0.3 m / +/-30 deg envelope above.
        mw_xy = np.array(self._MW_XY, dtype=np.float64)
        if self.config.get("microwave_randomize_xy", True):
            if self.config.get("microwave_seeded_variants", True):
                idx = int(getattr(self, "_episode_seed", 0)) % len(self._MW_SEEDED_VARIANTS)
                dx, dy, yaw_deg = self._MW_SEEDED_VARIANTS[idx]
                mw_xy += np.array([dx, dy], dtype=np.float64)
                mw_yaw = np.deg2rad(float(yaw_deg))
            else:
                mw_xy += np.array([
                    np.random.uniform(*self._MW_RANDOMIZE_X_RANGE),
                    np.random.uniform(*self._MW_RANDOMIZE_Y_RANGE),
                ])
                mw_yaw = np.deg2rad(np.random.uniform(*self._MW_RANDOMIZE_YAW_RANGE_DEG))
        else:
            mw_yaw = 0.0
        self._mw_xy = mw_xy.copy()
        self._mw_yaw = float(mw_yaw)
        self._mw_xy_delta = self._mw_xy - np.asarray(self._MW_XY, dtype=np.float64)
        avatar_pos = self._BASE_AVATAR_INIT_POS.copy()
        avatar_pos[:2] += self._mw_xy_delta
        self.avatar_init_pos = avatar_pos

        self._mw_root_pos = np.array([mw_xy[0], mw_xy[1], z_root])
        # The URDF already contains the upright body rotation internally; yawing
        # the fixed root around world Z keeps it standing while varying heading.
        self._mw_root_quat = t3d.euler.euler2quat(0.0, 0.0, self._mw_yaw, axes="sxyz")

        urdf_path, load_scale, merge_fixed_links = (
            prepare_articulated_urdf_for_renderer(
                urdf_path,
                renderer=self.config.get("renderer", ""),
                scale=scale,
            )
        )
        urdf_kwargs = dict(
            file=str(urdf_path.absolute()),
            pos=tuple(self._mw_root_pos),
            quat=tuple(self._mw_root_quat),
            scale=float(load_scale),
            fixed=True,
            visualization=True,
            collision=True,
        )
        if merge_fixed_links is not None:
            urdf_kwargs["merge_fixed_links"] = merge_fixed_links
        entity = self.scene.add_entity(gs.morphs.URDF(**urdf_kwargs))
        self.microwave = ArticulationActor(entity, {}, "044_microwave")
        try:
            entity.set_friction(self._CONTACT_FRICTION)
        except Exception:
            pass

        self.avatar_plate = self._load_avatar_plate(table_top)
        self._plate_attached = False
        self._robot_plate_collided = False
        self._robot_plate_collision_events = []

    def _get_link0_pose(self):
        """Get link_0 (door) world pose as (pos, quat)."""
        link0 = self.microwave.link_dict["link_0"]
        pos = to_numpy(link0.get_pos()).ravel()[:3]
        quat = to_numpy(link0.get_quat()).ravel()[:4]
        return pos, quat

    def _mw_yaw_matrix(self):
        """World-Z yaw rotation for task-level microwave heading randomization."""
        yaw = float(getattr(self, "_mw_yaw", 0.0))
        return t3d.axangles.axangle2mat([0.0, 0.0, 1.0], yaw)

    def _grasp_approach_world(self):
        approach = self._mw_yaw_matrix() @ np.asarray(
            self._GRASP_APPROACH_DIR_WORLD, dtype=np.float64)
        approach[2] = 0.0
        return approach / (np.linalg.norm(approach) + 1e-8)

    def _pregrasp_out_world(self):
        out_dir = self._mw_yaw_matrix() @ np.asarray(
            self._PRE_GRASP_OUT_DIR_WORLD, dtype=np.float64)
        out_dir[2] = 0.0
        return out_dir / (np.linalg.norm(out_dir) + 1e-8)

    def _grasp_tcp(self):
        """Compute grasp TCP pose at handle, approaching toward the door."""
        pos_link0, q_link0 = self._get_link0_pose()
        R_link0 = gu.quat_to_R(q_link0)

        # Handle position in world frame
        handle_pos = pos_link0 + R_link0 @ (_MW_HANDLE_CENTROID_LINK0_UNSCALED * self._MW_SCALE)

        approach = self._grasp_approach_world()

        # Build TCP frame: Z=approach, X=bar axis projected, Y=cross
        bar_world = R_link0 @ _MW_HANDLE_BAR_AXIS_LINK0
        tcp_z = approach
        tcp_x = bar_world - np.dot(bar_world, tcp_z) * tcp_z
        n = np.linalg.norm(tcp_x)
        tcp_x = tcp_x / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        tcp_y = np.cross(tcp_z, tcp_x)
        tcp_y /= np.linalg.norm(tcp_y) + 1e-8

        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        return Pose(handle_pos, t3d.quaternions.mat2quat(R))

    def _handle_bar_world(self):
        """Current microwave handle bar direction in world frame."""
        _, q_link0 = self._get_link0_pose()
        bar_world = gu.quat_to_R(q_link0) @ _MW_HANDLE_BAR_AXIS_LINK0
        return bar_world / (np.linalg.norm(bar_world) + 1e-8)

    def _pregrasp_tcp(self, grasp_tcp: Pose) -> Pose:
        """Horizontal pre-grasp offset outward from the door/handle."""
        out_dir = self._pregrasp_out_world()
        return Pose(grasp_tcp.p + out_dir * self._PRE_GRASP_DIST, grasp_tcp.q)

    def _door_angle(self):
        """Return the microwave door hinge qpos in radians."""
        qpos = to_numpy(self.microwave.entity.get_qpos()).ravel()
        return float(qpos[0]) if qpos.size else 0.0

    def step_sim(self):
        """One sim tick with task-local plate and door holds before capture."""
        self.scene.step()
        self.SIM_STEP_IDX += 1
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            self.avatar.step()
            if getattr(self, "_plate_attached", False):
                self.avatar.robot.update()
                self._plate_hold_pose()
        if getattr(self, "_plate_attached", False):
            self._update_robot_plate_collision()
        if getattr(self, "_door_hold_active", False):
            try:
                self.microwave.entity.set_qpos(
                    self._microwave_qpos_with_door(self._door_hold_target))
            except Exception:
                pass
        self._notify_interaction_sim_step()
        self._tick_avatar_checks()
        if self.vla_recorder is not None:
            self.vla_recorder.tick(self)
        self.capture_frame()
        if self._record_stride is not None:
            if self._record_tick % self._record_stride == 0:
                self.record_frame()
            self._record_tick += 1

    def _hold_door_at_current_angle(self):
        """Use joint PD to keep the physically-opened door from drifting shut."""
        angle = self._door_angle()
        entity = self.microwave.entity
        self._door_hold_target = angle
        self._door_hold_active = True
        try:
            kp = to_numpy(entity.get_dofs_kp()).copy()
            kv = to_numpy(entity.get_dofs_kv()).copy()
            kp[:] = self._DOOR_HOLD_KP
            kv[:] = self._DOOR_HOLD_KV
            set_dofs_kp_kv_compat(entity, kp=kp, kv=kv)
            entity.set_dofs_position(self._microwave_qpos_with_door(angle))
        except Exception:
            pass

    def _plate_hold_pose(self):
        """Place the plate flat at the avatar's two-hand midpoint."""
        if self.avatar is None or not hasattr(self, "avatar_plate"):
            return
        center, frame = self.avatar.robot._get_two_hand_frame(self._PLATE_TWO_HAND_OFFSET)
        plate_normal = np.array([0.0, 0.0, 1.0])
        plate_x = frame[:, 0] - np.dot(frame[:, 0], plate_normal) * plate_normal
        if np.linalg.norm(plate_x) < 1e-6:
            plate_x = np.array([1.0, 0.0, 0.0])
        plate_x = plate_x / (np.linalg.norm(plate_x) + 1e-8)
        plate_z = np.cross(plate_x, plate_normal)
        plate_z = plate_z / (np.linalg.norm(plate_z) + 1e-8)
        R = np.column_stack([plate_x, plate_normal, plate_z])
        entity = self.avatar_plate.entity
        entity.set_pos(np.asarray(center, dtype=np.float64).astype(float))
        entity.set_quat(t3d.quaternions.mat2quat(R).astype(float))

    def _attach_plate_to_avatar(self):
        """Attach the plate to both hands while preserving its flat pose."""
        if self.avatar is None or not hasattr(self, "avatar_plate"):
            return
        self._plate_hold_pose()
        self.avatar.robot.attach_object_to_two_hands(
            self.avatar_plate.entity,
            self._PLATE_TWO_HAND_OFFSET,
            snap_to_frame=False,
        )
        self.avatar.robot.update()
        self._plate_attached = self.avatar.robot.two_hand_attached_object is not None

    def _plate_hand_distance(self):
        if self.avatar is None or not hasattr(self, "avatar_plate"):
            return None
        try:
            plate_pos = to_numpy(self.avatar_plate.entity.get_pos()).ravel()[:3]
            hand_center, _ = self.avatar.robot._get_two_hand_frame(
                self._PLATE_TWO_HAND_OFFSET)
            return float(np.linalg.norm(plate_pos - np.asarray(hand_center, dtype=np.float64)))
        except Exception:
            return None

    def _avatar_plate_carried_ok(self) -> bool:
        if self.avatar is None or not hasattr(self, "avatar_plate"):
            return False
        attached = bool(getattr(self, "_plate_attached", False))
        two_hand_attached = self.avatar.robot.two_hand_attached_object is not None
        dist = self._plate_hand_distance()
        close_to_hands = dist is not None and dist <= self._PLATE_HAND_MAX_DIST
        return bool(attached and two_hand_attached and close_to_hands)

    def _plate_collision_geometry(self):
        """Return an inflated local plate cylinder used for arm-contact eval."""
        config = getattr(self.avatar_plate, "config", {}) or {}
        extents = np.asarray(config.get("extents", [0.23, 0.04, 0.23]), dtype=np.float64)
        scale = np.asarray(config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.size == 1:
            scale = np.repeat(scale.item(), 3)
        if extents.size != 3 or scale.size != 3:
            return 0.18, 0.03
        world_extents = extents * scale
        radius = 0.5 * max(float(world_extents[0]), float(world_extents[2]))
        half_thickness = 0.5 * float(world_extents[1])
        return radius, half_thickness

    def _robot_plate_sample_points(self, arm_tag="right"):
        """Sample arm link origins plus TCP for carried-plate collision checks."""
        points = []
        try:
            arm = self.robot.get_arm(arm_tag)
        except Exception:
            return points

        try:
            points.append(("tcp", Pose.from_pose7(arm.get_ee_pose()).p))
        except Exception:
            pass
        entity = getattr(arm, "entity", None)
        for link in getattr(entity, "links", []) or []:
            try:
                name = getattr(link, "name", None) or getattr(link, "_name", "link")
                points.append((str(name), to_numpy(link.get_pos()).ravel()[:3]))
            except Exception:
                continue
        return points

    def _update_robot_plate_collision(self):
        """Sticky failure flag if the robot arm intersects the carried plate."""
        if getattr(self, "_robot_plate_collided", False):
            return
        if not hasattr(self, "avatar_plate") or self.robot is None:
            return

        plate_pos = to_numpy(self.avatar_plate.entity.get_pos()).ravel()[:3]
        plate_quat = to_numpy(self.avatar_plate.entity.get_quat()).ravel()[:4]
        R_plate = gu.quat_to_R(plate_quat)
        radius, half_thickness = self._plate_collision_geometry()
        margin = float(self._PLATE_COLLISION_MARGIN)

        best = None
        for name, point in self._robot_plate_sample_points():
            point = np.asarray(point, dtype=np.float64)
            local = R_plate.T @ (point - plate_pos)
            radial = float(np.linalg.norm([local[0], local[2]]))
            normal = float(abs(local[1]))
            radial_clearance = radial - radius
            normal_clearance = normal - half_thickness
            score = max(radial_clearance, normal_clearance)
            if best is None or score < best["clearance"]:
                best = {
                    "name": name,
                    "point": point,
                    "local": local,
                    "clearance": score,
                    "radial_clearance": radial_clearance,
                    "normal_clearance": normal_clearance,
                }
            if radial_clearance <= margin and normal_clearance <= margin:
                event = {
                    "frame": int(self.FRAME_IDX),
                    "sample": name,
                    "point": point.tolist(),
                    "plate_pos": plate_pos.tolist(),
                    "radial_clearance": float(radial_clearance),
                    "normal_clearance": float(normal_clearance),
                }
                self._robot_plate_collision_events.append(event)
                self._robot_plate_collided = True
                return

        if best is not None:
            self._robot_plate_min_clearance = {
                "frame": int(self.FRAME_IDX),
                "sample": best["name"],
                "clearance": float(best["clearance"]),
                "radial_clearance": float(best["radial_clearance"]),
                "normal_clearance": float(best["normal_clearance"]),
            }

    def _microwave_load_opening_pos(self):
        """Approximate center of the loading opening in world coordinates."""
        return np.asarray([
            self._mw_root_pos[0],
            self._mw_root_pos[1],
            self.TABLE_TOP_Z + 0.25,
        ], dtype=np.float64)

    def _robot_blocks_microwave_load_path(self, arm_tag="right") -> bool:
        """Return True if final arm link samples occupy the plate insertion path."""
        if self.avatar is None or not hasattr(self, "avatar_plate"):
            return False
        plate_pos = to_numpy(self.avatar_plate.entity.get_pos()).ravel()[:3]
        opening = self._microwave_load_opening_pos()
        start_xy = plate_pos[:2]
        end_xy = opening[:2]
        seg = end_xy - start_xy
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-8:
            return False

        z_mid = 0.5 * (float(plate_pos[2]) + float(opening[2]))
        z_half = 0.5 * abs(float(plate_pos[2] - opening[2])) + self._LOAD_PATH_Z_MARGIN
        radius = float(self._LOAD_PATH_RADIUS)
        blocked = False
        min_dist = None
        closest = None
        for name, point in self._robot_plate_sample_points(arm_tag):
            point = np.asarray(point, dtype=np.float64)
            t = float(np.dot(point[:2] - start_xy, seg) / seg_len2)
            if t <= 0.02 or t >= 0.98:
                continue
            closest_xy = start_xy + np.clip(t, 0.0, 1.0) * seg
            dist_xy = float(np.linalg.norm(point[:2] - closest_xy))
            z_dist = abs(float(point[2]) - z_mid)
            if min_dist is None or dist_xy < min_dist:
                min_dist = dist_xy
                closest = {
                    "sample": name,
                    "dist_xy": dist_xy,
                    "z_dist": z_dist,
                    "point": point.tolist(),
                }
            if dist_xy <= radius and z_dist <= z_half:
                blocked = True
                closest = {
                    "sample": name,
                    "dist_xy": dist_xy,
                    "z_dist": z_dist,
                    "point": point.tolist(),
                }
                break
        self._robot_load_path_clearance = closest
        return blocked

    def _compute_pull_waypoint(self, arm_tag, delta_angle=None,
                               rotate_orientation=True):
        """Compute the next desired gripper waypoint along the handle arc.

        The microwave articulation is never posed here.  These are only
        end-effector targets; during execution the door must follow through
        gripper-handle contact.  Callers recompute this after every physical
        pull step so the next command tracks the actual hinge angle rather
        than an idealized door state.
        """
        if delta_angle is None:
            delta_angle = self._PULL_DELTA_ANGLE
        tcp_offset = self.robot.get_arm(arm_tag).tcp_offset
        hinge_pos, _ = self._get_link0_pose()

        # Waypoint: rotate the MEASURED TCP (position AND orientation) one
        # delta along the hinge arc.  NOTE: anchoring the command to the
        # door's live frame performs much worse — a door-anchored position
        # releases the pull tension, and door-frame orientation with a
        # 0.20 rad lead breaks the grip into branch-flip plans.  The co-rotating command with the
        # pull-loop guards (arc cap + non-latching probe + long-plan
        # rejection) is the best known configuration; the residual wrist
        # windup at the end of the arc is an open issue.
        current_tcp = Pose.from_pose7(self.robot.get_arm(arm_tag).get_ee_pose())
        current_R = current_tcp.to_matrix()[:3, :3]
        rel = current_tcp.p - hinge_pos
        Rz = t3d.axangles.axangle2mat([0.0, 0.0, -1.0], delta_angle)
        tcp_pos = hinge_pos + Rz @ rel
        # rotate_orientation=False (rescue mode): advance the POSITION
        # along the arc but keep the measured orientation — a small angular
        # slip around the handle bar the jaw gap tolerates, used only when
        # the co-rotating waypoint is IK-infeasible at every resolution.
        if rotate_orientation:
            tcp_quat = t3d.quaternions.mat2quat(Rz @ current_R)
        else:
            tcp_quat = t3d.quaternions.mat2quat(current_R)
        tcp_pose = Pose(tcp_pos, tcp_quat)
        return tcp_to_link_pose(tcp_pose, tcp_offset).to_pose7()

    # Like oil_bottle_recovery, the residual pull stalls are BRANCH-level —
    # the RRT's arrival configuration at the handle decides whether the full
    # arc is feasible.  With the base following the placement delta, the
    # base-frame grasp geometry is (near-)constant, so one calibrated
    # known-good arrival configuration works everywhere.  This configuration
    # gave fast clean successes with no fallback grinding; other arrival
    # branches needed long rescues or (e.g. a contorted j1=2.38 branch)
    # failed.  None disables.
    _REFERENCE_HANDLE_QPOS = (
        0.538615, 0.446751, -0.004745, -2.679618,
        -1.208132, 1.569825, -2.396511,
    )

    def _stage_to_reference_handle(self, arm_tag: str) -> bool:
        """Joint-plan to the calibrated pre-handle branch before grasping."""
        ref = self._REFERENCE_HANDLE_QPOS
        if ref is None:
            return False
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        if not hasattr(planner, "_mplib"):
            return False
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(
                    np.asarray(ref, dtype=np.float64),
                )],
                current_qpos=planner._pad_qpos(
                    np.asarray(arm.get_arm_qpos(), dtype=np.float64),
                ),
                time_step=1 / 250,
                planning_time=10,
                rrt_range=0.3,
                verbose=False,
            )
        except Exception:
            return False
        if not (result and result.get("status") == "Success"):
            print("[microwave] staging plan_qpos failed; unstaged grasp",
                  flush=True)
            return False
        from ..planning.base import PlanResult
        pos = result["position"]
        vel = result.get("velocity", np.zeros_like(pos))
        print(f"[microwave] staging to calibrated handle branch "
              f"({pos.shape[0]} steps)", flush=True)
        self.execute_plan(PlanResult(True, pos, vel), arm_tag)
        return True

    def _execute_handle_grasp(self, grasp_tcp: Pose, arm_tag: str):
        """Move to pre-grasp, then make a straight grasp approach."""
        door_grasp = GraspPose(
            name="door_handle",
            position=grasp_tcp.p,
            quaternion=grasp_tcp.q,
            pre_distance=self._PRE_GRASP_DIST,
            source="manual",
            scale_frame="runtime",
        )
        arm = self.robot.get_arm(arm_tag)
        tcp_world = door_grasp.to_world(Pose([0.0, 0.0, 0.0]), 1.0)
        pre_tcp = door_grasp.pre_grasp_world(Pose([0.0, 0.0, 0.0]), 1.0)
        grasp_link = tcp_to_link_pose(tcp_world, arm.tcp_offset)
        pre_link = tcp_to_link_pose(pre_tcp, arm.tcp_offset)

        current_qpos = arm.get_arm_qpos()
        result_pre = arm.planner.plan_path(current_qpos, pre_link.to_pose7())
        if not result_pre.success:
            return None
        end_pre_qpos = (
            result_pre.position[-1]
            if result_pre.position.size > 0 else current_qpos
        )

        result_grasp = None
        plan_screw = getattr(arm.planner, "plan_screw_path", None)
        if plan_screw is not None:
            result_grasp = plan_screw(end_pre_qpos, grasp_link.to_pose7())
        if result_grasp is None or not result_grasp.success:
            cart_ik = getattr(arm.planner, "solve_ik_cartesian", None)
            if cart_ik is None:
                return None
            result_grasp = cart_ik(
                end_pre_qpos,
                grasp_link.to_pose7(),
                num_waypoints=80,
            )
            if not result_grasp.success:
                return None

        grasp_final_qpos = (
            result_grasp.position[-1]
            if result_grasp.position.size > 0 else end_pre_qpos
        )
        saved = arm.entity.get_qpos()
        full_qpos = arm._build_qpos_with_arm(grasp_final_qpos)
        arm.entity.set_qpos(
            torch.tensor(full_qpos, dtype=torch.float32)
            if not hasattr(full_qpos, "cpu") else full_qpos
        )
        arm.entity.get_links_pos()
        ee_link_pos = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
        fk_err = float(np.linalg.norm(ee_link_pos - grasp_link.p))
        arm.entity.set_qpos(saved)
        arm.entity.get_links_pos()
        if fk_err > 0.03:
            return None

        self.execute_plan(result_pre, arm_tag)
        self.execute_plan(result_grasp, arm_tag)
        arrival_q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        print("[microwave] handle-grasp arrival qpos="
              + np.array2string(arrival_q[:7], precision=6, separator=", "),
              flush=True)
        # One MEASURED pre-close correction.  The fk_err gate above
        # only checks the PLAN endpoint; PD tracking can still leave the TCP
        # off the handle bar (the fingers then close on air and the arc
        # sweeps 1.20 rad with the door never moving).  A short screw
        # nudge back onto the grasp target before closing is a pre-close
        # correction, not a re-grasp.
        measured_tcp = Pose.from_pose7(arm.get_ee_pose())
        exec_err = float(np.linalg.norm(measured_tcp.p - tcp_world.p))
        if exec_err > 0.008:
            corr = None
            if plan_screw is not None:
                try:
                    corr = plan_screw(arm.get_arm_qpos(), grasp_link.to_pose7())
                except Exception:
                    corr = None
            if (corr is None or not getattr(corr, "success", False)
                    or getattr(corr, "position", np.zeros((0,))).shape[0] > 200):
                # The screw correction itself can fail instantly from the
                # executed arrival branch (the fingers then close on air and
                # the arc sweeps with the door never moving).  The correction
                # is ~1-2 cm; a joint-delta-gated
                # mplib IK interpolation covers it when screw refuses.
                try:
                    ik_corr = arm.planner.solve_ik(
                        arm.get_arm_qpos(), grasp_link.to_pose7(), log=False,
                    )
                except Exception:
                    ik_corr = None
                corr = None
                if (ik_corr is not None and getattr(ik_corr, "success", False)
                        and getattr(ik_corr, "position", np.zeros((0,))).shape[0] > 0):
                    start_q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
                    end_q = np.asarray(ik_corr.position[-1], dtype=np.float64)
                    n = min(len(start_q), len(end_q))
                    if float(np.abs(end_q[:n] - start_q[:n]).max()) < 0.5:
                        corr = ik_corr
            if corr is not None and getattr(corr, "success", False):
                print(f"[microwave] pre-close correction {exec_err*1000:.1f} mm",
                      flush=True)
                self.execute_plan(corr, arm_tag)
            else:
                print(f"[microwave] pre-close correction {exec_err*1000:.1f} mm "
                      "UNPLANNABLE (screw + gated IK both refused)", flush=True)
        return grasp_link, pre_link, door_grasp

    def _retreat_arm_after_open(self, arm_tag: str) -> bool:
        """Release the handle, move sideways out, then move back."""
        self.open_gripper(arm_tag, num_steps=80)
        arm = self.robot.get_arm(arm_tag)
        current_tcp = Pose.from_pose7(arm.get_ee_pose())
        side_offset = self._mw_yaw_matrix() @ self._ARM_SIDE_CLEAR_OFFSET
        side_tcp_pos = current_tcp.p + side_offset
        if not self._retreat_to_position_ik(side_tcp_pos, arm_tag):
            return False
        actual_side_tcp = Pose.from_pose7(arm.get_ee_pose())

        back_tcp_pos = actual_side_tcp.p + self._ARM_BACK_RETREAT_OFFSET
        if not self._retreat_to_position_ik(back_tcp_pos, arm_tag):
            return False
        actual_park_tcp = Pose.from_pose7(arm.get_ee_pose())
        base_tcp_pos = self._base_retreat_target(actual_park_tcp.p)
        if not self._retreat_to_position_ik(base_tcp_pos, arm_tag):
            return False
        return not self._robot_blocks_microwave_load_path(arm_tag)

    def _base_retreat_target(self, tcp_pos):
        """Move from the cleared side pose back toward the Franka base."""
        p = np.asarray(tcp_pos, dtype=np.float64).copy()
        try:
            bx, by, _ = self.robot.get_base_pose()
            base_xy = np.array([bx, by], dtype=np.float64)
        except Exception:
            base_xy = np.asarray(self.robot.get_arm("right").origin_pose.p[:2],
                                 dtype=np.float64)
        delta = base_xy - p[:2]
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            p[:2] += delta / norm * min(self._ARM_BASE_RETREAT_DIST, norm)
        p[2] += self._ARM_BASE_RETREAT_Z_OFFSET
        return p

    def _move_tcp_or_genesis(self, tcp_pose: Pose, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        retreat_link = tcp_to_link_pose(tcp_pose, arm.tcp_offset)
        if self.move_and_execute(retreat_link.to_pose7(), arm_tag) is None:
            # Full-pose IK is tight after the door has rotated.  Keep the
            # side goal and first try Genesis IK with the wrist orientation
            # preserved.  That avoids the position-only fallback choosing a
            # wrist sweep that pushes the door closed.
            self.plan_success = True
            if not self._retreat_with_genesis_ik(retreat_link, arm_tag, preserve_quat=True):
                if not self._retreat_with_genesis_ik(retreat_link, arm_tag, preserve_quat=False):
                    return False
        return True

    def _retreat_with_genesis_ik(self, link_pose: Pose, arm_tag: str, preserve_quat=True) -> bool:
        arm = self.robot.get_arm(arm_tag)
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return False
        measured = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=np.float64).ravel()
        try:
            kwargs = {
                "link": arm.ee_link,
                "pos": np.asarray(link_pose.p, dtype=np.float64),
                "init_qpos": measured,
            }
            if preserve_quat:
                kwargs["quat"] = np.asarray(link_pose.q, dtype=np.float64)
            qpos_sol = ik_fn(**kwargs)
        except Exception:
            return False
        if qpos_sol is None:
            return False
        return self._execute_full_qpos_retreat(qpos_sol, arm_tag)

    def _execute_full_qpos_retreat(self, qpos_sol, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        goal_full = np.asarray(to_numpy(qpos_sol), dtype=np.float64).ravel()
        measured = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=np.float64).ravel()
        start_full = getattr(arm, "_cached_target", None)
        if start_full is None:
            start_full = measured
        start_full = np.asarray(start_full, dtype=np.float64).copy()
        target_full = start_full.copy()
        for joint in arm.arm_joints:
            if joint is None:
                continue
            idx = getattr(joint, "dof_idx_local", None)
            if idx is None and hasattr(joint, "get_dof_indices_local"):
                indices = joint.get_dof_indices_local()
                idx = int(indices[0]) if len(indices) else None
            if idx is not None and 0 <= idx < len(target_full) and idx < len(goal_full):
                target_full[idx] = float(goal_full[idx])

        for i in range(1, self._ARM_RETREAT_IK_STEPS + 1):
            t = i / float(self._ARM_RETREAT_IK_STEPS)
            s = t * t * (3.0 - 2.0 * t)
            q = start_full * (1.0 - s) + target_full * s
            arm.entity.control_dofs_position(q)
            self.robot.set_gripper(1.0, arm_tag)
            for _ in range(self._ARM_RETREAT_IK_SIM_PER_STEP):
                self.step_sim()
        arm._cached_target = target_full.copy()
        return not bool(getattr(self, "_robot_plate_collided", False))

    def _retreat_to_position_ik(self, tcp_pos, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        current_tcp = Pose.from_pose7(arm.get_ee_pose())
        link_pos = np.asarray(tcp_pos, dtype=np.float64) - (
            gu.quat_to_R(current_tcp.q) @ arm.tcp_offset.p)
        link_pose = Pose(link_pos, current_tcp.q)
        return self._retreat_with_genesis_ik(link_pose, arm_tag, preserve_quat=False)

    def play_once(self) -> bool:
        """Scripted rollout: avatar approaches, arm grasps handle and opens door."""
        self._max_door_angle = self._door_angle()
        self._door_hold_active = False
        self._door_hold_target = 0.0
        self._robot_plate_collided = False
        self._robot_plate_collision_events = []
        self._robot_plate_min_clearance = None
        self._robot_load_path_clearance = None
        if self.avatar is not None:
            self.avatar.frame_ratio = self._AVATAR_FRAME_RATIO
            self._attach_plate_to_avatar()
            if not self._avatar_plate_carried_ok():
                return False
            self.avatar.play_animation("microwave")
            while not self.avatar.spare():
                self.step_sim()

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.entity.set_friction(self._CONTACT_FRICTION)
        except Exception:
            pass
        grasp_tcp = self._grasp_tcp()

        # Open gripper, move to pre-grasp, then grasp
        self.open_gripper(arm_tag, num_steps=50)

        self._stage_to_reference_handle(arm_tag)
        if self._execute_handle_grasp(grasp_tcp, arm_tag) is None:
            return False

        # The 7221 collision bar physically stops the fingers around the same
        # handle mesh that is rendered, so the bar remains between the pads.
        self.close_gripper(arm_tag, num_steps=100)

        # Pull loop (most pull failures occur at the TERMINAL arc pose):
        # the waypoint co-rotates the full TCP
        # orientation with the arc, and past ~1.30 rad of commanded arc no
        # Panda branch satisfies it within joint limits.  Three guards:
        #   1. cap the cumulative commanded arc at 1.30 rad (successes need
        #      1.20; the infeasibility onset is 1.20-1.32);
        #   2. probe each waypoint with a NON-LATCHING solve_ik before
        #      planning — a hard "IK Failed" inside move_to_pose latches
        #      plan_success=False and fails the episode even when the door
        #      is already past the success angle;
        #   3. reject grotesquely long pull plans (>450 steps): RRT "solves"
        #      the infeasible pose via a branch flip and executes a multi-
        #      second joint unwind with the gripper clamped on the handle,
        #      ripping the door.
        arm = self.robot.get_arm(arm_tag)
        theta_cmd = 0.0
        # 1.40: with the base following the microwave placement, the
        # infeasibility onset measured at a FIXED base (1.20-1.32) does not
        # apply; the screw-first steps + probe stop the pull safely, so
        # allow more arc.
        _MAX_PULL_ARC = 1.40
        _MAX_PULL_PLAN_STEPS = 450
        # 30: half-delta rescue steps mean more loop iterations can be
        # needed to cover the same arc.
        for i in range(30):
            if self._door_angle() >= self._PRE_RETREAT_OPEN_ANGLE:
                break
            if theta_cmd + self._PULL_DELTA_ANGLE > _MAX_PULL_ARC:
                print(f"[microwave] pull arc capped at {theta_cmd:.2f} rad "
                      f"(door {self._door_angle():.2f})", flush=True)
                break
            wp = self._compute_pull_waypoint(arm_tag)
            # Each pull step is a ~3 cm TCP arc, exactly what the
            # branch-continuous plan_screw is for.  An RRT-first loop fails
            # often: mplib's unseeded plan_pose "solves" the small arc step
            # via an 890-1628-step branch unwind, guard #3 then aborts the
            # WHOLE pull — the door can stay at 0.00-0.01 because the very
            # first pull plan flipped.  Screw-first keeps the wrist on the
            # arc; RRT (still probed + step-capped) is only a fallback.
            # When the full step's screw fails, try a HALF step before RRT —
            # a mid-arc screw failure's RRT fallback tends to branch-flip
            # and kill the pull.
            plan = None
            used_delta = self._PULL_DELTA_ANGLE
            planner = arm.planner
            # No position-only rescue steps: they buy only ~0.03 rad before
            # the same wall and can perturb a passing episode into a retreat
            # collision.  The stall is branch-level — addressed by the
            # calibrated staging before the handle grasp instead.
            for delta_try, rotate_o in (
                (self._PULL_DELTA_ANGLE, True),
                (self._PULL_DELTA_ANGLE / 2, True),
            ):
                wp_try = (
                    wp if (delta_try == self._PULL_DELTA_ANGLE and rotate_o)
                    else self._compute_pull_waypoint(
                        arm_tag, delta_angle=delta_try,
                        rotate_orientation=rotate_o,
                    )
                )
                wp_try7 = np.asarray(wp_try, dtype=float).ravel()[:7]
                screw = None
                if hasattr(planner, "_mplib"):
                    # Multi-resolution screw: plan_screw_path
                    # hardcodes qpos_step=0.1; the mixin's carry helper
                    # rescues its stalls by retrying at 0.05/0.02 — same
                    # recipe for the remaining mid-arc pull stalls.
                    from ..planning.base import PlanResult as _PR
                    goal = planner._to_mplib_pose(wp_try7)
                    for qpos_step in (0.1, 0.05, 0.02):
                        try:
                            res = planner._mplib.plan_screw(
                                goal_pose=goal,
                                current_qpos=planner._pad_qpos(
                                    arm.get_arm_qpos(),
                                ),
                                time_step=1 / 250,
                                qpos_step=qpos_step,
                                wrt_world=True,
                                verbose=False,
                            )
                        except Exception:
                            res = None
                        if res and res.get("status") == "Success":
                            pos = res["position"]
                            vel = res.get("velocity", np.zeros_like(pos))
                            screw = _PR(True, pos, vel)
                            break
                else:
                    pull_screw = getattr(planner, "plan_screw_path", None)
                    if pull_screw is not None:
                        try:
                            screw = pull_screw(arm.get_arm_qpos(), wp_try7)
                        except Exception:
                            screw = None
                if screw is not None and getattr(screw, "success", False):
                    plan = screw
                    used_delta = delta_try
                    break
            if plan is None:
                try:
                    probe = arm.planner.solve_ik(
                        arm.get_arm_qpos(),
                        np.asarray(wp, dtype=float).ravel()[:7],
                        log=False,
                    )
                    feasible = bool(getattr(probe, "success", False))
                except Exception:
                    feasible = True  # probe unavailable — let the planner decide
                if not feasible:
                    print(f"[microwave] pull waypoint infeasible at arc "
                          f"{theta_cmd:.2f} rad (door {self._door_angle():.2f}); "
                          "stopping pull", flush=True)
                    break
                plan = self.move_to_pose(wp, arm_tag)
                if plan is None:
                    break
                n_steps = int(getattr(plan, "position", np.zeros((0,))).shape[0])
                if n_steps > _MAX_PULL_PLAN_STEPS:
                    print(f"[microwave] pull plan {n_steps} steps = branch flip; "
                          "stopping pull", flush=True)
                    break
            self.execute_plan(plan, arm_tag)
            theta_cmd += used_delta
            self._max_door_angle = max(self._max_door_angle, self._door_angle())

        for _ in range(60):
            self.step_sim()
        self._hold_door_at_current_angle()
        for _ in range(10):
            self.step_sim()
        opened = self._door_open_success()
        retreated = self._retreat_arm_after_open(arm_tag)
        ok = bool(opened and retreated and self.check_success())
        # Gate attribution: a visibly open door can still fail — the episode
        # can fail on FOUR gates besides the door angle; print them all so failures are attributable from logs.
        print(
            f"[microwave] end: door={self._door_angle():.2f} "
            f"plan_success={self.plan_success} "
            f"plate_collided={bool(getattr(self, '_robot_plate_collided', False))} "
            f"load_path_blocked={self._robot_blocks_microwave_load_path(arm_tag)} "
            f"opened={opened} retreated={retreated} ok={ok}",
            flush=True,
        )
        return ok

    # ------------------------------------------------------------------
    # Blind ("no intent") expert
    # ------------------------------------------------------------------

    def _blind_sim_step(self) -> int:
        """Sim steps since play_blind_once started (monotonic core counter)."""
        return int(getattr(self, "_interaction_sim_step", 0)) - int(
            getattr(self, "_blind_sim_step0", 0))

    def _blind_refresh_planner_obstacles(self, arm_tag: str) -> None:
        """Replace the mplib point cloud with the avatar + plate as observed now.

        Body sampling matches ``human_transfer_planning.refresh_avatar_obstacles``
        (without its capsule-ignore filter) plus the carried plate disc.
        """
        from ..avatar.collision_checker import sample_capsule_planner_points

        collider = getattr(self, "avatar_collider", None)
        parts = []
        if collider is not None:
            body = sample_capsule_planner_points(
                list(collider.current_capsules()), inflate_factor=1.10)
            if body is not None:
                parts.append(np.asarray(body, dtype=np.float64).reshape(-1, 3))
        boxes = getattr(collider, "current_boxes", lambda: [])() if collider else []
        for _name, center, axes, half in boxes:
            coords = [np.linspace(-h - 0.005, h + 0.005,
                                  max(2, int(2 * h / 0.015) + 2))
                      for h in np.asarray(half, dtype=float)]
            grid = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
            parts.append(grid.reshape(-1, 3) @ np.asarray(axes).T + center)
        if self.avatar is not None and hasattr(self, "avatar_plate"):
            try:
                plate_pos = to_numpy(self.avatar_plate.entity.get_pos()).ravel()[:3]
                plate_R = gu.quat_to_R(
                    to_numpy(self.avatar_plate.entity.get_quat()).ravel()[:4])
                radius, _half = self._plate_collision_geometry()
                disc = []
                for r in np.arange(0.0, radius + 1e-6, 0.015):
                    n = max(1, int(2.0 * np.pi * r / 0.015))
                    for th in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
                        # plate local frame: normal is local y (see _plate_hold_pose)
                        disc.append([r * np.cos(th), 0.0, r * np.sin(th)])
                parts.append(np.asarray(disc) @ plate_R.T + plate_pos)
            except Exception:
                pass
        pts = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
        self.robot.get_arm(arm_tag).planner.update_obstacles(pts, resolution=0.015)

    def _blind_metric_human_ready(self) -> bool:
        """Metrics only: has the human finished the approach play_once waits for?"""
        return bool(
            self._eval_avatar_started
            and self.avatar is not None
            and self.avatar.spare()
        )

    def play_blind_once(self) -> bool:
        """Blind expert: play_once's robot script from reset, never waiting on the human.

        The human is driven only by ``_eval_step_avatar`` (random trigger in
        ``EVAL_TRIGGER_STEP_MIN/MAX``).  The robot never plays or reads the
        avatar's animation; the avatar body + carried plate as observed at
        each plan go into the mplib obstacle cloud.
        """
        self._blind_sim_step0 = int(getattr(self, "_interaction_sim_step", 0))
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        self._max_door_angle = self._door_angle()
        self._door_hold_active = False
        self._door_hold_target = 0.0
        self._robot_plate_collided = False
        self._robot_plate_collision_events = []
        self._robot_plate_min_clearance = None
        self._robot_load_path_clearance = None

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.entity.set_friction(self._CONTACT_FRICTION)
        except Exception:
            pass
        grasp_tcp = self._grasp_tcp()

        self.open_gripper(arm_tag, num_steps=50)

        self._blind_refresh_planner_obstacles(arm_tag)
        self._stage_to_reference_handle(arm_tag)
        self._blind_refresh_planner_obstacles(arm_tag)
        if self._execute_handle_grasp(grasp_tcp, arm_tag) is None:
            print("[microwave-blind] handle grasp unplannable", flush=True)
            return False

        self.close_gripper(arm_tag, num_steps=100)

        # Human-dependent step = the door pull.  Metrics only: the robot
        # never branches on this.
        self._blind_robot_interaction_sim_step = self._blind_sim_step()
        self._blind_human_ready_before_robot = self._blind_metric_human_ready()

        # play_once's pull loop plus the observed-avatar obstacle refresh.
        theta_cmd = 0.0
        _MAX_PULL_ARC = 1.40
        _MAX_PULL_PLAN_STEPS = 450
        for i in range(30):
            if self._door_angle() >= self._PRE_RETREAT_OPEN_ANGLE:
                break
            if theta_cmd + self._PULL_DELTA_ANGLE > _MAX_PULL_ARC:
                print(f"[microwave-blind] pull arc capped at {theta_cmd:.2f} rad "
                      f"(door {self._door_angle():.2f})", flush=True)
                break
            self._blind_refresh_planner_obstacles(arm_tag)
            wp = self._compute_pull_waypoint(arm_tag)
            plan = None
            used_delta = self._PULL_DELTA_ANGLE
            planner = arm.planner
            for delta_try, rotate_o in (
                (self._PULL_DELTA_ANGLE, True),
                (self._PULL_DELTA_ANGLE / 2, True),
            ):
                wp_try = (
                    wp if (delta_try == self._PULL_DELTA_ANGLE and rotate_o)
                    else self._compute_pull_waypoint(
                        arm_tag, delta_angle=delta_try,
                        rotate_orientation=rotate_o,
                    )
                )
                wp_try7 = np.asarray(wp_try, dtype=float).ravel()[:7]
                screw = None
                if hasattr(planner, "_mplib"):
                    from ..planning.base import PlanResult as _PR
                    goal = planner._to_mplib_pose(wp_try7)
                    for qpos_step in (0.1, 0.05, 0.02):
                        try:
                            res = planner._mplib.plan_screw(
                                goal_pose=goal,
                                current_qpos=planner._pad_qpos(
                                    arm.get_arm_qpos(),
                                ),
                                time_step=1 / 250,
                                qpos_step=qpos_step,
                                wrt_world=True,
                                verbose=False,
                            )
                        except Exception:
                            res = None
                        if res and res.get("status") == "Success":
                            pos = res["position"]
                            vel = res.get("velocity", np.zeros_like(pos))
                            screw = _PR(True, pos, vel)
                            break
                else:
                    pull_screw = getattr(planner, "plan_screw_path", None)
                    if pull_screw is not None:
                        try:
                            screw = pull_screw(arm.get_arm_qpos(), wp_try7)
                        except Exception:
                            screw = None
                if screw is not None and getattr(screw, "success", False):
                    plan = screw
                    used_delta = delta_try
                    break
            if plan is None:
                try:
                    probe = arm.planner.solve_ik(
                        arm.get_arm_qpos(),
                        np.asarray(wp, dtype=float).ravel()[:7],
                        log=False,
                    )
                    feasible = bool(getattr(probe, "success", False))
                except Exception:
                    feasible = True
                if not feasible:
                    print(f"[microwave-blind] pull waypoint infeasible at arc "
                          f"{theta_cmd:.2f} rad (door {self._door_angle():.2f}); "
                          "stopping pull", flush=True)
                    break
                plan = self.move_to_pose(wp, arm_tag)
                if plan is None:
                    break
                n_steps = int(getattr(plan, "position", np.zeros((0,))).shape[0])
                if n_steps > _MAX_PULL_PLAN_STEPS:
                    print(f"[microwave-blind] pull plan {n_steps} steps = branch "
                          "flip; stopping pull", flush=True)
                    break
            self.execute_plan(plan, arm_tag)
            theta_cmd += used_delta
            self._max_door_angle = max(self._max_door_angle, self._door_angle())

        for _ in range(60):
            self.step_sim()
        self._hold_door_at_current_angle()
        for _ in range(10):
            self.step_sim()
        opened = self._door_open_success()
        retreated = self._retreat_arm_after_open(arm_tag)
        ok = bool(opened and retreated and self.check_success())
        print(
            f"[microwave-blind] end: door={self._door_angle():.2f} "
            f"plan_success={self.plan_success} "
            f"plate_collided={bool(getattr(self, '_robot_plate_collided', False))} "
            f"load_path_blocked={self._robot_blocks_microwave_load_path(arm_tag)} "
            f"opened={opened} retreated={retreated} ok={ok} "
            f"human_started_sim={getattr(self, '_eval_human_started_sim_step', None)} "
            f"human_done_sim={getattr(self, '_eval_human_finished_sim_step', None)} "
            f"robot_pull_sim={self._blind_robot_interaction_sim_step} "
            f"human_ready={self._blind_human_ready_before_robot} "
            f"end_sim={self._blind_sim_step()}",
            flush=True,
        )
        return ok

    def _eval_step_avatar(self):
        """Start the plate-carrying avatar approach at a random policy step."""
        self._policy_step_count += 1
        if (self._eval_avatar_started
                and getattr(self, "_eval_human_finished_sim_step", None) is None
                and self.avatar is not None and self.avatar.spare()):
            # Metrics only: first tick at which the approach animation is done.
            self._eval_human_finished_sim_step = self._blind_sim_step()
        if self._eval_avatar_started or self._eval_avatar_failed:
            return
        if self._eval_trigger_step is None:
            return
        if self._policy_step_count < self._eval_trigger_step:
            return
        if self.avatar is None:
            self._eval_avatar_failed = True
            return
        self._eval_avatar_started = True
        self.avatar.frame_ratio = self._AVATAR_FRAME_RATIO
        if not getattr(self, "_plate_attached", False):
            self._attach_plate_to_avatar()
        if not self._avatar_plate_carried_ok():
            self._eval_avatar_failed = True
            return
        self._eval_human_started_sim_step = self._blind_sim_step()
        try:
            self.avatar.play_animation("microwave")
        except Exception:
            self._eval_avatar_failed = True

    def _door_open_success(self) -> bool:
        door_angle = self._door_angle()
        no_plate_collision = not bool(getattr(self, "_robot_plate_collided", False))
        return bool(
            self.plan_success
            and no_plate_collision
            and door_angle >= self._SUCCESS_OPEN_ANGLE
        )

    def _require_load_path_clear(self) -> bool:
        return bool(self.config.get("open_microwave_require_load_path_clear", True))

    def _load_path_success(self) -> bool:
        return bool(
            not self._require_load_path_clear()
            or not self._robot_blocks_microwave_load_path("right")
        )

    def check_success(self) -> bool:
        return bool(
            self._door_open_success()
            and self._load_path_success()
            and self._avatar_plate_carried_ok()
        )

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics["door_angle"] = float(self._door_angle())
        metrics["max_door_angle"] = float(max(
            self._door_angle(), getattr(self, "_max_door_angle", 0.0)))
        metrics["success_open_angle"] = float(self._SUCCESS_OPEN_ANGLE)
        metrics["robot_plate_collision"] = bool(getattr(self, "_robot_plate_collided", False))
        metrics["robot_plate_collision_events"] = list(
            getattr(self, "_robot_plate_collision_events", []))
        min_clearance = getattr(self, "_robot_plate_min_clearance", None)
        if min_clearance is not None:
            metrics["robot_plate_min_clearance"] = dict(min_clearance)
        metrics["robot_blocks_load_path"] = bool(
            self._robot_blocks_microwave_load_path("right"))
        metrics["avatar_plate_attached"] = bool(getattr(self, "_plate_attached", False))
        metrics["avatar_plate_two_hand_attached"] = bool(
            self.avatar is not None
            and self.avatar.robot.two_hand_attached_object is not None
        )
        plate_hand_dist = self._plate_hand_distance()
        metrics["avatar_plate_hand_distance"] = plate_hand_dist
        metrics["avatar_plate_hand_max_dist"] = float(self._PLATE_HAND_MAX_DIST)
        metrics["avatar_plate_carried_ok"] = bool(self._avatar_plate_carried_ok())
        load_path_clearance = getattr(self, "_robot_load_path_clearance", None)
        if load_path_clearance is not None:
            metrics["robot_load_path_clearance"] = dict(load_path_clearance)
        metrics["microwave_xy"] = np.asarray(
            getattr(self, "_mw_xy", self._MW_XY), dtype=np.float64).tolist()
        metrics["microwave_yaw_deg"] = float(np.rad2deg(
            getattr(self, "_mw_yaw", 0.0)))
        metrics["microwave_randomize_xy"] = bool(
            self.config.get("microwave_randomize_xy", True))
        metrics["microwave_randomize_yaw_range_deg"] = list(
            self._MW_RANDOMIZE_YAW_RANGE_DEG)
        metrics["eval_avatar_trigger_step"] = self._eval_trigger_step
        metrics["eval_avatar_started"] = bool(self._eval_avatar_started)
        metrics["eval_avatar_failed"] = bool(self._eval_avatar_failed)
        metrics["eval_policy_step_count"] = int(self._policy_step_count)
        metrics["reset_avatar_clearance"] = getattr(
            self, "_reset_avatar_clearance", None)
        if bool(self.config.get("eval_mode", False)):
            metrics["blind_intent"] = {
                "trigger_step": self._eval_trigger_step,
                "human_started": bool(self._eval_avatar_started),
                "human_started_sim_step": getattr(
                    self, "_eval_human_started_sim_step", None),
                "human_finished_sim_step": getattr(
                    self, "_eval_human_finished_sim_step", None),
                # Door pull start (gripper closed on the handle).
                "robot_interaction_sim_step": getattr(
                    self, "_blind_robot_interaction_sim_step", None),
                # Full-state play_once waits for avatar.spare() after the
                # "microwave" approach clip; True iff that clip was finished.
                "human_ready_before_robot": getattr(
                    self, "_blind_human_ready_before_robot", None),
                "sim_steps_per_policy_step": int(self.config.get(
                    "eval_parent_rollout_tick_interval", 5)),
            }
        return metrics
