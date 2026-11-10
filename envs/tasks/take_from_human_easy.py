"""Take from human (easy): grasp an object from the static human's hand and place it."""

import json
import numpy as np
import transforms3d as t3d
import genesis as gs
from ..base_task import BaseTask, TargetSpec
from ..interaction import IntentEvalModeController
from ..manipulation import TopDownPickPlaceMixin
from ..genesis_compat import (
    set_dofs_kp_kv_compat,
    update_dofs_force_range_compat,
)
from ..utils import Pose, load_object, create_primitive, to_numpy, ASSETS_PATH
from ..grasp import tcp_to_link_pose
from ..task_bases.human_transfer_planning import (
    refresh_avatar_obstacles, move_seeded,
)


def handoff_grip_ready(finger_q, finger_target, prev_finger_q, lateral_offset,
                       block_margin=None, stall_tol=0.001,
                       center_margin=0.004) -> bool:
    """Eval handoff grip gate: has the robot actually grasped the object?

    The human may only let go once the jaws are closed ON the object: the
    fingers have stopped moving since the previous policy step
    (``stall_tol``) and the object centre lies between the jaw pads along
    the closing axis (``lateral_offset``; pads at about +/- finger
    opening, minus ``center_margin``).  Optionally (``block_margin`` not
    None) every finger must also be held open past its commanded target.
    That test is off by default: against the kinematically held object the
    fingers sink to within ~2 mm of their target, so it is not a reliable
    contact signal.  A close beside the object, or one still in transit,
    keeps the object in the human's hand.
    """
    if finger_q is None or prev_finger_q is None:
        return False
    q = np.asarray(finger_q, dtype=np.float64).ravel()
    prev = np.asarray(prev_finger_q, dtype=np.float64).ravel()
    if q.size == 0 or q.shape != prev.shape:
        return False
    blocked = (block_margin is None
               or bool(np.all(q >= float(finger_target) + float(block_margin))))
    stalled = bool(np.max(np.abs(q - prev)) <= float(stall_tol))
    centered = abs(float(lateral_offset)) <= float(np.mean(q)) - float(center_margin)
    return blocked and stalled and centered


class _TakeLayoutController(IntentEvalModeController):
    def on_base_reset_ready(self, task):
        task._initialize_handover_layout()


class TakeFromHumanEasy(TopDownPickPlaceMixin, BaseTask):
    """Grasp a compact object from the human's hand and place it on the table."""

    INSTRUCTION = "take the {object} from the person and place it on the table"
    OBJECT_SET = "human_transfer_pool"

    use_avatar = True
    SIMULATION_DT = 0.001

    # Avatar base pose (no randomization). The avatar sits at +Y edge of the
    # table, facing -Y. Per-episode offsets are layered on in _sample_avatar_pose.
    AVATAR_BASE_POS = np.array([0.218, 0.768, -0.053])
    AVATAR_BASE_ROT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
    avatar_init_pos = AVATAR_BASE_POS
    avatar_init_rot = AVATAR_BASE_ROT

    table_offset = np.array([0.0, -0.1])

    # VLA primary view: near top-down over the full 0.99 x 1.31 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, -0.095, 2.42],
        "forward": [0.0, -0.005, -1.36],
    }]

    # Recording cam: side/back view with avatar visible.
    recording_camera_pos = [0.9, 1.8, 1.90]
    recording_camera_lookat = [0.0, 0.15, 1.40]

    # Side cam: pulled back along -Y past the table edge so the full table
    # width fits in the frame, looking up-and-across toward the avatar-arm
    # interaction center. Captures avatar's upper body + arm + the whole
    # table surface.
    side_camera_pos = [0.55, -1.10, 1.60]
    side_camera_lookat = [0.0, 0.35, 1.35]

    # Per-episode layout randomization. Avatar slides along the near table edge
    # (X axis only — Y stays close to table) with a modest yaw perturbation;
    # plate lives on the opposite (-Y) side so the arm has clear placing room.
    AVATAR_X_RANGE = (-0.12, 0.12)
    AVATAR_YAW_DEG_RANGE = (-30.0, 30.0)
    PLATE_X_RANGE = (0.15, 0.40)
    PLATE_Y_RANGE = (-0.50, -0.30)
    TABLE_OBJECT_Z_CLEARANCE = 0.02
    PLACE_Z_ABOVE_PLATE = 0.06
    # A full descent to PLACE_Z_ABOVE_PLATE crosses an unstable Franka IK
    # region.  Stop above it, but substantially below the transport height,
    # to reduce impact energy without losing the grasp during descent.
    RELEASE_DESCENT_ABOVE_PLACE = 0.14
    DECOR_REGION = (-0.40, 0.40, -0.65, 0.50)
    DECOR_PLATE_EXCLUSION_RADIUS = 0.18
    DECOR_OBJECT_COUNT = 3
    DECOR_TARGET_SIZE = 0.10
    ARM_MAX_REACH = 0.82
    # Max apple-to-arm-base distance accepted for the handover; beyond this
    # the avatar is stepped toward the base and the delivery replayed.
    HANDOVER_REACH_GUARD = 0.70
    # Where the avatar step-in AIMS (must be meaningfully deeper than the
    # guard): the old step targeted GUARD-0.02, and because the clip replay
    # moves the palm imprecisely the handover landed at 0.70-0.75 m on
    # nearly every seed — right at/over the guard, so the ARM_MAX_REACH
    # clamp still pulled the grasp ~2 cm off the apple and it fell at the
    # exchange (without it, apple episodes failed with dz -0.45..-0.70).
    HANDOVER_STEP_TARGET = 0.60
    PRE_GRASP_HEIGHT = 0.10
    PRE_GRASP_EXTRA_CLEARANCE = 0.05
    MID_WAYPOINT_Y_FROM_BASE = 0.35
    MID_WAYPOINT_Z_FROM_BASE = 0.10
    HELD_APPLE_GRIPPER_VALUE = 0.55
    # apple 0.45: closes of 0.55-0.70 drop the apple at the exchange
    # (post-lift dz -0.45..-0.77); 0.45 lifts it (dz +0.04..+0.07).
    OBJECT_GRIPPER_VALUES = {
        "035_apple": 0.45,
        # 0.35: 0.30 can eject the cube at the EXCHANGE (post-lift dz
        # -0.43/-0.23 — fingers overpower the kinematically-held cube and
        # squeeze it out).  Carry drops come from execute speed, not grip
        # depth (see ``_plan_exec_substeps``).
        "073_rubikscube": 0.35,
    }
    # kp 9000 / kv 250 matches the TopDownPickPlaceMixin grip that carries
    # the same apple in deliver_to_human; at 5000/500 the CARRY loses the
    # apple mid-transport.
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    # 300: at 200 the remaining failures are mid-carry drops of marginal
    # exchange grips.  The cube is rigid and the close
    # target already bounds the squeeze depth; a higher force cap only
    # firms the clamp under transport acceleration.
    FINGER_FORCE_LIMIT = 300.0
    SET_GRIPPER_STEPS = 100
    AVATAR_HAND_SETTLE_STEPS = 50
    GRASP_SETTLE_STEPS = 100
    POST_DETACH_SETTLE_STEPS = 50
    POST_LIFT_GRIP_HOLD_STEPS = 100
    LIFT_HEIGHT_FROM_APPLE = 0.10
    LIFT_CARTESIAN_STEPS = 40
    LONG_CARTESIAN_STEPS = 60
    RETRACT_CARTESIAN_STEPS = 20
    # 25: this task runs dt=0.001, so 10 steps/waypoint would make every Cartesian leg ~4-6x faster (wall-sim) than the other tasks'
    # transports — the PD lagged 0.14-0.21 m behind at the release point and
    # the held object slips/flings during the carry.
    CARTESIAN_SIM_PER_STEP = 25
    RELEASE_SETTLE_STEPS = 50
    FINAL_SETTLE_STEPS = 100
    # Allow post-impact rocking to finish. Quiet objects still return after
    # the same five consecutive samples; the timeout does not relax a gate.
    RELEASE_SETTLE_TIMEOUT_S = 2.0
    # Experimental until the original evaluation groups and impact review pass.
    # The pinned candidate runner opts in; benchmark defaults remain unchanged.
    GUARDED_RELEASE = False
    HAND_APPLE_OFFSET_LOCAL = np.array([0.0, 0.07, 0.03], dtype=np.float64)
    EVAL_APPLE_SUPPORT_HALF_SIZE = (0.04, 0.04, 0.003)
    # Success floor relative to place_pos (= plate top + 6 cm): an object
    # resting on the plate is at about -0.02..-0.04 and one on the table top
    # beside the plate at about -0.05; on the floor it is ~-0.8.
    SUCCESS_DZ_MIN = -0.08
    # Eval handoff: the human lets go once the policy gripper is commanded
    # below this value (1=open, 0=closed) at the object.  Policy outputs are
    # continuous, so the per-object scripted close value is too strict.
    EVAL_HANDOFF_GRIPPER_THRESHOLD = 0.6
    # Blind-expert offer start window in policy steps, used only when
    # eval_avatar_pre_policy is False (see play_blind_once for the choice).
    EVAL_TRIGGER_STEP_MIN = 0
    EVAL_TRIGGER_STEP_MAX = 68

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("collision_ignore_capsule_prefixes", (
            "r_palm",
            "r_thumb",
            "r_index",
            "r_middle",
            "r_ring",
            "r_pinky",
        ))
        super().__init__(cfg)
        # Sampled in load_actors after reset(seed) seeds numpy.
        self.obj_name = None
        self.obj_model_id = 0

    def create_interaction_controller(self) -> IntentEvalModeController:
        return _TakeLayoutController()

    def _refresh_planner_obstacles(self, arm_tag):
        margin = float(getattr(self, "_approach_obstacle_margin", 0.0) or 0.0)
        if margin <= 0.0 or getattr(self, "avatar_collider", None) is None:
            refresh_avatar_obstacles(self, arm_tag)
            return
        self.robot.get_arm(arm_tag).planner.update_obstacles(
            self._dense_body_cloud(margin), resolution=0.015,
        )

    # The transit to the pre-grasp is an RRT through the space
    # in front of the human, and the stock capsule cloud (8 points per 4 cm
    # ring, 1.10x radius) leaves gaps the hand slips through: a common failure
    # is the robot hand/link5 grazing r_forearm / r_upper_arm
    # during that move.  For that one leg the body capsules (hand capsules
    # stay ignored, as in the stock cloud) are sampled densely with an
    # absolute margin; the descent and everything after use the stock cloud.
    APPROACH_OBSTACLE_MARGIN_M = 0.04

    def _dense_body_cloud(self, margin: float) -> np.ndarray:
        ignored = set(self.config.get("collision_ignore_capsules", ()))
        prefixes = tuple(self.config.get("collision_ignore_capsule_prefixes", ()))
        pts = []
        for name, pa, pb, r in self.avatar_collider.current_capsules():
            if name in ignored or name.startswith(prefixes):
                continue
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            seg = pb - pa
            length = float(np.linalg.norm(seg))
            radius = float(r) + margin
            axis = seg / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(axis, tmp)
            u /= np.linalg.norm(u) + 1e-12
            v = np.cross(axis, u)
            n_ring = max(8, int(np.ceil(2.0 * np.pi * radius / 0.015)))
            th = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
            ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
            # cylinder rings, plus hemispherical end caps
            for t in np.linspace(0.0, 1.0, max(2, int(np.ceil(length / 0.015)) + 1)):
                c = pa + t * seg
                pts.append(c + radius * ring)
                pts.append(c[None, :])
            for end, sgn in ((pa, -1.0), (pb, 1.0)):
                for phi in np.linspace(0.0, np.pi / 2, 5)[1:]:
                    pts.append(end + sgn * radius * np.sin(phi) * axis
                               + radius * np.cos(phi) * ring)
        return np.concatenate(pts, axis=0) if pts else np.empty((0, 3))

    # Grasp-consistent pre-grasp.  plan_pose to the pre-grasp
    # picks whichever IK branch its random samples hit; on many pick
    # drops it lands on a wrist-flipped branch (joint 6 ~0.1 rad, at its
    # limit) from which the 15 cm descent re-configures the whole arm and
    # ends 4-9 cm off the cube with joint 5 stuck 0.2-0.6 rad off its command.
    # Solve the grasp IK first (closest to the current arm), then the
    # pre-grasp IK closest to that solution, and joint-plan both legs.
    GRASP_BRANCH_MAX_DQ = 0.8
    STAGED_ARRIVAL_TOL_RAD = 0.1

    def _mplib_ik(self, arm, link7, seed):
        planner = arm.planner
        backend = planner._mplib
        goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(link7)
        try:
            status, q = backend.IK(
                goal_pose=goal,
                start_qpos=planner._pad_qpos(np.asarray(seed, dtype=np.float64)),
                n_init_qpos=20, threshold=0.001, return_closest=True,
            )
        except Exception:
            return None
        if status != "Success" or q is None:
            return None
        return np.asarray(q, dtype=np.float64).ravel()[:arm.n_arm]

    def _plan_qpos_execute(self, arm_tag, q_goal, label) -> bool:
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(np.asarray(q_goal, dtype=np.float64))],
                current_qpos=planner._pad_qpos(
                    np.asarray(arm.get_arm_qpos(), dtype=np.float64)),
                time_step=1 / 250, planning_time=10, rrt_range=0.1, verbose=False,
            )
        except Exception as exc:
            print(f"[take] {label} plan_qpos exc: {exc}", flush=True)
            return False
        if not (result and result.get("status") == "Success"):
            print(f"[take] {label} plan_qpos failed", flush=True)
            return False
        from ..planning.base import PlanResult
        pos = result["position"]
        self.execute_plan(PlanResult(True, pos, result.get("velocity", np.zeros_like(pos))),
                          arm_tag)
        return True

    def _grasp_consistent_pregrasp(self, arm_tag, pre_link7, grasp_link7):
        """Returns the pre-grasp joint goal, or None (use plan_pose)."""
        arm = self.robot.get_arm(arm_tag)
        if not hasattr(getattr(arm, "planner", None), "_mplib"):
            return None
        q_now = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        self._refresh_planner_obstacles(arm_tag)
        q_grasp = self._mplib_ik(arm, grasp_link7, q_now)
        if q_grasp is None:
            return None
        self._approach_obstacle_margin = self.APPROACH_OBSTACLE_MARGIN_M
        self._refresh_planner_obstacles(arm_tag)
        q_pre = self._mplib_ik(arm, pre_link7, q_grasp)
        if q_pre is None:
            self._approach_obstacle_margin = 0.0
            self._refresh_planner_obstacles(arm_tag)
            q_pre = self._mplib_ik(arm, pre_link7, q_grasp)
        if q_pre is None or float(np.max(np.abs(q_pre - q_grasp))) > self.GRASP_BRANCH_MAX_DQ:
            self._approach_obstacle_margin = 0.0
            print("[take] no grasp-consistent pre-grasp branch; plan_pose", flush=True)
            return None
        self._staged_q_grasp = q_grasp
        return q_pre

    def move_to_pose(self, pose, arm_tag):
        self._refresh_planner_obstacles(arm_tag)
        return super().move_to_pose(pose, arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        return move_seeded(self, link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=0.0):
        self._refresh_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag, tcp_yaw=tcp_yaw)

    def _observe_handover(self, arm_tag, phase):
        if self.config.get("take_handover_diagnostics", False):
            from ..task_bases.take_handover_observe import observe_handover
            observe_handover(self, arm_tag, phase)

    def _hand_object_target(self):
        skin = self.avatar.robot.skin
        wrist = to_numpy(skin.get_global_translation("RightHand")[0]).ravel()[:3]
        middle = to_numpy(skin.get_global_translation("RightHandMiddle1")[0]).ravel()[:3]
        _, axes = self.avatar.robot._get_hand_frame(1)
        return (wrist + middle) / 2.0 + axes @ self.HAND_APPLE_OFFSET_LOCAL

    def _body_min_y(self):
        """Front of the torso and legs, excluding arms allowed above the table."""
        return min(
            min(float(a[1]), float(b[1])) - float(radius)
            for name, a, b, radius in self.avatar_collider.current_capsules()
            if name.startswith("torso") or name.endswith(("_thigh", "_shin"))
        )

    def _initialize_handover_layout(self):
        """Translate the handover beyond the fixed table edge during reset.

        Preview the existing avatar reach calibration without stepping physics,
        then move robot base, avatar, objects and targets together. This runs
        after scene-init replay and before the initial observation or actions;
        joint positions, contact scoring and relative geometry are preserved.
        """
        if self.avatar is None or self.avatar_collider is None:
            return
        arm = self.robot.get_arm("right")
        replay = (self.config.get("scene_init") or {}).get("task_attrs") or {}
        saved_offset = replay.get("_handover_layout_offset")
        if saved_offset is not None:
            # Scene-init restores translated objects/avatar but excludes the
            # robot entity. Restore only the missing robot-base translation.
            offset = np.asarray(saved_offset, dtype=float)
        else:
            original_trans = np.asarray(self.avatar.robot.global_trans, dtype=float).copy()
            original_rot = np.asarray(self.avatar.robot.global_rot, dtype=float).copy()
            arm_base = np.asarray(arm.origin_pose.p, dtype=float)
            body_front = float("inf")
            try:
                for attempt in range(4):
                    self.avatar.play_animation("deliver_to_human")
                    body_front = min(body_front, self._body_min_y())
                    while not self.avatar.spare():
                        self.avatar.step()
                        body_front = min(body_front, self._body_min_y())
                    handover = self._hand_object_target()
                    distance = float(np.linalg.norm(handover - arm_base))
                    if distance <= self.HANDOVER_REACH_GUARD or attempt == 3:
                        break
                    toward_base = arm_base[:2] - handover[:2]
                    norm = float(np.linalg.norm(toward_base))
                    if norm < 1e-6:
                        break
                    position = np.asarray(self.avatar.robot.global_trans, dtype=float).copy()
                    position[:2] += toward_base / norm * (distance - self.HANDOVER_STEP_TARGET)
                    self.avatar.reset(position, original_rot)
                table_edge = float(self._table_center_xy[1] + self._table_half_size_xy[1])
                offset = np.array([0.0, max(0.0, table_edge + 0.02 - body_front), 0.0])
                self._handover_body_table_clearance = body_front + offset[1] - table_edge
                print(f"[take-layout] body_front_y={body_front:.4f} "
                      f"table_edge_y={table_edge:.4f} translation_y={offset[1]:.4f} "
                      f"clearance={self._handover_body_table_clearance:.4f}", flush=True)
            finally:
                self.avatar.reset(original_trans, original_rot)
            for actor in (self.apple_actor, self.plate_actor):
                position = to_numpy(actor.entity.get_pos()).ravel()[:3]
                actor.entity.set_pos(position + offset)
            self.apple_pose = Pose(np.asarray(self.apple_pose.p) + offset, self.apple_pose.q)
            self.place_pos = np.asarray(self.place_pos) + offset
            self._plate_xy = np.asarray(self._plate_xy) + offset[:2]
            self.avatar_init_pos = np.asarray(self.avatar_init_pos) + offset
            self.avatar.reset(original_trans + offset, original_rot)
        self._handover_layout_offset = offset
        self.target.position = self.place_pos
        base = np.asarray(arm.origin_pose.p, dtype=float) + offset
        arm.entity.set_pos(base)
        arm.origin_pose.p = base
        planner = arm.planner
        planner._base_pose[:3] = base
        planner._mplib.set_base_pose(planner._to_mplib_pose(arm.origin_pose.to_pose7()))
        self.robot.set_origin_endpose()
        self.avatar_collider.update()

    def _setup_scene(self):
        """Use smaller dt for better collision response during grasping.

        Mirrors BaseTask._setup_scene but with dt=0.001 (vs 0.002) — and
        delegates renderer construction to `_build_renderer` so the
        `renderer: raytracer` config path works here too.
        """
        self._resolve_renderer()
        renderer = self._build_renderer()
        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=self.SIMULATION_DT),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        self.scene = gs.Scene(**scene_kwargs)
        self.scene.add_entity(gs.morphs.Plane(visualization=not self._hide_ground_visuals()))
        self._create_floor()

    def _load_robot(self):
        """Place Franka on the standard-height workstation."""
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.3, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _create_table(self, table_height=0.74):
        dx, dy = self.table_offset

        # Original table footprint; the reset-time coordinated shift keeps
        # the avatar beyond its human-facing edge without shortening it.
        half_x = 0.494
        half_y = 0.654
        center_y = dy
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=(half_x, half_y),
            center_xy=(dx, center_y),
            leg_inset=(half_x - 0.45, half_y - 0.60),
        )

    def _sample_avatar_pose(self):
        """Sample per-episode avatar (pos, rot). Uses the seeded np.random."""
        dx = float(np.random.uniform(*self.AVATAR_X_RANGE))
        yaw_deg = float(np.random.uniform(*self.AVATAR_YAW_DEG_RANGE))
        yaw = np.deg2rad(yaw_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        pos = self.AVATAR_BASE_POS + np.array([dx, 0.0, 0.0])
        rot = rot_z @ self.AVATAR_BASE_ROT
        return pos, rot, dx, yaw_deg

    def _sample_plate_xy(self):
        """Sample per-episode plate XY on the side opposite the avatar."""
        return np.array([
            float(np.random.uniform(*self.PLATE_X_RANGE)),
            float(np.random.uniform(*self.PLATE_Y_RANGE)),
        ])

    def _settle_scene(self, n_steps: int = None):
        if not getattr(self, "_avatar_head_hand_aligned", False):
            self._avatar_head_hand_yaw_deg = self._align_avatar_head_hand_to_xy(
                self.place_pos[:2], hand_id=1, motion_name="deliver_to_human",
            )
            self._avatar_head_hand_aligned = True
        super()._settle_scene(n_steps)

    def load_actors(self):
        self._release_settle_diagnostics = None
        entry = self.resolve_target_object(randomize=True)
        self.obj_name = entry.object_id or entry.key
        self.obj_model_id = int(self.config.get("model_id", entry.model_id))
        self._plate_xy = self._sample_plate_xy()
        self.avatar_init_pos, self.avatar_init_rot, _dx, _yaw_deg = self._sample_avatar_pose()

        table_top = self.TABLE_TOP_Z + self.TABLE_OBJECT_Z_CLEARANCE
        obj_q = np.array([1.0, 0.0, 0.0, 0.0])
        upright_q = np.array([0.707, 0.707, 0.0, 0.0])

        # Apple starts near the avatar's idle palm; attachment binds it to the
        # hand at play_once, so this pose is only an initial placement.
        self.apple_pose = Pose([0.0, 0.768, 0.69], obj_q)
        self.apple_actor = load_object(
            self.scene, self.apple_pose, self.obj_name,
            model_id=self.obj_model_id, convex=True, is_static=False,
            friction=4.0,
        )
        obj_radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        support_half = (max(0.025, 0.8 * obj_radius),) * 2 + (
            self.EVAL_APPLE_SUPPORT_HALF_SIZE[2],
        )
        self._eval_object_support_half_size = support_half
        self._eval_apple_support = create_primitive(
            self.scene, "box",
            Pose(p=[0.0, 0.0, -1.0]),
            size={"half_size": support_half},
            color=(0.0, 0.0, 0.0),
            is_static=True,
            collision=True,
            visual=False,
        )
        self._eval_apple_support_active = False

        plate_x, plate_y = float(self._plate_xy[0]), float(self._plate_xy[1])
        self.plate_actor = load_object(
            self.scene, Pose([plate_x, plate_y, table_top], upright_q), "003_plate",
            model_id=0, convex=True, is_static=False,
        )
        self.place_pos = np.array([
            plate_x, plate_y, table_top + self.PLACE_Z_ABOVE_PLATE,
        ])

        # Evaluation metric: apple→plate lateral distance.  The apple mesh
        # origin is offset ~27 mm from its bbox centre, so route the object
        # position through `_get_object_center_world` instead of the default
        # `entity.get_pos()`.
        self.target = TargetSpec(
            object=self.apple_actor,
            position=self.place_pos,
            label="plate",
            object_pos_fn=lambda: self._get_object_center_world(self.apple_actor),
            success_dist_xy=0.12,
            # z gate: the xy-only check passed cubes falling through the
            # plate's xy column (dz of -200 m).  Anything resting on the
            # plate or table top sits within a few cm of place_pos.
            success_dz_min=float(self.config.get(
                "take_success_dz_min", self.SUCCESS_DZ_MIN)),
        )

        self._clutter_actors = []

    # For top-down grasps on round objects, grip slightly below the equator so
    # contact normals push the object upward into the gripper palm.
    # 0.45: at 0.3 some seeds shed the cube early in the first screw leg —
    # an edge-biased exchange grip with too little pad below the center of
    # mass.  The extra ~4 mm of wrap-under gives upward support without reaching
    # the avatar palm (cube sits ~3 cm above it).
    GRASP_DEPTH_FRACTION = 0.45

    @staticmethod
    def _get_object_radius(object_name: str, model_id: int = 0) -> float:
        """Get approximate object radius from mesh bounding-box metadata."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return float(np.max(extents * scale) / 2)

    @staticmethod
    def _get_object_center_world(actor) -> np.ndarray:
        """Compute world-frame bounding-box center.

        `actor.entity.get_pos()` returns the mesh origin, which for many assets
        is offset from the visual/collision bounding-box center (see the
        `center` field in model_data.json). Applying the actor's current
        rotation to that offset gives the true center in world coordinates.
        """
        pos = to_numpy(actor.entity.get_pos()).astype(float).ravel()[:3]
        quat_wxyz = to_numpy(actor.entity.get_quat()).astype(float).ravel()[:4]
        center_local = np.asarray(actor.config.get("center", [0.0, 0.0, 0.0]), dtype=float)
        scale_raw = actor.config.get("scale", [1.0, 1.0, 1.0])
        scale = (np.array([scale_raw, scale_raw, scale_raw], dtype=float)
                 if np.ndim(scale_raw) == 0
                 else np.asarray(scale_raw, dtype=float))
        R = t3d.quaternions.quat2mat(quat_wxyz)
        return pos + R @ (center_local * scale)

    def _compute_grasp_poses(self, arm_tag: str):
        """Compute pre-grasp and grasp TCP poses for the apple.

        Uses a top-down approach with the grasp depth derived from the object's
        mesh bounding box — no per-object hardcoding needed.
        """
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p)

        # Use the bounding-box center (not mesh origin) — crucial for assets
        # like 035_apple whose mesh origin is offset ~27mm from true center.
        apple_pos = self._get_object_center_world(self.apple_actor)

        # Clamp to arm reach (Franka reach ~0.855m)
        to_apple = apple_pos - arm_base
        dist_3d = np.linalg.norm(to_apple)
        if dist_3d > self.ARM_MAX_REACH:
            grasp_pos = arm_base + to_apple * (self.ARM_MAX_REACH / dist_3d)
        else:
            grasp_pos = apple_pos.copy()

        # Gripper straight down.  NOTE: aligning the jaw yaw to the cube's
        # in-hand face normals makes the cube drop at the exchange itself —
        # the aligned jaws land on the same faces the avatar's fingers
        # occupy.  Keep yaw 0.
        self._handover_grasp_yaw = 0.0
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_down = t3d.quaternions.mat2quat(R_down)

        # Lower TCP below object center by a fraction of the object's radius,
        # so finger pads grip the lower hemisphere (upward contact forces).
        obj_radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        grasp_offset = obj_radius * self.GRASP_DEPTH_FRACTION
        grasp_pos[2] -= grasp_offset

        # Pre-grasp: well above the grasp point for clearance
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += self.PRE_GRASP_HEIGHT

        return Pose(pre_grasp_pos, q_down), Pose(grasp_pos, q_down)

    # ------------------------------------------------------------------
    # IK / teleport / cartesian helpers (adopted from categorize_objects_contribution)
    # ------------------------------------------------------------------

    def execute_plan(self, result, arm_tag: str, gripper_result=None):
        """Pace plan execution to the plan's own time parametrization.

        mplib trajectories are time-stamped at 1/250 s per waypoint, but
        this task steps sim at dt=0.001 (vs the BaseTask default 0.002), so
        the stock one-waypoint-per-sim-step execution plays every plan 4x
        overspeed.  While ``_plan_exec_substeps`` > 1 (set around the
        held-object carry legs), each waypoint is repeated that many times
        so the PD controller tracks the intended velocity profile instead
        of an inertial spike that sheds the grasped object.
        """
        reps = int(getattr(self, "_plan_exec_substeps", 1))
        if (reps > 1 and result is not None
                and getattr(result, "position", np.zeros((0,))).size > 0
                and gripper_result is None):
            from ..planning.base import PlanResult
            position = np.repeat(np.asarray(result.position), reps, axis=0)
            velocity = getattr(result, "velocity", None)
            if velocity is not None and np.asarray(velocity).size > 0:
                velocity = np.repeat(np.asarray(velocity), reps, axis=0) / reps
            result = PlanResult(True, position, velocity)
        return super().execute_plan(result, arm_tag, gripper_result)

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        # yaw 0 means "episode default": the handover grasp's aligned jaw
        # yaw (see _compute_grasp_poses), so lift and carry legs hold the
        # grasp orientation instead of twisting the wrist back to 0.
        if abs(float(yaw)) <= 1e-9:
            yaw = float(getattr(self, "_handover_grasp_yaw", 0.0))
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        if abs(float(yaw)) > 1e-9:
            Rz = t3d.euler.euler2mat(0.0, 0.0, float(yaw), axes="sxyz")
            R_down = Rz @ R_down
        q = t3d.quaternions.mat2quat(R_down)
        return Pose(np.asarray(pos, dtype=float), q)

    def _solve_ik(self, tcp_pose, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        self._refresh_planner_obstacles(arm_tag)
        link_pose = tcp_to_link_pose(tcp_pose, arm.tcp_offset)
        solved = arm.planner.solve_ik(
            arm.get_arm_qpos(), link_pose.to_pose7(), log=False,
        )
        if solved.success and solved.position.size:
            return np.asarray(solved.position[-1], dtype=float)[:arm.n_arm]
        return None

    def _boost_arm_pd(self, arm_tag: str = "right"):
        """Raise arm/finger gains so tracking and contact grasp are stable.

        With default kp=4000 the arm lags the planned trajectory by ~30cm at
        the end-effector near the edge of Franka's reach. kp=20000 brings that
        to millimetre accuracy. Latest Genesis can expose MJCF actuators as
        non-PD-reducible, so fall back to the direct actuator gain/bias API.
        """
        from ..robot.franka_robot import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        finger_indices = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_indices.append(idx)
        if finger_indices:
            arm._finger_dof_indices = finger_indices

        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
        except Exception:
            gain = to_numpy(arm.entity.get_dofs_act_gain()).copy()
            bias0_raw, bias1_raw, bias2_raw = arm.entity.get_dofs_act_bias()
            bias0 = to_numpy(bias0_raw).copy()
            bias1 = to_numpy(bias1_raw).copy()
            bias2 = to_numpy(bias2_raw).copy()
            for j in arm.arm_joints:
                if j is not None:
                    idx = _get_dof_idx(j)
                    if idx is not None:
                        gain[idx] = 20000.0
                        bias0[idx] = 0.0
                        bias1[idx] = -20000.0
                        bias2[idx] = -1000.0
            for idx in finger_indices:
                gain[idx] = self.FINGER_KP
                bias0[idx] = 0.0
                bias1[idx] = -self.FINGER_KP
                bias2[idx] = -self.FINGER_KV
            arm.entity.set_dofs_act_gain(gain)
            arm.entity.set_dofs_act_bias(bias0, bias1, bias2)
            self._set_finger_force_limit(arm, finger_indices)
            return

        for j in arm.arm_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 20000.0
                    kv[idx] = 1000.0
        for idx in finger_indices:
            kp[idx] = self.FINGER_KP
            kv[idx] = self.FINGER_KV
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)
        self._set_finger_force_limit(arm, finger_indices)

    def _set_finger_force_limit(self, arm, finger_indices):
        if not finger_indices:
            return
        update_dofs_force_range_compat(
            arm.entity,
            finger_indices,
            lower_value=-self.FINGER_FORCE_LIMIT,
            upper_value=+self.FINGER_FORCE_LIMIT,
        )

    def _move_cartesian(self, start_pos, end_pos, arm_tag, n_steps=50, sim_per_step=20):
        from ..robot.franka_robot import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        seed_target = base_target.copy()
        for j in arm.arm_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    seed_target[dof_idx] = float(measured_qpos[dof_idx])
        arm._cached_target = seed_target.copy()
        # Branch-jump guard (prevents sudden fast arm rotations): a seeded
        # IK solve can still return a solution on a neighboring branch
        # 1-3 rad away; commanding it as ONE waypoint at the boosted arm PD
        # (kp 20000) is a torque spike that flings the held object
        # (sometimes as a violent ejection).  Skip such waypoints instead of executing them
        # — same gate as factory_kit_packing._guarded_transit.
        _MAX_JOINT_DELTA = 0.30
        arm_dof_idxs = [
            _get_dof_idx(j) for j in arm.arm_joints
            if j is not None and _get_dof_idx(j) is not None
        ]
        prev_q = np.array(
            [seed_target[k] for k in arm_dof_idxs], dtype=float,
        )
        skipped = 0
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pos = np.asarray(start_pos) * (1 - t) + np.asarray(end_pos) * t
            arm_qpos = self._solve_ik(self._top_down_tcp(pos), arm_tag)
            if arm_qpos is None:
                print(f"[take] Cartesian IK failed at step {i}/{n_steps}",
                      flush=True)
                continue
            new_q = np.asarray(arm_qpos, dtype=float).ravel()[:len(arm_dof_idxs)]
            if float(np.max(np.abs(new_q - prev_q))) > _MAX_JOINT_DELTA:
                skipped += 1
                continue
            prev_q = new_q
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is not None:
                    dof_idx = _get_dof_idx(j)
                    if dof_idx is not None:
                        qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            arm._cached_target = qpos_full.copy()
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = qpos_full.copy()
        if skipped:
            print(f"[take] cartesian guard skipped {skipped}/{n_steps} "
                  f"branch-jump waypoints", flush=True)

    # When the branch guard refuses (nearly) every lift waypoint the arm
    # stays at the grasp pose ("skipped 40/40", dz -0.01).  Finish the
    # lift with the paced, branch-continuous screw used by the carry legs
    # instead of failing.
    STALLED_LIFT_FRACTION = 0.5

    def _finish_stalled_lift(self, arm_tag, grasp_pos, lift_pos, arm=None) -> None:
        arm = arm if arm is not None else self.robot.get_arm(arm_tag)
        tcp_z = float(np.asarray(arm.get_ee_pose(), dtype=float).ravel()[2])
        want = float(lift_pos[2]) - float(grasp_pos[2])
        if want <= 0 or tcp_z - float(grasp_pos[2]) >= self.STALLED_LIFT_FRACTION * want:
            return
        print(f"[take] lift stalled at +{tcp_z - float(grasp_pos[2]):.3f}m of "
              f"{want:.3f}m; paced screw lift", flush=True)
        prev = getattr(self, "_plan_exec_substeps", 1)
        self._plan_exec_substeps = 6
        try:
            self._move_screw(np.asarray(lift_pos, dtype=float), arm_tag)
        finally:
            self._plan_exec_substeps = prev

    # ------------------------------------------------------------------
    # Eval-mode hook (single-shot): replicate the avatar setup from
    # play_once — calibrate apple position relative to the avatar's right
    # hand, attach it, and queue the deliver_to_human animation.  After
    # _eval_at_reset returns, the policy starts stepping and the avatar
    # progresses through the animation in the background.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] _eval_at_reset begin", flush=True)
        self._boost_arm_pd("right")
        delayed_offer = not bool(self.config.get("eval_avatar_pre_policy", True))
        if delayed_offer:
            self._eval_place_human_in_reach()
        skin = self.avatar.robot.skin
        wrist = to_numpy(skin.get_global_translation("RightHand")[0]).ravel()[:3]
        mid1 = to_numpy(skin.get_global_translation("RightHandMiddle1")[0]).ravel()[:3]
        palm_center = (wrist + mid1) / 2.0
        _, R_hand = self.avatar.robot._get_hand_frame(1)
        apple_pos = palm_center + R_hand @ self.HAND_APPLE_OFFSET_LOCAL
        self.apple_actor.entity.set_pos(apple_pos.astype(float))
        self.avatar.attach_object_to_hand(self.apple_actor.entity, hand_id=1)
        self._eval_apple_attached_to_avatar = True
        self._eval_handoff_closed_steps = 0
        self._eval_handoff_debug = None
        self._eval_handoff_prev_finger_q = None
        self._eval_handoff_obj_z = None
        self._eval_avatar_retracted = False
        self._eval_trigger_step = None
        self._eval_human_started_sim_step = None
        self._eval_human_finished_sim_step = None
        self._eval_human_retract_sim_step = None
        if delayed_offer:
            # Delayed offer: the human holds the object at rest and starts
            # the delivery at a random policy step (see _eval_at_step).  A
            # dedicated per-seed RNG keeps the global numpy stream intact.
            lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                lo, hi = hi, lo
            rng = np.random.default_rng(
                [int(getattr(self, "_episode_seed", 0)), 0x7A4E]
            )
            self._eval_trigger_step = int(rng.integers(lo, hi + 1))
            return
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] play deliver_to_human", flush=True)
        self.avatar.play_animation("deliver_to_human")
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] _eval_at_reset done", flush=True)

    def _eval_place_human_in_reach(self):
        """Delayed-offer eval only: stand the human where play_once's step-in would.

        play_once steps the human toward the robot (up to three times) when
        the offered object lands beyond ``HANDOVER_REACH_GUARD`` of the arm
        base; without it the offer sits ~0.9 m away, past ``ARM_MAX_REACH``.
        Apply the same stance rule at reset as a kinematic avatar-only
        preview, then reset the human to idle at the chosen stance.
        """
        if self.avatar is None:
            return
        arm_base = np.asarray(self.robot.get_arm("right").origin_pose.p, dtype=float)
        trans = np.asarray(self.avatar.robot.global_trans, dtype=float).copy()
        rot = np.asarray(self.avatar.robot.global_rot, dtype=float).copy()
        start_xy = trans[:2].copy()
        try:
            for attempt in range(4):
                self.avatar.play_animation("deliver_to_human")
                while not self.avatar.spare():
                    self.avatar.step()
                handover = self._hand_object_target()
                dist = float(np.linalg.norm(handover - arm_base))
                if dist <= self.HANDOVER_REACH_GUARD or attempt == 3:
                    break
                to_base = arm_base[:2] - handover[:2]
                norm = float(np.linalg.norm(to_base))
                if norm < 1e-6:
                    break
                trans[:2] += to_base / norm * (dist - self.HANDOVER_STEP_TARGET)
                self.avatar.reset(trans.copy(), rot.copy())
        finally:
            self.avatar.reset(trans.copy(), rot.copy())
            if self.avatar_collider is not None:
                self.avatar_collider.update()
        self._eval_human_stance_shift_xy = (
            trans[:2] - start_xy
        ).tolist()

    def _place_eval_apple_support_and_detach(self):
        """In eval, replace the human-hand hold with a hidden support shelf.

        Policy control makes gripper-based detach timing ambiguous.  Once the
        handoff animation has finished, hold the apple from below with a small
        invisible static box, then release the avatar attachment so the policy
        sees a dynamic apple at the same offered pose.
        """
        support = getattr(self, "_eval_apple_support", None)
        if support is None or self.avatar is None:
            return
        if bool(getattr(self, "_eval_apple_support_active", False)):
            return

        apple_center = self._get_object_center_world(self.apple_actor)
        radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        support_half_z = float(self._eval_object_support_half_size[2])
        support_pos = np.array([
            apple_center[0],
            apple_center[1],
            apple_center[2] - radius - support_half_z,
        ], dtype=np.float64)
        support.set_pos(support_pos)
        self.avatar.robot.detach_object(1)
        self._eval_apple_attached_to_avatar = False
        self._eval_apple_support_active = True
        if self.config.get("debug_eval_handoff", False):
            print(
                "[take_eval] apple detached onto hidden support at "
                f"{support_pos.tolist()}",
                flush=True,
            )


    def _handover_close_value(self) -> float:
        """Per-object close, config-overridable for grip sweeps
        (``handover_close_<object_id>``)."""
        return float(self.config.get(
            f"handover_close_{self.obj_name}",
            self.OBJECT_GRIPPER_VALUES[self.obj_name],
        ))

    # After the handoff the demonstrations retract the human hand once the
    # robot has lifted the object (``play_once`` Phase 5); without it the
    # carry sweeps the object through the parked hand.
    EVAL_RETRACT_LIFT_DZ = 0.03

    def _eval_maybe_retract_after_lift(self):
        z0 = getattr(self, "_eval_handoff_obj_z", None)
        if (
            z0 is None
            or getattr(self, "_eval_avatar_retracted", False)
            or self.avatar is None
            or not bool(self.config.get("eval_retract_after_lift", True))
        ):
            return
        z = float(self._get_object_center_world(self.apple_actor)[2])
        if z - float(z0) < float(self.config.get(
                "eval_retract_lift_dz", self.EVAL_RETRACT_LIFT_DZ)):
            return
        if not self.avatar.spare():
            return
        self.avatar.play_animation_reverse("deliver_to_human")
        self._eval_avatar_retracted = True

    def _eval_finger_qpos(self, arm):
        idx = list(getattr(arm, "_finger_dof_indices", None) or [])
        entity = getattr(arm, "entity", None)
        if not idx or entity is None:
            return None
        q = np.asarray(to_numpy(entity.get_qpos()), dtype=np.float64).ravel()
        return q[idx]

    def _eval_handoff_grip_check(self, arm, ee_pos, apple_pos):
        """Grip gate for the eval handoff (see ``handoff_grip_ready``).

        A command-only gate lets go while the jaws are still in transit
        or have closed beside the object (e.g. TCP 2.3 cm off-centre,
        fingers fully shut through the kinematically held cube); the
        released cube is then squeezed out and dropped.  Robots without
        finger DOF indices keep the command-only gate.
        """
        if not bool(self.config.get("eval_handoff_require_grip", True)):
            return True, None
        fq = self._eval_finger_qpos(arm)
        if fq is None:
            return True, None
        prev = getattr(self, "_eval_handoff_prev_finger_q", None)
        self._eval_handoff_prev_finger_q = fq.copy()
        open_q = float(getattr(arm, "gripper_open", 0.04))
        closed_q = float(getattr(arm, "gripper_closed", 0.0))
        target = closed_q + float(arm.gripper_val) * (open_q - closed_q)
        quat = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[3:7]
        axis = t3d.quaternions.quat2mat(quat)[:, 1]
        lateral = float(np.dot(np.asarray(apple_pos, dtype=np.float64).ravel()[:3]
                               - np.asarray(ee_pos, dtype=np.float64).ravel()[:3], axis))
        ok = handoff_grip_ready(
            fq, target, prev, lateral,
            block_margin=self.config.get("eval_handoff_block_margin"),
            stall_tol=float(self.config.get("eval_handoff_stall_tol", 0.001)),
            center_margin=float(self.config.get("eval_handoff_center_margin", 0.004)),
        )
        return ok, {"finger_q": [float(x) for x in fq], "finger_target": target,
                    "lateral_offset": lateral}

    # "grip" (default): the human lets go only once the
    #     gripper has closed ON the object -- commanded closed, jaws stalled
    #     with the object centred between the pads (``handoff_grip_ready``)
    #     for the scripted close duration, TCP within
    #     EVAL_HANDOFF_CONTACT_DIST of the object -- as in the scripted
    #     expert / demonstrations (close, then the human detaches) and the
    #     take_from_human_safety knife (``_eval_maybe_release_knife``).
    # "contact" (opt-in): the human lets go as soon as the closing jaws
    #     reach the object's surface, so the fingers then squeeze a free
    #     object.  It releases earlier than the demos, and VLA grasps touch
    #     the human's forearm in 50-70% of episodes under it.  If the
    #     contact window is missed it falls back to "grip".
    EVAL_HANDOFF_MODE = "grip"
    EVAL_HANDOFF_CONTACT_SLACK = 0.003
    EVAL_HANDOFF_CONTACT_LATERAL_TOL = 0.012
    EVAL_HANDOFF_CONTACT_SINK = 0.006
    EVAL_HANDOFF_CONTACT_DIST = 0.035

    def _eval_handoff_at_contact(self, arm, ee_pos, apple_pos, ee_apple_dist,
                                 near_threshold, step_idx) -> bool:
        fq = self._eval_finger_qpos(arm)
        contact_dist = float(self.config.get("eval_handoff_contact_dist",
                                             self.EVAL_HANDOFF_CONTACT_DIST))
        if (fq is None or float(arm.gripper_val) >= 0.95
                or ee_apple_dist > min(near_threshold, contact_dist)):
            return False
        quat = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[3:7]
        axis = t3d.quaternions.quat2mat(quat)[:, 1]
        lateral = float(np.dot(np.asarray(apple_pos, dtype=np.float64).ravel()[:3]
                               - np.asarray(ee_pos, dtype=np.float64).ravel()[:3], axis))
        half_w = float(self._get_object_radius(self.obj_name, self.obj_model_id))
        slack = float(self.config.get("eval_handoff_contact_slack",
                                      self.EVAL_HANDOFF_CONTACT_SLACK))
        q_mean = float(np.mean(fq))
        # Both pads AT the object's faces: not still open, and not shut past
        # them (a close on air beside the object, e.g. 8 cm short, must not
        # release the cube).
        sink = float(self.config.get("eval_handoff_contact_sink",
                                     self.EVAL_HANDOFF_CONTACT_SINK))
        if float(np.min(fq)) < half_w - sink:
            return False
        lat_tol = float(self.config.get("eval_handoff_contact_lateral_tol",
                                        self.EVAL_HANDOFF_CONTACT_LATERAL_TOL))
        if q_mean > half_w + slack or abs(lateral) > lat_tol:
            return False
        self._eval_handoff_debug = {
            "step_idx": int(step_idx), "mode": "contact",
            "gripper": float(arm.gripper_val), "ee_apple_dist": ee_apple_dist,
            "finger_q": [float(x) for x in fq], "lateral_offset": lateral,
        }
        self.avatar.robot.detach_object(1)
        self._eval_apple_attached_to_avatar = False
        self._eval_handoff_obj_z = float(np.asarray(apple_pos).ravel()[2])
        return True
    def _eval_trigger_human(self, step_idx: int):
        """Delayed-offer human: offer at the trigger step, retract after release.

        The human starts ``deliver_to_human`` at its trigger step.  Once the
        offer has finished and the human has let go of the object (the
        handoff release below), the human withdraws the arm by playing the
        offer in reverse, as in the scripted expert's handover.
        """
        trigger = getattr(self, "_eval_trigger_step", None)
        if trigger is None:
            return
        now = int(getattr(self, "_interaction_sim_step", 0))
        if getattr(self, "_eval_human_started_sim_step", None) is None:
            if int(step_idx) < int(trigger):
                return
            self.avatar.play_animation("deliver_to_human")
            self._eval_human_started_sim_step = now
            return
        if not self.avatar.spare():
            return
        if getattr(self, "_eval_human_finished_sim_step", None) is None:
            self._eval_human_finished_sim_step = now
        if (getattr(self, "_eval_handoff_debug", None) is not None
                and getattr(self, "_eval_human_retract_sim_step", None) is None):
            self.avatar.play_animation_reverse("deliver_to_human")
            self._eval_human_retract_sim_step = now

    def _eval_at_step(self, step_idx: int):
        self._eval_trigger_human(step_idx)
        if not getattr(self, "_eval_apple_attached_to_avatar", False):
            self._eval_maybe_retract_after_lift()
            return
        arm = self.robot.get_arm("right")
        ee_pos = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        apple_pos = self._get_object_center_world(self.apple_actor)
        ee_apple_dist = float(np.linalg.norm(ee_pos - apple_pos))
        near_threshold = float(self.config.get("eval_handoff_near_threshold", 0.08))
        close_value = self._handover_close_value()
        threshold = max(
            close_value + 1e-3,
            float(self.config.get(
                "eval_handoff_gripper_threshold",
                self.EVAL_HANDOFF_GRIPPER_THRESHOLD,
            )),
        )
        closed = arm.gripper_val <= threshold
        # The finger-based gates need finger DOF readings; robots without them
        # keep the command-only gate below.
        if self._eval_finger_qpos(arm) is not None:
            if (str(self.config.get("eval_handoff_mode", self.EVAL_HANDOFF_MODE)) == "contact"
                    and self._eval_handoff_at_contact(arm, ee_pos, apple_pos, ee_apple_dist,
                                                      near_threshold, step_idx)):
                return
            # Grip gate (or contact mode's fallback when the jaws arrived
            # off-centre and were pulled onto the object while closing):
            # closed and stalled on the object, within a tight TCP-object
            # distance (a close on air 5-8 cm away must not release it).
            near_threshold = min(near_threshold, float(self.config.get(
                "eval_handoff_contact_dist", self.EVAL_HANDOFF_CONTACT_DIST)))
        grip_ok, grip_debug = self._eval_handoff_grip_check(arm, ee_pos, apple_pos)
        if closed and ee_apple_dist <= near_threshold and grip_ok:
            self._eval_handoff_closed_steps = (
                getattr(self, "_eval_handoff_closed_steps", 0) + 1
            )
            action_substeps = int(self.config.get("action_substeps", 25))
            required_closed_steps = int(np.ceil(self.SET_GRIPPER_STEPS / max(1, action_substeps)))
            required_closed_steps = max(
                1,
                int(self.config.get("eval_handoff_closed_steps", required_closed_steps)),
            )
            if self._eval_handoff_closed_steps >= required_closed_steps:
                self._eval_handoff_debug = {
                    "step_idx": int(step_idx),
                    "gripper": float(arm.gripper_val),
                    "ee_pos": ee_pos.tolist(),
                    "apple_pos": np.asarray(apple_pos, dtype=np.float64).ravel()[:3].tolist(),
                    "ee_apple_dist": ee_apple_dist,
                    "closed_steps": int(self._eval_handoff_closed_steps),
                    "required_closed_steps": int(required_closed_steps),
                }
                if grip_debug is not None:
                    self._eval_handoff_debug["grip"] = grip_debug
                self.avatar.robot.detach_object(1)
                self._eval_apple_attached_to_avatar = False
                self._eval_handoff_obj_z = float(np.asarray(apple_pos).ravel()[2])
        else:
            self._eval_handoff_closed_steps = 0
            if self.config.get("debug_eval_handoff", False) and closed:
                print(
                    "[take_eval] closed but not near: "
                    f"step={step_idx} gripper={arm.gripper_val:.3f} "
                    f"ee_apple_dist={ee_apple_dist:.3f}",
                    flush=True,
                )

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not bool(self.config.get("eval_avatar_pre_policy", True))
        ):
            return self.get_obs()
        while not self.avatar.spare():
            self.step_sim()
        # Match the demonstrations: the scripted expert steps the human in
        # until the handover is inside the arm's reach and squares the
        # object's yaw before the robot moves (``_deliver_object_in_reach``).
        # Without this the eval handover sat ~0.87 m from the arm base.
        if (self.robot is not None
                and bool(self.config.get("eval_handover_reach_guard", True))):
            self._deliver_object_in_reach(
                self.robot.get_arm("right"), first_delivery_done=True,
            )
        for _ in range(int(self.config.get(
            "eval_avatar_post_warmup_settle_steps",
            self.AVATAR_HAND_SETTLE_STEPS,
        ))):
            self.step_sim()
        # Default off: the hidden support shelf did not hold the object, so it
        # fell before the policy moved.  The human keeps holding it (as in the
        # demos) until the grasp-triggered release in ``_eval_at_step``.
        if bool(self.config.get("eval_detach_apple_to_support", False)):
            self._place_eval_apple_support_and_detach()
            for _ in range(int(self.config.get("eval_apple_support_settle_steps", 20))):
                self.step_sim()
        return self.get_obs()

    # Success ceiling relative to place_pos (plate top + 6 cm).  The scripted
    # release is from place_pos + 0.14 m and a cube resting on the plate sits
    # at about -0.03, so anything above +0.02 is still in the air or in the
    # gripper.  Without it a policy carrying the cube over the plate's xy
    # column (open-loop replay: dz +0.27..+0.30) scored a success mid-carry.
    SUCCESS_DZ_MAX = 0.02

    def check_success(self) -> bool:
        if not super().check_success():
            return False
        dz_max = self.config.get("take_success_dz_max", self.SUCCESS_DZ_MAX)
        if dz_max is None:
            return True
        metrics = self._compute_target_metrics()
        return metrics is not None and float(metrics["dz"]) <= float(dz_max)

    def evaluate(self) -> dict:
        out = super().evaluate()
        out["object_name"] = self.obj_name
        out["object_model_id"] = int(self.obj_model_id)
        debug = getattr(self, "_eval_handoff_debug", None)
        if debug is not None:
            out["eval_handoff_debug"] = debug
        out["eval_apple_attached_to_avatar"] = bool(
            getattr(self, "_eval_apple_attached_to_avatar", False)
        )
        out["eval_apple_support_active"] = bool(
            getattr(self, "_eval_apple_support_active", False)
        )
        diagnostics = getattr(self, "_release_settle_diagnostics", None)
        if diagnostics is not None:
            out["release_settled"] = bool(self._release_settled)
            out["release_settle_diagnostics"] = diagnostics
        # Only the delayed-offer eval (blind expert) has timing to report;
        # the default VLA eval output stays unchanged.
        if (bool(self.config.get("eval_mode", False))
                and not bool(self.config.get("eval_avatar_pre_policy", True))):
            out["blind_intent"] = {
                "trigger_step": getattr(self, "_eval_trigger_step", None),
                "human_started_sim_step": getattr(
                    self, "_eval_human_started_sim_step", None),
                "human_finished_sim_step": getattr(
                    self, "_eval_human_finished_sim_step", None),
                "robot_interaction_sim_step": getattr(
                    self, "_blind_robot_interaction_sim_step", None),
                "human_ready_before_robot": getattr(
                    self, "_blind_human_ready_before_robot", None),
                "apple_at_interaction": getattr(
                    self, "_blind_apple_at_interaction", None),
                "human_retract_sim_step": getattr(
                    self, "_eval_human_retract_sim_step", None),
                "human_stance_shift_xy": getattr(
                    self, "_eval_human_stance_shift_xy", None),
            }
        return out

    def _deliver_object_in_reach(self, arm, first_delivery_done: bool = False):
        """Deliver the object, step the avatar in until the handover is
        within reach, then canonicalize the presentation yaw.

        Shared by ``play_once`` and the eval warmup (which already played
        the first delivery in ``_eval_at_reset``) so the policy sees the
        same handover geometry as the demonstrations.
        """
        # Deliver, then verify the handover landed inside the arm's
        # dexterous envelope; if not, step the avatar toward the robot
        # base and replay the delivery (the clip-end
        # palm sat ~0.8 m from the base, the reach clamp then aimed the
        # gripper at air and the apple fell to the floor on release).
        arm_base_v = np.array(arm.origin_pose.p)
        for attempt in range(4):
            if not (first_delivery_done and attempt == 0):
                apple_pos = self._hand_object_target()
                self.apple_actor.entity.set_pos(apple_pos.astype(float))
                self.avatar.attach_object_to_hand(
                    self.apple_actor.entity, hand_id=1,
                )
                self.avatar.play_animation("deliver_to_human")
            while not self.avatar.spare():
                self.step_sim()
            handover = self._hand_object_target()
            dist = float(np.linalg.norm(handover - arm_base_v))
            if dist <= self.HANDOVER_REACH_GUARD or attempt == 3:
                if dist > self.HANDOVER_REACH_GUARD:
                    print(f"[take] WARNING handover stays {dist:.2f} m "
                          "from the arm base after avatar steps",
                          flush=True)
                break
            to_base = arm_base_v[:2] - handover[:2]
            norm = float(np.linalg.norm(to_base))
            if norm < 1e-6:
                break
            shift_xy = to_base / norm * (
                dist - self.HANDOVER_STEP_TARGET
            )
            print(f"[take] handover {dist:.2f} m from base; stepping "
                  f"avatar by {np.round(shift_xy, 3).tolist()} and "
                  "replaying the delivery", flush=True)
            self.avatar.robot.detach_object(1)
            trans = np.asarray(
                self.avatar.robot.global_trans, dtype=float,
            ).copy()
            trans[:2] += shift_xy
            rot = np.asarray(
                self.avatar.robot.global_rot, dtype=float,
            ).copy()
            self.avatar.reset(global_trans=trans, global_rot=rot)
            for _ in range(5):
                self.step_sim()
        # Canonicalize the presentation yaw.  The object
        # inherits the avatar hand's per-seed yaw through the delivery
        # animation, so on ~25% of seeds a box arrives CORNER-first and
        # the yaw-0 jaws pinch its edges — the grip survives statically
        # but sheds the object the moment transport accelerates
        # (deterministically, on the same close-handover seeds).
        # While the object is still kinematically held, rotate its
        # nearest face into jaw alignment over a short smooth motion —
        # the human adjusting the object in their hand before offering
        # it.  Init-stage kinematics on a hand-held object, before the
        # robot ever approaches.
        try:
            q0 = to_numpy(
                self.apple_actor.entity.get_quat()
            ).ravel()[:4].astype(np.float64)
            R_obj = t3d.quaternions.quat2mat(q0)
            best = None
            for k in range(3):
                ax = R_obj[:, k]
                horiz = float(np.hypot(ax[0], ax[1]))
                if best is None or horiz > best[0]:
                    best = (horiz, float(np.arctan2(ax[1], ax[0])))
            fold = float((best[1] + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            if abs(fold) > np.deg2rad(8.0):
                print(f"[take] canonicalizing presentation yaw by "
                      f"{-np.rad2deg(fold):.1f} deg", flush=True)
                c0 = self._get_object_center_world(self.apple_actor)
                n_adjust = 20
                for i in range(1, n_adjust + 1):
                    ang = -fold * (i / n_adjust)
                    q_step = t3d.quaternions.qmult(
                        t3d.quaternions.axangle2quat([0, 0, 1], ang), q0,
                    )
                    self.apple_actor.entity.set_quat(
                        np.asarray(q_step, dtype=float),
                    )
                    # keep the CENTER fixed (set_quat rotates about the
                    # mesh origin, which may be offset from the center)
                    c_now = self._get_object_center_world(self.apple_actor)
                    p_now = to_numpy(
                        self.apple_actor.entity.get_pos()
                    ).ravel()[:3]
                    self.apple_actor.entity.set_pos(
                        (p_now + (c0 - c_now)).astype(float),
                    )
                    self.avatar.attach_object_to_hand(
                        self.apple_actor.entity, hand_id=1,
                    )
                    self.step_sim()
        except Exception as exc:
            print(f"[take] presentation-yaw canonicalization skipped: "
                  f"{exc}", flush=True)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._release_settled = False
        self._release_settle_diagnostics = None
        self._boost_arm_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        # A previous episode's feasibility concessions may persist on the
        # instance; start each episode from the class values.
        self.ARM_MAX_REACH = type(self).ARM_MAX_REACH
        self.PRE_GRASP_HEIGHT = type(self).PRE_GRASP_HEIGHT
        # Plan pacing is enabled only around the held-object carry legs; a
        # mid-carry abort must not leave it on for the next episode.
        self._plan_exec_substeps = 1
        self._handover_grasp_yaw = 0.0
        self._staged_q_grasp = None
        self._approach_obstacle_margin = 0.0
        self.PRE_GRASP_EXTRA_CLEARANCE = type(self).PRE_GRASP_EXTRA_CLEARANCE

        if self.avatar is not None:
            self._deliver_object_in_reach(arm)
            # Leave the apple kinematically attached to the human hand while
            # the robot plans its approach. Detaching early would need a
            # support plane under the apple (an obstacle MPLib has to plan
            # around, which destroys grasp accuracy). Instead we detach right
            # before closing the gripper, so the apple only enters physics
            # simulation when the fingers are already in place to catch it.
            for _ in range(self.AVATAR_HAND_SETTLE_STEPS):
                self.step_sim()
        else:
            for _ in range(self.AVATAR_HAND_SETTLE_STEPS):
                self.step_sim()

        # Phase 2: Robot approaches apple in human's hand
        self.open_gripper(arm_tag)

        # Gripper-down orientation for waypoints and placing
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_down = t3d.quaternions.mat2quat(R_down)

        # Mid-height waypoint to extend arm before reaching toward human
        arm_base = np.array(arm.origin_pose.p)
        mid_tcp = Pose(
            arm_base + np.array([
                0.0,
                self.MID_WAYPOINT_Y_FROM_BASE,
                self.MID_WAYPOINT_Z_FROM_BASE,
            ]),
            q_down,
        )
        mid_link = tcp_to_link_pose(mid_tcp, tcp_offset)

        def _ik_feasible(pose7) -> bool:
            """Non-latching feasibility probe: the first
            mplib failure latches plan_success, so screen candidates with a
            bare IK solve before committing to move_and_execute."""
            try:
                probe = arm.planner.solve_ik(
                    arm.get_arm_qpos(), np.asarray(pose7, dtype=float).ravel()[:7],
                    log=False,
                )
                return bool(getattr(probe, "success", False))
            except Exception:
                return True  # probe unavailable — fall through to the planner

        # The sampled avatar pose can leave her extended arm inside the
        # obstacle cloud around the default mid waypoint; try shifted
        # variants before latching a failure.
        mid_candidates = [mid_link]
        for ddx, ddy, ddz in ((-0.10, 0.0, -0.05), (0.10, 0.0, -0.05),
                              (0.0, -0.08, 0.05)):
            alt = Pose(
                arm_base + np.array([
                    ddx,
                    self.MID_WAYPOINT_Y_FROM_BASE + ddy,
                    self.MID_WAYPOINT_Z_FROM_BASE + ddz,
                ]),
                q_down,
            )
            mid_candidates.append(tcp_to_link_pose(alt, tcp_offset))
        mid_choice = next(
            (c for c in mid_candidates if _ik_feasible(c.to_pose7())),
            mid_candidates[0],
        )
        result = self.move_and_execute(mid_choice.to_pose7(), arm_tag)
        if result is None:
            return False

        # Pre-flight BOTH the grasp pose and the actual raised pre-grasp
        # pose.  Order of concessions matters: lower the pre-grasp
        # clearance FIRST (keeping the grasp exactly on the object) and
        # only tighten the reach clamp as a last resort — clamping first
        # pulls the grasp ~2 cm short of the apple and it falls on release
        # (every apple episode).
        cls = type(self)
        candidates = (
            (cls.ARM_MAX_REACH, cls.PRE_GRASP_HEIGHT,
             cls.PRE_GRASP_EXTRA_CLEARANCE),
            (cls.ARM_MAX_REACH, 0.06, 0.02),
            (cls.ARM_MAX_REACH, 0.04, 0.0),
            (0.78, 0.06, 0.02),
            (0.74, 0.05, 0.0),
        )
        for reach, pre_h, extra in candidates:
            self.ARM_MAX_REACH = reach
            self.PRE_GRASP_HEIGHT = pre_h
            self.PRE_GRASP_EXTRA_CLEARANCE = extra
            probe_pre, probe_grasp = self._compute_grasp_poses(arm_tag)
            probe_hi = np.array(probe_pre.p) + np.array([
                0.0, 0.0, self.PRE_GRASP_EXTRA_CLEARANCE,
            ])
            if _ik_feasible(
                tcp_to_link_pose(
                    self._top_down_tcp(probe_hi), tcp_offset
                ).to_pose7()
            ) and _ik_feasible(
                tcp_to_link_pose(probe_grasp, tcp_offset).to_pose7()
            ):
                if (reach, pre_h, extra) != candidates[0]:
                    print(f"[take] pre-flight concession: reach={reach:.2f} "
                          f"pre_grasp={pre_h:.2f}+{extra:.2f}", flush=True)
                break
        else:
            self.ARM_MAX_REACH = cls.ARM_MAX_REACH
            self.PRE_GRASP_HEIGHT = cls.PRE_GRASP_HEIGHT
            self.PRE_GRASP_EXTRA_CLEARANCE = cls.PRE_GRASP_EXTRA_CLEARANCE

        # Move to pre-grasp (15cm above apple) via the motion planner.
        pre_grasp_tcp, _ = self._compute_grasp_poses(arm_tag)
        pre_pos_hi = np.array(pre_grasp_tcp.p) + np.array([
            0.0, 0.0, self.PRE_GRASP_EXTRA_CLEARANCE,
        ])
        pre_link = tcp_to_link_pose(self._top_down_tcp(pre_pos_hi), tcp_offset)
        q_pre = None
        if hasattr(getattr(arm, "planner", None), "_mplib"):
            _, grasp_tcp = self._compute_grasp_poses(arm_tag)
            grasp_link = tcp_to_link_pose(grasp_tcp, tcp_offset)
            q_pre = self._grasp_consistent_pregrasp(
                arm_tag, pre_link.to_pose7(), grasp_link.to_pose7())
        staged = q_pre is not None and self._plan_qpos_execute(arm_tag, q_pre, "pre-grasp")
        if staged:
            # A joint-space leg can end short of its goal (a wrist joint
            # lagging/wedged); re-plan from the measured state once.
            off = float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=np.float64) - q_pre)))
            if off > self.STAGED_ARRIVAL_TOL_RAD:
                print(f"[take] pre-grasp arrival off by {off:.2f} rad; re-planning", flush=True)
                self._plan_qpos_execute(arm_tag, q_pre, "pre-grasp retry")
        self._approach_obstacle_margin = 0.0
        if not staged and self.move_and_execute(pre_link.to_pose7(), arm_tag) is None:
            return False

        # Final descent via MPLib. With the support plane removed (apple is
        # still kinematically attached to the human hand), MPLib can plan
        # directly to the grasp pose without detouring around an obstacle.
        _, grasp_tcp = self._compute_grasp_poses(arm_tag)
        grasp_link = tcp_to_link_pose(grasp_tcp, tcp_offset)
        descended = False
        if staged:
            self._refresh_planner_obstacles(arm_tag)
            arm_q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
            q_grasp = self._mplib_ik(arm, grasp_link.to_pose7(), arm_q)
            if q_grasp is None or float(np.max(np.abs(q_grasp - arm_q))) > self.GRASP_BRANCH_MAX_DQ:
                q_grasp = getattr(self, "_staged_q_grasp", None)  # solved with the pre-grasp
            if (q_grasp is not None
                    and float(np.max(np.abs(q_grasp - arm_q))) <= self.GRASP_BRANCH_MAX_DQ):
                descended = self._plan_qpos_execute(arm_tag, q_grasp, "grasp descent")
            else:
                print("[take] staged descent: no nearby grasp IK "
                      f"({'none' if q_grasp is None else round(float(np.max(np.abs(q_grasp - arm_q))), 2)}); "
                      "plan_pose", flush=True)
        if not descended and self.move_and_execute(grasp_link.to_pose7(), arm_tag) is None:
            return False
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()
        apple_pre = to_numpy(self.apple_actor.entity.get_pos())
        self._observe_handover(arm_tag, "before_close")

        # Close the gripper while the apple is still kinematically held by the
        # human hand. The fingers encounter the apple as a fixed obstacle and
        # stop at its convex-hull surface (~joint=37mm), building up ~150N per
        # finger of PD contact force. Then detach the apple — it transitions
        # to physics already pinned between the fingers, so the grip holds
        # without the apple free-falling out from between the fingers.
        self.set_gripper(
            self._handover_close_value(),
            arm_tag,
            num_steps=self.SET_GRIPPER_STEPS,
        )
        self._observe_handover(arm_tag, "after_close_attached")
        if self.avatar is not None:
            self.avatar.robot.detach_object(1)
        for _ in range(self.POST_DETACH_SETTLE_STEPS):
            self.step_sim()
        self._observe_handover(arm_tag, "after_detach_settle")

        # Phase 4: Cartesian lift — slow motion preserves finger grip during
        # the transition from static close to moving grip. _move_cartesian
        # does per-step IK + control_dofs_position (no set_qpos).
        grasp_pos = np.array(grasp_tcp.p)
        lift_pos = grasp_pos.copy()
        lift_pos[2] = apple_pre[2] + self.LIFT_HEIGHT_FROM_APPLE
        self._move_cartesian(
            grasp_pos, lift_pos, arm_tag,
            n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )
        self._finish_stalled_lift(arm_tag, grasp_pos, lift_pos, arm=arm)
        object_lifted = to_numpy(self.apple_actor.entity.get_pos())
        self._observe_handover(arm_tag, "after_lift")
        lift_dz = float(object_lifted[2] - apple_pre[2])
        print(
            f"[take] {self.obj_name} post-lift dz={lift_dz:.4f}m "
            f"pos={np.asarray(object_lifted).round(4).tolist()}",
            flush=True,
        )
        if lift_dz < 0.03:
            return False
        # Let the grip stabilize at the lifted pose before the first
        # transport leg — the exchange grip is freshest right after
        # detach, and starting the carry mid-transient sheds the cube on
        # marginal seeds.
        hold_q = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        for _ in range(self.POST_LIFT_GRIP_HOLD_STEPS):
            arm.entity.control_dofs_position(hold_q)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()
        self._observe_handover(arm_tag, "after_lift_hold")

        # Lateral transport over a mid-waypoint (y = arm_base[1] + 0.35) to
        # keep the shoulder from twisting through its joint limit. Larger
        # n_steps here = slower velocity = less inertial slip at the fingers.
        # Clamp the carry height to the mixin's transport band
        # (TABLE+0.38).  A carry at apple-height+0.10 sits 54-65 cm above
        # the table near the Franka's reach/elevation boundary where
        # top-down IK branches are densest, and loses objects on this first
        # lateral leg.
        carry_z = min(float(lift_pos[2]), self.TABLE_TOP_Z + 0.38)
        mid_pos = np.array([
            0.0, arm_base[1] + self.MID_WAYPOINT_Y_FROM_BASE, carry_z,
        ])
        # The three carry legs use the mixin's `_move_screw` (mplib
        # plan_screw is branch-continuous by construction; its fallback is
        # the guarded seeded move).  A bespoke Cartesian transport hits IK
        # branch walls — apples shed on the descending first leg and the
        # rubik's release descent can have all 60 waypoints refused by the
        # guard.
        # Pace the carry plans to their own parametrization.  Screw plans
        # are time-stamped at 1/250 s per waypoint, but this task steps sim
        # at dt=0.001, so one-waypoint-per-step execution runs every
        # held-object leg 4x overspeed (the same issue
        # CARTESIAN_SIM_PER_STEP handles for `_move_cartesian`), and the
        # inertial spikes shed the cube mid-traverse.
        # 6 substeps: 4 matches the plan's own 1/250 pacing; 1.5x slower
        # still cuts the inertial load on marginal grips.
        # Phase 5 (before the transport): the human retracts their hand
        # while the robot holds the lifted object.  Otherwise, on close
        # handovers (0.80-0.82 m) the first transport leg sweeps the cube
        # through the avatar hand still parked at the exchange pose,
        # knocking it out toward the human.
        if self.avatar is not None:
            self.avatar.play_animation_reverse("deliver_to_human")
            while not self.avatar.spare():
                self.step_sim()

        self._observe_handover(arm_tag, "after_avatar_retraction")
        if self.config.get("take_post_handover_hold", False):
            from ..task_bases.take_hold import establish_carry_hold
            hold_ok = establish_carry_hold(self, arm_tag)
            self._observe_handover(arm_tag, "after_carry_hold")
            if not hold_ok:
                return False

        self._plan_exec_substeps = 6
        self._move_screw(mid_pos, arm_tag)
        self._observe_handover(arm_tag, "after_transport_to_mid")
        obj_now = to_numpy(self.apple_actor.entity.get_pos())
        print(f"[take] after transport-to-mid: obj={np.asarray(obj_now).round(3).tolist()}",
              flush=True)

        # Phase 6: Lateral traverse to a hover above the plate, followed by a
        # slow vertical last mile.  Releasing directly from the transport
        # height gives the apple enough energy to bounce and roll beyond the
        # plate footprint (seed 0 consistently settled about 14 cm away).
        # `_move_cartesian` seeds every IK solve from the previous command, so
        # keeping the top-down orientation fixed avoids a wrist-branch
        # jump while allowing a low-energy release.
        above_plate = np.array([self.place_pos[0], self.place_pos[1], carry_z])
        self._move_screw(above_plate, arm_tag)
        self._observe_handover(arm_tag, "after_traverse_to_plate")
        obj_now = to_numpy(self.apple_actor.entity.get_pos())
        print(f"[take] after traverse-to-plate: obj={np.asarray(obj_now).round(3).tolist()}",
              flush=True)
        release_pos = np.asarray(self.place_pos, dtype=float).copy()
        release_pos[2] += self.RELEASE_DESCENT_ABOVE_PLACE
        # The final descent is BEST-EFFORT by design — we release from
        # wherever it gets us.  Its guarded fallbacks can latch
        # plan_success=False even when the cube lands dead-center (target
        # dist_xy=0.007 m).  Preserve the
        # pre-descent plan state across these terminal legs; a genuinely
        # bad descent still fails on the physical outcome.
        _plan_ok_before_descent = bool(self.plan_success)
        self._move_screw(release_pos, arm_tag)
        # One measured retry.  When the traverse leg ends in a twisted
        # branch, the descent's screw AND its guarded fallback can both
        # refuse (e.g. joint_delta 4.76 rad) — the object is then released
        # from ~28 cm and bounces off the plate.  The arm state
        # after the refused attempt differs, so a single re-plan usually
        # lands a workable branch.  This is a planning retry, not a re-do.
        tcp_now = np.asarray(arm.get_ee_pose(), dtype=float).ravel()[:3]
        if float(tcp_now[2] - release_pos[2]) > 0.05:
            print(f"[take] release-descent short by "
                  f"{tcp_now[2] - release_pos[2]:.3f}m; one screw retry",
                  flush=True)
            self._move_screw(release_pos, arm_tag)
        # The retry can land on-height but laterally off (e.g. 11 cm from
        # the plate center, missing the rim).
        # One guarded seeded move re-centers before opening.
        tcp_now = np.asarray(arm.get_ee_pose(), dtype=float).ravel()[:3]
        xy_off = float(np.linalg.norm(tcp_now[:2] - np.asarray(release_pos[:2])))
        if xy_off > 0.03:
            print(f"[take] release xy off by {xy_off:.3f}m; guarded re-center",
                  flush=True)
            link = tcp_to_link_pose(
                self._top_down_tcp(release_pos), arm.tcp_offset,
            )
            self._move_seeded(link.to_pose7(), arm_tag)
        obj_now = to_numpy(self.apple_actor.entity.get_pos())
        print(f"[take] after release-descent: obj={np.asarray(obj_now).round(3).tolist()}",
              flush=True)
        self._plan_exec_substeps = 1
        self.plan_success = _plan_ok_before_descent

        # PD convergence wait before release: opening while the arm
        # still lags the commanded release pose drops the object short of the
        # plate.  Hold the last command until the TCP xy is actually there.
        release_xy = np.asarray(release_pos[:2], dtype=float)
        for _ in range(300):
            tcp_xy = np.asarray(arm.get_ee_pose()[:2], dtype=float)
            if float(np.linalg.norm(tcp_xy - release_xy)) < 0.008:
                break
            self.step_sim()

        if self.GUARDED_RELEASE:
            from ..task_bases.take_release import lower_for_release
            if not lower_for_release(self, arm_tag):
                return False

        # Release just above the plate so the object settles instead of
        # bouncing out of the success footprint.
        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        hold_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        for _ in range(self.FINAL_SETTLE_STEPS):
            arm.entity.control_dofs_position(hold_qpos)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        settled = self._settle_released_object()
        if self.vla_recorder is not None:
            self.vla_recorder.mark_episode_end(
                "take_from_human_easy_post_release" if settled
                else "take_from_human_easy_release_unsettled"
            )
        return settled

    def play_blind_once(self) -> bool:
        """Blind ("no intent") expert: play_once's robot motions on its own clock.

        The robot never plays, waits for, or reads the human's delivery: no
        offer playback, no avatar step-in, no in-hand yaw canonicalization,
        no detach call and no retraction.  The human is driven only by the
        eval hooks (the offer starts at a random policy step drawn from
        ``EVAL_TRIGGER_STEP_MIN/MAX``; ``_eval_at_step`` releases the object
        once the closed gripper holds it, then withdraws the arm).  Grasp
        targets are the object's observed pose at plan time.

        Trigger window (policy steps of action_substeps=25 sim steps, the
        VLA cadence): the offer clip lasts ~35 sim steps (~2 ticks) and the
        take begins once the arm reaches its mid waypoint, ~850 sim steps
        (~34 ticks) after reset.  The window [0, 68] is centred on that
        moment: 0 = the human offers at once, done long before the robot
        needs the object; 68 = one more take-approach duration after the
        robot has already committed.
        """
        arm_tag = "right"
        self._release_settled = False
        self._release_settle_diagnostics = None
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        self._blind_apple_at_interaction = None
        self._boost_arm_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        self.ARM_MAX_REACH = type(self).ARM_MAX_REACH
        self.PRE_GRASP_HEIGHT = type(self).PRE_GRASP_HEIGHT
        self._plan_exec_substeps = 1
        self._handover_grasp_yaw = 0.0
        self.PRE_GRASP_EXTRA_CLEARANCE = type(self).PRE_GRASP_EXTRA_CLEARANCE

        self.open_gripper(arm_tag)

        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_down = t3d.quaternions.mat2quat(R_down)
        arm_base = np.array(arm.origin_pose.p)

        def _ik_feasible(pose7) -> bool:
            try:
                probe = arm.planner.solve_ik(
                    arm.get_arm_qpos(), np.asarray(pose7, dtype=float).ravel()[:7],
                    log=False,
                )
                return bool(getattr(probe, "success", False))
            except Exception:
                return True

        mid_candidates = []
        for ddx, ddy, ddz in ((0.0, 0.0, 0.0), (-0.10, 0.0, -0.05),
                              (0.10, 0.0, -0.05), (0.0, -0.08, 0.05)):
            mid = Pose(
                arm_base + np.array([
                    ddx,
                    self.MID_WAYPOINT_Y_FROM_BASE + ddy,
                    self.MID_WAYPOINT_Z_FROM_BASE + ddz,
                ]),
                q_down,
            )
            mid_candidates.append(tcp_to_link_pose(mid, tcp_offset))
        mid_choice = next(
            (c for c in mid_candidates if _ik_feasible(c.to_pose7())),
            mid_candidates[0],
        )
        if self.move_and_execute(mid_choice.to_pose7(), arm_tag) is None:
            return False

        # The take starts here: everything below reads the object as it is
        # observed now, whatever the human is doing.
        self._blind_robot_interaction_sim_step = int(
            getattr(self, "_interaction_sim_step", 0)
        )
        self._blind_human_ready_before_robot = bool(
            self.avatar is not None
            and getattr(self, "_eval_human_started_sim_step", None) is not None
            and self.avatar.spare()
        )
        self._blind_apple_at_interaction = np.asarray(
            self._get_object_center_world(self.apple_actor), dtype=float,
        ).ravel()[:3].tolist()
        print(f"[take-blind] take begins at sim step "
              f"{self._blind_robot_interaction_sim_step}; human ready="
              f"{self._blind_human_ready_before_robot}; object="
              f"{np.round(self._blind_apple_at_interaction, 3).tolist()}",
              flush=True)

        cls = type(self)
        candidates = (
            (cls.ARM_MAX_REACH, cls.PRE_GRASP_HEIGHT,
             cls.PRE_GRASP_EXTRA_CLEARANCE),
            (cls.ARM_MAX_REACH, 0.06, 0.02),
            (cls.ARM_MAX_REACH, 0.04, 0.0),
            (0.78, 0.06, 0.02),
            (0.74, 0.05, 0.0),
        )
        for reach, pre_h, extra in candidates:
            self.ARM_MAX_REACH = reach
            self.PRE_GRASP_HEIGHT = pre_h
            self.PRE_GRASP_EXTRA_CLEARANCE = extra
            probe_pre, probe_grasp = self._compute_grasp_poses(arm_tag)
            probe_hi = np.array(probe_pre.p) + np.array([
                0.0, 0.0, self.PRE_GRASP_EXTRA_CLEARANCE,
            ])
            if _ik_feasible(
                tcp_to_link_pose(self._top_down_tcp(probe_hi), tcp_offset).to_pose7()
            ) and _ik_feasible(
                tcp_to_link_pose(probe_grasp, tcp_offset).to_pose7()
            ):
                break
        else:
            self.ARM_MAX_REACH = cls.ARM_MAX_REACH
            self.PRE_GRASP_HEIGHT = cls.PRE_GRASP_HEIGHT
            self.PRE_GRASP_EXTRA_CLEARANCE = cls.PRE_GRASP_EXTRA_CLEARANCE

        pre_grasp_tcp, _ = self._compute_grasp_poses(arm_tag)
        pre_pos_hi = np.array(pre_grasp_tcp.p) + np.array([
            0.0, 0.0, self.PRE_GRASP_EXTRA_CLEARANCE,
        ])
        pre_link = tcp_to_link_pose(self._top_down_tcp(pre_pos_hi), tcp_offset)
        if self.move_and_execute(pre_link.to_pose7(), arm_tag) is None:
            return False

        # Descent target re-read from the object as observed now.
        _, grasp_tcp = self._compute_grasp_poses(arm_tag)
        grasp_link = tcp_to_link_pose(grasp_tcp, tcp_offset)
        if self.move_and_execute(grasp_link.to_pose7(), arm_tag) is None:
            return False
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()
        apple_pre = to_numpy(self.apple_actor.entity.get_pos())
        self._observe_handover(arm_tag, "before_close")

        self.set_gripper(
            self._handover_close_value(),
            arm_tag,
            num_steps=self.SET_GRIPPER_STEPS,
        )
        # No detach here: the human lets go through _eval_at_step once the
        # closed gripper is at the object.
        for _ in range(self.POST_DETACH_SETTLE_STEPS):
            self.step_sim()
        self._observe_handover(arm_tag, "after_detach_settle")

        grasp_pos = np.array(grasp_tcp.p)
        lift_pos = grasp_pos.copy()
        lift_pos[2] = apple_pre[2] + self.LIFT_HEIGHT_FROM_APPLE
        self._move_cartesian(
            grasp_pos, lift_pos, arm_tag,
            n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )
        object_lifted = to_numpy(self.apple_actor.entity.get_pos())
        lift_dz = float(object_lifted[2] - apple_pre[2])
        print(
            f"[take-blind] {self.obj_name} post-lift dz={lift_dz:.4f}m "
            f"pos={np.asarray(object_lifted).round(4).tolist()}",
            flush=True,
        )
        if lift_dz < 0.03:
            return False
        hold_q = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        for _ in range(self.POST_LIFT_GRIP_HOLD_STEPS):
            arm.entity.control_dofs_position(hold_q)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        if self.config.get("take_post_handover_hold", False):
            from ..task_bases.take_hold import establish_carry_hold
            if not establish_carry_hold(self, arm_tag):
                return False

        carry_z = min(float(lift_pos[2]), self.TABLE_TOP_Z + 0.38)
        mid_pos = np.array([
            0.0, arm_base[1] + self.MID_WAYPOINT_Y_FROM_BASE, carry_z,
        ])
        self._plan_exec_substeps = 6
        self._move_screw(mid_pos, arm_tag)
        above_plate = np.array([self.place_pos[0], self.place_pos[1], carry_z])
        self._move_screw(above_plate, arm_tag)
        release_pos = np.asarray(self.place_pos, dtype=float).copy()
        release_pos[2] += self.RELEASE_DESCENT_ABOVE_PLACE
        _plan_ok_before_descent = bool(self.plan_success)
        self._move_screw(release_pos, arm_tag)
        tcp_now = np.asarray(arm.get_ee_pose(), dtype=float).ravel()[:3]
        if float(tcp_now[2] - release_pos[2]) > 0.05:
            self._move_screw(release_pos, arm_tag)
        tcp_now = np.asarray(arm.get_ee_pose(), dtype=float).ravel()[:3]
        if float(np.linalg.norm(tcp_now[:2] - np.asarray(release_pos[:2]))) > 0.03:
            link = tcp_to_link_pose(
                self._top_down_tcp(release_pos), arm.tcp_offset,
            )
            self._move_seeded(link.to_pose7(), arm_tag)
        self._plan_exec_substeps = 1
        self.plan_success = _plan_ok_before_descent

        release_xy = np.asarray(release_pos[:2], dtype=float)
        for _ in range(300):
            tcp_xy = np.asarray(arm.get_ee_pose()[:2], dtype=float)
            if float(np.linalg.norm(tcp_xy - release_xy)) < 0.008:
                break
            self.step_sim()

        if self.GUARDED_RELEASE:
            from ..task_bases.take_release import lower_for_release
            if not lower_for_release(self, arm_tag):
                return False

        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()
        hold_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        for _ in range(self.FINAL_SETTLE_STEPS):
            arm.entity.control_dofs_position(hold_qpos)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        settled = self._settle_released_object()
        if self.vla_recorder is not None:
            self.vla_recorder.mark_episode_end(
                "take_from_human_easy_post_release" if settled
                else "take_from_human_easy_release_unsettled"
            )
        return settled

    def _settle_released_object(self):
        """Keep recording real physics until the released object is quiet.

        The old 150-step tail lasts only 0.15 seconds at this task's dt,
        shorter than the fall from its reachable release height. Even an
        already-landed object may rock longer than the former 0.75s minimum.
        Observe up to two seconds with the same support/quiet checks; this
        does not move objects or change the target/collision thresholds.
        """
        entity = self.apple_actor.entity
        dt = self.SIMULATION_DT
        sample_steps = max(1, int(round(0.02 / dt)))
        sample_dt = sample_steps * dt
        previous = np.asarray(to_numpy(entity.get_pos()), dtype=float).ravel()[:3]
        previous_q = np.asarray(to_numpy(entity.get_quat()), dtype=float).ravel()[:4]
        aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float).reshape(2, 3)
        initial_geometry_valid = bool(np.all(np.isfinite(aabb)) and np.all(aabb[0] <= aabb[1]))
        # Budget whole observation windows so the advertised final interval
        # is actually simulated rather than truncated by the sampling stride.
        budget = int(np.ceil(self.RELEASE_SETTLE_TIMEOUT_S / sample_dt)) * sample_steps
        stable_samples = 0
        diagnostics = {}
        for count in range(sample_steps, budget + 1, sample_steps):
            for _ in range(sample_steps):
                self.step_sim()
            current = np.asarray(to_numpy(entity.get_pos()), dtype=float).ravel()[:3]
            current_q = np.asarray(to_numpy(entity.get_quat()), dtype=float).ravel()[:4]
            aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float).reshape(2, 3)
            plate = np.asarray(to_numpy(self.plate_actor.entity.get_AABB()), dtype=float).reshape(2, 3)
            table_center = np.asarray(self._table_center_xy, dtype=float)
            table_half = np.asarray(self._table_half_size_xy, dtype=float)
            values = np.concatenate([current, current_q, previous, previous_q,
                                     aabb.ravel(), plate.ravel(), table_center.ravel(),
                                     table_half.ravel(), [self.TABLE_TOP_Z]])
            if (not initial_geometry_valid or not np.all(np.isfinite(values))
                    or np.linalg.norm(current_q) < 1e-12
                    or np.linalg.norm(previous_q) < 1e-12
                    or np.any(table_half <= 0) or np.any(aabb[0] > aabb[1])
                    or np.any(plate[0] > plate[1])):
                diagnostics = {"steps": count, "budget": budget,
                               "supported": False, "reason": "invalid_object_state"}
                break
            speed = float(np.linalg.norm(current - previous) / sample_dt)
            quaternion_scale = float(np.linalg.norm(current_q) * np.linalg.norm(previous_q))
            angle = 2 * np.arccos(np.clip(abs(np.dot(current_q, previous_q)) / quaternion_scale, 0, 1))
            angular_speed = float(angle / sample_dt)
            previous, previous_q = current.copy(), current_q.copy()
            center = aabb.mean(axis=0)
            over_plate = bool(np.all(center[:2] >= plate[0, :2])
                              and np.all(center[:2] <= plate[1, :2]))
            support_z = max(float(self.TABLE_TOP_Z), float(plate[1, 2]) if over_plate
                            else float(self.TABLE_TOP_Z))
            table_low = table_center - table_half
            table_high = table_center + table_half
            supported = bool(
                np.all(aabb[0, :2] >= table_low - 0.002)
                and np.all(aabb[1, :2] <= table_high + 0.002)
                and self.TABLE_TOP_Z - 0.02 <= aabb[0, 2] <= support_z + 0.015
            )
            stable_samples = stable_samples + 1 if supported and speed < 0.03 and angular_speed < 0.3 else 0
            diagnostics = {"steps": count, "budget": budget, "supported": supported,
                           "bottom_z": float(aabb[0, 2]), "support_z": support_z,
                           "linear_speed": speed, "angular_speed": angular_speed}
            if stable_samples >= 5:
                self._release_settle_diagnostics = diagnostics
                self._release_settled = True
                print("[take-release] " + json.dumps(diagnostics), flush=True)
                return True
        self._release_settle_diagnostics = diagnostics
        self._release_settled = False
        self.plan_success = False
        print("[take-release] unsettled " + json.dumps(diagnostics), flush=True)
        return False
