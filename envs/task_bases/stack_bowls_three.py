"""Stack three bowls: Franka picks two bowls and stacks them on a
third (base) bowl on the table. Single-arm, no avatar — RoboTwin's
`stack_bowls_three` task scaled down to one Franka.

Three identical bowls (002_bowl, model_id=1, 11.4 × 4.9 × 11.4 cm)
spawn at random xy in a 0.40 × 0.40 m region. Bowl 0 is the base;
the robot picks bowls 1 and 2 in turn and drops them on the stack.

Pure physics throughout: PUBLIC actual-scale diagonal rim grasps, checked native
MPLib trajectories and physical PD control. No robot-object pose pinning.
Fingers pinch across the rim and open before vertical withdrawal.
"""

import json
import itertools
from functools import lru_cache
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..grasp import GraspPose
from pathlib import Path

from ..utils import (
    Actor, Pose, load_object, to_numpy, ASSETS_PATH, TABLE_HEIGHT,
    urdf_merge_fixed_links_kwargs,
)
from .stack_bowls_motion import BowlNativeMotionMixin, _BowlMotionRejected
from .resting_guard import held_by_anyone
from ..grasp import tcp_to_link_pose
from ..object_catalog import get_entry


# Target object identity from the central catalog (model_id=1 = the smaller
# 11.4 × 4.9 cm bowl variant).
_BOWL_ENTRY = get_entry("002_bowl")
_BOWL_ID = _BOWL_ENTRY.object_id
_BOWL_MODEL_ID = _BOWL_ENTRY.model_id
_NUM_BOWLS = 3

# Randomization region — 0.40 × 0.40 m. Bowl outer dia ~11.4 cm so
# 0.16 m min separation gives 4-5 cm clearance for the gripper.
_REGION_X = (-0.18, +0.22)
_REGION_Y = (-0.18, +0.18)
_MIN_SEP_M = 0.16
_SPAWN_SAMPLE_ATTEMPTS = 256
_SPAWN_FALLBACK_MARGIN = 0.05

def _read_model_data(name, model_id=0):
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name, model_id=0):
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


# Measured wall profile (wall_profile_v1): the 002_bowl wall is 2.6-3.0 mm thick,
# flares only 2-6 degrees from vertical between 20 and 44 mm height, and curls
# above 44 mm (47 mm tall). The Franka main pad spans TCP-9..+8 mm. A 30
# degree pinch meets that wall in a diagonal slice and a vertical pinch meets
# the 2-6 degree flare along one pad edge (measured contact cluster at TCP-8
# mm); both leave a line contact that cannot resist the gravity torque about
# the pinch, which loses the bowl mid-lift. Match the pad to the
# local wall slope and centre the band on the straight part of the wall
# (41 mm: 38 mm left only 28.4 mm palm clearance against the 30 mm guard).
_LIP_TILT_DEG = 4.0
_LIP_HEIGHT_M = .041


@lru_cache(maxsize=1)
def _bowl_lip_annotations():
    """Center each rim annotation between actual finite-pad first contacts."""
    import trimesh
    scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
    mesh = trimesh.load(ASSETS_PATH/'objects'/_BOWL_ID/f'visual/base{_BOWL_MODEL_ID}.glb',
                        force='mesh', process=False)
    mesh.vertices *= scale
    offsets = np.array([[x,0,z] for x,z in itertools.product(
        np.linspace(-.0055,.0055,11),np.linspace(-.0085,.0085,11))])
    annotations = []
    tilt = np.deg2rad(_LIP_TILT_DEG)
    up = np.array([0.,1.,0.])
    for degrees in range(0,360,45):
        angle = np.deg2rad(degrees)
        radial = np.array([np.cos(angle),0.,np.sin(angle)])
        closing = np.cos(tilt)*radial-np.sin(tilt)*up
        approach = -np.sin(tilt)*radial-np.cos(tilt)*up
        rotation = np.column_stack((np.cross(closing,approach),closing,approach))
        point = .054*radial+_LIP_HEIGHT_M*up
        centers = offsets@rotation.T+point
        hits,_,faces = mesh.ray.intersects_location(
            centers-.12*closing,np.tile(closing,(len(centers),1)),multiple_hits=True)
        local = (np.asarray(hits).reshape(-1,3)-point)@rotation
        near = np.abs(local[:,1])<.040
        values = local[near,1]
        if not np.any(values<0) or not np.any(values>0):
            continue
        shift = float((values.min()+values.max())/2)
        if abs(shift)>.004:
            continue
        normals = []
        for sign in (-1,1):
            edge = float(np.max(sign*values))
            selected = near & (sign*local[:,1]>=edge-.001)
            normal = mesh.face_normals[faces[selected]].mean(0)
            normals.append(normal/np.linalg.norm(normal))
        if np.dot(*normals)>-.8:
            continue
        point += shift*closing
        annotations.append((f'stack_slope_lip_{degrees}',tuple(point/scale),
                            tuple(t3d.quaternions.mat2quat(rotation))))
    return tuple(annotations)


# Fixed, offline convex decomposition of 002_bowl model 1. Runtime CoACD
# rebuilt a different 12- or 13-hull bowl in every process, with rim chords
# 5-9 mm thick against the 3.0 mm visual wall, so fingers pinched a wall that
# is not there and success depended on the hull count (3/5 on 12-hull processes,
# 2/15 on 13-hull ones). The package keeps the textured visual and fixes the collision hulls
# (authored y-up frame, mesh units, catalog scale in the URDF).
_FIXED_COLLISION_URDF = (Path(__file__).resolve().parent / "stack_bowls_collision"
                         / "bowl_002_model1" / "bowl.urdf")


def load_fixed_collision_bowl(scene, pose, object_id, model_id=_BOWL_MODEL_ID,
                              convex=True, is_static=False, friction=None,
                              density=None):
    """Same call shape as ``load_object`` for the bowl, with fixed collision hulls."""
    import genesis as gs
    if object_id != _BOWL_ID or model_id != _BOWL_MODEL_ID or convex is not True:
        raise ValueError("fixed collision package only covers 002_bowl model 1")
    material = {}
    if friction is not None:
        material["friction"] = friction
    if density is not None:
        material["rho"] = density
    entity = scene.add_entity(
        gs.morphs.URDF(file=str(_FIXED_COLLISION_URDF), pos=tuple(pose.p),
                       quat=tuple(pose.q), fixed=bool(is_static), convexify=True,
                       decompose_object_error_threshold=float("inf"),
                       **urdf_merge_fixed_links_kwargs(_FIXED_COLLISION_URDF)),
        material=gs.materials.Rigid(**material),
    )
    return Actor(entity, _read_model_data(object_id, model_id), object_id)


def _read_extents(name, model_id=0):
    return np.array(_read_model_data(name, model_id)["extents"], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


_EASY_BOWL_XY = ((-0.08, -0.06), (0.14, -0.16))


def _sample_bowl_xy(rng, n_bowls):
    if n_bowls == 2:
        return [tuple(xy) for xy in _EASY_BOWL_XY]
    for _ in range(_SPAWN_SAMPLE_ATTEMPTS):
        pts = []
        ok = True
        for _ in range(n_bowls):
            x = float(rng.uniform(*_REGION_X))
            y = float(rng.uniform(*_REGION_Y))
            for px, py in pts:
                if (x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2:
                    ok = False
                    break
            if not ok:
                break
            pts.append((x, y))
        if ok and len(pts) == n_bowls:
            return pts
    fallback = [
        (
            _REGION_X[0] + _SPAWN_FALLBACK_MARGIN,
            _REGION_Y[0] + _SPAWN_FALLBACK_MARGIN,
        ),
        (
            _REGION_X[1] - _SPAWN_FALLBACK_MARGIN,
            _REGION_Y[1] - _SPAWN_FALLBACK_MARGIN,
        ),
        (0.0, 0.0),
    ]
    return fallback[:n_bowls]


def _jitter_easy_bowl_xy(rng, positions, amount, min_sep=_MIN_SEP_M):
    """+-amount per-axis jitter on the fixed easy (2-bowl) layout.

    Draws are APPENDED to the bowl RandomState after the yaw draws, so the
    yaw stream (and the global np.random stream) are unchanged.  The two
    nominal centres are 0.242 m apart, so a 0.03 m jitter keeps them at least
    0.163 m apart (> the 0.16 m min-sep); the clamp below is a guard only.
    """
    amount = float(amount)
    if amount <= 0.0:
        return [tuple(map(float, xy)) for xy in positions]
    deltas = rng.uniform(-amount, amount, size=(len(positions), 2))
    out = [
        (float(x + d[0]), float(y + d[1]))
        for (x, y), d in zip(positions, deltas)
    ]
    for i in range(len(out)):
        for j in range(i):
            if np.hypot(out[i][0] - out[j][0], out[i][1] - out[j][1]) < float(min_sep):
                return [tuple(map(float, xy)) for xy in positions]
    return out


class StackBowlsThree(BowlNativeMotionMixin, BaseTask):
    """Robot picks two bowls and stacks them on a third (base) bowl."""

    INSTRUCTION = "stack the three bowls on top of each other"
    OBJECT_SET = ["002_bowl"]
    use_avatar = False
    EASY_NUM_OBJECTS = 2
    HARD_NUM_OBJECTS = _NUM_BOWLS

    # Three 200-hull bowls in contact overflow Genesis's default collision
    # buffers (150 pairs, broad phase 150 x 8, 512 solver contacts):
    # "Exceeding max number of broad phase candidate contact pairs" /
    # "candidate contact points" / "post-pruning contact points".
    RIGID_OPTIONS = {"max_collision_pairs": 600, "multiplier_collision_broad_phase": 16,
                     "max_contacts": 4096}

    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.6, 0.55)
    TABLE_CENTER_XY = (0.0, -0.35)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.55, -0.85),
        (0.55, -0.85),
        (-0.55, 0.15),
        (0.55, 0.15),
    )

    BOWL_UPRIGHT_QUAT = np.array(
        [0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64,
    )
    BOWL_YAW_LIMIT = np.pi / 4
    # Without jitter the easy (2-bowl) layout is a fixed pair of XY centres, so
    # an open-loop replay solves it.  Per-bowl +-3 cm jitter (config key
    # ``bowl_xy_jitter_m``; 0 restores the fixed layout).  Only variants that
    # use this class's ``load_actors`` (base + neutral) are affected; the
    # assist / interrupt variants have their own samplers.
    EASY_BOWL_XY_JITTER_M = 0.03
    BOWL_SPAWN_Z_CLEARANCE = 0.02
    # The built pair friction scales the pinned engine's contact regularizer
    # (measured diag = invweight*(1+mu^2)*2*mu^2*(1-imp)/imp), so mu 5.0 rows
    # are 325x softer than mu 1.0 rows. At high pair friction the
    # pads sink 1.1 mm per side into the 3.3 mm rim wall, reach their 0.5 mm
    # command, and hold about 0.02 N per pad, so every tiny lift rotates the
    # bowl out. A two-body drop test measured rest overlap 1.679 -> 0.027 mm
    # for pair mu 4 -> 1. Franka pad geometry is 1.0, so the pair takes 1.0.
    BOWL_FRICTION = 1.0
    BOWL_DENSITY = 10.0
    # Class switch: only variants that opt in load the fixed-collision bowl;
    # the default keeps runtime CoACD so siblings are unchanged.
    BOWL_FIXED_COLLISION = False
    # Class switch: offer finger-swapped rim grasps and require wrist
    # headroom for the carry. Default off keeps siblings unchanged.
    BOWL_WRIST_FLIP = False
    BOWL_WRIST_HEADROOM = 0.75

    LIFT_HEIGHT = 0.20
    # Plate-bottom clearance above the stack top at release. Releasing
    # right at contact (DROP_BOWL_DZ=0) jerks the plate when the
    # gripper opens; 3 cm gives the plate time to fall and settle on
    # plate i's top surface (28 mm thick disk catches it cleanly).
    DROP_BOWL_DZ = 0.030

    # Success: loose stacking metric. A stacked bowl's origin only needs
    # to land inside bowl 0's horizontal footprint and be higher than
    # bowl 0. This intentionally ignores uprightness and exact centering.
    SUCCESS_XY_TOL = 0.10
    SUCCESS_DZ_MIN = 0.0
    # Kept for old configs/diagnostics; check_success no longer uses it.
    SUCCESS_UP_DOT_MIN = 0.5
    SUCCESS_ELEVATED_DZ_MIN = 0.0
    SUCCESS_BASE_BOUNDS_MARGIN = 0.0
    # A stacked bowl rests on the stack: its origin (bottom) at most
    # (number of stacked bowls) x bowl height + this margin above bowl 0's,
    # and held by neither the avatar nor the gripper (closed gripper with the
    # TCP within SUCCESS_HELD_TCP_RADIUS_M of the bowl's mid-height centre;
    # the rim pinch sits ~one rim radius away).  Without it a bowl carried
    # over the base counted (SUCCESS_DZ_MIN = 0).
    SUCCESS_REST_DZ_MARGIN_M = 0.03
    SUCCESS_HELD_TCP_RADIUS_M = 0.09

    GRASP_MAX_CANDIDATES = 10
    GRASP_SETTLE_STEPS = 40
    # Diagonal rim pinch uses the previous external-grip command and PD.
    # Actual retention, not commanded aperture, determines lift acceptance.
    GRIPPER_ENTRY_VALUE = 1.0
    GRIPPER_HOLD_VALUE = 0.0125
    GRIPPER_CLOSE_SETTLE_STEPS = 120
    LIFT_SUCCESS_Z_MARGIN = 0.02
    ABOVE_STACK_SETTLE_STEPS = 80
    RELEASE_PARTIAL_OPEN_VALUE = 0.35
    RELEASE_PARTIAL_SETTLE_STEPS = 60
    RELEASE_OPEN_SETTLE_STEPS = 80
    RETREAT_Z_ABOVE_TABLE = 0.30
    INITIAL_SETTLE_STEPS = 60
    FINAL_SETTLE_STEPS = 180

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        # Native MPLib remains active through acquisition and placement.
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)
        self._num_bowls = self.difficulty_object_count()

    def _measure_held_offset(self, bowl_idx, arm_tag):
        """TCP→bowl offset right after close, used to aim the place pose.

        The bowl is held purely by friction — Genesis updates its world
        pose every step. We only sample once (post-close) so we know how
        the bowl sits in the gripper for placement targeting.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_now = np.asarray(arm.get_ee_pose()[:3], dtype=np.float64)
        bowl_p = np.asarray(self.bowls[bowl_idx].get_pose().p, dtype=np.float64)
        return bowl_p - tcp_now

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Wider table than default (matches place_bread_in_basket) so the
        # 0.40 × 0.40 m spawn region is fully inside the top.
        BaseTask._create_rectangular_table(
            self,
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
            table_thickness=self.TABLE_THICKNESS_VALUE,
            leg_radius=self.TABLE_LEG_RADIUS,
            leg_xy=self.TABLE_LEG_XY,
            remember_bounds_before_variant=False,
        )

    def load_actors(self):
        # mesh-y → world-z (002_bowl is y-up).
        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = _sample_bowl_xy(rng, self._num_bowls)

        self._bowl_scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
        bowl_extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        self._bowl_extents = bowl_extents
        # mesh-y is the bowl's vertical axis; after upright_q it is world z.
        self._bowl_height = float(bowl_extents[1]) * self._bowl_scale

        # Yaw clamped to ±π/4: the rim grasp candidates are written for
        # the four cardinal mesh directions (`cont_bowl_top_down_v1` set
        # in the shared YAML); the helper's approach-toward-robot score
        # picks whichever cardinal is most reachable, so a moderate yaw
        # spread doesn't hurt.
        yaws = [
            float(rng.uniform(-self.BOWL_YAW_LIMIT, self.BOWL_YAW_LIMIT))
            for _ in range(self._num_bowls)
        ]
        self._easy_bowl_jittered = False
        if self._num_bowls == 2:
            self._easy_bowl_jittered = float(
                self.config.get("bowl_xy_jitter_m", self.EASY_BOWL_XY_JITTER_M)) > 0.0
            positions = _jitter_easy_bowl_xy(
                rng, positions,
                float(self.config.get("bowl_xy_jitter_m", self.EASY_BOWL_XY_JITTER_M)),
            )

        # Lift mesh slightly above the table so the rim wall doesn't
        # collide on spawn.
        spawn_z = table_top + self.BOWL_SPAWN_Z_CLEARANCE
        self.bowls = []
        self._spawn_xy = []
        for i, ((cx, cy), yaw) in enumerate(zip(positions, yaws)):
            yaw_q = np.asarray(
                t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, yaw, "sxyz")),
                dtype=np.float64,
            )
            full_q = _quat_mul(yaw_q, upright_q)
            # density 10 kg/m³ is unrealistic but keeps plate mass
            # ~12 g; gripper torque during transit is the dominant
            # slip-cause on a rim grip and lower mass cuts it linearly.
            loader = (load_fixed_collision_bowl if self.BOWL_FIXED_COLLISION
                      else load_object)
            actor = loader(
                self.scene,
                Pose([cx, cy, spawn_z], full_q),
                _BOWL_ID, model_id=_BOWL_MODEL_ID,
                convex=True, is_static=False,
                friction=self.BOWL_FRICTION,
                density=self.BOWL_DENSITY,
            )
            self.bowls.append(actor)
            self._spawn_xy.append((cx, cy))

        # Bowl 0 = base (stays put); the rest get stacked on top.
        self._stacked_indices = list(range(1, self._num_bowls))

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def _bowl_xyz(self, idx):
        return to_numpy(self.bowls[idx].entity.get_pos()).ravel()[:3]

    def _bowl_xyz_str(self, idx):
        p = self._bowl_xyz(idx)
        return f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f})"

    def _bowl_up_dot(self, idx):
        quat = to_numpy(self.bowls[idx].entity.get_quat()).ravel()[:4]
        rot = t3d.quaternions.quat2mat(quat)
        # Mesh local +Y is the bowl opening/up direction.
        up_world = rot[:, 1]
        return float(np.dot(up_world, np.array([0.0, 0.0, 1.0])))

    def _bowl_xy_half_bounds(self):
        scale = float(getattr(self, "_bowl_scale", _read_scale(_BOWL_ID, _BOWL_MODEL_ID)))
        extents = getattr(self, "_bowl_extents", None)
        if extents is None:
            extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        extents = np.asarray(extents, dtype=np.float64)
        # 002_bowl is mesh-Y-up, so mesh X/Z are the table-plane axes.
        margin = float(getattr(self, "SUCCESS_BASE_BOUNDS_MARGIN", 0.0))
        return np.array([
            0.5 * float(extents[0]) * scale + margin,
            0.5 * float(extents[2]) * scale + margin,
        ], dtype=np.float64)

    def _pick_and_stack_one(self, src_idx, stack_position):
        try:
            with self._bowl_motion_scene("right"):
                return self._stack_bowl_physical(src_idx, stack_position)
        except _BowlMotionRejected:
            self.plan_success = False
            return False

    def _bowl_grasp_candidates(self):
        # PUBLIC still plans every approach and checks the real fixtures.
        candidates = []
        for name,position,quaternion in _bowl_lip_annotations():
            candidates.append(GraspPose(name,np.array(position),np.array(quaternion),
                                        pre_distance=.12,source='manual',scale_frame='mesh_unit'))
            if self.BOWL_WRIST_FLIP:
                # Same rim pinch with the fingers swapped: 180 deg about the
                # approach (TCP z) axis. Equal approach score, so the stable
                # ranking tries it right after the original.
                flipped = t3d.quaternions.qmult(quaternion, (0., 0., 0., 1.))
                candidates.append(GraspPose(f'{name}_flip',np.array(position),np.array(flipped),
                                            pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        return candidates

    def _bowl_wrist_headroom(self, plans, arm_tag):
        """Carry needs wrist travel: joint 7 rose up to 0.70 rad after closing
        in a 20-episode run, and every screw rejection was joint 7 crossing its upper
        limit. Keep that travel (plus .05 rad) available at the grasp."""
        if not self.BOWL_WRIST_FLIP:
            return True
        grasp = plans[1]
        final = np.asarray(getattr(grasp, 'position', ()), dtype=float)
        if final.ndim != 2 or not len(final) or final.shape[1] < 7:
            return False
        limits = np.asarray(self.robot.get_arm(arm_tag).planner._mplib.joint_limits, dtype=float)
        wrist = float(final[-1][6])
        return bool(np.isfinite(wrist) and limits[6][0] <= wrist
                    and wrist + self.BOWL_WRIST_HEADROOM <= limits[6][1])

    def _bowl_carry_move(self, carry, arm_tag):
        """Straight-line carry; on the jittered easy layout, an MPLib RRT
        fallback when the screw path itself is rejected before execution.

        With +-3 cm bowl jitter some carries hit a screw-path
        rejection (joint limit / collision along the straight line) that the
        fixed layout never produced.  The fallback is still MPLib
        (``plan_path`` = RRT-Connect + IK), passes the same collision, step
        and FK-endpoint gates, executes through PD control, and the caller's
        ``retained("carry")`` check still rejects a slipped bowl.  Variants
        with their own samplers never set ``_easy_bowl_jittered``.
        """
        try:
            return self._bowl_move(carry, arm_tag, "carry")
        except _BowlMotionRejected as exc:
            pre_execution = ("path, collision or FK rejected" in str(exc)
                             or "native screw rejected" in str(exc))
            if not (pre_execution and getattr(self, "_easy_bowl_jittered", False)):
                raise
        self.plan_success = True  # the screw rejection above cleared it
        arm = self.robot.get_arm(arm_tag)
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        self._bowl_refresh_avatar(arm_tag)
        link = tcp_to_link_pose(carry, arm.tcp_offset)
        for attempt in range(3):
            plan = arm.planner.plan_path(seed.copy(), link.to_pose7())
            if self._bowl_plan_valid(plan, seed, carry, arm_tag):
                print(f"[bowl-motion] carry: RRT fallback accepted (attempt {attempt + 1})",
                      flush=True)
                self.execute_plan(plan, arm_tag)
                for _ in range(60):
                    self.step_sim()
                actual = self._bowl_arrived(carry, arm_tag, "carry")
                print(f"[bowl-motion] carry: TCP={actual.p}", flush=True)
                return actual
        self._bowl_reject("carry: screw and RRT fallback rejected")

    def _stack_bowl_physical(self, src_idx, stack_position):
        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        src_pose = self.bowls[src_idx].get_pose()
        self.set_gripper(self.GRIPPER_ENTRY_VALUE, arm_tag)
        # Keep the real table gate and let PUBLIC selection reject
        # infeasible diagonal approaches before entering the rim.
        result = self.select_and_execute_grasp(
            object_name=_BOWL_ID, object_pose=src_pose, arm_tag=arm_tag,
            model_id=_BOWL_MODEL_ID, object_scale=self._bowl_scale,
            max_candidates=self.GRASP_MAX_CANDIDATES,
            candidate_grasps=self._bowl_grasp_candidates(), table_z=self.TABLE_TOP_Z,
            plan_attempts=3, screw_qpos_step=.02,
            post_plan_check=lambda plans: (self._bowl_grasp_plan_valid(plans, arm_tag)
                                           and self._bowl_wrist_headroom(plans, arm_tag)),
        )
        if result is None:
            return False
        grasp_link, _, _grasp = result
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()
        self._bowl_refine_open_grasp(grasp_link * arm.tcp_offset, arm_tag,
                                   lambda: self._bowl_xyz(src_idx))
        # Pure physical pinch: never pin, attach or overwrite a bowl pose.
        self.set_gripper(self.GRIPPER_HOLD_VALUE, arm_tag)
        for _ in range(self.GRIPPER_CLOSE_SETTLE_STEPS):
            self.step_sim()
        tcp = self._bowl_arrived(grasp_link * arm.tcp_offset, arm_tag, "closed grasp")
        pre_lift_z = float(self._bowl_xyz(src_idx)[2])
        local_held = tcp.to_matrix()[:3, :3].T @ (self._bowl_xyz(src_idx) - tcp.p)

        def retained(label):
            measured = np.asarray(arm.get_ee_pose(), dtype=float)
            if (measured.shape != (7,) or not np.all(np.isfinite(measured))
                    or not np.isclose(np.linalg.norm(measured[3:]), 1., atol=1e-4)):
                self._bowl_reject(f"{label}: invalid measured TCP")
            current = Pose.from_pose7(measured)
            local_now = current.to_matrix()[:3, :3].T @ (self._bowl_xyz(src_idx) - current.p)
            error = float(np.linalg.norm(local_now - local_held))
            if not np.isfinite(error) or error > .030:
                self._bowl_reject(f"{label}: lost bowl purchase, offset={error:.4f}m")
            print(f"[bowl-motion] {label}: held-offset change={error:.4f}m", flush=True)
            return current

        # Preserve the50mm acquisition check and unchanged native tiny-motion,
        #20mm body-rise and30mm retention gates before full lift or carry.
        tiny = Pose(tcp.p + np.array([0., 0., .050]), tcp.q)
        self._bowl_move(tiny, arm_tag, "tiny lift", tiny=True)
        if float(self._bowl_xyz(src_idx)[2]) - pre_lift_z < self.LIFT_SUCCESS_Z_MARGIN:
            self._bowl_reject("tiny lift: bowl did not leave support")
        retained("tiny lift")
        lift = Pose(tcp.p + np.array([0., 0., self.LIFT_HEIGHT]), tcp.q)
        self._bowl_move(lift, arm_tag, "lift")
        current = retained("lift")
        target_x, target_y, stack_top_z = np.asarray(stack_position, dtype=float)
        if not np.all(np.isfinite([target_x, target_y, stack_top_z])):
            self._bowl_reject("invalid stack destination")
        held = self._bowl_xyz(src_idx) - current.p
        carry = Pose([target_x - held[0], target_y - held[1], current.p[2]], current.q)
        self._bowl_carry_move(carry, arm_tag)
        current = retained("carry")
        held = self._bowl_xyz(src_idx) - current.p
        place = Pose([target_x - held[0], target_y - held[1],
                      stack_top_z + self.DROP_BOWL_DZ - held[2]], current.q)
        if place.p[2] > current.p[2]:
            self._bowl_reject("stack destination exceeds carry height")
        self._bowl_move(place, arm_tag, "descent")
        for _ in range(self.ABOVE_STACK_SETTLE_STEPS):
            self.step_sim()
        self._bowl_arrived(place, arm_tag, "release")
        retained("release")
        self.set_gripper(self.RELEASE_PARTIAL_OPEN_VALUE, arm_tag)
        for _ in range(self.RELEASE_PARTIAL_SETTLE_STEPS):
            self.step_sim()
        self.set_gripper(self.GRIPPER_ENTRY_VALUE, arm_tag)
        for _ in range(self.RELEASE_OPEN_SETTLE_STEPS):
            self.step_sim()
        actual = self._bowl_arrived(place, arm_tag, "open release")
        # Leave vertically from the measured release pose so fingers do not
        # sweep sideways across the nested bowls.
        retreat = Pose([actual.p[0], actual.p[1],
                        max(actual.p[2], self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE)], actual.q)
        self._bowl_move(retreat, arm_tag, "withdrawal")
        # The unchanged geometric scorer can accept a bowl still held above
        # the stack. Reject that physical outcome at the controller boundary.
        if float(self._bowl_xyz(src_idx)[2]) > stack_top_z + self.DROP_BOWL_DZ + .020:
            self._bowl_reject("withdrawal: bowl remains elevated after release")
        return True

    def play_once(self) -> bool:
        # Pre-task object settle; excluded from recordings.
        with self.suppress_recording():
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False

        # Pick each non-base bowl in turn and place above the base.
        # The stack-top reference is the highest already-placed bowl's
        # mesh-origin (= bowl bottom) + bowl_height. For the first
        # stacked bowl this is base_top; for the second it includes
        # whatever bowl 1 actually settled at.
        for stack_idx, src_idx in enumerate(self._stacked_indices):
            base_xyz = self._bowl_xyz(0)
            placed_so_far = [0] + self._stacked_indices[:stack_idx]
            top_z = max(self._bowl_xyz(i)[2] for i in placed_so_far)
            stack_top_z = top_z + self._bowl_height
            stack_position = (base_xyz[0], base_xyz[1], stack_top_z)
            if not self._pick_and_stack_one(src_idx, stack_position):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def check_success(self) -> bool:
        if any(b is None for b in self.bowls):
            return False
        positions = [self._bowl_xyz(i) for i in range(self._num_bowls)]
        base_xy = positions[0][:2]
        base_z = float(positions[0][2])
        half_bounds = self._bowl_xy_half_bounds()
        # Stacked bowl centers must be inside bowl 0's footprint and above it.
        for i in self._stacked_indices:
            delta_xy = np.abs(positions[i][:2] - base_xy)
            dz = float(positions[i][2]) - base_z
            if bool(np.any(delta_xy > half_bounds)):
                return False
            if dz <= self.SUCCESS_DZ_MIN:
                return False
            if not self._bowl_resting_on_stack(i, dz):
                return False
        return True

    def _bowl_resting_on_stack(self, idx: int, dz: float) -> bool:
        """Success guard for one stacked bowl (see SUCCESS_REST_DZ_MARGIN_M)."""
        height = float(getattr(self, "_bowl_height", 0.0) or 0.0)
        max_dz = len(self._stacked_indices) * height + self.SUCCESS_REST_DZ_MARGIN_M
        if float(dz) > max_dz:
            return False
        center = np.asarray(self._bowl_xyz(idx), dtype=float).copy()
        center[2] += 0.5 * height
        return not held_by_anyone(
            self, self.bowls[idx], center, radius_m=self.SUCCESS_HELD_TCP_RADIUS_M,
        )
