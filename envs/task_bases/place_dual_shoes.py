"""Place dual shoes: Franka picks 2 shoes off the table and drops them
into a stationary open-top shoebox.  Single-arm, no avatar — RoboTwin's
``place_dual_shoes`` task scaled down to one Franka.

**Pure physics** — for both robot and held objects:
  - Robot arm moves only via checked native MPLib → ``execute_plan`` (PD
    trajectory tracking).  No ``set_qpos`` / ``teleport_arm_joints`` /
    ``attach_to_gripper`` calls during the rollout.  The only mutations
    we apply to the arm are PD-gain bumps on the fingers
    (``set_dofs_kp/kv/force_range``) — control-gain changes, not state
    teleports.  This matches the recipe in ``place_burger_fries``.
  - Shoes are held entirely by friction × normal force × gripper PD.  No
    kinematic snap, no ``entity.set_pos``/``set_quat`` on shoes during
    play_once.

Skips RoboTwin's "tip pointing left" orientation gate — success here
just requires both shoes to land inside the shoebox footprint and
above its floor.

Two shoes + the shoebox xy are randomized inside a 0.40 × 0.40 m
region with a 0.22 m minimum pairwise separation (shoebox is ~26 cm
along its long axis; that sep keeps shoes from spawning under the
shoebox lip).
"""

import json
import os
from pathlib import Path

import genesis as gs
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..manipulation import TopDownPickPlaceMixin
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..utils import (
    ArticulationActor, Pose, load_object, to_numpy,
    ASSETS_PATH, TABLE_HEIGHT,
)
from ..object_catalog import get_entry
from .place_dual_shoes_motion import ShoeNativeMotionMixin, _ShoeMotionRejected
from .resting_guard import held_by_anyone


# Target object identity comes from the central catalog (single source);
# model_id=0 is the variant whose grasp_poses_franka.yml has 4 verified
# top-down grasps (center / heel × y000 / y090 axis).  The 2 toe variants
# in the YAML are blacklisted in data/grasp_helper_grid/041_shoe.json
# (rate=0/5 — toe is too thin for parallel-jaw friction).
_SHOE_ENTRY = get_entry("041_shoe")
_SHOE_ID = _SHOE_ENTRY.object_id
_SHOE_MODEL_ID = _SHOE_ENTRY.model_id
_BOX_ID = "007_shoe-box"   # receptacle (not a manipulated target)
# Inner floor at the visual mesh centre, in unscaled mesh-Y coordinates.
# The surface is slightly sloped; its variation across the interior is less
# than 0.05 mm at catalog scale. Generated-collision tests check the actual GLB.
_BOX_INNER_FLOOR_MESH_Y = 0.00503569
_SHOEBOX_CACHE_ROOT = Path("/tmp/genesis_hr_shoebox_v3_visible_floor")

# Randomization region — covers the Franka's reachable workspace with
# the base mounted at (0, -0.65, 0.75).  Slightly wider than
# place_bread_in_basket's 0.40 × 0.40 m to give the shoebox + 2 shoes
# more spread.
# Spawn region tightened to the Franka's *reachable annulus* (base at
# [0,-0.65]).  Under Genesis 1.0.0 the shoe's top-down pre-grasp IK
# succeeds out to ~0.52 m from base and fails (no mplib solution) at
# >=0.56 m; the old (-0.38,+0.06) y-band let shoes spawn up to 0.71 m
# away, so ~half of seeds were ungraspable.  The new band keeps every
# corner within reach:
#   nearest  (x=0,    y=-0.34) -> 0.31 m
#   farthest (x=0.20, y=-0.16) -> 0.53 m
# (Empirically the shoe's top-down pre-grasp IK starts failing at ~0.54 m.
# A still-tighter 0.36x0.15 band tested slightly worse — it forces the
# box+shoe layout sampler into its diagonal fallback too often — so this
# 0.40x0.18 band is kept.)  The shoebox (drop target, approached from
# above) is reachable across the whole band.
_REGION_X = (-0.20, +0.20)   # 0.40 m wide
_REGION_Y = (-0.34, -0.16)   # 0.18 m deep, within the reachable annulus
# Shoebox is 26 cm along its long axis; shoes are 22 cm long.  0.24 m
# min sep prevents the box's long edge (or a yawed corner) from
# clipping a shoe spawn — needs a bit more headroom than
# place_bread_in_basket's 0.18 m because of the larger box footprint.
_MIN_SEP_M = 0.24
_SAMPLE_BATCH_ATTEMPTS = 64
_SAMPLE_POINT_ATTEMPTS = 64
_SAMPLE_FALLBACK_MARGIN = 0.05
# Box yaw randomization range (about world-z, composed with upright_q).
# ±60° lets the box face roughly any direction while keeping its long
# axis broadly across the reach corridor.
_BOX_YAW_RANGE = (-np.pi / 3, +np.pi / 3)

# Shoe mesh contact is soft/thin enough that the generic Franka "strong grip"
# recipe visibly drives the fingers into the shoe.  Keep the gripper forceful
# enough to survive transit, but avoid saturating it as hard as box/cube tasks.
_SHOE_FINGER_KP = 8000.0
_SHOE_FINGER_KV = 250.0
# ADAPTIVE cage close.  set_gripper is NORMALIZED (1=open → 8 cm total gap
# on the Franka, 0=closed), so the original 0.04 commanded a ~3 mm gap on a
# ~7.3 cm-wide hollow shoe: at the old 55–80 N force cap the fingers could
# only satisfy it by crushing until they tunnelled through the thin CoACD
# side walls INTO the cavity — the shoe was caged from inside and a finger
# stayed wedged under the collar at release ("finger stuck inside the
# shoe").  Width-MATCHED gaps (0.65/0.60 sized from the visual mesh) then
# proved fragile the other way: the CoACD collision surface runs several
# mm NARROWER than the visual mesh and settled shoes sink/tilt 1–3 cm
# (fingers can reach a 49.5 mm gap through a visually
# 54–58 mm heel without contact), so fixed gaps miss.
# The robust recipe is a DEEP target with a MODERATE force cap: fingers
# advance until they stall on the real collision surface wherever it is
# (self-calibrating to CoACD shrink, sink, and tilt), and 35 N is well
# below the sustained-crush regime that tunnels through the walls while
# 35 N × friction 5 ≈ 175 N shear capacity carries the ~3 N shoe through
# transit accelerations (25 N held the lift but lost the shoe mid-carry
# on 2/8 seeds).
_SHOE_FINGER_FORCE_N = 35.0
_SHOE_CLOSE_TARGET = 0.30
_SHOE_RELEASE_FORCE_N = 100.0
# Grip-point offsets from the live-AABB centre along the shoe's length
# axis.  The ankle collar + opening span the MIDDLE-REAR of the shoe
# (model 0 top-rim z in [-0.091, +0.009] of a ±0.104 length, collar peak
# 9.1 cm): a centre descent puts the palm on the collar (palm face sits
# ~7 cm above the table at the near-sole grasp depth) and the fingers
# over the cavity.  Two grippable ends exist; the pick prefers the RAISED
# end of a pitched shoe, else the end FARTHER from the shoebox (a toe/heel
# resting against the box wall makes that end's descent land on the box
# rim):
#   - TOE/vamp  (+0.045): closed top 5.4–6.1 cm (palm clears at the
#     near-sole grasp depth), walls 66→54 mm across the pad band.
#   - HEEL      (-0.068): collar rim 7.3 cm above the sole, near-parallel
#     quarter-panel walls 56–58 mm.
# Both ends use the adaptive _SHOE_CLOSE_TARGET above and the top-anchored
# grasp height below.
_SHOE_GRIP_TOEWARD_M = 0.045
_SHOE_GRIP_HEELWARD_M = 0.068
_SHOE_MIN_LIFT_DZ = 0.05
# Fingertip height measured UP from the shoe's MESH-ORIGIN z (= the sole
# plane at the body centre; mesh y=0 is the sole).  Neither the table nor
# the AABB is a reliable anchor: a settled shoe's hull can sink 1.7-2.8 cm
# into the tabletop (load concentrates on a sharp CoACD sole piece, so a
# table-anchored fingertip lands 4-7 cm above the ACTUAL sole in the
# narrow collar zone → graze or collar-hook), and the AABB top/centre
# drift with tilt + yaw (AABB support inflation).  The origin z rides
# with the sink exactly.  Heel grasp sits 8 mm higher so the palm face
# (fingertip + ~5 cm) clears the 7.4 cm heel-collar rim.
_SHOE_GRASP_ABOVE_SOLE_TOE_M = 0.020
_SHOE_GRASP_ABOVE_SOLE_HEEL_M = 0.028
_SHOE_GRASP_TABLE_CLEAR_M = 0.015
_SHOE_TRANSPORT_Z_ABOVE_TABLE = 0.22


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> np.ndarray | float:
    """Return the asset scale.  Some assets (007_shoe-box) use per-axis
    scales — return a 3-vector in that case so the URDF helper can apply
    them per-axis to the visual mesh."""
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return float(raw[0])
        return np.array(raw, dtype=np.float64)
    return float(raw)


def _read_bbox_center(name: str, model_id: int = 0) -> np.ndarray:
    raw = _read_model_data(name, model_id).get("center", [0.0, 0.0, 0.0])
    return np.asarray(raw, dtype=np.float64)


def _ensure_shoebox_urdf(scale: np.ndarray) -> tuple[Path, np.ndarray, np.ndarray]:
    """Generate a one-link URDF for 007_shoe-box with **primitive
    box-wall collision** (floor + 4 sides) and the GLB mesh as visual.

    Why not CoACD on the GLB: the shoebox wall is ~5 mm thick at scale
    [0.47, 0.43, 0.40]; CoACD would split it into thin convex slabs
    that shoes can wedge through during the drop. Five boxes provide a simple
    collision fixture; contact penetration still requires physical validation.

    URDF dimensions are baked in METERS (visual mesh pre-scaled by
    `scale`, box sizes pre-multiplied by `scale`), so the URDF morph
    is loaded with scale=1.0 — no ambiguity about whether Genesis's
    URDF scale applies to box primitives.

    Returns ``(urdf_path, half_extents_world_meshframe,
    bbox_min_meshframe_meters)`` where:

      - ``half_extents_world_meshframe`` is the box's half-extents
        in (mesh-x, mesh-y, mesh-z) order, in meters.  After upright_q
        is applied to the URDF body these map to (world-x, world-z,
        world-y).
      - ``bbox_min_meshframe_meters`` is the bbox-min in mesh frame,
        meters; mesh-y component is the outer bbox bottom, not the inner
        floor surface.

    Cached at /tmp on first call.
    """
    # Per-axis scale vector.
    if isinstance(scale, (int, float)):
        scale_vec = np.array([scale, scale, scale], dtype=np.float64)
    else:
        scale_vec = np.asarray(scale, dtype=np.float64).reshape(3)
    if not np.isfinite(scale_vec).all() or np.any(scale_vec <= 0):
        raise ValueError("Shoebox scale must contain finite positive values")

    # Geometry version and exact scale are both part of the cache key. The
    # previous cache reused the first scale and placed the slab's top 10 mm
    # above the outer bbox bottom instead of at the visible inner floor.
    scale_key = "_".join(format(float(value), ".17g") for value in scale_vec)
    cache_dir = _SHOEBOX_CACHE_ROOT / scale_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    visual_obj = cache_dir / "shoebox_visual.obj"
    urdf_path = cache_dir / "shoebox.urdf"

    # Bbox computed via trimesh on the GLB (in mesh-unscaled units).
    # Numbers here came from a one-time trimesh.load + bounds query
    # (see initial implementation notes); cached as constants so the
    # task module doesn't import trimesh on every reset.
    bbox_min_mesh = np.array([-0.18601, 0.0, -0.31977], dtype=np.float64)
    bbox_max_mesh = np.array([+0.18601, 0.23583, +0.31977], dtype=np.float64)

    # Pre-scale visual mesh + bake URDF box dims in meters.
    bbox_min_m = bbox_min_mesh * scale_vec
    bbox_max_m = bbox_max_mesh * scale_vec
    ext_m = bbox_max_m - bbox_min_m
    cen_m = (bbox_min_m + bbox_max_m) / 2.0

    # Race-safe OBJ generation. Many task processes load the shoebox
    # concurrently (parallel collect/sweep). A bare ``if not exists: export``
    # lets process B read a half-written OBJ during process A's ``mesh.export``
    # -> trimesh "At least one mesh must be present in file". Mirror the
    # ensure_inertial_urdf recipe: reuse only a non-empty file, otherwise
    # export to a unique temp path and ``os.replace`` (atomic on POSIX, so a
    # reader sees either no file or a complete one — never a partial write).
    if not (visual_obj.exists() and visual_obj.stat().st_size > 0):
        import trimesh
        asset_dir = ASSETS_PATH / "objects" / _BOX_ID
        glb = asset_dir / "visual" / "base0.glb"
        mesh = trimesh.load(str(glb), force="mesh")
        # Per-axis pre-scale to meters so URDF morph scale=1.0.
        mesh.apply_scale(scale_vec)
        # Geometry-only OBJ: rebuild a bare Trimesh (drop UV/material) so
        # export writes a SINGLE .obj file with no sibling .mtl / texture
        # PNG. trimesh writes those siblings non-atomically, and a
        # concurrent reader loading the OBJ would read a half-written
        # texture -> "image file is truncated". The shoebox visual is
        # cosmetic (collision is the primitive box walls below), so a
        # plain untextured mesh is fine and races become impossible.
        geom = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            process=False,
        )
        tmp_obj = visual_obj.with_name(f"{visual_obj.stem}.{os.getpid()}.tmp.obj")
        geom.export(str(tmp_obj))
        os.replace(tmp_obj, visual_obj)

    wall_t = 0.01    # 1 cm wall thickness in world units
    floor_t = 0.01

    # Keep the slab thickness, but align its upper face to the inner visual
    # floor. The fixed fixture can extend below its outer visual surface.
    floor_top = _BOX_INNER_FLOOR_MESH_Y * scale_vec[1]
    floor_size = (ext_m[0], floor_t, ext_m[2])
    floor_center = (cen_m[0], floor_top - floor_t / 2, cen_m[2])

    # Side walls span y from top-of-floor to top-of-bbox.
    wall_y_lo = floor_top
    wall_y_hi = bbox_max_m[1]
    wall_y_size = wall_y_hi - wall_y_lo
    wall_y_center = (wall_y_lo + wall_y_hi) / 2

    # Left/right (mesh-x) walls: thin in x, full in z.
    side_size = (wall_t, wall_y_size, ext_m[2])
    left_center = (bbox_min_m[0] + wall_t / 2, wall_y_center, cen_m[2])
    right_center = (bbox_max_m[0] - wall_t / 2, wall_y_center, cen_m[2])

    # Front/back (mesh-z) walls: thin in z, full in x.
    end_size = (ext_m[0], wall_y_size, wall_t)
    front_center = (cen_m[0], wall_y_center, bbox_max_m[2] - wall_t / 2)
    back_center = (cen_m[0], wall_y_center, bbox_min_m[2] + wall_t / 2)

    def _box_block(name: str, size, center) -> str:
        return (
            f'    <collision name="{name}">\n'
            f'      <origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>\n'
            f'      <geometry>\n'
            f'        <box size="{size[0]} {size[1]} {size[2]}"/>\n'
            f'      </geometry>\n'
            f'    </collision>'
        )

    collisions = "\n".join([
        _box_block("floor", floor_size, floor_center),
        _box_block("left", side_size, left_center),
        _box_block("right", side_size, right_center),
        _box_block("front", end_size, front_center),
        _box_block("back", end_size, back_center),
    ])

    # Same race-safety for the URDF (atomic write). The URDF references the
    # OBJ by name; since the OBJ is written (atomically) before this, any
    # process that sees a complete URDF also sees a complete OBJ.
    if not (urdf_path.exists() and urdf_path.stat().st_size > 0):
        tmp_urdf = urdf_path.with_name(f"{urdf_path.stem}.{os.getpid()}.tmp.urdf")
        tmp_urdf.write_text(
            f"""<?xml version="1.0"?>
<robot name="shoebox">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.50"/>
      <inertia ixx="2e-2" ixy="0" ixz="0" iyy="2e-2" iyz="0" izz="2e-2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_obj.name}"/>
      </geometry>
    </visual>
{collisions}
  </link>
</robot>
"""
        )
        os.replace(tmp_urdf, urdf_path)
    half_ext_m = ext_m / 2.0
    return urdf_path, half_ext_m, bbox_min_m


def _sample_n_xy(
    n: int,
    rng: np.random.RandomState,
    max_outer: int = _SAMPLE_BATCH_ATTEMPTS,
):
    """Rejection-sample n xy positions inside the region with pairwise
    minimum separation."""
    for _outer in range(max_outer):
        positions: list[tuple[float, float]] = []
        ok = True
        for _ in range(n):
            placed = False
            for _ in range(_SAMPLE_POINT_ATTEMPTS):
                x = float(rng.uniform(*_REGION_X))
                y = float(rng.uniform(*_REGION_Y))
                if all(
                    (x - px) ** 2 + (y - py) ** 2 >= _MIN_SEP_M ** 2
                    for px, py in positions
                ):
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return positions
    # Fallback: spread points along the diagonal of the region.
    xs = np.linspace(
        _REGION_X[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_X[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    ys = np.linspace(
        _REGION_Y[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_Y[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    return [(float(xs[i]), float(ys[i])) for i in range(n)]


class DualShoesTaskCore(ShoeNativeMotionMixin, TopDownPickPlaceMixin, BaseTask):
    """Shared shoebox scene, shoe manipulation, and scoring mechanics.

    Public base, assist, and interrupt tasks are sibling scenarios over this
    private core. Interaction variants customize layout and lifecycle without
    inheriting another public scenario merely to reuse robot mechanics.
    """

    INSTRUCTION = "put the shoes in the shoebox"
    OBJECT_SET = ["041_shoe"]
    use_avatar = False

    _DUAL_SHOES_USE_DIFFICULTY_COUNT = True
    _DUAL_SHOES_ALLOW_TABLE_VARIANTS = True

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

    LIFT_HEIGHT = 0.20         # m above grasp before transit
    APPROACH_HEIGHT = 0.22     # m above box rim before lowering
    DROP_HEIGHT = 0.08         # m above box rim at release

    # Both shoes must end up inside the box footprint for success.
    # Shoebox interior is ~16 × 25 cm.  Half the diagonal ≈ 0.149 m;
    # 0.18 m gives some slack for shoes that drop near the rim.
    SUCCESS_DIST_XY_M = 0.18
    # Shoe must end up above the box floor (within 1 cm tolerance).
    # Shoes resting on the box floor sit ~5 cm above the floor (their
    # half-height); a stricter z-floor check would reject those.
    SUCCESS_Z_FLOOR_MARGIN_M = -0.01
    # A placed shoe rests in the box: its lowest point at most this far above
    # the box rim (a shoe lying on the other one stays below it), and held by
    # neither the avatar nor the gripper (closed gripper with the TCP within
    # SUCCESS_HELD_TCP_RADIUS_M of the shoe's bbox centre; the collar grasp
    # sits several cm from it).  Without it a shoe carried over the box
    # counted.
    SUCCESS_REST_MAX_ABOVE_RIM_M = 0.04
    SUCCESS_HELD_TCP_RADIUS_M = 0.12
    BOX_FLOOR_Z_CLEARANCE = 0.001
    SHOE_SPAWN_Z_CLEARANCE = 0.04   # sole ~1 cm above table (origin↔sole ≈0.028 m)
    SHOE_FRICTION = 5.0
    BOX_FRICTION = 2.0
    PRE_GRASP_SETTLE_STEPS = 30
    PRE_CLOSE_SETTLE_STEPS = 30
    CLOSE_SETTLE_STEPS = 80
    FAILED_GRASP_SETTLE_STEPS = 30
    RELEASE_GRIPPER_STEPS = 260
    RELEASE_SETTLE_STEPS = 240
    RELEASE_FORCE_N = _SHOE_RELEASE_FORCE_N
    POST_PLACE_SETTLE_STEPS = 60
    INITIAL_SETTLE_STEPS = 60

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        super().__init__(cfg)
        # Registry/override hook (validates config["object_name"]); the shoe is
        # a size-1 white-list so this always resolves to 041_shoe.
        self.resolve_target_object()
        if self._DUAL_SHOES_USE_DIFFICULTY_COUNT:
            self._num_shoes = self.difficulty_object_count()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _create_table(self, table_height=TABLE_HEIGHT):
        BaseTask._create_rectangular_table(
            self,
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
            table_thickness=self.TABLE_THICKNESS_VALUE,
            leg_radius=self.TABLE_LEG_RADIUS,
            leg_xy=self.TABLE_LEG_XY,
            remember_bounds_before_variant=False,
            allow_variant=self._DUAL_SHOES_ALLOW_TABLE_VARIANTS,
        )

    def _load_robot(self):
        super()._load_robot()
        # Raise the Franka home pose so the EE starts well above the table —
        # avoids the gripper colliding with shoes spawned upright (~21 cm
        # tall). Default home has j4=-150° / j6=+168°; this is the ready
        # pose: shoulder back, elbow up, wrist forward. Non-Franka
        # embodiments keep their own homestate.
        if self.config.get("robot_type") == "franka" and self.robot.right_arm is not None:
            self.robot.right_arm.homestate = [
                0.0, -np.pi / 4, 0.0, -3.0 * np.pi / 4,
                0.0, np.pi / 2, np.pi / 4,
            ]

    def _sample_layout_positions(self, rng):
        """Return ``[(box_xy), ...shoe_xy]`` for this mode.

        Interrupt/assist variants override this to constrain one actor
        while keeping the parent's shoebox + shoe loading logic.
        """
        return _sample_n_xy(1 + int(self._num_shoes), rng)

    def load_actors(self):
        # mesh-y maps to world-z under upright_q; the shoe and shoebox
        # GLBs are both y-up.
        upright_q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_layout_positions(rng)
        (box_x, box_y) = positions[0]
        shoe_xys = positions[1:]

        # ---- Shoebox (static receiver) ----
        # 1-link URDF with primitive walls; visual is the original GLB.
        # fixed=True so the box doesn't slide when shoes drop in — for
        # this task the box is just a target volume, not something the
        # robot touches.
        self._box_scale = _read_scale(_BOX_ID, 0)
        urdf_path, half_ext_mesh_m, bbox_min_mesh_m = _ensure_shoebox_urdf(
            self._box_scale,
        )
        # In mesh-frame meters: half_ext (mesh-x, mesh-y, mesh-z),
        # bbox_min (mesh-x, mesh-y, mesh-z).  After upright_q applied
        # to the URDF body, mesh-y → world-z and mesh-z → world-y, so:
        #   world-x extent = mesh-x extent  → box_w
        #   world-y extent = mesh-z extent  → box_l (along world y)
        #   world-z extent = mesh-y extent  → box_h
        box_w = float(2 * half_ext_mesh_m[0])
        box_h = float(2 * half_ext_mesh_m[1])
        box_l = float(2 * half_ext_mesh_m[2])
        # Random shoebox yaw (z-rotation, composed onto upright_q so the
        # box stays floor-down but rotated about world-z).  Compose order:
        # final = yaw ∘ upright (apply upright first to bring mesh-y →
        # world-z, then yaw spins about world-z).
        box_yaw = float(rng.uniform(*_BOX_YAW_RANGE))
        box_yaw_q = t3d.quaternions.axangle2quat(
            [0.0, 0.0, 1.0], box_yaw, is_normalized=True,
        )
        box_q = t3d.quaternions.qmult(box_yaw_q, upright_q)
        box_q /= np.linalg.norm(box_q)
        # mesh y = 0 is the box floor.  After upright_q the box's mesh
        # origin (at floor level) sits at the URDF's world position; so
        # placing the URDF at world-z = table_top puts the floor on the
        # table.
        self.box_pose = Pose([box_x, box_y, table_top + self.BOX_FLOOR_Z_CLEARANCE], box_q)
        self._box_yaw = box_yaw
        box_urdf_kwargs = dict(
            file=str(urdf_path.absolute()),
            pos=tuple(self.box_pose.p),
            quat=tuple(self.box_pose.q),
            scale=1.0,
            fixed=True,
        )
        # NYX renders a URDF as a sub-scene that re-parses the original file and
        # maps each file link 1:1 onto a Genesis link; Genesis's default
        # merge_fixed_links=True collapses fixed links and breaks that mapping
        # (RuntimeError at scene.build). Keep the un-merged tree only under NYX.
        if str(self.config.get("renderer", "")) == "nyx":
            box_urdf_kwargs["merge_fixed_links"] = False
        box_entity = self.scene.add_entity(
            gs.morphs.URDF(**box_urdf_kwargs),
            material=gs.materials.Rigid(friction=self.BOX_FRICTION),
        )
        self.shoebox = ArticulationActor(box_entity, {}, _BOX_ID)
        # Box rim in world: floor at table_top + ε, walls of height box_h.
        self._box_rim_z = float(table_top + self.BOX_FLOOR_Z_CLEARANCE + box_h)
        self._box_xy_half = np.array([box_w / 2, box_l / 2])

        # ---- Shoes (graspable) ----
        self._shoe_scale = _read_scale(_SHOE_ID, _SHOE_MODEL_ID)
        shoe_extents = np.array(
            _read_model_data(_SHOE_ID, _SHOE_MODEL_ID)["extents"],
            dtype=np.float64,
        )
        # World height after upright_q (mesh-y → world-z).  Shoes spawn
        # in their natural sole-down orientation (mesh-y axis vertical).
        # NOTE: spawning them STANDING (length
        # axis vertical via identity quat) makes every
        # grasp's pre-RRT fail because the existing mesh-frame grasps
        # produce a horizontal gripper approach with an inverted wrist
        # — outside the Franka's IK space at the available home pose.
        # The "easier to grasp" benefit of a taller shoe is offset by
        # the unreachable pre-grasp.  Sole-down + raised home pose (see
        # _load_robot) gives the gripper enough table clearance.
        if isinstance(self._shoe_scale, np.ndarray):
            shoe_h = float(shoe_extents[1] * self._shoe_scale[1])
        else:
            shoe_h = float(shoe_extents[1] * self._shoe_scale)
        self._shoe_h = shoe_h
        self._shoe_center_mesh = _read_bbox_center(_SHOE_ID, _SHOE_MODEL_ID)

        self.shoes = []
        self.shoe_poses = []
        # Random per-shoe yaw about world-z so the toe direction varies
        # seed-to-seed; composed onto upright_q (sole-down).
        for (sx, sy) in shoe_xys:
            yaw = float(rng.uniform(-np.pi, np.pi))
            yaw_q = t3d.quaternions.axangle2quat(
                [0.0, 0.0, 1.0], yaw, is_normalized=True,
            )
            pose_q = t3d.quaternions.qmult(yaw_q, upright_q)
            pose_q /= np.linalg.norm(pose_q)
            # Spawn the shoe's sole ~1 cm above the table so it drops cleanly
            # and rests on the surface.  The mesh origin sits ~0.028 m above
            # the sole, so the old origin-at-table_top+0.02 put the sole ~0.8 cm
            # INTO the table; that spawn-time overlap let the convex hull
            # penetrate ~7 cm and wedge (measured bottom z=0.697
            # vs table 0.765).  A clearance of 0.04 lifts the sole clear with a
            # ~1 cm settle drop (lifting by 0.5*shoe_h instead dropped it ~6 cm
            # and tipped it, regressing the grasp — run5).
            spawn_z = table_top + self.SHOE_SPAWN_Z_CLEARANCE
            pose = Pose([sx, sy, spawn_z], pose_q)
            actor = load_object(
                self.scene, pose, _SHOE_ID, model_id=_SHOE_MODEL_ID,
                convex=True, is_static=False,
                friction=self.SHOE_FRICTION,
            )
            self.shoe_poses.append(pose)
            self.shoes.append(actor)

        # Custom check_success — TargetSpec only handles a single
        # object.  Set a stub so BaseTask.evaluate() prints something
        # useful for the first shoe; check_success below ignores it.
        from ..base_task import TargetSpec
        self.target = TargetSpec(
            object=self.shoes[0].entity,
            position=lambda: to_numpy(self.shoebox.entity.get_pos()).ravel()[:3],
            label="shoebox",
        )

    # ------------------------------------------------------------------
    # Pick + place primitive
    # ------------------------------------------------------------------

    # In-box slot offset (along the box's mesh-x = SHORT axis, world-frame
    # rotated by box yaw).  ±2 cm keeps both drops well inside the rim;
    # the success metric only requires both shoes inside the box footprint.
    SLOT_OFFSET_M = 0.020

    def _box_slot_xy(
        self, slot_idx: int, slot_offset: float | None = None,
    ) -> np.ndarray:
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_quat = to_numpy(self.shoebox.entity.get_quat()).ravel()[:4]
        R_box = t3d.quaternions.quat2mat(box_quat)
        box_short_xy = R_box[:, 0][:2]
        box_short_xy /= max(np.linalg.norm(box_short_xy), 1e-9)
        offset_sign = -1.0 if int(slot_idx) == 0 else +1.0
        offset_m = self.SLOT_OFFSET_M if slot_offset is None else float(slot_offset)
        return (
            np.array([box_pos[0], box_pos[1]])
            + offset_sign * offset_m * box_short_xy
        )

    # ------------------------------------------------------------------
    # Robust top-down pick (live-AABB grip point + object-yaw alignment)
    # ------------------------------------------------------------------

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        """Override the mixin's fixed top-down TCP so every mixin motion
        (`_move_seeded` / `_move_screw`) descends with the jaws rotated to
        ``self._grip_yaw_deg`` about the approach axis.  Set the attribute
        before the pick, reset to 0 after.  yaw=0 → world-Y jaws.

        Accepts (and ignores) the mixin's own `yaw` kwarg — this override
        always uses `self._grip_yaw_deg` instead, but `_move_screw` /
        `_move_cartesian` unconditionally pass `yaw=tcp_yaw` (default
        0.0), so the signature must accept it or every screw/cartesian
        call raises TypeError (caught by scripts/collect.py's episode
        try/except, silently producing 0/N with no grasp-diagnostic
        prints at all — this task's smoke tests must show `[shoe_pick]`
        / `[shoe_dbg]` lines, not just bare FAILs)."""
        yaw = float(getattr(self, "_grip_yaw_deg", 0.0))
        if abs(yaw) < 1e-6:
            return super()._top_down_tcp(pos)
        yr = np.deg2rad(yaw)
        cz, sz = np.cos(yr), np.sin(yr)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        R = self._R_DOWN @ Rz
        return Pose(np.asarray(pos, dtype=float), t3d.quaternions.mat2quat(R))

    @staticmethod
    def _shoe_world_aabb(shoe):
        aabb = to_numpy(shoe.entity.get_AABB()).reshape(-1, 3)
        lo, hi = aabb[0], aabb[1]
        return lo, hi, 0.5 * (lo + hi)

    @staticmethod
    def _shoe_pos_quat(shoe):
        """Live base pose, read from the base LINK rather than the entity.

        After the avatar's kinematic attach (which writes entity
        set_pos/set_quat every frame during the inspect) the entity-level
        pos/quat can stay FROZEN at the detach-time values while the body
        physically falls and settles — the entity origin was measured
        20 cm from the collision AABB centre, impossible for this
        rigid body (its mesh origin is ≤4.5 cm from the AABB centre).
        The link state tracks the simulated body."""
        ent = shoe.entity
        links = getattr(ent, "links", None)
        if links:
            link = links[0]
            return (
                to_numpy(link.get_pos()).ravel()[:3],
                to_numpy(link.get_quat()).ravel()[:4],
            )
        return (
            to_numpy(ent.get_pos()).ravel()[:3],
            to_numpy(ent.get_quat()).ravel()[:4],
        )

    def _object_grip_yaw(self, shoe) -> float:
        """tcp_yaw (deg) that closes the top-down jaws across the shoe's
        measured SHORT horizontal (mesh-x = width) axis.

        Yaw convention (the naive mapping closes the gripper along the shoe's LONG
        axis instead of its thin axis): ``_top_down_tcp`` builds
        ``_R_DOWN @ Rz(tcp_yaw)``. ``_R_DOWN`` is a proper180-degree
        rotation about X; it reverses Y and Z, hence the effective
        horizontal yaw sign. The naive "tcp_yaw = theta - 90" mapping
        (theta = world-yaw of mesh-x) only happens to align the jaws
        with the width axis at yaw multiples of 90° — at every other
        shoe orientation it drifts away from width and toward length,
        worst case (45°/135°) grabbing squarely along the LONG axis.
        Verified analytically and by an exhaustive 15°-step sweep over
        all orientations: "90 - theta" aligns the closing axis with
        mesh-x for every angle (dot product ±1.000 throughout, vs.
        "theta - 90" which was correct only at 0/90/180/270 and
        perfectly WRONG — aligned with the long axis — at 45/135/etc).

        Wrapped to [-90, 90) — the jaws are 180°-symmetric, and raw
        values like -210° made the seeded IK pick contorted wrist
        branches."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        x_axis = R @ np.array([1.0, 0.0, 0.0])
        theta = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
        return ((180.0 - theta) % 180.0) - 90.0

    def _shoe_toe_dir_xy(self, shoe) -> np.ndarray:
        """Unit world-xy direction from heel to toe (mesh +z; the asset's
        orientation_point marks the toe at mesh z=+0.98).  Zero vector if
        the shoe is standing on end (length axis vertical)."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe = R @ np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(toe[:2]))
        if n < 1e-6:
            return np.zeros(2)
        return toe[:2] / n

    def _choose_grip_end(self, shoe) -> float:
        """+1.0 → grip the toe/vamp end, -1.0 → grip the heel end.

        A settled shoe is often PITCHED (one end dipped — load concentrates
        on a sharp CoACD sole piece and that end sinks into the table): the
        dipped end's grip band is partly below the tabletop, so prefer the
        RAISED end when the pitch is significant.  Otherwise pick the end
        whose grip point lies farther from the shoebox centre: the spawn
        separation (0.24 m centre-to-centre) still allows a shoe's toe or
        heel to rest against a yawed box corner, and a grip point on that
        end descends onto the box rim and closes on air."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe_z = float(R[2, 2])   # world-z component of mesh +z (toe dir)
        if abs(toe_z) > 0.04:    # pitched ≥ ~2.3°: grip the raised end
            return 1.0 if toe_z > 0 else -1.0
        # Untilted: default to the HEEL.  The heel is 5.5-6.0 cm wide →
        # ≥1 cm/side descent clearance inside the 8 cm jaw span, whereas
        # the 7.3 cm vamp leaves only ~3.5 mm/side and the descending
        # fingers graze the walls, pitching + sinking the shoe before the
        # close (every vamp-end pick of an untilted shoe
        # fails with dz≈0 while every heel-end pick lifts).  Fall back
        # to the vamp only when the heel grip point sits near the shoebox
        # (its descent would land on the box rim).
        origin_xy = self._shoe_pos_quat(shoe)[0][:2]
        toe_dir = self._shoe_toe_dir_xy(shoe)
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        heel_pt = origin_xy - _SHOE_GRIP_HEELWARD_M * toe_dir
        toe_pt = origin_xy + _SHOE_GRIP_TOEWARD_M * toe_dir
        d_heel = float(np.linalg.norm(heel_pt - box_xy))
        d_toe = float(np.linalg.norm(toe_pt - box_xy))
        if d_heel >= 0.16 or d_heel >= d_toe:
            return -1.0
        return 1.0

    def _shoe_grip_xy(self, shoe, grip_end: float) -> np.ndarray:
        """Grip point for the chosen end: the shoe's MESH ORIGIN shifted
        along the length axis onto the vamp (toe end) or the heel counter
        — both closed-top regions clear of the ankle opening.

        The origin, not the live-AABB centre: the mesh origin sits exactly
        at the body centre at sole level (bounds are x/z-symmetric with
        y=0 at the sole), so origin + offset is EXACT in the shoe frame.
        The AABB centre drifts with yaw (a yawed shoe's AABB support
        inflates by up to W·sinθ·cosθ ≈ 3.6 cm) and with tilt, which put
        the "vamp" grip 7.2 cm from the toe tip — over the collar front —
        on yawed seeds."""
        origin_xy = self._shoe_pos_quat(shoe)[0][:2]
        off = _SHOE_GRIP_TOEWARD_M if grip_end > 0 else -_SHOE_GRIP_HEELWARD_M
        return origin_xy + off * self._shoe_toe_dir_xy(shoe)

    def _close_until_contact(
        self, arm, arm_tag: str, shoe, bite_m: float | None = None,
    ) -> float:
        """Stage the gripper close and stop at FIRST CONTACT.

        Commanding the deep _SHOE_CLOSE_TARGET directly leaves a huge PD
        error after contact and the fingers creep ~15 mm/side through the
        thin CoACD walls (they visibly embed in the shoe;
        a finger-stall detector alone misses it because the
        walls never hard-stop the fingers at 35 N).  Contact signal:
        the SHOE MOVES at first real touch — watch its link pose during a
        fast staged close and freeze at the first displacement, then hold
        a fixed retention squeeze ``bite_m`` (total gap reduction across
        both fingers; default 28 mm = ~7 mm/side — the depth that held
        through the lift/carry without a retry in isolated testing).
        Finger-stall detection is kept as a backup (shoe braced against
        something).  A missed grasp reaches the deep floor target and the
        lift-dz guard catches it as before.

        ONE-SHOT: this data-collection pipeline must never retry a grasp
        (repeated attempts corrupt imitation-learning trajectories), so
        there is no escalating-bite mechanism here — a single fixed depth,
        chosen to favor retention over minimal visual penetration. If it
        fails to hold, the pick fails once and stays failed. Returns the
        held target.
        """
        if bite_m is None:
            # A 28 mm bite visibly embeds the fingers in the shoe
            # during carry/drop.  18 mm is the retention/appearance
            # compromise; override via config ``shoe_bite_m`` (0.028 restores
            # the old deep bite if retention regresses).
            bite_m = float(self.config.get("shoe_bite_m", 0.018))

        def _gap() -> float:
            q = to_numpy(arm.entity.get_qpos()).ravel()
            return float(q[-2] + q[-1])

        p0 = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
        target = 1.0
        prev_gap = _gap()
        stall = 0
        stage = 0
        touched = False
        while target > _SHOE_CLOSE_TARGET:
            target = max(_SHOE_CLOSE_TARGET, target - 0.06)
            self.set_gripper(target, arm_tag)
            for _ in range(8):
                self.step_sim()
            p = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
            # Motion counts as CONTACT only once the jaws are physically
            # near the body (residual settle/graze motion at wide gaps
            # fired a false touch at 68 mm > the 55-58 mm heel,
            # → fingers held 5 mm short of the surface).
            if (
                _gap() < 0.066
                and float(np.linalg.norm(p - p0)) > 0.002
            ):
                touched = True
                break
            gap = _gap()
            stage += 1
            if stage > 2 and prev_gap - gap < 0.0015:
                stall += 1
                if stall >= 2:
                    touched = True
                    break
            else:
                stall = 0
            prev_gap = gap
        # Carry grip: a pure surface pinch cannot lift this shoe — the
        # heel is a horizontal wedge and the clamp squirts it out along
        # the taper (lift dz=0 at 5 mm bite/70 N).  A
        # bite keys the fingertips into the wall compliance (prevents the
        # squirt) while staying visually near-surface — the position
        # target still bounds penetration regardless of force, so the
        # fingers can never sink deeper (the old deep target embedded
        # them ~15 mm/side).  Stiff PD + raised cap give the bite real
        # retention.
        hold = max(_SHOE_CLOSE_TARGET, (_gap() - bite_m) / 0.08)
        update_dofs_kp_kv_compat(
            arm.entity, arm._finger_dof_indices,
            kp_value=20000.0, kv_value=400.0,
        )
        update_dofs_force_range_compat(
            arm.entity, arm._finger_dof_indices,
            lower_value=-70.0, upper_value=70.0,
        )
        self.set_gripper(hold, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()
        return hold

    def _topdown_pick_shoe(self, shoe, arm_tag: str, label: str) -> bool:
        """PUBLIC acquisition, physical close and checked retained lift."""
        return self._shoe_native_pick(shoe, arm_tag, label)

    def _pick_and_place_shoe(
        self, shoe, arm_tag: str, label: str, slot_idx: int,
    ) -> bool:
        """One physical grasp with native fixture, arrival and release checks."""
        arm = self.robot.get_arm(arm_tag)
        for _ in range(self.PRE_GRASP_SETTLE_STEPS):
            self.step_sim()
        self.open_gripper(arm_tag)
        update_dofs_kp_kv_compat(
            arm.entity, arm._finger_dof_indices,
            kp_value=_SHOE_FINGER_KP, kv_value=_SHOE_FINGER_KV,
        )
        update_dofs_force_range_compat(
            arm.entity, arm._finger_dof_indices,
            lower_value=-_SHOE_FINGER_FORCE_N, upper_value=_SHOE_FINGER_FORCE_N,
        )
        try:
            with self._shoe_motion_scene(shoe, arm_tag):
                if not self._topdown_pick_shoe(shoe, arm_tag, label):
                    return False
                return self._shoe_native_place(shoe, arm_tag, label, slot_idx)
        except _ShoeMotionRejected:
            self.plan_success = False
            return False
        finally:
            self._grip_yaw_deg = 0.0

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Pre-task object settle; excluded from recordings.  NOTE: a
        # settled shoe's hull may legitimately rest 1-2.5 cm below the
        # nominal tabletop (load concentrates on a sharp CoACD sole
        # piece); re-seating cannot fix it (the sink is the contact
        # equilibrium — re-drops deterministically reproduce it), so the
        # pick anchors its grasp height to the live AABB TOP instead.
        with self.suppress_recording():
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"

        # Sort shoes so the closer-to-box one goes first; leaves more
        # room for the second drop and avoids stacking exactly.
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        order = sorted(
            range(len(self.shoes)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.shoes[i].entity.get_pos()).ravel()[:2] - box_xy,
            )),
        )

        any_success = False
        for slot, idx in enumerate(order):
            # slot 0/1 → -/+ side of the box short axis.  First (closer)
            # shoe takes the "-short" slot; second takes the "+short"
            # slot — purely cosmetic, the success criterion is symmetric.
            label = chr(ord("A") + idx)
            ok = self._pick_and_place_shoe(
                self.shoes[idx], arm_tag, label, slot_idx=slot,
            )
            any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _shoe_resting(self, shoe, pos) -> bool:
        """Success guard for one shoe (see SUCCESS_REST_MAX_ABOVE_RIM_M)."""
        center = np.asarray(pos, dtype=float)
        bottom = float(center[2])
        try:
            aabb = np.asarray(to_numpy(shoe.entity.get_AABB()), dtype=float)
            if aabb.shape == (2, 3) and np.all(np.isfinite(aabb)):
                center = 0.5 * (aabb[0] + aabb[1])
                bottom = float(aabb[0, 2])
        except Exception:
            pass
        rim_z = getattr(self, "_box_rim_z", None)
        if rim_z is not None and bottom > float(rim_z) + self.SUCCESS_REST_MAX_ABOVE_RIM_M:
            return False
        return not held_by_anyone(
            self, shoe, center, radius_m=self.SUCCESS_HELD_TCP_RADIUS_M,
        )

    def _shoe_in_box_metrics(self) -> dict:
        metrics = {}
        if not self.shoes or self.shoebox is None:
            return metrics

        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_xy = box_pos[:2]
        box_floor_z = self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        metrics["shoebox_pos"] = box_pos.tolist()
        metrics["shoebox_floor_z_threshold"] = float(box_floor_z)
        metrics["success_dist_xy_threshold"] = float(self.SUCCESS_DIST_XY_M)

        all_in = True
        for i, shoe in enumerate(self.shoes):
            label = chr(ord("A") + i)
            pos = np.asarray(self._shoe_pos_quat(shoe)[0])
            dist_xy = float(np.linalg.norm(pos[:2] - box_xy))
            above_floor = bool(pos[2] > box_floor_z)
            resting = bool(self._shoe_resting(shoe, pos))
            in_box = bool(
                dist_xy <= self.SUCCESS_DIST_XY_M and above_floor and resting
            )
            all_in = all_in and in_box
            prefix = f"shoe_{label.lower()}"
            metrics[f"{prefix}_pos"] = pos.tolist()
            metrics[f"{prefix}_dist_xy"] = dist_xy
            metrics[f"{prefix}_above_floor"] = above_floor
            metrics[f"{prefix}_resting"] = resting
            metrics[f"{prefix}_in_box"] = in_box
        metrics["all_shoes_in_box"] = bool(all_in)
        return metrics

    def check_success(self) -> bool:
        """Both shoes must end up within `SUCCESS_DIST_XY_M` (xy) of
        the shoebox centre, at z above the box floor, and resting there
        (``_shoe_resting``)."""
        if not self.plan_success:
            return False
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_xy = box_pos[:2]
        # Floor of the box is at table_top (URDF placed at table_top + ε).
        box_floor_z = self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        all_in = True
        for i, sh in enumerate(self.shoes):
            p = np.asarray(self._shoe_pos_quat(sh)[0])
            dxy = float(np.linalg.norm(p[:2] - box_xy))
            above_floor = p[2] > box_floor_z
            ok = (dxy <= self.SUCCESS_DIST_XY_M and above_floor
                  and self._shoe_resting(sh, p))
            if not ok:
                all_in = False
        return all_in


class PlaceDualShoes(DualShoesTaskCore):
    """Public robot-only dual-shoes task."""
