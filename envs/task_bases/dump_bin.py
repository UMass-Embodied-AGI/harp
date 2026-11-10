"""Dump-bin task: Franka picks loose objects off the table and drops each
into a stationary trash can placed on the floor beside the table.

Pure physics — no kinematic teleport.  Uses ``TopDownPickPlaceMixin``'s
proven ``_move_seeded`` (pre-grasp transit) / ``_move_screw`` (descent +
lift + drop) / ``_move_cartesian`` (slow Cartesian interpolation for the
horizontal transport leg, which keeps gripper PD steady so the cube
doesn't slip mid-transit).

Avatar disabled (``use_avatar = False``) — robot-only.
"""

import json
import numpy as np
import transforms3d as t3d

import genesis as gs

from ..base_task import BaseTask, TABLE_HEIGHT
from ..genesis_compat import (
    mesh_frame_kwargs,
    update_dofs_force_range_compat,
    update_dofs_kp_kv_compat,
)
from ..utils import (
    Pose, create_primitive, load_object, to_numpy, ASSETS_PATH,
)
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin, PickSpec, PlaceSpec
from .dump_bin_motion import DumpBinMotionMixin, DumpMotionRejected

_BIG_BIN_ID = "011_dustbin"
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

# Target objects live in the central catalog (envs/object_catalog.py),
# referenced via the ``OBJECT_SET`` pool set on the class below.


def _read_extents(name: str, model_id: int = 0):
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        d = json.load(f)
    return np.array(d["extents"], dtype=np.float64)


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> float:
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_center(name: str, model_id: int = 0) -> np.ndarray:
    """Mesh-origin → bbox-centre offset, **scaled** to world units (mesh
    ``center`` is in unscaled mesh units; load_object applies the asset scale,
    so the offset must be scaled to match — without this, objects whose mesh
    centre is far from the origin, e.g. 100_seal, spawn ~1 m off the table)."""
    md = _read_model_data(name, model_id)
    center = np.array(md.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    raw = md.get("scale", 1.0)
    scale = np.array(raw if isinstance(raw, (list, tuple)) else [raw, raw, raw], dtype=np.float64)
    return center * scale


class DumpBin(DumpBinMotionMixin, TopDownPickPlaceMixin, BaseTask):
    """Pick objects off the table and drop them into a trash can on the floor."""

    INSTRUCTION = "place each object into the trash can beside the table"
    OBJECT_SET = "dump_bin_pick_pool"  # no milk carton
    use_avatar = False

    # 011_dustbin at anisotropic scale (mesh-x ×0.12, mesh-y ×0.06, mesh-z ×0.12):
    #   raw extents (3.244, 3.694, 2.198) × scale → (0.389, 0.222, 0.264)
    #   after upright_q (mesh-y → world-z) + Rz(90°):
    #     world-x = mesh-z × 0.12 = 0.264 m
    #     world-y = mesh-x × 0.12 = 0.389 m
    #     height  = mesh-y × 0.06 = 0.222 m
    #   Net: 26 cm wide × 39 cm deep × 22 cm tall — opening doubled in
    #   each axis (4× area) so it's easy to throw objects in.
    BIG_BIN_SCALE_XY = 0.12
    BIG_BIN_SCALE_Z  = 0.06

    OBJECT_HALF = 0.020   # 4 cm cubes
    OBJECT_FRICTION = 4.0
    # At catalog friction 4.0 the held jar and apple creep out of the closed
    # jaws at a constant rate, independent of arm motion (measured drift
    # trace); Genesis contact rows soften with friction. 1.0 seats the pads.
    WIDTH_CLOSE_LABELS = ('035_apple', '031_jam-jar')
    HELD_CREEP_FRICTION = {"031_jam-jar": 1.0, "035_apple": 1.0}
    OBJECT_DENSITY = 800.0

    # Pick / drop tuning (adapted from place_burger_fries).
    CLOSE_VALUE = 0.30           # set_gripper target after grasp (0=closed, 1=open)
    BELOW_CENTER = 0.005         # grasp z = obj_center - 5 mm so fingers grip lower half
    TRANSPORT_Z_ABOVE_TABLE = 0.18   # lift / transit altitude
    # The bin is on the floor; with Franka base at (0, -0.65, 0.75), the
    # straight-line distance from base to (bin_xy=(0.65,-0.30), z=rim+0.05)
    # is ~0.87 m which exceeds Franka's ~0.85 m reach.  IK fails on the
    # descent and the seeded fallback lands the EE 6-10 cm off-centre →
    # cube released near a wall instead of above the centre.  Raising the
    # hover keeps the descent target reachable.  rim+0.20 (z≈0.43) is the
    # validated sweet spot: a gentle ~0.40 m drop that seats slender objects
    # (100_seal) cleanly — raising to rim+0.30 lets the seal tip over, and
    # lowering toward the rim exceeds Franka reach.  Generalizes the pool
    # from 3/8 (old 0.94 m free-fall) to 6/8 at seed 0.
    DROP_HOVER_Z_ABOVE_RIM = 0.20

    recording_camera_pos    = [1.20, 0.10, 1.85]
    recording_camera_lookat = [0.10, -0.30, 0.85]

    # Table geometry.
    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.5, 0.45)
    TABLE_CENTER_XY = (0.0, -0.35)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.45, -0.75),
        (0.45, -0.75),
        (-0.45, 0.05),
        (0.45, 0.05),
    )

    # Bin geometry.
    BIG_BIN_XY = (0.65, -0.30)
    BIG_BIN_FLOOR_Z = 0.0
    BIG_BIN_WALL_THICKNESS = 0.012
    BIG_BIN_FLOOR_THICKNESS = 0.010
    # Existing visual mesh's inner floor, measured in its unscaled Y axis.
    BIG_BIN_INNER_FLOOR_MODEL_Y = 0.1834166
    BIG_BIN_COLLISION_FLOOR_DEPTH = 0.050

    # Object layout and spawning.
    SPAWN_X_RANGE = (-0.15, 0.25)
    SPAWN_Y_RANGE = (-0.40, 0.00)
    LAYOUT_SAMPLE_ATTEMPTS = 40
    MIN_OBJECT_SPACING = 0.10
    PRIMITIVE_OBJECT_Z_CLEARANCE = 0.002
    MODEL_OBJECT_Z_CLEARANCE = 0.004

    # Robot timing and recovery.
    TABLE_SETTLE_STEPS = 80
    POST_DROP_SETTLE_STEPS = 300
    FAILURE_RECOVERY_STEPS = 40

    # Gripper tuning.
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 200.0

    # Success check.
    SUCCESS_RIM_MARGIN = 0.02
    # Mesh origins / convex hulls can settle a few millimetres below the
    # nominal collision-floor plane while the geometric center is visibly
    # inside the bin.  Keep a small contact tolerance instead of requiring an
    # exact positive world z.
    SUCCESS_FLOOR_Z_MIN = BIG_BIN_FLOOR_Z - 0.02

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        # ``dump_object_name`` is a back-compat alias for the catalog's
        # ``object_name`` forcing knob read by resolve_target_object.
        if cfg.get("dump_object_name") and not cfg.get("object_name"):
            cfg["object_name"] = cfg["dump_object_name"]
        super().__init__(cfg)
        # Catalog objects are the task default.  ``false`` remains available
        # only as a backwards-compatible debug mode for primitive cubes.
        self._dump_random_objects = bool(cfg.get("random_dump_objects", True))
        self._dump_cube_color = cfg.get("dump_cube_color")
        self.num_objects = self._resolve_num_objects()

    def _resolve_num_objects(self) -> int:
        """Number of robot-side objects for this episode."""
        return self.difficulty_object_count()

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
        )

    def load_actors(self):
        self._load_bin()
        self._load_cubes()

    def _load_bin(self):
        # ---- Trash can on the floor, aside the table ----
        # Two co-located entities so the scene LOOKS like the GLB dustbin
        # but the COLLISION is a clean primitive box bin (4 walls + floor).
        # Pure CoACD decomposition of the unclosed dustbin mesh produced
        # floating fragments that caught cubes mid-air; this split keeps
        # the appearance and the physics independent.
        upright_q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        rz90 = t3d.quaternions.axangle2quat([0, 0, 1], np.pi / 2)
        bin_quat = t3d.quaternions.qmult(rz90, upright_q)

        big_extents = _read_extents(_BIG_BIN_ID)
        # mesh-z (along world-x) and mesh-x (along world-y) get the XY scale.
        # mesh-y (along world-z, i.e. height) gets the Z scale.
        big_x_extent = float(big_extents[2]) * self.BIG_BIN_SCALE_XY  # 0.264 m
        big_y_extent = float(big_extents[0]) * self.BIG_BIN_SCALE_XY  # 0.389 m
        big_h        = float(big_extents[1]) * self.BIG_BIN_SCALE_Z   # 0.222 m
        # Bin pushed further out in +x because the doubled footprint
        # (x-extent 26 cm, half = 13 cm) needs more clearance from the
        # table at x_max = 0.5 — anchor at x = 0.65 ⇒ x_min = 0.52.
        big_x, big_y = self.BIG_BIN_XY
        bin_floor_z = float(getattr(self, "BIG_BIN_FLOOR_Z", 0.0))
        # The dustbin GLB has its mesh ORIGIN near the bottom of the bbox
        # (model_data0.json: center.y ≈ 1.825 along mesh-y, the "up" axis;
        #  bbox-y range [-0.022, 3.672]).  Genesis places the mesh ORIGIN at
        # `pos`, so to align the visual bin with the collision-wall floor at
        # z = floor_t = 0.010, we pin mesh origin just above the floor.  With
        # the prior `big_h/2 + 0.005` the visual mesh sat ~11 cm above the
        # collision walls (visual bbox z=[0.115, 0.336] vs collision z=[0,
        # 0.232]), so cubes appeared to settle below or beside the bin.
        floor_t_for_visual = self.BIG_BIN_FLOOR_THICKNESS
        self.big_bin_pose = Pose([big_x, big_y, bin_floor_z + floor_t_for_visual], bin_quat)
        # Rim height (world-z of the bin's top opening) — used to size a
        # reachable hover-and-release above the bin (vs a 0.9 m free-fall).
        self._big_bin_rim_z = float(bin_floor_z + big_h)

        # (a) Visual-only dustbin mesh with the same anisotropic scale.
        bin_mesh = ASSETS_PATH / "objects" / _BIG_BIN_ID / "visual" / "base0.glb"
        # Mesh-frame scale: mesh-x and mesh-z carry XY scale, mesh-y the
        # height scale (mesh-y is up before upright_q rotates it to world-z).
        mesh_scale = (
            self.BIG_BIN_SCALE_XY,  # mesh-x
            self.BIG_BIN_SCALE_Z,   # mesh-y (vertical)
            self.BIG_BIN_SCALE_XY,  # mesh-z
        )
        self.big_bin = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(bin_mesh.absolute()),
                scale=mesh_scale,
                pos=tuple(self.big_bin_pose.p),
                quat=tuple(self.big_bin_pose.q),
                visualization=True,
                collision=False,
                fixed=True,
                **mesh_frame_kwargs(bin_mesh, align=False),
            ),
            material=gs.materials.Rigid(),
        )

        # (b) Primitive box bin (collision-only, visualization=False) so
        # cubes have a clean rectangular cavity to fall into.
        wall_t = self.BIG_BIN_WALL_THICKNESS
        floor_t = self.BIG_BIN_FLOOR_THICKNESS
        support_z = self.big_bin_pose.p[2] + self.BIG_BIN_INNER_FLOOR_MODEL_Y*self.BIG_BIN_SCALE_Z
        floor_depth = max(float(floor_t), self.BIG_BIN_COLLISION_FLOOR_DEPTH)
        self._big_bin_support_z = float(support_z)
        # Extend downward from the visible interior floor. A 10mm slab let
        # falling objects cross its underside and become trapped on the plane.
        create_primitive(
            self.scene, "box",
            Pose(p=[big_x, big_y, support_z - floor_depth / 2]),
            size={"half_size": (big_x_extent / 2, big_y_extent / 2, floor_depth / 2)},
            color=(0.30, 0.30, 0.32), is_static=True, visual=False,
        )
        print(f'[dump-physical] bin support_z={support_z:.6f} depth={floor_depth:.3f}', flush=True)
        for sx, sy, hx, hy in [
            (+1, 0, wall_t / 2, big_y_extent / 2),
            (-1, 0, wall_t / 2, big_y_extent / 2),
            (0, +1, big_x_extent / 2, wall_t / 2),
            (0, -1, big_x_extent / 2, wall_t / 2),
        ]:
            wx = big_x + sx * (big_x_extent / 2 + wall_t / 2)
            wy = big_y + sy * (big_y_extent / 2 + wall_t / 2)
            create_primitive(
                self.scene, "box",
                Pose(p=[wx, wy, bin_floor_z + big_h / 2 + floor_t]),
                size={"half_size": (hx, hy, big_h / 2)},
                color=(0.30, 0.30, 0.32), is_static=True, visual=False,
            )
        self._big_bin_rim_z   = bin_floor_z + big_h + floor_t
        self._big_bin_xy_half = np.array([big_x_extent / 2, big_y_extent / 2])

    def _sample_cube_xys(self):
        """Return ``(xs, ys)`` arrays of length ``num_objects`` sampled
        uniformly in the 0.4 m × 0.4 m forward zone with min-spacing
        rejection.  Subclasses can override to constrain individual
        cubes (e.g. force one cube into an avatar-reachable subzone).
        """
        SX_LO, SX_HI = self.SPAWN_X_RANGE
        SY_LO, SY_HI = self.SPAWN_Y_RANGE
        spawn_xs = np.random.uniform(SX_LO, SX_HI, self.num_objects)
        spawn_ys = np.random.uniform(SY_LO, SY_HI, self.num_objects)
        min_spacing_sq = self.MIN_OBJECT_SPACING ** 2
        for _ in range(self.LAYOUT_SAMPLE_ATTEMPTS):
            ok = True
            for i in range(self.num_objects):
                for j in range(i + 1, self.num_objects):
                    if (spawn_xs[i] - spawn_xs[j]) ** 2 + \
                            (spawn_ys[i] - spawn_ys[j]) ** 2 < min_spacing_sq:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            spawn_xs = np.random.uniform(SX_LO, SX_HI, self.num_objects)
            spawn_ys = np.random.uniform(SY_LO, SY_HI, self.num_objects)
        return spawn_xs, spawn_ys

    def _load_cubes(self):
        """Spawn the N dump objects on the table.  Layout from
        ``_sample_cube_xys`` (overridable).  Center (0.05, -0.20);
        bounds chosen to (a) keep objects on the table (x ∈ [-0.5, 0.5],
        y ∈ [-0.8, 0.1]), (b) stay clear of the bin at x ≥ 0.52, and
        (c) stay inside Franka's reach (base at (0, -0.65, 0.75) ⇒
        farthest corner ≈ 0.66 m, well under 0.85 m).
        """
        spawn_xs, spawn_ys = self._sample_cube_xys()
        self.objects = []
        self._object_pick_radii = {}
        self._object_labels = {}
        self._object_model_ids = {}
        self._object_close_values = {}
        self._object_topdown_grasp_z_offsets = {}
        self._object_topdown_approach_xy_offsets = {}
        self._object_topdown_preopen_steps = {}
        forced = bool(self.config.get("object_name") or self.config.get("dump_object_name"))
        if forced or self._dump_random_objects:
            self._episode_dump_specs = [
                entry.as_spec()
                for entry in self.resolve_target_objects(self.num_objects, replace=False)
            ]
        else:
            self._episode_dump_specs = [
                {
                    "object_id": "cube",
                    "kind": "primitive",
                    "prim_half": self.OBJECT_HALF,
                    "friction": self.OBJECT_FRICTION,
                }
                for _ in range(self.num_objects)
            ]
        for i in range(self.num_objects):
            actor, radius, label = self._spawn_dump_object(
                float(spawn_xs[i]), float(spawn_ys[i]), i,
            )
            self.objects.append(actor)
            self._object_pick_radii[id(actor)] = float(radius)
            self._object_labels[id(actor)] = str(label)
            self._object_model_ids[id(actor)] = int(
                self._last_dump_object_spec.get("model_id", 0)
            )
            if "close_value" in self._last_dump_object_spec:
                self._object_close_values[id(actor)] = float(self._last_dump_object_spec["close_value"])
            if "topdown_grasp_z_offset" in self._last_dump_object_spec:
                self._object_topdown_grasp_z_offsets[id(actor)] = float(
                    self._last_dump_object_spec["topdown_grasp_z_offset"]
                )
            if "topdown_approach_xy_offset" in self._last_dump_object_spec:
                self._object_topdown_approach_xy_offsets[id(actor)] = tuple(
                    self._last_dump_object_spec["topdown_approach_xy_offset"]
                )
            if "topdown_preopen_steps" in self._last_dump_object_spec:
                self._object_topdown_preopen_steps[id(actor)] = int(
                    self._last_dump_object_spec["topdown_preopen_steps"]
                )

    def _sample_dump_object_spec(self, idx: int) -> dict:
        # The episode list is sampled once without replacement so a normal
        # multi-object scene contains distinct target types.  A forced object
        # override is intentionally repeated by ``resolve_target_objects``.
        specs = getattr(self, "_episode_dump_specs", None)
        if specs is None:
            specs = [
                entry.as_spec()
                for entry in self.resolve_target_objects(self.num_objects, replace=False)
            ]
            self._episode_dump_specs = specs
        spec = dict(specs[int(idx)])
        object_id = str(spec.get("object_id", ""))
        if object_id in self.HELD_CREEP_FRICTION:
            spec["friction"] = float(self.HELD_CREEP_FRICTION[object_id])
        if "topdown_close_value" in spec:
            spec["close_value"] = float(spec["topdown_close_value"])
        if object_id == "035_apple":
            # Narrowing below36mm pressed the fruit downward during lift.
            # Retain this aperture while selecting the contact direction.
            spec["close_value"] = 0.45
        elif object_id == "031_jam-jar":
            # The jar's body widens above its geometric midpoint.  The generic
            # catalog already records this +2 cm grasp bias; apply it to the
            # dump task's Cartesian top-down picker as well.
            spec["topdown_grasp_z_offset"] = float(
                spec.get("grasp_z_offset", 0.02)
            )
            if int(spec.get('model_id', 0)) == 0:
                # The60mm gap lost pad pressure as the shoulder pivoted.
                # Add1mm closing travel per finger at the same shoulder pose.
                spec['close_value'] = .725
        return spec

    def _spawn_dump_object(self, x: float, y: float, idx: int):
        spec = self._sample_dump_object_spec(idx)
        self._last_dump_object_spec = spec
        object_id = str(spec["object_id"])
        if spec.get("kind") == "primitive":
            z = self.TABLE_TOP_Z + self.OBJECT_HALF + self.PRIMITIVE_OBJECT_Z_CLEARANCE
            color = self._dump_cube_color
            if color is None:
                color = (0.85, 0.30 + 0.10 * (idx % 3), 0.20)
            actor = create_primitive(
                self.scene, "box",
                Pose([x, y, z]),
                size={"half_size": (self.OBJECT_HALF,) * 3},
                color=tuple(color),
            )
            return actor, self.OBJECT_HALF, "cube"

        model_id = int(spec.get("model_id", 0))
        quat = np.asarray(spec.get("quat", _Q_UPRIGHT), dtype=np.float64)
        scale = _read_scale(object_id, model_id)
        extents = _read_extents(object_id, model_id) * scale
        world_extents = np.abs(t3d.quaternions.quat2mat(quat)) @ extents
        desired_center = np.array([
            x,
            y,
            self.TABLE_TOP_Z + float(world_extents[2]) / 2.0 + self.MODEL_OBJECT_Z_CLEARANCE,
        ], dtype=np.float64)
        center_offset = t3d.quaternions.quat2mat(quat) @ _read_center(object_id, model_id)
        actor = load_object(
            self.scene,
            Pose(desired_center - center_offset, quat),
            object_id,
            model_id=model_id,
            convex=True,
            is_static=False,
            friction=float(spec.get("friction", self.OBJECT_FRICTION)),
        )
        radius = float(min(world_extents[0], world_extents[1]) / 2.0)
        return actor, radius, object_id

    def _object_pos(self, actor) -> np.ndarray:
        if hasattr(actor, "get_pos"):
            return to_numpy(actor.get_pos()).ravel()[:3]
        if hasattr(actor, "get_pose"):
            return np.asarray(actor.get_pose().p, dtype=float).ravel()[:3]
        if hasattr(actor, "entity") and hasattr(actor.entity, "get_pos"):
            return to_numpy(actor.entity.get_pos()).ravel()[:3]
        raise AttributeError(f"cannot read position from {type(actor).__name__}")

    def _dump_object_center(self, actor) -> np.ndarray:
        """Return the live geometric center used for both picking and scoring."""
        label = str(getattr(self, "_object_labels", {}).get(id(actor), "cube"))
        if label == "cube":
            return self._object_pos(actor)
        model_id = int(
            getattr(self, "_object_model_ids", {}).get(id(actor), 0)
        )
        return self._get_object_world_center(actor, label, model_id)

    # ------------------------------------------------------------------
    # Gripper PD boost (same recipe as place_burger_fries)
    # ------------------------------------------------------------------

    def _boost_finger_pd(self, arm_tag: str = "right"):
        arm = self.robot.get_arm(arm_tag)
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=self.FINGER_KP,
            kv_value=self.FINGER_KV,
        )
        update_dofs_force_range_compat(
            arm.entity,
            arm._finger_dof_indices,
            lower_value=-self.FINGER_FORCE_LIMIT,
            upper_value=+self.FINGER_FORCE_LIMIT,
        )

    # ------------------------------------------------------------------
    # Pick + drop one cube via the shared TopDownPickPlaceMixin helper.
    # `pick_and_place` handles IK-seed continuity (re-seed from MEASURED
    # qpos at the start of every pick), pre-grasp → grasp branch-flip
    # refinement (post-descent XY correction), and a single straight-line
    # `_move_screw` for the lateral place-transit (avoids the
    # `_move_cartesian` PD-undershoot that left earlier drops on one
    # bin edge instead of the centre).
    # ------------------------------------------------------------------

    # Side-grasp constants shared by the checked side-transport helpers in
    # dump_bin_milk.py (also used by the 031_jam-jar model-0 path).
    MILK_SIDE_GRASP_NAME = "grasp_065"
    MILK_SIDE_HOLD_STEPS = 100

    def _pick_and_drop_cube(self, cube_entity, arm_tag: str, label: str = "cube") -> bool:
        previous = getattr(self, "_dump_active_actor", None)
        self._dump_active_actor = cube_entity
        try:
            return self._pick_and_drop_cube_impl(cube_entity, arm_tag, label)
        except DumpMotionRejected:
            return False
        finally:
            self._dump_active_actor = previous

    def _pick_and_drop_cube_impl(self, cube_entity, arm_tag: str, label: str = "cube") -> bool:
        big_xy = self.big_bin_pose.p[:2]
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE

        pick_radius = float(getattr(self, "_object_pick_radii", {}).get(id(cube_entity), self.OBJECT_HALF))
        pick_label = str(getattr(self, "_object_labels", {}).get(id(cube_entity), label))
        pick_model_id = int(
            getattr(self, "_object_model_ids", {}).get(id(cube_entity), 0)
        )

        def get_pick_center():
            return self._dump_object_center(cube_entity)

        close_value = getattr(self, "_object_close_values", {}).get(id(cube_entity))
        if pick_label == "048_stapler" and self.config.get("robot_type") == "franka":
            # Both permitted stapler yaws close along world Y. The generic
            # 10 mm finger-position floor barely contacts this narrow body.
            entity = getattr(cube_entity, "entity", cube_entity)
            aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
            if aabb.shape != (2, 3) or not np.all(np.isfinite(aabb)):
                raise DumpMotionRejected("invalid stapler acquisition bounds")
            width = float(aabb[1, 1] - aabb[0, 1])
            if width <= 0:
                raise DumpMotionRejected("invalid stapler acquisition width")
            close_value = float(np.clip((width - 0.008) / 0.08, 0.0, 1.0))
            print(f"[dump_bin] stapler width={width:.4f}m, close={close_value:.3f}")
        if pick_label in self.WIDTH_CLOSE_LABELS and self.config.get("robot_type", "franka") == "franka":
            # Close to the measured cross-section minus 8 mm (4 mm per pad), the
            # rule that holds the stapler. The fixed commands squeezed the jar
            # 6 mm and the apple 9 mm per side and both crept out of the jaws.
            entity = getattr(cube_entity, "entity", cube_entity)
            aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
            if aabb.shape != (2, 3) or not np.all(np.isfinite(aabb)):
                raise DumpMotionRejected(f"invalid {pick_label} acquisition bounds")
            width = float(np.min(aabb[1, :2] - aabb[0, :2]))
            if width <= 0:
                raise DumpMotionRejected(f"invalid {pick_label} acquisition width")
            close_value = float(np.clip((width - 0.008) / 0.08, 0.0, 1.0))
            print(f"[dump_bin] {pick_label} width={width:.4f}m, close={close_value:.3f}")
        if (pick_label == "086_woodenblock" and pick_model_id == 1
                and self.config.get("robot_type") == "franka" and close_value is not None):
            # Model 1 is narrower than the inherited model-0 grip opening.
            # The default top-down jaws close along world Y; size only a
            # non-contacting command from that live physical cross-section.
            entity = getattr(cube_entity, "entity", cube_entity)
            aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
            width = float(aabb[1, 1] - aabb[0, 1])
            if 0.08 * float(close_value) >= width:
                close_value = float(np.clip((width - 0.008) / 0.08, 0.0, 1.0))
                print(f"[dump_bin] block width={width:.4f}m, close={close_value:.3f}")

        pick = PickSpec(
            get_center=get_pick_center,
            radius=pick_radius,
            label=pick_label,
            close_value=close_value,
            grasp_z_offset=getattr(
                self, "_object_topdown_grasp_z_offsets", {}
            ).get(id(cube_entity), 0.0),
            approach_xy_offset=getattr(
                self, "_object_topdown_approach_xy_offsets", {}
            ).get(id(cube_entity), (0.0, 0.0)),
            preopen_steps=getattr(
                self, "_object_topdown_preopen_steps", {}
            ).get(id(cube_entity), 0),
            require_lift=True,
            min_lift_delta=0.03,
            transport_retention_tolerance=0.03,
        )
        # Start with the nominal low hover. The family release planner checks
        # the tabletop and can choose a reachable pose up to 20cm higher;
        # actual arrival, retention and final support are required separately.
        rim_z = float(getattr(self, "_big_bin_rim_z", self.TABLE_TOP_Z))
        drop_from_z = rim_z + self.DROP_HOVER_Z_ABOVE_RIM
        place = PlaceSpec(
            pos=np.array([big_xy[0], big_xy[1], drop_from_z], dtype=float),
            label="bin",
            transport_z=transport_z,
            drop_from_z=drop_from_z,
            release=True,
            # Floor bin is at the Franka reach edge — the straight screw descent
            # crosses a wrist singularity and misplaces (round objects roll out).
            # Seeded IK descent preserves the branch and lands centred.
            descent_seeded=True,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False

        # Settle so the cube finishes falling before check_success peeks at it.
        for _ in range(self.POST_DROP_SETTLE_STEPS):
            self.step_sim()

        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _ensure_final_gripper_z(self, arm_tag: str = "right") -> None:
        """Optionally lift the final TCP above a configured world-z floor."""
        if "min_final_gripper_z" not in self.config:
            return
        arm = self.robot.get_arm(arm_tag)
        ee_pos = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        min_z = float(self.config["min_final_gripper_z"])
        final_xy = self.config.get("final_gripper_xy")
        if final_xy is None and ee_pos[2] >= min_z:
            return
        target = ee_pos.copy()
        if final_xy is not None:
            target[0] = float(final_xy[0])
            target[1] = float(final_xy[1])
        target[2] = max(float(target[2]), min_z)
        link = tcp_to_link_pose(self._top_down_tcp(target), arm.tcp_offset)
        try:
            result = self._move_seeded(link.to_pose7(), arm_tag)
        except Exception:
            result = None
        after_seed = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        if result is None or float(np.linalg.norm(after_seed[:2] - target[:2])) > 0.02:
            try:
                self._move_cartesian(
                    after_seed,
                    target,
                    arm_tag,
                    n_steps=int(self.config.get("final_gripper_lift_steps", 30)),
                    sim_per_step=int(self.config.get("final_gripper_lift_sim_per_step", 10)),
                )
            except Exception:
                pass
        final_ee = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        xy_err = float(np.linalg.norm(final_ee[:2] - target[:2]))
        z_ok = bool(final_ee[2] >= min_z)
        self._final_gripper_pose_check = {
            "target": target.tolist(),
            "actual": final_ee.tolist(),
            "xy_err": xy_err,
            "xy_threshold": float(self.config.get("final_gripper_xy_threshold", 0.05)),
            "z_ok": z_ok,
        }
        print(f"[dump_bin] final gripper target={target} actual={final_ee}", flush=True)
        for _ in range(int(self.config.get("final_gripper_hold_steps", 60))):
            self.step_sim()

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # Settle objects on the table (pre-task; excluded from recordings).
        with self.suppress_recording():
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()

        # A failed manipulation is incomplete even if an object was knocked
        # into the bin. Keep geometric scoring separate from this return value.
        for i, cube in enumerate(self.objects):
            ok = self._pick_and_drop_cube(cube, arm_tag)
            if not ok:
                return False
        self._ensure_final_gripper_z(arm_tag)
        return True

    # ------------------------------------------------------------------
    # Success: count cubes inside the trash can
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        big_x, big_y = self.big_bin_pose.p[:2]
        rim_z = self._big_bin_rim_z
        x_half, y_half = self._big_bin_xy_half

        in_count = 0
        for i, cube in enumerate(self.objects):
            p = self._dump_object_center(cube)
            in_xy   = abs(p[0] - big_x) < x_half and abs(p[1] - big_y) < y_half
            below_r = p[2] < rim_z + self.SUCCESS_RIM_MARGIN
            above_f = p[2] > self.SUCCESS_FLOOR_Z_MIN
            inside  = in_xy and below_r and above_f
            if inside:
                in_count += 1

        threshold = max(1, (self.num_objects + 1) // 2)
        # plan_success can flip False on internal mplib retry fallbacks even
        # when the visible motion succeeds, so final cube positions are the
        # authoritative success signal.
        return in_count >= threshold

    def evaluate(self) -> dict:
        out = super().evaluate()
        check = getattr(self, "_final_gripper_pose_check", None)
        if check is not None:
            xy_ok = bool(check["xy_err"] <= check["xy_threshold"])
            pose_ok = bool(xy_ok and check["z_ok"])
            out["final_gripper_pose"] = {**check, "xy_ok": xy_ok, "success": pose_ok}
            if not pose_ok:
                out["success"] = False
        return out
