"""Put-object-in-cabinet (single Franka, no avatar).

RoboTwin's ``put_object_cabinet`` is bimanual: one arm opens a drawer,
the other places an object inside.  This task collapses it to a single
arm: the arm first opens the cabinet's *top drawer*, then picks an
object off the table and drops it into the open drawer.

Pure physics for the robot — no kinematic teleport, no
``attach_to_gripper``.  Object pick goes through
``BaseTask.select_and_execute_grasp``.  Drawer open is a hand-coded
horizontal handle approach, since the only manual grasp annotation on
``036_cabinet`` is on the closed-drawer handle.

Coordinate notes (cabinet asset 036_cabinet/46653):

  - URDF has a fixed ``joint_0`` with rpy (1.57, 0, -1.57) so URDF-Z
    (the prismatic drawer axis) maps to world -X under an *identity*
    cabinet quat.
  - We rotate the cabinet by Rz(+π/2) so URDF-Z maps to world -Y, i.e.
    the drawers slide *toward* the Franka (which is at world Y < 0,
    facing +Y).  After that:
        cabinet world X span ≈ ±0.43  (left/right of cabinet origin)
        cabinet world Y span ≈ [-0.47, +0.43]  (front-to-back depth)
        cabinet world Z span ≈ [-0.80, +0.81]  (height around origin)
  - We anchor the cabinet so it stands on the floor with the front
    face facing the robot, *behind* the table.

The asset has 3 prismatic drawers (link_1, link_2, link_3) all with
joint origin (0, 0, 0) and the same range [0, 0.66].  They differ only
in mesh translation, so which one is the "top drawer" must be detected
at runtime by reading link world Z.  ``_pick_top_drawer`` does that.
"""

from __future__ import annotations

import json
import numpy as np
import transforms3d as t3d

import genesis as gs

from ..base_task import BaseTask
from ..avatar.collision_checker import sample_capsule_planner_points
from ..genesis_compat import (
    control_dof_force_compat,
    control_dof_force_toward_position_compat,
    hold_dof_position_compat,
    prepare_articulated_urdf_for_renderer,
    set_dofs_kp_kv_compat,
)
from ..utils import (
    Actor, ArticulationActor, Pose, create_primitive, load_object,
    to_numpy, ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin, PickSpec, PlaceSpec
from .cabinet_ordinary_motion import CabinetOrdinaryMotionMixin


_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
_Q_STAPLER = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

# Target objects now live in the central catalog (envs/object_catalog.py);
# this task references them via the ``cabinet_drawer_items`` category set on
# the class below.


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> float:
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_extents(name: str, model_id: int = 0) -> np.ndarray:
    return np.array(_read_model_data(name, model_id)["extents"], dtype=np.float64)


def _cabinet_motion_pose_valid(pose: Pose) -> bool:
    """Reject invalid motion poses before transforms, FK gates, or arm drive."""
    try:
        position = np.asarray(pose.p, dtype=float)
        quaternion = np.asarray(pose.q, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        position.shape == (3,) and quaternion.shape == (4,)
        and np.all(np.isfinite(position)) and np.all(np.isfinite(quaternion))
        and np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4)
    )


def _cabinet_motion_path(position, velocity, n_arm):
    """Return a finite arm-only trajectory, or None for a rejected plan."""
    try:
        position = np.asarray(position, dtype=float)
        velocity = np.zeros_like(position) if velocity is None else np.asarray(velocity, dtype=float)
    except (TypeError, ValueError):
        return None
    if (position.ndim != 2 or not len(position)
            or position.shape[1] != n_arm or velocity.shape != position.shape
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))):
        return None
    return position, velocity


class CabinetTaskCore(TopDownPickPlaceMixin, BaseTask):
    """Shared scene, drawer, manipulation, and scoring mechanics.

    The public base, assist, and interrupt tasks are sibling scenarios over
    this core. Keeping family mechanics here prevents an interaction variant
    from inheriting another variant's scripted lifecycle merely to reuse actor
    loading and robot-control code.
    """

    INSTRUCTION = "place the object inside the open cabinet drawer"
    OBJECT_SET = "tabletop_pick_pool"
    # The open-drawer phase is NOT physically driven on this single-arm
    # config — the cabinet handle's IK lands on a Franka-7DOF qpos that
    # mplib + Genesis IK can't solve consistently (repeated attempts
    # left the arm 15-30 cm off target with non-deterministic
    # joint solutions across runs).  Instead the drawer is pre-opened
    # via cabinet.set_qpos at scene init (kinematic placement is acceptable
    # for non-robot objects) and the robot's job is to
    # pick the table object and drop it in the open drawer.
    use_avatar = False

    _CABINET_TASK_NAME = "put_object_cabinet"
    _CABINET_ALLOW_TABLE_VARIANTS = True

    # ------------------------------------------------------------------
    # Cabinet placement (all world frame).
    # Cabinet is rescaled to 20% → world dims ≈
    # 17 × 18 × 32 cm.  Small enough to sit ON the table at the back
    # edge (no pedestal needed).  Drawer cavity at 20% scale ≈
    # 14 × 16 × 11 cm — comfortably holds the 9.7 cm stapler.
    #
    # Cabinet sits at the BACK of the table with quat = Rz(-π/2) so the
    # drawer slides in world +Y (toward the robot at y=0).  The handle
    # bar runs in world X — perpendicular to the mixin top-down jaws
    # (which span world Y).  Stapler spawns at the FRONT of the table,
    # well in front of the cabinet, so the arm has a clear top-down
    # approach to the stapler with no cabinet body in the line.
    # ------------------------------------------------------------------
    _CABINET_SCALE = 0.20
    # Cabinet sits ON the table at the back-RIGHT corner, OFFSET in
    # +X so the arm's forward reach (along x=0, y<0) doesn't intersect
    # the cabinet body.  Cabinet half-X ≈ 0.086 m, so anchor at x=+0.30
    # puts body x in [+0.214, +0.386] — well clear of the arm's
    # central workspace.  Cabinet body half-height at scale 0.20 ≈
    # 0.16 m; anchor z=0.927 puts bottom at z=0.765 (= table top, no
    # pedestal needed) and top at z=1.089.
    _CABINET_POS = np.array([0.30, -0.65, 0.927])
    _CABINET_X_RANGE = (0.24, 0.36)
    _CABINET_Y_RANGE = (-0.67, -0.61)
    # Rz(-π/2): drawer slides world +Y (toward robot at y=0).
    _CABINET_QUAT = np.array(
        [0.7071067811865476, 0.0, 0.0, -0.7071067811865476],
        dtype=np.float64,
    )

    # Pull qpos: prismatic joint friction in Genesis is very low — even
    # small contact forces from the gripper push the drawer past the
    # commanded waypoint by ~2x (target 0.06 → actual 0.119;
    # target 0.030 → actual 0.045).  Set target 0.050 so overshoot
    # lands drawer at ~0.08 — leaves ~3 cm exposed cavity in front of
    # the cabinet face for the stapler drop, with margin for gripper
    # body without pushing the drawer back.
    _DRAWER_OPEN_QPOS = 0.050
    _DRAWER_OPEN_MIN_QPOS = 0.075
    _DRAWER_OBJECT_CLEARANCE = 0.055
    # Keep the payload well behind the front lip.  Positive clearance moves
    # the target in the drawer-closing direction (world -Y for this asset).
    _DRAWER_PLACEMENT_CLEARANCE = 0.050
    _DRAWER_HELD_INWARD_ALLOWANCE = 0.005
    _DRAWER_SUPPORT_CLEARANCE = 0.005
    _DRAWER_LIMIT_MARGIN = 0.004
    _DRAWER_PULL_TOLERANCE = 0.008
    _DRAWER_HOLD_KP = 10000.0
    _DRAWER_HOLD_KV = 1000.0
    _DRAWER_HOLD_FORCE = 2000.0
    _DRAWER_PULL_DIR = np.array([0.0, +1.0, 0.0])

    # Object spawn — front-LEFT of the table, mirrored away from the
    # cabinet (x in 0.24-0.36) AND its drawer-open approach corridor: the
    # arm opens the drawer first and retreats through x~=cab_x, y~=-0.41,
    # and with a (-0.02, +0.14) band the sweep to/from the handle
    # knocks the target object over.  The left mirror
    # keeps the same base distances, so reachability is unchanged.
    _SPAWN_X = (-0.14, -0.02)
    _SPAWN_Y = (-0.34, -0.20)

    # Phase tunings (small cabinet → tighter clearances).
    _HANDLE_PRE_DIST = 0.08            # m, front retreat before horizontal handle grasp
    _HANDLE_RETREAT_DIST = 0.10        # m, back off after pulling
    _OBJECT_LIFT_HEIGHT = 0.15         # m, lift after picking
    # Release just above drawer floor — stapler half-height (2.4 cm)
    # plus 1 cm safety margin so the gripper doesn't penetrate the
    # floor (which would cause Genesis to eject the stapler downward
    # through the floor, e.g. with hover=0.005).
    _DROP_HOVER_ABOVE_DRAWER_FLOOR = 0.035
    # Family strategy composition. The three switches
    # below keep this core's default behaviour byte-for-byte unchanged
    # for every variant that does not override them; only the
    # interrupt variant sets them (see envs/tasks/put_object_in_cabinet_interruption.py).
    #  - CABINET_PAIR_FRICTION: None -> catalog payload friction and the 4.0
    #    drawer-floor proxy; a float builds both at that pair friction (the
    #    pinned engine softens contact rows as (1+mu^2)*mu^2 and uses the max
    #    of the two geometries, as measured). Family-wide
    #    1.0 regresses assist's ordinary radius-grasp approach (4/5 vs
    #    3/5, 1/5, 1/5), so it is interrupt-only.
    #  - DRAWER_OPENING_MAX_EXTENT: size the opening from the payload's
    #    largest horizontal extent instead of its spawn y-extent (a
    #    top-down grasp and the carry turn rotate long payloads).
    #  - DRAWER_TRANSLATION_PULL: the Franka pure-translation handle pull
    #    (+10 mm handle offset, full close on the bar, rail-follows-hand,
    #    plan_screw translation legs). Keeps the default behaviour for the
    #    base/assist/neutral variants; the interrupt disables it because the
    #    same code rejected seal 18's first pull and stranded stapler 16's
    #    lift there (family_v46 sanity 3/5 vs 5/5).
    CABINET_PAIR_FRICTION = None
    DRAWER_OPENING_MAX_EXTENT = False
    DRAWER_TRANSLATION_PULL = True
    # Straight-pull screw integration steps, tried in order (the first is the
    # original step, so a pull that planned before plans identically).
    DRAWER_PULL_SCREW_QPOS_STEPS = (.01, .0093, .0107)

    # Elevated front-right view: the sight line to the cabinet passes above
    # and to the right of every avatar zone in the assist / interrupt /
    # neutral variants (assist stands right of the cabinet, interrupt leans
    # over the front table edge, neutral works the front-left corner), so the
    # cabinet + open drawer stay visible for the whole episode.  The old
    # [0.50, 0.30, 1.30] view was blocked by the interrupt avatar's body.
    recording_camera_pos = [0.95, +0.35, 1.60]
    recording_camera_lookat = [0.10, -0.50, 0.80]
    # Default side camera ([1.5, -0.3, 1.2]) is blocked by the assist avatar
    # at (0.82, -0.56); this front-right eye-level view clears all variants.
    side_camera_pos = [1.35, +0.55, 1.15]
    side_camera_lookat = [0.05, -0.50, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)
        # config["object_name"] forcing is handled by resolve_target_object.
        self._object_id = "048_stapler"
        self._object_model_id = 0
        self._object_is_primitive = False
        self._object_close_value = None
        self._object_scale = _read_scale(self._object_id, self._object_model_id)
        self._object_extents = _read_extents(self._object_id, self._object_model_id) * self._object_scale
        self._object_world_extents = self._object_extents.copy()
        self._object_radius = float(np.min(self._object_world_extents[:2]) / 2.0)
        self._cabinet_pos = self._CABINET_POS.copy()

    def _sample_cabinet_pos(self) -> np.ndarray:
        """Sample a reachable cabinet pose on the robot-side table edge.

        The orientation stays fixed so the drawer faces inward and opens
        along +Y.  X is biased to the arm's right half, matching the
        non-overlap recipe that keeps the cabinet out of the Franka's
        central approach corridor.
        """
        return np.array([
            float(np.random.uniform(*self._CABINET_X_RANGE)),
            float(np.random.uniform(*self._CABINET_Y_RANGE)),
            float(self._CABINET_POS[2]),
        ], dtype=np.float64)

    def _sample_object_spec(self) -> dict:
        # Sampled (or config["object_name"]-forced) from the catalog category
        # ``cabinet_drawer_items``; .as_spec() yields the legacy dict shape the
        # downstream spawn code in load_actors expects.
        return self.resolve_target_object(randomize=True).as_spec()

    def _object_world_extents_from_quat(self, object_id: str, model_id: int,
                                        quat: np.ndarray) -> np.ndarray:
        local_ext = _read_extents(object_id, model_id) * _read_scale(object_id, model_id)
        R = np.abs(t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64)))
        return R @ local_ext

    def _get_object_world_center(self, actor, object_name, model_id=0):
        """World-frame object centre, robust to the Genesis-1.0.0 z-up GLB load.

        Some GLBs (e.g. 100_seal) load in a frame whose vertical axis doesn't
        match model_data's y-up ``center``/``extents``, so the mixin's
        model_data-derived centre can sit several cm off the real object — the
        top-down radius grasp then closes on air and nothing lifts.  When the
        live simulator AABB centre disagrees with the model_data centre by more
        than 1.5 cm in z, trust the AABB (ground truth from the physics).  For
        well-loaded meshes (stapler, block, cube, …) the two agree, so this is a
        no-op and other objects' grasps are unchanged.
        """
        if getattr(self, "_object_is_primitive", False):
            return np.asarray(actor.get_pose().p, dtype=float)
        model_c = super()._get_object_world_center(actor, object_name, model_id)
        try:
            aabb = to_numpy(actor.entity.get_AABB())
            aabb_c = 0.5 * (aabb[0] + aabb[1])
            if abs(float(aabb_c[2]) - float(model_c[2])) > 0.015:
                return np.asarray(aabb_c, dtype=float)
        except Exception:
            pass
        return model_c

    def _live_object_center(self) -> np.ndarray:
        """Return the live payload center without assuming a mesh origin."""
        if getattr(self, "_object_is_primitive", False):
            return np.asarray(self.object.get_pose().p, dtype=float)
        return self._get_object_world_center(
            self.object, self._object_id, self._object_model_id,
        )

    def _payload_pick_options(self) -> dict:
        options = dict(self._object_topdown_pick_options)
        if (self._object_id != "086_woodenblock" or self._object_model_id != 1
                or self.config.get("robot_type") != "franka"):
            return options
        # Model 1's live square cross-section is ~35 mm wide. Its inherited
        # model-0 close target (0.52) leaves a 41.6 mm opening, so neither
        # finger contacts it. Measure the physical footprint and retain the
        # catalog's 8 mm total squeeze only when that target cannot contact.
        aabb = np.asarray(to_numpy(self.object.entity.get_AABB()), dtype=float)
        yaw_reader = getattr(self, "_object_relative_grasp_yaw", None)
        yaw = float(yaw_reader()) if callable(yaw_reader) else 0.0
        square_projection = abs(float(np.cos(yaw))) + abs(float(np.sin(yaw)))
        width = float(np.min(aabb[1, :2] - aabb[0, :2])) / square_projection
        close = options.get("close_value")
        if close is not None and 0.08 * float(close) >= width:
            options["close_value"] = float(np.clip((width - 0.008) / 0.08, 0.0, 1.0))
            print(f"[cabinet] block width={width:.4f}m, "
                  f"close={float(close):.3f}->{options['close_value']:.3f}")
        return options

    def _sample_object_xy(self, world_extents: np.ndarray,
                          cabinet_pos: np.ndarray) -> tuple[float, float]:
        """Sample around table centre, rejecting starts near the cabinet."""
        obj_clear = float(max(world_extents[0], world_extents[1]) / 2.0 + 0.04)
        cab_half_x = 0.086 + obj_clear
        cab_half_y = 0.090 + obj_clear
        spawn_x = getattr(self, "_spawn_x_range", self._SPAWN_X)
        spawn_y = getattr(self, "_spawn_y_range", self._SPAWN_Y)
        for _ in range(200):
            x = float(np.random.uniform(*spawn_x))
            y = float(np.random.uniform(*spawn_y))
            if (abs(x - cabinet_pos[0]) < cab_half_x
                    and abs(y - cabinet_pos[1]) < cab_half_y):
                continue
            return x, y
        # Deterministic fallback near table centre-left, away from the
        # cabinet's right-edge footprint.
        return -0.10, -0.35

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Match dump_bin's table geometry exactly — cube pick
        # works reliably there with the same Franka.  Top z = 0.765.
        # Y span [-0.80, +0.10] (90 cm front-to-back), X span [-0.50, +0.50].
        BaseTask._create_rectangular_table(
            self,
            table_height=table_height,
            half_size=(0.50, 0.45),
            center_xy=(0.0, -0.35),
            table_thickness=0.05,
            leg_radius=0.025,
            leg_xy=(
                (-0.45, -0.75),
                (0.45, -0.75),
                (-0.45, 0.05),
                (0.45, 0.05),
            ),
            remember_bounds_before_variant=False,
            allow_variant=self._CABINET_ALLOW_TABLE_VARIANTS,
        )

    def load_actors(self):
        self._cabinet_pos = self._sample_cabinet_pos()

        # ---- Cabinet (articulated, fixed base) — sits directly on
        # the table top, no pedestal.
        cabinet_dir = ASSETS_PATH / "objects" / "036_cabinet"
        urdf = None
        for sub in sorted(cabinet_dir.iterdir()):
            if sub.is_dir() and (sub / "mobility.urdf").exists():
                urdf = sub / "mobility.urdf"
                break
        if urdf is None:
            raise RuntimeError(
                f"[{self._CABINET_TASK_NAME}] cabinet URDF not found"
            )

        # The SAPIEN 036_cabinet URDF ships with no <inertial> tags. Under
        # Genesis 1.0.0 the auto-computed inertia is degenerate and the scene
        # nan-crashes on the first settle step. Inject explicit inertials
        # (heavier body, lighter drawers) — no-op on the old Genesis fork.
        urdf, load_scale, merge_fixed_links = prepare_articulated_urdf_for_renderer(
            urdf,
            renderer=self.config.get("renderer", ""),
            scale=self._CABINET_SCALE,
            link_overrides={
                "link_0": {"mass": 8.0, "ixx": 0.3, "iyy": 0.3, "izz": 0.3},
                "link_1": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
                "link_2": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
                "link_3": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
            },
        )

        cabinet_urdf_kwargs = dict(
            file=str(urdf.absolute()),
            pos=tuple(self._cabinet_pos),
            quat=tuple(self._CABINET_QUAT),
            scale=float(load_scale),
            fixed=True,
        )
        if merge_fixed_links is not None:
            cabinet_urdf_kwargs["merge_fixed_links"] = merge_fixed_links
        cabinet_entity = self.scene.add_entity(gs.morphs.URDF(**cabinet_urdf_kwargs))
        self.cabinet = ArticulationActor(cabinet_entity, {}, "036_cabinet")
        # ---- Object on the table ----
        spec = self._sample_object_spec()
        self._object_topdown_pick_options = self.topdown_pick_options(spec)
        if hasattr(self, "_object_grasp_z_offset"):
            self._object_grasp_z_offset = float(
                self._object_topdown_pick_options.get("grasp_z_offset", 0.0)
            )
        self._object_id = str(spec["object_id"])
        self._object_model_id = int(spec.get("model_id", 0))
        self._object_is_primitive = spec.get("kind") == "primitive"
        # Catalog grip-firmness hint (None → radius-based close); firmer for
        # objects that slip the partial-close grip (apple, woodenblock).
        self._object_close_value = self._object_topdown_pick_options["close_value"]
        obj_quat = np.asarray(spec["quat"], dtype=np.float64)

        if self._object_is_primitive:
            # Procedural cube: bbox = (2*half)^3, origin == centre, grasp by
            # radius (the TopDownPickPlaceMixin path needs no mesh/grasp YAML).
            half = float(spec["prim_half"])
            self._object_scale = 1.0
            self._object_extents = np.array([2 * half] * 3, dtype=np.float64)
            self._object_world_extents = self._object_extents.copy()
            self._object_radius = half
            ox, oy = self._sample_object_xy(self._object_world_extents, self._cabinet_pos)
            desired_center = np.array([
                ox, oy, self.TABLE_TOP_Z + half + 0.01,
            ], dtype=np.float64)
            self._object_pose0 = Pose(desired_center, np.array([1.0, 0.0, 0.0, 0.0]))
            entity = create_primitive(
                self.scene, "box", self._object_pose0,
                size={"half_size": (half, half, half)},
                color=(0.85, 0.35, 0.20),
            )
            self.object = Actor(entity, {}, self._object_id)
        else:
            self._object_scale = _read_scale(self._object_id, self._object_model_id)
            self._object_extents = _read_extents(self._object_id, self._object_model_id) * self._object_scale
            self._object_world_extents = self._object_world_extents_from_quat(
                self._object_id, self._object_model_id, obj_quat,
            )
            self._object_radius = float(np.min(self._object_world_extents[:2]) / 2.0)
            ox, oy = self._sample_object_xy(self._object_world_extents, self._cabinet_pos)
            obj_h = float(self._object_world_extents[2])
            desired_center = np.array([
                ox, oy, self.TABLE_TOP_Z + obj_h / 2 + 0.01,
            ], dtype=np.float64)
            center_offset = (
                t3d.quaternions.quat2mat(obj_quat)
                @ self._get_model_center(self._object_id, self._object_model_id)
            )
            origin_pos = desired_center - center_offset
            self._object_pose0 = Pose(origin_pos, obj_quat)
            self.object = load_object(
                self.scene, self._object_pose0, self._object_id,
                model_id=self._object_model_id,
                convex=True, is_static=False,
                friction=(float(spec.get("friction", 4.0))
                          if self.CABINET_PAIR_FRICTION is None
                          else float(self.CABINET_PAIR_FRICTION)),
            )

        # Add support after the payload so its rigid-body/contact ordering is
        # unchanged from the established grasp path.
        self._drawer_floor_proxy = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.115, 0.110, 0.006), pos=(0.0, 0.0, -5.0),
                fixed=True, visualization=False, collision=True,
            ),
            material=gs.materials.Rigid(
                friction=(4.0 if self.CABINET_PAIR_FRICTION is None
                          else float(self.CABINET_PAIR_FRICTION)),
            ),
        )

        # Filled in after scene.build() in ``play_once`` (need link poses
        # at runtime).  Stored as instance state so check_success can use
        # them.
        self._drawer_link = None
        self._drawer_dof_idx = None
        self._drawer_handle_world0 = None
        self._drawer_floor_z = None
        self._drawer_inside_xy_half = None
        self._drawer_open_qpos_target = float(self._DRAWER_OPEN_QPOS)
        self._drawer_held_qpos = None

    # ------------------------------------------------------------------
    # Cabinet helpers — runtime introspection
    # ------------------------------------------------------------------

    def _pick_drawer(self):
        """Identify the *top* prismatic drawer — only one with clear
        top-down handle access (middle/bottom drawers have the next
        drawer above them, blocking the gripper's vertical approach).
        """
        entity = self.cabinet.entity
        # Prismatic joints all live on links named link_1, link_2, link_3.
        prismatic = []
        for link in entity.links:
            name = str(link.name)
            if name.startswith("link_") and name not in ("link_0",):
                z = float(to_numpy(link.get_pos()).ravel()[2])
                prismatic.append((z, link, name))
        prismatic.sort(key=lambda t: t[0])  # ascending z
        if not prismatic:
            raise RuntimeError(
                f"[{self._CABINET_TASK_NAME}] no drawer links found"
            )
        # Top drawer = highest z (last after ascending sort).
        # Note: at qpos=0 all links report cabinet origin z, so sort is
        # tie-stable in URDF order link_1, link_2, link_3 → last = link_3.
        top_z, top_link, top_name = prismatic[-1]
        self._drawer_link = top_link

        # Resolve DOF index for the joint whose child is this link.  Genesis
        # entities expose joints via .joints with names like "joint_1".
        link_idx = top_name.split("_")[-1]
        joint_name = f"joint_{link_idx}"
        dof_idx = None
        for j in entity.joints:
            if str(j.name) == joint_name:
                # DOFs are flat-indexed; for a single-DOF prismatic joint,
                # the joint owns one DOF.  Use the joint's dof_idx_local /
                # dof_start (Genesis convention).
                if hasattr(j, "dof_idx_local") and j.dof_idx_local is not None:
                    dof_idx = int(j.dof_idx_local)
                elif hasattr(j, "dof_start"):
                    dof_idx = int(j.dof_start)
                break
        if dof_idx is None:
            # Fallback: assume joints are added in order so the i-th
            # prismatic joint owns DOF i (joint_1 → dof 0, etc).
            dof_idx = int(link_idx) - 1
        self._drawer_dof_idx = dof_idx

        # Per-drawer geometry — derived from the live link AABB so we
        # don't have to hardcode per-drawer URDF Y.  The drawer body's
        # AABB gives front-face Y (most positive Y in world under our
        # Rz(-π/2) cabinet quat), top/bottom Z, and X span.  The handle
        # bar is on the front face near the top edge.
        s = float(self._CABINET_SCALE)
        cabinet_xyz = self._cabinet_pos
        try:
            aabb = to_numpy(top_link.get_AABB())
            xmin = float(aabb[0, 0]); xmax = float(aabb[1, 0])
            ymin = float(aabb[0, 1]); ymax = float(aabb[1, 1])
            zmin = float(aabb[0, 2]); zmax = float(aabb[1, 2])
            xmid = 0.5 * (xmin + xmax)
            # Handle mesh bounds at scale 0.20 are a thin bar near the
            # drawer AABB's front/top: y ≈ ymax - 5 mm, z ≈ zmax - 13 mm.
            # Using the drawer body's centroid or 75% height lands below
            # the actual bar for a horizontal grasp.
            self._drawer_handle_world0 = np.array([
                xmid, ymax - 0.005, zmax - 0.013,
            ])
            # Cavity floor ≈ 1 cm above drawer bottom (drawer floor
            # thickness); rim ≈ drawer top.
            self._drawer_floor_z = zmin + 0.010
            self._drawer_rim_z = zmax
            # Inside half-extents — shrink AABB by a small wall margin.
            self._drawer_inside_xy_half = np.array([
                max(0.005, 0.5 * (xmax - xmin) - 0.010),
                max(0.005, 0.5 * (ymax - ymin) - 0.010),
            ])
        except Exception as e:
            self._drawer_handle_world0 = np.array([
                float(cabinet_xyz[0]),
                float(cabinet_xyz[1]) + 0.440 * s,
                float(cabinet_xyz[2]) + 0.740 * s,
            ])
            self._drawer_floor_z = float(cabinet_xyz[2]) - 0.20 * s
            self._drawer_rim_z = float(cabinet_xyz[2]) + 0.176 * s
            self._drawer_inside_xy_half = np.array([0.30 * s, 0.28 * s])
        link_pos = to_numpy(top_link.get_pos()).ravel()[:3]
        object_aabb = to_numpy(self.object.entity.get_AABB())
        payload_extents = object_aabb[1, :2] - object_aabb[0, :2]
        payload_depth = (float(np.max(payload_extents))
                         if self.DRAWER_OPENING_MAX_EXTENT
                         else float(payload_extents[1]))
        self._drawer_open_qpos_target = self.object_aware_prismatic_open_target(
            entity,
            self._drawer_dof_idx,
            payload_depth,
            minimum_open=self._DRAWER_OPEN_MIN_QPOS,
            clearance=self._DRAWER_OBJECT_CLEARANCE,
            limit_margin=self._DRAWER_LIMIT_MARGIN,
        )
        self._drawer_held_qpos = None
        print(
            f"[drawer] payload_depth={payload_depth:.4f}m, "
            f"open_target={self._drawer_open_qpos_target:.4f}m"
        )

    def _handle_tcp(self, handle_pos: np.ndarray) -> Pose:
        """Horizontal drawer-handle TCP.

        The handle bar runs in world X.  TCP +Z is the approach axis, so
        set it to world -Y (from the robot/front side into the cabinet).
        TCP +Y is the Franka jaw axis; setting it to world +Z makes the
        fingers close vertically around the handle instead of pinching it
        from above.
        """
        tcp_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # along handle bar
        tcp_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)   # vertical jaw close
        tcp_z = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # horizontal approach
        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        variant = getattr(self, "_handle_approach_variant", None)
        if variant is not None:
            R = self._handle_variant_rotation(*variant) @ R
        return Pose(np.asarray(handle_pos, dtype=float), t3d.quaternions.mat2quat(R))

    # Full-state expert option (off by default; the neutral variant opts in):
    # when the horizontal handle pre-grasp has no collision-free IK because
    # the human or their props occupy the wrist corridor, try the same bar
    # grasp turned about the vertical (yaw), tilted to approach from above
    # (pitch), and/or with the fingers swapped, in order of deviation. Each
    # variant is used only if mplib finds collision-free IK for its pre-grasp,
    # handle grasp and full pull before the arm moves; every motion still
    # goes through the ordinary planned, gated and arrival-checked waypoints.
    HANDLE_APPROACH_VARIANTS = False
    HANDLE_VARIANT_GATE_WAIT_STEPS = 500   # per variant segment (1 s at dt .002)
    HANDLE_APPROACH_VARIANT_ORDER = (
        # (finger swap, yaw deg about world Z, pitch deg down about world X)
        (False, 20, 0), (False, -35, 0), (True, 20, 0), (True, -35, 0),
        (False, -20, 0), (True, -20, 0),
        (False, -35, 20), (True, -35, 20), (False, -20, 20), (True, -20, 20),
        (False, 20, 20), (True, 20, 20),
        (False, -20, 35), (True, -20, 35), (False, -35, 35), (True, -35, 35),
        (False, 0, 35), (True, 0, 35),
    )

    @staticmethod
    def _handle_variant_rotation(flip: bool, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """World rotation applied to the horizontal handle TCP frame."""
        R = np.eye(3)
        if flip:
            # 180 deg about the TCP approach axis (world -Y): swaps the fingers.
            R = t3d.axangles.axangle2mat([0.0, -1.0, 0.0], np.pi) @ R
        R = t3d.axangles.axangle2mat([1.0, 0.0, 0.0], np.deg2rad(pitch_deg)) @ R
        return t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg)) @ R

    def _handle_pre_front(self, handle_grasp: np.ndarray) -> np.ndarray:
        """Pre-grasp TCP: back off along the (possibly turned) approach axis."""
        if getattr(self, "_handle_approach_variant", None) is None:
            return handle_grasp + np.array([0.0, self._HANDLE_PRE_DIST, 0.0])
        approach = self._handle_tcp(handle_grasp).to_matrix()[:3, 2]
        return handle_grasp - approach * float(self._HANDLE_PRE_DIST)

    def _handle_variant_feasible(self, handle_grasp: np.ndarray, arm_tag: str) -> str | None:
        """Collision-free mplib IK for the pre-grasp, grasp and pull (no motion).

        Returns None when feasible, else the name of the first blocked pose."""
        arm = self.robot.get_arm(arm_tag)
        seed = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        self._refresh_avatar_planner_obstacles(arm_tag)
        pull_end = handle_grasp + np.array([0.0, float(self._drawer_open_qpos_target), 0.0])
        for name, position in (("pre", self._handle_pre_front(handle_grasp)),
                               ("grasp", handle_grasp), ("pull", pull_end)):
            pose7 = tcp_to_link_pose(self._handle_tcp(position), arm.tcp_offset).to_pose7()
            try:
                ik = arm.planner.solve_ik(seed, pose7, num_waypoints=2, log=False)
            except RuntimeError:
                return name
            path = _cabinet_motion_path(ik.position, getattr(ik, "velocity", None), arm.n_arm) \
                if ik is not None and ik.success else None
            if path is None:
                return name
            seed = path[0][-1].copy()
        return None

    def _handle_approach_fallback(self, handle_grasp: np.ndarray, arm_tag: str) -> bool:
        """After an IK-rejected or gate-withheld horizontal pre-grasp (nothing
        grasped yet), reach the
        pre-grasp with the first feasible handle-approach variant."""
        if (not self.HANDLE_APPROACH_VARIANTS
                or getattr(self, "_drawer_last_failure", None) not in ("ik", "gate")):
            return False
        # Nothing is grasped yet: every retry plans from the measured joints
        # and runs through the same gate. A gate that waited for a seated or
        # static human is not given its full wait again for each variant.
        previous_cap = getattr(self, "_neutral_aware_max_seg_override", None)
        self._neutral_aware_max_seg_override = self.HANDLE_VARIANT_GATE_WAIT_STEPS
        try:
            blocked = {}
            for variant in self.HANDLE_APPROACH_VARIANT_ORDER:
                self._handle_approach_variant = tuple(variant)
                stage = self._handle_variant_feasible(handle_grasp, arm_tag)
                if stage is not None:
                    blocked[stage] = blocked.get(stage, 0) + 1
                    continue
                pre_front = self._handle_pre_front(handle_grasp)
                print(f"[drawer] handle approach variant flip={int(variant[0])} "
                      f"yaw={variant[1]:+d} pitch={variant[2]:d}", flush=True)
                self.plan_success = True
                if self._drive_to_tcp(
                    pre_front, arm_tag, n_settle=300, tol=0.025,
                    tcp_pose=self._handle_tcp(pre_front),
                ):
                    return True
                if getattr(self, "_drawer_last_failure", None) not in ("ik", "gate"):
                    break   # executed but missed the pre-grasp: stop here
                if getattr(self, "avatar_collided", False):
                    break
        finally:
            self._neutral_aware_max_seg_override = previous_cap
        print(f"[drawer] no handle approach variant reached the pre-grasp; "
              f"IK-blocked stages={blocked}", flush=True)
        self._handle_approach_variant = None
        self.plan_success = False
        return False

    def _drawer_support_surface_z(self) -> float:
        # Cabinet46653's visible bottom panel is0.067411 mesh units above
        # its AABB minimum. Keep the legacy grading reference separate from
        # the physical support surface, as in the interrupt variant.
        return float(self._drawer_floor_z) - 0.010 + 0.067411 * float(self._CABINET_SCALE)

    def _hold_drawer_open(self) -> bool:
        """Keep the opened fixture at its intended PD target during placement."""
        target = (
            self._drawer_open_qpos_target
            if self._drawer_held_qpos is None
            else self._drawer_held_qpos
        )
        ok = hold_dof_position_compat(
            self.cabinet.entity,
            self._drawer_dof_idx,
            target,
            kp=self._DRAWER_HOLD_KP,
            kv=self._DRAWER_HOLD_KV,
            force_limit=self._DRAWER_HOLD_FORCE,
        )
        if ok:
            self.sync_support_proxy(
                self._drawer_floor_proxy, self._drawer_link, self._drawer_support_surface_z(),
            )
        return ok

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _boost_finger_pd(self, arm_tag: str = "right"):
        """Stiffen the gripper finger PD so closed jaws hold the
        stapler.  Arm-joint PD intentionally LEFT at default — bumping
        it (e.g. kp=10000) makes the arm overshoot and throws the
        seeded-IK plan_qpos trajectories off by 14 cm.  Same finger-only
        recipe as ``dump_bin`` and ``place_bread_in_basket``.
        """
        self.configure_finger_pd(arm_tag, force_limit=200.0)

    def _soften_object_gripper_pd(self, arm_tag: str = "right"):
        """Build contact gently before switching to the transport hold."""
        self.configure_finger_pd(arm_tag, force_limit=30.0)

    # ------------------------------------------------------------------
    # Phase A — physics-based drawer open
    # ------------------------------------------------------------------

    def _set_arm_pd(self, arm_tag: str, arm_kp: float, arm_kv: float):
        """Temporarily set the arm-joint PD gains.  Default Franka kp=4000
        is too soft to drive the EE to the cabinet handle (PD finds an
        equilibrium 13–18 cm short of the IK-solved goal qpos).  Bumping
        to kp~15000 lets PD reach the goal without the brittle jerks that
        kp=100000 produces."""
        from envs.robot.base import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
            for j in arm.arm_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None:
                    continue
                kp[idx] = float(arm_kp)
                kv[idx] = float(arm_kv)
            set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)
        except Exception as e:
            pass

    def _drive_to_tcp(self, tcp_pos, arm_tag: str, n_settle: int = 250,
                       tol: float = 0.015, tcp_pose: Pose | None = None,
                       max_joint_delta: float | None = None,
                       straight_pull: bool = False) -> bool:
        """Reach a drawer waypoint through a verified, collision-free plan."""
        arm = self.robot.get_arm(arm_tag)
        self._drawer_last_failure = None
        tcp_pose = tcp_pose if tcp_pose is not None else self._top_down_tcp(tcp_pos)
        if (not _cabinet_motion_pose_valid(tcp_pose)
                or not _cabinet_motion_pose_valid(arm.tcp_offset)):
            print("[drawer] invalid waypoint pose")
            return False
        if (not np.isfinite(tol) or tol <= 0
                or (max_joint_delta is not None
                    and (not np.isfinite(max_joint_delta) or max_joint_delta <= 0))):
            print("[drawer] invalid waypoint motion bound")
            return False
        tcp_pos = np.asarray(tcp_pose.p, dtype=np.float64)
        link_pose = tcp_to_link_pose(tcp_pose, arm.tcp_offset)
        if not _cabinet_motion_pose_valid(link_pose):
            print("[drawer] invalid waypoint link pose")
            return False
        pose7 = link_pose.to_pose7()
        planner = arm.planner
        try:
            current_qpos = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        except (TypeError, ValueError):
            print("[drawer] invalid current arm joints")
            return False
        if current_qpos.shape != (arm.n_arm,) or not np.all(np.isfinite(current_qpos)):
            print("[drawer] invalid current arm joints")
            return False
        dt = float(self.scene.sim.dt)
        if not np.isfinite(dt) or dt <= 0 or dt > .004:
            print("[drawer] invalid physical planning clock")
            return False
        branch_bound = min(3.5, float(max_joint_delta)) if max_joint_delta is not None else 3.5
        self._refresh_avatar_planner_obstacles(arm_tag)
        if straight_pull:
            try:
                # Translate the clamped hand without asking the screw solver
                # to remove a tiny measured rotation. MPLib0.2.1 rounds such
                # rotations to zero but retains their translation component,
                # which can move a short pull past its endpoint. Validate the
                # resulting path against the original nominal pose below.
                start_tcp = self._drawer_plan_endpoint_tcp(arm_tag, current_qpos)
                if not _cabinet_motion_pose_valid(start_tcp):
                    print("[drawer] invalid straight-pull starting FK")
                    return False
                translation_link = tcp_to_link_pose(
                    Pose(tcp_pos.copy(), start_tcp.q.copy()), arm.tcp_offset,
                )
                # MPLib's fixed-step screw integration can stop on a <1e-4
                # leftover twist and report failure for a feasible straight
                # pull; the same goal re-integrated at a slightly different
                # step then plans. Every result is validated below.
                for qpos_step in self.DRAWER_PULL_SCREW_QPOS_STEPS:
                    result = planner._mplib.plan_screw(
                        goal_pose=planner._to_mplib_pose(translation_link.to_pose7()),
                        current_qpos=planner._pad_qpos(current_qpos),
                        time_step=dt, qpos_step=qpos_step, wrt_world=True, verbose=False,
                    )
                    if result and result.get("status") == "Success":
                        break
            except RuntimeError as exc:
                print(f"[drawer] mplib straight pull failed: {exc}")
                return False
        else:
            try:
                ik = planner.solve_ik(
                    current_qpos, pose7, num_waypoints=2, log=False,
                )
            except RuntimeError as exc:
                print(f"[drawer] mplib waypoint IK failed: {exc}")
                return False
            if ik is None or not ik.success:
                print("[drawer] mplib IK rejected waypoint")
                self._drawer_last_failure = "ik"
                return False
            ik_path = _cabinet_motion_path(
                ik.position, getattr(ik, "velocity", None), arm.n_arm,
            )
            if ik_path is None:
                print("[drawer] invalid mplib waypoint IK path")
                return False
            arm_qpos_goal = ik_path[0][-1].copy()
            if np.linalg.norm(arm_qpos_goal - current_qpos) > branch_bound:
                print("[drawer] waypoint IK branch jump rejected")
                self._drawer_last_failure = "ik"
                return False
            try:
                result = planner._mplib.plan_qpos(
                    goal_qposes=[planner._pad_qpos(arm_qpos_goal)],
                    current_qpos=planner._pad_qpos(current_qpos),
                    time_step=dt,
                    planning_time=10,
                    rrt_range=0.15,
                    verbose=False,
                )
            except RuntimeError as exc:
                print(f"[drawer] mplib waypoint planning failed: {exc}")
                return False
        if not result or result.get("status") != "Success":
            print("[drawer] collision-free waypoint plan unavailable")
            return False

        path = _cabinet_motion_path(
            result.get("position"), result.get("velocity"), arm.n_arm,
        )
        if path is None:
            print("[drawer] invalid mplib waypoint trajectory")
            return False
        pos_traj, vel_traj = path
        if (np.linalg.norm(pos_traj[-1] - current_qpos) > branch_bound
                or np.max(np.abs(np.diff(np.vstack([current_qpos, pos_traj]), axis=0))) > .30):
            print("[drawer] waypoint trajectory continuity rejected")
            return False
        endpoint = self._drawer_plan_endpoint_tcp(arm_tag, pos_traj[-1])
        if not _cabinet_motion_pose_valid(endpoint):
            print("[drawer] nonfinite or malformed waypoint FK pose")
            return False
        endpoint_error = float(np.linalg.norm(endpoint.p - tcp_pos))
        quat_dot = abs(float(np.dot(endpoint.q, tcp_pose.q)))
        orientation_error = 2.0 * float(np.arccos(np.clip(quat_dot, 0.0, 1.0)))
        if endpoint_error > min(float(tol), 0.015) or orientation_error > np.deg2rad(10):
            print(f"[drawer] plan endpoint rejected: xyz={endpoint_error:.4f}m, "
                  f"angle={np.rad2deg(orientation_error):.1f}deg")
            return False

        if straight_pull and not self._drawer_pull_path_is_straight(
            arm_tag, np.vstack([current_qpos, pos_traj]), tcp_pose,
        ):
            print("[drawer] pull trajectory leaves the handle line")
            return False

        from ..planning.base import PlanResult
        # This plan already uses the physical clock. Payload adapters normally
        # expand native4ms samples and must not expand this route a second time.
        previous_clock = getattr(self, "_cabinet_plan_is_physical_clock", False)
        self._cabinet_plan_is_physical_clock = True
        aborts_before = self._human_aware_aborts()
        try:
            self.execute_plan(PlanResult(True, pos_traj, vel_traj), arm_tag)
        finally:
            self._cabinet_plan_is_physical_clock = previous_clock
        if self._human_aware_aborts() != aborts_before:
            # The full-state expert's human-aware gate withheld this plan
            # (predicted to meet the human). Holding the goal below would be
            # exactly the unplanned drive the gate refused.
            print("[drawer] human-aware gate withheld the waypoint plan")
            self._drawer_last_failure = "gate"
            return False
        arm_qpos_goal = pos_traj[-1, :arm.n_arm]

        # Hold only an endpoint that the collision-free trajectory reached.
        # A failed plan must never turn into an unplanned drive to its goal.
        for _ in range(n_settle):
            self.robot.set_arm_joints(arm_qpos_goal, arm_tag)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        try:
            ee = np.asarray(arm.get_ee_pose(), dtype=float)
        except (TypeError, ValueError):
            print("[drawer] invalid measured waypoint TCP pose")
            return False
        if (ee.shape != (7,) or not np.all(np.isfinite(ee))
                or not np.isclose(np.linalg.norm(ee[3:]), 1.0, atol=1e-4)):
            print("[drawer] invalid measured waypoint TCP pose")
            return False
        err = float(np.linalg.norm(ee[:3] - tcp_pos))
        angle = 2.0 * float(np.arccos(np.clip(abs(np.dot(ee[3:], tcp_pose.q)), 0., 1.)))
        arrival_tol = min(float(tol), .015)
        if err > arrival_tol or angle > np.deg2rad(10):
            print(f"[drawer] waypoint tracking error={err:.4f}m/{angle:.4f}rad")
            return False
        return True

    def _human_aware_aborts(self) -> int:
        """Segments the full-state human-aware gate has aborted so far."""
        stats = getattr(self, "_neutral_aware_stats", None)
        enabled = getattr(self, "_neutral_aware_enabled", None)
        if stats is None or enabled is None or not enabled():
            return 0
        return int(stats().get("aborted", 0))

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        collider = getattr(self, "avatar_collider", None)
        if collider is None:
            return
        points = sample_capsule_planner_points(
            collider.current_capsules(), inflate_factor=1.8,
        )
        if points is not None and points.size:
            self.robot.get_arm(arm_tag).planner.update_obstacles(points, resolution=0.04)

    def _drawer_plan_endpoint_tcp(self, arm_tag: str, qpos) -> Pose | None:
        """Compute a planned TCP pose without changing the physics state."""
        arm = self.robot.get_arm(arm_tag)
        backend = arm.planner._mplib
        backend.pinocchio_model.compute_forward_kinematics(
            arm.planner._pad_qpos(np.asarray(qpos, dtype=float)[:arm.n_arm]),
        )
        link = backend.pinocchio_model.get_link_pose(backend.move_group_link_id)
        if (not _cabinet_motion_pose_valid(link)
                or not _cabinet_motion_pose_valid(arm.origin_pose)
                or not _cabinet_motion_pose_valid(arm.tcp_offset)):
            return None
        return arm.origin_pose * Pose(np.asarray(link.p), np.asarray(link.q)) * arm.tcp_offset

    def _drawer_pull_path_is_straight(self, arm_tag: str, path, goal: Pose) -> bool:
        """Reject lateral or rotational detours while clamped to the handle."""
        start = self._drawer_plan_endpoint_tcp(arm_tag, path[0])
        if not _cabinet_motion_pose_valid(start):
            return False
        segment = goal.p - start.p
        squared_length = float(np.dot(segment, segment))
        for qpos in path:
            pose = self._drawer_plan_endpoint_tcp(arm_tag, qpos)
            if not _cabinet_motion_pose_valid(pose):
                return False
            fraction = (float(np.dot(pose.p - start.p, segment)) / squared_length
                        if squared_length > 1e-12 else 0.)
            nearest = start.p + np.clip(fraction, 0., 1.) * segment
            angle = 2. * np.arccos(np.clip(abs(np.dot(pose.q, goal.q)), 0., 1.))
            if np.linalg.norm(pose.p - nearest) > .002 or angle > np.deg2rad(2):
                return False
        return True

    def _drawer_translation_pull(self) -> bool:
        """Franka pure-translation handle pull, unless the variant opts out."""
        return bool(self.DRAWER_TRANSLATION_PULL) and self.config.get("robot_type") == "franka"

    def _open_drawer_physics(self, arm_tag: str) -> bool:
        """Approach handle horizontally, close on bar, pull straight in
        +Y to slide the drawer open, then release and retreat.  Uses
        ``_drive_to_tcp`` verifies each mplib trajectory before PD execution.
        """
        arm = self.robot.get_arm(arm_tag)

        def _ee_str(tag: str):
            ee = arm.get_ee_pose()
            return (f"[{tag}] ee=({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})")

        handle_world = self._drawer_handle_world0.copy()

        # 1) Open gripper.
        self.open_gripper(arm_tag)

        # 2) Move to a front-side pre-grasp with the palm vertical, then
        # drive horizontally onto the handle.
        handle_grasp = handle_world.copy()
        if self._drawer_translation_pull():
            # The old target puts the open lower finger through the drawer
            # front. Native mesh/pad checks leave the bar within the fingers
            # at this10mm outward offset, while clearing the front panel.
            handle_grasp[1] += .010
        self._handle_approach_variant = None
        pre_front = self._handle_pre_front(handle_grasp)
        if not self._drive_to_tcp(
            pre_front, arm_tag, n_settle=300, tol=0.025,
            tcp_pose=self._handle_tcp(pre_front),
        ) and not self._handle_approach_fallback(handle_grasp, arm_tag):
            print("[drawer] physical pre-grasp failed")
            return False

        # 3) Approach the handle horizontally from +Y toward -Y.
        if not self._drive_to_tcp(
            handle_grasp, arm_tag, n_settle=300, tol=0.025,
            tcp_pose=self._handle_tcp(handle_grasp),
        ):
            print("[drawer] physical handle grasp failed")
            return False

        # 4) The measured bar is3.386mm high. Franka's .08 command leaves
        # its pad corners6.4mm apart, so allow closure until the bar stops
        # the physical fingers. Keep the existing force and PD profile.
        self.set_gripper(0.0 if self._drawer_translation_pull() else 0.08, arm_tag)
        for _ in range(120):
            self.step_sim()
        try:
            qpos_full = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception:
            pass

        # 5) Pull straight in +Y while a bounded rail force is applied to the
        # drawer DOF. The force command is active only while the gripper is
        # clamped and moving with the handle; qpos is never assigned here.
        # This keeps the interaction physically integrated even though the
        # source cabinet mesh has unreliable handle collision geometry.
        pull_pos = handle_grasp + np.array([0.0, +self._drawer_open_qpos_target, 0.0])
        n_pull_steps = 8
        physical_pull_ok = True
        rail_follows_hand = self._drawer_translation_pull()
        rail_target = self._drawer_open_qpos_target
        for i in range(1, n_pull_steps + 1):
            waypoint_offset = self._drawer_open_qpos_target * (i / n_pull_steps)
            if rail_follows_hand:
                # A final-opening target during a short hand step pushes the
                # drawer ahead through the fingers. Assist only this waypoint.
                rail_target = waypoint_offset
            force_ok, dof_before, _, force = (
                control_dof_force_toward_position_compat(
                    self.cabinet.entity,
                    self._drawer_dof_idx,
                    rail_target,
                )
            )
            if not force_ok:
                physical_pull_ok = False
                break
            wp = handle_grasp + np.array([
                0.0, waypoint_offset, 0.0,
            ])
            if not self._drive_to_tcp(
                wp, arm_tag, n_settle=80, tol=0.035,
                tcp_pose=self._handle_tcp(wp),
                straight_pull=self._drawer_translation_pull(),
            ):
                physical_pull_ok = False
                break
            dof_now = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
                self._drawer_dof_idx])
            print(
                f"[drawer] physical pull step={i}/{n_pull_steps}, "
                f"qpos={dof_before:.4f}->{dof_now:.4f}, force={force:+.3f}N"
            )
            if dof_now >= self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE:
                break
        for _ in range(60):
            if rail_follows_hand and not physical_pull_ok:
                break
            force_ok, _, _, _ = control_dof_force_toward_position_compat(
                self.cabinet.entity,
                self._drawer_dof_idx,
                rail_target,
            )
            physical_pull_ok = physical_pull_ok and force_ok
            if rail_follows_hand and not force_ok:
                break
            self.step_sim()
        control_dof_force_compat(self.cabinet.entity, self._drawer_dof_idx, 0.0)

        pulled_qpos = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
            self._drawer_dof_idx])
        if (
            not physical_pull_ok
            or pulled_qpos < self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE
        ):
            print(
                f"[drawer] physical pull failed: qpos={pulled_qpos:.4f}, "
                f"target={self._drawer_open_qpos_target:.4f}"
            )
            self.open_gripper(arm_tag)
            return False
        # Lock only the position reached by the force-integrated pull before
        # releasing the handle. This prevents passive rail drift during the
        # retreat without letting position control complete the opening.
        self._drawer_held_qpos = pulled_qpos
        if not self._hold_drawer_open():
            self.open_gripper(arm_tag)
            return False

        # 6) Release the bar.
        self.open_gripper(arm_tag)
        for _ in range(60):
            self.step_sim()

        # 7) Retreat horizontally first so the gripper clears the handle,
        # then move up to the top-down clearance pose for the object pick.
        retreat_front = pull_pos + np.array([0.0, self._HANDLE_RETREAT_DIST, 0.0])
        self._drive_to_tcp(
            retreat_front, arm_tag, n_settle=220, tol=0.04,
            tcp_pose=self._handle_tcp(retreat_front),
        )
        retreat = retreat_front + np.array([0.0, 0.0, +0.15])
        self._drive_to_tcp(retreat, arm_tag, n_settle=300, tol=0.04)
        for _ in range(40):
            self.step_sim()
        achieved = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
            self._drawer_dof_idx])
        if achieved < self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE:
            print(
                f"[drawer] opened drawer drifted closed: qpos={achieved:.4f}, "
                f"target={self._drawer_open_qpos_target:.4f}"
            )
            return False
        # Stabilization may hold only what the physical pull achieved. It is
        # never allowed to finish opening a drawer after the gripper retreats.
        self._drawer_held_qpos = achieved
        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _play_robot_only_once(self) -> bool:
        # Settle the object on the table.
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(80):
                self.step_sim()

        if self.cabinet is None or self.object is None:
            return False

        self._pick_drawer()

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # Phase A — physics-based drawer open with bumped arm PD.
        # kp=15000 is the sweet spot: with descend 4 cm above
        # bar the gripper still snags the bar during the +Y pull
        # micro-steps, dragging the drawer to dof=0.065 (target 0.06).
        # kp=50000 is WORSE (drawer only opens to 0.020)
        # — likely the high-kp jerks bounce the gripper off the
        # bar before contact friction grips.  Restored to default after
        # for the pick (which works at default kp).
        self._set_arm_pd(arm_tag, arm_kp=15000.0, arm_kv=800.0)
        ok_open = self._open_drawer_physics(arm_tag)
        self._set_arm_pd(arm_tag, arm_kp=4000.0, arm_kv=300.0)
        if not ok_open:
            return False
        if not self._hold_drawer_open():
            return False
        try:
            qpos_after = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception as e:
            return False

        # Phase B+C — pick the randomized object and drop into open drawer via the
        # mixin's ``pick_and_place`` helper.  This uses the proven
        # seeded-IK + xy_err retry pattern from dump_bin /
        # place_bread_in_basket; bypasses the silent IK relax that
        # affects plan_path-based moves.
        def _object_center():
            if getattr(self, "_object_is_primitive", False):
                # Primitive origin == geometric centre (no mesh-origin offset).
                return np.asarray(self.object.get_pose().p, dtype=float).copy()
            return self._get_object_world_center(
                self.object, self._object_id, self._object_model_id,
            ).copy()

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        # Cross-section radius from the randomized object's world
        # footprint; this drives the mixin's adaptive close target.
        object_radius = float(self._object_radius)

        # Drop target = inside the open drawer cavity.  Inset from the front
        # edge by the payload's live half-depth plus clearance, so the entire
        # footprint (not merely its center) ends up behind the drawer lip.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        drop_x = 0.5 * (float(drawer_aabb[0, 0]) + float(drawer_aabb[1, 0]))
        object_aabb = to_numpy(self.object.entity.get_AABB())
        object_depth_y = float(object_aabb[1, 1] - object_aabb[0, 1])
        drop_y = self.contained_center_from_edge(
            float(drawer_aabb[1, 1]),
            object_depth_y,
            inward_sign=-1.0,
            clearance=self._DRAWER_PLACEMENT_CLEARANCE,
            held_inward_offset=self._DRAWER_HELD_INWARD_ALLOWANCE,
        )
        drop_xyz = np.array([
            drop_x, drop_y,
            self._drawer_support_surface_z() + self._DROP_HOVER_ABOVE_DRAWER_FLOOR,
        ], dtype=float)

        pick = PickSpec(
            get_center=_object_center,
            radius=object_radius,
            label=self._object_id,
            get_half_height=_object_half_height,
            after_close=lambda: self._boost_finger_pd(arm_tag),
            require_lift=True,
            transport_retention_tolerance=0.08,
            # Ejection guard: a contact-solver ejection at
            # close can launch the object away (e.g. one block ended
            # 12 m below the floor while the arm kept executing).  Interrupt
            # shares the same gate with base/neutral/assist.
            max_grasp_offset=0.10,
            **self._payload_pick_options(),
        )
        # transport_z must clear the cabinet top (≈ rim_z + 0.03 = 0.99 m)
        # AND the drawer rim during the lateral transit + hover-release.
        place = PlaceSpec(
            pos=drop_xyz,
            label="drawer",
            transport_z=self._drawer_rim_z + 0.10,
            support_z=self._drawer_support_surface_z(),
            support_clearance=self._DRAWER_SUPPORT_CLEARANCE,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False

        # Let everything settle so check_success reads steady state.
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _object_in_drawer_metrics(self) -> dict:
        if self._drawer_link is None and self.cabinet is not None:
            try:
                self._pick_drawer()
            except Exception as exc:
                return {
                    "object_in_drawer": False,
                    "object_in_drawer_error": f"drawer init failed: {exc}",
                }
        if self._drawer_link is None or self.object is None:
            return {
                "object_in_drawer": False,
                "object_in_drawer_error": "missing drawer link or object",
            }

        if getattr(self, "_object_is_primitive", False):
            obj_p = np.asarray(self.object.get_pose().p, dtype=float)
        else:
            obj_p = self._get_object_world_center(
                self.object, self._object_id, self._object_model_id,
            )
        # Object must lie inside the *current* drawer cavity, defined
        # by the live drawer-body AABB (which moves with the drawer's
        # prismatic joint).  Using ``get_pos()`` (joint origin) instead
        # would let an object sitting BEHIND the open drawer (inside
        # the cabinet body) pass.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        margin = 0.005  # 5 mm wall margin
        in_aabb_xy = (
            drawer_aabb[0, 0] + margin < obj_p[0] < drawer_aabb[1, 0] - margin
            and drawer_aabb[0, 1] + margin < obj_p[1] < drawer_aabb[1, 1] - margin
        )
        # Cabinet may have a ceiling above the drawer body (the
        # next-drawer-up's top panel for non-top drawers, or the
        # cabinet's own top panel above the top drawer).  Require the
        # object to ALSO be in front of the cabinet's body so it sits
        # in the *exposed* cavity slice — otherwise it's wedged
        # against / on the ceiling.  Cabinet front face is at
        # _CABINET_POS.y + half-Y * cabinet_scale.
        cabinet_front_y = (float(self._cabinet_pos[1])
                           + 0.094)  # body half-Y at scale 0.20
        exposed = bool(obj_p[1] > cabinet_front_y)
        in_xy = bool(in_aabb_xy and exposed)
        drawer_p = to_numpy(self._drawer_link.get_pos()).ravel()[:3]
        rim_z = self._drawer_rim_z
        in_z = bool(obj_p[2] > self._drawer_floor_z - 0.05
                    and obj_p[2] < rim_z + 0.02)
        ok = bool(in_xy and in_z)
        try:
            drawer_qpos = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
                self._drawer_dof_idx])
        except Exception:
            drawer_qpos = None
        ee_pose = np.asarray(self.robot.right_arm.get_ee_pose(), dtype=np.float64).ravel()
        ee_pos = ee_pose[:3]
        ee_in_drawer_aabb = bool(
            drawer_aabb[0, 0] < ee_pos[0] < drawer_aabb[1, 0]
            and drawer_aabb[0, 1] < ee_pos[1] < drawer_aabb[1, 1]
            and drawer_aabb[0, 2] < ee_pos[2] < drawer_aabb[1, 2]
        )
        ee_to_drawer_center = float(np.linalg.norm(ee_pos - drawer_p))
        return {
            "object_in_drawer": ok,
            "object_in_drawer_xy": in_xy,
            "object_in_drawer_aabb_xy": bool(in_aabb_xy),
            "object_in_drawer_exposed": exposed,
            "object_in_drawer_z": in_z,
            "object_name": self._object_id,
            "object_model_id": int(self._object_model_id),
            "object_center": obj_p.tolist(),
            "drawer_center": drawer_p.tolist(),
            "drawer_aabb_min": drawer_aabb[0].tolist(),
            "drawer_aabb_max": drawer_aabb[1].tolist(),
            "drawer_floor_z": float(self._drawer_floor_z),
            "drawer_rim_z": float(rim_z),
            "drawer_qpos": drawer_qpos,
            "cabinet_pos": self._cabinet_pos.tolist(),
            "cabinet_front_y": float(cabinet_front_y),
            "ee_pose": ee_pose.tolist(),
            "ee_in_drawer_aabb": ee_in_drawer_aabb,
            "ee_to_drawer_center": ee_to_drawer_center,
            "gripper_open_value": float(self.robot.right_arm.gripper_val),
        }

    def check_success(self) -> bool:
        metrics = self._object_in_drawer_metrics()
        if "object_in_drawer_error" in metrics:
            return False
        return bool(metrics["object_in_drawer"])

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._object_in_drawer_metrics())
        return metrics


class PutObjectCabinet(CabinetOrdinaryMotionMixin, CabinetTaskCore):
    """Robot opens the drawer, then places the table object inside it."""

    def play_once(self) -> bool:
        return self._play_robot_only_once()
