"""Shared mechanics for the categorize task family."""

import json
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import transforms3d as t3d
import genesis as gs

from ..base_task import BaseTask
from ..genesis_compat import set_dofs_kp_kv_compat
from ..utils import Actor, Pose, load_object, create_primitive, ASSETS_PATH
from ..grasp import tcp_to_link_pose
from ..object_catalog import object_set_pairs, get_entry
from .resting_guard import entity_bottom_z, held_by_anyone
from .categorize_support.profile import CategorizeTaskProfile
from .categorize_support.transport import milk_continuation_check
from .categorize_support.carry import milk_corner_carry, uses_milk_corner
from .categorize_support.motion import CategorizeMotionMixin, stop_on_motion_rejection
from .categorize_support.radius import pick_and_place_radius
from .categorize_support.release import lower_release, uses_lower_release
from .categorize_support.seal import (
    pick_and_place_seal, execute_seal_selector_plan, execute_seal_hold_plan,
)
from ..manipulation import (
    TopDownPickPlaceMixin,
    PickSpec as RadiusPickSpec,
    PlaceSpec as RadiusPlaceSpec,
)


@dataclass
class PickSpec:
    """Describes what to pick.

    ``get_center`` is called multiple times during the pick (initial plan,
    re-read after pre-grasp in case approach bumped the object, and for debug
    logging after grip/lift).  Keep it cheap.

    ``radius`` is the cross-section half-extent used to size the partial
    gripper close (``radius * 0.7``) and the below-equator grasp offset.

    ``close_target_m`` (optional) overrides the formula-derived gripper
    close target (in metres, finger half-gap).  Read from the object's
    `model_data{N}.json` `grasp_close_target_m` field.  When None, the
    fallback formula in `_grasp_object` runs.

    ``actor`` (optional) is the Genesis actor for the picked object —
    used by ``_update_planner_obstacles`` to skip its aabb during the
    pre-grasp / place transit so the planner doesn't reject the goal
    as in-collision with the held object.
    """

    get_center: Callable[[], np.ndarray]
    radius: float
    label: str = "object"
    close_target_m: Optional[float] = None
    actor: Optional[object] = None


@dataclass
class PlaceSpec:
    """Describes where to drop.

    ``pos`` is the world (x, y) of the drop target (z field is ignored when
    ``drop_from_z`` is set — which it always is for container-style drops).
    ``drop_from_z`` set = hover at that height and release (basket pattern).
    ``drop_from_z`` None = descend TCP to ``pos.z`` before release.

    ``basket_idx`` (optional) is the destination basket index — passed to
    ``_update_planner_obstacles`` as ``skip_basket_idx`` during the descent
    so the basket's own walls don't block the drop path.
    """

    pos: np.ndarray
    label: str = "place"
    drop_from_z: Optional[float] = None
    basket_idx: Optional[int] = None


class CategorizeTaskCore(
    CategorizeTaskProfile, CategorizeMotionMixin, TopDownPickPlaceMixin, BaseTask,
):
    """Shared scene, manipulation, and robot-only expert mechanics.

    Three categories, each with a basket containing one static example object
    and one loose object on the table that the robot must sort into the correct
    basket.  The avatar cooperates by sorting some objects on its side
    (placeholder until motion is available).

    Scene layout (top view, +Y points from robot toward avatar):
      - Robot (Franka) at y=-0.5
      - Large table centered at y=0.1
      - Three loose objects near robot side (y=-0.05)
      - Three baskets further from robot (y=0.20) with example objects
      - Avatar at y=1.2
    """

    # Release part-way down / from hover when the low basket
    # release has no checked descent (categorize_support.release).
    CATEGORIZE_SHALLOW_RELEASE = True

    INSTRUCTION = "sort the objects into the matching baskets"

    def execute_plan(self, result, arm_tag: str, gripper_result=None):
        if id(result) in getattr(self, '_categorize_grasp_targets', {}):
            return self._execute_categorize_plan(result, arm_tag, gripper_result)
        context = getattr(self, "_seal_selector_execution", None)
        controller = getattr(self, "_seal_hold_controller", None)
        if controller is not None:
            return execute_seal_hold_plan(
                self, super().execute_plan, result, arm_tag, gripper_result, controller,
            )
        if context is None:
            return super().execute_plan(result, arm_tag, gripper_result=gripper_result)
        return execute_seal_selector_plan(
            self, super().execute_plan, result, arm_tag, gripper_result, context,
        )

    def _setup_scene(self):
        """Use smaller dt for better collision response during grasping.
        Honours --config raytracer.yml by building the renderer from config;
        without this, gs.Scene defaulted to rasterizer and our LuisaRender
        config was silently ignored.
        """
        self._resolve_renderer()
        renderer = self._build_renderer()
        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=0.001),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        self.scene = gs.Scene(**scene_kwargs)
        self.scene.add_entity(gs.morphs.Plane(visualization=not self._hide_ground_visuals()))
        self._create_floor()

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, float(self._CATEGORIZE_ROBOT_Y), 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _boost_arm_pd(self):
        """Increase arm joint PD gains for better positioning accuracy."""
        from ..robot.franka_robot import _get_dof_idx, to_numpy
        arm = self.robot.get_arm("right")
        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
        except Exception:
            return
        for j in arm.arm_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 20000.0
                    kv[idx] = 1000.0
        # Medium finger kp — kp=15000 penetrates meshes (~390N per finger),
        # kp=800 is too weak.  kp=2500 holds ~45N per finger at full
        # compression: firm enough for 50g objects without crushing.
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 2500.0
                    kv[idx] = float(self._CATEGORIZE_FINGER_KV)
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)

    def _create_table(self, table_height=0.74):
        """Larger table to accommodate both robot and avatar sides."""
        if self._CATEGORIZE_COLUMN_LAYOUT:
            dx, dy = self.table_offset
            return self._create_rectangular_table(
                table_height=table_height,
                half_size=(0.80, self.TABLE_HALF_SIZE_Y),
                center_xy=(dx, dy),
            )
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        dx, dy = self.table_offset
        if self._try_create_table_variant(table_height, half_size=(0.80, self._TABLE_HALF_SIZE_Y), center_xy=(dx, dy)):
            return

        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (0.80, self._TABLE_HALF_SIZE_Y, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65),
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        _leg_y = self._TABLE_HALF_SIZE_Y - 0.05
        for x, y in [(-0.75, -_leg_y), (0.75, -_leg_y), (-0.75, _leg_y), (0.75, _leg_y)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )

    @staticmethod
    def _get_object_cross_section_radius(object_name: str, model_id: int = 0) -> float:
        if object_name == "cube":
            return float(get_entry("cube").prim_half)
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return float(np.min(extents * scale) / 2)

    @staticmethod
    def _get_grasp_close_target(object_name: str, model_id: int = 0) -> Optional[float]:
        """Per-object gripper close target in metres (finger half-gap).
        Returns None if the object's model_data has no `grasp_close_target_m`
        field — caller falls back to the radius-based formula."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        if not md_path.exists():
            return None
        with open(md_path) as f:
            md = json.load(f)
        v = md.get("grasp_close_target_m")
        return float(v) if v is not None else None

    @staticmethod
    def _get_object_scale(object_name: str, model_id: int = 0) -> float:
        """Runtime mesh scale applied by load_object for this model variant."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        if not md_path.exists():
            return 1.0
        with open(md_path) as f:
            md = json.load(f)
        raw = md.get("scale", 1.0)
        if isinstance(raw, (list, tuple)):
            return float(raw[0])
        return float(raw)

    def _select_and_execute_task_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        *,
        model_id: int,
        object_scale: float,
        max_candidates: int = 20,
        preferred_names=None,
        post_plan_check=None,
    ):
        """Run the standard grasp helper with task-local candidate priority."""
        exclusive_names = self.EXCLUSIVE_GRASP_NAMES.get(object_name)
        if object_name == "038_milk-box" and self.MILK_CONTINUATION_PREFLIGHT:
            return self.select_and_execute_grasp(
                object_name, object_pose, arm_tag,
                model_id=model_id, max_candidates=max_candidates,
                table_z=self.TABLE_TOP_Z, robot_type="franka",
                object_scale=object_scale, grasp_names=exclusive_names,
                plan_attempts=2, post_plan_check=post_plan_check,
            )
        if object_name == "035_apple":
            x = float(np.asarray(object_pose.p, dtype=float)[0])
            exclusive_names = (
                self.APPLE_CENTER_GRASP_NAMES
                if x >= self.APPLE_CENTER_MIN_X
                else self.APPLE_UNDERHAND_GRASP_NAMES
            )
        if exclusive_names is not None:
            preferred_names = exclusive_names
        elif preferred_names is None:
            active = getattr(self, "_active_preferred_grasp_names", None)
            if active is not None and object_name in active:
                preferred_names = active[object_name]
            else:
                preferred_names = self.PREFERRED_GRASP_NAMES.get(object_name, ())
        for grasp_name in preferred_names:
            result = self.select_and_execute_grasp(
                object_name,
                object_pose,
                arm_tag,
                model_id=model_id,
                robot_type="franka",
                object_scale=object_scale,
                grasp_names=(grasp_name,), table_z=self.TABLE_TOP_Z,
                post_plan_check=post_plan_check,
            )
            if result is not None:
                return result

        # An exclusive allowlist is also a ban list for everything omitted.
        # Do not let the generic ranker silently reintroduce a banned milk-box
        # top grasp or an unvalidated apple/Rubik pose after these candidates
        # fail planning.
        if exclusive_names is not None:
            return None

        return self.select_and_execute_grasp(
            object_name, object_pose, arm_tag,
            model_id=model_id, max_candidates=max_candidates,
            table_z=self.TABLE_TOP_Z, robot_type="franka",
            object_scale=object_scale,
        )

    @staticmethod
    def _get_model_center(object_name: str, model_id: int = 0) -> np.ndarray:
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        center = np.array(md["center"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return center * scale

    def _get_object_world_center(self, actor, object_name: str, model_id: int = 0) -> np.ndarray:
        if object_name == "cube":
            # Primitive origin == geometric centre (no mesh-origin offset).
            return np.asarray(actor.get_pose().p, dtype=float)
        model_center = self._get_model_center(object_name, model_id)
        pose = actor.get_pose()
        R = t3d.quaternions.quat2mat(np.array(pose.q))
        model_c = np.array(pose.p) + R @ model_center
        # Live-AABB centre override (mirrors put_object_cabinet): some GLBs load
        # in a frame whose vertical axis doesn't match model_data's y-up
        # center/extents, so the model_data centre sits several cm off the real
        # object and the top-down radius grasp closes on air.  When the live
        # simulator AABB centre disagrees by >1.5 cm in z, trust the physics.
        try:
            from ..robot.franka_robot import to_numpy as _to_numpy
            aabb = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            aabb_c = 0.5 * (aabb[0] + aabb[1])
            if abs(float(aabb_c[2]) - float(model_c[2])) > 0.015:
                return aabb_c
        except Exception:
            pass
        return model_c

    def _spawn_categorize_object(self, obj_name, model_id, x, y, table_top, quat):
        """Spawn one loose sortable object — a procedural cube for the
        ``cube`` pool entry (origin == centre, sits on the table), else the
        mesh asset."""
        if obj_name == "cube":
            half = float(get_entry("cube").prim_half)
            entity = create_primitive(
                self.scene, "box",
                Pose([x, y, table_top + half], np.array([1.0, 0.0, 0.0, 0.0])),
                size={"half_size": (half, half, half)},
                color=(0.85, 0.35, 0.20),
            )
            return Actor(entity, {}, "cube")
        # Use the catalog's per-object validated friction (4.0–5.0) instead of
        # a flat 2.5 — the loose categorize objects were slipping out of the
        # parallel-jaw grip on the lift under Genesis 1.2.0 because this task
        # ignored the per-object friction the cabinet/dump grasp tunings rely
        # on.  Cap at the Genesis Rigid friction limit (5.0).
        model_key = (obj_name, model_id)
        friction_key = model_key if model_key in self.SPAWN_FRICTION_OVERRIDE else obj_name
        if friction_key in self.SPAWN_FRICTION_OVERRIDE:
            obj_friction = float(self.SPAWN_FRICTION_OVERRIDE[friction_key])
        else:
            try:
                obj_friction = float(min(5.0, get_entry(obj_name).friction))
            except Exception:
                obj_friction = 2.5
        if ((getattr(self, "config", None) or {}).get("categorize_payload_friction_v1") is True
                and model_key in self.PAYLOAD_FRICTION_V1_MODELS):
            obj_friction = float(self.PAYLOAD_FRICTION_V1_VALUE)
            print(f"[categorize] payload_friction_v1: {obj_name}@{model_id} "
                  f"spawn friction={obj_friction}", flush=True)
        return load_object(
            self.scene, Pose([x, y, table_top], quat),
            obj_name, model_id=model_id, convex=True,
            is_static=False, friction=obj_friction,
        )

    def load_actors(self):
        if self._CATEGORIZE_COLUMN_LAYOUT:
            return self._load_column_actors()
        return self._load_cooperative_actors()

    def _load_cooperative_actors(self):
        """Cooperative layout (+Y = robot → human):

            robot (y=-0.35)
              │
              ▼ robot-side loose objects (random XY in y∈[-0.20,-0.05])
              │
              ▼ boxes (y=+0.15)
              │
              ▼ avatar-side loose objects (random XY in y∈[+0.32,+0.42])
              │
            human (y=+0.52)

        Per-seed randomization (via np.random.seed set in BaseTask.reset):
          - `basket_perm` permutes which CATEGORY sits in which basket
            column.
          - `robot_perm` and `avatar_perm` are INDEPENDENT permutations
            that pick which CATEGORY each robot-side / avatar-side loose
            object carries.  Each object's ``basket_idx`` is derived by
            category match (basket holding the matching example) — both
            agents must categorise, and the avatar-side and robot-side
            orderings are decoupled from each other.
          - Robot-side and avatar-side loose objects are placed at
            **random XY** within their respective Y strips (rejection-
            sampled with min-separation + basket-clear constraints) — no
            longer aligned to a fixed row of column X's.
          - Basket column X still gets ±2 cm jitter.
          - Universal BaseTask clutter handles optional table-random objects.

        Boxes are built from 5 primitives (floor + 4 walls) — same pattern
        as categorize_interrupt — because the articulated 100194 URDF has
        lid pieces that trap thrown objects on the rim.
        """
        table_top = self.TABLE_TOP_Z + 0.02
        upright_q = np.asarray(self.SPAWN_UPRIGHT_QUAT, dtype=float)

        # Per-episode: sample NUM_CATEGORIES distinct objects from the shared
        # catalog pool, one per basket (each a distinct category).  cube is a
        # primitive (handled via the radius-based pick path below).
        pool = object_set_pairs(self.OBJECT_SET)
        pick = np.random.choice(len(pool), self.NUM_CATEGORIES, replace=False)
        cats = [pool[int(i)] for i in pick]
        if self.simplified_mode_enabled():
            cats = [cats[int(np.random.randint(0, len(cats)))]]
        self._cats = cats
        # Per-seed basket category permutation.
        basket_perm = np.random.permutation(len(cats))
        # Robot-side and avatar-side each get their OWN independent
        # category-to-slot permutation; each object's ``basket_idx`` is
        # then derived by category match via ``basket_col_of_cat``.
        robot_perm = np.random.permutation(len(cats))
        avatar_perm = np.random.permutation(len(cats))
        basket_col_of_cat = np.argsort(basket_perm)

        # Per-seed X jitter on each basket column (±2 cm).  Loose
        # objects no longer share these column X's — they are sampled
        # freely within an XY region per side.
        base_col_xs = np.linspace(-0.25, 0.25, len(cats))
        col_xs = base_col_xs + np.random.uniform(-0.02, 0.02, size=len(cats))
        col_xs = [float(x) for x in col_xs]

        # --- Three open-top boxes (primitive floor + 4 walls) -----------
        box_y = 0.15
        box_half_x = 0.096
        box_half_y = 0.096
        box_half_z = 0.056
        wall_t = 0.005
        box_color = (0.7, 0.5, 0.3)
        self.basket_poses = []
        self.basket_actors = []
        # Baskets sit at the column X positions, in basket_perm order.
        # Each basket holds its category example (visible inside the box).
        # Anchor flush with TABLE_TOP_Z (not table_top = TABLE_TOP_Z + 0.02)
        # — the +0.02 is a spawn offset for loose objects to fall under
        # gravity; static baskets must sit on the table, not float above.
        for col_idx, bx in enumerate(col_xs):
            cz = self.TABLE_TOP_Z + box_half_z
            self.basket_poses.append(
                Pose([bx, box_y, cz], np.array([1.0, 0.0, 0.0, 0.0]))
            )
            # Floor.
            create_primitive(
                self.scene, "box",
                Pose(p=[bx, box_y, cz - box_half_z + wall_t]),
                size={"half_size": (box_half_x, box_half_y, wall_t)},
                color=box_color, is_static=True,
            )
            # Four walls.
            for wx, wy, hx, hy in [
                (-box_half_x + wall_t, 0.0, wall_t, box_half_y),
                (+box_half_x - wall_t, 0.0, wall_t, box_half_y),
                (0.0, -box_half_y + wall_t, box_half_x, wall_t),
                (0.0, +box_half_y - wall_t, box_half_x, wall_t),
            ]:
                create_primitive(
                    self.scene, "box",
                    Pose(p=[bx + wx, box_y + wy, cz]),
                    size={"half_size": (hx, hy, box_half_z)},
                    color=box_color, is_static=True,
                )
        self._example_actors = []  # no preloaded examples in this layout

        # --- Rejection-sample XY positions per side --------------------
        # Robot-side strip: y ∈ [-0.20, -0.05] (well clear of basket
        # back edge at y≈+0.054 and the robot base at y=-0.35).
        # Avatar-side strip: y ∈ [+0.32, +0.42] (well clear of basket
        # front edge at y≈+0.246 and avatar body at y≈+0.55).
        # X stays inside ±0.32 — wider than the previous ±0.25 columns
        # but still inside Franka's reliable top-down IK reach and the
        # avatar's nearest-hand reach (left hand serves x≤0, right
        # x>0; worst-case shoulder→hand ≈ 0.62 m, under the ~0.65 m
        # palm-reach budget).
        loose_min_sep = 0.10        # 10 cm pairwise separation
        basket_clear = 0.13         # keep clear of basket centres
        robot_region = (-0.32, 0.32, -0.20, -0.05)
        avatar_region = (-0.32, 0.32, 0.32, 0.42)

        def _sample_xy(region, n_existing_pts, *, avoid_robot_base=False,
                        max_tries=400):
            xmin, xmax, ymin, ymax = region
            for _try in range(max_tries):
                x = float(np.random.uniform(xmin, xmax))
                y = float(np.random.uniform(ymin, ymax))
                # Stay clear of basket centres.
                if any(np.hypot(x - bx, y - 0.15) < basket_clear for bx in col_xs):
                    continue
                # Pairwise separation against earlier loose-object picks.
                if any(np.hypot(x - px, y - py) < loose_min_sep
                       for px, py in n_existing_pts):
                    continue
                # Robot-base clearance (only matters for the robot strip).
                if avoid_robot_base and np.hypot(x, y - (-0.35)) < 0.22:
                    continue
                return x, y
            i = len(n_existing_pts)
            return col_xs[i % len(col_xs)], (ymin + ymax) / 2

        robot_pts = []
        avatar_pts = []
        for _ in range(len(cats)):
            robot_pts.append(_sample_xy(robot_region, robot_pts,
                                         avoid_robot_base=True))
        for _ in range(len(cats)):
            avatar_pts.append(_sample_xy(avatar_region, avatar_pts))

        # --- Robot-side loose objects (independent permutation) --------
        # basket_idx is computed by category match — each object goes to
        # whichever basket holds the matching example, regardless of
        # where it spawned.
        self._sort_targets = []  # (actor, name, model_id, basket_idx)
        for slot_i in range(len(cats)):
            cat_idx = int(robot_perm[slot_i])
            obj_name, model_id = cats[cat_idx]
            if self.ROBOT_MATCH_BASKET_COLUMNS:
                basket_col = int(basket_col_of_cat[cat_idx])
                ox = float(col_xs[basket_col] + np.random.uniform(
                    -self.ROBOT_COLUMN_X_JITTER,
                    self.ROBOT_COLUMN_X_JITTER,
                ))
                oy = float(np.random.uniform(*self.ROBOT_COLUMN_Y_RANGE))
            else:
                ox, oy = robot_pts[slot_i]
            if obj_name == "086_woodenblock" and not self.ROBOT_MATCH_BASKET_COLUMNS:
                # The block is held by a standard friction side-clamp.  Keep
                # it near its destination column so the robust grasp does not
                # need a long cross-table carry that can let it slip.
                basket_col = int(basket_col_of_cat[cat_idx])
                ox = float(col_xs[basket_col])
                if basket_col == 1:
                    ox -= 0.06
                elif basket_col == 2:
                    ox -= 0.045
                oy = -0.085
            if obj_name == "100_seal" and not self.ROBOT_MATCH_BASKET_COLUMNS:
                # Seal grasps are reach-sensitive in Franka layouts; reuse the
                # tight strip validated for the standard helper instead of the
                # wider free robot-side sampler.
                ox = float(np.random.uniform(0.00, 0.05))
                oy = float(np.random.uniform(-0.10, -0.07))
            actor = self._spawn_categorize_object(obj_name, model_id, ox, oy, table_top, upright_q)
            basket_idx = int(basket_col_of_cat[cat_idx])
            self._sort_targets.append((actor, obj_name, model_id, basket_idx))

        # --- Avatar-side loose objects (independent permutation, free XY)
        self._avatar_sort_targets = []
        if not self.no_human_enabled():
            for slot_i in range(len(cats)):
                cat_idx = int(avatar_perm[slot_i])
                obj_name, model_id = cats[cat_idx]
                ox, oy = avatar_pts[slot_i]
                actor = self._spawn_categorize_object(obj_name, model_id, ox, oy, table_top, upright_q)
                basket_idx = int(basket_col_of_cat[cat_idx])
                self._avatar_sort_targets.append((actor, obj_name, model_id, basket_idx))

        # Universal BaseTask clutter handles table-random objects.
        self._clutter_actors = []

    def _load_column_actors(self):
        """Boxes on robot side (y=-0.20), loose objects on avatar side,
        same X column per category so each picked object moves along a roughly
        straight line back to its matching box.

        Per-seed randomization (seeded through BaseTask.reset via np.random.seed):
          - CATEGORIES are permuted
          - The interrupted robot target is sampled from all three objects
          - Column X gets ±2 cm jitter
          - Loose-object Y gets ±1.5 cm jitter
          - Universal BaseTask clutter handles optional table-random objects.
        """
        table_top = self.TABLE_TOP_Z + 0.02
        upright_q = np.asarray(self.SPAWN_UPRIGHT_QUAT, dtype=float)

        # Per-episode: sample NUM_CATEGORIES distinct objects from the shared
        # catalog pool (one per basket), then permute (and reduce for the
        # simplified single-object mode).
        pool = object_set_pairs(self.OBJECT_SET)
        sampled = np.random.choice(len(pool), self.NUM_CATEGORIES, replace=False)
        cats = [pool[int(i)] for i in sampled]
        perm = np.random.permutation(len(cats))
        if self.simplified_mode_enabled():
            perm = perm[:1]
        cats = [cats[i] for i in perm]
        self._cats = cats

        # Pick which robot-side object the avatar will inspect.  Default is a
        # random ordinal from the scripted robot pick order; review/debug runs
        # can force a pick ordinal, spawned index, or object name.
        forced_obj = (
            self.config.get("interrupt_target_object")
            or self.config.get("interrupt_object_name")
        )
        pick_order = sorted(range(len(cats)), reverse=True)
        if forced_obj:
            matches = [i for i, (obj_name, _) in enumerate(cats) if obj_name == forced_obj]
            if not matches:
                raise ValueError(
                    f"interrupt_target_object={forced_obj!r} not in spawned categories "
                    f"{[obj_name for obj_name, _ in cats]}"
                )
            self._interrupted_idx = int(matches[0])
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))
        elif "interrupt_target_idx" in self.config:
            self._interrupted_idx = int(self.config["interrupt_target_idx"]) % len(cats)
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))
        elif "interrupt_pick_ordinal" in self.config:
            self._interrupted_pick_ordinal = (
                int(self.config["interrupt_pick_ordinal"]) % len(pick_order)
            )
            self._interrupted_idx = int(pick_order[self._interrupted_pick_ordinal])
        elif self.config.get("random_interrupt_target", True):
            self._interrupted_pick_ordinal = int(np.random.randint(0, len(pick_order)))
            self._interrupted_idx = int(pick_order[self._interrupted_pick_ordinal])
        else:
            self._interrupted_idx = self.INTERRUPTED_IDX
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))

        # Per-seed X jitter on each column (±2 cm) — boxes and loose objects
        # share the same jittered column X so the transport line is straight.
        col_xs = np.linspace(-0.25, 0.25, len(cats)) + np.random.uniform(
            -0.02, 0.02, size=len(cats)
        )
        print(f"[layout] cat_perm={list(perm)}, col_xs={col_xs}")

        box_y = -0.20
        # 0.8x of prior 24x24x14cm -> 19.2x19.2x11.2cm, so boxes look less
        # crowded on the table.
        box_half_x = 0.096
        box_half_y = 0.096
        box_half_z = 0.056
        wall_t = 0.005

        self.basket_poses = []
        self.basket_actors = []
        box_color = (0.7, 0.5, 0.3)
        for bx in col_xs:
            bx = float(bx)
            # Anchor basket bottom flush with TABLE_TOP_Z (not table_top, which
            # is TABLE_TOP_Z + 0.02 — that 2 cm offset is for spawning loose
            # objects so they fall under gravity; static baskets must NOT use
            # it or they appear to float above the table.)
            cz = self.TABLE_TOP_Z + box_half_z
            # basket_pos stored here is the outer box CENTER — the place/drop
            # logic inherited from CategorizeObjectsContribution adds +0.20 to its z
            # to get a drop position above the rim, which is 7 cm above
            # center, so drop ends up 13 cm above the rim.
            self.basket_poses.append(Pose([bx, box_y, cz], np.array([1.0, 0.0, 0.0, 0.0])))

            # Floor (thin slab at the bottom).
            create_primitive(
                self.scene, "box",
                Pose(p=[bx, box_y, cz - box_half_z + wall_t]),
                size={"half_size": (box_half_x, box_half_y, wall_t)},
                color=box_color, is_static=True,
            )
            # Four walls — thin boxes standing around the floor, full height.
            for wx, wy, hx, hy in [
                (-box_half_x + wall_t, 0.0, wall_t, box_half_y),  # -X wall
                (+box_half_x - wall_t, 0.0, wall_t, box_half_y),  # +X wall
                (0.0, -box_half_y + wall_t, box_half_x, wall_t),  # -Y wall
                (0.0, +box_half_y - wall_t, box_half_x, wall_t),  # +Y wall
            ]:
                create_primitive(
                    self.scene, "box",
                    Pose(p=[bx + wx, box_y + wy, cz]),
                    size={"half_size": (hx, hy, box_half_z)},
                    color=box_color, is_static=True,
                )

        # Loose objects on the avatar side, same X columns as boxes (with
        # independent ±1.5 cm Y jitter per object).
        obj_y = self.INTERRUPT_OBJECT_Y
        self._sort_targets = []
        for i, (obj_name, model_id) in enumerate(cats):
            oy = obj_y + float(np.random.uniform(-0.015, 0.015))
            # _spawn_categorize_object (inherited) handles the cube primitive.
            actor = self._spawn_categorize_object(
                obj_name, model_id, float(col_xs[i]), oy, table_top, upright_q,
            )
            self._sort_targets.append((actor, obj_name, model_id, i))

        self._clutter_actors = []

        # Parent's check_success iterates these lists — keep them empty so
        # only the three sort_targets are evaluated.
        self._example_actors = []
        self._avatar_sort_targets = []



    def _top_down_tcp(self, pos, yaw=None):
        """Top-down TCP with optional world-Z yaw for anisotropic objects."""
        if yaw is None:
            yaw = getattr(self, '_categorize_grasp_yaw', 0.0)
        Rz = t3d.euler.euler2mat(0.0, 0.0, float(yaw), axes="sxyz")
        q = t3d.quaternions.mat2quat(Rz @ self._R_DOWN)
        return Pose(np.asarray(pos, dtype=float), q)

    def _move_to_safe(self, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.array(arm.origin_pose.p)
        # Must clear 28cm-tall basket rim (box top at TABLE_TOP_Z + 0.30).
        safe_pos = np.array([arm_base[0], arm_base[1] + 0.30, self.TABLE_TOP_Z + 0.45])
        safe_link = tcp_to_link_pose(self._top_down_tcp(safe_pos), tcp_offset)
        self.move_and_execute(safe_link.to_pose7(), arm_tag)

    # Checked motion primitives live in CategorizeMotionMixin.

    @staticmethod
    def _sample_aabb_points(center, half, step=0.025):
        xs = np.arange(center[0] - half[0], center[0] + half[0] + 1e-6, step)
        ys = np.arange(center[1] - half[1], center[1] + half[1] + 1e-6, step)
        zs = np.arange(center[2] - half[2], center[2] + half[2] + 1e-6, step)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def _basket_aabb_points(self, basket_idx):
        bp = np.array(self.basket_poses[basket_idx].p, dtype=float)
        half = np.array([
            self._BASKET_HALF_X + self._OBSTACLE_MARGIN_XY,
            self._BASKET_HALF_Y + self._OBSTACLE_MARGIN_XY,
            self._BASKET_HALF_Z + self._OBSTACLE_MARGIN_Z,
        ])
        return self._sample_aabb_points(bp, half, step=0.025)

    def _actor_aabb_points(self, actor, name, model_id):
        r = float(self._get_object_cross_section_radius(name, model_id))
        center = np.asarray(self._get_object_world_center(actor, name, model_id), dtype=float)
        half = np.array([
            r + self._OBSTACLE_MARGIN_XY,
            r + self._OBSTACLE_MARGIN_XY,
            r + self._OBSTACLE_MARGIN_Z,
        ])
        return self._sample_aabb_points(center, half, step=0.02)

    @staticmethod
    def _to_numpy_array(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=float)

    def _actor_footprint_aabb(self, actor, obj_name, model_id):
        entity = getattr(actor, "entity", None)
        if entity is not None and hasattr(entity, "get_AABB"):
            try:
                aabb = self._to_numpy_array(entity.get_AABB()).reshape(-1, 3)
                lo = np.min(aabb, axis=0)
                hi = np.max(aabb, axis=0)
                return lo[:2], hi[:2]
            except Exception:
                pass

        center = self._get_object_world_center(actor, obj_name, model_id)
        ext = np.asarray(self._get_model_extents(obj_name, model_id), dtype=float)
        half_xy = np.max(ext[:2]) / 2.0
        return center[:2] - half_xy, center[:2] + half_xy

    @staticmethod
    def _get_model_extents(object_name: str, model_id: int = 0) -> np.ndarray:
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"], dtype=float)
        scale = np.array(md.get("scale", [1, 1, 1]), dtype=float)
        return extents * scale

    def _update_planner_obstacles(self, arm_tag, skip_actor=None, skip_basket_idx=None):
        self._categorize_skip_actor = skip_actor
        self._categorize_skip_basket = skip_basket_idx
        self._categorize_refresh_obstacles(arm_tag)

    def _categorize_scene_points(self, skip_actor=None, skip_basket_idx=None):
        """Refresh mplib's obstacle cloud with inflated baskets + on-table objects.

        ``skip_actor`` (held object) and ``skip_basket_idx`` (destination basket
        during descent) are excluded so the active place-target is not rejected
        as in-collision.
        """
        pts = []
        for i in range(len(self.basket_poses)):
            if skip_basket_idx is not None and i == skip_basket_idx:
                continue
            pts.append(self._basket_aabb_points(i))
        for actor, obj_name, model_id, _ in self._sort_targets:
            if skip_actor is not None and actor is skip_actor:
                continue
            center = np.asarray(self._get_object_world_center(actor, obj_name, model_id), dtype=float)
            if any(np.linalg.norm(center[:2] - np.asarray(bp.p, dtype=float)[:2]) <= 0.11
                   for bp in self.basket_poses):
                continue
            pts.append(self._actor_aabb_points(actor, obj_name, model_id))
        pc = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
        return pc

    def pick_and_place(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
        self._categorize_active_pick = pick
        self._categorize_hold_arm = arm_tag
        self._categorize_hold_local = None
        self._categorize_skip_actor = pick.actor
        self._categorize_skip_basket = None
        self._categorize_grasp_yaw = 0.0
        self._categorize_pick_center = np.asarray(pick.get_center(), dtype=float).copy()
        try:
            if not self._pick_and_place_checked(pick, place, arm_tag):
                self._reject_categorize_motion(f'{pick.label}: incomplete pick/place')
            return True
        finally:
            self._categorize_active_pick = None
            self._categorize_hold_local = None
            self._categorize_skip_actor = None
            self._categorize_skip_basket = None
            self._categorize_grasp_yaw = 0.0

    def _pick_and_place_checked(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
        """Dispatch to radius grasp by default and explicit YAML exceptions."""
        if pick.label == "100_seal" and self.SEAL_MEASURED_GRASP:
            return pick_and_place_seal(self, pick, place, arm_tag)
        if pick.label == "035_apple":
            center = np.asarray(pick.get_center(), dtype=float)
            if float(center[0]) >= self.APPLE_YAML_MIN_X:
                return self._pick_and_place_yaml(pick, place, arm_tag)
        if pick.label in self.YAML_ONLY_GRASP_LABELS:
            return self._pick_and_place_yaml(pick, place, arm_tag)
        if not self.USE_RADIUS_GRASP:
            return self._pick_and_place_yaml(pick, place, arm_tag)

        actor = pick.actor
        if actor is None:
            return False
        # Skip-list: round objects the parallel jaw can't hold (they eject).
        if pick.label in self.RADIUS_SKIP_LABELS:
            return False

        # --- Regression guards (so ungraspable objects fail GRACEFULLY) -------
        # (a) Height cap: objects taller than a stable top-down grip can hold
        #     (the 12 cm milk-box carton) are left standing on the table rather
        #     than grabbed near the top and tipped.
        from ..robot.franka_robot import to_numpy as _to_numpy
        try:
            aabb0 = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            obj_height = float(aabb0[1][2] - aabb0[0][2])
        except Exception:
            obj_height = 0.0
        if obj_height > self.RADIUS_MAX_GRASP_HEIGHT:
            return False
        # Live cross-section radius from the simulator AABB (robust to the
        # model_data extents being off under the z-up GLB load); fall back to
        # the model_data cross-section if the AABB read fails.
        obj_radius = float(pick.radius)
        live_top_z = None
        try:
            aabb = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            ext_xy = (aabb[1] - aabb[0])[:2]
            obj_radius = float(np.min(ext_xy) / 2.0)
            live_top_z = float(aabb[1][2])
        except Exception:
            pass
        # Per-object close fraction.  Priority: categorize-local override (tuned
        # for THIS pool's models + the radius below-equator scoop) → catalog
        # ``close_value`` → mixin radius formula.  The local map exists because
        # the catalog values are shared with put_object_cabinet, where a tighter
        # apple close (0.35) holds; here the radius scoop on apple@1 ejects the
        # round fruit at 0.35, so it needs a looser cage.
        close_value = None
        if pick.label in self.RADIUS_CLOSE_OVERRIDE:
            close_value = self.RADIUS_CLOSE_OVERRIDE[pick.label]
        else:
            try:
                close_value = get_entry(pick.label).close_value
            except Exception:
                close_value = None
        close_value = self._radius_close_for_live_width(close_value, obj_radius)
        # Per-object grasp-height lift.  Priority: categorize-local override
        # (OBJECT_GRASP_Z_OFFSET) → catalog ``grasp_z_offset``.  Without this
        # the radius scoop lands ``radius*0.3`` below center, which on tall
        # bodies like 031_jam-jar grips near the base.
        grasp_z_offset = 0.0
        if pick.label in self.OBJECT_GRASP_Z_OFFSET:
            grasp_z_offset = float(self.OBJECT_GRASP_Z_OFFSET[pick.label])
        else:
            try:
                grasp_z_offset = float(get_entry(pick.label).grasp_z_offset or 0.0)
            except Exception:
                grasp_z_offset = 0.0
        # A task-independent height offset can put the fingertips above a
        # thin object's entire body after it settles flat.  The stapler's
        # +20 mm catalog offset was calibrated for its upright pose; the
        # cooperative layout sometimes leaves a 17 mm-high resting body.
        if live_top_z is not None:
            center_z = float(pick.get_center()[2])
            grasp_z_offset = self._radius_offset_below_live_top(
                grasp_z_offset, center_z, obj_radius, live_top_z,
            )
        radius_pick = RadiusPickSpec(
            get_center=pick.get_center,
            radius=obj_radius,
            label=pick.label,
            close_value=close_value,
            grasp_z_offset=grasp_z_offset,
            get_grasp_yaw=(
                (lambda yaw=float(self.RADIUS_GRASP_YAW[pick.label]): yaw)
                if pick.label in self.RADIUS_GRASP_YAW
                else None
            ),
        )
        radius_place = RadiusPlaceSpec(
            pos=np.asarray(place.pos, dtype=float),
            label=place.label,
            drop_from_z=place.drop_from_z,
            release=True,
        )
        model_id = next((mid for candidate, name, mid, _ in
                         getattr(self, '_sort_targets', [])
                         if candidate is actor and name == pick.label), None)
        if pick.label == 'cube' or (pick.label, model_id) in {
            ('086_woodenblock', 1), ('035_apple', 1),
        } or (pick.label, model_id) in getattr(self, 'RADIUS_SELECTOR_MODELS', ()):
            return pick_and_place_radius(
                self, radius_pick, radius_place, actor, arm_tag,
                object_name=pick.label, model_id=0 if pick.label == 'cube' else model_id,
                object_scale=1. if pick.label == 'cube' else self._get_object_scale(pick.label, model_id),
            )
        # Keep the legacy radius geometry, but route every yaw through the
        # family adapter. The shared mixin's explicit oriented calls bypass
        # virtual screw/cartesian dispatch and permit a joint-route fallback.
        self._categorize_grasp_yaw = (
            float(radius_pick.get_grasp_yaw()) if radius_pick.get_grasp_yaw else 0.0
        )
        radius_pick = replace(
            radius_pick, get_grasp_yaw=None,
            after_close=lambda: self._categorize_begin_hold(arm_tag),
            require_lift=True, transport_retention_tolerance=.03,
        )
        checked_release = uses_lower_release(self, pick.label)
        if checked_release:
            radius_place = replace(radius_place, release=False)
        completed = bool(TopDownPickPlaceMixin.pick_and_place(self, radius_pick, radius_place, arm_tag))
        if completed and checked_release:
            return lower_release(self, actor, pick.label, arm_tag)
        return completed

    @staticmethod
    def _radius_close_for_live_width(close_value, radius):
        """Keep a catalog preset only when it actually contacts this model.

        Catalog entries identify an asset family, while the categorize pool
        may select another model.  In particular woodenblock@1 is only about
        35 mm wide; the @0 preset opens a 41.6 mm gap and never pinches it.
        If a preset cannot touch the live cross section, use a 4 mm bite per
        finger, matching the shallow clamp used for the larger model.
        """
        if close_value is None or float(close_value) * 0.04 < float(radius):
            return close_value
        return float(np.clip((float(radius) - 0.004) / 0.04, 0.0, 1.0))

    @staticmethod
    def _radius_offset_below_live_top(offset, center_z, radius, top_z):
        """Leave an 8 mm fingertip purchase beneath the actual resting top."""
        return min(
            float(offset),
            float(top_z) - 0.008 - float(center_z) + 0.3 * float(radius),
        )

    def _categorize_yaml_preflight(self, pick, place, arm_tag, safe_z):
        return (
            milk_continuation_check(self, pick, place, arm_tag, safe_z)
            if self.MILK_CONTINUATION_PREFLIGHT and pick.label == "038_milk-box"
            else None
        )

    def _pick_and_place_yaml(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
        """Pick a loose object (described by ``pick``) and drop it at ``place``.

        Semantics:
          - Top-down approach, below-equator grasp offset (``radius * 0.3``).
          - Two-phase transit: lateral at safe_z, vertical descent to pre-grasp.
          - Partial close sized to ``radius * 0.7`` (avoids mesh penetration).
          - Hover-and-release over ``place.pos`` if ``drop_from_z`` is set;
            otherwise descend to ``place.pos.z`` before opening the gripper.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # Reset IK seed to MEASURED qpos at the start of every pick.  The
        # prior pick's final ``_cached_target`` is a qpos at a *different*
        # column's basket; feeding it to Genesis IK for the NEW pre-grasp
        # target can lock onto a bad local minimum where FK doesn't match
        # the target (seen on seed 20 D3 apple — 28 cm err, never touched
        # the object).  Measured qpos is always a valid starting config.
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float
        ).ravel().copy()

        obj_center = pick.get_center()
        obj_radius = pick.radius

        # Transport ABOVE the drop target.  Default safe_z clears the basket
        # rim (baskets are 28cm tall; top at TABLE_TOP_Z + 30cm).  Callers
        # drop into deeper containers by passing a higher ``drop_from_z``.
        safe_z = place.drop_from_z if place.drop_from_z is not None else self.TABLE_TOP_Z + 0.38

        # Open gripper
        self.open_gripper(arm_tag)

        if not self.USE_YAML_GRASPS:
            return False
        if pick.label in self.YAML_GRASP_SKIP:
            return False

        if self._USE_OBSTACLE_INFLATION:
            self._update_planner_obstacles(arm_tag, skip_actor=pick.actor)
        actor = pick.actor
        if actor is None:
            return False

        model_id = None
        for _a, _n, _m, _ in self._sort_targets:
            if _a is actor:
                model_id = _m
                break
        if model_id is None:
            return False

        obj_pose_for_grasp = actor.get_pose()
        object_scale = self._get_object_scale(pick.label, model_id)
        result = self._select_and_execute_task_grasp(
            pick.label, obj_pose_for_grasp, arm_tag,
            model_id=model_id, object_scale=object_scale,
            max_candidates=20,
            post_plan_check=self._categorize_yaml_preflight(pick, place, arm_tag, safe_z),
        )
        if result is None:
            return False

        _grasp_link, _pre_link, grasp = result
        ee_pose = np.array(arm.get_ee_pose(), dtype=float)
        grasp_pos = ee_pose[:3].copy()
        grasp_quat = ee_pose[3:7].copy()
        obj_pre_close = pick.get_center()
        print(
            f"[categorize] YAML grasp {pick.label}@{model_id}/{grasp.name}: "
            f"tcp={grasp_pos}, object={obj_pre_close}",
            flush=True,
        )

        # Close target priority:
        #   1. Per-object metadata `grasp_close_target_m` (set on PickSpec from
        #      the object's `model_data{N}.json`).  Used to override the
        #      formula for objects that need a tighter clamp than the size
        #      heuristic gives (e.g. tall cylinders like 100_seal where
        #      friction along the axis must hold the object's weight).
        #   2. Adaptive formula.  Round/small meshes (r<3 cm, e.g. apple) need
        #      a shallow squeeze (`r-0.005`) — deeper commands let fingers
        #      overpower mesh contact before caging, shoving the object
        #      sideways.  Flat/large meshes (r>=3 cm, e.g. wooden block) need
        #      a deep squeeze (`r*0.7`) — fingers contact a flat face
        #      squarely, the block wins the PD push and stops fingers at the
        #      face with ~100 N firm grip.  Verified: deliver apple → 4/4
        #      with shallow; cate-coop block → 3/3 with deep.
        r = float(obj_radius)
        ct_meta = getattr(pick, "close_target_m", None)
        if ct_meta is not None:
            close_target = max(0.010, float(ct_meta))
        elif r < 0.030:
            close_target = max(0.010, r - 0.005)
        else:
            close_target = max(0.010, r * 0.7)
        self.set_gripper(close_target / 0.04, arm_tag)
        # Settle 300 ticks (was 200) so the gripper-object contact reaches
        # static equilibrium before the lift accelerates the object — too
        # short a settle leaves residual finger velocity that gets reversed
        # by the lift's upward command, breaking the friction grip.
        for _ in range(300):
            self.step_sim()
        self._categorize_begin_hold(arm_tag)
        # Use the FRESH object centre for the lift target — `obj_center` was
        # captured at top-of-pick and is stale if the pre-grasp transit
        # bumped the object.
        # Re-read object centre for the lift target — `obj_center_now` is
        # only set in the synthetic-grasp branch; for the YAML path we need
        # a fresh read here regardless.
        lift_obj_center = pick.get_center()
        lift_mid = grasp_pos.copy()
        lift_mid[2] = lift_obj_center[2] + obj_radius  # lift to just above object top
        if not self._move_tcp_screw(Pose(lift_mid, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        # Debug: check grip
        from ..robot.franka_robot import _get_dof_idx, to_numpy
        qpos_after = to_numpy(arm.entity.get_qpos())
        finger_qpos = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_qpos.append(float(qpos_after[idx]))
        obj_pos_after = pick.get_center()

        # Column-layout milk: one measured corner rise plus a checked diagonal
        # replaces the vertical lift and horizontal carry, shortening the time
        # the side grasp must hold the carton. A preflight failure keeps the
        # unchanged route below; an execution failure rejects without opening.
        corner_carried = False
        if uses_milk_corner(self, pick.label, model_id, place):
            corner_carried = milk_corner_carry(self, pick, place, actor, arm_tag, safe_z)

        # Lift to safe height via screw motion (straight-line, gripper held).
        # Keep the YAML grasp orientation through transport; switching back
        # to the synthetic top-down TCP can rotate round objects out of the
        # fingers after an otherwise good grasp.
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z])
        if not corner_carried and not self._move_tcp_screw(Pose(lift_pos, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        obj_pos_lifted = pick.get_center()
        lift_dz = float(obj_pos_lifted[2] - obj_pos_after[2])
        print(
            f"[categorize] YAML lift {pick.label}: close={close_target / 0.04:.4f}, "
            f"fingers={finger_qpos}, object={obj_pos_lifted}, dz={lift_dz:.4f}",
            flush=True,
        )
        if pick.label != "100_seal" and lift_dz < 0.04:
            self.open_gripper(arm_tag)
            for _ in range(120):
                self.step_sim()
            return False

        # Transport to above drop target, then release. No multi-segment
        # drop — a second screw motion into the container occasionally
        # flips the wrist and hurls the object.  For hover-and-release
        # (basket pattern) the object falls cleanly from safe_z.
        # Keep the destination basket in the obstacle set — the path goes
        # ABOVE all baskets at safe_z, so a planner detour that dips toward
        # another basket gets repelled.
        if not corner_carried and self._USE_OBSTACLE_INFLATION:
            self._update_planner_obstacles(arm_tag, skip_actor=pick.actor)
        # Annotated side grasps hold the object off the TCP centerline.  Aim
        # the object center at the basket, using the measured offset after
        # lift while its orientation is held fixed.  A TCP-centered target
        # left milk cartons 4–6 cm off center and perched on the basket rim.
        held_center_offset = np.asarray(obj_pos_lifted, dtype=float) - np.asarray(
            arm.get_ee_pose()[:3], dtype=float,
        )
        place_xyz = self._yaml_transport_target(
            place.pos, safe_z, held_center_offset,
        )
        if not corner_carried and not self._move_tcp_screw(Pose(place_xyz, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        # Settle so the object stops swinging in the gripper.
        for _ in range(100):
            self.step_sim()

        obj_pos_transport = pick.get_center()
        print(
            f"[categorize] YAML carry {pick.label}: target={place_xyz}, "
            f"object={obj_pos_transport}, tcp={arm.get_ee_pose()[:3]}",
            flush=True,
        )

        # If caller wants a descent-before-release (e.g. place on a plate),
        # move TCP down to place.pos.z first.
        if place.drop_from_z is None and len(place.pos) >= 3:
            # Drop the destination basket from obstacles so its own walls
            # don't block the descent.
            if self._USE_OBSTACLE_INFLATION:
                self._update_planner_obstacles(
                    arm_tag,
                    skip_actor=pick.actor,
                    skip_basket_idx=place.basket_idx,
                )
            descend_pos = np.array([place.pos[0], place.pos[1], float(place.pos[2])])
            if not self._move_tcp_screw(Pose(descend_pos, grasp_quat), arm_tag):
                self.open_gripper(arm_tag)
                for _ in range(80):
                    self.step_sim()
                return False
            for _ in range(50):
                self.step_sim()

        # YAML-only objects (including Rubik) bypass the radius dispatcher.
        if uses_lower_release(self, pick.label):
            return lower_release(self, pick.actor, pick.label, arm_tag)

        # Release.  Let the object drop under gravity.
        self.open_gripper(arm_tag)
        for _ in range(250):
            self.step_sim()

        obj_pos_release = pick.get_center()

        return True

    @staticmethod
    def _yaml_transport_target(basket_pos, safe_z, held_center_offset):
        target = np.array([basket_pos[0], basket_pos[1], safe_z], dtype=float)
        target[:2] -= np.asarray(held_center_offset, dtype=float)[:2]
        return target

    def _open_basket_lids(self):
        """Set basket lid joints to URDF-default (open) pose and hold with
        high PD gains so falling objects don't push the lids closed."""
        if not self._CATEGORIZE_OPEN_LIDS:
            return
        for actor in self.basket_actors:
            entity = getattr(actor, "entity", None)
            if entity is None:
                continue
            try:
                q = entity.get_qpos()
                q_np = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
                q_np = np.asarray(q_np).ravel().astype(float)
                q_np[:] = 0.0
                entity.set_qpos(q_np)
                # High kp on every lid DoF so they stay open
                try:
                    kp = entity.get_dofs_kp()
                    kv = entity.get_dofs_kv()
                    kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
                    kv_np = kv.cpu().numpy() if hasattr(kv, "cpu") else np.asarray(kv)
                    kp_np = np.asarray(kp_np).ravel().astype(float)
                    kv_np = np.asarray(kv_np).ravel().astype(float)
                    kp_np[:] = 500.0
                    kv_np[:] = 50.0
                    set_dofs_kp_kv_compat(entity, kp=kp_np, kv=kv_np)
                    entity.control_dofs_position(q_np)
                except Exception as e:
                    pass
            except Exception as e:
                pass

    @stop_on_motion_rejection
    def play_blind_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()
        self._move_to_safe(arm_tag)

        def _robot_sort_idx(idx):
            actor, obj_name, model_id, basket_idx = self._sort_targets[idx]
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            ok = self.pick_and_place(pick, place, arm_tag)
            self._move_to_safe(arm_tag)
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            return bool(ok and dist_xy <= 0.10)

        any_success = False
        self._blind_robot_order = [
            int(i) for i in np.random.permutation(len(self._sort_targets))
        ]
        for idx in self._blind_robot_order:
            ok_robot = _robot_sort_idx(idx)
            any_success = any_success or ok_robot
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 2400:
                self.step_sim()
                tail += 1
        for _ in range(200):
            self.step_sim()
        return any_success

    def _success_target_groups(self):
        return [("robot", self._sort_targets)]

    # A sorted object rests in its basket: its lowest point at most this far
    # above the basket rim (objects stacked inside a basket stay below it),
    # and held by neither the avatar nor the gripper.  The xy-only footprint
    # rule alone accepted an object carried over the basket (the scripted
    # drop height is basket centre + 0.18 m).
    SUCCESS_REST_MAX_ABOVE_RIM_M = 0.04

    def _object_resting_in_basket(self, actor, obj_center, basket_pos) -> bool:
        rim_z = float(basket_pos[2]) + float(self._BASKET_HALF_Z)
        bottom = entity_bottom_z(actor, float(obj_center[2]))
        if bottom > rim_z + self.SUCCESS_REST_MAX_ABOVE_RIM_M:
            return False
        return not held_by_anyone(self, actor, obj_center)

    def check_success(self) -> bool:
        if not self.plan_success:
            return False

        # Success requires the object footprint, not just its center, to sit
        # inside the basket.  Center-only checks accepted long objects perched
        # on a rim (e.g. seal seed 2: center dist 7 cm, visible failure).
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        all_correct = True
        for side, targets in self._success_target_groups():
            for actor, obj_name, model_id, basket_idx in targets:
                obj_center = self._get_object_world_center(actor, obj_name, model_id)
                basket_pos = np.array(self.basket_poses[basket_idx].p)
                dist_xy = np.linalg.norm(obj_center[:2] - basket_pos[:2])
                lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
                rel_lo = lo_xy - basket_pos[:2]
                rel_hi = hi_xy - basket_pos[:2]
                footprint_ok = bool(
                    np.all(rel_lo >= -footprint_half) and
                    np.all(rel_hi <= footprint_half)
                )
                center_ok = bool(dist_xy <= center_threshold)
                resting_ok = self._object_resting_in_basket(
                    actor, obj_center, basket_pos,
                )
                if not (center_ok and footprint_ok and resting_ok):
                    all_correct = False
                else:
                    pass

        # When avatar-collision tracking is enabled, any robot↔avatar contact
        # during the episode is a failure regardless of basket placement.
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            all_correct = False

        return all_correct

    def _basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for side, targets in self._success_target_groups():
            for actor, obj_name, model_id, basket_idx in targets:
                obj_center = self._get_object_world_center(actor, obj_name, model_id)
                basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
                dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
                lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
                rel_lo = lo_xy - basket_pos[:2]
                rel_hi = hi_xy - basket_pos[:2]
                center_ok = bool(dist_xy <= center_threshold)
                footprint_ok = bool(
                    np.all(rel_lo >= -footprint_half)
                    and np.all(rel_hi <= footprint_half)
                )
                resting_ok = self._object_resting_in_basket(
                    actor, obj_center, basket_pos,
                )
                ok = bool(center_ok and footprint_ok and resting_ok)
                all_correct = all_correct and ok
                per_object.append({
                    "side": side,
                    "object": obj_name,
                    "model_id": int(model_id),
                    "basket_idx": int(basket_idx),
                    "object_pos": obj_center.tolist(),
                    "basket_pos": basket_pos.tolist(),
                    "dist_xy": dist_xy,
                    "rel_aabb_lo_xy": rel_lo.tolist(),
                    "rel_aabb_hi_xy": rel_hi.tolist(),
                    "center_success": center_ok,
                    "footprint_success": footprint_ok,
                    "resting_success": resting_ok,
                    "success": ok,
                })
        return {
            "basket_center_threshold": center_threshold,
            "basket_footprint_half_xy": float(footprint_half),
            "basket_all_correct": bool(all_correct),
            "basket_objects": per_object,
        }

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._basket_metrics())
        return metrics
