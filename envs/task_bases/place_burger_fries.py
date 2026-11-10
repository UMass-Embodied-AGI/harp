"""Place burger and fries.

Single-arm Franka adaptation of RoboTwin's bimanual ``place_burger_fries``
(https://robotwin-platform.github.io/doc/tasks/place_burger_fries.html).
The robot picks a hamburger and a french-fries carton from the right side
of the table and drops them onto a tray on the left side, sequentially with
a single arm.  No avatar.

Randomization (per-reset, seeded by ``np.random.seed(seed)``):
- One burger + one fries carton (default), or two of each in hard mode.
- Object positions drawn from per-type spawn rectangles with rejection
  sampling so spawned items never start within 10 cm of each other.
- Tray xy drawn from a small range (~15 cm wide) on the left, kept inside
  the Franka's dexterous workspace.
- Tray colour drawn from a uniform RGB box.  Genesis's ``Default`` surface
  override is applied even though the GLB ships its own material — visual
  fidelity depends on whether the renderer respects the override (it does
  for the rasterizer; raytracer behaviour is untested).

Drop slots (positions on the tray) scale with item count: 1→centre,
2→left/right, 3→top-pair + bottom-centre, 4→2×2 grid.

Picking uses the lower-level helpers from ``TopDownPickPlaceMixin``
(``_top_down_tcp`` / ``_move_seeded`` / ``_move_screw`` / ``_move_cartesian``)
with a custom close + lift step so we can pin a per-object close target,
verify post-lift that the object actually came with the gripper, and use
slow Cartesian interpolation for the long horizontal transit (plan_screw
oscillates the gripper PD enough to slip the slim 3.3 cm fries box).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

import transforms3d as t3d

from ..base_task import BaseTask
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..manipulation import TopDownPickPlaceMixin
from ..utils import (
    Actor,
    Pose,
    load_object,
    to_numpy,
)
from ..grasp import tcp_to_link_pose
from .resting_guard import avatar_holds, entity_bottom_z, robot_holds


_HAMBURG_ID = "006_hamburg"
_FRIES_ID = "005_french-fries"
_TRAY_ID = "008_tray"


@dataclass
class FoodSpec:
    """Per-food parameters: asset id, model variant, spawn pose, grasp."""
    name: str
    asset_id: str
    model_id: int
    spawn_quat: np.ndarray
    spawn_z_offset: float       # z above TABLE_TOP_Z at spawn
    friction: float
    close_value: float          # set_gripper target (0=closed, 1=open)
    below_center: float         # grip pose z = obj_center.z - below_center
    tcp_yaw_deg: float          # rotate top-down gripper about approach axis
    spawn_xy_range: tuple       # ((xlo, xhi), (ylo, yhi))
    # Per-spec horizontal transit override (0 = use the class default).  A
    # big flat box needs many small Cartesian steps so it cannot pitch out
    # of the grip; a compact item is fine with the coarser class default and
    # creeps if held for the extra sim time, so the two are tuned separately.
    transit_steps: int = 0
    transit_sim: int = 0
    # When True, rotate the top-down grip so the jaws close across the
    # object's MEASURED short horizontal (mesh-X) axis at grasp time, instead
    # of a fixed world-aligned tcp_yaw_deg.  A long/flat object that settles a
    # few degrees off its spawn yaw is otherwise gripped on a corner and
    # dangles/pitches out; aligning the jaws to its real axis grips it square.
    align_to_object_yaw: bool = False
    # Collision-decomposition override forwarded to the mesh loader.  None =
    # Genesis default (CoACD when hull volume error > 15%).  ``float("inf")``
    # forces a single convex hull — needed for light meshes whose multi-hull
    # base rests on several non-coplanar hull bottoms and rocks forever.
    decompose_error_threshold: float | None = None


# upright = mesh-Y → world-Z (most assets are y-up)
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
# legacy burger orientation = R_z(90°) * R_x(90°) — also mesh-Y → world-Z,
# differs only in xy yaw.  Kept for backwards compatibility.
_Q_BURGER = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)


# 006_hamburg model_0 is a wide flat 6.6 cm disc — a parallel jaw clamps its
# full diameter with near-zero compression and any wrench ejects it on lift
# (the non-caging-grasp limit of Genesis contact).
# model_5 is the ELONGATED patty variant (4.7 x 2.4 x 7.6 cm): under _Q_BURGER
# its 4.7 cm narrow axis lands on world-Y, exactly the default top-down finger
# closing axis, so the jaws clamp the narrow waist and the long axis gives a
# long contact line — the same flat-object approach as the bread loaf.
# close_value 0.50 (NOT tighter): on this thin mesh a firm squeeze crush-EJECTS
# the patty before it lifts (sweep: 0.35/0.40 fail, 0.45/0.50 hold; 0.50 best
# lift_dz +0.048). 0.50 ≈ 4.0 cm finger gap = ~0.7 cm gentle compression on the
# 4.7 cm waist. friction 5.0 (capped) carries the vertical load through transit.
# model_5 does not read as a hamburger on video.  Among the round models,
# model_1 ejects (0/6 at close
# 0.55/0.60 — the crush-eject wall above), but round model_4 (6.7 x 3.1 x
# 6.6 cm) at close 0.55 went 6/6 vs 5/6 for the model_5 baseline.  Override
# per-run via config burger_model_id / burger_close_value / burger_below_center.
_BURGER_SPEC = FoodSpec(
    name="burger", asset_id=_HAMBURG_ID, model_id=4,
    spawn_quat=_Q_BURGER,
    spawn_z_offset=0.05, friction=5.0,
    close_value=0.55, below_center=0.005, tcp_yaw_deg=0.0,
    spawn_xy_range=((+0.05, +0.20), (-0.08, +0.05)),
)

# 005_french-fries carton.  Tuned via a grasp-parameter sweep; the surviving
# recipe and the dead ends behind each knob:
#   * model_0 (7.06 x 9.62 x 3.27 cm), NOT model_1: model_1 is a 5.6 cm-wide
#     HOLLOW thin-walled carton (CoACD -> 11 pieces) whose walls flex and
#     slip; model_0's wider, more solid body gives the jaws a form-closure
#     grip.  (model_3/2 are >8 cm wide -> the fingers graze them, dz=0.)
#   * Laid flat with mesh +Y mapped to world +Z, gripped top-down across the
#     7.06 cm short axis.
#     Standing the carton never lifts (the jaws slide off thin walls /
#     tapered top, dz=0).
#   * align_to_object_yaw=True is the single biggest win: a fixed
#     world-aligned grip catches the settled (slightly-yawed) long box on a
#     corner and it dangles out (0/N); squaring the jaws to the box's
#     measured short axis took it to ~2/3 (sweep4/5).
#   * spawn_z_offset 0.005 (a low drop, NOT 0.02): dropping the box 2 cm lets
#     it bounce to a random yaw/pitch; settling it flat from ~rest cut the
#     per-seed variance and lifted success 2/5 -> 3/5 (sweep6).
#   * close 0.25 + friction 5.0 (capped); DEFAULT transit (smoothing the
#     Cartesian transit to 80-160 steps consistently made it WORSE, 0/N).
# Net: a marginal-but-real ~3/5 expert grasp on a genuinely hard hollow
# carton (the burger beside it is ~3/3, so the task is ~60% end-to-end).
_FRIES_SPEC = FoodSpec(
    name="fries", asset_id=_FRIES_ID, model_id=0,
    spawn_quat=_Q_UPRIGHT,
    spawn_z_offset=0.005, friction=5.0,
    close_value=0.25, below_center=0.005, tcp_yaw_deg=90.0,
    spawn_xy_range=((-0.02, +0.15), (-0.22, -0.08)),
    align_to_object_yaw=True,
    # Genesis's default 15% hull threshold CoACD-splits this
    # carton into 11 hulls, 4 of which touch the ground non-coplanarly; the
    # 28 g standing box then rocks in a contact-solver limit cycle forever
    # ("fries keep bouncing").  A single convex hull (only 23% fatter than
    # the mesh) gives one flat contact face and it rests still.
    decompose_error_threshold=float("inf"),
)


class PlaceBurgerFries(TopDownPickPlaceMixin, BaseTask):
    """Robot picks a hamburger and a french-fries carton (one of each in
    the default layout, two of each in hard mode) and places them all on a
    cafeteria tray.
    """

    INSTRUCTION = "place the hamburger and the french fries on the tray"

    use_avatar = False

    # Keep the original avatar-side edge at y=+0.35 and extend only toward the
    # Franka at y=-0.65. The robot-side edge at y=-0.80 places the whole base
    # 0.15 m inboard instead of leaving it suspended beyond the tabletop.
    TABLE_HALF_SIZE = (0.60, 0.575)
    TABLE_CENTER_XY = (0.0, -0.225)

    # Shared by neutral / assist / interrupt variants.
    # VLA primary view: near top-down over the full default 1.20 x 0.70 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, 0.005, 1.90],
        "forward": [0.0, -0.005, -1.05],
    }]

    # The tray is authored Y-up; map its thin/local-Y axis onto world Z so the
    # 30.7 x 18.7 cm face lies flat on the table.
    _Q_TRAY = _Q_UPRIGHT
    # Kept as a class attribute for the assist subclass which needs to set
    # the avatar-tossed burger's quaternion.
    _Q_BURGER = _Q_BURGER

    # ---- Layout ----------------------------------------------------------
    # Tray on the left.  Each food has its own spawn xy range (set in its
    # FoodSpec at module scope).
    TRAY_X_RANGE = (-0.25, -0.10)
    TRAY_Y_RANGE = (+0.00, +0.10)

    # Min separation between any two spawned food items (world XY).
    SPAWN_MIN_SEP = 0.085
    # Small margin around the tray footprint when sampling initial food poses.
    # Without this, the right edge of a randomized tray can overlap the food
    # spawn rectangles and some items start inside the tray.
    SPAWN_TRAY_CLEARANCE = 0.005

    # Tray colour range (uniform RGB) — kept for log only; GLB material
    # overrides ``surface=Default(color=)`` so this is informational.
    TRAY_COLOR_LO = (0.20, 0.20, 0.20)
    TRAY_COLOR_HI = (0.95, 0.95, 0.95)

    # ---- Tray geometry (008_tray GLB) ------------------------------------
    # World AABB: 30.7 × 18.7 × 3.1 cm.  Success tolerance is half-extent
    # + 1 cm overhang in each axis (rectangular check).
    TRAY_RIM_Z = 0.015         # ≈ half tray thickness
    TRAY_HALF_X = 0.165
    TRAY_HALF_Y = 0.105

    # ---- Pick parameters --------------------------------------------------
    TRANSPORT_Z_ABOVE_TABLE = 0.22
    DROP_HOVER_Z_ABOVE_TRAY = 0.06
    LIFT_HEIGHT_ABOVE_GRASP = 0.18

    # Single-shot pick: ONE attempt per object, no re-pick on slip.  VLA data
    # collection wants clean 1-shot pick-and-place demonstrations — a re-pick
    # (the arm going back to grab a slipped object) pollutes the trajectory, so
    # if an object isn't placed on the first try the episode simply fails.
    SPAWN_SAMPLE_ATTEMPTS = 1000
    SPAWN_FALLBACK_GRID_COUNT = 31
    TRAY_FRICTION = 2.0
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 80.0
    TABLE_FALL_Z_MARGIN = 0.05
    GRASP_TABLE_Z_MARGIN = 0.003
    PRE_GRASP_HEIGHT = 0.15
    GRASP_SETTLE_STEPS = 100
    GRASP_ALIGNMENT_XY_TOL = 0.015
    GRASP_ALIGNMENT_SETTLE_STEPS = 80
    CLOSE_SETTLE_STEPS = 200
    POST_LIFT_SETTLE_STEPS = 60
    LIFT_DZ_MIN = 0.05
    LIFT_FAIL_RELEASE_SETTLE_STEPS = 50
    # Default horizontal transit (compact items, e.g. the burger).  The big
    # flat fries carton overrides this per-spec with many small steps — see
    # FoodSpec.transit_steps / _FRIES_SPEC.
    TRANSIT_CARTESIAN_STEPS = 40
    TRANSIT_SIM_PER_STEP = 20
    POST_TRANSIT_SETTLE_STEPS = 50
    TRANSIT_SLIP_XY_TOL = 0.10
    DROP_SETTLE_STEPS = 40
    RELEASE_SETTLE_STEPS = 200
    FINAL_SETTLE_STEPS = 150
    SUCCESS_TABLE_Z_MARGIN = 0.02
    # A placed food rests on the tray: its lowest point is at most this far
    # above the tray surface (TABLE_TOP_Z + 2 * TRAY_RIM_Z), and neither the
    # avatar nor the gripper holds it.  Without this, a food the eval human
    # holds over the tray counted as placed.
    SUCCESS_REST_MAX_ABOVE_TRAY_M = 0.04

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

        self.tray_actor = None
        # Unified list of (Actor, FoodSpec) for all spawned food items.
        # Burger is always index 0; optionals follow in declared order.
        self.food_actors: list[tuple[Actor, FoodSpec]] = []
        self.food_spawn_xys: list[tuple[float, float]] = []
        self._tray_xy: tuple[float, float] = (-0.20, +0.05)
        self._tray_color: tuple = (0.5, 0.5, 0.5, 1.0)

    def _create_table(self, table_height=0.74):
        """Build the extended footprint identically for every table variant."""
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        )

    # ------------------------------------------------------------------
    # Drop slot layouts on the tray (offsets relative to tray centre xy).
    # ------------------------------------------------------------------
    @staticmethod
    def _drop_slots(n: int) -> list[tuple[float, float]]:
        if n == 1:
            return [(0.0, 0.0)]
        if n == 2:
            return [(-0.06, 0.0), (+0.06, 0.0)]
        if n == 3:
            return [(-0.07, +0.03), (+0.07, +0.03), (0.0, -0.04)]
        if n == 4:
            return [(-0.07, +0.03), (+0.07, +0.03),
                    (-0.07, -0.03), (+0.07, -0.03)]
        raise ValueError(f"unsupported drop slot count: {n}")

    # ------------------------------------------------------------------
    # Random non-overlapping spawn position.
    # ------------------------------------------------------------------
    def _xy_overlaps_tray(self, x: float, y: float) -> bool:
        tray_x, tray_y = self._tray_xy
        clearance = self.SPAWN_TRAY_CLEARANCE
        return (
            abs(float(x) - tray_x) <= self.TRAY_HALF_X + clearance
            and abs(float(y) - tray_y) <= self.TRAY_HALF_Y + clearance
        )

    def _sample_spawn_xy(self, xy_range, occupied, max_tries: int = None
                         ) -> tuple[float, float]:
        max_tries = self.SPAWN_SAMPLE_ATTEMPTS if max_tries is None else max_tries
        (xlo, xhi), (ylo, yhi) = xy_range
        for _ in range(max_tries):
            x = float(np.random.uniform(xlo, xhi))
            y = float(np.random.uniform(ylo, yhi))
            if self._xy_overlaps_tray(x, y):
                continue
            if all(float(np.hypot(x - ox, y - oy)) >= self.SPAWN_MIN_SEP
                   for ox, oy in occupied):
                return x, y
        best = None
        best_dist = -np.inf
        for x in np.linspace(xlo, xhi, self.SPAWN_FALLBACK_GRID_COUNT):
            for y in np.linspace(ylo, yhi, self.SPAWN_FALLBACK_GRID_COUNT):
                x = float(x)
                y = float(y)
                if self._xy_overlaps_tray(x, y):
                    continue
                min_dist = min(
                    (float(np.hypot(x - ox, y - oy)) for ox, oy in occupied),
                    default=np.inf,
                )
                if min_dist >= self.SPAWN_MIN_SEP and min_dist > best_dist:
                    best = (x, y)
                    best_dist = min_dist
        if best is not None:
            return best
        raise RuntimeError(
            f"could not find a non-overlapping spawn outside tray in {xy_range} "
            f"after {max_tries} tries (occupied={occupied}, tray_xy={self._tray_xy})"
        )

    # The plate is the ``008_tray`` GLB (rectangular cafeteria tray w/ rim,
    # ~30.7 × 18.7 × 3.1 cm).  Per-task colour is intentionally NOT applied
    # here: the GLB ships with baked PBR materials that the renderer
    # prefers over ``surface=Default(color=…)``, so ``self._tray_color`` is
    # kept around for the launcher log only.

    def _add_tray_with_color(self, pose: Pose, color):
        """Spawn the 008_tray GLB as a DYNAMIC, physics-simulated body.

        A static tray sat wherever it was placed — and because 008_tray's mesh
        origin is ~1.3 cm below its geometric centre, placing the origin at
        ``table_top + TRAY_RIM_Z`` left the tray hovering ~1.2 cm above the
        table.  As a dynamic body it just falls during the reset-settle and
        rests flat on the table under gravity (real contact), so it neither
        floats nor needs the origin offset hand-computed.  ``color`` is logged
        but unused (the GLB's baked material wins over a surface override)."""
        return load_object(
            self.scene, pose, _TRAY_ID,
            model_id=0, convex=True, is_static=False, friction=self.TRAY_FRICTION,
        )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _burger_spec(self) -> FoodSpec:
        """Burger spec with optional config overrides.

        ``burger_model_id`` / ``burger_close_value`` / ``burger_below_center``
        select the 006_hamburg variant and grip, e.g. to compare the recognizable
        round models (0-2/4) against the elongated default (5).
        """
        mid = int(self.config.get("burger_model_id", _BURGER_SPEC.model_id))
        close = float(
            self.config.get("burger_close_value", _BURGER_SPEC.close_value)
        )
        below = float(
            self.config.get("burger_below_center", _BURGER_SPEC.below_center)
        )
        if (mid, close, below) == (
            _BURGER_SPEC.model_id,
            _BURGER_SPEC.close_value,
            _BURGER_SPEC.below_center,
        ):
            return _BURGER_SPEC
        return dataclasses.replace(
            _BURGER_SPEC, model_id=mid, close_value=close, below_center=below
        )

    def _food_specs_for_episode(self) -> list[FoodSpec]:
        """One hamburger + one fries carton in the default (simplified)
        layout; a second burger + fries pair in hard mode."""
        burger = self._burger_spec()
        if self.hard_mode_enabled():
            return [burger, _FRIES_SPEC, burger, _FRIES_SPEC]
        return [burger, _FRIES_SPEC]

    # In the default (easy) layout the robot manipulates ONE
    # object.  The fries carton starts already on the tray (spawned over its
    # drop slot, settles onto the tray under gravity) and the robot only picks
    # the burger.  Hard mode keeps all foods loose.  The assist variant turns
    # this off and instead spawns a single robot food (robot 1 / avatar 1).
    PREPLACE_FOOD_ON_TRAY = True

    def _preplaced_indices(self, specs: list[FoodSpec]) -> set[int]:
        if (
            not self.PREPLACE_FOOD_ON_TRAY
            or self.hard_mode_enabled()
            or len(specs) < 2
        ):
            return set()
        return {1}  # the fries carton; the robot picks the burger (index 0)

    def _settle_scene(self, n_steps: int = None):
        # Initialization-time correction: the GLB origins
        # sit near mid-body, so spawning the origin at table_top + a small
        # offset buries the mesh in the tabletop for the first rendered
        # frames until physics ejects it.  Lift each actor so its AABB bottom
        # starts just above the table before the settle (init-time set_pos,
        # before play_once — allowed).
        def _lift(entity):
            try:
                aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
                deficit = (self.TABLE_TOP_Z + 0.002) - float(aabb[0][2])
                if deficit > 0.0:
                    pos = np.asarray(
                        to_numpy(entity.get_pos()), dtype=float
                    ).ravel()[:3]
                    entity.set_pos(
                        (pos + np.array([0.0, 0.0, deficit])).astype(float)
                    )
            except Exception:
                pass

        tray = getattr(self, "tray_actor", None)
        if tray is not None:
            _lift(getattr(tray, "entity", tray))
        for actor, _spec in getattr(self, "food_actors", None) or []:
            _lift(getattr(actor, "entity", actor))
        super()._settle_scene(n_steps)

        # Pre-placement guarantee: if a pre-placed food
        # slid off the tray while both bodies settled together, re-seat it
        # on its slot (init-time set_pos, before play_once — allowed) and
        # settle briefly.  Without this the robot picks the fries with its
        # marginal grasp and can drop it mid-carry.
        preplaced = getattr(self, "_preplaced_food_idxs", None) or set()
        if preplaced and self.food_actors:
            slots = self._drop_slots(len(self.food_actors))
            tray_x, tray_y = self._live_tray_center_xy()
            reseated = False
            for i in sorted(preplaced):
                actor, _spec = self.food_actors[i]
                if self._food_on_tray(actor):
                    continue
                dx, dy = slots[i]
                ent = getattr(actor, "entity", actor)
                try:
                    aabb = np.asarray(to_numpy(ent.get_AABB()), dtype=float)
                    pos = np.asarray(
                        to_numpy(ent.get_pos()), dtype=float,
                    ).ravel()[:3]
                    origin_above_bottom = float(pos[2] - aabb[0][2])
                    ent.set_pos(np.array([
                        tray_x + dx, tray_y + dy,
                        self.TABLE_TOP_Z + 2.0 * self.TRAY_RIM_Z
                        + origin_above_bottom + 0.004,
                    ], dtype=float))
                    ent.set_dofs_velocity(np.zeros(6, dtype=float))
                    reseated = True
                except Exception:
                    pass
            if reseated:
                for _ in range(120):
                    self.step_sim()

    def load_actors(self):
        table_top = self.TABLE_TOP_Z

        # ---- Random tray position ----
        tray_x = float(np.random.uniform(*self.TRAY_X_RANGE))
        tray_y = float(np.random.uniform(*self.TRAY_Y_RANGE))
        self._tray_xy = (tray_x, tray_y)

        rgb = [float(np.random.uniform(lo, hi))
               for lo, hi in zip(self.TRAY_COLOR_LO, self.TRAY_COLOR_HI)]
        self._tray_color = tuple(rgb + [1.0])

        tray_pose = Pose(
            [tray_x, tray_y, table_top + self.TRAY_RIM_Z],
            self._Q_TRAY,
        )
        self.tray_actor = self._add_tray_with_color(tray_pose, self._tray_color)

        # ---- Pick which foods to spawn this episode ----
        specs: list[FoodSpec] = self._food_specs_for_episode()

        # ---- Spawn each food, rejection-sampling its xy against earlier ones ----
        self.food_actors = []
        self.food_spawn_xys = []
        self._preplaced_food_idxs = self._preplaced_indices(specs)
        slots = self._drop_slots(len(specs))
        occupied: list[tuple[float, float]] = []
        for i, spec in enumerate(specs):
            if i in self._preplaced_food_idxs:
                # Starts on the tray: spawn over the item's
                # drop slot, slightly above the rim, and let it settle onto
                # the tray during the pre-task settle.
                dx, dy = slots[i]
                x, y = tray_x + dx, tray_y + dy
                z = table_top + 2.0 * self.TRAY_RIM_Z + spec.spawn_z_offset + 0.01
            else:
                x, y = self._sample_spawn_xy(spec.spawn_xy_range, occupied)
                z = table_top + spec.spawn_z_offset
            actor = load_object(
                self.scene,
                Pose([x, y, z], spec.spawn_quat),
                spec.asset_id, model_id=spec.model_id,
                convex=True, is_static=False, friction=spec.friction,
                decompose_object_error_threshold=spec.decompose_error_threshold,
            )
            self.food_actors.append((actor, spec))
            self.food_spawn_xys.append((x, y))
            occupied.append((x, y))

    # ------------------------------------------------------------------
    # Finger PD + force-range boost.
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
    # One-shot pick wrapper.
    # ------------------------------------------------------------------
    def _rotated_top_down_tcp(self, pos, yaw_deg: float) -> Pose:
        """Top-down gripper rotated about the approach (TCP-Z) axis by
        ``yaw_deg``.  Used to grip narrow boxy objects across their short
        dimension (yaw=90°): default top-down closes fingers along world Y;
        +90° yaw closes fingers along world X.
        """
        if abs(yaw_deg) < 1e-6:
            return self._top_down_tcp(pos)
        yaw = np.deg2rad(yaw_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        R = self._R_DOWN @ Rz
        q = t3d.quaternions.mat2quat(R)
        return Pose(np.asarray(pos, dtype=float), q)

    def _pick_and_drop(self, actor, asset_name: str, arm_tag: str,
                       close_value: float, below_center: float,
                       drop_xy: tuple[float, float],
                       tcp_yaw_deg: float = 0.0,
                       transit_steps: int = 0, transit_sim: int = 0,
                       align_to_object_yaw: bool = False) -> bool:
        return self._pick_once(
            actor, asset_name, arm_tag,
            close_value=close_value, below_center=below_center,
            drop_xy=drop_xy, tcp_yaw_deg=tcp_yaw_deg,
            transit_steps=transit_steps, transit_sim=transit_sim,
            align_to_object_yaw=align_to_object_yaw,
        )

    def _get_object_world_center(self, actor, object_name: str, model_id: int = 0):
        """For the fries carton, grasp at the LIVE world-AABB centre instead of
        the model_data centre. The simulator AABB centre is the authoritative
        geometric centre after settling, so the jaws close symmetrically even
        if this asset's metadata is stale. Scoped to the fries only; the
        burger's grasp tuning is unchanged.
        """
        if object_name == _FRIES_ID:
            try:
                aabb = to_numpy(actor.entity.get_AABB())
                return np.asarray(0.5 * (aabb[0] + aabb[1]), dtype=float)
            except Exception:
                pass
        # The pick primitive historically passed model 0 at every stage,
        # although the episode can load another burger model. Use the spec
        # of this actor: model 4's center is about 3 mm higher than model 0's
        # in the upright pose, enough to shift a shallow pinch on the bun.
        # Resolve by actor identity so mixed-model episodes remain correct.
        for food_actor, spec in getattr(self, "food_actors", ()):
            if food_actor is actor and spec.asset_id == object_name:
                model_id = spec.model_id
                break
        return super()._get_object_world_center(actor, object_name, model_id)

    def _object_grip_yaw(self, actor) -> float:
        """tcp_yaw (deg) that closes the top-down jaws across the object's
        measured short horizontal (mesh-X) axis.  tcp_yaw 0 closes along
        world-Y, so the offset is (world-yaw of mesh-X) - 90."""
        q = np.asarray(to_numpy(actor.get_pose().q), dtype=float).ravel()
        R = t3d.quaternions.quat2mat(q)
        x_axis = R @ np.array([1.0, 0.0, 0.0])
        theta = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
        return theta - 90.0

    # Panda hand and open fingers as boxes in the TCP frame (TCP-y closes the
    # jaws, TCP-z is the approach axis, so the hand sits at negative z).
    _HAND_BOXES_TCP = (
        (np.array([0.0, 0.0, -0.078]), np.array([0.032, 0.106, 0.034])),
        (np.array([0.0, 0.0, -0.028]), np.array([0.012, 0.050, 0.028])),
    )
    BURGER_PROP_AWARE_YAW = False
    BURGER_PROP_YAW_MIN_CLEARANCE_M = 0.02

    def _burger_hand_clearance(self, tcp_positions, yaw: float) -> float:
        """Smallest gap between the hand/finger boxes along the descent column
        and the human's props (their checker boxes) and current body capsules.
        Pure geometry: no planner call, so a seed that does not need this is
        left untouched."""
        boxes = getattr(self, "_neutral_object_aabb_boxes", None)
        try:
            props = list(boxes()) if boxes is not None else []
        except Exception:
            props = []
        # The seated human's current body (e.g. the hand on the mouse) too.
        capsules = []
        collider = getattr(self, "avatar_collider", None)
        if collider is not None:
            try:
                capsules = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float), float(r))
                            for _n, a, b, r in collider.current_capsules()]
            except Exception:
                capsules = []
        if not props and not capsules:
            return float("inf")
        best = float("inf")
        for pos in tcp_positions:
            R = Pose(np.zeros(3), self._top_down_tcp(pos, yaw=yaw).q).to_matrix()[:3, :3]
            for center, half in self._HAND_BOXES_TCP:
                g = [np.linspace(-h, h, 5) for h in half]
                local = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3) + center
                world = np.asarray(pos, dtype=float) + local @ R.T
                for _n, c, axes, h in props:
                    rel = (world - np.asarray(c, dtype=float)) @ np.asarray(axes, dtype=float)
                    out = np.maximum(np.abs(rel) - np.asarray(h, dtype=float), 0.0)
                    best = min(best, float(np.min(np.linalg.norm(out, axis=1))))
                for a, b, r in capsules:
                    ab = b - a
                    t = np.clip(((world - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
                    d = np.linalg.norm(world - (a + t[:, None] * ab), axis=1) - r
                    best = min(best, float(np.min(d)))
        return best

    def _burger_prop_clear_yaw(self, grasp_pos, above_pos):
        column = [np.array([grasp_pos[0], grasp_pos[1], z]) for z in
                  np.linspace(float(grasp_pos[2]), float(above_pos[2]), 6)]
        base = self._burger_hand_clearance(column, 0.0)
        if base >= self.BURGER_PROP_YAW_MIN_CLEARANCE_M:
            return None
        alts = [(self._burger_hand_clearance(column, y), y) for y in (np.pi / 2, -np.pi / 2)]
        clearance, yaw = max(alts)
        if clearance < base + 0.02:
            return None
        print(f"[burger] hand prop clearance {base:.3f} m at yaw 0; picking at "
              f"yaw {np.degrees(yaw):+.0f} deg (clearance {clearance:.3f} m)", flush=True)
        return float(yaw)

    def _pick_once(self, actor, asset_name: str, arm_tag: str,
                   close_value: float, below_center: float,
                   drop_xy: tuple[float, float],
                   tcp_yaw_deg: float = 0.0,
                   transit_steps: int = 0, transit_sim: int = 0,
                   align_to_object_yaw: bool = False) -> bool:
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        drop_z = self.TABLE_TOP_Z + self.TRAY_RIM_Z + self.DROP_HOVER_Z_ABOVE_TRAY

        obj_center = self._get_object_world_center(actor, asset_name, 0)
        eff_yaw = self._object_grip_yaw(actor) if align_to_object_yaw else tcp_yaw_deg

        grasp_pos = obj_center.copy()
        grasp_pos[2] -= below_center
        grasp_pos[2] = max(
            grasp_pos[2], self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN,
        )

        self.open_gripper(arm_tag)

        above_pos = np.array([obj_center[0], obj_center[1], transport_z])
        # Neutral only: pick the round patty with the hand turned
        # 90 deg when the default hand footprint would graze a human's prop.
        alt_yaw = None
        if (getattr(self, "BURGER_PROP_AWARE_YAW", False) and asset_name == _HAMBURG_ID
                and abs(float(eff_yaw)) < 1e-9):
            alt_yaw = self._burger_prop_clear_yaw(grasp_pos, above_pos)
        tcp_kw = {} if alt_yaw is None else {"tcp_yaw": float(alt_yaw)}

        def _tcp(pos):
            if alt_yaw is None:
                return self._rotated_top_down_tcp(pos, eff_yaw)
            return self._top_down_tcp(pos, yaw=float(alt_yaw))

        above_link = tcp_to_link_pose(_tcp(above_pos), tcp_offset)
        retry = getattr(self, "_neutral_aware_plan_retry", None)
        move_above = lambda: self._move_seeded(above_link.to_pose7(), arm_tag)  # noqa: E731
        above = (move_above() if retry is None
                 else retry(move_above, arm_tag, label=f"above {asset_name}"))
        if above is None:
            print(f"[burger] {asset_name}: no plan to above-object pose", flush=True)
            return False

        pre_pos = obj_center.copy()
        pre_pos[2] += self.PRE_GRASP_HEIGHT
        self._move_screw(pre_pos, arm_tag, **tcp_kw)

        obj_now = self._get_object_world_center(actor, asset_name, 0)
        grasp_pos[0] = obj_now[0]
        grasp_pos[1] = obj_now[1]
        self._move_screw(grasp_pos, arm_tag, **tcp_kw)
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()

        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        if xy_err > self.GRASP_ALIGNMENT_XY_TOL:
            grasp_link = tcp_to_link_pose(_tcp(grasp_pos), tcp_offset)
            self._move_seeded(grasp_link.to_pose7(), arm_tag)
            for _ in range(self.GRASP_ALIGNMENT_SETTLE_STEPS):
                self.step_sim()

        z_before = float(self._get_object_world_center(actor, asset_name, 0)[2])
        self.set_gripper(close_value, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()

        lift_pos = np.array([grasp_pos[0], grasp_pos[1], transport_z])
        if not self._move_screw(lift_pos, arm_tag, **tcp_kw):
            print(f"[burger] {asset_name}: lift screw failed", flush=True)
            return False
        for _ in range(self.POST_LIFT_SETTLE_STEPS):
            self.step_sim()
        last_pos = lift_pos

        z_after = float(self._get_object_world_center(actor, asset_name, 0)[2])
        dz = z_after - z_before
        if dz < self.LIFT_DZ_MIN:
            print(f"[burger] {asset_name}: lift dz={dz:.4f} < {self.LIFT_DZ_MIN}", flush=True)
            self.open_gripper(arm_tag)
            for _ in range(self.LIFT_FAIL_RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        above_drop = np.array([drop_xy[0], drop_xy[1], transport_z])
        n_steps = transit_steps if transit_steps else self.TRANSIT_CARTESIAN_STEPS
        sim_per_step = transit_sim if transit_sim else self.TRANSIT_SIM_PER_STEP
        self._move_cartesian(
            np.asarray(last_pos, dtype=float),
            above_drop, arm_tag,
            n_steps=n_steps,
            sim_per_step=sim_per_step,
            **tcp_kw,
        )
        for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
            self.step_sim()

        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        obj_xy = self._get_object_world_center(actor, asset_name, 0)[:2]
        post_transit_xy = float(np.linalg.norm(ee_xy - obj_xy))
        if post_transit_xy > self.TRANSIT_SLIP_XY_TOL:
            print(f"[burger] {asset_name}: transit slip xy={post_transit_xy:.4f}", flush=True)
            return False

        drop_pos = np.array([drop_xy[0], drop_xy[1], drop_z])
        self._move_screw(drop_pos, arm_tag, **tcp_kw)
        for _ in range(self.DROP_SETTLE_STEPS):
            self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        n_total = len(self.food_actors)
        slots = self._drop_slots(n_total)
        tray_x, tray_y = self._tray_xy

        for i, (actor, spec) in enumerate(self.food_actors):
            if self._food_on_tray(actor):
                # Pre-placed or already delivered — the
                # robot only manipulates foods that are still loose.
                continue
            dx, dy = slots[i]
            if not self._pick_and_drop(
                actor, spec.asset_id, arm_tag,
                close_value=spec.close_value,
                below_center=spec.below_center,
                drop_xy=(tray_x + dx, tray_y + dy),
                tcp_yaw_deg=spec.tcp_yaw_deg,
                transit_steps=spec.transit_steps,
                transit_sim=spec.transit_sim,
                align_to_object_yaw=spec.align_to_object_yaw,
            ):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success: every spawned food item lies within the tray footprint and
    # is at or above the table top (didn't fall on the floor).
    # ------------------------------------------------------------------
    def _live_tray_center_xy(self) -> tuple[float, float]:
        """Live world xy of the (now dynamic) tray's geometric centre.  Falls
        back to the spawn xy if the AABB read fails."""
        if self.tray_actor is not None:
            try:
                aabb = to_numpy(self.tray_actor.entity.get_AABB())
                c = 0.5 * (aabb[0] + aabb[1])
                return float(c[0]), float(c[1])
            except Exception:
                pass
        return self._tray_xy

    def _food_on_tray(self, actor) -> bool:
        """Live per-food version of the success predicate (same tolerances)."""
        tray_x, tray_y = self._live_tray_center_xy()
        pos = to_numpy(actor.get_pose().p).ravel()[:3]
        return (
            abs(float(pos[0]) - tray_x) <= self.TRAY_HALF_X
            and abs(float(pos[1]) - tray_y) <= self.TRAY_HALF_Y
            and float(pos[2]) >= self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
        )

    def _food_resting_on_tray(self, actor) -> bool:
        """Success predicate for one food: on the tray (``_food_on_tray``),
        resting on its surface, and held by neither the avatar nor the
        gripper."""
        if not self._food_on_tray(actor):
            return False
        pos = to_numpy(actor.get_pose().p).ravel()[:3]
        tray_top = self.TABLE_TOP_Z + 2.0 * self.TRAY_RIM_Z
        if entity_bottom_z(actor, pos[2]) > tray_top + self.SUCCESS_REST_MAX_ABOVE_TRAY_M:
            return False
        if avatar_holds(self, actor) or robot_holds(self, actor, pos):
            return False
        return True

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        return all(self._food_resting_on_tray(actor) for actor, _spec in self.food_actors)
