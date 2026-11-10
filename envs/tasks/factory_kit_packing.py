"""Factory kit packing (Contribution): human and robot pack one shared shipping
kit in parallel at the factory work cell (listed in ``docs/factory_task_inventory.md``).

Scene (``FactorySceneMixin`` bench, top z = 0.765): an open-top cardboard
shipping box mid-bench and three distinct products sampled from the shared
factory product pool.

Robot: top-down pick-and-place of its 2 items into box slots on the box's
robot-facing half, reusing the proven ``PlaceBurgerFries`` pick pipeline
(``_pick_and_drop``: seeded RRT to above-object, screw descents, pinned
close target, post-lift grip verification, slow Cartesian transit with a
slip detector, hover release).  The generic mixin ``pick_and_place``
is not used: its screw-planned transit can fall back to ``plan_pose`` and
hurl the can off the bench, which the burger pipeline's Cartesian transit
avoids.  The avatar capsule pcd is pushed into
the planner obstacles before each pick.

Human: replays the KIT mocap clip ``put_objects_in_bowl``
("putting cans into a plate" — here: putting kit items into the box).
The clip carries two right-hand pick→put event pairs (clip frames
175→222 and 289→379).  Per-episode geometry mirrors the co-presence
avatar machinery (``envs/task_bases/neutral_avatar_table_work.py``) without
depending on it:

  1. Dry-run the clip once at an offscreen pose (``frame_ratio=1``, one
     clip frame per sim step) and record the palm hold anchor
     (palm center + hold_offset * palm normal) at each event frame.
  2. Place the avatar body so the first *put* anchor lands at the box's
     worker-facing half; the *pick* anchors (+ the same shift) become the
     spawn xy of the avatar's 2 items.  The clip's work area extends to
     the actor's RIGHT (measured: rel anchors ~(+0.65, -0.07..-0.45) at
     the factory base rotation), so the body is yawed
     ``AVATAR_ASSIST_YAW_DEG`` (+90°) to point that work area at the
     bench (+y).
  3. Replay the clip slowed by ``AVATAR_MOTION_SLOW`` with a patched
     ``step_sim`` that fires ``attach_object_to_hand`` / ``detach_object``
     when ``motion.at_frame`` crosses each event's scaled frame
     (avatar ``attach_obj`` kinematic holds are sanctioned; the robot
     side stays pure physics).

The robot's picks run concurrently: every ``step_sim`` tick inside the
robot's planned trajectories advances the avatar replay.

Success: all 4 items inside the box interior AABB (geometric check), and
no robot-avatar collision when ``track_avatar_collision`` is on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import transforms3d as t3d

from ..avatar.collision_checker import sample_capsule_planner_points
from ..base_task import BaseTask
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..object_catalog import get_entry, resolve_object_set
from ..scenes.factory import FactorySceneMixin
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ..utils import ASSETS_PATH, Pose, create_primitive, load_object, to_numpy


_ROOT_PATH = Path(__file__).resolve().parents[2]
_NEUTRAL_MOTION_PKL = str(
    _ROOT_PATH / "assets" / "neutral_motions" / "raw_blender_interaction_clips.pkl"
)

# KIT "put objects in mixing bowl" clip — see
# assets/neutral_motions/raw_blender_interaction_clips.json.
_CLIP_NAME = "put_objects_in_bowl"
# The raw clip has large root motion: the actor walks ~0.97 m in during
# frames 0-100 and walks away after ~frame 425 (measured from the pkl's
# y-up ``trans``).  Replaying that walk dragged the worker straight
# through the robot's workspace, so the task replays only the
# stationary working segment — root drift < 2 cm across all 4 events.
_CLIP_TRIM = (105, 425)
_CLIP_NUM_FRAMES = _CLIP_TRIM[1] - _CLIP_TRIM[0]
# (type, clip_frame-in-trimmed-clip, hand_id, item_idx, hold_offset).
# Source frames 175/222/289/379 minus the trim start.  hold_offset is
# meters along the hand-local +Z palm normal (same as the clip JSON).
_CLIP_EVENTS = (
    ("pick", 175 - _CLIP_TRIM[0], 1, 0, 0.05),
    ("put", 222 - _CLIP_TRIM[0], 1, 0, 0.05),
    ("pick", 289 - _CLIP_TRIM[0], 1, 1, 0.05),
    ("put", 379 - _CLIP_TRIM[0], 1, 1, 0.05),
)

_FACTORY_Q_FLAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
_FACTORY_Q_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _factory_kit_object_pool():
    # Per-task robot grasp tuning.  Membership comes exclusively from the
    # shared catalog pool; the worker side is kinematic and needs no tuning.
    # Robot friction 2.0: at/near Genesis's friction cap the
    # contact solve rests these boxes 5-10 mm INSIDE the bench with a visible
    # standing oscillation (probe: milk-box fric 5.0 -> -7.4 mm, fric 2.0 ->
    # -0.4 mm, still).  The width-sized cage closes don't rely on friction.
    params = {
        "038_milk-box": (0.25, 0.010, 2.0, _FACTORY_Q_FLAT),
        # Laid flat this carton exposes a 7.44 cm face —
        # only 2.8 mm/side inside the Franka's 8.0 cm span, less than the
        # task's own alignment tolerance, so fingers ended up INSIDE the box.
        # Upright (like factory_inspect_pack spawns it) the grasp face is
        # 6.62 cm.
        "113_coffee-box": (0.25, -0.005, 2.0, _Q_UPRIGHT),
        "112_tea-box": (0.25, -0.005, 2.0, _FACTORY_Q_IDENT),
        "023_tissue-box": (0.25, -0.005, 2.0, _FACTORY_Q_FLAT),
        "073_rubikscube": (0.43, 0.005, 2.0, _FACTORY_Q_IDENT),
        "086_woodenblock": (0.44, 0.005, 2.0, _FACTORY_Q_IDENT),
    }
    return tuple(
        (entry.object_id, entry.model_id, 2.0, params[entry.key])
        for entry in (
            get_entry(token) for token in resolve_object_set("factory_product_pool")
        )
    )

# upright = mesh-Y → world-Z (these assets are y-up)
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)


def _read_model_data(name: str, mid: int) -> tuple[float, np.ndarray]:
    """(scale, extents) from model_data{mid}.json."""
    p = ASSETS_PATH / "objects" / name / f"model_data{mid}.json"
    with open(p) as f:
        d = json.load(f)
    raw = d.get("scale", 1.0)
    scale = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
    return scale, np.asarray(d["extents"], dtype=np.float64)


class FactoryKitPacking(FactorySceneMixin, TopDownPickPlaceMixin, BaseTask):
    """Human + robot pack three sampled products into one shipping box:
    the worker packs two and the robot packs the remaining one."""

    INSTRUCTION = (
        "pack the remaining item into the shipping box "
        "while the human packs the other two"
    )

    use_avatar = True
    OBJECT_SET = "factory_product_pool"

    # ---- Shipping box (5 static primitives, kraft cardboard) -------------
    # The worker's stance is clip-anchored ~0.38 m behind the put target
    # (measured: pelvis rel-y +0.291, put rel-y +0.670), so the box's y
    # sets her bench distance.  y=-0.23 stands her pelvis ~0.13 m off the
    # bench front edge (a natural working lean) with the box still fully
    # on the bench and both her puts inside the interior.
    BOX_XY = (0.0, -0.23)
    BOX_INNER_HALF = (0.19, 0.17)      # 38 x 34 cm interior
    BOX_WALL_H = 0.07
    BOX_WALL_T = 0.015
    BOX_FLOOR_T = 0.012
    BOX_COLOR = (0.72, 0.56, 0.38)

    # ---- Robot side -------------------------------------------------------
    # (asset, model_id, spawn xy range, friction, close_value, slot dx/dy)
    # Spawn bands live in the Franka's proven 0.40-0.60 m annulus around
    # the base at (+0.60, -0.30) AND on the box side of the base, so the
    # lift -> pre-drop carry is one short straight line that never crosses
    # the inner-workspace boundary.  A milk-box on the far side
    # of the base made the carry skirt the base column at ~0.33 m radius
    # and the waypoint IK wrist-flipped mid-transit, flinging the item
    # (looked like grip slip; it wasn't).  Bands also keep >= 0.12 m from
    # the measured avatar pick anchors and the box footprint.
    # Close targets are sized against the MEASURED collision hulls
    # (milk-box 6.53 cm — honest; can 5.08 cm vs 7.1 declared) for a
    # ~0.9 cm/side squeeze: a deeper 1.5 cm/side command
    # (close 0.45) stores an 80 N pinch that ejects the flat-faced carton
    # mid-carry like a melon seed (perfect grasp, solid lift, then a
    # 1.7 m fling with the worker idle and zero waypoint-IK failures).
    # Slots sit on the box's NORTH half — the south half is where the
    # worker's stand-by hands hover (a south slot produced repeated
    # robot-hand/forearm contacts during the drop descent).
    # Slots hug the box's EAST edge: the worker's stand-by palm hovers
    # 0.07-0.13 m from the box CENTER in every frame of this clip
    # (measured — the arm never retracts), so east-edge drops maximize
    # the carry's clearance from her kinematic arm.
    # Band placement threads three measured constraints:
    #   - x >= ~0.40 escapes the worker's sweep envelope — items inside it
    #     are knocked/tilted by her kinematic arms during her phase (the
    #     can rolled into the box wall, center z sank 3 cm; grasps then
    #     failed on displaced geometry).  The can's seed-0 spot
    #     (0.333, 0.004) is the empirical exception — observed untouched
    #     across repeated runs.
    #   - >= 0.35 m from the base (near-base IK dead zone).
    #   - carry azimuth sweep small: a 65° carry null-space-scooped
    #     the box onto the robot's wrist; the can's proven carries sweep
    #     ~40°.  Milk-box drops at the box NE corner to cut its carry to
    #     ~51° / 0.42 m — the best this bench geometry allows.
    # The can's band is pinned to base-frame azimuth >= -20° (east of
    # the wrist wind-up wall, where its no-unwind carry is proven):
    # azimuth ~-27° is a dead band where neither wrist state crosses.
    # The milk-box band (azimuth -39..-46°) uses the -90° unwind +
    # yaw-follow carry.
    # ---- Object pool + per-episode random 2-avatar / 1-robot split -------
    # The worker packs ANY 2 of the 3 objects (to
    # the motion's two put positions) and the robot packs the remaining
    # one.  Each object carries robot-grasp params used when it happens
    # to be the robot's item.  All three rest as flat cartons (~7 cm
    # tall) with a small horizontal dimension along world-Y, so the
    # top-down gripper closes across a flat face.  The worker is kinematic
    # (attach_obj), so any object works for her with no grasp tuning.
    #
    # Each entry: (asset, model_id, avatar_friction, robot_params)
    #   robot_params = (close_value, below_center, robot_friction,
    #                   spawn_quat | None)  — None uses ROBOT_SPAWN_QUAT.
    _Q_FLAT = _FACTORY_Q_FLAT
    _Q_IDENT = _FACTORY_Q_IDENT
    # A shared pool of compact factory products.  Round jars are
    # excluded: a smooth 7 cm cylinder gives a parallel-jaw gripper only a
    # tangent line-contact and slips out on the lift in every
    # orientation/grip (0/3); a box grips on a flat face and is solid (3/3).
    # below_center = -0.005 grasps ~2 cm higher than 0.015, which sat too
    # low on the carton.
    # The higher grip is near the top edge, so the close is FIRMER (0.20
    # vs 0.25) to hold the box through the lateral carry — the higher
    # grasp at close 0.25 lifted fine but sheared out mid-transit.
    OBJECT_POOL = _factory_kit_object_pool()
    # Bench spawn band for whichever object is the robot's, with ~4 cm
    # jitter.  x~0.30 is the squarest top-down grasp zone; y >= +0.10 is
    # NORTH of the worker's pick anchors (top out at y~+0.07) so her arm
    # never bumps the box and the robot reaches clear of her frozen arm.
    # Widened to hold up to two robot items with ROBOT_ITEM_MIN_SEP spacing.
    ROBOT_SPAWN_BAND = ((0.24, 0.38), (+0.08, +0.16))
    ROBOT_ITEM_MIN_SEP = 0.11
    ROBOT_DROP_SLOT = (+0.10, +0.05)
    # R_z(90°): lays each object flat/on-side, small dim along world-Y.
    ROBOT_SPAWN_QUAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
    # Out-of-sweep parking pose for the robot TCP while the worker packs:
    # her clip sweep reaches x <= 0.30 — the default home hand sits right
    # in it (a deterministic -4.7 cm forearm contact occurred DURING HER
    # PHASE against a hand parked there, identical to 12 digits).
    ROBOT_PARK_TCP = (0.55, 0.08, 1.25)
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 80.0

    # Pick-pipeline knobs consumed by PlaceBurgerFries._pick_once.
    # Drop z = TABLE_TOP_Z + TRAY_RIM_Z + DROP_HOVER_Z_ABOVE_TRAY;
    # TRAY_RIM_Z doubles as the box wall height so the release clears it.
    # Low transport height: the transit is purely lateral at this height
    # and the item is RELEASED here (no descent), so it must be low
    # enough that the flat box drops gently into the box and stays in,
    # yet high enough to clear the 7 cm box walls on the way over.
    TRANSPORT_Z_ABOVE_TABLE = 0.18
    # Lateral carry height: carries at 0.18 from the far
    # bench band start right at an IK branch wall — the guarded transit's
    # first waypoints already branch-jump and the arm never leaves the bench
    # (the stall rate is very sensitive to settled poses shifting a few
    # mm).  0.30 keeps the elbow-up branch well-conditioned across the whole
    # bench->box corridor; the release still happens from TRANSPORT_Z via a
    # safe vertical descent over the box.
    CARRY_Z_ABOVE_TABLE = 0.30
    TRAY_RIM_Z = 0.07
    DROP_HOVER_Z_ABOVE_TRAY = 0.09
    TRANSIT_MAX_JOINT_DELTA = 0.30   # branch-jump reject for the guarded
                                     # lateral transit (rad/waypoint)
    TABLE_FALL_Z_MARGIN = 0.05
    GRASP_TABLE_Z_MARGIN = 0.003
    PRE_GRASP_HEIGHT = 0.15
    GRASP_SETTLE_STEPS = 100
    GRASP_ALIGNMENT_XY_TOL = 0.015
    GRASP_ALIGNMENT_SETTLE_STEPS = 80
    CLOSE_SETTLE_STEPS = 400   # let the grip force fully build before lift
    POST_LIFT_SETTLE_STEPS = 60
    LIFT_DZ_MIN = 0.05
    LIFT_FAIL_RELEASE_SETTLE_STEPS = 50
    TRANSIT_CARTESIAN_STEPS = 60
    TRANSIT_SIM_PER_STEP = 25
    LIFT_CARTESIAN_STEPS = 24    # branch-guarded straight-up lift
    LIFT_SIM_PER_STEP = 20
    POST_TRANSIT_SETTLE_STEPS = 50
    TRANSIT_SLIP_XY_TOL = 0.10
    DROP_SETTLE_STEPS = 40
    RELEASE_SETTLE_STEPS = 200

    # ---- Avatar side ------------------------------------------------------
    ASSIST_CLIP = _CLIP_NAME
    AVATAR_MOTION_SLOW = 10.0          # replay frames per clip frame
    # Yaw applied on top of the factory base rotation so the clip's
    # right-side work area points at the bench (+y).
    AVATAR_ASSIST_YAW_DEG = 90.0
    AVATAR_MEASURE_POS = np.array([6.0, 6.0, -0.18])
    AVATAR_DROP_OFFSET_XY = (-0.08, 0.0)   # put target rel. to box center
    # Bench-edge clearance for the worker.  Her stance pelvis sits forward
    # of the reset point (clip-internal root offset), so the dry-run also
    # measures the pelvis at the first put frame and the body is pushed
    # south until pelvis_y <= bench_front - clearance (otherwise she
    # stands half-inside the bench).  The shift is capped so her puts
    # (which move with her) stay inside the deepened box.
    BENCH_FRONT_Y = -0.475
    AVATAR_BENCH_CLEARANCE = 0.13
    AVATAR_BODY_SHIFT_MAX = 0.03
    # Final hard guard on the reset point.
    AVATAR_BODY_Y_MAX = -0.575
    AVATAR_BODY_X_BOUNDS = (-0.75, 0.15)
    AVATAR_PRE_MOTION_SETTLE_STEPS = 60
    AVATAR_FINISH_WAIT_STEPS = 20000
    AVATAR_ITEM_BOX_CLEARANCE = 0.06   # push spawn xy out of the box footprint
    # Workspace de-confliction.  The robot starts after the avatar's LAST
    # put: running the robot's carry while her kinematic (unstoppable)
    # arm transports her own item across the same corridor lets the arm
    # knock the carried milk-box out of the gripper and onto her torso.
    # She packs the first half of the episode, the robot the second; the
    # hand-clearance gates below remain as safety nets for her retract
    # tail and stand-by pose.
    ROBOT_START_AFTER_EVENT = 3        # index into _CLIP_EVENTS (put #2)
    ROBOT_START_WAIT_STEPS = 8000
    AVATAR_HAND_CLEARANCE = 0.30
    AVATAR_HAND_CLEAR_TIMEOUT = 5000
    # Robot drop point: ROBOT_DROP_RADIUS into the box from centre, on
    # the side away from the worker's static end-frame hand; kept this
    # far from the interior wall so the laid-flat box footprint fits.
    ROBOT_DROP_RADIUS = 0.11
    ROBOT_DROP_EDGE_MARGIN = 0.08

    INITIAL_SETTLE_STEPS = 80
    RELEVEL_SETTLE_STEPS = 30   # settle after re-levelling the robot box
    FINAL_SETTLE_STEPS = 150
    SUCCESS_Z_MAX_ABOVE_TABLE = 0.30

    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.25

    # Mount the Franka yawed +150° (same fix as factory_line_feeding,
    # which traced its lateral IK drift / branch flips to this).  With
    # the mixin's identity quat the shipping box sits at azimuth ~179°
    # from the base — outside the joint-1 limit (±166°) — and the carry
    # toward it crossed the near-limit wall, where the per-waypoint IK
    # branch-jumped and whipped the held item off the gripper (perfect
    # 4 mm grasps, solid lifts, then a mid-transit "slip" fling — also
    # with the worker idle, so not her doing).
    # Yawed +150°, the box and both spawn bands sit at azimuth -46°..+29°.
    FACTORY_ROBOT_KWARGS = {
        "pos": [0.60, -0.30, 0.765],
        "quat": [0.2588190, 0.0, 0.0, 0.9659258],
    }

    # Reuse the proven, basic PlaceBurgerFries top-down pick pipeline
    # wholesale (plain unbound-method reuses): above → pre → grasp →
    # close → lift → transit → drop.
    _boost_finger_pd = PlaceBurgerFries._boost_finger_pd
    _rotated_top_down_tcp = PlaceBurgerFries._rotated_top_down_tcp
    _pick_and_drop = PlaceBurgerFries._pick_and_drop
    # `_pick_once` is defined below — a copy of PlaceBurgerFries._pick_once
    # whose ONLY change is the final drop descent: straight-line
    # `_move_cartesian` instead of `_move_screw`, which on this bench
    # geometry failed plan_screw and replanned to a different IK branch,
    # swinging the arm in the "last mile".

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        avatar_cfg = dict(cfg.get("avatar") or {})
        avatar_cfg.setdefault("generated_motion_data", _NEUTRAL_MOTION_PKL)
        avatar_cfg.setdefault("frame_ratio", float(self.AVATAR_MOTION_SLOW))
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)
        self.robot_item_actors = []    # [(actor, asset, mid, radius, half_h, friction, close, slot)]
        self.avatar_item_actors = []   # [(actor, asset, mid, half_h)]

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0):
        if hasattr(self, "_kit_original_step_sim"):
            self.step_sim = self._kit_original_step_sim
            del self._kit_original_step_sim
        return super().reset(seed=seed)

    def _build_shipping_box(self):
        bx, by = self.BOX_XY
        ihx, ihy = self.BOX_INNER_HALF
        t, wh, ft = self.BOX_WALL_T, self.BOX_WALL_H, self.BOX_FLOOR_T
        z0 = self.TABLE_TOP_Z
        ohx, ohy = ihx + t, ihy + t
        self.box_parts = [create_primitive(
            self.scene, "box", Pose(p=[bx, by, z0 + ft / 2]),
            size={"half_size": (ohx, ohy, ft / 2)},
            color=self.BOX_COLOR, is_static=True,
        )]
        wz = z0 + wh / 2
        for sx in (-1, +1):
            self.box_parts.append(create_primitive(
                self.scene, "box", Pose(p=[bx + sx * (ihx + t / 2), by, wz]),
                size={"half_size": (t / 2, ohy, wh / 2)},
                color=self.BOX_COLOR, is_static=True,
            ))
        for sy in (-1, +1):
            self.box_parts.append(create_primitive(
                self.scene, "box", Pose(p=[bx, by + sy * (ihy + t / 2), wz]),
                size={"half_size": (ohx, t / 2, wh / 2)},
                color=self.BOX_COLOR, is_static=True,
            ))

    def load_actors(self):
        self.build_factory()
        self._build_shipping_box()

        # ---- Per-episode random 2-avatar / 1-robot split ----
        # Seeded by BaseTask.reset (np.random.seed(seed)).  Pick which of
        # the 3 pool objects the robot packs; the worker packs the other
        # two.  Logged so the assignment is reproducible from the seed.
        # config["object_name"]=<asset substring> forces the robot's object.
        pool = list(self.OBJECT_POOL)
        forced = str(self.config.get("object_name") or "").strip()
        # 2 avatar items + 1-2 robot items (a lone,
        # late-starting robot item reads as "the robot does nothing").
        n_robot = int(np.clip(int(self.config.get("kit_robot_items", 2)), 1, 2))
        n_objects = 2 + n_robot
        if forced:
            forced_idx = next(
                (i for i, o in enumerate(pool) if forced in o[0]), None,
            )
            if forced_idx is None:
                raise ValueError(
                    f"object_name={forced!r} not in factory_product_pool"
                )
            other = [i for i in np.random.permutation(len(pool)) if i != forced_idx]
            episode_pool = [pool[forced_idx]] + [
                pool[int(i)] for i in other[: n_objects - 1]
            ]
        else:
            chosen = np.random.permutation(len(pool))[:n_objects]
            episode_pool = [pool[int(i)] for i in chosen]
        robot_objs = episode_pool[:n_robot]
        avatar_objs = episode_pool[n_robot:]
        print(f"[kit] episode split: robot={[o[0] for o in robot_objs]} "
              f"avatar={[o[0] for o in avatar_objs]}", flush=True)

        # ---- Robot items: random non-overlapping spots in the bench band ----
        self.robot_item_actors = []
        self._robot_spawn_pose_by_actor = {}
        (xlo, xhi), (ylo, yhi) = self.ROBOT_SPAWN_BAND
        placed_xy: list[tuple[float, float]] = []
        for item_i, robot_obj in enumerate(robot_objs):
            asset, mid, _avf, rp = robot_obj
            close, below, rfric = rp[0], rp[1], rp[2]
            obj_quat = rp[3] if len(rp) > 3 and rp[3] is not None else None
            spawn_quat = (np.asarray(obj_quat, dtype=float) if obj_quat is not None
                          else getattr(self, "ROBOT_SPAWN_QUAT", _Q_UPRIGHT))
            scale, extents = _read_model_data(asset, mid)
            R = t3d.quaternions.quat2mat(np.asarray(spawn_quat, dtype=float))
            world_ext = np.abs(R) @ (np.asarray(extents, dtype=float) * scale)
            half_h = float(world_ext[2]) / 2.0
            radius = float(min(world_ext[0], world_ext[1])) / 2.0
            for _ in range(200):
                x = float(np.random.uniform(xlo, xhi))
                y = float(np.random.uniform(ylo, yhi))
                if all(
                    np.hypot(x - px, y - py) >= self.ROBOT_ITEM_MIN_SEP
                    for px, py in placed_xy
                ):
                    break
            placed_xy.append((x, y))
            actor = load_object(
                self.scene,
                Pose([x, y, self.TABLE_TOP_Z + half_h + 0.001], spawn_quat),
                asset, model_id=mid,
                convex=True, is_static=False, friction=rfric,
            )
            # Remember the spawn pose so each box can be re-levelled to a
            # flat resting pose just before its grasp (the worker's arm
            # drifts/tilts it during her phase; a tilted box slips the
            # top-down grip, a pristine one grips solid — proven in the
            # robot-first parallel runs, dz=0.18 0-slip).
            self._robot_spawn_pose_by_actor[id(actor)] = (
                float(x), float(y), float(half_h),
                np.asarray(spawn_quat, dtype=float).copy(),
            )
            # tuple: (actor, asset, mid, radius, half_h, fric, close, slot,
            #         below_center)
            self.robot_item_actors.append(
                (actor, asset, mid, radius, half_h, rfric, close,
                 self.ROBOT_DROP_SLOT, below)
            )

        # ---- Avatar items spawn offscreen; placed at the measured clip
        # pick anchors at the start of play_once ----
        self.avatar_item_actors = []
        for i, (asset, mid, fric, _rp) in enumerate(avatar_objs):
            scale, extents = _read_model_data(asset, mid)
            half_h = float(extents[1]) * scale / 2.0
            actor = load_object(
                self.scene,
                Pose([7.0 + 0.7 * i, 7.5, half_h + 0.01], _Q_UPRIGHT),
                asset, model_id=mid,
                convex=True, is_static=False, friction=fric,
            )
            self.avatar_item_actors.append((actor, asset, mid, half_h))

    @staticmethod
    def _entity(actor):
        return getattr(actor, "entity", actor)

    # ------------------------------------------------------------------
    # Avatar: clip measurement + replay with event-driven attach/detach
    # ------------------------------------------------------------------
    def _ensure_runtime_clip(self) -> str:
        """Install the trimmed (stationary working segment) subclip into the
        avatar's motion data under a task-local name; idempotent."""
        dst = f"{_CLIP_NAME}_kit_trim_{_CLIP_TRIM[0]}_{_CLIP_TRIM[1]}"
        if dst not in self.avatar.motion_data:
            md = self.avatar.motion_data[_CLIP_NAME]
            lo, hi = _CLIP_TRIM
            self.avatar.motion_data[dst] = {k: v[lo:hi].copy() for k, v in md.items()}
        self._runtime_clip = dst
        return dst

    def _assist_rot(self) -> np.ndarray:
        yaw = np.deg2rad(float(self.AVATAR_ASSIST_YAW_DEG))
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ np.asarray(self.avatar_init_rot, dtype=np.float64)

    def _hold_anchor(self, hand_id: int, hold_offset: float) -> np.ndarray:
        palm = np.asarray(
            self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64,
        )
        _, hand_frame = self.avatar.robot._get_hand_frame(int(hand_id))
        normal = np.asarray(hand_frame[:, 2], dtype=np.float64)
        return palm + float(hold_offset) * normal

    def _measure_clip_anchors(self) -> dict[int, np.ndarray]:
        """Dry-run the clip offscreen at frame_ratio=1; return hold anchors
        (relative to the measurement body position) keyed by clip frame."""
        self._ensure_runtime_clip()
        base_pos = np.asarray(self.AVATAR_MEASURE_POS, dtype=np.float64)
        base_rot = self._assist_rot()
        self.avatar.reset(global_trans=base_pos, global_rot=base_rot)
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(self._runtime_clip)
        needed = {int(f): (h, off) for (_t, f, h, _i, off) in _CLIP_EVENTS}
        anchors: dict[int, np.ndarray] = {}
        step = 0
        max_step = max(needed) + 2
        first_put_frame = next(
            int(f) for (t, f, _h, _i, _o) in _CLIP_EVENTS if t == "put"
        )
        self._pelvis_rel = None
        # Candidate stand-by frames: clip start ("about to reach" — right
        # arm extended over the box, a robot-contact source) vs the
        # end-of-retract tail (arm pulled back).  The palm is measured at
        # each and play_once parks her on whichever is farthest from the
        # box.
        standby_candidates = (0, _CLIP_NUM_FRAMES - 20, _CLIP_NUM_FRAMES - 2)
        self._standby_palm_rel = {}
        max_step = max(max_step, _CLIP_NUM_FRAMES - 1)
        while not self.avatar.spare() and step <= max_step:
            self.scene.step()
            self.avatar.step()
            if step in needed:
                hand_id, off = needed[step]
                anchors[step] = self._hold_anchor(hand_id, off) - base_pos
            if step == first_put_frame:
                try:
                    pelvis = np.asarray(
                        self.avatar.robot.skin.get_global_translation("Hips")[0]
                    ).ravel()[:3]
                    self._pelvis_rel = pelvis - base_pos
                    print(f"[kit] stance pelvis rel: "
                          f"{self._pelvis_rel.round(3).tolist()}", flush=True)
                except Exception as exc:
                    print(f"[kit] WARNING pelvis measurement failed: {exc}",
                          flush=True)
            if step in standby_candidates:
                palm = np.asarray(
                    self.avatar.robot.get_palm_center(1), dtype=np.float64,
                )
                self._standby_palm_rel[step] = palm - base_pos
            step += 1
        self.avatar.frame_ratio = old_ratio
        missing = sorted(set(needed) - set(anchors))
        if missing:
            raise RuntimeError(
                f"[kit] clip dry-run missed event frames {missing} "
                f"(stopped at step {step})"
            )
        for f in sorted(anchors):
            print(f"[kit] clip anchor rel frame {f}: {anchors[f].round(3).tolist()}",
                  flush=True)
        return anchors

    def _push_xy_outside_box(self, xy: np.ndarray) -> np.ndarray:
        """If xy falls inside the box footprint (+clearance), push it out
        along the smaller-penetration axis so items never spawn in the box."""
        bx, by = self.BOX_XY
        hx = self.BOX_INNER_HALF[0] + self.BOX_WALL_T + self.AVATAR_ITEM_BOX_CLEARANCE
        hy = self.BOX_INNER_HALF[1] + self.BOX_WALL_T + self.AVATAR_ITEM_BOX_CLEARANCE
        dx, dy = float(xy[0]) - bx, float(xy[1]) - by
        if abs(dx) >= hx or abs(dy) >= hy:
            return xy
        out = np.asarray(xy, dtype=np.float64).copy()
        push_x = hx - abs(dx)
        push_y = hy - abs(dy)
        if push_x <= push_y:
            out[0] = bx + np.sign(dx or 1.0) * hx
        else:
            out[1] = by + np.sign(dy or 1.0) * hy
        print(f"[kit] avatar item spawn pushed out of box: {xy.round(3).tolist()} "
              f"-> {out.round(3).tolist()}", flush=True)
        return out

    def _apply_clip_frame(self, frame_idx: int, motion_name: str = None) -> None:
        """Force the avatar skin to an exact replay frame (stand naturally
        after the clip ends instead of freezing on the last frame)."""
        name = motion_name or self._runtime_clip
        motion = self.avatar.motion_modules.get(name)
        if motion is None or not getattr(motion, "data", None):
            return
        idx = int(np.clip(int(frame_idx), 0, len(motion.data) - 1))
        motion.at_frame = idx
        self.avatar.robot.pose = motion.data[idx]
        self.avatar.robot.node_trans = motion.node_data[idx]
        self.avatar.robot.global_mat = motion.global_mat
        self.avatar.robot.global_mat_inv = motion.global_mat_inv
        self.avatar.robot.update()

    def _hold_clip_frame(self, frame_idx: int) -> None:
        """Pose the worker on an exact frame of HER OWN working clip and
        freeze there (NO_ACTION), so the rest pose has the same body
        rotation as the adjacent clip motion — no orientation jump.

        A separate ``idle`` clip for the end stand-by would snap her
        orientation (the idle clip's root faces a different way than the
        working clip).  Holding her own clip's begin/end frame keeps
        begin-rest = begin-frame and end-rest = end-frame, exactly aligned."""
        from ..avatar.utils import AvatarState, ActionStatus
        if self._runtime_clip not in self.avatar.motion_modules:
            self.avatar.play_animation(self._runtime_clip)
        self._apply_clip_frame(frame_idx)
        self.avatar.robot.action_state = AvatarState.NO_ACTION
        self.avatar.robot.action_status = ActionStatus.INIT

    def _start_avatar_packing(self) -> None:
        """Play the clip (slowed) and patch step_sim to fire attach/detach
        events as the replay crosses each event's scaled frame."""
        slow = float(self.AVATAR_MOTION_SLOW)
        self.avatar.frame_ratio = slow
        self.avatar.play_animation(self._runtime_clip)

        scaled_len = max(1, int(_CLIP_NUM_FRAMES * slow))
        scaled_to_clip = np.round(
            np.linspace(0, _CLIP_NUM_FRAMES - 1, scaled_len)
        ).astype(int)
        events = []
        for (etype, clip_frame, hand_id, item_idx, _off) in _CLIP_EVENTS:
            matches = np.flatnonzero(scaled_to_clip >= int(clip_frame))
            events.append({
                "type": etype,
                "hand_id": int(hand_id),
                "item": int(item_idx),
                "scaled_frame": int(matches[0]) if matches.size else scaled_len - 1,
            })

        state = {"handled": set(), "final_pose_restored": False}
        self._kit_event_state = state
        original_step_sim = self.step_sim
        self._kit_original_step_sim = original_step_sim

        def patched_step_sim():
            original_step_sim()
            motion = self.avatar.motion_modules.get(self._runtime_clip)
            at_frame = int(getattr(motion, "at_frame", -1)) if motion is not None else -1
            for idx, event in enumerate(events):
                if idx in state["handled"] or at_frame < event["scaled_frame"]:
                    continue
                ent = self._entity(self.avatar_item_actors[event["item"]][0])
                if event["type"] == "pick":
                    self.avatar.robot.attach_object_to_hand(event["hand_id"], ent)
                    self.avatar.robot.update()
                else:
                    self.avatar.robot.detach_object(event["hand_id"])
                    # Guided put (avoids the "avatar put
                    # bounce"): the clip's put anchor releases the item
                    # 15-25 cm above the box floor, sometimes right beside
                    # the low wall — from there it tips/bounces OUT.  The
                    # hold is kinematic anyway (attach_obj abstraction), so
                    # clamp the release pose into the box interior just
                    # above the wall before the free fall.
                    try:
                        bx, by = self.BOX_XY
                        ihx, ihy = self.BOX_INNER_HALF
                        pos = np.asarray(
                            to_numpy(ent.get_pos()), dtype=float,
                        ).ravel()[:3].copy()
                        aabb = np.asarray(
                            to_numpy(ent.get_AABB()), dtype=float,
                        )
                        half = 0.5 * (aabb[1] - aabb[0])
                        m = 0.015
                        pos[0] = float(np.clip(
                            pos[0],
                            bx - ihx + half[0] + m, bx + ihx - half[0] - m,
                        ))
                        pos[1] = float(np.clip(
                            pos[1],
                            by - ihy + half[1] + m, by + ihy - half[1] - m,
                        ))
                        pos[2] = min(
                            pos[2],
                            self.TABLE_TOP_Z + self.BOX_WALL_H
                            + float(half[2]) + 0.01,
                        )
                        ent.set_pos(pos)
                        ent.set_dofs_velocity(np.zeros(6, dtype=float))
                    except Exception:
                        pass
                print(f"[kit] avatar {event['type']} item{event['item']} "
                      f"at scaled frame {at_frame}", flush=True)
                state["handled"].add(idx)
            if (
                len(state["handled"]) == len(events)
                and self.avatar.spare()
                and not state["final_pose_restored"]
            ):
                # Hold HER clip's last frame (end-rest = end-frame, no
                # rotation jump).  The robot drops away from her static
                # hand (see _robot_drop_slot), so no separate arms-down
                # pose is needed.
                motion = self.avatar.motion_modules.get(self._runtime_clip)
                last = (len(motion.data) - 1) if motion is not None else 0
                self._hold_clip_frame(last)
                state["final_pose_restored"] = True

        self.step_sim = patched_step_sim

    def _restore_step_sim(self) -> None:
        if hasattr(self, "_kit_original_step_sim"):
            self.step_sim = self._kit_original_step_sim
            del self._kit_original_step_sim

    # ------------------------------------------------------------------
    # Robot: avatar capsules as planner obstacles
    # ------------------------------------------------------------------
    def _avatar_obstacle_points(self) -> np.ndarray | None:
        if self.avatar_collider is None:
            return None
        return sample_capsule_planner_points(
            self.avatar_collider.current_capsules(),
            inflate_factor=self._AVATAR_INFLATE_FACTOR,
        )

    def _refresh_planner_obstacles(self, arm_tag: str) -> None:
        pts = self._avatar_obstacle_points()
        if pts is None:
            return
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception:
            pass

    def _wait_avatar_event(self, event_idx: int, timeout: int) -> None:
        """Step the sim until clip event ``event_idx`` has fired."""
        if self.avatar is None:
            return
        waited = 0
        while waited < int(timeout):
            state = getattr(self, "_kit_event_state", None)
            if state is None or int(event_idx) in state["handled"]:
                return
            self.step_sim()
            waited += 1
        print(f"[kit] WARNING wait for clip event {event_idx} timed out", flush=True)

    def _wait_avatar_hand_clear(self, target_xy, clearance: float, timeout: int) -> None:
        """Step the sim until the avatar's right palm is at least
        ``clearance`` (xy) away from ``target_xy``."""
        if self.avatar is None:
            return
        target = np.asarray(target_xy, dtype=np.float64)[:2]
        waited = 0
        while waited < int(timeout):
            try:
                palm = np.asarray(
                    self.avatar.robot.get_palm_center(1), dtype=np.float64,
                )[:2]
            except Exception:
                return
            if float(np.linalg.norm(palm - target)) >= float(clearance):
                if waited:
                    print(f"[kit] waited {waited} steps for avatar hand to clear "
                          f"{target.round(3).tolist()}", flush=True)
                return
            self.step_sim()
            waited += 1
        print(f"[kit] WARNING avatar-hand-clear wait timed out near "
              f"{target.round(3).tolist()}", flush=True)

    # Commanded finger bite per side when sizing the close from the measured
    # face width: ``close = (width - 2*bite) / 0.08``.
    # A fixed close=0.25 commands a 2 cm jaw gap on 6-7 cm boxes —
    # 2+ cm of interpenetration per side held at the 80 N force cap, which
    # renders as fingers sunk INTO the object.  6 mm of commanded bite gives
    # ~54 N of grip (kp 9000) with barely visible penetration.
    GRIP_BITE_M = 0.006
    # Per-item bite overrides (empty: coffee-box dz=0 slips come from a
    # grasp-HEIGHT frame mismatch — model_data vs live AABB centre — not
    # insufficient bite; `_pick_once` grasps mid-body from the live AABB).
    GRIP_BITE_OVERRIDES: dict = {}

    def _relevel_robot_item(self, actor) -> None:
        """Reset the robot's object to its flat spawn pose just before the
        grasp.  The worker's kinematic arm drifts/tilts it during her
        phase, and a tilted box slips the top-down grip; a flat, settled
        box grips solidly (robot-first parallel runs: dz=0.18, 0 slips).
        The object is loose (not held by the robot), so this is a
        legitimate scene reset — the same set_pos mechanism the task uses
        to stage the worker's items."""
        poses = getattr(self, "_robot_spawn_pose_by_actor", None) or {}
        pose = poses.get(id(actor))
        if pose is None:
            return
        x, y, half_h, quat = pose
        ent = self._entity(actor)
        ent.set_pos(np.array([x, y, self.TABLE_TOP_Z + half_h + 0.002], dtype=float))
        ent.set_quat(np.asarray(quat, dtype=float))
        try:
            ent.set_dofs_velocity(np.zeros(6, dtype=float))
        except Exception:
            pass
        for _ in range(self.RELEVEL_SETTLE_STEPS):
            self.step_sim()
        # Seat the item on the tabletop.  ``TABLE_TOP_Z + half_h`` places
        # the mesh ORIGIN — for bottom-origin assets (113_coffee-box) that
        # leaves the box hovering ~half_h in the air, the 30-step settle
        # doesn't land it, and the settled-grasp helper then measures a
        # FLOATING AABB whose mid-body grasp is really the box top (the
        # observed dz=0 coffee-box slips).
        try:
            aabb = np.asarray(to_numpy(ent.get_AABB()), dtype=float)
            drop = float(aabb[0][2]) - (self.TABLE_TOP_Z + 0.001)
            if abs(drop) > 0.003:
                pos = np.asarray(
                    to_numpy(ent.get_pos()), dtype=float,
                ).ravel()[:3]
                ent.set_pos(pos - np.array([0.0, 0.0, drop], dtype=float))
                ent.set_dofs_velocity(np.zeros(6, dtype=float))
                for _ in range(self.RELEVEL_SETTLE_STEPS):
                    self.step_sim()
        except Exception:
            pass

    def _robot_pack_items(self, arm_tag: str) -> bool:
        """Pack each robot item with the BASIC top-down pick pipeline
        (``PlaceBurgerFries._pick_and_drop``): above-object → pre-grasp →
        grasp → close → lift → transit → drop.  No custom planner-IK /
        joint-lerp / yaw-sweep machinery: with the +150° mount (good joint
        range) and the laid-flat milk-box it is unnecessary and makes the
        arm reconfigure repeatedly before the grasp."""
        bx, by = self.BOX_XY
        ok = True
        for item_i, (
            actor, asset, _mid, _radius, _half_h, _fric, _close, (dx, dy),
            _below,
        ) in enumerate(self.robot_item_actors):
            # Safety net: never begin a pick while the worker's palm is
            # still near this item (her retract tail can overlap the first
            # pick now that the robot starts on her final put event).
            item_xy = self._get_object_world_center(actor, asset, _mid)[:2]
            self._wait_avatar_hand_clear(
                item_xy, self.AVATAR_HAND_CLEARANCE,
                self.AVATAR_HAND_CLEAR_TIMEOUT,
            )
            self._refresh_planner_obstacles(arm_tag)
            drops = getattr(self, "_robot_drop_xys", None)
            if drops:
                drop_xy = drops[min(item_i, len(drops) - 1)]
            else:
                drop_xy = getattr(self, "_robot_drop_xy", (bx + dx, by + dy))
            # Re-level ONCE (the single visible scene snap per item), THEN
            # measure the grasp from the restored pose via the shared
            # settled-grasp helper: narrowest truly
            # horizontal face, hard jaw-span gate, live-AABB grasp height.
            self._relevel_robot_item(actor)
            grasp = self.settled_topdown_grasp(
                self._entity(actor), asset, _mid,
                span_margin=0.005,
                table_z=self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN,
            )
            if not grasp["feasible"]:
                print(f"[kit] pack {asset} SKIP: {grasp['reason']}",
                      flush=True)
                ok = False
                continue
            yaw, width = grasp["yaw_deg"], grasp["width"]
            bite = float(self.GRIP_BITE_OVERRIDES.get(asset, self.GRIP_BITE_M))
            close = float(np.clip(
                (width - 2.0 * bite) / 0.08, 0.10, 0.95,
            ))
            below_live = float(grasp["center"][2] - grasp["grasp_z"])
            print(f"[kit] pack {asset}: yaw={yaw:.0f} deg "
                  f"width={round(width, 3)} close={close:.2f} "
                  f"drop={np.round(drop_xy, 3).tolist()}",
                  flush=True)
            try:
                # Call the LOCAL _pick_once directly: the borrowed
                # PlaceBurgerFries._pick_and_drop forwards transit kwargs the
                # local pick does not accept, so every robot pack would raise
                # TypeError and silently "fail" (leaving the robot idle).
                this_ok = self._pick_once(
                    actor, asset, arm_tag,
                    close_value=close,
                    below_center=below_live,
                    drop_xy=drop_xy,
                    tcp_yaw_deg=yaw,
                )
            except Exception as exc:
                print(f"[kit] robot pack of {asset} raised: {exc}", flush=True)
                this_ok = False
            if not this_ok:
                print(f"[kit] robot pack of {asset} failed", flush=True)
                ok = False
        return ok

    def _guarded_transit(self, start_pos, end_pos, arm_tag, n_steps,
                         sim_per_step, tcp_yaw_deg: float = 0.0):
        """Straight-line Cartesian transit with a branch-jump GUARD: a
        waypoint whose top-down IK solution moves any arm joint more than
        ``TRANSIT_MAX_JOINT_DELTA`` from the previous one is a branch flip
        — skip it (hold) instead of executing the wild swing that flings
        the held item.  Otherwise identical to the mixin ``_move_cartesian``
        (finger grip preserved, arm reseeded from measured qpos at entry)."""
        from ..robot.franka_robot import to_numpy as _tn, _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _tn(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured = np.asarray(_tn(arm.entity.get_qpos()), dtype=float).ravel()
        idxs = [_get_dof_idx(j) for j in arm.arm_joints
                if j is not None and _get_dof_idx(j) is not None]
        seed = base_target.copy()
        for k in idxs:
            seed[k] = float(measured[k])
        arm._cached_target = seed
        prev_q = np.array([seed[k] for k in idxs], dtype=float)
        start_pos = np.asarray(start_pos, float)
        end_pos = np.asarray(end_pos, float)
        skipped = 0
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pos = start_pos * (1 - t) + end_pos * t
            arm_qpos = self._solve_ik(
                self._rotated_top_down_tcp(pos, tcp_yaw_deg), arm_tag,
            )
            if arm_qpos is None:
                continue
            new_q = np.asarray(arm_qpos, dtype=float).ravel()[:len(idxs)]
            if float(np.max(np.abs(new_q - prev_q))) > self.TRANSIT_MAX_JOINT_DELTA:
                skipped += 1
                continue
            qpos_full = base_target.copy()
            for j, k in enumerate(idxs):
                qpos_full[k] = float(new_q[j])
            arm.entity.control_dofs_position(qpos_full)
            arm._cached_target = qpos_full.copy()
            prev_q = new_q
            for _ in range(sim_per_step):
                self.step_sim()
        if skipped:
            print(f"[kit] transit guard skipped {skipped}/{n_steps} "
                  f"branch-jump waypoints", flush=True)

    def _gated_qpos_hop(self, target_pos, arm_tag: str) -> bool:
        """One seeded joint-RRT hop to a yaw-0 top-down pose while HOLDING.

        Used only when the guarded Cartesian carry stalls on a branch wall.
        Refuses (returns False) rather than risk a fling: needs seeded IK
        within the branch gate AND a successful ``plan_qpos`` — there is NO
        ``plan_pose`` fallback here by design."""
        from ..robot.franka_robot import to_numpy as _tn
        arm = self.robot.get_arm(arm_tag)
        link = tcp_to_link_pose(self._top_down_tcp(target_pos), arm.tcp_offset)
        pose7 = link.to_pose7()
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return False
        seed = np.asarray(
            _tn(arm.entity.get_qpos()), dtype=np.float64,
        ).ravel()
        try:
            sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=seed,
            )
        except Exception:
            return False
        if sol is None:
            return False
        goal = _tn(sol).ravel()[:arm.n_arm]
        if len(goal) >= 7:
            j7d = float(goal[6]) - float(seed[6])
            if j7d > np.pi:
                goal[6] -= 2 * np.pi
            elif j7d < -np.pi:
                goal[6] += 2 * np.pi
        if float(np.linalg.norm(goal - seed[:arm.n_arm])) > 3.5:
            print("[kit] qpos-hop refused: IK branch flip", flush=True)
            return False
        planner = arm.planner
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(np.asarray(goal, dtype=np.float64))],
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
            print("[kit] qpos-hop refused: no joint plan", flush=True)
            return False
        from ..planning.base import PlanResult
        pos = result["position"]
        vel = result.get("velocity", np.zeros_like(pos))
        print("[kit] qpos-hop executing gated joint plan", flush=True)
        self.execute_plan(PlanResult(True, pos, vel), arm_tag)
        self._reseed_cached_arm_target(arm)
        return True

    def _pick_once(self, actor, asset_name: str, arm_tag: str,
                   close_value: float, below_center: float,
                   drop_xy: tuple[float, float],
                   tcp_yaw_deg: float = 0.0) -> bool:
        """Faithful copy of ``PlaceBurgerFries._pick_once`` with TWO
        changes for this bench geometry: (1) the lateral transit uses a
        branch-jump-guarded Cartesian move (``_guarded_transit``) so a
        mid-carry IK flip can't fling the held box; (2) the item is
        released directly at the hover over the box — NO descent leg,
        which on this geometry swung the arm in the last mile."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        # NOTE: the item was already re-levelled ONCE by `_robot_pack_items`
        # (immediately before the grasp measurement).  A second snap here
        # renders as the box visibly teleporting twice per pick.
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        drop_z = self.TABLE_TOP_Z + self.TRAY_RIM_Z + self.DROP_HOVER_Z_ABOVE_TRAY

        # Live-AABB centre, NOT the model_data centre: `below_center` comes
        # from the settled-grasp helper's live AABB, and mixing the two
        # frames put the coffee-box grasp ~3 cm high (2 cm from the carton
        # top -> occasional dz=0 slips).
        ent = self._entity(actor)

        def _live_center() -> np.ndarray:
            return np.asarray(self.live_entity_aabb_center(ent), dtype=float)

        obj_center = _live_center()
        grasp_pos = obj_center.copy()
        grasp_pos[2] -= below_center
        grasp_pos[2] = max(grasp_pos[2], self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN)

        self.open_gripper(arm_tag)

        above_pos = np.array([obj_center[0], obj_center[1], transport_z])
        above_link = tcp_to_link_pose(self._rotated_top_down_tcp(above_pos, tcp_yaw_deg), tcp_offset)
        if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
            print(f"[kit] {asset_name} FAIL: above-object approach failed", flush=True)
            return False

        pre_pos = obj_center.copy()
        pre_pos[2] += self.PRE_GRASP_HEIGHT
        self._move_screw(pre_pos, arm_tag)

        obj_now = _live_center()
        grasp_pos[0] = obj_now[0]
        grasp_pos[1] = obj_now[1]
        self._move_screw(grasp_pos, arm_tag)
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()

        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        print(f"[kit] {asset_name} grasp xy_err={xy_err:.3f} "
              f"target={grasp_pos.round(3).tolist()}", flush=True)
        if xy_err > self.GRASP_ALIGNMENT_XY_TOL:
            # ONE-SHOT policy: retreat vertically and fail
            # instead of re-approaching the grasp (the visible "re-try").
            print(f"[kit] {asset_name} FAIL: grasp off-centre "
                  f"({xy_err:.3f} m); aborting untouched", flush=True)
            self._guarded_transit(
                np.array([ee_pos[0], ee_pos[1], ee_pos[2]]),
                np.array([ee_pos[0], ee_pos[1], transport_z]),
                arm_tag, n_steps=self.LIFT_CARTESIAN_STEPS,
                sim_per_step=self.LIFT_SIM_PER_STEP,
            )
            return False

        z_before = float(_live_center()[2])
        self.set_gripper(close_value, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()

        # Branch-guarded straight-up lift (held item).  ``_move_screw``'s
        # plan_screw fallback could replan to a different branch and
        # swing/fling the box during the lift (dz=-0.72, box flung off
        # the bench).  The guarded Cartesian lift is a pure
        # vertical line that can't branch-jump.
        # NOTE the lift stays at yaw 0: the pre/grasp descents run through
        # the mixin ``_move_screw`` whose waypoints are yaw-0, so the wrist
        # is already canonical at close time — a yaw-held lift here fought
        # that and stalled the guard (rubikscube dz=0).
        carry_z = self.TABLE_TOP_Z + self.CARRY_Z_ABOVE_TABLE
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], carry_z])
        self._guarded_transit(
            np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2]]), lift_pos,
            arm_tag, n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.LIFT_SIM_PER_STEP,
        )
        for _ in range(self.POST_LIFT_SETTLE_STEPS):
            self.step_sim()
        last_pos = lift_pos

        z_after = float(_live_center()[2])
        print(f"[kit] {asset_name} lift dz={z_after - z_before:.3f}", flush=True)
        if (z_after - z_before) < self.LIFT_DZ_MIN:
            print(f"[kit] {asset_name} FAIL: grip slipped on lift", flush=True)
            self.open_gripper(arm_tag)
            for _ in range(self.LIFT_FAIL_RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        # Re-orient the wrist to the canonical yaw-0 top-down config while
        # the item is lifted clear (same fix as factory_inspect_pack).  The
        # transit legs below command yaw-0 waypoints; from an oriented grasp
        # (e.g. the rubikscube's -90 deg) those are a large wrist change that
        # the branch guard reads as a flip at EVERY waypoint — it skips the
        # whole carry and the arm "releases" the box back on the bench
        # instead.  A single in-place re-orient up high (the box
        # spins in hand, harmless for these symmetric packages) unifies the
        # wrist before the carry.
        ee_q = np.array(arm.get_ee_pose()[3:7], dtype=float)
        canon_q = t3d.quaternions.mat2quat(self._R_DOWN)
        if abs(float(np.dot(ee_q, canon_q))) < 0.985:
            canon_link = tcp_to_link_pose(
                self._top_down_tcp(lift_pos), tcp_offset,
            )
            self._move_seeded(canon_link.to_pose7(), arm_tag)
            for _ in range(self.POST_LIFT_SETTLE_STEPS):
                self.step_sim()

        # Purely LATERAL transit over the box at transport_z (the proven
        # smooth burger move — no descent), then release directly.  NO
        # in-box descent: that leg branch-flipped and swung/flung the arm.
        # transport_z is set low enough
        # (TRANSPORT_Z_ABOVE_TABLE) that the flat box drops gently into
        # the open box and stays in.
        # Transit via a radially-bowed midpoint (line_feeding's fix for the
        # same geometry): a straight carry passing r~0.5 m of the base flips
        # the seeded IK branch at a consistent spot, and the guard then
        # SKIPS every waypoint — the arm never leaves the bench and the
        # "release" put the box back on its spawn (observed: coffee
        # ended at [0.30, 0.13], rubik at [0.49, 0.05], both still passing
        # the slip check because arm and box were together on the bench).
        above_drop = np.array([drop_xy[0], drop_xy[1], carry_z])
        base_xy = np.asarray(arm.origin_pose.p, dtype=float)[:2]
        mid_xy = 0.5 * (np.asarray(last_pos[:2]) + above_drop[:2])
        v = mid_xy - base_xy
        r = float(np.linalg.norm(v))
        if r < 0.58:
            mid_xy = base_xy + v / max(r, 1e-6) * 0.58
        via = np.array([mid_xy[0], mid_xy[1], carry_z])
        for leg_a, leg_b in ((np.asarray(last_pos, dtype=float), via),
                             (via, above_drop)):
            self._guarded_transit(
                leg_a, leg_b, arm_tag,
                n_steps=self.TRANSIT_CARTESIAN_STEPS // 2 + 5,
                sim_per_step=self.TRANSIT_SIM_PER_STEP,
            )
        for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
            self.step_sim()

        # Arrival check: a fully guard-skipped leg leaves the arm short of
        # the box.  One CONTINUATION of the remaining leg (fresh seed often
        # heals the branch); if the guard still refuses (a genuine branch
        # wall in the Cartesian corridor), fall back to ONE gated joint-RRT
        # hop: seeded IK with the branch gate and ``plan_qpos`` ONLY —
        # never ``plan_pose``, whose unseeded IK is what flings held items.
        # Then fail the one-shot delivery if still short.
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        if float(np.linalg.norm(ee_now[:2] - above_drop[:2])) > 0.05:
            print(f"[kit] {asset_name} transit fell short "
                  f"(ee={ee_now[:2].round(3).tolist()}); continuing leg",
                  flush=True)
            self._guarded_transit(
                ee_now, above_drop, arm_tag,
                n_steps=self.TRANSIT_CARTESIAN_STEPS,
                sim_per_step=self.TRANSIT_SIM_PER_STEP,
            )
            for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
                self.step_sim()
            ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            if float(np.linalg.norm(ee_now[:2] - above_drop[:2])) > 0.05:
                if self._gated_qpos_hop(above_drop, arm_tag):
                    for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
                        self.step_sim()
                ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            if float(np.linalg.norm(ee_now[:2] - above_drop[:2])) > 0.05:
                print(f"[kit] {asset_name} FAIL: cannot reach the box "
                      f"(ee={ee_now[:2].round(3).tolist()})", flush=True)
                self.open_gripper(arm_tag)
                for _ in range(self.RELEASE_SETTLE_STEPS):
                    self.step_sim()
                return False

        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        obj_xy = _live_center()[:2]
        slip = float(np.linalg.norm(ee_xy - obj_xy))
        if slip > self.TRANSIT_SLIP_XY_TOL:
            print(f"[kit] {asset_name} FAIL: transit slip {slip:.3f}", flush=True)
            return False

        # Safe vertical descent from the high carry to the release height —
        # a pure vertical guarded line over the box can't branch-jump.
        release_pos = np.array([drop_xy[0], drop_xy[1], transport_z])
        self._guarded_transit(
            np.array([drop_xy[0], drop_xy[1], carry_z]), release_pos,
            arm_tag, n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.LIFT_SIM_PER_STEP,
        )
        for _ in range(self.POST_LIFT_SETTLE_STEPS):
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
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if self.avatar is None:
            return self._robot_pack_items(arm_tag)

        # ---- Geometry from per-episode clip dry-run ----------------------
        rel = self._measure_clip_anchors()
        put_frames = [f for (t, f, _h, _i, _o) in _CLIP_EVENTS if t == "put"]
        pick_frames = [f for (t, f, _h, _i, _o) in _CLIP_EVENTS if t == "pick"]
        target_put = np.array([
            self.BOX_XY[0] + self.AVATAR_DROP_OFFSET_XY[0],
            self.BOX_XY[1] + self.AVATAR_DROP_OFFSET_XY[1],
        ])
        body = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        body[:2] = target_put - rel[put_frames[0]][:2]
        # Bench-edge clearance: her working pelvis sits forward of the
        # reset point; push the body south until it clears the bench (her
        # puts move with her — the shift cap keeps them inside the box).
        pelvis_rel = getattr(self, "_pelvis_rel", None)
        if pelvis_rel is not None:
            pelvis_y = float(body[1] + pelvis_rel[1])
            limit_y = self.BENCH_FRONT_Y - self.AVATAR_BENCH_CLEARANCE
            deficit = pelvis_y - limit_y
            if deficit > 0.0:
                shift = min(deficit, self.AVATAR_BODY_SHIFT_MAX)
                body[1] -= shift
                print(f"[kit] bench clearance: stance pelvis y {pelvis_y:.3f} > "
                      f"{limit_y:.3f}, body shifted south {shift:.3f}"
                      + (" (CAPPED — residual overlap likely)"
                         if shift < deficit else ""), flush=True)
        raw_body = body.copy()
        body[0] = float(np.clip(body[0], *self.AVATAR_BODY_X_BOUNDS))
        body[1] = min(body[1], self.AVATAR_BODY_Y_MAX)
        if np.linalg.norm(body[:2] - raw_body[:2]) > 1e-6:
            print(f"[kit] WARNING avatar body clamped "
                  f"{raw_body[:2].round(3).tolist()} -> {body[:2].round(3).tolist()}; "
                  f"puts will land off the box target by the same amount", flush=True)
        for f in put_frames:
            land = body[:2] + rel[f][:2]
            print(f"[kit] put frame {f} lands at {land.round(3).tolist()} "
                  f"(z {float(body[2] + rel[f][2]):.3f}), box={self.BOX_XY}", flush=True)

        # ---- Avatar items at the (shifted) pick anchors -------------------
        for (actor, _asset, _mid, half_h), f in zip(self.avatar_item_actors, pick_frames):
            xy = self._push_xy_outside_box(body[:2] + rel[f][:2])
            ent = self._entity(actor)
            ent.set_pos(np.array([xy[0], xy[1], self.TABLE_TOP_Z + half_h + 0.002]))
            ent.set_quat(_Q_UPRIGHT.astype(float))
            try:
                ent.set_dofs_velocity(np.zeros(6, dtype=float))
            except Exception:
                pass
            print(f"[kit] avatar item at pick anchor f{f}: {xy.round(3).tolist()} "
                  f"(anchor z {float(body[2] + rel[f][2]):.3f})", flush=True)

        # Robot drop point: inside the box on the side AWAY from the
        # worker's static end-of-clip hand, so the robot's drop never
        # reaches near her.  (She now holds her own clip's last frame as
        # the end rest pose — arm still over the box — so the robot
        # avoids her in SPACE rather than her moving out of the way.)
        bx, by = self.BOX_XY
        palms = getattr(self, "_standby_palm_rel", {}) or {}
        if palms:
            hand_xy = body[:2] + palms[max(palms)][:2]
            away = np.asarray([bx, by], dtype=float) - hand_xy
            nrm = float(np.linalg.norm(away))
            away = away / nrm if nrm > 1e-6 else np.array([0.5, 0.87])
            drop = np.array([bx, by], dtype=float) + away * self.ROBOT_DROP_RADIUS
            mx = self.BOX_INNER_HALF[0] - self.ROBOT_DROP_EDGE_MARGIN
            my = self.BOX_INNER_HALF[1] - self.ROBOT_DROP_EDGE_MARGIN
            drop[0] = float(np.clip(drop[0], bx - mx, bx + mx))
            drop[1] = float(np.clip(drop[1], by - my, by + my))
            self._robot_drop_xy = (float(drop[0]), float(drop[1]))
            # Second robot item: a slot displaced perpendicular to the
            # away-from-worker direction, still inside the box margins.
            perp = np.array([-away[1], away[0]])
            drop2 = np.array([bx, by], dtype=float) + away * (
                self.ROBOT_DROP_RADIUS * 0.5
            ) + perp * 0.09
            drop2[0] = float(np.clip(drop2[0], bx - mx, bx + mx))
            drop2[1] = float(np.clip(drop2[1], by - my, by + my))
            self._robot_drop_xys = [
                self._robot_drop_xy, (float(drop2[0]), float(drop2[1])),
            ]
            print(f"[kit] worker end-frame hand {hand_xy.round(3).tolist()} "
                  f"-> robot drops {np.round(self._robot_drop_xys, 3).tolist()}",
                  flush=True)
        else:
            self._robot_drop_xy = (bx + 0.10, by + 0.05)
            self._robot_drop_xys = [(bx + 0.10, by + 0.05), (bx + 0.10, by - 0.05)]

        # Park the robot hand out of her sweep volume before she starts —
        # the default home pose leaves the hand inside her reach corridor.
        arm = self.robot.get_arm(arm_tag)
        park_link = tcp_to_link_pose(
            self._top_down_tcp(np.asarray(self.ROBOT_PARK_TCP, dtype=float)),
            arm.tcp_offset,
        )
        self._move_seeded(park_link.to_pose7(), arm_tag)

        self.avatar.reset(global_trans=body, global_rot=self._assist_rot())
        # Begin-rest = begin-frame: hold her clip's frame 0 during the
        # settle (instead of the default stop_pose, whose orientation
        # differs and snaps when the clip starts).
        self._hold_clip_frame(0)
        self.avatar_collided = False
        self.avatar_collision_log = []
        for _ in range(self.AVATAR_PRE_MOTION_SETTLE_STEPS):
            self.step_sim()

        # ---- Worker packs FULLY, THEN the robot ----------------------
        # The robot waits for the worker to be COMPLETELY done (spare(),
        # i.e. through her retract to the end-frame rest pose) before
        # picking.  Waiting only for her last put left her still
        # animating the retract during the robot's pick, and her moving
        # kinematic arm re-bumped the box AFTER the pre-grasp re-level —
        # so the one-shot grasp slipped on a displaced box.  With her
        # fully static, the re-levelled box stays pristine through the
        # grasp.  (Avatar-first throughout avoids the robot ever reaching
        # toward her; robot-first collided with her frozen begin arm.)
        self._start_avatar_packing()
        # Overlap with the worker's retract tail: start the robot once her
        # FINAL put event has fired (waiting for the full
        # clip leaves the robot visibly idle for most of the episode).  The
        # per-item hand-clear gate inside _robot_pack_items keeps her
        # retracting arm out of the first pick's corridor.
        self._wait_avatar_event(
            self.ROBOT_START_AFTER_EVENT, self.ROBOT_START_WAIT_STEPS,
        )
        ok_robot = self._robot_pack_items(arm_tag)
        tail = 0
        while not self.avatar.spare() and tail < self.AVATAR_FINISH_WAIT_STEPS:
            self.step_sim()
            tail += 1

        self._restore_step_sim()
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_robot

    # ------------------------------------------------------------------
    # Success: every kit item inside the box interior AABB.
    # ------------------------------------------------------------------
    def _item_in_box(self, pos: np.ndarray) -> bool:
        bx, by = self.BOX_XY
        ihx, ihy = self.BOX_INNER_HALF
        z0 = self.TABLE_TOP_Z
        return (
            abs(float(pos[0]) - bx) <= ihx
            and abs(float(pos[1]) - by) <= ihy
            and z0 - 0.01 <= float(pos[2]) <= z0 + self.SUCCESS_Z_MAX_ABOVE_TABLE
        )

    def check_success(self) -> bool:
        # Per-item box verdict (printed so robot-only debug runs, whose
        # avatar items are never packed, are still interpretable).
        for actor, asset, mid, *_rest in self.robot_item_actors:
            c = self._get_object_world_center(actor, asset, mid)
            print(f"[kit] FINAL robot {asset} pos={c.round(3).tolist()} "
                  f"in_box={self._item_in_box(c)}", flush=True)
        for actor, asset, _mid, _half_h in self.avatar_item_actors:
            p = to_numpy(self._entity(actor).get_pos()).ravel()[:3]
            print(f"[kit] FINAL avatar {asset} pos={p.round(3).tolist()} "
                  f"in_box={self._item_in_box(p)}", flush=True)
        if (self.config.get("track_avatar_collision", True)
                and getattr(self, "avatar_collided", False)):
            return False
        for actor, asset, mid, *_rest in self.robot_item_actors:
            if not self._item_in_box(self._get_object_world_center(actor, asset, mid)):
                return False
        for actor, _asset, _mid, _half_h in self.avatar_item_actors:
            pos = to_numpy(self._entity(actor).get_pos()).ravel()[:3]
            if not self._item_in_box(pos):
                return False
        return True

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        per_item = {}
        for actor, asset, mid, *_rest in self.robot_item_actors:
            c = self._get_object_world_center(actor, asset, mid)
            per_item[f"robot_{asset}"] = bool(self._item_in_box(c))
        for actor, asset, _mid, _half_h in self.avatar_item_actors:
            pos = to_numpy(self._entity(actor).get_pos()).ravel()[:3]
            per_item[f"avatar_{asset}"] = bool(self._item_in_box(pos))
        metrics["items_in_box"] = per_item
        metrics["avatar_collided"] = bool(getattr(self, "avatar_collided", False))
        return metrics
