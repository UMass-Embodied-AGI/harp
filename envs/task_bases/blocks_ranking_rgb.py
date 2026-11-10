"""Blocks ranking RGB: arrange red, green, and blue cubes left-to-right.

Robot-only v1 of RoboTwin's ``blocks_ranking_rgb`` task.  A single Franka
sorts three primitive cubes into a fixed row ordered by color: red, green,
blue along increasing world X.
"""

import numpy as np

from ..base_task import BaseTask, TABLE_HEIGHT
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..interaction.block_ranking import (
    BlockEpisodePlan,
    BlockEpisodeProvider,
    BlockRankingAssistController,
    BlockRankingInterruptController,
    BlockSpawnSpec,
)
from ..grasp import tcp_to_link_pose
from ..manipulation import PickSpec, PlaceSpec, TopDownPickPlaceMixin
from ..utils import Pose, create_primitive, to_numpy
from .resting_guard import held_by_anyone
from .block_grasp import (
    block_neighbor_obstacles,
    endpoint_matches,
    execute_grasp_leg,
    move_block_seeded,
    plan_block_grasp,
    refresh_block_obstacles,
    solve_block_lift_ik,
)


class BlocksRankingRGB(TopDownPickPlaceMixin, BaseTask):
    """Robot places red, green, and blue blocks in left-to-right order."""

    INSTRUCTION = (
        "place the red block, green block, and blue block in a row from "
        "left to right"
    )
    use_avatar = False

    CUBE_HALF = 0.020
    CUBE_COLORS = {
        "red": (0.85, 0.08, 0.06),
        "green": (0.10, 0.60, 0.18),
        "blue": (0.08, 0.22, 0.85),
    }

    # World-X row: red, green, blue.  The row sits forward of the random
    # spawn zone so the task is visually legible and leaves clean approach
    # columns for top-down placement.
    SLOT_XS = {
        "red": -0.16,
        "green": 0.0,
        "blue": 0.16,
    }
    SLOT_Y = -0.08

    TRANSPORT_Z_ABOVE_TABLE = 0.20
    RELEASE_Z_ABOVE_TABLE = 0.080
    SUCCESS_DIST_XY = 0.055
    MIN_BLOCK_SPACING = 0.105
    ON_TABLE_HEIGHT_FRACTION = 0.35
    # A placed block rests on the table: its centre at most this far above
    # its resting height (table + half size), and held by neither the avatar
    # nor the gripper.  Without it a block carried over its slot counted.
    SUCCESS_REST_MAX_ABOVE_TABLE_M = 0.02

    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.55, 0.50)
    TABLE_CENTER_XY = (0.0, -0.30)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.50, -0.75),
        (0.50, -0.75),
        (-0.50, 0.15),
        (0.50, 0.15),
    )

    SPAWN_X_RANGE = (-0.24, 0.24)
    SPAWN_Y_RANGE = (-0.38, -0.22)
    SPAWN_SAMPLE_ATTEMPTS = 80
    CUBE_Z_CLEARANCE = 0.002

    TABLE_SETTLE_STEPS = 80
    FAILURE_RECOVERY_STEPS = 60
    FINAL_SETTLE_STEPS = 200

    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 200.0

    BLOCK_GRASP_PREFLIGHT = True
    # When no forward branch reaches the cube, try the grasp-IK-first
    # (backward) preflight before giving the block up.
    BLOCK_GRASP_BACKWARD_PREFLIGHT = True

    recording_camera_pos = [0.85, -0.65, 1.55]
    recording_camera_lookat = [0.0, -0.22, 0.82]
    side_camera_pos = [0.0, -0.25, 2.45]
    side_camera_lookat = [0.0, -0.22, 0.78]

    # Real collection setting: rotate the authored sim layout by -90 deg
    # around world Z so the robot side moves from world y- to world x-.
    REAL_AXIS_ROT = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        self._collection_setting = str(
            cfg.get("setting", cfg.get("collection_setting", "sim"))
        ).strip().lower()
        self._configure_collection_layout()
        super().__init__(cfg)

    def _real_setting_enabled(self) -> bool:
        return self._collection_setting == "real"

    def _layout_xy(self, xy) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        if not self._real_setting_enabled():
            return arr.copy()
        return np.array([arr[1], -arr[0]], dtype=np.float64)

    def _layout_xyz(self, xyz) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float64).ravel()[:3]
        out = arr.copy()
        out[:2] = self._layout_xy(arr[:2])
        return out

    def _layout_xys(self, xys) -> np.ndarray:
        arr = np.asarray(xys, dtype=np.float64)
        if not self._real_setting_enabled():
            return arr.copy()
        out = arr.copy()
        out[:, 0] = arr[:, 1]
        out[:, 1] = -arr[:, 0]
        return out

    def _configure_collection_layout(self) -> None:
        if not self._real_setting_enabled():
            return
        cls = type(self)
        self.TABLE_CENTER_XY = tuple(self._layout_xy(cls.TABLE_CENTER_XY))
        self.TABLE_HALF_SIZE = (
            float(cls.TABLE_HALF_SIZE[1]),
            float(cls.TABLE_HALF_SIZE[0]),
        )
        self.TABLE_LEG_XY = tuple(
            tuple(self._layout_xy(xy)) for xy in cls.TABLE_LEG_XY
        )
        for attr in (
            "recording_camera_pos",
            "recording_camera_lookat",
            "side_camera_pos",
            "side_camera_lookat",
        ):
            value = getattr(cls, attr, None)
            if value is not None:
                setattr(self, attr, self._layout_xyz(value).tolist())
        if hasattr(cls, "avatar_init_pos"):
            self.avatar_init_pos = self._layout_xyz(cls.avatar_init_pos)
        if hasattr(cls, "avatar_init_rot"):
            self.avatar_init_rot = self.REAL_AXIS_ROT @ np.asarray(
                cls.avatar_init_rot, dtype=np.float64
            )
        self._R_DOWN = self.REAL_AXIS_ROT @ np.asarray(cls._R_DOWN, dtype=np.float64)

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

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka" and self._real_setting_enabled():
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", self._layout_xyz([0.0, -0.65, 0.75]))
            kwargs.setdefault("quat", [1.0, 0.0, 0.0, 0.0])
        super()._load_robot()

    def _sample_spawn_xys(self):
        """Sample a random non-overlapping layout away from the target row."""
        x_lo, x_hi = self.SPAWN_X_RANGE
        # Avoid the near-base band around y < -0.40: top-down Franka IK can
        # fail there for left-side cubes even though the position is close.
        y_lo, y_hi = self.SPAWN_Y_RANGE
        for _ in range(self.SPAWN_SAMPLE_ATTEMPTS):
            xys = np.column_stack([
                np.random.uniform(x_lo, x_hi, 3),
                np.random.uniform(y_lo, y_hi, 3),
            ])
            ok = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if np.linalg.norm(xys[i] - xys[j]) < self.MIN_BLOCK_SPACING:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return self._layout_xys(xys)
        raise RuntimeError("[blocks_rgb] could not sample non-overlapping cubes")

    def _target_slot_pos(self, key: str, z: float) -> np.ndarray:
        # Slot protection is handled on the neutral-avatar side: the blocks
        # neutral variants set NEUTRAL_SLOTS_ARE_PHYSICAL so the layout
        # solver (which runs AFTER these slots are materialized) keeps the
        # avatar's work footprint off the target row.
        xy = self._layout_xy([self.SLOT_XS[key], self.SLOT_Y])
        return np.array([xy[0], xy[1], z], dtype=float)

    def _order_axis_value(self, pos) -> float:
        p = np.asarray(pos, dtype=np.float64).ravel()
        if self._real_setting_enabled():
            # Authored +X maps to world -Y under the real-axis rotation.
            return float(-p[1])
        return float(p[0])

    def block_keys(self) -> tuple[str, ...]:
        """Return the objective's canonical placement order."""

        return tuple(self.CUBE_COLORS)

    def block_half_size(self, key: str) -> float:
        return float(self.CUBE_HALF)

    def block_pick_label(self, key: str) -> str:
        return f"{key} block"

    def block_center(self, key: str) -> np.ndarray:
        """Return a block center as a CPU NumPy vector."""

        return to_numpy(self.blocks[key].get_pos()).ravel()[:3]

    def _base_block_episode_plan(self) -> BlockEpisodePlan:
        keys = self.block_keys()
        order = tuple(np.random.permutation(keys))
        xys = self._sample_spawn_xys()
        return BlockEpisodePlan(
            blocks=tuple(
                BlockSpawnSpec(
                    key=key,
                    color_name=key,
                    half_size=self.block_half_size(key),
                    xy=(float(xys[index, 0]), float(xys[index, 1])),
                )
                for index, key in enumerate(order)
            )
        )

    def _spawn_block_episode(self, plan: BlockEpisodePlan) -> None:
        self.blocks = {}
        records_distinct_colors = any(
            spec.key != spec.color_name for spec in plan.blocks
        )
        if records_distinct_colors:
            self.block_colors = {}
        for spec in plan.blocks:
            cube_z = (
                self.TABLE_TOP_Z + spec.half_size + self.CUBE_Z_CLEARANCE
            )
            cube = create_primitive(
                self.scene,
                "box",
                Pose([spec.xy[0], spec.xy[1], cube_z]),
                size={"half_size": (spec.half_size,) * 3},
                color=self.CUBE_COLORS[spec.color_name],
                is_static=False,
            )
            self.blocks[spec.key] = cube
            if records_distinct_colors:
                self.block_colors[spec.key] = spec.color_name

        self.target_slots = {
            key: self._target_slot_pos(
                key, self.TABLE_TOP_Z + self.block_half_size(key)
            )
            for key in self.block_keys()
        }

    def load_actors(self):
        controller = self.interaction_controller
        if isinstance(controller, BlockEpisodeProvider):
            plan = controller.create_block_episode(self)
        else:
            plan = self._base_block_episode_plan()
        self._spawn_block_episode(plan)

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

    def _pick_and_place_block(
        self, color_name: str, arm_tag: str = "right"
    ) -> bool:
        key = color_name
        cube = self.blocks[key]
        target = self.target_slots[key]
        half = self.block_half_size(key)
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        release_z = self.TABLE_TOP_Z + self.RELEASE_Z_ABOVE_TABLE

        pick = PickSpec(
            get_center=lambda c=cube: to_numpy(c.get_pos()).ravel()[:3],
            radius=half,
            label=self.block_pick_label(key),
        )
        place = PlaceSpec(
            pos=np.array([target[0], target[1], release_z], dtype=float),
            label=f"{key} slot",
            transport_z=transport_z,
            release=True,
        )
        if not self.BLOCK_GRASP_PREFLIGHT:
            return bool(self.pick_and_place(pick, place, arm_tag))
        with block_neighbor_obstacles(self, key, arm_tag):
            preflight = plan_block_grasp(
                self, pick.get_center(), half, transport_z, arm_tag,
            )
            rounds = 0
            while preflight is None:
                # Optional neutral fallback: None means it declined or found
                # no route without moving; True/False is an executed outcome.
                fallback = getattr(self, "_block_preflight_fallback", None)
                if fallback is not None:
                    outcome = fallback(pick, place, arm_tag)
                    if outcome is not None:
                        return bool(outcome)
                # Optional bounded hold: the moving avatar may be occupying
                # the cube right now. Re-run the unchanged preflight after
                # ordinary PD/simulation steps; the hook owns its own bound.
                hold = getattr(self, "_block_preflight_hold", None)
                if hold is None or not hold(rounds, arm_tag):
                    return False
                rounds += 1
                preflight = plan_block_grasp(
                    self, pick.get_center(), half, transport_z, arm_tag,
                )
            wait = getattr(self, "_block_wait_for_grasp_site", None)
            if wait is not None and wait(preflight, arm_tag):
                # The human moved on while the arm held still: re-check the
                # approach against the current clouds before committing.
                replanned = plan_block_grasp(
                    self, pick.get_center(), half, transport_z, arm_tag,
                )
                if replanned is not None:
                    preflight = replanned
            self._active_block_grasp_plan = preflight
            self._block_place_shift = np.zeros(2)
            try:
                return bool(self.pick_and_place(pick, place, arm_tag))
            finally:
                self._active_block_grasp_plan = None
                self._block_place_shift = np.zeros(2)

    # Full-state neutral expert only: the scripted human can sweep
    # through a cube while the arm descends, closes and makes the tiny lift
    # (~600 sim steps parked at the cube).  Before the approach, look ahead
    # along the human's clip over that whole window and hold still (arm at its
    # current, clear pose) until the grasp site stays clear.  Bounded; on
    # budget exhaustion the unchanged pick proceeds.
    BLOCK_GRASP_SITE_WAIT = True
    BLOCK_GRASP_SITE_DWELL_STEPS = 600
    BLOCK_GRASP_SITE_CLEARANCE_M = 0.12
    BLOCK_GRASP_SITE_MAX_WAIT_STEPS = 4000

    def _block_site_clearance_m(self) -> float:
        return float((self.config or {}).get("block_grasp_site_clearance_m", self.BLOCK_GRASP_SITE_CLEARANCE_M))

    def _block_grasp_site_clearance(self, preflight, horizon_from: int, horizon_to: int) -> float:
        from .neutral_human_aware import point_capsule_distance, predict_human_capsules
        arm = self.robot.get_arm("right")
        grasp_link = Pose.from_pose7(np.asarray(preflight.legs[-1][0], dtype=float))
        pre_link = Pose.from_pose7(np.asarray(preflight.legs[-2][0], dtype=float))
        low = (grasp_link * arm.tcp_offset).p
        high = (pre_link * arm.tcp_offset).p
        site = [low + f * (high - low) for f in np.linspace(0.0, 1.0, 4)]
        offsets = list(range(int(horizon_from), int(horizon_to) + 1, 25))
        best = float("inf")
        for caps in predict_human_capsules(self, offsets):
            for _n, a, b, r in caps:
                for p in site:
                    best = min(best, point_capsule_distance(p, a, b, r))
        return best

    # The scripted human may keep working over one cube; pick a cube whose
    # grasp site the look-ahead shows clear first (full-state neutral only;
    # the canonical order is kept whenever its next cube is clear).
    BLOCK_SITE_ORDER_WINDOW = (800, 2600)

    def _block_center_site_clearance(self, key: str) -> float:
        from .neutral_human_aware import point_capsule_distance, predict_human_capsules
        center = self.block_center(key).astype(float)
        site = [center + np.array([0.0, 0.0, dz]) for dz in (0.0, 0.075, 0.15)]
        lo, hi = self.BLOCK_SITE_ORDER_WINDOW
        best = float("inf")
        for caps in predict_human_capsules(self, list(range(int(lo), int(hi) + 1, 50))):
            for _n, a, b, r in caps:
                for p in site:
                    best = min(best, point_capsule_distance(p, a, b, r))
        return best

    def _next_block_key(self, pending):
        enabled = getattr(self, "_neutral_aware_enabled", None)
        if len(pending) < 2 or enabled is None or not enabled():
            return pending[0]
        try:
            first = self._block_center_site_clearance(pending[0])
            if first >= self._block_site_clearance_m():
                return pending[0]
            scored = [(self._block_center_site_clearance(k), k) for k in pending[1:]]
        except Exception as exc:
            print(f"[blocks-grasp] site order look-ahead unavailable: {exc}", flush=True)
            return pending[0]
        best, key = max(scored, key=lambda item: item[0])
        if best >= self._block_site_clearance_m():
            print(f"[blocks-grasp] {pending[0]} site contested ({first:.3f} m); "
                  f"placing {key} first ({best:.3f} m)", flush=True)
            return key
        return pending[0]

    def _block_wait_for_grasp_site(self, preflight, arm_tag: str) -> bool:
        enabled = getattr(self, "_neutral_aware_enabled", None)
        if (not self.BLOCK_GRASP_SITE_WAIT or enabled is None or not enabled()
                or preflight is None or len(preflight.legs) < 3):
            return False
        lengths = [len(np.asarray(plan.position)) for _t, plan in preflight.legs]
        start = int(sum(lengths[:-2]))
        end = int(sum(lengths)) + int(self.BLOCK_GRASP_SITE_DWELL_STEPS)
        arm = self.robot.get_arm(arm_tag)
        hold = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
        waited = 0
        tick = 25
        while True:
            try:
                clearance = self._block_grasp_site_clearance(preflight, start, end)
            except Exception as exc:
                print(f"[blocks-grasp] grasp-site look-ahead unavailable: {exc}", flush=True)
                return waited > 0
            if clearance >= self._block_site_clearance_m():
                break
            if waited + tick > self.BLOCK_GRASP_SITE_MAX_WAIT_STEPS or getattr(self, "avatar_collided", False):
                print(f"[blocks-grasp] grasp-site wait budget spent (clearance {clearance:.3f} m)", flush=True)
                break
            # Never hold still where the human's clip is about to reach the
            # arm itself (the human never adapts): then go on with the pick.
            threat_fn = getattr(self, "_neutral_aware_robot_threat", None)
            if threat_fn is not None:
                try:
                    prm = self._neutral_aware_params()
                    threat = float(threat_fn(arm_tag, prm))
                except Exception:
                    threat = float("inf")
                if threat < prm["clearance"]:
                    print(f"[blocks-grasp] grasp-site wait stopped: the human reaches the "
                          f"holding arm (clearance {threat:.3f} m)", flush=True)
                    break
            if waited == 0:
                print(f"[blocks-grasp] grasp site contested by the human (clearance {clearance:.3f} m); "
                      f"holding", flush=True)
            arm.set_arm_joints(hold.copy())
            for _ in range(tick):
                self.step_sim()
            waited += tick
        if waited:
            print(f"[blocks-grasp] grasp-site wait {waited} steps", flush=True)
        return waited > 0

    def _block_straight_move(self, tcp_pos, arm_tag: str) -> bool:
        """One checked straight (screw) move of the TCP, top-down at the
        current wrist yaw; never an RRT.  False (nothing moved) on no path."""
        arm = self.robot.get_arm(arm_tag)
        refresh_block_obstacles(self, arm_tag)
        q = np.asarray(arm.get_arm_qpos(), dtype=float)
        ee = np.asarray(arm.get_ee_pose(), dtype=float)
        target = tcp_to_link_pose(Pose(np.asarray(tcp_pos, dtype=float), ee[3:7]), arm.tcp_offset).to_pose7()
        for step in (None, .05, .02):
            plan = (arm.planner.plan_screw_path(q, target, log=False) if step is None
                    else arm.planner.plan_screw_path(q, target, log=False, qpos_step=step))
            if endpoint_matches(arm.planner, plan, target):
                self.execute_plan(plan, arm_tag)
                return True
        return False

    # After opening over the slot, rise straight to transport height
    # before the next approach.  Planning the next RRT from inside the opened
    # jaws around the placed cube starts "in collision" with that neighbour;
    # MPLib then perturbs the start and the executed path can sweep it away.
    BLOCK_RETREAT_AFTER_RELEASE = True
    BLOCK_RETREAT_SETTLE_STEPS = 60

    def _after_pick_place_release(self, pick, place, arm_tag, release_pos):
        if not self.BLOCK_RETREAT_AFTER_RELEASE or getattr(self, "_active_block_grasp_plan", None) is None:
            return
        for _ in range(self.BLOCK_RETREAT_SETTLE_STEPS):
            self.step_sim()
        arm = self.robot.get_arm(arm_tag)
        ee = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        if ee[2] >= transport_z - 0.02 or getattr(self, "avatar_collided", False):
            return
        try:
            if not self._block_straight_move(np.array([ee[0], ee[1], transport_z]), arm_tag):
                print("[blocks-grasp] post-release retreat: no straight path", flush=True)
        except Exception as exc:
            print(f"[blocks-grasp] post-release retreat skipped: {exc}", flush=True)

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        # Keep PickSpec.get_grasp_yaw unset: the mixin's oriented-grasp path
        # explicitly dispatches its own primitives, bypassing this queue.
        preflight = getattr(self, "_active_block_grasp_plan", None)
        if preflight is not None:
            yaw += preflight.yaw
        return super()._top_down_tcp(pos, yaw=yaw)

    def _move_seeded(self, link_pose7, arm_tag):
        preflight = getattr(self, "_active_block_grasp_plan", None)
        if preflight is not None and preflight.legs:
            return execute_grasp_leg(self, preflight, link_pose7, arm_tag)
        if getattr(self, "_block_straight_only", False):
            # Slot-offset retries accept straight screw paths only: a joint
            # RRT beside a human's prop can swing a link into it under PD lag.
            self.plan_success = False
            raise RuntimeError("slot-offset retry has no straight path")
        retry = (None if getattr(self, "_block_skip_hold_retry", False)
                 else getattr(self, "_neutral_aware_plan_retry", None))
        if retry is not None:
            # Full-state neutral expert: hold (block in hand) and re-plan once
            # the human leaves the target; no-op for every other expert.
            result = retry(lambda: move_block_seeded(self, link_pose7, arm_tag), arm_tag,
                           label="block seeded move")
        else:
            result = move_block_seeded(self, link_pose7, arm_tag)
        if result is None and preflight is not None:
            # The inherited screw fallback ignores a None return. Abort the
            # active pickup through its normal recovery instead of continuing.
            raise RuntimeError("block seeded fallback has no checked MPLib path")
        return result

    def _solve_ik(self, tcp_pose, arm_tag):
        return solve_block_lift_ik(self, tcp_pose, arm_tag)

    def _move_screw(self, target_pos, arm_tag, tcp_yaw: float = 0.0):
        preflight = getattr(self, "_active_block_grasp_plan", None)
        arm = self.robot.get_arm(arm_tag)
        if preflight is not None and float(arm.gripper_val) < 0.999:
            preflight.closed = True
        if preflight is not None and not preflight.closed:
            target = tcp_to_link_pose(
                self._top_down_tcp(target_pos, yaw=tcp_yaw), arm.tcp_offset,
            ).to_pose7()
            if preflight.legs:
                execute_grasp_leg(self, preflight, target, arm_tag)
                return True
            # The shared helper may refine a millimetre-scale tracking error
            # before closing.  Keep that correction straight as well.
            refresh_block_obstacles(self, arm_tag)
            plan = arm.planner.plan_screw_path(arm.get_arm_qpos(), target, log=False)
            if not endpoint_matches(arm.planner, plan, target):
                # A tiny final integration increment can make MPLib reject an
                # otherwise converged correction. Retry at finer resolution;
                # collision checks and endpoint quality remain mandatory.
                print("[blocks-grasp] pre-close correction retry qpos_step=0.03", flush=True)
                plan = arm.planner.plan_screw_path(
                    arm.get_arm_qpos(), target, log=False, qpos_step=.03,
                )
                if not endpoint_matches(arm.planner, plan, target):
                    # A sub-centimetre centring move is optional
                    # for a cube between 8 cm open jaws; close where the arm
                    # already is instead of abandoning the block.
                    try:
                        tcp_now = np.asarray(arm.get_ee_pose()[:3], dtype=float)
                        shift = float(np.linalg.norm(
                            tcp_now[:2] - np.asarray(target_pos, dtype=float)[:2]))
                    except Exception:
                        shift = float("inf")
                    if shift <= float(getattr(self, "BLOCK_OPTIONAL_CORRECTION_M", 0.0)):
                        print(f"[blocks-grasp] pre-close correction skipped (shift={shift:.4f}m, "
                              f"no straight path)", flush=True)
                        return True
                    raise RuntimeError("block pre-close correction has no straight path")
            self.execute_plan(plan, arm_tag)
            return True
        refresh_block_obstacles(self, arm_tag)
        stage = self._block_transfer_stage(preflight, target_pos, arm)
        if stage is not None:
            return self._block_transfer_with_shifts(stage, preflight, target_pos, arm_tag, tcp_yaw)
        return super()._move_screw(target_pos, arm_tag, tcp_yaw=tcp_yaw)

    def _block_transfer(self, stage, preflight, target_pos, arm_tag, tcp_yaw, with_hold=True,
                        half_turns=True):
        if stage == "release" and getattr(self, "_neutral_aware_enabled", lambda: False)():
            return self._carry_with_yaw_alternatives(
                preflight, target_pos, arm_tag, tcp_yaw,
                move=lambda: self._aware_place_descent(target_pos, arm_tag, tcp_yaw),
                with_hold=with_hold, half_turns=half_turns)
        return self._carry_with_yaw_alternatives(preflight, target_pos, arm_tag, tcp_yaw,
                                                 with_hold=with_hold, half_turns=half_turns)

    # The slot tolerance is SUCCESS_DIST_XY (5.5 cm).  When neither
    # the carry over the slot nor the release descent has a checked path at
    # any wrist yaw (a human's laptop/bottle box or arm next to the slot),
    # aim the held block up to 3 cm off the slot centre, away from the
    # nearest obstacle.  Later legs of this placement keep the same offset.
    BLOCK_PLACE_SHIFTS_M = (0.015, 0.030)
    BLOCK_PLACE_SHIFT_TURNS_DEG = (0.0, 50.0, -50.0, 100.0, -100.0)
    BLOCK_HIGH_RELEASE_RAISES_M = (0.04, 0.08)

    def _block_shift_away_dir(self, slot_xy):
        slot_xy = np.asarray(slot_xy, dtype=float)[:2]
        sources = []
        boxes = getattr(self, "_neutral_object_aabb_boxes", None)
        if boxes is not None:
            try:
                sources += [np.asarray(c, dtype=float)[:2] for _n, c, _a, _h in boxes()]
            except Exception:
                pass
        collider = getattr(self, "avatar_collider", None)
        if collider is not None:
            try:
                for _n, a, b, _r in collider.current_capsules():
                    for p in (a, b):
                        p = np.asarray(p, dtype=float)
                        if p[2] < self.TABLE_TOP_Z + 0.35:
                            sources.append(p[:2])
            except Exception:
                pass
        near = [p for p in sources if np.all(np.isfinite(p)) and np.linalg.norm(p - slot_xy) < 0.35]
        if near:
            nearest = min(near, key=lambda p: float(np.linalg.norm(p - slot_xy)))
            away = slot_xy - nearest
            if np.linalg.norm(away) > 1e-6:
                return away / np.linalg.norm(away)
        return np.array([0.0, -1.0])

    def _block_transfer_with_shifts(self, stage, preflight, target_pos, arm_tag, tcp_yaw):
        arm = self.robot.get_arm(arm_tag)
        shift = np.asarray(getattr(self, "_block_place_shift", np.zeros(2)), dtype=float)
        base = np.asarray(target_pos, dtype=float).copy()
        try:
            goal = base.copy()
            goal[:2] += shift
            return self._block_transfer(stage, preflight, goal, arm_tag, tcp_yaw)
        except RuntimeError as exc:
            if (np.linalg.norm(shift) > 0 or not self.BLOCK_PLACE_SHIFTS_M
                    or getattr(self, "avatar_collided", False)):
                raise
            first = exc
        q0 = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
        away = self._block_shift_away_dir(base[:2])
        for dist in self.BLOCK_PLACE_SHIFTS_M:
            for turn in self.BLOCK_PLACE_SHIFT_TURNS_DEG:
                t = np.deg2rad(turn)
                d = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ away
                candidate = float(dist) * d
                goal = base.copy()
                goal[:2] += candidate
                print(f"[blocks-grasp] {stage} retry with slot offset "
                      f"[{candidate[0]:+.3f}, {candidate[1]:+.3f}] m", flush=True)
                self.plan_success = True
                self._block_straight_only = True
                try:
                    result = self._block_transfer(stage, preflight, goal, arm_tag, tcp_yaw,
                                                  with_hold=False, half_turns=False)
                except RuntimeError:
                    moved = float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=float) - q0)))
                    if moved > 0.02 or getattr(self, "avatar_collided", False):
                        raise
                    continue
                finally:
                    self._block_straight_only = False
                self._block_place_shift = candidate
                return result
        if stage == "release" and self.BLOCK_HIGH_RELEASE_RAISES_M:
            # No checked descent anywhere inside the slot tolerance
            # (a human's laptop box beside the slot).  Release higher above
            # the slot centre instead, straight paths only; as a last resort
            # open where the arm already holds the cube over the slot.
            for raise_m in self.BLOCK_HIGH_RELEASE_RAISES_M:
                goal = base.copy()
                goal[2] += float(raise_m)
                self.plan_success = True
                self._block_straight_only = True
                try:
                    result = self._block_transfer(stage, preflight, goal, arm_tag, tcp_yaw,
                                                  with_hold=False, half_turns=False)
                except RuntimeError:
                    moved = float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=float) - q0)))
                    if moved > 0.02 or getattr(self, "avatar_collided", False):
                        raise
                    continue
                finally:
                    self._block_straight_only = False
                print(f"[blocks-grasp] release from {raise_m:.2f} m above the release height", flush=True)
                return result
            ee = np.asarray(arm.get_ee_pose()[:3], dtype=float)
            if float(np.linalg.norm(ee[:2] - base[:2])) <= 0.02:
                print("[blocks-grasp] no checked descent; releasing from transport height", flush=True)
                self.plan_success = True
                return True
        self.plan_success = False
        raise first

    def _block_transfer_stage(self, preflight, target_pos, arm):
        """'carry' (lateral move at transport height to the slot), 'release'
        (descent to the release height) or None, while holding a block."""
        if preflight is None or not preflight.closed:
            return None
        table = getattr(self, "TABLE_TOP_Z", None)
        if table is None:
            return None
        z = float(target_pos[2])
        if abs(z - (table + self.RELEASE_Z_ABOVE_TABLE)) < 1e-4:
            return "release"
        if abs(z - (table + self.TRANSPORT_Z_ABOVE_TABLE)) < 1e-4:
            try:
                lateral = float(np.linalg.norm(np.asarray(arm.get_ee_pose()[:2], dtype=float)
                                               - np.asarray(target_pos[:2], dtype=float)))
            except Exception:
                return None
            return "carry" if lateral > 0.05 else None
        return None

    # A cube is symmetric under 90 deg yaw.  When the carry to the
    # slot, or the descent into it, has no checked path with the grasp yaw
    # (e.g. the 20 cm-long hand clips a human's laptop beside the slot), turn
    # the wrist and re-plan; later legs keep that yaw.  Nothing moved yet.
    BLOCK_CARRY_YAW_ALTERNATIVES = (np.pi / 2, -np.pi / 2, np.pi)

    def _carry_with_yaw_alternatives(self, preflight, target_pos, arm_tag, tcp_yaw, move=None,
                                     with_hold=True, half_turns=True):
        if move is None:
            move = lambda: super(BlocksRankingRGB, self)._move_screw(  # noqa: E731
                target_pos, arm_tag, tcp_yaw=tcp_yaw)
        arm = self.robot.get_arm(arm_tag)
        base_yaw = float(preflight.yaw)
        q0 = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
        # Every yaw without the aware hold first, then the grasp yaw with it.
        attempts = [(0.0, True)] + [(d, True) for d in self.BLOCK_CARRY_YAW_ALTERNATIVES
                                    if half_turns or abs(abs(float(d)) - np.pi) > 1e-6]
        if with_hold:
            attempts.append((0.0, False))
        last = None
        for delta, skip_hold in attempts:
            preflight.yaw = base_yaw + float(delta)
            self.plan_success = True
            if delta:
                print(f"[blocks-grasp] carry retry with wrist yaw {np.degrees(delta):+.0f} deg", flush=True)
            self._block_skip_hold_retry = skip_hold
            try:
                return move()
            except RuntimeError as exc:
                moved = float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=float) - q0)))
                if moved > 0.02 or getattr(self, "avatar_collided", False):
                    raise
                last = exc
            finally:
                self._block_skip_hold_retry = False
        preflight.yaw = base_yaw
        raise last

    # Full-state neutral expert only.  When the human works next to
    # the slot, the predictive gate may never clear the last centimetres of
    # the place descent; release from the lowest height it does clear.
    BLOCK_AWARE_RELEASE_RAISES_M = (0.0, 0.04, 0.08)

    def _aware_place_descent(self, target_pos, arm_tag, tcp_yaw):
        ok = True
        for raise_m in self.BLOCK_AWARE_RELEASE_RAISES_M:
            goal = np.asarray(target_pos, dtype=float).copy()
            goal[2] += float(raise_m)
            self._neutral_aware_optional = True
            self._neutral_aware_skipped = False
            # Short waits only: idling beside a human who never adapts invites
            # contact, while a release a few cm higher is harmless for a cube.
            self._neutral_aware_max_seg_override = 500
            try:
                ok = super(BlocksRankingRGB, self)._move_screw(goal, arm_tag, tcp_yaw=tcp_yaw)
            finally:
                self._neutral_aware_optional = False
                self._neutral_aware_max_seg_override = None
            if not self._neutral_aware_skipped:
                if raise_m:
                    print(f"[blocks-grasp] aware place descent stopped {raise_m:.2f} m high", flush=True)
                return ok
        print("[blocks-grasp] aware place descent skipped; releasing from transport height", flush=True)
        return True

    def _run_block_ranking_script(
        self,
        keys,
        *,
        boost_fingers: bool = True,
        include_settle: bool,
        wait_for_interaction: bool,
    ) -> bool:
        arm_tag = "right"
        if boost_fingers:
            self._boost_finger_pd(arm_tag)

        if include_settle:
            with self.suppress_recording():
                for _ in range(self.TABLE_SETTLE_STEPS):
                    self.step_sim()

        ok_all = True
        pending = list(keys)
        while pending:
            key = self._next_block_key(pending)
            pending.remove(key)
            ok_all = self._place_block_with_recovery(key, arm_tag) and ok_all

        if wait_for_interaction:
            controller = self.interaction_controller
            if isinstance(controller, BlockRankingAssistController):
                controller.wait_for_scripted_completion(self)
        if not ok_all:
            ok_all = self._retry_unplaced_blocks(keys, arm_tag)
        park = getattr(self, "_neutral_aware_park", None)
        if park is not None:
            park(arm_tag)
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_all

    def _prepare_block_ranking_script(self, arm_tag: str = "right") -> None:
        """Apply family control gains and the recording-excluded pre-settle."""

        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()

    def _place_block_with_recovery(
        self,
        key: str,
        arm_tag: str = "right",
    ) -> bool:
        """Execute one placement with the release-v1 broad recovery path."""

        try:
            ok = self._pick_and_place_block(key, arm_tag)
        except Exception as exc:
            print(f"[blocks-grasp] {key} rejected: {exc}", flush=True)
            ok = False
            if self._block_retry_allowed(arm_tag, key):
                ok = self._retry_block_after_retreat(key, arm_tag)
        if not ok:
            self.open_gripper(arm_tag)
            for _ in range(self.FAILURE_RECOVERY_STEPS):
                self.step_sim()
        return bool(ok)

    # One fresh attempt when an acquisition is rejected
    # before the fingers close (the block shifted after its preflight, or the
    # straight pre-close correction had no path).  The arm first retreats
    # straight up with open jaws; the retry re-runs the full preflight.
    BLOCK_PRE_CLOSE_RETRIES = 1
    BLOCK_OPTIONAL_CORRECTION_M = 0.012

    def _block_retry_allowed(self, arm_tag: str, key: str | None = None) -> bool:
        if int((self.config or {}).get("block_pre_close_retries", self.BLOCK_PRE_CLOSE_RETRIES)) <= 0:
            return False
        if getattr(self, "avatar_collided", False):
            return False
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        try:
            if float(robot.get_arm(arm_tag).gripper_val) >= 0.999:
                # Before the close: the commanded jaws are still open.
                return bool(getattr(self, "plan_success", True))
            # After the close but the lift was rejected before moving: the
            # block still rests on the table, so opening hands nothing over.
            if key is None:
                return False
            pos = np.asarray(to_numpy(self.blocks[key].get_pos()), dtype=float).ravel()[:3]
            return bool(pos[2] <= self.TABLE_TOP_Z + self.block_half_size(key) + 0.01)
        except Exception:
            return False

    def _retry_block_after_retreat(self, key: str, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        if float(arm.gripper_val) < 0.999:
            self.open_gripper(arm_tag)
            for _ in range(60):
                self.step_sim()
        # A planning-only rejection (no motion) may have cleared the flag.
        self.plan_success = True
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        ee = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        if ee[2] < transport_z - 0.02:
            try:
                self._move_screw(np.array([ee[0], ee[1], transport_z]), arm_tag)
            except Exception as exc:
                print(f"[blocks-grasp] {key} retry retreat failed: {exc}", flush=True)
                return False
            for _ in range(60):
                self.step_sim()
        if not getattr(self, "plan_success", True) or getattr(self, "avatar_collided", False):
            return False
        lifted = float(np.asarray(arm.get_ee_pose()[:3], dtype=float)[2])
        if lifted < transport_z - 0.05:
            print(f"[blocks-grasp] {key} retry: retreat reached z={lifted:.3f} only", flush=True)
            return False
        self._block_retries = int(getattr(self, "_block_retries", 0)) + 1
        print(f"[blocks-grasp] {key} retry after retreat (z={lifted:.3f})", flush=True)
        try:
            return bool(self._pick_and_place_block(key, arm_tag))
        except Exception as exc:
            print(f"[blocks-grasp] {key} retry rejected: {exc}", flush=True)
            return False

    # One more pass over this rollout's blocks that are still not in
    # their slot (a block whose approach had no checked path, e.g. because a
    # neighbour or the human was in the way at that moment).  Each retry is
    # the ordinary retreat + full preflight; nothing is retried after a
    # human contact, for a block someone holds, or for one off the table.
    BLOCK_FINAL_RETRY_PASS = True

    def _block_needs_retry(self, key: str) -> bool:
        cube = self.blocks[key]
        pos = np.asarray(to_numpy(cube.get_pos()), dtype=float).ravel()[:3]
        half = float(self.block_half_size(key))
        if not np.all(np.isfinite(pos)):
            return False
        on_table = abs(float(pos[2]) - (self.TABLE_TOP_Z + half)) <= 0.01
        in_slot = float(np.linalg.norm(pos[:2] - self.target_slots[key][:2])) <= self.SUCCESS_DIST_XY
        if in_slot or not on_table:
            return False
        try:
            if held_by_anyone(self, cube, pos):
                return False
        except Exception:
            return False
        return True

    def _retry_unplaced_blocks(self, keys, arm_tag: str = "right") -> bool:
        if not bool((self.config or {}).get("block_final_retry_pass", self.BLOCK_FINAL_RETRY_PASS)):
            return False
        pending = [key for key in keys if self._block_needs_retry(key)]
        if not pending or getattr(self, "avatar_collided", False):
            return False
        ok_all = True
        for key in pending:
            if getattr(self, "avatar_collided", False):
                return False
            print(f"[blocks-grasp] final pass: retrying {key}", flush=True)
            ok = self._retry_block_after_retreat(key, arm_tag)
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()
            ok_all = bool(ok) and ok_all
        return ok_all

    def _move_above_block_for_interrupt(
        self,
        key: str,
        arm_tag: str = "right",
    ) -> None:
        """Show intent above the current live block pose before yielding."""

        block_pos = self.block_center(key).astype(float)
        arm = self.robot.get_arm(arm_tag)
        self.open_gripper(arm_tag)
        intent_pos = np.array(
            [
                block_pos[0],
                block_pos[1],
                self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE,
            ]
        )
        above_link = tcp_to_link_pose(
            self._top_down_tcp(intent_pos),
            arm.tcp_offset,
        )
        self._move_seeded(above_link.to_pose7(), arm_tag)

    def play_once(self) -> bool:
        """Run the robot-only family expert.

        Keeping this method unconditionally robot-only preserves the public
        ``expert`` evaluator's parent-method fallback for assist scenarios.
        The assist task's thin public class delegates to the composed path.
        """

        return self._run_block_ranking_script(
            self.block_keys(),
            include_settle=True,
            wait_for_interaction=False,
        )

    def play_assist_once(self) -> bool:
        """Run the shared avatar/robot work-allocation choreography."""

        controller = self.interaction_controller
        if not isinstance(controller, BlockRankingAssistController):
            raise TypeError("play_assist_once requires block-ranking assist")

        # Preserve the legacy direct-construction edge: assist variants
        # without an avatar performed their own settle and then delegated to
        # the base rollout, which settled a second time.
        self._boost_finger_pd("right")
        with self.suppress_recording():
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()
        if self.avatar is None:
            return self._run_block_ranking_script(
                self.block_keys(),
                boost_fingers=True,
                include_settle=True,
                wait_for_interaction=False,
            )
        keys = controller.scripted_robot_keys(self)
        self._assist_yield_to_human()
        return self._run_block_ranking_script(
            keys,
            boost_fingers=False,
            include_settle=False,
            wait_for_interaction=True,
        )

    # Full-state assist expert only (the blind expert has its own
    # rollout): the scripted human fetches its block and carries it across
    # the slot row, through the space the robot's first approach sweeps.
    # The human never adapts, so the robot holds still until the human's
    # assist motion is over (bounded), then places its own blocks.
    ASSIST_YIELD_TO_HUMAN = True
    ASSIST_YIELD_MAX_STEPS = 3000

    # Wait at the interrupt variants' retreat pose (base + 0.25 m, 0.30 m above
    # the table), not at home: the human's reach can pass the home hand.
    ASSIST_YIELD_RETREAT_Y_FROM_BASE = 0.25
    ASSIST_YIELD_RETREAT_Z_ABOVE_TABLE = 0.30

    def _assist_yield_to_human(self) -> None:
        avatar = getattr(self, "avatar", None)
        if not self.ASSIST_YIELD_TO_HUMAN or not callable(getattr(avatar, "spare", None)):
            return
        if not avatar.spare():
            try:
                arm = self.robot.get_arm("right")
                base = np.asarray(arm.origin_pose.p, dtype=float)
                xy = base[:2] + self._layout_xy([0.0, self.ASSIST_YIELD_RETREAT_Y_FROM_BASE])
                goal = np.array([xy[0], xy[1], self.TABLE_TOP_Z + self.ASSIST_YIELD_RETREAT_Z_ABOVE_TABLE])
                self._block_straight_move(goal, "right")
            except Exception as exc:
                print(f"[blocks-assist] yield retreat skipped: {exc}", flush=True)
        waited = 0
        while not avatar.spare() and waited < self.ASSIST_YIELD_MAX_STEPS:
            self.step_sim()
            waited += 1
        print(f"[blocks-assist] robot yielded {waited} steps until the human finished", flush=True)

    def play_blind_assist_once(self) -> bool:
        """Blind (no-intent) assist expert: never reads which block the human
        takes.  At t=0 it commits to the robot's share size (all blocks but
        one) by a human-independent rule (the blocks nearest the gripper),
        places them in the family order, then waits for the human and
        settles.  No retarget, yield or leftovers pass."""

        controller = self.interaction_controller
        if not isinstance(controller, BlockRankingAssistController):
            raise TypeError("play_blind_assist_once requires block-ranking assist")
        keys = (
            controller.choose_blind_robot_keys(self)
            if self.avatar is not None else tuple(self.block_keys())
        )
        self._boost_finger_pd("right")
        with self.suppress_recording():
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()
        return self._run_block_ranking_script(
            keys,
            boost_fingers=False,
            include_settle=False,
            wait_for_interaction=self.avatar is not None,
        )

    def play_interrupt_once(self) -> bool:
        """Run shared inspect/yield choreography for an interrupt variant."""

        controller = self.interaction_controller
        if not isinstance(controller, BlockRankingInterruptController):
            raise TypeError("play_interrupt_once requires block-ranking interrupt")
        return controller.run_scripted(self)

    def _block_resting(self, cube, pos, half: float) -> bool:
        """Success guard for one block: resting on the table (centre at most
        ``SUCCESS_REST_MAX_ABOVE_TABLE_M`` above table + ``half``) and held by
        neither the avatar nor the gripper."""
        if float(pos[2]) > self.TABLE_TOP_Z + float(half) + self.SUCCESS_REST_MAX_ABOVE_TABLE_M:
            return False
        return not held_by_anyone(self, cube, pos)

    def check_success(self) -> bool:
        positions = {}
        slot_ok = True
        on_table_ok = True

        for color_name, cube in self.blocks.items():
            p = to_numpy(cube.get_pos()).ravel()[:3]
            positions[color_name] = p
            target = self.target_slots[color_name]
            dist_xy = float(np.linalg.norm(p[:2] - target[:2]))
            in_slot = dist_xy <= self.SUCCESS_DIST_XY
            on_table = (
                p[2] >= self.TABLE_TOP_Z + self.CUBE_HALF * self.ON_TABLE_HEIGHT_FRACTION
                and self._block_resting(cube, p, self.CUBE_HALF)
            )
            slot_ok = slot_ok and in_slot
            on_table_ok = on_table_ok and on_table

        order_ok = (
            self._order_axis_value(positions["red"]) <
            self._order_axis_value(positions["green"]) <
            self._order_axis_value(positions["blue"])
        )
        success = bool(self.plan_success and slot_ok and on_table_ok and order_ok)
        return success

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        positions = {}
        dist_xy = {}
        slot_ok = {}
        on_table_ok = {}
        resting_ok = {}

        for color_name, cube in self.blocks.items():
            p = to_numpy(cube.get_pos()).ravel()[:3]
            target = self.target_slots[color_name]
            dxy = float(np.linalg.norm(p[:2] - target[:2]))
            positions[color_name] = p.tolist()
            dist_xy[color_name] = dxy
            slot_ok[color_name] = bool(dxy <= self.SUCCESS_DIST_XY)
            on_table_ok[color_name] = bool(
                p[2] >= self.TABLE_TOP_Z + self.CUBE_HALF * self.ON_TABLE_HEIGHT_FRACTION
            )
            resting_ok[color_name] = bool(self._block_resting(cube, p, self.CUBE_HALF))

        order_ok = bool(
            self._order_axis_value(positions["red"]) <
            self._order_axis_value(positions["green"]) <
            self._order_axis_value(positions["blue"])
        )
        metrics.update({
            "block_positions": positions,
            "block_dist_xy": dist_xy,
            "block_slot_ok": slot_ok,
            "block_on_table_ok": on_table_ok,
            "block_all_slots_ok": bool(all(slot_ok.values())),
            "block_all_on_table_ok": bool(all(on_table_ok.values())),
            "block_order_ok": order_ok,
            "block_success_dist_xy": self.SUCCESS_DIST_XY,
            "block_resting_ok": resting_ok,
        })
        return metrics
