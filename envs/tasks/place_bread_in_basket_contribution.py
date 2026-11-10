"""Place bread in basket (Contribution, ``place_bread_in_basket_contribution``):
variant of ``place_bread_in_basket`` where a human avatar and the robot share one pool
of two loaves in one region beside the robot; the human takes one (drawn
uniformly at reset) and the robot places the other.

Success: both loaves in the basket AND no robot-avatar contact.

The robot objective stays here (pure-physics grasp + place, no kinematic
snap of held objects).  ``play_once`` (full-state expert) reads the human's
choice and takes the other loaf concurrently; ``play_blind_once`` (blind
expert) picks one loaf by a human-independent rule at t=0 and commits, so
about half the episodes contest the same loaf.  The human side -- which loaf
the avatar takes, its single stance, its ``AvatarController.pick_and_place``
motion (``attach_obj`` is the sanctioned avatar mechanism) and the eval-mode
trigger -- is owned by the composed :class:`PlaceBreadInBasketAssistController`.
"""

import contextlib

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..interaction.place_bread_in_basket import (
    PlaceBreadInBasketAssistController,
    bread_world_center,
)
from ..utils import to_numpy
from ..task_bases.place_bread_in_basket import (
    PlaceBreadInBasket,
)


# The default avatar y=0.55 puts its
# collision capsule slightly through the +y edge of the table.  Slide
# avatar / robot / object spawn region together by +Y_SHIFT so the
# avatar clears the table; the table itself is unchanged.  Objects stay
# 0.05 m inside the table's +y edge (y=+0.20).
_Y_SHIFT = 0.30
_TABLE_Y_MAX = +0.20
_OBJ_TABLE_MARGIN = 0.05
# Shared-pool layout (assist 50-50).  At its home pose the Franka's hand,
# fingers and links 5-7 hover over x in [-0.13, 0.13], y in [-0.13, 0.13]
# (z 0.83-1.27), i.e. right above the legacy centred region, so a human
# reaching there sweeps the parked robot whatever policy runs.  The pool sits
# to the robot's +x side instead, laid out loaf | basket | loaf along x so the
# human's reach and carry (to the middle) stay on its own side and never pass
# over the other loaf while the robot works on it concurrently.  Accepted
# layouts have
#   * every object >= 0.22 m (xy) from the parked hand,
#   * every object 0.38-0.60 m from the robot base (Franka top-down band;
#     near-base dead zone < 0.35 m),
#   * loaves at y <= 0.02 (robot pre-grasps clear of the human's body/arms in
#     the planner cloud) and >= 0.15 m from the basket along x,
#   * each loaf + the basket drop reachable (<= 0.63 m) from one shoulder at
#     body y 0.38, so the stance chosen at reset for the human's loaf never
#     changes during the episode.
# The sampler keeps 2 draws per object in the legacy order (loaf index, then
# basket, then each loaf as (x offset, y)); ~0.5% of attempts are accepted.
# Basket at y <= 0.0 and objects <= 0.60 m from the base: with the basket
# at y >= 0.03 the robot's above-basket pose was IK-blocked by the human
# standing in front of it, and loaves at 0.60-0.62 m slipped on lift.
# Hard mode (3 loaves) does not fit and takes a best-effort fallback.
_ASSIST_SAMPLE_ATTEMPTS = 3000
_ROBOT_BASE_XY = (0.0, -0.65 + _Y_SHIFT)
_ROBOT_HOME_HAND_XY = (0.0, 0.08)
_ROBOT_HOME_KEEPOUT_M = 0.22
_POOL_LAYOUT = {
    "basket_x": (0.33, 0.40),
    "basket_y": (-0.06, 0.0),
    "offset": (0.15, 0.20),
    "loaf_y": (-0.14, 0.02),
}
_POOL_REGION_X = (0.13, 0.60)   # bounding box of the structured layout
_POOL_REGION_Y = (-0.14, 0.08)
_POOL_LOAF_Y_MAX = 0.02
_ROBOT_REACH_ANNULUS = (0.38, 0.60)
_POOL_MIN_SEP_M = 0.15
_POOL_STANCE_Y = 0.38
_HUMAN_REACH = {
    "shoulder_z": 1.20,  # measured 1.18-1.21 across the avatar pool
    "radius": 0.63,
    "basket_z": 0.91,    # rim + drop height + palm-down attach offset
    "loaf_z": 0.80,
}
# Deterministic valid layout used when rejection sampling fails (basket
# first; the 4th entry is hard mode only, best effort).
_ASSIST_FALLBACK_POSITIONS = (
    (0.37, -0.01),
    (0.21, -0.02),
    (0.53, -0.10),
    (0.37, -0.12),
)


class PlaceBreadInBasketContribution(PlaceBreadInBasket):
    """Avatar places one bread while the robot places the rest."""

    INSTRUCTION = "put one bread into the basket while the human puts the other bread into the basket"
    # Success = both loaves in the basket AND no robot-avatar contact.
    VLA_ALLOW_AVATAR_COLLISION = False
    use_avatar = True
    EASY_NUM_OBJECTS = 2
    HARD_NUM_OBJECTS = 3

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Mesh-+X (avatar forward) maps to world-(-Y).  Same constants as
    # the interrupt sibling.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55 + _Y_SHIFT, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # The avatar's bread can spawn anywhere in the parent's full table
    # region — `AvatarController.pick_and_place(natural=True)` will
    # auto-position the avatar body per-episode so FABRIK reach covers
    # the chosen pick + place midpoint (see _NATURAL_BODY_* below).

    # Avatar pick-and-place pacing.  Defaults from the design doc /
    # categorize_cooperative — 320/400/320 frames at 0.001s dt is ~1s
    # of motion which gives the FABRIK chain time to interpolate
    # smoothly without skin pop.
    AVATAR_APPROACH_FRAMES = 320
    AVATAR_TRANSPORT_FRAMES = 400
    AVATAR_RETRACT_FRAMES = 320

    # Drop the bread this far above the basket rim before release.  The
    # parent now uses the small model-2 loaf (graspable); like the robot's
    # own placement, releasing it high lets it bounce back out over the rim
    # (robot sweep: +0.06 -> 12.8 cm out, +0.02 -> 2.5 cm in).  Seat the
    # avatar's loaf low too — a touch above the robot's 0.02 to absorb
    # FABRIK reach error.
    AVATAR_DROP_ABOVE_RIM = 0.03

    # Shared pool (assist 50-50): the human starts as soon as it is triggered
    # -- no extra hold that would let the robot lead.  The delay hook stays
    # for callers that pass add_start_delay=True; it is a no-op at 0.
    AVATAR_START_DELAY_STEPS = 0

    # `AvatarController.pick_and_place(natural=True)` knobs.  The
    # controller stands the avatar `body_margin` m behind the pick/place
    # midpoint (along its forward axis), yawing up to `yaw_limit_deg`,
    # and clips body xy into the optional bounds.  body_y_bounds clamps
    # the avatar y to (avatar's MIN_Y, init y) — keeps the body from
    # being pushed further onto/under the table than is physically
    # meaningful.  body_x_bounds left unbounded so it follows the
    # target laterally.
    AVATAR_NATURAL_BODY_MARGIN = 0.40
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (-0.20 + _Y_SHIFT, 0.55 + _Y_SHIFT)
    AVATAR_NATURAL_SETTLE_STEPS = 20
    AVATAR_FINISH_SETTLE_STEPS = 120
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    # Shared-pool assist: success requires both loaves in the basket AND no
    # robot-avatar contact, so collision tracking is on by default.
    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        cfg.setdefault("success_require_no_avatar_collision", True)
        super().__init__(cfg)

    def create_interaction_controller(self) -> PlaceBreadInBasketAssistController:
        return PlaceBreadInBasketAssistController(
            region_x=_POOL_REGION_X,
            region_y=_POOL_REGION_Y,
            min_separation=_POOL_MIN_SEP_M,
            sample_attempts=_ASSIST_SAMPLE_ATTEMPTS,
            fallback_positions=_ASSIST_FALLBACK_POSITIONS,
            start_delay_steps=self.AVATAR_START_DELAY_STEPS,
            robot_base_xy=_ROBOT_BASE_XY,
            robot_reach_annulus=_ROBOT_REACH_ANNULUS,
            pool_stance_y=_POOL_STANCE_Y,
            human_reach=_HUMAN_REACH,
            loaf_y_max=_POOL_LOAF_Y_MAX,
            pool_layout=_POOL_LAYOUT,
            keepout_xy=_ROBOT_HOME_HAND_XY,
            keepout_radius=_ROBOT_HOME_KEEPOUT_M,
        )

    def _assist_controller(self) -> PlaceBreadInBasketAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, PlaceBreadInBasketAssistController):
            raise TypeError("task does not use bread-basket assist behavior")
        return controller

    # Robot base shifted by +_Y_SHIFT in the avatar direction so the
    # avatar/robot/objects keep the same relative geometry while the
    # table stays put — see _Y_SHIFT comment.
    def _load_robot(self):
        if self.config.get("robot_type", "franka") == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            # Parent default is (0.0, -0.65, 0.75); shift y by +_Y_SHIFT so the
            # robot keeps the same relative geometry to the +Y-shifted avatar /
            # objects.  This (0.0, -0.35, 0.75) mount matches the other avatar
            # tasks (categorize_cooperative, deliver_to_human_easy).
            kwargs.setdefault("pos", [0.0, -0.65 + _Y_SHIFT, 0.75])
        super()._load_robot()
        # Use the DEFAULT Franka home pose (no override) so this task's initial
        # robot pose matches every other task / avatar variant.

    # ------------------------------------------------------------------
    # Sampling — the controller draws which bread the avatar picks (the
    # robot gets all remaining breads) and the layout, in the legacy order.
    # ------------------------------------------------------------------

    def _sample_positions(self, rng):
        return self._assist_controller().plan_positions(self, rng)

    # ------------------------------------------------------------------
    # Legacy private helpers, delegated to the controller
    # ------------------------------------------------------------------

    def _bread_world_center(self, bread):
        return bread_world_center(self, bread)

    def _delay_avatar_motion_start(self, delay_steps: int) -> None:
        self._assist_controller().delay_avatar_motion(self, delay_steps)

    def _start_avatar_assist(self, add_start_delay: bool = True) -> None:
        self._assist_controller().start_avatar_assist(
            self, add_start_delay=add_start_delay,
        )

    def _eval_at_reset(self) -> None:
        self._assist_controller().eval_reset(self)

    def _eval_at_step(self, step_idx: int) -> None:
        self._assist_controller().eval_step(self, step_idx)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        """Full-state expert: knows which loaf the human took and places the
        other loaf (loaves), planning around the live avatar body.  Its only
        coordination is privileged timing: it starts once the human's
        scheduled motion frames are past (read from the schedule)."""
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.breads or self.basket is None:
            return False
        arm_tag = "right"

        if self.avatar is None:
            return super().play_once()

        controller = self._assist_controller()
        # Scripted (non-eval) rollouts start the human here; eval mode has
        # already started it at reset.
        if not getattr(self, "_eval_assist_fired", False):
            self._start_avatar_assist(add_start_delay=False)

        # Privileged timing (full-state only): the human's pick-and-place is
        # a fixed frame schedule, so hold the arm until its scheduled
        # carry/drop/retract frames are past -- read from the schedule, not a
        # reactive proximity wait.
        for _ in range(self._human_frames_remaining()):
            self.step_sim()

        ok_robot = True
        with self._avatar_in_planner(arm_tag), self._basket_replan():
            for robot_idx in controller.scripted_robot_indices(self):
                controller.note_robot_target(robot_idx)
                try:
                    ok_robot = self._pick_and_place_bread(
                        self.breads[robot_idx], arm_tag, chr(ord("A") + robot_idx),
                    ) and ok_robot
                except Exception:
                    ok_robot = False

        self._wait_for_avatar_spare()
        for _ in range(self.AVATAR_FINISH_SETTLE_STEPS):
            self.step_sim()
        return ok_robot

    # Avatar capsules enter the planner cloud grown by an absolute margin
    # (a 1.8x factor grew the torso by 11 cm, swallowing loaves and the
    # basket at the human's table edge), sampled at this voxel resolution.
    _AVATAR_PLANNER_MARGIN_M = 0.04
    _BLIND_AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_SPARE_MAX_STEPS = 4000

    @contextlib.contextmanager
    def _avatar_in_planner(self, arm_tag: str):
        """Refresh the live avatar body into the planner cloud before every
        RRT plan.  This task registers no static cloud, so replacing the
        cloud on each refresh drops nothing."""
        planner = self.robot.get_arm(arm_tag).planner
        plan_path = planner.plan_path

        def plan_path_with_avatar(*args, **kwargs):
            pts = None
            if self.avatar_collider is not None:
                capsules = [
                    (name, pa, pb, float(r) + self._AVATAR_PLANNER_MARGIN_M)
                    for name, pa, pb, r in self.avatar_collider.current_capsules()
                ]
                pts = sample_capsule_planner_points(capsules, inflate_factor=1.0)
            if pts is not None and pts.size:
                planner.update_obstacles(pts, resolution=self._BLIND_AVATAR_OBSTACLE_RES)
            return plan_path(*args, **kwargs)

        planner.plan_path = plan_path_with_avatar
        try:
            yield
        finally:
            del planner.plan_path

    def _human_frames_remaining(self) -> int:
        """Frames left in the human's scheduled pick-and-place motion."""
        motion = getattr(getattr(self, "avatar", None), "motion_modules", {}).get(
            "pick_and_place"
        )
        frames = getattr(motion, "frames", None)
        if not frames:
            return 0
        return max(0, len(frames) - int(getattr(motion, "at_frame", 0)))

    # Above-basket moves only: a planner-cloud transient (the human's arm
    # near the shared basket) may make the plan briefly infeasible; hold
    # position and re-plan every 20 steps for up to 300 steps.
    _BASKET_REPLAN_EVERY = 20
    _BASKET_REPLAN_MAX_STEPS = 300
    _basket_replan_active = False

    @contextlib.contextmanager
    def _basket_replan(self):
        prev = self.__dict__.get("_basket_replan_active")
        self._basket_replan_active = True
        try:
            yield
        finally:
            if prev is None:
                self.__dict__.pop("_basket_replan_active", None)
            else:
                self._basket_replan_active = prev

    def _targets_basket(self, pose) -> bool:
        if pose is None or getattr(self, "basket", None) is None:
            return False
        xy = np.asarray(pose, dtype=np.float64).ravel()[:2]
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        return float(np.linalg.norm(xy - basket_xy)) <= 0.05

    def move_to_pose(self, pose, arm_tag: str):
        was_ok = bool(self.plan_success)
        result = super().move_to_pose(pose, arm_tag)
        if (result is not None or not was_ok or not self._basket_replan_active
                or not self._targets_basket(pose)):
            return result
        waited = 0
        while waited < self._BASKET_REPLAN_MAX_STEPS:
            for _ in range(self._BASKET_REPLAN_EVERY):
                self.step_sim()
            waited += self._BASKET_REPLAN_EVERY
            self.plan_success = True
            result = super().move_to_pose(pose, arm_tag)
            if result is not None:
                print(f"[bread_assist] above-basket move re-planned after {waited} steps",
                      flush=True)
                return result
        return None  # plan_success stays False: an honest failure

    def _wait_for_avatar_spare(self, max_steps=None) -> None:
        if self.avatar is None:
            return
        budget = self._AVATAR_SPARE_MAX_STEPS if max_steps is None else int(max_steps)
        while not self.avatar.spare() and budget > 0:
            self.step_sim()
            budget -= 1

    def _blind_pick_rule(self) -> int:
        """Human-independent rule fixed at t=0: the loaf nearest the robot's
        gripper (xy), ties to the lower index."""
        tcp = np.asarray(self.robot.get_ee_pose("right"), dtype=np.float64)[:2]
        return min(range(len(self.breads)), key=lambda i: (round(float(np.linalg.norm(
            to_numpy(self.breads[i].entity.get_pos()).ravel()[:2] - tcp,
        )), 4), i))

    def play_blind_once(self) -> bool:
        """Expert baseline without the human's intent or any allocation.

        Picks ONE loaf by a human-independent rule at t=0 and commits: no
        retarget, yield, pause or leftovers pass.  It plans with the live
        avatar body in the planner cloud (repo rule); a failed plan/grasp is
        an honest failure.  Then waits (bounded) for the human and settles.
        """
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if not self.breads or self.basket is None:
            return False
        if self.avatar is None:
            return super().play_once()

        arm_tag = "right"
        controller = self._assist_controller()
        target = self._blind_pick_rule()
        controller.note_robot_target(target)
        with self._avatar_in_planner(arm_tag), self._basket_replan():
            try:
                ok = self._pick_and_place_bread(
                    self.breads[target], arm_tag, chr(ord("A") + target),
                )
            except Exception:
                ok = False
        self._wait_for_avatar_spare()
        for _ in range(self.AVATAR_FINISH_SETTLE_STEPS):
            self.step_sim()
        return bool(ok)

    # ------------------------------------------------------------------
    # Success — parent's "all breads in basket" + optional collision gate
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        # Tracking and rejection are deliberately separate.  Collection turns
        # tracking on for telemetry, while this task's known capsule overlap
        # must not override the geometric all-breads-in-basket verdict.
        if (self.config.get("success_require_no_avatar_collision", False)
                and self.avatar_collision_checker is not None
                and self.avatar_collided):
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        all_breads_in_basket = True
        for idx, bread in enumerate(self.breads):
            pos = to_numpy(bread.entity.get_pos()).ravel()[:3]
            dist_xy = float(np.linalg.norm(pos[:2] - basket_xy))
            above_table = bool(
                pos[2] > self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
            )
            in_basket = bool(
                dist_xy <= self.SUCCESS_DIST_XY_M and above_table
            )
            metrics[f"bread_{idx}_pos"] = pos.tolist()
            metrics[f"bread_{idx}_dist_xy"] = dist_xy
            metrics[f"bread_{idx}_above_table"] = above_table
            metrics[f"bread_{idx}_in_basket"] = in_basket
            all_breads_in_basket = all_breads_in_basket and in_basket
        metrics["all_breads_in_basket"] = bool(all_breads_in_basket)
        metrics.update(self.interaction_controller.interaction_metrics(self))
        # BaseTask.evaluate() conservatively clears success whenever collision
        # telemetry contains a hit.  Restore this task's explicit success
        # contract, which already honors success_require_no_avatar_collision.
        metrics["success"] = bool(self.check_success())
        return metrics
