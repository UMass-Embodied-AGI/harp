"""Stack three bowls (Contribution, ``stack_three_bowls_contribution``): variant
of ``stack_bowls_three``.

Same scene + grasp pipeline as the parent ``StackBowlsThree``.  Bowl 0
stays put as the base.  Every other bowl is a SHARED candidate in one band
both the avatar and the robot reach (no human side / robot side split).
The human picks ONE candidate uniformly at random and goes for it; the robot
stacks the rest.  Nothing resolves a conflict: if both go for the same bowl,
the episode honestly fails (collision, empty grasp, or a bowl missing from
the stack).

The avatar uses the sanctioned ``attach_obj`` mechanism via
``AvatarController.pick_and_place``.  The robot uses the parent's
pure-physics rim grasp + lift + place primitive (``_pick_and_stack_one``).

Experts:

* ``play_once`` (full state): reads the human's bowl and stacks the others,
  concurrently with the human.
* ``play_blind_once`` (blind): commits at t=0 to the candidate nearest the
  gripper (independent of the human), stacks it, waits for the avatar to
  finish, settles.

Success: every candidate resting nested on the base (parent footprint +
higher-z check; a bowl still pinched in the closed gripper does not count)
and no robot-avatar contact.
"""

import numpy as np
import transforms3d as t3d

from ..interaction.stack_bowls_three import StackBowlsThreeAssistController
from ..utils import Pose, load_object
from ..task_bases.stack_bowls_motion import _BowlMotionRejected
from ..task_bases.stack_bowls_three import (
    StackBowlsThree,
    _BOWL_ID, _BOWL_MODEL_ID, _NUM_BOWLS,
    _MIN_SEP_M, _REGION_X, _REGION_Y,
    _read_extents, _read_scale, _quat_mul, load_fixed_collision_bowl,
)


class StackThreeBowlsContribution(StackBowlsThree):
    """Human and robot stack bowls from one shared candidate pool."""

    BOWL_FIXED_COLLISION = True
    BOWL_WRIST_FLIP = True

    INSTRUCTION = "stack one bowl on the base bowl while the human stacks another bowl on the base"
    use_avatar = True
    EASY_NUM_OBJECTS = _NUM_BOWLS
    HARD_NUM_OBJECTS = _NUM_BOWLS + 1

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Mesh-+X (avatar forward) maps to world-(-Y).  Same convention as
    # place_bread_in_basket_contribution.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # Layout.  The Franka is mounted at (0, -0.65) and its home TCP rests
    # low over the table at about (0, -0.19, 0.84).  Rim grasps are reliable
    # up to ~0.57 m from the mount (bowls 0.64 m out failed the measured rim
    # alignment 4/5 times; a carry to a base at y=0 failed the screw
    # plan), and a human reach that ends ~0.13 m from the home TCP clipped
    # the idle robot hand.  Every candidate is therefore sampled in one band
    # both agents reach, inside the robot reach and clear of the home hand;
    # the base (goal) sits between them, nearer the human.
    ROBOT_MOUNT_XY = (0.0, -0.65)
    ROBOT_HOME_TCP_XY = (0.0, -0.19)
    ROBOT_REACH_M = 0.58
    CANDIDATE_HOME_CLEARANCE_M = 0.18
    # Three candidates cannot all clear the home hand at 16 cm separation;
    # hard mode (not in the eval rows) keeps only the reach limit.
    HARD_CANDIDATE_HOME_CLEARANCE_M = 0.0
    BASE_X = (-0.06, 0.06)
    BASE_Y = (-0.06, -0.02)
    SHARED_X = (-0.24, 0.24)
    SHARED_Y = (-0.16, -0.06)
    HARD_SHARED_X = (-0.20, 0.20)
    HARD_SHARED_Y = (-0.20, -0.08)
    ASSIST_SAMPLE_ATTEMPTS = 4096
    # Default (simplified, used by the eval rows) layout and sampling
    # fallback: base in the middle, one candidate on each side, mirror
    # images of each other in the shared band.
    # Candidates at x = +-0.20.  +-0.18 lets a do-nothing robot come within
    # 1-3 cm of the human's fingers; +-0.22 clears it by >= 5 cm but breaks
    # the scripted expert's far-bowl "tiny lift" (seed 9000000).  +-0.20 keeps ~3.9 cm
    # human clearance with the pick lift and drop offset below.
    SHARED_FIXED_XY = (
        (0.0, -0.04),
        (0.20, -0.12),
        (-0.20, -0.12),
    )
    HARD_FALLBACK_XY = (
        (0.0, -0.04),
        (0.18, -0.12),
        (-0.18, -0.12),
        (0.0, -0.21),
    )
    # Legacy avatar-band names (kept for external readers).
    AVATAR_X_LO, AVATAR_X_HI = -0.18, 0.18
    AVATAR_Y_LO, AVATAR_Y_HI = -0.10, 0.04

    # Avatar pick-and-place pacing.
    AVATAR_APPROACH_FRAMES = 80
    AVATAR_TRANSPORT_FRAMES = 120
    AVATAR_RETRACT_FRAMES = 80

    # Palm aim above the bowl's mesh origin (= bowl bottom) at pick time
    # (0.08 before the idle-robot clearance fix).
    AVATAR_PICK_LIFT = 0.12
    # The human's drop palm sits this far (xy) from the base centre, on the
    # human's side (+y), and AVATAR_DROP_CLEARANCE_M higher (0.10, was
    # 0.06): with the drop right over the base centre the fingers of a
    # palm-down drop reached within ~3 cm of the parked robot's wrist.  The
    # bowl still nests on the base (~1 cm off centre; a 5 cm offset left it
    # far enough off centre that the robot's bowl did not nest).
    AVATAR_DROP_XY_OFFSET = (0.0, 0.025)

    # Human drop: a couple of bowl heights above the table plus clearance;
    # gravity nests the bowl on whatever is stacked by then.
    AVATAR_DROP_BOWLS_ABOVE = 2.0   # in units of bowl_height
    AVATAR_DROP_CLEARANCE_M = 0.10

    # The parent table is centered at y=-0.35 and spans up to y=+0.20.
    AVATAR_TABLE_Y_MAX = 0.20
    AVATAR_MIN_BODY_Y = AVATAR_TABLE_Y_MAX + 0.18
    AVATAR_MAX_BODY_Y = 0.62
    AVATAR_BODY_MARGIN = 0.32
    AVATAR_BODY_X_BOUNDS = (-0.22, 0.22)
    AVATAR_YAW_LIMIT_DEG = 30.0
    AVATAR_FINISH_WAIT_STEPS = 1500
    # Success: a bowl whose centre is this close to the TCP while the gripper
    # is closed below this value is still held, not stacked.
    HELD_TCP_RADIUS_M = 0.09
    HELD_GRIPPER_BELOW = 0.5
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)

    def create_interaction_controller(self) -> StackBowlsThreeAssistController:
        return StackBowlsThreeAssistController(
            region_x=_REGION_X, region_y=_REGION_Y, min_separation=_MIN_SEP_M,
        )

    def _assist_controller(self) -> StackBowlsThreeAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, StackBowlsThreeAssistController):
            raise TypeError("task does not use stack-bowls assist behavior")
        return controller

    # ------------------------------------------------------------------
    # Layout — base and every candidate in one shared band.
    # ------------------------------------------------------------------

    def _sample_assist_xy(self, rng):
        return self._assist_controller().sample_assist_xy(self, rng)

    def load_actors(self):
        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        # Child RNG: draws below change only this task's layout, never the
        # global stream other components read after load_actors.
        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_assist_xy(rng)

        self._bowl_scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
        bowl_extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        self._bowl_extents = bowl_extents
        self._bowl_height = float(bowl_extents[1]) * self._bowl_scale

        yaws = [
            float(rng.uniform(-self.BOWL_YAW_LIMIT, self.BOWL_YAW_LIMIT))
            for _ in range(self._num_bowls)
        ]
        spawn_z = table_top + self.BOWL_SPAWN_Z_CLEARANCE
        self.bowls = []
        self._spawn_xy = []
        for i, ((cx, cy), yaw) in enumerate(zip(positions, yaws)):
            yaw_q = np.asarray(
                t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, yaw, "sxyz")),
                dtype=np.float64,
            )
            full_q = _quat_mul(yaw_q, upright_q)
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

        # Bowl 0 = base; the rest form the shared pool.  The human's uniform
        # random pick comes from the same child RNG.
        controller = self._assist_controller()
        controller.allocate(self)
        controller.draw_human_choice(self, rng)

    def _start_avatar_assist(self) -> bool:
        return self._assist_controller().start_avatar_assist(self)

    # ------------------------------------------------------------------
    # Robot primitives
    # ------------------------------------------------------------------

    # A long screw carry from the left candidate across the table is
    # sometimes rejected while its halves are not (a kinematic fallback, no
    # waiting): retry the same carry as 2 then 3 straight segments.
    _REPLANNABLE = ("native screw rejected", "path, collision or FK rejected")
    CARRY_SEGMENTS = (1, 2, 3)

    def _bowl_move(self, target, arm_tag, label, *, tiny=False, **kwargs):
        if label != "carry":
            return super()._bowl_move(target, arm_tag, label, tiny=tiny, **kwargs)
        for attempt, segments in enumerate(self.CARRY_SEGMENTS):
            plan_before = getattr(self, "plan_success", True)
            try:
                if segments == 1:
                    return super()._bowl_move(target, arm_tag, label, tiny=tiny)
                start = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)[:3]
                goal = np.asarray(target.p, dtype=float)
                out = None
                for k in range(1, segments + 1):
                    waypoint = Pose(start + (goal - start) * (k / segments), target.q)
                    out = super()._bowl_move(waypoint, arm_tag, label, tiny=tiny)
                return out
            except _BowlMotionRejected as exc:
                if (attempt == len(self.CARRY_SEGMENTS) - 1
                        or not any(r in str(exc) for r in self._REPLANNABLE)):
                    raise
                self.plan_success = plan_before

    def _robot_stack(self, idx: int, stack_position) -> bool:
        try:
            return bool(self._pick_and_stack_one(idx, stack_position))
        except Exception:
            return False

    def _observed_stack_top_xy(self, exclude_idx: int) -> np.ndarray:
        """xy of the highest resting bowl on the base stack (same rule as
        ``_observed_stack_top_z``); the base's xy when nothing rests on it."""
        base = self._bowl_xyz(0)
        half_bounds = self._bowl_xy_half_bounds()
        best_z, best_xy = float(base[2]), np.asarray(base[:2], dtype=float)
        for i in range(self._num_bowls):
            if i in (0, exclude_idx):
                continue
            p = self._bowl_xyz(i)
            on_base = not bool(np.any(np.abs(p[:2] - base[:2]) > half_bounds))
            resting = float(p[2]) - float(base[2]) <= 2.5 * self._bowl_height
            if on_base and resting and float(p[2]) > best_z:
                best_z, best_xy = float(p[2]), np.asarray(p[:2], dtype=float)
        return best_xy

    def _observed_stack_top_z(self, exclude_idx: int) -> float:
        """Highest resting bowl on the base stack, read from live poses.

        A bowl counts as stacked when its centre is inside the base bowl's
        footprint (the success criterion) and it is no more than 2.5 bowl
        heights above the base (nested bowls rise ~4 cm each), i.e. it rests on the stack rather than being
        carried over it.
        """
        base = self._bowl_xyz(0)
        half_bounds = self._bowl_xy_half_bounds()
        top_z = float(base[2])
        for i in range(self._num_bowls):
            if i in (0, exclude_idx):
                continue
            p = self._bowl_xyz(i)
            on_base = not bool(np.any(np.abs(p[:2] - base[:2]) > half_bounds))
            resting = float(p[2]) - float(base[2]) <= 2.5 * self._bowl_height
            if on_base and resting:
                top_z = max(top_z, float(p[2]))
        return top_z

    # Full-state expert release height (it waits for the human, so the
    # observed stack is final).  A nested 002_bowl rises 4.2 cm over the bowl
    # below; the blind rule (one full bowl height over the observed top,
    # never below two over the base) released the robot's bowl ~6 cm above
    # its nest onto the human's tilted bowl, and it tipped off toward the
    # robot (seeds 9000001/06/23).
    FULL_STATE_NEST_RISE_M = 0.042
    # Release the bowl 1.5 cm (not 3) above its target: dropped from ~5 cm
    # over the nest it still tipped off the human's tilted bowl toward the
    # robot; 1.5 cm nests it (seeds 9000001/06/23).
    DROP_BOWL_DZ = 0.015

    def _stack_position(self, idx: int):
        base = self._bowl_xyz(0)
        if getattr(self, "_full_state_nest_release", False):
            top_xy = self._observed_stack_top_xy(idx)
            top = self._observed_stack_top_z(idx) + self.FULL_STATE_NEST_RISE_M
            return (float(top_xy[0]), float(top_xy[1]), top)
        # One bowl height over the observed top, never lower than two over
        # the base: the human's bowl may land on the base meanwhile.
        top = max(self._observed_stack_top_z(idx) + self._bowl_height,
                  float(base[2]) + 2.0 * self._bowl_height)
        # Nest on the bowl actually on top (the human's lands ~1-2 cm off the
        # base centre); aiming at the base centre slid the robot's bowl off
        # its rim (robot bowl 5-7 cm off the base).
        top_xy = self._observed_stack_top_xy(idx)
        return (float(top_xy[0]), float(top_xy[1]), top)

    def _wait_avatar_and_settle(self) -> None:
        tail = 0
        while not self.avatar.spare() and tail < self.AVATAR_FINISH_WAIT_STEPS:
            self.step_sim()
            tail += 1
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()

    def _run_robot(self, order) -> bool:
        pool = self._shared_pool
        pool["robot_target"] = int(order[0]) if order else None
        ok = True
        for idx in order:
            ok = self._robot_stack(idx, self._stack_position(idx)) and ok
        self._wait_avatar_and_settle()
        return ok

    # ------------------------------------------------------------------
    # Experts
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        """Full state: knows the human's bowl and stacks the others once
        the human (who starts at t=0, as at an eval reset) has nested its
        bowl and retracted.  Stacking concurrently, the human's drop landed
        on or knocked the robot's bowl off the base on 9/11 failed
        episodes; the robot yields instead."""
        if (self.avatar is not None and all(b is not None for b in self.bowls)
                and not getattr(self, "_eval_assist_fired", False)):
            self._start_avatar_assist()
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False
        if self.avatar is None:
            return super().play_once()
        human = self._human_target
        tail = 0
        while not self.avatar.spare() and tail < self.AVATAR_FINISH_WAIT_STEPS:
            self.step_sim()
            tail += 1
        # The human is spare as soon as it lets go; its bowl may still be
        # falling onto the base.  Read the stack only once every bowl rests
        # (a mid-air read put the release 2.4 cm too high, GPU 9000003).
        self._wait_bowls_at_rest()
        self._full_state_nest_release = True
        return self._run_robot([i for i in self._stacked_indices if i != human])

    BOWL_REST_WINDOW_STEPS = 10
    BOWL_REST_MAX_STEPS = 300
    BOWL_REST_TOL_M = 0.0005

    def _wait_bowls_at_rest(self) -> None:
        def poses():
            return np.asarray([self._bowl_xyz(i) for i in range(self._num_bowls)], dtype=float)
        prev = poses()
        for _ in range(self.BOWL_REST_MAX_STEPS // self.BOWL_REST_WINDOW_STEPS):
            for _ in range(self.BOWL_REST_WINDOW_STEPS):
                self.step_sim()
            now = poses()
            if float(np.max(np.abs(now - prev))) < self.BOWL_REST_TOL_M:
                return
            prev = now

    def _blind_order(self):
        """Candidates by xy distance to the gripper at t=0 (index tie-break);
        independent of the human."""
        tcp = np.asarray(self.robot.get_arm("right").get_ee_pose(), dtype=float)[:2]

        def key(i):
            return (round(float(np.linalg.norm(self._bowl_xyz(i)[:2] - tcp)), 6), i)

        return sorted(self._stacked_indices, key=key)

    def play_blind_once(self) -> bool:
        """Blind: never reads the human's choice.  Commits at t=0 to the
        candidate(s) nearest the gripper (all but one, i.e. one in easy
        mode) and stacks them; no retarget, yield or waiting."""
        order = self._blind_order() if self.avatar is not None else None
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False
        if self.avatar is None:
            return super().play_once()
        return self._run_robot(order[:max(1, len(order) - 1)])

    def _eval_at_reset(self) -> None:
        self._assist_controller().eval_reset(self)

    def _eval_at_step(self, step_idx: int) -> None:
        self._assist_controller().eval_step(self, step_idx)

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        # Any robot-avatar contact fails the episode whenever the collision
        # checker ran (the eval rows enable it).
        if (getattr(self, "avatar_collision_checker", None) is not None
                or self.config.get("track_avatar_collision", False)) \
                and getattr(self, "avatar_collided", False):
            return False
        if not super().check_success():
            return False
        # A bowl still pinched in the closed gripper (e.g. a rejected release
        # left it held over the stack) is not stacked.
        return not any(self._bowl_in_gripper(i) for i in self._stacked_indices)

    def _bowl_in_gripper(self, idx: int) -> bool:
        try:
            arm = self.robot.get_arm("right")
            tcp = np.asarray(arm.get_ee_pose(), dtype=float)[:3]
            closed = float(arm.gripper_val) < self.HELD_GRIPPER_BELOW
        except Exception:
            return False
        center = np.asarray(self._bowl_xyz(idx), dtype=float)
        center[2] += 0.5 * float(self._bowl_height)
        return closed and float(np.linalg.norm(tcp - center)) < self.HELD_TCP_RADIUS_M

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        if all(b is not None for b in getattr(self, "bowls", []) or [None]):
            base = np.asarray(self._bowl_xyz(0), dtype=float)
            metrics["bowl_offsets_from_base"] = [
                [round(float(v), 4) for v in (np.asarray(self._bowl_xyz(i)) - base)]
                for i in self._stacked_indices
            ]
            metrics["avatar_collided_flag"] = bool(getattr(self, "avatar_collided", False))
        return metrics
