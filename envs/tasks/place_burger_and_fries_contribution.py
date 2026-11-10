"""Place burger and fries (Contribution, ``place_burger_and_fries_contribution``):
same scene as ``place_burger_fries`` but the hamburgers are a SHARED pool that
the robot and a human avatar put on the tray together (simple 50-50 design).

* Easy layout: two identical hamburgers spawn in ONE shared region between
  the robot and the human (reachable by both; no human/robot side split).
  Hard mode keeps the parent's foods; its burgers are the shared pool and
  the fries are robot-only.
* Human (``PlaceBurgerFriesAssistController``): starts at reset, takes ONE
  shared burger uniformly at random and carries it onto the tray no matter
  what the robot does.
* Full-state expert (``play_once``): reads the human's choice and places the
  OTHER burger (plus any robot-only food) concurrently with the human,
  planning around the human's current body.
* Blind expert (``play_blind_once``): commits at t=0 to the burger nearest
  its gripper (index tie-break), independent of the human, and runs the
  base task's one-shot pick; then waits (bounded) for the human, settles.

Success: every food (both burgers) on the tray AND no robot-avatar contact.
Per-episode ``shared_pool`` metrics: human_target, robot_target,
same_target.
"""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..interaction.place_burger_fries import PlaceBurgerFriesAssistController
from ..utils import to_numpy
from ..task_bases.place_burger_fries import (
    PlaceBurgerFries,
    _HAMBURG_ID,
)


_GENERATED_MOTIONS_PKL = "avatars/motions/generated_motions.pkl"


class PlaceBurgerAndFriesContribution(PlaceBurgerFries):
    INSTRUCTION = "place the hamburger on the tray while the human adds one food item"

    use_avatar = True

    # No pre-placed tray food: every food is loose (the shared burgers in
    # the easy layout).
    PREPLACE_FOOD_ON_TRAY = False

    # One shared spawn region for both burgers, between the robot (base at
    # y=-0.65) and the human (y>=+0.28).  Its far corner (0.22, 0.06) is
    # ~0.74 m from the Franka base (the base task's burger region reaches
    # 0.73 m; the tray drop is ~0.70 m); the human's palm reaches its near
    # corner from the natural stance at <0.6 m.  The sampler still rejects
    # points over the tray and enforces SPAWN_MIN_SEP between burgers.
    SHARED_REGION_X = (0.02, 0.22)
    SHARED_REGION_Y = (-0.08, 0.06)

    def _burger_spec(self):
        spec = super()._burger_spec()
        return dataclasses.replace(
            spec, spawn_xy_range=(self.SHARED_REGION_X, self.SHARED_REGION_Y),
        )

    def _food_specs_for_episode(self):
        if self.hard_mode_enabled():
            return super()._food_specs_for_episode()
        burger = self._burger_spec()
        return [burger, burger]

    def _shared_pool_indices(self) -> list[int]:
        return [
            i for i, (_actor, spec) in enumerate(self.food_actors)
            if spec.asset_id == _HAMBURG_ID
        ]

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([-0.20, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT
    AVATAR_YAW_DEG = 0.0
    AVATAR_LIFT_Z = 0.05
    AVATAR_BACK_OFFSET = 0.05
    AVATAR_PALM_FORWARD_OFFSET = 0.04

    # Trash_clip dry-run helpers (kept for diagnostics / delegators).
    ASSIST_MOTION_NAME = "Trash_clip"
    ATTACH_FRAME = 25
    DETACH_FRAME = 57
    AVATAR_HAND_ID = 0

    # Human drop point on the tray (offset from the live tray centre) and
    # the robot's drop slots; disjoint so the two never stack.
    AVATAR_TRAY_OFFSET = (-0.10, +0.06)
    ROBOT_TRAY_SLOTS = (
        (+0.07, -0.03),
        (-0.03, -0.05),
        (+0.07, +0.05),
        (-0.01, +0.03),
    )
    AVATAR_PICK_LIFT = 0.020
    AVATAR_PLACE_LIFT = 0.090
    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 360
    AVATAR_RETRACT_FRAMES = 180
    AVATAR_NATURAL_BODY_MARGIN = 0.34
    AVATAR_NATURAL_Y_BOUNDS = (0.28, 0.62)
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_SETTLE_STEPS = 20
    AVATAR_PICKUP_Z_CLEARANCE = 0.050
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    AVATAR_FINISH_MAX_STEPS = 3000

    # Blind/full-state planners see the live human body, inflated like
    # place_bread_in_basket_assist.
    _AVATAR_PCD_INFLATE_FACTOR = 1.8
    _AVATAR_PCD_RES = 0.04

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        avatar_cfg = dict(cfg.get("avatar", {}))
        avatar_cfg.setdefault("generated_motion_data", _GENERATED_MOTIONS_PKL)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)
        # The burger the human is currently carrying / reaching for.
        self.avatar_burger = None
        self._avatar_burger_drop_xy = None
        self._reset_shared_state()

    def _reset_shared_state(self) -> None:
        self._shared_pool = None
        self._human_target_idx = None
        self.avatar_burger = None

    def create_interaction_controller(self) -> PlaceBurgerFriesAssistController:
        return PlaceBurgerFriesAssistController()

    def _assist_controller(self) -> PlaceBurgerFriesAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, PlaceBurgerFriesAssistController):
            raise TypeError("task does not use burger/fries assist behavior")
        return controller

    def load_actors(self):
        self._reset_shared_state()
        super().load_actors()

    # ------------------------------------------------------------------
    # Legacy private helpers, delegated to the controller
    # ------------------------------------------------------------------

    def _avatar_init_pos_lifted(self) -> np.ndarray:
        return self._assist_controller().avatar_init_pos_lifted(self)

    def _avatar_rot_with_yaw(self) -> np.ndarray:
        return self._assist_controller().avatar_rot_with_yaw(self)

    def _avatar_palm_center(self, hand_id: int) -> np.ndarray:
        return self._assist_controller().avatar_palm_center(self, hand_id)

    def _avatar_wrist(self, hand_id: int) -> np.ndarray:
        return self._assist_controller().avatar_wrist(self, hand_id)

    def _avatar_dry_run(self):
        return self._assist_controller().avatar_dry_run(self)

    def _start_avatar_burger_assist(self):
        return self._assist_controller().start_avatar_burger_assist(self)

    # ------------------------------------------------------------------
    # Robot side
    # ------------------------------------------------------------------

    def _food_pos(self, idx: int) -> np.ndarray:
        actor, _spec = self.food_actors[idx]
        return to_numpy(actor.get_pose().p).ravel()[:3].astype(float)

    def _robot_ee_xy(self, arm_tag: str = "right") -> np.ndarray:
        arm = self.robot.get_arm(arm_tag)
        return np.asarray(arm.get_ee_pose()[:2], dtype=float)

    def _blind_target(self) -> int:
        """Fixed rule independent of the human: the shared burger nearest
        the gripper (xy) right now; index tie-break."""
        ee_xy = self._robot_ee_xy()
        return min(
            self._shared_pool_indices(),
            key=lambda i: (
                round(float(np.linalg.norm(self._food_pos(i)[:2] - ee_xy)), 6), i,
            ),
        )

    @contextlib.contextmanager
    def _avatar_in_planner(self, arm_tag: str):
        """Refresh the live avatar body into the planner cloud before every
        RRT plan (this task registers no static cloud)."""
        planner = self.robot.get_arm(arm_tag).planner
        plan_path = planner.plan_path

        def plan_path_with_avatar(*args, **kwargs):
            pts = None
            if self.avatar_collider is not None:
                pts = sample_capsule_planner_points(
                    self.avatar_collider.current_capsules(),
                    inflate_factor=self._AVATAR_PCD_INFLATE_FACTOR,
                )
            if pts is not None and pts.size:
                planner.update_obstacles(pts, resolution=self._AVATAR_PCD_RES)
            return plan_path(*args, **kwargs)

        planner.plan_path = plan_path_with_avatar
        try:
            yield
        finally:
            del planner.plan_path

    def _place_foods(self, indices, arm_tag: str = "right") -> bool:
        """Base task's one-shot pick-and-drop for each food, in order, onto
        the robot's tray slots.  A failure is an honest failure."""
        for k, idx in enumerate(indices):
            actor, spec = self.food_actors[idx]
            slot = self.ROBOT_TRAY_SLOTS[min(k, len(self.ROBOT_TRAY_SLOTS) - 1)]
            tray_x, tray_y = self._live_tray_center_xy()
            if not self._pick_and_drop(
                actor, spec.asset_id, arm_tag,
                close_value=spec.close_value,
                below_center=spec.below_center,
                drop_xy=(tray_x + slot[0], tray_y + slot[1]),
                tcp_yaw_deg=spec.tcp_yaw_deg,
                transit_steps=spec.transit_steps,
                transit_sim=spec.transit_sim,
                align_to_object_yaw=spec.align_to_object_yaw,
            ):
                return False
        return True

    def _robot_only_indices(self) -> list[int]:
        shared = set(self._shared_pool_indices())
        return [i for i in range(len(self.food_actors)) if i not in shared]

    def _wait_for_human(self) -> None:
        if self.avatar is None:
            return
        steps = 0
        while not self.avatar.spare() and steps < self.AVATAR_FINISH_MAX_STEPS:
            self.step_sim()
            steps += 1

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        """Full-state expert: reads the human's choice and places the other
        burger concurrently with the human."""
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        if self.avatar is None:
            return super().play_once()
        controller = self._assist_controller()
        if not bool(self.config.get("eval_mode", False)):
            self._start_avatar_burger_assist()
        human = controller.human_target(self)
        others = [i for i in self._shared_pool_indices() if i != human]
        if others:
            controller.shared_pool(self)["robot_target"] = int(others[0])
        with self._avatar_in_planner(arm_tag):
            ok = self._place_foods(others + self._robot_only_indices(), arm_tag)
        if not ok:
            return False
        self._wait_for_human()
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def play_blind_once(self) -> bool:
        """Blind expert: commits at t=0 to the burger nearest its gripper,
        never looks at the human's choice, never re-targets."""
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        if self.avatar is None:
            return super().play_once()
        target = self._blind_target()
        self._assist_controller().shared_pool(self)["robot_target"] = int(target)
        with self._avatar_in_planner(arm_tag):
            ok = self._place_foods([target] + self._robot_only_indices(), arm_tag)
        if not ok:
            return False
        self._wait_for_human()
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def check_success(self) -> bool:
        if (self.avatar_collision_checker is not None
                and bool(getattr(self, "avatar_collided", False))):
            return False
        return super().check_success()

    # ------------------------------------------------------------------
    # Eval-mode avatar assist — owned by the controller
    # ------------------------------------------------------------------

    def _eval_at_reset(self) -> None:
        self._assist_controller().eval_reset(self)

    def _eval_at_step(self, step_idx: int) -> None:
        self._assist_controller().eval_step(self, step_idx)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
