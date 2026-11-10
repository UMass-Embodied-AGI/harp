"""Place food in skillet (Contribution, ``place_food_in_skillet_contribution``) over the
shared skillet task core."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..base_task import TargetSpec
from ..interaction.place_food_in_skillet import PlaceFoodInSkilletAssistController
# Explicit self-aliases preserve imports and old pickle lookups from the former
# standalone module while keeping one implementation in the family core.
from ..task_bases.place_food_in_skillet import (
    FoodSpec,
    SkilletTaskCore,
    _APPLE_ID,
    _BREAD_ID,
    _BURGER_ID,
    _PAN_ID,
    _Q_BURGER as _Q_BURGER,
    _Q_BURGER_MANUAL as _Q_BURGER_MANUAL,
    _Q_UPRIGHT,
    _read_model_data,
    _read_scale as _read_scale,
    _scaled_center as _scaled_center,
    _scaled_extents as _scaled_extents,
)
from ..grasp import tcp_to_link_pose
from ..planning.base import PlanResult
from ..utils import Actor, ASSETS_PATH, Pose, load_mesh, load_object, to_numpy


class PlaceFoodInSkilletContribution(SkilletTaskCore):
    """Robot and avatar cooperatively place foods into the skillet.

    The human side (food allocation, the avatar pick-and-place with its start
    delay, the eval trigger) is owned by the composed
    :class:`PlaceFoodInSkilletAssistController`; this class keeps the robot
    choreography and thin ``_``-prefixed delegators.
    """

    INSTRUCTION = (
        "place the skillet on the stove and put the foods in it while the "
        "human helps place one food"
    )
    use_avatar = True
    SKILLET_LOG_LABEL = "skillet_assist"

    # Preserve the assist's camera surface instead of inheriting the core's
    # top-down static camera.
    static_camera_list = None
    recording_camera_pos = [0.85, -1.05, 1.55]
    recording_camera_lookat = [-0.10, 0.06, 0.88]
    side_camera_pos = [-1.10, -0.50, 1.35]
    side_camera_lookat = [-0.10, 0.06, 0.88]

    # Avatar stands on the +Y side of the table, facing the task region.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.72, -0.18], dtype=np.float64)
    avatar_init_rot = AVATAR_BASE_ROT

    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 320
    AVATAR_RETRACT_FRAMES = 240
    # 0.12 let the avatar-dropped food free-fall ~15 cm and punch through the
    # compliant pan-bottom hull into the cooktop (measured: burger came to
    # rest at z 0.763, embedded under the pan).  Drop from just above the rim.
    AVATAR_DROP_ABOVE_RIM = 0.05
    AVATAR_START_DELAY_RANGE = (40, 180)
    AVATAR_NATURAL_BODY_MARGIN = 0.36
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.42, 0.82)
    AVATAR_NATURAL_YAW_LIMIT_DEG = 35.0
    AVATAR_NATURAL_SETTLE_STEPS = 20
    EVAL_TRIGGER_STEP_MIN = 20
    # After its pick-and-place the human glides back to the start-stance
    # depth over this many sim steps (0 disables); see the controller.
    HUMAN_STEP_BACK_SIM_STEPS = 400
    EVAL_TRIGGER_STEP_MAX = 60

    # Exempt human contact.  The human's pan drop
    # sweeps its hand past the parked robot's wrist by design (task audit:
    # 3/3 hold-still seeds, r_palm/r_thumb vs link7 at policy steps 23-27).
    # A contact check is exempt when (1) the human is in its drop window,
    # i.e. its pick_and_place clip frame is within [detach - PRE, detach +
    # POST] frames of the food release, and (2) the robot holds still (every
    # arm joint slower than EXEMPT_ROBOT_STILL_RAD_S since the previous
    # check).  Exempt checks are rewritten to collided=False, so they fail
    # neither the hold-still screening rule nor the collision metrics, and
    # are kept under ``exempt_human_contact`` (metrics) / the record's
    # ``exempt_pairs``.  A robot that moves into the human is never exempt.
    # 220: the hand meets the wrist ~165 frames before the release (CPU
    # hold-still seed 9000000, first contact at sim step 495, release 658).
    EXEMPT_DROP_PRE_FRAMES = 220
    EXEMPT_DROP_POST_FRAMES = 120
    EXEMPT_ROBOT_STILL_RAD_S = 0.05

    # play_once (full-state expert): longest wait for the human's
    # pick-and-place before the robot's own pick (start delay <= 180 frames
    # + approach/transport/retract 820 frames + margin).
    EXPERT_HUMAN_WAIT_MAX_SIM_STEPS = 1500

    FOOD_TRANSPORT_ABOVE_TABLE = 0.42

    # Robot apple grasp: the family's top-down primitive, as in the
    # Co-presence variant, instead of the one-shot side grasp.  This task's
    # apple spawns up to y 0.38 (0.73 m from the Franka base), where the
    # horizontal side approach stops 5-6 cm above its grasp and misses
    # (0/10 robot-apple seeds, CPU).  Close 0.50 with no post-lift re-grip
    # is the Co-presence variant's measured apple close.  Spawn, friction and the
    # human's apple pick are not affected.
    ROBOT_APPLE_TOP_DOWN = True
    ROBOT_APPLE_CLOSE_VALUE = 0.50
    APPLE_POST_LIFT_CLOSE_VALUE = None
    # play_once (full-state expert) parks the arm back over its base, out
    # of the human's reach to the pan, before the human starts: from the
    # reset pose (TCP ~(0, 0.11, 0.85), beside the pan) the human's
    # transport swept into the parked wrist outside the exempt drop window
    # (CPU seeds 9000011/35: first contact at sim step 490-540).
    EXPERT_PARK_TCP_FROM_BASE = (0.0, 0.20, 0.40)   # dx, dy from base, dz above table
    # The robot's own apple is released with its centre this far above the
    # pan rim plus its half-size, not lowered to the pan floor: a round apple
    # squeezed in the jaws and pressed toward the floor popped out sideways
    # during the descent (CPU seeds 9000011/32), while one let go above the
    # pan settled in it.
    ROBOT_APPLE_RELEASE_ABOVE_RIM = 0.02
    # Robot burger grasp: this task spawns the boxy
    # model_5 burger at _Q_UPRIGHT, which lays its 7.75 cm long axis along
    # world y, the jaw-closing axis of the default tcp yaw 0.  The 8 cm open
    # jaws then landed on the burger's ends and pressed it into the table
    # (~9 mm) instead of straddling it, and the lift lost it or flung it
    # through the table (the main robot-burger failure).
    # The expert closes across the burger's live narrowest horizontal
    # axis (its 4.8 cm waist, as the core's _Q_BURGER pairing does) with the
    # core's burger close 0.50.  Spawn, friction and the human are not affected.
    ROBOT_BURGER_SETTLED_YAW = True
    ROBOT_BURGER_CLOSE_VALUE = 0.50
    # The close is width-matched to the burger's live collision width
    # across the jaws (model_data extents can be off): gap = width minus
    # this squeeze, the core pairing's 4.8 cm waist at close 0.50 (4.0 cm
    # gap); jaw gap = 0.08 * close value.  Clamped to ROBOT_BURGER_CLOSE_RANGE.
    ROBOT_BURGER_SQUEEZE_M = 0.008
    ROBOT_BURGER_CLOSE_RANGE = (0.30, 0.55)
    # The held burger slid out of the jaws in the fast core transit (36 x 18
    # sim steps) and fell through the table; it transits at the apple's pace.
    ROBOT_BURGER_TRANSIT = (90, 30)
    ROBOT_BURGER_PREFERRED_YAW_DEG = -90.0
    # False keeps the burger at tcp yaw 0 (along its long axis) with the
    # fixed ROBOT_BURGER_CLOSE_VALUE, the slow transit and the on-food place.
    ROBOT_BURGER_WAIST_GRASP = True
    # Release gap between the held burger's bottom and the top of the food
    # already in the pan (see _gentle_place_burger_on_food).
    ROBOT_BURGER_ABOVE_SUPPORT = 0.012
    FOOD_DROP_ABOVE_RIM = 0.12

    FOOD_SPAWN_JITTER_XY = 0.030
    FOOD_RANDOM_REGIONS = (
        ((-0.44, -0.28), (-0.15, 0.16)),
        ((0.12, 0.28), (-0.12, 0.34)),
        ((-0.10, 0.10), (0.30, 0.38)),
    )
    APPLE_RANDOM_REGIONS = (
        ((0.12, 0.28), (0.18, 0.34)),
        ((-0.08, 0.08), (0.30, 0.38)),
    )
    BREAD_RANDOM_REGIONS = (
        ((0.12, 0.28), (0.18, 0.34)),
    )
    FOOD_MIN_SEP = 0.18
    AVATAR_RANDOM_FOOD_POOL = ("burger", "apple")

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", False)
        # The assist story needs one food for the human and at least one for
        # the robot regardless of difficulty; the simplified default of a
        # single burger would degenerate to a robot-only episode.
        cfg.setdefault("foods", ["burger", "bread"])
        super().__init__(cfg)
        if "force_dynamic_pan" in cfg:
            self._force_dynamic_pan = bool(cfg["force_dynamic_pan"])
        self._avatar_food_idx = None
        self._robot_food_indices: list[int] = []

    def create_interaction_controller(self) -> PlaceFoodInSkilletAssistController:
        return PlaceFoodInSkilletAssistController()

    def _assist_controller(self) -> PlaceFoodInSkilletAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, PlaceFoodInSkilletAssistController):
            raise TypeError("task does not use skillet assist behavior")
        return controller

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _enabled_food_specs(self) -> list[FoodSpec]:
        pool = [name for name in self.AVATAR_RANDOM_FOOD_POOL if name != "bread"]
        avatar_food = str(np.random.choice(pool))
        requested = list(np.random.permutation(["bread", avatar_food]))
        all_specs = {
            # Bread model 1 is the smaller bun-like variant used by
            # place_burger_fries; it is much easier to pick and fits the pan.
            "bread": FoodSpec(
                "bread", _BREAD_ID, 1, (-0.34, -0.10), _Q_UPRIGHT,
                0.055, 4.0, 0.30, 0.005, 0.0, False, False,
            ),
            "apple": FoodSpec(
                # Match the known-good place_burger_fries apple primitive:
                # model 0, top-down grasp, shallow close.  Put it just
                # behind/right of the cooktop, outside both the cooktop
                # footprint and the skillet handle's transfer path.
                "apple", _APPLE_ID, 0, (0.14, 0.34), _Q_UPRIGHT,
                0.045, 4.0, 0.30, 0.005, 0.0, False, True,
            ),
            "burger": FoodSpec(
                # model_5 = boxy wrapped burger. Map authored local +Y onto
                # world +Z so it lies flat in the pan. Box grips reliably via
                # the top-down primitive (use_grasp_helper=False); close 0.40
                # cages without crushing; friction 5.0.  Mirrors the base fix
                # (envs/task_bases/place_food_in_skillet.py).
                "burger", _BURGER_ID, 5, (0.14, 0.34), _Q_UPRIGHT,
                0.055, 5.0, 0.40, 0.005, 0.0, False, False,
            ),
        }
        specs = [all_specs[name] for name in requested]
        if self.ROBOT_APPLE_TOP_DOWN:
            specs = [
                replace(spec, side_grasp=False,
                        close_value=float(self.ROBOT_APPLE_CLOSE_VALUE))
                if spec.name == "apple" else spec
                for spec in specs
            ]
        return specs

    def _sample_food_spawn_xys(
        self,
        specs: list[FoodSpec],
    ) -> list[tuple[float, float]]:
        if not self.config.get("randomize_initial_positions", True):
            return []
        for _ in range(80):
            out = []
            for spec in specs:
                regions = (
                    self.APPLE_RANDOM_REGIONS
                    if spec.asset_id == _APPLE_ID
                    else self.BREAD_RANDOM_REGIONS
                    if spec.asset_id == _BREAD_ID
                    else self.FOOD_RANDOM_REGIONS
                )
                xr, yr = regions[int(np.random.randint(len(regions)))]
                out.append((
                    float(np.random.uniform(*xr)),
                    float(np.random.uniform(*yr)),
                ))
            ok = True
            for i in range(len(out)):
                for j in range(i + 1, len(out)):
                    if np.linalg.norm(np.asarray(out[i]) - np.asarray(out[j])) < self.FOOD_MIN_SEP:
                        ok = False
                        break
                if not ok:
                    break
            # The dynamic pan spawns at ~(0.25, 0.03) with an AABB footprint of
            # x +-0.08 / y +-0.135 around its origin: reject food samples that
            # would land inside it (they end up leaning on the bowl rim and
            # foul the handle grasp).
            if ok:
                pan_xy = np.asarray(
                    getattr(self, "_pan_spawn_xy", (0.25, 0.03)), dtype=float,
                )
                for x, y in out:
                    if abs(x - pan_xy[0]) < 0.14 and abs(y - pan_xy[1]) < 0.20:
                        ok = False
                        break
            if ok:
                return out
        return [
            self._jitter_xy(spec.spawn_xy, self.FOOD_SPAWN_JITTER_XY)
            for spec in specs
        ]

    def load_actors(self):
        self.build_kitchen()
        table_top = self.TABLE_TOP_Z

        # Easy (default): the pan is pre-placed on the cooktop and only the
        # foods are manipulated.  Hard / force_dynamic_pan: it starts on the
        # right-front counter for the robot to pick.  CoACD is forced so the
        # pan interior is not treated as one filled convex hull.
        if self._pan_starts_on_cooktop():
            origin = self._pan_origin_target_on_cooktop()
            pan_x, pan_y = float(origin[0]), float(origin[1])
            pan_pose = Pose(origin + np.array([0.0, 0.0, 0.015]), _Q_UPRIGHT)
        else:
            pan_x, pan_y = self._jitter_xy((0.25, 0.03), self.PAN_SPAWN_JITTER_XY)
            pan_pose = Pose([pan_x, pan_y, table_top + 0.050], _Q_UPRIGHT)
        self._pan_spawn_xy = (pan_x, pan_y)
        # friction 1.5 / density 800: the handle grip is a geometric pinch —
        # the old 5.0/2500 tipped the pan when centring it on the cooktop and
        # drooped it ~47 deg in the pinch (mirrors the base task; probe19).
        pan_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _PAN_ID / "visual" / "base0.glb",
            pan_pose,
            scale=(self._pan_scale,) * 3,
            convex=True,
            is_static=False,
            friction=1.5,
            density=800.0,
            decompose_object_error_threshold=0.0,
        )
        self.pan_actor = Actor(pan_entity, _read_model_data(_PAN_ID, 0), _PAN_ID)

        self.food_actors = []
        specs = self._enabled_food_specs()
        sampled_xys = self._sample_food_spawn_xys(specs)
        for i, spec in enumerate(specs):
            if sampled_xys:
                x, y = sampled_xys[i]
                spec = replace(spec, spawn_xy=(x, y))
            else:
                x, y = self._jitter_xy(spec.spawn_xy, self.FOOD_SPAWN_JITTER_XY)
                spec = replace(spec, spawn_xy=(x, y))
            actor = load_object(
                self.scene,
                Pose([x, y, table_top + spec.spawn_z_offset], spec.spawn_quat),
                spec.asset_id,
                model_id=spec.model_id,
                convex=True,
                is_static=False,
                friction=spec.friction,
            )
            self.food_actors.append((actor, spec))

        self.target = TargetSpec(
            object=self.pan_actor.entity,
            position=lambda: np.array([self._cooktop_xy[0], self._cooktop_xy[1], self.TABLE_TOP_Z]),
            label="cooktop",
        )

    # ------------------------------------------------------------------
    # Assist-specific core hooks
    # ------------------------------------------------------------------

    def _food_center(self, actor: Actor, spec: FoodSpec) -> np.ndarray:
        return self._actor_center(actor, spec.asset_id, spec.model_id)

    def _pick_and_drop_food(self, actor: Actor, spec: FoodSpec, arm_tag: str) -> bool:
        if (spec.name == "burger" and self.ROBOT_BURGER_SETTLED_YAW
                and not spec.side_grasp and not spec.use_grasp_helper):
            grasp = self.settled_topdown_grasp(actor.entity, spec.asset_id, spec.model_id)
            if grasp["feasible"]:
                # A parallel-jaw grasp is the same at yaw and yaw +- 180.
                # settled_topdown_grasp wraps into [-90, 90], so float noise
                # chose +90 or -90 for the same burger; +90 lost the burger
                # in transit on all 5 such CPU episodes (seeds 9000033/39/
                # 54/79/118), -90 held it on 12 of 17.  Use the equivalent
                # yaw closest to ROBOT_BURGER_PREFERRED_YAW_DEG.
                yaw = min((float(grasp["yaw_deg"]) + k for k in (-180.0, 0.0, 180.0)),
                          key=lambda y: abs(y - self.ROBOT_BURGER_PREFERRED_YAW_DEG))
                width = self._live_width_across_jaws(actor, yaw)
                close = float(self.ROBOT_BURGER_CLOSE_VALUE)
                if width is not None:
                    lo, hi = self.ROBOT_BURGER_CLOSE_RANGE
                    close = float(np.clip((width - self.ROBOT_BURGER_SQUEEZE_M) / 0.08, lo, hi))
                if not self.ROBOT_BURGER_WAIST_GRASP:
                    yaw, close = 0.0, float(self.ROBOT_BURGER_CLOSE_VALUE)
                print(f"[{self.SKILLET_LOG_LABEL}] robot burger: close {close:.3f} across "
                      f"{grasp['width'] * 100:.1f} cm (live "
                      f"{'n/a' if width is None else f'{width * 100:.1f}'} cm) at tcp yaw "
                      f"{yaw:.1f} deg", flush=True)
                spec = replace(spec, tcp_yaw_deg=yaw, close_value=close)
        return super()._pick_and_drop_food(actor, spec, arm_tag)

    def _live_width_across_jaws(self, actor: Actor, yaw_deg: float) -> float | None:
        """Extent of the actor's live collision vertices along the jaw-closing
        axis of tcp yaw ``yaw_deg`` ((-sin t, -cos t)), or None."""
        try:
            verts = np.concatenate([
                np.asarray(to_numpy(g.get_verts()), dtype=float).reshape(-1, 3)
                for g in actor.entity.geoms
            ], 0)
        except Exception:
            return None
        t = np.deg2rad(float(yaw_deg))
        proj = verts[:, :2] @ np.array([-np.sin(t), -np.cos(t)])
        width = float(proj.max() - proj.min()) if len(proj) else float("nan")
        return width if np.isfinite(width) and width > 0 else None

    def _food_transit_profile(self, spec: FoodSpec) -> tuple[int, int]:
        if spec.name == "burger" and self.ROBOT_BURGER_SETTLED_YAW:
            return tuple(self.ROBOT_BURGER_TRANSIT)
        return super()._food_transit_profile(spec)

    def _retreat_after_food_release(
        self,
        release_tcp: Pose,
        retreat_tcp: Pose,
        arm_tag: str,
        tcp_offset,
    ) -> None:
        """Retain the assist scenario's authored Cartesian retreat."""
        del tcp_offset
        self._move_tcp_cartesian(
            release_tcp,
            retreat_tcp,
            arm_tag,
            n_steps=36,
            sim_per_step=18,
        )

    # ------------------------------------------------------------------
    # Cooperative allocation, rollout, and evaluation
    # ------------------------------------------------------------------

    def _choose_avatar_food(self) -> tuple[int, list[int]]:
        return self._assist_controller().choose_avatar_food(
            self, pan_id=_PAN_ID, avatar_only_ids=(_BURGER_ID, _APPLE_ID),
        )

    def _delay_avatar_motion_start(self, delay_steps: int) -> None:
        self._assist_controller().delay_avatar_motion(self, delay_steps)

    def _start_avatar_food_place(self, actor: Actor, spec: FoodSpec) -> None:
        self._assist_controller().start_avatar_food_place(self, actor, spec)
        # The controller restarted the collision log; the release frame
        # (start delay included) anchors the exempt drop window.
        self._exempt_contacts = []
        motion = self._human_pick_place_motion()
        detach = getattr(motion, "detach_frame", None)
        self._exempt_drop_frame = None if detach is None else int(detach)

    # ------------------------------------------------------------------
    # Exempt human contact during the human's drop (task-scoped)
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        self._human_step_back = None
        self._exempt_contacts: list[dict] = []
        self._exempt_drop_frame: int | None = None
        self._exempt_prev_arm: tuple[int, np.ndarray] | None = None
        return super().reset(seed)

    def _human_pick_place_motion(self):
        modules = getattr(getattr(self, "avatar", None), "motion_modules", None) or {}
        return modules.get("pick_and_place")

    def _human_in_drop_window(self) -> bool:
        drop = getattr(self, "_exempt_drop_frame", None)
        motion = self._human_pick_place_motion()
        if drop is None or motion is None:
            return False
        frame = int(getattr(motion, "at_frame", -1))
        return (drop - int(self.EXEMPT_DROP_PRE_FRAMES)
                <= frame <= drop + int(self.EXEMPT_DROP_POST_FRAMES))

    def _robot_still_since_last_check(self) -> bool:
        """Every arm joint slower than EXEMPT_ROBOT_STILL_RAD_S since the
        previous collision check (False on the first check)."""
        try:
            q = np.asarray(self.robot.get_arm("right").get_arm_qpos(), dtype=np.float64).ravel()
        except Exception:
            return False
        step = int(self.SIM_STEP_IDX)
        prev = getattr(self, "_exempt_prev_arm", None)
        self._exempt_prev_arm = (step, q)
        if prev is None or step <= prev[0] or prev[1].shape != q.shape:
            return False
        speed = float(np.max(np.abs(q - prev[1]))) / ((step - prev[0]) * self._sim_dt())
        return speed < float(self.EXEMPT_ROBOT_STILL_RAD_S)

    def _tick_avatar_checks(self):
        log_before = self.avatar_collision_log
        n_before = len(log_before)
        collided_before = bool(self.avatar_collided)
        super()._tick_avatar_checks()
        log = self.avatar_collision_log
        if log is not log_before or len(log) <= n_before:
            return
        self._apply_exempt_human_contact(log[-1], collided_before)

    def _apply_exempt_human_contact(self, record: dict, collided_before: bool) -> bool:
        """Rewrite one fresh check record when its contact is exempt."""
        still = self._robot_still_since_last_check()
        if not record.get("collided") or not still or not self._human_in_drop_window():
            return False
        record["collided"] = False
        record["exempt_human_contact"] = True
        record["exempt_pairs"] = record.get("pairs")
        record["pairs"] = []
        self.avatar_collided = collided_before
        motion = self._human_pick_place_motion()
        self._exempt_contacts.append({
            "sim_step": record.get("sim_step"),
            "human_frame": int(getattr(motion, "at_frame", -1)),
            "drop_frame": self._exempt_drop_frame,
            "min_distance": record.get("min_distance"),
            "pairs": [list(p[:2]) for p in (record.get("exempt_pairs") or [])],
        })
        return True

    def exempt_human_contact_metrics(self) -> dict:
        contacts = list(getattr(self, "_exempt_contacts", None) or [])
        return {
            "any": bool(contacts),
            "n_checks": len(contacts),
            "first_sim_step": contacts[0]["sim_step"] if contacts else None,
            "drop_frame": getattr(self, "_exempt_drop_frame", None),
            "window_frames": [int(self.EXEMPT_DROP_PRE_FRAMES), int(self.EXEMPT_DROP_POST_FRAMES)],
            "robot_still_rad_s": float(self.EXEMPT_ROBOT_STILL_RAD_S),
            "pairs": sorted({tuple(p) for c in contacts for p in c["pairs"]}),
        }

    def _eval_at_reset(self) -> None:
        self._assist_controller().eval_reset(self)

    def _eval_at_step(self, step_idx: int) -> None:
        self._assist_controller().eval_step(self, step_idx)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if not self._pan_starts_on_cooktop() and not self._place_pan_on_cooktop(
            arm_tag
        ):
            return False

        if self.avatar is None or len(self.food_actors) < 2:
            ok_any = False
            for actor, spec in self.food_actors:
                ok = self._pick_and_drop_food(actor, spec, arm_tag)
                ok_any = ok_any or ok
                if not ok:
                    pass
            for _ in range(180):
                self.step_sim()
            return ok_any

        avatar_idx, robot_indices = self._choose_avatar_food()
        if avatar_idx < 0:
            return False
        avatar_actor, avatar_spec = self.food_actors[avatar_idx]

        self._park_arm_for_human(arm_tag)
        self._start_avatar_food_place(avatar_actor, avatar_spec)
        # Full-state expert yields: it knows the human's food (fair coin), so
        # it holds still until the human's pick-and-place has ended before
        # reaching for its own.  Moving concurrently, the arm met the human's
        # hand on 13/14 failed episodes; a
        # still robot during the human's pan drop is the exempt case above.
        waited = 0
        while self._human_busy() and waited < self.EXPERT_HUMAN_WAIT_MAX_SIM_STEPS:
            self.step_sim()
            waited += 1
        self._expert_human_wait_sim_steps = waited
        if getattr(self, "_expert_parked_for_human", False):
            # Picks are tuned from the reset arm configuration; descending
            # from the parked one took another IK branch (seed 9000039).
            self._return_arm_to_qpos(self._expert_home_qpos, arm_tag)

        ok_any = False
        for idx in robot_indices:
            actor, spec = self.food_actors[idx]
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok
            if not ok:
                pass
        if not self.avatar.spare():
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(180):
            self.step_sim()
        return ok_any

    def _human_busy(self) -> bool:
        """The human's pick-and-place or its step back is still running."""
        return (not self.avatar.spare()
                or self._assist_controller().step_back_active(self))

    def _return_arm_to_qpos(self, goal_q, arm_tag: str) -> bool:
        """Joint-space MPLib plan back to ``goal_q`` (no-op on failure)."""
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        backend = getattr(planner, "_mplib", None)
        if backend is None:
            return False
        current = np.asarray(arm.get_arm_qpos(), dtype=float)
        try:
            res = backend.plan_qpos(
                [planner._pad_qpos(np.asarray(goal_q, dtype=float))],
                planner._pad_qpos(current), time_step=1 / 250, planning_time=5,
            )
        except Exception as exc:  # planner failure keeps the arm parked
            print(f"[skillet_assist] return to reset qpos failed: {exc}", flush=True)
            return False
        if not res or res.get("status") != "Success":
            print(f"[skillet_assist] return to reset qpos: {res and res.get('status')}", flush=True)
            return False
        pos = np.asarray(res["position"])
        self.execute_plan(PlanResult(True, pos, np.asarray(res.get("velocity", np.zeros_like(pos)))), arm_tag)
        return True

    def _pan_food_support_top(self, actor) -> float | None:
        """Top z of the highest other food resting over the pan bowl, or None."""
        bowl = self._pan_bowl_center()
        top = None
        for other, other_spec in self.food_actors:
            if other is actor or other_spec.asset_id == _PAN_ID:
                continue
            aabb = np.asarray(to_numpy(other.entity.get_AABB()), dtype=float).reshape(2, 3)
            center = aabb.mean(axis=0)
            if float(np.linalg.norm(center[:2] - bowl[:2])) <= self.FOOD_SUCCESS_XY:
                top = float(aabb[1, 2]) if top is None else max(top, float(aabb[1, 2]))
        return top

    def _gentle_place_food(self, actor, spec, arm_tag, hover_tcp, make_tcp):
        if (spec.name == "burger" and self.ROBOT_BURGER_SETTLED_YAW
                and self._pan_food_support_top(actor) is not None):
            return self._gentle_place_burger_on_food(actor, spec, arm_tag, hover_tcp, make_tcp)
        if spec.name != "apple" or self.ROBOT_APPLE_RELEASE_ABOVE_RIM is None:
            return super()._gentle_place_food(actor, spec, arm_tag, hover_tcp, make_tcp)
        bowl = self._pan_bowl_center()
        food_z = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        held_off = food_z - float(np.asarray(hover_tcp.p, dtype=float)[2])
        food_half = 0.5 * float(np.min(_scaled_extents(spec.asset_id, spec.model_id)))
        target_food_z = self._pan_rim_z() + food_half + float(self.ROBOT_APPLE_RELEASE_ABOVE_RIM)
        target_tcp_z = max(target_food_z - held_off, self._pan_rim_z() - 0.01)
        release_tcp = make_tcp(np.array([bowl[0], bowl[1], target_tcp_z]))
        if not self._move_tcp_cartesian(hover_tcp, release_tcp, arm_tag, n_steps=30, sim_per_step=22):
            return None
        target_xy = np.asarray(release_tcp.p, dtype=float)[:2]
        for _ in range(200):
            self.step_sim()
            tcp_xy = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose()[:2], dtype=float)
            if float(np.linalg.norm(tcp_xy - target_xy)) < 0.006:
                break
        self.open_gripper(arm_tag, num_steps=100)
        for _ in range(180):
            self.step_sim()
        return release_tcp

    def _gentle_place_burger_on_food(self, actor, spec, arm_tag, hover_tcp, make_tcp):
        """Core gentle place, but lowered onto the food already in the pan.

        The core lowers the held food to just above the bowl floor; with the
        human's food already at the bowl centre that drove the robot's
        burger into it and the solver diverged (NaN on seeds
        9000029/54/112 once the burger grasp held).  The burger is lowered
        until its bottom is just above the other food's live top instead."""
        bowl = self._pan_bowl_center()
        food_z = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        held_off = food_z - float(np.asarray(hover_tcp.p, dtype=float)[2])
        food_half = 0.5 * float(np.min(_scaled_extents(spec.asset_id, spec.model_id)))
        support = max(float(bowl[2]), float(self._pan_food_support_top(actor)))
        target_tcp_z = max(support + food_half + self.ROBOT_BURGER_ABOVE_SUPPORT - held_off,
                           self._pan_rim_z() - 0.01)
        release_tcp = make_tcp(np.array([bowl[0], bowl[1], target_tcp_z]))
        if not self._move_tcp_cartesian(hover_tcp, release_tcp, arm_tag, n_steps=44, sim_per_step=22):
            return None
        target_xy = np.asarray(release_tcp.p, dtype=float)[:2]
        for _ in range(200):
            self.step_sim()
            tcp_xy = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose()[:2], dtype=float)
            if float(np.linalg.norm(tcp_xy - target_xy)) < 0.006:
                break
        self.open_gripper(arm_tag, num_steps=100)
        for _ in range(180):
            self.step_sim()
        return release_tcp

    def _park_arm_for_human(self, arm_tag: str) -> bool:
        """Full-state expert: move the arm back over its base (mplib) before
        the human's pick-and-place starts.  A failed plan keeps the arm
        where it is."""
        arm = self.robot.get_arm(arm_tag)
        self._expert_home_qpos = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
        base = np.asarray(arm.origin_pose.p, dtype=float)
        dx, dy, dz = self.EXPERT_PARK_TCP_FROM_BASE
        park = self._top_down_tcp(np.array([
            base[0] + dx, base[1] + dy, self.TABLE_TOP_Z + dz,
        ]))
        ok = self.move_and_execute(
            tcp_to_link_pose(park, arm.tcp_offset).to_pose7(), arm_tag,
        ) is not None
        self._expert_parked_for_human = bool(ok)
        return ok

    def play_blind_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        seed = int(self.config.get("eval_seed", 0))
        rng = np.random.RandomState(seed + 50005)
        order = [int(i) for i in rng.permutation(len(self.food_actors))]
        self._blind_robot_food_order = order
        self._blind_robot_food_attempts = []
        self._robot_motion_failed = False
        for _ in range(60):
            self.step_sim()

        if not self._pan_starts_on_cooktop() and not self._place_pan_on_cooktop(
            arm_tag
        ):
            return False

        if (
            self.avatar is not None
            and bool(self.config.get("eval_mode", False))
            and not getattr(self, "_eval_assist_fired", False)
            and not getattr(self, "_eval_assist_failed", False)
        ):
            self._eval_at_step(int(getattr(
                self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN,
            )))

        ok_any = False
        # With a human the blind expert commits to ONE food, the first of its
        # private random order (independent of the human's coin): no second
        # pick after watching what the human placed.  Robot-only debug
        # configs still place every food.
        commit = order[:1] if self.avatar is not None else order
        for attempt_i, idx in enumerate(commit):
            actor, spec = self.food_actors[idx]
            already_done = self._food_in_skillet(actor, spec)
            self._blind_robot_food_attempts.append({
                "idx": int(idx),
                "name": spec.name,
                "already_in_skillet": bool(already_done),
            })
            if already_done and attempt_i > 0:
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok
            if self.check_success() or getattr(self, "_robot_motion_failed", False):
                break
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(180):
            self.step_sim()
        return ok_any

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        metrics["exempt_human_contact"] = self.exempt_human_contact_metrics()
        metrics["human_step_back"] = getattr(self, "_human_step_back", None)
        return metrics
