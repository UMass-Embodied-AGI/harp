"""Contribution scenario for the categorize family (``categorize_objects_contribution``).

Shared-pool layout (50-50 Contribution): two loose objects of different categories
lie in ONE shared strip in front of the three open boxes, reachable by both
the robot and the human.  There is no robot side / human side split and no
fixed allocation:

* the human (``CategorizeCooperativeController``) picks one of the two
  objects uniformly at random at reset and goes for it, no matter what;
* the full-state expert (``play_once``) reads the human's choice and sorts
  the other object concurrently;
* the blind expert (``play_blind_once``) commits to the object nearest its
  gripper at t=0 and never reacts to the human (~50% same target).

Success requires both shared objects in their matching boxes and no
robot-avatar collision.  The no-human mapping (``categorize``) keeps the
legacy robot-only layout and rollout from ``CategorizeTaskCore`` unchanged.
"""

import numpy as np

from ..interaction.categorize import CategorizeCooperativeController
from ..base_task_parts.grasp_select import mplib_world_link_pose
from ..grasp import tcp_to_link_pose
from ..planning.base import PlanResult
from ..object_catalog import object_set_pairs
from ..task_bases.categorize import CategorizeTaskCore, PickSpec, PlaceSpec
from ..task_bases.categorize_support.cooperative_lift import cooperative_lift_check
from ..task_bases.categorize_support.motion import (
    CategorizeMotionRejected, stop_on_motion_rejection,
)
from ..task_bases.categorize_support.stack import add_supported_milk_stacks
from ..task_bases import categorize as _categorize_core
from ..utils import Pose


__all__ = ["CategorizeObjectsContribution", "PickSpec", "PlaceSpec"]


class CategorizeObjectsContribution(CategorizeTaskCore):
    """Robot and human sort a shared pool of two objects."""

    INSTRUCTION = "sort the objects into the matching baskets"
    use_avatar = True

    # Free-XY spawns can make the legacy single-yaw descent unreachable after
    # its pre-grasp transit. Check the two equivalent finger branches through
    # the shared selector before executing either leg, as for cube and wood.
    # The runtime radius candidates preserve grasp depth and the jaw axis.
    RADIUS_SELECTOR_MODELS = frozenset({
        ("048_stapler", 0), ("031_jam-jar", 0), ("100_seal", 1),
    })

    # ------------------------------------------------------------------
    # Shared-pool layout (human present only; the no-human mapping keeps
    # the legacy layout).
    # ------------------------------------------------------------------
    # Categories: the three objects whose categorize robot grasps are
    # already reliable (screened100 full-state robot placement: cube 22/22,
    # rubikscube@1 20/21, woodenblock@1 53/54).  Two distinct ones are drawn
    # per episode; each has its own open box.
    SHARED_POOL_OBJECTS = ("cube", "073_rubikscube@1", "086_woodenblock@1")
    SHARED_POOL_SIZE = 2
    # Geometry.  Robot base (legacy y=-0.35) and human (y=0.58) face each
    # other across ~1 m, so the shared strip sits mid-table and the two boxes
    # flank it left/right at the same depth.  Constraints this satisfies:
    #  * strip 0.40-0.50 m from the robot base (top-down picks fold up below
    #    ~0.35 m) and within ~0.55 m of the human's shoulder (FABRIK palm);
    #  * box drops <= ~0.6 m from the robot base;
    #  * no box in front of / behind the strip: a top-down Franka hand is
    #    ~0.21 m long along world y and collides with an inflated box wall
    #    within ~0.22 m in y.
    SHARED_BOX_XS = (-0.36, 0.36)
    SHARED_BOX_Y = 0.15
    SHARED_BOX_JITTER = 0.02
    # Shared strip between the boxes (x_min, x_max, y_min, y_max).
    SHARED_REGION = (-0.15, 0.15, 0.15, 0.22)
    SHARED_MIN_SEP = 0.15
    # Each object spawns on the strip half nearest its own box (carries stay
    # <= ~0.36 m; cross-body carries of the block slipped and long screw
    # transports in front of the base failed).  Which object the human or
    # the robot goes for is still open: both halves are reachable by both.
    SHARED_NEAR_BOX_SIDE = True
    SHARED_ROBOT_Y = -0.35
    SHARED_SAFE_DY = 0.30
    # Avatar stands a little closer than the legacy 0.67 so the shared strip
    # is inside FABRIK palm reach.
    avatar_init_pos = np.array([0.0, 0.58, -0.18])

    # Bounded wait for the human to finish after the robot's own object.
    HUMAN_WAIT_MAX_STEPS = 2400

    @property
    def _shared_pool_active(self) -> bool:
        return not self.no_human_enabled()

    def _categorize_yaml_preflight(self, pick, place, arm_tag, safe_z):
        existing = super()._categorize_yaml_preflight(pick, place, arm_tag, safe_z)
        return cooperative_lift_check(self, pick, arm_tag, safe_z, existing)

    def create_interaction_controller(self) -> CategorizeCooperativeController:
        return CategorizeCooperativeController()

    def _cooperative_controller(self) -> CategorizeCooperativeController:
        controller = self.interaction_controller
        if not isinstance(controller, CategorizeCooperativeController):
            raise TypeError("task does not use categorize cooperative behavior")
        return controller

    def _avatar_pick_and_place(self, actor, obj_name, model_id, basket_idx,
                                wait=True,
                                approach_frames=320, transport_frames=400,
                                retract_frames=320):
        return self._cooperative_controller().avatar_pick_and_place(
            self, actor, obj_name, model_id, basket_idx, wait=wait,
            approach_frames=approach_frames,
            transport_frames=transport_frames,
            retract_frames=retract_frames,
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    # Franka "ready" pose (as place_dual_shoes): the default MJCF home parks
    # the fingers at about (0, 0.10, 0.89) -- right over the shared strip,
    # inside the objects' planning cloud and the human's reach.  The ready
    # pose starts the TCP near the scripted safe pose above the robot side.
    SHARED_FRANKA_HOME = (0.0, -np.pi / 4, 0.0, -3.0 * np.pi / 4,
                          0.0, np.pi / 2, np.pi / 4)

    def _load_robot(self):
        if self._shared_pool_active:
            self.config.setdefault("robot_type", "franka")
            if self.config["robot_type"] == "franka":
                kwargs = self.config.setdefault("robot_kwargs", {})
                kwargs.setdefault("pos", [0.0, float(self.SHARED_ROBOT_Y), 0.75])
        super()._load_robot()
        if (self._shared_pool_active
                and self.config.get("robot_type") == "franka"
                and getattr(self.robot, "right_arm", None) is not None):
            self.robot.right_arm.homestate = list(self.SHARED_FRANKA_HOME)

    def _move_to_safe(self, arm_tag):
        """Shared pool: return to the fixed home joint configuration.

        A pose-space safe target lets MPLib pick any wrist-roll branch; from
        some branches the next top-down grasp needs a >3.5 rad joint swing and
        is rejected.  Planning in joint space to the home configuration keeps
        every pick starting from the same branch.  Falls back to the legacy
        pose target when the joint-space plan is unavailable.
        """
        if not self._shared_pool_active:
            return super()._move_to_safe(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        home = np.asarray(arm.homestate, dtype=float)[:arm.n_arm]
        endpoint = mplib_world_link_pose(planner, home) if hasattr(planner, "_mplib") else None
        if endpoint is not None:
            _arm, current, target = self._categorize_inputs(endpoint.to_pose7(), arm_tag)
            if float(np.max(np.abs(current - home))) < 0.01:
                return True
            for _attempt in range(3):
                try:
                    result = planner._mplib.plan_qpos(
                        goal_qposes=[planner._pad_qpos(home.copy())],
                        current_qpos=planner._pad_qpos(current.copy()),
                        time_step=1 / 250,
                        planning_time=getattr(planner, "_timeout", None) or 10,
                        rrt_range=0.3,
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"[categorize_coop] home plan_qpos exc: {exc}", flush=True)
                    result = None
                if not result or result.get("status") != "Success":
                    continue
                pos = np.asarray(result["position"], dtype=float)[:, :arm.n_arm]
                vel = np.asarray(result.get("velocity", np.zeros_like(pos)), dtype=float)[:, :arm.n_arm]
                plan = PlanResult(True, pos, vel)
                if self._categorize_plan_valid(arm, plan, current, target):
                    self._execute_categorize_plan(plan, arm_tag)
                    return True
        base = np.array(arm.origin_pose.p)
        safe_pos = np.array([base[0], base[1] + self.SHARED_SAFE_DY, self.TABLE_TOP_Z + 0.45])
        safe_link = tcp_to_link_pose(self._top_down_tcp(safe_pos), arm.tcp_offset)
        return self.move_and_execute(safe_link.to_pose7(), arm_tag)

    # Held lateral carries to a side box can run the wrist roll (joint 7)
    # into its limit along the straight screw line even though the endpoint
    # is reachable.  For a HELD object at transport height only, retry the
    # same checked endpoint with a joint-space path (same validation,
    # same arrival and held-drift checks).  Acquisition and lifts never use it.
    SHARED_TRANSPORT_RRT_FALLBACK = True
    SHARED_TRANSPORT_MIN_DZ = 0.30

    def _move_tcp_screw(self, target_tcp, arm_tag):
        plan_ok_before = self.plan_success
        try:
            return super()._move_tcp_screw(target_tcp, arm_tag)
        except CategorizeMotionRejected as exc:
            if (not self._shared_pool_active
                    or not self.SHARED_TRANSPORT_RRT_FALLBACK
                    or "straight MPLib path/FK" not in str(exc)
                    or getattr(self, "_categorize_hold_local", None) is None):
                raise
            arm = self.robot.get_arm(arm_tag)
            floor = self.TABLE_TOP_Z + self.SHARED_TRANSPORT_MIN_DZ
            if (float(arm.get_ee_pose()[2]) < floor or float(target_tcp.p[2]) < floor):
                raise
            target = tcp_to_link_pose(target_tcp, arm.tcp_offset).to_pose7()
            _arm, current, target = self._categorize_inputs(target, arm_tag)
            candidates = []
            # 1) Joint-linear path to the IK solution closest to the current
            #    configuration (keeps the elbow/wrist branch); 2) MPLib RRT.
            ik = arm.planner.solve_ik(current.copy(), target, num_waypoints=2, log=False)
            if ik.success:
                goal = np.asarray(ik.position[-1], dtype=float)
                n_wp = max(50, int(np.ceil(float(np.max(np.abs(goal - current))) / 0.004)) + 1)
                candidates.append(lambda n=n_wp: arm.planner.solve_ik(
                    current.copy(), target, num_waypoints=n, log=False))
            candidates += [lambda: arm.planner.plan_path(current.copy(), target)] * 3
            for make in candidates:
                result = make()
                if self._categorize_plan_valid(arm, result, current, target):
                    print("[categorize_coop] held transport: screw blocked, "
                          "using checked joint-space path", flush=True)
                    self.plan_success = plan_ok_before
                    self._execute_categorize_plan(result, arm_tag)
                    self._categorize_arrival(target, arm_tag)
                    return True
            raise

    # Grasp branch preference: equivalent top-down candidates (half-turn
    # finger swaps / flips) differ in wrist roll (joint 7).  A side-box carry
    # rotates the base joint by up to ~1 rad and the wrist must counter-rotate
    # to hold the object's yaw, so a grasp that starts near the +-2.9 rad
    # joint-7 limit cannot complete the straight carry.  Prefer branches with
    # |q7| below this soft limit; fall back to the unrestricted selector.
    SHARED_WRIST_SOFT_LIMIT = 2.0

    def select_and_execute_grasp(self, *args, **kwargs):
        if not self._shared_pool_active:
            return super().select_and_execute_grasp(*args, **kwargs)
        extra = kwargs.pop("post_plan_check", None)
        limit = float(self.SHARED_WRIST_SOFT_LIMIT)

        def wrist_ok(plans):
            if extra is not None and not extra(plans):
                return False
            grasp = plans[1]
            try:
                q7 = float(np.asarray(grasp.position)[-1][6])
            except Exception:
                return True
            if abs(q7) > limit:
                print(f"[categorize_coop] grasp branch q7={q7:.2f} near the wrist "
                      f"limit; trying the next branch", flush=True)
                return False
            return True

        result = super().select_and_execute_grasp(*args, post_plan_check=wrist_ok, **kwargs)
        if result is None:
            result = super().select_and_execute_grasp(*args, post_plan_check=extra, **kwargs)
        return result

    def load_actors(self):
        if not self._shared_pool_active:
            return super().load_actors()
        return self._load_shared_pool_actors()

    def _build_open_box(self, bx, by):
        box_half_x, box_half_y = self._BASKET_HALF_X, self._BASKET_HALF_Y
        box_half_z, wall_t = self._BASKET_HALF_Z, 0.005
        color = (0.7, 0.5, 0.3)
        cz = self.TABLE_TOP_Z + box_half_z
        self.basket_poses.append(Pose([bx, by, cz], np.array([1.0, 0.0, 0.0, 0.0])))
        _categorize_core.create_primitive(
            self.scene, "box", Pose(p=[bx, by, cz - box_half_z + wall_t]),
            size={"half_size": (box_half_x, box_half_y, wall_t)},
            color=color, is_static=True,
        )
        for wx, wy, hx, hy in [
            (-box_half_x + wall_t, 0.0, wall_t, box_half_y),
            (+box_half_x - wall_t, 0.0, wall_t, box_half_y),
            (0.0, -box_half_y + wall_t, box_half_x, wall_t),
            (0.0, +box_half_y - wall_t, box_half_x, wall_t),
        ]:
            _categorize_core.create_primitive(
                self.scene, "box", Pose(p=[bx + wx, by + wy, cz]),
                size={"half_size": (hx, hy, box_half_z)},
                color=color, is_static=True,
            )

    def _sample_shared_xy(self, name, box_x, placed, max_tries=400):
        xmin, xmax, ymin, ymax = self.SHARED_REGION
        lo, hi = xmin, xmax
        if self.SHARED_NEAR_BOX_SIDE:
            lo, hi = (0.0, xmax) if box_x > 0 else (xmin, 0.0)
        for _ in range(max_tries):
            x = float(np.random.uniform(lo, hi))
            y = float(np.random.uniform(ymin, ymax))
            if all(np.hypot(x - px, y - py) >= self.SHARED_MIN_SEP for px, py in placed):
                return x, y
        # Deterministic fallback: opposite ends of the strip.
        x = xmin if len(placed) % 2 == 0 else xmax
        return float(x), float(0.5 * (ymin + ymax))

    def _load_shared_pool_actors(self):
        """Two open boxes (one per category) and two shared loose objects.

        Per-seed draws (global np.random, seeded by BaseTask.reset): the two
        categories, the box side of each, box jitter, then each object's XY
        in the shared strip.
        """
        table_top = self.TABLE_TOP_Z + 0.02
        upright_q = np.asarray(self.SPAWN_UPRIGHT_QUAT, dtype=float)
        pool = object_set_pairs(list(self.SHARED_POOL_OBJECTS))
        chosen = np.random.choice(len(pool), self.SHARED_POOL_SIZE, replace=False)
        cats = [pool[int(i)] for i in chosen]
        self._cats = cats
        basket_perm = np.random.permutation(len(cats))
        box_of_cat = np.argsort(basket_perm)
        jitter = np.random.uniform(-self.SHARED_BOX_JITTER, self.SHARED_BOX_JITTER,
                                   size=(len(cats), 2))
        box_xy = [(float(self.SHARED_BOX_XS[i] + jitter[i, 0]),
                   float(self.SHARED_BOX_Y + jitter[i, 1])) for i in range(len(cats))]
        self.basket_poses = []
        self.basket_actors = []
        for bx, by in box_xy:
            self._build_open_box(bx, by)
        self._example_actors = []

        order = list(range(len(cats)))
        xy = [None] * len(cats)
        placed = []
        for k in order:
            xy[k] = self._sample_shared_xy(cats[k][0], box_xy[int(box_of_cat[k])][0], placed)
            placed.append(xy[k])

        self._sort_targets = []
        for k, (obj_name, model_id) in enumerate(cats):
            x, y = xy[k]
            actor = self._spawn_categorize_object(obj_name, model_id, x, y, table_top, upright_q)
            self._sort_targets.append((actor, obj_name, model_id, int(box_of_cat[k])))
        # No side-specific copies in the shared-pool layout.
        self._avatar_sort_targets = []
        self._clutter_actors = []
        print(f"[categorize_coop] shared pool: "
              f"{[(n, round(x, 3), round(y, 3), b) for (_a, n, _m, b), (x, y) in zip(self._sort_targets, xy)]}"
              f" boxes={[(round(x, 3), round(y, 3)) for x, y in box_xy]}", flush=True)

    # ------------------------------------------------------------------
    # Eval-mode hooks (loop pattern) — owned by the controller.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        self._cooperative_controller().eval_reset(self)

    def _eval_at_step(self, step_idx: int):
        self._cooperative_controller().eval_step(self, step_idx)

    def _eval_fire_next_avatar_motion(self):
        self._cooperative_controller().fire_next_avatar_motion(self)

    # ------------------------------------------------------------------
    # Robot primitives
    # ------------------------------------------------------------------
    def _robot_sort(self, idx, arm_tag="right", targets=None) -> bool:
        targets = self._sort_targets if targets is None else targets
        actor, obj_name, model_id, basket_idx = targets[idx]
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

    def _robot_rule_target(self, arm_tag="right"):
        """Blind rule, independent of the human: the object nearest (xy) to
        the gripper at t=0; ties -> lower index."""
        tcp = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose()[:3], dtype=float)
        ctrl = self._cooperative_controller()
        return min(
            range(len(self._sort_targets)),
            key=lambda i: (round(float(np.linalg.norm(
                ctrl.object_center(self, i)[:2] - tcp[:2])), 4), i),
        )

    def _settle(self, steps=200):
        for _ in range(steps):
            self.step_sim()

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    def _start_human_if_needed(self):
        ctrl = self._cooperative_controller()
        if self.avatar is not None and not ctrl.human_started:
            ctrl.start_human(self)

    @stop_on_motion_rejection
    def play_once(self) -> bool:
        """Full-state expert: reads the human's choice, sorts the OTHER object
        concurrently with the human (no ordering, no waiting)."""
        if not self._shared_pool_active:
            return self._play_once_no_human()
        arm_tag = "right"
        ctrl = self._cooperative_controller()
        self._boost_arm_pd()
        self._open_basket_lids()
        self._start_human_if_needed()
        human = ctrl.human_target
        ok = True
        for idx in range(len(self._sort_targets)):
            if human is not None and idx == int(human):
                continue
            ctrl.note_robot_target(idx)
            ok = self._robot_sort(idx, arm_tag) and ok
        ctrl.wait_for_human(self, self.HUMAN_WAIT_MAX_STEPS)
        self._settle()
        return ok

    @stop_on_motion_rejection
    def play_blind_once(self) -> bool:
        """Blind expert: commits to the object nearest its gripper at t=0
        and never reads, waits for, or reacts to the human.  It still plans
        with the human's current body in the planning cloud."""
        if not self._shared_pool_active:
            return super().play_blind_once()
        arm_tag = "right"
        ctrl = self._cooperative_controller()
        self._boost_arm_pd()
        self._open_basket_lids()
        # In eval mode the controller already started the human at reset.
        self._start_human_if_needed()
        target = self._robot_rule_target(arm_tag)
        ctrl.note_robot_target(target)
        ok = self._robot_sort(target, arm_tag)
        ctrl.wait_for_human(self, self.HUMAN_WAIT_MAX_STEPS)
        self._settle()
        return ok

    def _play_once_no_human(self) -> bool:
        """Legacy robot-only rollout (no-human ``categorize`` mapping)."""
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()
        self._move_to_safe(arm_tag)
        any_success = False
        for idx in range(len(self._sort_targets)):
            any_success = self._robot_sort(idx, arm_tag) or any_success
        if not self.no_human_enabled():
            for idx in range(len(getattr(self, "_avatar_sort_targets", []))):
                any_success = self._robot_sort(
                    idx, arm_tag, targets=self._avatar_sort_targets,
                ) or any_success
        self._settle()
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    def _success_target_groups(self):
        # The shared-pool layout publishes no side-specific copies; a legacy
        # robot/avatar split (scene replay, fixtures) keeps its two groups.
        if self._shared_pool_active and not getattr(self, "_avatar_sort_targets", None):
            return [("shared", self._sort_targets)]
        groups = [("robot", self._sort_targets)]
        if not self.no_human_enabled():
            groups.append(("avatar", self._avatar_sort_targets))
        return groups

    def _basket_metrics(self) -> dict:
        return add_supported_milk_stacks(self, super()._basket_metrics())

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        if self.avatar_collision_checker is not None and self.avatar_collided:
            return False
        return bool(self._basket_metrics()["basket_all_correct"])

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
