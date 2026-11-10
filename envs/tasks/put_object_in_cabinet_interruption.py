"""Put object in cabinet (Interruption, ``put_object_in_cabinet_interruption``;
single Franka + avatar).

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

from contextlib import contextmanager

import numpy as np

from ..avatar.inspect_motion_mixin import INSPECT_MOTION_SPECS
from ..genesis_compat import hold_dof_position_compat
from ..grasp import GraspPose, tcp_to_link_pose
from ..interaction.interrupt_progress import interrupt_episode, interrupt_fired
from ..interaction.put_object_cabinet import CabinetInterruptController
from ..manipulation import PickSpec, PlaceSpec
from ..planning.base import PlanResult
from ..task_bases.put_object_cabinet import (
    CabinetTaskCore, _cabinet_motion_path, _cabinet_motion_pose_valid,
)
from ..utils import Pose, to_numpy
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class _CabinetMotionRejected(RuntimeError):
    """An expected planning rejection, distinct from a task runtime error."""


def _cabinet_plan_on_sim_clock(plan, sim_dt, gripper=None):
    """Hold every native4ms sample for its planned physical duration."""
    if not np.isfinite(sim_dt) or sim_dt <= 0:
        raise _CabinetMotionRejected("invalid cabinet execution clock")
    ratio = .004 / float(sim_dt)
    repeats = int(round(ratio))
    if repeats < 1 or repeats > 16 or not np.isclose(ratio, repeats, rtol=0, atol=1e-8):
        raise _CabinetMotionRejected("cabinet clock must divide native4ms samples")
    pos = np.asarray(plan.position, dtype=float)
    vel = np.asarray(plan.velocity, dtype=float)
    if (not plan.success or pos.ndim != 2 or not len(pos)
            or vel.shape != pos.shape or not np.all(np.isfinite(pos))
            or not np.all(np.isfinite(vel))):
        raise _CabinetMotionRejected("invalid cabinet timed trajectory")
    if gripper is not None:
        gripper = np.asarray(gripper, dtype=float)
        if gripper.shape != (len(pos),) or not np.all(np.isfinite(gripper)):
            raise _CabinetMotionRejected("invalid cabinet gripper timeline")
    if repeats == 1:
        return plan, gripper
    # Repetition preserves every checked configuration and its ordering; no
    # new joint interpolation or unverified geometric shortcut is introduced.
    return (PlanResult(True, np.repeat(pos, repeats, axis=0),
                       np.repeat(vel, repeats, axis=0)),
            None if gripper is None else np.repeat(gripper, repeats))


# Body payloads whose descent may use the checked joint-space route when the
# Cartesian screw cannot integrate it. Native probes on seeds 16/19: the
# release pose has 32/37 same-branch IK solutions (.67/.87 rad away) and
# plan_qpos routes with 6-8 mm lateral drift while the screw fails outright.
_BODY_DESCENT_FALLBACK_OBJECTS = ("048_stapler", "086_woodenblock", "035_apple", "031_jam-jar", "038_milk-box")


def _collision_pairs(hits):
    """Compact link/object names from native collision results (diagnostic text only)."""
    names = []
    for hit in list(hits)[:6]:
        parts = [str(getattr(hit, key)) for key in ("link_name1", "object_name1", "link_name2", "object_name2")
                 if getattr(hit, key, None) not in (None, "")]
        names.append("/".join(parts) if parts else str(hit)[:60])
    return "; ".join(names) if names else "unnamed"


class PutObjectInCabinetInterruption(InspectMotionInterruptFacade, CabinetTaskCore):
    """Robot opens the top drawer of a cabinet, picks an object off the
    table after a human inspection, and drops it inside the open drawer."""

    INSTRUCTION = "place the object inside the cabinet drawer after the human inspects it"
    # Shared tabletop pool minus 038_milk-box (see object_catalog).
    OBJECT_SET = "cabinet_interrupt_pick_pool"
    # The open-drawer phase is NOT physically driven on this single-arm
    # config — the cabinet handle's IK lands on a Franka-7DOF qpos that
    # mplib + Genesis IK can't solve consistently (repeated attempts
    # left the arm 15-30 cm off target with non-deterministic
    # joint solutions across runs).  Instead the drawer is pre-opened
    # via cabinet.set_qpos at scene init (kinematic control is allowed
    # for non-robot articulations) and the robot's job is to
    # pick the table object and drop it in the open drawer.
    use_avatar = True

    _CABINET_TASK_NAME = "put_object_in_cabinet_interruption"
    _CABINET_ALLOW_TABLE_VARIANTS = False

    # Interrupt-only family strategy (see the core's switch comments): pair
    # friction 1.0 for the payload/drawer-floor proxy, largest-extent drawer
    # opening, and the waypoint/IK drawer pull instead of the Franka
    # translation pull.
    CABINET_PAIR_FRICTION = 1.0
    DRAWER_OPENING_MAX_EXTENT = True
    DRAWER_TRANSLATION_PULL = False

    # Avatar approaches from the +Y side of the table, facing -Y.  This
    # keeps the body out of the back-right cabinet/drop corridor while
    # still letting the right hand reach the front table object.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # Use the shorter inspect clip by default so calibration/rollout stays
    # tractable; randomize_inspect can still sample the legacy long clip.
    INSPECT_POOL = [
        INSPECT_MOTION_SPECS["Inspect2"].as_pool_entry(),
        INSPECT_MOTION_SPECS["Inspect1"].as_pool_entry(),
        INSPECT_MOTION_SPECS["Inspect4"].as_pool_entry(),
        {"name": "Inspect2_touch_hold_reverse", "attach": 31, "detach": 31, "attach_object": False},
        {"name": "Inspect1_touch_hold_reverse", "attach": 34, "detach": 34, "attach_object": False},
    ]
    INSPECT_MOTION_NAME = "Inspect2"
    INSPECT_ATTACH_FRAME = 31
    INSPECT_DETACH_FRAME = 84
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04
    # Clear the hand from the released object before the robot closes around
    # it.  The object detaches at its authored frame; only the avatar pose
    # transitions back to idle over these frames.
    INSPECT_RETURN_TO_IDLE_FRAMES = 30

    # Eval mode (progress-trigger interrupt contract, envs/interaction/
    # interrupt_progress.py): the human interrupts when the robot commits to
    # the table object after the drawer is open (progress trigger, no time
    # fallback), glides in, inspects, and returns to idle.  The bounds below
    # belong to the retired random-step trigger; only its RNG draw remains.
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    # Fitted to the demonstrated intent hover (TABLE_TOP_Z + 0.20 m, 0.15-0.18
    # m from the object, progress 0.40-0.50): 0.5 fired only after the hover,
    # on the grasp descent; 0.30 fires 5-8 steps before the demo human starts
    # on all 4 probed demos.
    INTERRUPT_COMMIT_RADIUS_M = 0.22
    INTERRUPT_TRIGGER_PROGRESS = 0.30
    INTERRUPT_MIN_POLICY_STEPS = 20
    # Phase gate: drawer joint opened at least this far (nominal open qpos).
    INTERRUPT_DRAWER_OPEN_QPOS = 0.050
    # Payloads include a milk box; widen the "held" test accordingly.
    INTERRUPT_HELD_TCP_RADIUS_M = 0.08

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)
        self._spawn_x_range = tuple(float(v) for v in cfg.get("object_spawn_x_range", self._SPAWN_X))
        self._spawn_y_range = tuple(float(v) for v in cfg.get("object_spawn_y_range", self._SPAWN_Y))
        self.EVAL_TRIGGER_STEP_MIN = int(cfg.get(
            "eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
        self.EVAL_TRIGGER_STEP_MAX = int(cfg.get(
            "eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))

    def create_interaction_controller(self):
        return CabinetInterruptController()

    def _interrupt_controller(self) -> CabinetInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, CabinetInterruptController):
            raise TypeError("task does not use cabinet interrupt behavior")
        return controller

    # Reset ordering (inspect resolve, drawer pick, eval arming or calibration)
    # lives in the controller's after_base_reset_observation hook.

    def _reject_cabinet_motion(self, reason):
        self.plan_success = False
        print(f"[cabinet-motion] {reason}", flush=True)
        if getattr(self, "_cabinet_pick_active", False):
            raise _CabinetMotionRejected(reason)
        return None

    def pick_and_place(self, pick, place, arm_tag):
        """Stop a rejected motion even where the shared helper ignores None."""
        previous = getattr(self, "_cabinet_pick_active", False)
        self._cabinet_pick_active = True
        try:
            if ((getattr(self, "_object_id", None), getattr(self, "_object_model_id", None))
                    in (("038_milk-box", 0), ("031_jam-jar", 0), ("035_apple", 1),
                        ("048_stapler", 0), ("086_woodenblock", 1))
                    and self.config.get("robot_type") == "franka"
                    and place.support_z is not None and place.release):
                with self._body_acquisition_obstacles(arm_tag):
                    return self._pick_and_place_body(pick, place, arm_tag)
            return super().pick_and_place(pick, place, arm_tag)
        except _CabinetMotionRejected:
            self.plan_success = False
            return False
        finally:
            self._cabinet_pick_active = previous

    def execute_plan(self, result, arm_tag, gripper_result=None):
        if (result is not None and getattr(self, "_cabinet_pick_active", False)
                and not getattr(self, "_cabinet_plan_is_physical_clock", False)):
            arm = self.robot.get_arm(arm_tag)
            if not hasattr(arm.planner, "_mplib"):
                return self._reject_cabinet_motion("payload timing requires native MPLib")
            result, gripper_result = _cabinet_plan_on_sim_clock(
                result, float(self.scene.sim.dt), gripper_result,
            )
        return super().execute_plan(result, arm_tag, gripper_result=gripper_result)

    @contextmanager
    def _body_acquisition_obstacles(self, arm_tag):
        """Keep fixture geometry alongside avatar clouds during acquisition."""
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject

        if getattr(self, "_body_fixture_scene", None) is not None:
            raise _CabinetMotionRejected("nested body acquisition scene")
        planner = self.robot.get_arm(arm_tag).planner
        if not hasattr(planner, "_mplib"):
            raise _CabinetMotionRejected("body grasp requires native MPLib")
        backend = planner._mplib
        world = backend.planning_world
        names = ("cabinet_body_acquisition_table", "cabinet_body_acquisition_fixture")
        boxes = []
        for name, entity in zip(names, (self.table, self.cabinet.entity)):
            bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
            if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                    or np.any(bounds[1] <= bounds[0]) or world.has_object(name)
                    or backend.acm.has_entry(name)):
                raise _CabinetMotionRejected("invalid/conflicting acquisition fixture")
            boxes.append(CollisionObject(Box(*(bounds[1] - bounds[0])),
                MplibPose(p=bounds.mean(axis=0), q=np.array([1., 0, 0, 0]))))
        state = {"backend": backend, "names": [], "arm_tag": arm_tag}
        self._body_fixture_scene = state
        try:
            for name, box in zip(names, boxes):
                world.add_object(name, box)
                state["names"].append(name)
            # The fixed base is mounted through the tabletop. No moving link
            # or avatar pair is excluded; remove this local entry on exit.
            backend.acm.set_entry("panda_link0", names[0], True)
            yield
        finally:
            self._clear_body_acquisition_obstacles()

    def _clear_body_acquisition_obstacles(self):
        state = getattr(self, "_body_fixture_scene", None)
        if state is None:
            return
        backend = state["backend"]
        try:
            for name in state["names"]:
                backend.planning_world.remove_object(name)
                backend.acm.remove_entry(name)
        finally:
            self._body_fixture_scene = None

    def _body_path_clear(self, positions, arm_tag):
        state = getattr(self, "_body_fixture_scene", None)
        if state is None:
            return True
        if state["arm_tag"] != arm_tag:
            print("[cabinet-preflight] acquisition fixture arm mismatch", flush=True)
            return False
        arm = self.robot.get_arm(arm_tag)
        # Closed padding misses open-finger/table interference. Check the
        # full80mm opening throughout acquisition and vertical lift; this
        # conservative envelope also contains the closed transport fingers.
        for waypoint, q in enumerate(positions):
            full = arm.planner._pad_qpos(np.asarray(q, dtype=float))
            if full.shape != (9,) or not np.all(np.isfinite(full)):
                print(f"[cabinet-preflight] invalid full joints at waypoint {waypoint}", flush=True)
                return False
            full[7:9] = .04
            collisions = state["backend"].check_for_env_collision(full)
            if collisions:
                pairs = sorted({(str(getattr(c, "link_name1", "unknown")),
                                 str(getattr(c, "object_name2", "unknown")),
                                 str(getattr(c, "link_name2", "unknown"))) for c in collisions})
                print(f"[cabinet-preflight] fixture/avatar collision at waypoint {waypoint}: {pairs}", flush=True)
                return False
        return True

    def _payload_arrived_pose(self, target, arm_tag, label, *, tiny=False):
        raw = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        if (raw.shape != (7,) or not np.all(np.isfinite(raw))
                or not np.isclose(np.linalg.norm(raw[3:]), 1.0, atol=1e-4)):
            return self._reject_cabinet_motion(f"payload {label}: invalid measured TCP")
        actual = Pose.from_pose7(raw)
        distance = float(np.linalg.norm(actual.p - target.p))
        angle = 2 * float(np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1)))
        if distance > (0.01 if tiny else 0.015) or angle > (0.05 if tiny else np.deg2rad(10)):
            return self._reject_cabinet_motion(
                f"payload {label}: arrival xyz={distance:.4f}m, angle={angle:.4f}rad",
            )
        return actual

    def _payload_move_pose(self, target, arm_tag, label, *, tiny=False, reseed=True):
        """Checked native screw motion for payload transfer and withdrawal."""
        arm = self.robot.get_arm(arm_tag)
        if (not _cabinet_motion_pose_valid(target)
                or not _cabinet_motion_pose_valid(arm.tcp_offset)
                or not hasattr(arm.planner, "_mplib")):
            return self._reject_cabinet_motion(f"payload {label}: invalid native motion")
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            return self._reject_cabinet_motion(f"payload {label}: invalid joint seed")
        self._refresh_avatar_planner_obstacles(arm_tag)
        link = tcp_to_link_pose(target, arm.tcp_offset)
        try:
            with self._body_transfer_profile(arm, tiny=tiny):
                plan = arm.planner.plan_screw_path(seed.copy(), link.to_pose7(), qpos_step=0.02)
                # The coarse screw step can fail natively, or jump a joint by
                # more than the continuity bound near a wrist singularity, on a
                # short payload leg. Re-solve the same straight leg more finely
                # from the same measured seed; every check below still applies.
                for finer in (0.01, 0.005):
                    if self._payload_screw_continuous(plan, seed, tiny=tiny):
                        break
                    candidate = arm.planner.plan_screw_path(seed.copy(), link.to_pose7(), qpos_step=finer)
                    if candidate is not None and candidate.success:
                        plan = candidate
                # The empty-hand release retreat is object independent (e.g. the
                # Rubik's cube uses the mixin pick path and never reaches the
                # descent route, but still needs the retreat).
                if ((plan is None or not plan.success) and not tiny
                        and self.config.get("robot_type") == "franka"
                        and (label == "release retreat"
                             or (label == "descent"
                                 and getattr(self, "_object_id", None) in _BODY_DESCENT_FALLBACK_OBJECTS))):
                    plan = self._stapler_descent_plan(seed, target, arm_tag)
        except RuntimeError as exc:
            return self._reject_cabinet_motion(f"payload {label}: screw rejected: {exc}")
        if plan is None or not plan.success:
            return self._reject_cabinet_motion(f"payload {label}: screw unavailable")
        path = _cabinet_motion_path(plan.position, plan.velocity, arm.n_arm)
        if path is None:
            return self._reject_cabinet_motion(f"payload {label}: invalid screw path")
        positions, _ = path
        if not self._body_path_clear(np.vstack([seed, positions]), arm_tag):
            return self._reject_cabinet_motion(f"payload {label}: fixture collision")
        delta = positions[-1] - seed
        if (np.linalg.norm(delta) > 3.5
                or (tiny and np.max(np.abs(delta)) > self._payload_tiny_joint_bound(seed, positions))
                or np.max(np.abs(np.diff(np.vstack([seed, positions]), axis=0))) > 0.30):
            return self._reject_cabinet_motion(f"payload {label}: joint continuity rejected")
        endpoint = self._drawer_plan_endpoint_tcp(arm_tag, positions[-1])
        if not _cabinet_motion_pose_valid(endpoint):
            return self._reject_cabinet_motion(f"payload {label}: invalid FK")
        distance = float(np.linalg.norm(endpoint.p - target.p))
        angle = 2 * float(np.arccos(np.clip(abs(np.dot(endpoint.q, target.q)), 0, 1)))
        if distance > (0.01 if tiny else 0.015) or angle > (0.05 if tiny else np.deg2rad(10)):
            return self._reject_cabinet_motion(
                f"payload {label}: endpoint xyz={distance:.4f}m, angle={angle:.4f}rad rejected",
            )
        self.execute_plan(plan, arm_tag)
        for _ in range(60):
            self.step_sim()
        if self._payload_arrived_pose(target, arm_tag, label, tiny=tiny) is None:
            return False
        if reseed:
            self._reseed_cached_arm_target(arm)
        print(f"[cabinet-payload] {label}: TCP={target.p}", flush=True)
        return True

    @staticmethod
    def _payload_tiny_joint_bound(seed, positions):
        """Largest joint change allowed over a 30 mm tiny lift.

        0.30 rad, or 0.60 rad when the native screw path is dense (every
        waypoint step <= 0.05 rad): near the elbow/wrist limits a smooth
        straight lift legitimately turns joints 3/5/7 by 0.3-0.6 rad, which
        is not the branch flip the bound guards against."""
        steps = np.abs(np.diff(np.vstack([seed, positions]), axis=0))
        return 0.60 if steps.size and float(np.max(steps)) <= 0.05 else 0.30

    @staticmethod
    def _payload_screw_continuous(plan, seed, *, tiny=False):
        """The joint-continuity gate of ``_payload_move_pose`` (no side effects)."""
        if plan is None or not plan.success:
            return False
        positions = np.asarray(plan.position, dtype=float)
        if positions.ndim != 2 or not len(positions) or not np.all(np.isfinite(positions)):
            return False
        if positions.shape[1] != len(seed):
            return False
        delta = positions[-1] - seed
        return not (np.linalg.norm(delta) > 3.5
                    or (tiny and np.max(np.abs(delta))
                        > PutObjectInCabinetInterruption._payload_tiny_joint_bound(seed, positions))
                    or np.max(np.abs(np.diff(np.vstack([seed, positions]), axis=0))) > 0.30)

    def _refine_open_body_grasp(self, target, arm_tag):
        """Correct measured open-jaw tracking bias before body acquisition."""
        entity = self.object.entity
        reference = Pose(to_numpy(entity.get_pos()), to_numpy(entity.get_quat()))
        if not _cabinet_motion_pose_valid(target) or not _cabinet_motion_pose_valid(reference):
            return self._reject_cabinet_motion("body alignment: invalid target/object pose")
        correction = np.zeros(3)
        for attempt in range(3):
            actual = self._payload_arrived_pose(target, arm_tag, "body alignment", tiny=True)
            if actual is None:
                return False
            current = Pose(to_numpy(entity.get_pos()), to_numpy(entity.get_quat()))
            if (not _cabinet_motion_pose_valid(current)
                    or np.linalg.norm(current.p - reference.p) > .003
                    or 2 * np.arccos(np.clip(abs(np.dot(current.q, reference.q)), 0, 1)) > .05):
                return self._reject_cabinet_motion("body alignment: object moved before closing")
            error = target.p - actual.p
            distance = float(np.linalg.norm(error))
            print(f"[cabinet-body-alignment] attempt={attempt}, error={distance:.6f}m", flush=True)
            if distance <= .001:
                return True
            if attempt == 2 or distance > .008:
                return self._reject_cabinet_motion("body alignment: measured target not reached")
            correction += error
            if np.linalg.norm(correction) > .012:
                return self._reject_cabinet_motion("body alignment: correction bound exceeded")
            # Keep the checked planned joint target while settling/closing.
            # Reseeding it to gravity-deflected joints would erase this bias.
            if not self._payload_move_pose(Pose(target.p + correction, target.q), arm_tag,
                                           "open grasp alignment", tiny=True, reseed=False):
                return False
            for _ in range(100):
                self.step_sim()

    def _body_descent_rejected(self, reason):
        print(f"[cabinet-body] descent route: {reason}", flush=True)
        return None

    def _stapler_descent_plan(self, seed, target, arm_tag):
        """Plan a bounded vertical leg (descent, or the release retreat)
        around the screw planner's joint-limit branch: same-column IK, checked
        plan_qpos, monotone toward the target, 10 mm lateral, .05 rad."""
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        start = self._drawer_plan_endpoint_tcp(arm_tag, seed)
        if (not _cabinet_motion_pose_valid(start)
                or abs(float(target.p[2]) - float(start.p[2])) < .005
                or np.linalg.norm(target.p[:2] - start.p[:2]) > .010
                or 2 * np.arccos(np.clip(abs(np.dot(target.q, start.q)), 0, 1)) > .05):
            return self._body_descent_rejected("route check failed before execution")
        link = tcp_to_link_pose(target, arm.tcp_offset)
        # Prefer the CLOSEST of 100 IK restarts. The 20-restart wrapper can
        # return a far (mirrored-branch) solution whose joint-space route swings
        # 15-60 mm laterally and fails the 10 mm bound, while the closest
        # solution's route swings 6 mm (native probes).
        ik = None
        if hasattr(getattr(planner, "_mplib", None), "IK"):
            backend = planner._mplib
            goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(link.to_pose7())
            status, solutions = backend.IK(goal_pose=goal, start_qpos=planner._pad_qpos(seed.copy()),
                                           n_init_qpos=100, threshold=.001, return_closest=True)
            if status == "Success" and solutions is not None:
                goal_q = np.asarray(solutions, dtype=float).ravel()[:arm.n_arm]
                if goal_q.shape == (arm.n_arm,) and np.all(np.isfinite(goal_q)):
                    ik = PlanResult(True, np.stack([seed, goal_q]), np.zeros((2, arm.n_arm)))
        if ik is None or not ik.success:
            ik = planner.solve_ik(seed.copy(), link.to_pose7(), num_waypoints=2, log=False)
        if ik is None or not ik.success:
            return self._body_descent_rejected("no IK solution for the release pose")
        path = _cabinet_motion_path(ik.position, ik.velocity, arm.n_arm)
        if path is None or np.linalg.norm(path[0][-1] - seed) > 3.5:
            return self._body_descent_rejected("IK solution is not in a continuous branch")
        result = planner._mplib.plan_qpos(
            goal_qposes=[planner._pad_qpos(path[0][-1])],
            current_qpos=planner._pad_qpos(seed), time_step=.004,
            planning_time=10, rrt_range=.15, verbose=False,
        )
        if not result or result.get("status") != "Success":
            return self._body_descent_rejected("route check failed before execution")
        path = _cabinet_motion_path(result.get("position"), result.get("velocity"), arm.n_arm)
        if path is None:
            return self._body_descent_rejected("route check failed before execution")
        positions, velocities = path
        complete = np.vstack([seed, positions])
        if (np.linalg.norm(positions[-1] - seed) > 3.5
                or np.max(np.abs(np.diff(complete, axis=0))) > .30):
            return self._body_descent_rejected("route check failed before execution")
        # Progress coordinate: increases toward the target in either direction.
        sign = -1.0 if float(target.p[2]) < float(start.p[2]) else 1.0
        progress = sign * float(start.p[2])
        goal_progress = sign * float(target.p[2])
        for q in complete:
            pose = self._drawer_plan_endpoint_tcp(arm_tag, q)
            if not _cabinet_motion_pose_valid(pose):
                return self._body_descent_rejected("path sample has invalid FK")
            angle = 2 * np.arccos(np.clip(abs(np.dot(pose.q, start.q)), 0, 1))
            lateral = float(np.linalg.norm(pose.p[:2] - start.p[:2]))
            current = sign * float(pose.p[2])
            if (lateral > .010 or angle > .05
                    or current < progress - .002 or current > goal_progress + .002):
                return self._body_descent_rejected(
                    f"path sample bound: lateral={lateral*1000:.1f}mm angle={angle:.4f}rad "
                    f"z={pose.p[2]:.4f} progress={sign*progress:.4f} target={target.p[2]:.4f} "
                    f"ikL2={np.linalg.norm(positions[-1]-seed):.3f}rad")
            progress = max(progress, current)
            full = planner._pad_qpos(q)
            if full.shape != (9,) or not np.all(np.isfinite(full)):
                return self._body_descent_rejected("path sample has an invalid full configuration")
            full[7:9] = .04
            hits = planner._mplib.check_for_self_collision(full)
            if hits:
                return self._body_descent_rejected("path sample self collision: " + _collision_pairs(hits))
            hits = planner._mplib.check_for_env_collision(full)
            if hits:
                return self._body_descent_rejected("path sample environment collision: " + _collision_pairs(hits))
        # Endpoint FK and measured arrival are checked by _payload_move_pose.
        return PlanResult(True, positions, velocities)

    @contextmanager
    def _body_transfer_profile(self, arm, *, tiny=False):
        """Explicit native timing for non-seal transfers after acquisition."""
        active = (not tiny and getattr(self, "_cabinet_pick_active", False)
                  and getattr(self, "_object_id", None) in
                  ("038_milk-box", "031_jam-jar", "035_apple", "048_stapler", "086_woodenblock")
                  and self.config.get("robot_type") == "franka")
        if not active:
            yield
            return
        backend = arm.planner._mplib
        velocity, acceleration = backend.joint_vel_limits, backend.joint_acc_limits
        vel, acc = np.asarray(velocity, dtype=float), np.asarray(acceleration, dtype=float)
        if (vel.shape != (arm.n_arm,) or acc.shape != vel.shape
                or not np.all(np.isfinite(vel)) or not np.all(np.isfinite(acc))
                or np.any(vel <= 0) or np.any(acc <= 0)):
            raise _CabinetMotionRejected("invalid native transfer limits")
        # Respect externally configured limits. This profile only replaces
        # MPLib0.2.1's documented unit defaults; no robot/global setting changes.
        if not (np.array_equal(vel, np.ones(arm.n_arm)) and np.array_equal(acc, np.ones(arm.n_arm))):
            yield
            return
        backend.joint_vel_limits = np.full(arm.n_arm, 2.)
        backend.joint_acc_limits = np.full(arm.n_arm, 4.)
        try:
            yield
        finally:
            backend.joint_vel_limits = velocity
            backend.joint_acc_limits = acceleration

    def _after_pick_place_release(self, pick, place, arm_tag, release_pos):
        """Clear the open fingers from drawer walls before final settling."""
        if place.support_z is None or not place.release:
            return
        raw = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        if (raw.shape != (7,) or not np.all(np.isfinite(raw))
                or not np.isclose(np.linalg.norm(raw[3:]), 1.0, atol=1e-4)):
            return self._reject_cabinet_motion("release retreat: invalid measured TCP")
        actual = Pose.from_pose7(raw)
        height = float(place.transport_z)
        if not np.isfinite(height) or height < actual.p[2]:
            return self._reject_cabinet_motion("release retreat: invalid transport height")
        target = Pose([actual.p[0], actual.p[1], height], actual.q)
        if not self._payload_move_pose(target, arm_tag, "release retreat"):
            return self._reject_cabinet_motion("release retreat rejected")

    def _body_grasp_plan_valid(self, plans, arm_tag):
        """Apply cabinet continuity/FK gates before the public helper executes."""
        def reject(reason):
            print(f"[cabinet-preflight] {reason}", flush=True)
            return False

        arm = self.robot.get_arm(arm_tag)
        pre, grasp, grasp_link, pre_link = plans
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            return reject("invalid initial arm joints")
        for label, plan, link in (("pre-grasp", pre, pre_link), ("grasp", grasp, grasp_link)):
            if plan is None or not plan.success or not _cabinet_motion_pose_valid(link):
                return reject(f"{label}: unavailable plan or invalid target")
            path = _cabinet_motion_path(plan.position, plan.velocity, arm.n_arm)
            if path is None:
                return reject(f"{label}: invalid path shape or nonfinite values")
            positions, _ = path
            if not self._body_path_clear(np.vstack([seed, positions]), arm_tag):
                return reject(f"{label}: acquisition fixture check failed")
            if (np.linalg.norm(positions[-1] - seed) > 3.5
                    or np.max(np.abs(np.diff(np.vstack([seed, positions]), axis=0))) > 0.30):
                return reject(f"{label}: joint continuity failed")
            actual = self._drawer_plan_endpoint_tcp(arm_tag, positions[-1])
            target = link * arm.tcp_offset
            if not _cabinet_motion_pose_valid(actual):
                return reject(f"{label}: invalid endpoint FK")
            distance = float(np.linalg.norm(actual.p - target.p))
            angle = 2 * float(np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1)))
            if distance > 0.015 or angle > np.deg2rad(10):
                return reject(f"{label}: endpoint xyz={distance:.6f}m angle={angle:.6f}rad")
            seed = positions[-1].copy()
        return True

    def _body_grasp_candidates(self, object_pose):
        if self._object_id == "038_milk-box":
            # Center on the carton axis at its estimated mass height. The
            # previous4mm radial bias allowed rolling during a slow turn;
            # native palm/body and table margins here are4.2/18mm.
            return [GraspPose(
                "cabinet_milk_body_centered", [0, 0.050, 0],
                [0.5, 0.5, -0.5, -0.5],
                pre_distance=0.12, source="manual", scale_frame="mesh_unit",
            )]
        aabb = np.asarray(to_numpy(self.object.entity.get_AABB()), dtype=float)
        scale = float(self._object_scale)
        if (aabb.shape != (2, 3) or not np.all(np.isfinite(aabb))
                or np.any(aabb[1] <= aabb[0]) or not np.isfinite(scale) or scale <= 0
                or not _cabinet_motion_pose_valid(object_pose)):
            return self._reject_cabinet_motion("invalid payload grasp geometry")
        center = aabb.mean(axis=0)
        if self._object_id in ("035_apple", "048_stapler", "086_woodenblock"):
            # Preserve the existing8mm palm margin. The nearly spherical
            # apple admits orthogonal jaws at the same TCP; this lets the
            # public selector reject an infeasible straight approach before
            # any physical grasp, without a curved near-object fallback.
            yaw = float(self._object_relative_grasp_yaw())
            if not np.isfinite(yaw):
                return self._reject_cabinet_motion("invalid apple grasp yaw")
            kind = {"035_apple": "apple", "048_stapler": "stapler", "086_woodenblock": "wood"}[self._object_id]
            # Stapler's old11mm roof depth grips a narrow upper sliver.
            # A20mm depth leaves17mm palm clearance. Wood uses the same8mm
            # palm margin as apple. Keep the established jaw axis first;
            # model1 also has measured broad faces on its orthogonal axis.
            depth = .020 if kind == "stapler" else .029
            alternate = np.pi/2 if kind == "apple" else np.pi
            suffix = "orthogonal" if kind == "apple" else "equivalent"
            orientations = [(f"cabinet_{kind}_palm_clear", yaw),
                            (f"cabinet_{kind}_palm_clear_{suffix}", yaw + alternate)]
            if kind == "apple" and self._object_model_id == 1:
                # The retained apple slides off the finite finger pads.
                # Actual-scale sections at this azimuth offer broader
                # opposing purchase at the same palm-clear depth. Keep the
                # original alternatives when PUBLIC rejects this approach.
                orientations.insert(0, ("cabinet_apple_palm_clear_45", yaw - np.pi / 4))
                # Reversing the same jaw axis avoids a bounded-wrist branch
                # that required >5rad during pre-grasp in the native trace.
                # Keep PUBLIC ranking and every continuity/fixture gate.
                orientations.insert(1, ("cabinet_apple_palm_clear_45_equivalent", yaw + 3 * np.pi / 4))
            if kind == "wood" and self._object_model_id == 1:
                # Recorded19 rejects both original straight approaches.
                # A quarter turn retains32.5–34.5mm finite-pad sections and
                # 8.4mm palm clearance while admitting a native approach.
                # Append to the input list; PUBLIC ranks all candidates
                # and may select either axis first. Closure stays unchanged.
                orientations.extend([
                    ("cabinet_wood_palm_clear_orthogonal", yaw + np.pi / 2),
                    ("cabinet_wood_palm_clear_orthogonal_equivalent", yaw + 3 * np.pi / 2),
                ])
            candidates = []
            for name, angle in orientations:
                target = self._top_down_tcp([center[0], center[1], aabb[1,2] - depth], yaw=angle)
                local = object_pose.inv() * target
                candidates.append(GraspPose(name, local.p / scale, local.q,
                    pre_distance=.12, source="manual", scale_frame="mesh_unit"))
            return candidates
        # Jar0's horizontal approach leaves only~1mm wrist/table clearance,
        # lost under measured tracking. Topdown at20mm below the roof leaves
        # ~17mm palm/body and232mm wrist/table clearance in recorded states.
        # PUBLIC selects the feasible jaw axis; physical retention is still
        # required after the unchanged close, tiny lift and every transfer.
        candidates = []
        for name, yaw in (("cabinet_jar_topdown", 0.),
                          ("cabinet_jar_topdown_orthogonal", np.pi / 2)):
            target = self._top_down_tcp([center[0], center[1], aabb[1, 2] - .020], yaw=yaw)
            local = object_pose.inv() * target
            candidates.append(GraspPose(name, local.p / scale, local.q,
                pre_distance=.12, source="manual", scale_frame="mesh_unit"))
        return candidates

    def _close_body_payload(self, pick, arm_tag):
        """Close on the body payload; the apple uses a stall-detected staged close.

        The fixed .675 apple command (27 mm per jaw) holds on the
        45-degree grasp axis (stall 28.5 mm) but never loads the orthogonal
        grasp axis (seeds 7/13: lift -0.1 mm), because the hull width differs
        per axis. Stage the command down in .06 steps, watch the measured gap
        and the apple, stop at the first stall or 2 mm displacement, then hold
        the measured gap minus a 3 mm bite (1.5 mm per side, about 14 N per pad
        at the finger PD). Floor .30 keeps the shared missed-grasp lift gate.
        Other payloads keep the single catalog/close-target command.
        """
        close = self._body_close_target(pick)
        if not ((self._object_id, self._object_model_id) == ("035_apple", 1)
                and self.config.get("robot_type") == "franka"):
            self.set_gripper(close, arm_tag)
            for _ in range(200):
                self.step_sim()
            return close
        arm = self.robot.get_arm(arm_tag)

        def gap():
            q = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
            return float(q[-2] + q[-1])

        start = np.asarray(pick.get_center(), dtype=float).copy()
        target, previous, stall, stage = 1.0, gap(), 0, 0
        while target > .30:
            target = max(.30, target - .06)
            self.set_gripper(target, arm_tag)
            for _ in range(8):
                self.step_sim()
            current = gap()
            moved = float(np.linalg.norm(np.asarray(pick.get_center(), dtype=float) - start))
            if current < .066 and moved > .002:
                break
            stage += 1
            if stage > 2 and previous - current < .0015:
                stall += 1
                if stall >= 2:
                    break
            else:
                stall = 0
            previous = current
        hold = float(np.clip((gap() - .003) / .08, .30, 1.0))
        print(f"[cabinet-body] apple staged close: gap={gap()*1000:.1f}mm hold={hold:.3f}", flush=True)
        self.set_gripper(hold, arm_tag)
        for _ in range(200):
            self.step_sim()
        return hold

    def _body_close_target(self, pick):
        # Preserve the shared default when the catalog omits a target.
        if pick.close_value is None:
            radius = float(pick.radius)
            close = max(.010, radius-.005 if radius<.030 else radius*.7)/.04
        else:
            close = float(np.clip(pick.close_value,0.,1.))
        if not np.isfinite(close):
            raise _CabinetMotionRejected('invalid payload close target')
        if ((self._object_id,self._object_model_id)==('035_apple',1)
                and np.isclose(close,.50,rtol=0,atol=1e-12)):
            # At pair friction 1 the pads no longer sink into the apple:
            # they stall at 28.5 mm per jaw on the collision hull, and the .45
            # command (18 mm) then drives 62 -> 160 N into a round body, which
            # squirts out during the tiny lift. Width-match the cage instead:
            # .675 commands 27 mm per jaw, a 1.5 mm bite past the measured
            # stall (about 14 N per pad at the finger PD), the regime that
            # holds round food elsewhere. The .45 command assumes sinking pads.
            close = .675
        return close

    def _pick_and_place_body(self, pick, place, arm_tag):
        """Acquire a body grasp publicly, then use checked physical transfer."""
        arm = self.robot.get_arm(arm_tag)
        if not hasattr(arm.planner, "_mplib"):
            return self._reject_cabinet_motion("body grasp requires native MPLib")
        entity = self.object.entity
        self.open_gripper(arm_tag)
        for _ in range(max(0, int(pick.preopen_steps))):
            self.step_sim()
        object_pose = Pose(to_numpy(entity.get_pos()), to_numpy(entity.get_quat()))
        self._refresh_avatar_planner_obstacles(arm_tag)
        side_grasps = self._body_grasp_candidates(object_pose)
        if not side_grasps:
            return False
        result = self.select_and_execute_grasp(
            object_name=self._object_id, object_pose=object_pose, arm_tag=arm_tag,
            model_id=self._object_model_id, object_scale=self._object_scale,
            max_candidates=len(side_grasps), table_z=self.TABLE_TOP_Z, candidate_grasps=side_grasps,
            plan_attempts=3,
            screw_qpos_step=0.02,
            post_plan_check=lambda plans: self._body_grasp_plan_valid(plans, arm_tag),
        )
        if result is None:
            return False
        grasp_link, _, grasp = result
        for _ in range(100):
            self.step_sim()
        tcp = self._payload_arrived_pose(grasp_link * arm.tcp_offset, arm_tag, "grasp")
        if tcp is None:
            return False
        if ((self._object_id, self._object_model_id) in (("086_woodenblock", 1), ("035_apple", 1))
                and self.config.get("robot_type") == "franka"
                and not self._refine_open_body_grasp(grasp_link * arm.tcp_offset, arm_tag)):
            return False
        self._close_body_payload(pick, arm_tag)
        if pick.after_close is not None:
            pick.after_close()
            for _ in range(40):
                self.step_sim()
        start_center = np.asarray(pick.get_center(), dtype=float)
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        if np.linalg.norm(start_center - tcp.p) > float(pick.max_grasp_offset):
            return False
        # Preserve the tight tiny-lift joint/FK/arrival checks before transport.
        tiny_target = Pose(tcp.p + np.array([0, 0, 0.03]), tcp.q)
        if not self._payload_move_pose(tiny_target, arm_tag, "tiny lift", tiny=True):
            return False
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        local_held = tcp.to_matrix()[:3, :3].T @ (np.asarray(pick.get_center()) - tcp.p)

        def retained(label):
            measured = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
            local_now = measured.to_matrix()[:3, :3].T @ (
                np.asarray(pick.get_center()) - measured.p)
            error = float(np.linalg.norm(local_now - local_held))
            print(f"[cabinet-body] {label}: held-offset change={error:.4f}m", flush=True)
            return np.isfinite(error) and error <= float(pick.transport_retention_tolerance)

        safe_z = float(place.transport_z)
        lift = Pose(np.array([tcp.p[0], tcp.p[1], safe_z]), tcp.q)
        if not self._payload_move_pose(lift, arm_tag, "lift"):
            return False
        lift_delta = float(np.asarray(pick.get_center())[2] - start_center[2])
        minimum_lift = float(pick.min_lift_delta) if pick.min_lift_delta is not None else max(0.03, 0.5 * pick.radius)
        print(f"[cabinet-body] grasp={grasp.name}, lift={lift_delta:.4f}m", flush=True)
        if lift_delta < minimum_lift or not retained("lift"):
            return False
        # The conservative cabinet envelope is only for acquisition outside
        # the fixture. Drawer entry needs the cavity, which this box fills.
        self._clear_body_acquisition_obstacles()
        # Turning above the rim lets the narrow top-down gripper enter the
        # drawer. The carton may lie flat; the placement contract permits this.
        downward = float((tcp.to_matrix()[:3, :3])[:, 2] @ np.array([0., 0., -1.]))
        if downward < np.cos(np.deg2rad(10)):
            # A circular jar can roll in the side pinch. Following its new
            # yaw adds an unnecessary wrist twist (the saved-state probe
            # inferred about-1rad). The canonical turn remains feasible
            # from that state; do not derive its orientation from payload slip.
            turn_yaw = (0.0 if (self._object_id, self._object_model_id) == ("031_jam-jar", 0)
                        else self._object_relative_grasp_yaw())
            topdown_tcp = self._top_down_tcp(lift.p, yaw=turn_yaw)
            if not self._payload_move_pose(topdown_tcp, arm_tag, "turn") or not retained("turn"):
                return False
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        held = np.asarray(pick.get_center()) - tcp.p
        destination = np.array([place.pos[0] - held[0], place.pos[1] - held[1], safe_z])
        if not self._payload_move_pose(Pose(destination, tcp.q), arm_tag, "carry") or not retained("carry"):
            return False
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        if self._object_id == "086_woodenblock":
            # Exact retained-state MPLib probes reject wood's current wrist
            # branch on descent. A quarter-turn above the rim admits a
            # continuous lower route, with less joint travel than yaw zero.
            # High releases instead tunnel the thin drawer support.
            turn = self._top_down_tcp(tcp.p, yaw=np.pi / 2)
            if not self._payload_move_pose(turn, arm_tag, "release turn") or not retained("release turn"):
                return False
            tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        held = np.asarray(pick.get_center()) - tcp.p
        # Recompute held height after the physical turn, including any change
        # in the object's orientation; never reuse its pre-turn half-height.
        release_z = self.support_release_tcp_z(
            place.support_z, pick.get_half_height(), held[2], place.support_clearance,
        )
        release_z = self._body_release_height(release_z, safe_z)
        release_xy = (np.asarray(place.pos[:2]) - held[:2]
                      if self._object_id == "086_woodenblock" else tcp.p[:2])
        release = Pose(np.array([*release_xy, release_z]), tcp.q)
        release = self._descend_with_branch_retry(release, pick, place, arm_tag, retained, safe_z)
        if release is None:
            return False
        self.open_gripper(arm_tag)
        for _ in range(40):
            self.step_sim()
        retreat = Pose(np.array([release.p[0], release.p[1], safe_z]), release.q)
        if not self._payload_move_pose(retreat, arm_tag, "release retreat"):
            return False
        for _ in range(210):
            self.step_sim()
        print(f"[cabinet-milk] after release: {pick.get_center()}", flush=True)
        return True

    def _descend_with_branch_retry(self, release, pick, place, arm_tag, retained, safe_z):
        """Descend to the release pose; after a pre-execution planning rejection,
        mirror the jaw yaw above the rim once and retry from the other wrist branch.

        The stapler (seed 16) and wooden block (seed 19) can carry into a branch
        with joint2 within .16 rad of its limit, so the 12 cm descent has no IK
        solution (a branch with .6 rad of room descends the same seeds). A
        parallel jaw is symmetric under a 180 degree yaw, so the mirrored
        top-down pose holds the same object and reaches the same release pose
        from the opposite branch. Only rejections raised before execution are
        retried; a measured-arrival failure after execution is not. Retention
        gates apply to every leg.
        """
        try:
            if not self._payload_move_pose(release, arm_tag, "descent"):
                return None
            return release if retained("descent") else None
        except _CabinetMotionRejected as exc:
            if not getattr(self, "_cabinet_pick_active", False):
                raise
            print(f"[cabinet-body] descent branch retry after: {exc}", flush=True)
        arm = self.robot.get_arm(arm_tag)
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        rotation = tcp.to_matrix()[:3, :3]
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        turn = self._top_down_tcp(np.array([tcp.p[0], tcp.p[1], float(safe_z)]), yaw=yaw + np.pi)
        if not self._payload_move_pose(turn, arm_tag, "branch turn") or not retained("branch turn"):
            return None
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
        held = np.asarray(pick.get_center()) - tcp.p
        release_z = self.support_release_tcp_z(
            place.support_z, pick.get_half_height(), held[2], place.support_clearance,
        )
        release_z = self._body_release_height(release_z, safe_z)
        release_xy = (np.asarray(place.pos[:2]) - held[:2]
                      if self._object_id == "086_woodenblock" else tcp.p[:2])
        release = Pose(np.array([*release_xy, release_z]), tcp.q)
        if not self._payload_move_pose(release, arm_tag, "descent") or not retained("descent"):
            return None
        return release

    def _body_release_height(self, support_release_z, safe_z):
        # The native palm spans204mm, wider than this drawer. Its lower face
        # is37.44mm above a topdown TCP. Clear the rim before opening rather
        # than driving the palm through the walls to achieve a smaller drop.
        values = np.asarray([support_release_z, safe_z, self._drawer_rim_z], dtype=float)
        if not np.all(np.isfinite(values)):
            raise _CabinetMotionRejected("invalid body release heights")
        height = max(float(support_release_z), float(self._drawer_rim_z) + .008 - .037)
        if height > safe_z:
            raise _CabinetMotionRejected("body release clearance exceeds transport height")
        return height

    def _move_seeded(self, link_pose7, arm_tag):
        """Reuse native cabinet MPLib planning without extra settle ticks."""
        try:
            target = np.asarray(link_pose7, dtype=float)
        except (TypeError, ValueError):
            return self._reject_cabinet_motion("invalid seeded link target")
        if (target.shape != (7,) or not np.all(np.isfinite(target))
                or not np.isclose(np.linalg.norm(target[3:]), 1.0, atol=1e-4)):
            return self._reject_cabinet_motion("invalid seeded link target")
        arm = self.robot.get_arm(arm_tag)
        if not _cabinet_motion_pose_valid(arm.tcp_offset):
            return self._reject_cabinet_motion("invalid seeded TCP offset")
        tcp_pose = Pose.from_pose7(target.copy()) * arm.tcp_offset
        ok = self._drive_to_tcp(
            tcp_pose.p, arm_tag, n_settle=0, tol=0.05, tcp_pose=tcp_pose,
            max_joint_delta=3.5,
        )
        if not ok:
            return self._reject_cabinet_motion("seeded MPLib motion rejected")
        self._reseed_cached_arm_target(arm)
        return True

    def _solve_ik(self, tcp_pose, arm_tag):
        """Keep the shared tiny-lift PD loop, with checked MPLib IK only."""
        if not _cabinet_motion_pose_valid(tcp_pose):
            return self._reject_cabinet_motion("invalid tiny-lift TCP target")
        arm = self.robot.get_arm(arm_tag)
        if not _cabinet_motion_pose_valid(arm.tcp_offset):
            return self._reject_cabinet_motion("invalid tiny-lift TCP offset")
        cached = getattr(arm, "_cached_target", None)
        try:
            seed = np.asarray(arm.get_arm_qpos() if cached is None else cached, dtype=float)
        except (TypeError, ValueError):
            return self._reject_cabinet_motion("invalid tiny-lift joint seed")
        if (seed.ndim != 1 or len(seed) < arm.n_arm or not np.all(np.isfinite(seed))):
            return self._reject_cabinet_motion("invalid tiny-lift joint seed")
        seed = seed[:arm.n_arm].copy()
        link_pose = tcp_to_link_pose(tcp_pose, arm.tcp_offset)
        if not _cabinet_motion_pose_valid(link_pose):
            return self._reject_cabinet_motion("invalid tiny-lift link target")
        self._refresh_avatar_planner_obstacles(arm_tag)
        try:
            result = arm.planner.solve_ik(
                seed.copy(), link_pose.to_pose7(), num_waypoints=2, log=False,
            )
        except RuntimeError as exc:
            return self._reject_cabinet_motion(f"tiny-lift MPLib IK rejected: {exc}")
        if result is None or not result.success:
            return self._reject_cabinet_motion("tiny-lift MPLib IK rejected")
        path = _cabinet_motion_path(
            result.position, getattr(result, "velocity", None), arm.n_arm,
        )
        if path is None:
            return self._reject_cabinet_motion("invalid tiny-lift MPLib IK path")
        goal = path[0][-1]
        if np.max(np.abs(goal - seed)) > 0.30:
            return self._reject_cabinet_motion("tiny-lift IK branch jump exceeds 0.30 rad")
        endpoint = self._drawer_plan_endpoint_tcp(arm_tag, goal)
        if not _cabinet_motion_pose_valid(endpoint):
            return self._reject_cabinet_motion("invalid tiny-lift FK pose")
        distance = float(np.linalg.norm(endpoint.p - tcp_pose.p))
        angle = 2.0 * float(np.arccos(np.clip(abs(np.dot(endpoint.q, tcp_pose.q)), 0, 1)))
        if distance > 0.01 or angle > 0.05:
            return self._reject_cabinet_motion(
                f"tiny-lift FK rejected: xyz={distance:.4f}m, angle={angle:.4f}rad",
            )
        return goal.copy()

    def _pick_drawer(self):
        super()._pick_drawer()
        print(
            f"[drawer] physical support_z={self._drawer_support_surface_z():.7f}, "
            f"grading floor_z={self._drawer_floor_z:.7f}", flush=True,
        )
        self._object_grasp_reference_quat = np.asarray(
            to_numpy(self.object.entity.get_quat()), dtype=float,
        ).reshape(-1)[:4].copy()

    def _object_relative_grasp_yaw(self) -> float:
        reference = getattr(self, "_object_grasp_reference_quat", None)
        if reference is None:
            return 0.0
        current = np.asarray(
            to_numpy(self.object.entity.get_quat()), dtype=float,
        ).reshape(-1)[:4]
        if (self._object_id == "031_jam-jar" and self._object_model_id == 0
                and self.config.get("robot_type") == "franka"
                and abs(Pose([0, 0, 0], current).to_matrix()[2, 1]) >= np.cos(np.deg2rad(10))):
            # The loaded jar's body is circular to within 0.4 mm. Following
            # inspection yaw needlessly rotates the wrist into an infeasible
            # descent. Keep the canonical jaws for an upright jar; tilted
            # payloads still use the existing relative-orientation policy.
            return 0.0
        return self.relative_parallel_jaw_yaw(current, reference)

    def _payload_pick_options(self) -> dict:
        options = super()._payload_pick_options()
        if ((self._object_id, self._object_model_id) != ("035_apple", 1)
                or self.config.get("robot_type") != "franka"):
            return options
        # Contact traces show the palm touching the apple before close.
        # The pinned Franka hand collision mesh ends 37.44 mm above its
        # down-facing TCP. Keep an
        # 8 mm approach margin using a conservative 37 mm separation, while
        # leaving the catalog's jaw gap and finger PD profile unchanged.
        half_height = self.live_entity_half_height(self.object.entity)
        roof_clear_offset = half_height + 0.3 * float(self._object_radius) - 0.037 + 0.008
        options["grasp_z_offset"] = max(
            float(options.get("grasp_z_offset", 0.0)), roof_clear_offset,
        )
        return options

    def _release_object_from_avatar(self) -> None:
        if self.avatar is not None:
            self.avatar.robot.detach_object(self.AVATAR_HAND_ID)

    def _hold_drawer_open(self) -> bool:
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
            self._sync_drawer_floor_proxy()
        return ok

    def _sync_drawer_floor_proxy(self) -> None:
        proxy = getattr(self, "_drawer_floor_proxy", None)
        if proxy is None or self._drawer_link is None or self._drawer_floor_z is None:
            return
        self.sync_support_proxy(proxy, self._drawer_link, self._drawer_support_surface_z())

    def _drawer_support_surface_z(self) -> float:
        # Cabinet46653's top drawer bottom panel is0.067411 mesh units above
        # its AABB minimum, not the old10mm estimate at runtime scale0.20.
        # Align physical support/placement to the visible floor. Preserve the
        # existing grading reference and thresholds for separate root review.
        return float(self._drawer_floor_z) - 0.010 + 0.067411 * float(self._CABINET_SCALE)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    # Park relative to the arm base (collection choreography and yields).
    INTERRUPT_PARK_FORWARD_M = 0.25
    INTERRUPT_PARK_HEIGHT_M = 0.30
    # Full-state expert yield only: at the centred park the human's hand can
    # brush the robot hand (2-10 mm) while lifting the object to inspect it
    # (10 of 14 baseline failures). While the human is active the expert
    # watches its predicted capsules; if they come within the watch distance
    # of the parked hand, it moves to a park 10 cm toward the cabinet side and
    # 10 cm higher (18 cm put the open fingers in the cabinet fixture). Moving
    # there unconditionally lost 4 seeds that the centred park passes.
    INTERRUPT_EXPERT_PARK_WATCH_M = 0.10
    INTERRUPT_EXPERT_PARK_WATCH_HORIZON = 150     # sim steps of prediction
    INTERRUPT_EXPERT_PARK_WATCH_EVERY = 10        # sim steps between checks
    INTERRUPT_EXPERT_PARK_LATERAL_M = 0.10
    INTERRUPT_EXPERT_PARK_HEIGHT_M = 0.40

    def _retreat_safe(self, arm_tag: str, lateral: float = 0.0, height: float | None = None):
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        park_forward = float(self.INTERRUPT_PARK_FORWARD_M)
        if (getattr(self, "_object_id", None) == "038_milk-box"
                and self._object_model_id == 0
                and self.config.get("robot_type") == "franka"):
            # At the old park, the index capsule touches the hand during
            # the avatar's upward inspection lift, before robot pickup.
            park_forward = min(park_forward, 0.20)
        retreat_pos = np.array([
            arm_base[0] + float(lateral),
            arm_base[1] + park_forward,
            self.TABLE_TOP_Z + float(self.INTERRUPT_PARK_HEIGHT_M if height is None else height),
        ], dtype=float)
        if self._move_screw(retreat_pos, arm_tag) is None:
            link = tcp_to_link_pose(self._top_down_tcp(retreat_pos), arm.tcp_offset)
            self._move_seeded(link.to_pose7(), arm_tag)

    # ------------------------------------------------------------------
    # Eval-mode avatar interrupt
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        self._interrupt_controller().init_eval_state(self)

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def _is_object_on_table(self, obj_pos: np.ndarray) -> bool:
        return self._interrupt_controller().is_object_on_table(self, obj_pos)

    def play_once(self) -> bool:
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
        # kp=15000 works best, though the gripper, descending 4 cm above
        # the bar, can still snag the bar during the +Y pull micro-steps,
        # dragging the drawer to dof=0.065 (target 0.06).  kp=50000 is WORSE
        # (drawer only opens to 0.020) — likely the high-kp jerks bounce the
        # gripper off the bar before contact friction grips.  Restored to
        # default after for the pick (which works at default kp).
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

        # Phase B — show intent on the table object, then let the avatar
        # inspect it before the robot commits to grasping.
        def _object_center():
            return self.live_entity_aabb_center(self.object.entity)

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        if self.avatar is not None:
            obj_center = _object_center()
            intent_z = self.TABLE_TOP_Z + 0.20
            intent_pos = np.array([obj_center[0], obj_center[1], intent_z])
            intent_link = tcp_to_link_pose(
                self._top_down_tcp(intent_pos),
                self.robot.get_arm(arm_tag).tcp_offset,
            )
            self.open_gripper(arm_tag)
            if self._move_seeded(intent_link.to_pose7(), arm_tag) is None:
                pass

            self._play_inspect_motion(
                attach_obj=self.object.entity,
                hand_id=self.AVATAR_HAND_ID,
                attach_frame=self._attach_frame_scaled,
                detach_frame=self._detach_frame_scaled,
                return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
            )
            self._retreat_safe(arm_tag)
            self._wait_until_avatar_attach()
            for _ in range(100):
                self.step_sim()

            if not self.avatar.spare():
                while not self.avatar.spare():
                    self.step_sim()
            for _ in range(120):
                self.step_sim()

            # The authored inspect clip has already detached the object at its
            # release frame.  Leave it physical and settled here; retaining a
            # kinematic hand constraint during robot finger closure can inject
            # contact energy and expel the object when that constraint drops.
        else:
            pass

        # Phase C — pick the object from its latest pose and drop into the
        # open drawer via the mixin's ``pick_and_place`` helper.
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

        def _release_avatar_and_boost_grip():
            self._release_object_from_avatar()
            self._boost_finger_pd(arm_tag)

        pick = PickSpec(
            get_center=_object_center,
            radius=object_radius,
            label=self._object_id,
            get_half_height=_object_half_height,
            get_grasp_yaw=self._object_relative_grasp_yaw,
            after_close=_release_avatar_and_boost_grip,
            require_lift=True,
            max_grasp_offset=0.10,
            transport_retention_tolerance=0.08,
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

    def _drawer_ready(self) -> bool:
        """Drawer opened to its scripted target (within pull tolerance)."""
        q = self._interrupt_controller().drawer_qpos(self)
        if q is None or getattr(self, "_drawer_open_qpos_target", None) is None:
            return False
        return q >= float(self._drawer_open_qpos_target) - float(self._DRAWER_PULL_TOLERANCE)

    def _open_drawer_for_rollout(self, arm_tag: str) -> bool:
        self._set_arm_pd(arm_tag, arm_kp=15000.0, arm_kv=800.0)
        ok_open = self._open_drawer_physics(arm_tag)
        self._set_arm_pd(arm_tag, arm_kp=4000.0, arm_kv=300.0)
        if not ok_open:
            return False
        return bool(self._hold_drawer_open())

    def _pick_object_into_drawer(self, arm_tag: str) -> bool:
        def _object_center():
            return self.live_entity_aabb_center(self.object.entity)

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        def _release_avatar_and_boost_grip():
            self._release_object_from_avatar()
            self._boost_finger_pd(arm_tag)

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
            radius=float(self._object_radius),
            label=self._object_id,
            get_half_height=_object_half_height,
            get_grasp_yaw=self._object_relative_grasp_yaw,
            after_close=_release_avatar_and_boost_grip,
            require_lift=True,
            max_grasp_offset=0.10,
            transport_retention_tolerance=0.08,
            **self._payload_pick_options(),
        )
        place = PlaceSpec(
            pos=drop_xyz,
            label="drawer",
            transport_z=self._drawer_rim_z + 0.10,
            support_z=self._drawer_support_surface_z(),
            support_clearance=self._DRAWER_SUPPORT_CLEARANCE,
        )
        return bool(self.pick_and_place(pick, place, arm_tag))

    def play_blind_once(self) -> bool:
        """No-intent expert: open the drawer and put the object in it from its
        live pose, without reading interrupt or avatar state and without
        yielding.  Eval ticks come from the expert harness
        (``ScriptedRolloutDriver`` at ``action_substeps`` cadence)."""
        for _ in range(80):
            self.step_sim()

        if self.cabinet is None or self.object is None:
            return False

        self._pick_drawer()

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        if not self._open_drawer_for_rollout(arm_tag):
            return False
        if not self._pick_object_into_drawer(arm_tag):
            return False
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # ------------------------------------------------------------------

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent rollout: open the drawer only if it is not open yet,
        then pick the object from its live pose unless it is already in the
        drawer."""
        arm_tag = "right"
        if self.cabinet is None or self.object is None:
            return False
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            with self.suppress_recording():
                for _ in range(80):
                    self.step_sim()
            self._pick_drawer()
            self._boost_finger_pd(arm_tag)
        if not self._drawer_ready():
            if not self._open_drawer_for_rollout(arm_tag):
                return False
        elif not self._hold_drawer_open():
            return False
        if not bool(self._object_in_drawer_metrics().get("object_in_drawer", False)):
            if not interrupt_fired(self):
                # Approach like the collection choreography: hover above the
                # object first (the progress trigger normally fires here and
                # unwinds this motion), then pick from the live pose.
                self._interrupt_expert_hover(arm_tag)
            if not self._pick_object_into_drawer(arm_tag):
                return False
        for _ in range(60):
            self.step_sim()
        return True

    INTERRUPT_EXPERT_HOVER_Z = 0.20
    INTERRUPT_EXPERT_HOVER_DWELL_STEPS = 50

    def _interrupt_expert_hover(self, arm_tag: str) -> None:
        center = self.live_entity_aabb_center(self.object.entity)
        hover = np.array([
            center[0], center[1], self.TABLE_TOP_Z + self.INTERRUPT_EXPERT_HOVER_Z,
        ], dtype=float)
        link = tcp_to_link_pose(
            self._top_down_tcp(hover), self.robot.get_arm(arm_tag).tcp_offset,
        )
        self.open_gripper(arm_tag)
        self._move_seeded(link.to_pose7(), arm_tag)
        for _ in range(self.INTERRUPT_EXPERT_HOVER_DWELL_STEPS):
            self.step_sim()

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")
        self._interrupt_expert_watch_park("right")

    def _interrupt_expert_hand_points(self, arm_tag: str) -> list[np.ndarray]:
        arm = self.robot.get_arm(arm_tag)
        points = [np.asarray(arm.get_ee_pose(), dtype=float)[:3]]
        for link in getattr(arm.entity, "links", []) or []:
            if str(link.name) in ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link7"):
                points.append(np.asarray(to_numpy(link.get_pos()), dtype=float).ravel()[:3])
        return points

    def _interrupt_expert_watch_park(self, arm_tag: str) -> None:
        """Hold the park while the human is active; step aside once (see
        INTERRUPT_EXPERT_PARK_WATCH_M) if its predicted motion closes in."""
        from ..task_bases.neutral_human_aware import (
            point_capsule_distance, predict_human_capsules,
        )
        every = max(1, int(self.INTERRUPT_EXPERT_PARK_WATCH_EVERY))
        horizon = max(0, int(self.INTERRUPT_EXPERT_PARK_WATCH_HORIZON))
        offsets = list(range(0, horizon + 1, max(1, horizon // 6 or 1)))
        cap = int((self.config or {}).get("interrupt_expert_wait_max_sim_steps", 12000))
        n = 0
        episode = interrupt_episode(self)
        # From the trigger (the human first waits out its warning, then glides
        # in) until its whole sequence is done.
        while (episode is not None and episode.fired and not episode.human.done
               and n < cap):
            points = self._interrupt_expert_hand_points(arm_tag)
            try:
                frames = predict_human_capsules(self, offsets)
            except Exception:
                frames = []
            near = min((point_capsule_distance(p, a, b, r)
                        for caps in frames for _n, a, b, r in caps for p in points),
                       default=np.inf)
            if near < float(self.INTERRUPT_EXPERT_PARK_WATCH_M):
                print(f"[cabinet-interrupt] human within {near:.3f} m of the park: "
                      f"stepping aside", flush=True)
                self._retreat_safe(arm_tag, lateral=self.INTERRUPT_EXPERT_PARK_LATERAL_M,
                                   height=self.INTERRUPT_EXPERT_PARK_HEIGHT_M)
                return
            for _ in range(every):
                self.step_sim()
            n += every

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _object_in_drawer_metrics(self) -> dict:
        if self._drawer_link is None or self.object is None:
            return {
                "object_in_drawer": False,
                "object_in_drawer_error": "missing drawer link or object",
            }

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
        }

    def check_success(self) -> bool:
        metrics = self._object_in_drawer_metrics()
        if "object_in_drawer_error" in metrics:
            return False
        return bool(metrics["object_in_drawer"])

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._object_in_drawer_metrics())
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
