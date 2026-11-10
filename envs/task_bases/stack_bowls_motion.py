"""Native MPLib checks and physical execution for bowl stacking."""
from contextlib import contextmanager

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..grasp import tcp_to_link_pose
from ..planning.base import PlanResult
from ..utils import Pose, to_numpy


class _BowlMotionRejected(RuntimeError):
    """A rejected physical motion, distinct from an evaluator error."""


def _valid_pose(pose):
    try:
        p, q = np.asarray(pose.p), np.asarray(pose.q)
        return (p.shape == (3,) and q.shape == (4,)
                and np.all(np.isfinite(p)) and np.all(np.isfinite(q))
                and np.isclose(np.linalg.norm(q), 1., atol=1e-4))
    except (AttributeError, TypeError, ValueError):
        return False


def _pose_matches(actual, target, tiny=False, tiny_tol=.010):
    if not _valid_pose(actual) or not _valid_pose(target):
        return False
    angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1))
    return (np.linalg.norm(actual.p - target.p) <= (tiny_tol if tiny else .015)
            and angle <= (.05 if tiny else np.deg2rad(10)))


class BowlNativeMotionMixin:
    """Bowl-local motion policy; grasp acquisition stays in the public helper."""

    # Open-grasp refinement limits.  The parked arm sags 6.5-8.7 mm under PD
    # at the rim (measured on every arrival), so an 8 mm
    # first-error limit rejected sound approaches on the interrupt layouts;
    # the correction bound follows the larger first error.
    BOWL_REFINE_MAX_ERROR_M = .012
    BOWL_REFINE_MAX_CORRECTION_M = .018
    # Measured arrival tolerance of tiny moves (the FK endpoint check of
    # their plans stays at 10 mm).  Same sag as above: on five interrupt
    # layouts the held tiny lift settled 10.1-10.7 mm under its target.
    BOWL_TINY_ARRIVAL_M = .012
    # Screw re-integration steps tried in order (first = the nominal .02).
    BOWL_SCREW_QPOS_STEPS = (.02, .0193, .0207)

    def _bowl_reject(self, reason):
        self.plan_success = False
        print(f"[bowl-motion] {reason}", flush=True)
        raise _BowlMotionRejected(reason)

    def _bowl_refresh_avatar(self, arm_tag):
        collider = getattr(self, "avatar_collider", None)
        if collider is not None:
            points = sample_capsule_planner_points(collider.current_capsules(), inflate_factor=1.8)
            if points is not None and points.size:
                self.robot.get_arm(arm_tag).planner.update_obstacles(points, resolution=.04)

    @contextmanager
    def _bowl_motion_scene(self, arm_tag):
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject

        arm = self.robot.get_arm(arm_tag)
        if self.config.get("robot_type") != "franka" or not hasattr(arm.planner, "_mplib"):
            self._bowl_reject("stacking requires native Franka MPLib")
        backend = arm.planner._mplib
        name = "stack_bowls_table"
        bounds = np.asarray(to_numpy(self.table.get_AABB()), dtype=float)
        if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                or np.any(bounds[1] <= bounds[0]) or backend.planning_world.has_object(name)
                or backend.acm.has_entry(name)):
            self._bowl_reject("invalid or conflicting table fixture")
        previous = getattr(self, "_bowl_motion_active", False)
        if previous:
            self._bowl_reject("nested bowl motion")
        box = CollisionObject(Box(*(bounds[1] - bounds[0])),
                              MplibPose(p=bounds.mean(axis=0), q=np.array([1., 0, 0, 0])))
        backend.planning_world.add_object(name, box)
        self._bowl_motion_active = True
        try:
            # Only the fixed mounted base intersects the tabletop by design.
            backend.acm.set_entry("panda_link0", name, True)
            self._bowl_refresh_avatar(arm_tag)
            yield
        finally:
            backend.planning_world.remove_object(name)
            backend.acm.remove_entry(name)
            self._bowl_motion_active = previous

    def _bowl_fk_tcp(self, arm_tag, joints):
        arm = self.robot.get_arm(arm_tag)
        backend = arm.planner._mplib
        backend.pinocchio_model.compute_forward_kinematics(arm.planner._pad_qpos(joints))
        link = backend.pinocchio_model.get_link_pose(backend.move_group_link_id)
        if not all(_valid_pose(p) for p in (link, arm.origin_pose, arm.tcp_offset)):
            return None
        return arm.origin_pose * Pose(np.asarray(link.p), np.asarray(link.q)) * arm.tcp_offset

    def _bowl_plan_valid(self, plan, seed, target, arm_tag, *, tiny=False):
        arm = self.robot.get_arm(arm_tag)
        if plan is None or not plan.success or not _valid_pose(target):
            return False
        pos, vel = np.asarray(plan.position), np.asarray(plan.velocity)
        seed = np.asarray(seed)
        if (pos.ndim != 2 or pos.shape[1] != arm.n_arm or not len(pos)
                or vel.shape != pos.shape or seed.shape != (arm.n_arm,)
                or not all(np.all(np.isfinite(x)) for x in (pos, vel, seed))):
            return False
        path = np.vstack([seed, pos])
        if (np.linalg.norm(pos[-1] - seed) > 3.5
                or np.max(np.abs(np.diff(path, axis=0))) > .30
                or (tiny and np.max(np.abs(pos[-1] - seed)) > .30)):
            return False
        for joints in path:
            full = arm.planner._pad_qpos(joints)
            if full.shape != (9,) or not np.all(np.isfinite(full)):
                return False
            full = full.copy()
            full[7:9] = .04
            if arm.planner._mplib.check_for_env_collision(full):
                return False
        return _pose_matches(self._bowl_fk_tcp(arm_tag, pos[-1]), target, tiny)

    def _bowl_grasp_plan_valid(self, plans, arm_tag):
        pre, grasp, grasp_link, pre_link = plans
        arm = self.robot.get_arm(arm_tag)
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        for plan, link in ((pre, pre_link), (grasp, grasp_link)):
            if not _valid_pose(link) or not _valid_pose(arm.tcp_offset):
                return False
            if not self._bowl_plan_valid(plan, seed, link * arm.tcp_offset, arm_tag):
                return False
            seed = plan.position[-1].copy()
        return True

    def _bowl_arrived(self, target, arm_tag, label, *, tiny=False):
        raw = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        if (raw.shape != (7,) or not np.all(np.isfinite(raw))
                or not np.isclose(np.linalg.norm(raw[3:]), 1., atol=1e-4)):
            self._bowl_reject(f"{label}: invalid measured TCP")
        actual = Pose.from_pose7(raw)
        if not _pose_matches(actual, target, tiny, self.BOWL_TINY_ARRIVAL_M):
            self._bowl_reject(f"{label}: measured arrival rejected")
        return actual

    def _bowl_move(self, target, arm_tag, label, *, tiny=False, check_arrival=True):
        arm = self.robot.get_arm(arm_tag)
        if not _valid_pose(target) or not _valid_pose(arm.tcp_offset) or not hasattr(arm.planner, "_mplib"):
            self._bowl_reject(f"{label}: invalid native target")
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            self._bowl_reject(f"{label}: invalid joint seed")
        self._bowl_refresh_avatar(arm_tag)
        link = tcp_to_link_pose(target, arm.tcp_offset)
        # mplib's screw integration can stall on a leftover twist for a
        # reachable straight move; re-integrating with a slightly different
        # step resolves that quantisation case (same straight path).
        plan, error = None, None
        for step in self.BOWL_SCREW_QPOS_STEPS:
            try:
                plan = arm.planner.plan_screw_path(seed.copy(), link.to_pose7(), qpos_step=step)
            except RuntimeError as exc:
                plan, error = None, exc
                continue
            if self._bowl_plan_valid(plan, seed, target, arm_tag, tiny=tiny):
                break
        else:
            if plan is None and error is not None:
                self._bowl_reject(f"{label}: native screw rejected: {error}")
            self._bowl_reject(f"{label}: path, collision or FK rejected")
        self.execute_plan(plan, arm_tag)
        for _ in range(60):
            self.step_sim()
        if not check_arrival:
            return None
        actual = self._bowl_arrived(target, arm_tag, label, tiny=tiny)
        print(f"[bowl-motion] {label}: TCP={actual.p}", flush=True)
        return actual

    def _bowl_refine_open_grasp(self, target, arm_tag, get_center):
        """Correct measured PD lag after PUBLIC acquisition, before closing."""
        center = np.asarray(get_center(), dtype=float).copy()
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            self._bowl_reject("grasp refinement: invalid bowl center")
        correction = np.zeros(3)
        for attempt in range(4):
            actual = self._bowl_arrived(target, arm_tag, "grasp refinement")
            current_center = np.asarray(get_center(), dtype=float)
            if (current_center.shape != (3,) or not np.all(np.isfinite(current_center))
                    or np.linalg.norm(current_center - center) > .003):
                self._bowl_reject("grasp refinement: bowl moved before closing")
            error = target.p - actual.p
            distance = float(np.linalg.norm(error))
            angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1))
            if distance <= .001 and angle <= .05:
                return actual
            if attempt == 3 or distance > self.BOWL_REFINE_MAX_ERROR_M or angle > .05:
                self._bowl_reject("grasp refinement: measured rim alignment rejected")
            correction += error
            if np.linalg.norm(correction) > self.BOWL_REFINE_MAX_CORRECTION_M:
                self._bowl_reject("grasp refinement: correction bound exceeded")
            # The command is biased by the measured sag, so the arm settles
            # about one sag away from it; the next pass checks the arrival
            # against the true rim target instead.
            self._bowl_move(Pose(target.p + correction, target.q), arm_tag,
                            "open grasp correction", tiny=True, check_arrival=False)

    def execute_plan(self, result, arm_tag, gripper_result=None):
        if result is not None and getattr(self, "_bowl_motion_active", False):
            dt = float(self.scene.sim.dt)
            if not np.isfinite(dt) or dt <= 0:
                self._bowl_reject("invalid execution clock")
            ratio = .004 / dt
            repeats = int(round(ratio))
            if repeats < 1 or repeats > 16 or not np.isclose(ratio, repeats, rtol=0, atol=1e-8):
                self._bowl_reject("physical clock does not divide native4ms samples")
            pos, vel = np.asarray(result.position), np.asarray(result.velocity)
            if (not result.success or pos.ndim != 2 or not len(pos) or vel.shape != pos.shape
                    or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel))):
                self._bowl_reject("invalid execution trajectory")
            if gripper_result is not None:
                gripper_result = np.asarray(gripper_result)
                if gripper_result.shape != (len(pos),) or not np.all(np.isfinite(gripper_result)):
                    self._bowl_reject("invalid gripper trajectory")
                gripper_result = np.repeat(gripper_result, repeats)
            result = PlanResult(True, np.repeat(pos, repeats, axis=0), np.repeat(vel, repeats, axis=0))
        return super().execute_plan(result, arm_tag, gripper_result=gripper_result)
