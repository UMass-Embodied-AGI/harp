"""Checked MPLib execution for categorize; no shared controller changes."""
from functools import wraps

import numpy as np

from ...avatar.collision_checker import sample_capsule_planner_points
from ...base_task_parts.grasp_select import mplib_world_link_pose
from ...grasp import tcp_to_link_pose
from ...planning.base import PlanResult
from ...utils import Pose


class CategorizeMotionRejected(RuntimeError):
    """Expected planning, tracking or retention rejection."""


def stop_on_motion_rejection(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except CategorizeMotionRejected as exc:
            self.plan_success = False
            print(f'[categorize-motion] stopped: {exc}', flush=True)
            return False
    return checked


def valid_pose(pose):
    values = np.asarray(pose, dtype=float)
    return (values.shape == (7,) and np.all(np.isfinite(values))
            and np.isclose(np.linalg.norm(values[3:]), 1., atol=1e-4))


def pose_errors(actual, target):
    if not valid_pose(actual) or not valid_pose(target):
        return float('inf'), float('inf')
    return (float(np.linalg.norm(np.asarray(actual[:3]) - target[:3])),
            float(2 * np.arccos(np.clip(abs(np.dot(actual[3:], target[3:])), 0., 1.))))


def timed_plan(plan, sim_dt, slowdown=1., gripper=None):
    """Resample the native 4ms timeline on the actual physical clock."""
    if (not np.isfinite(sim_dt) or not 0 < sim_dt <= .004
            or not np.isfinite(slowdown) or slowdown < 1):
        raise ValueError('invalid categorize execution clock')
    pos, vel = np.asarray(plan.position), np.asarray(plan.velocity)
    if (not plan.success or pos.ndim != 2 or not len(pos) or vel.shape != pos.shape
            or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel))):
        raise ValueError('invalid categorize trajectory')
    old = np.arange(len(pos)) * .004 * slowdown
    new = np.linspace(0., old[-1], int(np.ceil(old[-1] / sim_dt)) + 1)
    def sample(values):
        return np.column_stack([np.interp(new, old, values[:, j]) for j in range(values.shape[1])])
    grip = gripper
    if grip is not None:
        grip = np.asarray(grip, dtype=float)
        if grip.shape != (len(pos),) or not np.all(np.isfinite(grip)):
            raise ValueError('gripper timeline does not match motion')
        grip = np.interp(new, old, grip)
    return PlanResult(True, sample(pos), sample(vel) / slowdown), grip


class CategorizeMotionMixin:
    # Native duration, with a bounded terminal hold for physical PD tracking.
    _REPLAY_SUBSTEPS = 1

    def select_and_execute_grasp(self, *args, **kwargs):
        arm_tag = args[2] if len(args) > 2 else kwargs['arm_tag']
        self._categorize_refresh_obstacles(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        extra_check = kwargs.pop('post_plan_check', None)
        previous = getattr(self, '_categorize_grasp_targets', {})
        self._categorize_grasp_targets = {}

        def check(plans):
            pre, grasp, grasp_link, pre_link = plans
            current = np.asarray(arm.get_arm_qpos(), dtype=float)
            if (not self._categorize_plan_valid(arm, pre, current, pre_link.to_pose7())
                    or not self._categorize_plan_valid(arm, grasp, pre.position[-1], grasp_link.to_pose7())):
                return False
            if extra_check is not None and not extra_check(plans):
                return False
            self._categorize_grasp_targets = {
                id(pre): pre_link.to_pose7(), id(grasp): grasp_link.to_pose7(),
            }
            return True

        try:
            return super().select_and_execute_grasp(*args, post_plan_check=check, **kwargs)
        finally:
            self._categorize_grasp_targets = previous

    def _reject_categorize_motion(self, reason):
        self.plan_success = False
        raise CategorizeMotionRejected(reason)

    def _categorize_refresh_obstacles(self, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        if not hasattr(arm.planner, '_mplib'):
            self._reject_categorize_motion('scripted categorize requires MPLib')
        parts = [self._categorize_scene_points(
            skip_actor=getattr(self, '_categorize_skip_actor', None),
            skip_basket_idx=getattr(self, '_categorize_skip_basket', None),
        )]
        collider = getattr(self, 'avatar_collider', None)
        if collider is not None:
            points = sample_capsule_planner_points(collider.current_capsules(), inflate_factor=1.10)
            if points is not None and points.size:
                parts.append(points)
            for _name, center, axes, half in collider.current_boxes():
                coordinates = [np.linspace(-h-.005, h+.005, max(2, int(2*h/.015)+2)) for h in half]
                grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1).reshape(-1, 3)
                parts.append(grid @ np.asarray(axes).T + center)
        neutral = getattr(self, '_neutral_object_obstacle_points', None)
        if neutral is not None:
            points = neutral()
            if points is not None and points.size:
                parts.append(points)
        cloud = np.concatenate(parts)
        arm.planner.update_obstacles(cloud, resolution=self._OBSTACLE_RESOLUTION)
        # Record the exact cloud used by the opt-in native split planner.
        from .split import enabled, identity_of_cloud
        if enabled(self):
            self._categorize_last_cloud_identity = identity_of_cloud(cloud)
        self._categorize_install_table(arm.planner._mplib)

    def _categorize_install_table(self, backend):
        if getattr(self, '_categorize_table_backend', None) is backend:
            return
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject
        half_y = self.TABLE_HALF_SIZE_Y if self._CATEGORIZE_COLUMN_LAYOUT else self._TABLE_HALF_SIZE_Y
        center = np.r_[self.table_offset, self.TABLE_TOP_Z - self.TABLE_THICKNESS / 2]
        size = np.array([1.60, 2 * half_y, self.TABLE_THICKNESS])
        if not np.all(np.isfinite(np.r_[center, size])) or np.any(size <= 0):
            self._reject_categorize_motion('invalid table bounds')
        backend.planning_world.add_object('categorize_tabletop', CollisionObject(
            Box(*size), MplibPose(p=center, q=np.array([1., 0., 0., 0.])),
        ))
        roots = [name for name, idx in backend.link_name_2_idx.items() if idx == 0]
        if len(roots) != 1:
            self._reject_categorize_motion('cannot identify fixed table-mounted root')
        backend.planning_world.get_allowed_collision_matrix().set_entry(roots[0], 'categorize_tabletop', True)
        self._categorize_table_backend = backend

    def _categorize_inputs(self, target, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        current, target = np.asarray(arm.get_arm_qpos(), dtype=float), np.asarray(target, dtype=float)
        if not valid_pose(target) or current.shape != (arm.n_arm,) or not np.all(np.isfinite(current)):
            self._reject_categorize_motion('invalid pose or measured joints')
        self._categorize_refresh_obstacles(arm_tag)
        return arm, current.copy(), target.copy()

    def _categorize_plan_valid(self, arm, plan, current, target):
        if plan is None or not plan.success:
            return False
        pos, vel = np.asarray(plan.position), np.asarray(plan.velocity)
        if (pos.ndim != 2 or pos.shape[1] != arm.n_arm or not len(pos)
                or vel.shape != pos.shape or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel))):
            return False
        if (np.max(np.abs(np.diff(np.vstack([current, pos]), axis=0))) > .30
                or np.max(np.abs(pos - current)) > 3.5):
            return False
        backend = arm.planner._mplib
        limits = np.asarray(backend.joint_limits)[np.asarray(backend.move_group_joint_indices)]
        if np.any(pos < limits[:, 0] - .001) or np.any(pos > limits[:, 1] + .001):
            return False
        endpoint = mplib_world_link_pose(arm.planner, pos[-1])
        if endpoint is None:
            return False
        distance, angle = pose_errors(endpoint.to_pose7(), target)
        if distance > .01 or angle > .05:
            return False
        # Native TOPP can overshoot collision-free geometric screw vertices.
        for joints in pos:
            full = arm.planner._pad_qpos(joints.copy())
            if backend.check_for_self_collision(full) or backend.check_for_env_collision(full):
                return False
        return True

    def _categorize_arrival(self, target, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        tcp = np.asarray(arm.get_ee_pose(), dtype=float)
        if not valid_pose(tcp):
            self._reject_categorize_motion('invalid measured TCP')
        actual = tcp_to_link_pose(Pose.from_pose7(tcp), arm.tcp_offset).to_pose7()
        distance, angle = pose_errors(actual, target)
        if distance > .01 or angle > .05:
            self._reject_categorize_motion(f'arrival xyz={distance:.4f}m angle={angle:.4f}rad')

    def _execute_categorize_plan(self, result, arm_tag, gripper_result=None):
        arm = self.robot.get_arm(arm_tag)
        if result is None or not result.success or not np.asarray(result.position).size:
            self._reject_categorize_motion('empty or unsuccessful execution plan')
        endpoint = mplib_world_link_pose(arm.planner, result.position[-1])
        if endpoint is None:
            self._reject_categorize_motion('execution endpoint has no FK')
        expected = getattr(self, '_categorize_grasp_targets', {}).get(id(result), endpoint.to_pose7())
        arm, current, target = self._categorize_inputs(expected, arm_tag)
        if not self._categorize_plan_valid(arm, result, current, target):
            self._reject_categorize_motion('execution path failed validation')
        timed, grip = timed_plan(result, float(self.scene.sim.dt), float(self._REPLAY_SUBSTEPS), gripper_result)
        # Dispatch the timed trajectory through the existing task execution
        # hook. Seal controllers keep their lazy, per-step gripper sequence.
        self.execute_plan(timed, arm_tag, gripper_result=grip)
        hold = arm.gripper_val
        for _ in range(int(np.ceil(.10 / float(self.scene.sim.dt)))):
            self.robot.set_arm_joints(timed.position[-1], arm_tag)
            self.robot.set_gripper(hold, arm_tag)
            self.step_sim()
        self._categorize_arrival(target, arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        arm, current, target = self._categorize_inputs(link_pose7, arm_tag)
        result = arm.planner.plan_path(current.copy(), target)
        if not self._categorize_plan_valid(arm, result, current, target):
            self._reject_categorize_motion('MPLib transit path/FK')
        self._execute_categorize_plan(result, arm_tag)
        self._categorize_arrival(target, arm_tag)
        return result

    def move_and_execute(self, pose, arm_tag):
        return self._move_seeded(pose, arm_tag)

    def _move_tcp_screw(self, target_tcp, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        target = tcp_to_link_pose(target_tcp, arm.tcp_offset).to_pose7()
        arm, current, target = self._categorize_inputs(target, arm_tag)
        pick = getattr(self, '_categorize_active_pick', None)
        if pick is not None and getattr(self, '_categorize_hold_local', None) is None:
            actual, center = np.asarray(arm.get_ee_pose()), np.asarray(pick.get_center())
            if min(actual[2], target_tcp.p[2]) <= center[2] + pick.radius + .015:
                delta = np.linalg.norm(actual[:2] - target_tcp.p[:2])
                if not np.isfinite(delta) or delta > .01:
                    self._reject_categorize_motion('pre-close lateral correction exceeds 10mm')
        first_lift = bool(getattr(self, '_categorize_split_lift_armed', False))
        if pick is not None and getattr(self, '_categorize_hold_local', None) is not None:
            self._categorize_split_lift_armed = False
        for precision in (.005, .002, .001):
            result = arm.planner.plan_screw_path(current.copy(), target, qpos_step=precision)
            if self._categorize_plan_valid(arm, result, current, target):
                self._execute_categorize_plan(result, arm_tag)
                self._categorize_arrival(target, arm_tag)
                return True
        # Only the first post-close vertical milk lift may use the checked
        # midpoint split; acquisition never falls back to a curved RRT path.
        from .split import enabled, small_lift_split
        if first_lift and enabled(self) and small_lift_split(self, target_tcp, arm_tag):
            return True
        self._reject_categorize_motion('straight MPLib path/FK')

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=None):
        return self._move_tcp_screw(self._top_down_tcp(target_pos, yaw=tcp_yaw), arm_tag)

    def _move_cartesian(self, start_pos, end_pos, arm_tag, **kwargs):
        return self._move_screw(end_pos, arm_tag, tcp_yaw=kwargs.get('tcp_yaw'))

    def _teleport_to_tcp(self, tcp_pose, arm_tag):
        # Historical name, always a checked straight physical motion.
        return self._move_tcp_screw(tcp_pose, arm_tag)

    def _joint_interp_to(self, pose7, arm_tag, **kwargs):
        arm = self.robot.get_arm(arm_tag)
        return self._move_tcp_screw(Pose.from_pose7(pose7) * arm.tcp_offset, arm_tag)

    def _solve_ik(self, tcp_pose, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        target = tcp_to_link_pose(tcp_pose, arm.tcp_offset).to_pose7()
        arm, current, target = self._categorize_inputs(target, arm_tag)
        result = arm.planner.solve_ik(current.copy(), target, num_waypoints=2, log=False)
        if not self._categorize_plan_valid(arm, result, current, target):
            self._reject_categorize_motion('MPLib IK/FK/continuity')
        return result.position[-1].copy()

    def _slow_replay(self, pos, vel, arm_tag):
        return self.execute_plan(PlanResult(True, pos, vel), arm_tag)

    def _categorize_begin_hold(self, arm_tag):
        pick = self._categorize_active_pick
        tcp = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        center = np.asarray(pick.get_center(), dtype=float)
        if not valid_pose(tcp) or center.shape != (3,) or not np.all(np.isfinite(center)):
            self._reject_categorize_motion('invalid closed-grasp geometry')
        self._categorize_hold_local = Pose.from_pose7(tcp).to_matrix()[:3, :3].T @ (center - tcp[:3])
        self._categorize_split_lift_armed = True

    def step_sim(self):
        result = super().step_sim()
        local = getattr(self, '_categorize_hold_local', None)
        if local is not None:
            pick = self._categorize_active_pick
            tcp = np.asarray(self.robot.get_arm(self._categorize_hold_arm).get_ee_pose(), dtype=float)
            if not valid_pose(tcp):
                self._reject_categorize_motion('invalid held TCP')
            expected = Pose.from_pose7(tcp).to_matrix()[:3, :3] @ local
            drift = np.linalg.norm(np.asarray(pick.get_center()) - tcp[:3] - expected)
            if not np.isfinite(drift) or drift > .03:
                self._reject_categorize_motion(f'held object/TCP drift={drift:.4f}m')
        return result

    def open_gripper(self, arm_tag, num_steps=200):
        self._categorize_hold_local = None
        return super().open_gripper(arm_tag, num_steps=num_steps)

    def set_gripper(self, value, arm_tag, num_steps=200):
        pick = getattr(self, '_categorize_active_pick', None)
        if pick is not None and value < 1. and getattr(self, '_categorize_hold_local', None) is None:
            moved = np.linalg.norm(np.asarray(pick.get_center()) - self._categorize_pick_center)
            if not np.isfinite(moved) or moved > .01:
                self._reject_categorize_motion(f'object moved {moved:.4f}m before closure')
        return super().set_gripper(value, arm_tag, num_steps=num_steps)
