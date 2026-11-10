"""Dump-family MPLib motion checks; shared manipulation remains unchanged."""

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..base_task_parts.grasp_select import mplib_world_link_pose
from ..grasp import tcp_to_link_pose
from ..planning.base import PlanResult
from ..utils import Pose


class DumpMotionRejected(RuntimeError):
    """Expected motion rejection, distinct from an unexpected runtime error."""


class DumpSlipOverBin(RuntimeError):
    """The held payload slipped past the retention gate while above the bin."""


def valid_pose(pose):
    values = np.asarray(pose, dtype=float)
    return (values.shape == (7,) and np.all(np.isfinite(values))
            and np.isclose(np.linalg.norm(values[3:]), 1.0, atol=1e-4))


def pose_errors(actual, target):
    if not valid_pose(actual) or not valid_pose(target):
        return float('inf'), float('inf')
    return (float(np.linalg.norm(actual[:3] - target[:3])),
            float(2 * np.arccos(np.clip(abs(np.dot(actual[3:], target[3:])), 0, 1))))


def retime_plan(plan, sim_dt, gripper=None):
    """Preserve MPLib's 4ms trajectory times on the task's physical clock."""
    if not np.isfinite(sim_dt) or sim_dt <= 0 or sim_dt > .004:
        raise ValueError('dump trajectory execution needs a finite dt in (0, 0.004]')
    position = np.asarray(plan.position, dtype=float)
    velocity = np.asarray(plan.velocity, dtype=float)
    if (position.ndim != 2 or not len(position) or velocity.shape != position.shape
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity))):
        raise ValueError('invalid dump trajectory')
    if len(position) < 2 or np.isclose(sim_dt, .004):
        return plan, gripper
    original_time = np.arange(len(position)) * .004
    times = np.linspace(0., original_time[-1], int(np.ceil(original_time[-1] / sim_dt)) + 1)
    sample = lambda values: np.column_stack([
        np.interp(times, original_time, values[:, j]) for j in range(values.shape[1])
    ])
    timed_gripper = gripper
    if gripper is not None:
        values = np.asarray(gripper, dtype=float)
        if values.shape != (len(position),) or not np.all(np.isfinite(values)):
            raise ValueError('dump gripper timeline must match the motion timeline')
        timed_gripper = np.interp(times, original_time, values)
    return PlanResult(True, sample(position), sample(velocity)), timed_gripper


def release_chord_vertices(path):
    """Parameterize dense release screw vertices by joint-space distance."""
    vertices = np.asarray(path, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 7
            or not 2 <= len(vertices) <= 4096 or not np.all(np.isfinite(vertices))):
        raise ValueError('invalid release screw vertices')
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    if np.any(lengths > .05):
        raise ValueError('release screw vertices are not a dense path')
    keep = np.r_[True, lengths > 1e-12]
    unique = vertices[keep]
    if len(unique) < 2:
        return vertices.copy()
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(unique, axis=0), axis=1))]
    samples = np.linspace(0., distance[-1], len(vertices))
    result = np.column_stack([np.interp(samples, distance, unique[:, j]) for j in range(7)])
    result[[0, -1]] = vertices[[0, -1]]
    return result


def plan_release_screw(arm, current, target):
    """Use native timing on uniform-distance input, then check the timed path."""
    planner, backend = arm.planner, arm.planner._mplib
    original_topp = backend.TOPP
    previous = vars(backend).get('TOPP')

    def chord_topp(path, *args, **kwargs):
        # Native TOPP uses vertex index for its spline parameter. A very short
        # terminal screw segment can create a long near-stationary interval.
        # Change only its input parameterization; native limits and timestamps
        # still determine every physical command and simulation step.
        vertices = release_chord_vertices(path)
        result = original_topp(vertices, *args, **kwargs)
        print(f'[dump-release] chord vertices={len(vertices)} samples={len(result[1])}'
              f' native_duration={float(result[4]):.6f}s', flush=True)
        return result

    backend.TOPP = chord_topp
    try:
        result = planner.plan_screw_path(current.copy(), target, qpos_step=.005, log=False)
    finally:
        if previous is None:
            del backend.TOPP
        else:
            backend.TOPP = previous
    if result is None or not result.success:
        return result
    position, velocity = np.asarray(result.position), np.asarray(result.velocity)
    if (position.ndim != 2 or position.shape[1] != 7 or not len(position)
            or velocity.shape != position.shape or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))):
        return PlanResult(False)
    for joints in position:
        full = planner._pad_qpos(joints.copy())
        if backend.check_for_self_collision(full) or backend.check_for_env_collision(full):
            return PlanResult(False)
    return result


class DumpBinMotionMixin:
    """Checked adapters for the legacy dump pick sequence, without IK fallback."""
    UNLOADED_ARRIVAL_ACCEPT = .015
    WRIST_ROLL_MAX_EXCURSION = 4.2
    # Final open-jaw centering corrects the steady 6-10 mm PD sag (was capped at 6 mm).
    CENTER_CORRECTION_CEILING = .012
    CENTER_CORRECTION_BUDGET = .014

    def step_sim(self):
        result = super().step_sim()
        arm_tag = getattr(self, '_dump_retention_arm', None)
        if arm_tag is not None:
            self._dump_check_retention(arm_tag)
        return result

    def _reject_dump_motion(self, reason):
        self.plan_success = False
        print(f'[dump-motion] rejected: {reason}', flush=True)
        raise DumpMotionRejected(reason)

    def execute_plan(self, result, arm_tag, gripper_result=None):
        if result is None:
            return super().execute_plan(result, arm_tag, gripper_result=gripper_result)
        expected = getattr(self, '_dump_selector_targets', {}).get(id(result))
        timed, gripper = retime_plan(result, float(self.scene.sim.dt), gripper_result)
        super().execute_plan(timed, arm_tag, gripper_result=gripper)
        self._dump_settle_endpoint(timed, arm_tag)
        if expected is not None:
            self._dump_arrival(expected, arm_tag)
            center = self._dump_object_center(self._dump_active_actor)
            shift = float(np.linalg.norm(center - self._dump_selector_center))
            if not np.isfinite(shift) or shift > .01:
                self._reject_dump_motion(f'object moved {shift:.4f}m before closure')

    def _dump_settle_endpoint(self, result, arm_tag):
        """Allow bounded physical tracking time without changing the target."""
        arm = self.robot.get_arm(arm_tag)
        target = mplib_world_link_pose(arm.planner, result.position[-1])
        if target is None:
            self._reject_dump_motion('settle endpoint has no finite FK')
        target = target.to_pose7()
        def errors():
            tcp = np.asarray(arm.get_ee_pose(), dtype=float)
            if not valid_pose(tcp):
                return float('inf'), float('inf')
            return pose_errors(tcp_to_link_pose(Pose.from_pose7(tcp), arm.tcp_offset).to_pose7(), target)
        before = errors()
        hold_gripper = arm.gripper_val
        for _ in range(int(np.ceil(.10 / float(self.scene.sim.dt)))):
            self.robot.set_arm_joints(result.position[-1], arm_tag)
            self.robot.set_gripper(hold_gripper, arm_tag)
            self.step_sim()
            self._dump_check_retention(arm_tag)
        after = errors()
        print(f'[dump-motion] endpoint settle before={before} after={after} target={target[:3]}', flush=True)

    def _install_dump_table(self, arm_tag):
        """Keep the real tabletop separate from the refreshed avatar cloud."""
        backend = self.robot.get_arm(arm_tag).planner._mplib
        installed = getattr(self, '_dump_table_backends', [])
        if any(value is backend for value in installed):
            return
        self._dump_add_table_box(backend, 'dump_tabletop', margin=0.)
        self._dump_table_backends = [*installed, backend]

    def _dump_add_table_box(self, backend, name, *, margin):
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject

        center, half = np.asarray(self.TABLE_CENTER_XY), np.asarray(self.TABLE_HALF_SIZE)
        low = np.r_[center-half, self.TABLE_TOP_Z-self.TABLE_THICKNESS_VALUE] - margin
        high = np.r_[center+half, self.TABLE_TOP_Z] + margin
        # The tabletop is a solid box. Voxelizing fitted grids made its outer
        # collision boundary depend on float32 reset rounding. Keep the exact
        # physical bounds (and explicit release margin) without that aliasing.
        if not np.all(np.isfinite(np.r_[low, high])) or np.any(high <= low):
            self._reject_dump_motion('invalid table obstacle bounds')
        box = CollisionObject(Box(*(high-low)),
                              MplibPose(p=(low+high)/2, q=np.array([1., 0., 0., 0.])))
        backend.planning_world.add_object(name, box)
        # The robot is mounted on this table. Only the fixed root may touch it;
        # moving links and every avatar pair retain normal collision checks.
        roots = [link for link, index in backend.link_name_2_idx.items() if index == 0]
        if len(roots) != 1:
            self._reject_dump_motion('cannot identify fixed planner root for table mount')
        backend.planning_world.get_allowed_collision_matrix().set_entry(roots[0], name, True)

    def _set_dump_release_clearance(self, arm_tag):
        """Allow for tracking sag while lowering beside the table."""
        backend = self.robot.get_arm(arm_tag).planner._mplib
        if getattr(self, '_dump_release_table_backend', None) is backend:
            return
        self._clear_dump_release_clearance()
        self._dump_add_table_box(backend, 'dump_release_table_margin', margin=.015)
        self._dump_release_table_backend = backend

    def _clear_dump_release_clearance(self):
        backend = getattr(self, '_dump_release_table_backend', None)
        if backend is not None:
            backend.planning_world.remove_object('dump_release_table_margin')
            self._dump_release_table_backend = None

    def _refresh_dump_obstacles(self, arm_tag):
        self._install_dump_table(arm_tag)
        parts = []
        # Neutral variant (opt-in): plan around the human's props as exact
        # MPLib boxes (checker OBB + 1 cm) instead of voxelized shells, which
        # added ~1-2 cm and closed grasps beside a chair or bowl.  Still a
        # strict superset of the boxes the collision checker scores.
        boxes_enabled = getattr(self, '_neutral_object_planner_boxes_enabled', None)
        prop_boxes = bool(boxes_enabled is not None and boxes_enabled()
                          and self._sync_neutral_object_planner_boxes(arm_tag))
        collider = getattr(self, 'avatar_collider', None)
        if collider is not None:
            points = sample_capsule_planner_points(collider.current_capsules(), inflate_factor=1.10)
            if points is not None and points.size:
                parts.append(points)
            for _name, center, axes, half in collider.current_boxes():
                if prop_boxes and str(_name).startswith('neutral_obj:'):
                    continue
                coordinates = [np.linspace(-h-.005, h+.005, max(2, int(2*h/.015)+2)) for h in half]
                grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1).reshape(-1, 3)
                parts.append(grid @ np.asarray(axes).T + center)
        # Neutral props are separate from avatar capsules in some snapshots.
        neutral_points = None if prop_boxes else getattr(self, '_neutral_object_obstacle_points', None)
        if neutral_points is not None:
            points = neutral_points()
            if points is not None and points.size:
                parts.append(points)
        self.robot.get_arm(arm_tag).planner.update_obstacles(
            np.concatenate(parts) if parts else np.empty((0, 3)), resolution=.015,
        )

    def _dump_motion_inputs(self, target, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        current = np.asarray(arm.get_arm_qpos(), dtype=float)
        target = np.asarray(target, dtype=float)
        if (not valid_pose(target) or current.shape != (arm.n_arm,)
                or not np.all(np.isfinite(current))):
            self._reject_dump_motion('invalid pose or measured joints')
        if not hasattr(arm.planner, '_mplib'):
            self._reject_dump_motion('scripted dump requires MPLib')
        self._refresh_dump_obstacles(arm_tag)
        return arm, current.copy(), target.copy()

    def _dump_plan_valid(self, arm, plan, current, target, *, max_step=.30):
        if plan is None or not plan.success:
            return False
        pos, vel = np.asarray(plan.position), np.asarray(plan.velocity)
        if (pos.ndim != 2 or pos.shape[1] != arm.n_arm or not len(pos)
                or vel.shape != pos.shape or not np.all(np.isfinite(pos))
                or not np.all(np.isfinite(vel))):
            return False
        if np.max(np.abs(np.diff(np.vstack([current, pos]), axis=0))) > max_step:
            print('[dump-plan] rejected waypoint continuity', flush=True)
            return False
        # A large swing of any joint that moves the arm through space is
        # rejected; the wrist roll (last joint) only turns the hand in place,
        # so during top-down grasp selection it may swing further (unwinding
        # to the turned grasp yaw).  Side/body grasps keep the 3.5 rad bound:
        # their retained lift depends on the arrival branch.
        excursion = np.max(np.abs(pos - current), axis=0)
        limits_by_joint = np.full(excursion.shape, 3.5)
        if getattr(self, '_dump_allow_wrist_roll', False):
            limits_by_joint[-1] = self.WRIST_ROLL_MAX_EXCURSION
        if np.any(excursion > limits_by_joint):
            joint = int(np.argmax(excursion - limits_by_joint))
            print(f'[dump-plan] rejected joint excursion={float(excursion[joint]):.6f}rad joint={joint + 1}', flush=True)
            return False
        backend = arm.planner._mplib
        limits = np.asarray(backend.joint_limits)[np.asarray(backend.move_group_joint_indices)]
        if np.any(pos < limits[:, 0] - .001) or np.any(pos > limits[:, 1] + .001):
            print('[dump-plan] rejected joint limits', flush=True)
            return False
        endpoint = mplib_world_link_pose(arm.planner, pos[-1])
        if endpoint is None:
            return False
        distance, angle = pose_errors(endpoint.to_pose7(), target)
        if distance > .01 or angle > .05:
            print(f'[dump-motion] FK xyz={distance:.4f}m angle={angle:.4f}rad', flush=True)
            return False
        return True

    def _dump_target_at_transport(self, target, arm):
        """True when the arrival target's TCP is at or above the transport altitude."""
        try:
            floor = float(self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE) - .005
            tcp_z = float((Pose.from_pose7(np.asarray(target, dtype=float)) * arm.tcp_offset).p[2])
        except Exception:
            return False
        return bool(np.isfinite(tcp_z) and tcp_z >= floor)

    def _dump_arrival(self, target, arm_tag, *, center_grasp=False):
        arm = self.robot.get_arm(arm_tag)
        correction = np.zeros(3)
        tolerance = .002 if center_grasp else .01
        for attempt in range(4):
            tcp = np.asarray(arm.get_ee_pose(), dtype=float)
            if not valid_pose(tcp):
                self._reject_dump_motion('nonfinite measured TCP')
            actual_link = tcp_to_link_pose(Pose.from_pose7(tcp), arm.tcp_offset)
            distance, angle = pose_errors(actual_link.to_pose7(), target)
            self._dump_check_retention(arm_tag)
            if distance <= tolerance and angle <= .05:
                return
            # Small loaded tracking residuals can persist after a fixed hold.
            # Correct them with checked physical motions, keeping the original
            # arrival gate. Final apple centering has a separate, tighter
            # budget and cannot rescue a large open-jaw approach error.
            # An unloaded miss is normally rejected outright, because a
            # correction near the object could sweep it (the picked object is
            # not in the planner cloud). At the task's own transport altitude
            # nothing on the table is within reach of a bounded correction, and
            # the miss there is a steady-state PD sag of ~8-10 mm that the loaded
            # path already corrects. So a transport-altitude unloaded miss gets
            # the same bounded, collision-checked correction; lower down it does not.
            unloaded = getattr(self, '_dump_hold_offset', None) is None and not center_grasp
            # An empty-hand approach miss up to 15 mm is accepted as is; the
            # later close-range centering and pre-closure motion checks still apply.
            if unloaded and distance <= self.UNLOADED_ARRIVAL_ACCEPT and angle <= .05:
                print(f'[dump-motion] accepted unloaded arrival xyz={distance:.4f}m', flush=True)
                return
            if (attempt == 3 or distance > (self.CENTER_CORRECTION_CEILING if center_grasp else .025) or angle > .05
                    or (unloaded and not self._dump_target_at_transport(target, arm))):
                self._reject_dump_motion(f'arrival xyz={distance:.4f}m angle={angle:.4f}rad')
            error = np.asarray(target[:3]) - actual_link.p
            correction += error * min(1., (.004 if center_grasp else .008) / np.linalg.norm(error))
            budget = self.CENTER_CORRECTION_BUDGET if center_grasp else .020
            if np.linalg.norm(correction) > budget:
                self._reject_dump_motion(f'arrival correction exceeds {budget*1000:g}mm budget')
            commanded = np.asarray(target, dtype=float).copy()
            commanded[:3] += correction
            arm, current, commanded = self._dump_motion_inputs(commanded, arm_tag)
            # The fine step fails natively near the wrist limits; the coarser step
            # the other checked straight moves use is tried only after it fails.
            for qpos_step in (.002, .005):
                result = arm.planner.plan_screw_path(current.copy(), commanded, qpos_step=qpos_step)
                if self._dump_plan_valid(arm, result, current, commanded):
                    break
            else:
                self._reject_dump_motion('arrival correction has no checked straight path')
            print(f'[dump-motion] arrival correction={correction} residual={distance:.4f}m', flush=True)
            self.execute_plan(result, arm_tag)
            # Closing the fingers reuses the cached arm PD target. Keep the
            # checked compensation through closure; measured-joint reseeding
            # here would discard it and restore the unloaded tracking bias.
            if not center_grasp:
                self._reseed_cached_arm_target(arm)
            if center_grasp:
                center = self._dump_object_center(self._dump_active_actor)
                if (not np.all(np.isfinite(center))
                        or np.linalg.norm(center-self._dump_selector_center) > .01):
                    self._reject_dump_motion('object moved before centered closure')

    def _dump_check_retention(self, arm_tag):
        pick = getattr(self, '_dump_active_pick', None)
        offset = getattr(self, '_dump_hold_offset', None)
        if pick is None or offset is None:
            return
        center = np.asarray(pick.get_center(), dtype=float)
        tcp = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        local = getattr(self, '_dump_hold_local', None)
        if local is not None:
            if not valid_pose(tcp):
                self._reject_dump_motion('invalid retained TCP rotation')
            offset = Pose.from_pose7(tcp).to_matrix()[:3, :3] @ local
        drift = float(np.linalg.norm(center - tcp[:3] - offset))
        if (np.isfinite(drift) and drift > .03
                and getattr(self, '_dump_release_descent', False)
                and self._dump_center_over_bin(center)):
            # Lowering straight down above the bin, a payload sliding in the
            # jaws can only fall into the bin; the final support check decides.
            if not getattr(self, '_dump_descent_slip_noted', False):
                self._dump_descent_slip_noted = True
                print(f'[dump-motion] payload slipping over bin during release descent '
                      f'drift={drift:.4f}m; continuing', flush=True)
            return
        if (np.isfinite(drift) and drift > .03 and not getattr(self, '_dump_releasing', False)
                and self._dump_center_over_bin(center, margin=.03)):
            # Slipping while already above the bin: release it here instead
            # of carrying a loose payload further; support is checked after.
            print(f'[dump-motion] payload slipping above bin drift={drift:.4f}m; releasing here', flush=True)
            raise DumpSlipOverBin(f'drift={drift:.4f}m')
        if not np.isfinite(drift) or drift > .03:
            self._reject_dump_motion(f'held object/TCP drift={drift:.4f}m')

    def _dump_center_over_bin(self, center, margin=.01):
        if getattr(self, 'big_bin_pose', None) is None or getattr(self, '_big_bin_xy_half', None) is None:
            return False
        center = np.asarray(center, dtype=float)
        bin_xy = np.asarray(self.big_bin_pose.p[:2], dtype=float)
        half = np.asarray(self._big_bin_xy_half, dtype=float)
        return bool(center.shape == (3,) and np.all(np.isfinite(center))
                    and np.all(np.abs(center[:2] - bin_xy) < half - margin))

    def _move_seeded(self, link_pose7, arm_tag):
        arm, current, target = self._dump_motion_inputs(link_pose7, arm_tag)
        result = arm.planner.plan_path(current, target)
        if not self._dump_plan_valid(arm, result, current, target):
            self._reject_dump_motion('MPLib transit/descent plan or FK')
        self.execute_plan(result, arm_tag)
        self._reseed_cached_arm_target(arm)
        self._dump_arrival(target, arm_tag)
        return result

    def move_and_execute(self, pose, arm_tag):
        return self._move_seeded(pose, arm_tag)

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=0.0):
        arm = self.robot.get_arm(arm_tag)
        target_tcp = self._top_down_tcp(target_pos, yaw=tcp_yaw)
        target_link = tcp_to_link_pose(target_tcp, arm.tcp_offset).to_pose7()
        arm, current, target = self._dump_motion_inputs(target_link, arm_tag)
        pick = getattr(self, '_dump_active_pick', None)
        if pick is not None and getattr(self, '_dump_hold_offset', None) is None:
            actual = np.asarray(arm.get_ee_pose(), dtype=float)
            center = np.asarray(pick.get_center(), dtype=float)
            # At object height, a large correction is an unsafe lateral sweep.
            if min(actual[2], target_tcp.p[2]) <= center[2] + pick.radius + .015:
                delta = float(np.linalg.norm(actual[:2] - target_tcp.p[:2]))
                if not np.isfinite(delta) or delta > .01:
                    self._reject_dump_motion(f'pre-close lateral correction={delta:.4f}m > 0.010m')
        for precision in (.005, .002, .001):
            result = arm.planner.plan_screw_path(current.copy(), target, qpos_step=precision)
            if self._dump_plan_valid(arm, result, current, target):
                self.execute_plan(result, arm_tag)
                self._reseed_cached_arm_target(arm)
                self._dump_arrival(target, arm_tag)
                return True
        result = self._dump_straight_joint_path(arm, current, target)
        if result is None and self._dump_gate_checks_human():
            # The arm starts inside the human's inflated cloud (a hand beside
            # the grasp).  Check this straight move against the table and
            # props only; the human-aware gate checks the predicted human
            # (capsules, synchronously) before it executes.
            try:
                self.robot.get_arm(arm_tag).planner.update_obstacles(np.empty((0, 3)), resolution=.015)
                result = self._dump_straight_joint_path(arm, current, target)
            finally:
                self._refresh_dump_obstacles(arm_tag)
            if result is not None:
                print('[dump-motion] straight joint path checked without the human cloud; gate checks the human', flush=True)
        if result is not None and self._dump_plan_valid(arm, result, current, target):
            self.execute_plan(result, arm_tag)
            self._reseed_cached_arm_target(arm)
            self._dump_arrival(target, arm_tag)
            return True
        self._reject_dump_motion('straight MPLib motion has no checked path')

    # Last resort for a straight move the screw integrator cannot follow
    # (diverges or stalls near joint limits after an unusual grasp branch):
    # the MPLib IK goal nearest the current joints, joint-space interpolation
    # in small steps, accepted only if every row stays on the straight TCP
    # segment (1 cm, 0.05 rad) and is free of self/environment collision.
    def _dump_gate_checks_human(self):
        """True when a predictive human gate validates every executed plan."""
        return False

    STRAIGHT_JOINT_STEP = .003
    STRAIGHT_JOINT_TOLERANCE = (.01, .05)

    def _dump_straight_joint_path(self, arm, current, target):
        planner, backend = arm.planner, arm.planner._mplib
        ik = planner.solve_ik(current.copy(), target, num_waypoints=2, log=False)
        if ik is None or not ik.success or not np.all(np.isfinite(ik.position)):
            return None
        goal = np.asarray(ik.position[-1], dtype=float)
        # Cosine ease in/out; its peak rate is pi/2 times the mean.
        rows = int(np.ceil(np.max(np.abs(goal - current)) * np.pi / 2 / self.STRAIGHT_JOINT_STEP)) + 1
        if rows > 2000:
            return None
        t = (.5 - .5 * np.cos(np.pi * np.linspace(0., 1., max(2, rows))))[:, None]
        position = (1 - t) * current + t * goal
        start = mplib_world_link_pose(planner, current)
        if start is None:
            return None
        start = start.to_pose7()
        tolerance, angle_tolerance = self.STRAIGHT_JOINT_TOLERANCE
        for k, joints in enumerate(position):
            link = mplib_world_link_pose(planner, joints)
            if link is None:
                return None
            link = link.to_pose7()
            alpha = float(t[k, 0])
            expect = start[:3] + alpha * (np.asarray(target[:3]) - start[:3])
            _, angle = pose_errors(link, target)
            _, angle0 = pose_errors(link, start)
            if (np.linalg.norm(link[:3] - expect) > tolerance
                    or min(angle, angle0) > angle_tolerance + pose_errors(start, target)[1]):
                print(f'[dump-motion] straight joint path leaves the segment at row {k}', flush=True)
                return None
            full = planner._pad_qpos(joints.copy())
            if backend.check_for_self_collision(full) or backend.check_for_env_collision(full):
                print(f'[dump-motion] straight joint path collides at row {k}', flush=True)
                return None
        velocity = np.zeros_like(position)
        velocity[:-1] = np.diff(position, axis=0) / .004
        print(f'[dump-motion] straight joint path rows={len(position)}', flush=True)
        return PlanResult(True, position, velocity)

    def _solve_ik(self, tcp_pose, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        target = tcp_to_link_pose(tcp_pose, arm.tcp_offset).to_pose7()
        arm, current, target = self._dump_motion_inputs(target, arm_tag)
        cached = getattr(arm, '_cached_target', None)
        seed = current if cached is None else np.asarray(cached, dtype=float)[:arm.n_arm]
        if seed.shape != current.shape or not np.all(np.isfinite(seed)):
            self._reject_dump_motion('invalid tiny-lift seed')
        result = arm.planner.solve_ik(seed.copy(), target, num_waypoints=2, log=False)
        if (not self._dump_plan_valid(arm, result, seed, target)
                or np.max(np.abs(result.position[-1] - seed)) > .30):
            self._reject_dump_motion('tiny-lift MPLib IK/FK/branch')
        return result.position[-1].copy()

    def _move_cartesian(self, start_pos, end_pos, arm_tag, **kwargs):
        super()._move_cartesian(start_pos, end_pos, arm_tag, **kwargs)
        arm = self.robot.get_arm(arm_tag)
        target = tcp_to_link_pose(self._top_down_tcp(end_pos, yaw=kwargs.get('tcp_yaw', 0.0)), arm.tcp_offset)
        self._dump_arrival(target.to_pose7(), arm_tag)

    def pick_and_place(self, pick, place, arm_tag):
        from .dump_bin_grasp import pick_and_drop_topdown
        previous = getattr(self, '_dump_active_pick', None)
        previous_hold = getattr(self, '_dump_hold_offset', None)
        self._dump_active_pick, self._dump_hold_offset = pick, None
        previous_local = getattr(self, '_dump_hold_local', None)
        self._dump_hold_local = None
        previous_arm = getattr(self, '_dump_retention_arm', None)
        self._dump_retention_arm = arm_tag
        try:
            return pick_and_drop_topdown(self, pick, place, arm_tag)
        except DumpSlipOverBin:
            from .dump_bin_grasp import release_and_settle
            return release_and_settle(self, pick, arm_tag)
        except DumpMotionRejected:
            return False
        finally:
            self._dump_active_pick, self._dump_hold_offset = previous, previous_hold
            self._dump_hold_local = previous_local
            self._dump_retention_arm = previous_arm
            self._dump_idle_guard = False
            self._clear_dump_release_clearance()
