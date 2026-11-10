"""Checked body acquisition and retracted lift for dump milk and model-0 jars."""
import numpy as np
from contextlib import contextmanager, nullcontext

from ..grasp import GraspPose, get_grasp_pose, tcp_to_link_pose
from ..base_task_parts.grasp_select import mplib_world_link_pose
from ..planning.base import PlanResult
from ..utils import Pose, to_numpy


def _densify_milk_pre_path(path):
    """Keep native TOPP splines close to the collision-planned RRT segments."""
    path = np.asarray(path, dtype=float)
    if (path.ndim != 2 or path.shape[1] != 7 or not 2 <= len(path) <= 1024
            or not np.all(np.isfinite(path))):
        raise ValueError('invalid milk pre-approach RRT vertices')
    deltas = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    if np.any(deltas > 7.):
        raise ValueError('unbounded milk pre-approach RRT segment')
    counts = np.maximum(1, np.ceil(deltas/.025).astype(int))
    if counts.sum()+1 > 4096:
        raise ValueError('too many milk pre-approach RRT vertices')
    return np.concatenate([path[:1]] + [
        np.linspace(start, end, count+1)[1:]
        for start, end, count in zip(path[:-1], path[1:], counts)])


@contextmanager
def milk_approach_planning(task, arm, *, target_margin=.008):
    """Keep the open hand clear of the payload during the free RRT approach."""
    from mplib.pymp import Pose as MplibPose
    from mplib.pymp.collision_detection.fcl import Box, CollisionObject

    if not np.isfinite(target_margin) or target_margin < 0:
        task._reject_dump_motion('invalid body approach obstacle margin')
    planner, backend = arm.planner, arm.planner._mplib
    original = planner.plan_path
    previous = vars(planner).get('plan_path')
    name = 'dump_milk_pre_target'

    def pre_path(current, target, **kwargs):
        entity = getattr(task._dump_active_actor, 'entity', task._dump_active_actor)
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                or np.any(bounds[1] <= bounds[0]) or arm.n_arm != 7
                or not np.isclose(arm.gripper_val, 1.)
                or backend.planning_world.has_object(name)):
            task._reject_dump_motion('invalid open-jaw milk approach state')
        full = planner._pad_qpos(np.asarray(current, dtype=float).copy())
        if full.shape != (9,) or not np.all(np.isfinite(full)):
            task._reject_dump_motion('invalid Franka approach joints')
        full[7:] = .04
        def obstacle(margin):
            return CollisionObject(Box(*(bounds[1]-bounds[0]+2*margin)),
                                   MplibPose(p=bounds.mean(axis=0), q=np.array([1., 0., 0., 0.])))
        # Native TOPP fits a cubic spline through sparse RRT vertices. Densify
        # its input without changing native timing or omitting output checks.
        original_topp = backend.TOPP
        previous_topp = vars(backend).get('TOPP')
        def dense_topp(path, *args, **kwargs):
            return original_topp(_densify_milk_pre_path(path), *args, **kwargs)
        backend.planning_world.add_object(name, obstacle(target_margin))
        backend.TOPP = dense_topp
        try:
            result = original(full, target, **kwargs)
            if result is not None and result.success:
                for joints in result.position:
                    checked = planner._pad_qpos(np.asarray(joints, dtype=float).copy())
                    if checked.shape != (9,) or not np.all(np.isfinite(checked)):
                        return PlanResult(False)
                    checked[7:] = .04
                    if backend.check_for_env_collision(checked):
                        return PlanResult(False)
            return result
        finally:
            if previous_topp is None:
                del backend.TOPP
            else:
                backend.TOPP = previous_topp
            backend.planning_world.remove_object(name)

    planner.plan_path = pre_path
    try:
        yield
    finally:
        if previous is None:
            del planner.plan_path
        else:
            planner.plan_path = previous


def transport_targets(task, tcp, place, *, final_yaw=0., carry_center=None, retract_y=None):
    raised = tcp.p + np.array([0., 0., .05])
    retracted = raised + np.array([0., -.05, 0.])
    # Starts close to the mount need room for the high lift and wrist turn.
    retracted[1] = max(retracted[1], -.15)
    if retract_y is not None:
        if not np.isfinite(retract_y):
            task._reject_dump_motion('invalid body retraction station')
        retracted[1] = float(retract_y)
    high = np.r_[retracted[:2], max(float(place.transport_z), retracted[2])]
    if carry_center is not None:
        # The close-to-base milk grasp reaches a poor branch on a vertical
        # lift. Move slightly toward the bin while rising, with the same
        # orientation and checked straight path.
        high[0] += .05
    targets = (Pose(raised, tcp.q), Pose(retracted, tcp.q), Pose(high, tcp.q))
    if carry_center is not None:
        center = np.asarray(carry_center, dtype=float)
        bin_center = np.asarray(task.big_bin_pose.p, dtype=float)
        if (center.shape != (3,) or bin_center.shape != (3,)
                or not np.all(np.isfinite(np.r_[center, bin_center]))):
            task._reject_dump_motion('invalid milk side-carry target')
        # Carry the carton upright. Turning at the bin avoids the long lateral
        # transit while gravity pivots the carton against the turned pads.
        high = np.r_[bin_center[:2] - (center-tcp.p)[:2], high[2]]
        targets += (Pose(high, tcp.q),)
    final = task._top_down_tcp(high, yaw=final_yaw)
    if carry_center is not None:
        # The quarter turn still lets the carton pivot against both pads.
        # Preserving the side orientation cannot reach a checked descent.
        # Three shortest-arc midpoints retain an eighth turn, with the same
        # continuation and release checks.
        start_q, end_q = np.asarray(tcp.q), np.asarray(final.q)
        if np.dot(start_q, end_q) < 0:
            end_q = -end_q
        turn_q = end_q.copy()
        for _ in range(3):
            turn_q = start_q + turn_q
            turn_q /= np.linalg.norm(turn_q)
        final = Pose(high, turn_q)
    return (*targets, final)


# Approach sides for the body grasps: the original side first, then the
# opposite side, then the two quarter turns.
SIDE_TURNS = (0., np.pi, np.pi/2, -np.pi/2)


def turn_about_vertical(grasp, object_pose, scale, center, turn):
    """The same grasp approached from another side: rotate the TCP pose by
    ``turn`` about the vertical axis through the object's measured center."""
    tcp = grasp.to_world(object_pose, scale)
    center = np.asarray(center, dtype=float)
    rotation = Pose(np.zeros(3), [np.cos(turn/2), 0., 0., np.sin(turn/2)])
    turned = Pose(center + rotation.to_matrix()[:3, :3] @ (np.asarray(tcp.p) - center),
                  (rotation * Pose(np.zeros(3), tcp.q)).q)
    local = object_pose.inv() * turned
    return GraspPose(f'{grasp.name}_side{int(round(np.degrees(turn)))}', local.p, local.q,
                     pre_distance=grasp.pre_distance, source='manual', scale_frame='runtime')


def milk_body_grasp(task, object_pose, scale, model_id):
    """Keep the catalog approach, with the pads below the tapered shoulder."""
    catalog = get_grasp_pose('038_milk-box', task.MILK_SIDE_GRASP_NAME,
                             model_id, robot_type='franka')
    tcp = catalog.to_world(object_pose, scale)
    local = object_pose.inv()*Pose(tcp.p-[0., 0., .020], tcp.q)
    return GraspPose('dump_milk_lower_body', local.p, local.q,
                     pre_distance=.060, source='manual', scale_frame='runtime')


def jar_near_mount(arm, center):
    """Identify the short-reach workspace using measured geometry, not IDs."""
    center, base = np.asarray(center, dtype=float), np.asarray(arm.origin_pose.p, dtype=float)
    if center.shape != (3,) or base.shape != (3,) or not np.all(np.isfinite(np.r_[center, base])):
        raise ValueError('invalid jar or mount center')
    return bool(np.linalg.norm(center[:2]-base[:2]) < .32)


def jar_body_grasp(task, actor, object_pose, center, *, away_from_mount=False):
    """Oblique robot-facing body grip; keep wrist above the original table."""
    entity = getattr(actor, 'entity', actor)
    bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
    if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])):
        task._reject_dump_motion('invalid jar body acquisition bounds')
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        task._reject_dump_motion('invalid jar body acquisition center')
    # Pinch the broad body with a nearly horizontal approach. The sampled
    # palm gap is6.6mm; the smaller offset reduces the initial rolling moment.
    position = np.array([center[0], center[1]-.006, bounds[0, 2]+.055])
    down = task._top_down_tcp(np.zeros(3), yaw=np.pi/2)
    # Ten degrees downward from horizontal clears the palm and tabletop.
    tilt = Pose(np.zeros(3), [np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0., 0.])
    # The unrolled far-side grasp reaches the final orientation with an
    # 80-degree turn, avoiding the previous 180-degree inversion.
    tcp = Pose(position, (tilt*down).q)
    if away_from_mount:
        # Close to the mount, approaching from the opposite side leaves room
        # for the wrist. Rotate the offset with the pose about the jar center.
        opposite = Pose(np.zeros(3), [0., 0., 0., 1.])
        tcp = Pose(center+opposite.to_matrix()[:3, :3]@(tcp.p-center),
                   (opposite*Pose(np.zeros(3), tcp.q)*opposite).q)
        # Measured near-side contacts sit about2mm beyond the center along Y.
        # Recenter that lever while preserving the checked wrist orientation.
        position = tcp.p.copy()
        position[1] = center[1]+.004
        tcp = Pose(position, tcp.q)
    local = object_pose.inv()*tcp
    return GraspPose('dump_jar_body_oblique', local.p, local.q,
                     pre_distance=.050,
                     source='manual', scale_frame='runtime')


def select_body_transport(task, arm, current, target_options, *, screw_first, rotation_options):
    """Choose a complete checked continuation before executing acquisition."""
    if not 1 <= len(target_options) <= 2:
        task._reject_dump_motion('invalid body continuation option count')
    for targets in target_options:
        cursor = np.asarray(current, dtype=float).copy()
        for index, tcp in enumerate(targets):
            target = tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7()
            path = (plan_body_rotation(task, arm, cursor, tcp, screw_first=screw_first,
                                       **rotation_options) if index == len(targets)-1 else
                    plan_body_screw(task, arm, cursor, target))
            if not task._dump_plan_valid(arm, path, cursor, target):
                print(f'[dump-body] rejected continuation leg={index} target={target}', flush=True)
                break
            cursor = path.position[-1].copy()
        else:
            return targets
    return None


def plan_high_rotation(task, arm, current, tcp, *, preferred_goal=None, endpoint_check=None):
    """Bounded native RRT for a retained, high rotation; never a grasp descent."""
    planner, backend = arm.planner, arm.planner._mplib
    target = tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7()
    full = planner._pad_qpos(current.copy())
    native = backend.robot.get_base_pose().inv()*planner._to_mplib_pose(target)
    status, goals = backend.IK(native, full, n_init_qpos=20, return_closest=False)
    indices = np.asarray(backend.move_group_joint_indices)
    ranked = (sorted(np.asarray(goals).reshape(-1, len(full)),
                     key=lambda goal: np.linalg.norm(goal[indices]-current))
              if status == 'Success' and goals is not None else [])
    if preferred_goal is not None:
        preferred = np.asarray(preferred_goal, dtype=float)
        if preferred.shape == full.shape:
            ranked.insert(0, preferred.copy())
    for goal in ranked[:6]:
        if not np.all(np.isfinite(goal)) or np.max(np.abs(goal[indices]-current)) > 3.5:
            continue
        raw = backend.plan_qpos(goal_qposes=[goal], current_qpos=full,
                               time_step=.004, planning_time=5, rrt_range=.1, verbose=False)
        if not raw or raw.get('status') != 'Success':
            continue
        path = PlanResult(True, np.asarray(raw['position']), np.asarray(raw['velocity']))
        if not task._dump_plan_valid(arm, path, current, target):
            continue
        positions = []
        for joints in path.position:
            link = mplib_world_link_pose(planner, joints)
            if link is None or not np.all(np.isfinite(link.to_pose7())):
                break
            positions.append((link*arm.tcp_offset).p)
        if len(positions) != len(path.position):
            continue
        positions = np.asarray(positions)
        minimum = max(task.TABLE_TOP_Z+.15, float(tcp.p[2])-.03)
        if (np.all(np.isfinite(positions)) and np.min(positions[:, 2]) >= minimum
                and np.max(np.linalg.norm(positions[:, :2]-tcp.p[:2], axis=1)) <= .10):
            if endpoint_check is None or endpoint_check(path.position[-1].copy(), goal.copy()):
                return path
    return None


def plan_body_screw(task, arm, current, target):
    """Retry native screw discretization from the same measured start."""
    for step in (.005, .002, .001):
        path = arm.planner.plan_screw_path(current.copy(), target, qpos_step=step, log=False)
        if task._dump_plan_valid(arm, path, current, target):
            return path
    return None


def plan_body_rotation(task, arm, current, tcp, *, screw_first=False, **rotation_options):
    if screw_first:
        target = tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7()
        path = plan_body_screw(task, arm, current, target)
        if path is not None:
            return path
    return plan_high_rotation(task, arm, current, tcp, **rotation_options)


def move_checked(task, tcp, arm_tag, *, high_rotation=False, screw_first=False, **rotation_options):
    arm = task.robot.get_arm(arm_tag)
    target = tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7()
    arm, current, target = task._dump_motion_inputs(target, arm_tag)
    result = (plan_body_rotation(task, arm, current, tcp, screw_first=screw_first, **rotation_options) if high_rotation else
              plan_body_screw(task, arm, current, target))
    if not task._dump_plan_valid(arm, result, current, target):
        task._reject_dump_motion('body retained motion has no checked path')
    task.execute_plan(result, arm_tag)
    task._reseed_cached_arm_target(arm)
    task._dump_arrival(target, arm_tag)


def jar_rotation_release_check(task, place, arm_tag, reference_tcp, center, bounds, accept_goal):
    """Check a rigidly predicted payload without changing its physical state."""
    from .dump_bin_grasp import plan_release_from_state
    arm = task.robot.get_arm(arm_tag)
    center, bounds = np.asarray(center, dtype=float).copy(), np.asarray(bounds, dtype=float).copy()
    if (center.shape != (3,) or bounds.shape != (2, 3)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])):
        task._reject_dump_motion('invalid predicted jar release bounds')
    reference = reference_tcp.to_matrix()[:3, :3]
    half = (bounds[1]-bounds[0])/2
    def check(current, goal):
        link = mplib_world_link_pose(arm.planner, current)
        if link is None:
            return False
        tcp = link*arm.tcp_offset
        rotation = tcp.to_matrix()[:3, :3]@reference.T
        predicted_center = tcp.p+rotation@(center-reference_tcp.p)
        box_center = tcp.p+rotation@(bounds.mean(axis=0)-reference_tcp.p)
        predicted_half = np.abs(rotation)@half
        predicted_bounds = np.array([box_center-predicted_half, box_center+predicted_half])
        plans = plan_release_from_state(task, place, arm_tag, current, tcp.to_pose7(),
                                        predicted_center, predicted_bounds, tcp.q)
        if plans is None:
            return False
        accept_goal(goal.copy())
        return True
    return check


def pick_and_drop_side(task, pick, place, arm_tag):
    from .dump_bin import _read_scale
    from .dump_bin_grasp import plan_release, finish_drop, wait_for_jar_rest
    actor, arm = task._dump_active_actor, task.robot.get_arm(arm_tag)
    is_jar = pick.label == '031_jam-jar'
    task._dump_idle_guard = True     # stationary pre-planning waits
    try:
        task.open_gripper(arm_tag)
        for _ in range(80):
            task.step_sim()
        if is_jar:
            wait_for_jar_rest(task, pick)
    finally:
        task._dump_idle_guard = False
    object_pose = actor.get_pose()
    model_id = int(getattr(task, '_object_model_ids', {}).get(id(actor), 0))
    scale = _read_scale(pick.label, model_id)
    center_before = np.asarray(pick.get_center(), dtype=float)
    away = (is_jar and task.config.get('robot_type', 'franka') == 'franka'
            and jar_near_mount(arm, center_before))
    base_grasp = (jar_body_grasp(task, actor, object_pose, center_before, away_from_mount=away) if is_jar else
                  milk_body_grasp(task, object_pose, scale, model_id))
    far_jar = is_jar and not away

    def plan_set(turn):
        """Candidate grasps and continuations with the whole side grasp turned
        by ``turn`` about the object's vertical axis (approach from another side)."""
        grasp, final_yaw = base_grasp, (np.pi/2 if is_jar else 0.)
        if turn:
            grasp = turn_about_vertical(grasp, object_pose, scale, center_before, turn)
            final_yaw += turn
        grasps = [grasp]
        if far_jar:
            grasps.append(GraspPose(grasp.name.replace('oblique', 'oblique_long_pre', 1), grasp.pose.p,
                                    grasp.pose.q, pre_distance=.060, source='manual', scale_frame='runtime'))
        expected_tcp = grasp.to_world(object_pose, scale)
        options = [transport_targets(task, expected_tcp, place, final_yaw=final_yaw,
                                     carry_center=None if is_jar else center_before,
                                     retract_y=float(arm.origin_pose.p[1])+.35 if away else None)]
        if far_jar:
            options.append(transport_targets(task, expected_tcp, place, final_yaw=final_yaw,
                                             retract_y=float(arm.origin_pose.p[1])+.35))
        return grasps, expected_tcp, options

    task._refresh_dump_obstacles(arm_tag)
    preferred_rotation = None
    def accept_rotation(goal):
        nonlocal preferred_rotation
        preferred_rotation = goal.copy()

    def rotation_options(reference_tcp):
        if not is_jar:
            return {}
        entity = getattr(actor, 'entity', actor)
        return dict(preferred_goal=preferred_rotation,
                    endpoint_check=jar_rotation_release_check(
                        task, place, arm_tag, reference_tcp, pick.get_center(),
                        to_numpy(entity.get_AABB()), accept_rotation))

    def preflight(plans):
        nonlocal targets
        pre, descent, grasp_link, pre_link = plans
        current = np.asarray(arm.get_arm_qpos(), dtype=float)
        if (not task._dump_plan_valid(arm, pre, current, pre_link.to_pose7())
                or not task._dump_plan_valid(arm, descent, pre.position[-1], grasp_link.to_pose7())):
            print('[dump-body] rejected acquisition preflight', flush=True)
            return False
        selected = select_body_transport(task, arm, descent.position[-1], target_options,
                                        screw_first=not is_jar, rotation_options=rotation_options(expected_tcp))
        if selected is None:
            return False
        targets = selected
        task._dump_selector_targets = {id(pre): pre_link.to_pose7(), id(descent): grasp_link.to_pose7()}
        return True

    result = None
    # Other approach sides are tried only when every candidate of the previous
    # side failed to plan; nothing has executed at that point.
    for turn in SIDE_TURNS:
        grasps, expected_tcp, target_options = plan_set(turn)
        targets = target_options[0]
        task._dump_selector_center = center_before.copy()
        task._dump_selector_targets = {}
        try:
            approach = (milk_approach_planning(task, arm, target_margin=.003 if is_jar else .008)
                        if task.config.get('robot_type') == 'franka' else nullcontext())
            with approach:
                result = task.select_and_execute_grasp(
                    object_name=pick.label, object_pose=object_pose, arm_tag=arm_tag,
                    model_id=model_id, object_scale=scale, candidate_grasps=grasps,
                    max_candidates=len(grasps), plan_attempts=3, screw_qpos_step=.03 if is_jar else .005,
                    post_plan_check=preflight,
                )
        finally:
            task._dump_selector_targets = {}
        if result is not None:
            break
    if result is None:
        return False
    task._dump_arrival(tcp_to_link_pose(expected_tcp, arm.tcp_offset).to_pose7(), arm_tag)
    if is_jar:
        # Resolve unloaded tracking error before closing, preserving the
        # compensated PD target through the gripper command.
        task._dump_arrival(tcp_to_link_pose(expected_tcp, arm.tcp_offset).to_pose7(),
                           arm_tag, center_grasp=True)
    task.set_gripper(float(pick.close_value), arm_tag)
    for _ in range(200):
        task.step_sim()
    measured = Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float))
    task._dump_hold_offset = np.asarray(pick.get_center()) - measured.p
    task._dump_hold_local = measured.to_matrix()[:3, :3].T @ task._dump_hold_offset
    for _ in range(task.MILK_SIDE_HOLD_STEPS):
        task.step_sim()
    for index, target in enumerate(targets):
        options = (rotation_options(Pose.from_pose7(np.asarray(arm.get_ee_pose(), dtype=float)))
                   if is_jar and index == len(targets)-1 else {})
        move_checked(task, target, arm_tag, high_rotation=index == len(targets)-1,
                     screw_first=not is_jar, **options)
        if index == 0 and float(np.asarray(pick.get_center())[2]-center_before[2]) < .03:
            task._reject_dump_motion(f'{pick.label} small lift did not retain the body')
    dz = float(np.asarray(pick.get_center())[2] - center_before[2])
    print(f'[dump-physical] lift {pick.label} object={pick.get_center()} dz={dz:.4f}', flush=True)
    if not np.isfinite(dz) or dz < .05:
        task._reject_dump_motion(f'{pick.label} body lift below 50mm')
    plans = plan_release(task, pick, place, arm_tag, targets[-1].q)
    if plans is None:
        task._reject_dump_motion(f'{pick.label} has no retained carry and controlled descent path')
    return finish_drop(task, pick, plans, arm_tag)
