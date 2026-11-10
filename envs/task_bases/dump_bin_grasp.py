"""Dump-specific acquisition and controlled floor-bin release.

Grasp planning uses BaseTask's public selector. All other trajectories use the
family's checked MPLib adapters and physical PD execution.
"""
import numpy as np

from ..grasp import GraspPose, tcp_to_link_pose
from ..base_task_parts.grasp_select import mplib_world_link_pose
from ..utils import Pose, to_numpy
from .dump_bin_motion import DumpMotionRejected, pose_errors, plan_release_screw


# Objects that can still be rocking or rolling when acquisition starts: the jar
# rocks through upright, and at friction 1.0 an apple nudged by the human keeps
# rolling, so the plan would target a position the object has already left.
REST_BEFORE_ACQUISITION = ('031_jam-jar', '035_apple')


def wait_for_jar_rest(task, pick):
    previous = np.asarray(pick.get_center(), dtype=float)
    quiet = 0
    # A jar can still be rocking through upright after the first two seconds.
    # Allow passive damping for six seconds without relaxing the quiet gate.
    for window in range(30):
        for _ in range(100):
            task.step_sim()
        center = np.asarray(pick.get_center(), dtype=float)
        if not np.all(np.isfinite(center)):
            task._reject_dump_motion('nonfinite jar center while settling')
        shift = float(np.linalg.norm(center-previous))
        quiet = quiet+1 if shift <= .0005 else 0
        print(f'[dump-rest] window={window+1} shift={shift:.6f}m quiet={quiet}', flush=True)
        if quiet >= 2:
            return
        previous = center
    task._reject_dump_motion('jar did not settle before acquisition')


def select_topdown(task, pick, place, arm_tag):
    from .dump_bin import _read_scale
    actor = task._dump_active_actor
    arm = task.robot.get_arm(arm_tag)
    # Stationary pre-planning waits: a neutral task may guard them (idle guard).
    task._dump_idle_guard = True
    try:
        task.open_gripper(arm_tag)
        for _ in range(max(0, int(pick.preopen_steps))):
            task.step_sim()
        if pick.label in REST_BEFORE_ACQUISITION:
            wait_for_jar_rest(task, pick)
    finally:
        task._dump_idle_guard = False
    center = np.asarray(pick.get_center(), dtype=float)
    object_pose = actor.get_pose() if hasattr(actor, 'get_pose') else Pose(center)
    model_id = int(getattr(task, '_object_model_ids', {}).get(id(actor), 0))
    scale = _read_scale(pick.label, model_id) if pick.label != 'cube' else 1.
    grasp_pos = center.copy()
    grasp_pos[:2] += np.asarray(pick.approach_xy_offset)
    height_offset = pick.grasp_z_offset
    if pick.label == 'cube' and task.config.get('robot_type', 'franka') == 'franka':
        # Native and physical fingertips extend 8.85mm below the TCP. The
        # original low cube target grazes the tabletop collision envelope.
        height_offset += .010
    grasp_pos[2] = max(task.TABLE_TOP_Z + .010,
                       center[2] - .3 * pick.radius + height_offset)
    if pick.label == '031_jam-jar' and task.config.get('robot_type', 'franka') == 'franka':
        entity = getattr(actor, 'entity', actor)
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            task._reject_dump_motion('invalid jar bounds before acquisition')
        # Model0's upper 2–6mm form a ~64mm cylindrical lid band. The
        # wider cap command grips there with the body hanging below the pads.
        below_top = .005 if model_id == 0 else .027
        grasp_pos[2] = max(grasp_pos[2], float(bounds[1, 2]) - below_top)
    if (pick.label == '035_apple' and model_id == 1
            and task.config.get('robot_type', 'franka') == 'franka'):
        entity = getattr(actor, 'entity', actor)
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            task._reject_dump_motion('invalid apple bounds before acquisition')
        # Center the pads at the broadest body section. The lower band wedged
        # the fruit upward into the palm; the old upper band pressed it down.
        # The27mm band still slides down at yaw0;29mm wedged upward. Use the
        # intervening body band without increasing closure or force.
        grasp_pos[2] = max(grasp_pos[2], float(bounds[1, 2]) - .028)
    apple_body_grip = (pick.label == '035_apple' and model_id == 1
                        and task.config.get('robot_type', 'franka') == 'franka')
    # The diagonal apple contact line let the fruit roll sideways during lift.
    # Keep the measured body height and use the axial top-down direction.
    yaw = float(pick.get_grasp_yaw()) if pick.get_grasp_yaw is not None else 0.
    tcp = task._top_down_tcp(grasp_pos, yaw=yaw)
    pre_distance = (.10 if apple_body_grip or pick.label in ('031_jam-jar', '048_stapler')
                    else max(.15, float(place.transport_z) - grasp_pos[2]))
    candidates = []
    # A parallel-jaw top-down grasp turned a half turn about the vertical
    # closes on the same contact line. It is tried only after the nominal
    # yaw fails to plan, and it often avoids the wrist-limit screw failure
    # of the nominal descent.
    yaws = (yaw, yaw + np.pi)
    if pick.label == '031_jam-jar':
        # The long carry runs mostly along world X. Close across that axis so
        # lateral load is supported by the jaws instead of the narrow pad edge.
        yaws = (yaw + np.pi/2, yaw - np.pi/2)
    for index, candidate_yaw in enumerate(yaws):
        local = object_pose.inv() * task._top_down_tcp(grasp_pos, yaw=candidate_yaw)
        candidates.append(GraspPose(f'dump_catalog_topdown_{index}', local.p, local.q,
                                   pre_distance=pre_distance, source='manual', scale_frame='runtime'))
    task._refresh_dump_obstacles(arm_tag)

    def preflight(plans):
        pre, descent, grasp_link, pre_link = plans
        current = np.asarray(arm.get_arm_qpos(), dtype=float)
        if not task._dump_plan_valid(arm, pre, current, pre_link.to_pose7()):
            return False
        if not task._dump_plan_valid(arm, descent, pre.position[-1], grasp_link.to_pose7()):
            return False
        task._dump_selector_targets = {id(pre): pre_link.to_pose7(), id(descent): grasp_link.to_pose7()}
        return True

    task._dump_selector_center = center.copy()
    task._dump_selector_targets = {}
    # Labels that always offered both yaws keep one selector call. For the
    # rest the turned grasp is a separate, later call: the selector sorts by
    # an approach score that is ~0 for both yaws, so float noise could
    # otherwise put it first. Nothing has executed when a call returns None.
    groups = ([candidates] if pick.label in ('048_stapler', '073_rubikscube', 'cube', '031_jam-jar')
              else [[candidate] for candidate in candidates])
    result = None
    task._dump_allow_wrist_roll = True
    try:
        for group in groups:
            result = task.select_and_execute_grasp(
                object_name=pick.label, object_pose=object_pose, arm_tag=arm_tag,
                model_id=model_id, object_scale=scale, max_candidates=len(group),
                candidate_grasps=group, plan_attempts=3,
                post_plan_check=preflight, screw_qpos_step=.005,
            )
            if result is not None:
                break
    finally:
        task._dump_selector_targets = {}
        task._dump_allow_wrist_roll = False
    if result is None:
        return None
    tcp = result[2].to_world(object_pose, scale)
    task._dump_arrival(tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7(), arm_tag)
    if apple_body_grip:
        # A few millimeters of unloaded PD sag puts the contact line beside
        # the fruit's center. Correct only a small, already-validated arrival.
        task._dump_arrival(tcp_to_link_pose(tcp, arm.tcp_offset).to_pose7(), arm_tag,
                           center_grasp=True)
    return tcp


def release_centers(bin_center, half, object_half):
    """Near-center alternatives that keep the complete object inside the bin."""
    center, half, object_half = (np.asarray(v, dtype=float) for v in (bin_center, half, object_half))
    if (center.shape != (2,) or half.shape != (2,) or object_half.shape != (2,)
            or not np.all(np.isfinite(np.concatenate([center, half, object_half])))
            or np.any(half <= 0) or np.any(object_half <= 0)):
        raise ValueError('invalid dump release footprint')
    for dx, dy in ((0, 0), (0, -.06), (-.04, 0), (-.04, -.06),
                   (0, -.12), (-.08, 0), (-.08, -.06), (-.04, -.12), (-.08, -.12)):
        offset = np.array([dx, dy])
        if np.all(np.abs(offset) + object_half + .008 < half):
            yield center + offset


def plan_release(task, pick, place, arm_tag, tcp_quaternion=None):
    """Preflight both carry and descent before moving a held object laterally."""
    actor, arm = task._dump_active_actor, task.robot.get_arm(arm_tag)
    center = np.asarray(pick.get_center(), dtype=float)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    quaternion = task._top_down_tcp(np.zeros(3)).q if tcp_quaternion is None else np.asarray(tcp_quaternion)
    entity = getattr(actor, 'entity', actor)
    bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
    current = np.asarray(arm.get_arm_qpos(), dtype=float)
    return plan_release_from_state(task, place, arm_tag, current, tcp, center, bounds, quaternion,
                                   allow_fallback=True)


# Fallback carry: when no straight carry at the transport altitude
# clears the obstacle cloud (a human prop or arm on the line to the bin, or a
# grasp branch the straight screw cannot unwind), carry higher or along an
# MPLib RRT path that never dips below the carry altitude.
FALLBACK_CARRY_RAISES = (.10, .20)
FALLBACK_RRT_CANDIDATES = 3
FALLBACK_MAX_DIP = .03


def _tcp_path_min_z(arm, positions, stride=10):
    """Lowest TCP height along a joint path (MPLib FK)."""
    positions = np.asarray(positions, dtype=float)
    lowest = np.inf
    for index in list(range(0, len(positions), stride)) + [len(positions)-1]:
        link = mplib_world_link_pose(arm.planner, positions[index])
        if link is None:
            return -np.inf
        lowest = min(lowest, float((link*arm.tcp_offset).p[2]))
    return lowest


def _plan_fallback_carry(task, arm, current, tcp, release_xy, quaternion, place, target):
    """Carry legs ending above the release point, plus its descent, or None."""
    above = Pose(np.r_[release_xy, place.transport_z], quaternion)
    above_target = tcp_to_link_pose(above, arm.tcp_offset).to_pose7()
    for raise_by in FALLBACK_CARRY_RAISES:
        z = max(float(tcp[2]), float(place.transport_z)) + raise_by
        legs, q = [], current
        for leg_tcp in (Pose(np.r_[tcp[:2], z], quaternion), Pose(np.r_[release_xy, z], quaternion), above):
            leg_target = tcp_to_link_pose(leg_tcp, arm.tcp_offset).to_pose7()
            leg = plan_release_screw(arm, q, leg_target)
            if not task._dump_plan_valid(arm, leg, q, leg_target):
                break
            legs.append((leg, leg_target))
            q = leg.position[-1]
        else:
            descent = plan_release_screw(arm, q, target)
            if task._dump_plan_valid(arm, descent, q, target):
                print(f'[dump-physical] fallback high carry +{raise_by:.2f}m to {release_xy}', flush=True)
                return (*legs, (descent, target))
    floor = min(float(tcp[2]), float(place.transport_z)) - FALLBACK_MAX_DIP
    carry = arm.planner.plan_path(current.copy(), above_target, log=False)
    if (task._dump_plan_valid(arm, carry, current, above_target)
            and _tcp_path_min_z(arm, carry.position) >= floor):
        descent = plan_release_screw(arm, carry.position[-1], target)
        if task._dump_plan_valid(arm, descent, carry.position[-1], target):
            print(f'[dump-physical] fallback RRT carry rows={len(carry.position)} to {release_xy}', flush=True)
            return ((carry, above_target), (descent, target))
    return None


def plan_release_from_state(task, place, arm_tag, current, tcp, center, bounds, quaternion,
                            allow_fallback=False):
    """Query a continuation from measured or predicted state without moving it."""
    arm = task.robot.get_arm(arm_tag)
    current, tcp, center, bounds, quaternion = (
        np.asarray(value, dtype=float).copy() for value in (current, tcp, center, bounds, quaternion))
    if (current.shape != (7,) or tcp.shape != (7,) or center.shape != (3,)
            or bounds.shape != (2, 3) or quaternion.shape != (4,)
            or not all(np.all(np.isfinite(value)) for value in (current, tcp, center, bounds, quaternion))
            or np.any(bounds[1] <= bounds[0]) or not np.isclose(np.linalg.norm(quaternion), 1.)):
        task._reject_dump_motion('invalid held object bounds or release state')
    offset = center - tcp[:3]
    object_half = np.maximum(center[:2] - bounds[0, :2], bounds[1, :2] - center[:2])
    candidates = list(release_centers(task.big_bin_pose.p[:2], task._big_bin_xy_half, object_half))
    task._refresh_dump_obstacles(arm_tag)
    task._set_dump_release_clearance(arm_tag)
    try:
        attempts = 0
        reachable = []
        # The nominal low hover can put the forearm through the tabletop. Search
        # up to 20cm higher with that obstacle present; reach the selected target
        # before opening and require stable floor support after the bounded drop.
        for rise in (0., .05, .10, .15, .20):
            for object_xy in candidates:
                release = Pose(np.r_[object_xy - offset[:2], place.drop_from_z + rise], quaternion)
                target = tcp_to_link_pose(release, arm.tcp_offset).to_pose7()
                ik = arm.planner.solve_ik(current.copy(), target, num_waypoints=2, log=False)
                if ik is None or not ik.success or not np.all(np.isfinite(ik.position)):
                    continue
                actual = mplib_world_link_pose(arm.planner, ik.position[-1])
                if actual is None:
                    continue
                distance, angle = pose_errors(actual.to_pose7(), target)
                if distance > .01 or angle > .05:
                    continue
                attempts += 1
                reachable.append((object_xy - offset[:2], target))
                above = Pose(np.r_[object_xy - offset[:2], place.transport_z], quaternion)
                above_target = tcp_to_link_pose(above, arm.tcp_offset).to_pose7()
                carry = plan_release_screw(arm, current, above_target)
                if task._dump_plan_valid(arm, carry, current, above_target):
                    descent = plan_release_screw(arm, carry.position[-1], target)
                    if task._dump_plan_valid(arm, descent, carry.position[-1], target):
                        print(f'[dump-physical] planned release tcp={release.p} object_xy={object_xy}', flush=True)
                        return ((carry, above_target), (descent, target))
                if attempts >= 18:
                    break
            if attempts >= 18:
                break
        if allow_fallback:
            # Highest reachable releases first: the low hovers rarely clear the table.
            for release_xy, target in list(reversed(reachable))[:FALLBACK_RRT_CANDIDATES]:
                plans = _plan_fallback_carry(task, arm, current, tcp, release_xy, quaternion, place, target)
                if plans is not None:
                    return plans
        return None
    finally:
        task._clear_dump_release_clearance()


def pick_and_drop_topdown(task, pick, place, arm_tag):
    if (pick.label == '031_jam-jar'
            and int(getattr(task, '_object_model_ids', {}).get(id(task._dump_active_actor), 0)) == 0
            and task.config.get('robot_type', 'franka') == 'franka'):
        from .dump_bin_milk import pick_and_drop_side
        return pick_and_drop_side(task, pick, place, arm_tag)
    arm = task.robot.get_arm(arm_tag)
    center_before = np.asarray(pick.get_center(), dtype=float)
    tcp = select_topdown(task, pick, place, arm_tag)
    if tcp is None:
        return False
    relative_rotation = tcp.to_matrix()[:3, :3] @ task._R_DOWN.T
    yaw = float(np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0]))
    close = pick.close_value
    if close is None:
        close = max(.010, pick.radius-.005 if pick.radius < .030 else pick.radius*.7) / .04
    task.set_gripper(float(np.clip(close, 0., 1.)), arm_tag)
    for _ in range(200):
        task.step_sim()
    if pick.after_close is not None:
        pick.after_close()
    task._dump_hold_offset = np.asarray(pick.get_center(), dtype=float) - np.asarray(arm.get_ee_pose())[:3]
    for _ in range(40):
        task.step_sim()
    mid = tcp.p.copy()
    mid[2] = max(mid[2], center_before[2] + pick.radius)
    try:
        task._move_cartesian(tcp.p, mid, arm_tag, n_steps=30, sim_per_step=30, tcp_yaw=yaw)
    except DumpMotionRejected as error:
        # A per-waypoint IK branch flip on the tiny lift: finish the same
        # vertical lift with the checked straight move instead.
        if 'tiny-lift' not in str(error):
            raise
        print(f'[dump-physical] tiny lift rejected ({error}); checked straight lift instead', flush=True)
        task.plan_success = True
        task._move_screw(mid, arm_tag, tcp_yaw=yaw)
    task._move_screw(np.r_[tcp.p[:2], place.transport_z], arm_tag, tcp_yaw=yaw)
    lifted = np.asarray(pick.get_center(), dtype=float)
    dz = float(lifted[2] - center_before[2])
    print(f'[dump-physical] lift object={lifted} dz={dz:.4f}', flush=True)
    if not np.isfinite(dz) or dz < max(.03, float(pick.min_lift_delta or .03)):
        return False
    plans = plan_release(task, pick, place, arm_tag, tcp.q)
    if plans is None:
        task._reject_dump_motion('no retained carry and controlled descent path')
    return finish_drop(task, pick, plans, arm_tag)


def finish_drop(task, pick, plans, arm_tag):
    """Execute a checked transfer and verify that an opened payload settles."""
    arm = task.robot.get_arm(arm_tag)
    task._dump_descent_slip_noted = False
    phases = ['carry'] * (len(plans) - 1) + ['descent']
    for (result, target), phase in zip(plans, phases):
        task._dump_release_descent = phase == 'descent'
        try:
            task.execute_plan(result, arm_tag)
            task._reseed_cached_arm_target(arm)
            task._dump_arrival(target, arm_tag)
            for _ in range(100 if phase == 'carry' else 50):
                task.step_sim()
            task._dump_check_retention(arm_tag)
        finally:
            task._dump_release_descent = False
        print(f'[dump-physical] {phase} object={pick.get_center()} tcp={arm.get_ee_pose()[:3]}', flush=True)
    entity = getattr(task._dump_active_actor, 'entity', task._dump_active_actor)
    bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
    bin_xy = np.asarray(task.big_bin_pose.p[:2])
    half = np.asarray(task._big_bin_xy_half)
    if getattr(task, '_dump_descent_slip_noted', False) and task._dump_center_over_bin(pick.get_center()):
        # A payload that slid during the descent may already be falling or
        # tilted against a wall; release it and let the support check decide.
        pass
    elif (not np.all(np.isfinite(bounds)) or np.any(bounds[0, :2] <= bin_xy-half+.002)
            or np.any(bounds[1, :2] >= bin_xy+half-.002)):
        task._reject_dump_motion('held footprint outside bin before release')
    return release_and_settle(task, pick, arm_tag)


def release_and_settle(task, pick, arm_tag):
    """Open the jaws and require the payload to come to rest on the bin floor."""
    entity = getattr(task._dump_active_actor, 'entity', task._dump_active_actor)
    bin_xy = np.asarray(task.big_bin_pose.p[:2])
    half = np.asarray(task._big_bin_xy_half)
    task._dump_hold_offset = None
    task._dump_hold_local = None
    task._dump_releasing = True
    task._dump_idle_guard = True
    try:
        task.open_gripper(arm_tag)
        for _ in range(200):
            task.step_sim()
    finally:
        task._dump_releasing = False
    earlier = np.asarray(pick.get_center(), dtype=float)
    # Rocking can return to the same center at successive window endpoints.
    # Require sustained support and bound travel within each observation too.
    quiet_windows = 0
    for settle_window in range(20):
        traveled = 0.
        previous = earlier.copy()
        for _ in range(100):
            task.step_sim()
            center = np.asarray(pick.get_center(), dtype=float)
            traveled += float(np.linalg.norm(center-previous))
            previous = center.copy()
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        stable = (np.all(np.isfinite(center)) and np.linalg.norm(center-earlier) <= .01
                  and np.isfinite(traveled) and traveled <= .001)
        supported = (np.all(np.isfinite(bounds)) and -.01 <= bounds[0, 2] <= task.BIG_BIN_FLOOR_THICKNESS+.03
                     and np.all(center[:2] > bin_xy-half) and np.all(center[:2] < bin_xy+half))
        quiet_windows = quiet_windows+1 if stable and supported else 0
        if quiet_windows >= 3:
            break
        earlier = center.copy()
    task._dump_idle_guard = False
    complete = quiet_windows >= 3
    print(f'[dump-physical] released object={center} stable={complete} floor_support={supported} '
          f'quiet_windows={quiet_windows} window_travel={traveled:.6f}m', flush=True)
    return bool(complete)
