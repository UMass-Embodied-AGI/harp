"""Shorter checked carry for a retained apple; preserve physical hold gates."""
import numpy as np
from transforms3d.quaternions import qmult, quat2mat

from ...grasp import tcp_to_link_pose
from ...utils import Pose, to_numpy
from .milk_support import bounds_corners, payload_obstacles
from .motion import valid_pose
from .release import checked_plan


def uses_direct_carry(task, object_name, model_id, place):
    return bool(getattr(task, '_CATEGORIZE_COLUMN_LAYOUT', False)
                and object_name == '035_apple' and model_id == 1
                and place.drop_from_z is not None and place.transport_z is None)


def carry_and_release(task, pick, place, actor, arm_tag):
    """Preflight one diagonal from the retained small lift, then release.

    All native samples include the measured payload and full finger sweep.
    The existing per-step30mm hold anchor stays active until normal opening.
    """
    arm = task.robot.get_arm(arm_tag)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    if not valid_pose(tcp) or getattr(task, '_categorize_hold_local', None) is None:
        task._reject_categorize_motion('direct carry requires a retained grasp')
    bounds = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
    corners = bounds_corners(bounds)
    if bounds[0, 2] <= task.TABLE_TOP_Z + task._OBSTACLE_MARGIN_Z:
        task._reject_categorize_motion('direct carry payload has not cleared table')
    before = np.asarray(pick.get_center(), dtype=float)
    destination = np.r_[np.asarray(place.pos)[:2], float(place.drop_from_z), tcp[3:]]
    if not valid_pose(destination) or not .04 < destination[2]-tcp[2] < .40:
        task._reject_categorize_motion('direct carry destination outside lifted envelope')
    local = (corners-tcp[:3])@quat2mat(tcp[3:])
    target = tcp_to_link_pose(Pose.from_pose7(destination), arm.tcp_offset).to_pose7()
    arm, current, target = task._categorize_inputs(target, arm_tag)
    plan = checked_plan(task, arm, current, target, local, payload_obstacles(task, actor))
    if plan is None:
        # Fallback: rise straight up to the release height, then one
        # level leg to the release pose.  Both legs are preflighted with the
        # same payload/finger/obstacle checks before anything moves.
        corner = tcp_to_link_pose(Pose(np.r_[tcp[:2], destination[2]], tcp[3:]),
                                  arm.tcp_offset).to_pose7()
        rise = checked_plan(task, arm, current, corner, local, payload_obstacles(task, actor))
        level = (None if rise is None else checked_plan(
            task, arm, rise.position[-1], target, local, payload_obstacles(task, actor)))
        if rise is None or level is None:
            # The wrist roll (joint 7) often runs into its limit on
            # this carry.  The apple is round, so turn the hand a quarter turn
            # about the vertical along the same checked straight carry.
            turned = _turned_carry(task, arm, current, destination, local, actor)
            if turned is None:
                task._reject_categorize_motion('direct carry has no checked native path')
            plan, target = turned
        else:
            print('[categorize] direct carry via rise + level leg', flush=True)
            task._execute_categorize_plan(rise, arm_tag)
            task._categorize_arrival(corner, arm_tag)
            tcp_now = np.asarray(arm.get_ee_pose(), dtype=float)
            if not valid_pose(tcp_now):
                task._reject_categorize_motion('direct carry invalid measured TCP after rise')
            live = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
            local = (bounds_corners(live) - tcp_now[:3]) @ quat2mat(tcp_now[3:])
            arm, current, target = task._categorize_inputs(target, arm_tag)
            plan = checked_plan(task, arm, current, target, local, payload_obstacles(task, actor))
            if plan is None:
                task._reject_categorize_motion('direct carry level leg lost its checked path after rise')
    task._execute_categorize_plan(plan, arm_tag)
    task._categorize_arrival(target, arm_tag)
    if float(np.asarray(pick.get_center())[2]-before[2]) < max(.03, .5*float(pick.radius)):
        task._reject_categorize_motion('direct carry did not retain vertical lift')
    for _ in range(100):
        task.step_sim()
    print('[categorize] direct apple carry completed with retained lift', flush=True)
    task.open_gripper(arm_tag)
    for _ in range(250):
        task.step_sim()
    return True


CARRY_TURN_ALTERNATIVES = (np.pi / 2, -np.pi / 2)


def _turned_carry(task, arm, current, destination, local, actor):
    """Same straight checked carry, ending with the hand turned about z."""
    for turn in CARRY_TURN_ALTERNATIVES:
        c, s = np.cos(turn / 2), np.sin(turn / 2)
        quat = qmult(np.array([c, 0., 0., s]), np.asarray(destination[3:], dtype=float))
        turned = np.r_[destination[:3], quat / np.linalg.norm(quat)]
        if not valid_pose(turned):
            continue
        target = tcp_to_link_pose(Pose.from_pose7(turned), arm.tcp_offset).to_pose7()
        plan = checked_plan(task, arm, current, target, local, payload_obstacles(task, actor))
        if plan is not None:
            print(f'[categorize] direct carry with hand turned {np.degrees(turn):+.0f} deg', flush=True)
            return plan, target
    return None


def uses_milk_corner(task, object_name, model_id, place):
    """Column-layout milk carton model 0 with a hover release only."""
    return bool(getattr(task, '_CATEGORIZE_COLUMN_LAYOUT', False)
                and object_name == '038_milk-box' and model_id == 0
                and getattr(place, 'drop_from_z', None) is not None)


def milk_corner_geometry(tcp, aabb, target_xy, safe_z, basket_bounds, clearance=.010):
    """Corner height so a straight diagonal keeps the payload above the rim.

    Returns ``(fraction, required_tcp_z, corner_z)`` or ``None`` when the
    translated payload never enters the padded basket footprint, enters it at
    the start, or the corner cannot sit between the current TCP and ``safe_z``.
    ``fraction`` is the share of the XY translation at which the payload's
    footprint first overlaps the padded basket; ``required_tcp_z`` keeps the
    measured payload bottom ``clearance`` above the padded basket top there.
    """
    tcp, aabb = np.asarray(tcp, dtype=float), np.asarray(aabb, dtype=float)
    low, high = np.asarray(basket_bounds, dtype=float)
    delta = np.asarray(target_xy, dtype=float) - tcp[:2]
    entry, exit_ = -np.inf, np.inf
    for axis in range(2):
        gap_low, gap_high = low[axis] - aabb[1, axis], high[axis] - aabb[0, axis]
        if abs(delta[axis]) < 1e-9:
            if gap_low > 0 or gap_high < 0:
                return None
            continue
        first, second = gap_low / delta[axis], gap_high / delta[axis]
        entry, exit_ = max(entry, min(first, second)), min(exit_, max(first, second))
    if not np.isfinite(entry) or exit_ <= entry or not 0 < entry < 1:
        return None
    required = float(high[2] - (aabb[0, 2] - tcp[2]) + clearance)
    corner = max(float(tcp[2]), (required - entry * float(safe_z)) / (1 - entry))
    if not np.isfinite(corner) or not tcp[2] <= corner < safe_z:
        return None
    return float(entry), required, float(corner)


def milk_corner_carry(task, pick, place, actor, arm_tag, safe_z):
    """Rise to a measured corner, then one checked diagonal to the release pose.

    Called after the retained small lift. Every native sample of both legs is
    checked with the measured payload, all finger widths and the current
    obstacle world (baskets, other objects, avatar, neutral clutter) before
    anything moves; a preflight failure returns ``False`` so the caller keeps
    its unchanged route. Once the first leg has executed, a lost second-leg
    path or an unretained lift rejects without opening. The per-step hold
    guard and the existing release height are unchanged.
    """
    arm = task.robot.get_arm(arm_tag)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    if not valid_pose(tcp) or getattr(task, '_categorize_hold_local', None) is None:
        print('[categorize] milk corner skipped: no retained grasp', flush=True)
        return False
    matches = [i for item, _name, _mid, i in task._sort_targets if item is actor]
    if len(matches) != 1:
        print('[categorize] milk corner skipped: cannot identify assigned basket', flush=True)
        return False
    bounds = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
    corners = bounds_corners(bounds)
    if bounds[0, 2] <= task.TABLE_TOP_Z + task._OBSTACLE_MARGIN_Z:
        print('[categorize] milk corner skipped: payload has not cleared table', flush=True)
        return False
    obstacles = payload_obstacles(task, actor)
    before = np.asarray(pick.get_center(), dtype=float)
    target_xy = np.asarray(place.pos, dtype=float)[:2] - (before - tcp[:3])[:2]
    geometry = milk_corner_geometry(tcp, bounds, target_xy, float(safe_z), obstacles[matches[0]])
    if geometry is None:
        print('[categorize] milk corner skipped: no corner geometry', flush=True)
        return False
    fraction, required, corner_z = geometry
    local = (corners - tcp[:3]) @ quat2mat(tcp[3:])
    final = tcp_to_link_pose(Pose(np.r_[target_xy, float(safe_z)], tcp[3:]), arm.tcp_offset).to_pose7()
    if not valid_pose(final) or not .005 < safe_z - tcp[2] < .40:
        print('[categorize] milk corner skipped: destination outside lifted envelope', flush=True)
        return False
    arm, current, final = task._categorize_inputs(final, arm_tag)
    first, corner_target = None, None
    if corner_z - tcp[2] >= .005:
        corner_target = tcp_to_link_pose(Pose(np.r_[tcp[:2], corner_z], tcp[3:]), arm.tcp_offset).to_pose7()
        first = checked_plan(task, arm, current, corner_target, local, obstacles)
        if first is None:
            print('[categorize] milk corner skipped: no checked rise', flush=True)
            return False
        start = first.position[-1]
    else:
        start = current
    second = checked_plan(task, arm, start, final, local, obstacles)
    if second is None:
        print('[categorize] milk corner skipped: no checked diagonal', flush=True)
        return False
    print(f'[categorize] milk corner accepted: fraction={fraction:.4f} required_tcp_z={required:.4f} '
          f'corner_z={corner_z:.4f} rise_samples={0 if first is None else len(first.position)} '
          f'diagonal_samples={len(second.position)}', flush=True)
    if first is not None:
        task._execute_categorize_plan(first, arm_tag)
        task._categorize_arrival(corner_target, arm_tag)
        tcp = np.asarray(arm.get_ee_pose(), dtype=float)
        if not valid_pose(tcp):
            task._reject_categorize_motion('milk corner invalid measured TCP after rise')
        live = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
        local = (bounds_corners(live) - tcp[:3]) @ quat2mat(tcp[3:])
        arm, current, final = task._categorize_inputs(final, arm_tag)
        second = checked_plan(task, arm, current, final, local, payload_obstacles(task, actor))
        if second is None:
            task._reject_categorize_motion('milk corner diagonal lost its checked path after rise')
    task._execute_categorize_plan(second, arm_tag)
    task._categorize_arrival(final, arm_tag)
    if float(np.asarray(pick.get_center())[2] - before[2]) < .04:
        task._reject_categorize_motion('milk corner carry did not retain vertical lift')
    print('[categorize] milk corner carry completed with retained lift', flush=True)
    return True
