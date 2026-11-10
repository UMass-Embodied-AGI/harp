"""Checked post-lift milk turn so the lower finger can support its weight."""
import itertools

import numpy as np
from transforms3d.quaternions import qinverse, qmult, quat2mat

from ...base_task_parts.grasp_select import mplib_world_link_pose
from ...grasp import tcp_to_link_pose
from ...utils import Pose, to_numpy
from .motion import valid_pose


def bounds_corners(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] < bounds[0]):
        raise ValueError('invalid payload or obstacle bounds')
    return np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))


def payload_obstacles(task, actor):
    """Conservative solid bounds with the existing family/avatar inflation."""
    obstacles = []
    margin = np.array([task._OBSTACLE_MARGIN_XY]*2+[task._OBSTACLE_MARGIN_Z])
    for basket in task.basket_poses:
        half = np.array([task._BASKET_HALF_X, task._BASKET_HALF_Y, task._BASKET_HALF_Z])+margin
        obstacles.append(np.array([basket.p-half, basket.p+half]))
    for other, _name, _mid, _basket in task._sort_targets:
        if other is not actor:
            bounds = np.asarray(to_numpy(other.entity.get_AABB()), dtype=float)
            obstacles.append(bounds+np.array([-margin, margin]))
    collider = getattr(task, 'avatar_collider', None)
    if collider is not None:
        for _name, start, end, radius in collider.current_capsules():
            points = np.array([start, end], dtype=float)
            half = 1.10*float(radius)
            obstacles.append(np.array([points.min(axis=0)-half, points.max(axis=0)+half]))
        for _name, center, axes, half in collider.current_boxes():
            half = np.abs(np.asarray(axes))@(np.asarray(half)+.005)
            obstacles.append(np.array([np.asarray(center)-half, np.asarray(center)+half]))
    neutral = getattr(task, '_neutral_object_obstacle_points', None)
    if neutral is not None:
        points = neutral()
        if points is not None and np.asarray(points).size:
            points = np.asarray(points, dtype=float)
            obstacles.append(np.array([points.min(axis=0), points.max(axis=0)]))
    for bounds in obstacles:
        bounds_corners(bounds)
    return obstacles


def payload_clear(local_corners, tcp, obstacles, table_z):
    world = np.asarray(local_corners)@quat2mat(tcp.q).T+tcp.p
    if not np.all(np.isfinite(world)) or world[:, 2].min() <= table_z:
        return False
    low, high = world.min(axis=0), world.max(axis=0)
    return not any(np.all(high >= bounds[0]) and np.all(low <= bounds[1]) for bounds in obstacles)


def orient_milk_for_support(task, pick, arm_tag):
    """Turn only after retained lift; reject without opening or reanchoring.

    The two quarter turns preserve the selected side-grasp approach axis
    and closure. All native path samples must clear with measured finger
    coordinates and the rigid conservative payload envelope before execution.
    Actual object/TCP angular retention is checked before continuing the carry.
    """
    arm = task.robot.get_arm(arm_tag)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    if not valid_pose(tcp) or getattr(task, '_categorize_hold_local', None) is None:
        task._reject_categorize_motion('milk support turn requires a retained grasp')
    if abs(quat2mat(tcp[3:])[2, 1]) >= .95:
        return tcp[3:].copy()
    actor = pick.actor
    corners = bounds_corners(to_numpy(actor.entity.get_AABB()))
    local = (corners-tcp[:3])@quat2mat(tcp[3:])
    obstacles = payload_obstacles(task, actor)
    relative_before = qmult(qinverse(tcp[3:]), actor.get_pose().q)
    actual_q = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float)
    indices = list(arm._finger_dof_indices)
    native_names = arm.planner._mplib.robot.get_user_joint_names()
    native_indices = [i for i, name in enumerate(native_names) if 'finger' in name]
    if len(indices) != 2 or len(native_indices) != 2:
        task._reject_categorize_motion('milk support turn requires two measured fingers')
    fingers = actual_q[indices]
    if not np.all(np.isfinite(fingers)):
        task._reject_categorize_motion('milk support turn has invalid finger positions')
    for sign in (-1., 1.):
        quaternion = qmult(tcp[3:], [np.sqrt(.5), 0., 0., sign*np.sqrt(.5)])
        if abs(quat2mat(quaternion)[2, 1]) < .95:
            continue
        target = tcp_to_link_pose(Pose(tcp[:3], quaternion), arm.tcp_offset).to_pose7()
        arm, current, target = task._categorize_inputs(target, arm_tag)
        for precision in (.005, .002, .001):
            plan = arm.planner.plan_screw_path(current.copy(), target, qpos_step=precision)
            if not task._categorize_plan_valid(arm, plan, current, target):
                continue
            safe = True
            for joints in plan.position:
                full = arm.planner._pad_qpos(joints.copy())
                full[native_indices] = fingers
                backend = arm.planner._mplib
                pose = mplib_world_link_pose(arm.planner, joints)
                if (pose is None or backend.check_for_self_collision(full)
                        or backend.check_for_env_collision(full)
                        or not payload_clear(local, pose*arm.tcp_offset, obstacles,
                                             task.TABLE_TOP_Z+task._OBSTACLE_MARGIN_Z)):
                    safe = False
                    break
            if not safe:
                continue
            task._execute_categorize_plan(plan, arm_tag)
            task._categorize_arrival(target, arm_tag)
            measured = np.asarray(arm.get_ee_pose(), dtype=float)
            relative_after = qmult(qinverse(measured[3:]), actor.get_pose().q)
            angle = 2*np.arccos(np.clip(abs(np.dot(relative_before, relative_after)), 0., 1.))
            if not np.isfinite(angle) or angle > .15:
                task._reject_categorize_motion(f'milk support angular slip={angle:.4f}rad')
            if abs(quat2mat(measured[3:])[2, 1]) < .95:
                task._reject_categorize_motion('milk support closing axis not vertical')
            print(f'[categorize] milk support turn sign={sign:+.0f}, angular_slip={angle:.4f}rad', flush=True)
            return measured[3:].copy()
    task._reject_categorize_motion('milk support turn has no checked native path')
