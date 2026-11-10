"""Preflight a lower basket release and empty retreat through real walls."""
import numpy as np
from transforms3d.quaternions import quat2mat

from ...base_task_parts.grasp_select import mplib_world_link_pose
from ...grasp import tcp_to_link_pose
from ...utils import Pose, to_numpy
from .milk_support import bounds_corners, payload_clear, payload_obstacles
from .motion import valid_pose


def uses_lower_release(task, label):
    return bool(getattr(task, '_CATEGORIZE_COLUMN_LAYOUT', False) and label in
                {'cube', '086_woodenblock', '073_rubikscube'})


def basket_solids(task, index):
    """Match the column basket's floor/four walls; retain existing padding."""
    basket = task.basket_poses[index]
    if not np.allclose(basket.q, [1., 0., 0., 0.], atol=1e-6):
        task._reject_categorize_motion('lower release needs axis-aligned column basket')
    hx, hy, hz = task._BASKET_HALF_X, task._BASKET_HALF_Y, task._BASKET_HALF_Z
    wall = .005
    primitives = [([0, 0, -hz+wall], [hx, hy, wall]),
                  ([-hx+wall, 0, 0], [wall, hy, hz]),
                  ([hx-wall, 0, 0], [wall, hy, hz]),
                  ([0, -hy+wall, 0], [hx, wall, hz]),
                  ([0, hy-wall, 0], [hx, wall, hz])]
    margin = np.array([task._OBSTACLE_MARGIN_XY]*2+[task._OBSTACLE_MARGIN_Z])
    bounds = []
    for delta, half in primitives:
        center = np.asarray(basket.p)+delta
        extent = np.asarray(half)+margin
        box = np.array([center-extent, center+extent])
        bounds_corners(box)
        bounds.append(box)
    return bounds


def install_solids(world, bounds):
    from mplib.pymp import Pose as MplibPose
    from mplib.pymp.collision_detection.fcl import Box, CollisionObject
    names = []
    try:
        for i, (low, high) in enumerate(bounds):
            name = f'categorize_lower_release_{i}'
            world.add_object(name, CollisionObject(Box(*(high-low)),
                MplibPose(p=(low+high)/2, q=np.array([1., 0., 0., 0.]))))
            names.append(name)
    except Exception:
        for name in names:
            world.remove_object(name)
        raise
    return names


def checked_plan(task, arm, current, target, local, obstacles):
    backend = arm.planner._mplib
    names = backend.robot.get_user_joint_names()
    fingers = [i for i, name in enumerate(names) if 'finger' in name]
    if len(fingers) != 2:
        task._reject_categorize_motion('lower release needs two native fingers')
    for precision in (.005, .004, .002, .001):
        plan = arm.planner.plan_screw_path(current.copy(), target, qpos_step=precision)
        if not task._categorize_plan_valid(arm, plan, current, target):
            continue
        safe = True
        for joints in plan.position:
            for width in np.linspace(0., .04, 11):
                full = arm.planner._pad_qpos(joints.copy())
                full[fingers] = width
                if backend.check_for_self_collision(full) or backend.check_for_env_collision(full):
                    safe = False
                    break
            if not safe:
                break
            if local is not None:
                pose = mplib_world_link_pose(arm.planner, joints)
                if pose is None or not payload_clear(local, pose*arm.tcp_offset, obstacles,
                                                      task.TABLE_TOP_Z+task._OBSTACLE_MARGIN_Z):
                    safe = False
                    break
        if safe:
            return plan
    return None


SHALLOW_RELEASE_FRACTIONS = (.66, .33)


def _shallow_release(task, arm, arm_tag, actor, label, current, tcp, destination, local,
                     obstacles, floor):
    high = tcp_to_link_pose(Pose.from_pose7(tcp), arm.tcp_offset).to_pose7()
    for fraction in SHALLOW_RELEASE_FRACTIONS:
        goal = tcp.copy()
        goal[2] = float(tcp[2] - fraction * (tcp[2] - destination[2]))
        target = tcp_to_link_pose(Pose.from_pose7(goal), arm.tcp_offset).to_pose7()
        descent = checked_plan(task, arm, current, target, local, obstacles)
        if descent is None:
            continue
        retreat = checked_plan(task, arm, descent.position[-1], high, None, obstacles)
        if retreat is None:
            continue
        task._execute_categorize_plan(descent, arm_tag)
        actual = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
        print(f'[categorize] lower release {label}: shallow ({fraction:.2f}) '
              f'payload_floor_gap={float(actual[0, 2] - floor):.4f}m', flush=True)
        task.open_gripper(arm_tag)
        for _ in range(250):
            task.step_sim()
        task._execute_categorize_plan(retreat, arm_tag)
        return True
    print(f'[categorize] lower release {label}: no checked descent; releasing from hover', flush=True)
    task.open_gripper(arm_tag)
    for _ in range(250):
        task.step_sim()
    return True


def lower_release(task, actor, label, arm_tag):
    """Keep the grasp until descent AND empty retreat have passed all checks."""
    if not uses_lower_release(task, label) or getattr(task, '_categorize_hold_local', None) is None:
        task._reject_categorize_motion('lower release requires retained column-basket payload')
    matches = [i for item, _name, _mid, i in task._sort_targets if item is actor]
    if len(matches) != 1:
        task._reject_categorize_motion('lower release cannot identify assigned basket')
    index = matches[0]
    arm = task.robot.get_arm(arm_tag)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    if not valid_pose(tcp):
        task._reject_categorize_motion('lower release invalid measured TCP')
    basket = task.basket_poses[index]
    if np.linalg.norm(tcp[:2]-basket.p[:2]) > .10:
        task._reject_categorize_motion('lower release must begin above assigned basket')
    aabb = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
    local = (bounds_corners(aabb)-tcp[:3])@quat2mat(tcp[3:])
    floor = float(basket.p[2]-task._BASKET_HALF_Z+.010)
    gap = .075 if label == 'cube' else .050
    destination = tcp.copy()
    destination[2] += floor+gap-aabb[0, 2]
    if not .005 < tcp[2]-destination[2] < .30:
        task._reject_categorize_motion('lower release descent outside measured envelope')
    solids = basket_solids(task, index)
    obstacles = payload_obstacles(task, actor)
    # The first entries correspond exactly to basket_poses, in order.
    obstacles = obstacles[:index]+obstacles[index+1:]+solids
    old_skip = getattr(task, '_categorize_skip_basket', None)
    world = arm.planner._mplib.planning_world
    names = install_solids(world, solids)
    task._categorize_skip_basket = index
    try:
        target = tcp_to_link_pose(Pose.from_pose7(destination), arm.tcp_offset).to_pose7()
        arm, current, target = task._categorize_inputs(target, arm_tag)
        descent = checked_plan(task, arm, current, target, local, obstacles)
        high = tcp_to_link_pose(Pose.from_pose7(tcp), arm.tcp_offset).to_pose7()
        split = None
        if descent is None:
            # The opt-in wood descent split runs only after all original
            # checked straight-path attempts have failed.
            from .split import descent_split, enabled
            if enabled(task):
                split = descent_split(task, arm, arm_tag, actor, current, target, local, obstacles, high)
            if split is None and not getattr(task, 'CATEGORIZE_SHALLOW_RELEASE', False):
                task._reject_categorize_motion('lower release has no checked descent')
            if split is None:
                # No checked descent to the low release height.
                # Release from part-way down, or from the hover height the
                # unchanged basket drop already uses, instead of failing.
                return _shallow_release(task, arm, arm_tag, actor, label, current, tcp,
                                        destination, local, obstacles, floor)
        if split is not None and split['plan'] is not None:
            descent, split = split['plan'], None
        if split is None:
            retreat = checked_plan(task, arm, descent.position[-1], high, None, obstacles)
            if retreat is None:
                task._reject_categorize_motion('lower release has no checked empty retreat')
            task._execute_categorize_plan(descent, arm_tag)
        else:
            from .split import execute_split
            retreat = split['retreat']

            def local_fn(at_tcp):
                bounds = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
                return (bounds_corners(bounds) - at_tcp[:3]) @ quat2mat(at_tcp[3:])

            def obstacles_fn():
                refreshed = payload_obstacles(task, actor)
                return refreshed[:index] + refreshed[index + 1:] + solids

            execute_split(task, arm_tag, split, actor, local_fn, obstacles_fn, 'descent')
        actual = np.asarray(to_numpy(actor.entity.get_AABB()), dtype=float)
        bounds_corners(actual)
        measured_gap = float(actual[0, 2]-floor)
        if abs(measured_gap-gap) > .015:
            task._reject_categorize_motion(f'lower release payload gap={measured_gap:.4f}m')
        print(f'[categorize] lower release {label}: payload_floor_gap={measured_gap:.4f}m', flush=True)
        task.open_gripper(arm_tag)
        for _ in range(250):
            task.step_sim()
        task._execute_categorize_plan(retreat, arm_tag)
        return True
    finally:
        for name in names:
            world.remove_object(name)
        task._categorize_skip_basket = old_skip
        task._categorize_refresh_obstacles(arm_tag)
