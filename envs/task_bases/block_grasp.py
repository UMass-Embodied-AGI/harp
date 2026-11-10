"""Plan block-ranking approaches completely before moving near a cube."""

from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..planning.base import PlanResult
from ..utils import Pose, to_numpy


@dataclass
class BlockGraspPlan:
    yaw: float
    legs: list
    closed: bool = False


_BACKWARD_IK_DRAWS = 20
_BACKWARD_SOLUTIONS_PER_YAW = 8
_GRASP_NEIGHBOR_CLEARANCE_M = 0.01


@contextmanager
def block_neighbor_obstacles(task, target_key, arm_tag):
    """Own only this pickup's named neighbor objects in the planning world."""
    if getattr(task, "_block_neighbor_scene", None) is not None:
        raise RuntimeError("nested block neighbor planning scene")
    world = task.robot.get_arm(arm_tag).planner._mplib.planning_world
    state = {"target": target_key, "arm_tag": arm_tag, "world": world, "names": []}
    task._block_neighbor_scene = state
    try:
        yield
    finally:
        try:
            for name in state["names"]:
                world.remove_object(name)
        finally:
            task._block_neighbor_scene = None


def _refresh_block_neighbors(task, arm_tag):
    """Refresh exact oriented cube geometry without replacing human clouds."""
    state = getattr(task, "_block_neighbor_scene", None)
    if state is None:
        return
    if state["arm_tag"] != arm_tag:
        raise RuntimeError("block neighbor scene belongs to a different arm")
    from mplib.pymp import Pose as MplibPose
    from mplib.pymp.collision_detection.fcl import Box, CollisionObject

    objects = []
    for key, cube in task.blocks.items():
        if key == state["target"]:
            continue
        center = np.asarray(to_numpy(cube.get_pos()), dtype=float).ravel()
        quat = np.asarray(to_numpy(cube.get_quat()), dtype=float).ravel()
        half = float(task.block_half_size(key))
        if (center.shape != (3,) or quat.shape != (4,)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(quat))
                or not np.isclose(np.linalg.norm(quat), 1., atol=1e-4)
                or not np.isfinite(half) or half <= 0):
            raise RuntimeError("invalid block neighbor geometry")
        name = f"block_ranking_neighbor_{key}"
        objects.append((name, CollisionObject(Box(2 * half, 2 * half, 2 * half),
                                             MplibPose(p=center, q=quat))))
    world = state["world"]
    for name in state["names"]:
        world.remove_object(name)
    state["names"].clear()
    for name, obj in objects:
        world.add_object(name, obj)
        state["names"].append(name)


def _reverse_block_plan(plan):
    """Reverse both time order and velocity sign without aliasing cached data."""
    return PlanResult(True, plan.position[::-1].copy(), -plan.velocity[::-1].copy())


def _backward_path_valid(arm, plan, start_qpos, target, vertical_tcp=None):
    """Validate a candidate entirely in MPLib; never probe simulator poses.

    ``vertical_tcp`` bounds a low straight segment by its two TCP targets.
    The 1 cm / 0.05 rad gates and 1 mrad joint-limit tolerance are the existing
    block/MPLib gates, not relaxed recovery tolerances.
    """
    if plan is None or not plan.success:
        return False
    position = np.asarray(plan.position, dtype=float)
    velocity = np.asarray(plan.velocity, dtype=float)
    start_qpos = np.asarray(start_qpos, dtype=float)
    if (position.ndim != 2 or len(position) == 0
            or position.shape[1:] != start_qpos.shape
            or velocity.shape != position.shape
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(start_qpos))
            or np.linalg.norm(position[0] - start_qpos) > 1e-6):
        return False
    planner = arm.planner
    if not endpoint_matches(planner, plan, target):
        return False
    backend = planner._mplib
    limits = np.asarray(backend.joint_limits, dtype=float)
    if (limits.ndim != 2 or limits.shape[1] != 2
            or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] > limits[:, 1])):
        return False
    for qpos in position:
        full = np.asarray(planner._pad_qpos(qpos.copy()), dtype=float)
        if (full.shape != (len(limits),) or not np.all(np.isfinite(full))
                or np.any(full < limits[:, 0] - .001)
                or np.any(full > limits[:, 1] + .001)):
            return False
        backend.planning_world.set_qpos_all(full[backend.move_group_joint_indices])
        if backend.planning_world.is_state_colliding():
            return False
        if vertical_tcp is not None:
            backend.pinocchio_model.compute_forward_kinematics(full)
            link = backend.robot.get_base_pose() * backend.pinocchio_model.get_link_pose(
                backend.link_name_2_idx[backend.move_group],
            )
            tcp = Pose(link.p, link.q) * arm.tcp_offset
            low, high = vertical_tcp
            distance = float(np.linalg.norm(tcp.p[:2] - low.p[:2]))
            angle = float(2 * np.arccos(np.clip(abs(np.dot(tcp.q, low.q)), 0., 1.)))
            if (not np.all(np.isfinite(tcp.to_pose7()))
                    or not np.isfinite(distance) or not np.isfinite(angle)
                    or distance > .01 or angle > .05
                    or tcp.p[2] < min(low.p[2], high.p[2]) - .01
                    or tcp.p[2] > max(low.p[2], high.p[2]) + .01):
                return False
    return True


def _plan_block_grasp_backward(task, current, points, arm_tag):
    """Bounded low-grasp IK → straight lift → exact high-joint RRT preflight.

    The caller retains the original avatar/prop cloud. Nothing advances the
    simulator while these reverse candidates are planned or checked. Missing
    backend capabilities or malformed plans fail closed, with no fallback IK.
    """
    try:
        arm = task.robot.get_arm(arm_tag)
        planner = arm.planner
        backend = planner._mplib
        current = np.asarray(current, dtype=float)
        limits = np.asarray(backend.joint_limits, dtype=float)
        if (current.ndim != 1 or not len(current) or not np.all(np.isfinite(current))
                or limits.ndim != 2 or limits.shape[1] != 2
                or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] > limits[:, 1])):
            return None
        for yaw in (0., np.pi / 2, -np.pi / 2, np.pi):
            tcp = [TopDownPickPlaceMixin._top_down_tcp(task, point, yaw=yaw) for point in points]
            targets = [tcp_to_link_pose(pose, arm.tcp_offset).to_pose7() for pose in tcp]
            if not np.all(np.isfinite(targets)):
                return None
            goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(targets[2])
            status, solutions = backend.IK(
                goal, planner._pad_qpos(current.copy()), n_init_qpos=_BACKWARD_IK_DRAWS,
                threshold=.001, return_closest=False, verbose=False,
            )
            if status != "Success" or solutions is None:
                continue
            valid_solutions = []
            for solution in solutions:
                qpos = np.asarray(solution, dtype=float)
                if (qpos.shape == (len(limits),) and np.all(np.isfinite(qpos))
                        and np.all(qpos >= limits[:, 0] - .001)
                        and np.all(qpos <= limits[:, 1] + .001)):
                    valid_solutions.append(qpos[:len(current)].copy())
            valid_solutions.sort(key=lambda q: np.linalg.norm(q - current))
            for candidate_index, grasp_qpos in enumerate(valid_solutions[:_BACKWARD_SOLUTIONS_PER_YAW]):
                seed = PlanResult(True, grasp_qpos[None, :])
                if not _backward_path_valid(arm, seed, grasp_qpos, targets[2], (tcp[2], tcp[2])):
                    continue
                qpos = grasp_qpos
                upward = []
                for goal_index, previous_index in ((1, 2), (0, 1)):
                    plan = planner.plan_screw_path(qpos, targets[goal_index], log=False)
                    if not _backward_path_valid(
                            arm, plan, qpos, targets[goal_index], (tcp[previous_index], tcp[goal_index])):
                        break
                    upward.append(plan)
                    qpos = plan.position[-1]
                if len(upward) != 2:
                    continue
                transit = backend.plan_qpos(
                    [planner._pad_qpos(qpos.copy())], planner._pad_qpos(current.copy()),
                    time_step=1 / 250, planning_time=5, rrt_range=.1,
                    fix_joint_limits=False, verbose=False,
                )
                if transit is None or transit.get("status") != "Success":
                    continue
                approach = PlanResult(True, transit.get("position"), transit.get("velocity"))
                if (not _backward_path_valid(arm, approach, current, targets[0])
                        or np.linalg.norm(approach.position[-1] - qpos) > 1e-6):
                    continue
                if not _neighbor_endpoint_clearance_ok(task, planner, seed):
                    continue
                # Exact joint continuity makes reversing the checked screw
                # paths safe to queue; never solve another above-pose IK.
                legs = [(targets[0], approach),
                        (targets[1], _reverse_block_plan(upward[1])),
                        (targets[2], _reverse_block_plan(upward[0]))]
                print(f"[blocks-grasp] backward preflight accepted yaw={np.degrees(yaw):+.0f} "
                      f"candidate={candidate_index + 1}", flush=True)
                return BlockGraspPlan(float(yaw), legs)
    except Exception as exc:
        print(f"[blocks-grasp] backward preflight unavailable: {exc}", flush=True)
    return None


def refresh_block_obstacles(task, arm_tag):
    """Include current neighbors, human and neutral props in every preflight."""
    _refresh_block_neighbors(task, arm_tag)
    neutral_refresh = getattr(task, "_refresh_avatar_planner_obstacles", None)
    if neutral_refresh is not None:
        neutral_refresh(arm_tag)
        return
    collider = getattr(task, "avatar_collider", None)
    if collider is None:
        task.robot.get_arm(arm_tag).planner.update_obstacles(
            np.empty((0, 3)), resolution=0.015,
        )
        return
    points = sample_capsule_planner_points(
        collider.current_capsules(), inflate_factor=1.10,
    )
    parts = [points] if points is not None else []
    for _name, center, axes, half in collider.current_boxes():
        coordinates = [
            np.linspace(-h - 0.005, h + 0.005, max(2, int(2 * h / 0.015) + 2))
            for h in np.asarray(half, dtype=float)
        ]
        grid = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
        parts.append(grid.reshape(-1, 3) @ np.asarray(axes).T + center)
    task.robot.get_arm(arm_tag).planner.update_obstacles(
        np.concatenate(parts) if parts else np.empty((0, 3)), resolution=0.015,
    )


def endpoint_matches(planner, plan, target):
    """Check the planned endpoint in MPLib without touching simulator state."""
    if plan is None or not plan.success or not plan.position.size:
        return False
    backend = planner._mplib
    qpos = planner._pad_qpos(plan.position[-1])
    backend.pinocchio_model.compute_forward_kinematics(qpos)
    actual = backend.pinocchio_model.get_link_pose(
        backend.link_name_2_idx[backend.move_group],
    )
    goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(target)
    distance = float(np.linalg.norm(np.asarray(actual.p) - np.asarray(goal.p)))
    angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, goal.q)), 0.0, 1.0))
    valid = bool(distance <= 0.01 and angle <= 0.05)
    if not valid:
        print(f"[blocks-grasp] reject FK position_error={distance:.4f} "
              f"orientation_error={angle:.4f}", flush=True)
    return valid


def _motion_plan_valid(planner, plan, target, n_arm):
    """Validate new block motion candidates before FK or physical execution."""
    if plan is None or not plan.success:
        return False
    position = np.asarray(plan.position, dtype=float)
    velocity = np.asarray(plan.velocity, dtype=float)
    return bool(
        position.ndim == 2 and position.shape[1] == n_arm and len(position)
        and velocity.shape == position.shape
        and np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))
        and endpoint_matches(planner, plan, target)
    )


def _motion_inputs(arm, target, *, cached_seed=False):
    target = np.asarray(target, dtype=float)
    current = np.asarray(arm.get_arm_qpos(), dtype=float)
    if (target.shape != (7,) or not np.all(np.isfinite(target))
            or not np.isclose(np.linalg.norm(target[3:]), 1., atol=1e-4)
            or current.shape != (arm.n_arm,) or not np.all(np.isfinite(current))):
        raise ValueError("nonfinite or malformed block motion target/current joints")
    seed = getattr(arm, "_cached_target", None) if cached_seed else None
    seed = current if seed is None else np.asarray(seed, dtype=float)
    if (seed.ndim != 1 or len(seed) < arm.n_arm or not np.all(np.isfinite(seed))):
        raise ValueError("nonfinite or malformed block motion seed")
    return target.copy(), current.copy(), seed[:arm.n_arm].copy()


def solve_block_lift_ik(task, tcp_pose, arm_tag):
    """Preserve the tiny lift's PD/timing loop, using checked MPLib IK only.

    Raise on rejection: the inherited Cartesian loop silently skips ``None``
    waypoints, which would continue a failed lift. Task recovery owns opening.
    """
    try:
        arm = task.robot.get_arm(arm_tag)
        target, current, seed = _motion_inputs(
            arm, tcp_to_link_pose(tcp_pose, arm.tcp_offset).to_pose7(), cached_seed=True,
        )
        initial_seed = seed.copy()
        wait_enabled = bool(
            getattr(task, "BLOCK_CAN_LIFT_WAIT_RECOVERY", False)
            and getattr(task, "_neutral_avatar_started", False)
            and (getattr(task, "_neutral_avatar_job", None) or {}).get("kind")
            == "put_objects_in_bowl"
        )
        for attempt in range(9):
            refresh_block_obstacles(task, arm_tag)
            plan = arm.planner.solve_ik(seed.copy(), target, log=False)
            if not _motion_plan_valid(arm.planner, plan, target, arm.n_arm):
                raise ValueError("tiny-lift IK/FK rejected")
            goal = np.asarray(plan.position[-1], dtype=float)
            if (np.max(np.abs(goal - seed)) <= .30
                    and np.max(np.abs(goal - initial_seed)) <= .30):
                return goal.copy()
            if not wait_enabled or attempt == 8:
                raise ValueError("tiny-lift IK branch jump exceeds 0.30 rad")
            if getattr(task, "avatar_collided", False):
                raise ValueError("avatar collision before tiny-lift wait")
            # Hold through ordinary PD and simulation. This API preserves the
            # commanded fingers; the held block remains governed by physics.
            arm.set_arm_joints(current.copy())
            print(f"[blocks-grasp] tiny-lift wait {attempt + 1}/8: 30 steps",
                  flush=True)
            for _ in range(30):
                task.step_sim()
                if getattr(task, "avatar_collided", False):
                    raise ValueError("avatar collision during tiny-lift wait")
            target, current, seed = _motion_inputs(arm, target, cached_seed=True)
    except Exception as exc:
        task.plan_success = False
        raise RuntimeError(f"block MPLib tiny lift rejected: {exc}") from exc


def move_block_seeded(task, link_pose7, arm_tag):
    """Collision-aware MPLib IK → joint RRT, with one checked pose fallback.

    Only arm joints are planned. Existing execution/reseeding retains the
    commanded finger targets, and every route checks physical link arrival.
    """
    try:
        arm = task.robot.get_arm(arm_tag)
        planner = arm.planner
        target, current, _seed = _motion_inputs(arm, link_pose7)
        refresh_block_obstacles(task, arm_tag)
        plan = None
        solved = planner.solve_ik(current.copy(), target, log=False)
        if _motion_plan_valid(planner, solved, target, arm.n_arm):
            goal = np.asarray(solved.position[-1], dtype=float)
            if np.linalg.norm(goal - current) <= 3.5:
                try:
                    result = planner._mplib.plan_qpos(
                        goal_qposes=[planner._pad_qpos(goal.copy())],
                        current_qpos=planner._pad_qpos(current.copy()),
                        time_step=1 / 250,
                        planning_time=getattr(planner, "_timeout", None) or 10,
                        rrt_range=.3, verbose=False,
                    )
                    if result and result.get("status") == "Success":
                        plan = PlanResult(True, result.get("position"), result.get("velocity"))
                except Exception as exc:
                    print(f"[blocks-grasp] joint RRT unavailable: {exc}", flush=True)
        if not _motion_plan_valid(planner, plan, target, arm.n_arm):
            plan = planner.plan_path(current.copy(), target, log=False)
        if not _motion_plan_valid(planner, plan, target, arm.n_arm):
            raise ValueError("seeded motion has no FK-checked MPLib path")
        task.execute_plan(plan, arm_tag)
        task._reseed_cached_arm_target(arm)
        measured = np.asarray(to_numpy(arm.ee_link.get_pos()), dtype=float)
        if (measured.shape != (3,) or not np.all(np.isfinite(measured))
                or np.linalg.norm(measured - target[:3]) > .05):
            raise ValueError("seeded motion misses physical link arrival by >5 cm")
        return plan
    except Exception as exc:
        task.plan_success = False
        print(f"[blocks-grasp] seeded motion rejected: {exc}", flush=True)
        return None


# A neighbour already resting in its own slot may be passed closer.
# Brushing it by a few millimetres cannot take it out of its 5.5 cm slot
# tolerance, while 1 cm of room made a block returned next to the slot row
# (interrupt variants) unreachable.  Exact boxes; no collision is allowed.
_PLACED_NEIGHBOR_CLEARANCE_M = 0.002
_PLACED_NEIGHBOR_SLOT_MARGIN_M = 0.02


def _neighbor_is_placed(task, key):
    cube = task.blocks[key]
    slot = np.asarray(task.target_slots[key], dtype=float)[:2]
    pos = np.asarray(to_numpy(cube.get_pos()), dtype=float).ravel()[:3]
    tolerance = float(task.SUCCESS_DIST_XY) - _PLACED_NEIGHBOR_SLOT_MARGIN_M
    return bool(np.linalg.norm(pos[:2] - slot) <= tolerance)


def _placed_neighbor_nearby(task):
    state = getattr(task, "_block_neighbor_scene", None)
    if state is None:
        return False
    try:
        return any(_neighbor_is_placed(task, key) for key in task.blocks if key != state["target"])
    except Exception:
        return False


def _neighbor_clearance_limit(task, name):
    if not getattr(task, "_block_relaxed_neighbor_clearance", False):
        return _GRASP_NEIGHBOR_CLEARANCE_M
    key = name[len("block_ranking_neighbor_"):] if name.startswith("block_ranking_neighbor_") else None
    try:
        cube = task.blocks[key]
        slot = np.asarray(task.target_slots[key], dtype=float)[:2]
        pos = np.asarray(to_numpy(cube.get_pos()), dtype=float).ravel()[:3]
        tolerance = float(task.SUCCESS_DIST_XY) - _PLACED_NEIGHBOR_SLOT_MARGIN_M
        if np.linalg.norm(pos[:2] - slot) <= tolerance:
            return _PLACED_NEIGHBOR_CLEARANCE_M
    except Exception:
        pass
    return _GRASP_NEIGHBOR_CLEARANCE_M


def _neighbor_endpoint_clearance_ok(task, planner, plan):
    """Reserve tracking room at the grasp without expanding transit obstacles."""
    state = getattr(task, "_block_neighbor_scene", None)
    if state is None:
        return True
    if not state["names"]:
        return False
    robot = None
    saved = None
    try:
        backend = planner._mplib
        robot = backend.robot
        saved = np.asarray(robot.get_qpos()).copy()
        closed = np.asarray(planner._pad_qpos(plan.position[-1]), dtype=float).copy()
        # The existing planner pads the two fingers closed, but the queued
        # acquisition executes with fully open jaws. Check both envelopes.
        expected = [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint1", "panda_finger_joint2"]
        if list(robot.get_user_joint_names()) != expected or closed.shape != (9,):
            raise ValueError("unsupported block gripper joint mapping")
        arm = task.robot.get_arm(state["arm_tag"])
        aperture = float(arm.gripper_open)
        limits = np.asarray(backend.joint_limits, dtype=float)[-2:]
        if (not np.isfinite(aperture) or aperture <= 0.
                or np.any(aperture < limits[:, 0]) or np.any(aperture > limits[:, 1])):
            raise ValueError("invalid block gripper open aperture")
        opened = closed.copy()
        opened[-2:] = aperture
        model = robot.get_fcl_model()
        world = state["world"]
        acm = world.get_allowed_collision_matrix()
        for label, qpos in (("closed", closed), ("open", opened)):
            robot.set_qpos(qpos, True)
            for name in state["names"]:
                neighbor = world.get_object(name)
                if neighbor is None:
                    return False
                distance = float(model.distance_with(neighbor, acm=acm).min_distance)
                limit = _neighbor_clearance_limit(task, name)
                if not np.isfinite(distance) or distance < limit:
                    print(f"[blocks-grasp] reject neighbor clearance {name} "
                          f"fingers={label} distance={distance:.6f}m", flush=True)
                    return False
        return True
    except Exception as exc:
        print(f"[blocks-grasp] neighbor clearance unavailable: {exc}", flush=True)
        return False
    finally:
        if saved is not None:
            robot.set_qpos(saved, True)


def backward_preflight_enabled(task):
    value = (getattr(task, "config", {}) or {}).get("block_grasp_backward_preflight", None)
    if value is None:
        value = getattr(task, "BLOCK_GRASP_BACKWARD_PREFLIGHT", False)
    return value is True


def plan_block_grasp(task, center, half, transport_z, arm_tag):
    """Try four cube-symmetric yaws, with two bounded branch draws each.

    All three legs must be feasible before any arm motion: RRT to above the
    cube, then two straight screw segments down to its grasp.  This prevents
    a late descent failure from substituting an RRT beside the cube.
    """
    arm = task.robot.get_arm(arm_tag)
    planner = arm.planner
    current = np.asarray(arm.get_arm_qpos(), dtype=float)
    center = np.asarray(center, dtype=float)
    above = center.copy()
    above[2] = float(transport_z)
    pre = center.copy()
    pre[2] += 0.15
    grasp = center.copy()
    grasp[2] = max(float(center[2] - 0.3 * half), task.TABLE_TOP_Z + 0.002)
    refresh_block_obstacles(task, arm_tag)

    def forward():
        for yaw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            targets = [
                tcp_to_link_pose(
                    TopDownPickPlaceMixin._top_down_tcp(task, point, yaw=yaw),
                    arm.tcp_offset,
                ).to_pose7()
                for point in (above, pre, grasp)
            ]
            for attempt in range(2):
                legs = []
                qpos = current
                for index, target in enumerate(targets):
                    plan_fn = planner.plan_path if index == 0 else planner.plan_screw_path
                    plan = plan_fn(qpos, target, log=False)
                    if not endpoint_matches(planner, plan, target):
                        print(
                            f"[blocks-grasp] preflight rejected yaw={np.degrees(yaw):+.0f} "
                            f"attempt={attempt + 1} leg={index + 1}", flush=True,
                        )
                        break
                    legs.append((np.asarray(target, dtype=float), plan))
                    qpos = plan.position[-1]
                if len(legs) == len(targets):
                    if not _neighbor_endpoint_clearance_ok(task, planner, legs[-1][1]):
                        continue
                    print(
                        f"[blocks-grasp] preflight accepted yaw={np.degrees(yaw):+.0f} "
                        f"attempt={attempt + 1}", flush=True,
                    )
                    return BlockGraspPlan(float(yaw), legs)
        return None

    found = forward()
    if found is not None:
        return found
    # Only after every forward branch failed: solve the grasp IK first, then
    # lift straight up and RRT to it (the forward RRT branch can
    # leave the elbow/wrist too close to a limit for the straight descent).
    # The config key overrides the per-class default.
    backward = backward_preflight_enabled(task)
    if backward:
        found = _plan_block_grasp_backward(task, current, (above, pre, grasp), arm_tag)
        if found is not None:
            return found
    # Last resort: pass a neighbour already resting in its slot at
    # 2 mm instead of 1 cm (only when nothing passed with the full margin,
    # since the tighter branch leaves no room for a pre-close correction).
    if backward and _placed_neighbor_nearby(task):
        task._block_relaxed_neighbor_clearance = True
        try:
            print("[blocks-grasp] retrying with placed-neighbour clearance 2 mm", flush=True)
            found = forward()
            if found is None:
                found = _plan_block_grasp_backward(task, current, (above, pre, grasp), arm_tag)
        finally:
            task._block_relaxed_neighbor_clearance = False
        if found is not None:
            return found
    print("[blocks-grasp] no complete approach/descent; arm stays clear", flush=True)
    return None


def execute_grasp_leg(task, grasp_plan, target, arm_tag):
    """Consume only the exact preflighted segment; never replan beside a cube."""
    expected, plan = grasp_plan.legs[0]
    target = np.asarray(target, dtype=float)
    if np.linalg.norm(target[:3] - expected[:3]) > 0.002:
        raise RuntimeError("block moved after preflight; refusing a lateral grasp replan")
    if abs(float(np.dot(target[3:7], expected[3:7]))) < np.cos(0.005):
        raise RuntimeError("block grasp orientation changed after preflight")
    grasp_plan.legs.pop(0)
    task.execute_plan(plan, arm_tag)
    return plan
