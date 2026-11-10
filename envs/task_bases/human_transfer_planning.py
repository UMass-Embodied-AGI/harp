"""Collision-aware MPLib motions shared by the two human-transfer experts."""

import numpy as np

from ..avatar.collision_checker import sample_capsule_planner_points
from ..planning.base import PlanResult
from ..utils import to_numpy


def refresh_avatar_obstacles(task, arm_tag):
    """Match the existing contact contract, including oriented palm boxes."""
    collider = getattr(task, "avatar_collider", None)
    if collider is None:
        task.robot.get_arm(arm_tag).planner.update_obstacles(
            np.empty((0, 3)), resolution=0.015,
        )
        return
    ignored = set(task.config.get("collision_ignore_capsules", ()))
    prefixes = tuple(task.config.get("collision_ignore_capsule_prefixes", ()))

    def included(name):
        return name not in ignored and not name.startswith(prefixes)

    capsules = [c for c in collider.current_capsules() if included(c[0])]
    points = sample_capsule_planner_points(capsules, inflate_factor=1.10)
    parts = [points] if points is not None else []
    for name, center, axes, half in getattr(collider, "current_boxes", lambda: [])():
        if not included(name):
            continue
        coordinates = [np.linspace(-h - 0.005, h + 0.005,
                                   max(2, int(2 * h / 0.015) + 2))
                       for h in np.asarray(half, dtype=float)]
        grid = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
        parts.append(grid.reshape(-1, 3) @ np.asarray(axes).T + center)
    task.robot.get_arm(arm_tag).planner.update_obstacles(
        np.concatenate(parts) if parts else np.empty((0, 3)), resolution=0.015,
    )


def endpoint_reaches_goal(planner, plan, link_pose7):
    """Check MPLib FK without moving the simulated robot to the candidate."""
    if plan is None or not plan.success or not plan.position.size:
        return False
    backend = planner._mplib
    qpos = backend.pad_move_group_qpos(np.asarray(plan.position[-1], dtype=float))
    backend.pinocchio_model.compute_forward_kinematics(qpos)
    actual = backend.pinocchio_model.get_link_pose(backend.link_name_2_idx[backend.move_group])
    goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(link_pose7)
    distance = float(np.linalg.norm(np.asarray(actual.p) - np.asarray(goal.p)))
    dot = float(abs(np.dot(np.asarray(actual.q), np.asarray(goal.q))))
    angle = float(2 * np.arccos(np.clip(dot, 0.0, 1.0)))
    if distance > 0.01 or angle > 0.05:
        print(f"[handover-plan] reject FK position_error={distance:.4f} "
              f"orientation_error={angle:.4f}", flush=True)
        return False
    return True


def move_seeded(task, link_pose7, arm_tag):
    """Reach a nearby MPLib IK solution through collision-checked joint RRT."""
    refresh_avatar_obstacles(task, arm_tag)
    arm = task.robot.get_arm(arm_tag)
    planner = arm.planner

    def execute_checked(plan):
        task.execute_plan(plan, arm_tag)
        task._reseed_cached_arm_target(arm)
        # A valid FK endpoint can still be physically blocked. Every route,
        # including plan_pose fallbacks, must check the measured arrival.
        desired = np.asarray(link_pose7, dtype=float)[:3]
        measured = to_numpy(arm.ee_link.get_pos()).astype(float).reshape(-1)[:3]
        if float(np.linalg.norm(measured - desired)) > 0.05:
            task.plan_success = False
            return None
        return plan

    def checked_fallback():
        plan = task.move_to_pose(link_pose7, arm_tag)
        if not endpoint_reaches_goal(planner, plan, link_pose7):
            task.plan_success = False
            return None
        return execute_checked(plan)

    current = np.asarray(arm.get_arm_qpos(), dtype=float)
    solved = planner.solve_ik(current, np.asarray(link_pose7), log=False)
    if not solved.success or not solved.position.size:
        return checked_fallback()
    goal = np.asarray(solved.position[-1], dtype=float)
    if float(np.linalg.norm(goal - current)) > 3.5:
        return checked_fallback()
    backend = planner._mplib
    result = backend.plan_qpos(
        goal_qposes=[planner._pad_qpos(goal)],
        current_qpos=planner._pad_qpos(current),
        time_step=1 / 250, planning_time=getattr(planner, "_timeout", None) or 10,
        rrt_range=0.3, verbose=False,
    )
    if not result or result.get("status") != "Success":
        return checked_fallback()
    position = result["position"]
    plan = PlanResult(True, position, result.get("velocity", np.zeros_like(position)))
    if not endpoint_reaches_goal(planner, plan, link_pose7):
        task.plan_success = False
        return None
    return execute_checked(plan)
