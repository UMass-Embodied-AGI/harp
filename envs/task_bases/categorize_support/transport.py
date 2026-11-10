"""Non-executing continuation checks for the categorize milk side grasps."""

import numpy as np

from ...grasp import tcp_to_link_pose
from ...utils import Pose


def endpoint_matches(planner, plan, target):
    if plan is None or not plan.success or not plan.position.size:
        return False
    backend = planner._mplib
    backend.pinocchio_model.compute_forward_kinematics(
        planner._pad_qpos(plan.position[-1]),
    )
    actual = backend.pinocchio_model.get_link_pose(
        backend.link_name_2_idx[backend.move_group],
    )
    goal = backend.robot.get_base_pose().inv() * planner._to_mplib_pose(target)
    distance = float(np.linalg.norm(np.asarray(actual.p) - np.asarray(goal.p)))
    angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, goal.q)), 0.0, 1.0))
    return bool(distance <= 0.01 and angle <= 0.05)


def milk_continuation_check(task, pick, place, arm_tag, safe_z):
    """Return the standard grasp helper's optional post-plan predicate.

    A candidate must support the existing initial lift, vertical clearance,
    and basket carry, all with straight screw paths and fixed orientation.
    Only MPLib state is touched; no simulator joints or object poses change.
    """
    arm = task.robot.get_arm(arm_tag)

    def check(plans):
        _approach, grasp_plan, grasp_link, _pre_link = plans
        if not grasp_plan.position.size:
            return False
        grasp_tcp = grasp_link * arm.tcp_offset
        center = np.asarray(pick.get_center(), dtype=float)
        mid = np.asarray(grasp_tcp.p, dtype=float).copy()
        mid[2] = center[2] + float(pick.radius)
        lifted = np.asarray(grasp_tcp.p, dtype=float).copy()
        lifted[2] = safe_z
        carry = task._yaml_transport_target(
            place.pos, safe_z, center - np.asarray(grasp_tcp.p),
        )
        qpos = grasp_plan.position[-1]
        for label, point in (("initial lift", mid), ("clearance", lifted), ("carry", carry)):
            target = tcp_to_link_pose(Pose(point, grasp_tcp.q), arm.tcp_offset).to_pose7()
            plan = arm.planner.plan_screw_path(qpos, target, log=False)
            if not endpoint_matches(arm.planner, plan, target):
                print(f"[categorize] milk continuation rejected: {label}", flush=True)
                return False
            qpos = plan.position[-1]
        print("[categorize] milk continuation accepted before approach", flush=True)
        return True

    return check
