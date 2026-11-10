"""Optional, non-executing lift checks for cooperative YAML grasp branches."""
import numpy as np

from ...grasp import tcp_to_link_pose
from ...utils import Pose


OPT_IN = "categorize_cooperative_lift_preflight_v1"
MODELS = frozenset({("038_milk-box", 0), ("073_rubikscube", 1)})
PRECISIONS = (.005, .002, .001)


def cooperative_lift_check(task, pick, arm_tag, safe_z, existing_check=None):
    """Compose a lift predicate with the caller's existing selector check.

    Only the cooperative facade calls this hook. The authored small/high lift
    targets are checked from the proposed grasp endpoint, using the current
    planner obstacle snapshot and the same native screw/FK/path checks as
    execution. Nothing is executed and native kinematic state is restored.
    """
    if (task.config.get(OPT_IN) is not True
            or bool(getattr(task, "_CATEGORIZE_COLUMN_LAYOUT", False))):
        return existing_check
    model = next(((name, int(model_id)) for actor, name, model_id, _basket
                  in task._sort_targets if actor is pick.actor), None)
    if model not in MODELS:
        return existing_check

    arm = task.robot.get_arm(arm_tag)

    def check(plans):
        backend = arm.planner._mplib
        saved = np.array(backend.robot.get_qpos(), dtype=float, copy=True)
        move_indices = np.asarray(backend.move_group_joint_indices, dtype=int)
        try:
            if existing_check is not None and not existing_check(plans):
                return False
            _pre, grasp_plan, grasp_link, _pre_link = plans
            if grasp_plan is None or not grasp_plan.success:
                return False
            position = np.asarray(grasp_plan.position, dtype=float)
            center = np.asarray(pick.get_center(), dtype=float)
            if (position.ndim != 2 or position.shape[1] != arm.n_arm or not len(position)
                    or not np.all(np.isfinite(position)) or center.shape != (3,)
                    or not np.all(np.isfinite(center)) or not np.isfinite(pick.radius)
                    or not np.isfinite(safe_z)):
                return False
            tcp = grasp_link * arm.tcp_offset
            small = np.asarray(tcp.p, dtype=float).copy()
            small[2] = center[2] + float(pick.radius)
            high = np.asarray(tcp.p, dtype=float).copy()
            high[2] = float(safe_z)
            current = position[-1].copy()
            for phase, point in (("small lift", small), ("high lift", high)):
                target = tcp_to_link_pose(Pose(point, tcp.q), arm.tcp_offset).to_pose7()
                accepted = None
                for precision in PRECISIONS:
                    plan = arm.planner.plan_screw_path(
                        current.copy(), target.copy(), qpos_step=precision, log=False,
                    )
                    if task._categorize_plan_valid(arm, plan, current, target):
                        accepted = plan
                        break
                if accepted is None:
                    print(f"[categorize] cooperative lift preflight rejected "
                          f"{model[0]}@{model[1]}: {phase}", flush=True)
                    return False
                current = np.asarray(accepted.position[-1], dtype=float).copy()
            print(f"[categorize] cooperative lift preflight accepted "
                  f"{model[0]}@{model[1]} before approach", flush=True)
            return True
        finally:
            backend.robot.set_qpos(saved, True)
            backend.planning_world.set_qpos_all(saved[move_indices])
            backend.pinocchio_model.compute_forward_kinematics(saved)

    return check
