"""Runtime radius contact geometry through the shared grasp selector."""
import numpy as np

from ...grasp import GraspPose
from ...utils import Pose
from .carry import carry_and_release, uses_direct_carry
from .release import lower_release, uses_lower_release


QUARTER_TURN_RETRY = True


def radius_candidates(task, pick, object_pose, quarter=False):
    center = np.asarray(pick.get_center(), dtype=float).copy()
    center[2] = max(task.TABLE_TOP_Z + .002,
                    center[2] - .3 * pick.radius + pick.grasp_z_offset)
    candidates = []
    get_yaw = getattr(pick, 'get_grasp_yaw', None)
    original_yaw = float(get_yaw()) if get_yaw is not None else 0.
    branches = ([('radius_half', original_yaw + np.pi / 2), ('radius_mhalf', original_yaw - np.pi / 2)]
                if quarter else
                [('radius_pi', original_yaw + np.pi), ('radius_zero', original_yaw)])
    for name, yaw in branches:
        local = object_pose.inv() * task._top_down_tcp(center, yaw=yaw)
        candidates.append(GraspPose(name, local.p, local.q, pre_distance=.15,
                                    source='manual', scale_frame='runtime'))
    return candidates


def pick_and_place_radius(task, pick, place, actor, arm_tag, *,
                          object_name='cube', model_id=0, object_scale=1.):
    """Preserve radius post-grasp motion, selecting a feasible jaw branch first.

    Runtime annotations use measured geometry and the loaded model scale. Both
    candidates share contact position and approach axis; a half turn swaps
    the parallel fingers. No task-private grasp planner is used.
    """
    arm = task.robot.get_arm(arm_tag)
    task.open_gripper(arm_tag)
    object_pose = actor.get_pose()
    get_q = getattr(arm, 'get_arm_qpos', None)
    q0 = None if get_q is None else np.asarray(get_q(), dtype=float).copy()
    result = task.select_and_execute_grasp(
        object_name, object_pose, arm_tag, model_id=model_id, object_scale=object_scale,
        candidate_grasps=radius_candidates(task, pick, object_pose),
        max_candidates=2, plan_attempts=3, screw_qpos_step=.005,
        table_z=task.TABLE_TOP_Z,
    )
    if (result is None and QUARTER_TURN_RETRY and q0 is not None
            and float(np.max(np.abs(np.asarray(get_q(), dtype=float) - q0))) < .02):
        # Both jaw branches stalled or ran joint 7 into its limit.
        # These pool objects are round or square in plan, so the jaws may
        # equally close a quarter turn away; nothing has moved yet.
        print(f'[categorize] radius {object_name}: retrying with quarter-turn jaws', flush=True)
        object_pose = actor.get_pose()
        result = task.select_and_execute_grasp(
            object_name, object_pose, arm_tag, model_id=model_id, object_scale=object_scale,
            candidate_grasps=radius_candidates(task, pick, object_pose, quarter=True),
            max_candidates=2, plan_attempts=3, screw_qpos_step=.005,
            table_z=task.TABLE_TOP_Z,
        )
    if result is None:
        return False
    grasp_link, _, grasp = result
    target = grasp_link * arm.tcp_offset
    task._categorize_arrival(grasp_link.to_pose7(), arm_tag)
    print(f'[categorize] radius {object_name}@{model_id} selected {grasp.name}', flush=True)
    for _ in range(100):
        task.step_sim()
    radius = float(pick.radius)
    close = pick.close_value
    if close is None:
        close = max(.010, radius - .005 if radius < .030 else radius * .7) / .04
    task.set_gripper(float(np.clip(close, 0., 1.)), arm_tag)
    for _ in range(200):
        task.step_sim()
    task._categorize_begin_hold(arm_tag)
    for _ in range(40):
        task.step_sim()

    def move(position):
        task._move_tcp_screw(Pose(position, target.q), arm_tag)

    lift_mid = np.asarray(target.p, dtype=float).copy()
    lift_mid[2] = float(pick.get_center()[2]) + radius
    move(lift_mid)
    if uses_direct_carry(task, object_name, model_id, place):
        return carry_and_release(task, pick, place, actor, arm_tag)
    after_mid = np.asarray(pick.get_center(), dtype=float).copy()
    safe_z = task.TABLE_TOP_Z + .38 if place.transport_z is None else place.transport_z
    move(np.array([target.p[0], target.p[1], safe_z]))
    lifted = np.asarray(pick.get_center(), dtype=float)
    if float(lifted[2] - after_mid[2]) < max(.03, .5 * radius):
        task._reject_categorize_motion(f'{object_name} did not retain vertical lift')
    held_offset = lifted - np.asarray(arm.get_ee_pose()[:3])
    move(np.array([place.pos[0], place.pos[1], safe_z]))
    for _ in range(100):
        task.step_sim()
    drift = (np.asarray(pick.get_center()) - np.asarray(arm.get_ee_pose()[:3])
             - held_offset)
    if not np.all(np.isfinite(drift)) or np.linalg.norm(drift) > .03:
        task._reject_categorize_motion(f'{object_name} carry offset changed')
    release_z = float(place.drop_from_z if place.drop_from_z is not None else place.pos[2])
    if abs(release_z - safe_z) > 1e-4:
        move(np.array([place.pos[0], place.pos[1], release_z]))
        for _ in range(50):
            task.step_sim()
    if uses_lower_release(task, object_name):
        return lower_release(task, actor, object_name, arm_tag)
    task.open_gripper(arm_tag)
    for _ in range(250):
        task.step_sim()
    return True
