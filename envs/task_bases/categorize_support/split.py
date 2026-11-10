"""categorize_split_screw_v1: one exact translation-midpoint split for two scoped screws.

Literal config opt-in ``categorize_split_screw_v1=True``. Scope: column layout,
milk model 0 first post-close vertical small lift and wood model 1 lower-release
descent. Every original full-target attempt runs first and unchanged. Only when
this module's own raw native calls at the original target return well-formed
native non-success is ONE arithmetic translation midpoint proposed, with the
original orientation and the original exact final target. Exceptions, malformed
results, rejected returned starts and guard-rejected native successes fail
closed. No recursion, jitter, retries-until-pass, RRT or Genesis IK. Legs run
as separate stopped legs through the existing timed executor; the second leg is
re-planned from the measured midpoint state before it moves. No shared wrapper,
installed library or global state is edited.
"""
import hashlib
import time

import numpy as np
from transforms3d.quaternions import quat2mat

from ...base_task_parts.grasp_select import mplib_world_link_pose
from ...grasp import tcp_to_link_pose
from ...planning.base import PlanResult
from ...robot.franka_robot import _get_dof_idx, to_numpy
from ...utils import Pose
from .milk_support import bounds_corners, payload_clear, payload_obstacles
from .motion import pose_errors, valid_pose

OPT_IN = 'categorize_split_screw_v1'
SCOPE = {('038_milk-box', 0): 'small_lift', ('086_woodenblock', 1): 'descent'}
NATIVE_STEP_S = 1 / 250
# Existing frozen policy: joint-limit slack already used by _categorize_plan_valid
# and by the installed screw planner's own check. Not a new tolerance.
LIMIT_SLACK = .001
LATERAL_VERTICAL_TOLERANCE = .001


def enabled(task):
    config = getattr(task, 'config', None) or {}
    return config.get(OPT_IN) is True and bool(getattr(task, '_CATEGORIZE_COLUMN_LAYOUT', False))


def scoped_phase(task, actor):
    for item, name, model_id, _basket in getattr(task, '_sort_targets', []):
        if item is actor:
            return SCOPE.get((name, int(model_id)))
    return None


def log_entry(task, **fields):
    entries = getattr(task, '_categorize_split_log', None)
    if entries is None:
        entries = []
        task._categorize_split_log = entries
    entries.append(dict(fields))


# ---------------------------------------------------------------- state ----

def read_full9(arm):
    """Measured arm and both finger positions in the native user-joint order.

    Returns (full, move_idx, finger_idx). Rejects unsupported mappings, wrong
    shapes, non-finite values and materially out-of-limit starts. Measured
    asymmetric finger values are preserved exactly; nothing is substituted.
    """
    backend = arm.planner._mplib
    names = list(backend.robot.get_user_joint_names())
    qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
    joints = {}
    for joint in list(getattr(arm, 'arm_joints', [])) + list(getattr(arm, 'finger_joints', [])):
        if joint is not None:
            joints[str(joint.name)] = joint
    full = np.empty(len(names), dtype=float)
    for i, name in enumerate(names):
        joint = joints.get(name) or joints.get(name.replace('panda_', ''))
        if joint is None:
            raise ValueError(f'unsupported native joint mapping: {name}')
        idx = _get_dof_idx(joint)
        if idx is None or not 0 <= idx < len(qpos):
            raise ValueError(f'unsupported dof index for {name}')
        full[i] = float(qpos[idx])
    if not np.all(np.isfinite(full)):
        raise ValueError('non-finite measured joint state')
    limits = np.asarray(backend.joint_limits, dtype=float)
    if limits.shape != (len(names), 2):
        raise ValueError('unexpected native joint limit shape')
    if np.any(full < limits[:, 0] - LIMIT_SLACK) or np.any(full > limits[:, 1] + LIMIT_SLACK):
        raise ValueError('measured start outside native joint limits')
    move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
    finger_idx = np.asarray([i for i, n in enumerate(names) if 'finger' in n], dtype=int)
    if len(finger_idx) != 2 or len(move_idx) != arm.n_arm:
        raise ValueError('native model does not expose seven arm joints and two fingers')
    return full, move_idx, finger_idx


def cloud_identity(task):
    return getattr(task, '_categorize_last_cloud_identity', None)


# ----------------------------------------------------------------- seam ----

def raw_screw(arm, full, target_link_pose7, qpos_step):
    """Task-local checked native call returning the raw outcome kind.

    kind is one of 'success', 'native_failure', 'exception', 'malformed'. Only
    'native_failure' (a well-formed non-Success status dict) may trigger the
    split. The native robot/planning-world state is restored after the call.
    """
    backend = arm.planner._mplib
    goal = arm.planner._to_mplib_pose(np.asarray(target_link_pose7, dtype=float))
    move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
    start = np.array(full, dtype=float)
    saved = None
    try:
        saved = np.array(backend.robot.get_qpos(), dtype=float, copy=True)
    except Exception:
        saved = None
    started = time.perf_counter()
    try:
        raw = backend.plan_screw(goal_pose=goal, current_qpos=start.copy(), time_step=NATIVE_STEP_S,
                                 qpos_step=float(qpos_step), wrt_world=True, verbose=False)
    except Exception as exc:
        return dict(kind='exception', status=f'{type(exc).__name__}: {exc}', elapsed=time.perf_counter() - started)
    finally:
        if saved is not None:
            try:
                backend.robot.set_qpos(saved, True)
                backend.planning_world.set_qpos_all(saved[move_idx])
            except Exception:
                pass
    elapsed = time.perf_counter() - started
    if not isinstance(raw, dict) or not isinstance(raw.get('status'), str):
        return dict(kind='malformed', status='non-dict or missing status', elapsed=elapsed)
    if raw['status'] != 'Success':
        return dict(kind='native_failure', status=raw['status'], elapsed=elapsed)
    pos, vel = raw.get('position'), raw.get('velocity')
    try:
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float) if vel is not None else None
    except Exception:
        return dict(kind='malformed', status='non-numeric path', elapsed=elapsed)
    if (pos.ndim != 2 or pos.shape[1] != len(move_idx) or not len(pos) or not np.all(np.isfinite(pos))
            or vel is None or vel.shape != pos.shape or not np.all(np.isfinite(vel))):
        return dict(kind='malformed', status='malformed Success arrays', elapsed=elapsed)
    if not np.allclose(pos[0], start[move_idx], atol=1e-12, rtol=0.):
        return dict(kind='malformed', status='returned start differs from native start', elapsed=elapsed)
    return dict(kind='success', status='Success', elapsed=elapsed, plan=PlanResult(True, pos, vel),
                samples=int(len(pos)), native_duration_s=float((len(pos) - 1) * NATIVE_STEP_S))


# ----------------------------------------------------------- validation ----

def payload_floor(task, phase, start_tcp, local):
    """Lowest allowed payload corner height for this phase.

    Descents keep the existing table margin. The first post-close lift starts
    with the payload resting on the table, so its floor is the payload's own
    measured starting bottom (it may never go lower than where it started),
    capped by the table margin.
    """
    margin = float(task.TABLE_TOP_Z + task._OBSTACLE_MARGIN_Z)
    if phase != 'small_lift' or local is None:
        return margin
    start = np.asarray(start_tcp, dtype=float)
    world = np.asarray(local, dtype=float) @ quat2mat(start[3:]).T + start[:3]
    return float(min(margin, world[:, 2].min() - 1e-6))


def full_path_checks(task, arm, plan, full, finger_idx, local, obstacles, floor_z=None):
    """Measured-finger, full finger sweep, self/env and payload checks on every sample."""
    if floor_z is None:
        floor_z = float(task.TABLE_TOP_Z + task._OBSTACLE_MARGIN_Z)
    backend = arm.planner._mplib
    pad = arm.planner._pad_qpos
    measured = np.asarray(full, dtype=float)[finger_idx]
    for index, joints in enumerate(np.asarray(plan.position)):
        base = pad(np.asarray(joints, dtype=float).copy())
        for widths in ([measured], [np.array([w, w]) for w in np.linspace(0., .04, 11)]):
            for w in widths:
                state = base.copy()
                state[finger_idx] = w
                if backend.check_for_self_collision(state) or backend.check_for_env_collision(state):
                    return dict(ok=False, reason='finger sweep collision', sample=index, widths=np.asarray(w).tolist())
        if local is not None:
            pose = mplib_world_link_pose(arm.planner, joints)
            if pose is None or not payload_clear(local, pose * arm.tcp_offset, obstacles, floor_z):
                return dict(ok=False, reason='payload clearance', sample=index, floor_z=float(floor_z))
    return dict(ok=True)


def endpoint_errors(arm, plan, target):
    end = mplib_world_link_pose(arm.planner, np.asarray(plan.position)[-1])
    if end is None:
        return None
    distance, angle = pose_errors(end.to_pose7(), np.asarray(target, dtype=float))
    return dict(xyz=float(distance), angle=float(angle))


def plan_leg(task, arm, full, move_idx, finger_idx, target, local, obstacles, precisions, phase, leg, floor_z=None):
    """Original-precision attempts through the seam; returns (record, plan or None).

    A well-formed native failure at one precision moves to the next. Any
    exception/malformed/rejected-Success outcome fails the leg closed.
    """
    current = np.asarray(full, dtype=float)[move_idx]
    attempts = []
    for precision in precisions:
        outcome = raw_screw(arm, full, target, precision)
        row = dict(precision=float(precision), kind=outcome['kind'], status=outcome['status'], elapsed=outcome['elapsed'])
        if outcome['kind'] == 'native_failure':
            attempts.append(row)
            continue
        if outcome['kind'] != 'success':
            attempts.append(row)
            return dict(leg=leg, phase=phase, decision='fail_closed', attempts=attempts), None
        plan = outcome['plan']
        row.update(samples=outcome['samples'], native_duration_s=outcome['native_duration_s'],
                   endpoint=endpoint_errors(arm, plan, target))
        if not task._categorize_plan_valid(arm, plan, current, np.asarray(target, dtype=float)):
            row['validation'] = 'rejected native Success (plan validity)'
            attempts.append(row)
            return dict(leg=leg, phase=phase, decision='fail_closed', attempts=attempts), None
        checks = full_path_checks(task, arm, plan, full, finger_idx, local, obstacles, floor_z)
        if not checks['ok']:
            row['validation'] = checks
            attempts.append(row)
            return dict(leg=leg, phase=phase, decision='fail_closed', attempts=attempts), None
        row['validation'] = 'accepted'
        attempts.append(row)
        return dict(leg=leg, phase=phase, decision='accepted', attempts=attempts), plan
    return dict(leg=leg, phase=phase, decision='native_failure_all_precisions', attempts=attempts), None


def midpoint_geometry(start_tcp, final_tcp):
    """Exact arithmetic translation midpoint with the original orientation.

    Only pure vertical, orientation-preserving routes are in scope. Returns
    (mid Pose, final Pose) or None.
    """
    start, final = np.asarray(start_tcp, dtype=float), np.asarray(final_tcp, dtype=float)
    if not valid_pose(start) or not valid_pose(final):
        return None
    if np.linalg.norm(final[:2] - start[:2]) > LATERAL_VERTICAL_TOLERANCE:
        return None
    if 2 * np.arccos(np.clip(abs(np.dot(start[3:], final[3:])), 0., 1.)) > .05:
        return None
    mid = Pose((start[:3] + final[:3]) / 2., final[3:].copy())
    return mid, Pose(final[:3].copy(), final[3:].copy())


# ------------------------------------------------------------ proposal ----

def propose_split(task, arm, arm_tag, final_tcp, local, obstacles, precisions, phase, retreat_target=None):
    """Original full-target attempts through the seam, then the one midpoint proposal.

    Returns a dict with keys plan (full-target plan if it succeeded), leg1,
    leg2, mid_target, final_target, retreat, record; or None when the proposal
    fails closed or the original target succeeds natively through the seam.
    """
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    full, move_idx, finger_idx = read_full9(arm)
    final_link = tcp_to_link_pose(Pose.from_pose7(np.asarray(final_tcp, dtype=float)), arm.tcp_offset).to_pose7()
    record = dict(phase=phase, start_tcp=tcp.tolist(), full9=full.tolist(), fingers=full[finger_idx].tolist(),
                  final_tcp=np.asarray(final_tcp, dtype=float).tolist(), final_link=np.asarray(final_link).tolist(),
                  cloud=cloud_identity(task), payload_corners_local=None if local is None else np.asarray(local).tolist(),
                  obstacle_count=len(obstacles))
    floor_z = payload_floor(task, phase, tcp, local)
    record['payload_floor_z'] = floor_z
    original, plan = plan_leg(task, arm, full, move_idx, finger_idx, final_link, local, obstacles, precisions, phase, 'original', floor_z)
    record['original'] = original
    if plan is not None:
        record['decision'] = 'original_target_accepted_through_seam'
        log_entry(task, **record)
        return dict(plan=plan, leg1=None, leg2=None, record=record)
    if original['decision'] != 'native_failure_all_precisions':
        record['decision'] = 'fail_closed_original_' + original['decision']
        log_entry(task, **record)
        return None
    geometry = midpoint_geometry(tcp, final_tcp)
    if geometry is None:
        record['decision'] = 'fail_closed_not_vertical_or_rotated'
        log_entry(task, **record)
        return None
    mid, final = geometry
    mid_link = tcp_to_link_pose(mid, arm.tcp_offset).to_pose7()
    record.update(mid_tcp=mid.to_pose7().tolist(), mid_link=np.asarray(mid_link).tolist())
    leg1_record, leg1 = plan_leg(task, arm, full, move_idx, finger_idx, mid_link, local, obstacles, precisions, phase, 'leg1', floor_z)
    record['leg1'] = leg1_record
    if leg1 is None:
        record['decision'] = 'fail_closed_leg1_' + leg1_record['decision']
        log_entry(task, **record)
        return None
    hypothetical = np.array(full, dtype=float)
    hypothetical[move_idx] = np.asarray(leg1.position)[-1]
    record['leg2_start'] = dict(hypothetical=True, full9=hypothetical.tolist())
    leg2_record, leg2 = plan_leg(task, arm, hypothetical, move_idx, finger_idx, final_link, local, obstacles, precisions, phase, 'leg2', floor_z)
    record['leg2'] = leg2_record
    if leg2 is None:
        record['decision'] = 'fail_closed_leg2_' + leg2_record['decision']
        log_entry(task, **record)
        return None
    retreat = None
    if retreat_target is not None:
        from .release import checked_plan
        retreat = checked_plan(task, arm, np.asarray(leg2.position)[-1], retreat_target, None, obstacles)
        record['retreat'] = 'accepted' if retreat is not None else 'rejected'
        if retreat is None:
            record['decision'] = 'fail_closed_retreat'
            log_entry(task, **record)
            return None
    record['decision'] = 'split_preflighted'
    record['native_durations_s'] = dict(leg1=float((len(leg1.position) - 1) * NATIVE_STEP_S),
                                        leg2=float((len(leg2.position) - 1) * NATIVE_STEP_S))
    log_entry(task, **record)
    return dict(plan=None, leg1=leg1, leg2=leg2, mid_target=mid_link, final_target=final_link,
                mid_tcp=mid, final_tcp=final, retreat=retreat, record=record, precisions=tuple(precisions),
                floor_z=floor_z)


def execute_split(task, arm_tag, proposal, actor, local_fn, obstacles_fn, phase):
    """Separate stopped legs: leg1, measured midpoint arrival, refresh, re-plan leg2, leg2.

    ``local_fn(tcp)`` recomputes the payload corners in the current TCP frame
    from the live geometry; ``obstacles_fn()`` recomputes the obstacle set.
    Any rejection raises through the existing rejection path; nothing opens.
    """
    arm = task.robot.get_arm(arm_tag)
    hold_steps = int(np.ceil(.10 / float(task.scene.sim.dt)))
    task._execute_categorize_plan(proposal['leg1'], arm_tag)
    task._categorize_arrival(proposal['mid_target'], arm_tag)
    arm, current, final = task._categorize_inputs(proposal['final_target'], arm_tag)
    full, move_idx, finger_idx = read_full9(arm)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    local = local_fn(tcp)
    obstacles = obstacles_fn()
    leg2_record, leg2 = plan_leg(task, arm, full, move_idx, finger_idx, final, local, obstacles, proposal['precisions'], phase, 'leg2_measured',
                                 proposal['floor_z'])
    log_entry(task, phase=phase, event='midpoint_replan', measured_tcp=tcp.tolist(), full9=full.tolist(),
              cloud=cloud_identity(task), leg2=leg2_record,
              extra_hold_s=float(hold_steps * float(task.scene.sim.dt)))
    if leg2 is None:
        task._reject_categorize_motion(f'split {phase}: remaining leg lost its checked path after the measured midpoint')
    task._execute_categorize_plan(leg2, arm_tag)
    task._categorize_arrival(final, arm_tag)
    return True


# ---------------------------------------------------------------- hooks ----

def small_lift_split(task, target_tcp, arm_tag):
    """Milk model 0 first post-close vertical small lift; returns True on success."""
    pick = getattr(task, '_categorize_active_pick', None)
    actor = getattr(pick, 'actor', None) if pick is not None else None
    if actor is None or scoped_phase(task, actor) != 'small_lift':
        return False
    if getattr(task, '_categorize_hold_local', None) is None:
        return False
    arm = task.robot.get_arm(arm_tag)
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    entity = getattr(actor, 'entity', actor)

    def local_fn(at_tcp):
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        return (bounds_corners(bounds) - at_tcp[:3]) @ quat2mat(at_tcp[3:])

    def obstacles_fn():
        return payload_obstacles(task, actor)

    arm, _current, _target = task._categorize_inputs(
        tcp_to_link_pose(target_tcp, arm.tcp_offset).to_pose7(), arm_tag)
    proposal = propose_split(task, arm, arm_tag, target_tcp.to_pose7(), local_fn(tcp), obstacles_fn(),
                             (.005, .002, .001), 'small_lift')
    if proposal is None:
        return False
    if proposal['plan'] is not None:
        task._execute_categorize_plan(proposal['plan'], arm_tag)
        task._categorize_arrival(tcp_to_link_pose(target_tcp, arm.tcp_offset).to_pose7(), arm_tag)
        return True
    return execute_split(task, arm_tag, proposal, actor, local_fn, obstacles_fn, 'small_lift')


def descent_split(task, arm, arm_tag, actor, current, target, local, obstacles, high):
    """Wood model 1 lower-release descent; returns a proposal dict or None.

    Called by lower_release only after its original checked_plan attempts
    returned None. The retreat from the hypothetical leg-2 endpoint is
    preflighted here before anything moves.
    """
    if scoped_phase(task, actor) != 'descent':
        return None
    tcp = np.asarray(arm.get_ee_pose(), dtype=float)
    final_tcp = (Pose.from_pose7(np.asarray(target, dtype=float)) * arm.tcp_offset).to_pose7()
    return propose_split(task, arm, arm_tag, final_tcp, local, obstacles, (.005, .004, .002, .001), 'descent',
                         retreat_target=high)


def identity_of_cloud(points):
    points = np.asarray(points, dtype=float)
    return dict(count=int(len(points)), sha256=hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest())
