"""Opt-in, physically tracked last descent for the human-take expert.

Keep the established grasp/carry and contact parameters. Never substitute a
high release when the native MPLib descent or its measured arrival fails.
"""

import json
import numpy as np

from ..grasp import link_to_tcp_pose, tcp_to_link_pose
from ..utils import Pose, to_numpy
from .take_release_plan import validate_release_plan


def _array(value, shape):
    result = np.asarray(to_numpy(value), dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid release observation: expected {shape}")
    return result


def _bounds(entity):
    result = _array(entity.get_AABB(), (2, 3))
    if np.any(result[0] >= result[1]):
        raise ValueError("empty release geometry")
    return result


def _angle(a, b):
    a, b = _array(a, (4,)), _array(b, (4,))
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        raise ValueError("zero release quaternion")
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)) /
                                      (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1)))


def _state(task, arm):
    entity = task.apple_actor.entity
    fingers = [link for link in arm.entity.links if "finger" in link.name]
    if len(fingers) != 2:
        raise ValueError("unsupported release finger geometry")
    tcp = _array(arm.get_ee_pose(), (7,))
    quat = _array(entity.get_quat(), (4,))
    if any(not np.isclose(np.linalg.norm(q), 1., atol=1e-4, rtol=0)
           for q in (tcp[3:], quat)):
        raise ValueError("nonunit release quaternion")
    return dict(tcp=tcp,
                position=_array(entity.get_pos(), (3,)),
                quaternion=quat,
                bounds=_bounds(entity), fingers=np.stack([_bounds(f) for f in fingers]))


def release_target(state, plate):
    """Use live held geometry, not the object's possibly offset mesh origin."""
    low = min(state["bounds"][0, 2], state["fingers"][:, 0, 2].min())
    delta = np.r_[plate.mean(axis=0)[:2] - state["bounds"].mean(axis=0)[:2],
                  plate[1, 2] + .005 - low]
    if not (-.25 <= delta[2] <= 0) or np.linalg.norm(delta[:2]) > .06:
        raise ValueError("release is outside the bounded downward corridor")
    if state["bounds"][0, 2] + delta[2] - plate[1, 2] > .03:
        raise ValueError("finger clearance would require another high release")
    target = state["tcp"].copy()
    target[:3] += delta
    return target


def _retained(state, initial, plate):
    offset = state["position"] - state["tcp"][:3]
    original = initial["position"] - initial["tcp"][:3]
    if np.linalg.norm(offset - original) > .015 or _angle(
            state["quaternion"], initial["quaternion"]) > .15:
        raise ValueError("held object moved relative to the gripper")
    if _angle(state["tcp"][3:], initial["tcp"][3:]) > .05:
        raise ValueError("measured descent changed gripper orientation")
    if min(state["bounds"][0, 2], state["fingers"][:, 0, 2].min()) < plate[1, 2] + .002:
        raise ValueError("measured held geometry entered the plate-rim clearance")


def _require_no_retreat(task):
    # The retreat controller may put down/open a held object itself. This
    # experimental phase supports only the fixed-set, retreat-disabled mode;
    # reject it read-only instead of invoking that actuator and then aborting.
    if task._avatar_retreat_enabled() or getattr(task, "avatar_retreat_checker", None) is not None:
        raise ValueError("avatar retreat is incompatible with guarded release")


def _terminal_snapshot(task, arm, state, initial, target, plate, command, progress, checked):
    """Best-effort terminal reads; never plan, run FK, step, or set a target."""
    result = {"progress": dict(progress), "observation_errors": {}}

    def read(name, getter):
        try:
            value = getter()
            json.dumps(value, allow_nan=False)
            result[name] = value
            return value
        except Exception as exc:
            result["observation_errors"][name] = f"{type(exc).__name__}: {exc}"
            return None

    # For retention rejection, use exactly the state that failed its gate.
    # Earlier failures may not have a valid state; reads remain best-effort.
    if state is None:
        try:
            state = _state(task, arm)
        except Exception as exc:
            result["observation_errors"]["measured"] = f"{type(exc).__name__}: {exc}"
    if state is not None:
        read("measured", lambda: {key: value.tolist() for key, value in state.items()})
        if initial is not None:
            read("origin_offset_drift_world_m", lambda: (
                (state["position"] - state["tcp"][:3])
                - (initial["position"] - initial["tcp"][:3])).tolist())
            read("object_rotation_from_initial_rad", lambda: _angle(
                state["quaternion"], initial["quaternion"]))
            read("tcp_rotation_from_initial_rad", lambda: _angle(
                state["tcp"][3:], initial["tcp"][3:]))
        if target is not None:
            read("measured_tcp_minus_target_xyz_m", lambda: (state["tcp"][:3] - target[:3]).tolist())
        if plate is not None:
            read("bottom_above_reference_rim_m", lambda: float(state["bounds"][0, 2] - plate[1, 2]))
            read("finger_bottom_above_reference_rim_m", lambda: float(state["fingers"][:, 0, 2].min() - plate[1, 2]))
    read("live_plate_bounds", lambda: _bounds(task.plate_actor.entity).tolist())
    read("full_qpos", lambda: _array(arm.entity.get_qpos(), (9,)).tolist())
    read("cached_qpos", lambda: None if arm._cached_target is None
         else _array(arm._cached_target, (9,)).tolist())
    read("commanded_waypoint", lambda: None if command is None else _array(command, (7,)).tolist())
    read("gripper_command", lambda: float(arm.gripper_val))
    read("physical_time_s", lambda: progress["physics_steps"] * float(task.SIMULATION_DT))
    read("video_frames", lambda: len(task._video_frames))
    native = checked.get("endpoint_world_link_pose7") if checked is not None else None
    if native is None:
        result["native_endpoint_fk"] = {"available": False,
                                         "reason": "validator did not publish an endpoint pose"}
    else:
        def native_values():
            pose = _array(native, (7,))
            tcp = _array(link_to_tcp_pose(Pose(pose[:3], pose[3:]), arm.tcp_offset).to_pose7(), (7,))
            value = {"available": True, "world_link_pose7": pose.tolist(), "world_tcp_pose7": tcp.tolist(),
                     "comparison": "planned endpoint, not instantaneous waypoint tracking error"}
            if target is not None:
                value["native_endpoint_tcp_minus_target_xyz_m"] = (tcp[:3] - target[:3]).tolist()
            if state is not None:
                value["measured_tcp_minus_native_endpoint_xyz_m"] = (state["tcp"][:3] - tcp[:3]).tolist()
            return value
        read("native_endpoint_fk", native_values)
    return result


def lower_for_release(task, arm_tag):
    """One native screw path, six PD substeps, then five quiet arrival samples."""
    diagnostic = {"stage": "start"}
    arm = initial = observed = target = plate = command = checked = None
    progress = {"phase": "before_plan", "physics_steps": 0,
                "waypoint_index": None, "waypoint_count": None, "substep": None}

    def observe():
        nonlocal observed
        observed = None
        observed = _state(task, arm)
        return observed

    def emit(stage, **values):
        if (stage in ("ready_to_open", "rejected_without_opening")
                and getattr(task, "config", {}).get("guarded_release_diagnostics", False)):
            try:
                terminal = _terminal_snapshot(task, arm, observed, initial, target, plate,
                                              command, progress, checked)
                json.dumps(terminal, allow_nan=False)
            except Exception as exc:
                terminal = {"observation_errors": {"snapshot": f"{type(exc).__name__}: {exc}"}}
            values["terminal_snapshot"] = terminal
        updated = {**diagnostic, "stage": stage, **values}
        line = json.dumps(updated, allow_nan=False)
        diagnostic.update(updated)
        task._guarded_release_diagnostics = diagnostic.copy()
        print("[take-guarded-release] " + line, flush=True)

    try:
        arm = task.robot.get_arm(arm_tag)
        if not task.plan_success:
            raise ValueError("an earlier planning failure is still latched")
        _require_no_retreat(task)
        planner = arm.planner
        if planner._bounded_paths_debug_enabled():
            raise ValueError("guarded release requires native MPLib")
        initial = observe()
        plate, table = _bounds(task.plate_actor.entity), _bounds(task.table)
        target = release_target(initial, plate)
        grip = float(arm.gripper_val)
        if not np.isfinite(grip):
            raise ValueError("invalid held gripper command")
        dt = float(task.SIMULATION_DT)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid simulation step")
        emit("before_plan", initial={key: value.tolist() for key, value in initial.items()},
             target_tcp=target.tolist(), plate_bounds=plate.tolist(), table_bounds=table.tolist(),
             arm_qpos=np.asarray(arm.get_arm_qpos()).tolist(),
             full_qpos=np.asarray(to_numpy(arm.entity.get_qpos())).tolist(), gripper=grip)
        task._refresh_planner_obstacles(arm_tag)
        link = tcp_to_link_pose(Pose(target[:3], target[3:]), arm.tcp_offset).to_pose7()
        plan = planner.plan_screw_path(arm.get_arm_qpos(), link, qpos_step=.02, log=True)
        checked = validate_release_plan(arm, plan, link, table_bounds=table, plate_bounds=plate)
        emit("plan_checked", plan_check=checked)
        # Direct arm PD targets preserve cached finger targets; no pose setters.
        # Observe every physical substep, rather than only the plan endpoint.
        progress["phase"] = "descent"
        progress["waypoint_count"] = len(plan.position)
        for waypoint_index, waypoint in enumerate(plan.position):
            for substep in range(6):
                _require_no_retreat(task)
                arm.set_arm_joints(waypoint)
                command = waypoint
                progress.update(waypoint_index=waypoint_index, substep=substep)
                task.robot.set_gripper(grip, arm_tag)
                task.step_sim()
                progress["physics_steps"] += 1
                _retained(observe(), initial, plate)
        previous = observe()
        quiet = 0
        stride = max(1, round(.02 / dt))
        progress["phase"] = "quiet_wait"
        for sample in range(max(1, round(.6 / (stride * dt)))):
            for substep in range(stride):
                _require_no_retreat(task)
                progress.update(sample_index=sample, substep=substep)
                task.step_sim()
                progress["physics_steps"] += 1
                _retained(observe(), initial, plate)
            current = observe()
            speed = np.linalg.norm(current["position"] - previous["position"]) / (stride * dt)
            spin = _angle(current["quaternion"], previous["quaternion"]) / (stride * dt)
            error = float(np.linalg.norm(current["tcp"][:3] - target[:3]))
            centered = np.linalg.norm(current["bounds"].mean(axis=0)[:2] - plate.mean(axis=0)[:2]) < .008
            gap = float(current["bounds"][0, 2] - plate[1, 2])
            quiet = quiet + 1 if (error < .008 and centered and .002 <= gap <= .03
                                  and speed < .03 and spin < .3) else 0
            previous = current
            if quiet >= 5:
                if not task.plan_success:
                    raise ValueError("planning failure latched during descent")
                emit("ready_to_open", samples=sample + 1, tcp_error=error,
                     bottom_above_rim=gap, linear_speed=float(speed), angular_speed=spin,
                     measured={key: value.tolist() for key, value in current.items()})
                return True
        raise ValueError("no stable measured low-release arrival within 0.6 seconds")
    except Exception as exc:
        task.plan_success = False
        emit("rejected_without_opening", reason=str(exc))
        return False
