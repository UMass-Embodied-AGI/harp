"""Opt-in, read-only checkpoints for diagnosing the human-take grasp.

The pinned Genesis 1.2 collider stores contacts from the last scene.step(),
including ``penetration`` and ``normal``. Its public getter currently fails
when torch.gather receives int32 permutation indices. Native observations
therefore copy the live permutation and contact fields without calling that
getter (which also populates a simulator cache). Nothing is detected again.
Contact rows below retain global geom/link indices and are copied immediately;
they are telemetry, never scoring, contact control, or a retention certificate.
"""

import json

import numpy as np

from envs.contact import (
    _copy as _copy_array,
    _read_quadrants,
    copied_single_scene_contacts,
)


def _array(value, shape=None):
    result = _copy_array(value).astype(float, copy=False)
    if ((shape is not None and result.shape != shape)
            or not np.isfinite(result).all()):
        raise ValueError(f"invalid finite observation array; expected {shape}")
    return result


def _integer(value):
    number = _array(value, ()).item()
    if number < 0 or number != int(number):
        raise ValueError("expected nonnegative integer")
    return int(number)


def _positive_dt(value):
    number = _array(value, ()).item()
    if number <= 0:
        raise ValueError("expected positive simulation dt")
    return number


def _gripper_command(value):
    number = _array(value, ()).item()
    if not 0 <= number <= 1:
        raise ValueError("gripper command outside normalized range")
    return number


def _bounds(entity):
    bounds = _array(entity.get_AABB(), (2, 3))
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("invalid nonempty AABB")
    return bounds.tolist()


def _pose(position, quaternion):
    position, quaternion = _array(position, (3,)), _array(quaternion, (4,))
    if not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-4, rtol=0):
        raise ValueError("invalid unit quaternion")
    return np.r_[position, quaternion].tolist()


def _entity_identity(entity):
    first, end = _integer(entity.geom_start), _integer(entity.geom_end)
    if end < first:
        raise ValueError("invalid geom index range")
    return {"entity_idx": _integer(entity.idx), "geom_range": [first, end]}


def _finger_indices(arm):
    if len(arm.finger_joints) != 2:
        raise ValueError("expected two Franka finger joints")
    rows = []
    for joint in arm.finger_joints:
        qidx, didx = list(joint.qs_idx_local), list(joint.dofs_idx_local)
        if len(qidx) != 1 or len(didx) != 1:
            raise ValueError("expected single-DOF finger joint")
        qidx, didx = _integer(qidx[0]), _integer(didx[0])
        if max(qidx, didx) >= 9:
            raise ValueError("finger joint index outside Franka state")
        rows.append({"name": str(joint.name), "qpos_index": qidx,
                     "dof_index": didx})
    if any(len({row[key] for row in rows}) != 2 for key in ("qpos_index", "dof_index")):
        raise ValueError("duplicate finger indices")
    return rows


def _copied_single_scene_contacts(collider, *, field_reader=None):
    """Shared passive snapshot, with the observer's float64 telemetry schema."""
    if _integer(collider._solver.n_envs) != 0:
        raise ValueError("batched contact observations are unsupported")
    read = _read_quadrants if field_reader is None else field_reader
    contacts = copied_single_scene_contacts(collider, field_reader=read)
    identifiers = ("geom_a", "geom_b", "link_a", "link_b")
    return {key: value if key in identifiers else _array(value)
            for key, value in contacts.items()}


def _contacts(entity):
    """Copy one entity's last-step contacts; intentionally single-scene only."""
    solver = entity._solver
    if _integer(solver.n_envs) != 0:
        raise ValueError("batched contact observations are unsupported")
    collider = getattr(solver, "collider", None)
    source = "RigidEntity.get_contacts; last scene.step"
    if collider is None:
        raw = entity.get_contacts()
    else:
        # Avoid even the public getter's cache.setdefault side effect. Apply
        # exactly its default entity filter (either side, including self-pairs).
        raw = _copied_single_scene_contacts(collider)
        first, end = _entity_identity(entity)["geom_range"]
        mask = (((raw["geom_a"] >= first) & (raw["geom_a"] < end))
                | ((raw["geom_b"] >= first) & (raw["geom_b"] < end)))
        raw = {key: value[mask].copy() for key, value in raw.items()}
        source = "copied_int64_contact_permutation; last scene.step"
    if not isinstance(raw, dict):
        raise ValueError("contact getter did not return a dictionary")
    if "valid_mask" in raw:
        raise ValueError("unexpected batched contact mask")
    arrays = {key: _array(raw[key]) for key in (
        "geom_a", "geom_b", "link_a", "link_b", "penetration", "position",
        "normal", "force_a", "force_b")}
    if arrays["geom_a"].ndim != 1:
        raise ValueError("unexpected contact index dimensions")
    count = len(arrays["geom_a"])
    for key, value in arrays.items():
        expected = (count, 3) if key in ("position", "normal", "force_a", "force_b") else (count,)
        if value.shape != expected:
            raise ValueError(f"invalid contact field shape: {key}")
    result = []
    for index in range(count):
        row = {key: value[index].tolist() for key, value in arrays.items()}
        for key in ("geom_a", "geom_b", "link_a", "link_b"):
            row[key] = _integer(row[key])
        names, entities = [], []
        for key in ("link_a", "link_b"):
            link = solver.links[row[key]]
            if _integer(link.idx) != row[key]:
                raise ValueError("global contact link index mismatch")
            names.append(str(link.name))
            entities.append(_integer(link.entity.idx))
        row.update(link_names=names, entity_indices=entities)
        result.append(row)
    return {"source": source,
            "count": count, "rows": result}


def observe_handover(task, arm_tag, phase):
    """Emit a best-effort snapshot, or do nothing when the opt-in is disabled.

    Returns the copied record for diagnostics/tests only. No state is stored on
    the task; getter, schema, and output errors never alter task control flow.
    """
    try:
        enabled = task.config.get("take_handover_diagnostics", False) is True
    except Exception:
        return None
    if not enabled:
        return None
    result = {"schema": 1, "observation_errors": {}}

    def read(name, getter):
        try:
            value = getter()
            json.dumps(value, allow_nan=False)
            result[name] = value
            return value
        except Exception as exc:
            result["observation_errors"][name] = f"{type(exc).__name__}: {exc}"
            return None

    read("phase", lambda: str(phase))
    read("arm_tag", lambda: str(arm_tag))
    step = read("sim_step", lambda: _integer(task._interaction_sim_step))
    dt = read("simulation_dt_s", lambda: _positive_dt(task.SIMULATION_DT))
    if step is not None and dt is not None:
        read("sim_time_s", lambda: step * dt)
    result["clock_source"] = "_interaction_sim_step * SIMULATION_DT"
    frames = read("video_frames", lambda: len(task._video_frames))
    if frames is not None:
        result["last_video_frame"] = frames - 1 if frames else None

    # Object lookup failures should not suppress independent robot telemetry.
    try:
        obj = task.apple_actor.entity
    except Exception as exc:
        result["observation_errors"]["object"] = f"{type(exc).__name__}: {exc}"
    else:
        read("object_identity", lambda: _entity_identity(obj))
        read("object_pose7", lambda: _pose(obj.get_pos(), obj.get_quat()))
        read("object_aabb", lambda: _bounds(obj))
        read("object_contacts", lambda: _contacts(obj))

    try:
        arm = task.robot.get_arm(arm_tag)
    except Exception as exc:
        result["observation_errors"]["arm"] = f"{type(exc).__name__}: {exc}"
    else:
        read("robot_identity", lambda: _entity_identity(arm.entity))
        measured = read("full_qpos", lambda: _array(arm.entity.get_qpos(), (9,)).tolist())
        cached = read("cached_qpos", lambda: None if arm._cached_target is None
                      else _array(arm._cached_target, (9,)).tolist())
        read("gripper_command", lambda: _gripper_command(arm.gripper_val))
        indices = read("finger_joints", lambda: _finger_indices(arm))
        if indices is not None:
            if measured is not None:
                result["measured_finger_qpos"] = [measured[row["qpos_index"]] for row in indices]
            if cached is not None:
                result["commanded_finger_qpos"] = [cached[row["qpos_index"]] for row in indices]

        def tcp_pose():
            value = _array(arm.get_ee_pose(), (7,))
            return _pose(value[:3], value[3:])

        read("tcp_pose7", tcp_pose)
        read("robot_aabb", lambda: _bounds(arm.entity))

        def finger_bounds():
            links = [link for link in arm.entity.links if "finger" in link.name]
            if len(links) != 2:
                raise ValueError("expected two Franka finger links")
            return [{"name": str(link.name), "link_idx": _integer(link.idx),
                     "aabb": _bounds(link)} for link in links]

        read("finger_aabbs", finger_bounds)
        read("robot_contacts", lambda: _contacts(arm.entity))
    try:
        print("[take-handover-state] " + json.dumps(result, allow_nan=False), flush=True)
    except Exception as exc:
        result["observation_errors"]["emit"] = f"{type(exc).__name__}: {exc}"
    return result
