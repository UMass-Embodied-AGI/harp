"""Fail-closed, simulator-read-only preflight for Franka's final TAKE lowering.

This checks returned trajectory samples, not continuous swept volumes or the
held object's geometry. The caller owns measured object/arrival/release gates.
Only MPLib model state is temporarily changed; fixtures and ACM are untouched.
"""

import os

import numpy as np


_ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
_COLLISION_LINKS = {f"panda_link{i}" for i in range(8)} | {
    "panda_hand", "panda_leftfinger", "panda_rightfinger",
}


def _array(value, shape, label):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: expected finite shape {shape}")
    return result.copy()


def _pose7(pose, label):
    result = np.concatenate([_array(pose.p, (3,), label),
                             _array(pose.q, (4,), label)])
    if not np.isclose(np.linalg.norm(result[3:]), 1., atol=1e-5, rtol=0):
        raise ValueError(f"{label}: nonunit quaternion")
    return result


def _fixture(bounds, label):
    from mplib.pymp import Pose
    from mplib.pymp.collision_detection.fcl import Box, CollisionObject, FCLObject

    bounds = _array(bounds, (2, 3), label)
    size = bounds[1] - bounds[0]
    if np.any(size <= 0):
        raise ValueError(f"{label}: empty or inverted bounds")
    identity = Pose(p=[0., 0., 0.], q=[1., 0., 0., 0.])
    return FCLObject(label, Pose(p=bounds.mean(axis=0), q=identity.q),
                     [CollisionObject(Box(*size), identity)], [identity])


def validate_release_plan(arm, plan, link_pose7, *, table_bounds, plate_bounds):
    """Return diagnostics or raise ValueError; never move the simulated arm.

    Native MPLib/Franka only. Checks measured and available commanded finger
    states at the measured start and every returned waypoint. Direct FCL
    fixture queries ignore no moving links or existing-world collision pairs.
    """
    try:
        return _validate(arm, plan, link_pose7, table_bounds, plate_bounds)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"release preflight unavailable: {type(exc).__name__}: {exc}") from exc


def _validate(arm, plan, target, table_bounds, plate_bounds):
    import mplib
    from mplib.pymp.collision_detection.fcl import distance

    planner = arm.planner
    backend = planner._mplib
    if (type(backend) is not mplib.Planner or arm.n_arm != 7 or planner.n_arm != 7
            or os.environ.get("DEBUG_BOUNDED_JOINT_PATHS", "").strip() == "1"):
        raise ValueError("native seven-joint Franka MPLib required")
    if plan is None or not plan.success:
        raise ValueError("unsuccessful plan")
    position = np.asarray(plan.position, dtype=float)
    if position.ndim != 2 or not len(position) or position.shape[1] != 7:
        raise ValueError("position: expected nonempty N x 7 path")
    position = _array(position, position.shape, "position")
    _array(plan.velocity, position.shape, "velocity")
    current = _array(arm.get_arm_qpos(), (7,), "measured arm")
    target = _array(target, (7,), "target")
    if not np.isclose(np.linalg.norm(target[3:]), 1., atol=1e-5, rtol=0):
        raise ValueError("target: nonunit quaternion")
    if not np.allclose(position[0], current, atol=1e-6, rtol=0):
        raise ValueError("plan does not start at measured arm")
    path = np.vstack([current, position])
    steps = np.abs(np.diff(path, axis=0))
    travel = steps.sum(axis=0)
    excursion = np.max(np.abs(path - current), axis=0)
    if np.max(steps) > .30 or np.max(travel) > 2.0 or np.max(excursion) > 1.75:
        raise ValueError("joint branch continuity/travel bound exceeded")

    robot, pin, world = backend.robot, backend.pinocchio_model, backend.planning_world
    names = list(robot.get_user_joint_names())
    if len(names) != 9 or set(names) != set(_ARM_NAMES + _FINGER_NAMES):
        raise ValueError("unsupported full Franka joint mapping")
    move = np.asarray(backend.move_group_joint_indices)
    if (move.shape != (7,) or not np.issubdtype(move.dtype, np.integer)
            or np.any(move < 0) or np.any(move >= 9)
            or [names[i] for i in move] != _ARM_NAMES):
        raise ValueError("unsupported move-group ordering")
    limits = _array(backend.joint_limits, (9, 2), "joint limits")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("inverted joint limits")
    sim = arm.entity.get_qpos()
    if hasattr(sim, "detach"):
        sim = sim.detach().cpu().numpy()
    sim = np.asarray(sim, dtype=float)
    if sim.ndim != 1 or not np.all(np.isfinite(sim)):
        raise ValueError("invalid measured full simulator qpos")
    mapping = planner._arm_joint_index
    indices = [mapping[name.removeprefix("panda_")] for name in names]
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            or i < 0 or i >= len(sim) for i in indices) or len(set(indices)) != 9):
        raise ValueError("invalid simulator-to-planner joint mapping")
    measured = sim[indices].copy()
    if not np.allclose(measured[move], current, atol=1e-6, rtol=0):
        raise ValueError("inconsistent measured joint snapshots")
    fingers = [names.index(name) for name in _FINGER_NAMES]
    configurations = [("measured", measured)]
    cached = getattr(arm, "_cached_target", None)
    if cached is not None:
        commanded = measured.copy()
        commanded[fingers] = _array(cached, sim.shape, "cached PD target")[
            np.asarray(indices)[fingers]]
        configurations.append(("cached_command", commanded))
    if all(hasattr(arm, key) for key in ("gripper_val", "gripper_open", "gripper_closed")):
        values = _array([arm.gripper_val, arm.gripper_open, arm.gripper_closed],
                        (3,), "gripper command")
        if not 0 <= values[0] <= 1 or values[1] < values[2]:
            raise ValueError("invalid gripper command")
        commanded = measured.copy()
        commanded[fingers] = values[2] + values[0] * (values[1] - values[2])
        configurations.append(("gripper_command", commanded))
    aperture = float(_array(arm.gripper_open, (), "open aperture"))
    if not 0 <= aperture <= .04:
        raise ValueError("unsupported Franka open aperture")

    def states():
        for label, configuration in configurations:
            for waypoint, joints in enumerate(path):
                full = configuration.copy()
                full[move] = joints
                yield label, waypoint, full
            # Check the entire intended opening at final arm pose, with at
            # most 2 mm change per finger. The held endpoint is checked above.
            count = max(1, int(np.ceil(np.max(np.abs(configuration[fingers] - aperture)) / .002)))
            for alpha in np.linspace(0., 1., count + 1)[1:]:
                full = configuration.copy()
                full[move] = path[-1]
                full[fingers] = (1. - alpha) * configuration[fingers] + alpha * aperture
                yield f"opening_{label}", len(path) - 1, full

    fixtures = [(_fixture(table_bounds, "take_release_table"), "table"),
                (_fixture(plate_bounds, "take_release_plate"), "plate")]
    fcl = robot.get_fcl_model()
    objects = list(fcl.get_collision_objects())
    link_names = [obj.name for obj in objects]
    if (len(set(link_names)) != len(link_names)
            or not _COLLISION_LINKS.issubset(link_names)
            or set(link_names) != set(fcl.get_collision_link_names())
            or any(not obj.shapes for obj in objects)):
        raise ValueError("incomplete robot collision geometry")
    # Exact root of the checked-in Panda URDF, not a prefix or link1 exemption.
    if list(pin.get_chain_joint_name("panda_link0")):
        raise ValueError("panda_link0 is not a fixed root")
    planned = list(world.get_planned_articulations())
    if len(planned) != 1 or planned[0].get_name() != robot.get_name():
        raise ValueError("unsupported planning-world articulation layout")
    saved = _array(robot.get_qpos(), (9,), "saved MPLib state")
    nearest = {"table": float("inf"), "plate": float("inf")}
    checked = 0
    opening_checked = 0
    try:
        for label, waypoint, full in states():
            if np.any(full < limits[:, 0] - .001) or np.any(full > limits[:, 1] + .001):
                raise ValueError(f"joint limit: {label} waypoint {waypoint}")
            robot.set_qpos(full, True)
            world.set_qpos_all(full[move])
            evaluated = _array(robot.get_qpos(), (9,), "MPLib evaluated state")
            if not np.allclose(evaluated, full, atol=1e-9, rtol=0):
                raise ValueError("planning world changed full/finger configuration")
            if world.is_state_colliding():
                raise ValueError(f"world collision: {label} waypoint {waypoint}")
            # Do not assume pybind's earlier FCLObject list has live poses.
            sample_objects = list(fcl.get_collision_objects())
            if (len(sample_objects) != len(objects)
                    or {obj.name for obj in sample_objects} != set(link_names)
                    or any(not obj.shapes for obj in sample_objects)):
                raise ValueError("collision geometry changed during preflight")
            for obj in sample_objects:
                for fixture, kind in fixtures:
                    if obj.name == "panda_link0" and kind == "table":
                        continue
                    gap = float(distance(obj, fixture).min_distance)
                    if not np.isfinite(gap) or gap < .002:
                        raise ValueError(f"{kind} clearance: {obj.name} {gap!r}m "
                                         f"{label} waypoint {waypoint}")
                    nearest[kind] = min(nearest[kind], gap)
            checked += 1
            opening_checked += label.startswith("opening_")
        pin.compute_forward_kinematics(full)
        endpoint_pose = pin.get_link_pose(backend.link_name_2_idx[backend.move_group])
        actual = _pose7(endpoint_pose, "FK")
        goal = _pose7(robot.get_base_pose().inv() * planner._to_mplib_pose(target), "goal")
        error = float(np.linalg.norm(actual[:3] - goal[:3]))
        angle = float(2 * np.arccos(np.clip(abs(np.dot(actual[3:], goal[3:])), 0., 1.)))
        if not np.isfinite(error + angle) or error > .01 or angle > .05:
            raise ValueError(f"FK endpoint: position={error}m orientation={angle}rad")
        endpoint_world = _pose7(robot.get_base_pose() * endpoint_pose, "world FK")
        return {"checked_states": checked, "opening_checked_states": opening_checked,
                "finger_states": [label for label, _ in configurations],
                "min_fixture_distance_m": nearest, "max_joint_step_rad": float(np.max(steps)),
                "joint_travel_rad": travel.tolist(), "joint_excursion_rad": excursion.tolist(),
                "endpoint_position_error_m": error, "endpoint_orientation_error_rad": angle,
                "endpoint_world_link_pose7": endpoint_world.tolist(),
                "endpoint_world_link_error_xyz_m": (endpoint_world[:3] - target[:3]).tolist(),
                "table_exempt_link": "panda_link0", "continuous_collision_checked": False}
    finally:
        # Also refresh attached-body poses through the world. No Genesis setters.
        try:
            robot.set_qpos(saved, True)
            world.set_qpos_all(saved[move])
        finally:
            try:
                robot.set_qpos(saved, True)
            finally:
                pin.compute_forward_kinematics(saved)
