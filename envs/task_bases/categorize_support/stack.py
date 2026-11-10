"""The cooperative milk-only, physically supported stacking success rule.

The ordinary full-footprint rule still applies to every other placement.
Stacking may overhang the rim, but must rest on a correctly sorted carton;
mere proximity, a rim-only perch, or a held carton is not sufficient.
"""

import numpy as np

from ...contact import entity_contacts
from ...utils import to_numpy


SCORING_VERSION = "cooperative_milk_stack_v1"
MILK_MODEL = ("038_milk-box", 0)
MAX_LINEAR_SPEED = 0.03  # m/s, both cartons must be nearly at rest
MAX_ANGULAR_SPEED = 0.5  # rad/s
MIN_UPWARD_SUPPORT_FORCE = 0.01  # N; not just a zero-force collision pair
FLOOR_TOLERANCE = 0.005  # m; mesh/contact penetration, not a second elevated stack


def _speed(entity, getter):
    velocity = np.asarray(to_numpy(getattr(entity, getter)()), dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError(f"invalid {getter} observation")
    return float(np.linalg.norm(velocity))


def _in_entity(indices, entity):
    start, end = int(entity.geom_start), int(entity.geom_end)
    if start < 0 or end <= start:
        raise ValueError("invalid entity geometry range")
    return (indices >= start) & (indices < end)


def _support_evidence(task, upper, lower, basket_idx):
    """Read the last physics step; never advance or modify the simulation."""
    result = {"success": False, "reason": "support_unavailable"}
    try:
        entities = (upper.entity, lower.entity)
        bounds = np.asarray(to_numpy(entities[1].get_AABB()), dtype=float)
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                or np.any(bounds[0] >= bounds[1])):
            raise ValueError("invalid lower carton bounds")
        floor = float(task.basket_poses[basket_idx].p[2] - task._BASKET_HALF_Z + .010)
        result["lower_floor_gap_m"] = float(bounds[0, 2] - floor)
        if abs(result["lower_floor_gap_m"]) > FLOOR_TOLERANCE:
            return dict(result, reason="lower_not_on_floor")
        avatar = task.avatar
        if avatar is not None:
            rig = avatar.robot
            attachments = [*rig.attached_object, rig.two_hand_attached_object]
            if any(item is not None and any(item[0] is e for e in entities)
                   for item in attachments):
                return dict(result, reason="avatar_attached")

        result["linear_speeds_m_s"] = [_speed(e, "get_vel") for e in entities]
        result["angular_speeds_rad_s"] = [_speed(e, "get_ang") for e in entities]
        if (max(result["linear_speeds_m_s"]) > MAX_LINEAR_SPEED
                or max(result["angular_speeds_rad_s"]) > MAX_ANGULAR_SPEED):
            return dict(result, reason="not_settled")

        arms = [arm for arm in (task.robot.left_arm, task.robot.right_arm)
                if arm is not None]
        if not arms:
            raise ValueError("no robot entities available")
        contacts = [entity_contacts(e) for e in entities]
        for rows in contacts:
            for arm in arms:
                if np.any(_in_entity(rows["geom_a"], arm.entity)
                          | _in_entity(rows["geom_b"], arm.entity)):
                    return dict(result, reason="robot_contact")

        rows = contacts[0]
        a_upper = _in_entity(rows["geom_a"], entities[0])
        b_upper = _in_entity(rows["geom_b"], entities[0])
        a_lower = _in_entity(rows["geom_a"], entities[1])
        b_lower = _in_entity(rows["geom_b"], entities[1])
        support_z = float(
            rows["force_a"][a_upper & b_lower, 2].sum()
            + rows["force_b"][b_upper & a_lower, 2].sum()
        )
        result["upward_support_force_n"] = support_z
        supported = bool(np.isfinite(support_z) and support_z > MIN_UPWARD_SUPPORT_FORCE)
        return dict(result, success=supported,
                    reason="supported" if supported else "no_upward_milk_support")
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        # Missing contact/velocity/attachment evidence cannot certify a stack.
        # Keep the ordinary footprint result; expose the diagnostic explicitly.
        return dict(result, error=f"{type(error).__name__}: {error}")


def add_supported_milk_stacks(task, metrics):
    """Extend ordinary metrics in place, preserving the measured footprint.

    Only the cooperative facade calls this. The support carton must pass the
    ORIGINAL center + footprint rule, so there are no circular stack passes.
    Both actors must belong to the same assigned basket. The existing center
    tolerance still bounds the upper carton; no enlarged basket is invented.
    """
    metrics["basket_scoring_version"] = SCORING_VERSION
    rows = metrics["basket_objects"]
    targets = {
        (side, name, int(model), int(basket)): actor
        for side, group in task._success_target_groups()
        for actor, name, model, basket in group
    }

    def key(row):
        return row["side"], row["object"], row["model_id"], row["basket_idx"]

    def finite_position(row):
        position = np.asarray(row["object_pos"], dtype=float)
        return position.shape == (3,) and bool(np.isfinite(position).all())

    for row in rows:
        if (row["object"], row["model_id"]) != MILK_MODEL:
            continue
        row["stack_success"] = False
        if row["success"]:
            continue
        if not finite_position(row):
            row["stack_support"] = {"success": False, "reason": "invalid_position"}
            continue
        if not row["center_success"]:
            row["stack_support"] = {"success": False, "reason": "center_outside"}
            continue
        row["stack_support"] = {"success": False, "reason": "no_valid_lower_carton"}
        for support in rows:
            if (support is row
                    or (support["object"], support["model_id"]) != MILK_MODEL
                    or support["basket_idx"] != row["basket_idx"]
                    or not finite_position(support)
                    or not (support["center_success"] and support["footprint_success"])
                    or not row["object_pos"][2] > support["object_pos"][2]):
                continue
            evidence = _support_evidence(
                task, targets[key(row)], targets[key(support)], row["basket_idx"],
            )
            evidence["side"] = support["side"]
            row["stack_support"] = evidence
            if evidence["success"]:
                row["success"] = row["stack_success"] = True
                break
    metrics["basket_all_correct"] = all(row["success"] for row in rows)
    return metrics
