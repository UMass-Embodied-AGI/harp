"""Success guards: an object counts as placed only while nobody holds it.

Placement success checks that test "object xy inside the container and above
the table" also accept an object that the eval human (or the gripper) is
holding over the container.  These helpers let a task's success predicate
require the object to rest there instead.
"""

from __future__ import annotations

import numpy as np

from ..interaction.interrupt_progress import robot_grasping, robot_tcp
from ..utils import to_numpy

DEFAULT_HELD_TCP_RADIUS_M = 0.07
# Open fingers still inside the object (review clip 08: open fingers inside
# the top bowl at the end of a "successful" stack) count as holding it: any
# finger point within this of the object's live AABB.
DEFAULT_FINGER_AABB_MARGIN_M = 0.01
_FINGER_LINK_KEYS = ("finger", "pad")


def _entity(obj):
    return getattr(obj, "entity", obj)


def avatar_holds(task, obj) -> bool:
    """True when the avatar has ``obj`` attached to one or both hands."""
    rig = getattr(getattr(task, "avatar", None), "robot", None)
    if rig is None:
        return False
    entity = _entity(obj)
    attachments = list(getattr(rig, "attached_object", None) or [])
    attachments.append(getattr(rig, "two_hand_attached_object", None))
    return any(item is not None and item[0] is entity for item in attachments)


def _entity_aabb(obj):
    try:
        aabb = np.asarray(to_numpy(_entity(obj).get_AABB()), dtype=float).reshape(-1, 2, 3)[0]
    except Exception:
        return None
    return aabb if np.all(np.isfinite(aabb)) else None


def finger_points(task, arm_tag: str = "right"):
    """World points along each finger link (root, middle, tip), or None.

    The finger links' origins sit at the finger roots; the tip of each finger
    is its root moved along the gripper's approach axis to the TCP level
    (TCP minus the mean of the roots), which holds for parallel grippers."""
    try:
        arm = task.robot.get_arm(arm_tag)
        links = [lk for lk in arm.entity.links
                 if any(k in str(lk.name).lower() for k in _FINGER_LINK_KEYS)]
        if not links:
            return None
        roots = np.array([np.asarray(to_numpy(lk.get_pos()), dtype=float).ravel()[:3]
                          for lk in links])
        tcp = robot_tcp(task, arm_tag)
    except Exception:
        return None
    axis = tcp - roots.mean(axis=0)
    return np.concatenate([roots, roots + 0.5 * axis, roots + axis])


def fingers_in_object(task, obj, *, margin_m: float = DEFAULT_FINGER_AABB_MARGIN_M,
                      arm_tag: str = "right") -> bool:
    """True when any finger point lies inside ``obj``'s live AABB grown by
    ``margin_m`` (open or closed gripper)."""
    aabb = _entity_aabb(obj)
    pts = finger_points(task, arm_tag)
    if aabb is None or pts is None:
        return False
    lo, hi = aabb[0] - float(margin_m), aabb[1] + float(margin_m)
    return bool(np.any(np.all((pts >= lo) & (pts <= hi), axis=1)))


def robot_holds(task, obj, center, *, radius_m: float = DEFAULT_HELD_TCP_RADIUS_M) -> bool:
    """True when ``obj`` is kinematically attached to a gripper, a closed
    gripper's TCP is within ``radius_m`` of ``center``, or (open or closed)
    the TCP is within ``radius_m`` and a finger is inside / within
    DEFAULT_FINGER_AABB_MARGIN_M of the object's live AABB."""
    entity = _entity(obj)
    for item in (getattr(task, "_gripper_attached", None) or {}).values():
        if item and item[0] is entity:
            return True
    if center is None:
        return False
    center = np.asarray(center, dtype=float)
    if robot_grasping(task, center, radius_m=radius_m):
        return True
    try:
        tcp = robot_tcp(task, "right")
    except Exception:
        return False
    if float(np.linalg.norm(tcp - center.ravel()[:3])) >= float(radius_m):
        return False
    return fingers_in_object(task, obj)


def entity_bottom_z(obj, fallback_z: float) -> float:
    """Lowest AABB z of ``obj``; ``fallback_z`` when no AABB is available."""
    try:
        aabb = np.asarray(to_numpy(_entity(obj).get_AABB()), dtype=float)
        if aabb.shape == (2, 3) and np.all(np.isfinite(aabb)):
            return float(aabb[0, 2])
    except Exception:
        pass
    return float(fallback_z)


def held_by_anyone(task, obj, center, *, radius_m: float = DEFAULT_HELD_TCP_RADIUS_M) -> bool:
    """``avatar_holds`` or ``robot_holds``: someone still has ``obj``."""
    return bool(avatar_holds(task, obj) or robot_holds(task, obj, center, radius_m=radius_m))
