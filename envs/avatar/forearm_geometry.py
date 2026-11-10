"""Conservative arm envelopes derived from the avatar's rendered skin.

Capsules remain the broad phase. A convex hull of every forearm-owned skin
triangle (including its mixed-weight boundary vertices) can reject capsule
overlaps in empty space or neighboring hand space without cutting wrist skin
off at an assumed joint plane. Missing/invalid skin data retains the capsule.
Upper arms include all spatially overlapping skin triangles, regardless of
bone ownership, so shoulder and elbow skin cannot be erased by refinement.
"""
import re
import warnings

import numpy as np
from scipy.spatial import ConvexHull

from ..utils import to_numpy


def _family(name):
    name = str(name).split(":")[-1]
    for hand in ("LeftHand", "RightHand"):
        if name.startswith(hand):
            return hand
    return re.sub(r"\d+$", "", name)


def _validated_indices(values, upper_bound, *, ndim, label):
    """Reject malformed indices before a cast can truncate or wrap them."""
    raw = np.asarray(values)
    if (raw.ndim != ndim or raw.size == 0 or raw.dtype.kind not in "iuf"
            or not np.all(np.isfinite(raw)) or np.any(raw < 0)
            or np.any(raw >= upper_bound)
            or (raw.dtype.kind == "f" and np.any(raw != np.floor(raw)))):
        raise ValueError(f"Invalid avatar skin {label} indices")
    return raw.astype(np.int64, copy=False)


def forearm_vertex_indices(joint_names, joint_indices, weights, faces, family):
    """Include complete triangles incident to forearm-dominant vertices.

    Twist joints belong to their forearm family. Dominance is compared with
    every other joint family, not just with the hand. Including whole mixed
    triangles preserves the skin seam rather than fitting only its centers.
    """
    names = np.asarray([_family(name) for name in joint_names])
    indices = _validated_indices(joint_indices, len(names), ndim=2, label="joint")
    weights = np.asarray(weights, dtype=np.float64)
    faces = _validated_indices(faces, len(indices), ndim=2, label="triangle")
    if (weights.shape != indices.shape
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or faces.shape[1] != 3):
        raise ValueError("Invalid avatar skin weights/triangle indexing")
    vertex_families = names[indices]
    target_weight = np.sum(weights * (vertex_families == family), axis=1)
    dominant = target_weight > 0
    for other in np.unique(names):
        if other != family:
            dominant &= target_weight >= np.sum(weights * (vertex_families == other), axis=1)
    incident = faces[np.any(dominant[faces], axis=1)]
    if not len(incident):
        raise ValueError(f"No rendered skin triangles attributed to {family}")
    return np.unique(incident)


def convex_halfspaces(vertices):
    """Unit-normal halfspaces containing all supplied surface vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Invalid forearm surface vertices")
    equations = np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
    if not np.all(np.isfinite(equations)):
        raise ValueError("Nonfinite forearm hull")
    equations /= np.linalg.norm(equations[:, :3], axis=1)[:, None]
    # Cover floating-point hull residuals without changing a physical margin.
    equations[:, 3] -= 1e-9
    if np.max(vertices @ equations[:, :3].T + equations[:, 3]) > 1e-8:
        raise ValueError("Forearm hull does not contain its source surface")
    return equations


class ForearmMeshRefiner:
    """Lazy skin/pose cache; failure conservatively returns no refinement."""

    def __init__(self, skin):
        self.skin = skin
        self._indices = {}
        self._vertices = None
        self._hulls = {}
        self._spatial_vertices = None
        self._spatial_hulls = {}
        self._warned = False

    def capsule_halfspaces(self, start, end, radius, margin=0.0):
        """Contain all skin that could contact an old capsule candidate.

        Any triangle intersecting the capsule's AABB has an overlapping
        triangle AABB. Include every vertex of each such complete triangle,
        not just vertices inside the box or belonging to a particular bone.
        For positive margins the old candidate region extends by ``margin``;
        possible skin within that distance extends by another ``margin``.
        The hull is conservative even at joints or for cross-body poses.
        """
        vertices = getattr(self.skin, "_latest_render_vverts", None)
        if vertices is None:
            return None
        try:
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            radius, margin = float(radius), float(margin)
            if (start.shape != (3,) or end.shape != (3,)
                    or not np.all(np.isfinite([start, end]))
                    or not np.isfinite(radius) or radius <= 0
                    or not np.isfinite(margin)):
                raise ValueError("Invalid capsule bounds")
            if vertices is not self._spatial_vertices:
                surface = np.asarray(vertices, dtype=np.float64)
                if (surface.ndim != 2 or surface.shape[1] != 3
                        or not np.all(np.isfinite(surface))):
                    raise ValueError("Invalid rendered skin vertices")
                faces = _validated_indices(
                    self.skin._entity._nyx_custom_vfaces, len(surface),
                    ndim=2, label="triangle",
                )
                if faces.shape[1] != 3:
                    raise ValueError("Invalid rendered skin triangles")
                triangles = surface[faces]
                self._spatial_surface = surface
                self._spatial_faces = faces
                self._triangle_lo = triangles.min(axis=1)
                self._triangle_hi = triangles.max(axis=1)
                self._spatial_hulls = {}
                self._spatial_vertices = vertices
            extent = radius + 2 * max(margin, 0.0)
            key = (tuple(start), tuple(end), extent)
            if key not in self._spatial_hulls:
                lower = np.minimum(start, end) - extent
                upper = np.maximum(start, end) + extent
                overlap = (np.all(self._triangle_hi >= lower, axis=1)
                           & np.all(self._triangle_lo <= upper, axis=1))
                indices = np.unique(self._spatial_faces[overlap])
                if len(indices) == 0:
                    raise ValueError("No skin triangles near capsule")
                self._spatial_hulls[key] = convex_halfspaces(np.vstack([
                    self._spatial_surface[indices], start, end,
                ]))
            return self._spatial_hulls[key]
        except Exception as exc:
            if not self._warned:
                warnings.warn(f"Arm mesh unavailable; retaining capsule: {exc}", RuntimeWarning)
                self._warned = True
            return None

    def halfspaces(self, family, start, end):
        skin = self.skin
        vertices = getattr(skin, "_latest_render_vverts", None)
        if vertices is None:
            return None
        try:
            if family not in self._indices:
                nodes = skin._nodes[1]
                palette = _validated_indices(
                    to_numpy(skin._skin_joints), len(nodes), ndim=1, label="palette",
                )
                joint_names = [nodes[index].name for index in palette]
                faces = skin._entity._nyx_custom_vfaces
                indices = forearm_vertex_indices(
                    joint_names, to_numpy(skin._vert_joints_cuda),
                    to_numpy(skin._vert_weights_cuda), faces, family,
                )
                if len(vertices) != len(skin._vert_joints_cuda):
                    raise ValueError("Rendered skin and skinning weights are misaligned")
                self._indices[family] = indices
            # KinematicAvatarSkin.update_mesh replaces this world-space array
            # for every pose. Never reuse a hull across different arrays.
            if vertices is not self._vertices:
                self._vertices = vertices
                self._hulls = {}
            key = (family, tuple(start), tuple(end))
            if key not in self._hulls:
                surface = np.asarray(vertices, dtype=np.float64)[self._indices[family]]
                self._hulls[key] = convex_halfspaces(np.vstack([surface, start, end]))
            return self._hulls[key]
        except Exception as exc:
            # A failed/unsupported mesh must never erase a capsule contact.
            if not self._warned:
                warnings.warn(f"Forearm mesh unavailable; retaining capsule: {exc}", RuntimeWarning)
                self._warned = True
            return None
