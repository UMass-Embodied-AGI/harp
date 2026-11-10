"""Robot ↔ avatar collision detection with analytic proxies and skin refinement.

Why analytic (not Genesis's `detect_collision()`): Genesis does not expose
a "query-only / phantom collider" mode — any geom with `collision=True`
is fully in the physics world and will push the robot.  To detect
collisions *without* perturbing the physics we stay out of the collision
world entirely and do the distance test in Python.

Algorithm per call (`check()`):
  1. For every robot collision geom, read its current world-space mesh
     vertices via `geom.get_verts()`.  A pre-chosen subsample (~30 pts
     per geom by default) keeps the work bounded.
  2. For every avatar capsule `(A, B, radius)` and oriented box
     `(center, axes, half_extents)`, compute each point's signed
     distance. Rigs with validated joint boundaries may explicitly
     request flat distal ends for selected segments. Default arms refine
     candidate overlaps against conservative rendered-skin hulls.
  3. A pair `(robot_link, capsule)` is in collision iff its minimum
     signed distance is ≤ `margin` (default 0). Configured allowed
     contacts are kept separate and cannot mask a forbidden contact.

Proxy distances are vectorized via NumPy, ~O(N_robot_pts × N_capsules).
Arm skin hulls are built lazily for candidate overlaps and cached per
avatar pose; they do not alter the physical simulation.

Typical usage:

    from envs.avatar import AvatarCollider, AvatarCollisionChecker
    self.collider = AvatarCollider(scene, self.avatar.robot)
    # ... scene.build(), avatar.reset() ...
    self.collision_checker = AvatarCollisionChecker(
        self.collider, self.robot.get_arm("right").entity
    )

    # Each sim step:
    self.collider.update()
    result = self.collision_checker.check()
    if result["collided"]:
        print("hit", result["pairs"])  # [(link, capsule, depth_m), ...]
"""
import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def sample_capsule_planner_points(
    capsules, *, inflate_factor: float
) -> np.ndarray | None:
    """Sample ordered planner points from world-space avatar capsules."""
    pts = []
    for _name, pa, pb, r in capsules:
        pa = np.asarray(pa, dtype=np.float64)
        pb = np.asarray(pb, dtype=np.float64)
        seg = pb - pa
        length = float(np.linalg.norm(seg))
        r_inflated = float(r) * inflate_factor
        if length < 1e-6:
            pts.append(pa)
            continue
        axis = seg / length
        tmp = (
            np.array([0.0, 0.0, 1.0])
            if abs(axis[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        u = np.cross(axis, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(axis, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        n_axis = max(2, int(length / 0.04) + 1)
        for t in np.linspace(0.0, 1.0, n_axis):
            center = pa + t * seg
            pts.append(center)
            for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                pts.append(
                    center
                    + r_inflated * (np.cos(theta) * u + np.sin(theta) * v)
                )
    return np.asarray(pts, dtype=np.float64) if pts else None


def _capsule_distances(points, start, end, radius, *, flat_distal=False):
    """Signed distances to a capsule, optionally ending flat at ``end``.

    The flat end keeps the proximal round cap and the full shaft radius.
    This models a segment ending at an anatomical joint without assigning
    the neighboring body part to its distal hemispherical cap. Unlike just
    masking points past the joint, the distance remains meaningful when a
    caller requests a positive near-contact margin.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return np.linalg.norm(points - start, axis=1) - float(radius)
    axis = segment / length
    relative = points - start
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
    if not flat_distal:
        closest = start + np.clip(axial, 0.0, length)[:, None] * axis
        return np.linalg.norm(points - closest, axis=1) - float(radius)

    shaft_distance = radial - float(radius)
    proximal_distance = np.sqrt(radial ** 2 + axial ** 2) - float(radius)
    before_end = np.maximum(
        np.where(axial < 0.0, proximal_distance, shaft_distance), axial - length,
    )
    past_end = np.sqrt(np.maximum(shaft_distance, 0.0) ** 2 + (axial - length) ** 2)
    return np.where(axial > length, past_end, before_end)


class AvatarCollisionChecker:
    """Analytic point-vs-capsule collision checker.

    Parameters
    ----------
    avatar_collider : AvatarCollider
        Source of capsules (`current_capsules()` returns (name, A, B, r)).
    robot_entity : genesis RigidEntity OR iterable of RigidEntity
        Robot(s) to check.  Each entity's collision geoms contribute
        sample points.  Pass a list to cover multiple arms.
    n_points_per_geom : int
        Target subsample count per collision geom.  The module takes a
        stride over `init_verts` so small geoms keep more coverage and
        large geoms don't explode the point count.
    margin : float
        Distance threshold for "in collision" (metres).  Set `0.0` for
        strict overlap, positive to flag near-misses (e.g. 0.01 → within
        1 cm).  Negative to allow small interpenetration without
        tripping (unusual).
    """

    def __init__(
        self,
        avatar_collider,
        robot_entity,
        n_points_per_geom: int = 30,
        margin: float = 0.0,
        ignored_capsules=(),
        ignored_capsule_prefixes=(),
    ):
        self.collider = avatar_collider
        self.margin = float(margin)
        self.ignored_capsules = {str(x) for x in (ignored_capsules or ())}
        self.ignored_capsule_prefixes = tuple(
            str(x) for x in (ignored_capsule_prefixes or ())
        )

        entities = robot_entity if isinstance(robot_entity, (list, tuple)) \
            else [robot_entity]

        # [(link_name, geom, numpy-int32 sub-indices)] — indices applied
        # to get_verts() output each step.
        self._geom_refs = []
        for ent in entities:
            for link in ent.links:
                link_geoms = getattr(link, "_geoms", None) or []
                for g in link_geoms:
                    try:
                        n_v = int(g.init_verts.shape[0])
                    except Exception:
                        continue
                    if n_v == 0:
                        continue
                    step = max(1, n_v // n_points_per_geom)
                    idxs = np.arange(0, n_v, step, dtype=np.int64)
                    # Always include the last vertex (covers extremes).
                    if idxs[-1] != n_v - 1:
                        idxs = np.append(idxs, n_v - 1)
                    self._geom_refs.append((link.name, g, idxs))

    def _is_ignored_capsule(self, name: str) -> bool:
        if name in self.ignored_capsules:
            return True
        return any(name.startswith(prefix) for prefix in self.ignored_capsule_prefixes)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _collect_points(self):
        """Stack world-space sample points across all robot geoms.

        Returns (M, 3) points and a length-M array of originating link
        names (used to build the per-link collision report).
        """
        pts_parts = []
        link_parts = []
        for link_name, geom, idxs in self._geom_refs:
            verts = _to_numpy(geom.get_verts())[idxs]
            pts_parts.append(verts)
            link_parts.extend([link_name] * len(verts))
        if not pts_parts:
            return np.zeros((0, 3)), []
        return np.concatenate(pts_parts, axis=0), link_parts

    def check(self) -> dict:
        """Run one collision query.

        Returns
        -------
        dict with keys
            collided      : bool — any signed-distance ≤ margin
            min_distance  : float — minimum signed distance over all
                            (point, capsule) pairs.  Negative means
                            interpenetration depth; positive means
                            closest approach.
            pairs         : list of (link_name, capsule_name, depth) for
                            every (link, capsule) pair where at least
                            one sample point is within `margin`.  Depth
                            is the most-negative signed distance found
                            for that pair.
            closest_pair  : (link, capsule, depth) for the single
                            deepest / closest point (always populated
                            when there are points and capsules).
        """
        capsules = self.collider.current_capsules() if self.collider else []
        boxes = []
        if self.collider is not None and hasattr(self.collider, "current_boxes"):
            boxes = self.collider.current_boxes()
        pts, link_names = self._collect_points()
        if (not capsules and not boxes) or len(pts) == 0:
            return {"collided": False, "min_distance": float("inf"),
                    "scored_min_distance": float("inf"), "scored_closest_pair": None,
                    "pairs": [], "closest_pair": None}

        # For each point, track (min_signed_distance, capsule_name).
        min_d = np.full(len(pts), np.inf)
        which_cap = np.array(["" for _ in range(len(pts))], dtype=object)
        scored_d = np.full(len(pts), np.inf)
        scored_cap = np.array(["" for _ in range(len(pts))], dtype=object)
        ignored_d = np.full(len(pts), np.inf)
        ignored_cap = np.array(["" for _ in range(len(pts))], dtype=object)

        def include_proxy(name, distances):
            # An allowed hand contact must not hide a simultaneous forearm
            # contact merely because the hand proxy is the deeper overlap.
            mask = distances < min_d
            min_d[mask] = distances[mask]
            which_cap[mask] = name
            if self._is_ignored_capsule(name):
                mask = distances < ignored_d
                ignored_d[mask] = distances[mask]
                ignored_cap[mask] = name
            else:
                mask = distances < scored_d
                scored_d[mask] = distances[mask]
                scored_cap[mask] = name

        flat_distal = getattr(self.collider, "distal_clipped_capsules", ())

        for cname, A, B, r in capsules:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
            AB = B - A
            ab_sq = float(np.dot(AB, AB))
            if ab_sq < 1e-12:
                continue
            dist = _capsule_distances(pts, A, B, r, flat_distal=cname in flat_distal)
            candidates = dist <= self.margin
            refine = getattr(self.collider, "forearm_mesh_halfspaces", None)
            spatial_refine = getattr(self.collider, "capsule_mesh_halfspaces", None)
            if np.any(candidates) and (spatial_refine is not None or refine is not None):
                equations = (spatial_refine(cname, A, B, r, self.margin)
                             if spatial_refine is not None else refine(cname, A, B))
                if equations is not None:
                    # Max unit-plane residual is a conservative distance
                    # bound (exact containment sign). Positive contact
                    # margins therefore remain conservative at corners.
                    residual = np.max(pts[candidates] @ equations[:, :3].T + equations[:, 3], axis=1)
                    dist[candidates] = np.maximum(dist[candidates], residual)
            include_proxy(cname, dist)

        for bname, center, axes, half_extents in boxes:
            center = np.asarray(center, dtype=np.float64)
            axes = np.asarray(axes, dtype=np.float64)
            half_extents = np.asarray(half_extents, dtype=np.float64)
            local = (pts - center) @ axes
            q = np.abs(local) - half_extents[None, :]
            outside = np.maximum(q, 0.0)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside_dist = np.minimum(np.max(q, axis=1), 0.0)
            dist = outside_dist + inside_dist
            include_proxy(bname, dist)

        # Build per-pair (link, capsule) min-depth report for pts that
        # breach `margin`.
        def pairs_for(distances, names):
            pair_depth = {}
            for i in np.flatnonzero(distances <= self.margin):
                key = (link_names[i], str(names[i]))
                depth = float(distances[i])
                if key not in pair_depth or depth < pair_depth[key]:
                    pair_depth[key] = depth
            return sorted(((link, name, depth) for (link, name), depth in pair_depth.items()),
                          key=lambda item: item[2])

        pairs = pairs_for(scored_d, scored_cap)
        ignored_pairs = pairs_for(ignored_d, ignored_cap)

        arg_closest = int(np.argmin(min_d))
        closest_pair = pairs[0] if pairs else (
            link_names[arg_closest],
            str(which_cap[arg_closest]),
            float(min_d[arg_closest]),
        )

        # True minimum over scored (non-ignored) shapes, whether or not in
        # contact.  `min_distance` keeps its legacy meaning for existing
        # consumers; envs/safety_metrics.py reads the scored fields.
        if np.isfinite(scored_d).any():
            arg_scored = int(np.argmin(scored_d))
            scored_min = float(scored_d[arg_scored])
            scored_closest = (link_names[arg_scored], str(scored_cap[arg_scored]), scored_min)
        else:
            scored_min, scored_closest = float("inf"), None

        return {
            "collided": bool(pairs),
            "min_distance": float(pairs[0][2] if pairs else min_d.min()),
            "scored_min_distance": scored_min,
            "scored_closest_pair": scored_closest,
            "pairs": pairs,
            "ignored_pairs": ignored_pairs,
            "closest_pair": closest_pair,
        }
