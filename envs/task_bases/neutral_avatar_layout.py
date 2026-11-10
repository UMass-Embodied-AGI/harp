"""Neutral-avatar layout solver: motion footprints, work-xy candidates,
side/region selection, forbidden regions, and clearance checks.
"""


from __future__ import annotations


import numpy as np
import transforms3d as t3d

from ..utils import (
    to_numpy,
)




class _NeutralAvatarLayoutMixin:
    """Split from NeutralAvatarTableWorkMixin (verbatim member move); composed back into it in neutral_avatar_table_work.py."""

    def _neutral_avatar_compute_motion_footprint(self, job: dict) -> dict:
        """Dry-run the neutral animation and return object-swept XY offsets.

        The returned AABB is relative to the first attached/picked object's
        target point, so it can be translated to candidate table positions.
        It includes held-object sweeps and static helper objects such as the
        bowl used by the put-objects-in-bowl motion.
        """
        base_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        base_rot = np.asarray(
            getattr(self, "_neutral_avatar_measure_body_rot", None)
            if getattr(self, "_neutral_avatar_measure_body_rot", None) is not None
            else self.avatar_init_rot,
            dtype=np.float64,
        ).copy()
        self.avatar.reset(base_pos, base_rot)
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(job["motion"])

        events_by_frame: dict[int, list[dict]] = {}
        for event in job["events"]:
            events_by_frame.setdefault(int(event["clip_frame"]), []).append(event)
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_frame = int(first_event["clip_frame"])
        first_hand = int(first_event["hand_id"])
        first_obj = str(first_event["object"])

        active: dict[str, tuple[int, float, dict]] = {}
        points = []
        region_samples = []
        arm_sweep = ({} if getattr(self, "BLOCK_NEUTRAL_LAYOUT_RECOVERY", False)
                     and job.get("kind") == "put_objects_in_bowl" else None)
        first_anchor = None
        first_attach_by_object: set[str] = set()
        max_frame = max(
            int(job["clip"].get("num_frames", 0)),
            max(events_by_frame) + 2 if events_by_frame else 0,
        )
        step = 0
        while not self.avatar.spare() and step <= max_frame + 2:
            self.scene.step()
            self.avatar.step()
            if arm_sweep is not None:
                self._neutral_can_collect_arm_sweep(arm_sweep)
            for event in events_by_frame.get(step, []):
                obj_name = str(event["object"])
                hand_id = int(event["hand_id"])
                radius = self._neutral_prop_radius(obj_name)
                if event["type"] in ("attach", "pick"):
                    active[obj_name] = (hand_id, radius, event)
                    p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                    points.append((p[:2], radius, obj_name))
                    region_samples.append((p[:2].copy(), radius, obj_name))
                    first_attach_by_object.add(obj_name)
                    if step == first_frame and hand_id == first_hand and obj_name == first_obj:
                        first_anchor = p[:2].copy()
                elif event["type"] in ("detach", "put"):
                    if job.get("kind") in ("sponge", "mouse"):
                        active_event = active.get(obj_name, (hand_id, radius, event))[2]
                        hold_event = (
                            event
                            if "hold_offset" in event or "hold_offset_local" in event
                            else active_event
                        )
                        p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, hold_event)
                        points.append((p[:2], radius, obj_name))
                        region_samples.append((p[:2].copy(), radius, obj_name))
                    elif (
                        job.get("kind") == "put_objects_in_bowl"
                        and bool((job.get("clip") or {}).get("reverse"))
                    ):
                        # Reverse take-objects clip: the prop is SET DOWN here,
                        # so its target landing spot on the table must clear the
                        # task regions — the mirror of the forward clip clearing
                        # the can INIT (pick) positions added above.
                        p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                        points.append((p[:2], radius, obj_name))
                        region_samples.append((p[:2].copy(), radius, obj_name))
                    active.pop(obj_name, None)
                elif event["type"] == "switch":
                    if obj_name in active:
                        active[obj_name] = (hand_id, radius, event)

            for obj_name, (hand_id, radius, event) in active.items():
                if job.get("kind") == "bottle":
                    continue
                if job.get("kind") == "put_objects_in_bowl" and obj_name in first_attach_by_object:
                    continue
                p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                points.append((p[:2], radius, obj_name))
                region_samples.append((p[:2].copy(), radius, obj_name))
            step += 1

        self.avatar.frame_ratio = old_ratio
        if first_anchor is None:
            first_anchor = np.asarray(points[0][0], dtype=np.float64) if points else np.zeros(2)

        mins = []
        maxs = []
        for xy, radius, _obj_name in points:
            offset = np.asarray(xy, dtype=np.float64) - first_anchor
            mins.append(offset - radius)
            maxs.append(offset + radius)

        bowl_center_offset = None
        if job["kind"] == "put_objects_in_bowl":
            bowl_event_type = self._neutral_bowl_event_type(job)
            put_offsets = []
            for event in job["events"]:
                if event["type"] != bowl_event_type:
                    continue
                # Use the event hold position as the center of the bowl target.
                frame = int(event["clip_frame"])
                hand_id = int(event["hand_id"])
                self.avatar.reset(base_pos, base_rot)
                self.avatar.frame_ratio = 1.0
                self.avatar.play_animation(job["motion"])
                for _ in range(frame + 1):
                    self.scene.step()
                    self.avatar.step()
                put_offsets.append(
                    self._neutral_avatar_event_anchor_pos(
                        job, hand_id, str(event["object"]), event
                    )[:2]
                    - first_anchor
                )
                self.avatar.frame_ratio = old_ratio
            if put_offsets:
                bowl_center = np.mean(np.asarray(put_offsets), axis=0)
                bowl_center_offset = np.asarray(bowl_center, dtype=np.float64).copy()
                bowl_abs = first_anchor + bowl_center
                mins.append(bowl_center - self.NEUTRAL_BOWL_HALF_XY)
                maxs.append(bowl_center + self.NEUTRAL_BOWL_HALF_XY)
                bowl_radius = float(np.linalg.norm(self.NEUTRAL_BOWL_HALF_XY))
                region_samples.append((bowl_abs, bowl_radius, "bowl"))

        self.avatar.frame_ratio = old_ratio

        if not mins:
            mins = [np.array([-0.08, -0.08], dtype=np.float64)]
            maxs = [np.array([0.08, 0.08], dtype=np.float64)]

        min_offset = np.min(np.asarray(mins), axis=0)
        max_offset = np.max(np.asarray(maxs), axis=0)
        footprint = {
            "motion": job["base_motion"],
            "kind": job["kind"],
            "min_offset": min_offset,
            "max_offset": max_offset,
            "size": max_offset - min_offset,
        }
        if bowl_center_offset is not None:
            footprint["bowl_center_offset"] = bowl_center_offset
        region_points = []
        for xy, radius, label in region_samples:
            offset = np.asarray(xy, dtype=np.float64).ravel()[:2] - first_anchor
            region_points.append({
                "offset": offset.tolist(),
                "radius": float(radius),
                "label": str(label),
            })
        footprint["region_points"] = region_points
        if arm_sweep is not None:
            footprint["can_base_sweep"] = [
                {"name": name, "radius": radius,
                 "vertices": self._neutral_xy_convex_hull(
                     np.asarray(samples) - first_anchor).tolist()}
                for name, (radius, samples) in sorted(arm_sweep.items())
            ]
        return footprint

    @staticmethod
    def _neutral_xy_convex_hull(points):
        """Convex envelope of measured XY endpoints (including degenerate sets)."""
        points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
        if len(points) <= 1:
            return np.asarray(points, dtype=float).reshape(-1, 2)
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        def chain(rows):
            out = []
            for p in rows:
                while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                    out.pop()
                out.append(p)
            return out
        return np.asarray(chain(points)[:-1] + chain(points[::-1])[:-1])

    def _neutral_can_collect_arm_sweep(self, sweep):
        """Read current arm geometry; never update collider or simulation here.

        Separate convex envelopes per segment conservatively include its whole
        measured motion. Palm boxes use their actual eight projected corners.
        Capsule radii are retained, including conservative round distal ends.
        """
        collider = getattr(self, "avatar_collider", None)
        if collider is None:
            raise RuntimeError("can layout requires measured avatar arm geometry")
        observed = set()
        def add(name, radius, points):
            points = np.asarray(points, dtype=float)
            if not np.isfinite(points).all() or not np.isfinite(radius) or radius < 0:
                raise RuntimeError("invalid can arm sweep geometry")
            old_radius, samples = sweep.setdefault(name, (float(radius), []))
            samples.extend(points[:, :2].copy())
            sweep[name] = (max(old_radius, float(radius)), samples)
            observed.add(name)
        for name, start, end, radius in collider.current_capsules():
            if name.startswith(("l_", "r_")) and not name.endswith(("thigh", "shin")):
                add(name, radius, [start, end])
        for name, center, axes, half_extents in collider.current_boxes():
            if name in ("l_palm", "r_palm"):
                signs = np.asarray([(x, y, z) for x in (-1, 1)
                                    for y in (-1, 1) for z in (-1, 1)])
                add(name, 0., np.asarray(center) +
                    (signs * np.asarray(half_extents)) @ np.asarray(axes).T)
        if not {"l_forearm", "r_forearm", "l_palm", "r_palm"} <= observed:
            raise RuntimeError("incomplete can arm sweep geometry")

    @staticmethod
    def _neutral_can_base_sweep_is_clear(xy, footprint, task_regions):
        """Conservative XY separation from existing robot-base disks only.

        This does not certify moving robot links or continuous avatar motion.
        Other object/table/side/body/gripper checks remain independent gates.
        """
        sweep = footprint.get("can_base_sweep")
        if sweep is None:
            # Non-can footprints and existing synthetic geometry callers.
            # Every production can measurement above supplies this field.
            return True
        if not sweep:
            return False
        anchor = np.asarray(xy, dtype=float).ravel()[:2]
        for center, base_radius, label in task_regions:
            if not str(label).startswith("robot_base:"):
                continue
            center = np.asarray(center, dtype=float).ravel()[:2] - anchor
            for item in sweep:
                vertices = np.asarray(item["vertices"], dtype=float)
                if len(vertices) == 0 or not np.isfinite(vertices).all():
                    return False
                edges = np.roll(vertices, -1, axis=0) - vertices
                delta = center - vertices
                length2 = np.sum(edges * edges, axis=1)
                t = np.divide(np.sum(delta * edges, axis=1), length2,
                              out=np.zeros_like(length2), where=length2 > 0)
                distances = np.linalg.norm(delta - np.clip(t, 0., 1.)[:, None] * edges, axis=1)
                cross = edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]
                inside = len(vertices) >= 3 and bool(np.all(cross >= 0))
                if inside or float(np.min(distances)) < float(base_radius) + float(item["radius"]):
                    return False
        return True

    def _neutral_avatar_work_candidates(self, job: dict) -> list[np.ndarray]:
        footprint = getattr(self, "_neutral_avatar_motion_footprint", None)
        motion_candidates = self._neutral_avatar_motion_work_candidates(job, footprint)
        if motion_candidates:
            return [np.asarray(xy, dtype=np.float64).ravel()[:2] for xy in motion_candidates]

        if footprint is None and job.get("kind") in ("mouse", "sponge"):
            side_aware = self._neutral_avatar_side_aware_work_candidates(job)
            if side_aware:
                return [np.asarray(xy, dtype=np.float64).ravel()[:2] for xy in side_aware]

        if job["kind"] == "sponge":
            attr = "NEUTRAL_SPONGE_WORK_CANDIDATES"
        elif job["kind"] == "mouse":
            attr = "NEUTRAL_MOUSE_WORK_CANDIDATES"
        else:
            attr = "NEUTRAL_WORK_CANDIDATES"
        raw = getattr(self, attr, None)
        raw_is_absolute_override = raw is not None
        if raw is None:
            if job["kind"] == "sponge":
                region_attr = "NEUTRAL_SPONGE_WORK_REGION"
            elif job["kind"] == "mouse":
                region_attr = "NEUTRAL_MOUSE_WORK_REGION"
            else:
                region_attr = "NEUTRAL_WORK_REGION"
            region = getattr(self, region_attr, None)
            if region is not None:
                arr = np.asarray(region, dtype=np.float64).ravel()
                if arr.size != 4:
                    raise ValueError(
                        f"{region_attr} must be (x_min, x_max, y_min, y_max); got {region!r}"
                    )
                x0, x1, y0, y1 = [float(v) for v in arr]
                n = int(self.config.get("neutral_avatar_region_candidates", 12))
                n = max(1, n)
                raw = [
                    np.array([
                        np.random.uniform(x0, x1),
                        np.random.uniform(y0, y1),
                    ], dtype=np.float64)
                    for _ in range(n)
                ]
                # Deterministic fallbacks keep selection robust when the
                # sampled points all clip a task object or table boundary.
                raw.extend([
                    np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64),
                    np.array([x0, (y0 + y1) * 0.5], dtype=np.float64),
                    np.array([x1, (y0 + y1) * 0.5], dtype=np.float64),
                ])
        if raw is None:
            base = self._neutral_avatar_default_work_xy(job)
            offsets = np.array([
                [0.00, 0.00],
                [0.00, 0.08],
                [0.00, -0.08],
                [-0.10, 0.00],
                [0.10, 0.00],
                [-0.10, 0.08],
                [0.10, 0.08],
            ], dtype=np.float64)
            raw = [base + off for off in offsets]
            raw_is_absolute_override = False
        shift = (
            self._neutral_avatar_work_candidate_shift_xy(job)
            if raw_is_absolute_override
            else np.zeros(2, dtype=np.float64)
        )
        return [np.asarray(xy, dtype=np.float64).ravel()[:2] + shift for xy in raw]

    def _neutral_avatar_allowed_work_regions(self, job: dict) -> list[tuple[float, float, float, float]]:
        """No task-specific side rectangles.

        Neutral side placement is shared across tasks: candidates come from
        the generic left/back/right table bands and are then filtered against
        task objects, robot regions, and the motion footprint.
        """
        return []

    def _neutral_avatar_allowed_side_work_regions(self, job: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Task-specific side rectangles are intentionally disabled."""
        return []

    def _neutral_avatar_blocked_sides(self, job: dict | None = None) -> set[str]:
        raw = self.config.get(
            "neutral_avatar_blocked_sides",
            getattr(self, "NEUTRAL_AVATAR_BLOCKED_SIDES", ()),
        )
        if isinstance(raw, str):
            raw = [raw]
        return {str(side) for side in (raw or [])}

    def _neutral_avatar_default_side_work_regions(
        self,
        job: dict | None = None,
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Generic left/back/right anchor bands for motions with no task override."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        xmin, xmax, ymin, ymax = [float(v) for v in (xmin, xmax, ymin, ymax)]
        x_span = max(0.01, xmax - xmin)
        y_span = max(0.01, ymax - ymin)
        side_w = min(0.25, 0.28 * x_span)
        back_h = min(0.22, 0.24 * y_span)
        y0 = ymin + 0.24 * y_span
        y1 = ymax - 0.12 * y_span
        x0 = xmin + 0.18 * x_span
        x1 = xmax - 0.18 * x_span
        if y0 >= y1:
            cy = 0.5 * (ymin + ymax)
            half = min(0.10, 0.18 * y_span)
            y0, y1 = cy - half, cy + half
        if x0 >= x1:
            cx = 0.5 * (xmin + xmax)
            half = min(0.12, 0.18 * x_span)
            x0, x1 = cx - half, cx + half
        edge_gap = 0.030
        side_regions = [
            ("right", (xmax - side_w, xmax - edge_gap, y0, y1)),
            ("back", (x0, x1, ymax - back_h, ymax - edge_gap)),
            ("left", (xmin + edge_gap, xmin + side_w, y0, y1)),
        ]
        blocked = self._neutral_avatar_blocked_sides(job)
        if blocked:
            side_regions = [
                (side, region)
                for side, region in side_regions
                if str(side) not in blocked
            ]
        return side_regions

    def _neutral_table_xy_bounds(self) -> tuple[float, float, float, float]:
        center = getattr(self, "TABLE_CENTER_XY", None)
        half = getattr(self, "TABLE_HALF_SIZE", None)
        if center is not None and half is not None:
            cx, cy = [float(v) for v in np.asarray(center, dtype=np.float64).ravel()[:2]]
            hx, hy = [float(v) for v in np.asarray(half, dtype=np.float64).ravel()[:2]]
            if hx > 0.0 and hy > 0.0:
                return (cx - hx, cx + hx, cy - hy, cy + hy)
        return tuple(float(v) for v in getattr(
            self, "NEUTRAL_TABLE_XY_BOUNDS", self.NEUTRAL_TABLE_XY_BOUNDS
        ))

    def _neutral_put_objects_plate_center_is_safe(
        self, xy: np.ndarray, footprint: dict | None
    ) -> bool:
        if footprint is None or "bowl_center_offset" not in footprint:
            return True
        center = (
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            + np.asarray(footprint["bowl_center_offset"], dtype=np.float64).ravel()[:2]
        )
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        margin_xy = (
            np.asarray(self.NEUTRAL_BOWL_HALF_XY, dtype=np.float64).ravel()[:2]
            + float(self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN)
        )
        return bool(
            float(xmin) + margin_xy[0] <= center[0] <= float(xmax) - margin_xy[0]
            and float(ymin) + margin_xy[1] <= center[1] <= float(ymax) - margin_xy[1]
        )

    def _neutral_put_objects_footprint_is_table_safe(
        self, xy: np.ndarray, footprint: dict | None
    ) -> bool:
        if footprint is None:
            return True
        min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
        max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + min_offset
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + max_offset
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        margin = float(
            getattr(
                self,
                "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
            )
        )
        return bool(
            float(xmin) + margin <= min_xy[0] <= max_xy[0] <= float(xmax) - margin
            and float(ymin) + margin <= min_xy[1] <= max_xy[1] <= float(ymax) - margin
        )

    def _neutral_filter_put_objects_candidates(
        self, candidates: list[np.ndarray], footprint: dict | None
    ) -> list[np.ndarray]:
        return [
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            for xy in candidates
            if self._neutral_put_objects_footprint_is_table_safe(xy, footprint)
        ]

    def _neutral_avatar_table_side_for_xy(self, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        return self._neutral_avatar_default_table_side_for_xy(arr)

    def _neutral_avatar_default_table_side_for_xy(self, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, _ymin, ymax = self._neutral_table_xy_bounds()
        dists = {
            "left": abs(float(arr[0]) - float(xmin)),
            "right": abs(float(xmax) - float(arr[0])),
            "back": abs(float(ymax) - float(arr[1])),
        }
        blocked = self._neutral_avatar_blocked_sides(
            getattr(self, "_neutral_avatar_job", None),
        )
        allowed = {side: distance for side, distance in dists.items()
                   if side not in blocked}
        return min(allowed, key=allowed.get) if allowed else ""

    def _neutral_avatar_side_for_work_xy(self, job: dict, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        key = (round(float(arr[0]), 6), round(float(arr[1]), 6))
        if key in getattr(self, "_neutral_avatar_default_side_candidates", set()):
            return self._neutral_avatar_default_table_side_for_xy(arr)
        side = getattr(self, "_neutral_avatar_candidate_sides", {}).get(
            key
        )
        if side:
            return str(side)
        side = self._neutral_avatar_table_side_for_xy(xy)
        if side:
            return side
        if job.get("kind") in ("put_objects_in_bowl", "mouse", "sponge"):
            return self._neutral_avatar_default_table_side_for_xy(xy)
        return ""

    def _neutral_avatar_motion_work_candidates(
        self,
        job: dict,
        footprint: dict | None,
    ) -> list[np.ndarray] | None:
        """Motion-level candidate generator.

        The default samples anchor points from task-provided allowed regions.
        Motion/task subclasses can override for clips whose body or prop sweep
        needs special treatment.
        """
        if job.get("kind") == "put_objects_in_bowl" and footprint is not None:
            candidate_sides: dict[tuple[float, float], str] = {}
            self._neutral_avatar_candidate_sides = candidate_sides
            self._neutral_avatar_default_side_candidates = set()

            def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
                arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
                out_list.append(arr)
                candidate_sides[
                    (round(float(arr[0]), 6), round(float(arr[1]), 6))
                ] = str(side_name)

            explicit_side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
            default_side_regions = list(self._neutral_avatar_default_side_work_regions())
            side_regions = list(explicit_side_regions)
            default_side_keys = set()
            if not side_regions:
                side_regions = list(default_side_regions)
                default_side_keys = {
                    (str(side), tuple(float(v) for v in region))
                    for side, region in default_side_regions
                }
            elif self._neutral_avatar_forced_motion_requested():
                known = {
                    (str(side), tuple(float(v) for v in region))
                    for side, region in side_regions
                }
                for side, region in default_side_regions:
                    key = (str(side), tuple(float(v) for v in region))
                    if key in known:
                        continue
                    side_regions.append((side, region))
                    default_side_keys.add(key)
            if side_regions:
                n = int(self.config.get("neutral_avatar_region_candidates", 16))
                n = max(1, n)
                out: list[np.ndarray] = []
                rects = []
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                margin = float(
                    getattr(
                        self,
                        "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                        self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
                    )
                )
                old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
                try:
                    for side, region in side_regions:
                        region_key = (str(side), tuple(float(v) for v in region))
                        x0, x1, y0, y1 = self._neutral_put_objects_side_region_for_sampling(
                            side, region
                        )
                        self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(side, job)
                        _hold = self._neutral_avatar_measure_hold_points(job)
                        side_fp = self._neutral_avatar_compute_motion_footprint(job)
                        min_offset = np.asarray(side_fp["min_offset"], dtype=np.float64).ravel()[:2]
                        max_offset = np.asarray(side_fp["max_offset"], dtype=np.float64).ravel()[:2]
                        table_lo = (
                            np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64)
                            - min_offset
                        )
                        table_hi = (
                            np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64)
                            - max_offset
                        )
                        lo = np.maximum(np.array([x0, y0], dtype=np.float64), table_lo)
                        hi = np.minimum(np.array([x1, y1], dtype=np.float64), table_hi)
                        if np.all(lo <= hi):
                            rects.append((lo, hi, side_fp, side, region_key in default_side_keys))
                finally:
                    self._neutral_avatar_measure_body_rot = old_measure_rot
                if rects:
                    for lo, hi, _side_fp, side, is_default_region in rects:
                        center = np.array([
                            (float(lo[0]) + float(hi[0])) * 0.5,
                            (float(lo[1]) + float(hi[1])) * 0.5,
                        ], dtype=np.float64)
                        add_candidate(out, side, center)
                        if is_default_region:
                            self._neutral_avatar_default_side_candidates.add((
                                round(float(center[0]), 6),
                                round(float(center[1]), 6),
                            ))
                    for _ in range(n):
                        lo, hi, _side_fp, side, is_default_region = rects[int(np.random.randint(0, len(rects)))]
                        candidate = np.array([
                            np.random.uniform(float(lo[0]), float(hi[0])),
                            np.random.uniform(float(lo[1]), float(hi[1])),
                        ], dtype=np.float64)
                        add_candidate(out, side, candidate)
                        if is_default_region:
                            self._neutral_avatar_default_side_candidates.add((
                                round(float(candidate[0]), 6),
                                round(float(candidate[1]), 6),
                            ))
                    return out
                if self.config.get("neutral_avatar_force_side"):
                    centers = []
                    for _side, region in side_regions:
                        x0, x1, y0, y1 = self._neutral_put_objects_side_region_for_sampling(
                            _side, region
                        )
                        center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                        add_candidate(centers, _side, center)
                    return centers or None

            xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
            margin = float(self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN)
            min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
            max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
            lo = np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64) - min_offset
            hi = np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64) - max_offset
            if np.all(lo <= hi):
                n = int(self.config.get("neutral_avatar_region_candidates", 16))
                n = max(1, n)
                out = [
                    np.array([
                        np.random.uniform(float(lo[0]), float(hi[0])),
                        np.random.uniform(float(lo[1]), float(hi[1])),
                    ], dtype=np.float64)
                    for _ in range(n)
                ]
                out.append((lo + hi) * 0.5)
                filtered = self._neutral_filter_put_objects_candidates(out, footprint)
                if filtered:
                    return filtered
                return out

        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(job)
        side_regions = list(explicit_side_regions)
        default_side_keys = set()
        if job.get("kind") in ("bottle", "mouse", "sponge"):
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key in {(str(s), tuple(float(v) for v in r)) for s, r in side_regions}:
                    continue
                default_side_keys.add(key)
                side_regions.append((side, region))
        if not side_regions:
            return None
        candidate_sides: dict[tuple[float, float], str] = {}
        self._neutral_avatar_candidate_sides = candidate_sides
        self._neutral_avatar_default_side_candidates = set()

        def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            out_list.append(arr)
            candidate_sides[
                (round(float(arr[0]), 6), round(float(arr[1]), 6))
            ] = str(side_name)

        default_n = 48 if job.get("kind") in ("bottle", "mouse", "sponge") else 16
        n = int(self.config.get("neutral_avatar_region_candidates", default_n))
        n = max(1, n)
        out: list[np.ndarray] = []
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        task_regions_for_anchor = self.neutral_task_regions()
        for side, region in side_regions:
            key = (str(side), tuple(float(v) for v in region))
            is_default_region = key in default_side_keys
            side_footprint = footprint
            if side_footprint is None:
                try:
                    self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(str(side), job)
                    _hold = self._neutral_avatar_measure_hold_points(job)
                    side_footprint = self._neutral_avatar_compute_motion_footprint(job)
                finally:
                    self._neutral_avatar_measure_body_rot = old_measure_rot
            if side_footprint is None:
                continue
            table_margin = self._neutral_motion_table_margin(side_footprint)
            min_offset = np.asarray(side_footprint["min_offset"], dtype=np.float64).ravel()[:2]
            max_offset = np.asarray(side_footprint["max_offset"], dtype=np.float64).ravel()[:2]
            lo = (
                np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                - min_offset
            )
            hi = (
                np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                - max_offset
            )
            if not np.all(lo <= hi):
                continue
            forbidden_rects = self._neutral_anchor_forbidden_rects(
                side_footprint, task_regions_for_anchor,
            )
            candidates = self._neutral_empty_anchor_rect_candidates(
                lo, hi, forbidden_rects,
            )
            accepted = 0
            for candidate in candidates:
                arr = np.asarray(candidate, dtype=np.float64).ravel()[:2]
                # Default candidates are later oriented by their nearest
                # allowed table side. Only spend this side's budget on
                # anchors whose eventual orientation matches the footprint
                # used above. Otherwise a few large cells on another side
                # can exhaust every side's quota, and identical coordinates
                # overwrite one another in candidate_sides.
                if (is_default_region
                        and self._neutral_avatar_default_table_side_for_xy(arr) != str(side)):
                    continue
                add_candidate(out, str(side), arr)
                if bool(is_default_region):
                    self._neutral_avatar_default_side_candidates.add((
                        round(float(arr[0]), 6),
                        round(float(arr[1]), 6),
                    ))
                accepted += 1
                if accepted >= n:
                    break
        return out

    def _neutral_avatar_use_layout_recovery(self, job: dict) -> bool:
        """Task-level opt-in to the existing measured-pose fallback search."""
        return bool(getattr(self, "BLOCK_NEUTRAL_LAYOUT_RECOVERY", False))

    def _neutral_avatar_fallback_work_candidates(self, job, pose_eval_cache, task_regions):
        """Lazily retry failed sponge/can layouts with measured nearby headings.

        Existing successful searches never enter this generator. Every yielded
        pose still goes through the caller's unchanged body, side, table and
        initial-prop checks, including the real gripper keep-out regions.
        """
        if (not self._neutral_avatar_use_layout_recovery(job)
                or job.get("kind") not in ("sponge", "put_objects_in_bowl")
                or self._neutral_avatar_allowed_side_work_regions(job)):
            return
        if not bool(self.config.get("neutral_use_region_placement", self.NEUTRAL_USE_REGION_PLACEMENT)):
            return
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        sides = self._neutral_avatar_default_side_work_regions(job)
        measured_can_poses = []
        # Denser anchors at the original heading come first. The bounded yaw
        # choices rotate the same full clip; no frame or prop is substituted.
        for degrees in getattr(self, "NEUTRAL_LAYOUT_RECOVERY_DEGREES", (0, -30, -15, 15, 30)):
            angle = np.deg2rad(degrees)
            c, s = float(np.cos(angle)), float(np.sin(angle))
            turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            for side, _region in sides:
                body_rot = turn @ self._neutral_avatar_body_rot_for_side(side, job)
                rot_key = tuple(np.round(body_rot.ravel(), 6))
                cached = pose_eval_cache.get(rot_key)
                if cached is None:
                    old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
                    try:
                        self._neutral_avatar_measure_body_rot = body_rot
                        hold = self._neutral_avatar_measure_hold_points(job)
                        footprint = self._neutral_avatar_compute_motion_footprint(job)
                        cached = (hold, footprint)
                        pose_eval_cache[rot_key] = cached
                    finally:
                        self._neutral_avatar_measure_body_rot = old_measure_rot
                _hold, footprint = cached
                if job.get("kind") == "put_objects_in_bowl":
                    measured_can_poses.append((side, _region, body_rot, footprint))
                margin = self._neutral_motion_table_margin(footprint)
                lo = np.array([xmin + margin, ymin + margin]) - footprint["min_offset"]
                hi = np.array([xmax - margin, ymax - margin]) - footprint["max_offset"]
                if not np.all(lo + 2e-7 <= hi):
                    continue
                for x in np.linspace(lo[0] + 1e-7, hi[0] - 1e-7, 41):
                    for y in np.linspace(lo[1] + 1e-7, hi[1] - 1e-7, 41):
                        xy = np.array([x, y], dtype=np.float64)
                        if self._neutral_avatar_default_table_side_for_xy(xy) != side:
                            continue
                        if not self._neutral_motion_region_is_clear(xy, footprint, task_regions):
                            continue
                        # Only existing default sides are considered. There
                        # is no new side-band exemption or persistent yaw map.
                        yield xy, body_rot

        # Preserve all existing selections: this pass is reached only after
        # every old heading/grid candidate has failed the caller's gates.
        yield from self._neutral_can_band_work_candidates(measured_can_poses, task_regions)

    def _neutral_can_band_work_candidates(self, measured_poses, task_regions):
        """Sample a failed can search inside unchanged side/table intersections.

        A grid over the whole feasible table can undersample the narrow side
        bands. Reuse each heading's actual measured geometry; retain the same
        grid size and the caller's complete strict acceptance checks.
        """
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        for side, region, rotation, footprint in measured_poses:
            margin = max(self._neutral_motion_table_margin(footprint), float(
                self.NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN))
            lo = np.array([xmin + margin, ymin + margin]) - footprint["min_offset"]
            hi = np.array([xmax - margin, ymax - margin]) - footprint["max_offset"]
            x0, x1, y0, y1 = region
            lo = np.maximum(lo, [x0, y0])
            hi = np.minimum(hi, [x1, y1])
            if not np.all(lo + 2e-7 <= hi):
                continue
            for x in np.linspace(lo[0] + 1e-7, hi[0] - 1e-7, 41):
                for y in np.linspace(lo[1] + 1e-7, hi[1] - 1e-7, 41):
                    xy = np.array([x, y], dtype=np.float64)
                    if self._neutral_avatar_default_table_side_for_xy(xy) != side:
                        continue
                    if self._neutral_motion_region_is_clear(xy, footprint, task_regions):
                        yield xy, rotation

    def _neutral_anchor_forbidden_rects(
        self,
        footprint: dict,
        task_regions: list[tuple[np.ndarray, float, str]],
    ) -> list[tuple[float, float, float, float]]:
        """Convert object regions into forbidden anchor rectangles.

        For a fixed motion AABB, an obstacle disk forbids anchors whose
        translated AABB overlaps that disk. This is the usual configuration
        space obstacle expansion specialized to axis-aligned rectangles.
        """
        min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
        max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        rects: list[tuple[float, float, float, float]] = []
        for center, radius, label in task_regions:
            if not self._neutral_task_region_is_physical(label):
                continue
            c = np.asarray(center, dtype=np.float64).ravel()[:2]
            r = float(radius) + margin
            rects.append((
                float(c[0] - r - max_offset[0]),
                float(c[0] + r - min_offset[0]),
                float(c[1] - r - max_offset[1]),
                float(c[1] + r - min_offset[1]),
            ))
        return rects

    def _neutral_empty_anchor_rect_candidates(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        forbidden_rects: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        """Return deterministic centers of free arrangement cells, largest first."""
        lo = np.asarray(lo, dtype=np.float64).ravel()[:2]
        hi = np.asarray(hi, dtype=np.float64).ravel()[:2]
        if not np.all(lo <= hi):
            return []
        xs = [float(lo[0]), float(hi[0])]
        ys = [float(lo[1]), float(hi[1])]
        clipped = []
        for x0, x1, y0, y1 in forbidden_rects:
            cx0, cx1 = max(float(x0), float(lo[0])), min(float(x1), float(hi[0]))
            cy0, cy1 = max(float(y0), float(lo[1])), min(float(y1), float(hi[1]))
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            clipped.append((cx0, cx1, cy0, cy1))
            xs.extend([cx0, cx1])
            ys.extend([cy0, cy1])
        xs = sorted(set(round(float(x), 6) for x in xs))
        ys = sorted(set(round(float(y), 6) for y in ys))
        cells: list[tuple[float, list[np.ndarray]]] = []
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x1 <= x0:
                continue
            for y0, y1 in zip(ys[:-1], ys[1:]):
                if y1 <= y0:
                    continue
                center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                blocked = False
                for rx0, rx1, ry0, ry1 in clipped:
                    if rx0 <= center[0] <= rx1 and ry0 <= center[1] <= ry1:
                        blocked = True
                        break
                if blocked:
                    continue
                area = float((x1 - x0) * (y1 - y0))
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                probes = [
                    center,
                    np.array([x0 + 0.20 * dx, y0 + 0.20 * dy], dtype=np.float64),
                    np.array([x0 + 0.80 * dx, y0 + 0.20 * dy], dtype=np.float64),
                    np.array([x0 + 0.20 * dx, y0 + 0.80 * dy], dtype=np.float64),
                    np.array([x0 + 0.80 * dx, y0 + 0.80 * dy], dtype=np.float64),
                ]
                cells.append((area, probes))
        if not cells:
            # Returning ONLY the anchor-box centre here makes
            # the layout solver test 3 poses total (1 per side) before
            # raising "no neutral layout candidate ..." — the coarse C-space
            # used to GENERATE candidates inflates every task disk by the
            # whole motion sweep AABB and can wipe out the free set, while
            # the fine per-point ACCEPT test is far more permissive.  Return
            # a dense uniform grid + corners so acceptance gets a fair shot;
            # scenes with free cells are unchanged (this branch never runs).
            center = (lo + hi) * 0.5
            span = hi - lo
            fallback: list[np.ndarray] = [center]
            for fx in (0.1, 0.3, 0.5, 0.7, 0.9):
                for fy in (0.1, 0.3, 0.5, 0.7, 0.9):
                    fallback.append(np.array([
                        lo[0] + fx * span[0], lo[1] + fy * span[1],
                    ], dtype=np.float64))
            return fallback
        cells.sort(key=lambda item: item[0], reverse=True)
        out: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()
        for _area, probes in cells:
            for probe in probes:
                key = (round(float(probe[0]), 6), round(float(probe[1]), 6))
                if key in seen:
                    continue
                seen.add(key)
                out.append(probe)
        return out

    # Opt-in (default off): tasks whose placement slots must stay clear of
    # the avatar's motion region set this True, so the layout solver treats
    # "slot:*"/"target" regions as physical obstacles.  Used by the blocks
    # ranking neutrals, where the avatar sweeping over the target row
    # can knock over placed blocks.
    NEUTRAL_SLOTS_ARE_PHYSICAL = False

    def _neutral_task_region_is_physical(self, label: object) -> bool:
        """Return whether a task region represents real occupied geometry."""
        text = str(label)
        if text.startswith("slot:") or text == "target":
            return bool(self.NEUTRAL_SLOTS_ARE_PHYSICAL)
        return True

    def _neutral_put_objects_side_region_for_sampling(
        self, side: str, region: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Use a narrower anchor strip for the plate/bowl neutral clip.

        The put-objects motion has a wider object sweep than mouse/sponge, so
        sampling the full edge strip can put held objects visually too close to
        the table boundary even when the swept AABB technically fits.
        """
        x0, x1, y0, y1 = [float(v) for v in region]
        frac = float(self.config.get("neutral_put_objects_side_width_fraction", 0.5))
        frac = min(1.0, max(0.05, frac))
        if side == "left":
            x0 = x1 - (x1 - x0) * frac
        elif side == "right":
            x1 = x0 + (x1 - x0) * frac
        elif side == "back":
            y0 = y1 - (y1 - y0) * frac
        return (x0, x1, y0, y1)

    def _neutral_avatar_side_aware_work_candidates(self, job: dict) -> list[np.ndarray] | None:
        """Sample anchors so the side-specific swept footprint fits a side band."""
        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(job)
        side_regions = explicit_side_regions
        if not side_regions and job.get("kind") in ("mouse", "sponge"):
            side_regions = self._neutral_avatar_default_side_work_regions()
        if not side_regions:
            return None
        candidate_sides: dict[tuple[float, float], str] = {}
        self._neutral_avatar_candidate_sides = candidate_sides
        self._neutral_avatar_default_side_candidates = set()

        def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            out_list.append(arr)
            candidate_sides[
                (round(float(arr[0]), 6), round(float(arr[1]), 6))
            ] = str(side_name)

        n = int(self.config.get("neutral_avatar_region_candidates", 16))
        n = max(1, n)
        out: list[np.ndarray] = []
        old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        try:
            for side, region in side_regions:
                if explicit_side_regions:
                    x0, x1, y0, y1 = [float(v) for v in region]
                    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                else:
                    xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                    center = np.array([
                        0.5 * (float(xmin) + float(xmax)),
                        0.5 * (float(ymin) + float(ymax)),
                    ], dtype=np.float64)
                self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_work_xy(job, center)
                _hold = self._neutral_avatar_measure_hold_points(job)
                fp = self._neutral_avatar_compute_motion_footprint(job)
                min_offset = np.asarray(fp["min_offset"], dtype=np.float64).ravel()[:2]
                max_offset = np.asarray(fp["max_offset"], dtype=np.float64).ravel()[:2]
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                table_margin = self._neutral_motion_table_margin(fp)
                if explicit_side_regions:
                    lo = np.array([x0, y0], dtype=np.float64) - min_offset
                    hi = np.array([x1, y1], dtype=np.float64) - max_offset
                else:
                    lo = np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64) - min_offset
                    hi = np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64) - max_offset
                table_lo = (
                    np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                    - min_offset
                )
                table_hi = (
                    np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                    - max_offset
                )
                lo = np.maximum(lo, table_lo)
                hi = np.minimum(hi, table_hi)
                if not np.all(lo <= hi):
                    continue
                per_region = max(1, int(np.ceil(float(n) / float(len(side_regions)))))
                for _ in range(per_region):
                    add_candidate(out, side, np.array([
                        np.random.uniform(float(lo[0]), float(hi[0])),
                        np.random.uniform(float(lo[1]), float(hi[1])),
                    ], dtype=np.float64))
                add_candidate(out, side, (lo + hi) * 0.5)
        finally:
            self._neutral_avatar_measure_body_rot = old_measure_rot
        if explicit_side_regions and job.get("kind") in ("mouse", "sponge"):
            old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
            try:
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                table_center = np.array([
                    0.5 * (float(xmin) + float(xmax)),
                    0.5 * (float(ymin) + float(ymax)),
                ], dtype=np.float64)
                for side, _region in self._neutral_avatar_default_side_work_regions():
                    self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(side, job)
                    _hold = self._neutral_avatar_measure_hold_points(job)
                    fp = self._neutral_avatar_compute_motion_footprint(job)
                    min_offset = np.asarray(fp["min_offset"], dtype=np.float64).ravel()[:2]
                    max_offset = np.asarray(fp["max_offset"], dtype=np.float64).ravel()[:2]
                    table_margin = self._neutral_motion_table_margin(fp)
                    lo = (
                        np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                        - min_offset
                    )
                    hi = (
                        np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                        - max_offset
                    )
                    if not np.all(lo <= hi):
                        continue
                    per_region = max(1, int(np.ceil(float(n) / 3.0)))
                    center_candidate = np.clip(table_center, lo, hi)
                    add_candidate(out, side, center_candidate)
                    self._neutral_avatar_default_side_candidates.add((
                        round(float(center_candidate[0]), 6),
                        round(float(center_candidate[1]), 6),
                    ))
                    for _ in range(per_region):
                        candidate = np.array([
                            np.random.uniform(float(lo[0]), float(hi[0])),
                            np.random.uniform(float(lo[1]), float(hi[1])),
                        ], dtype=np.float64)
                        add_candidate(out, side, candidate)
                        self._neutral_avatar_default_side_candidates.add((
                            round(float(candidate[0]), 6),
                            round(float(candidate[1]), 6),
                        ))
            finally:
                self._neutral_avatar_measure_body_rot = old_measure_rot
        return out or None

    def _neutral_footprint_inside_side_region(self, xy: np.ndarray, footprint: dict, side: str) -> bool:
        if not side:
            return True
        arr_for_key = np.asarray(xy, dtype=np.float64).ravel()[:2]
        if (
            round(float(arr_for_key[0]), 6),
            round(float(arr_for_key[1]), 6),
        ) in getattr(self, "_neutral_avatar_default_side_candidates", set()):
            return True
        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(self._neutral_avatar_job or {})
        if not explicit_side_regions and (self._neutral_avatar_job or {}).get("kind") != "put_objects_in_bowl":
            return True
        if (self._neutral_avatar_job or {}).get("kind") == "put_objects_in_bowl":
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            side_regions = (
                explicit_side_regions
                or self._neutral_avatar_default_side_work_regions()
            )
            for reg_side, region in side_regions:
                if str(reg_side) != str(side):
                    continue
                x0, x1, y0, y1 = [float(v) for v in region]
                if (
                    x0 - 1e-6 <= arr[0] <= x1 + 1e-6
                    and y0 - 1e-6 <= arr[1] <= y1 + 1e-6
                ):
                    return self._neutral_put_objects_footprint_is_table_safe(arr, footprint)
            return False
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["min_offset"], dtype=np.float64
        ).ravel()[:2]
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["max_offset"], dtype=np.float64
        ).ravel()[:2]
        side_regions = (
            explicit_side_regions
            or self._neutral_avatar_default_side_work_regions()
        )
        for reg_side, region in side_regions:
            if str(reg_side) != str(side):
                continue
            x0, x1, y0, y1 = [float(v) for v in region]
            if (
                min_xy[0] >= x0 - 1e-6
                and max_xy[0] <= x1 + 1e-6
                and min_xy[1] >= y0 - 1e-6
                and max_xy[1] <= y1 + 1e-6
            ):
                return True
        return False

    def _neutral_avatar_spawn_exclude_regions(self) -> list[tuple[float, float, float]]:
        """Approximate neutral work coverage before parent tasks place clutter."""
        if self._neutral_avatar_job is None:
            try:
                self._resolve_neutral_avatar_job()
            except Exception:
                return []
        job = self._neutral_avatar_job
        if not job:
            return []

        regions: list[tuple[float, float, float]] = []
        if job["kind"] == "writing":
            seen = set()
            for side, xy in self._neutral_writing_side_candidates(job):
                if side in seen:
                    continue
                seen.add(side)
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions
        if job["kind"] == "sitting_clap":
            seen = set()
            for side, xy in self._neutral_clap_side_candidates(job):
                if side in seen:
                    continue
                seen.add(side)
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions
        if job["kind"] == "sponge":
            raw_region = getattr(self, "NEUTRAL_SPONGE_WORK_REGION", None)
            if raw_region is not None:
                x0, x1, y0, y1 = [float(v) for v in np.asarray(raw_region).ravel()[:4]]
                center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                radius = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + 0.28
                regions.append((float(center[0]), float(center[1]), radius))
                return regions
            candidates = self._neutral_avatar_work_candidates(job)
            for xy in candidates:
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions

        xy = self._neutral_avatar_default_work_xy(job)
        radius = 0.24 if job["kind"] == "put_objects_in_bowl" else 0.18
        regions.append((float(xy[0]), float(xy[1]), radius))
        return regions

    def _select_neutral_avatar_work_xy(self, job: dict, footprint: dict) -> np.ndarray:
        candidates = self._neutral_avatar_work_candidates(job)
        forbidden = self._neutral_avatar_forbidden_regions()
        for xy in candidates:
            if self._neutral_footprint_is_clear(xy, footprint, forbidden):
                self._neutral_debug(
                    f"[neutral_avatar] selected work_xy={xy.round(3).tolist()} "
                    f"for {job['base_motion']}"
                )
                return xy
        xy = candidates[0]
        self._neutral_debug(
            f"[neutral_avatar] WARNING: no clear neutral candidate for "
            f"{job['base_motion']}; using {xy.round(3).tolist()}"
        )
        return xy

    def _neutral_footprint_is_clear(
        self, xy: np.ndarray, footprint: dict, forbidden: list[tuple[np.ndarray, float, str]]
    ) -> bool:
        min_xy = xy + np.asarray(footprint["min_offset"], dtype=np.float64)
        max_xy = xy + np.asarray(footprint["max_offset"], dtype=np.float64)
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        table_margin = self._neutral_motion_table_margin(footprint)
        if (
            min_xy[0] < float(xmin) + table_margin
            or max_xy[0] > float(xmax) - table_margin
            or min_xy[1] < float(ymin) + table_margin
            or max_xy[1] > float(ymax) - table_margin
        ):
            return False
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        for center, radius, _label in forbidden:
            center = np.asarray(center, dtype=np.float64).ravel()[:2]
            clamped = np.minimum(np.maximum(center, min_xy), max_xy)
            if float(np.linalg.norm(center - clamped)) < float(radius) + margin:
                return False
        return True

    def _neutral_footprint_inside_table(self, xy: np.ndarray, footprint: dict) -> bool:
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["min_offset"], dtype=np.float64
        ).ravel()[:2]
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["max_offset"], dtype=np.float64
        ).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        table_margin = self._neutral_motion_table_margin(footprint)
        return bool(
            min_xy[0] >= float(xmin) + table_margin
            and max_xy[0] <= float(xmax) - table_margin
            and min_xy[1] >= float(ymin) + table_margin
            and max_xy[1] <= float(ymax) - table_margin
        )

    def _neutral_motion_region_is_clear(
        self,
        xy: np.ndarray,
        footprint: dict,
        task_regions: list[tuple[np.ndarray, float, str]],
    ) -> bool:
        """Check the swept neutral region against table bounds and task regions.

        ``region_points`` are stored as offsets from the neutral anchor.  Each
        point has a radius, which lets us represent arm keypoints and neutral
        objects without collapsing everything into one oversized AABB.
        """
        anchor = np.asarray(xy, dtype=np.float64).ravel()[:2]
        if not self._neutral_can_base_sweep_is_clear(anchor, footprint, task_regions):
            return False
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        points = footprint.get("region_points") or []
        if not points:
            return self._neutral_footprint_is_clear(anchor, footprint, task_regions)
        for item in points:
            if isinstance(item, dict):
                offset = np.asarray(item.get("offset", [0.0, 0.0]), dtype=np.float64).ravel()[:2]
                radius = float(item.get("radius", 0.05))
            else:
                offset = np.asarray(item[:2], dtype=np.float64).ravel()[:2]
                radius = float(item[2])
                label = str(item[3]) if len(item) > 3 else ""
            if isinstance(item, dict):
                label = str(item.get("label", ""))
            center = anchor + offset
            table_margin = self._neutral_motion_table_margin(footprint)
            if not label.startswith("arm:") and (
                center[0] - radius < float(xmin) + table_margin
                or center[0] + radius > float(xmax) - table_margin
                or center[1] - radius < float(ymin) + table_margin
                or center[1] + radius > float(ymax) - table_margin
            ):
                return False
            if label.startswith("arm:") and str(footprint.get("kind", "")) in ("sponge", "mouse", "bottle"):
                continue
            for task_center, task_radius, task_label in task_regions:
                if not self._neutral_task_region_is_physical(task_label):
                    continue
                task_center = np.asarray(task_center, dtype=np.float64).ravel()[:2]
                if float(np.linalg.norm(center - task_center)) < radius + float(task_radius):
                    return False
        return True

    def _neutral_initial_prop_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        obj_name: str,
        forbidden: list[tuple[np.ndarray, float, str]],
    ) -> float:
        """Signed XY clearance for the prop at its reset pose.

        Positive means the reset prop center is outside all forbidden task
        objects by at least the configured margin; negative means overlap.
        """
        prop_xy = (
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)[:2]
        )
        prop_radius = self._neutral_prop_radius(obj_name)
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        best = float("inf")
        for center, radius, _label in forbidden:
            center = np.asarray(center, dtype=np.float64).ravel()[:2]
            clearance = float(np.linalg.norm(prop_xy - center)) - float(radius) - prop_radius - margin
            best = min(best, clearance)
        return best if np.isfinite(best) else 1.0

    def _neutral_initial_prop_layout(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
    ) -> list[tuple[str, np.ndarray, float, bool]]:
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_hold = hold_by_frame[int(first_event["clip_frame"])]
        first_target = np.array(
            [
                float(np.asarray(xy, dtype=np.float64).ravel()[0]),
                float(np.asarray(xy, dtype=np.float64).ravel()[1]),
                float(self._neutral_avatar_prop_center_z(str(first_event["object"]))),
            ],
            dtype=np.float64,
        )
        shift = first_target - first_hold
        seen = set()
        out: list[tuple[str, np.ndarray, float, bool]] = []
        for event in job["events"]:
            if event["type"] not in ("attach", "pick"):
                continue
            obj_name = str(event["object"])
            if obj_name in seen:
                continue
            seen.add(obj_name)
            pos = hold_by_frame[int(event["clip_frame"])] + shift
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)
            if job["kind"] == "put_objects_in_bowl":
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            elif obj_name == "sponge":
                pos[2] = self._neutral_avatar_prop_center_z("sponge")
            is_can = (
                job.get("kind") == "put_objects_in_bowl"
                and str(obj_name).startswith("object_")
            )
            radius = (
                self._neutral_avatar_can_body_radius(obj_name)
                if is_can
                else self._neutral_prop_radius(obj_name)
            )
            out.append((obj_name, np.asarray(pos, dtype=np.float64), float(radius), bool(is_can)))
        return out

    def _neutral_initial_props_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
        forbidden: list[tuple[np.ndarray, float, str]],
    ) -> float:
        """Signed initial-layout clearance across all neutral props.

        For neutral can props, require a 5 cm buffer outside the can body
        radius. This prevents the can from starting in contact with task
        objects or the other neutral prop, which otherwise makes the can
        topple during the short neutral-layout settle.
        """
        layout = self._neutral_initial_prop_layout(job, xy, body_pos, hold_by_frame)
        if not layout:
            first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
            return self._neutral_initial_prop_clearance(
                job, xy, body_pos, str(first_event["object"]), forbidden
            )
        best = float("inf")
        default_margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        can_margin = float(self.config.get(
            "neutral_can_clearance_margin",
            self.NEUTRAL_CAN_CLEARANCE_MARGIN,
        ))
        for obj_name, pos, radius, is_can in layout:
            margin = can_margin if is_can else default_margin
            for center, task_radius, _label in forbidden:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clearance = (
                    float(np.linalg.norm(pos[:2] - center))
                    - float(task_radius)
                    - float(radius)
                    - margin
                )
                best = min(best, clearance)
        for i, (_name_a, pos_a, radius_a, is_can_a) in enumerate(layout):
            for _name_b, pos_b, radius_b, is_can_b in layout[i + 1:]:
                margin = can_margin if (is_can_a or is_can_b) else default_margin
                clearance = (
                    float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
                    - float(radius_a)
                    - float(radius_b)
                    - margin
                )
                best = min(best, clearance)
        return best if np.isfinite(best) else 1.0

    def _neutral_initial_props_table_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
    ) -> float:
        """Signed clearance from reset-layout props to the real table edge."""
        layout = self._neutral_initial_prop_layout(job, xy, body_pos, hold_by_frame)
        if not layout:
            return 1.0
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        best = float("inf")
        for obj_name, pos, radius, _is_can in layout:
            p = np.asarray(pos, dtype=np.float64).ravel()[:2]
            r = max(
                float(radius) + self._neutral_motion_table_margin(
                    getattr(self, "_neutral_avatar_motion_footprint", None)
                ),
                self._neutral_avatar_table_margin_for_prop(job, str(obj_name)),
            )
            best = min(
                best,
                p[0] - (float(xmin) + r),
                (float(xmax) - r) - p[0],
                p[1] - (float(ymin) + r),
                (float(ymax) - r) - p[1],
            )
        return best if np.isfinite(best) else 1.0

    def neutral_task_regions(self) -> list[tuple[np.ndarray, float, str]]:
        """Task occupied/safety regions used by neutral placement.

        Task subclasses can override this API with more precise geometry. The
        default starts from existing task-object regions and adds one coarse
        base disk for each robot arm entity.
        """
        regions = list(self._neutral_avatar_forbidden_regions())
        robot = getattr(self, "robot", None)
        seen = set()
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(robot, arm_name, None)
            ent = getattr(arm, "entity", None)
            if ent is None or id(ent) in seen:
                continue
            seen.add(id(ent))
            pos = None
            for getter in (
                lambda: to_numpy(ent.get_pos()).ravel()[:2],
                lambda: np.asarray(getattr(arm, "base_pos"), dtype=np.float64).ravel()[:2],
            ):
                try:
                    candidate = np.asarray(getter(), dtype=np.float64).ravel()[:2]
                except Exception:
                    continue
                if np.all(np.isfinite(candidate)):
                    pos = candidate
                    break
            if pos is None:
                links = getattr(ent, "links", []) or []
                if links:
                    try:
                        candidate = to_numpy(links[0].get_pos()).ravel()[:2]
                        if np.all(np.isfinite(candidate)):
                            pos = np.asarray(candidate, dtype=np.float64)
                    except Exception:
                        pass
            if pos is not None:
                regions.append((
                    np.asarray(pos, dtype=np.float64),
                    float(self.config.get(
                        "neutral_robot_base_region_radius",
                        self.NEUTRAL_ROBOT_BASE_REGION_RADIUS,
                    )),
                    f"robot_base:{arm_name}",
                ))
        return regions

    def _neutral_avatar_prop_forbidden_regions(
        self, regions: list[tuple[np.ndarray, float, str]]
    ) -> list[tuple[np.ndarray, float, str]]:
        """Physical object regions used by reset-time prop overlap checks."""
        out: list[tuple[np.ndarray, float, str]] = []
        for center, radius, label in regions:
            label = str(label)
            if label.startswith("robot:"):
                continue
            if (label.startswith("slot:") or label == "target") and (
                not self.NEUTRAL_SLOTS_ARE_PHYSICAL
            ):
                continue
            out.append((center, radius, label))
        out.extend(self._neutral_robot_gripper_forbidden_regions())
        return out

    def _neutral_robot_gripper_forbidden_regions(self) -> list[tuple[np.ndarray, float, str]]:
        """Hand/finger/TCP regions to keep neutral cans out of at reset."""
        robot = getattr(self, "robot", None)
        out: list[tuple[np.ndarray, float, str]] = []
        radius = float(self.config.get(
            "neutral_can_gripper_clearance_radius",
            self.NEUTRAL_CAN_GRIPPER_CLEARANCE_RADIUS,
        ))
        seen = set()
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(robot, arm_name, None)
            ent = getattr(arm, "entity", None)
            if ent is None or id(ent) in seen:
                continue
            seen.add(id(ent))
            for i, link in enumerate(getattr(ent, "links", []) or []):
                name = str(getattr(link, "name", "")).lower()
                if not any(token in name for token in ("hand", "finger", "gripper")):
                    continue
                try:
                    pos = to_numpy(link.get_pos()).ravel()[:2]
                except Exception:
                    continue
                if np.all(np.isfinite(pos)):
                    out.append((
                        np.asarray(pos, dtype=np.float64),
                        radius,
                        f"gripper:{arm_name}:{i}:{name}",
                    ))
            try:
                tcp = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
                if np.all(np.isfinite(tcp[:2])):
                    out.append((
                        tcp[:2].astype(np.float64),
                        radius,
                        f"gripper:{arm_name}:tcp",
                    ))
            except Exception:
                pass
        return out

    def _neutral_avatar_forbidden_regions(self) -> list[tuple[np.ndarray, float, str]]:
        regions: list[tuple[np.ndarray, float, str]] = []

        def add_xy(xy, radius: float, label: str):
            try:
                arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            except Exception:
                return
            if arr.shape[0] == 2 and np.all(np.isfinite(arr)):
                regions.append((arr, float(radius), label))

        def add_entity(obj, radius: float, label: str):
            ent = getattr(obj, "entity", obj)
            try:
                add_xy(to_numpy(ent.get_pos()).ravel()[:2], radius, label)
            except Exception:
                try:
                    add_xy(np.asarray(obj.get_pose().p, dtype=np.float64)[:2], radius, label)
                except Exception:
                    pass

        def entity_xy_radius(obj, fallback: float) -> tuple[np.ndarray, float] | None:
            ent = getattr(obj, "entity", obj)
            try:
                aabb = to_numpy(ent.get_AABB()).reshape(-1, 3)
                lo = np.min(aabb, axis=0)
                hi = np.max(aabb, axis=0)
                center = 0.5 * (lo[:2] + hi[:2])
                half = 0.5 * (hi[:2] - lo[:2])
                radius = float(max(half[0], half[1]))
                if np.all(np.isfinite(center)) and np.isfinite(radius) and radius > 1e-5:
                    return center.astype(np.float64), radius
            except Exception:
                pass
            try:
                pos = to_numpy(ent.get_pos()).ravel()[:2]
                if np.all(np.isfinite(pos)):
                    return pos.astype(np.float64), float(fallback)
            except Exception:
                pass
            return None

        if hasattr(self, "blocks"):
            for name, ent in getattr(self, "blocks", {}).items():
                add_entity(ent, 0.055, f"block:{name}")
        if hasattr(self, "target_slots"):
            for name, pos in getattr(self, "target_slots", {}).items():
                add_xy(np.asarray(pos)[:2], 0.060, f"slot:{name}")
        for attr, radius in (
            ("objects", 0.060),
            ("breads", 0.070),
            ("shoes", 0.090),
            ("bowls", 0.100),
        ):
            for i, obj in enumerate(getattr(self, attr, []) or []):
                add_entity(obj, radius, f"{attr}:{i}")
        if hasattr(self, "basket"):
            add_entity(self.basket, 0.130, "basket")
        if hasattr(self, "shoebox"):
            add_entity(self.shoebox, 0.150, "shoebox")
        if hasattr(self, "food_actors"):
            for actor, spec in getattr(self, "food_actors", []) or []:
                add_entity(actor, 0.070, f"food:{getattr(spec, 'name', '?')}")
        if hasattr(self, "tray_actor"):
            add_entity(self.tray_actor, 0.140, "tray")
        if hasattr(self, "pan_actor"):
            add_entity(self.pan_actor, 0.130, "pan")
        if hasattr(self, "_cooktop_xy"):
            add_xy(self._cooktop_xy, 0.180, "cooktop")
        if hasattr(self, "_cabinet_pos"):
            add_xy(np.asarray(self._cabinet_pos)[:2], 0.140, "cabinet")
        if hasattr(self, "object"):
            add_entity(self.object, 0.070, "cabinet_object")
        if hasattr(self, "basket_poses"):
            for i, pose in enumerate(getattr(self, "basket_poses", []) or []):
                half_x = float(getattr(self, "_BASKET_HALF_X", 0.096))
                half_y = float(getattr(self, "_BASKET_HALF_Y", 0.096))
                add_xy(np.asarray(pose.p)[:2], max(half_x, half_y), f"categorize_basket:{i}")
        if hasattr(self, "_sort_targets"):
            for i, item in enumerate(getattr(self, "_sort_targets", []) or []):
                actor, obj_name, model_id, _basket_idx = item
                radius = None
                get_radius = getattr(self, "_get_object_cross_section_radius", None)
                if get_radius is not None:
                    try:
                        radius = float(get_radius(str(obj_name), int(model_id)))
                    except Exception:
                        radius = None
                region = entity_xy_radius(actor, radius if radius is not None else 0.040)
                if region is not None:
                    center, actual_radius = region
                    add_xy(center, min(actual_radius, radius) if radius is not None else actual_radius, f"categorize_obj:{i}")
                else:
                    add_entity(actor, radius if radius is not None else 0.040, f"categorize_obj:{i}")

        target = getattr(self, "target", None)
        pos = getattr(target, "position", None)
        if pos is not None:
            try:
                add_xy(pos() if callable(pos) else pos, 0.100, "target")
            except Exception:
                pass
        return regions

    def _neutral_avatar_forward_left_xy(self, body_pos: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        if rot is None:
            rot = getattr(self, "_neutral_avatar_body_rot", None)
        if rot is not None:
            forward = self._neutral_avatar_semantic_forward_xy(
                getattr(self, "_neutral_avatar_job", None),
                np.asarray(rot, dtype=np.float64),
            )
            norm = float(np.linalg.norm(forward))
            forward = forward / norm if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float64)
        elif body_pos is None:
            forward = np.array([0.0, -1.0], dtype=np.float64)
        else:
            forward = np.asarray(self.neutral_avatar_target[:2], dtype=np.float64) - np.asarray(
                body_pos[:2], dtype=np.float64
            )
            norm = float(np.linalg.norm(forward))
            if norm < 1e-6:
                forward = np.array([0.0, -1.0], dtype=np.float64)
            else:
                forward = forward / norm
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        return forward, left

    def _neutral_avatar_semantic_forward_xy(
        self, job: dict | None, rot: np.ndarray
    ) -> np.ndarray:
        forward = np.asarray(rot, dtype=np.float64)[:2, 0].copy()
        offset = self._neutral_avatar_motion_visual_yaw_offset(job)
        if abs(float(offset)) > 1e-9:
            c, s = float(np.cos(-offset)), float(np.sin(-offset))
            forward = np.array(
                [c * forward[0] - s * forward[1], s * forward[0] + c * forward[1]],
                dtype=np.float64,
            )
        return forward

    def _neutral_avatar_initial_object_offset(
        self, job: dict, obj_name: str, body_pos: np.ndarray | None
    ) -> np.ndarray:
        task_placement = self._neutral_avatar_task_placement(job)
        offset = np.zeros(3, dtype=np.float64)
        right_by_name = dict(task_placement.get("object_image_right_by_name", {}) or {})
        up_by_name = dict(task_placement.get("object_image_up_by_name", {}) or {})
        image_right = float(right_by_name.get(str(obj_name), 0.0))
        image_up = float(up_by_name.get(str(obj_name), 0.0))
        if image_right or image_up:
            forward, left = self._neutral_avatar_forward_left_xy(body_pos)
            offset[:2] += image_right * left - image_up * forward
        xyz_by_name = dict(task_placement.get("object_xyz_offset_by_name", {}) or {})
        if str(obj_name) in xyz_by_name:
            arr = np.asarray(xyz_by_name[str(obj_name)], dtype=np.float64).ravel()
            if arr.size != 3:
                raise ValueError(
                    f"object_xyz_offset_by_name[{obj_name!r}] must have 3 values"
                )
            offset += arr
        return offset

    def _neutral_avatar_clamp_table_xy(self, xy: np.ndarray, margin: float | None = None) -> np.ndarray:
        margin = self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN if margin is None else float(margin)
        x0, x1, y0, y1 = [float(v) for v in self._neutral_table_xy_bounds()]
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2].copy()
        arr[0] = float(np.clip(arr[0], x0 + margin, x1 - margin))
        arr[1] = float(np.clip(arr[1], y0 + margin, y1 - margin))
        return arr

    def _place_neutral_avatar_props(
        self,
        job: dict,
        hold_by_frame: dict[int, np.ndarray],
        shift: np.ndarray,
        body_pos: np.ndarray | None = None,
    ) -> None:
        for event in job["events"]:
            if event["type"] not in ("attach", "pick"):
                continue
            obj_name = str(event["object"])
            obj = self.neutral_avatar_props[obj_name]
            pos = hold_by_frame[int(event["clip_frame"])] + shift
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)
            if job["kind"] == "put_objects_in_bowl":
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            elif obj_name == "sponge":
                pos[2] = self._neutral_avatar_prop_center_z("sponge")
            elif obj_name == "bottle" or str(obj_name).startswith("bottle_"):
                pos[:2] = self._neutral_avatar_clamp_table_xy(
                    pos[:2],
                    margin=self._neutral_avatar_table_margin_for_prop(job, obj_name),
                )
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
            self._neutral_zero_entity_velocity(obj)
            if job["kind"] == "put_objects_in_bowl":
                # Record the resting spawn pose so the patched step-sim can pin
                # a not-yet-picked prop in place. Near-spherical props (apples)
                # otherwise roll off the grab spot during the long approach.
                cache = getattr(self, "_neutral_put_objects_spawn_pose", None)
                if cache is None:
                    cache = {}
                    self._neutral_put_objects_spawn_pose = cache
                cache[obj_name] = (
                    pos.astype(float).copy(),
                    self._neutral_avatar_prop_quat(obj_name).astype(float).copy(),
                )

        if job["kind"] == "put_objects_in_bowl" and hasattr(self, "neutral_avatar_bowl"):
            bowl_event_type = self._neutral_bowl_event_type(job)
            put_points = [
                hold_by_frame[int(e["clip_frame"])] + shift
                for e in job["events"]
                if e["type"] == bowl_event_type
            ]
            if put_points:
                xy = np.mean(np.asarray(put_points)[:, :2], axis=0)
                z = float(self.TABLE_TOP_Z + 0.020)
                self.neutral_avatar_bowl.set_pos(np.array([xy[0], xy[1], z], dtype=float))
                self.neutral_avatar_bowl.set_quat(self._neutral_avatar_prop_quat("plate"))
                self._neutral_zero_entity_velocity(self.neutral_avatar_bowl)

    def _neutral_reset_can_props_upright(self, job: dict) -> None:
        if job.get("kind") != "put_objects_in_bowl":
            return
        for obj_name, obj in getattr(self, "neutral_avatar_props", {}).items():
            if not str(obj_name).startswith("object_"):
                continue
            try:
                pos = to_numpy(obj.get_pos()).ravel()[:3].astype(np.float64)
            except Exception:
                continue
            pos[2] = self._neutral_avatar_prop_center_z(str(obj_name))
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat(str(obj_name)))
            self._neutral_zero_entity_velocity(obj)

    def _neutral_can_prop_is_upright(self, obj_name: str, obj) -> bool:
        try:
            quat = to_numpy(obj.get_quat()).ravel()[:4]
            pos = to_numpy(obj.get_pos()).ravel()[:3]
        except Exception:
            return False
        axis = t3d.quaternions.quat2mat(quat) @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        dot = float(axis[2])
        z_min = float(self.TABLE_TOP_Z - 0.050)
        return bool(dot >= float(self.NEUTRAL_CAN_UPRIGHT_DOT_MIN) and float(pos[2]) >= z_min)

    def _neutral_assert_can_upright_before_pick(self, obj_name: str, obj) -> None:
        if not (
            str(obj_name).startswith("object_")
            and (self._neutral_avatar_job or {}).get("kind") == "put_objects_in_bowl"
        ):
            return
        # The upright gate is can-specific (a can must stand on its base). Props
        # without a canonical upright axis (e.g. apples) are exempt.
        if self._neutral_put_objects_asset() != "071_can":
            return
        if self._neutral_can_prop_is_upright(str(obj_name), obj):
            return
        self.plan_success = False
        try:
            pos = to_numpy(obj.get_pos()).ravel()[:3].astype(float)
            quat = to_numpy(obj.get_quat()).ravel()[:4].astype(float)
        except Exception:
            pos = np.full(3, np.nan)
            quat = np.full(4, np.nan)
        msg = (
            f"neutral can {obj_name} is not upright before avatar pick; "
            f"pos={np.round(pos, 4).tolist()} quat={np.round(quat, 4).tolist()}"
        )
        self._neutral_debug(f"[neutral_avatar] FAIL: {msg}")
        # Scored as an episode failure (the robot disturbed the human's
        # prop), not an exception: raising would abort the whole rollout,
        # leaving the episode with no metrics.  The human still picks the
        # can up.
        disturbed = getattr(self, "_neutral_prop_disturbed", None)
        if not disturbed:
            self._neutral_prop_disturbed = {
                "object": str(obj_name),
                "pos": np.round(pos, 4).tolist(),
                "quat": np.round(quat, 4).tolist(),
                "reason": "can_not_upright_before_avatar_pick",
            }

    def _neutral_avatar_prop_quat(self, obj_name: str) -> np.ndarray:
        upright = np.array([0.7071067811865476, 0.7071067811865475, 0.0, 0.0])
        if obj_name == "mouse":
            task_placement = self._neutral_avatar_task_placement()
            yaw_deg = float(
                dict(task_placement.get("object_yaw_deg_by_name", {}) or {}).get("mouse", -90.0)
            )
            if bool(task_placement.get("object_yaw_relative_to_avatar", False)):
                rot = np.asarray(
                    getattr(self, "_neutral_avatar_body_rot", self.avatar_init_rot),
                    dtype=np.float64,
                )
                forward = rot[:, 0].copy()
                yaw_deg += float(np.rad2deg(np.arctan2(forward[1], forward[0])))
            yaw = np.deg2rad(yaw_deg)
            yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
            return t3d.quaternions.qmult(yaw_quat, upright)
        if str(obj_name).startswith("object_"):
            return upright
        if obj_name in ("bottle", "sponge", "apple", "plate") or str(obj_name).startswith("bottle_"):
            return upright
        if obj_name == "phone":
            # Orient the phone in the LEFT palm with the screen toward the head
            # (back facing out) — built from the live hand frame, so it looks
            # like a natural phone-call hold once attached.  Same math as
            # the talking-motion calibration.
            import genesis.utils.geom as geom_utils
            try:
                robot = self.avatar.robot
                _, R_hand = robot._get_hand_frame(0)
                palm = np.asarray(robot.get_palm_center(0), dtype=np.float64)
                head = np.asarray(
                    robot.skin.get_global_translation("Head")[0], dtype=np.float64
                ).ravel()[:3]
            except Exception:
                return np.array([1.0, 0.0, 0.0, 0.0])
            e2 = np.asarray(R_hand[:, 1], dtype=np.float64)
            y = head - palm
            y = y / (np.linalg.norm(y) + 1e-8)
            z = e2 - np.dot(e2, y) * y
            if np.linalg.norm(z) < 1e-6:
                z = np.array([0.0, 0.0, 1.0]) - y[2] * y
            z = z / (np.linalg.norm(z) + 1e-8)
            x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-8
            z = np.cross(x, y)
            R = np.column_stack([x, y, z])
            return np.asarray(geom_utils.R_to_quat(R)).astype(np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0])
