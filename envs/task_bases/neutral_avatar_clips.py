"""Per-clip neutral-avatar preparation: seated writing, standing laptop,
seated typing, seated clapping, and standing phone-talker layouts.
"""


from __future__ import annotations


import numpy as np
import transforms3d as t3d

from ..genesis_compat import ensure_inertial_urdf, ensure_nyx_urdf
from ..utils import (
    ASSETS_PATH,
    Pose,
    load_mesh,
    load_object,
    load_urdf,
)
from .neutral_avatar_clip_transforms import (
    install_clap_subclip,
    install_laptop_pingpong_subclip,
    install_typing_subclip,
)




class _NeutralAvatarClipsMixin:
    """Split from NeutralAvatarTableWorkMixin (verbatim member move); composed back into it in neutral_avatar_table_work.py."""

    # ------------------------------------------------------------------
    # Seated-writing neutral clip
    # ------------------------------------------------------------------

    def _neutral_writing_dz(self) -> float:
        """Vertical shift from the probe reference tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_WRITING_TABLE_TOP_REF)

    def _neutral_writing_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_WRITING_SEAT_HEIGHT_REF) + self._neutral_writing_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _neutral_urdf_merge_fixed_links(self) -> bool | None:
        """NYX renders URDF entities as sub-scenes that re-parse the file and
        need the file's link tree to map 1:1 onto the simulated links, so the
        chair / laptop URDFs must keep their fixed links un-merged under NYX.
        None keeps Genesis's default (merged) for every other renderer."""
        return False if str(self.config.get("renderer", "")) == "nyx" else None

    def _neutral_chair_urdf(self):
        """sapien-chair-179 mobility.urdf ships no <inertial> tags; with the
        NYX un-merged link tree Genesis 1.x auto-inertia goes degenerate and
        the first settle step nan-crashes (same failure as 036_cabinet)."""
        return ensure_inertial_urdf(
            ASSETS_PATH / "objects" / "sapien-chair-179" / "mobility.urdf"
        )

    def _neutral_laptop_urdf(self):
        """015_laptop/9960 mobility.urdf also ships without <inertial> tags."""
        return ensure_inertial_urdf(
            ASSETS_PATH / "objects" / "015_laptop" / "9960" / "mobility.urdf"
        )

    def _neutral_load_prop_urdf(self, urdf_path, pose: Pose, scale: float):
        """Load a fixed chair/laptop prop URDF, NYX-safe.

        NYX's sub-scene re-parse drops fixed-joint child links (the chair
        body / laptop keyboard hang under a fixed root joint → they vanish,
        leaving floating wheels / a detached lid) and ignores Genesis's
        runtime ``scale``. Under NYX, load a sibling file with the fixed
        joints pre-merged and the scale baked in (see ensure_nyx_urdf).
        """
        merge = self._neutral_urdf_merge_fixed_links()
        if merge is False:
            urdf_path = ensure_nyx_urdf(urdf_path, scale)
            # Default on: the laptop's many visual nodes abort NYX scene import
            # (rc=-6) without the visual-only mesh pack. Collision is unchanged.
            if (self.config.get("nyx_laptop_mesh_pack_v1", True) is not False
                    and "015_laptop" in urdf_path.parts):
                from ..nyx_mesh_pack import pack_urdf_visual_meshes
                urdf_path = pack_urdf_visual_meshes(urdf_path)
            scale = 1.0
        return load_urdf(
            self.scene, urdf_path, pose,
            scale=float(scale), fix_root=True, merge_fixed_links=merge,
        )

    def _load_neutral_writing_props(self, job: dict) -> None:
        """Create the pen / notebook / chair, parked until layout selection.

        All three are repositioned by `_prepare_neutral_writing_layout`; the
        pen is fixed-base because the avatar hand attachment drives it purely
        kinematically (set_pos/set_quat each step).
        """
        chair_scale = self._neutral_writing_chair_scale()
        self.neutral_avatar_props = {
            "pen": load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "058_markpen" / "visual" / "base0.glb",
                Pose([6.0, 6.4, 1.0]),
                scale=float(self.NEUTRAL_WRITING_PEN_SCALE),
                is_static=True,
                collision=False,
            )
        }
        self.neutral_avatar_prop = self.neutral_avatar_props["pen"]
        self.neutral_avatar_book = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / "092_notebook" / "visual" / "base0.glb",
            Pose([6.0, 6.8, 1.0]),
            scale=float(self.NEUTRAL_WRITING_BOOK_SCALE),
            is_static=True,
            collision=False,
        )
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self.neutral_avatar_target = np.array(
            [6.0, 6.8, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    _NEUTRAL_WRITING_SIDE_INWARD = {
        "back": (0.0, -1.0),
        "left": (1.0, 0.0),
        "right": (-1.0, 0.0),
    }

    def _neutral_writing_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, book-center) candidates at the fixed edge inset, center-out.

        The book inset from the table edge is fixed, so the only placement
        freedom is the lateral slide along the edge.  Lateral spans come from
        the task's side work regions (which encode "not the robot side"),
        falling back to the generic side bands.
        """
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_WRITING_BOOK_EDGE_INSET)
        half_long = float(self.NEUTRAL_WRITING_BOOK_HALF_LONG)
        edge_margin = float(self.NEUTRAL_WRITING_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_WRITING_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        elif self._neutral_avatar_forced_motion_requested():
            known = {
                (str(side), tuple(float(v) for v in region))
                for side, region in side_regions
            }
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key not in known:
                    side_regions.append((side, region))
                    known.add(key)
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) + half_long + edge_margin)
                lat_hi = min(x1, float(xmax) - half_long - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) + half_long + edge_margin)
                lat_hi = min(y1, float(ymax) - half_long - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _neutral_writing_footprint(self, side: str) -> dict:
        """Neutral region of the writing motion = the book's region."""
        clearance = float(self.NEUTRAL_WRITING_BOOK_CLEARANCE)
        half_long = float(self.NEUTRAL_WRITING_BOOK_HALF_LONG) + clearance
        half_depth = float(self.NEUTRAL_WRITING_BOOK_HALF_DEPTH) + clearance
        if side == "back":
            half = np.array([half_long, half_depth], dtype=np.float64)
        else:
            half = np.array([half_depth, half_long], dtype=np.float64)
        return {
            "motion": "writing",
            "kind": "writing",
            "min_offset": -half,
            "max_offset": half,
            "size": 2.0 * half,
            "region_points": [],
        }

    def _prepare_neutral_writing_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated writing clip.

        Pose chain (all fixed): book center = anchor on the table at the edge
        inset; avatar root = anchor - R_body @ BOOK_LOCAL_XY (which puts the
        body at a fixed gap outside the table edge); chair under the avatar;
        pen attached near-vertical at the page under the right hand.
        """
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_writing_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_writing_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no writing-side candidates: table bounds/side regions leave "
                "no room for the notebook at the fixed edge inset"
            )

        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        book_local = np.asarray(self.NEUTRAL_WRITING_BOOK_LOCAL_XY, dtype=np.float64)
        body_points = [
            (str(label), np.asarray(p, dtype=np.float64).ravel()[:2], float(r))
            for label, p, r in self.NEUTRAL_WRITING_BODY_POINTS
        ]
        robot_prefixes = tuple(self.NEUTRAL_WRITING_ROBOT_REGION_PREFIXES)
        rot2_by_side = {}
        # The robot's planned sweeps are not captured by the static task-object
        # disks, so among valid candidates pick the one farthest from the
        # robot regions (base + gripper/TCP) — the seated writer leans well
        # over the tabletop and cannot dodge the arm.
        selected = None        # best (book + body) clear candidate
        best_book_only = None  # (robot_dist, side, xy, footprint, record)
        best_any = None        # (book clearance, side, xy, footprint, record)
        debug_records = []
        writing_valid = []     # every (book + body) clear candidate, in order
        for side, xy in candidates:
            footprint = self._neutral_writing_footprint(side)
            book_clear = self._neutral_footprint_is_clear(xy, footprint, task_regions)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            rot2 = rot2_by_side.get(side)
            if rot2 is None:
                rot2 = np.asarray(
                    self._neutral_avatar_body_rot_for_side(side, job),
                    dtype=np.float64,
                )[:2, :2]
                rot2_by_side[side] = rot2
            pts_world = [
                (label, xy + rot2 @ (p - book_local), r)
                for label, p, r in body_points
            ]
            body_clear = True
            book_worst = float("inf")
            robot_dist = float("inf")
            for center, radius, label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                radius = float(radius)
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                d_book = float(np.linalg.norm(center - clamped)) - radius - margin
                book_worst = min(book_worst, d_book)
                d_body = min(
                    float(np.linalg.norm(center - pt)) - radius - r_pt - margin
                    for _lbl, pt, r_pt in pts_world
                )
                if str(label).startswith(robot_prefixes):
                    robot_dist = min(robot_dist, d_book, d_body)
                    # The arm sweeps, so keep the seated body envelope a clear
                    # margin from the robot base/gripper regions.  Short
                    # tabletop objects are NOT gated here: the head/hands lean
                    # ABOVE them (verified 0 robot-avatar collisions), and the
                    # book footprint already prevents table-surface overlap.
                    if d_body < float(self.NEUTRAL_WRITING_BODY_SWEEP_MARGIN):
                        body_clear = False
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(book_clear),
                "neutral_region_clear": bool(book_clear),
                "footprint_table_ok": True,
                "side_region_ok": True,
                "body_ok": bool(body_clear),
                "prop_clearance": robot_dist if np.isfinite(robot_dist) else 1.0,
                "prop_table_clearance": book_worst if np.isfinite(book_worst) else 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if book_clear and body_clear:
                writing_valid.append((robot_dist, side, xy, footprint, record))
                if selected is None or robot_dist > selected[0]:
                    selected = (robot_dist, side, xy, footprint, record)
            elif book_clear:
                if best_book_only is None or robot_dist > best_book_only[0]:
                    best_book_only = (robot_dist, side, xy, footprint, record)
            if best_any is None or book_worst > best_any[0]:
                best_any = (book_worst, side, xy, footprint, record)
        self._neutral_avatar_region_debug_records = debug_records
        if self._neutral_contested_requested():
            # Contested layout: among the same body-clear writing
            # anchors, take the one nearest the robot corridor band instead
            # of the farthest-from-robot one.
            chosen = self._neutral_contested_select(
                [entry[4] for entry in writing_valid], task_regions,
            )
            if chosen is not None:
                selected = next(entry for entry in writing_valid if entry[4] is chosen)
        if selected is None:
            # Body-clear is mandatory under strict layout: a grazing-margin
            # anchor reads as a robot-through-avatar collision later, and a
            # raise lets the collector retry with a new task-object seed.
            if bool(self.config.get("neutral_avatar_strict_layout", True)):
                raise RuntimeError(
                    "no writing layout candidate keeps the book region and the "
                    "seated body envelope clear of task objects; motion=writing"
                )
            if best_book_only is not None:
                selected = best_book_only
                self._neutral_debug(
                    "[neutral_avatar] writing: body envelope grazes a task "
                    "region on every clear candidate; using the "
                    "farthest-from-robot one"
                )
        if selected is None:
            _worst, side, xy, footprint, record = best_any
            record["selected"] = True
            selected = (None, side, xy, footprint, record)
            self._neutral_debug(
                f"[neutral_avatar] WARNING: no clear writing candidate; using "
                f"{np.asarray(xy).round(3).tolist()} (clearance={_worst:.3f})"
            )
        _robot_dist, side, anchor_xy, footprint, record = selected
        record["selected"] = True
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[writing_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for center, radius, label in task_regions:
                print(
                    f"[writing_layout] region {label}: "
                    f"center={np.round(np.asarray(center, dtype=float).ravel()[:2], 3).tolist()} "
                    f"r={float(radius):.3f}"
                )
            for rec in debug_records:
                print(
                    f"[writing_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} book={rec['footprint_clear']} "
                    f"body={rec['body_ok']} robot_dist={float(rec['prop_clearance']):.3f}"
                    + (" <- selected" if rec.get("selected") else "")
                )
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        body_rot = self._neutral_avatar_body_rot_for_side(side, job)
        rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
        book_local = np.asarray(self.NEUTRAL_WRITING_BOOK_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ book_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        # Chair under the avatar, wheels on the floor, backrest behind her.
        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_writing_chair_scale()
        chair_yaw = facing + np.pi  # asset faces -X
        chair.set_pos(np.array([
            body_xy[0],
            body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale,
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)],
            dtype=float,
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            # Wheel / gas-lift joints sag over long rollouts; re-pin per step.
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        # Notebook flat on the table, long edge across the writing direction.
        book_R = (
            t3d.euler.euler2mat(0.0, 0.0, facing + np.pi / 2.0, "sxyz")
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        self.neutral_avatar_book.set_pos(np.array(
            [anchor_xy[0], anchor_xy[1], table_top + 0.001], dtype=float
        ))
        self.neutral_avatar_book.set_quat(
            t3d.quaternions.mat2quat(book_R).astype(float)
        )

        # Seat the avatar and hold the first writing frame.
        self.avatar.reset(body_pos, np.asarray(body_rot, dtype=np.float64))
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        # Pen near-vertical at the page: tip just above the tabletop, slightly
        # ahead of the palm along the finger direction, leaned back 18 deg.
        pen = self.neutral_avatar_props["pen"]
        hand_pos, hand_R = self.avatar.robot._get_hand_frame(1)
        finger = np.asarray(hand_R, dtype=np.float64)[:, 1].copy()
        finger[2] = 0.0
        norm = float(np.linalg.norm(finger))
        finger = finger / norm if norm > 1e-6 else np.array([forward[0], forward[1], 0.0])
        pen_fwd = float(self.NEUTRAL_WRITING_PEN_FWD)
        tip = np.array([
            float(hand_pos[0]) + finger[0] * pen_fwd,
            float(hand_pos[1]) + finger[1] * pen_fwd,
            table_top + 0.002,
        ], dtype=np.float64)
        tilt_axis = np.array([-finger[1], finger[0], 0.0], dtype=np.float64)
        pen_R = (
            t3d.axangles.axangle2mat(
                tilt_axis, np.deg2rad(float(self.NEUTRAL_WRITING_PEN_TILT_DEG))
            )
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        pen.set_pos(tip.astype(float))
        pen.set_quat(t3d.quaternions.mat2quat(pen_R).astype(float))
        self.avatar.robot.attach_object_to_hand(1, pen)
        # The pen is held for the whole clip; nothing left for the event loop.
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "writing",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = np.asarray(body_rot, dtype=np.float64)
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] writing layout side={side} "
            f"book={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Standing-laptop neutral clip
    # ------------------------------------------------------------------

    def _neutral_laptop_dz(self) -> float:
        """Vertical shift from the calibration tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_LAPTOP_TABLE_TOP_REF)

    def _load_neutral_laptop_props(self, job: dict) -> None:
        """Create the laptop, parked until layout selection.

        The laptop is fixed-base (a static scene prop); its base pose is set
        by `_prepare_neutral_laptop_layout` and its lid joint is re-held at
        the open angle every sim step by the patched step loop.
        """
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        laptop = self._neutral_load_prop_urdf(
            self._neutral_laptop_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale + 1.0]),
            scale,
        )
        self.neutral_avatar_props = {"laptop": laptop}
        self.neutral_avatar_prop = laptop
        self._neutral_laptop_hold_state = None
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _install_neutral_laptop_pingpong_subclip(self, job: dict) -> None:
        """Trim to the stationary typing window and ping-pong loop it."""
        install_laptop_pingpong_subclip(self, job)

    def _neutral_laptop_world_yaws(self, side: str) -> tuple[float, float]:
        """(laptop_yaw, avatar_yaw) in world for a table side.

        The laptop asset's display faces its local -X, so its +X axis points
        inward (away from the avatar standing outside the edge).
        """
        inward = self._NEUTRAL_WRITING_SIDE_INWARD[str(side)]
        laptop_yaw = float(np.arctan2(float(inward[1]), float(inward[0])))
        avatar_yaw = laptop_yaw - float(np.deg2rad(self.NEUTRAL_LAPTOP_YAW_LOCAL_DEG))
        return laptop_yaw, avatar_yaw

    def _neutral_laptop_footprint(self, side: str) -> dict:
        """Neutral region of the laptop motion = the laptop's region."""
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        clearance = float(self.NEUTRAL_LAPTOP_CLEARANCE)
        x0, x1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_X_BOUNDS)
        y0, y1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_Y_BOUNDS)
        laptop_yaw, _ = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(laptop_yaw)), float(np.sin(laptop_yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        corners = (rot @ np.array(
            [[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float64
        ).T).T
        min_off = corners.min(axis=0) - clearance
        max_off = corners.max(axis=0) + clearance
        # The laptop's front edge overhangs the table edge by a few cm at the
        # calibrated inset (by design, per the offline calibration).  Clamp
        # the outward extent to the edge so the recorded neutral region is the
        # on-table part and the inside-table placement check can pass.
        inset = float(self.NEUTRAL_LAPTOP_EDGE_INSET) - 1e-3
        if side == "back":
            max_off[1] = min(float(max_off[1]), inset)
        elif side == "left":
            min_off[0] = max(float(min_off[0]), -inset)
        elif side == "right":
            max_off[0] = min(float(max_off[0]), inset)
        return {
            "motion": "using_a_fax",
            "kind": "laptop",
            "min_offset": min_off,
            "max_offset": max_off,
            "size": max_off - min_off,
            "region_points": [],
        }

    def _neutral_laptop_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, laptop-origin) candidates at the fixed edge inset, center-out."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_LAPTOP_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_LAPTOP_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_LAPTOP_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        elif self._neutral_avatar_forced_motion_requested():
            known = {
                (str(side), tuple(float(v) for v in region))
                for side, region in side_regions
            }
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key not in known:
                    side_regions.append((side, region))
                    known.add(key)
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            footprint = self._neutral_laptop_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _prepare_neutral_laptop_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the standing laptop clip.

        Pose chain (all fixed): laptop origin = anchor on the table at the
        edge inset, display facing outward; avatar root = anchor -
        R(avatar_yaw) @ LAPTOP_LOCAL_XY (a fixed gap outside the table edge);
        avatar root z keeps the typing hands hovering over the keyboard.
        """
        if self._neutral_contested_requested():
            self._neutral_contested_mark_unsupported(f"kind_unsupported:{job.get('kind')}")
        self._install_neutral_laptop_pingpong_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_laptop_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_laptop_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no laptop-side candidates: table bounds/side regions leave "
                "no room for the laptop at the fixed edge inset"
            )

        # Pick the candidate that puts the laptop FARTHEST from every task
        # object/corridor region (max clearance), not just the first that
        # doesn't overlap.  A few-mm gap still lets the gripper clip the
        # laptop while it grasps/carries an adjacent object, so we maximise
        # the gap subject to a small non-overlap floor.  Different tasks have
        # different headroom (dump_bin's dense corridors leave ~9 cm; the
        # blocks tasks only ~4-5 cm), so a fixed threshold would either
        # under-protect one or starve the other — max-clearance adapts.
        min_clearance = float(self.config.get(
            "neutral_laptop_min_clearance",
            self.NEUTRAL_LAPTOP_TASK_CLEARANCE,
        ))
        debug_records = []
        scored = []  # (clearance, side, xy, footprint, record)
        for side, xy in candidates:
            footprint = self._neutral_laptop_footprint(side)
            in_table = self._neutral_footprint_inside_table(xy, footprint)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(
                    worst,
                    float(np.linalg.norm(center - clamped)) - float(radius),
                )
            clearance = worst if np.isfinite(worst) else 1.0
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": bool(in_table),
                "side_region_ok": True,
                "body_ok": True,
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored.append((clearance, side, xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[laptop_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for center, radius, label in task_regions:
                print(
                    f"[laptop_layout] region {label}: "
                    f"center={np.round(np.asarray(center, dtype=float).ravel()[:2], 3).tolist()} "
                    f"r={float(radius):.3f}"
                )
            for rec in debug_records:
                print(
                    f"[laptop_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} clear={rec['footprint_clear']} "
                    f"clearance={float(rec['prop_clearance']):.3f}"
                )
        feasible = [c for c in scored if c[0] >= min_clearance]
        pool = feasible or scored
        if not pool:
            raise RuntimeError(
                "no laptop layout candidate fits inside the table at the fixed "
                "edge inset; motion=using_a_fax"
            )
        best = max(pool, key=lambda c: c[0])
        if not feasible and bool(self.config.get("neutral_avatar_strict_layout", True)):
            raise RuntimeError(
                "no laptop layout candidate clears task objects by the minimum "
                f"clearance {min_clearance:.3f} m; best={best[0]:.3f} m; "
                "motion=using_a_fax"
            )
        _clear, side, anchor_xy0, footprint, record = best
        record["selected"] = True
        selected = (side, anchor_xy0, footprint)
        self._neutral_debug(
            f"[neutral_avatar] laptop max-clearance pick {np.asarray(anchor_xy0).round(3).tolist()} "
            f"side={side} clearance={_clear:.3f}"
        )
        side, anchor_xy, footprint = selected
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        laptop_yaw, avatar_yaw = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        laptop_local = np.asarray(self.NEUTRAL_LAPTOP_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ laptop_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)

        # Laptop on the table at the anchor, display facing the avatar; the
        # lid hold below keeps the revolute lid at the open angle.
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        laptop = self.neutral_avatar_props["laptop"]
        laptop.set_pos(np.array([
            anchor_xy[0],
            anchor_xy[1],
            table_top + float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale,
        ], dtype=float))
        laptop.set_quat(np.array(
            [np.cos(laptop_yaw / 2.0), 0.0, 0.0, np.sin(laptop_yaw / 2.0)],
            dtype=float,
        ))
        lid_qpos = None
        if getattr(laptop, "n_qs", 0):
            lid_qpos = np.full(
                (int(laptop.n_qs),),
                float(self.NEUTRAL_LAPTOP_LID_QPOS),
                dtype=np.float64,
            )
            laptop.set_qpos(lid_qpos)
            try:
                # PD-hold the lid so it stays open even before the patched
                # step loop starts re-applying the qpos every step.
                laptop.set_dofs_kp(np.full((int(laptop.n_qs),), 50.0))
                laptop.set_dofs_kv(np.full((int(laptop.n_qs),), 5.0))
                laptop.control_dofs_position(lid_qpos)
            except Exception:
                pass
        self._neutral_laptop_hold_state = (laptop, lid_qpos)

        # Stand the avatar at the fixed relative pose, holding frame 0.
        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "laptop",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] laptop layout side={side} "
            f"laptop={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Seated-typing neutral clip (chair + laptop + mouse)
    # ------------------------------------------------------------------

    def _neutral_typing_dz(self) -> float:
        """Vertical shift from the calibration tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_TYPING_TABLE_TOP_REF)

    def _neutral_typing_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_TYPING_SEAT_HEIGHT_REF) + self._neutral_typing_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _load_neutral_typing_props(self, job: dict) -> None:
        """Create the chair / laptop / mouse, parked until layout selection.

        Chair + laptop are repositioned by `_prepare_neutral_typing_layout`.
        The laptop is fixed-base (static prop) with its lid held open; the
        mouse is a static prop on the table; the chair is pinned per step.
        Nothing is held in hand (the hands just type), so there are no events.
        """
        scale = float(self.NEUTRAL_TYPING_SCALE)
        chair_scale = self._neutral_typing_chair_scale()
        laptop = self._neutral_load_prop_urdf(
            self._neutral_laptop_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale + 1.0]),
            scale,
        )
        mouse = load_object(
            self.scene,
            Pose([6.0, 6.4, 1.0], self._neutral_typing_mouse_quat(0.0)),
            "047_mouse",
            model_id=int(self.NEUTRAL_TYPING_MOUSE_SCALE_MODEL_ID),
            convex=True,
            is_static=True,
        ).entity
        self.neutral_avatar_props = {"laptop": laptop, "mouse": mouse}
        self.neutral_avatar_prop = laptop
        self.neutral_avatar_mouse = mouse
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self._neutral_laptop_hold_state = None
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _neutral_typing_mouse_quat(self, mouse_yaw: float) -> np.ndarray:
        """Map local +Y up to world +Z, then yaw about world +Z."""
        mouse_R = (
            t3d.euler.euler2mat(0.0, 0.0, float(mouse_yaw), "sxyz")
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        return t3d.quaternions.mat2quat(mouse_R).astype(float)

    def _install_neutral_typing_subclip(self, job: dict) -> None:
        """Trim the T-pose blend-in/out frames before the central loop wrap."""
        install_typing_subclip(self, job)

    def _neutral_typing_world_yaws(self, side: str) -> tuple[float, float]:
        """(laptop_yaw, avatar_yaw) in world for a table side.

        Laptop display faces its local -X, so +X points inward (toward the
        table, away from the seated avatar outside the edge).
        """
        inward = self._NEUTRAL_WRITING_SIDE_INWARD[str(side)]
        laptop_yaw = float(np.arctan2(float(inward[1]), float(inward[0])))
        avatar_yaw = laptop_yaw - float(np.deg2rad(self.NEUTRAL_TYPING_LAPTOP_YAW_LOCAL_DEG))
        return laptop_yaw, avatar_yaw

    def _neutral_typing_footprint(self, side: str) -> dict:
        """On-table neutral region = laptop AABB unioned with the mouse AABB.

        Offsets are relative to the anchor (laptop origin world xy).
        """
        scale = float(self.NEUTRAL_TYPING_SCALE)
        clearance = float(self.NEUTRAL_TYPING_CLEARANCE)
        laptop_yaw, avatar_yaw = self._neutral_typing_world_yaws(side)
        x0, x1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_X_BOUNDS)
        y0, y1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_Y_BOUNDS)
        cl, sl = float(np.cos(laptop_yaw)), float(np.sin(laptop_yaw))
        rot_l = np.array([[cl, -sl], [sl, cl]], dtype=np.float64)
        lap_corners = (rot_l @ np.array(
            [[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float64
        ).T).T
        ca, sa = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot_a = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
        mouse_off = rot_a @ (
            np.asarray(self.NEUTRAL_TYPING_MOUSE_LOCAL_XY, dtype=np.float64)
            - np.asarray(self.NEUTRAL_TYPING_LAPTOP_LOCAL_XY, dtype=np.float64)
        )
        mh = float(self.NEUTRAL_TYPING_MOUSE_HALF)
        mouse_corners = mouse_off + np.array(
            [[-mh, -mh], [-mh, mh], [mh, -mh], [mh, mh]], dtype=np.float64
        )
        corners = np.vstack([lap_corners, mouse_corners])
        min_off = corners.min(axis=0) - clearance
        max_off = corners.max(axis=0) + clearance
        # The laptop's front edge sits ~at the table edge at the calibrated
        # inset (by design, like the standing laptop kind).  Clamp the OUTWARD
        # extent (toward the avatar/edge) to the inset so the recorded region
        # is the on-table part and the inside-table check can pass.
        inset = float(self.NEUTRAL_TYPING_EDGE_INSET) - 1e-3
        if side == "back":
            max_off[1] = min(float(max_off[1]), inset)
        elif side == "left":
            min_off[0] = max(float(min_off[0]), -inset)
        elif side == "right":
            max_off[0] = min(float(max_off[0]), inset)
        return {
            "motion": "Typing",
            "kind": "typing",
            "min_offset": min_off,
            "max_offset": max_off,
            "size": max_off - min_off,
            "region_points": [],
        }

    def _neutral_typing_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, laptop-origin) candidates at the fixed edge inset, center-out."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_TYPING_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_TYPING_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_TYPING_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            footprint = self._neutral_typing_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _prepare_neutral_typing_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated typing clip.

        Pose chain (all fixed): laptop origin = anchor on the table at the
        edge inset, display facing the avatar; avatar root = anchor -
        R(avatar_yaw) @ LAPTOP_LOCAL_XY (a fixed gap outside the table edge);
        chair under the avatar; mouse on the table to the avatar's right.
        """
        if self._neutral_contested_requested():
            self._neutral_contested_mark_unsupported(f"kind_unsupported:{job.get('kind')}")
        self._install_neutral_typing_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_typing_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_typing_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no typing-side candidates: table bounds/side regions leave "
                "no room for the laptop+mouse at the fixed edge inset"
            )

        # Max-clearance anchor: farthest from every task region, subject to a
        # small non-overlap floor (same policy as the laptop kind).
        min_clearance = float(self.config.get(
            "neutral_typing_min_clearance",
            self.NEUTRAL_TYPING_TASK_CLEARANCE,
        ))
        debug_records = []
        scored = []  # (clearance, side, xy, footprint, record)
        for side, xy in candidates:
            footprint = self._neutral_typing_footprint(side)
            in_table = self._neutral_footprint_inside_table(xy, footprint)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(
                    worst,
                    float(np.linalg.norm(center - clamped)) - float(radius),
                )
            clearance = worst if np.isfinite(worst) else 1.0
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": bool(in_table),
                "side_region_ok": True,
                "body_ok": True,
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored.append((clearance, side, xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[typing_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for rec in debug_records:
                print(
                    f"[typing_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} clear={rec['footprint_clear']} "
                    f"clearance={float(rec['prop_clearance']):.3f}"
                )
        feasible = [c for c in scored if c[0] >= min_clearance]
        pool = feasible or scored
        if not pool:
            raise RuntimeError(
                "no typing layout candidate fits inside the table at the fixed "
                "edge inset; motion=Typing"
            )
        best = max(pool, key=lambda c: c[0])
        if not feasible and bool(self.config.get("neutral_avatar_strict_layout", True)):
            raise RuntimeError(
                "no typing layout candidate clears task objects by the minimum "
                f"clearance {min_clearance:.3f} m; best={best[0]:.3f} m; motion=Typing"
            )
        _clear, side, anchor_xy, footprint, record = best
        record["selected"] = True
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
        self._neutral_debug(
            f"[neutral_avatar] typing max-clearance pick {anchor_xy.round(3).tolist()} "
            f"side={side} clearance={_clear:.3f}"
        )

        laptop_yaw, avatar_yaw = self._neutral_typing_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        laptop_local = np.asarray(self.NEUTRAL_TYPING_LAPTOP_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ laptop_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        # Laptop on the table at the anchor, display facing the avatar.
        scale = float(self.NEUTRAL_TYPING_SCALE)
        laptop = self.neutral_avatar_props["laptop"]
        laptop.set_pos(np.array([
            anchor_xy[0],
            anchor_xy[1],
            table_top + float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale,
        ], dtype=float))
        laptop.set_quat(np.array(
            [np.cos(laptop_yaw / 2.0), 0.0, 0.0, np.sin(laptop_yaw / 2.0)],
            dtype=float,
        ))
        lid_qpos = None
        if getattr(laptop, "n_qs", 0):
            lid_qpos = np.full(
                (int(laptop.n_qs),), float(self.NEUTRAL_TYPING_LID_QPOS), dtype=np.float64
            )
            laptop.set_qpos(lid_qpos)
            try:
                laptop.set_dofs_kp(np.full((int(laptop.n_qs),), 50.0))
                laptop.set_dofs_kv(np.full((int(laptop.n_qs),), 5.0))
                laptop.control_dofs_position(lid_qpos)
            except Exception:
                pass
        self._neutral_laptop_hold_state = (laptop, lid_qpos)

        # Mouse on the table to the avatar's right, flat (upright + yaw).
        mouse_world = body_xy + rot2 @ np.asarray(
            self.NEUTRAL_TYPING_MOUSE_LOCAL_XY, dtype=np.float64
        )
        mouse_yaw = avatar_yaw + float(np.deg2rad(self.NEUTRAL_TYPING_MOUSE_YAW_LOCAL_DEG))
        self.neutral_avatar_mouse.set_pos(np.array([
            mouse_world[0], mouse_world[1],
            table_top + float(self.NEUTRAL_TYPING_MOUSE_Z_OFFSET),
        ], dtype=float))
        self.neutral_avatar_mouse.set_quat(self._neutral_typing_mouse_quat(mouse_yaw))

        # Chair under the avatar, wheels on the floor, backrest behind her.
        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_typing_chair_scale()
        chair_yaw = facing + np.pi  # asset faces -X
        chair.set_pos(np.array([
            body_xy[0], body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale
            + float(self.NEUTRAL_CLAP_CHAIR_Z_OFFSET),
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)], dtype=float
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            # Wheel / gas-lift joints sag over long rollouts; re-pin per step.
            # Reuse the writing chair-pin attribute so reset() restores it.
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        # Seat the avatar and hold the first typing frame.
        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)
        # Nothing is held; the hands just type.
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "typing",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] typing layout side={side} "
            f"laptop={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Seated-clapping neutral clip (chair only)
    # ------------------------------------------------------------------

    def _neutral_clap_dz(self) -> float:
        """Vertical shift from the clap review tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_CLAP_TABLE_TOP_REF)

    def _neutral_clap_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_CLAP_SEAT_HEIGHT_REF) + self._neutral_clap_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _neutral_clap_body_z_offset(self) -> float:
        return float(self.config.get(
            "neutral_clap_body_z_offset",
            self.NEUTRAL_CLAP_BODY_Z_OFFSET,
        ))

    def _neutral_clap_chair_z_offset(self) -> float:
        return float(self.config.get(
            "neutral_clap_chair_z_offset",
            self.NEUTRAL_CLAP_CHAIR_Z_OFFSET,
        ))

    def _neutral_clap_edge_anchor_local_xy(self) -> np.ndarray:
        if "neutral_clap_table_distance" in self.config:
            return np.array([
                float(self.config["neutral_clap_table_distance"]),
                0.0,
            ], dtype=np.float64)
        return np.asarray(self.NEUTRAL_CLAP_EDGE_ANCHOR_LOCAL_XY, dtype=np.float64)

    def _load_neutral_sitting_clap_props(self, job: dict) -> None:
        chair_scale = self._neutral_clap_chair_scale()
        self.neutral_avatar_props = {}
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _neutral_clap_footprint(self, side: str) -> dict:
        body_rot = np.asarray(
            self._neutral_avatar_body_rot_for_side(str(side), {"kind": "sitting_clap"}),
            dtype=np.float64,
        )
        rot2 = body_rot[:2, :2]
        points = []
        mins = []
        maxs = []
        for label, local_xy, radius in self.NEUTRAL_CLAP_LOCAL_REGION_POINTS:
            offset = rot2 @ np.asarray(local_xy, dtype=np.float64).ravel()[:2]
            radius = float(radius)
            points.append({
                "offset": offset.tolist(),
                "radius": radius,
                "label": str(label),
            })
            mins.append(offset - radius)
            maxs.append(offset + radius)
        min_offset = np.min(np.asarray(mins), axis=0)
        max_offset = np.max(np.asarray(maxs), axis=0)
        return {
            "motion": "sitting_clap",
            "kind": "sitting_clap",
            "min_offset": min_offset,
            "max_offset": max_offset,
            "size": max_offset - min_offset,
            "region_points": points,
        }

    def _neutral_clap_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_CLAP_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_CLAP_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_CLAP_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_default_side_work_regions())
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD:
                continue
            if forced_side and side != forced_side:
                continue
            footprint = self._neutral_clap_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _install_neutral_clap_subclip(self, job: dict) -> None:
        """Trim to the selected clapping period before ping-pong looping."""
        install_clap_subclip(self, job)

    def _prepare_neutral_sitting_clap_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated clapping clip."""
        self._install_neutral_clap_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_clap_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_clap_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no sitting-clap side candidates: table bounds/side regions "
                "leave no room for the seated clapping footprint"
            )

        min_clearance = float(self.config.get(
            "neutral_clap_min_clearance",
            self.NEUTRAL_CLAP_TASK_CLEARANCE,
        ))
        anchor_local = self._neutral_clap_edge_anchor_local_xy()
        robot_prefixes = tuple(self.NEUTRAL_WRITING_ROBOT_REGION_PREFIXES)
        debug_records = []
        scored = []
        scored_any = []
        for side, anchor_xy in candidates:
            anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
            footprint = self._neutral_clap_footprint(side)
            # The clapping footprint describes the seated body/hands around
            # the table edge.  Parts of that envelope intentionally live
            # outside the tabletop, unlike laptop/book props, so do not gate
            # it through the generic on-table AABB check.
            in_table = True
            min_xy = anchor_xy + np.asarray(footprint["min_offset"], dtype=np.float64)
            max_xy = anchor_xy + np.asarray(footprint["max_offset"], dtype=np.float64)
            body_rot = self._neutral_avatar_body_rot_for_side(side, job)
            rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
            body_xy = anchor_xy - rot2 @ anchor_local
            body_clear = True
            worst = float("inf")
            robot_dist = float("inf")
            for center, radius, label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                d = float(np.linalg.norm(center - clamped)) - float(radius)
                worst = min(worst, d)
                if str(label).startswith(robot_prefixes):
                    body_d = float(np.linalg.norm(center - body_xy)) - float(radius) - 0.18
                    robot_dist = min(robot_dist, d, body_d)
                    if body_d < float(self.NEUTRAL_CLAP_BODY_SWEEP_MARGIN):
                        body_clear = False
            clearance = worst if np.isfinite(worst) else 1.0
            if not np.isfinite(robot_dist):
                robot_dist = clearance
            record = {
                "xy": anchor_xy.copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": True,
                "side_region_ok": True,
                "body_ok": bool(body_clear),
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored_any.append((clearance, robot_dist, side, anchor_xy, footprint, record))
            if in_table and body_clear:
                scored.append((clearance, robot_dist, side, anchor_xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        pool = [c for c in scored if c[0] >= min_clearance] or scored
        contested_pick = None
        if pool and self._neutral_contested_requested():
            # Contested layout: same clear pool, nearest-corridor pick.
            chosen = self._neutral_contested_select([c[5] for c in pool], task_regions)
            if chosen is not None:
                contested_pick = next(c for c in pool if c[5] is chosen)
        if not pool:
            if bool(self.config.get("neutral_avatar_strict_layout", True)) or not scored_any:
                raise RuntimeError(
                    "no sitting-clap layout candidate clears task objects and the "
                    f"seated body envelope; motion={job['base_motion']}"
                )
            pool = scored_any
            self._neutral_debug(
                "[neutral_avatar] sitting_clap: body envelope grazes every "
                "candidate; using the best in-table placement"
            )
        _clear, _robot_dist, side, anchor_xy, footprint, record = (
            contested_pick if contested_pick is not None
            else max(pool, key=lambda c: (c[0], c[1]))
        )
        record["selected"] = True
        body_rot = self._neutral_avatar_body_rot_for_side(side, job)
        rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
        body_xy = anchor_xy - rot2 @ anchor_local
        body_pos = np.array([
            body_xy[0],
            body_xy[1],
            dz + self._neutral_clap_body_z_offset(),
        ], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_clap_chair_scale()
        chair_yaw = facing + np.pi
        chair.set_pos(np.array([
            body_xy[0], body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale
            + self._neutral_clap_chair_z_offset(),
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)],
            dtype=float,
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        self.avatar.reset(body_pos, np.asarray(body_rot, dtype=np.float64))
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "sitting_clap",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = np.asarray(body_rot, dtype=np.float64)
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] sitting_clap layout side={side} "
            f"anchor={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Standing phone-talker neutral clip ("talking")
    # ------------------------------------------------------------------
    # Avatar root in the table-edge anchor's frame (avatar local +x = forward,
    # toward the table).  ~0.2 m closer to the table than the laptop typist
    # (0.79575) so the standing talker reads as standing AT the table.
    NEUTRAL_PHONE_LOCAL_XY = (0.396, -0.054)   # ~0.4 m closer than laptop typist
    NEUTRAL_PHONE_BODY_Z = 0.0          # avatar root z (feet on the floor)
    NEUTRAL_PHONE_DOWN = 0.04           # lower the phone in world -z (matches the talking-motion calibration)

    def _prepare_neutral_phone_layout(self, job: dict) -> bool:
        """Fixed-pose layout for the standing 'talking on the phone' clip.

        The avatar stands beside the table (no on-table prop) holding the phone
        in the LEFT hand for the whole clip.  Reuses the laptop side/yaw
        selection to face the table, but stands ~0.2 m closer with nothing on
        the table; the phone is snapped into the palm by the event system.
        """
        if self._neutral_contested_requested():
            self._neutral_contested_mark_unsupported(f"kind_unsupported:{job.get('kind')}")
        self._install_neutral_loop_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))

        # Pick the table-edge anchor whose on-table footprint is FARTHEST from
        # every task object/corridor (max clearance) so the standing avatar
        # ends up on the side away from the robot's work — same proxy the
        # laptop layout uses, which keeps the arm out of the avatar.
        candidates = self._neutral_laptop_side_candidates(job)
        task_regions = self.neutral_task_regions()
        best = None  # (clearance, side, xy)
        for side_i, xy_i in candidates:
            fp = self._neutral_laptop_footprint(side_i)
            min_xy = np.asarray(xy_i) + fp["min_offset"]
            max_xy = np.asarray(xy_i) + fp["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(worst, float(np.linalg.norm(center - clamped)) - float(radius))
            clearance = worst if np.isfinite(worst) else 1.0
            if best is None or clearance > best[0]:
                best = (clearance, str(side_i), np.asarray(xy_i, dtype=np.float64))
        if best is not None:
            _clear, side, anchor_xy = best
        else:
            xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
            side = "back"
            anchor_xy = np.array(
                [0.5 * (xmin + xmax), float(ymax) - float(self.NEUTRAL_LAPTOP_EDGE_INSET)],
                dtype=np.float64,
            )
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        _laptop_yaw, avatar_yaw = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        local = np.asarray(self.NEUTRAL_PHONE_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ local
        body_pos = np.array(
            [body_xy[0], body_xy[1], float(self.NEUTRAL_PHONE_BODY_Z)], dtype=np.float64
        )

        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        half = 0.22
        footprint = {
            "motion": str(job["base_motion"]),
            "kind": "phone",
            "min_offset": np.array([-half, -half], dtype=np.float64),
            "max_offset": np.array([half, half], dtype=np.float64),
            "size": np.array([2 * half, 2 * half], dtype=np.float64),
            "region_points": [],
        }
        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "phone",
            "min_offset": footprint["min_offset"].tolist(),
            "max_offset": footprint["max_offset"].tolist(),
            "size": footprint["size"].tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] phone layout side={side} "
            f"anchor={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, self.neutral_task_regions())
        return True

    def _neutral_laptop_hold_lid(self) -> None:
        state = getattr(self, "_neutral_laptop_hold_state", None)
        if not state:
            return
        laptop, lid_qpos = state
        if lid_qpos is None:
            return
        try:
            laptop.set_qpos(lid_qpos)
        except Exception:
            pass
