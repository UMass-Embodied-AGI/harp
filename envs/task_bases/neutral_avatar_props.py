"""Neutral-avatar prop machinery: prop creation and identity, job
resolution, start-layout preparation, body placement, and hold-point
measurement.
"""


from __future__ import annotations

import json

import numpy as np

from ..avatar.utils import ActionStatus, AvatarState
from ..base_task import BaseTask
from ..utils import (
    ASSETS_PATH,
    Pose,
    create_primitive,
    load_mesh,
    load_object,
    to_numpy,
)
from .neutral_avatar_clip_transforms import (
    install_mouse_subclip,
    install_reverse_subclip,
    install_sponge_pingpong_subclip,
)
from .neutral_avatar_spec import (
    _NEUTRAL_MOTION_META,
    _resolve_repo_path,
)




class _NeutralAvatarPropsMixin:
    """Split from NeutralAvatarTableWorkMixin (verbatim member move); composed back into it in neutral_avatar_table_work.py."""

    def _load_neutral_avatar_region_debug_tiles(self) -> None:
        if not bool(self.config.get("neutral_avatar_debug_regions")):
            self._neutral_avatar_region_debug_tiles = {}
            return
        park = np.array([8.0, 8.0, self.TABLE_TOP_Z + 0.20], dtype=float)
        half = float(self.config.get("neutral_avatar_debug_region_tile_half", 0.025))
        count = int(self.config.get("neutral_avatar_debug_region_tile_count", 260))
        count = max(1, count)
        colors = {
            "table": (0.10, 0.25, 0.95, 0.30),
            "forbidden": (0.90, 0.10, 0.10, 0.35),
            "blocked": (1.0, 0.45, 0.05, 0.35),
            "clear": (0.05, 0.75, 0.20, 0.30),
            # Selected neutral motion swept region.
            "selected": (1.0, 0.92, 0.05, 0.45),
            # Reset-time neutral object footprint; separate from motion sweep.
            "object": (0.05, 0.85, 1.0, 0.45),
        }
        pools = {}
        for name, color in colors.items():
            pools[name] = [
                create_primitive(
                    self.scene,
                    "box",
                    Pose(park + np.array([0.0, 0.0, 0.002 * i])),
                    size={"half_size": (half, half, 0.002)},
                    color=color,
                    is_static=True,
                    collision=False,
                )
                for i in range(count)
            ]
        self._neutral_avatar_region_debug_tiles = pools

    def _create_neutral_avatar_prop(self, obj_name: str, pos: np.ndarray):
        z = float(pos[2])
        if obj_name == "sponge":
            return load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "cc0_sponge_3" / "visual" / "base0.glb",
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                scale=1.0,
                is_static=False,
                collision=True,
                friction=3.0,
                density=80.0,
            )
        if obj_name == "mouse":
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "047_mouse",
                model_id=0,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        if obj_name == "bottle" or str(obj_name).startswith("bottle_"):
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "001_bottle",
                model_id=13,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        if obj_name == "phone":
            # Held-in-hand phone for the standing "talking" neutral motion;
            # snapped into the palm by the event system, never on the table.
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "077_phone",
                model_id=0,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        actor = load_object(
            self.scene,
            Pose(
                [float(pos[0]), float(pos[1]), self._neutral_avatar_prop_center_z(obj_name)],
                self._neutral_avatar_prop_quat(obj_name),
            ),
            self._neutral_put_objects_asset(),
            model_id=self._neutral_avatar_can_model_id(obj_name),
            convex=False,
            is_static=False,
            friction=4.0,
            density=250.0,
        )
        return actor.entity

    def _neutral_put_objects_clip_defaults(self) -> dict:
        """Per-clip prop overrides baked into the motion (e.g. apples for the
        take-objects clip). Resolution order is config > clip > class default."""
        job = self._neutral_avatar_job or {}
        return dict(((job.get("clip") or {}).get("prop_defaults") or {}))

    def _neutral_put_objects_asset(self) -> str:
        defaults = self._neutral_put_objects_clip_defaults()
        return str(self.config.get(
            "neutral_put_objects_object",
            defaults.get("object", self.NEUTRAL_PUT_OBJECTS_OBJECT),
        ))

    def _neutral_avatar_can_model_id(self, obj_name: str) -> int:
        defaults = self._neutral_put_objects_clip_defaults()
        ids = list(self.config.get(
            "neutral_avatar_can_model_ids",
            defaults.get("model_ids", self.NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS),
        ))
        ids = [int(v) for v in ids]
        if not ids:
            ids = list(self.NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS)
        cache = getattr(self, "_neutral_avatar_can_model_ids", None)
        if cache is None:
            cache = {}
            self._neutral_avatar_can_model_ids = cache
        key = str(obj_name)
        if key not in cache:
            seed = int(getattr(self, "_neutral_avatar_reset_seed", 0))
            obj_offset = 1 if key == "object_1" else 2
            rng = np.random.default_rng(seed + 80409 + obj_offset * 1009)
            cache[key] = ids[int(rng.integers(0, len(ids)))]
        return int(cache[key])

    def _neutral_avatar_can_body_radius(self, obj_name: str) -> float:
        model_id = self._neutral_avatar_can_model_id(obj_name)
        md_path = ASSETS_PATH / "objects" / self._neutral_put_objects_asset() / f"model_data{model_id}.json"
        try:
            with open(md_path) as f:
                md = json.load(f)
            extents = np.asarray(md.get("extents", [1.6, 2.0, 1.6]), dtype=np.float64)
            scale = np.asarray(md.get("scale", [0.05, 0.05, 0.05]), dtype=np.float64)
            if scale.size == 1:
                scale = np.repeat(float(scale[0]), 3)
            return float(max(extents[0] * scale[0], extents[2] * scale[2]) * 0.5)
        except Exception:
            return float(self._neutral_prop_radius(obj_name))

    def _neutral_avatar_prebuild_prop_xy(
        self, job: dict, work_xy: np.ndarray, obj_name: str, obj_idx: int
    ) -> np.ndarray:
        """Keep reset-time neutral props stable before layout selection.

        Neutral props are created in ``load_actors()``, before BaseTask's
        initial settle and before ``_prepare_neutral_avatar_start_layout()``
        selects the final work anchor.  Clamp these temporary positions to the
        physical table footprint so bottle/sponge clips cannot start with a
        prop hanging over an edge and falling before the neutral layout pass.
        """
        xy = np.asarray(work_xy, dtype=np.float64).ravel()[:2].copy()
        prop_margin = self._neutral_avatar_table_margin_for_prop(job, obj_name)
        if job.get("kind") != "put_objects_in_bowl":
            return self._neutral_avatar_clamp_table_xy(xy, margin=prop_margin)
        offsets = (
            np.array([-0.075, 0.000], dtype=np.float64),
            np.array([0.075, 0.000], dtype=np.float64),
            np.array([0.000, 0.095], dtype=np.float64),
            np.array([0.000, -0.095], dtype=np.float64),
        )
        xy = xy + offsets[int(obj_idx) % len(offsets)]
        return self._neutral_avatar_clamp_table_xy(
            xy,
            margin=max(float(self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN), prop_margin),
        )

    @staticmethod
    def _neutral_object_identity_tokens(value) -> set[str]:
        """Return comparable identity tokens for a target or neutral prop.

        Catalog entries may include a model suffix (``035_apple@1``), while
        motion metadata usually uses a short noun (``apple``).  Keeping both
        the full asset ID and its numeric-prefix-free noun makes those forms
        compare equal without conflating distinct objects such as an oil
        bottle and the generic water bottle.
        """
        if value is None:
            return set()
        raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        raw = raw.split("@", 1)[0]
        if not raw:
            return set()
        tokens = {raw}
        prefix, sep, tail = raw.partition("_")
        if sep and prefix.isdigit() and tail:
            tokens.add(tail)
        return tokens

    def _neutral_clip_object_ids(self, clip: dict) -> set[str]:
        """Actual prop identities used by one neutral-motion clip."""
        handles = {
            str(event.get("object"))
            for event in (clip.get("events") or [])
            if event.get("object") is not None
        }
        handles.update(str(obj) for obj in (clip.get("objects") or []))

        kind = str(clip.get("kind") or "")
        if not kind and any(
            str(event.get("type")) == "pick"
            for event in (clip.get("events") or [])
        ):
            kind = "put_objects_in_bowl"
        if kind == "writing":
            handles.update(("pen", "notebook", "chair"))
        elif kind == "laptop":
            handles.add("laptop")
        elif kind == "typing":
            handles.update(("laptop", "mouse", "chair"))
        elif kind == "sitting_clap":
            handles.add("chair")

        prop_defaults = dict(clip.get("prop_defaults") or {})
        if kind == "put_objects_in_bowl" and bool(self.config.get(
            "neutral_put_objects_plate",
            prop_defaults.get("use_plate", True),
        )):
            handles.add("plate")
        put_objects_asset = str(self.config.get(
            "neutral_put_objects_object",
            prop_defaults.get("object", self.NEUTRAL_PUT_OBJECTS_OBJECT),
        ))
        identities: set[str] = set()
        for handle in handles:
            if handle.startswith("object_"):
                asset_id = put_objects_asset
            elif handle.startswith("bottle_"):
                asset_id = self.NEUTRAL_PROP_ASSET_IDS["bottle"]
            else:
                asset_id = self.NEUTRAL_PROP_ASSET_IDS.get(handle, handle)
            identities.update(self._neutral_object_identity_tokens(asset_id))
        return identities

    def _neutral_robot_target_object_ids(self) -> set[str]:
        """Collect identities of the robot targets spawned for this episode.

        Neutral task families expose targets in a few established shapes:
        catalog ``_target_entry`` for single-object tasks, sampled dump specs,
        categorize ``_sort_targets``, and food ``(actor, FoodSpec)`` pairs.
        This is intentionally episode-specific, so a pooled task only loses a
        motion when its sampled target actually conflicts.
        """
        identities: set[str] = set()

        def add(value) -> None:
            identities.update(self._neutral_object_identity_tokens(value))

        entry = getattr(self, "_target_entry", None)
        if entry is not None:
            add(getattr(entry, "key", None))
            add(getattr(entry, "object_id", None))

        for spec in (getattr(self, "_episode_dump_specs", None) or []):
            if isinstance(spec, dict):
                add(spec.get("object_id") or spec.get("key"))

        for target in (getattr(self, "_sort_targets", None) or []):
            if isinstance(target, (tuple, list)) and len(target) >= 2:
                add(target[1])

        for item in (getattr(self, "food_actors", None) or []):
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            spec = item[1]
            add(getattr(spec, "asset_id", None))
            add(getattr(spec, "object_id", None))

        return identities

    def _ensure_neutral_avatar_job_is_target_distinct(self) -> None:
        """Enforce that the avatar's props differ from robot target objects."""
        job = self._neutral_avatar_job or {}
        targets = self._neutral_robot_target_object_ids()
        conflicts = targets & self._neutral_clip_object_ids(job.get("clip") or {})
        if not conflicts:
            return
        motion = str(job.get("base_motion") or job.get("motion") or "")
        if self._neutral_avatar_forced_motion_requested():
            raise ValueError(
                f"neutral_avatar_motion={motion!r} duplicates robot target "
                f"object(s) {sorted(conflicts)}; choose a distinct neutral motion"
            )
        self._resolve_neutral_avatar_job(excluded_target_ids=targets)

    def _resolve_neutral_avatar_job(self, excluded_target_ids: set[str] | None = None):
        meta_path = _resolve_repo_path(self.config.get("neutral_avatar_motion_meta", _NEUTRAL_MOTION_META))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clips = meta["clips"]
        forced = self.config.get("neutral_avatar_motion")
        keys = sorted(clips)
        if forced:
            forced = str(forced)
            forced = str(self.NEUTRAL_AVATAR_MOTION_ALIASES.get(forced, forced))
            matches = [k for k in keys if k == forced or k.startswith(f"{forced}_")]
            if not matches:
                raise ValueError(
                    "neutral_avatar_motion must be a clip name/prefix or one "
                    f"of aliases {sorted(self.NEUTRAL_AVATAR_MOTION_ALIASES)}; "
                    f"got {forced!r}. Available clips: {keys}"
                )
            motion_name = matches[0]
        else:
            choices = list(self.config.get(
                "neutral_avatar_motion_choices",
                self.NEUTRAL_AVATAR_MOTION_CHOICES,
            ))
            # Resolve aliases (e.g. the retired ``wipe_table_1``) like the
            # forced branch does, instead of silently dropping them.
            # Duplicates collapse so each distinct clip keeps equal weight.
            resolved: list[str] = []
            for name in choices:
                name = str(self.NEUTRAL_AVATAR_MOTION_ALIASES.get(str(name), str(name)))
                if name in clips and name not in resolved:
                    resolved.append(name)
            choices = resolved
            if not choices:
                choices = keys
            if excluded_target_ids:
                choices = [
                    name for name in choices
                    if not (
                        set(excluded_target_ids)
                        & self._neutral_clip_object_ids(clips[name])
                    )
                ]
                if not choices:
                    raise ValueError(
                        "no neutral avatar motion remains after excluding robot "
                        f"target object(s) {sorted(excluded_target_ids)}"
                    )
                # A private generator avoids perturbing scene/layout RNG after
                # the parent task has already spawned its episode targets.
                seed = int(getattr(self, "_neutral_avatar_reset_seed", 0)) + 904_219
                rng = np.random.default_rng(seed)
                motion_name = choices[int(rng.integers(len(choices)))]
            else:
                motion_name = choices[int(np.random.randint(len(choices)))]
        clip = clips[motion_name]
        placement = dict(clip.get("neutral_placement", {}) or {})
        events = [dict(e) for e in clip["events"]]
        objects = []
        for event in events:
            obj = str(event["object"])
            if obj not in objects:
                objects.append(obj)
        if not objects:
            objects = [str(o) for o in (clip.get("objects") or [])]
        if clip.get("kind") is not None:
            kind = str(clip["kind"])
        else:
            kind = "put_objects_in_bowl" if any(e["type"] == "pick" for e in events) else objects[0]
        self._neutral_avatar_job = {
            "motion": motion_name,
            "base_motion": motion_name,
            "clip": clip,
            "events": events,
            "objects": objects,
            "kind": kind,
            "placement": placement,
        }

    def _neutral_avatar_task_placement(self, job: dict | None = None) -> dict:
        job = self._neutral_avatar_job if job is None else job
        if not job:
            return {}
        return dict((job.get("placement") or {}).get("task", {}) or {})

    def _neutral_avatar_raw_preview_adjustment(self, job: dict | None = None) -> dict:
        job = self._neutral_avatar_job if job is None else job
        if not job:
            return {}
        return dict((job.get("placement") or {}).get("raw_preview_adjustment", {}) or {})

    def _install_neutral_avatar_subclip(self, job: dict) -> None:
        install_mouse_subclip(self, job)

    def _install_neutral_avatar_sponge_pingpong_subclip(self, job: dict) -> None:
        install_sponge_pingpong_subclip(self, job)

    def _install_neutral_reverse_subclip(self, job: dict) -> None:
        """Time-reverse the put-objects clip into a take-objects-out clip.

        The ``take_objects_from_bowl`` motion reuses the put-objects-in-bowl
        machinery but plays the source frames backward, so the avatar lifts
        the two cans out of the plate and sets them down on the table.  The
        clip's events are authored directly in reversed-frame indexing with
        swapped attach/detach (``pick`` at the plate, ``put`` on the table),
        so this method only builds the reversed ``motion_data`` entry and
        repoints the job at it.  No-op unless the resolved clip declares
        ``reverse: true``.
        """
        install_reverse_subclip(self, job)

    def _neutral_bowl_event_type(self, job: dict) -> str:
        """Event type whose hand position anchors the plate/bowl.

        Forward put-objects releases the cans into the plate (``put``); the
        reversed take-objects clip lifts them out (``pick``).  The plate sits
        at that event's hold position in either direction.
        """
        clip = (job or {}).get("clip", {}) or {}
        return "pick" if bool(clip.get("reverse")) else "put"

    def _neutral_avatar_default_work_xy(self, job: dict) -> np.ndarray:
        if job["kind"] == "sponge" and self.NEUTRAL_SPONGE_WORK_XY is not None:
            return np.asarray(self.NEUTRAL_SPONGE_WORK_XY, dtype=np.float64)
        if job["kind"] == "mouse":
            return np.asarray(self.NEUTRAL_MOUSE_WORK_XY, dtype=np.float64)
        xy = np.asarray(self.NEUTRAL_WORK_XY, dtype=np.float64)
        if job["kind"] == "put_objects_in_bowl":
            xy = xy + np.asarray(self.NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY, dtype=np.float64)
        return xy

    def _neutral_avatar_work_candidate_shift_xy(self, job: dict) -> np.ndarray:
        if job["kind"] == "put_objects_in_bowl":
            return np.asarray(self.NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY, dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

    def _neutral_avatar_work_xy(self, job: dict) -> np.ndarray:
        if self._neutral_avatar_selected_work_xy is not None:
            return np.asarray(self._neutral_avatar_selected_work_xy, dtype=np.float64)
        return self._neutral_avatar_default_work_xy(job)

    def _neutral_zero_entity_velocity(self, entity) -> None:
        try:
            entity.set_dofs_velocity(np.zeros(6, dtype=float))
        except Exception:
            pass

    def _neutral_debug(self, message: str) -> None:
        return None

    def _neutral_avatar_body_offset(self, job: dict) -> np.ndarray:
        offset = np.zeros(3, dtype=np.float64)
        offset[2] += float(self.NEUTRAL_AVATAR_BODY_Z_OFFSET)
        # Extra body raise: config > clip default > 0 (no effect elsewhere).
        offset[2] += float(self.config.get(
            "neutral_avatar_body_z_offset",
            self._neutral_put_objects_clip_defaults().get("avatar_body_z_offset", 0.0),
        ))
        task_placement = self._neutral_avatar_task_placement(job)
        offset[2] += float(task_placement.get("avatar_body_z_offset", 0.0))
        if job.get("kind") == "bottle":
            offset += np.asarray(
                task_placement.get(
                    "bottle_body_extra_offset",
                    self.NEUTRAL_AVATAR_BOTTLE_EXTRA_BODY_OFFSET,
                ),
                dtype=np.float64,
            )
            offset[2] += float(task_placement.get(
                "bottle_body_z_offset",
                self.NEUTRAL_AVATAR_BOTTLE_BODY_Z_OFFSET,
            ))
        if job.get("kind") == "mouse":
            offset[2] += float(task_placement.get(
                "mouse_body_z_offset",
                self.NEUTRAL_AVATAR_MOUSE_BODY_Z_OFFSET,
            ))
        if job["kind"] == "put_objects_in_bowl":
            offset[2] += float(self.NEUTRAL_PUT_OBJECTS_AVATAR_TABLE_Z_OFFSET)
        return offset

    def _prepare_neutral_avatar_start_layout(self) -> bool:
        if self.avatar is None or getattr(self, "_neutral_avatar_layout_ready", False):
            return False
        job = self._neutral_avatar_job
        if not job:
            return False

        if job["kind"] == "writing":
            return self._prepare_neutral_writing_layout(job)

        if job["kind"] == "laptop":
            return self._prepare_neutral_laptop_layout(job)

        if job["kind"] == "phone":
            return self._prepare_neutral_phone_layout(job)

        if job["kind"] == "typing":
            return self._prepare_neutral_typing_layout(job)

        if job["kind"] == "sitting_clap":
            return self._prepare_neutral_sitting_clap_layout(job)

        if job["kind"] == "mouse":
            self._install_neutral_avatar_subclip(job)
        if job["kind"] == "sponge":
            self._install_neutral_avatar_sponge_pingpong_subclip(job)
        if job["kind"] == "put_objects_in_bowl":
            self._install_neutral_reverse_subclip(job)

        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        if (
            job["kind"] == "put_objects_in_bowl"
            and self._neutral_avatar_motion_footprint is None
        ):
            prelim_xy = self._neutral_avatar_default_work_xy(job)
            prelim_rot = self._neutral_avatar_body_rot_for_work_xy(job, prelim_xy)
            self._neutral_avatar_measure_body_rot = prelim_rot
            _prelim_hold = self._neutral_avatar_measure_hold_points(job)
            self._neutral_avatar_motion_footprint = self._neutral_avatar_compute_motion_footprint(job)
            self._neutral_avatar_measure_body_rot = None
        candidates = self._neutral_avatar_work_candidates(job)
        task_regions = self.neutral_task_regions()
        forbidden = task_regions
        prop_forbidden = self._neutral_avatar_prop_forbidden_regions(task_regions)
        selected = None
        debug_records = []
        candidate_states = []
        pose_eval_cache: dict[tuple[float, ...], tuple[dict[int, np.ndarray], dict]] = {}
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        def candidate_poses():
            for xy in candidates:
                yield xy, None
            if contested and contested_valid:
                # A primary candidate passed: legacy never reaches recovery.
                return
            if self._neutral_avatar_use_layout_recovery(job):
                yield from self._neutral_avatar_fallback_work_candidates(job, pose_eval_cache, task_regions)

        # Contested layout: on contested seeds keep scanning the
        # PRIMARY candidates (already generated, so no extra np.random draws)
        # and choose among every valid one afterwards.  Recovery candidates
        # keep the legacy first-fit order so their lazy generation is unchanged.
        contested = self._neutral_contested_requested()
        contested_valid: list[dict] = []
        for xy, fallback_rot in candidate_poses():
            body_rot_i = (self._neutral_avatar_body_rot_for_work_xy(job, xy)
                          if fallback_rot is None else fallback_rot)
            if fallback_rot is not None:
                self._neutral_avatar_measure_body_rot = body_rot_i
            rot_key = tuple(np.round(np.asarray(body_rot_i, dtype=np.float64).ravel(), 6))
            cached_eval = pose_eval_cache.get(rot_key)
            if cached_eval is None:
                self._neutral_avatar_measure_body_rot = body_rot_i
                hold_i = self._neutral_avatar_measure_hold_points(job)
                footprint_i = self._neutral_avatar_compute_motion_footprint(job)
                pose_eval_cache[rot_key] = (hold_i, footprint_i)
            else:
                hold_i, footprint_i = cached_eval
            body_pos_i = self._neutral_avatar_body_pos_for_work_xy(
                job, xy, hold_i,
            )
            footprint_clear = self._neutral_footprint_is_clear(xy, footprint_i, forbidden)
            neutral_region_clear = self._neutral_motion_region_is_clear(xy, footprint_i, task_regions)
            body_ok = self._neutral_avatar_body_pose_is_valid(job, xy, body_pos_i)
            side_fn = getattr(self, "_neutral_sponge_side_for_xy", None)
            side = side_fn(xy) if side_fn is not None else self._neutral_avatar_side_for_work_xy(job, xy)
            side_region_ok = self._neutral_footprint_inside_side_region(xy, footprint_i, side)
            footprint_table_ok = self._neutral_footprint_inside_table(xy, footprint_i)
            prop_clearance = self._neutral_initial_props_clearance(
                job, xy, body_pos_i, hold_i, prop_forbidden
            )
            prop_table_clearance = self._neutral_initial_props_table_clearance(
                job, xy, body_pos_i, hold_i
            )
            candidate_state = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": str(side),
                "footprint": footprint_i,
                "footprint_clear": bool(footprint_clear),
                "neutral_region_clear": bool(neutral_region_clear),
                "footprint_table_ok": bool(footprint_table_ok),
                "side_region_ok": bool(side_region_ok),
                "body_ok": bool(body_ok),
                "prop_clearance": float(prop_clearance),
                "prop_table_clearance": float(prop_table_clearance),
                "selected": False,
                "selected_tuple": (xy, body_rot_i, hold_i, footprint_i, body_pos_i),
            }
            candidate_states.append(candidate_state)
            debug_records.append(candidate_state)
            use_region_placement = bool(self.config.get(
                "neutral_use_region_placement",
                self.NEUTRAL_USE_REGION_PLACEMENT,
            ))
            placement_clear = neutral_region_clear if use_region_placement else footprint_clear
            placement_side_ok = side_region_ok
            props_ok = prop_clearance >= 0.0 and prop_table_clearance >= 0.0
            if placement_clear and placement_side_ok and footprint_table_ok and body_ok and props_ok:
                if contested and fallback_rot is None:
                    contested_valid.append(candidate_state)
                    continue
                debug_records[-1]["selected"] = True
                selected = (xy, body_rot_i, hold_i, footprint_i, body_pos_i)
                side_text = f" side={side}" if side else ""
                self._neutral_debug(
                    f"[neutral_avatar] selected work_xy={xy.round(3).tolist()} "
                    f"for {job['base_motion']}{side_text}"
                )
                break
        if contested and selected is None and contested_valid:
            chosen = self._neutral_contested_select(contested_valid, task_regions)
            if chosen is None:
                # No candidate reaches the corridor band: keep the legacy
                # first-fit choice (contest_failed is recorded in metrics).
                chosen = contested_valid[0]
            chosen["selected"] = True
            selected = chosen["selected_tuple"]
            self._neutral_debug(
                f"[neutral_avatar] selected work_xy={np.asarray(chosen['xy']).round(3).tolist()} "
                f"for {job['base_motion']} side={chosen.get('side')} "
                f"(contested={self._neutral_contested_state.get('applied')}, "
                f"gap={chosen.get('contest_gap')})"
            )
        elif contested and selected is None:
            self._neutral_contested_mark_unsupported("no_valid_candidate")
        elif contested and not contested_valid:
            # No primary candidate was valid; the legacy first-fit recovery
            # candidate was taken, so the layout is not contested.
            self._neutral_contested_mark_unsupported("recovery_candidate_only")
        if selected is None:
            if not candidate_states and job.get("kind") in ("sponge", "mouse", "bottle"):
                for xy in self._neutral_avatar_work_candidates(job):
                    body_rot_i = self._neutral_avatar_body_rot_for_work_xy(job, xy)
                    self._neutral_avatar_measure_body_rot = body_rot_i
                    hold_i = self._neutral_avatar_measure_hold_points(job)
                    footprint_i = self._neutral_avatar_compute_motion_footprint(job)
                    body_pos_i = self._neutral_avatar_body_pos_for_work_xy(job, xy, hold_i)
                    side = self._neutral_avatar_side_for_work_xy(job, xy)
                    candidate_state = {
                        "xy": np.asarray(xy, dtype=np.float64).copy(),
                        "side": str(side),
                        "footprint": footprint_i,
                        "footprint_clear": bool(self._neutral_footprint_is_clear(xy, footprint_i, forbidden)),
                        "neutral_region_clear": bool(self._neutral_motion_region_is_clear(xy, footprint_i, task_regions)),
                        "footprint_table_ok": bool(self._neutral_footprint_inside_table(xy, footprint_i)),
                        "side_region_ok": bool(self._neutral_footprint_inside_side_region(xy, footprint_i, side)),
                        "body_ok": bool(self._neutral_avatar_body_pose_is_valid(job, xy, body_pos_i)),
                        "prop_clearance": float(self._neutral_initial_props_clearance(
                            job, xy, body_pos_i, hold_i, prop_forbidden
                        )),
                        "prop_table_clearance": float(self._neutral_initial_props_table_clearance(
                            job, xy, body_pos_i, hold_i
                        )),
                        "selected": False,
                        "selected_tuple": (xy, body_rot_i, hold_i, footprint_i, body_pos_i),
                    }
                    candidate_states.append(candidate_state)
                    debug_records.append(candidate_state)
                self._neutral_avatar_measure_body_rot = None
            self._neutral_avatar_region_debug_records = debug_records
            safe_prop = [
                rec for rec in candidate_states
                if (
                    float(rec.get("prop_clearance", -1e9)) >= 0.0
                    and float(rec.get("prop_table_clearance", -1e9)) >= 0.0
                    and bool(rec.get("footprint_table_ok"))
                    and bool(rec.get("side_region_ok"))
                    and bool(rec.get("body_ok"))
                    and (
                        (job.get("kind") not in ("sponge", "mouse", "bottle")
                         and not (getattr(self, "BLOCK_NEUTRAL_LAYOUT_RECOVERY", False)
                                  and job.get("kind") == "put_objects_in_bowl"))
                        or bool(rec.get("neutral_region_clear"))
                    )
                )
            ]
            strict_layout = bool(self.config.get("neutral_avatar_strict_layout", True))
            if (
                strict_layout
                and job.get("kind") == "put_objects_in_bowl"
                and not safe_prop
            ):
                raise RuntimeError(
                    "no neutral can layout candidate clears task objects, "
                    "table bounds, and the robot gripper region"
                )
            if strict_layout and job.get("kind") in ("sponge", "mouse", "bottle") and not safe_prop:
                raise RuntimeError(
                    "no neutral layout candidate clears task objects, table bounds, "
                    f"and avatar body edge gap; motion={job['base_motion']} "
                    f"kind={job['kind']}"
                )
            if safe_prop:
                by_side: dict[str, list[dict]] = {}
                for rec in safe_prop:
                    by_side.setdefault(str(rec.get("side", "")), []).append(rec)
                sides = sorted(by_side)
                seed = int(getattr(self, "_neutral_avatar_reset_seed", 0))
                side = sides[seed % len(sides)]
                group = by_side[side]
                best = group[(seed // max(1, len(sides))) % len(group)]
            else:
                best = max(
                    candidate_states,
                    key=lambda rec: (
                        float(rec.get("prop_clearance", -1e9)) >= 0.0,
                        bool(rec.get("side_region_ok")),
                        bool(rec.get("body_ok")),
                        bool(rec.get("footprint_table_ok")),
                        bool(rec.get("footprint_clear")),
                        float(rec.get("prop_clearance", -1e9)),
                        float(rec.get("prop_table_clearance", -1e9)),
                    ),
                )
            best["selected"] = True
            selected = best["selected_tuple"]
            xy = selected[0]
            self._neutral_debug(
                f"[neutral_avatar] WARNING: no clear neutral candidate for "
                f"{job['base_motion']}; using {xy.round(3).tolist()} "
                f"(prop_clearance={float(best.get('prop_clearance', 0.0)):.3f})"
            )
        self._neutral_avatar_measure_body_rot = None
        self._neutral_avatar_region_debug_records = debug_records
        if job.get("kind") == "put_objects_in_bowl":
            xy, body_rot, hold_by_frame, footprint, body_pos = selected
            if not self._neutral_put_objects_footprint_is_table_safe(xy, footprint):
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                margin = float(
                    getattr(
                        self,
                        "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                        self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
                    )
                )
                min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
                max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
                lo = np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64) - min_offset
                hi = np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64) - max_offset
                if np.all(lo <= hi):
                    xy = np.clip(np.asarray(xy, dtype=np.float64).ravel()[:2], lo, hi)
                    body_pos = self._neutral_avatar_body_pos_for_work_xy(job, xy, hold_by_frame)
                    selected = (xy, body_rot, hold_by_frame, footprint, body_pos)
        (
            self._neutral_avatar_selected_work_xy,
            body_rot,
            hold_by_frame,
            footprint,
            body_pos,
        ) = selected
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": str(footprint["kind"]),
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [
                float(self._neutral_avatar_selected_work_xy[0]),
                float(self._neutral_avatar_selected_work_xy[1]),
                float(self.TABLE_TOP_Z),
            ],
            dtype=np.float64,
        )

        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_hold = hold_by_frame[int(first_event["clip_frame"])]
        first_target = np.array(
            [
                float(self._neutral_avatar_work_xy(job)[0]),
                float(self._neutral_avatar_work_xy(job)[1]),
                float(self._neutral_avatar_prop_center_z(str(first_event["object"]))),
            ],
            dtype=np.float64,
        )
        shift = first_target - first_hold
        body_rot = np.asarray(body_rot, dtype=np.float64).copy()

        self._neutral_avatar_body_rot = body_rot
        self._place_neutral_avatar_props(job, hold_by_frame, shift, body_pos=body_pos)
        self.avatar.reset(body_pos, body_rot)
        settle_steps = int(self.config.get("settle_steps", BaseTask.SETTLE_STEPS))
        for _ in range(max(0, settle_steps)):
            self.step_sim()
        if not bool(self.config.get("neutral_avatar_skip_reset_props_upright", False)):
            self._neutral_reset_can_props_upright(job)
        self.avatar.reset(body_pos, body_rot)
        if not bool(self.config.get("neutral_avatar_skip_hold_motion_frame_idle", False)):
            self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._create_neutral_avatar_region_debug_overlays(job, forbidden)
        return True

    def _neutral_avatar_body_rot_for_work_xy(self, job: dict, xy: np.ndarray) -> np.ndarray:
        task_placement = self._neutral_avatar_task_placement(job)
        raw_adjustment = self._neutral_avatar_raw_preview_adjustment(job)
        side = self._neutral_avatar_side_for_work_xy(job, xy)
        if side:
            return self._neutral_avatar_body_rot_for_side(side, job)
        if bool(task_placement.get("use_raw_preview_avatar_rotation", False)):
            if raw_adjustment.get("avatar_face") == "image_down":
                return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
            return np.eye(3, dtype=np.float64)
        return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()

    def _neutral_avatar_motion_visual_yaw_offset(self, job: dict | None = None) -> float:
        if job and job.get("kind") == "sponge":
            return np.pi / 2.0
        if job and job.get("kind") == "put_objects_in_bowl":
            return np.pi / 2.0
        if job and job.get("kind") == "sitting_clap":
            if "neutral_clap_visual_yaw_offset_deg" in self.config:
                return float(np.deg2rad(self.config["neutral_clap_visual_yaw_offset_deg"]))
            return 0.0
        return 0.0

    def _neutral_avatar_body_rot_for_side(self, side: str, job: dict | None = None) -> np.ndarray:
        if side == "back":
            yaw = 0.0
        elif side == "left":
            yaw = np.pi / 2.0
        elif side == "right":
            yaw = -np.pi / 2.0
        else:
            return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
        yaw += self._neutral_avatar_motion_visual_yaw_offset(job)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rz = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rz @ np.asarray(self.avatar_init_rot, dtype=np.float64)

    def _neutral_avatar_body_pos_for_work_xy(
        self, job: dict, xy: np.ndarray, hold_by_frame: dict[int, np.ndarray]
    ) -> np.ndarray:
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
        body_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
            + shift
            + self._neutral_avatar_body_offset(job)
        )
        task_placement = self._neutral_avatar_task_placement(job)
        if "avatar_root_table_z_offset" in task_placement:
            body_pos[2] = (
                float(self.TABLE_TOP_Z)
                + float(task_placement["avatar_root_table_z_offset"])
            )
        return body_pos

    def _neutral_avatar_clamp_body_edge_gap(
        self, body_pos: np.ndarray, target_xy: np.ndarray, side: str | None = None
    ) -> np.ndarray:
        body_pos = np.asarray(body_pos, dtype=np.float64).copy()
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        max_gap = float(self.config.get(
            "neutral_avatar_max_table_edge_gap",
            self.NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP,
        ))
        min_gap = float(self.config.get(
            "neutral_avatar_min_table_edge_gap",
            self.NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP,
        ))
        if max_gap <= 0.0 and min_gap <= 0.0:
            return body_pos
        max_gap = max(float(max_gap), float(min_gap))
        side = str(side or "")
        if side == "left":
            edge = 0
        elif side == "right":
            edge = 1
        elif side == "back":
            edge = 3
        else:
            dists = np.array([
                target[0] - float(xmin),
                float(xmax) - target[0],
                target[1] - float(ymin),
                float(ymax) - target[1],
            ], dtype=np.float64)
            edge = int(np.argmin(dists))
        if edge == 0:
            body_pos[0] = float(np.clip(body_pos[0], float(xmin) - max_gap, float(xmin) - min_gap))
        elif edge == 1:
            body_pos[0] = float(np.clip(body_pos[0], float(xmax) + min_gap, float(xmax) + max_gap))
        elif edge == 2:
            body_pos[1] = float(np.clip(body_pos[1], float(ymin) - max_gap, float(ymin) - min_gap))
        elif edge == 3:
            body_pos[1] = float(np.clip(body_pos[1], float(ymax) + min_gap, float(ymax) + max_gap))
        return body_pos

    def _neutral_avatar_body_edge_gap_is_valid(
        self, side: str, target_xy: np.ndarray, body_xy: np.ndarray
    ) -> bool:
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        body = np.asarray(body_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        min_gap = float(self.config.get(
            "neutral_avatar_min_table_edge_gap",
            self.NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP,
        ))
        if side == "back":
            gap = body[1] - float(ymax)
            return bool(min_gap <= gap and target[1] - body[1] < -0.18)
        if side == "right":
            gap = body[0] - float(xmax)
            return bool(min_gap <= gap and target[0] - body[0] < -0.18)
        if side == "left":
            gap = float(xmin) - body[0]
            return bool(min_gap <= gap and target[0] - body[0] > 0.18)
        return bool(
            body[0] <= float(xmin) - min_gap
            or body[0] >= float(xmax) + min_gap
            or body[1] <= float(ymin) - min_gap
            or body[1] >= float(ymax) + min_gap
        )

    def _neutral_avatar_push_body_outside_table(
        self, body_pos: np.ndarray, target_xy: np.ndarray, edge_gap: float
    ) -> np.ndarray:
        body_pos = np.asarray(body_pos, dtype=np.float64).copy()
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        dists = np.array([
            target[0] - float(xmin),
            float(xmax) - target[0],
            target[1] - float(ymin),
            float(ymax) - target[1],
        ], dtype=np.float64)
        edge = int(np.argmin(dists))
        gap = float(edge_gap)
        if edge == 0:
            body_pos[0] = float(xmin) - gap
        elif edge == 1:
            body_pos[0] = float(xmax) + gap
        elif edge == 2:
            body_pos[1] = float(ymin) - gap
        else:
            body_pos[1] = float(ymax) + gap
        return body_pos

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge") and not bool(self.config.get(
            "neutral_avatar_require_surface_body_edge_gap",
            False,
        )):
            return True
        if job.get("kind") == "put_objects_in_bowl":
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            side = self._neutral_avatar_side_for_work_xy(job, xy)
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            return self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
        if job.get("kind") == "bottle":
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            side = self._neutral_avatar_side_for_work_xy(job, xy)
            return bool(
                self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
                and np.linalg.norm(target - body) > 0.18
            )
        side = self._neutral_avatar_side_for_work_xy(job, xy)
        if side:
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            return self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
        return True

    def _create_neutral_avatar_region_debug_overlays(
        self, job: dict, forbidden: list[tuple[np.ndarray, float, str]]
    ) -> None:
        if not bool(self.config.get("neutral_avatar_debug_regions")):
            return
        base_z = float(self.TABLE_TOP_Z + 0.020)
        # Top-down region priority is encoded by height so overlapping tiles
        # render deterministically. Requested order:
        # red forbidden > yellow selected > orange rejected > blue table.
        # Green clear-but-not-selected stays below orange and above blue.
        layer_z = {
            "table": base_z,
            "clear": base_z + 0.010,
            "blocked": base_z + 0.020,
            "selected": base_z + 0.030,
            "object": base_z + 0.035,
            "forbidden": base_z + 0.040,
        }
        pools = getattr(self, "_neutral_avatar_region_debug_tiles", {}) or {}
        used = {name: 0 for name in pools}
        tile_half = float(self.config.get("neutral_avatar_debug_region_tile_half", 0.025))
        step = 2.0 * tile_half
        park = np.array([8.0, 8.0, base_z], dtype=float)

        for pool in pools.values():
            for ent in pool:
                ent.set_pos(park)

        def add_region(min_xy, max_xy, pool_name: str):
            pool = pools.get(pool_name)
            if not pool:
                return
            min_xy = np.asarray(min_xy, dtype=np.float64)
            max_xy = np.asarray(max_xy, dtype=np.float64)
            xs = np.arange(float(min_xy[0]) + tile_half, float(max_xy[0]), step)
            ys = np.arange(float(min_xy[1]) + tile_half, float(max_xy[1]), step)
            if xs.size == 0:
                xs = np.array([0.5 * (float(min_xy[0]) + float(max_xy[0]))])
            if ys.size == 0:
                ys = np.array([0.5 * (float(min_xy[1]) + float(max_xy[1]))])
            for x in xs:
                for y in ys:
                    idx = used.get(pool_name, 0)
                    if idx >= len(pool):
                        return
                    z = float(layer_z.get(pool_name, base_z))
                    pool[idx].set_pos(np.array([float(x), float(y), z], dtype=float))
                    used[pool_name] = idx + 1

        # Lowest to highest visual priority:
        # blue table, green clear, orange blocked, yellow selected, cyan object,
        # red forbidden.
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        add_region((xmin, ymin), (xmax, ymax), "table")

        # Candidate swept AABBs: blocked orange, clear green, selected yellow.
        records_by_pool = {"clear": [], "blocked": [], "selected": []}
        for rec in getattr(self, "_neutral_avatar_region_debug_records", []) or []:
            if rec.get("selected"):
                pool_name = "selected"
            elif rec.get("footprint_clear") and rec.get("side_region_ok", True) and rec.get("body_ok"):
                pool_name = "clear"
            else:
                pool_name = "blocked"
            records_by_pool[pool_name].append(rec)
        for pool_name in ("clear", "blocked", "selected"):
            for rec in records_by_pool[pool_name]:
                xy = np.asarray(rec["xy"], dtype=np.float64).ravel()[:2]
                fp = rec["footprint"]
                min_xy = xy + np.asarray(fp["min_offset"], dtype=np.float64)
                max_xy = xy + np.asarray(fp["max_offset"], dtype=np.float64)
                add_region(min_xy, max_xy, pool_name)

        # Initial neutral prop/object regions used by reset-time layout
        # clearance. These are cyan so they can be inspected independently
        # from the larger swept motion footprint.
        for obj_name, obj in sorted(getattr(self, "neutral_avatar_props", {}).items()):
            try:
                center = to_numpy(obj.get_pos()).ravel()[:2].astype(np.float64)
            except Exception:
                continue
            radius = float(self._neutral_prop_radius(str(obj_name)))
            if (
                job.get("kind") == "put_objects_in_bowl"
                and str(obj_name).startswith("object_")
            ):
                radius = float(self._neutral_avatar_can_body_radius(str(obj_name)))
            add_region(center - radius, center + radius, "object")
        if hasattr(self, "neutral_avatar_bowl"):
            try:
                center = to_numpy(self.neutral_avatar_bowl.get_pos()).ravel()[:2].astype(np.float64)
                half = np.asarray(self.NEUTRAL_BOWL_HALF_XY, dtype=np.float64).ravel()[:2]
                add_region(center - half, center + half, "object")
            except Exception:
                pass

        # Forbidden object/corridor regions are red and intentionally highest.
        for center, radius, _label in forbidden:
            c = np.asarray(center, dtype=np.float64).ravel()[:2]
            r = float(radius)
            add_region(c - r, c + r, "forbidden")

    def _neutral_avatar_hold_motion_frame_idle(self, motion_name: str, frame_idx: int) -> None:
        """Pose the avatar on a motion frame without advancing the animation."""
        if self.avatar is None:
            return
        if motion_name not in self.avatar.motion_modules:
            self.avatar.play_animation(motion_name)
        self._neutral_avatar_apply_motion_frame(motion_name, frame_idx)
        self.avatar.robot.action_state = AvatarState.NO_ACTION
        self.avatar.robot.action_status = ActionStatus.INIT

    def _start_neutral_avatar_table_work(self, add_start_delay: bool = True) -> bool:
        if self.avatar is None or getattr(self, "_neutral_avatar_started", False):
            return False
        if (
            add_start_delay
            and bool(self.config.get("eval_mode", False))
            and bool(self.config.get("neutral_eval_human_from_ticks", False))
        ):
            # Eval-mode expert: the human starts at the eval trigger step,
            # ticked at policy cadence (NeutralTableWorkController.
            # on_scripted_rollout_tick), exactly as for the policies; the
            # scripted play_once start (with its demo start delay) is skipped.
            return False
        job = self._neutral_avatar_job
        if not job:
            return False

        # All of this is pre-task scene/avatar setup (incl. the layout settle
        # that lets props fall to rest); none of it should land in the video or
        # the training data.
        with self.suppress_recording():
            self._prepare_neutral_avatar_start_layout()
            self._install_neutral_loop_subclip(job)
            body_pos = np.asarray(
                getattr(self, "_neutral_avatar_body_pos", self.avatar_init_pos),
                dtype=np.float64,
            )
            footprint = self._neutral_avatar_motion_footprint
            self.avatar.play_animation(job["motion"])
            event_frame_offset = 0
            if add_start_delay:
                event_frame_offset = self._delay_neutral_avatar_motion_start(
                    job["motion"],
                    int(self.config.get("neutral_avatar_start_delay_steps", 0))
                )
            self._patch_neutral_avatar_step_sim(job, event_frame_offset=event_frame_offset)
            self._neutral_avatar_apply_motion_frame(job["motion"], 0)
        self._neutral_avatar_started = True
        # Never wipe task-phase contacts when the human
        # starts.  In eval mode the start happens at a policy step, so robot
        # contacts/distances logged before it are real and must be scored.
        # Only the pre-task settle records (tagged ``pretask`` by step_sim
        # under suppress_recording) are dropped.
        self.avatar_collision_log = [
            record for record in (getattr(self, "avatar_collision_log", None) or [])
            if not record.get("pretask")
        ]
        self.avatar_collided = any(
            bool(record.get("collided")) for record in self.avatar_collision_log
        )
        self._neutral_debug(
            f"[neutral_avatar] started {job['motion']} at {self._neutral_avatar_motion_slow():g}x slow, "
            f"body={body_pos.round(3).tolist()}, target={self.neutral_avatar_target.round(3).tolist()}, "
            f"footprint={np.round(footprint['min_offset'], 3).tolist()}.."
            f"{np.round(footprint['max_offset'], 3).tolist()}"
        )
        return True

    def _neutral_avatar_apply_motion_frame(self, motion_name: str, frame_idx: int) -> None:
        """Force the avatar skin to an exact animation frame."""
        motion = self.avatar.motion_modules.get(motion_name) if self.avatar is not None else None
        if motion is None or not getattr(motion, "data", None):
            return
        idx_raw = int(frame_idx)
        if idx_raw < 0:
            idx_raw = len(motion.data) + idx_raw
        idx = int(np.clip(idx_raw, 0, len(motion.data) - 1))
        motion.at_frame = idx
        self.avatar.robot.pose = motion.data[idx]
        self.avatar.robot.node_trans = motion.node_data[idx]
        self.avatar.robot.global_mat = motion.global_mat
        self.avatar.robot.global_mat_inv = motion.global_mat_inv
        self.avatar.robot.update()

    def _neutral_avatar_event_hold_offset(self, event: dict | None, obj_name: str) -> float:
        if event is not None and "hold_offset" in event:
            return float(event["hold_offset"])
        return (
            self.NEUTRAL_AVATAR_BOTTLE_HOLD_OFFSET
            if str(obj_name) == "bottle"
            else self.NEUTRAL_AVATAR_HOLD_OFFSET
        )

    def _neutral_avatar_hold_pos(
        self, hand_id: int, obj_name: str = "", event: dict | None = None
    ) -> np.ndarray:
        palm = np.asarray(self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64)
        _, hand_frame = self.avatar.robot._get_hand_frame(int(hand_id))
        palm_normal = np.asarray(hand_frame[:, 2], dtype=np.float64)
        pos = palm + self._neutral_avatar_event_hold_offset(event, obj_name) * palm_normal
        local = None if event is None else event.get("hold_offset_local")
        if local is not None:
            local_arr = np.asarray(local, dtype=np.float64).ravel()
            if local_arr.size != 3:
                raise ValueError(f"hold_offset_local must have 3 values, got {local!r}")
            pos = pos + hand_frame @ local_arr
        return pos

    def _neutral_avatar_palm_center_pos(
        self, hand_id: int, event: dict | None = None
    ) -> np.ndarray:
        pos = np.asarray(self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64).copy()
        if event is not None:
            _, hand_frame = self.avatar.robot._get_hand_frame(int(hand_id))
            local = event.get("hold_offset_local")
            if local is not None:
                local_arr = np.asarray(local, dtype=np.float64).ravel()
                if local_arr.size != 3:
                    raise ValueError(f"hold_offset_local must have 3 values, got {local!r}")
                pos = pos + hand_frame @ local_arr
        return pos

    def _neutral_avatar_surface_prop_pos(
        self, job: dict, obj_name: str, hand_id: int, event: dict
    ) -> np.ndarray:
        # Match render_raw_blender_interactions.py's default
        # surface_attach_strategy="palm_center_table_xy".
        pos = self._neutral_avatar_palm_center_pos(int(hand_id), event)
        pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, None)
        pos[2] = self._neutral_avatar_prop_center_z(obj_name)
        return pos

    def _neutral_avatar_event_anchor_pos(
        self, job: dict, hand_id: int, obj_name: str, event: dict
    ) -> np.ndarray:
        if str(obj_name) in ("sponge", "mouse"):
            return self._neutral_avatar_palm_center_pos(int(hand_id), event)
        return self._neutral_avatar_hold_pos(int(hand_id), obj_name, event)

    def _neutral_avatar_snap_prop_to_event_pose(
        self, job: dict, obj_name: str, hand_id: int, event: dict
    ) -> None:
        obj = self.neutral_avatar_props[obj_name]
        if obj_name == "phone":
            # Seat the phone in the palm per the talking-motion calibration
            # (palm centre, lowered a touch; screen faces the head via prop_quat).
            palm = np.asarray(
                self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64
            )
            pos = palm + np.array([0.0, 0.0, -self.NEUTRAL_PHONE_DOWN], dtype=np.float64)
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat("phone"))
            self._neutral_zero_entity_velocity(obj)
            return
        if obj_name in ("sponge", "mouse"):
            pos = self._neutral_avatar_surface_prop_pos(job, obj_name, hand_id, event)
        else:
            pos = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, None)
            if obj_name == "bottle" or str(obj_name).startswith("bottle_"):
                pos[:2] = self._neutral_avatar_clamp_table_xy(
                    pos[:2],
                    margin=self._neutral_avatar_table_margin_for_prop(job, obj_name),
                )
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
        obj.set_pos(pos.astype(float))
        obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
        self._neutral_zero_entity_velocity(obj)

    def _neutral_avatar_transition_frames(self) -> int:
        return max(
            1,
            int(round(
                float(self.NEUTRAL_AVATAR_TRANSITION_BASE_FRAMES)
                * self._neutral_avatar_motion_slow()
            )),
        )

    def _neutral_avatar_measure_hold_points(self, job: dict) -> dict[int, np.ndarray]:
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
        needed = {
            int(e["clip_frame"]): e
            for e in job["events"]
            if e["type"] in ("attach", "pick", "put")
        }
        hold_by_frame = {}
        step = 0
        max_step = max(needed) + 2 if needed else 0
        while not self.avatar.spare() and step <= max_step:
            self.scene.step()
            self.avatar.step()
            if step in needed:
                event = needed[step]
                hand_id = int(event["hand_id"])
                obj_name = str(event["object"])
                hold_by_frame[step] = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
            step += 1
        self.avatar.frame_ratio = old_ratio
        missing = sorted(set(needed) - set(hold_by_frame))
        if missing:
            raise RuntimeError(f"neutral avatar dry-run missed frames {missing}")
        return hold_by_frame

    def _neutral_prop_radius(self, obj_name: str) -> float:
        return float(self.NEUTRAL_PROP_RADII.get(str(obj_name), 0.055))

    def _neutral_motion_table_margin(self, footprint: dict | None = None) -> float:
        kind = str((footprint or {}).get("kind", ""))
        if kind == "sponge":
            return float(self.config.get(
                "neutral_sponge_table_margin",
                self.NEUTRAL_SPONGE_TABLE_MARGIN,
            ))
        return 0.0

    def _neutral_avatar_table_margin_for_prop(self, job: dict, obj_name: str) -> float:
        is_can = (
            job.get("kind") == "put_objects_in_bowl"
            and str(obj_name).startswith("object_")
        )
        radius = (
            self._neutral_avatar_can_body_radius(obj_name)
            if is_can
            else self._neutral_prop_radius(obj_name)
        )
        margin = float(radius) + float(self.config.get("neutral_prop_table_margin", 0.010))
        if str(obj_name) == "bottle" or str(obj_name).startswith("bottle_"):
            margin = max(
                margin,
                float(self.config.get(
                    "neutral_bottle_table_margin",
                    self.NEUTRAL_BOTTLE_TABLE_MARGIN,
                )),
            )
        return margin

    def _neutral_avatar_prop_center_z(self, obj_name: str) -> float:
        if str(obj_name) == "sponge":
            return float(self.TABLE_TOP_Z + self.NEUTRAL_SPONGE_HALF_THICKNESS)
        if str(obj_name).startswith("object_"):
            defaults = self._neutral_put_objects_clip_defaults()
            off = self.config.get(
                "neutral_put_objects_object_rest_z_offset",
                defaults.get("rest_z_offset", self.NEUTRAL_PUT_OBJECTS_OBJECT_REST_Z_OFFSET),
            )
            if off is None:
                off = self.NEUTRAL_PUT_OBJECTS_CAN_REST_Z_OFFSET
            return float(self.TABLE_TOP_Z + float(off))
        if str(obj_name) == "bottle" or str(obj_name).startswith("bottle_"):
            return float(self.TABLE_TOP_Z + 0.060)
        if str(obj_name) == "mouse":
            task_placement = self._neutral_avatar_task_placement()
            offsets = dict(task_placement.get("object_center_z_offsets", {}) or {})
            return float(self.TABLE_TOP_Z + float(offsets.get(
                "mouse",
                self.NEUTRAL_MOUSE_CENTER_Z_OFFSET,
            )))
        return float(self.TABLE_TOP_Z + self.NEUTRAL_PROP_HALF + 0.002)
