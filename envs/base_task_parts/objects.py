"""Target-object selection (see envs/object_catalog.py).

Split from BaseTask (verbatim member move); composed back into it in
envs/base_task.py.
"""


import numpy as np
from pathlib import Path

from ..utils import place_decorative_objects, to_numpy, ASSETS_PATH

from .shared import TABLE_HEIGHT


class _TargetObjectMixin:
    # ------------------------------------------------------------------
    # Target-object selection (see envs/object_catalog.py)
    # ------------------------------------------------------------------

    def resolve_target_object(self, *, randomize: bool = True):
        """Resolve this task's target object from ``OBJECT_SET``.

        ``config["object_name"]`` forces a specific key (validated against the
        task's set); otherwise one is sampled (``randomize=True``) or the first
        key is used (``randomize=False``, for tasks with a stable default).

        Returns the catalog ``ObjectEntry``.  Also stores it on
        ``self._target_entry`` and re-templates ``self.instruction`` so any
        ``{object}`` slot is filled with the object's display name.
        """
        from ..object_catalog import resolve_object_set, get_entry

        if self.OBJECT_SET is None:
            raise ValueError(
                f"{type(self).__name__} called resolve_target_object but does "
                f"not declare OBJECT_SET"
            )
        keys = resolve_object_set(self.OBJECT_SET)
        forced = self.config.get("object_name")
        if forced:
            forced = str(forced)
            # Accept either a bare key or an ``id@model`` token already in the set.
            match = next(
                (k for k in keys if k == forced or k.split("@", 1)[0] == forced),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{type(self).__name__}: object_name={forced!r} not in "
                    f"OBJECT_SET; choices={keys}"
                )
            token = match
        elif randomize:
            token = keys[int(np.random.randint(0, len(keys)))]
        else:
            token = keys[0]

        entry = get_entry(token)
        self._target_entry = entry
        self._apply_instruction_template(entry)
        return entry

    def resolve_target_objects(self, count: int, *, replace: bool = False):
        """Resolve several entries from ``OBJECT_SET`` for one episode.

        Normal sampling is seeded and without replacement by default.  An
        explicit ``config["object_name"]`` override intentionally repeats the
        forced object, which keeps single-object debug sweeps useful for
        multi-object tasks.
        """
        from ..object_catalog import resolve_object_set, get_entry

        count = int(count)
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if self.OBJECT_SET is None:
            raise ValueError(
                f"{type(self).__name__} called resolve_target_objects but does "
                f"not declare OBJECT_SET"
            )
        keys = resolve_object_set(self.OBJECT_SET)
        forced = self.config.get("object_name")
        if forced:
            forced = str(forced)
            match = next(
                (k for k in keys if k == forced or k.split("@", 1)[0] == forced),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{type(self).__name__}: object_name={forced!r} not in "
                    f"OBJECT_SET; choices={keys}"
                )
            tokens = [match] * count
        else:
            if not replace and count > len(keys):
                raise ValueError(
                    f"{type(self).__name__}: cannot sample {count} distinct "
                    f"objects from OBJECT_SET of size {len(keys)}"
                )
            indices = np.random.choice(len(keys), size=count, replace=replace)
            tokens = [keys[int(i)] for i in np.asarray(indices).reshape(-1)]

        entries = [get_entry(token) for token in tokens]
        if entries:
            self._target_entry = entries[0]
            self._apply_instruction_template(entries[0])
        return entries

    def _apply_instruction_template(self, entry) -> None:
        """Fill a ``{object}`` slot in the base instruction with ``entry``'s
        display name.  No-op when the instruction has no such slot (or a fixed
        instruction was passed via config)."""
        template = self._instruction_template
        if isinstance(template, str) and "{object}" in template:
            name = entry.display_name or entry.key
            self.instruction = template.format(object=name)

    @classmethod
    def default_instruction(cls):
        """Static (no-instantiation) instruction for this task.

        If ``INSTRUCTION`` contains a ``{object}`` slot, fill it with the first
        object in ``OBJECT_SET`` as a representative — so static instruction
        lookups (VLA client, data tooling) never surface a raw ``{object}``
        placeholder.  Per-episode the real object's name is filled into
        ``self.instruction`` by ``resolve_target_object``; VLA eval refreshes
        the policy from that per reset.
        """
        instr = cls.INSTRUCTION
        if isinstance(instr, str) and "{object}" in instr and cls.OBJECT_SET is not None:
            try:
                from ..object_catalog import resolve_object_set, get_entry
                keys = resolve_object_set(cls.OBJECT_SET)
                name = get_entry(keys[0]).display_name or keys[0]
                return instr.format(object=name)
            except Exception:
                return instr
        return instr

    def task_irrelevant_objects_enabled(self) -> bool:
        """Whether this rollout should include random background table objects."""
        return bool(self.table_random_objects)

    def table_random_objects_enabled(self) -> bool:
        """Canonical flag for task-irrelevant/random tabletop objects."""
        return self.task_irrelevant_objects_enabled()

    def hard_mode_enabled(self) -> bool:
        """Whether this rollout should use the task's hard layout."""
        return bool(self.hard)

    def simplified_mode_enabled(self) -> bool:
        """Whether this rollout should use the task's default easy layout."""
        return not self.hard_mode_enabled()

    def easy_mode_enabled(self) -> bool:
        """Backward-compatible alias for the simplified default layout."""
        return self.simplified_mode_enabled()

    def difficulty_value(self, *, easy, hard):
        """Return the value selected by the task's easy/hard mode."""
        return hard if self.hard_mode_enabled() else easy

    def difficulty_count(self, *, easy: int, hard: int) -> int:
        """Return a positive integer count selected by easy/hard mode."""
        return max(1, int(self.difficulty_value(easy=easy, hard=hard)))

    def difficulty_object_count(self) -> int:
        """Return the task-relevant object count selected by easy/hard mode."""
        return self.difficulty_count(
            easy=self.EASY_NUM_OBJECTS,
            hard=self.HARD_NUM_OBJECTS,
        )

    def _remember_table_bounds(
        self,
        half_size: tuple[float, float] = (0.6, 0.35),
        center_xy: tuple[float, float] | None = None,
    ) -> None:
        """Record tabletop XY bounds for universal random-object placement."""
        if center_xy is None:
            center_xy = getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0)))
        self._table_center_xy = np.asarray(center_xy, dtype=float).ravel()[:2]
        self._table_half_size_xy = np.asarray(half_size, dtype=float).ravel()[:2]

    def table_random_object_bounds(self) -> tuple[float, float, float, float]:
        """Return full tabletop XY bounds used by universal clutter placement."""
        center = np.asarray(
            getattr(
                self,
                "_table_center_xy",
                getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0))),
            ),
            dtype=float,
        ).ravel()[:2]
        half = np.asarray(getattr(self, "_table_half_size_xy", (0.6, 0.35)), dtype=float).ravel()[:2]
        if half.size < 2 or np.any(half <= 0.0):
            half = np.asarray((0.6, 0.35), dtype=float)
        return (
            float(center[0] - half[0]),
            float(center[0] + half[0]),
            float(center[1] - half[1]),
            float(center[1] + half[1]),
        )

    def table_random_object_region(self) -> tuple[float, float, float, float]:
        """Placement region for universal random tabletop objects."""
        if "table_random_object_region" in self.config:
            return tuple(float(v) for v in self.config["table_random_object_region"][:4])
        xmin, xmax, ymin, ymax = self.table_random_object_bounds()
        margin = float(self.config.get(
            "table_random_object_table_margin",
            self.TABLE_RANDOM_OBJECT_TABLE_MARGIN,
        ))
        return (xmin + margin, xmax - margin, ymin + margin, ymax - margin)

    def register_table_random_object_exclusion(self, exclusion) -> None:
        """Add a circle/rectangle exclusion consumed by the clutter spawner."""
        self._table_random_object_exclusions.append(exclusion)

    def table_random_object_exclusions(self) -> list:
        """Default universal exclusions: task actors, robot base, neutral motion."""
        exclusions = list(getattr(self, "_table_random_object_exclusions", []) or [])
        exclusions.extend(self.config.get("table_random_object_exclusions", []) or [])
        exclusions.extend(self._table_random_object_robot_exclusions())
        exclusions.extend(self._table_random_object_task_actor_exclusions())
        if hasattr(self, "_neutral_avatar_spawn_exclude_regions"):
            try:
                exclusions.extend(list(self._neutral_avatar_spawn_exclude_regions()))
            except Exception:
                pass
        return exclusions

    def _table_random_object_robot_exclusions(self) -> list[tuple[float, float, float]]:
        radius = float(self.config.get(
            "table_random_object_robot_base_radius",
            self.TABLE_RANDOM_OBJECT_ROBOT_BASE_RADIUS,
        ))
        exclusions = []
        seen = set()

        def add_xy(xy):
            key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
            if key not in seen:
                seen.add(key)
                exclusions.append((float(xy[0]), float(xy[1]), radius))

        robot = getattr(self, "robot", None)
        if robot is None:
            return exclusions
        try:
            bx, by, _ = robot.get_base_pose()
            add_xy((bx, by))
        except Exception:
            pass
        for tag in ("left", "right"):
            try:
                arm = robot.get_arm(tag)
            except Exception:
                arm = None
            if arm is not None and getattr(arm, "origin_pose", None) is not None:
                add_xy(np.asarray(arm.origin_pose.p, dtype=float).ravel()[:2])
        return exclusions

    def _table_random_object_task_actor_exclusions(self) -> list[tuple[float, float, float]]:
        """Approximate task object areas from actor/entity attributes."""
        radius = float(self.config.get(
            "table_random_object_exclusion_radius",
            self.TABLE_RANDOM_OBJECT_EXCLUSION_RADIUS,
        ))
        table_z = float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025))
        exclusions = []
        seen = set()
        skip_names = {
            "scene", "robot", "avatar", "avatar_collider", "avatar_collision_checker",
            "avatar_safety_checker", "avatar_retreat_checker", "cameras", "table",
            "floor", "table_variant_entity", "table_variant_surface_entity",
            "_table_random_object_actors",
        }

        def add_pos(pos):
            arr = np.asarray(pos, dtype=float).ravel()
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return
            if arr[2] < table_z - 0.20 or arr[2] > table_z + 0.80:
                return
            key = (round(float(arr[0]), 4), round(float(arr[1]), 4))
            if key in seen:
                return
            seen.add(key)
            exclusions.append((float(arr[0]), float(arr[1]), radius))

        def walk(value, depth=0):
            if depth > 3 or value is None:
                return
            if hasattr(value, "entity"):
                walk(getattr(value, "entity"), depth + 1)
                return
            if hasattr(value, "get_pos"):
                try:
                    add_pos(to_numpy(value.get_pos()).ravel()[:3])
                except Exception:
                    pass
                return
            if isinstance(value, dict):
                for v in value.values():
                    walk(v, depth + 1)
                return
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    walk(v, depth + 1)

        for name, value in vars(self).items():
            if name in skip_names or name.startswith("_table_random_object"):
                continue
            walk(value)
        return exclusions

    def _table_random_object_task_type_names(self) -> set[str]:
        """Return task-related asset ids/types that clutter should not reuse."""
        excluded: set[str] = set()
        asset_names = {
            p.name for p in (ASSETS_PATH / "objects").iterdir()
            if p.is_dir()
        }
        aliases = {
            str(k).lower(): tuple(v)
            for k, v in self.TABLE_RANDOM_OBJECT_TYPE_ALIASES.items()
        }

        def add_name(value) -> None:
            if value is None:
                return
            if isinstance(value, Path):
                raw = value.name
            else:
                raw = str(value)
            raw = raw.strip()
            if not raw:
                return
            candidates = {raw, raw.strip("/").split("/")[-1]}
            for candidate in list(candidates):
                stem = candidate.lower()
                candidates.add(stem)
                if stem in aliases:
                    candidates.update(aliases[stem])
            for candidate in candidates:
                if candidate in asset_names:
                    excluded.add(candidate)

        def add_hint(name: str) -> None:
            lowered = str(name).lower()
            tokens = lowered.replace("-", "_").split("_")
            compact = lowered.replace("-", "").replace("_", "")
            for key in aliases:
                key_compact = key.replace("-", "").replace("_", "")
                if key and (key in tokens or key == lowered or key_compact == compact):
                    for alias in aliases[key]:
                        add_name(alias)

        for manual in self.config.get("table_random_object_exclude_types", []) or []:
            add_name(manual)

        hook = getattr(self, "table_random_object_exclude_types", None)
        if callable(hook):
            try:
                for item in hook() or []:
                    add_name(item)
            except Exception:
                pass

        def walk(value, depth=0, attr_name: str | None = None) -> None:
            if depth > 4 or value is None:
                return
            if attr_name:
                add_hint(attr_name)
            if isinstance(value, (str, Path)):
                add_name(value)
                return
            if hasattr(value, "name"):
                try:
                    add_name(getattr(value, "name"))
                except Exception:
                    pass
            if hasattr(value, "entity"):
                return
            for field_name in (
                "asset_id", "object_id", "object_name", "dump_object_name",
                "_object_id", "_forced_object_id", "_forced_dump_object_id",
            ):
                if hasattr(value, field_name):
                    try:
                        add_name(getattr(value, field_name))
                    except Exception:
                        pass
            if isinstance(value, dict):
                for key, item in value.items():
                    key_s = str(key)
                    if any(token in key_s.lower() for token in ("asset", "object", "name", "id", "type")):
                        walk(item, depth + 1, key_s)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item, depth + 1, attr_name)
                return
            mod = getattr(type(value), "__module__", "")
            if mod.startswith("envs.") and hasattr(value, "__dict__"):
                for key, item in vars(value).items():
                    if not key.startswith("__"):
                        walk(item, depth + 1, key)

        from ..base_task import BaseTask  # call-time import; module-level would cycle

        for cls in type(self).mro():
            if cls is BaseTask:
                break
            for key, value in vars(cls).items():
                if key.startswith("__"):
                    continue
                if key.isupper() or key.startswith("_"):
                    walk(value, 0, key)
                else:
                    add_hint(key)

        for key, value in vars(self).items():
            if key in {"config", "scene", "robot", "avatar"}:
                continue
            if key.startswith("_table_random_object"):
                continue
            walk(value, 0, key)

        return excluded

    def table_random_object_pool(self) -> list[str] | None:
        """Return clutter pool after removing task-related object types."""
        configured_pool = self.config.get("table_random_object_pool")
        if configured_pool is None:
            return None
        excluded = self._table_random_object_task_type_names()
        return [name for name in configured_pool if str(name) not in excluded]

    def _spawn_table_random_objects(self) -> None:
        """Spawn universal physical clutter after task actors are in the scene."""
        self._table_random_object_actors = []
        if not self.table_random_objects_enabled():
            return
        if not self.use_table or self.table is None:
            return
        if bool(self.config.get("disable_universal_table_random_objects", False)):
            return
        count = int(self.config.get("table_random_object_count", self.TABLE_RANDOM_OBJECT_COUNT))
        if count <= 0:
            return
        configured_pool = self.config.get("table_random_object_pool")
        excluded_types = self._table_random_object_task_type_names()
        object_pool = None
        if configured_pool is not None:
            object_pool = [name for name in configured_pool if str(name) not in excluded_types]
        elif excluded_types:
            from ..utils import table_object_pool
            object_pool = [name for name in table_object_pool() if name not in excluded_types]
        if object_pool is not None and not object_pool:
            return
        self._table_random_object_actors = place_decorative_objects(
            self.scene,
            region=self.table_random_object_region(),
            height=float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025)),
            exclude_regions=self.table_random_object_exclusions(),
            num_objects=count,
            target_size=float(self.config.get(
                "table_random_object_target_size",
                self.TABLE_RANDOM_OBJECT_TARGET_SIZE,
            )),
            seed=int(np.random.randint(0, 10_000)),
            object_pool=object_pool,
            collision=bool(self.config.get("table_random_object_collision", True)),
            is_static=bool(self.config.get("table_random_object_static", False)),
            friction=self.config.get("table_random_object_friction"),
            density=self.config.get("table_random_object_density"),
            min_spacing=float(self.config.get("table_random_object_min_spacing", 0.02)),
        )

