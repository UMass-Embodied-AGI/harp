"""Pool-driven inspect-motion mixin for interrupt tasks.

Interrupt tasks such as `categorize_interrupt`,
`place_burger_fries_interrupt`, and `cont_bowl_cutlery_fetch` used to
hardcode three class attrs:

    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34
    INSPECT_DETACH_FRAME  = 187

``InspectMotionRuntime`` implements the scene-independent operations, while
``InspectMotionMixin`` remains the compatibility adapter for existing task
MROs. Dump-bin and bread-basket interrupts consume the same runtime through
family-specific composed controllers. Together they replace hardcoded metadata
with an ``INSPECT_POOL`` of
``{"name", "attach", "detach", optional "slow"}`` entries.  The resolver
runs once per ``reset()`` (after ``BaseTask.reset`` seeds ``np.random``)
and writes those same three attribute names — plus optionally
``AVATAR_MOTION_SLOW`` — onto the instance, shadowing the class-level
defaults.  Existing read sites (`self.INSPECT_MOTION_NAME` etc.) keep
working unchanged.

When ``config["randomize_inspect"]`` is True and the pool has more than
one entry, the resolver samples uniformly via ``np.random.randint`` —
deterministic per ``reset(seed=…)``.  Otherwise the first entry wins, so
existing tasks that don't override ``INSPECT_POOL`` keep playing
``"Inspect1"`` exactly as before.  The old ``"Inspect"`` name remains a
compatibility alias for existing configs.

MRO note: place ``InspectMotionMixin`` before the task / other mixins so
``super().reset`` chains correctly:

    class PlaceBurgerAndFriesInterruption(InspectMotionMixin, PlaceBurgerFries):
        INSPECT_POOL = None  # → INSPECT_POOL_DEFAULT
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectMotionSpec:
    """Motion-local metadata for one inspect animation.

    ``attach`` / ``detach`` are source-frame indices in the original motion.
    If ``clip`` is set, the mixin installs a runtime subclip and converts
    attach/detach to clip-local frame indices before task code sees them.
    """

    name: str
    attach: int
    detach: int
    hand_id: int = 1
    clip: tuple[int, int] | None = None
    slow: float | None = None
    attach_object: bool = True

    def as_pool_entry(self) -> dict:
        entry = {
            "name": self.name,
            "attach": int(self.attach),
            "detach": int(self.detach),
            "hand_id": int(self.hand_id),
            "attach_object": bool(self.attach_object),
        }
        if self.clip is not None:
            entry["clip"] = (int(self.clip[0]), int(self.clip[1]))
        if self.slow is not None:
            entry["slow"] = float(self.slow)
        return entry


INSPECT_MOTION_SPECS = {
    "Inspect1": InspectMotionSpec("Inspect1", attach=34, detach=187),
    "Inspect2": InspectMotionSpec("Inspect2", attach=31, detach=84),
    "Inspect3": InspectMotionSpec("Inspect3", attach=33, detach=164),
    # Source clip from retarget/raw_blender/inspect.blend.  Runtime tasks play
    # the clipped submotion but metadata stays in original source-frame units.
    "Inspect4": InspectMotionSpec("Inspect4", attach=61, detach=173, clip=(25, 196)),
    # Source clip from retarget/raw_blender/inspect5.blend.
    "Inspect5": InspectMotionSpec("Inspect5", attach=55, detach=133, clip=(15, 170)),
    # Source clip from retarget/raw_blender/inspect6.blend.
    "Inspect6": InspectMotionSpec("Inspect6", attach=94, detach=184, clip=(43, 212)),
    # Source clip from retarget/raw_blender/inspect7.blend.
    "Inspect7": InspectMotionSpec("Inspect7", attach=52, detach=196, clip=(29, 229)),
    # Source clip from retarget/raw_blender/inspect10.blend.
    "Inspect8": InspectMotionSpec(
        "Inspect8", attach=74, detach=185, clip=(19, 231),
    ),
    # Source clip from retarget/raw_blender/inspect9.blend.
    "Inspect9": InspectMotionSpec(
        "Inspect9", attach=69, detach=196, clip=(21, 247),
    ),
    # Source clip from retarget/raw_blender/inspect10.blend.
    "Inspect10": InspectMotionSpec(
        "Inspect10", attach=36, detach=184, clip=(17, 225),
    ),
}

INSPECT_MOTION_ALIASES = {
    "Inspect": "Inspect1",
}

STANDARD_INSPECT_POOL = tuple(
    INSPECT_MOTION_SPECS[name].as_pool_entry()
    for name in (
        "Inspect1", "Inspect2", "Inspect3", "Inspect4", "Inspect5",
        "Inspect6", "Inspect7", "Inspect8", "Inspect9", "Inspect10",
    )
)


# First entry is the legacy default — picked when randomization is off so
# existing scripted-flow tasks behave unchanged.  Add new motions to the
# end so randomized runs see them; keep frame indices in *original* (pre-
# slow) units — the read sites multiply by ``self.AVATAR_MOTION_SLOW``.
# Optional ``slow`` per-entry overrides the task's class-level
# ``AVATAR_MOTION_SLOW`` for that motion only.  ``attach=False`` means the
# avatar reaches the calibrated contact frame but does not kinematically
# attach the object; this is used by touch/withdraw interrupt variants.
INSPECT_POOL_DEFAULT = [
    *STANDARD_INSPECT_POOL,
    {"name": "Inspect1_touch_hold_reverse", "attach": 34, "detach": 34, "attach_object": False},
    {"name": "Inspect2_touch_hold_reverse", "attach": 31, "detach": 31, "attach_object": False},
]


class InspectMotionRuntime:
    """Stateless inspect-motion operations for composed and legacy tasks.

    Every method receives the owning task explicitly.  The runtime therefore
    retains neither task references nor episode state and can be shared by the
    legacy MRO adapter below and by composed interaction controllers.
    """

    __slots__ = ()

    def resolve(self, task, *, prepare_entry=None) -> None:
        """Resolve and publish the selected inspect motion on ``task``."""

        pool = task.INSPECT_POOL or INSPECT_POOL_DEFAULT
        if not pool:
            return
        forced_name = task.config.get("inspect_motion_name")
        if forced_name:
            forced_name = INSPECT_MOTION_ALIASES.get(forced_name, forced_name)
            matches = [entry for entry in pool if entry["name"] == forced_name]
            if not matches:
                raise ValueError(
                    f"config['inspect_motion_name']={forced_name!r} "
                    f"not in pool: {[entry['name'] for entry in pool]}"
                )
            entry = matches[0]
            print(f"[inspect] forced motion={entry['name']} (config override)")
        elif task.config.get("randomize_inspect", False) and len(pool) > 1:
            entry = pool[int(np.random.randint(len(pool)))]
            print(f"[inspect] randomized motion={entry['name']}")
        else:
            entry = pool[0]

        entry = dict(entry)
        if prepare_entry is None:
            entry = self.prepare_entry(task, entry)
        else:
            entry = prepare_entry(entry)
        task.INSPECT_MOTION_NAME = entry["name"]
        task.INSPECT_ATTACH_FRAME = int(entry["attach"])
        task.INSPECT_DETACH_FRAME = int(entry["detach"])
        task.AVATAR_HAND_ID = int(
            entry.get("hand_id", getattr(task, "AVATAR_HAND_ID", 1))
        )
        task.INSPECT_ATTACH_OBJECT = bool(entry.get("attach_object", True))
        if "slow" in entry:
            task.AVATAR_MOTION_SLOW = float(entry["slow"])

    def prepare_entry(self, task, entry: dict) -> dict:
        """Install a configured source-frame subclip and return runtime frames."""

        clip = entry.get("clip")
        if clip is None:
            return entry
        if getattr(task, "avatar", None) is None:
            return entry
        src_name = str(entry["name"])
        start, end = (int(clip[0]), int(clip[1]))
        if src_name not in task.avatar.motion_data:
            return entry
        md = task.avatar.motion_data[src_name]
        n = int(md["trans"].shape[0])
        if start < 0 or end < start or end >= n:
            raise ValueError(
                f"Invalid inspect clip {start}..{end} for {src_name} length {n}"
            )
        dst_name = f"{src_name}_clip_{start}_{end}"
        if dst_name not in task.avatar.motion_data:
            sl = slice(start, end + 1)
            clipped = {key: value[sl].copy() for key, value in md.items()}
            task.avatar.motion_data[dst_name] = clipped
        entry["name"] = dst_name
        entry["source_name"] = src_name
        entry["source_attach"] = int(entry["attach"])
        entry["source_detach"] = int(entry["detach"])
        entry["attach"] = int(entry["attach"]) - start
        entry["detach"] = int(entry["detach"]) - start
        if entry["attach"] < 0 or entry["detach"] < entry["attach"]:
            raise ValueError(
                f"Inspect attach/detach must fall inside clip for {src_name}: {entry}"
            )
        return entry

    @staticmethod
    def inspect_attach_frame_scaled(task) -> int:
        return int(round(task.INSPECT_ATTACH_FRAME * task.AVATAR_MOTION_SLOW))

    @staticmethod
    def inspect_detach_frame_scaled(task) -> int:
        return int(round(task.INSPECT_DETACH_FRAME * task.AVATAR_MOTION_SLOW))

    def avatar_pos_for_object(
        self,
        task,
        object_pos,
        extra_z: float | None = None,
    ):
        """Return the avatar root that fixes the selected palm at an object."""

        palm_default = task._dryrun_capture_palm()
        if palm_default is None:
            return None
        target = np.asarray(object_pos, dtype=np.float64).ravel()[:3].copy()
        shift = target - np.asarray(palm_default, dtype=np.float64).ravel()[:3]
        shift[2] += float(
            getattr(task, "AVATAR_EXTRA_Z", 0.0)
            if extra_z is None
            else extra_z
        )
        return np.asarray(task.avatar_init_pos, dtype=np.float64).copy() + shift

    def reset_avatar_for_object(
        self,
        task,
        object_pos,
        extra_z: float | None = None,
        settle_steps: int | None = None,
        *,
        avatar_pos_for_object=None,
        set_motion_root_pos=None,
    ) -> bool:
        """Place the avatar using the fixed selected-motion relation."""

        if getattr(task, "avatar", None) is None:
            return False
        if avatar_pos_for_object is None:
            avatar_pos = self.avatar_pos_for_object(
                task, object_pos, extra_z=extra_z
            )
        else:
            avatar_pos = avatar_pos_for_object(object_pos, extra_z=extra_z)
        if avatar_pos is None:
            return False
        if set_motion_root_pos is None:
            idle_pos = self.set_motion_root_pos(task, avatar_pos)
        else:
            idle_pos = set_motion_root_pos(avatar_pos)
        task.avatar.reset(
            idle_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        n = int(
            getattr(task, "CALIBRATION_SETTLE_STEPS", 0)
            if settle_steps is None
            else settle_steps
        )
        for _ in range(max(0, n)):
            task.step_sim()
        task.avatar_collided = False
        task.avatar_collision_log = []
        return True

    @staticmethod
    def idle_position_for_motion_root(
        task,
        motion_root_pos,
        motion_frame: int = 0,
    ) -> np.ndarray:
        """Return a global idle root whose mesh matches a motion frame."""

        if getattr(task, "avatar", None) is None:
            return np.asarray(motion_root_pos, dtype=np.float64).copy()
        md = task.avatar.motion_data.get(task.INSPECT_MOTION_NAME)
        idle = task.avatar.motion_data.get("idle")
        if md is None or idle is None:
            return np.asarray(motion_root_pos, dtype=np.float64).copy()
        n = int(md["trans"].shape[0])
        frame = int(np.clip(int(motion_frame), 0, n - 1))
        motion_trans = np.asarray(md["trans"][frame], dtype=np.float64)
        idle_trans = np.asarray(idle["trans"][0], dtype=np.float64)
        rot = np.asarray(task.avatar_init_rot, dtype=np.float64)
        base_rot = np.asarray(task.avatar.robot.base_rot, dtype=np.float64)
        return (
            np.asarray(motion_root_pos, dtype=np.float64).copy()
            + rot @ base_rot @ (motion_trans - idle_trans)
        )

    def set_motion_root_pos(
        self,
        task,
        motion_root_pos,
        *,
        idle_position_for_motion_root=None,
    ) -> np.ndarray:
        """Store the playback root and return the matching idle root."""

        root = np.asarray(motion_root_pos, dtype=np.float64).copy()
        task._inspect_motion_root_pos = root
        if idle_position_for_motion_root is None:
            idle_start = self.idle_position_for_motion_root(
                task, root, motion_frame=0
            )
        else:
            idle_start = idle_position_for_motion_root(root, motion_frame=0)
        task._inspect_idle_start_pos = idle_start
        return idle_start

    @staticmethod
    def play(
        task,
        attach_obj=None,
        hand_id=1,
        attach_frame=None,
        detach_frame=None,
        **kwargs,
    ):
        """Play the selected motion while honoring touch-only entries."""

        if not getattr(task, "INSPECT_ATTACH_OBJECT", True):
            attach_obj = None
            attach_frame = None
            detach_frame = None
        motion_root_pos = getattr(task, "_inspect_motion_root_pos", None)
        if motion_root_pos is not None:
            task.avatar.reset(
                np.asarray(motion_root_pos, dtype=np.float64).copy(),
                np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
            )
        return task.avatar.play_animation(
            task.INSPECT_MOTION_NAME,
            attach_obj=attach_obj,
            hand_id=hand_id,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            **kwargs,
        )


_INSPECT_MOTION_RUNTIME = InspectMotionRuntime()


class InspectMotionMixin:
    """See module docstring."""

    INSPECT_POOL = None  # Override per-task; None → INSPECT_POOL_DEFAULT.

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        self._resolve_inspect_motion()
        return obs

    def _resolve_inspect_motion(self) -> None:
        return _INSPECT_MOTION_RUNTIME.resolve(
            self,
            prepare_entry=self._prepare_inspect_motion_entry,
        )

    def _prepare_inspect_motion_entry(self, entry: dict) -> dict:
        """Install a configured source-frame subclip and return runtime frames."""
        return _INSPECT_MOTION_RUNTIME.prepare_entry(self, entry)

    @property
    def _inspect_attach_frame_scaled(self) -> int:
        return _INSPECT_MOTION_RUNTIME.inspect_attach_frame_scaled(self)

    @property
    def _inspect_detach_frame_scaled(self) -> int:
        return _INSPECT_MOTION_RUNTIME.inspect_detach_frame_scaled(self)

    @property
    def _attach_frame_scaled(self) -> int:
        return self._inspect_attach_frame_scaled

    @property
    def _detach_frame_scaled(self) -> int:
        return self._inspect_detach_frame_scaled

    def inspect_avatar_pos_for_object(self, object_pos, extra_z: float | None = None):
        """Return avatar root position that gives this motion a fixed object pose.

        This preserves the motion-specific avatar-object relative placement:
        the object can be placed first or the avatar can be placed first, but
        one side should be derived from this fixed relation.
        """
        return _INSPECT_MOTION_RUNTIME.avatar_pos_for_object(
            self,
            object_pos,
            extra_z=extra_z,
        )

    def reset_avatar_for_inspect_object(
        self,
        object_pos,
        extra_z: float | None = None,
        settle_steps: int | None = None,
    ) -> bool:
        """Place the avatar using the fixed relation for the selected motion."""
        return _INSPECT_MOTION_RUNTIME.reset_avatar_for_object(
            self,
            object_pos,
            extra_z=extra_z,
            settle_steps=settle_steps,
            avatar_pos_for_object=self.inspect_avatar_pos_for_object,
            set_motion_root_pos=self._set_inspect_motion_root_pos,
        )

    def _inspect_idle_position_for_motion_root(
        self,
        motion_root_pos,
        motion_frame: int = 0,
    ) -> np.ndarray:
        """Return a global idle root whose mesh matches a motion frame.

        This adjusts only the avatar's global idle position.  It never edits
        the authored motion arrays.
        """
        return _INSPECT_MOTION_RUNTIME.idle_position_for_motion_root(
            self,
            motion_root_pos,
            motion_frame=motion_frame,
        )

    def _set_inspect_motion_root_pos(self, motion_root_pos) -> np.ndarray:
        """Store the root used for playback and return the matching idle root."""
        return _INSPECT_MOTION_RUNTIME.set_motion_root_pos(
            self,
            motion_root_pos,
            idle_position_for_motion_root=self._inspect_idle_position_for_motion_root,
        )

    def _play_inspect_motion(self, attach_obj=None, hand_id=1,
                             attach_frame=None, detach_frame=None, **kwargs):
        """Play the selected inspect motion, honoring touch-only entries."""
        return _INSPECT_MOTION_RUNTIME.play(
            self,
            attach_obj=attach_obj,
            hand_id=hand_id,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            **kwargs,
        )
