"""Composable human interaction shared by block-ranking task families."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from ..avatar.inspect_motion_mixin import InspectMotionRuntime
from ..utils import to_numpy
from .base import AssistInteractionController, InterruptInteractionController
from .interrupt_progress import (
    InterruptEpisode,
    InterruptHumanSequence,
    ProgressTrigger,
    burn_legacy_trigger_draw,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    robot_tcp,
)


@dataclass(frozen=True)
class BlockSpawnSpec:
    """One block entity requested by an interaction episode plan."""

    key: str
    color_name: str
    half_size: float
    xy: tuple[float, float]


@dataclass(frozen=True)
class BlockEpisodePlan:
    """Scene-independent block layout and work-allocation plan."""

    blocks: tuple[BlockSpawnSpec, ...]
    avatar_key: str | None = None
    robot_keys: tuple[str, ...] = ()


class BlockEpisodeProvider:
    """Capability implemented by controllers that own block scene planning."""

    def create_block_episode(self, task) -> BlockEpisodePlan:
        raise NotImplementedError


class BlockAssistSampling(str, Enum):
    """Versioned global-NumPy draw orders retained for release parity."""

    RGB_V1 = "rgb_v1"
    SIZE_V1 = "size_v1"


class BlockRankingAssistController(BlockEpisodeProvider, AssistInteractionController):
    """One avatar-owned block plus a robot-owned complement.

    The controller deliberately samples through NumPy's legacy global stream
    at the family ``load_actors`` boundary.  That preserves release-v1 seeded
    scene layouts and random-call order.  New framework code should normally
    use :meth:`episode_rng`; this compatibility controller is explicit about
    why it cannot do so without a versioned dataset change.
    """

    rng_namespace = "block_ranking.assist.v1"
    handles_policy_ticks = True
    STATE_SCHEMA = "genesis_hr_bench.block_ranking_assist_state.v1"

    def __init__(
        self,
        *,
        keys: Sequence[str],
        owner_keys: Sequence[str] | None = None,
        delay_range: tuple[int, int],
        sampling: BlockAssistSampling | str,
    ) -> None:
        self.keys = tuple(str(key) for key in keys)
        if not self.keys:
            raise ValueError("block-ranking assist requires at least one key")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("block-ranking assist keys must be unique")
        self.owner_keys = tuple(
            str(key) for key in (self.keys if owner_keys is None else owner_keys)
        )
        if not self.owner_keys or not set(self.owner_keys) <= set(self.keys):
            raise ValueError("owner_keys must be a non-empty subset of keys")
        if len(set(self.owner_keys)) != len(self.owner_keys):
            raise ValueError("block-ranking owner_keys must be unique")
        self.delay_range = (int(delay_range[0]), int(delay_range[1]))
        if self.delay_range[0] < 0 or self.delay_range[0] > self.delay_range[1]:
            raise ValueError("delay_range must be ordered and non-negative")
        try:
            self.sampling = BlockAssistSampling(sampling)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported block assist sampling: {sampling!r}"
            ) from exc
        self.avatar_key: str | None = None
        self.robot_keys: tuple[str, ...] = ()
        self.avatar_start_delay_steps = 0
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = False
        self.eval_assist_failed = False
        self.policy_step_count = 0
        self.eval_start_delay_steps = 0
        self.blind_robot_keys: tuple[str, ...] = ()

    def on_episode_seeded(self, task, seed: int) -> None:
        self.avatar_key = None
        self.robot_keys = ()
        self.avatar_start_delay_steps = 0
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = False
        self.eval_assist_failed = False
        self.policy_step_count = 0
        self.eval_start_delay_steps = 0
        self.blind_robot_keys = ()

    def on_base_reset_ready(self, task) -> None:
        """Import release-v1 work allocation after scene-init replay.

        Scene-init v1 stores the legacy task attributes rather than an
        explicit controller state. BaseTask restores them after actors are
        built and before this hook, so the controller adopts those values for
        expert and evaluation behavior.
        """

        noun = str(getattr(task, "BLOCK_KEY_KIND", "block"))
        avatar_key = getattr(task, f"_avatar_{noun}", self.avatar_key)
        robot_keys = getattr(task, f"_robot_{noun}s", self.robot_keys)
        delay = getattr(
            task,
            "_avatar_start_delay_steps",
            self.avatar_start_delay_steps,
        )
        if avatar_key in self.owner_keys:
            restored_robot_keys = tuple(str(key) for key in robot_keys)
            expected = {key for key in self.keys if key != avatar_key}
            if (
                len(restored_robot_keys) == len(expected)
                and set(restored_robot_keys) == expected
            ):
                self.avatar_key = str(avatar_key)
                self.robot_keys = restored_robot_keys
        try:
            delay = int(delay)
        except (TypeError, ValueError):
            delay = self.avatar_start_delay_steps
        if delay >= 0:
            self.avatar_start_delay_steps = delay
        self._publish_work_allocation(task)

    def create_block_episode(self, task) -> BlockEpisodePlan:
        if self.sampling is BlockAssistSampling.SIZE_V1:
            # Release-v1 size order: colors, creation order, owner, delay,
            # then scalar layout draws.
            colors = tuple(np.random.permutation(tuple(task.CUBE_COLORS)))
            creation_order = tuple(np.random.permutation(self.keys))
        else:
            colors = tuple(task.CUBE_COLORS)
            creation_order = None

        self.avatar_key = str(np.random.choice(self.owner_keys))
        self.robot_keys = tuple(key for key in self.keys if key != self.avatar_key)
        lo, hi = self.delay_range
        self.avatar_start_delay_steps = int(np.random.randint(lo, hi + 1))
        # The legacy task classes exposed their allocation before attempting
        # layout sampling. Preserve that observable failure-path contract too.
        self._publish_work_allocation(task)
        xys = self._sample_spawn_xys(task, self.avatar_key)

        if creation_order is None:
            creation_order = tuple(np.random.permutation(self.keys))
        else:
            creation_order = tuple(str(key) for key in creation_order)
        if len(creation_order) != len(self.keys) or set(creation_order) != set(
            self.keys
        ):
            raise ValueError("creation order must be a permutation of keys")
        blocks = tuple(
            BlockSpawnSpec(
                key=key,
                color_name=(
                    key
                    if self.sampling is BlockAssistSampling.RGB_V1
                    else str(colors[index])
                ),
                half_size=float(task.block_half_size(key)),
                xy=(float(xys[key][0]), float(xys[key][1])),
            )
            for index, key in enumerate(creation_order)
        )
        self._publish_work_allocation(task)
        return BlockEpisodePlan(
            blocks=blocks,
            avatar_key=self.avatar_key,
            robot_keys=self.robot_keys,
        )

    def _sample_spawn_xys(self, task, avatar_key: str) -> dict[str, np.ndarray]:
        for _ in range(int(task.AVATAR_LAYOUT_SAMPLE_ATTEMPTS)):
            candidate = {
                avatar_key: task._layout_xy(
                    [
                        np.random.uniform(*task.AVATAR_SPAWN_X),
                        np.random.uniform(*task.AVATAR_SPAWN_Y),
                    ]
                )
            }
            for key in self.keys:
                if key == avatar_key:
                    continue
                candidate[key] = task._layout_xy(
                    [
                        np.random.uniform(*task.ROBOT_SPAWN_X),
                        np.random.uniform(*task.ROBOT_SPAWN_Y),
                    ]
                )
            if all(
                np.linalg.norm(candidate[first] - candidate[second])
                >= task.MIN_BLOCK_SPACING
                for index, first in enumerate(self.keys)
                for second in self.keys[index + 1 :]
            ):
                return candidate
        raise RuntimeError(
            f"[{getattr(task, 'ASSIST_LAYOUT_ERROR_LABEL', type(task).__name__)}] "
            "could not sample non-overlapping cubes"
        )

    def start_avatar_assist(self, task, *, add_start_delay: bool) -> None:
        if self.avatar_key is None:
            raise RuntimeError("assist episode has not been planned")
        cube = task.blocks[self.avatar_key]
        half = float(task.block_half_size(self.avatar_key))
        pick_pos = task.block_center(self.avatar_key).copy()
        pick_pos[2] += float(task.AVATAR_PICK_LIFT)
        slot = task.target_slots[self.avatar_key]
        place_pos = np.array(
            [
                slot[0],
                slot[1],
                task.TABLE_TOP_Z + half + task.AVATAR_PLACE_LIFT,
            ],
            dtype=float,
        )
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=cube,
            hand_id=None,
            approach_frames=task.AVATAR_APPROACH_FRAMES,
            transport_frames=task.AVATAR_TRANSPORT_FRAMES,
            retract_frames=task.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=task.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=task.AVATAR_NATURAL_BODY_Y_BOUNDS,
            yaw_limit_deg=task.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=task.AVATAR_NATURAL_SETTLE_STEPS,
        )
        task.avatar_collided = False
        task.avatar_collision_log = []
        if add_start_delay:
            self._delay_avatar_motion(task, self.avatar_start_delay_steps)

    @staticmethod
    def _delay_avatar_motion(task, delay_steps: int) -> None:
        if delay_steps <= 0:
            return
        motion = task.avatar.motion_modules.get("pick_and_place")
        if motion is None or not motion.frames:
            return
        hold = motion.frames[0]
        motion.frames = [hold] * int(delay_steps) + motion.frames
        motion.attach_frame += int(delay_steps)
        motion.detach_frame += int(delay_steps)

    def draw_eval_start_delay(self, task) -> int:
        """Eval-mode start delay in sim frames, a fair draw from the demos'
        window (``delay_range``; 100-200 frames = policy steps 4-8 for RGB,
        100-400 = 4-16 for size) from the controller's private per-episode
        stream, so it never consumes the global RNG."""

        lo, hi = self.delay_range
        return int(self.episode_rng(task, "eval_start_delay").integers(lo, hi + 1))

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        if not self._eval_active(task):
            return
        self.eval_assist_trigger_step = 0
        self.eval_assist_fired = True
        self.eval_assist_failed = False
        # The human is queued at reset (intent shown before the robot can
        # commit) but holds its first frame for a delay drawn from the
        # training demos' window: it used to start at frame 0, while every
        # demo human started at policy step 4-8 (RGB) or 4-16 (size).
        self.eval_start_delay_steps = self.draw_eval_start_delay(task)
        task._eval_avatar_start_delay_steps = self.eval_start_delay_steps
        try:
            self.start_avatar_assist(task, add_start_delay=False)
            self._delay_avatar_motion(task, self.eval_start_delay_steps)
        except Exception as exc:
            self.eval_assist_failed = True
            if task.config.get("debug_eval_avatar", False):
                print(
                    f"[{type(task).__name__}/eval] avatar assist failed: {exc}",
                    flush=True,
                )
        self._publish_eval_state(task)

    def on_policy_tick(self, task, *, step_index: int) -> None:
        if not self._eval_active(task):
            return
        # EvalModeAvatarMixin incremented this compatibility counter at the
        # absolute start of every policy call, including calls after success.
        self.policy_step_count = int(step_index)
        if self.eval_assist_fired:
            self._publish_eval_state(task)
            return
        if step_index < self.eval_assist_trigger_step:
            self._publish_eval_state(task)
            return
        self.eval_assist_fired = True
        self.start_avatar_assist(task, add_start_delay=False)
        self._publish_eval_state(task)

    @staticmethod
    def _eval_active(task) -> bool:
        return task.avatar is not None and bool(task.config.get("eval_mode", False))

    def scripted_robot_keys(self, task) -> tuple[str, ...]:
        if task.avatar is None:
            return tuple(task.block_keys())
        self.start_avatar_assist(task, add_start_delay=True)
        return self.robot_keys

    def choose_blind_robot_keys(self, task) -> tuple[str, ...]:
        """Blind expert's share, fixed at t=0 by a human-independent rule:
        the ``len(keys) - 1`` blocks nearest the gripper (xy, ties to the
        family key order), placed in the family key order.  It never reads
        the human's block, so it contests it whenever that block is among
        the nearest ones."""

        keys = tuple(task.block_keys())
        tcp = np.asarray(task.robot.get_arm("right").get_ee_pose(), dtype=float).ravel()[:2]

        def distance(key):
            xy = np.asarray(task.block_center(key), dtype=float).ravel()[:2]
            return round(float(np.linalg.norm(xy - tcp)), 6)

        nearest = sorted(keys, key=lambda key: (distance(key), keys.index(key)))
        share = set(nearest[: max(1, len(keys) - 1)])
        self.blind_robot_keys = tuple(key for key in keys if key in share)
        task._blind_robot_keys = self.blind_robot_keys
        return self.blind_robot_keys

    def wait_for_scripted_completion(self, task) -> None:
        if task.avatar is None:
            return
        while not task.avatar.spare():
            task.step_sim()

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        extra = {}
        if self.blind_robot_keys:
            # Only after play_blind_once ran (the legacy surface is unchanged).
            extra = {
                "blind_robot_keys": list(self.blind_robot_keys),
                "blind_contests_human_block": bool(
                    self.avatar_key in self.blind_robot_keys
                ),
            }
        # RGB assist historically inherited the base evaluator unchanged.
        # Preserve that release-v1 surface; size assist had explicit metrics.
        if str(getattr(task, "BLOCK_KEY_KIND", "block")) != "size":
            return extra
        return {**extra, **{
            "assist_avatar_size": self.avatar_key,
            "assist_robot_sizes": list(self.robot_keys),
            "assist_avatar_start_delay_steps": int(
                self.avatar_start_delay_steps
            ),
            "assist_avatar_spawn_y_band": list(task.AVATAR_SPAWN_Y),
            "assist_robot_spawn_y_band": list(task.ROBOT_SPAWN_Y),
            "eval_mode": bool(task.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(
                getattr(task, "_eval_assist_trigger_step", -1)
            ),
            "eval_assist_fired": bool(
                getattr(task, "_eval_assist_fired", False)
            ),
            "eval_policy_step_count": int(
                getattr(task, "_eval_policy_step_count", 0)
            ),
        }}

    def state_dict(self) -> dict[str, Any]:
        """Serialize controller bookkeeping and compatibility projections.

        This does not snapshot simulator physics or queued avatar motion; a
        full mid-episode continuation still requires the episode recorder.
        """

        return {
            "schema_version": self.STATE_SCHEMA,
            "avatar_key": self.avatar_key,
            "robot_keys": list(self.robot_keys),
            "avatar_start_delay_steps": int(self.avatar_start_delay_steps),
            "eval_assist_trigger_step": int(self.eval_assist_trigger_step),
            "eval_assist_fired": bool(self.eval_assist_fired),
            "eval_assist_failed": bool(self.eval_assist_failed),
            "policy_step_count": int(self.policy_step_count),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        """Restore validated v1 assist state without silently defaulting."""

        if not isinstance(value, Mapping):
            raise TypeError("block-ranking assist state must be a mapping")
        required = {
            "schema_version",
            "avatar_key",
            "robot_keys",
            "avatar_start_delay_steps",
            "eval_assist_trigger_step",
            "eval_assist_fired",
            "eval_assist_failed",
            "policy_step_count",
        }
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError("malformed assist state; missing: " + ", ".join(missing))
        if extra:
            raise ValueError("malformed assist state; unknown: " + ", ".join(extra))
        if value["schema_version"] != self.STATE_SCHEMA:
            raise ValueError(
                "unsupported block-ranking assist state schema: "
                f"{value['schema_version']!r}"
            )

        avatar_key = value["avatar_key"]
        if avatar_key is not None:
            avatar_key = str(avatar_key)
            if avatar_key not in self.owner_keys:
                raise ValueError(f"invalid assist avatar_key: {avatar_key!r}")
        robot_keys_raw = value["robot_keys"]
        if not isinstance(robot_keys_raw, (list, tuple)):
            raise TypeError("assist robot_keys must be a list or tuple")
        robot_keys = tuple(str(key) for key in robot_keys_raw)
        expected_robot = (
            set() if avatar_key is None else {
                key for key in self.keys if key != avatar_key
            }
        )
        if (
            len(robot_keys) != len(expected_robot)
            or set(robot_keys) != expected_robot
        ):
            raise ValueError("assist robot_keys do not complement avatar_key")

        delay = self._validated_state_int(
            value["avatar_start_delay_steps"],
            name="avatar_start_delay_steps",
        )
        trigger = self._validated_state_int(
            value["eval_assist_trigger_step"],
            name="eval_assist_trigger_step",
        )
        policy_step_count = self._validated_state_int(
            value["policy_step_count"],
            name="policy_step_count",
        )
        for name in ("eval_assist_fired", "eval_assist_failed"):
            if not isinstance(value[name], bool):
                raise TypeError(f"{name} must be a bool")

        self.avatar_key = avatar_key
        self.robot_keys = robot_keys
        self.avatar_start_delay_steps = delay
        self.eval_assist_trigger_step = trigger
        self.eval_assist_fired = value["eval_assist_fired"]
        self.eval_assist_failed = value["eval_assist_failed"]
        self.policy_step_count = policy_step_count

    def restore_task_state(self, task, value: Mapping[str, Any]) -> None:
        """Restore controller state and its release-v1 task projections."""

        self.load_state_dict(value)
        task._interaction_policy_step = int(self.policy_step_count)
        self._publish_eval_state(task)

    @staticmethod
    def _validated_state_int(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} cannot be negative")
        return result

    def _publish_work_allocation(self, task) -> None:
        """Keep the release-v1 diagnostic and scene-init attributes."""

        if self.avatar_key is not None:
            noun = str(getattr(task, "BLOCK_KEY_KIND", "block"))
            setattr(task, f"_avatar_{noun}", self.avatar_key)
            setattr(task, f"_robot_{noun}s", self.robot_keys)
        task._avatar_start_delay_steps = self.avatar_start_delay_steps

    def _publish_eval_state(self, task) -> None:
        self._publish_work_allocation(task)
        task._eval_assist_trigger_step = self.eval_assist_trigger_step
        task._eval_assist_fired = self.eval_assist_fired
        task._eval_assist_failed = self.eval_assist_failed
        task._eval_policy_step_count = int(self.policy_step_count)


class BlockInterruptSampling(str, Enum):
    """Versioned interrupt layout draw orders retained for release parity."""

    RGB_V1 = "rgb_v1"
    SIZE_V1 = "size_v1"


class BlockRankingInterruptController(
    BlockEpisodeProvider,
    InterruptInteractionController,
):
    """Shared inspect choreography for RGB- and size-ranking tasks.

    Scripted (collection) target and interruption-time decisions use the same
    independent seeded generators as the legacy task classes. Block creation
    and layout retain their exact global-NumPy draw order so existing
    scene-init seeds remain stable.

    Eval mode follows the shared Interruption eval contract
    (:mod:`envs.interaction.interrupt_progress`): the human reaches for the
    block the robot is approaching once the shared progress trigger fires,
    glides in, inspects the block, returns to idle and glides back.  Policy
    directives stay ``PROCEED``: the robot is never overridden.
    """

    rng_namespace = "block_ranking.interrupt.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = True
    STATE_SCHEMA = "genesis_hr_bench.block_ranking_interrupt_state.v2"

    def __init__(
        self,
        *,
        keys: Sequence[str],
        sampling: BlockInterruptSampling | str,
    ) -> None:
        self.keys = tuple(str(key) for key in keys)
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("block-ranking interrupt keys must be non-empty and unique")
        try:
            self.sampling = BlockInterruptSampling(sampling)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported block interrupt sampling: {sampling!r}"
            ) from exc
        self.inspect_runtime = InspectMotionRuntime()
        self._target_rng = None
        self.interrupted_key: str | None = None
        self.scripted_prefix_count = 0
        self.eval_palm_default: np.ndarray | None = None
        # Value of the burned release-v1 random-trigger draw (RNG parity
        # only; it no longer schedules anything).
        self.eval_legacy_trigger_draw: int | None = None
        self.eval_interrupt_fired = False
        self.eval_target_key: str | None = None
        self.eval_interrupt_started = False
        self.released_interrupted_pos: np.ndarray | None = None
        self.policy_step_count = 0
        self.eval_episode: InterruptEpisode | None = None
        self.hover_run = 0

    @property
    def is_size_profile(self) -> bool:
        return self.sampling is BlockInterruptSampling.SIZE_V1

    def on_episode_seeded(self, task, seed: int) -> None:
        self._target_rng = np.random.default_rng(
            int(seed) + int(task.SCRIPT_TARGET_SEED_OFFSET)
        )
        time_rng = np.random.default_rng(
            int(seed) + int(task.SCRIPT_TIME_SEED_OFFSET)
        )
        self.scripted_prefix_count = int(time_rng.integers(0, 3))
        self.interrupted_key = None
        self.eval_palm_default = None
        self.eval_interrupt_fired = False
        self.eval_target_key = None
        self.policy_step_count = 0
        self.eval_episode = None
        if self.is_size_profile:
            self.eval_interrupt_started = False
            self.released_interrupted_pos = None
            task._released_interrupted_pos = None
            task._eval_target_size = None
            task._eval_interrupt_started = False
        task._script_target_rng = self._target_rng
        task._scripted_prefix_count = self.scripted_prefix_count

    def create_block_episode(self, task) -> BlockEpisodePlan:
        if self.is_size_profile:
            colors = tuple(np.random.permutation(tuple(task.CUBE_COLORS)))
            creation_order = tuple(
                str(key) for key in np.random.permutation(self.keys)
            )
        else:
            colors = tuple(task.CUBE_COLORS)
            creation_order = tuple(
                str(key) for key in np.random.permutation(self.keys)
            )

        if self._target_rng is None:
            interrupted_index = int(np.random.randint(len(self.keys)))
        else:
            interrupted_index = int(self._target_rng.integers(0, len(self.keys)))
        self.interrupted_key = self.keys[interrupted_index]
        self._publish_episode_state(task)
        xys = self._sample_spawn_xys(task, creation_order)
        blocks = tuple(
            BlockSpawnSpec(
                key=key,
                color_name=(
                    str(colors[index]) if self.is_size_profile else key
                ),
                half_size=float(task.block_half_size(key)),
                xy=(float(xys[index, 0]), float(xys[index, 1])),
            )
            for index, key in enumerate(creation_order)
        )
        return BlockEpisodePlan(blocks=blocks)

    def _sample_spawn_xys(
        self,
        task,
        creation_order: Sequence[str],
    ) -> np.ndarray:
        x_lo, x_hi = task.AVATAR_REACH_SPAWN_X_RANGE
        y_lo, y_hi = task.AVATAR_REACH_SPAWN_Y_RANGE
        for _ in range(int(task.SPAWN_SAMPLE_ATTEMPTS)):
            xys = np.zeros((len(creation_order), 2), dtype=float)
            for index, _key in enumerate(creation_order):
                xys[index, 0] = np.random.uniform(x_lo, x_hi)
                xys[index, 1] = np.random.uniform(y_lo, y_hi)
            if all(
                np.linalg.norm(xys[first] - xys[second])
                >= task.MIN_BLOCK_SPACING
                for first in range(len(creation_order))
                for second in range(first + 1, len(creation_order))
            ):
                return task._layout_xys(xys)
        label = getattr(task, "INTERRUPT_LAYOUT_ERROR_LABEL", type(task).__name__)
        raise RuntimeError(f"[{label}] could not sample non-overlapping cubes")

    def on_base_reset_ready(self, task) -> None:
        """Adopt scene-init-v1 task projections after replay."""

        noun = str(getattr(task, "BLOCK_KEY_KIND", "block"))
        restored_key = getattr(task, f"_interrupted_{noun}", self.interrupted_key)
        if restored_key in self.keys:
            self.interrupted_key = str(restored_key)
        try:
            restored_prefix = int(
                getattr(task, "_scripted_prefix_count", self.scripted_prefix_count)
            )
        except (TypeError, ValueError):
            restored_prefix = self.scripted_prefix_count
        self.scripted_prefix_count = restored_prefix
        if self.is_size_profile:
            released = getattr(task, "_released_interrupted_pos", None)
            self.released_interrupted_pos = self._optional_vector(released)
            target_key = getattr(task, "_eval_target_size", None)
            if target_key in self.keys:
                self.eval_target_key = str(target_key)
            self.eval_interrupt_started = bool(
                getattr(task, "_eval_interrupt_started", False)
            )
        self._publish_episode_state(task)

    def after_base_reset_observation(self, task, *, observation: Any) -> None:
        # InspectMotionMixin historically resolved after BaseTask had already
        # captured and returned the initial observation, even with no avatar.
        resolver = getattr(task, "_resolve_inspect_motion", None)
        if callable(resolver):
            resolver()
        else:
            self.inspect_runtime.resolve(task)
        if task.avatar is None:
            return
        task.avatar.frame_ratio = task.AVATAR_MOTION_SLOW
        if bool(task.config.get("eval_mode", False)):
            initialize = self._task_hook(
                task,
                "_init_eval_interrupt_state",
                lambda: self.init_eval_state(task),
            )
            initialize()
        else:
            calibrate = self._task_hook(
                task,
                "_calibrate_avatar_for_inspect",
                lambda: self.calibrate_avatar(task),
            )
            calibrate()

    def attach_frame_scaled(self, task) -> int:
        return self.inspect_runtime.inspect_attach_frame_scaled(task)

    def detach_frame_scaled(self, task) -> int:
        return self.inspect_runtime.inspect_detach_frame_scaled(task)

    def play_inspect_motion(self, task, **kwargs):
        return self.inspect_runtime.play(task, **kwargs)

    @staticmethod
    def _task_hook(task, name: str, controller_method):
        """Honor task-level compatibility overrides without recursing."""

        if name in vars(task):
            return getattr(task, name)
        task_method = getattr(type(task), name, None)
        facade_method = None
        for base in type(task).__mro__:
            if base.__name__ == "BlockRankingInterruptFacade":
                facade_method = base.__dict__.get(name)
                break
        if task_method is not None and task_method is not facade_method:
            return getattr(task, name)
        return controller_method

    def dryrun_capture_palm(self, task):
        """Capture the selected motion's attach-frame palm in default space."""

        cache_key = (
            task.INSPECT_MOTION_NAME,
            float(task.AVATAR_MOTION_SLOW),
            task.AVATAR_HAND_ID,
            tuple(np.asarray(task.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(task), "_palm_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        from ..avatar.utils import (
            Mixamo_data_to_controller_pose,
            Mixamo_node_processing,
        )

        motion_data = task.avatar.motion_data.get(task.INSPECT_MOTION_NAME)
        if motion_data is None:
            return None
        source_length = int(motion_data["trans"].shape[0])
        scaled_length = max(
            1,
            int(source_length * task.AVATAR_MOTION_SLOW),
        )
        scaled_indices = np.round(
            np.linspace(0, source_length - 1, scaled_length)
        ).astype(int)
        frame = min(int(task._attach_frame_scaled), scaled_length - 1)
        source_index = int(scaled_indices[frame])

        pose = Mixamo_data_to_controller_pose(
            motion_data["trans"][source_index],
            motion_data["rot"][source_index],
            motion_data["joint"][source_index],
        )
        global_mat = motion_data["mat"][0]
        global_mat_inv = np.array([np.linalg.inv(matrix) for matrix in global_mat])
        vgeom = task.avatar.robot.skin.links[0]._vgeoms[0]
        node_trans = Mixamo_node_processing(
            vgeom,
            pose,
            global_mat,
            global_mat_inv,
        )

        task.avatar.robot.pose = pose
        task.avatar.robot.node_trans = node_trans
        task.avatar.robot.global_mat = global_mat
        task.avatar.robot.global_mat_inv = global_mat_inv
        task.avatar.robot.update()
        palm_default = np.asarray(
            task.avatar.robot.get_palm_center(task.AVATAR_HAND_ID),
            dtype=np.float64,
        ).copy()

        task.avatar.reset(
            np.asarray(task.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(int(task.CALIBRATION_SETTLE_STEPS)):
            task.step_sim()
        task.avatar_collided = False
        task.avatar_collision_log = []

        cache = getattr(type(task), "_palm_default_cache", None)
        if cache is None:
            cache = {}
            type(task)._palm_default_cache = cache
        cache[cache_key] = palm_default.copy()
        return palm_default

    def calibrate_avatar(self, task) -> None:
        if self.interrupted_key is None:
            raise RuntimeError("interrupt episode has not been planned")
        # Capture the target before the dry run, matching the legacy timing.
        cube = task.blocks[self.interrupted_key]
        target_palm = to_numpy(cube.get_pos()).ravel()[:3].astype(float)
        capture_palm = self._task_hook(
            task,
            "_dryrun_capture_palm",
            lambda: self.dryrun_capture_palm(task),
        )
        palm_default = capture_palm()
        if palm_default is None:
            return
        shift = target_palm - palm_default
        shift[2] += task.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        task.avatar.reset(
            shifted_pos.copy(),
            np.asarray(task.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(int(task.POST_CALIBRATION_SETTLE_STEPS)):
            task.step_sim()

    # -- eval-mode interrupt contract ------------------------

    def init_eval_state(self, task) -> None:
        """Capture the palm reference, burn the retired trigger draw, and arm
        the shared progress trigger from the reset TCP pose.

        The avatar stays where reset left it (``avatar_init_pos``); it is
        only moved by the post-trigger glide.
        """
        capture_palm = self._task_hook(
            task,
            "_dryrun_capture_palm",
            lambda: self.dryrun_capture_palm(task),
        )
        self.eval_palm_default = capture_palm()
        episode = new_interrupt_episode(
            task, hand_ids=(int(task.AVATAR_HAND_ID),),
        )
        self.eval_episode = episode
        self.eval_target_key = None
        self.eval_interrupt_started = False
        self.policy_step_count = 0
        self.hover_run = 0
        dwell = self.hover_dwell_params(task)
        if dwell is not None:
            episode.extra["hover_dwell"] = dict(dwell)
        if self.eval_palm_default is None:
            # Calibration failed: no human interruption is possible.
            self.eval_interrupt_fired = True
            episode.extra["calibration_failed"] = True
            episode.trigger.block_reason = "avatar_calibration_failed"
        else:
            # The release-v1 random-step trigger drew here; keep the draw so
            # the global RNG tail per seed is unchanged.
            self.eval_legacy_trigger_draw = burn_legacy_trigger_draw(task)
            self.eval_interrupt_fired = False
            episode.trigger.arm(robot_tcp(task), self.eligible_blocks(task))
        self._publish_eval_state(task)

    def on_policy_tick(self, task, *, step_index: int) -> None:
        advance = self._task_hook(
            task,
            "_eval_step_avatar",
            lambda: self.advance_eval_step(task),
        )
        advance()

    def advance_eval_step(self, task) -> None:
        """Compatibility entry point for the former private task hook."""

        if task.avatar is None or not bool(task.config.get("eval_mode", False)):
            return
        self.policy_step_count += 1
        self._advance_eval_state(task)
        self._publish_eval_state(task)

    def _advance_eval_state(self, task) -> None:
        episode = self.eval_episode
        if episode is None:
            return
        episode.policy_steps = int(self.policy_step_count)
        step = int(self.policy_step_count)
        if episode.human.phase != "idle":
            episode.human.tick(task, step)
            return
        if episode.fired or not episode.trigger.armed:
            return
        candidates = self.eligible_blocks(task)
        gate_ok, gate_reason = True, "phase_gate"
        holding = self.robot_holding_any(task)
        if holding:
            gate_ok, gate_reason = False, "robot_holding_object"
        dwell = self.hover_dwell_params(task)
        if dwell is not None:
            # Hover/dwell trigger: only the reset-chosen block, and only once
            # the TCP has hovered over it for ``steps`` consecutive policy
            # steps (as the demos do before the human starts).  Passing near
            # it while picking or placing another block does not count.
            candidates = {
                k: v for k, v in candidates.items() if k == self.interrupted_key
            }
            hovering = (not holding) and self._hovering_over(
                task, candidates.get(self.interrupted_key), dwell,
            )
            self.hover_run = self.hover_run + 1 if hovering else 0
            episode.extra["hover_dwell_max_run"] = max(
                int(episode.extra.get("hover_dwell_max_run", 0)), self.hover_run,
            )
            if gate_ok and self.hover_run < int(dwell["steps"]):
                gate_ok, gate_reason = False, "no_hover_dwell"
        key = episode.trigger.update(
            step, robot_tcp(task), candidates,
            gate_ok=gate_ok, gate_reason=gate_reason,
        )
        if key is None:
            return
        self.eval_interrupt_fired = True
        self.start_human(task, str(key), step)

    @staticmethod
    def hover_dwell_params(task) -> dict | None:
        """Hover/dwell trigger parameters, or None when the task uses the
        plain progress trigger.  Enabled per task by
        ``INTERRUPT_HOVER_DWELL_STEPS`` (config ``interrupt_hover_dwell_steps``);
        both block-ranking interrupt tasks set it."""
        config = getattr(task, "config", None) or {}
        steps = config.get(
            "interrupt_hover_dwell_steps",
            getattr(task, "INTERRUPT_HOVER_DWELL_STEPS", None),
        )
        if steps is None or int(steps) <= 0:
            return None
        xy = config.get(
            "interrupt_hover_xy_m", getattr(task, "INTERRUPT_HOVER_XY_M", 0.06),
        )
        dz = config.get(
            "interrupt_hover_dz_range_m",
            getattr(task, "INTERRUPT_HOVER_DZ_RANGE_M", (0.08, 0.20)),
        )
        return {
            "steps": int(steps),
            "xy_m": float(xy),
            "dz_min_m": float(dz[0]),
            "dz_max_m": float(dz[1]),
        }

    @staticmethod
    def _hovering_over(task, block_pos, dwell: Mapping[str, Any]) -> bool:
        if block_pos is None:
            return False
        tcp = robot_tcp(task)
        pos = np.asarray(block_pos, dtype=float)
        dz = float(tcp[2] - pos[2])
        return bool(
            float(np.linalg.norm(tcp[:2] - pos[:2])) <= float(dwell["xy_m"])
            and float(dwell["dz_min_m"]) <= dz <= float(dwell["dz_max_m"])
        )

    @staticmethod
    def _block_pos(task, key: str) -> np.ndarray:
        return to_numpy(task.blocks[key].get_pos()).ravel()[:3].astype(float)

    def eligible_blocks(self, task) -> dict:
        """Blocks on the table, not in their goal slot, not held."""
        out = {}
        blocks = getattr(task, "blocks", None) or {}
        for key in self.keys:
            if blocks.get(key) is None:
                continue
            pos = self._block_pos(task, key)
            target = np.asarray(task.target_slots[key], dtype=float)
            if np.linalg.norm(pos[:2] - target[:2]) <= task.SUCCESS_DIST_XY:
                continue
            if (
                pos[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or pos[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            if robot_grasping(task, pos):
                continue
            out[key] = pos
        return out

    def robot_holding_any(self, task) -> bool:
        """The robot carries a block (lifted off the table, gripper closed
        around it): it is transporting, not approaching."""
        blocks = getattr(task, "blocks", None) or {}
        for key in self.keys:
            if blocks.get(key) is None:
                continue
            pos = self._block_pos(task, key)
            lifted = pos[2] > (
                task.TABLE_TOP_Z + 2.0 * float(task.block_half_size(key)) + 0.02
            )
            if lifted and robot_grasping(task, pos):
                return True
        return False

    def _make_start_clip(self, task, key: str):
        def start_clip() -> bool:
            episode = self.eval_episode
            cube = task.blocks[key]
            # Play from the glided root: never let a stored motion root
            # teleport the avatar at clip start.
            task._inspect_motion_root_pos = None
            play_inspect = self._task_hook(
                task,
                "_play_inspect_motion",
                lambda **kwargs: self.play_inspect_motion(task, **kwargs),
            )
            play_inspect(
                attach_obj=cube,
                hand_id=task.AVATAR_HAND_ID,
                attach_frame=int(task._attach_frame_scaled),
                detach_frame=int(task._detach_frame_scaled),
                return_to_idle_after=episode.human.idle_return_frames,
            )
            episode.human.set_attach_guard(
                motion_name=task.INSPECT_MOTION_NAME,
                object_pos=lambda: self._block_pos(task, key),
                grasped=lambda: robot_grasping(task, self._block_pos(task, key)),
            )
            if self.is_size_profile:
                self.eval_interrupt_started = True
            return True

        return start_clip

    def start_human(self, task, key: str, step: int) -> None:
        episode = self.eval_episode
        block_pos = self._block_pos(task, key)
        shift = block_pos - np.asarray(self.eval_palm_default, dtype=np.float64)
        shift[2] += task.AVATAR_EXTRA_Z
        target_root = np.asarray(task.avatar_init_pos, dtype=np.float64) + shift
        self.eval_target_key = key
        episode.human.start(
            step=step,
            start_root=np.asarray(task.avatar.robot.global_trans, dtype=np.float64),
            target_root=target_root,
            rot=np.asarray(task.avatar_init_rot, dtype=np.float64),
            target_key=key,
            target_pos=block_pos,
            start_clip=self._make_start_clip(task, key),
        )

    def on_base_sim_step(self, task, *, step_index: int) -> None:
        episode = self.eval_episode
        if episode is not None and getattr(task, "avatar", None) is not None:
            episode.human.on_sim_step(task)

    def find_closest_table_block(self, task) -> str | None:
        arm = task.robot.get_arm("right")
        ee_pos = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        best_key = None
        best_distance = float("inf")
        for key, cube in task.blocks.items():
            pos = to_numpy(cube.get_pos()).ravel()[:3]
            target = task.target_slots[key]
            in_slot = (
                np.linalg.norm(pos[:2] - target[:2]) <= task.SUCCESS_DIST_XY
            )
            if in_slot:
                continue
            if (
                pos[2] < task.TABLE_TOP_Z - task.EVAL_TABLE_Z_LOWER_MARGIN
                or pos[2] > task.TABLE_TOP_Z + task.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            distance = float(np.linalg.norm(pos[:2] - ee_pos[:2]))
            if distance < best_distance:
                best_distance = distance
                best_key = key
        return best_key

    def run_scripted(self, task) -> bool:
        """Run the release-v1 robot/avatar interruption choreography."""

        arm_tag = "right"
        task._prepare_block_ranking_script(arm_tag)
        if task.avatar is None:
            # Directly constructing an avatar variant without an avatar
            # historically performed a second boost and settle in the core.
            return task._run_block_ranking_script(
                task.block_keys(),
                boost_fingers=True,
                include_settle=True,
                wait_for_interaction=False,
            )
        if self.interrupted_key is None:
            raise RuntimeError("interrupt episode has not been planned")

        interrupted_key = self.interrupted_key
        interrupted_cube = task.blocks[interrupted_key]
        non_interrupted = [
            key for key in task.block_keys() if key != interrupted_key
        ]
        prefix_count = min(
            max(int(self.scripted_prefix_count), 0),
            len(non_interrupted),
        )
        before_interrupt = non_interrupted[:prefix_count]
        during_interrupt = non_interrupted[prefix_count:]
        ok_all = True
        task.plan_success = True

        for key in before_interrupt:
            ok_all = task._place_block_with_recovery(key, arm_tag) and ok_all

        task._move_above_block_for_interrupt(interrupted_key, arm_tag)
        play_inspect = self._task_hook(
            task,
            "_play_inspect_motion",
            lambda **kwargs: self.play_inspect_motion(task, **kwargs),
        )
        play_inspect(
            attach_obj=interrupted_cube,
            hand_id=task.AVATAR_HAND_ID,
            attach_frame=int(task._attach_frame_scaled),
            detach_frame=int(task._detach_frame_scaled),
            return_to_idle_after=task.INSPECT_RETURN_TO_IDLE_FRAMES,
        )
        retreat = self._task_hook(
            task,
            "_retreat_safe",
            lambda tag: self.retreat_safe(task, tag),
        )
        retreat(arm_tag)
        wait_attach = self._task_hook(
            task,
            "_wait_until_avatar_attach",
            lambda: self.wait_until_avatar_attach(task),
        )
        wait_attach()
        for _ in range(int(task.POST_ATTACH_WAIT_MARGIN_STEPS)):
            task.step_sim()

        for key in during_interrupt:
            ok_all = task._place_block_with_recovery(key, arm_tag) and ok_all

        while not task.avatar.spare():
            task.step_sim()
        for _ in range(int(task.POST_RELEASE_SETTLE_STEPS)):
            task.step_sim()

        if self.is_size_profile:
            self.released_interrupted_pos = to_numpy(
                interrupted_cube.get_pos()
            ).ravel()[:3].copy()
            task._released_interrupted_pos = self.released_interrupted_pos.copy()
        ok_all = (
            task._place_block_with_recovery(interrupted_key, arm_tag) and ok_all
        )
        if not ok_all:
            ok_all = task._retry_unplaced_blocks(task.block_keys(), arm_tag)
        for _ in range(int(task.FINAL_SETTLE_STEPS)):
            task.step_sim()
        self._publish_episode_state(task)
        return ok_all

    @staticmethod
    def retreat_safe(task, arm_tag: str) -> None:
        arm = task.robot.get_arm(arm_tag)
        arm_base = np.asarray(arm.origin_pose.p, dtype=float)
        retreat_xy = arm_base[:2] + task._layout_xy(
            [0.0, task.RETREAT_Y_FROM_BASE]
        )
        retreat_pos = np.array(
            [
                retreat_xy[0],
                retreat_xy[1],
                task.TABLE_TOP_Z + task.RETREAT_Z_ABOVE_TABLE,
            ]
        )
        task._move_screw(retreat_pos, arm_tag)

    def wait_until_avatar_attach(self, task) -> None:
        motion = task.avatar.motion_modules.get(task.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not task.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self.attach_frame_scaled(task)
        ):
            task.step_sim()

    # -- full-state expert under the eval-mode contract ----------------

    def interrupt_expert_attempt(self, task) -> bool:
        """Idempotent scripted rollout from live state: place every block
        that is not yet in its slot, re-reading block poses each time."""
        arm_tag = "right"
        if not getattr(task, "_interrupt_expert_started", False):
            task._interrupt_expert_started = True
            task._prepare_block_ranking_script(arm_tag)
            task.plan_success = True
        ok_all = True
        for key in task.block_keys():
            pos = self._block_pos(task, key)
            target = np.asarray(task.target_slots[key], dtype=float)
            in_slot = np.linalg.norm(pos[:2] - target[:2]) <= task.SUCCESS_DIST_XY
            if in_slot and not robot_grasping(task, pos):
                continue
            ok_all = task._place_block_with_recovery(key, arm_tag) and ok_all
        for _ in range(int(task.FINAL_SETTLE_STEPS)):
            task.step_sim()
        return ok_all

    def interrupt_expert_retreat(self, task) -> None:
        arm_tag = "right"
        task.set_gripper(1.0, arm_tag, 20)
        retreat = self._task_hook(
            task,
            "_retreat_safe",
            lambda tag: self.retreat_safe(task, tag),
        )
        retreat(arm_tag)

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        eval_mode = bool(task.config.get("eval_mode", False))
        metrics: dict[str, Any] = {}
        episode = self.eval_episode
        trigger_step = None if episode is None else episode.trigger.trigger_step
        if self.is_size_profile:
            metrics.update({
                "interrupted_size": self.interrupted_key,
                "interrupted_color": task.block_colors.get(
                    self.interrupted_key,
                    None,
                ),
                "scripted_prefix_count": int(self.scripted_prefix_count),
                "interrupt_motion": getattr(task, "INSPECT_MOTION_NAME", None),
                "interrupt_attach_frame": int(
                    getattr(task, "INSPECT_ATTACH_FRAME", -1)
                ),
                "interrupt_detach_frame": int(
                    getattr(task, "INSPECT_DETACH_FRAME", -1)
                ),
                "eval_mode": eval_mode,
                # the policy step at which the progress trigger
                # fired (None if it never fired), not a pre-drawn step.
                "eval_trigger_step": trigger_step,
                "eval_policy_step_count": int(self.policy_step_count),
                "eval_interrupt_fired": bool(self.eval_interrupt_fired),
                "eval_interrupt_started": bool(self.eval_interrupt_started),
                "eval_target_size": self.eval_target_key,
            })
            if self.released_interrupted_pos is not None:
                metrics["released_interrupted_pos"] = np.asarray(
                    self.released_interrupted_pos
                ).tolist()
        if eval_mode:
            metrics["interrupt"] = interrupt_metrics(task)
            metrics["pace_interrupt"] = legacy_pace_block(task)
        return metrics

    def _publish_episode_state(self, task) -> None:
        task._script_target_rng = self._target_rng
        task._scripted_prefix_count = int(self.scripted_prefix_count)
        if self.interrupted_key is not None:
            noun = str(getattr(task, "BLOCK_KEY_KIND", "block"))
            setattr(task, f"_interrupted_{noun}", self.interrupted_key)
        if self.is_size_profile:
            task._released_interrupted_pos = (
                None
                if self.released_interrupted_pos is None
                else self.released_interrupted_pos.copy()
            )

    def _publish_eval_state(self, task) -> None:
        self._publish_episode_state(task)
        task._eval_palm_default = (
            None
            if self.eval_palm_default is None
            else self.eval_palm_default.copy()
        )
        episode = self.eval_episode
        if episode is not None:
            task._interrupt = episode
        task._eval_trigger_step = (
            None if episode is None else episode.trigger.trigger_step
        )
        task._eval_interrupt_fired = bool(self.eval_interrupt_fired)
        task._eval_glide_active = bool(
            episode is not None and episode.human.phase == "glide_in"
        )
        task._eval_target_block = (
            None
            if self.eval_target_key is None
            else task.blocks.get(self.eval_target_key)
        )
        task._policy_step_count = int(self.policy_step_count)
        if self.is_size_profile:
            task._eval_target_size = self.eval_target_key
            task._eval_interrupt_started = bool(self.eval_interrupt_started)

    @staticmethod
    def _copy_optional(value):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64).copy()

    @classmethod
    def _optional_vector(cls, value):
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64).ravel()
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return None
        return vector.copy()

    def state_dict(self) -> dict[str, Any]:
        """Serialize controller bookkeeping, not simulator/avatar physics."""

        def vector(value):
            return None if value is None else np.asarray(value).tolist()

        episode = self.eval_episode
        return {
            "schema_version": self.STATE_SCHEMA,
            "sampling": self.sampling.value,
            "interrupted_key": self.interrupted_key,
            "scripted_prefix_count": int(self.scripted_prefix_count),
            "eval_palm_default": vector(self.eval_palm_default),
            "eval_legacy_trigger_draw": self.eval_legacy_trigger_draw,
            "eval_interrupt_fired": bool(self.eval_interrupt_fired),
            "eval_target_key": self.eval_target_key,
            "eval_interrupt_started": bool(self.eval_interrupt_started),
            "released_interrupted_pos": vector(self.released_interrupted_pos),
            "policy_step_count": int(self.policy_step_count),
            "interrupt_trigger": (
                None if episode is None else episode.trigger.state_dict()
            ),
            "interrupt_human": (
                None if episode is None else episode.human.state_dict()
            ),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("block-ranking interrupt state must be a mapping")
        required = set(self.state_dict())
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError(
                "malformed interrupt state; missing: " + ", ".join(missing)
            )
        if extra:
            raise ValueError(
                "malformed interrupt state; unknown: " + ", ".join(extra)
            )
        if value["schema_version"] != self.STATE_SCHEMA:
            raise ValueError(
                "unsupported block-ranking interrupt state schema: "
                f"{value['schema_version']!r}"
            )
        if value["sampling"] != self.sampling.value:
            raise ValueError(
                "interrupt state sampling profile does not match controller"
            )

        interrupted_key = self._validated_optional_key(
            value["interrupted_key"],
            name="interrupted_key",
        )
        target_key = self._validated_optional_key(
            value["eval_target_key"],
            name="eval_target_key",
        )
        prefix = self._validated_state_int(
            value["scripted_prefix_count"],
            name="scripted_prefix_count",
        )
        legacy_draw = self._validated_optional_int(
            value["eval_legacy_trigger_draw"],
            name="eval_legacy_trigger_draw",
        )
        policy_count = self._validated_state_int(
            value["policy_step_count"],
            name="policy_step_count",
        )
        for name in ("eval_interrupt_fired", "eval_interrupt_started"):
            if not isinstance(value[name], bool):
                raise TypeError(f"{name} must be a bool")

        eval_palm_default = self._validated_state_vector(
            value["eval_palm_default"],
            name="eval_palm_default",
        )
        released_pos = self._validated_state_vector(
            value["released_interrupted_pos"],
            name="released_interrupted_pos",
        )
        fired = value["eval_interrupt_fired"]
        interrupt_started = value["eval_interrupt_started"]

        trigger_state = value["interrupt_trigger"]
        human_state = value["interrupt_human"]
        if (trigger_state is None) != (human_state is None):
            raise ValueError(
                "interrupt trigger and human state must be restored together"
            )
        episode = None
        if trigger_state is not None:
            trigger = ProgressTrigger()
            trigger.load_state_dict(trigger_state)
            human = InterruptHumanSequence()
            human.load_state_dict(human_state)
            if human.phase != "idle":
                if not trigger.fired or not fired:
                    raise ValueError(
                        "an active interrupt human requires a fired trigger"
                    )
                if target_key is None or str(human.target_key) != target_key:
                    raise ValueError(
                        "an active interrupt human requires its block key"
                    )
            if trigger.armed and not trigger.fired and eval_palm_default is None:
                raise ValueError(
                    "armed interrupt state requires a calibrated palm position"
                )
            episode = InterruptEpisode(
                trigger=trigger, human=human, policy_steps=policy_count,
            )
        if not self.is_size_profile and (
            interrupt_started or released_pos is not None
        ):
            raise ValueError("RGB interrupt state cannot contain size-only fields")

        # Commit only after the complete payload has passed cross-field
        # validation.  A rejected checkpoint must not leave a live controller
        # partially restored.
        self.interrupted_key = interrupted_key
        self.scripted_prefix_count = prefix
        self.eval_palm_default = eval_palm_default
        self.eval_legacy_trigger_draw = legacy_draw
        self.eval_interrupt_fired = fired
        self.eval_target_key = target_key
        self.eval_interrupt_started = interrupt_started
        self.released_interrupted_pos = released_pos
        self.policy_step_count = policy_count
        self.eval_episode = episode

    def restore_task_state(self, task, value: Mapping[str, Any]) -> None:
        if isinstance(value, Mapping):
            target_key = value.get("eval_target_key")
            if target_key in self.keys:
                blocks = getattr(task, "blocks", None)
                if not hasattr(blocks, "get") or blocks.get(target_key) is None:
                    raise RuntimeError(
                        "cannot restore an interrupt target before task blocks "
                        "have been initialized"
                    )
        self.load_state_dict(value)
        if self.eval_episode is not None and self.eval_target_key is not None:
            # Family callbacks are not serialized: rebind the clip start.
            self.eval_episode.human.rebind(
                self._make_start_clip(task, self.eval_target_key)
            )
        self._publish_eval_state(task)
        task._interaction_policy_step = int(self.policy_step_count)


    def _validated_optional_key(self, value, *, name: str) -> str | None:
        if value is None:
            return None
        result = str(value)
        if result not in self.keys:
            raise ValueError(f"invalid {name}: {result!r}")
        return result

    @staticmethod
    def _validated_state_int(value, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} cannot be negative")
        return result

    @classmethod
    def _validated_optional_int(cls, value, *, name: str) -> int | None:
        if value is None:
            return None
        return cls._validated_state_int(value, name=name)

    @staticmethod
    def _validated_state_vector(value, *, name: str) -> np.ndarray | None:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be null or a finite 3-vector")
        return vector.copy()
