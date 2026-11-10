"""Scene-free release contracts for block-ranking interrupt behavior.

These tests intentionally characterize the release-v1 task implementation
without requiring Genesis scene construction.  The small ownership adapters
allow the same assertions to follow the behavior when it moves from the
legacy task classes into ``BlockRankingInterruptController``.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect

import numpy as np
import pytest

from envs.interaction import InteractionMode
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.task_bases.blocks_ranking_size import BlocksRankingSize
from envs.tasks.rank_blocks_by_color_contribution import RankBlocksByColorContribution
from envs.tasks.rank_blocks_by_color_interruption import RankBlocksByColorInterruption
from envs.tasks.rank_blocks_by_size_contribution import RankBlocksBySizeContribution
from envs.tasks.rank_blocks_by_size_interruption import RankBlocksBySizeInterruption
from envs.utils import Pose
from scripts.eval_expert_set import (
    configure_parent_rollout_eval,
    policy_play_mode,
    run_with_eval_step_driver,
)


EPISODE_SEEDS = (0, 8, 42)
INDEPENDENT_STREAM_GOLDENS = {
    # seed: (target index from seed + 7919, prefix from seed + 13)
    0: (1, 2),
    1: (0, 0),
    2: (2, 2),
    7: (1, 2),
    8: (2, 0),
    42: (2, 2),
    113: (0, 0),
}


def _is_composed(task) -> bool:
    mode = getattr(task.interaction_controller, "mode", None)
    return getattr(mode, "value", mode) == InteractionMode.INTERRUPT.value


def _owner(task):
    """Return the object that owns interrupt state in this implementation."""

    return task.interaction_controller if _is_composed(task) else task


def _set_on_task_and_owner(task, name: str, value) -> None:
    setattr(task, name, value)
    setattr(_owner(task), name.lstrip("_"), value)
    setattr(_owner(task), name, value)


def _read_any(task, *names, default=None):
    for source in (task, _owner(task)):
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    return default


def _interrupted_key(task):
    noun = "size" if isinstance(task, BlocksRankingSize) else "color"
    return _read_any(
        task,
        f"_interrupted_{noun}",
        f"interrupted_{noun}",
        "_interrupted_key",
        "interrupted_key",
    )


def _scripted_prefix(task) -> int:
    return int(
        _read_any(
            task,
            "_scripted_prefix_count",
            "scripted_prefix_count",
            default=0,
        )
    )


def _seed_episode(task, seed: int) -> None:
    """Reproduce the interrupt state available just before actor loading."""

    seed = int(seed)
    np.random.seed(seed)
    task._episode_seed = seed
    task._episode_rngs = {}
    if _is_composed(task):
        task.interaction_controller.on_episode_seeded(task, seed)
        return

    task._script_target_rng = np.random.default_rng(
        seed + task.SCRIPT_TARGET_SEED_OFFSET
    )
    task._scripted_prefix_count = int(
        np.random.default_rng(seed + task.SCRIPT_TIME_SEED_OFFSET).integers(0, 3)
    )
    if isinstance(task, BlocksRankingSize):
        task._released_interrupted_pos = None
        task._eval_target_size = None
        task._eval_interrupt_started = False


def _set_interrupt_allocation(task, interrupted_key: str, prefix: int) -> None:
    noun = "size" if isinstance(task, BlocksRankingSize) else "color"
    _set_on_task_and_owner(task, f"_interrupted_{noun}", interrupted_key)
    _set_on_task_and_owner(task, "_interrupted_key", interrupted_key)
    _set_on_task_and_owner(task, "_scripted_prefix_count", int(prefix))


class _FakeBlock:
    def __init__(self, pos, **metadata):
        self._pos = np.asarray(pos, dtype=np.float64)
        self.metadata = metadata

    def get_pos(self):
        return self._pos.copy()


class _FakeAvatar:
    def __init__(self, events=None):
        self.events = events
        self.reset_positions = []
        self.frame_ratio = None
        self.motion_modules = {}

    def reset(self, pos, rot, **_kwargs):
        p = np.asarray(pos, dtype=np.float64).copy()
        self.reset_positions.append(p)
        if self.events is not None:
            self.events.append(("avatar_reset", p.copy()))

    def spare(self):
        return True


def _legacy_interrupt_layout(task, order):
    """Copy the release-v1 scalar draw and retry order."""

    x_lo, x_hi = task.AVATAR_REACH_SPAWN_X_RANGE
    y_lo, y_hi = task.AVATAR_REACH_SPAWN_Y_RANGE
    for attempt in range(1, int(task.SPAWN_SAMPLE_ATTEMPTS) + 1):
        xys = np.zeros((3, 2), dtype=float)
        for index, _key in enumerate(order):
            xys[index, 0] = np.random.uniform(x_lo, x_hi)
            xys[index, 1] = np.random.uniform(y_lo, y_hi)
        if all(
            np.linalg.norm(xys[first] - xys[second])
            >= task.MIN_BLOCK_SPACING
            for first in range(3)
            for second in range(first + 1, 3)
        ):
            return task._layout_xys(xys), attempt
    raise RuntimeError("legacy interrupt layout found no valid sample")


def _legacy_rgb_episode(task, seed: int):
    np.random.seed(seed)
    keys = tuple(task.CUBE_COLORS)
    order = tuple(np.random.permutation(keys))
    target_index = int(
        np.random.default_rng(seed + task.SCRIPT_TARGET_SEED_OFFSET).integers(
            0, len(keys)
        )
    )
    xys, attempts = _legacy_interrupt_layout(task, order)
    tail = np.random.random_sample(8)
    return {
        "order": order,
        "colors": {key: key for key in order},
        "xys": dict(zip(order, xys)),
        "attempts": attempts,
        "interrupted_key": keys[target_index],
        "tail": tail,
    }


def _legacy_size_episode(task, seed: int):
    np.random.seed(seed)
    colors = tuple(np.random.permutation(tuple(task.CUBE_COLORS)))
    keys = tuple(task.SIZE_ORDER)
    order = tuple(np.random.permutation(keys))
    target_index = int(
        np.random.default_rng(seed + task.SCRIPT_TARGET_SEED_OFFSET).integers(
            0, len(keys)
        )
    )
    xys, attempts = _legacy_interrupt_layout(task, order)
    tail = np.random.random_sample(8)
    return {
        "order": order,
        "colors": dict(zip(order, colors)),
        "xys": dict(zip(order, xys)),
        "attempts": attempts,
        "interrupted_key": keys[target_index],
        "tail": tail,
    }


def _load_fake_episode(task, seed: int, monkeypatch):
    from envs.task_bases import blocks_ranking_rgb as rgb_core_module
    from envs.tasks import rank_blocks_by_color_interruption as rgb_module
    from envs.tasks import rank_blocks_by_size_interruption as size_module

    spawned = []

    def create_primitive(_scene, shape, pose, **kwargs):
        assert shape == "box"
        block = _FakeBlock(pose.p, **kwargs)
        spawned.append(block)
        return block

    # The first target is the legacy implementation; the core target is the
    # composed implementation.  ``raising=False`` keeps this adapter harmless
    # after the duplicate task-local imports are deleted.
    monkeypatch.setattr(rgb_module, "create_primitive", create_primitive, raising=False)
    monkeypatch.setattr(size_module, "create_primitive", create_primitive, raising=False)
    monkeypatch.setattr(
        rgb_core_module, "create_primitive", create_primitive, raising=False
    )

    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    _seed_episode(task, seed)
    task.load_actors()
    tail = np.random.random_sample(8)
    return spawned, tail


@pytest.mark.parametrize("seed", EPISODE_SEEDS)
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_rgb_interrupt_preserves_global_rng_layout_and_retry(seed, setting, monkeypatch):
    task = RankBlocksByColorInterruption({"setting": setting})
    expected = _legacy_rgb_episode(task, seed)
    _spawned, tail = _load_fake_episode(task, seed, monkeypatch)

    assert tuple(task.blocks) == expected["order"]
    assert _interrupted_key(task) == expected["interrupted_key"]
    for key, block in task.blocks.items():
        np.testing.assert_array_equal(block.get_pos()[:2], expected["xys"][key])
    np.testing.assert_array_equal(tail, expected["tail"])
    if seed == 0:
        assert expected["attempts"] == 3


@pytest.mark.parametrize("seed", EPISODE_SEEDS)
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_size_interrupt_preserves_global_rng_color_layout_and_retry(
    seed, setting, monkeypatch
):
    task = RankBlocksBySizeInterruption({"setting": setting})
    expected = _legacy_size_episode(task, seed)
    _spawned, tail = _load_fake_episode(task, seed, monkeypatch)

    assert tuple(task.blocks) == expected["order"]
    assert task.block_colors == expected["colors"]
    assert _interrupted_key(task) == expected["interrupted_key"]
    for key, block in task.blocks.items():
        np.testing.assert_array_equal(block.get_pos()[:2], expected["xys"][key])
    np.testing.assert_array_equal(tail, expected["tail"])
    if seed == 0:
        assert expected["attempts"] == 3


@pytest.mark.parametrize("seed,golden", INDEPENDENT_STREAM_GOLDENS.items())
@pytest.mark.parametrize(
    "task_cls,keys",
    (
        (RankBlocksByColorInterruption, ("red", "green", "blue")),
        (RankBlocksBySizeInterruption, ("large", "medium", "small")),
    ),
)
def test_target_and_prefix_use_independent_seeded_stream_goldens(
    task_cls, keys, seed, golden, monkeypatch
):
    task = task_cls({})
    _load_fake_episode(task, seed, monkeypatch)
    target_index, prefix = golden

    assert _interrupted_key(task) == keys[target_index]
    assert _scripted_prefix(task) == prefix


@pytest.mark.parametrize(
    "task_cls,core,module_name,instruction",
    (
        (
            RankBlocksByColorInterruption,
            BlocksRankingRGB,
            "envs.tasks.rank_blocks_by_color_interruption",
            "place the red, green, and blue blocks in left-to-right order; "
            "wait while the human inspects one block, then continue",
        ),
        (
            RankBlocksBySizeInterruption,
            BlocksRankingSize,
            "envs.tasks.rank_blocks_by_size_interruption",
            "place the RGB blocks in largest-to-smallest order; wait while the "
            "human inspects one block, then continue",
        ),
    ),
)
def test_interrupt_public_class_paths_signatures_and_family_boundary(
    task_cls, core, module_name, instruction
):
    assert task_cls.__module__ == module_name
    assert issubclass(task_cls, core)
    assert task_cls.use_avatar is True
    assert task_cls.INSTRUCTION == instruction
    task = task_cls({})
    assert _is_composed(task)
    assert type(task.interaction_controller).__name__ == (
        "BlockRankingInterruptController"
    )
    assert task.interaction_controller.handles_policy_ticks is True
    assert task.interaction_controller.handles_scripted_rollout_ticks is True

    constructor = inspect.signature(task_cls)
    assert tuple(constructor.parameters) == ("config",)
    assert constructor.parameters["config"].default is None

    reset = inspect.signature(task_cls.reset)
    assert tuple(reset.parameters) == ("self", "seed")
    assert reset.parameters["seed"].default == 0
    action = inspect.signature(task_cls.take_action)
    assert tuple(action.parameters) == ("self", "action", "action_type")
    assert action.parameters["action_type"].default == "qpos"
    pick = inspect.signature(task_cls._pick_and_place_block)
    assert tuple(pick.parameters) == (
        "self",
        "color_name" if core is BlocksRankingRGB else "size_name",
        "arm_tag",
    )
    assert pick.parameters["arm_tag"].default == "right"
    assert tuple(inspect.signature(task_cls.play_once).parameters) == ("self",)


INTERRUPT_ONLY_ATTRIBUTES = (
    "_attach_frame_scaled",
    "_detach_frame_scaled",
    "_inspect_attach_frame_scaled",
    "_inspect_detach_frame_scaled",
    "_resolve_inspect_motion",
    "_prepare_inspect_motion_entry",
    "_play_inspect_motion",
    "_dryrun_capture_palm",
    "_calibrate_avatar_for_inspect",
    "_init_eval_interrupt_state",
    "_eval_step_avatar",
    "_find_closest_table_block",
    "_retreat_safe",
    "_wait_until_avatar_attach",
    "inspect_avatar_pos_for_object",
    "reset_avatar_for_inspect_object",
    "_inspect_idle_position_for_motion_root",
    "_set_inspect_motion_root_pos",
)


@pytest.mark.parametrize(
    "task_cls",
    (
        BlocksRankingRGB,
        BlocksRankingSize,
        RankBlocksByColorContribution,
        RankBlocksBySizeContribution,
    ),
)
def test_robot_only_and_assist_tasks_do_not_leak_interrupt_facade(task_cls):
    task = task_cls({})

    # This deliberately uses hasattr rather than static class inspection: the
    # former core-level facade exposed properties whose lookup raised a
    # TypeError on non-interrupt tasks.
    for name in INTERRUPT_ONLY_ATTRIBUTES:
        assert not hasattr(task, name), name


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption)
)
def test_interrupt_tasks_retain_inspect_and_reset_compatibility_api(task_cls):
    task = task_cls({})

    assert task._attach_frame_scaled == 340
    assert task._detach_frame_scaled == 1870
    assert task._inspect_attach_frame_scaled == 340
    assert task._inspect_detach_frame_scaled == 1870
    for name in INTERRUPT_ONLY_ATTRIBUTES[4:]:
        assert callable(getattr(task, name)), name


def _active_size_interrupt_checkpoint() -> dict:
    # The checkpoint carries the shared progress trigger and the
    # human sequence (glide-in in progress) instead of the legacy glide fields.
    from envs.interaction.interrupt_progress import (
        InterruptEpisode,
        InterruptHumanSequence,
        ProgressTrigger,
    )

    task = RankBlocksBySizeInterruption({})
    controller = task.interaction_controller
    controller.interrupted_key = "medium"
    controller.scripted_prefix_count = 2
    controller.eval_palm_default = np.array([0.01, 0.02, 0.03])
    controller.eval_legacy_trigger_draw = 137
    controller.eval_interrupt_fired = True
    controller.eval_target_key = "medium"
    controller.eval_interrupt_started = True
    controller.released_interrupted_pos = np.array([0.11, 0.07, 0.79])
    controller.policy_step_count = 29
    trigger = ProgressTrigger()
    trigger.arm(np.array([0.0, -0.3, 1.1]), {"medium": np.array([0.12, 0.08, 0.82])})
    assert trigger.update(25, np.array([0.12, 0.05, 0.95]),
                          {"medium": np.array([0.12, 0.08, 0.82])}) == "medium"
    human = InterruptHumanSequence(glide_steps=30)
    human.start(step=25, start_root=np.array([0.0, 0.55, -0.38]),
                target_root=np.array([0.12, 0.08, 0.82]), rot=np.eye(3),
                target_key="medium", target_pos=np.array([0.12, 0.08, 0.82]),
                start_clip=lambda: True)
    human._k = 4
    controller.eval_episode = InterruptEpisode(trigger=trigger, human=human,
                                               policy_steps=29)
    return controller.state_dict()


def test_size_interrupt_active_glide_state_roundtrip_and_task_projection():
    checkpoint = _active_size_interrupt_checkpoint()
    assert checkpoint["schema_version"].endswith(".v2")
    task = RankBlocksBySizeInterruption({})
    task.blocks = {"medium": _FakeBlock([0.12, 0.08, 0.82])}
    controller = task.interaction_controller

    controller.restore_task_state(task, checkpoint)

    assert controller.state_dict() == checkpoint
    assert task._interaction_policy_step == 29
    assert task._interrupted_size == "medium"
    assert task._scripted_prefix_count == 2
    assert task._eval_trigger_step == 25
    assert task._eval_interrupt_fired is True
    assert task._eval_glide_active is True
    assert task._eval_target_block is task.blocks["medium"]
    assert task._eval_target_size == "medium"
    assert task._eval_interrupt_started is True
    assert task._interrupt is controller.eval_episode
    # The clip callback is rebound on restore.
    assert controller.eval_episode.human._start_clip is not None
    np.testing.assert_array_equal(task._eval_palm_default, [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(
        task._released_interrupted_pos, [0.11, 0.07, 0.79]
    )


def _malformed(field, value):
    checkpoint = _active_size_interrupt_checkpoint()
    if field == "human_phase_without_fire":
        checkpoint["eval_interrupt_fired"] = False
    elif field == "human_without_target":
        checkpoint["eval_target_key"] = None
    elif field == "trigger_without_human":
        checkpoint["interrupt_human"] = None
    else:
        checkpoint[field] = value
    return checkpoint


@pytest.mark.parametrize(
    "field,value",
    (
        ("human_phase_without_fire", None),
        ("human_without_target", None),
        ("trigger_without_human", None),
        ("policy_step_count", -1),
    ),
)
def test_malformed_active_glide_state_is_rejected_atomically(field, value):
    task = RankBlocksBySizeInterruption({})
    controller = task.interaction_controller
    before = controller.state_dict()
    malformed = _malformed(field, value)

    with pytest.raises(ValueError):
        controller.load_state_dict(malformed)

    assert controller.state_dict() == before


def test_v1_interrupt_state_schema_is_rejected():
    task = RankBlocksByColorInterruption({})
    controller = task.interaction_controller
    state = controller.state_dict()
    state["schema_version"] = "genesis_hr_bench.block_ranking_interrupt_state.v1"
    with pytest.raises(ValueError, match="schema"):
        controller.load_state_dict(state)


def test_armed_interrupt_state_requires_calibrated_palm_atomically():
    from envs.interaction.interrupt_progress import (
        InterruptHumanSequence,
        ProgressTrigger,
    )

    task = RankBlocksBySizeInterruption({})
    controller = task.interaction_controller
    before = controller.state_dict()
    trigger = ProgressTrigger()
    trigger.arm(np.zeros(3), {})
    malformed = dict(before)
    malformed.update(
        eval_interrupt_fired=False,
        eval_palm_default=None,
        interrupt_trigger=trigger.state_dict(),
        interrupt_human=InterruptHumanSequence().state_dict(),
    )

    with pytest.raises(ValueError, match="calibrated palm"):
        controller.load_state_dict(malformed)

    assert controller.state_dict() == before


def test_restore_with_target_requires_initialized_blocks_before_mutation():
    checkpoint = _active_size_interrupt_checkpoint()
    task = RankBlocksBySizeInterruption({})
    controller = task.interaction_controller
    before = controller.state_dict()

    with pytest.raises(RuntimeError, match="before task blocks"):
        controller.restore_task_state(task, checkpoint)

    assert controller.state_dict() == before
    assert not hasattr(task, "_eval_target_block")


@pytest.mark.parametrize(
    "field,value",
    (
        ("eval_interrupt_started", True),
        ("released_interrupted_pos", [0.1, -0.2, 0.8]),
    ),
)
def test_rgb_interrupt_rejects_size_only_state_atomically(field, value):
    task = RankBlocksByColorInterruption({})
    controller = task.interaction_controller
    before = controller.state_dict()
    malformed = dict(before)
    malformed["interrupted_key"] = "green"
    malformed[field] = value

    with pytest.raises(ValueError, match="size-only"):
        controller.load_state_dict(malformed)

    assert controller.state_dict() == before


class _FakeTcpArm:
    gripper_val = 1.0

    def __init__(self, tcp=(0.0, -0.35, 1.10)):
        self.tcp = np.asarray(tcp, dtype=np.float64)

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


class _FakeTcpRobot:
    left_arm = None

    def __init__(self, tcp=(0.0, -0.35, 1.10)):
        self.arm = _FakeTcpArm(tcp)

    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return self.arm


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption)
)
def test_eval_trigger_config_override_is_effective_and_defaults_remain_legacy(
    task_cls,
):
    # Eval_trigger_step_min/max only bound the burned legacy RNG
    # draw (scene RNG parity); they no longer schedule the interrupt.  The
    # progress trigger takes its thresholds from INTERRUPT_* class attributes
    # and interrupt_* config overrides.
    assert task_cls.EVAL_TRIGGER_STEP_MIN == 100
    assert task_cls.EVAL_TRIGGER_STEP_MAX == 200
    assert task_cls.INTERRUPT_COMMIT_RADIUS_M == 0.20

    override_task = task_cls(
        {
            "eval_mode": True,
            "eval_trigger_step_min": 5,
            "eval_trigger_step_max": 5,
            "interrupt_commit_radius_m": 0.23,
        }
    )
    override_task.robot = _FakeTcpRobot()
    override_task._dryrun_capture_palm = lambda: np.zeros(3)
    override_task.interaction_controller.init_eval_state(override_task)
    controller = override_task.interaction_controller
    assert controller.eval_legacy_trigger_draw == 5
    assert override_task._eval_trigger_step is None
    assert controller.eval_episode.trigger.params.commit_radius_m == 0.23

    default_task = task_cls({"eval_mode": True})
    default_task.robot = _FakeTcpRobot()
    default_task._dryrun_capture_palm = lambda: np.zeros(3)
    np.random.seed(113)
    expected = int(np.random.randint(100, 201))
    tail = np.random.rand()
    np.random.seed(113)
    default_task.interaction_controller.init_eval_state(default_task)
    assert default_task.interaction_controller.eval_legacy_trigger_draw == expected
    assert np.random.rand() == tail  # RNG tail unchanged
    params = default_task.interaction_controller.eval_episode.trigger.params
    assert params.commit_radius_m == 0.20 and params.min_policy_steps == 20


@pytest.mark.parametrize(
    "task_cls,noun,key",
    (
        (RankBlocksByColorInterruption, "color", "green"),
        (RankBlocksBySizeInterruption, "size", "medium"),
    ),
)
def test_scene_init_work_allocation_projection_is_adopted(task_cls, noun, key):
    task = task_cls({})
    controller = task.interaction_controller
    setattr(task, f"_interrupted_{noun}", key)
    task._scripted_prefix_count = 1
    if noun == "size":
        task._released_interrupted_pos = np.array([0.1, -0.2, 0.8])
        task._eval_target_size = "small"
        task._eval_interrupt_started = True

    controller.on_base_reset_ready(task)

    assert controller.interrupted_key == key
    assert controller.scripted_prefix_count == 1
    assert getattr(task, f"_interrupted_{noun}") == key
    assert task._scripted_prefix_count == 1
    if noun == "size":
        np.testing.assert_array_equal(
            controller.released_interrupted_pos, [0.1, -0.2, 0.8]
        )
        assert controller.eval_target_key == "small"
        assert controller.eval_interrupt_started is True


class _FakeScene:
    def build(self):
        return None

    def step(self):
        return None


class _FakeTable:
    def get_pos(self):
        return np.array([0.0, 0.0, 0.74])


class _FakeResetRobot(_FakeTcpRobot):
    left_arm = None

    def __init__(self):
        super().__init__()

    def init_joints(self, _scene):
        return None

    def move_to_homestate(self):
        return None

    def open_gripper(self, _arm):
        return None

    def set_origin_endpose(self):
        return None


def _stub_reset_dependencies(task, monkeypatch, events):
    from envs import base_task as base_module

    avatar = _FakeAvatar(events)
    observation = object()

    def setup_scene():
        task.scene = _FakeScene()

    def create_table():
        task.table = _FakeTable()
        task.TABLE_THICKNESS = 0.05
        task.TABLE_TOP_Z = 0.765
        task._table_top_z_fixed = False

    def load_robot():
        task.robot = _FakeResetRobot()

    task._setup_scene = setup_scene
    task._create_table = create_table
    task._load_robot = load_robot
    task._load_cameras = lambda: setattr(task, "cameras", None)
    task.load_actors = lambda: events.append(("load_actors", _scripted_prefix(task)))
    task._spawn_table_random_objects = lambda: None
    task._init_avatar = lambda: setattr(task, "avatar", avatar)
    task._init_avatar_collider = lambda: None
    task._init_avatar_collision_checker = lambda: None
    task._setup_batch_render_cache = lambda: None
    task._apply_policy_gripper_gains = lambda: None
    task._settle_scene = lambda _steps: events.append(("settle", None))
    capture_palm = lambda *_args: (
        events.append(("eval_init", None)),
        np.zeros(3, dtype=np.float64),
    )[1]
    task._dryrun_capture_palm = capture_palm
    if _is_composed(task):
        task.interaction_controller.dryrun_capture_palm = capture_palm

    def get_obs():
        events.append(("initial_observation", None))
        return observation

    task.get_obs = get_obs
    monkeypatch.setattr(base_module, "apply_scene_init_pre_avatar", lambda _task: None)
    monkeypatch.setattr(
        base_module,
        "apply_scene_init_post_settle",
        lambda _task: {"enabled": False},
    )
    return avatar, observation


@pytest.mark.parametrize("task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption))
def test_eval_reset_seeds_choreography_before_scene_and_initializes_after_observation(
    task_cls, monkeypatch
):
    task = task_cls({"eval_mode": True})
    events = []
    avatar, observation = _stub_reset_dependencies(task, monkeypatch, events)
    controller = task.interaction_controller
    original_after_observation = controller.after_base_reset_observation

    def after_observation(bound_task, *, observation):
        events.append(("after_observation_hook", observation))
        return original_after_observation(bound_task, observation=observation)

    controller.after_base_reset_observation = after_observation
    returned = task.reset(seed=0)

    assert returned is observation
    assert events[0] == ("load_actors", 2)
    labels = [event[0] for event in events]
    assert labels.index("initial_observation") < labels.index("after_observation_hook")
    assert labels.index("initial_observation") < labels.index("eval_init")
    assert avatar.frame_ratio == task.AVATAR_MOTION_SLOW
    assert int(_read_any(task, "_policy_step_count", "policy_step_count", default=0)) == 0
    assert _read_any(task, "_eval_interrupt_fired", "eval_interrupt_fired") is False
    # The legacy draw is burned (not scheduled); nothing fired yet.
    assert task.EVAL_TRIGGER_STEP_MIN <= int(
        task.interaction_controller.eval_legacy_trigger_draw
    ) <= task.EVAL_TRIGGER_STEP_MAX
    assert task._eval_trigger_step is None
    assert task._interrupt.trigger.armed


@pytest.mark.parametrize("task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption))
def test_eval_policy_tick_occurs_before_early_success_observation(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    events = []
    controller = task.interaction_controller

    if _is_composed(task):
        def tick(bound_task, *, step_index):
            assert bound_task is task
            events.append(("tick", step_index))

        controller.on_policy_tick = tick
    else:
        task._policy_step_count = 0

        def tick():
            task._policy_step_count += 1
            events.append(("tick", task._policy_step_count))

        task._eval_step_avatar = tick

    observation = object()
    task.get_obs = lambda: (events.append(("observation", None)), observation)[1]

    assert task.take_action(np.zeros(8), action_type="qpos") is observation
    assert events == [("tick", 1), ("observation", None)]


@pytest.mark.parametrize(
    "eval_mode,hook_name",
    (
        (True, "_init_eval_interrupt_state"),
        (False, "_calibrate_avatar_for_inspect"),
    ),
)
def test_reset_honors_task_level_interrupt_compatibility_overrides(
    eval_mode, hook_name
):
    task = RankBlocksByColorInterruption({"eval_mode": eval_mode})
    task.avatar = type("Avatar", (), {"frame_ratio": None})()
    events = []
    task._resolve_inspect_motion = lambda: events.append("resolve")
    setattr(task, hook_name, lambda: events.append(hook_name))

    task.interaction_controller.after_base_reset_observation(
        task, observation=None
    )

    assert events == ["resolve", hook_name]


def test_policy_tick_honors_task_level_interrupt_compatibility_override():
    task = RankBlocksByColorInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append("custom_tick")

    task.interaction_controller.on_policy_tick(task, step_index=17)

    assert events == ["custom_tick"]
    assert task.interaction_controller.policy_step_count == 0


class _FakeScriptArm:
    tcp_offset = Pose()


class _FakeScriptRobot:
    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return _FakeScriptArm()


def _prepare_script_task(task, interrupted_key: str, prefix: int):
    _set_interrupt_allocation(task, interrupted_key, prefix)
    task.TABLE_TOP_Z = 0.765
    task.blocks = {
        key: _FakeBlock([0.07 * index, -0.02 + 0.02 * index, 0.80])
        for index, key in enumerate(task.block_keys())
    }
    task.target_slots = {
        key: np.array([-0.16 + 0.16 * index, -0.08, 0.80])
        for index, key in enumerate(task.block_keys())
    }
    if isinstance(task, BlocksRankingSize):
        task.block_colors = dict(zip(task.SIZE_ORDER, ("red", "green", "blue")))
        task._released_interrupted_pos = None
    task.robot = _FakeScriptRobot()
    task.avatar = _FakeAvatar()
    task.TABLE_SETTLE_STEPS = 2
    task.POST_ATTACH_WAIT_MARGIN_STEPS = 2
    task.POST_RELEASE_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 2


def _install_script_trace(task):
    events = []
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim", None))
    task._pick_and_place_block = lambda key, arm: (
        events.append(("pick", key)),
        True,
    )[1]
    task.open_gripper = lambda arm: events.append(("open", arm))
    task._top_down_tcp = lambda pos: Pose(pos)
    task._move_seeded = lambda _pose, arm: events.append(("intent", arm))
    task._play_inspect_motion = lambda **kwargs: events.append(
        ("inspect", next(key for key, value in task.blocks.items() if value is kwargs["attach_obj"]))
    )
    task._retreat_safe = lambda arm: events.append(("retreat", arm))
    task._wait_until_avatar_attach = lambda: events.append(("wait_attach", None))
    if _is_composed(task):
        controller = task.interaction_controller
        controller.play_inspect_motion = lambda _task, **kwargs: events.append(
            (
                "inspect",
                next(
                    key
                    for key, value in task.blocks.items()
                    if value is kwargs["attach_obj"]
                ),
            )
        )
        controller.retreat_safe = lambda _task, arm: events.append(
            ("retreat", arm)
        )
        controller.wait_until_avatar_attach = lambda _task: events.append(
            ("wait_attach", None)
        )
    return events


@pytest.mark.parametrize("prefix", (0, 1, 2))
@pytest.mark.parametrize(
    "task_cls,interrupted_key",
    (
        (RankBlocksByColorInterruption, "green"),
        (RankBlocksBySizeInterruption, "medium"),
    ),
)
def test_scripted_interrupt_event_trace_for_every_prefix(
    task_cls, interrupted_key, prefix
):
    task = task_cls({})
    _prepare_script_task(task, interrupted_key, prefix)
    events = _install_script_trace(task)
    other_keys = tuple(key for key in task.block_keys() if key != interrupted_key)

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        *[("pick", key) for key in other_keys[:prefix]],
        ("open", "right"),
        ("intent", "right"),
        ("inspect", interrupted_key),
        ("retreat", "right"),
        ("wait_attach", None),
        ("sim", None),
        ("sim", None),
        *[("pick", key) for key in other_keys[prefix:]],
        ("sim", None),
        ("sim", None),
        ("pick", interrupted_key),
        ("sim", None),
        ("sim", None),
    ]
    if isinstance(task, BlocksRankingSize):
        np.testing.assert_array_equal(
            task._released_interrupted_pos,
            task.blocks[interrupted_key].get_pos(),
        )


@pytest.mark.parametrize("task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption))
def test_direct_no_avatar_interrupt_preserves_double_boost_and_settle(task_cls):
    task = task_cls({})
    task.avatar = None
    task.TABLE_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 1
    events = _install_script_trace(task)

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        ("boost", "right"),
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        *[("pick", key) for key in task.block_keys()],
        ("sim", None),
    ]


class _FakeAvatarRobot:
    def __init__(self):
        self.global_trans = np.array([0.0, 0.55, -0.38])

    def get_palm_center(self, hand_id):
        # Idle hand hangs ~1 m from any block.
        return self.global_trans + np.array([0.0, 0.0, 2.0])


class _FakeEvalAvatar(_FakeAvatar):
    def __init__(self):
        super().__init__()
        self.robot = _FakeAvatarRobot()
        self.busy = False

    def reset(self, pos, rot, **_kwargs):
        super().reset(pos, rot)
        self.robot.global_trans = np.asarray(pos, dtype=np.float64).copy()

    def spare(self):
        return not self.busy


def _initialize_eval_state(task):
    task.interaction_controller.after_base_reset_observation(
        task, observation=None
    )


def _eval_tick(task):
    assert task.advance_interaction_policy_step() is True


@pytest.mark.parametrize(
    "task_cls,target_key",
    (
        (RankBlocksByColorInterruption, "green"),
        (RankBlocksBySizeInterruption, "medium"),
    ),
)
def test_eval_progress_trigger_glide_inspect_and_idle_return(task_cls, target_key):
    # Replaces the random-step trigger + 20-tick glide to the block
    # closest to the gripper.  The trigger fires on robot progress toward the
    # approached block, the avatar glides from its current root for 30 ticks
    # (no teleport), the clip returns to idle, and the avatar glides back.
    task = task_cls({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    robot = _FakeTcpRobot(tcp=(0.0, -0.35, 1.10))
    task.robot = robot
    task.avatar = _FakeEvalAvatar()
    task._resolve_inspect_motion = lambda: None
    task._interaction_policy_step = 0
    keys = task.block_keys()
    target_positions = {
        keys[0]: np.array([-0.16, -0.08, 0.80]),  # already sorted
        keys[1]: np.array([0.12, 0.02, 0.80]),    # eligible, approached
        keys[2]: np.array([-0.10, 0.05, 0.80]),   # eligible, not approached
    }
    task.blocks = {key: _FakeBlock(target_positions[key]) for key in keys}
    task.target_slots = {
        keys[0]: target_positions[keys[0]].copy(),
        keys[1]: np.array([0.80, 0.80, 0.80]),
        keys[2]: np.array([-0.80, 0.80, 0.80]),
    }
    if isinstance(task, BlocksRankingSize):
        task.block_colors = dict(zip(keys, ("red", "green", "blue")))
    palm_default = np.array([0.01, 0.02, 0.03])
    inspect_calls = []
    controller = task.interaction_controller
    controller.dryrun_capture_palm = lambda _task: palm_default.copy()
    controller.play_inspect_motion = (
        lambda _task, **kwargs: (inspect_calls.append(kwargs),
                                 setattr(task.avatar, "busy", True))
    )

    np.random.seed(17)
    _initialize_eval_state(task)
    assert task._interrupt.trigger.armed

    # 25 ticks at the reset pose: never fires (robot did not move).
    for _ in range(25):
        _eval_tick(task)
    assert not task._eval_interrupt_fired
    assert task._interrupt.trigger.block_reason == "robot_not_moved"
    assert task.avatar.reset_positions == []

    # Hover over the approached block: fires on the next tick (RGB: once the
    # hover over its reset-chosen block has lasted the dwell).
    dwell = controller.hover_dwell_params(task)
    if dwell is not None:
        controller.interrupted_key = keys[1]
    n_hover = dwell["steps"] if dwell is not None else 1
    robot.arm.tcp = target_positions[keys[1]] + np.array([0.0, 0.0, 0.15])
    for _ in range(n_hover):
        _eval_tick(task)
    assert task._eval_interrupt_fired
    assert task._interrupt.trigger.trigger_step == 25 + n_hover
    assert task._interrupt.trigger.trigger_target == target_key
    assert task._eval_target_block is task.blocks[target_key]

    start = np.array([0.0, 0.55, -0.38])
    expected_target = (
        np.asarray(task.avatar_init_pos, dtype=np.float64)
        + target_positions[keys[1]] - palm_default
    )
    expected_target[2] += task.AVATAR_EXTRA_Z
    for glide_tick in range(1, 31):
        _eval_tick(task)
        expected = start + (expected_target - start) * (glide_tick / 30.0)
        np.testing.assert_allclose(task.avatar.reset_positions[-1], expected)
        assert len(inspect_calls) == (1 if glide_tick == 30 else 0)
    jumps = np.linalg.norm(np.diff(np.asarray(task.avatar.reset_positions), axis=0), axis=1)
    assert float(jumps.max()) <= np.linalg.norm(expected_target - start) / 30 + 1e-9

    call = inspect_calls[0]
    assert call["attach_obj"] is task.blocks[target_key]
    assert call["hand_id"] == task.AVATAR_HAND_ID
    assert call["attach_frame"] == 340
    assert call["detach_frame"] == 1870
    assert call["return_to_idle_after"] == task._interrupt.human.idle_return_frames
    if isinstance(task, BlocksRankingSize):
        assert task._eval_interrupt_started

    # Clip (with its idle blend) ends; the avatar glides back to its root.
    task.avatar.busy = False
    for _ in range(1 + task._interrupt.human.glide_out_steps):
        _eval_tick(task)
    assert task._interrupt.human.done
    np.testing.assert_allclose(task.avatar.reset_positions[-1], start)
    metrics = task.interaction_controller.interaction_metrics(task)
    assert metrics["interrupt"]["fired"] is True
    assert metrics["interrupt"]["trigger_step"] == 25 + n_hover
    assert metrics["pace_interrupt"]["trigger_mode"] == "progress"


@pytest.mark.parametrize("task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption))
def test_eval_hold_still_robot_never_interrupts(task_cls):
    task = task_cls({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    # Reset TCP right above a block: the retired force_dist trigger fired here.
    task.robot = _FakeTcpRobot(tcp=(0.12, 0.02, 0.92))
    task.avatar = _FakeEvalAvatar()
    task._resolve_inspect_motion = lambda: None
    task._interaction_policy_step = 0
    keys = task.block_keys()
    task.blocks = {
        key: _FakeBlock([0.12 - 0.1 * i, 0.02, 0.80]) for i, key in enumerate(keys)
    }
    task.target_slots = {key: np.array([0.9, 0.9, 0.8]) for key in keys}
    if isinstance(task, BlocksRankingSize):
        task.block_colors = dict(zip(keys, ("red", "green", "blue")))
    task.interaction_controller.dryrun_capture_palm = lambda _task: np.zeros(3)
    _initialize_eval_state(task)
    for _ in range(300):
        _eval_tick(task)
    assert not task._eval_interrupt_fired
    assert task.avatar.reset_positions == []
    metrics = task.interaction_controller.interaction_metrics(task)
    assert metrics["interrupt"]["fired"] is False
    assert metrics["interrupt"]["not_fired_reason"] == "robot_not_moved"


def _set_metric_state(task) -> None:
    _set_interrupt_allocation(task, "medium", 2)
    task.block_colors = {"large": "red", "medium": "green", "small": "blue"}
    _set_on_task_and_owner(task, "_eval_trigger_step", 137)
    _set_on_task_and_owner(task, "_policy_step_count", 9)
    _set_on_task_and_owner(task, "_eval_interrupt_fired", True)
    _set_on_task_and_owner(task, "_eval_interrupt_started", True)
    _set_on_task_and_owner(task, "_eval_target_size", "medium")
    _set_on_task_and_owner(
        task, "_released_interrupted_pos", np.array([0.1, -0.2, 0.8])
    )
    if _is_composed(task):
        controller = task.interaction_controller
        controller.eval_target_key = "medium"
        controller.released_interrupted_pos = np.array([0.1, -0.2, 0.8])


def test_rgb_interrupt_adds_only_interrupt_contract_metrics(monkeypatch):
    # Every interrupt task reports the uniform interrupt block in
    # eval mode (fired / not fired + reason); outside eval mode nothing.
    monkeypatch.setattr(BlocksRankingRGB, "evaluate", lambda _self: {"base": 1})
    task = RankBlocksByColorInterruption({"eval_mode": True})
    _set_interrupt_allocation(task, "green", 2)

    out = task.evaluate()
    assert set(out) == {"base", "interrupt", "pace_interrupt"}
    assert out["interrupt"]["fired"] is False
    assert RankBlocksByColorInterruption({}).interaction_controller.interaction_metrics(
        RankBlocksByColorInterruption({})
    ) == {}


def test_size_interrupt_metrics_preserve_exact_release_surface(monkeypatch):
    monkeypatch.setattr(BlocksRankingSize, "evaluate", lambda _self: {"base": 1})
    task = RankBlocksBySizeInterruption({"eval_mode": True})
    _set_metric_state(task)
    expected_additions = {
        "interrupted_size": "medium",
        "interrupted_color": "green",
        "scripted_prefix_count": 2,
        "interrupt_motion": "Inspect1",
        "interrupt_attach_frame": 34,
        "interrupt_detach_frame": 187,
        "eval_mode": True,
        # The step the progress trigger fired at (none here).
        "eval_trigger_step": None,
        "eval_policy_step_count": 9,
        "eval_interrupt_fired": True,
        "eval_interrupt_started": True,
        "eval_target_size": "medium",
        "released_interrupted_pos": [0.1, -0.2, 0.8],
    }

    out = task.evaluate()
    # Plus the uniform interrupt blocks in eval mode.
    assert {k: v for k, v in out.items() if k not in ("interrupt", "pace_interrupt")} == {
        "base": 1, **expected_additions
    }
    assert out["interrupt"]["not_fired_reason"] == "interrupt_contract_inactive"


@pytest.mark.parametrize(
    "task_name,task_cls,parent",
    (
        (
            "rank_blocks_by_color_interruption",
            RankBlocksByColorInterruption,
            BlocksRankingRGB.play_once,
        ),
        (
            "rank_blocks_by_size_interruption",
            RankBlocksBySizeInterruption,
            BlocksRankingSize.play_once,
        ),
    ),
)
def test_plain_expert_routes_to_unprivileged_family_core(
    task_name, task_cls, parent
):
    assert policy_play_mode(task_name, task_cls, "expert") == (
        "parent_play_once",
        parent,
    )
    # The full-state expert runs the eval-mode interrupt rollout,
    # and the harness no longer overrides the task's trigger.
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        "interrupt_full_state",
        None,
    )
    cfg = {}
    configure_parent_rollout_eval(cfg, task_name, "parent_play_once")
    assert cfg == {}


@pytest.mark.parametrize("task_cls", (RankBlocksByColorInterruption, RankBlocksBySizeInterruption))
def test_parent_expert_driver_ticks_at_policy_cadence_and_restores_step(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    ticks = []
    original_step = lambda: None
    task.step_sim = original_step
    controller = task.interaction_controller

    if _is_composed(task):
        assert controller.handles_scripted_rollout_ticks is True

        def tick(bound_task, *, step_index):
            assert bound_task is task
            ticks.append(step_index)

        controller.on_policy_tick = tick
    else:
        task._eval_step_avatar = lambda: ticks.append(len(ticks) + 1)

    def rollout():
        for _ in range(11):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(
        task, rollout, {"eval_parent_rollout_tick_interval": 3}
    )
    assert ticks == [1, 2, 3]
    assert task.step_sim is original_step
