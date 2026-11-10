"""Scene-free release contracts for composed block-ranking assist behavior."""

from __future__ import annotations

from contextlib import contextmanager
import gc
from types import SimpleNamespace
import weakref
import zlib

import numpy as np
import pytest

from envs.avatar.eval_mode_mixin import EvalModeAvatarMixin
from envs.interaction import InteractionMode
from envs.interaction.block_ranking import BlockRankingAssistController
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.task_bases.blocks_ranking_size import BlocksRankingSize
from envs.tasks.rank_blocks_by_color_contribution import RankBlocksByColorContribution
from envs.tasks.rank_blocks_by_size_contribution import RankBlocksBySizeContribution
from scripts.eval_expert_set import policy_play_mode, run_with_eval_step_driver


# Includes rejection/retry layouts in both families (RGB seed 0, size seed 8).
SEEDS = (0, 1, 7, 8, 42, 113)


def _legacy_assist_layout(task, keys, avatar_key):
    """Copy the release-v1 scalar draw order, not controller internals."""

    for attempt in range(int(task.AVATAR_LAYOUT_SAMPLE_ATTEMPTS)):
        candidate = {
            avatar_key: task._layout_xy(
                [
                    np.random.uniform(*task.AVATAR_SPAWN_X),
                    np.random.uniform(*task.AVATAR_SPAWN_Y),
                ]
            )
        }
        for key in keys:
            if key == avatar_key:
                continue
            candidate[key] = task._layout_xy(
                [
                    np.random.uniform(*task.ROBOT_SPAWN_X),
                    np.random.uniform(*task.ROBOT_SPAWN_Y),
                ]
            )

        valid = True
        for index, first in enumerate(keys):
            for second in keys[index + 1 :]:
                if (
                    np.linalg.norm(candidate[first] - candidate[second])
                    < task.MIN_BLOCK_SPACING
                ):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return candidate, attempt + 1
    raise RuntimeError("legacy assist layout did not find a valid sample")


def _legacy_rgb_episode(task, seed):
    np.random.seed(seed)
    keys = tuple(task.CUBE_COLORS)
    avatar_key = str(np.random.choice(("red", "green", "blue")))
    robot_keys = tuple(key for key in keys if key != avatar_key)
    delay = int(
        np.random.randint(
            task.AVATAR_START_DELAY_STEPS[0],
            task.AVATAR_START_DELAY_STEPS[1] + 1,
        )
    )
    xys, attempts = _legacy_assist_layout(task, keys, avatar_key)
    creation_order = tuple(np.random.permutation(list(task.CUBE_COLORS)))
    tail = np.random.random_sample(8)
    return {
        "avatar_key": avatar_key,
        "robot_keys": robot_keys,
        "delay": delay,
        "xys": xys,
        "attempts": attempts,
        "creation_order": creation_order,
        "colors": {key: key for key in creation_order},
        "tail": tail,
    }


def _legacy_size_episode(task, seed):
    np.random.seed(seed)
    keys = tuple(task.SIZE_ORDER)
    color_order = tuple(np.random.permutation(list(task.CUBE_COLORS)))
    creation_order = tuple(np.random.permutation(list(task.SIZE_ORDER)))
    avatar_key = str(np.random.choice(task.SIZE_ORDER))
    robot_keys = tuple(key for key in task.SIZE_ORDER if key != avatar_key)
    delay = int(
        np.random.randint(
            task.AVATAR_START_DELAY_STEPS[0],
            task.AVATAR_START_DELAY_STEPS[1] + 1,
        )
    )
    xys, attempts = _legacy_assist_layout(task, keys, avatar_key)
    tail = np.random.random_sample(8)
    return {
        "avatar_key": avatar_key,
        "robot_keys": robot_keys,
        "delay": delay,
        "xys": xys,
        "attempts": attempts,
        "creation_order": creation_order,
        "colors": dict(zip(creation_order, color_order)),
        "tail": tail,
    }


def _actual_episode(task, seed):
    np.random.seed(seed)
    controller = task.interaction_controller
    plan = controller.create_block_episode(task)
    tail = np.random.random_sample(8)
    return controller, plan, tail


def _assert_episode_matches_legacy(task, expected, controller, plan, tail):
    assert plan.avatar_key == expected["avatar_key"]
    assert plan.robot_keys == expected["robot_keys"]
    assert controller.avatar_key == expected["avatar_key"]
    assert controller.robot_keys == expected["robot_keys"]
    assert controller.avatar_start_delay_steps == expected["delay"]
    assert tuple(spec.key for spec in plan.blocks) == expected["creation_order"]
    assert {spec.key: spec.color_name for spec in plan.blocks} == expected["colors"]
    for spec in plan.blocks:
        np.testing.assert_array_equal(
            np.asarray(spec.xy), expected["xys"][spec.key]
        )
        assert spec.half_size == task.block_half_size(spec.key)
    np.testing.assert_array_equal(tail, expected["tail"])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_rgb_assist_preserves_legacy_global_numpy_stream(seed, setting):
    task = RankBlocksByColorContribution({"setting": setting})

    expected = _legacy_rgb_episode(task, seed)
    controller, plan, tail = _actual_episode(task, seed)

    _assert_episode_matches_legacy(task, expected, controller, plan, tail)
    assert task._avatar_color == expected["avatar_key"]
    assert task._robot_colors == expected["robot_keys"]
    assert task._avatar_start_delay_steps == expected["delay"]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_size_assist_preserves_legacy_global_numpy_stream(seed, setting):
    task = RankBlocksBySizeContribution({"setting": setting})

    expected = _legacy_size_episode(task, seed)
    controller, plan, tail = _actual_episode(task, seed)

    _assert_episode_matches_legacy(task, expected, controller, plan, tail)
    assert task._avatar_size == expected["avatar_key"]
    assert task._robot_sizes == expected["robot_keys"]
    assert task._avatar_start_delay_steps == expected["delay"]


def test_assist_variants_use_direct_core_inheritance_and_assist_controller():
    assert RankBlocksByColorContribution.__bases__ == (BlocksRankingRGB,)
    assert RankBlocksBySizeContribution.__bases__ == (BlocksRankingSize,)
    assert EvalModeAvatarMixin not in RankBlocksByColorContribution.__mro__
    assert EvalModeAvatarMixin not in RankBlocksBySizeContribution.__mro__
    assert RankBlocksByColorContribution.AVATAR_COLOR_CHOICES == (
        "red",
        "green",
        "blue",
    )

    for task in (RankBlocksByColorContribution({}), RankBlocksBySizeContribution({})):
        controller = task.interaction_controller
        assert isinstance(controller, BlockRankingAssistController)
        assert controller.mode is InteractionMode.ASSIST
        assert controller.handles_policy_ticks is True
        assert task.config["track_avatar_collision"] is False


@pytest.mark.parametrize(
    ("task_cls", "owner_attr", "robot_attr"),
    (
        (RankBlocksByColorContribution, "_avatar_color", "_robot_colors"),
        (RankBlocksBySizeContribution, "_avatar_size", "_robot_sizes"),
    ),
)
def test_failed_layout_still_publishes_legacy_work_allocation(
    task_cls, owner_attr, robot_attr
):
    task = task_cls({})
    task.AVATAR_LAYOUT_SAMPLE_ATTEMPTS = 0

    with pytest.raises(RuntimeError, match="could not sample"):
        task.interaction_controller.create_block_episode(task)

    owner = getattr(task, owner_attr)
    robots = getattr(task, robot_attr)
    assert owner in task.block_keys()
    assert tuple(robots) == tuple(key for key in task.block_keys() if key != owner)
    assert task._avatar_start_delay_steps >= 0


class _FakeBlock:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()


class _FakeAvatar:
    def __init__(self):
        self.calls = []
        self.hold_frame = object()
        self.second_frame = object()
        self.third_frame = object()
        self.motion = SimpleNamespace(
            frames=[self.hold_frame, self.second_frame, self.third_frame],
            attach_frame=17,
            detach_frame=31,
        )
        self.motion_modules = {"pick_and_place": self.motion}

    def pick_and_place(self, **kwargs):
        self.calls.append(kwargs)


def _prepare_motion_task(task, seed=7):
    controller, plan, _tail = _actual_episode(task, seed)
    task.TABLE_TOP_Z = 0.735
    task.blocks = {
        key: _FakeBlock([0.11 * index, -0.21 - 0.03 * index, 0.81])
        for index, key in enumerate(task.block_keys())
    }
    task.target_slots = {
        key: np.array([-0.16 + 0.16 * index, -0.08, 0.0])
        for index, key in enumerate(task.block_keys())
    }
    task.avatar = _FakeAvatar()
    task.avatar_collided = True
    task.avatar_collision_log = ["old collision"]
    return controller, plan


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
def test_avatar_motion_parameters_and_scripted_delay_match_legacy(task_cls):
    task = task_cls({})
    controller, _plan = _prepare_motion_task(task)
    owner = controller.avatar_key
    avatar = task.avatar
    original_attach = avatar.motion.attach_frame
    original_detach = avatar.motion.detach_frame

    controller.start_avatar_assist(task, add_start_delay=True)

    assert len(avatar.calls) == 1
    call = avatar.calls[0]
    expected_pick = task.blocks[owner].get_pos()
    expected_pick[2] += task.AVATAR_PICK_LIFT
    expected_place = np.array(
        [
            task.target_slots[owner][0],
            task.target_slots[owner][1],
            task.TABLE_TOP_Z
            + task.block_half_size(owner)
            + task.AVATAR_PLACE_LIFT,
        ]
    )
    np.testing.assert_array_equal(call["pick_pos"], expected_pick)
    np.testing.assert_array_equal(call["place_pos"], expected_place)
    assert call["attach_obj"] is task.blocks[owner]
    assert call["hand_id"] is None
    assert call["approach_frames"] == task.AVATAR_APPROACH_FRAMES
    assert call["transport_frames"] == task.AVATAR_TRANSPORT_FRAMES
    assert call["retract_frames"] == task.AVATAR_RETRACT_FRAMES
    assert call["natural"] is True
    assert call["body_margin"] == task.AVATAR_NATURAL_BODY_MARGIN
    assert call["body_y_bounds"] == task.AVATAR_NATURAL_BODY_Y_BOUNDS
    assert call["yaw_limit_deg"] == task.AVATAR_NATURAL_YAW_LIMIT_DEG
    assert call["settle_steps"] == task.AVATAR_NATURAL_SETTLE_STEPS
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []

    delay = controller.avatar_start_delay_steps
    assert avatar.motion.frames[: delay + 1] == [avatar.hold_frame] * (delay + 1)
    assert avatar.motion.frames[delay + 1 :] == [
        avatar.second_frame,
        avatar.third_frame,
    ]
    assert avatar.motion.attach_frame == original_attach + delay
    assert avatar.motion.detach_frame == original_detach + delay


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
def test_eval_starts_at_reset_with_private_demo_window_delay_and_ticks_before_success_return(
    task_cls,
):
    task = task_cls({"eval_mode": True})
    controller, _plan = _prepare_motion_task(task)
    avatar = task.avatar
    observation = object()

    global_state = np.random.get_state()
    controller.after_base_reset_observation(task, observation=observation)
    after_state = np.random.get_state()

    # The delay is a private per-episode draw from the demos' window and
    # never touches the global stream.
    assert after_state[0] == global_state[0]
    np.testing.assert_array_equal(after_state[1], global_state[1])
    assert after_state[2:] == global_state[2:]
    delay = controller.eval_start_delay_steps
    lo, hi = task.AVATAR_START_DELAY_STEPS
    assert lo <= delay <= hi
    assert task._eval_avatar_start_delay_steps == delay
    assert delay == int(np.random.default_rng(np.random.SeedSequence([
        task._episode_seed,
        zlib.crc32(
            b"interaction:block_ranking.assist.v1:eval_start_delay"
        ) & 0xFFFFFFFF,
    ])).integers(lo, hi + 1))
    assert len(avatar.calls) == 1
    assert len(avatar.motion.frames) == 3 + delay
    assert avatar.motion.frames[: delay + 1] == [avatar.hold_frame] * (delay + 1)
    assert avatar.motion.attach_frame == 17 + delay
    assert avatar.motion.detach_frame == 31 + delay
    assert task._eval_assist_trigger_step == 0
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is False
    assert task._eval_policy_step_count == 0

    returned_observation = object()
    task.eval_success = True
    task.get_obs = lambda: returned_observation
    result = task.take_action(np.zeros(8), action_type="qpos")

    assert result is returned_observation
    assert task._interaction_policy_step == 1
    assert task._eval_policy_step_count == 1
    assert len(avatar.calls) == 1


def test_assist_metrics_preserve_the_exact_legacy_public_surface():
    rgb = RankBlocksByColorContribution({})
    rgb_controller, _rgb_plan, _tail = _actual_episode(rgb, seed=42)
    assert rgb_controller.interaction_metrics(rgb) == {}

    size = RankBlocksBySizeContribution({})
    size_controller, _size_plan, _tail = _actual_episode(size, seed=42)
    assert size_controller.interaction_metrics(size) == {
        "assist_avatar_size": size._avatar_size,
        "assist_robot_sizes": list(size._robot_sizes),
        "assist_avatar_start_delay_steps": size._avatar_start_delay_steps,
        "assist_avatar_spawn_y_band": list(size.AVATAR_SPAWN_Y),
        "assist_robot_spawn_y_band": list(size.ROBOT_SPAWN_Y),
        "eval_mode": False,
        "eval_assist_trigger_step": -1,
        "eval_assist_fired": False,
        "eval_policy_step_count": 0,
    }

    eval_rgb = RankBlocksByColorContribution({"eval_mode": True})
    eval_rgb_controller, _plan = _prepare_motion_task(eval_rgb, seed=42)
    eval_rgb_controller.after_base_reset_observation(eval_rgb, observation=None)
    assert eval_rgb_controller.interaction_metrics(eval_rgb) == {}

    eval_size = RankBlocksBySizeContribution({"eval_mode": True})
    eval_size_controller, _plan = _prepare_motion_task(eval_size, seed=42)
    eval_size_controller.after_base_reset_observation(eval_size, observation=None)
    assert eval_size_controller.interaction_metrics(eval_size) == {
        "assist_avatar_size": eval_size._avatar_size,
        "assist_robot_sizes": list(eval_size._robot_sizes),
        "assist_avatar_start_delay_steps": eval_size._avatar_start_delay_steps,
        "assist_avatar_spawn_y_band": list(eval_size.AVATAR_SPAWN_Y),
        "assist_robot_spawn_y_band": list(eval_size.ROBOT_SPAWN_Y),
        "eval_mode": True,
        "eval_assist_trigger_step": 0,
        "eval_assist_fired": True,
        "eval_policy_step_count": 0,
    }


@pytest.mark.parametrize(
    "task_cls,noun",
    (
        (RankBlocksByColorContribution, "color"),
        (RankBlocksBySizeContribution, "size"),
    ),
)
def test_scene_init_v1_work_allocation_is_adopted_by_controller(task_cls, noun):
    task = task_cls({})
    controller, _plan, _tail = _actual_episode(task, seed=7)
    restored_owner = next(
        key for key in task.block_keys() if key != controller.avatar_key
    )
    restored_robot = tuple(reversed(tuple(
        key for key in task.block_keys() if key != restored_owner
    )))
    setattr(task, f"_avatar_{noun}", restored_owner)
    setattr(task, f"_robot_{noun}s", list(restored_robot))
    restored_delay = task.AVATAR_START_DELAY_STEPS[1] + 37
    task._avatar_start_delay_steps = restored_delay

    controller.on_base_reset_ready(task)

    assert controller.avatar_key == restored_owner
    assert controller.robot_keys == restored_robot
    assert controller.avatar_start_delay_steps == restored_delay
    assert getattr(task, f"_avatar_{noun}") == restored_owner
    assert getattr(task, f"_robot_{noun}s") == restored_robot



@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
def test_controller_state_round_trip_is_complete_and_strict(task_cls):
    task = task_cls({})
    controller, _plan, _tail = _actual_episode(task, seed=42)
    controller.eval_assist_trigger_step = 9
    controller.eval_assist_fired = True
    controller.eval_assist_failed = True
    controller.policy_step_count = 12
    state = controller.state_dict()

    restored_task = task_cls({})
    restored = restored_task.interaction_controller
    restored.restore_task_state(restored_task, state)
    assert restored.state_dict() == state
    assert restored_task._interaction_policy_step == 12
    assert restored_task._eval_policy_step_count == 12
    noun = restored_task.BLOCK_KEY_KIND
    assert getattr(restored_task, f"_avatar_{noun}") == state["avatar_key"]
    assert tuple(getattr(restored_task, f"_robot_{noun}s")) == tuple(
        state["robot_keys"]
    )

    malformed = dict(state)
    malformed.pop("avatar_key")
    with pytest.raises(ValueError, match="missing"):
        restored.load_state_dict(malformed)
    future = dict(state, schema_version="future.v2")
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict(future)
    wrong = dict(state, robot_keys=[state["avatar_key"]])
    with pytest.raises(ValueError, match="complement"):
        restored.load_state_dict(wrong)


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
def test_controller_does_not_retain_owning_task(task_cls):
    task = task_cls({})
    controller = task.interaction_controller
    task_ref = weakref.ref(task)

    del task
    gc.collect()

    assert task_ref() is None
    assert controller.keys


@pytest.mark.parametrize(
    "task_name,task_cls,parent",
    (
        ("rank_blocks_by_color_contribution", RankBlocksByColorContribution, BlocksRankingRGB.play_once),
        ("rank_blocks_by_size_contribution", RankBlocksBySizeContribution, BlocksRankingSize.play_once),
    ),
)
def test_blind_expert_routes_to_play_blind_once(task_name, task_cls, parent):
    # The blind expert commits to the robot's share instead of
    # falling back to the robot-only parent that places all three blocks.
    assert parent is not task_cls.play_blind_once
    mode, selected = policy_play_mode(task_name, task_cls, "expert")
    assert mode == "play_blind_once"
    assert selected is None
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        "play_once",
        None,
    )


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
@pytest.mark.parametrize("seed", SEEDS)
def test_eval_start_delay_is_seeded_and_spans_the_demo_window(task_cls, seed):
    task = task_cls({"eval_mode": True})
    controller = task.interaction_controller
    lo, hi = task.AVATAR_START_DELAY_STEPS
    draws = []
    for s in range(400):
        task._episode_seed = s
        task._episode_rngs = {}
        draws.append(controller.draw_eval_start_delay(task))
    assert min(draws) >= lo and max(draws) <= hi
    # Covers the window (not a constant) and is roughly uniform.
    assert max(draws) - min(draws) > 0.8 * (hi - lo)
    assert abs(float(np.mean(draws)) - 0.5 * (lo + hi)) < 0.1 * (hi - lo)
    task._episode_seed = seed
    task._episode_rngs = {}
    first = controller.draw_eval_start_delay(task)
    task._episode_rngs = {}
    assert controller.draw_eval_start_delay(task) == first


class _FakeArm:
    def __init__(self, tcp):
        self._tcp = np.asarray(tcp, dtype=np.float64)

    def get_ee_pose(self):
        return np.concatenate([self._tcp, [1.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "task_cls", (RankBlocksByColorContribution, RankBlocksBySizeContribution)
)
def test_blind_share_is_the_nearest_blocks_to_the_gripper_in_family_order(task_cls):
    task = task_cls({"eval_mode": True})
    controller, _plan = _prepare_motion_task(task)
    keys = tuple(task.block_keys())
    xy = {keys[0]: (0.20, -0.30), keys[1]: (0.00, 0.05), keys[2]: (-0.05, -0.25)}
    task.blocks = {k: _FakeBlock([x, y, 0.76]) for k, (x, y) in xy.items()}
    task.block_center = lambda key: task.blocks[key].get_pos()
    task.robot = SimpleNamespace(get_arm=lambda tag: _FakeArm([0.0, -0.19, 0.84]))

    chosen = controller.choose_blind_robot_keys(task)

    # Distances to (0, -0.19): k0 0.226, k1 0.240, k2 0.078 -> k2 and k0.
    assert chosen == (keys[0], keys[2])
    assert task._blind_robot_keys == chosen
    # Independent of the human: the same layout gives the same share
    # whichever block the human owns.
    for owner in keys:
        controller.avatar_key = owner
        assert controller.choose_blind_robot_keys(task) == chosen
        metrics = controller.interaction_metrics(task)
        assert metrics["blind_robot_keys"] == list(chosen)
        assert metrics["blind_contests_human_block"] is (owner in chosen)


def test_blind_rollout_places_only_the_committed_share_then_waits():
    task = RankBlocksByColorContribution({"eval_mode": True})
    controller, _plan = _prepare_motion_task(task)
    task.TABLE_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 1
    events = []
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))

    @contextmanager
    def suppress_recording():
        events.append(("suppress",))
        yield

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("step",))
    controller.choose_blind_robot_keys = lambda t: ("red", "blue")
    task._place_block_with_recovery = lambda key, arm: events.append(("place", key)) or True
    controller.wait_for_scripted_completion = lambda t: events.append(("wait",))

    assert task.play_blind_once() is True
    assert events == [
        ("boost", "right"), ("suppress",), ("step",), ("step",),
        ("place", "red"), ("place", "blue"), ("wait",), ("step",),
    ]


def test_parent_expert_driver_does_not_change_legacy_size_metric_counter():
    task = RankBlocksBySizeContribution({"eval_mode": True})
    task.avatar = object()
    task._eval_policy_step_count = 0
    task.step_sim = lambda: None

    def rollout():
        for _ in range(5):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(
        task, rollout, {"eval_parent_rollout_tick_interval": 1}
    )
    assert task._interaction_policy_step == 0
    assert task._eval_policy_step_count == 0


def _legacy_base_rgb_episode(task, seed):
    np.random.seed(seed)
    order = tuple(np.random.permutation(list(task.CUBE_COLORS)))
    xys = task._sample_spawn_xys()
    tail = np.random.random_sample(8)
    return order, {key: xys[index] for index, key in enumerate(order)}, tail


def _legacy_base_size_episode(task, seed):
    np.random.seed(seed)
    colors = tuple(np.random.permutation(list(task.CUBE_COLORS)))
    order = tuple(np.random.permutation(list(task.SIZE_ORDER)))
    xys = task._sample_spawn_xys()
    tail = np.random.random_sample(8)
    return order, dict(zip(order, colors)), {
        key: xys[index] for index, key in enumerate(order)
    }, tail


@pytest.mark.parametrize("seed", (0, 8, 42))
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_rgb_base_plan_preserves_rng_and_layout(seed, setting):
    task = BlocksRankingRGB({"setting": setting})
    expected_order, expected_xys, expected_tail = _legacy_base_rgb_episode(
        task, seed
    )
    np.random.seed(seed)
    plan = task._base_block_episode_plan()
    actual_tail = np.random.random_sample(8)

    assert tuple(spec.key for spec in plan.blocks) == expected_order
    for spec in plan.blocks:
        np.testing.assert_array_equal(spec.xy, expected_xys[spec.key])
    np.testing.assert_array_equal(actual_tail, expected_tail)


@pytest.mark.parametrize("seed", (0, 8, 42))
@pytest.mark.parametrize("setting", ("sim", "real"))
def test_size_base_plan_preserves_rng_color_and_layout(seed, setting):
    task = BlocksRankingSize({"setting": setting})
    order, colors, expected_xys, expected_tail = _legacy_base_size_episode(
        task, seed
    )
    np.random.seed(seed)
    plan = task._base_block_episode_plan()
    actual_tail = np.random.random_sample(8)

    assert tuple(spec.key for spec in plan.blocks) == order
    assert {spec.key: spec.color_name for spec in plan.blocks} == colors
    for spec in plan.blocks:
        np.testing.assert_array_equal(spec.xy, expected_xys[spec.key])
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_scripted_assist_order_matches_legacy_choreography():
    task = RankBlocksByColorContribution({})
    controller, _plan, _tail = _actual_episode(task, seed=7)
    task.avatar = object()
    task.TABLE_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 2
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
        events.append(("pick", key)), True
    )[1]
    controller.scripted_robot_keys = lambda owner: (
        events.append(("avatar", "start")), controller.robot_keys
    )[1]
    controller.wait_for_scripted_completion = lambda owner: events.append(
        ("avatar", "wait")
    )

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        ("avatar", "start"),
        *[("pick", key) for key in controller.robot_keys],
        ("avatar", "wait"),
        ("sim", None),
        ("sim", None),
    ]


def test_full_state_assist_yields_until_the_human_finished():
    task = RankBlocksByColorContribution({})
    controller, _plan, _tail = _actual_episode(task, seed=7)
    busy = [True, True, True]
    task.avatar = SimpleNamespace(spare=lambda: not busy)
    task.TABLE_SETTLE_STEPS = 0
    task.FINAL_SETTLE_STEPS = 0
    events = []
    task._boost_finger_pd = lambda arm: None
    task.step_sim = lambda: (busy.pop() if busy else None, events.append("sim"))
    task._pick_and_place_block = lambda key, arm: (events.append(("pick", key)), True)[1]
    controller.wait_for_scripted_completion = lambda owner: events.append("wait")
    controller.scripted_robot_keys = lambda owner: controller.robot_keys

    @contextmanager
    def suppress_recording():
        yield

    task.suppress_recording = suppress_recording
    assert task.play_once() is True
    first_pick = next(i for i, e in enumerate(events) if isinstance(e, tuple))
    assert events[:first_pick] == ["sim"] * 3
    task.ASSIST_YIELD_TO_HUMAN = False
    busy[:] = [True]
    events.clear()
    assert task.play_once() is True
    assert isinstance(events[0], tuple)


def test_direct_no_avatar_assist_preserves_double_boost_and_settle():
    task = RankBlocksByColorContribution({})
    _controller, _plan, _tail = _actual_episode(task, seed=7)
    task.avatar = None
    task.TABLE_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 1
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
        events.append(("pick", key)), True
    )[1]

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
        ("pick", "red"),
        ("pick", "green"),
        ("pick", "blue"),
        ("sim", None),
    ]


def test_spawn_plan_preserves_rgb_task_attrs_and_size_color_mapping(monkeypatch):
    from envs.task_bases import blocks_ranking_rgb as core_module

    monkeypatch.setattr(
        core_module,
        "create_primitive",
        lambda *args, **kwargs: object(),
    )

    rgb = BlocksRankingRGB({})
    rgb.scene = object()
    rgb.TABLE_TOP_Z = 0.75
    np.random.seed(7)
    rgb_plan = rgb._base_block_episode_plan()
    rgb._spawn_block_episode(rgb_plan)
    assert not hasattr(rgb, "block_colors")

    size = BlocksRankingSize({})
    size.scene = object()
    size.TABLE_TOP_Z = 0.75
    np.random.seed(7)
    size_plan = size._base_block_episode_plan()
    size._spawn_block_episode(size_plan)
    assert size.block_colors == {
        spec.key: spec.color_name for spec in size_plan.blocks
    }
