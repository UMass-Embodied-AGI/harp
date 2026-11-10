"""Public, scene-free interaction contracts for dump-bin and bread tasks."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from envs.avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from envs.interaction import (
    DumpBinInterruptController,
    InteractionMode,
    PlaceBreadInBasketInterruptController,
)
from envs.task_bases.dump_bin import DumpBin
from envs.task_bases.place_bread_in_basket import PlaceBreadInBasket
from envs.tasks import (
    NO_HUMAN_TASK_MAP,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
from envs.tasks.dump_bin_contribution import DumpBinContribution
from envs.tasks.dump_bin_interruption import DumpBinInterruption
from envs.tasks._inspect_motion_interrupt_facade import (
    InspectMotionInterruptFacade,
)
from envs.tasks.place_bread_in_basket_contribution import PlaceBreadInBasketContribution
from envs.tasks.place_bread_in_basket_interruption import PlaceBreadInBasketInterruption
from scripts.eval_expert_set import (
    configure_parent_rollout_eval,
    policy_play_mode,
    run_with_eval_step_driver,
)


PUBLIC_TASKS = (
    (
        "dump_bin",
        "envs.task_bases.dump_bin",
        "DumpBin",
        DumpBin,
        "dump_bin",
        TaskVariant.BASE,
        True,
    ),
    (
        "dump_bin_contribution",
        "envs.tasks.dump_bin_contribution",
        "DumpBinContribution",
        DumpBinContribution,
        "dump_bin",
        TaskVariant.ASSIST,
        False,
    ),
    (
        "dump_bin_interruption",
        "envs.tasks.dump_bin_interruption",
        "DumpBinInterruption",
        DumpBinInterruption,
        "dump_bin",
        TaskVariant.INTERRUPT,
        False,
    ),
    (
        "place_bread_in_basket",
        "envs.task_bases.place_bread_in_basket",
        "PlaceBreadInBasket",
        PlaceBreadInBasket,
        "place_bread_in_basket",
        TaskVariant.BASE,
        True,
    ),
    (
        "place_bread_in_basket_contribution",
        "envs.tasks.place_bread_in_basket_contribution",
        "PlaceBreadInBasketContribution",
        PlaceBreadInBasketContribution,
        "place_bread_in_basket",
        TaskVariant.ASSIST,
        False,
    ),
    (
        "place_bread_in_basket_interruption",
        "envs.tasks.place_bread_in_basket_interruption",
        "PlaceBreadInBasketInterruption",
        PlaceBreadInBasketInterruption,
        "place_bread_in_basket",
        TaskVariant.INTERRUPT,
        False,
    ),
)


@pytest.mark.parametrize(
    "name,module_name,class_name,task_cls,family,variant,no_human_canonical",
    PUBLIC_TASKS,
)
def test_public_class_paths_signatures_and_registry_metadata(
    name,
    module_name,
    class_name,
    task_cls,
    family,
    variant,
    no_human_canonical,
):
    module = importlib.import_module(module_name)
    assert getattr(module, class_name) is task_cls
    assert task_cls.__module__ == module_name
    assert task_cls.__name__ == class_name

    constructor = inspect.signature(task_cls.__init__)
    assert tuple(constructor.parameters) == ("self", "config")
    assert constructor.parameters["config"].default is None
    for method_name, parameters in (
        ("reset", ("self", "seed")),
        ("take_action", ("self", "action", "action_type")),
        ("play_once", ("self",)),
        ("check_success", ("self",)),
        ("evaluate", ("self",)),
    ):
        signature = inspect.signature(getattr(task_cls, method_name))
        assert tuple(signature.parameters) == parameters
    assert inspect.signature(task_cls.reset).parameters["seed"].default == 0
    assert (
        inspect.signature(task_cls.take_action).parameters["action_type"].default
        == "qpos"
    )

    descriptor = TASK_DESCRIPTOR_MAP[name]
    assert descriptor.task_cls is task_cls
    assert descriptor.family == family
    assert descriptor.variant is variant
    assert descriptor.registration is TaskRegistration.CANONICAL
    assert descriptor.no_human_name == family
    assert descriptor.no_human_canonical is no_human_canonical
    assert descriptor.instruction == task_cls.default_instruction()
    assert TASK_MAP[name] is task_cls

    no_human_cls = DumpBin if family == "dump_bin" else PlaceBreadInBasket
    assert NO_HUMAN_TASK_MAP[name] is no_human_cls
    assert resolve_task_class(name) == (name, task_cls)
    assert resolve_task_class(name, no_human=True) == (family, no_human_cls)


def test_constructor_defaults_and_input_copy_contracts():
    dump_input = {"avatar": {"frame_ratio": 3}}
    dump_assist = DumpBinContribution(dump_input)
    assert dump_input == {"avatar": {"frame_ratio": 3}}
    assert dump_assist.config["robot_type"] == "franka"
    assert dump_assist.config["robot_single_arm"] is True
    assert dump_assist.config["avatar"] == {
        "frame_ratio": 3,
        "generated_motion_data": "avatars/motions/generated_motions.pkl",
    }
    assert dump_assist.num_objects == 1
    assert dump_assist.use_avatar is True

    dump_interrupt = DumpBinInterruption({"hard": True})
    assert dump_interrupt.num_objects == 3
    assert dump_interrupt.use_avatar is True

    bread = PlaceBreadInBasket({"hard": True})
    assert bread.config["robot_type"] == "franka"
    assert bread._target_entry.key == "075_bread"
    assert bread._num_breads == 2

    bread_input = {"robot_kwargs": {"custom": 1}}
    bread_assist = PlaceBreadInBasketContribution(bread_input)
    assert bread_input == {"robot_kwargs": {"custom": 1}}
    assert bread_assist.config["track_avatar_collision"] is True
    assert bread_assist.config["success_require_no_avatar_collision"] is True
    assert bread_assist._num_breads == 2
    assert bread_assist.use_avatar is True

    bread_interrupt = PlaceBreadInBasketInterruption({"track_avatar_collision": False})
    assert bread_interrupt.config["track_avatar_collision"] is False
    assert bread_interrupt.use_avatar is True


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_assist_controller_and_task_hook_compatibility_surface(task_cls):
    task = task_cls({})
    controller = task.interaction_controller
    assert getattr(controller.mode, "value", controller.mode) == InteractionMode.ASSIST.value
    assert controller.handles_policy_ticks is True
    for name in ("_eval_at_reset", "_eval_at_step"):
        assert callable(getattr(task, name))
    assert tuple(inspect.signature(task._eval_at_step).parameters) == ("step_idx",)


@pytest.mark.parametrize(
    "task_cls",
    (DumpBinInterruption, PlaceBreadInBasketInterruption),
)
def test_interrupt_inspect_and_eval_compatibility_surface(task_cls):
    task = task_cls({})
    assert task.INSPECT_MOTION_NAME == "Inspect1"
    assert task.INSPECT_ATTACH_FRAME == 34
    assert task.INSPECT_DETACH_FRAME == 187
    assert task.AVATAR_MOTION_SLOW == 10.0
    assert task.AVATAR_HAND_ID == 1
    assert task._attach_frame_scaled == 340
    assert task._detach_frame_scaled == 1870
    for name in (
        "_init_eval_interrupt_state",
        "_eval_step_avatar",
        "_play_inspect_motion",
        "_wait_until_avatar_attach",
    ):
        assert callable(getattr(task, name))
    if task_cls is DumpBinInterruption:
        # The hooks remain as inert shims for scripts/eval.py; the
        # env never skips the policy or requests a policy reset.
        assert callable(task.eval_skip_policy_action)
        assert callable(task.consume_eval_policy_reset_requested)
        task._eval_reset_policy_after_pace = True
        assert task.consume_eval_policy_reset_requested() is False
        assert task.eval_skip_policy_action() is False


@pytest.mark.parametrize(
    "task_cls,family_core,controller_cls",
    (
        (DumpBinInterruption, DumpBin, DumpBinInterruptController),
        (
            PlaceBreadInBasketInterruption,
            PlaceBreadInBasket,
            PlaceBreadInBasketInterruptController,
        ),
    ),
)
def test_interrupt_composition_excludes_inspect_mixin_and_uses_family_controller(
    task_cls, family_core, controller_cls
):
    assert issubclass(task_cls, family_core)
    assert InspectMotionMixin not in task_cls.__mro__
    assert InspectMotionInterruptFacade in task_cls.__mro__

    task = task_cls({})
    controller = task.interaction_controller
    assert type(controller) is controller_cls
    assert controller.mode is InteractionMode.INTERRUPT
    assert controller.handles_policy_ticks is False
    assert controller.handles_scripted_rollout_ticks is False


def test_inspect_facade_does_not_own_lifecycle_or_expert_routing_methods():
    # The facade must not participate in lifecycle or expert routing. Those
    # methods remain owned by each family task/core.
    for name in ("reset", "take_action", "play_once", "play_blind_once"):
        assert name not in InspectMotionInterruptFacade.__dict__


INSPECT_COMPATIBILITY_API = (
    "_resolve_inspect_motion",
    "_prepare_inspect_motion_entry",
    "inspect_avatar_pos_for_object",
    "reset_avatar_for_inspect_object",
    "_inspect_idle_position_for_motion_root",
    "_set_inspect_motion_root_pos",
    "_play_inspect_motion",
)


@pytest.mark.parametrize(
    "task_cls",
    (
        DumpBinInterruption,
        PlaceBreadInBasketInterruption,
    ),
)
def test_composed_interrupts_retain_inspect_facade_properties_and_callables(task_cls):
    task = task_cls({})
    for name in INSPECT_COMPATIBILITY_API:
        assert callable(getattr(task, name)), name

    assert task._inspect_attach_frame_scaled == 340
    assert task._inspect_detach_frame_scaled == 1870
    assert task._attach_frame_scaled == task._inspect_attach_frame_scaled
    assert task._detach_frame_scaled == task._inspect_detach_frame_scaled

    # These are live compatibility properties, not values cached at class
    # creation or controller construction.
    task.INSPECT_ATTACH_FRAME = 31
    task.INSPECT_DETACH_FRAME = 84
    task.AVATAR_MOTION_SLOW = 2.5
    assert task._inspect_attach_frame_scaled == 78
    assert task._inspect_detach_frame_scaled == 210
    assert task._attach_frame_scaled == 78
    assert task._detach_frame_scaled == 210


@pytest.mark.parametrize(
    "task_cls",
    (
        DumpBinInterruption,
        PlaceBreadInBasketInterruption,
    ),
)
def test_inspect_facade_resolver_preserves_forced_alias_and_selected_metadata(task_cls):
    task = task_cls({"inspect_motion_name": "Inspect"})
    task._resolve_inspect_motion()

    assert task.INSPECT_MOTION_NAME == "Inspect1"
    assert task.INSPECT_ATTACH_FRAME == 34
    assert task.INSPECT_DETACH_FRAME == 187
    assert task.AVATAR_HAND_ID == 1
    assert task.INSPECT_ATTACH_OBJECT is True


class _FakeInspectPlaybackAvatar:
    def __init__(self):
        self.calls = []
        self.resets = []

    def reset(self, pos, rot):
        self.resets.append((np.asarray(pos).copy(), np.asarray(rot).copy()))

    def play_animation(self, motion_name, **kwargs):
        self.calls.append((motion_name, kwargs))
        return "played"


def test_inspect_facade_playback_preserves_root_and_touch_only_compatibility():
    task = DumpBinInterruption({})
    task.avatar = _FakeInspectPlaybackAvatar()
    task._inspect_motion_root_pos = np.array([0.1, 0.2, 0.3])
    attached = object()

    assert task._play_inspect_motion(
        attach_obj=attached,
        hand_id=1,
        attach_frame=340,
        detach_frame=1870,
        return_to_idle_after=None,
    ) == "played"
    np.testing.assert_array_equal(
        task.avatar.resets[-1][0], task._inspect_motion_root_pos
    )
    motion_name, kwargs = task.avatar.calls[-1]
    assert motion_name == "Inspect1"
    assert kwargs == {
        "attach_obj": attached,
        "hand_id": 1,
        "attach_frame": 340,
        "detach_frame": 1870,
        "return_to_idle_after": None,
    }

    task.INSPECT_ATTACH_OBJECT = False
    assert task._play_inspect_motion(
        attach_obj=attached,
        hand_id=1,
        attach_frame=340,
        detach_frame=1870,
    ) == "played"
    assert task.avatar.calls[-1][1] == {
        "attach_obj": None,
        "hand_id": 1,
        "attach_frame": None,
        "detach_frame": None,
    }


class _FakeScene:
    def build(self):
        return None

    def step(self):
        return None


class _FakeTable:
    def get_pos(self):
        return np.array([0.0, 0.0, 0.74])


class _FakeResetRobot:
    left_arm = None

    def init_joints(self, _scene):
        return None

    def move_to_homestate(self):
        return None

    def open_gripper(self, _arm):
        return None

    def set_origin_endpose(self):
        return None


class _FakeResetAvatar:
    def __init__(self, events):
        self.events = events
        self.frame_ratio = None
        self.reset_positions = []
        self.motion_data = {}

    def reset(self, pos, _rot, **_kwargs):
        copied = np.asarray(pos, dtype=np.float64).copy()
        self.reset_positions.append(copied)
        self.events.append(("avatar_reset", copied))


def _stub_reset_dependencies(task, monkeypatch, events, *, with_avatar=True):
    from envs import base_task as base_module

    avatar = _FakeResetAvatar(events) if with_avatar else None
    observations = []

    def setup_scene():
        task.scene = _FakeScene()

    def create_table():
        task.table = _FakeTable()
        task.TABLE_THICKNESS = 0.05
        task.TABLE_TOP_Z = 0.765
        task._table_top_z_fixed = False

    task._setup_scene = setup_scene
    task._create_table = create_table
    task._load_robot = lambda: setattr(task, "robot", _FakeResetRobot())
    task._load_cameras = lambda: setattr(task, "cameras", None)
    task.load_actors = lambda: events.append(("load_actors", None))
    task._spawn_table_random_objects = lambda: None
    task._init_avatar = lambda: setattr(task, "avatar", avatar)
    task._init_avatar_collider = lambda: None
    task._init_avatar_collision_checker = lambda: None
    task._setup_batch_render_cache = lambda: None
    task._apply_policy_gripper_gains = lambda: None
    task._settle_scene = lambda _steps: events.append(("settle", None))

    def get_obs():
        observation = object()
        observations.append(observation)
        events.append(("initial_observation", len(observations)))
        return observation

    task.get_obs = get_obs
    monkeypatch.setattr(base_module, "apply_scene_init_pre_avatar", lambda _task: None)
    monkeypatch.setattr(
        base_module,
        "apply_scene_init_post_settle",
        lambda _task: {"enabled": False},
    )
    return avatar, observations


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_assist_eval_reset_hook_runs_after_initial_observation(
    task_cls, monkeypatch
):
    task = task_cls({"eval_mode": True})
    events = []
    _avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)

    def eval_reset_hook():
        events.append(("assist_eval_reset", None))
        task._eval_assist_trigger_step = 0
        task._eval_assist_fired = True
        task._eval_assist_failed = False

    task._eval_at_reset = eval_reset_hook
    returned = task.reset(seed=7)

    assert returned is observations[0]
    labels = [event[0] for event in events]
    assert labels.index("initial_observation") < labels.index("assist_eval_reset")
    assert task._eval_policy_step_count == 0
    assert task._eval_assist_trigger_step == 0
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is False


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_assist_eval_reset_is_inactive_without_avatar(task_cls, monkeypatch):
    task = task_cls({"eval_mode": True})
    events = []
    _avatar, observations = _stub_reset_dependencies(
        task, monkeypatch, events, with_avatar=False
    )
    task._eval_policy_step_count = 13
    task._eval_at_reset = lambda: events.append(("unexpected_eval_reset", None))

    returned = task.reset(seed=7)

    assert returned is observations[0]
    assert ("unexpected_eval_reset", None) not in events
    assert task._eval_policy_step_count == 13


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_assist_active_policy_counter_ticks_before_base_success_short_circuit(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    task._eval_policy_step_count = 0
    task._interaction_policy_step = 0
    events = []
    observation = object()
    task._eval_at_step = lambda step: events.append(("eval_tick", step))
    task.get_obs = lambda: (events.append(("observation", None)), observation)[1]

    returned = task.take_action(np.zeros(8), action_type="qpos")

    assert returned is observation
    assert events == [("eval_tick", 1), ("observation", None)]
    assert task._eval_policy_step_count == 1
    assert task._interaction_policy_step == 1


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_assist_inactive_policy_call_does_not_increment_compatibility_counter(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = None
    task.eval_success = True
    task._eval_policy_step_count = 4
    task._eval_at_step = lambda _step: pytest.fail("inactive hook ran")
    observation = object()
    task.get_obs = lambda: observation

    assert task.take_action(np.zeros(8), action_type="qpos") is observation
    assert task._eval_policy_step_count == 4


@pytest.mark.parametrize(
    "task_cls",
    (
        DumpBinInterruption,
        PlaceBreadInBasketInterruption,
    ),
)
def test_interrupt_eval_reset_initializes_after_initial_observation(
    task_cls, monkeypatch
):
    task = task_cls({"eval_mode": True})
    events = []
    avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)

    def initialize():
        events.append(("interrupt_eval_init", None))
        task._policy_step_count = 0
        task._eval_interrupt_fired = False

    # The controller must dispatch through the task-facing compatibility hook,
    # so scene-init/custom subclasses can still override motion resolution.
    task._resolve_inspect_motion = lambda: events.append(
        ("resolve_inspect", None)
    )
    task._init_eval_interrupt_state = initialize
    returned = task.reset(seed=9)

    assert returned is observations[0]
    assert len(observations) == 1
    labels = [event[0] for event in events]
    assert (
        labels.index("initial_observation")
        < labels.index("resolve_inspect")
        < labels.index("interrupt_eval_init")
    )
    assert avatar.frame_ratio == task.AVATAR_MOTION_SLOW
    assert task._policy_step_count == 0
    assert task._eval_interrupt_fired is False


@pytest.mark.parametrize(
    "task_cls",
    (
        DumpBinInterruption,
        PlaceBreadInBasketInterruption,
    ),
)
def test_interrupt_scripted_reset_resolves_after_observation_before_calibration(
    task_cls, monkeypatch
):
    task = task_cls({"eval_mode": False})
    events = []
    avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)
    task._resolve_inspect_motion = lambda: events.append(
        ("resolve_inspect", None)
    )
    task._calibrate_avatar_for_inspect = lambda: events.append(
        ("scripted_calibration", None)
    )

    returned = task.reset(seed=11)

    assert returned is observations[0]
    assert len(observations) == 1
    labels = [event[0] for event in events]
    assert (
        labels.index("initial_observation")
        < labels.index("resolve_inspect")
        < labels.index("scripted_calibration")
    )
    assert avatar.frame_ratio == task.AVATAR_MOTION_SLOW


@pytest.mark.parametrize(
    "task_cls",
    (
        DumpBinInterruption,
        PlaceBreadInBasketInterruption,
    ),
)
@pytest.mark.parametrize("seed", (0, 42))
def test_randomized_inspect_reset_preserves_draw_position_rng_tail_and_task_state(
    task_cls, seed, monkeypatch
):
    task = task_cls({"eval_mode": True, "randomize_inspect": True})
    events = []
    _avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)
    observed_layout_draws = []

    def load_actors():
        observed_layout_draws.extend(np.random.random_sample(4))
        # These are release/scene-init-facing task projections; resolving the
        # inspect motion must not replace or relocate their ownership.
        task._interrupted_idx = 2
        task._interrupt_pick_ordinal = 1
        events.append(("load_actors", None))

    task.load_actors = load_actors
    task._init_eval_interrupt_state = lambda: events.append(
        ("interrupt_eval_init", None)
    )

    np.random.seed(seed)
    expected_layout_draws = np.random.random_sample(4)
    expected_entry = STANDARD_INSPECT_POOL[
        int(np.random.randint(len(STANDARD_INSPECT_POOL)))
    ]
    expected_tail = np.random.random_sample(8)

    returned = task.reset(seed=seed)
    actual_tail = np.random.random_sample(8)

    assert returned is observations[0]
    np.testing.assert_array_equal(observed_layout_draws, expected_layout_draws)
    assert task.INSPECT_MOTION_NAME == expected_entry["name"]
    assert task.INSPECT_ATTACH_FRAME == expected_entry["attach"]
    assert task.INSPECT_DETACH_FRAME == expected_entry["detach"]
    np.testing.assert_array_equal(actual_tail, expected_tail)
    assert task._interrupted_idx == 2
    assert task._interrupt_pick_ordinal == 1


def test_dump_interrupt_hidden_avatar_refreshes_returned_observation(monkeypatch):
    task = DumpBinInterruption({
        "eval_mode": True,
        "eval_hide_avatar_until_trigger": True,
    })
    events = []
    avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)
    task._resolve_inspect_motion = lambda: events.append(
        ("resolve_inspect", None)
    )
    task._init_eval_interrupt_state = lambda: events.append(
        ("interrupt_eval_init", None)
    )

    returned = task.reset(seed=3)

    assert len(observations) == 2
    assert returned is observations[1]
    labels = [event[0] for event in events]
    first_observation = labels.index("initial_observation")
    second_observation = labels.index("initial_observation", first_observation + 1)
    assert (
        first_observation
        < labels.index("resolve_inspect")
        < labels.index("interrupt_eval_init")
        < second_observation
    )
    np.testing.assert_array_equal(
        avatar.reset_positions[-1], task.EVAL_HIDDEN_AVATAR_POS
    )


def test_dump_interrupt_hidden_avatar_skip_reset_obs_does_not_capture_again(
    monkeypatch,
):
    task = DumpBinInterruption({
        "eval_mode": True,
        "eval_hide_avatar_until_trigger": True,
        "skip_reset_obs": True,
    })
    events = []
    avatar, observations = _stub_reset_dependencies(task, monkeypatch, events)
    task._resolve_inspect_motion = lambda: events.append(
        ("resolve_inspect", None)
    )
    task._init_eval_interrupt_state = lambda: events.append(
        ("interrupt_eval_init", None)
    )

    assert task.reset(seed=3) is None
    assert observations == []
    labels = [event[0] for event in events]
    assert labels.index("resolve_inspect") < labels.index("interrupt_eval_init")
    np.testing.assert_array_equal(
        avatar.reset_positions[-1], task.EVAL_HIDDEN_AVATAR_POS
    )


@pytest.mark.parametrize(
    "task_cls",
    (DumpBinInterruption, PlaceBreadInBasketInterruption),
)
def test_interrupt_policy_hook_runs_once_before_base_success_short_circuit(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    events = []
    observation = object()
    task._eval_step_avatar = lambda: events.append(("interrupt_tick", None))
    if task_cls is DumpBinInterruption:
        task._apply_eval_pace_override = lambda: False
    task.get_obs = lambda: (events.append(("observation", None)), observation)[1]

    assert task.take_action(np.zeros(8), action_type="qpos") is observation
    assert events == [("interrupt_tick", None), ("observation", None)]


def test_dump_interrupt_never_replaces_the_policy_action(monkeypatch):
    # Replaces the yield-substitution contract; the policy action
    # reaches the robot unmodified while the interrupt ticks first.
    task = DumpBinInterruption({"eval_mode": True})
    task.avatar = object()
    ticks = []
    task._eval_step_avatar = lambda: ticks.append(1)
    task._eval_override_last_cmd = np.arange(7, dtype=np.float64) + 0.5
    captured = {}

    def base_action(_self, action, action_type="qpos"):
        captured["action"] = np.asarray(action).copy()
        captured["action_type"] = action_type
        return "observation"

    monkeypatch.setattr(DumpBin, "take_action", base_action)

    assert task.take_action(np.ones(8), action_type="qpos") == "observation"
    np.testing.assert_array_equal(captured["action"], np.ones(8))
    assert captured["action_type"] == "qpos"
    assert ticks == [1]


@pytest.mark.parametrize("pace_no_human", (False, True))
def test_dump_interrupt_avatar_none_never_ticks(pace_no_human, monkeypatch):
    # The pace_no_human_ablation path was removed, so without an
    # avatar the interrupt never ticks.
    task = DumpBinInterruption({
        "eval_mode": True,
        "pace_no_human_ablation": pace_no_human,
    })
    task.avatar = None
    ticks = []
    task._eval_step_avatar = lambda: ticks.append(1)
    monkeypatch.setattr(
        DumpBin,
        "take_action",
        lambda _self, _action, action_type="qpos": action_type,
    )

    assert task.take_action(np.zeros(8), action_type="qpos") == "qpos"
    assert ticks == []


@pytest.mark.parametrize(
    "task_name,task_cls,expected_mode,expected_parent",
    (
        ("dump_bin_contribution", DumpBinContribution, "play_blind_once", None),
        ("dump_bin_interruption", DumpBinInterruption, "play_blind_once", None),
        (
            "place_bread_in_basket_contribution",
            PlaceBreadInBasketContribution,
            "play_blind_once",
            None,
        ),
        (
            "place_bread_in_basket_interruption",
            PlaceBreadInBasketInterruption,
            "parent_play_once",
            PlaceBreadInBasket.play_once,
        ),
    ),
)
def test_expert_policy_routing_preserves_blind_vs_full_state_choreography(
    task_name, task_cls, expected_mode, expected_parent
):
    assert policy_play_mode(task_name, task_cls, "expert") == (
        expected_mode,
        expected_parent,
    )
    # Interrupt tasks run the full-state expert in eval mode under
    # the interrupt contract (yield on the fired flag); other tasks keep the
    # scripted play_once choreography.
    expected_full_state = (
        ("interrupt_full_state", None)
        if task_name.endswith("_interruption")
        else ("play_once", None)
    )
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        expected_full_state
    )


def test_parent_interrupt_eval_configuration_defaults_are_preserved():
    dump_cfg = {}
    configure_parent_rollout_eval(
        dump_cfg, "dump_bin_interruption", "play_blind_once"
    )
    # No harness trigger overrides for dump either.
    assert dump_cfg == {}

    bread_cfg = {}
    configure_parent_rollout_eval(
        bread_cfg, "place_bread_in_basket_interruption", "parent_play_once"
    )
    # The harness no longer overrides interrupt triggers; the
    # progress trigger belongs to the task.
    assert bread_cfg == {}


def test_dump_blind_expert_does_not_self_drive_ticks(monkeypatch):
    # The harness (ScriptedRolloutDriver at action_substeps) drives
    # the eval tick for every interrupt no-intent expert; play_blind_once runs
    # the plain dump script and must not tick a second time.
    task = DumpBinInterruption({"eval_mode": True, "action_substeps": 3})
    task.avatar = object()
    events = []
    original_step = lambda: events.append("sim")
    task.step_sim = original_step
    task._eval_step_avatar = lambda: events.append("tick")

    def rollout(_task):
        for _ in range(7):
            _task.step_sim()
        return True

    monkeypatch.setattr(DumpBin, "play_once", rollout)

    assert task.play_blind_once() is True
    assert events == ["sim"] * 7
    assert task.step_sim is original_step


def _controller_owns_policy_ticks(task) -> bool:
    return bool(getattr(task.interaction_controller, "handles_policy_ticks", False))


def test_parent_driver_ticks_after_policy_cadence_and_restores_step():
    task = PlaceBreadInBasketInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    original_step = lambda: events.append("sim")
    task.step_sim = original_step

    if _controller_owns_policy_ticks(task):
        controller = task.interaction_controller
        controller.handles_scripted_rollout_ticks = True
        controller.on_policy_tick = lambda _task, *, step_index: events.append(
            f"tick{step_index}"
        )
    else:
        task._eval_step_avatar = lambda: events.append(
            f"tick{sum(str(event).startswith('tick') for event in events) + 1}"
        )

    def rollout():
        for _ in range(7):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(
        task, rollout, {"eval_parent_rollout_tick_interval": 3}
    )
    assert events == [
        "sim", "sim", "sim", "tick1", "sim", "sim", "sim", "tick2", "sim"
    ]
    assert task.step_sim is original_step


@pytest.mark.parametrize("task_cls", (DumpBinContribution, PlaceBreadInBasketContribution))
def test_parent_driver_does_not_change_legacy_assist_metric_counter(task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    task._eval_policy_step_count = 0
    original_step = lambda: None
    task.step_sim = original_step

    def rollout():
        for _ in range(7):
            task.step_sim()
        return True

    assert run_with_eval_step_driver(
        task, rollout, {"eval_parent_rollout_tick_interval": 2}
    )
    assert task._eval_policy_step_count == 0
    assert task.step_sim is original_step
