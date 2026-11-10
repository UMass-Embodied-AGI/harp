import math
from types import MappingProxyType

import pytest

from envs.base_task import BaseTask
from envs.interaction import (
    AssistEvalModeController,
    AssistInteractionController,
    InteractionController,
    InteractionMode,
    IntentInteractionController,
    InterruptInteractionController,
    LegacyInteractionController,
    NeutralInteractionController,
    PolicyDirective,
    PolicyDisposition,
)


def test_policy_directive_is_immutable_and_validated():
    directive = PolicyDirective(
        disposition=PolicyDisposition.YIELD,
        reason="human occupies target",
        hold_steps=3,
        metadata={"target": "cube"},
    )

    assert directive.hold_steps == 3
    assert isinstance(directive.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        directive.metadata["target"] = "bowl"
    with pytest.raises(ValueError, match="hold_steps"):
        PolicyDirective(hold_steps=-1)
    with pytest.raises(TypeError, match="integer"):
        PolicyDirective(hold_steps=True)
    with pytest.raises(TypeError, match="integer"):
        PolicyDirective(hold_steps=1.5)
    with pytest.raises(TypeError, match="reason"):
        PolicyDirective(reason=3)
    with pytest.raises(ValueError, match="disposition"):
        PolicyDirective(disposition="invented")

    round_trip = PolicyDirective.from_dict(directive.to_dict())
    assert round_trip == directive
    with pytest.raises(ValueError, match="schema"):
        PolicyDirective.from_dict(
            {"schema_version": "genesis_hr_bench.policy_directive.v2"}
        )
    with pytest.raises(ValueError, match="missing fields"):
        PolicyDirective.from_dict(
            {"schema_version": "genesis_hr_bench.policy_directive.v1"}
        )
    payload = directive.to_dict()
    payload["future_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        PolicyDirective.from_dict(payload)


def test_policy_directive_deep_freezes_json_metadata():
    directive = PolicyDirective(metadata={"nested": {"items": [1, 2]}})
    with pytest.raises(TypeError):
        directive.metadata["nested"]["new"] = True
    with pytest.raises(TypeError):
        directive.metadata["nested"]["items"][0] = 3
    assert directive.to_dict()["metadata"] == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError, match="JSON-compatible"):
        PolicyDirective(metadata={"bad": object()})
    with pytest.raises(TypeError, match="keys"):
        PolicyDirective(metadata={1: "bad"})
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(TypeError, match="finite"):
            PolicyDirective(metadata={"bad": value})


def test_controller_defaults_are_safe_noops():
    controller = InteractionController()
    task = object()

    assert controller.mode is InteractionMode.BASE
    assert controller.before_base_policy_action(
        task, step_index=1, action=[0.0], action_type="qpos"
    ) == PolicyDirective()
    state = controller.state_dict()
    assert state == {"schema_version": "genesis_hr_bench.interaction_state.v1"}
    controller.load_state_dict(state)
    with pytest.raises(ValueError, match="schema"):
        controller.load_state_dict(
            {"schema_version": "genesis_hr_bench.interaction_state.v2"}
        )
    with pytest.raises(ValueError, match="missing fields"):
        controller.load_state_dict({})
    with pytest.raises(ValueError, match="unknown fields"):
        controller.load_state_dict(
            {
                "schema_version": "genesis_hr_bench.interaction_state.v1",
                "future": True,
            }
        )
    with pytest.raises(TypeError, match="mapping"):
        controller.load_state_dict([])


def test_semantic_controller_types_expose_stable_modes():
    assert AssistInteractionController().mode is InteractionMode.ASSIST
    assert InterruptInteractionController().mode is InteractionMode.INTERRUPT
    assert NeutralInteractionController().mode is InteractionMode.NEUTRAL
    assert IntentInteractionController().mode is InteractionMode.INTENT


def test_assist_eval_controller_preserves_legacy_hook_timing_and_active_count():
    events = []

    class Task:
        avatar = object()
        config = {"eval_mode": True}

        def _eval_at_reset(self):
            events.append(("reset", self._eval_policy_step_count))

        def _eval_at_step(self, step_index):
            events.append(("step", step_index))

    task = Task()
    controller = AssistEvalModeController()

    assert controller.mode is InteractionMode.ASSIST
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False
    controller.after_base_reset_observation(task, observation={"initial": True})
    controller.on_policy_tick(task, step_index=1)
    controller.on_policy_tick(task, step_index=2)
    assert events == [("reset", 0), ("step", 1), ("step", 2)]

    # Legacy EvalModeAvatarMixin counted only calls made while both eval mode
    # and the avatar were active, not the absolute BaseTask policy index.
    task.avatar = None
    controller.on_policy_tick(task, step_index=3)
    task.avatar = object()
    controller.on_policy_tick(task, step_index=4)
    assert events[-1] == ("step", 3)


def test_assist_eval_controller_requires_explicit_family_hooks():
    class MissingHooks:
        avatar = object()
        config = {"eval_mode": True}

    controller = AssistEvalModeController()
    with pytest.raises(TypeError, match="_eval_at_reset"):
        controller.after_base_reset_observation(
            MissingHooks(), observation=None
        )


def test_eval_mode_assist_tasks_use_composed_lifecycle_controller():
    from envs.avatar.eval_mode_mixin import EvalModeAvatarMixin
    from envs.tasks.dump_bin_contribution import DumpBinContribution
    from envs.tasks.place_bread_in_basket_contribution import PlaceBreadInBasketContribution
    from envs.tasks.place_burger_and_fries_contribution import PlaceBurgerAndFriesContribution
    from envs.tasks.place_food_in_skillet_contribution import PlaceFoodInSkilletContribution
    from envs.tasks.stack_three_bowls_contribution import StackThreeBowlsContribution

    classes = (
        DumpBinContribution,
        PlaceBreadInBasketContribution,
        PlaceBurgerAndFriesContribution,
        PlaceFoodInSkilletContribution,
        StackThreeBowlsContribution,
    )
    for task_cls in classes:
        assert EvalModeAvatarMixin not in task_cls.__mro__
        task = object.__new__(task_cls)
        controller = task_cls.create_interaction_controller(task)
        assert isinstance(controller, AssistEvalModeController)

    # Keep subclass initialization on the task side of the new controller
    # method boundary; a misplaced method definition can otherwise make these
    # post-super compatibility fields unreachable without a syntax error.
    burger = PlaceBurgerAndFriesContribution({})
    assert burger.avatar_burger is None
    assert burger._avatar_burger_drop_xy is None
    skillet = PlaceFoodInSkilletContribution({})
    assert skillet.pan_actor is None
    assert skillet.food_actors == []
    assert skillet._avatar_food_idx is None


def test_controller_randomness_requires_a_stable_namespace():
    class Task:
        def __init__(self):
            self.streams = []

        def episode_rng(self, stream):
            self.streams.append(stream)
            return object()

    task = Task()
    with pytest.raises(RuntimeError, match="rng_namespace"):
        InteractionController().episode_rng(task, "layout")

    class StableController(InteractionController):
        rng_namespace = "example_family.assist.v1"

    rng = StableController().episode_rng(task, "layout")
    assert rng is not None
    assert task.streams == ["interaction:example_family.assist.v1:layout"]


def test_unmigrated_tasks_use_explicit_legacy_adapter():
    task = BaseTask({})

    assert isinstance(task.interaction_controller, LegacyInteractionController)
    assert task.interaction_controller.mode is InteractionMode.LEGACY
    assert task.last_policy_directive == PolicyDirective()


def test_base_task_lifecycle_contains_controller_seams():
    reset_names = set(BaseTask.reset.__code__.co_names)
    action_names = set(BaseTask.take_action.__code__.co_names)
    sim_names = set(BaseTask.step_sim.__code__.co_names)

    assert {
        "on_episode_seeded",
        "augment_scene_before_build",
        "on_avatar_created_before_build",
        "on_base_reset_ready",
        "after_base_reset_observation",
    } <= reset_names
    assert {
        "before_base_policy_action",
        "after_base_policy_action",
    } <= action_names
    assert "_notify_interaction_sim_step" in sim_names


def test_public_policy_tick_driver_is_opt_in_and_monotonic():
    class RecordingController(InteractionController):
        handles_policy_ticks = True

        def __init__(self):
            self.steps = []

        def on_policy_tick(self, task, *, step_index):
            self.steps.append(step_index)

    task = BaseTask({})
    controller = RecordingController()
    task._interaction_controller = controller

    assert task.advance_interaction_policy_step() is True
    assert task.advance_interaction_policy_step() is True
    assert controller.steps == [1, 2]

    legacy = BaseTask({})
    assert legacy.advance_interaction_policy_step() is False
    assert legacy._interaction_policy_step == 0


def test_policy_tick_occurs_before_take_action_success_short_circuit():
    class RecordingController(InteractionController):
        handles_policy_ticks = True

        def __init__(self):
            self.steps = []

        def on_policy_tick(self, task, *, step_index):
            self.steps.append(step_index)

    task = BaseTask({})
    controller = RecordingController()
    task._interaction_controller = controller
    task.eval_success = True
    expected = {"already": "successful"}
    task.get_obs = lambda: expected

    assert task.take_action([0.0], action_type="qpos") is expected
    assert task.take_action([0.0], action_type="qpos") is expected
    assert controller.steps == [1, 2]
    assert task.take_action_cnt == 0


@pytest.mark.parametrize("skip_observation", (False, True))
def test_after_base_reset_observation_hook_has_exact_timing(skip_observation):
    events = []

    class RecordingController(InteractionController):
        def after_base_reset_observation(self, task, *, observation):
            events.append(("after", observation))

    class FakeScene:
        def build(self):
            return None

    class FakeRobot:
        left_arm = None

        def init_joints(self, scene):
            return None

        def move_to_homestate(self):
            return None

        def open_gripper(self, arm):
            return None

        def set_origin_endpose(self):
            return None

    class MinimalResetTask(BaseTask):
        use_table = False

        def create_interaction_controller(self):
            return RecordingController()

        def _setup_scene(self):
            self.scene = FakeScene()

        def _load_robot(self):
            self.robot = FakeRobot()

        def _load_cameras(self):
            self.cameras = None

        def load_actors(self):
            return None

        def _spawn_table_random_objects(self):
            return None

        def _init_avatar(self):
            self.avatar = None

        def _init_avatar_collider(self):
            return None

        def _setup_batch_render_cache(self):
            return None

        def _init_avatar_collision_checker(self):
            return None

        def _apply_policy_gripper_gains(self):
            return None

        def _settle_scene(self, n_steps=None):
            return None

        def get_obs(self):
            events.append(("get_obs", None))
            return {"initial": True}

    task = MinimalResetTask({"skip_reset_obs": skip_observation})
    observation = task.reset(seed=3)

    if skip_observation:
        assert observation is None
        assert events == [("after", None)]
    else:
        assert observation == {"initial": True}
        assert events == [
            ("get_obs", None),
            ("after", {"initial": True}),
        ]


def test_scene_init_capture_excludes_interaction_runtime_counters():
    from envs.scene_init import _capture_task_attrs

    task = BaseTask({})
    task._interaction_sim_step = 17
    task._interaction_policy_step = 9

    attrs = _capture_task_attrs(task)

    assert not any(name.startswith("_interaction") for name in attrs)


def test_sim_step_notification_is_monotonic_and_override_safe():
    class RecordingController(InteractionController):
        def __init__(self):
            self.steps = []

        def on_base_sim_step(self, task, *, step_index):
            self.steps.append(step_index)

    task = BaseTask({})
    controller = RecordingController()
    task._interaction_controller = controller
    task._notify_interaction_sim_step()
    task._notify_interaction_sim_step()
    assert controller.steps == [1, 2]

    # Registered direct step_sim overrides must emit the same core hook even
    # though they cannot inherit BaseTask.step_sim's implementation.
    from envs.task_bases.frying_scene import FryingSceneMixin
    from envs.tasks.oil_bottle_recovery import OilBottleRecovery
    from envs.tasks.open_microwave import OpenMicrowave
    from envs.tasks.feed_documents import FeedDocuments

    assert "_notify_interaction_sim_step" in OpenMicrowave.step_sim.__code__.co_names
    assert "_notify_interaction_sim_step" in FeedDocuments.step_sim.__code__.co_names
    assert (
        "_notify_interaction_sim_step"
        in FryingSceneMixin.step_sim.__code__.co_names
    )
    assert OilBottleRecovery.step_sim is FryingSceneMixin.step_sim


def test_controller_factory_is_lazy_until_subclass_init_finishes():
    events = []

    class DeferredTask(BaseTask):
        def __init__(self):
            super().__init__({})
            self.subclass_ready = True

        def create_interaction_controller(self):
            events.append(self.subclass_ready)
            return InteractionController()

    task = DeferredTask()
    assert events == []
    assert isinstance(task.interaction_controller, InteractionController)
    assert events == [True]


def test_non_proceed_directive_fails_closed_and_remains_observable():
    rejected = PolicyDirective(
        disposition=PolicyDisposition.YIELD,
        reason="human owns target",
        hold_steps=2,
    )

    class YieldController(InteractionController):
        def before_base_policy_action(self, *args, **kwargs):
            return rejected

    task = BaseTask({})
    task._interaction_controller = YieldController()
    with pytest.raises(NotImplementedError, match="phase-0"):
        task.take_action([0.0], action_type="qpos")

    assert task.take_action_cnt == 0
    assert task.last_policy_directive == rejected
    assert task.robot is None
