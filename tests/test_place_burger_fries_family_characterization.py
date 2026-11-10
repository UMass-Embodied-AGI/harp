"""Scene-free characterization for the burger/fries task family.

The tests freeze the public family contract, seeded layout stream, interrupt
lifecycle, scripted choreography, and metric surfaces without constructing a
Genesis scene, renderer, planner, robot, or avatar asset.
"""

from __future__ import annotations

from dataclasses import fields
import inspect
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.inspect_motion_mixin import InspectMotionMixin
from envs.base_task import BaseTask
from envs.interaction.eval_mode import AssistEvalModeController
import envs.interaction.inspect_motion as inspect_interaction
from envs.manipulation import TopDownPickPlaceMixin
import envs.task_bases.place_burger_fries as base_module
from envs.task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
)
from envs.task_bases.place_burger_fries import (
    FoodSpec,
    PlaceBurgerFries,
    _BURGER_SPEC,
    _FRIES_ID,
    _FRIES_SPEC,
    _HAMBURG_ID,
    _TRAY_ID,
)
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
from envs.tasks._inspect_motion_interrupt_facade import (
    InspectMotionInterruptFacade,
)
from envs.tasks.place_burger_and_fries_contribution import PlaceBurgerAndFriesContribution
from envs.tasks.place_burger_and_fries_interruption import PlaceBurgerAndFriesInterruption
from envs.tasks.place_burger_and_fries_copresence import PlaceBurgerAndFriesCopresence
from scripts.eval_expert_set import policy_play_mode


ALL_CLASSES = (
    PlaceBurgerFries,
    PlaceBurgerAndFriesContribution,
    PlaceBurgerAndFriesInterruption,
    PlaceBurgerAndFriesCopresence,
)


@pytest.mark.parametrize(
    ("task_cls", "module", "instruction", "uses_avatar"),
    (
        (
            PlaceBurgerFries,
            "envs.task_bases.place_burger_fries",
            "place the hamburger and the french fries on the tray",
            False,
        ),
        (
            PlaceBurgerAndFriesContribution,
            "envs.tasks.place_burger_and_fries_contribution",
            "place the hamburger on the tray while the human adds one "
            "food item",
            True,
        ),
        (
            PlaceBurgerAndFriesInterruption,
            "envs.tasks.place_burger_and_fries_interruption",
            "place the hamburger and the french fries on the tray; if the "
            "human inspects one item midway through, wait and then continue",
            True,
        ),
        (
            PlaceBurgerAndFriesCopresence,
            "envs.tasks.place_burger_and_fries_copresence",
            "place the meal items on the tray while the human works nearby "
            "on the same table",
            True,
        ),
    ),
)
def test_public_identity_instruction_and_surface(
    task_cls,
    module,
    instruction,
    uses_avatar,
):
    assert task_cls.__module__ == module
    assert issubclass(task_cls, BaseTask)
    assert task_cls.INSTRUCTION == instruction
    assert bool(task_cls.use_avatar) is uses_avatar

    signature = inspect.signature(task_cls)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method in (
        "_create_table",
        "_food_specs_for_episode",
        "_sample_spawn_xy",
        "_drop_slots",
        "load_actors",
        "play_once",
        "check_success",
        "evaluate",
        "take_action",
    ):
        assert callable(getattr(task_cls, method))


def test_direct_family_inheritance_and_interrupt_controller_contract():
    controller_cls = getattr(
        inspect_interaction,
        "PlaceBurgerFriesInterruptController",
        None,
    )

    assert PlaceBurgerFries.__bases__ == (TopDownPickPlaceMixin, BaseTask)
    assert PlaceBurgerAndFriesContribution.__bases__ == (PlaceBurgerFries,)
    assert PlaceBurgerAndFriesCopresence.__bases__ == (
        NeutralAvatarTableWorkMixin,
        PlaceBurgerFries,
    )
    assert isinstance(
        PlaceBurgerAndFriesContribution({}).interaction_controller,
        AssistEvalModeController,
    )

    # This suite lands before the final interrupt-family migration. Accept the
    # exact legacy graph and its single intended composed replacement.
    if controller_cls is None:
        assert PlaceBurgerAndFriesInterruption.__bases__ == (
            InspectMotionMixin,
            PlaceBurgerFries,
        )
        return

    assert PlaceBurgerAndFriesInterruption.__bases__ == (
        InspectMotionInterruptFacade,
        PlaceBurgerFries,
    )
    assert InspectMotionMixin not in PlaceBurgerAndFriesInterruption.__mro__
    controller = PlaceBurgerAndFriesInterruption({}).interaction_controller
    assert isinstance(controller, controller_cls)
    assert controller.handles_policy_ticks is False
    # The expert harness ticks this controller through the base
    # on_scripted_rollout_tick -> _eval_step_avatar adapter.
    assert controller.handles_scripted_rollout_ticks is False
    assert controller.rng_namespace == "place_burger_fries.interrupt.v1"


def test_registry_variants_and_no_human_collapse_are_stable():
    expected = {
        "place_burger_fries": (PlaceBurgerFries, TaskVariant.BASE),
        "place_burger_and_fries_contribution": (
            PlaceBurgerAndFriesContribution,
            TaskVariant.ASSIST,
        ),
        "place_burger_and_fries_interruption": (
            PlaceBurgerAndFriesInterruption,
            TaskVariant.INTERRUPT,
        ),
        "place_burger_and_fries_copresence": (
            PlaceBurgerAndFriesCopresence,
            TaskVariant.NEUTRAL,
        ),
    }
    for name, (task_cls, variant) in expected.items():
        descriptor = TASK_DESCRIPTOR_MAP[name]
        assert descriptor.task_cls is task_cls
        assert descriptor.family == "place_burger_fries"
        assert descriptor.variant is variant
        assert descriptor.registration is TaskRegistration.CANONICAL
        assert TASK_MAP[name] is task_cls
        assert resolve_task_class(name) == (name, task_cls)

    assert (
        NO_HUMAN_CANONICAL_TASK_MAP["place_burger_fries"]
        is PlaceBurgerFries
    )
    for name in expected:
        assert NO_HUMAN_VARIANT_ALIASES.get(
            name,
            "place_burger_fries",
        ) == "place_burger_fries"
        assert resolve_task_class(name, no_human=True) == (
            "place_burger_fries",
            PlaceBurgerFries,
        )


def test_food_spec_public_shape_constants_and_pickle_compatibility():
    assert FoodSpec.__module__ == "envs.task_bases.place_burger_fries"
    assert [field.name for field in fields(FoodSpec)] == [
        "name",
        "asset_id",
        "model_id",
        "spawn_quat",
        "spawn_z_offset",
        "friction",
        "close_value",
        "below_center",
        "tcp_yaw_deg",
        "spawn_xy_range",
        "transit_steps",
        "transit_sim",
        "align_to_object_yaw",
        "decompose_error_threshold",
    ]
    assert (_HAMBURG_ID, _FRIES_ID, _TRAY_ID) == (
        "006_hamburg",
        "005_french-fries",
        "008_tray",
    )
    assert (_BURGER_SPEC.name, _BURGER_SPEC.model_id) == ("burger", 4)
    assert (_FRIES_SPEC.name, _FRIES_SPEC.model_id) == ("fries", 0)
    assert _FRIES_SPEC.align_to_object_yaw is True
    assert _FRIES_SPEC.decompose_error_threshold == float("inf")

    restored = pickle.loads(pickle.dumps(_FRIES_SPEC))
    assert type(restored) is FoodSpec
    assert restored.name == _FRIES_SPEC.name
    assert restored.asset_id == _FRIES_SPEC.asset_id
    np.testing.assert_array_equal(restored.spawn_quat, _FRIES_SPEC.spawn_quat)

    from envs.tasks import PlaceBurgerFries as public_base
    from envs.tasks import PlaceBurgerAndFriesContribution as public_assist
    from envs.tasks import PlaceBurgerAndFriesInterruption as public_interrupt
    from envs.tasks import PlaceBurgerAndFriesCopresence as public_neutral

    assert (
        public_base,
        public_assist,
        public_interrupt,
        public_neutral,
    ) == ALL_CLASSES


class _FakeEntity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self):
        return self._quat.copy()

    def get_AABB(self):
        return np.array([
            self._pos - [0.10, 0.10, 0.01],
            self._pos + [0.10, 0.10, 0.04],
        ])


class _FakeActor:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.entity = _FakeEntity(pos, quat)

    def get_pose(self):
        return SimpleNamespace(
            p=self.entity.get_pos(),
            q=self.entity.get_quat(),
        )


def _capture_actor_layout(monkeypatch, task_cls, config, seed):
    events = []

    def load_object(_scene, pose, object_id, **kwargs):
        events.append({
            "id": object_id,
            "model_id": kwargs["model_id"],
            "pos": np.asarray(pose.p, dtype=np.float64).copy(),
            "quat": np.asarray(pose.q, dtype=np.float64).copy(),
            "friction": kwargs["friction"],
            "is_static": kwargs["is_static"],
        })
        return _FakeActor(pose.p, pose.q)

    monkeypatch.setattr(base_module, "load_object", load_object)

    task = task_cls(config)
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    if isinstance(task, PlaceBurgerAndFriesCopresence):
        # Resolve the real neutral job to freeze its leading RNG draw, while
        # keeping this characterization independent of prop assets.
        task._load_neutral_avatar_props = lambda: None
        task._load_neutral_avatar_region_debug_tiles = lambda: None

    np.random.seed(seed)
    task.load_actors()
    tail = np.random.random_sample(8)
    neutral_job = getattr(task, "_neutral_avatar_job", None) or {}
    return task, events, neutral_job.get("motion"), tail


# In the easy layout the fries carton is PRE-PLACED on its
# tray drop slot (x = tray_x + 0.06 slot offset, z = table + 2*rim + 0.045)
# and the robot only picks the burger; the assist variant instead spawns two
# shared burgers in one region (the assist 50-50 split).  The interrupt's
# inspected index is always the loose (non-pre-placed) food.  The round
# model_4 is the default burger.
_LAYOUT_CASES = (
    (
        PlaceBurgerFries,
        {},
        0,
        (
            (_TRAY_ID, 0, (-0.167677974411, 0.071518936637, 0.780)),
            (_HAMBURG_ID, 4, (0.146884116960, -0.023113662536, 0.815)),
            (_FRIES_ID, 0, (-0.107677974411, 0.071518936637, 0.810)),
        ),
        None,
        None,
        (
            0.891773000782,
            0.963662760501,
            0.383441518826,
            0.791725038083,
            0.528894919753,
            0.568044561094,
            0.925596638293,
            0.071036058198,
        ),
    ),
    (
        PlaceBurgerFries,
        {"hard": True},
        7,
        (
            (_TRAY_ID, 0, (-0.238553756594, 0.077991879224, 0.780)),
            (_HAMBURG_ID, 4, (0.130774380562, -0.014854339724, 0.815)),
            (_FRIES_ID, 0, (-0.007751307329, -0.182418542786, 0.770)),
            (_HAMBURG_ID, 4, (0.050214032084, -0.068005895040, 0.815)),
            (_FRIES_ID, 0, (0.100597046933, -0.146591616453, 0.770)),
        ),
        None,
        None,
        (
            0.696160463517,
            0.955468323003,
            0.682913854375,
            0.053128690673,
            0.308852684864,
            0.592594687323,
            0.235120407257,
            0.964970999536,
        ),
    ),
    (
        PlaceBurgerAndFriesContribution,
        {},
        42,
        (
            (_TRAY_ID, 0, (-0.193818982173, 0.095071430641, 0.780)),
            (_HAMBURG_ID, 4, (0.051198904067, -0.071868294296, 0.815)),
            (_HAMBURG_ID, 4, (0.193235229155, 0.004156101644, 0.815)),
        ),
        None,
        None,
        (
            0.708072577796,
            0.020584494296,
            0.969909852162,
            0.832442640800,
            0.212339110678,
            0.181824967207,
            0.183404509853,
            0.304242242960,
        ),
    ),
    (
        PlaceBurgerAndFriesInterruption,
        {"randomize_inspect": True},
        42,
        (
            (_TRAY_ID, 0, (-0.193818982173, 0.095071430641, 0.780)),
            (_HAMBURG_ID, 4, (-0.011001369916, -0.090706622053, 0.815)),
            (_FRIES_ID, 0, (-0.133818982173, 0.095071430641, 0.810)),
        ),
        0,
        None,
        (
            0.866176145775,
            0.601115011743,
            0.708072577796,
            0.020584494296,
            0.969909852162,
            0.832442640800,
            0.212339110678,
            0.181824967207,
        ),
    ),
    (
        PlaceBurgerAndFriesCopresence,
        {},
        0,
        (
            (_TRAY_ID, 0, (-0.161073307266, 0.084426574858, 0.780)),
            (_HAMBURG_ID, 4, (0.107657256094, -0.041320501149, 0.815)),
            (_FRIES_ID, 0, (-0.101073307266, 0.084426574858, 0.810)),
        ),
        None,
        # The "wipe_table_1" choice now resolves to
        # wipe_table_2_reverse instead of being dropped (4 choices, not 3).
        "wipe_table_2_reverse",
        (
            0.056712977317,
            0.272656294580,
            0.477665117321,
            0.812168728775,
            0.479977172375,
            0.392784796101,
            0.836078763537,
            0.337396160417,
        ),
    ),
)


@pytest.mark.parametrize(
    (
        "task_cls",
        "config",
        "seed",
        "expected_events",
        "expected_interrupted",
        "expected_neutral_job",
        "expected_tail",
    ),
    _LAYOUT_CASES,
)
def test_seeded_actor_layout_and_global_numpy_tail(
    monkeypatch,
    task_cls,
    config,
    seed,
    expected_events,
    expected_interrupted,
    expected_neutral_job,
    expected_tail,
):
    task, events, neutral_job, tail = _capture_actor_layout(
        monkeypatch,
        task_cls,
        config,
        seed,
    )

    assert len(events) == len(expected_events)
    for actual, (object_id, model_id, pos) in zip(events, expected_events):
        assert actual["id"] == object_id
        assert actual["model_id"] == model_id
        assert actual["is_static"] is False
        expected_friction = 2.0 if object_id == _TRAY_ID else 5.0
        assert actual["friction"] == expected_friction
        np.testing.assert_allclose(actual["pos"], pos, rtol=0.0, atol=5e-12)
    assert getattr(task, "_interrupted_food_idx", None) == expected_interrupted
    assert neutral_job == expected_neutral_job
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-12)


def test_interrupt_reset_draw_and_hook_order_match_legacy(monkeypatch):
    events = []

    def load_object(_scene, pose, object_id, **_kwargs):
        events.append(("load", object_id))
        return _FakeActor(pose.p, pose.q)

    monkeypatch.setattr(base_module, "load_object", load_object)
    task = PlaceBurgerAndFriesInterruption({
        "eval_mode": True,
        "randomize_inspect": True,
        "eval_trigger_step_min": 20,
        "eval_trigger_step_max": 100,
    })
    task.avatar = SimpleNamespace(frame_ratio=None, motion_data={})

    original_prepare = task._prepare_inspect_motion_entry

    def prepare(entry):
        events.append(("inspect", entry["name"]))
        return original_prepare(entry)

    task._prepare_inspect_motion_entry = prepare
    task._calibrate_avatar_for_inspect = lambda: events.append(("calibrate",))
    # After the (burned) legacy trigger draw the controller arms
    # the progress trigger; stub it so the scene-free test needs no robot.
    task.interaction_controller.init_eval_state = (
        lambda _task: events.append(("arm",))
    )
    observation = {"initial": 42}

    def fake_base_reset(self, seed=0):
        np.random.seed(seed)
        self.scene = object()
        self.TABLE_TOP_Z = 0.765
        self.load_actors()
        events.append(("interrupted", self._interrupted_food_idx))
        events.append(("observation",))
        return observation

    controller_cls = getattr(
        inspect_interaction,
        "PlaceBurgerFriesInterruptController",
        None,
    )
    if controller_cls is None:
        monkeypatch.setattr(PlaceBurgerFries, "reset", fake_base_reset)
        actual_observation = task.reset(seed=42)
    else:
        actual_observation = fake_base_reset(task, seed=42)
        task.interaction_controller.after_base_reset_observation(
            task,
            observation=actual_observation,
        )

    # The legacy draw is burned (value kept for audit only);
    # _eval_trigger_step stays None until the progress trigger fires.
    assert task._eval_trigger_step is None
    events.append(("trigger", task._eval_legacy_trigger_draw))
    tail = np.random.random_sample(8)

    assert actual_observation is observation
    # The pre-placed fries is never the interrupt target, so
    # the draw lands on the loose burger (index 0); the sampled motion/trigger
    # and the frozen tail follow from that RNG stream.
    assert events == [
        ("load", _TRAY_ID),
        ("load", _HAMBURG_ID),
        ("load", _FRIES_ID),
        ("interrupted", 0),
        ("observation",),
        ("inspect", "Inspect8"),
        ("calibrate",),
        ("arm",),
        ("trigger", 43),
    ]
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW
    assert task.INSPECT_MOTION_NAME == "Inspect8"
    assert task._eval_policy_step_count == 0
    assert task._eval_interrupt_fired is False
    assert task._eval_interrupt_failed is False
    np.testing.assert_allclose(
        tail,
        (
            0.650888472949,
            0.056411579027,
            0.721998772267,
            0.938552709016,
            0.000778765841,
            0.992211559291,
            0.617481509628,
            0.611653160488,
        ),
        rtol=0.0,
        atol=5e-12,
    )


def test_interrupt_policy_tick_precedes_parent_action(monkeypatch):
    events = []
    task = PlaceBurgerAndFriesInterruption({"eval_mode": True})
    task.avatar = object()
    task._eval_step_avatar = lambda: events.append(("eval",))

    def parent_action(_self, action, action_type="qpos"):
        events.append(("parent", action, action_type))
        return {"obs": True}

    monkeypatch.setattr(PlaceBurgerFries, "take_action", parent_action)
    action = object()
    assert task.take_action(action, action_type="qpos_abs") == {"obs": True}
    assert events == [("eval",), ("parent", action, "qpos_abs")]


def test_expert_policy_routing_uses_unprivileged_parent_rollouts():
    mode, parent_play_once = policy_play_mode(
        "place_burger_and_fries_interruption", PlaceBurgerAndFriesInterruption, "expert",
    )
    assert mode == "parent_play_once"
    assert parent_play_once is PlaceBurgerFries.play_once
    # Assist 50-50: the blind expert has its own observation-only rollout.
    assert policy_play_mode(
        "place_burger_and_fries_contribution", PlaceBurgerAndFriesContribution, "expert",
    ) == ("play_blind_once", None)

    assert policy_play_mode(
        "place_burger_and_fries_copresence",
        PlaceBurgerAndFriesCopresence,
        "expert",
    ) == ("play_once", None)


def test_base_scripted_rollout_preserves_food_order_slots_and_parameters():
    task = PlaceBurgerFries({})
    burger = _FakeActor([0.1, 0.0, 0.815])
    fries = _FakeActor([0.0, -0.1, 0.770])
    task.food_actors = [(burger, _BURGER_SPEC), (fries, _FRIES_SPEC)]
    task._tray_xy = (-0.20, 0.05)
    task.FINAL_SETTLE_STEPS = 2
    events = []

    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.step_sim = lambda: events.append(("sim",))

    def pick(actor, asset_id, arm, **kwargs):
        events.append(("pick", actor, asset_id, arm, kwargs))
        return True

    task._pick_and_drop = pick
    assert task.play_once() is True
    assert [event[1] for event in events if event[0] == "pick"] == [
        burger,
        fries,
    ]
    picks = [event for event in events if event[0] == "pick"]
    assert picks[0][4]["drop_xy"] == pytest.approx((-0.26, 0.05))
    assert picks[1][4]["drop_xy"] == pytest.approx((-0.14, 0.05))
    assert picks[0][4]["align_to_object_yaw"] is False
    assert picks[1][4]["align_to_object_yaw"] is True
    assert events[0] == ("boost", "right")
    assert events[-2:] == [("sim",), ("sim",)]


def test_assist_scripted_rollout_runs_human_and_robot_in_parallel():
    # Assist 50-50: no avatar-first ordering.  The human is started, the
    # robot places the OTHER shared burger while the human moves, then the
    # robot waits (bounded) for the human and settles.
    task = PlaceBurgerAndFriesContribution({})
    task.avatar = object()
    task.FINAL_SETTLE_STEPS = 2
    events = []
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        planner=SimpleNamespace(plan_path=lambda *a, **k: None),
    ))
    task.avatar_collider = None
    task.food_actors = [
        (_FakeActor([0.05, 0.0, 0.79]), _BURGER_SPEC),
        (_FakeActor([0.18, 0.0, 0.79]), _BURGER_SPEC),
    ]
    task._human_target_idx = 1
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task._start_avatar_burger_assist = lambda: events.append(("human",))
    task._place_foods = lambda indices, arm_tag="right": (
        events.append(("robot", list(indices))), True,
    )[1]
    task._wait_for_human = lambda: events.append(("wait",))
    task.step_sim = lambda: events.append(("sim",))

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        ("human",),
        ("robot", [0]),
        ("wait",),
        ("sim",),
        ("sim",),
    ]


class _FakeArm:
    tcp_offset = object()


class _FakeRobot:
    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return _FakeArm()


class _FakeLinkPose:
    def __init__(self, value):
        self.value = value

    def to_pose7(self):
        return ("pose7", self.value)


class _SpareSequenceAvatar:
    def __init__(self, values):
        self._values = iter(values)

    def spare(self):
        return next(self._values, True)


def test_interrupt_scripted_rollout_preserves_phase_and_pick_order(
    monkeypatch,
):
    import envs.grasp as grasp_module

    task = PlaceBurgerAndFriesInterruption({})
    task.PRE_INTERRUPT_SETTLE_STEPS = 2
    task.PRE_ATTACH_WAIT_MARGIN_STEPS = 1
    task.POST_RELEASE_SETTLE_STEPS = 1
    task.FINAL_SETTLE_STEPS = 2
    task.TABLE_TOP_Z = 0.765
    task._tray_xy = (-0.20, 0.05)
    burger = _FakeActor([0.1, 0.0, 0.815])
    fries = _FakeActor([0.0, -0.1, 0.770])
    task.food_actors = [(burger, _BURGER_SPEC), (fries, _FRIES_SPEC)]
    task._interrupted_food_idx = 0
    task.robot = _FakeRobot()
    task.avatar = _SpareSequenceAvatar([False, False, True])
    events = []

    monkeypatch.setattr(
        grasp_module,
        "tcp_to_link_pose",
        lambda pose, _offset: _FakeLinkPose(pose),
    )
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.step_sim = lambda: events.append(("sim",))
    task.open_gripper = lambda arm: events.append(("open", arm))
    task._top_down_tcp = lambda pos: ("tcp", tuple(pos))
    task._move_seeded = lambda pose, arm: events.append(
        ("intent", pose, arm)
    )
    task._play_inspect_motion = lambda **kwargs: events.append(
        ("inspect", kwargs)
    )
    task._retreat_safe = lambda arm: events.append(("retreat", arm))
    task._wait_until_avatar_attach = lambda: events.append(("wait_attach",))

    def pick(actor, asset_id, arm, **kwargs):
        events.append(("pick", actor, asset_id, arm, kwargs))
        return True

    task._pick_and_drop = pick
    assert task.play_once() is True

    labels = [event[0] for event in events]
    assert labels == [
        "boost",
        "sim",
        "sim",
        "open",
        "intent",
        "inspect",
        "retreat",
        "wait_attach",
        "sim",
        "pick",
        "sim",
        "sim",
        "pick",
        "sim",
        "sim",
    ]
    picks = [event for event in events if event[0] == "pick"]
    assert [event[1] for event in picks] == [fries, burger]
    assert picks[0][4]["drop_xy"] == pytest.approx((-0.14, 0.05))
    assert picks[1][4]["drop_xy"] == pytest.approx((-0.26, 0.05))
    inspect_call = next(event[1] for event in events if event[0] == "inspect")
    assert inspect_call["attach_obj"] is burger.entity
    assert inspect_call["hand_id"] == task.AVATAR_HAND_ID
    assert inspect_call["attach_frame"] == task._attach_frame_scaled
    assert inspect_call["detach_frame"] == task._detach_frame_scaled
    assert inspect_call["return_to_idle_after"] is None


def _configure_scoring(task, food_positions):
    task.TABLE_TOP_Z = 0.765
    task._tray_xy = (-0.20, 0.05)
    task.tray_actor = _FakeActor([-0.20, 0.05, 0.765])
    specs = (
        (_BURGER_SPEC, _BURGER_SPEC)
        if isinstance(task, PlaceBurgerAndFriesContribution)
        else (_BURGER_SPEC, _FRIES_SPEC)
    )
    task.food_actors = [
        (_FakeActor(position), spec)
        for position, spec in zip(food_positions, specs)
    ]
    task.plan_success = True


def test_success_boundaries_and_variant_gates_are_stable():
    base = PlaceBurgerFries({})
    _configure_scoring(
        base,
        [(-0.20, 0.05, 0.77), (-0.04, 0.15, 0.77)],
    )
    assert base.check_success() is True
    base.food_actors[1][0].entity.set_pos([-0.0349, 0.15, 0.77])
    assert base.check_success() is False
    base.food_actors[1][0].entity.set_pos([-0.20, 0.05, 0.7449])
    assert base.check_success() is False
    base.plan_success = False
    assert base.check_success() is False

    assist = PlaceBurgerAndFriesContribution({})
    _configure_scoring(
        assist,
        [(-0.20, 0.05, 0.77), (-0.30, 0.10, 0.77)],
    )
    assert assist.check_success() is True
    # Both shared burgers must end on the tray, whoever carried them.
    assist.food_actors[1][0].entity.set_pos([0.0, 0.10, 0.77])
    assert assist.check_success() is False

    interrupt = PlaceBurgerAndFriesInterruption({})
    _configure_scoring(
        interrupt,
        [(-0.20, 0.05, 0.77), (-0.10, 0.05, 0.77)],
    )
    interrupt.avatar_collided = True
    assert interrupt.check_success() is True

    neutral = PlaceBurgerAndFriesCopresence({})
    _configure_scoring(
        neutral,
        [(-0.20, 0.05, 0.77), (-0.10, 0.05, 0.77)],
    )
    neutral.avatar_collided = False
    assert neutral.check_success() is True
    neutral.avatar_collided = True
    assert neutral.check_success() is False


@pytest.mark.parametrize(
    ("task_cls", "config", "extra_state", "expected_keys"),
    (
        (
            PlaceBurgerFries,
            {},
            {},
            (
                "plan_success",
                "success",
                "avatar_collision",
                "safe_distance",
                "avatar_safety",
                "avatar_safety_intervention",
                # BaseTask safety-metrics block.
                "safety",
            ),
        ),
        (
            PlaceBurgerAndFriesContribution,
            {"eval_mode": True},
            {
                "_eval_assist_trigger_step": 0,
                "_eval_assist_fired": True,
                "_eval_assist_failed": False,
                "_eval_policy_step_count": 7,
            },
            (
                "plan_success",
                "success",
                "avatar_collision",
                "safe_distance",
                "avatar_safety",
                "avatar_safety_intervention",
                # BaseTask safety-metrics block.
                "safety",
                "eval_mode",
                "eval_assist_trigger_step",
                "eval_assist_fired",
                "eval_assist_failed",
                "eval_policy_step_count",
                "shared_pool",
            ),
        ),
        (
            PlaceBurgerAndFriesInterruption,
            {"eval_mode": True},
            {
                "_eval_trigger_step": 41,
                "_eval_policy_step_count": 8,
                "_eval_interrupt_fired": True,
                "_eval_interrupt_failed": False,
                "_interrupted_food_idx": 1,
            },
            (
                "plan_success",
                "success",
                "avatar_collision",
                "safe_distance",
                "avatar_safety",
                "avatar_safety_intervention",
                # BaseTask safety-metrics block.
                "safety",
                "eval_mode",
                "eval_trigger_step",
                "eval_policy_step_count",
                "eval_interrupt_fired",
                "eval_interrupt_failed",
                "interrupted_food_idx",
                # Uniform interrupt metrics in eval mode.
                "interrupt",
                "pace_interrupt",
            ),
        ),
        (
            PlaceBurgerAndFriesCopresence,
            {"eval_mode": True},
            {
                "_neutral_eval_trigger_step": 12,
                "_neutral_eval_policy_step_count": 5,
                "_neutral_avatar_started": True,
                "_neutral_eval_avatar_failed": False,
            },
            (
                "plan_success",
                "success",
                "avatar_collision",
                "safe_distance",
                "avatar_safety",
                "avatar_safety_intervention",
                # BaseTask safety-metrics block.
                "safety",
                "avatar_collided",
                "neutral_object_collided",
                "neutral_eval_trigger_step",
                "neutral_eval_policy_step_count",
                "neutral_avatar_started",
                "neutral_eval_avatar_failed",
                # A human prop the robot knocked over (scored failure).
                "neutral_prop_disturbed",
                # Contested co-presence layout metrics.
                "neutral_contested",
                "neutral_contested_requested",
                "neutral_contested_reason",
                "neutral_layout",
            ),
        ),
    ),
)
def test_evaluate_metric_values_and_key_order(
    task_cls,
    config,
    extra_state,
    expected_keys,
):
    task = task_cls(config)
    _configure_scoring(
        task,
        [(-0.20, 0.05, 0.77), (-0.10, 0.05, 0.77)],
    )
    for name, value in extra_state.items():
        setattr(task, name, value)

    metrics = task.evaluate()

    assert tuple(metrics) == expected_keys
    assert metrics["plan_success"] is True
    assert metrics["success"] is True
    if task_cls is PlaceBurgerAndFriesContribution:
        assert metrics["eval_assist_trigger_step"] == 0
        assert metrics["eval_policy_step_count"] == 7
        assert metrics["shared_pool"]["same_target"] is None
    elif task_cls is PlaceBurgerAndFriesInterruption:
        assert metrics["eval_trigger_step"] == 41
        assert metrics["interrupted_food_idx"] == 1
    elif task_cls is PlaceBurgerAndFriesCopresence:
        assert metrics["neutral_eval_trigger_step"] == 12
        assert metrics["neutral_avatar_started"] is True


class _HeldArm:
    def __init__(self, tcp, gripper):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "task_cls", [PlaceBurgerFries, PlaceBurgerAndFriesContribution, PlaceBurgerAndFriesInterruption,
                 PlaceBurgerAndFriesCopresence],
)
def test_food_held_over_the_tray_is_not_placed(task_cls):
    # Regression (interrupt_check stage 2): the eval human inspecting a food
    # carried it over the tray and the episode scored success mid-inspection.
    task = task_cls({})
    positions = [(-0.20, 0.05, 0.77), (-0.12, 0.08, 0.77)]
    _configure_scoring(task, positions)
    task.avatar_collision_checker = None
    held = task.food_actors[1][0]
    rig = SimpleNamespace(attached_object=[None, None], two_hand_attached_object=None)
    task.avatar = SimpleNamespace(robot=rig)
    assert task.check_success() is True

    # Human holds the food over the tray (one- or two-handed).
    rig.attached_object[1] = (held.entity, np.zeros(3), np.eye(3))
    assert task.check_success() is False
    rig.attached_object[1] = None
    rig.two_hand_attached_object = (held.entity, np.zeros(3))
    assert task.check_success() is False
    rig.two_hand_attached_object = None
    assert task.check_success() is True

    # Food hanging 10 cm above the tray surface is not resting on it.
    held.entity.set_pos([-0.12, 0.08, 0.765 + 0.03 + 0.10])
    assert task.check_success() is False
    held.entity.set_pos(positions[1])

    # Food still in the closed gripper is not placed; an open gripper is fine.
    arm = _HeldArm(positions[1], gripper=0.3)
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    assert task.check_success() is False
    arm.gripper_val = 1.0
    assert task.check_success() is True


def test_skillet_food_held_over_the_pan_is_not_placed():
    from envs.task_bases.place_food_in_skillet import PlaceFoodInSkillet

    task = PlaceFoodInSkillet({})
    task.TABLE_TOP_Z = 0.765  # set by the scene at reset
    bowl = np.array([0.10, 0.20, 0.80])
    food = SimpleNamespace(entity=object())
    spec = SimpleNamespace(asset_id="006_hamburg", model_id=0)
    center = {"p": bowl + [0.0, 0.0, 0.03]}
    task._pan_bowl_center = lambda: bowl.copy()
    task._actor_center = lambda actor, asset_id, model_id: center["p"].copy()
    rig = SimpleNamespace(attached_object=[None, None], two_hand_attached_object=None)
    task.avatar = SimpleNamespace(robot=rig)
    assert task._food_resting_in_skillet(food, spec) is True
    rig.attached_object[1] = (food.entity, np.zeros(3), np.eye(3))
    assert task._food_resting_in_skillet(food, spec) is False
    rig.attached_object[1] = None
    center["p"] = bowl + [0.0, 0.0, 0.25]  # held high over the pan
    assert task._food_resting_in_skillet(food, spec) is False
