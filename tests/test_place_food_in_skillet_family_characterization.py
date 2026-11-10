"""Scene-free characterization for the skillet task family.

The skillet variants deliberately share robot manipulation while retaining
different layout, allocation, and human-interaction policies.  These tests
freeze that public behavior at small fake boundaries; they never construct a
Genesis scene, renderer, planner, robot, or avatar asset.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.inspect_motion_mixin import InspectMotionMixin
from envs.base_task import BaseTask
from envs.interaction.eval_mode import AssistEvalModeController
import envs.interaction.inspect_motion as inspect_interaction
from envs.manipulation import TopDownPickPlaceMixin
from envs.scenes.kitchen import KitchenSceneMixin
import envs.task_bases.place_food_in_skillet as base_module
from envs.task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
)
from envs.task_bases.place_food_in_skillet import (
    FoodSpec,
    PlaceFoodInSkillet,
    _APPLE_ID,
    _BREAD_ID,
    _BURGER_ID,
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
import envs.tasks.place_food_in_skillet_contribution as assist_module
from envs.tasks.place_food_in_skillet_contribution import PlaceFoodInSkilletContribution
from envs.tasks.place_food_in_skillet_interruption import (
    PlaceFoodInSkilletInterruption,
)
from envs.tasks.place_food_in_skillet_copresence import PlaceFoodInSkilletCopresence
from scripts.eval_expert_set import policy_play_mode


ALL_CLASSES = (
    PlaceFoodInSkillet,
    PlaceFoodInSkilletContribution,
    PlaceFoodInSkilletInterruption,
    PlaceFoodInSkilletCopresence,
)


@pytest.mark.parametrize(
    ("task_cls", "module", "instruction", "uses_avatar"),
    (
        (
            PlaceFoodInSkillet,
            "envs.task_bases.place_food_in_skillet",
            "place the skillet on the stove and put the foods in it",
            False,
        ),
        (
            PlaceFoodInSkilletContribution,
            "envs.tasks.place_food_in_skillet_contribution",
            "place the skillet on the stove and put the foods in it while the "
            "human helps place one food",
            True,
        ),
        (
            PlaceFoodInSkilletInterruption,
            "envs.tasks.place_food_in_skillet_interruption",
            "place the skillet on the stove and put the foods in it; the human "
            "may inspect one food item while you are reaching for it",
            True,
        ),
        (
            PlaceFoodInSkilletCopresence,
            "envs.tasks.place_food_in_skillet_copresence",
            "place the food in the skillet while the human works nearby on the "
            "same table",
            True,
        ),
    ),
)
def test_public_identity_and_family_surface(
    task_cls, module, instruction, uses_avatar,
):
    assert task_cls.__module__ == module
    assert issubclass(task_cls, BaseTask)
    assert task_cls.INSTRUCTION == instruction
    assert bool(task_cls.use_avatar) is uses_avatar

    signature = inspect.signature(task_cls)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method in (
        "load_actors",
        "play_once",
        "check_success",
        "evaluate",
        "take_action",
        "_enabled_food_specs",
        "_place_pan_on_cooktop",
        "_pick_and_drop_food",
        "_food_in_skillet",
    ):
        assert callable(getattr(task_cls, method))


def test_shared_core_sibling_architecture_and_controller_contracts():
    core = getattr(base_module, "SkilletTaskCore", None)
    controller_cls = getattr(
        inspect_interaction, "SkilletInterruptController", None,
    )

    # This characterization lands before the extraction.  Accept the exact
    # legacy graph at that checkpoint and the one intended replacement graph;
    # no third inheritance shape is compatible with the release contract.
    if core is None:
        assert PlaceFoodInSkillet.__bases__ == (
            TopDownPickPlaceMixin,
            KitchenSceneMixin,
            BaseTask,
        )
        assert PlaceFoodInSkilletContribution.__bases__ == (
            TopDownPickPlaceMixin,
            KitchenSceneMixin,
            BaseTask,
        )
        assert PlaceFoodInSkilletInterruption.__bases__ == (
            InspectMotionMixin,
            PlaceFoodInSkillet,
        )
        assert PlaceFoodInSkilletCopresence.__bases__ == (
            NeutralAvatarTableWorkMixin,
            PlaceFoodInSkillet,
        )
        assert isinstance(
            PlaceFoodInSkilletContribution({}).interaction_controller,
            AssistEvalModeController,
        )
        assert controller_cls is None
        return

    assert core.__bases__ == (
        TopDownPickPlaceMixin,
        KitchenSceneMixin,
        BaseTask,
    )
    assert PlaceFoodInSkillet.__bases__ == (core,)
    assert PlaceFoodInSkilletContribution.__bases__ == (core,)
    assert PlaceFoodInSkilletInterruption.__bases__ == (
        InspectMotionInterruptFacade,
        core,
    )
    assert PlaceFoodInSkilletCopresence.__bases__ == (
        NeutralAvatarTableWorkMixin,
        core,
    )
    assert all(issubclass(task_cls, core) for task_cls in ALL_CLASSES)
    assert not issubclass(PlaceFoodInSkilletContribution, PlaceFoodInSkillet)
    assert not issubclass(PlaceFoodInSkilletInterruption, PlaceFoodInSkillet)
    assert not issubclass(PlaceFoodInSkilletCopresence, PlaceFoodInSkillet)
    assert InspectMotionMixin not in PlaceFoodInSkilletInterruption.__mro__

    assist = PlaceFoodInSkilletContribution({}).interaction_controller
    interrupt = PlaceFoodInSkilletInterruption({}).interaction_controller
    assert isinstance(assist, AssistEvalModeController)
    assert controller_cls is not None
    assert isinstance(interrupt, controller_cls)
    assert interrupt.handles_policy_ticks is False
    assert interrupt.handles_scripted_rollout_ticks is False


def test_registry_variants_and_no_human_collapse_are_stable():
    expected = {
        "place_food_in_skillet": (PlaceFoodInSkillet, TaskVariant.BASE),
        "place_food_in_skillet_contribution": (
            PlaceFoodInSkilletContribution,
            TaskVariant.ASSIST,
        ),
        "place_food_in_skillet_interruption": (
            PlaceFoodInSkilletInterruption,
            TaskVariant.INTERRUPT,
        ),
        "place_food_in_skillet_copresence": (
            PlaceFoodInSkilletCopresence,
            TaskVariant.NEUTRAL,
        ),
    }
    for name, (task_cls, variant) in expected.items():
        descriptor = TASK_DESCRIPTOR_MAP[name]
        assert descriptor.task_cls is task_cls
        assert descriptor.family == "place_food_in_skillet"
        assert descriptor.variant is variant
        assert descriptor.registration is TaskRegistration.CANONICAL
        assert TASK_MAP[name] is task_cls
        assert resolve_task_class(name) == (name, task_cls)

    assert (
        NO_HUMAN_CANONICAL_TASK_MAP["place_food_in_skillet"]
        is PlaceFoodInSkillet
    )
    for name in expected:
        assert NO_HUMAN_VARIANT_ALIASES.get(
            name, "place_food_in_skillet",
        ) == "place_food_in_skillet"
        assert resolve_task_class(name, no_human=True) == (
            "place_food_in_skillet",
            PlaceFoodInSkillet,
        )


@pytest.mark.parametrize("task_cls", ALL_CLASSES)
def test_constructor_copies_config_and_keeps_robot_defaults(task_cls):
    config = {"renderer": "rasterizer"}
    original = dict(config)
    task = task_cls(config)

    assert config == original
    assert task.config["robot_type"] == "franka"
    assert task.config["robot_single_arm"] is True
    assert task.pan_actor is None
    assert task.food_actors == []
    np.testing.assert_array_equal(task._cooktop_xy, [-0.12, 0.08])


def test_variant_configuration_and_geometry_compatibility_fields():
    base = PlaceFoodInSkillet({})
    assist = PlaceFoodInSkilletContribution({})
    interrupt = PlaceFoodInSkilletInterruption({})
    neutral = PlaceFoodInSkilletCopresence({})

    assert "track_avatar_collision" not in base.config
    assert assist.config["track_avatar_collision"] is False
    assert interrupt.config["track_avatar_collision"] is True
    assert neutral.config["random_object"] is True
    # Easy default: variants pre-place the pan on the
    # cooktop unless hard mode / force_dynamic_pan opts into the pick.
    assert interrupt._force_dynamic_pan is False
    assert interrupt._pan_starts_on_cooktop() is True
    assert assist._pan_starts_on_cooktop() is True
    assert interrupt.config["foods"] == ["burger", "bread"]
    assert assist.config["foods"] == ["burger", "bread"]
    hard_interrupt = PlaceFoodInSkilletInterruption({"hard": True})
    assert hard_interrupt._pan_starts_on_cooktop() is False
    forced_assist = PlaceFoodInSkilletContribution({"force_dynamic_pan": True})
    assert forced_assist._pan_starts_on_cooktop() is False
    assert interrupt._attach_frame_scaled == 340
    assert interrupt._detach_frame_scaled == 1870
    assert assist._avatar_food_idx is None
    assert assist._robot_food_indices == []

    assert base.FOOD_TRANSPORT_ABOVE_TABLE == 0.28
    assert base.FOOD_DROP_ABOVE_RIM == 0.055
    # Base/neutral/interrupt food jitter is 0.035 (assist keeps 0.030).
    assert base.FOOD_SPAWN_JITTER_XY == 0.035
    assert assist.FOOD_TRANSPORT_ABOVE_TABLE == 0.42
    assert assist.FOOD_DROP_ABOVE_RIM == 0.12
    assert assist.FOOD_SPAWN_JITTER_XY == 0.030
    assert assist.FOOD_MIN_SEP == 0.18
    assert assist.AVATAR_RANDOM_FOOD_POOL == ("burger", "apple")

    spec = neutral.NEUTRAL_TASK_SPEC
    assert spec.work_xy == (0.58, 0.26)
    assert spec.mouse_work_xy == (0.58, 0.32)
    assert spec.table_xy_bounds == (-0.80, 0.80, -0.45, 0.55)
    assert spec.max_table_edge_gap == 0.67


class _FakeEntity:
    def __init__(self, pos, quat):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return self._quat.copy()

    def get_AABB(self):
        return np.array([
            self._pos - [0.10, 0.10, 0.01],
            self._pos + [0.10, 0.10, 0.04],
        ])


def _capture_actor_layout(monkeypatch, task_cls, config, seed):
    events = []

    def load_mesh(_scene, _path, pose, **kwargs):
        events.append({
            "kind": "pan",
            "pos": pose.p.copy(),
            "quat": pose.q.copy(),
            "friction": kwargs["friction"],
            "density": kwargs["density"],
        })
        return _FakeEntity(pose.p, pose.q)

    def load_object(_scene, pose, object_id, **kwargs):
        events.append({
            "kind": "food",
            "id": object_id,
            "model_id": kwargs["model_id"],
            "pos": pose.p.copy(),
            "quat": pose.q.copy(),
            "friction": kwargs["friction"],
        })
        return _FakeEntity(pose.p, pose.q)

    for module in (base_module, assist_module):
        monkeypatch.setattr(module, "load_mesh", load_mesh)
        monkeypatch.setattr(module, "load_object", load_object)

    task = task_cls(config)
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    task.build_kitchen = lambda: None
    np.random.seed(seed)
    task.load_actors()
    tail = np.random.random_sample(8)
    return task, events, tail


# Food positions for FOOD_SPAWN_JITTER_XY = 0.035 (the uniform draws of a
# 0.020 layout scaled by 1.75 about each spawn centre; RNG tails unchanged).
_LAYOUT_CASES = (
    (
        PlaceFoodInSkillet,
        {},
        0,
        ("burger",),
        (
            ("pan", None, None, (-0.12, 0.135, 0.8224)),
            ("food", _BURGER_ID, 5, (0.203416945275, 0.015063255646, 0.82)),
        ),
        (
            0.602763376072,
            0.544883182997,
            0.423654799339,
            0.645894113067,
            0.437587211263,
            0.891773000782,
            0.963662760501,
            0.383441518826,
        ),
    ),
    (
        PlaceFoodInSkillet,
        {"hard": True},
        7,
        ("burger", "bread"),
        (
            ("pan", None, None, (0.237289248681, 0.038397563767, 0.815)),
            ("food", _BURGER_ID, 5, (0.415688646201, -0.104357437552, 0.82)),
            ("food", _BREAD_ID, 1, (-0.306540734160, -0.097305289071, 0.82)),
        ),
        (
            0.501120463660,
            0.072051133360,
            0.268438980102,
            0.499882500826,
            0.679229996121,
            0.803739036104,
            0.380941133149,
            0.065936346906,
        ),
    ),
    # Easy default: the interrupt/assist pan is
    # pre-placed at the cooktop origin, the interrupt burger is pulled to
    # the avatar-reachable band, and both avatar variants default to two
    # foods (burger + bread).
    (
        PlaceFoodInSkilletInterruption,
        {},
        42,
        ("burger", "bread"),
        # The bread is PRE-PLACED in the skillet — it spawns
        # over the cooktop centre (z = table + cooktop 0.04 + 0.05) and the
        # robot manipulates only the inspected burger.
        (
            ("pan", None, None, (-0.12, 0.135, 0.8224)),
            ("food", _BURGER_ID, 5, (0.391217808319, 0.081550001449, 0.82)),
            ("food", _BREAD_ID, 1, (-0.12, 0.08, 0.855)),
        ),
        (
            0.731993941811,
            0.598658484197,
            0.156018640442,
            0.155994520336,
            0.058083612168,
            0.866176145775,
            0.601115011743,
            0.708072577796,
        ),
    ),
    (
        PlaceFoodInSkilletContribution,
        {},
        0,
        None,
        (
            ("pan", None, None, (-0.12, 0.135, 0.8224)),
            ("food", _BREAD_ID, 1, (0.234430298620, 0.276442140171, 0.82)),
            ("food", _BURGER_ID, 5, (0.219770191486, 0.056815585355, 0.82)),
        ),
        (
            0.297534606544,
            0.056712977317,
            0.272656294580,
            0.477665117321,
            0.812168728775,
            0.479977172375,
            0.392784796101,
            0.836078763537,
        ),
    ),
    (
        PlaceFoodInSkilletContribution,
        {},
        42,
        None,
        (
            ("pan", None, None, (-0.12, 0.135, 0.8224)),
            ("food", _BREAD_ID, 1, (0.272114289026, 0.297119030690, 0.82)),
            ("food", _BURGER_ID, 5, (-0.344503974729, -0.011791846615, 0.82)),
        ),
        (
            0.099974915818,
            0.459248891966,
            0.333708611139,
            0.142866817922,
            0.650888472949,
            0.056411579027,
            0.721998772267,
            0.938552709016,
        ),
    ),
)


@pytest.mark.parametrize(
    (
        "task_cls",
        "config",
        "seed",
        "selected_names",
        "expected_events",
        "expected_tail",
    ),
    _LAYOUT_CASES,
)
def test_seeded_actor_layout_and_global_rng_tail(
    monkeypatch,
    task_cls,
    config,
    seed,
    selected_names,
    expected_events,
    expected_tail,
):
    task, events, tail = _capture_actor_layout(
        monkeypatch, task_cls, config, seed,
    )

    if selected_names is not None:
        assert task.selected_food_names == selected_names
    assert len(events) == len(expected_events)
    for actual, expected in zip(events, expected_events):
        kind, object_id, model_id, pos = expected
        assert actual["kind"] == kind
        if kind == "pan":
            assert actual["friction"] == 1.5
            assert actual["density"] == 800.0
        else:
            assert actual["id"] == object_id
            assert actual["model_id"] == model_id
        np.testing.assert_allclose(actual["pos"], pos, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-13)


@pytest.mark.parametrize(
    ("seed", "expected_name", "expected_tail"),
    (
        (
            0,
            "burger",
            (
                0.592844618225,
                0.844265748581,
                0.857945617623,
                0.847251738784,
            ),
        ),
        (
            42,
            "apple",
            (
                0.796542986860,
                0.183434789866,
                0.779691000273,
                0.596850157946,
            ),
        ),
    ),
)
def test_neutral_single_food_sampling_and_rng_tail(
    seed, expected_name, expected_tail,
):
    task = PlaceFoodInSkilletCopresence({})
    np.random.seed(seed)
    specs = task._enabled_food_specs()
    tail = np.random.random_sample(4)

    assert [spec.name for spec in specs] == [expected_name]
    assert all(isinstance(spec, FoodSpec) for spec in specs)
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-13)


@pytest.mark.parametrize(
    ("seed", "expected_avatar", "expected_robot", "expected_tail"),
    (
        (
            0,
            0,
            [1, 2],
            (0.592844618225, 0.844265748581, 0.857945617623),
        ),
        (
            7,
            2,
            [0, 1],
            (0.227339079826, 0.318972225773, 0.978222897079),
        ),
    ),
)
def test_assist_food_assignment_and_rng_tail(
    seed, expected_avatar, expected_robot, expected_tail,
):
    # expected_avatar/expected_robot are the legacy
    # allocation: the human always took a burger/apple.  The human's food is
    # now a fair private coin over every food; the legacy global draw is still
    # consumed, so the global tail is unchanged.
    del expected_avatar, expected_robot
    task = PlaceFoodInSkilletContribution({})
    task._episode_seed = 9000000 + seed
    task._episode_rngs = {}
    task.interaction_controller.on_episode_seeded(task, task._episode_seed)
    task.food_actors = [
        (object(), SimpleNamespace(asset_id=asset_id))
        for asset_id in (_BURGER_ID, _BREAD_ID, _APPLE_ID)
    ]

    np.random.seed(seed)
    avatar_idx, robot_indices = task._choose_avatar_food()
    tail = np.random.random_sample(3)

    stream = zlib.crc32(b"interaction:place_food_in_skillet.assist.v1:human_food") & 0xFFFFFFFF
    coin = int(np.random.default_rng(np.random.SeedSequence(
        [task._episode_seed, stream])).integers(3))
    assert avatar_idx == coin
    assert robot_indices == [i for i in range(3) if i != coin]
    assert task._avatar_food_idx == coin
    assert task._robot_food_indices == robot_indices
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-13)
    # Drawn once per episode: a second allocation call agrees.
    assert task._choose_avatar_food()[0] == coin


def test_assist_food_coin_is_balanced_and_includes_bread():
    task = PlaceFoodInSkilletContribution({})
    picks = []
    for seed in range(9000000, 9000400):
        task._episode_seed = seed
        task._episode_rngs = {}
        task.interaction_controller.on_episode_seeded(task, seed)
        task.food_actors = [
            (object(), SimpleNamespace(asset_id=asset_id))
            for asset_id in (_BREAD_ID, _BURGER_ID)
        ]
        picks.append(task._choose_avatar_food()[0])
    # 400 fair flips between bread (0) and the other food (1).
    assert set(picks) == {0, 1}
    assert abs(float(np.mean(picks)) - 0.5) < 4 * 0.5 / np.sqrt(len(picks))


def test_assist_blind_commits_to_one_food_with_a_human():
    task = PlaceFoodInSkilletContribution({"eval_seed": 7, "eval_mode": False})
    task.food_actors = [
        _rollout_food("bread", _BREAD_ID),
        _rollout_food("burger", _BURGER_ID),
    ]
    task.avatar = SimpleNamespace(spare=lambda: True)
    task._boost_finger_pd = lambda _arm: None
    task.step_sim = lambda: None
    task._place_pan_on_cooktop = lambda _arm: True
    task._pan_starts_on_cooktop = lambda: True
    task._food_in_skillet = lambda _actor, _spec: False
    attempts = []
    task._pick_and_drop_food = lambda actor, _spec, _arm: (
        attempts.append(next(
            i for i, (candidate, _candidate_spec) in enumerate(task.food_actors)
            if candidate is actor
        ))
        or True
    )
    task.check_success = lambda: False

    expected = [int(i) for i in np.random.RandomState(50012).permutation(2)]
    assert task.play_blind_once() is True
    assert task._blind_robot_food_order == expected
    assert attempts == expected[:1]


def test_assist_controller_reset_and_policy_hook_order(monkeypatch):
    task = PlaceFoodInSkilletContribution({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_at_reset = lambda: events.append(("reset",))
    task._eval_at_step = lambda index: events.append(("avatar", index))

    task.interaction_controller.after_base_reset_observation(
        task, observation="observation",
    )

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = object()
    task.advance_interaction_policy_step()
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert events == [
        ("reset",),
        ("avatar", 1),
        ("robot", action, "qpos_abs"),
    ]
    assert task._eval_policy_step_count == 1


@pytest.mark.parametrize(
    ("eval_mode", "expected_last"),
    ((False, "calibrate"), (True, "eval")),
)
def test_interrupt_controller_reset_hook_order(
    eval_mode, expected_last, monkeypatch,
):
    controller_cls = getattr(
        inspect_interaction, "SkilletInterruptController", None,
    )
    task = PlaceFoodInSkilletInterruption({"eval_mode": eval_mode})
    task.avatar = SimpleNamespace(frame_ratio=None)
    events = []
    task._resolve_inspect_motion = lambda: events.append("inspect")
    task._select_interrupted_food = lambda: events.append("select")
    task._calibrate_avatar_for_inspect = lambda: events.append("calibrate")
    task._init_eval_interrupt_state = lambda: events.append("eval")

    if controller_cls is None:
        monkeypatch.setattr(
            BaseTask,
            "reset",
            lambda _self, seed=0: events.append("base") or "observation",
        )
        assert task.reset(seed=17) == "observation"
    else:
        events.append("base")
        task.interaction_controller.after_base_reset_observation(
            task, observation="observation",
        )

    assert events == ["base", "inspect", "select", expected_last]
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW


class _FakeArm:
    def __init__(self, tcp, gripper=1.0):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


def _interrupt_eval_task(tcp=(0.0, -0.35, 1.10), config=None):
    cfg = {"eval_mode": True}
    cfg.update(config or {})
    task = PlaceFoodInSkilletInterruption(cfg)
    task.TABLE_TOP_Z = 0.75
    arm = _FakeArm(tcp)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    food = (SimpleNamespace(entity="burger_entity"),
            SimpleNamespace(asset_id=_BURGER_ID, model_id=5))
    task.food_actors = [food]
    task._actor_center = lambda actor, _asset, _model: np.array([0.40, 0.05, 0.78])
    task._pan_bowl_center = lambda: np.array([-0.30, 0.0, 0.80])
    task._cooktop_xy = np.array([-0.30, 0.0])
    task._dryrun_capture_palm = lambda: np.array([0.1, -0.2, 0.9])
    return task, arm


def _fake_avatar(starts=None):
    starts = [] if starts is None else starts
    return SimpleNamespace(
        robot=SimpleNamespace(
            global_trans=np.array([0.0, 1.05, -0.34]),
            get_palm_center=lambda _h: np.array([5.0, 5.0, 5.0]),
        ),
        reset=lambda pos, rot: starts.append(np.asarray(pos).copy()),
        spare=lambda: True,
        motion_modules={},
    )


def test_interrupt_eval_state_burns_legacy_draw_and_arms_progress_trigger():
    # The random-step trigger is gone; its draw is burned so the
    # global RNG tail per seed is unchanged, and the progress trigger is
    # armed from the reset TCP pose.
    task, _arm = _interrupt_eval_task(config={
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
    })

    reference = np.random.RandomState(42)
    expected_trigger = int(reference.randint(7, 12))
    expected_tail = reference.random_sample(8)
    np.random.seed(42)
    task._init_eval_interrupt_state()
    actual_tail = np.random.random_sample(8)

    assert task._eval_trigger_step == expected_trigger
    assert task._eval_interrupt_fired is False
    assert task._policy_step_count == 0
    assert task._eval_target_food is None
    assert task._interrupt.trigger.armed is True
    assert task._interrupt.human.phase == "idle"
    assert task._interrupt.trigger.params.commit_radius_m == 0.32
    assert task._interrupt.trigger.params.trigger_progress == 0.25
    assert task._interrupt.trigger.params.min_policy_steps == 10
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_interrupt_progress_trigger_needs_motion_progress_and_pan_ready():
    # No force_dist at the start pose, no random step, no step
    # fallback; the human glides in toward the food the robot approaches.
    task, arm = _interrupt_eval_task()
    np.random.seed(0)
    task._init_eval_interrupt_state()
    starts = []
    task.avatar = _fake_avatar(starts)
    for _ in range(40):  # robot holds still: never fires
        task._eval_step_avatar()
    assert task._interrupt.fired is False
    assert task._interrupt.trigger.block_reason == "robot_not_moved"

    arm.tcp = np.array([0.40, 0.05, 0.95])  # hovering 0.17 m above the food
    task._pan_bowl_center = lambda: np.array([-0.30, 0.0, 0.70])  # pan not up
    task._eval_step_avatar()
    assert task._interrupt.fired is False
    assert task._interrupt.trigger.block_reason == "pan_not_on_cooktop"

    task._pan_bowl_center = lambda: np.array([-0.30, 0.0, 0.80])
    task._eval_step_avatar()
    assert task._interrupt.fired is True and task._eval_interrupt_fired is True
    assert task._interrupt.trigger.trigger_step == 42
    assert task._interrupt.human.phase == "glide_in"
    assert task._interrupted_idx == 0
    assert starts == []
    task._eval_step_avatar()
    assert len(starts) == 1  # first glide step, not a teleport to the target
    home = np.array([0.0, 1.05, -0.34])
    target = home + (
        np.array([0.40, 0.05, 0.78]) - np.array([0.1, -0.2, 0.9])
    ) + np.array([0.0, 0.0, task.AVATAR_EXTRA_Z])
    expected_first = home + (target - home) / task._interrupt.human.glide_steps
    np.testing.assert_allclose(starts[0], expected_first)


def test_interrupt_trigger_fires_on_the_demonstrated_intent_hover():
    # The skillet demos hover the TCP at TABLE_TOP_Z + 0.28 m over the food
    # (~0.26 m 3-D from its centre) before retreating, and the human reaches
    # in then.  The trigger must fire on that hover, not only on the later
    # pick descent; retreat/home poses must not fire it.
    from envs.interaction.interrupt_progress import (
        ProgressTrigger,
        ProgressTriggerParams,
    )

    food = np.array([0.40, 0.05, 0.78])
    reset_tcp = food + np.array([-0.405, 0.105, 0.09])  # ~0.43 m, as in the demos
    hover = np.array([food[0] - 0.05, food[1] + 0.07, 0.75 + 0.28])
    retreat = food + np.array([-0.40, -0.10, 0.30])

    task, arm = _interrupt_eval_task(tcp=reset_tcp)
    np.random.seed(0)
    task._init_eval_interrupt_state()
    task.avatar = _fake_avatar([])

    arm.tcp = retreat.copy()
    for _ in range(12):
        task._eval_step_avatar()
    assert task._interrupt.fired is False
    assert task._interrupt.trigger.block_reason == "no_progress_within_commit_radius"

    arm.tcp = hover.copy()
    task._eval_step_avatar()
    assert task._interrupt.fired is True
    assert task._interrupt.trigger.trigger_step == 13
    assert 0.25 <= task._interrupt.trigger.trigger_progress < 0.5
    assert 0.22 < task._interrupt.trigger.trigger_dist_m <= 0.32

    # Early hover is held until min_policy_steps (10).
    task, arm = _interrupt_eval_task(tcp=reset_tcp)
    task._init_eval_interrupt_state()
    task.avatar = _fake_avatar([])
    arm.tcp = hover.copy()
    for _ in range(9):
        task._eval_step_avatar()
    assert task._interrupt.fired is False
    assert task._interrupt.trigger.block_reason == "min_policy_steps"
    task._eval_step_avatar()
    assert task._interrupt.fired is True
    assert task._interrupt.trigger.trigger_step == 10

    # The generic contract thresholds (0.22 m / 0.5) cannot fire on the hover.
    generic = ProgressTrigger(ProgressTriggerParams())
    generic.arm(reset_tcp, {0: food})
    for step in range(1, 40):
        generic.update(step, hover, {0: food})
    assert generic.fired is False


def test_interrupt_take_action_ticks_then_applies_policy_action_unmodified(monkeypatch):
    # No retreat substitution; the policy keeps control.
    task = PlaceFoodInSkilletInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("avatar",))
    action = np.array([0.3, 0.2, 0.1])

    def base_action(_self, act, action_type="qpos"):
        events.append(("robot", act, action_type))
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    assert task.take_action(action, action_type="ee") == "obs"
    assert [event[0] for event in events] == ["avatar", "robot"]
    assert events[-1][1] is action and events[-1][2] == "ee"
    assert not hasattr(task, "_apply_eval_robot_retreat")


def test_neutral_eval_tick_precedes_robot_action(monkeypatch):
    task = PlaceFoodInSkilletCopresence({"eval_mode": True})
    task.avatar = object()
    task._neutral_eval_trigger_step = 1
    task._neutral_eval_policy_step_count = 0
    task._neutral_avatar_started = False
    task._neutral_eval_avatar_failed = False
    events = []
    task._start_neutral_avatar_table_work = lambda **kwargs: events.append(
        ("avatar", kwargs),
    )

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = object()
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert events == [
        ("avatar", {"add_start_delay": False}),
        ("robot", action, "qpos_abs"),
    ]
    assert task._neutral_eval_policy_step_count == 1


@contextmanager
def _recording_scope(events):
    events.append("recording_enter")
    try:
        yield
    finally:
        events.append("recording_exit")


def _rollout_food(name, asset_id):
    return object(), SimpleNamespace(
        name=name,
        asset_id=asset_id,
        model_id=0,
        tcp_yaw_deg=0.0,
    )


def test_base_scripted_rollout_preserves_dynamic_pan_and_food_order():
    task = PlaceFoodInSkillet({"hard": True})
    first = _rollout_food("burger", _BURGER_ID)
    second = _rollout_food("bread", _BREAD_ID)
    task.food_actors = [first, second]
    events = []
    sim_steps = {"count": 0}
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.suppress_recording = lambda: _recording_scope(events)
    task.step_sim = lambda: sim_steps.__setitem__(
        "count", sim_steps["count"] + 1,
    )
    task._place_pan_on_cooktop = (
        lambda arm: events.append(("pan", arm)) or True
    )
    task._food_in_skillet = lambda actor, _spec: actor is first[0]
    task._pick_and_drop_food = lambda actor, _spec, arm: (
        events.append(("pick", "second" if actor is second[0] else "first", arm))
        or True
    )

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        "recording_enter",
        "recording_exit",
        ("pan", "right"),
        ("pick", "second", "right"),
    ]
    assert sim_steps["count"] == 240


def test_assist_scripted_rollout_starts_avatar_before_robot_food():
    task = PlaceFoodInSkilletContribution({})
    robot_food = _rollout_food("bread", _BREAD_ID)
    avatar_food = _rollout_food("burger", _BURGER_ID)
    task.food_actors = [robot_food, avatar_food]
    task.avatar = SimpleNamespace(spare=lambda: True)
    events = []
    sim_steps = {"count": 0}
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.suppress_recording = lambda: _recording_scope(events)
    task.step_sim = lambda: sim_steps.__setitem__(
        "count", sim_steps["count"] + 1,
    )
    task._place_pan_on_cooktop = (
        lambda arm: events.append(("pan", arm)) or True
    )
    task._choose_avatar_food = (
        lambda: events.append(("assign",)) or (1, [0])
    )
    task._start_avatar_food_place = lambda actor, _spec: events.append(
        ("avatar", "avatar_food" if actor is avatar_food[0] else "wrong"),
    )
    task._pick_and_drop_food = lambda actor, _spec, arm: (
        events.append(("robot", "robot_food" if actor is robot_food[0] else "wrong", arm))
        or True
    )
    # The full-state expert parks its arm before the human
    # starts (a failed park keeps the arm where it is).
    task._park_arm_for_human = lambda arm: events.append(("park", arm)) or False

    assert task.play_once() is True
    # Easy default (issue 10): the pan is pre-placed on the cooktop, so the
    # scripted rollout goes straight from the settle to food assignment.
    assert events == [
        ("boost", "right"),
        "recording_enter",
        "recording_exit",
        ("assign",),
        ("park", "right"),
        ("avatar", "avatar_food"),
        ("robot", "robot_food", "right"),
    ]
    assert sim_steps["count"] == 240


def test_interrupt_scripted_choreography_resumes_interrupted_food_last():
    task = PlaceFoodInSkilletInterruption({"hard": True})
    interrupted = _rollout_food("burger", _BURGER_ID)
    other = _rollout_food("bread", _BREAD_ID)
    task.food_actors = [interrupted, other]
    task._interrupted_idx = 0
    task.avatar = SimpleNamespace(spare=lambda: True)
    events = []
    sim_steps = {"count": 0}
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.suppress_recording = lambda: _recording_scope(events)
    task.step_sim = lambda: sim_steps.__setitem__(
        "count", sim_steps["count"] + 1,
    )
    task._place_pan_on_cooktop = (
        lambda arm: events.append(("pan", arm)) or True
    )
    task._move_above_food_intent = lambda actor, _spec, arm: events.append(
        ("intent", actor is interrupted[0], arm),
    )
    task._play_inspect_motion = lambda **kwargs: events.append(
        ("inspect", kwargs["attach_obj"] is interrupted[0].entity)
    )
    # The fake actor is a bare object; give the choreography an entity handle.
    interrupted = (
        SimpleNamespace(entity=object()),
        interrupted[1],
    )
    other = (SimpleNamespace(entity=object()), other[1])
    task.food_actors = [interrupted, other]
    task._retreat_safe = lambda arm: events.append(("retreat", arm))
    task._wait_until_avatar_attach = lambda: events.append(("wait_attach",))
    # The phase-4 loop skips foods already in the pan
    # (pre-placed).  Neither fake food is in the pan here.
    task._food_in_skillet = lambda _actor, _spec: False
    task._pick_and_drop_food = lambda actor, _spec, arm: (
        events.append(("pick", 0 if actor is interrupted[0] else 1, arm))
        or True
    )

    assert task.play_once() is True
    assert events == [
        ("boost", "right"),
        "recording_enter",
        "recording_exit",
        ("pan", "right"),
        ("intent", True, "right"),
        ("inspect", True),
        ("retreat", "right"),
        ("wait_attach",),
        ("pick", 1, "right"),
        ("pick", 0, "right"),
    ]
    assert sim_steps["count"] == 460


def test_assist_blind_order_uses_eval_seed_local_stream():
    task = PlaceFoodInSkilletContribution({"eval_seed": 7})
    task.food_actors = [
        _rollout_food("bread", _BREAD_ID),
        _rollout_food("burger", _BURGER_ID),
        _rollout_food("apple", _APPLE_ID),
    ]
    task.avatar = None
    task._boost_finger_pd = lambda _arm: None
    task.step_sim = lambda: None
    task._place_pan_on_cooktop = lambda _arm: True
    task._food_in_skillet = lambda _actor, _spec: False
    attempts = []
    task._pick_and_drop_food = lambda actor, _spec, _arm: (
        attempts.append(next(
            i for i, (candidate, _candidate_spec) in enumerate(task.food_actors)
            if candidate is actor
        ))
        or True
    )
    task.check_success = lambda: False

    expected = [int(i) for i in np.random.RandomState(50012).permutation(3)]
    np.random.seed(113)
    expected_global_tail = np.random.random_sample(8)
    np.random.seed(113)
    assert task.play_blind_once() is True
    actual_global_tail = np.random.random_sample(8)

    assert task._blind_robot_food_order == expected
    assert attempts == expected
    assert [item["idx"] for item in task._blind_robot_food_attempts] == expected
    np.testing.assert_array_equal(actual_global_tail, expected_global_tail)


@pytest.mark.parametrize(
    ("task_name", "task_cls", "expected_mode"),
    (
        ("place_food_in_skillet", PlaceFoodInSkillet, "play_once"),
        (
            "place_food_in_skillet_contribution",
            PlaceFoodInSkilletContribution,
            "play_blind_once",
        ),
        (
            "place_food_in_skillet_interruption",
            PlaceFoodInSkilletInterruption,
            "parent_play_once",
        ),
        (
            "place_food_in_skillet_copresence",
            PlaceFoodInSkilletCopresence,
            "play_once",
        ),
    ),
)
def test_expert_routing_surface(task_name, task_cls, expected_mode):
    mode, parent = policy_play_mode(task_name, task_cls, "expert")
    assert mode == expected_mode
    if expected_mode == "parent_play_once":
        core = getattr(base_module, "SkilletTaskCore", PlaceFoodInSkillet)
        assert parent is core.play_once
    else:
        assert parent is None
    # Interrupt rows run the full-state expert in eval mode under
    # the interrupt contract.
    expected_full = (
        "interrupt_full_state" if task_name.endswith("_interruption") else "play_once"
    )
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        expected_full,
        None,
    )


def _configure_success(task):
    inside = object()
    task.TABLE_TOP_Z = 0.75
    task.plan_success = True
    task._cooktop_xy = np.array([0.0, 0.0])
    task._pan_bowl_center = lambda: np.array([0.03, 0.04, 0.80])
    task.food_actors = [
        (inside, SimpleNamespace(asset_id=_BURGER_ID, model_id=5)),
    ]
    task._actor_center = lambda actor, _asset, _model: (
        np.array([0.04, 0.02, 0.80])
        if actor is inside
        else np.array([0.30, 0.30, 0.80])
    )


def test_success_surfaces_keep_variant_collision_gates():
    base = PlaceFoodInSkillet({})
    _configure_success(base)
    base.avatar_collided = True
    assert base.check_success() is True
    base.plan_success = False
    assert base.check_success() is False

    assist = PlaceFoodInSkilletContribution({})
    _configure_success(assist)
    assist.avatar_collided = True
    assert assist.check_success() is True

    interrupt = PlaceFoodInSkilletInterruption({})
    _configure_success(interrupt)
    interrupt.avatar_collided = False
    assert interrupt.check_success() is True
    interrupt.avatar_collided = True
    assert interrupt.check_success() is False

    neutral = PlaceFoodInSkilletCopresence({})
    _configure_success(neutral)
    neutral.avatar_collided = False
    assert neutral.check_success() is True
    neutral.avatar_collided = True
    assert neutral.check_success() is False


def test_assist_evaluate_preserves_compatibility_metrics(monkeypatch):
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": 1})
    task = PlaceFoodInSkilletContribution({"eval_mode": True})
    task._eval_assist_trigger_step = 31
    task._eval_assist_fired = True
    task._eval_assist_failed = False
    task._eval_policy_step_count = 9
    task._avatar_food_idx = 1
    task._blind_robot_food_order = [1, 0]
    task._blind_robot_food_attempts = [{"idx": 1}]

    assert task.evaluate() == {
        "base": 1,
        "eval_mode": True,
        "eval_assist_trigger_step": 31,
        "eval_assist_fired": True,
        "eval_assist_failed": False,
        "eval_policy_step_count": 9,
        "assist_avatar_food_idx": 1,
        "blind_robot_food_order": [1, 0],
        "blind_robot_food_attempts": [{"idx": 1}],
        "blind_robot_avatar_same_first": True,
        # Human contact during the human's pan drop is exempt (but recorded).
        "exempt_human_contact": {
            "any": False, "n_checks": 0, "first_sim_step": None,
            "drop_frame": None, "window_frames": [220, 120],
            "robot_still_rad_s": 0.05, "pairs": [],
        },
        # The human's post-place step back (none yet).
        "human_step_back": None,
    }


def test_interrupt_evaluate_reports_progress_trigger_metrics(monkeypatch):
    # Uniform ``interrupt`` block plus the legacy
    # ``pace_interrupt`` key surface (trigger_mode is always "progress").
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": 1})
    task, arm = _interrupt_eval_task()
    np.random.seed(0)
    task._init_eval_interrupt_state()
    task.avatar = _fake_avatar()
    task._pace_pan_ready = lambda: True
    out = task.evaluate()
    assert out["base"] == 1
    assert out["interrupt"]["fired"] is False
    assert out["interrupt"]["not_fired_reason"] == "min_policy_steps"
    assert out["pace_interrupt"]["trigger_mode"] == "progress"
    assert out["pace_interrupt"]["fired"] is False
    assert out["pace_interrupt"]["pan_ready"] is True

    for _ in range(20):
        task._eval_step_avatar()
    arm.tcp = np.array([0.40, 0.05, 0.95])
    task._eval_step_avatar()
    out = task.evaluate()
    assert out["interrupt"]["fired"] is True
    assert out["interrupt"]["trigger_step"] == 21
    assert out["interrupt"]["trigger_target"] == 0
    assert out["pace_interrupt"]["trigger_reason"] == "progress"
