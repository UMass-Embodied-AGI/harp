"""Scene-free characterization for the place-dual-shoes task family.

These tests intentionally describe the release behavior before the family is
moved onto a shared core/controller architecture. They use small fakes only;
no Genesis scene, renderer, planner, robot, or GPU is created.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import zlib

import numpy as np
import pytest

from envs.avatar.inspect_motion_mixin import InspectMotionMixin
from envs.base_task import BaseTask
from envs.interaction.eval_mode import DualShoesAssistController
from envs.interaction.inspect_motion import DualShoesInterruptController
from envs.manipulation import TopDownPickPlaceMixin
from envs.task_bases.place_dual_shoes_motion import ShoeNativeMotionMixin
import envs.task_bases.place_dual_shoes as base_module
import envs.task_bases.place_dual_shoes_motion as motion_module
from envs.task_bases.place_dual_shoes import (
    DualShoesTaskCore,
    PlaceDualShoes,
)
from envs.tasks.place_pair_of_shoes_contribution import PlacePairOfShoesContribution
from envs.tasks._inspect_motion_interrupt_facade import InspectMotionInterruptFacade
from envs.tasks.place_pair_of_shoes_interruption import PlacePairOfShoesInterruption
from envs.tasks.place_pair_of_shoes_copresence import PlacePairOfShoesCopresence
from scripts.eval_expert_set import policy_play_mode


ALL_CLASSES = (
    PlaceDualShoes,
    PlacePairOfShoesContribution,
    PlacePairOfShoesInterruption,
    PlacePairOfShoesCopresence,
)


@pytest.mark.parametrize(
    ("task_cls", "module", "instruction", "uses_avatar"),
    (
        (
            PlaceDualShoes,
            "envs.task_bases.place_dual_shoes",
            "put the shoes in the shoebox",
            False,
        ),
        (
            PlacePairOfShoesContribution,
            "envs.tasks.place_pair_of_shoes_contribution",
            "the human will put one shoe in the shoebox; the robot puts the "
            "other shoe in the shoebox — work in parallel",
            True,
        ),
        (
            PlacePairOfShoesInterruption,
            "envs.tasks.place_pair_of_shoes_interruption",
            "put the shoes in the shoebox; the human will inspect one shoe "
            "midway through - wait, then continue",
            True,
        ),
        (
            PlacePairOfShoesCopresence,
            "envs.tasks.place_pair_of_shoes_copresence",
            "place both shoes in the shoebox while the human works nearby on "
            "the same table",
            True,
        ),
    ),
)
def test_public_identity_and_surface(task_cls, module, instruction, uses_avatar):
    assert task_cls.__module__ == module
    assert task_cls.__name__.startswith(("PlaceDualShoes", "PlacePairOfShoes"))
    assert issubclass(task_cls, BaseTask)
    assert task_cls.INSTRUCTION == instruction
    assert task_cls.OBJECT_SET == ["041_shoe"]
    assert bool(task_cls.use_avatar) is uses_avatar

    signature = inspect.signature(task_cls)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method in (
        "_create_table",
        "_sample_layout_positions",
        "load_actors",
        "play_once",
        "check_success",
        "evaluate",
        "take_action",
    ):
        assert callable(getattr(task_cls, method))


def test_release_mro_and_method_ownership():
    assert DualShoesTaskCore.__bases__ == (ShoeNativeMotionMixin, TopDownPickPlaceMixin, BaseTask)
    assert PlaceDualShoes.__bases__ == (DualShoesTaskCore,)
    assert PlacePairOfShoesContribution.__bases__ == (DualShoesTaskCore,)
    assert PlacePairOfShoesInterruption.__bases__ == (
        InspectMotionInterruptFacade,
        DualShoesTaskCore,
    )
    assert PlacePairOfShoesCopresence.__bases__[1] is PlaceDualShoes
    assert all(
        issubclass(task_cls, DualShoesTaskCore)
        for task_cls in ALL_CLASSES
    )
    assert not issubclass(PlacePairOfShoesContribution, PlaceDualShoes)
    assert not issubclass(PlacePairOfShoesInterruption, PlaceDualShoes)
    assert InspectMotionMixin not in PlacePairOfShoesInterruption.__mro__
    assert isinstance(
        PlacePairOfShoesContribution({}).interaction_controller,
        DualShoesAssistController,
    )
    assert isinstance(
        PlacePairOfShoesInterruption({}).interaction_controller,
        DualShoesInterruptController,
    )

    assert "_sample_layout_positions" in DualShoesTaskCore.__dict__
    assert "_sample_layout_positions" not in PlaceDualShoes.__dict__
    assert "_sample_layout_positions" in PlacePairOfShoesContribution.__dict__
    assert "_sample_layout_positions" in PlacePairOfShoesInterruption.__dict__
    assert "_sample_layout_positions" not in PlacePairOfShoesCopresence.__dict__
    assert "_pick_and_place_shoe" in DualShoesTaskCore.__dict__
    assert "_pick_and_place_shoe" not in PlaceDualShoes.__dict__
    assert "_pick_and_place_shoe" not in PlacePairOfShoesContribution.__dict__
    assert "_pick_and_place_shoe" not in PlacePairOfShoesInterruption.__dict__
    assert "_pick_and_place_shoe" not in PlacePairOfShoesCopresence.__dict__


@pytest.mark.parametrize("task_cls", ALL_CLASSES)
def test_constructor_copies_config_and_keeps_franka_default(task_cls):
    config = {"renderer": "rasterizer"}
    original = dict(config)
    task = task_cls(config)

    assert config == original
    assert task.config["robot_type"] == "franka"
    assert task._target_entry.object_id == "041_shoe"
    assert task._target_entry.model_id == 0


def test_object_count_and_table_policies(monkeypatch):
    assert PlaceDualShoes({})._num_shoes == 1
    assert PlaceDualShoes({"hard": True})._num_shoes == 2
    assert PlacePairOfShoesInterruption({"difficulty": "hard"})._num_shoes == 2
    assert PlacePairOfShoesCopresence({})._num_shoes == 1
    assert not hasattr(PlacePairOfShoesContribution({"hard": True}), "_num_shoes")

    assert "_create_table" in DualShoesTaskCore.__dict__
    assert "_create_table" not in PlaceDualShoes.__dict__
    assert "_create_table" not in PlacePairOfShoesContribution.__dict__
    assert "_create_table" not in PlacePairOfShoesInterruption.__dict__
    assert "_create_table" not in PlacePairOfShoesCopresence.__dict__
    assert PlaceDualShoes.TABLE_HALF_SIZE == (0.6, 0.55)
    assert PlaceDualShoes.TABLE_CENTER_XY == (0.0, -0.35)
    assert PlaceDualShoes.TABLE_THICKNESS_VALUE == 0.05
    assert PlaceDualShoes.TABLE_LEG_XY == (
        (-0.55, -0.85),
        (0.55, -0.85),
        (-0.55, 0.15),
        (0.55, 0.15),
    )

    calls = []
    monkeypatch.setattr(
        BaseTask,
        "_create_rectangular_table",
        lambda _self, **kwargs: calls.append(kwargs),
    )
    PlaceDualShoes({})._create_table(table_height=0.713)
    assert calls == [{
        "table_height": 0.713,
        "half_size": (0.6, 0.55),
        "center_xy": (0.0, -0.35),
        "table_thickness": 0.05,
        "leg_radius": 0.025,
        "leg_xy": PlaceDualShoes.TABLE_LEG_XY,
        "remember_bounds_before_variant": False,
        "allow_variant": True,
    }]


_LAYOUT_CASES = (
    (
        PlaceDualShoes,
        False,
        0,
        [
            (-0.10498331015822743, -0.2924266797097482),
            (0.19545387416694232, -0.30327129519813306),
        ],
        {},
    ),
    (
        PlaceDualShoes,
        True,
        7,
        [
            (-0.15000000000000002, -0.29000000000000004),
            (0.0, -0.25),
            (0.15000000000000002, -0.21000000000000002),
        ],
        {},
    ),
    (
        PlacePairOfShoesContribution,
        False,
        0,
        [
            (-0.00613917787051782, -0.11142876824889775),
            (0.21365346229110346, -0.1365284307805333),
            (-0.21964688618759673, -0.04370051462209802),
        ],
        {},  # the human's shoe is a private coin (see the coin tests below)
    ),
    (
        PlacePairOfShoesContribution,
        False,
        7,
        [
            (-0.19547023604957073, -0.12980326479637255),
            (0.0045618230384994185, -0.04288518411168192),
            (0.17552298986187206, -0.14987159341945783),
        ],
        {},
    ),
    (
        PlacePairOfShoesInterruption,
        True,
        0,
        [(-0.18, -0.34), (0.16, 0.02), (-0.04, 0.04)],
        {"_interrupted_idx": 0},
    ),
    (
        PlacePairOfShoesInterruption,
        True,
        7,
        [(-0.18, -0.34), (0.16, 0.02), (-0.04, 0.04)],
        {"_interrupted_idx": 1},
    ),
)


@pytest.mark.parametrize(
    ("task_cls", "hard", "seed", "expected_layout", "expected_state"),
    _LAYOUT_CASES,
)
def test_seeded_layout_local_rng_handoff_and_global_tail(
    task_cls, hard, seed, expected_layout, expected_state,
):
    task = task_cls({"hard": hard})

    reference = np.random.RandomState(seed)
    expected_local_seed = int(reference.randint(0, 2**31 - 1))
    expected_global_tail = reference.random_sample(4)

    np.random.seed(seed)
    local_seed = int(np.random.randint(0, 2**31 - 1))
    actual_layout = task._sample_layout_positions(
        np.random.RandomState(local_seed),
    )
    actual_global_tail = np.random.random_sample(4)

    assert local_seed == expected_local_seed
    np.testing.assert_array_equal(actual_layout, expected_layout)
    np.testing.assert_array_equal(actual_global_tail, expected_global_tail)
    for name, expected in expected_state.items():
        assert getattr(task, name) == expected


def _coin_reference(seed):
    stream = zlib.crc32(b"interaction:place_dual_shoes.assist.v1:human_shoe") & 0xFFFFFFFF
    return int(np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, stream])).integers(2))


def _seed_episode(task, seed):
    task._episode_seed = int(seed)
    task._episode_rngs = {}
    task.interaction_controller.on_episode_seeded(task, int(seed))


@pytest.mark.parametrize("seed", (0, 7, 9000000, 9000001))
def test_assist_human_shoe_is_private_coin_independent_of_layout_and_global_rng(seed):
    task = PlacePairOfShoesContribution({})
    _seed_episode(task, seed)
    np.random.seed(123)
    state = np.random.get_state()
    task._assign_shoes([(0.0, -0.1), (0.2, -0.1), (-0.2, -0.1)])
    first = task._avatar_shoe_idx
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], state[1])
    assert after[2] == state[2]
    assert first == _coin_reference(seed)
    assert task._robot_shoe_idx == 1 - first
    # Mirrored layout, same episode: same coin (it ignores the shoe sides).
    task._assign_shoes([(0.0, -0.1), (-0.2, -0.1), (0.2, -0.1)])
    assert task._avatar_shoe_idx == first
    # A new episode with the same seed redraws the same coin.
    _seed_episode(task, seed)
    task._assign_shoes([(0.0, -0.1), (-0.2, -0.1), (0.2, -0.1)])
    assert task._avatar_shoe_idx == first


def test_assist_human_shoe_coin_is_balanced():
    task = PlacePairOfShoesContribution({})
    picks = []
    for seed in range(9000000, 9000400):
        _seed_episode(task, seed)
        task._assign_shoes([(0.0, -0.1), (0.2, -0.1), (-0.2, -0.1)])
        picks.append(task._avatar_shoe_idx)
    # 400 fair flips: mean within 4 sigma of 0.5.
    assert abs(float(np.mean(picks)) - 0.5) < 4 * 0.5 / np.sqrt(len(picks))


@pytest.mark.parametrize("seed", (0, 1, 7, 42))
def test_assist_blind_order_is_global_permutation_stream(seed):
    task = PlacePairOfShoesContribution({})
    task.shoes = [object(), object()]

    reference = np.random.RandomState(seed)
    expected_order = [int(i) for i in reference.permutation(2)]
    expected_tail = reference.random_sample(4)

    np.random.seed(seed)
    actual_order = task._sample_blind_robot_order()
    actual_tail = np.random.random_sample(4)

    assert actual_order == expected_order
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_assist_reset_orders_base_blind_sampling_then_avatar_start(monkeypatch):
    task = PlacePairOfShoesContribution({"eval_mode": True})
    task.avatar = object()
    events = []
    monkeypatch.setattr(
        BaseTask,
        "reset",
        lambda _self, seed=0: events.append(("base", seed)) or "obs",
    )
    task._sample_blind_robot_order = (
        lambda: events.append(("blind",)) or [1, 0]
    )
    task._start_avatar_assist = (
        lambda: events.append(("avatar",)) or True
    )

    observation = task.reset(seed=17)
    task.interaction_controller.after_base_reset_observation(
        task, observation=observation,
    )
    assert observation == "obs"
    assert events == [("base", 17), ("blind",), ("avatar",)]
    assert task._policy_step_count == 0
    assert task._blind_robot_order == [1, 0]
    assert task._blind_robot_first_idx == 1
    assert task._eval_trigger_step == 0
    assert task._eval_avatar_started is True
    assert task._eval_avatar_failed is False


def test_assist_avatar_tick_precedes_robot_action(monkeypatch):
    task = PlacePairOfShoesContribution({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("avatar",))

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    action = object()
    task.advance_interaction_policy_step()
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert events == [("avatar",), ("robot", action, "qpos_abs")]


@pytest.mark.parametrize(
    ("eval_mode", "expected_hook"),
    ((False, "calibrate"), (True, "eval")),
)
def test_interrupt_reset_orders_base_inspect_then_variant_setup(
    eval_mode, expected_hook, monkeypatch,
):
    task = PlacePairOfShoesInterruption({"eval_mode": eval_mode})
    task.avatar = SimpleNamespace(frame_ratio=None)
    events = []
    monkeypatch.setattr(
        BaseTask,
        "reset",
        lambda _self, seed=0: events.append(("base", seed)) or "obs",
    )
    task._resolve_inspect_motion = lambda: events.append(("inspect",))
    task._calibrate_avatar_for_inspect = lambda: events.append(("calibrate",))
    task._init_eval_interrupt_state = lambda: events.append(("eval",))

    observation = task.reset(seed=17)
    task.interaction_controller.after_base_reset_observation(
        task, observation=observation,
    )
    assert observation == "obs"
    assert events == [("base", 17), ("inspect",), (expected_hook,)]
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW


def test_interrupt_tick_precedes_unmodified_policy_action(monkeypatch):
    # The retreat/return override is gone.  take_action ticks the
    # interrupt, then applies the policy's own action and action type.
    task = PlacePairOfShoesInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("avatar",))

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", action, action_type))
        return "obs"

    monkeypatch.setattr(BaseTask, "take_action", base_action)
    policy_action = object()
    assert task.take_action(policy_action, action_type="ee") == "obs"
    assert [event[0] for event in events] == ["avatar", "robot"]
    assert events[-1][1] is policy_action
    assert events[-1][2] == "ee"
    assert task.eval_skip_policy_action() is False
    assert task.consume_eval_policy_reset_requested() is False


@pytest.mark.parametrize(
    ("task_name", "task_cls", "expected_mode", "expected_parent"),
    (
        ("place_dual_shoes", PlaceDualShoes, "play_once", None),
        (
            "place_pair_of_shoes_contribution",
            PlacePairOfShoesContribution,
            "play_blind_once",
            None,
        ),
        (
            "place_pair_of_shoes_interruption",
            PlacePairOfShoesInterruption,
            "parent_play_once",
            DualShoesTaskCore.play_once,
        ),
        (
            "place_pair_of_shoes_copresence",
            PlacePairOfShoesCopresence,
            "play_once",
            None,
        ),
    ),
)
def test_expert_routing_surface(
    task_name, task_cls, expected_mode, expected_parent,
):
    mode, parent = policy_play_mode(task_name, task_cls, "expert")
    assert mode == expected_mode
    assert parent is expected_parent
    # The interrupt full-state expert runs in eval mode under the
    # interrupt contract (yields on the fired flag).
    expected_full_state = (
        ("interrupt_full_state", None)
        if task_name.endswith("_interruption")
        else ("play_once", None)
    )
    assert policy_play_mode(task_name, task_cls, "expert_full_state") == (
        expected_full_state
    )


class _FakeEntity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return self._quat.copy()


def _configure_scoring(task):
    task.TABLE_TOP_Z = 0.75
    task.plan_success = True
    task.shoebox = SimpleNamespace(entity=_FakeEntity([0.0, 0.0, 0.75]))
    task.shoes = [
        SimpleNamespace(entity=_FakeEntity([0.02, 0.03, 0.76])),
        SimpleNamespace(entity=_FakeEntity([-0.04, 0.01, 0.78])),
    ]


def test_success_collision_and_interrupt_lifecycle_gates():
    base = PlaceDualShoes({"hard": True})
    _configure_scoring(base)
    base.avatar_collided = True
    assert base.check_success() is True
    base.plan_success = False
    assert base.check_success() is False

    assist = PlacePairOfShoesContribution({})
    _configure_scoring(assist)
    assist.avatar_collided = True
    assist.avatar_collision_checker = None
    assert assist.check_success() is True
    assist.avatar_collision_checker = object()
    assert assist.check_success() is False
    assist.avatar_collided = False
    assert assist.check_success() is True

    # Success is gated while the human sequence is in flight
    # (glide/clip/idle return), not on retired retreat/return flags.
    from envs.interaction.interrupt_progress import new_interrupt_episode

    interrupt = PlacePairOfShoesInterruption({"eval_mode": True, "hard": True})
    _configure_scoring(interrupt)
    interrupt.avatar_collided = False
    assert interrupt.check_success() is True
    episode = new_interrupt_episode(interrupt)
    episode.human.phase = "clip"
    assert interrupt.check_success() is False
    episode.human.phase = "done"
    assert interrupt.check_success() is True
    interrupt.avatar_collided = True
    assert interrupt.check_success() is False


def test_scoring_helper_ownership_is_legacy_or_single_core_target():
    ownership = (
        "_shoe_in_box_metrics" in DualShoesTaskCore.__dict__,
        "_shoe_in_box_metrics" in PlacePairOfShoesContribution.__dict__,
        "_shoe_in_box_metrics" in PlacePairOfShoesInterruption.__dict__,
    )
    assert ownership in {
        # Release baseline: two value-for-value variant copies.
        (False, True, True),
        # Intended cleanup: one helper inherited from the family core.
        (True, False, False),
    }
    if ownership[0]:
        helper = DualShoesTaskCore.__dict__["_shoe_in_box_metrics"]
        assert PlacePairOfShoesContribution._shoe_in_box_metrics is helper
        assert PlacePairOfShoesInterruption._shoe_in_box_metrics is helper


@pytest.mark.parametrize(
    "task_cls", (PlacePairOfShoesContribution, PlacePairOfShoesInterruption),
)
def test_variant_shoe_metrics_surface(task_cls):
    task = task_cls({"eval_mode": True})
    _configure_scoring(task)
    metrics = task._shoe_in_box_metrics()

    assert list(metrics) == [
        "shoebox_pos",
        "shoebox_floor_z_threshold",
        "success_dist_xy_threshold",
        "shoe_a_pos",
        "shoe_a_dist_xy",
        "shoe_a_above_floor",
        "shoe_a_resting",
        "shoe_a_in_box",
        "shoe_b_pos",
        "shoe_b_dist_xy",
        "shoe_b_above_floor",
        "shoe_b_resting",
        "shoe_b_in_box",
        "all_shoes_in_box",
    ]
    assert metrics == {
        "shoebox_pos": [0.0, 0.0, 0.75],
        "shoebox_floor_z_threshold": 0.74,
        "success_dist_xy_threshold": 0.18,
        "shoe_a_pos": [0.02, 0.03, 0.76],
        "shoe_a_dist_xy": 0.03605551275463989,
        "shoe_a_above_floor": True,
        "shoe_a_resting": True,
        "shoe_a_in_box": True,
        "shoe_b_pos": [-0.04, 0.01, 0.78],
        "shoe_b_dist_xy": 0.04123105625617661,
        "shoe_b_above_floor": True,
        "shoe_b_resting": True,
        "shoe_b_in_box": True,
        "all_shoes_in_box": True,
    }


@pytest.mark.parametrize(
    "task_cls", (PlacePairOfShoesContribution, PlacePairOfShoesInterruption),
)
def test_variant_shoe_metrics_empty_scene_is_exactly_empty(task_cls):
    task = task_cls({})
    task.shoes = []
    task.shoebox = None
    assert task._shoe_in_box_metrics() == {}


def test_assist_success_preserves_collision_gate_then_core_super(monkeypatch):
    target_core_helper = "_shoe_in_box_metrics" in DualShoesTaskCore.__dict__
    calls = []

    def core_success(_task):
        calls.append("core")
        return True

    monkeypatch.setattr(DualShoesTaskCore, "check_success", core_success)
    task = PlacePairOfShoesContribution({})
    _configure_scoring(task)

    task.avatar_collided = True
    task.avatar_collision_checker = None
    assert task.check_success() is True
    assert calls == (["core"] if target_core_helper else [])

    task.avatar_collision_checker = object()
    assert task.check_success() is False
    assert calls == (["core"] if target_core_helper else [])

    task.avatar_collided = False
    assert task.check_success() is True
    assert calls == (["core", "core"] if target_core_helper else [])


def test_base_evaluate_does_not_gain_variant_shoe_metrics(monkeypatch):
    # The extraction adds a private scoring helper to the family core; it must
    # not make the robot-only base publish variant-only metric keys.
    assert "evaluate" not in DualShoesTaskCore.__dict__
    monkeypatch.setattr(BaseTask, "evaluate", lambda _task: {"base": 1})

    task = PlaceDualShoes({"hard": True})
    _configure_scoring(task)
    assert task.evaluate() == {"base": 1}


def test_assist_evaluate_adds_assignment_blind_and_eval_metrics(monkeypatch):
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": 1})
    task = PlacePairOfShoesContribution({"eval_mode": True})
    task._shoe_in_box_metrics = lambda: {"all_shoes_in_box": True}
    task._avatar_shoe_idx = 1
    task._robot_shoe_idx = 0
    task._blind_robot_order = [1, 0]
    task._blind_robot_first_idx = 1
    task._eval_trigger_step = 0
    task._policy_step_count = 9
    task._eval_avatar_started = True
    task._eval_avatar_failed = False

    assert task.evaluate() == {
        "base": 1,
        "all_shoes_in_box": True,
        "assist_avatar_shoe_idx": 1,
        "assist_robot_shoe_idx": 0,
        "blind_robot_order": [1, 0],
        "blind_robot_first_idx": 1,
        "blind_same_first_as_avatar": True,
        "eval_mode": True,
        "eval_trigger_step": 0,
        "eval_policy_step_count": 9,
        "eval_avatar_started": True,
        "eval_avatar_failed": False,
    }


class _FakeArm:
    tcp_offset = object()
    entity = object()
    _finger_dof_indices = (7, 8)

    def get_ee_pose(self):
        return np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])


class _FakeRobot:
    def __init__(self):
        self.arm = _FakeArm()

    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return self.arm


class _FakeLinkPose:
    def __init__(self, value):
        self.value = value

    def to_pose7(self):
        return np.asarray(self.value, dtype=np.float64)


@pytest.mark.parametrize(
    ("task_cls", "module", "expected_steps", "expected_forces"),
    (
        (
            PlaceDualShoes,
            base_module,
            420,
            [(-35.0, 35.0), (-100.0, 100.0)],
        ),
        (
            PlacePairOfShoesContribution,
            base_module,
            400,
            [(-35.0, 35.0)],
        ),
    ),
)
def test_release_force_and_settle_timing_contract(
    task_cls, module, expected_steps, expected_forces, monkeypatch,
):
    monkeypatch.delenv("SHOE_DEBUG", raising=False)
    task = task_cls({})
    task.robot = _FakeRobot()
    task.shoebox = SimpleNamespace(
        entity=_FakeEntity([0.0, 0.0, 0.75]),
    )
    task.shoebox.entity.get_AABB = lambda: np.array([[-.1,-.1,.75],[.1,.1,.82]])
    task.TABLE_TOP_Z = 0.75
    task._box_rim_z = 0.82
    task.plan_success = True
    events = []
    forces = []

    task.step_sim = lambda: events.append(("sim",))
    task.open_gripper = lambda arm, num_steps=None: events.append(
        ("open", arm, num_steps),
    )
    task._topdown_pick_shoe = lambda *_args: True
    task._shoe_motion_scene = lambda *_a: nullcontext()
    task._shoe_carry_quat = np.array([1.,0,0,0])
    task._shoe_move = lambda pose, arm, label, **_kw: events.append(("native",label)) or pose
    task._shoe_arrived = lambda pose, *_a: pose
    task._shoe_retained = lambda *_a: None
    task._shoe_anchor_world = lambda _shoe: np.array([.3,0,.8])
    task._shoe_world_aabb = lambda _shoe: (
        np.array([0.0, 0.0, 0.76]),
        np.array([0.1, 0.1, 0.90]),
        np.array([0.05, 0.05, 0.83]),
    )
    monkeypatch.setattr(
        module,
        "update_dofs_kp_kv_compat",
        lambda *_args, **_kwargs: events.append(("pd",)),
    )
    monkeypatch.setattr(
        module,
        "update_dofs_force_range_compat",
        lambda *_args, **kwargs: forces.append((
            kwargs["lower_value"], kwargs["upper_value"],
        )),
    )
    monkeypatch.setattr(motion_module, "update_dofs_force_range_compat", module.update_dofs_force_range_compat)

    assert task._pick_and_place_shoe(
        SimpleNamespace(), "right", "A", slot_idx=0,
    ) is True
    assert sum(event[0] == "sim" for event in events) == expected_steps
    assert [event for event in events if event[0] == "open"] == [
        ("open", "right", None),
        ("open", "right", 260),
    ]
    assert [event[1] for event in events if event[0] == "native"] == ["carry","descent","release retreat"]
    assert forces == expected_forces
