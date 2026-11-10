"""Scene-free release-v1 contracts for the categorize task family.

The suite freezes seeded layouts and public behavior while enforcing sibling
scenarios over one family core. Neutral retains historical RNG cadence without
inheriting or executing interrupt behavior.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.interaction import (
    AssistEvalModeController,
    CategorizeInterruptController,
    CategorizeNeutralController,
    InteractionMode,
)
from envs.task_bases import categorize as core_module
from envs.task_bases.categorize import CategorizeTaskCore
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    resolve_task_class,
)
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence
from envs.tasks.registry import TaskRegistration, TaskVariant
from envs.utils import Pose


class _Entity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self.pos.copy()

    def set_pos(self, pos):
        self.pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self):
        return self.quat.copy()


class _Actor:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.entity = _Entity(pos, quat)


class _Avatar:
    frame_ratio = 1.0

    @staticmethod
    def spare():
        return True


def _round(values):
    return tuple(round(float(value), 12) for value in np.asarray(values).ravel())


def _targets(targets):
    return [
        (
            name,
            int(model_id),
            int(basket_idx),
            _round(actor.entity.get_pos()),
            _round(actor.entity.get_quat()),
        )
        for actor, name, model_id, basket_idx in targets
    ]


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _capture_layout(monkeypatch, task_cls, config, seed):
    task = task_cls(dict(config))
    task.scene = object()
    task.TABLE_TOP_Z = 0.75
    primitives = []

    def create_primitive(_scene, _shape, pose, **_kwargs):
        primitives.append(_round(pose.p))
        return _Entity(pose.p, getattr(pose, "q", (1.0, 0.0, 0.0, 0.0)))

    monkeypatch.setattr(core_module, "create_primitive", create_primitive)
    task._spawn_categorize_object = (
        lambda name, model_id, x, y, z, quat: _Actor((x, y, z), quat)
    )

    if task_cls is CategorizeObjectsCopresence:
        task._neutral_avatar_reset_seed = int(seed)
        task._ensure_neutral_avatar_job_is_target_distinct = lambda: None
        task._load_neutral_avatar_props = lambda: None
        task._load_neutral_avatar_region_debug_tiles = lambda: None

    np.random.seed(seed)
    task.load_actors()
    tail = _round(np.random.random_sample(8))
    snapshot = {
        "categories": [(name, int(model_id)) for name, model_id in task._cats],
        "baskets": [_round(pose.p) for pose in task.basket_poses],
        "robot": _targets(task._sort_targets),
        "avatar": _targets(task._avatar_sort_targets),
        "primitives": primitives,
    }
    if task_cls is CategorizeObjectsInterruption:
        snapshot["interrupted"] = (
            int(task._interrupted_idx),
            int(task._interrupted_pick_ordinal),
        )
    if task_cls is CategorizeObjectsCopresence:
        snapshot["neutral_motion"] = task._neutral_avatar_job["motion"]
    return task, snapshot, tail


# (class, config, seed, layout sha256, next eight global np.random values)
LAYOUT_CASES = (
    (
        # Human present: shared-pool layout (assist 50-50, 2 boxes/objects).
        CategorizeObjectsContribution, {"hard": True}, 0,
        "58cd2d3da2bb90d07aae763baadb5c19b6c25a0aa51bbf73770d1d62906670ca",
        (0.479977172375, 0.392784796101, 0.836078763537, 0.337396160417,
         0.648171872051, 0.368241539841, 0.957155158953, 0.140350780413),
    ),
    (
        CategorizeObjectsContribution, {"hard": True}, 8,
        "8cec7ebf28b42dd7bbc179d3f812b03ac8d0a5f7c94e646979eee6d2f20291e4",
        (0.555356473822, 0.543386017543, 0.760895575603, 0.712374574082,
         0.619682096103, 0.426091770476, 0.289075028018, 0.9738552412),
    ),
    (
        CategorizeObjectsContribution, {}, 0,
        "58cd2d3da2bb90d07aae763baadb5c19b6c25a0aa51bbf73770d1d62906670ca",
        (0.479977172375, 0.392784796101, 0.836078763537, 0.337396160417,
         0.648171872051, 0.368241539841, 0.957155158953, 0.140350780413),
    ),
    (
        CategorizeObjectsContribution, {"hard": True, "no_human": True}, 8,
        "1ef233643b14df86bdb8411c565f955eb366a83170e63766543f4c4a76c6e87c",
        (0.43247593279, 0.270145741591, 0.801055886326, 0.638238806123,
         0.068690133458, 0.603644476369, 0.795574558429, 0.032008156426),
    ),
    (
        CategorizeObjectsInterruption, {"hard": True}, 0,
        "3cc61f0eaa604176d08cf08466ccc45bcda90c80ac6d82acaed182b2a0153d7c",
        (0.392784796101, 0.836078763537, 0.337396160417, 0.648171872051,
         0.368241539841, 0.957155158953, 0.140350780413, 0.870087258358),
    ),
    (
        CategorizeObjectsInterruption, {"hard": True}, 8,
        "719434a2968a907449a1bc1ad1072385c088c3763a1d163896d5946b37772cc2",
        (0.701527914114, 0.402677369955, 0.818265979139, 0.021281320051,
         0.352811994322, 0.482124082549, 0.87481589119, 0.053810784361),
    ),
    (
        CategorizeObjectsInterruption, {}, 0,
        "e3d508632b9ff4628727868a54c6d8b41279e23e806ecfa8eba21aa84108e0bb",
        (0.963662760501, 0.383441518826, 0.791725038083, 0.528894919753,
         0.568044561094, 0.925596638293, 0.071036058198, 0.087129299702),
    ),
    (
        CategorizeObjectsCopresence, {"hard": True}, 0,
        "712d164ee9c997626f700c678b31d691cdfed708e431c675c3b3e6e7da10109b",
        (0.392784796101, 0.836078763537, 0.337396160417, 0.648171872051,
         0.368241539841, 0.957155158953, 0.140350780413, 0.870087258358),
    ),
    (
        CategorizeObjectsCopresence, {"hard": True}, 8,
        "a364f4e9d5475ee80413879ecb1085f8e86a49d1ef3cfd536cb6c32df762da2b",
        (0.402677369955, 0.818265979139, 0.021281320051, 0.352811994322,
         0.482124082549, 0.87481589119, 0.053810784361, 0.267227136089),
    ),
)


def test_public_names_instructions_registry_and_no_human_mapping():
    cooperative = TASK_DESCRIPTOR_MAP["categorize_objects_contribution"]
    interrupt = TASK_DESCRIPTOR_MAP["categorize_objects_interruption"]
    neutral = TASK_DESCRIPTOR_MAP["categorize_objects_copresence"]

    assert cooperative.task_cls is CategorizeObjectsContribution
    assert cooperative.variant is TaskVariant.CONTRIBUTION
    assert cooperative.variant_label == "cooperative"
    assert cooperative.registration is TaskRegistration.CANONICAL
    assert interrupt.task_cls is CategorizeObjectsInterruption
    assert interrupt.variant is TaskVariant.INTERRUPTION
    assert interrupt.registration is TaskRegistration.CANONICAL
    assert neutral.task_cls is CategorizeObjectsCopresence
    assert neutral.variant is TaskVariant.COPRESENCE
    assert neutral.registration is TaskRegistration.CANONICAL

    shared = "sort the objects into the matching baskets"
    assert CategorizeObjectsContribution.INSTRUCTION == shared
    assert CategorizeObjectsInterruption.INSTRUCTION == shared
    assert CategorizeObjectsCopresence.INSTRUCTION == (
        "sort the objects into matching baskets while the human works nearby "
        "on the same table"
    )
    assert TASK_MAP["categorize_objects_contribution"] is CategorizeObjectsContribution
    assert TASK_MAP["categorize_objects_copresence"] is CategorizeObjectsCopresence
    assert TASK_MAP["categorize_objects_interruption"] is CategorizeObjectsInterruption
    assert "categorize_interrupt" not in TASK_MAP
    assert TASK_ALIASES["categorize_interrupt"] is CategorizeObjectsInterruption
    assert resolve_task_class("categorize_objects_interruption") == (
        "categorize_objects_interruption",
        CategorizeObjectsInterruption,
    )
    with pytest.raises(ValueError, match=r"^Unknown task: categorize\."):
        resolve_task_class("categorize")

    assert NO_HUMAN_CANONICAL_TASK_MAP["categorize"] is CategorizeObjectsContribution
    for name in (
        "categorize_objects_contribution",
        "categorize_objects_interruption",
        "categorize_objects_copresence",
    ):
        assert NO_HUMAN_VARIANT_ALIASES[name] == "categorize"
        assert resolve_task_class(name, no_human=True) == (
            "categorize",
            CategorizeObjectsContribution,
        )
    assert resolve_task_class("categorize", no_human=True) == (
        "categorize",
        CategorizeObjectsContribution,
    )


def test_family_core_sibling_architecture_and_controller_modes():
    assert issubclass(CategorizeObjectsContribution, CategorizeTaskCore)
    assert issubclass(CategorizeObjectsInterruption, CategorizeTaskCore)
    assert issubclass(CategorizeObjectsCopresence, CategorizeTaskCore)
    assert not issubclass(CategorizeObjectsInterruption, CategorizeObjectsContribution)
    assert not issubclass(CategorizeObjectsCopresence, CategorizeObjectsContribution)
    assert not issubclass(CategorizeObjectsCopresence, CategorizeObjectsInterruption)

    cooperative = CategorizeObjectsContribution({}).interaction_controller
    interrupt = CategorizeObjectsInterruption({}).interaction_controller
    neutral = CategorizeObjectsCopresence({}).interaction_controller
    assert isinstance(cooperative, AssistEvalModeController)
    assert cooperative.mode is InteractionMode.ASSIST
    assert isinstance(interrupt, CategorizeInterruptController)
    assert interrupt.mode is InteractionMode.INTERRUPT
    assert isinstance(neutral, CategorizeNeutralController)
    assert neutral.mode is InteractionMode.NEUTRAL
    assert cooperative.handles_policy_ticks is True
    assert interrupt.handles_policy_ticks is True
    assert neutral.handles_policy_ticks is False


@pytest.mark.parametrize(
    ("task_cls", "config", "seed", "expected_digest", "expected_tail"),
    LAYOUT_CASES,
)
def test_seeded_layout_and_global_rng_tail(
    monkeypatch, task_cls, config, seed, expected_digest, expected_tail
):
    _task, snapshot, tail = _capture_layout(monkeypatch, task_cls, config, seed)
    assert _digest(snapshot) == expected_digest
    assert tail == expected_tail


@pytest.mark.parametrize(
    ("config", "group_sizes"),
    (
        # Human present: one shared pool of two objects (assist 50-50).
        ({"hard": True}, {"shared": 2}),
        ({}, {"shared": 2}),
        # No-human mapping keeps the legacy robot-only layout.
        ({"hard": True, "no_human": True}, {"robot": 3}),
    ),
)
def test_normal_simplified_and_no_human_target_grouping(
    monkeypatch, config, group_sizes
):
    task, _snapshot, _tail = _capture_layout(
        monkeypatch, CategorizeObjectsContribution, config, 17
    )
    assert task._avatar_sort_targets == []
    groups = task._success_target_groups()
    assert {side: len(targets) for side, targets in groups} == group_sizes


def test_cooperative_eval_reset_starts_human_once_without_global_rng():
    task = CategorizeObjectsContribution({"eval_mode": True})
    events = []
    task._open_basket_lids = lambda: events.append("open")
    task._episode_seed = 42
    task._episode_rngs = {}
    task.basket_poses = [Pose((-0.36, 0.15, 0.8)), Pose((0.36, 0.15, 0.8))]
    task._sort_targets = [
        (_Actor((x, 0.18, 0.8)), name, 0, index)
        for index, (x, name) in enumerate(((-0.1, "cube"), (0.1, "block")))
    ]
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task.avatar = SimpleNamespace(
        pick_and_place=lambda **kw: events.append(kw["attach_obj"]),
        spare=lambda: True,
    )

    np.random.seed(42)
    expected_tail = np.random.random_sample(8)
    np.random.seed(42)
    task._eval_at_reset()
    actual_tail = np.random.random_sample(8)

    np.testing.assert_array_equal(actual_tail, expected_tail)
    human = task.interaction_controller.human_target
    assert events == ["open", task._sort_targets[human][0]]
    task._eval_at_step(1)
    task._eval_at_step(2)
    assert len(events) == 2


def test_interrupt_eval_trigger_and_rng_tail():
    task = CategorizeObjectsInterruption({
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
    })
    np.random.seed(42)
    expected = int(np.random.randint(7, 12))
    expected_tail = np.random.random_sample(8)
    np.random.seed(42)
    task._eval_at_reset()
    actual_tail = np.random.random_sample(8)
    # The legacy trigger draw is burned (same bounds, same RNG
    # tail); the progress trigger decides when the human moves.
    assert task._eval_legacy_trigger_draw == expected
    assert task._eval_inspect_fired is False
    assert task._interrupt.fired is False
    np.testing.assert_array_equal(actual_tail, expected_tail)


@pytest.mark.parametrize(
    ("task_cls", "eval_mode", "expected"),
    (
        (CategorizeObjectsInterruption, False, ("base", "inspect", "calibrate")),
        (CategorizeObjectsInterruption, True, ("base", "interrupt_eval", "inspect", "calibrate")),
    ),
)
def test_interrupt_controller_reset_hook_order(
    monkeypatch, task_cls, eval_mode, expected
):
    config = {"eval_mode": eval_mode}
    task = task_cls(config)
    task.avatar = SimpleNamespace(frame_ratio=None)
    task._interrupted_idx = 1
    events = ["base"]
    task._eval_at_reset = lambda: events.append("interrupt_eval")
    task._resolve_inspect_motion = lambda: events.append("inspect")
    task._calibrate_avatar_for_inspect = (
        lambda _index: events.append("calibrate")
    )

    task.interaction_controller.after_base_reset_observation(
        task, observation="observation"
    )
    assert tuple(events) == expected
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []


@pytest.mark.parametrize("task_cls", (CategorizeObjectsContribution, CategorizeObjectsInterruption))
def test_eval_action_hook_precedes_base_robot_action(monkeypatch, task_cls):
    task = task_cls({"eval_mode": True})
    task.avatar = object()
    task._eval_policy_step_count = 0
    events = []
    task._eval_at_step = lambda step: events.append(("interaction", step))
    monkeypatch.setattr(
        BaseTask,
        "take_action",
        lambda _self, action, action_type="qpos":
        events.append(("robot", action, action_type)) or "observation",
    )
    action = object()
    assert task.advance_interaction_policy_step() is True
    assert BaseTask.take_action(
        task, action, action_type="qpos_abs"
    ) == "observation"
    assert events == [
        ("interaction", 1),
        ("robot", action, "qpos_abs"),
    ]


def test_neutral_action_runs_neutral_then_robot_without_interrupt(monkeypatch):
    task = CategorizeObjectsCopresence({"eval_mode": True})
    task.avatar = object()
    task._neutral_eval_trigger_step = 1
    task._neutral_eval_policy_step_count = 0
    task._neutral_avatar_started = False
    task._neutral_eval_avatar_failed = False
    task._eval_policy_step_count = 0
    events = []
    task._start_neutral_avatar_table_work = (
        lambda add_start_delay=False:
        events.append(("neutral", add_start_delay))
    )
    task._eval_at_step = lambda step: events.append(("interrupt", step))
    monkeypatch.setattr(
        BaseTask,
        "take_action",
        lambda _self, action, action_type="qpos":
        events.append(("robot", action_type)) or "observation",
    )

    assert task.take_action(object(), action_type="qpos_abs") == "observation"
    assert events == [
        ("neutral", False),
        ("robot", "qpos_abs"),
    ]


def _install_sort_stubs(task, events, successful=()):
    task._boost_arm_pd = lambda: events.append(("boost",))
    task._open_basket_lids = lambda: events.append(("open",))
    task._move_to_safe = lambda arm: events.append(("safe", arm))
    task._get_object_cross_section_radius = lambda _name, _model=0: 0.03
    task._get_grasp_close_target = lambda _name, _model=0: 0.02
    task._get_object_world_center = (
        lambda actor, _name, _model=0: actor.entity.get_pos()
    )

    def pick_and_place(pick, place, arm):
        events.append(("robot", pick.label, int(place.basket_idx), arm))
        pick.actor.entity.set_pos(place.pos)
        return pick.label in successful

    task.pick_and_place = pick_and_place
    task.step_sim = lambda: events.append(("sim",))


def test_cooperative_expert_sorts_the_object_the_human_did_not_choose():
    for human in (0, 1):
        task = CategorizeObjectsContribution({})
        task.avatar = _Avatar()
        task.basket_poses = [Pose((-0.36, 0.15, 0.806)), Pose((0.36, 0.15, 0.806))]
        task._sort_targets = [
            (_Actor((-0.1, 0.18, 0.77)), "seal", 0, 0),
            (_Actor((0.1, 0.18, 0.77)), "apple", 0, 1),
        ]
        events = []
        _install_sort_stubs(task, events, successful={"seal", "apple"})
        task.pick_and_place = None
        task._robot_sort = (
            lambda idx, _arm="right", targets=None:
            events.append(("robot", task._sort_targets[idx][1])) or True
        )
        controller = task.interaction_controller
        controller.human_started = True
        controller.human_target = human
        assert task.play_once() is True
        assert [event for event in events if event[0] == "robot"] == [
            ("robot", task._sort_targets[1 - human][1])
        ]
        assert sum(event[0] == "sim" for event in events) == 200

    assert "play_blind_once" in CategorizeObjectsCopresence.__dict__
    assert CategorizeObjectsCopresence.play_blind_once is not CategorizeObjectsInterruption.play_blind_once


def _metric_task(task_cls):
    task = task_cls({})
    task.plan_success = True
    task.avatar_collision_checker = None
    task.avatar_collided = False
    task.basket_poses = [Pose((0.0, 0.0, 0.8)), Pose((0.25, 0.0, 0.8))]
    task._sort_targets = [
        (_Actor((0.0, 0.0, 0.8)), "robot", 1, 0),
    ]
    task._avatar_sort_targets = [
        (_Actor((0.25, 0.0, 0.8)), "avatar", 2, 1),
    ]
    task._get_object_world_center = (
        lambda actor, _name, _model=0: actor.entity.get_pos()
    )
    task._actor_footprint_aabb = (
        lambda actor, _name, _model=0:
        (actor.entity.get_pos()[:2] - 0.01, actor.entity.get_pos()[:2] + 0.01)
    )
    return task


def test_family_metric_item_shapes_and_success_gates():
    cooperative = _metric_task(CategorizeObjectsContribution)
    metrics = cooperative._basket_metrics()
    assert set(metrics) == {
        "basket_center_threshold",
        "basket_footprint_half_xy",
        "basket_all_correct",
        "basket_objects",
        "basket_scoring_version",
    }
    assert [item["side"] for item in metrics["basket_objects"]] == [
        "robot", "avatar"
    ]
    assert set(metrics["basket_objects"][0]) == {
        "side", "object", "model_id", "basket_idx", "object_pos",
        "basket_pos", "dist_xy", "rel_aabb_lo_xy", "rel_aabb_hi_xy",
        "center_success", "footprint_success", "resting_success", "success",
    }
    assert cooperative.check_success() is True
    cooperative.plan_success = False
    assert cooperative.check_success() is False
    cooperative.plan_success = True
    cooperative.avatar_collision_checker = object()
    cooperative.avatar_collided = True
    cooperative.avatar_collision_summary = lambda: {"any_collision": True}
    assert cooperative.check_success() is False

    interrupt = _metric_task(CategorizeObjectsInterruption)
    interrupt._avatar_sort_targets = []
    interrupt_metrics = interrupt._basket_metrics()
    assert "side" not in interrupt_metrics["basket_objects"][0]
    assert interrupt.check_success() is True

    neutral = _metric_task(CategorizeObjectsCopresence)
    neutral_metrics = neutral._robot_basket_metrics()
    assert neutral_metrics["basket_objects"][0]["side"] == "robot"
    assert neutral.check_success() is True


def test_evaluate_public_key_surfaces(monkeypatch):
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base_metric": True})

    cooperative = _metric_task(CategorizeObjectsContribution)
    cooperative.config["eval_mode"] = True
    cooperative._eval_trigger_step = 0
    cooperative._eval_policy_step_count = 3
    cooperative._eval_avatar_started = True
    cooperative._eval_avatar_target_idx = 1
    cooperative._eval_avatar_pick_order = [0]
    cooperative_metrics = cooperative.evaluate()
    assert set(cooperative_metrics) == {
        "base_metric", "basket_center_threshold", "basket_footprint_half_xy",
        "basket_all_correct", "basket_objects", "basket_scoring_version", "shared_pool",
        "eval_mode",
        "eval_trigger_step", "eval_policy_step_count", "eval_avatar_started",
        "eval_avatar_target_idx", "eval_avatar_order",
    }

    interrupt = _metric_task(CategorizeObjectsInterruption)
    interrupt.config["eval_mode"] = True
    interrupt._avatar_sort_targets = []
    interrupt._interrupted_idx = 0
    interrupt._interrupted_pick_ordinal = 0
    interrupt_metrics = interrupt.evaluate()
    assert {
        "interrupted_idx",
        "interrupted_pick_ordinal",
        "interrupted_object",
    } <= set(interrupt_metrics)
    assert interrupt_metrics["interrupted_object"] == "robot"
    assert interrupt_metrics["eval_avatar_started"] is False
    assert interrupt_metrics["eval_avatar_target_idx"] == 0
    assert interrupt_metrics["eval_avatar_order"] == []
    interrupt_keys = list(interrupt_metrics)
    assert interrupt_keys.index("eval_avatar_order") < interrupt_keys.index(
        "interrupted_idx"
    ) < interrupt_keys.index("eval_inspect_fired")

    neutral = _metric_task(CategorizeObjectsCopresence)
    neutral_metrics = neutral.evaluate()
    assert neutral_metrics["neutral_robot_only_grading"] is True
    assert "interrupted_idx" not in neutral_metrics
    assert "eval_inspect_fired" not in neutral_metrics

@pytest.mark.parametrize(
    ("randomize_inspect", "expected_tail"),
    (
        (
            False,
            (0.183434789866, 0.779691000273, 0.596850157946,
             0.445832752854, 0.099974915818, 0.459248891966,
             0.333708611139, 0.142866817922),
        ),
        (
            True,
            (0.596850157946, 0.445832752854, 0.099974915818,
             0.459248891966, 0.333708611139, 0.142866817922,
             0.650888472949, 0.056411579027),
        ),
    ),
)
def test_neutral_reset_legacy_draw_cadence_and_global_rng_tail(
    randomize_inspect, expected_tail
):
    """Burn legacy draws without publishing or running interrupt behavior."""

    task = CategorizeObjectsCopresence({
        "eval_mode": True,
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
        "neutral_eval_trigger_step_min": 13,
        "neutral_eval_trigger_step_max": 17,
        "randomize_inspect": randomize_inspect,
    })
    task.avatar = SimpleNamespace(frame_ratio=None, motion_data={})
    task._interrupted_idx = 1

    np.random.seed(42)
    task.interaction_controller.after_base_reset_observation(
        task, observation="observation"
    )
    task._neutral_eval_trigger_step = int(np.random.randint(13, 18))
    tail = _round(np.random.random_sample(8))

    assert task._neutral_eval_trigger_step == 17
    assert not hasattr(task, "_eval_trigger_step")
    assert not hasattr(task, "INSPECT_MOTION_NAME")
    assert not hasattr(task, "_calibrate_avatar_for_inspect")
    assert tail == expected_tail

def test_neutral_column_layout_robot_pd_avatar_and_camera_profile(monkeypatch):
    task = CategorizeObjectsCopresence({})

    table_calls = []
    task._create_rectangular_table = lambda **kwargs: table_calls.append(kwargs)
    task._create_table(table_height=0.74)
    assert len(table_calls) == 1
    assert table_calls[0]["table_height"] == 0.74
    assert tuple(table_calls[0]["half_size"]) == (0.80, 0.50)
    np.testing.assert_array_equal(
        table_calls[0]["center_xy"],
        np.array([0.0, -0.15]),
    )

    robot_loads = []
    monkeypatch.setattr(
        BaseTask,
        "_load_robot",
        lambda _self: robot_loads.append(dict(_self.config["robot_kwargs"])),
    )
    task._load_robot()
    assert robot_loads == [{
        "pos": [0.0, -0.5, 0.75],
        "quat": [0.707, 0.0, 0.0, 0.707],
    }]

    class _PdEntity:
        def __init__(self):
            self.kp = np.zeros(9, dtype=np.float64)
            self.kv = np.zeros(9, dtype=np.float64)

        def get_dofs_kp(self):
            return self.kp.copy()

        def get_dofs_kv(self):
            return self.kv.copy()

        def set_dofs_kp(self, values):
            self.kp = np.asarray(values, dtype=np.float64).copy()

        def set_dofs_kv(self, values):
            self.kv = np.asarray(values, dtype=np.float64).copy()

    entity = _PdEntity()
    joint = lambda index: SimpleNamespace(dof_idx_local=index)
    arm = SimpleNamespace(
        entity=entity,
        arm_joints=[joint(0), joint(1)],
        finger_joints=[joint(7), joint(8)],
    )
    task.robot = SimpleNamespace(get_arm=lambda _name: arm)
    task._boost_arm_pd()
    np.testing.assert_array_equal(entity.kp[[0, 1]], [20000.0, 20000.0])
    np.testing.assert_array_equal(entity.kv[[0, 1]], [1000.0, 1000.0])
    np.testing.assert_array_equal(entity.kp[[7, 8]], [2500.0, 2500.0])
    np.testing.assert_array_equal(entity.kv[[7, 8]], [200.0, 200.0])

    assert task._USE_OBSTACLE_INFLATION is True
    np.testing.assert_array_equal(task.avatar_init_pos, [0.08, 0.58, -0.18])
    assert task.static_camera_list == [{
        "name": "head_camera",
        "position": [0.0, -0.145, 2.85],
        "forward": [0.0, -0.005, -1.35],
    }]
    assert task.recording_camera_pos == [1.7, 0.05, 1.85]
    assert task.recording_camera_lookat == [0.0, 0.0, 1.55]
    assert task.side_camera_pos == [0.0, -0.25, 2.7]
    assert task.side_camera_lookat == [0.0, -0.25, 1.30]
