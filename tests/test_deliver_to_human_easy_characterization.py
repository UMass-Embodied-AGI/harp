"""Scene-free release characterization for ``deliver_to_human_easy``.

The task currently receives its eval-mode lifecycle from
``EvalModeAvatarMixin``.  These contracts also admit exactly one intended
replacement: an ``IntentEvalModeController`` driven through ``BaseTask``'s
post-observation and pre-action interaction hooks.  No test in this module
constructs a Genesis scene, robot, planner, renderer, or avatar asset.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.eval_mode_mixin import EvalModeAvatarMixin
from envs.base_task import BaseTask, TargetSpec
import envs.interaction as interaction_api
from envs.interaction import InteractionMode, LegacyInteractionController
import envs.interaction.eval_mode as interaction_eval
from envs.manipulation import TopDownPickPlaceMixin
from envs.scene_init import apply_scene_init_post_settle, capture_scene_init
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
import envs.tasks.deliver_to_human as deliver_module
from envs.tasks.deliver_to_human import DeliverToHuman
from scripts.eval_expert_set import policy_play_mode


TASK_NAME = "deliver_to_human"
INTENT_CONTROLLER = getattr(interaction_eval, "IntentEvalModeController", None)


def test_public_identity_signature_and_task_surface():
    assert DeliverToHuman.__module__ == "envs.tasks.deliver_to_human"
    assert DeliverToHuman.__name__ == "DeliverToHuman"
    assert issubclass(DeliverToHuman, BaseTask)
    assert DeliverToHuman.INSTRUCTION == (
        "pick up the {object} from the table and deliver it to the human"
    )
    # The apple is not in human_transfer_pool; the representative
    # first-pool-object is the rubik's cube.
    assert DeliverToHuman.default_instruction() == (
        "pick up the rubik's cube from the table and deliver it to the human"
    )
    assert DeliverToHuman.OBJECT_SET == "human_transfer_pool"
    assert DeliverToHuman.use_avatar is True

    signature = inspect.signature(DeliverToHuman)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is inspect.Parameter.empty
    for method_name in (
        "load_actors",
        "play_once",
        "check_success",
        "evaluate",
        "reset",
        "take_action",
        "eval_pre_policy_warmup",
        "_eval_at_reset",
        "_play_once_annotated",
        "_play_once_primitive",
    ):
        assert callable(getattr(DeliverToHuman, method_name))


def test_constructor_requires_config_copies_it_and_keeps_compatibility_fields():
    with pytest.raises(TypeError):
        DeliverToHuman()

    config = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    original = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    task = DeliverToHuman(config)

    assert config == original
    assert task.config is not config
    assert task.obj_name is None
    assert task.obj_model_id == task.DEFAULT_OBJ_MODEL_ID == 1
    assert task.random_object is True
    assert task.grasp_method_override is None
    assert task.instruction == DeliverToHuman.INSTRUCTION


def test_only_legacy_mixin_or_intended_intent_controller_graph_is_allowed():
    task = DeliverToHuman({})

    if INTENT_CONTROLLER is None:
        assert DeliverToHuman.__bases__ == (
            EvalModeAvatarMixin,
            TopDownPickPlaceMixin,
            BaseTask,
        )
        assert isinstance(task.interaction_controller, LegacyInteractionController)
        assert task.interaction_controller.mode is InteractionMode.LEGACY
        assert DeliverToHuman.__dict__.get("_eval_at_step") is None
        assert task._eval_at_step(19) is None
        return

    assert getattr(interaction_api, "IntentEvalModeController", None) is INTENT_CONTROLLER
    assert DeliverToHuman.__bases__ == (TopDownPickPlaceMixin, BaseTask)
    assert EvalModeAvatarMixin not in DeliverToHuman.__mro__
    controller = task.interaction_controller
    assert isinstance(controller, INTENT_CONTROLLER)
    assert controller.mode is InteractionMode.INTENT
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False
    # The blind-intent window adds a policy-step hook; with no trigger drawn
    # it is the legacy no-op.
    assert task._eval_at_step(19) is None


def test_registry_intent_identity_and_no_human_rejection_are_stable():
    descriptor = TASK_DESCRIPTOR_MAP[TASK_NAME]
    assert descriptor.task_cls is DeliverToHuman
    assert descriptor.family == "deliver_to_human"
    assert descriptor.variant is TaskVariant.INTENT
    assert descriptor.registration is TaskRegistration.CANONICAL
    assert descriptor.variant_label is None
    assert descriptor.no_human_name is None
    assert descriptor.no_human_canonical is False
    assert descriptor.instruction_template == DeliverToHuman.INSTRUCTION
    assert descriptor.instruction == DeliverToHuman.default_instruction()

    assert TASK_MAP[TASK_NAME] is DeliverToHuman
    assert resolve_task_class(TASK_NAME) == (TASK_NAME, DeliverToHuman)
    assert TASK_NAME not in NO_HUMAN_CANONICAL_TASK_MAP
    assert TASK_NAME not in NO_HUMAN_VARIANT_ALIASES
    assert TASK_NAME not in NO_HUMAN_TASK_MAP
    with pytest.raises(ValueError, match="has no no-human variant"):
        resolve_task_class(TASK_NAME, no_human=True)


class _FakeActor:
    def __init__(self, pos, quat):
        self.entity = self
        self._pose = SimpleNamespace(
            p=np.asarray(pos, dtype=np.float64),
            q=np.asarray(quat, dtype=np.float64),
        )

    def get_pose(self):
        return self._pose


def _capture_layout(monkeypatch, config, seed):
    events = []

    def read_quat(object_name, robot_type):
        events.append(("read_quat", object_name, robot_type))
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    def load_object(_scene, pose, object_name, **kwargs):
        events.append((
            "load_object",
            object_name,
            int(kwargs["model_id"]),
            tuple(float(value) for value in pose.p),
            tuple(float(value) for value in pose.q),
            bool(kwargs["convex"]),
            bool(kwargs["is_static"]),
        ))
        return _FakeActor(pose.p, pose.q)

    monkeypatch.setattr(deliver_module, "read_save_object_quat", read_quat)
    monkeypatch.setattr(deliver_module, "load_object", load_object)

    task = DeliverToHuman(dict(config))
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    np.random.seed(seed)
    task.load_actors()
    tail = np.random.random_sample(8)
    return task, events, tail


@pytest.mark.parametrize(
    (
        "config",
        "seed",
        "expected_name",
        "expected_model",
        "expected_instruction",
        "expected_pos",
        "expected_quat",
        "expected_quat_read",
        "expected_tail",
    ),
    (
        # 035_apple is not in human_transfer_pool (it drops mid-carry),
        # so every pool draw resolves to the rubik's cube and the
        # RNG stream follows a 1-element pool.
        (
            {},
            0,
            "073_rubikscube",
            1,
            "pick up the rubik's cube from the table and deliver it to the human",
            (0.03720337598183118, -0.07848106336275805, 0.785),
            (1.0, 0.0, 0.0, 0.0),
            False,
            (
                0.6027633760716439,
                0.5448831829968969,
                0.4236547993389047,
                0.6458941130666561,
                0.4375872112626925,
                0.8917730007820798,
                0.9636627605010293,
                0.3834415188257777,
            ),
        ),
        (
            {},
            1,
            "073_rubikscube",
            1,
            "pick up the rubik's cube from the table and deliver it to the human",
            (0.004255501175643495, -0.07796755065578419, 0.785),
            (1.0, 0.0, 0.0, 0.0),
            False,
            (
                0.00011437481734488664,
                0.30233257263183977,
                0.14675589081711304,
                0.0923385947687978,
                0.1862602113776709,
                0.34556072704304774,
                0.39676747423066994,
                0.538816734003357,
            ),
        ),
        (
            {"random_object": False},
            42,
            "073_rubikscube",
            1,
            "pick up the rubik's cube from the table and deliver it to the human",
            (-0.006364970288159383, -0.05492856935900839, 0.785),
            (1.0, 0.0, 0.0, 0.0),
            False,
            (
                0.7319939418114051,
                0.5986584841970366,
                0.15601864044243652,
                0.15599452033620265,
                0.05808361216819946,
                0.8661761457749352,
                0.6011150117432088,
                0.7080725777960455,
            ),
        ),
        (
            {"object_name": "073_rubikscube"},
            42,
            "073_rubikscube",
            1,
            "pick up the rubik's cube from the table and deliver it to the human",
            (-0.006364970288159383, -0.05492856935900839, 0.785),
            (1.0, 0.0, 0.0, 0.0),
            False,
            (
                0.7319939418114051,
                0.5986584841970366,
                0.15601864044243652,
                0.15599452033620265,
                0.05808361216819946,
                0.8661761457749352,
                0.6011150117432088,
                0.7080725777960455,
            ),
        ),
        (
            {"object_name": "073_rubikscube", "model_id": 7},
            7,
            "073_rubikscube",
            7,
            "pick up the rubik's cube from the table and deliver it to the human",
            (-0.08092292765651071, -0.07200812077598853, 0.785),
            (1.0, 0.0, 0.0, 0.0),
            False,
            (
                0.4384092314408935,
                0.7234651778309412,
                0.9779895119966027,
                0.5384958704104337,
                0.5011204636599379,
                0.07205113335976154,
                0.26843898010187117,
                0.49988250082555996,
            ),
        ),
    ),
)
def test_seeded_actor_layout_target_and_global_numpy_tail(
    monkeypatch,
    config,
    seed,
    expected_name,
    expected_model,
    expected_instruction,
    expected_pos,
    expected_quat,
    expected_quat_read,
    expected_tail,
):
    task, events, tail = _capture_layout(monkeypatch, config, seed)

    assert task.obj_name == expected_name
    assert task.obj_model_id == expected_model
    assert task.instruction == expected_instruction
    assert task.target.object is task.obj_actor
    assert task.target.label == "right_palm"
    assert task.target.success_dist_3d == task.SUCCESS_DIST_3D_M == 0.30
    assert task.target.success_dz_min == task.SUCCESS_DZ_MIN_M == -0.40
    assert task.target.success_require_no_avatar_collision is True

    read_events = [event for event in events if event[0] == "read_quat"]
    assert bool(read_events) is expected_quat_read
    if expected_quat_read:
        assert read_events == [("read_quat", expected_name, "franka")]
    load_events = [event for event in events if event[0] == "load_object"]
    assert len(load_events) == 1
    event = load_events[0]
    assert event[1:3] == (expected_name, expected_model)
    np.testing.assert_allclose(event[3], expected_pos, rtol=0.0, atol=5e-15)
    np.testing.assert_allclose(event[4], expected_quat, rtol=0.0, atol=5e-15)
    assert event[5:] == (True, False)
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-15)


class _WarmupAvatar:
    def __init__(self, events):
        self.events = events
        self._spare = iter((False, True))

    def play_animation(self, name):
        self.events.append(("eval_reset_animation", name))

    def spare(self):
        return next(self._spare)


def _run_eval_reset_boundary(task, events, monkeypatch):
    """Run the legacy wrapper or its sole intended controller equivalent."""

    observation = {"phase": "initial"}
    if INTENT_CONTROLLER is None:
        monkeypatch.setattr(
            BaseTask,
            "reset",
            lambda _task, seed=0: events.append(
                ("initial_observation", seed)
            ) or observation,
        )
        return task.reset(seed=17)

    events.append(("initial_observation", 17))
    task.interaction_controller.after_base_reset_observation(
        task,
        observation=observation,
    )
    return observation


def test_eval_reset_runs_after_initial_observation_and_before_outer_warmup(
    monkeypatch,
):
    events = []
    task = DeliverToHuman({
        "eval_mode": True,
        "eval_avatar_pre_policy": True,
        "eval_avatar_post_warmup_settle_steps": 2,
    })
    task.avatar = _WarmupAvatar(events)
    task._eval_policy_step_count = 99
    task.step_sim = lambda: events.append(("warmup_sim_step",))
    task._clear_avatar_collision_state = lambda: events.append(
        ("warmup_clear_collision",)
    )
    task.get_obs = lambda: events.append(("warmup_observation",)) or {
        "phase": "warm"
    }

    initial = _run_eval_reset_boundary(task, events, monkeypatch)
    assert initial == {"phase": "initial"}
    assert task._eval_policy_step_count == 0
    warmed = task.eval_pre_policy_warmup()

    assert warmed == {"phase": "warm"}
    assert events == [
        ("initial_observation", 17),
        ("eval_reset_animation", "take_from_human"),
        ("warmup_sim_step",),
        ("warmup_sim_step",),
        ("warmup_sim_step",),
        ("warmup_clear_collision",),
        ("warmup_observation",),
    ]


@pytest.mark.parametrize(
    ("eval_mode", "has_avatar"),
    ((False, True), (True, False)),
)
def test_eval_reset_and_warmup_are_inactive_without_both_gates(
    eval_mode,
    has_avatar,
    monkeypatch,
):
    events = []
    task = DeliverToHuman({"eval_mode": eval_mode})
    task.avatar = SimpleNamespace(
        play_animation=lambda name: events.append(("play", name)),
        spare=lambda: True,
    ) if has_avatar else None
    task._eval_policy_step_count = 91
    task.step_sim = lambda: events.append(("step",))
    task._clear_avatar_collision_state = lambda: events.append(("clear",))
    task.get_obs = lambda: events.append(("obs",)) or "observation"

    assert _run_eval_reset_boundary(task, events, monkeypatch) == {
        "phase": "initial"
    }
    assert task._eval_policy_step_count == 91
    assert task.eval_pre_policy_warmup() == "observation"
    assert events == [("initial_observation", 17), ("obs",)]


def test_active_policy_counter_is_one_based_and_precedes_early_success_return():
    task = DeliverToHuman({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    task.take_action_cnt = 0
    task._eval_policy_step_count = 0
    events = []
    task._eval_at_step = lambda step: events.append(("eval_step", step))
    task.get_obs = lambda: events.append(("observation",)) or "obs"

    action = object()
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert events == [
        ("eval_step", 1),
        ("observation",),
        ("eval_step", 2),
        ("observation",),
    ]
    assert task._eval_policy_step_count == 2
    # BaseTask's success short-circuit precedes its robot-action counter, but
    # not the legacy/composed avatar policy tick.
    assert task.take_action_cnt == 0


def test_composed_active_policy_tick_safely_preserves_inherited_noop_semantics():
    if INTENT_CONTROLLER is None:
        pytest.skip("composed intent controller has not landed yet")

    task = DeliverToHuman({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    task.take_action_cnt = 0
    task._eval_policy_step_count = 0
    task.get_obs = lambda: "obs"

    assert task.take_action(object()) == "obs"
    assert task._eval_policy_step_count == 1
    assert task.take_action_cnt == 0


def test_inactive_policy_call_does_not_create_or_increment_eval_counter():
    task = DeliverToHuman({"eval_mode": True})
    task.avatar = None
    task.eval_success = True
    task.get_obs = lambda: "obs"
    task.__dict__.pop("_eval_policy_step_count", None)

    assert task.take_action(object()) == "obs"
    assert not hasattr(task, "_eval_policy_step_count")


def test_expert_routing_stays_on_the_public_scripted_rollout():
    assert policy_play_mode(TASK_NAME, DeliverToHuman, "expert") == (
        "play_blind_once",
        None,
    )
    assert policy_play_mode(
        TASK_NAME,
        DeliverToHuman,
        "expert_full_state",
    ) == ("play_full_eval_once", None)


class _ScriptedAvatar:
    def __init__(self, events, active_steps):
        self.events = events
        self._states = iter([False] * active_steps + [True])

    def play_animation(self, name):
        self.events.append(("play", name))

    def spare(self):
        return next(self._states)


@pytest.mark.parametrize(
    ("method", "result"),
    (("annotated", True), ("primitive", False)),
)
def test_scripted_expert_presentation_settle_clear_and_method_dispatch(
    method,
    result,
):
    task = DeliverToHuman({"grasp_method": method})
    events = []
    task.avatar = _ScriptedAvatar(events, active_steps=2)
    steps = {"count": 0}
    task.step_sim = lambda: steps.__setitem__("count", steps["count"] + 1)
    task._clear_avatar_collision_state = lambda: events.append(
        ("clear", steps["count"])
    )
    task._play_once_annotated = lambda: events.append(
        ("annotated", steps["count"])
    ) or result
    task._play_once_primitive = lambda: events.append(
        ("primitive", steps["count"])
    ) or result

    assert task.play_once() is result
    assert steps["count"] == 52
    assert events == [
        ("play", "take_from_human"),
        ("clear", 52),
        (method, 52),
    ]


def test_scripted_expert_without_avatar_keeps_fifty_step_presentation_settle():
    task = DeliverToHuman({"grasp_method": "primitive"})
    task.avatar = None
    events = []
    task.step_sim = lambda: events.append("step")
    task._clear_avatar_collision_state = lambda: events.append("clear")
    task._play_once_primitive = lambda: events.append("primitive") or True

    assert task.play_once() is True
    assert events == [*(["step"] * 50), "clear", "primitive"]


def _configured_metric_task():
    task = DeliverToHuman({})
    task.obj_name = "035_apple"
    task.obj_model_id = 1
    task.plan_success = True
    task.target = TargetSpec(
        object=object(),
        position=np.array([0.2, 0.2, 1.0]),
        label="right_palm",
        object_pos_fn=lambda: np.array([0.1, 0.2, 0.9]),
        success_dist_3d=task.SUCCESS_DIST_3D_M,
        success_dz_min=task.SUCCESS_DZ_MIN_M,
        success_require_no_avatar_collision=True,
    )
    metrics = {
        "object_pos": np.array([0.1, 0.2, 0.9]),
        "target_pos": np.array([0.2, 0.2, 1.0]),
        "dist_xy": 0.1,
        "dist_3d": 0.1414213562373095,
        "dz": -0.1,
    }
    task._compute_target_metrics = lambda: metrics
    task._object_bottom_z = lambda: 0.84
    task._object_table_clearance = lambda: 0.075
    task._object_still_held = lambda: True
    task.avatar = SimpleNamespace(spare=lambda: True)   # hand presented
    task.avatar_collision_summary = lambda: {"any_collision": False}
    task.safe_distance_summary = lambda: {"enabled": False}
    task.avatar_safety_summary = lambda: {"enabled": False}
    task.avatar_safety_intervention_summary = lambda: {"enabled": False}
    return task, metrics


def test_success_uses_live_hold_collision_distance_and_height_gates():
    task, metrics = _configured_metric_task()
    assert task.check_success() is True

    task.plan_success = False
    assert task.check_success() is False
    task.plan_success = True

    task._object_still_held = lambda: False
    assert task.check_success() is False
    task._object_still_held = lambda: True

    # Success only at the PRESENTED palm.  Before the
    # eval trigger, or while the take_from_human clip still plays, an object
    # near the hand does not count.
    task._eval_trigger_step = 172
    task._blind_human_started_sim_step = None
    assert task.check_success() is False          # human not started
    task._blind_human_started_sim_step = 4300
    task.avatar = SimpleNamespace(spare=lambda: False)
    assert task.check_success() is False          # clip still playing
    task.avatar = SimpleNamespace(spare=lambda: True)
    assert task.check_success() is True           # hand held out
    task.avatar = None
    assert task.check_success() is False
    task.avatar = SimpleNamespace(spare=lambda: True)
    metrics["dist_3d"] = 0.45                     # inside the old 0.65 sphere
    assert task.check_success() is False
    metrics["dist_3d"] = 0.1414213562373095

    task.avatar_collision_summary = lambda: {"any_collision": True}
    assert task.check_success() is False
    task.avatar_collision_summary = lambda: {"any_collision": False}

    task._compute_target_metrics = lambda: None
    assert task.check_success() is False
    task._compute_target_metrics = lambda: metrics

    metrics["dist_3d"] = task.SUCCESS_DIST_3D_M + 1e-6
    assert task.check_success() is False
    metrics["dist_3d"] = task.SUCCESS_DIST_3D_M
    metrics["dz"] = task.SUCCESS_DZ_MIN_M - 1e-6
    assert task.check_success() is False
    metrics["dz"] = task.SUCCESS_DZ_MIN_M
    assert task.check_success() is True


def test_evaluate_metric_values_and_key_order_are_stable():
    task, _metrics = _configured_metric_task()
    task._stage_failed = None
    task._grasp_used = "grasp_007"

    result = task.evaluate()

    assert result["success"] is True
    assert result["object_name"] == "035_apple"
    assert result["object_model_id"] == 1
    assert result["grasp_method"] == "primitive"
    assert result["stage_failed"] is None
    assert result["grasp_used"] == "grasp_007"
    assert result["object_bottom_z"] == 0.84
    assert result["object_table_clearance"] == 0.075
    assert result["object_table_clearance_threshold"] == 0.05
    assert result["object_still_held"] is True
    assert list(result) == [
        "plan_success",
        "success",
        "target_label",
        "target_object_pos",
        "target_pos",
        "target_dist_xy",
        "target_dist_3d",
        "target_dz",
        "target_dist_3d_threshold",
        "target_dz_min_threshold",
        "target_require_no_avatar_collision",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
        "object_name",
        "object_model_id",
        "grasp_method",
        "stage_failed",
        "grasp_used",
        "object_bottom_z",
        "object_table_clearance",
        "object_table_clearance_threshold",
        "object_still_held",
    ]
    assert "eval_policy_step_count" not in result


def test_scene_init_captures_task_facing_state_but_not_controller_state():
    source = DeliverToHuman({"grasp_method": "primitive"})
    source.obj_name = "073_rubikscube"
    source.obj_model_id = 1
    source.instruction = (
        "pick up the rubik's cube from the table and deliver it to the human"
    )
    source._avatar_head_hand_aligned = True
    source._avatar_head_hand_yaw_deg = -12.5
    source._eval_policy_step_count = 7
    # Force lazy controller creation so the exclusion itself is characterized.
    assert source.interaction_controller is not None

    scene_init = capture_scene_init(source, episode_id="deliver-scene", seed=17)
    attrs = scene_init["task_attrs"]
    expected = {
        "instruction": source.instruction,
        "obj_name": "073_rubikscube",
        "obj_model_id": 1,
        "random_object": True,
        "grasp_method_override": "primitive",
        "_avatar_head_hand_aligned": True,
        "_avatar_head_hand_yaw_deg": -12.5,
        "_eval_policy_step_count": 7,
    }
    for name, value in expected.items():
        assert attrs[name] == value
    assert "_interaction_controller" not in attrs
    assert scene_init["avatar"] == {
        "init_pos": [0.0, 1.5, -0.18],
        "init_rot": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    }

    replay_attrs = dict(expected)
    restored = DeliverToHuman({
        "scene_init": {
            "id": "deliver-scene",
            "source_seed": 17,
            "task_attrs": replay_attrs,
            "entities": {},
        }
    })
    replay = apply_scene_init_post_settle(restored)
    assert replay == {
        "enabled": True,
        "id": "deliver-scene",
        "source_seed": 17,
        "applied_attrs": len(replay_attrs),
        "applied_entities": 0,
        "missing_entities": [],
    }
    for name, value in expected.items():
        assert getattr(restored, name) == value
