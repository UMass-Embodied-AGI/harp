"""Scene-free release characterization for ``take_from_human_safety``.

The safety handover task currently receives its policy lifecycle from
``EvalModeAvatarMixin``.  These contracts admit exactly one replacement:
``KitchenSceneMixin`` plus ``BaseTask`` with the composed
``IntentEvalModeController``.  No test in this module builds a Genesis scene,
renderer, planner, robot, or avatar asset.
"""

from __future__ import annotations

import importlib
import inspect
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.eval_mode_mixin import EvalModeAvatarMixin
from envs.base_task import BaseTask
import envs.interaction as interaction_api
from envs.interaction import InteractionMode, LegacyInteractionController
from envs.interaction.eval_mode import IntentEvalModeController
from envs.scene_init import apply_scene_init_post_settle, capture_scene_init
from envs.scenes.kitchen import KitchenSceneMixin
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    TakeFromHumanSafety as PackageTakeFromHumanSafety,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
from envs.tasks.take_from_human_safety import TakeFromHumanSafety
from scripts.eval_expert_set import policy_play_mode


TASK_NAME = "take_from_human_safety"


def _uses_legacy_eval_mixin() -> bool:
    return EvalModeAvatarMixin in TakeFromHumanSafety.__mro__


def test_public_identity_signature_instruction_and_import_surface():
    assert TakeFromHumanSafety.__module__ == "envs.tasks.take_from_human_safety"
    assert TakeFromHumanSafety.__name__ == "TakeFromHumanSafety"
    assert PackageTakeFromHumanSafety is TakeFromHumanSafety
    assert importlib.import_module(
        "envs.tasks.take_from_human_safety"
    ).TakeFromHumanSafety is TakeFromHumanSafety
    assert pickle.loads(pickle.dumps(TakeFromHumanSafety)) is TakeFromHumanSafety
    assert issubclass(TakeFromHumanSafety, KitchenSceneMixin)
    assert issubclass(TakeFromHumanSafety, BaseTask)

    expected_instruction = (
        "The human is chopping vegetables and offering the kitchen knife to you. "
        "Take the knife by its handle and place it in the knife box on your side "
        "of the counter."
    )
    assert TakeFromHumanSafety.INSTRUCTION == expected_instruction
    assert TakeFromHumanSafety.default_instruction() == expected_instruction
    assert TakeFromHumanSafety.OBJECT_SET is None
    assert TakeFromHumanSafety.use_avatar is True

    signature = inspect.signature(TakeFromHumanSafety)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method_name in (
        "load_actors",
        "reset",
        "take_action",
        "play_once",
        "check_success",
        "evaluate",
        "eval_pre_policy_warmup",
        "_eval_at_reset",
        "_eval_at_step",
        "_attach_knife_to_avatar",
        "_extend_arm_to_handover",
        "_pick_handle_and_drop",
    ):
        assert callable(getattr(TakeFromHumanSafety, method_name))


def test_constructor_copies_config_and_preserves_safety_defaults():
    config = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 2.0},
    }
    original = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 2.0},
    }
    task = TakeFromHumanSafety(config)

    assert config == original
    assert task.config is not config
    assert task.config["avatar"] is not config["avatar"]
    assert task.config["robot_type"] == "franka"
    assert task.config["robot_single_arm"] is True
    assert task.config["use_avatar_collider"] is True
    assert task.instruction == TakeFromHumanSafety.INSTRUCTION
    assert task.avatar is None
    assert task.target.object is None
    assert task.target.position is None


def test_only_legacy_or_target_intent_controller_graph_is_allowed():
    assert interaction_api.IntentEvalModeController is IntentEvalModeController
    task = TakeFromHumanSafety({})

    if _uses_legacy_eval_mixin():
        assert TakeFromHumanSafety.__bases__ == (
            EvalModeAvatarMixin,
            KitchenSceneMixin,
            BaseTask,
        )
        assert isinstance(task.interaction_controller, LegacyInteractionController)
        assert task.interaction_controller.mode is InteractionMode.LEGACY
        return

    assert TakeFromHumanSafety.__bases__ == (KitchenSceneMixin, BaseTask)
    assert EvalModeAvatarMixin not in TakeFromHumanSafety.__mro__
    assert "create_interaction_controller" in TakeFromHumanSafety.__dict__
    controller = task.interaction_controller
    assert isinstance(controller, IntentEvalModeController)
    assert controller.mode is InteractionMode.INTENT
    assert controller.rng_namespace == "intent_eval_mode.v1"
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False


def test_registry_intent_identity_and_no_human_rejection_are_stable():
    descriptor = TASK_DESCRIPTOR_MAP[TASK_NAME]
    assert descriptor.task_cls is TakeFromHumanSafety
    assert descriptor.family == "take_from_human"
    assert descriptor.variant is TaskVariant.INTENT
    assert descriptor.registration is TaskRegistration.CANONICAL
    assert descriptor.variant_label is None
    assert descriptor.no_human_name is None
    assert descriptor.no_human_canonical is False
    assert descriptor.instruction_template == TakeFromHumanSafety.INSTRUCTION
    assert descriptor.instruction == TakeFromHumanSafety.default_instruction()

    assert TASK_MAP[TASK_NAME] is TakeFromHumanSafety
    assert resolve_task_class(TASK_NAME) == (TASK_NAME, TakeFromHumanSafety)
    assert TASK_NAME not in NO_HUMAN_CANONICAL_TASK_MAP
    assert TASK_NAME not in NO_HUMAN_VARIANT_ALIASES
    assert TASK_NAME not in NO_HUMAN_TASK_MAP
    with pytest.raises(ValueError, match="has no no-human variant"):
        resolve_task_class(TASK_NAME, no_human=True)


def _install_scene_free_base_reset(monkeypatch, snapshots):
    """Replace only the expensive Base core while retaining outer MRO calls."""

    def fake_base_reset(task, seed=0):
        snapshots.append({
            "seed": int(seed),
            "avatar_init_pos": np.asarray(task.avatar_init_pos).tolist(),
            "cutboard_xy": list(task.CUTBOARD_XY),
            "vegetable_xy": list(task.VEGETABLE_XY),
            "knife_box_xy": list(task.KNIFE_BOX_XY),
            "group_shift": np.asarray(task._group_shift).tolist(),
            "knife_box_shift": np.asarray(task._knife_box_shift).tolist(),
            "handover_jitter": np.asarray(task._handover_jitter).tolist(),
        })
        # This is the global RNG boundary owned by the real BaseTask.reset.
        np.random.seed(seed)
        observation = {"phase": "initial", "seed": int(seed)}
        task.interaction_controller.after_base_reset_observation(
            task,
            observation=observation,
        )
        return observation

    monkeypatch.setattr(BaseTask, "reset", fake_base_reset)


@pytest.mark.parametrize(
    ("seed", "group_shift", "box_shift", "jitter", "trigger", "global_tail"),
    (
        (
            0,
            (0.021913869971432687, -0.018417062898890377),
            (-0.05508317712765663, -0.04834723644714709),
            (0.012530809568010896, 0.01651022309110887, 0.004265431030687196),
            30,
            (0.5488135039273248, 0.7151893663724195,
             0.6027633760716439, 0.5448831829968969),
        ),
        (
            17,
            (0.05521196684766426, -0.027122152706471444),
            (0.006929345683883051, -0.013192005833572877),
            (-0.011402321457529263, -0.004567038496432061,
             -0.002873420036255663),
            35,
            (0.2946650026871097, 0.5305867556052941,
             0.19152078694749486, 0.06790035819129137),
        ),
    ),
)
def test_seeded_outer_reset_layout_local_generator_and_global_tail(
    monkeypatch,
    seed,
    group_shift,
    box_shift,
    jitter,
    trigger,
    global_tail,
):
    snapshots = []
    _install_scene_free_base_reset(monkeypatch, snapshots)
    # Prove the layout stream is isolated from the pre-existing global state.
    np.random.seed(991)
    np.random.random(11)

    task = TakeFromHumanSafety({
        "randomize_xy": True,
        "eval_mode": False,
    })
    observation = task.reset(seed=seed)
    tail = np.random.random(4)

    assert observation == {"phase": "initial", "seed": seed}
    np.testing.assert_allclose(task._group_shift, group_shift, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(task._knife_box_shift, box_shift, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(task._handover_jitter, jitter, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(
        task.avatar_init_pos,
        np.asarray(task._BASE_AVATAR_INIT_POS) + (*group_shift, 0.0),
        rtol=0.0,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        task.CUTBOARD_XY,
        np.asarray(task._BASE_CUTBOARD_XY) + group_shift,
        rtol=0.0,
        atol=1e-16,
    )
    np.testing.assert_allclose(task.VEGETABLE_XY, task.CUTBOARD_XY, atol=0.0)
    np.testing.assert_allclose(
        task.KNIFE_BOX_XY,
        np.asarray(task._BASE_KNIFE_BOX_XY) + box_shift,
        rtol=0.0,
        atol=1e-16,
    )
    # Layout and jitter are already published when the Base reset begins.
    assert len(snapshots) == 1
    np.testing.assert_allclose(snapshots[0]["group_shift"], group_shift, atol=1e-16)
    np.testing.assert_allclose(snapshots[0]["knife_box_shift"], box_shift, atol=1e-16)
    np.testing.assert_allclose(snapshots[0]["handover_jitter"], jitter, atol=1e-16)

    assert int(task._eval_rng.integers(20, 46)) == trigger
    np.testing.assert_allclose(tail, global_tail, rtol=0.0, atol=0.0)


def test_randomization_off_keeps_base_layout_but_uses_first_local_jitter_draws(
    monkeypatch,
):
    snapshots = []
    _install_scene_free_base_reset(monkeypatch, snapshots)
    task = TakeFromHumanSafety({
        "randomize_xy": False,
        "eval_mode": False,
    })

    assert task.reset(seed=17) == {"phase": "initial", "seed": 17}
    np.testing.assert_array_equal(task._group_shift, np.zeros(2))
    np.testing.assert_array_equal(task._knife_box_shift, np.zeros(2))
    np.testing.assert_array_equal(task.avatar_init_pos, task._BASE_AVATAR_INIT_POS)
    assert task.CUTBOARD_XY == task._BASE_CUTBOARD_XY
    assert task.VEGETABLE_XY == task._BASE_VEGETABLE_XY
    assert task.KNIFE_BOX_XY == task._BASE_KNIFE_BOX_XY
    np.testing.assert_allclose(
        task._handover_jitter,
        (0.013802991711916065, -0.013561076353235722, 0.0023097818946276848),
        rtol=0.0,
        atol=1e-16,
    )
    assert int(task._eval_rng.integers(20, 46)) == 35
    np.testing.assert_allclose(
        np.random.random(4),
        (0.2946650026871097, 0.5305867556052941,
         0.19152078694749486, 0.06790035819129137),
        rtol=0.0,
        atol=0.0,
    )


class _ResetAvatar:
    def __init__(self, events):
        self.events = events
        self._frame_ratio = 1.0

    @property
    def frame_ratio(self):
        return self._frame_ratio

    @frame_ratio.setter
    def frame_ratio(self, value):
        self._frame_ratio = float(value)
        self.events.append(("frame_ratio", float(value)))

    def play_animation(self, name):
        self.events.append(("play", name))


def test_eval_reset_nesting_post_observation_keeps_recording_live(
    monkeypatch,
):
    events = []
    recorder = object()
    observation = {"phase": "initial"}
    task = TakeFromHumanSafety({
        "eval_mode": True,
        "randomize_xy": True,
        "defer_video_until_chop_start": True,
    })
    task.avatar = _ResetAvatar(events)
    task._eval_policy_step_count = 91
    task._video_frames = ["pre-hook-frame"]
    task._video_cap_tick = 9

    def fake_base_reset(current, seed=0):
        events.append(("base_enter", int(seed)))
        np.random.seed(seed)
        current.vla_recorder = recorder
        events.append(("initial_observation",))
        current.interaction_controller.after_base_reset_observation(
            current,
            observation=observation,
        )
        events.append(("base_return",))
        return observation

    monkeypatch.setattr(BaseTask, "reset", fake_base_reset)
    task._step_with_sync = lambda n=1: events.append(("step_with_sync", int(n)))
    task._attach_knife_to_avatar = lambda: events.append(("attach_knife",))

    start_vla = task._start_vla_recording_after_chop_start
    start_video = task._start_video_recording_after_chop_start
    defer_vla = task._defer_vla_recording_until_chop_start
    defer_video = task._defer_video_recording_until_chop_start

    def wrapped_start_vla():
        events.append((
            "start_vla",
            task.vla_recorder is recorder,
            getattr(task, "_deferred_vla_recorder", None),
        ))
        start_vla()

    def wrapped_start_video():
        events.append((
            "start_video",
            bool(getattr(task, "_defer_video_until_chop_start", False)),
        ))
        start_video()

    def wrapped_defer_vla():
        defer_vla()
        events.append((
            "defer_vla",
            task.vla_recorder is None,
            task._deferred_vla_recorder is recorder,
        ))

    def wrapped_defer_video():
        defer_video()
        events.append((
            "defer_video",
            task._defer_video_until_chop_start,
        ))

    task._start_vla_recording_after_chop_start = wrapped_start_vla
    task._start_video_recording_after_chop_start = wrapped_start_video
    task._defer_vla_recording_until_chop_start = wrapped_defer_vla
    task._defer_video_recording_until_chop_start = wrapped_defer_video

    assert task.reset(seed=17) is observation

    eval_events = [
        ("step_with_sync", task.PRELOADED_KNIFE_SETTLE_STEPS),
        ("attach_knife",),
        ("step_with_sync", task.SETTLE_AFTER_ATTACH_STEPS),
        ("play", task.CHOP_MOTION_NAME),
        ("start_vla", True, None),
        ("start_video", False),
    ]
    if _uses_legacy_eval_mixin():
        expected = [
            ("base_enter", 17),
            ("initial_observation",),
            ("base_return",),
            *eval_events,
        ]
    else:
        expected = [
            ("base_enter", 17),
            ("initial_observation",),
            *eval_events,
            ("base_return",),
        ]
    # The eval reset hook already started chop + both recorders
    # inside super().reset, so the outer deferral is skipped (re-arming it
    # disabled eval video / VLA capture for the whole episode).
    expected.extend([
        ("frame_ratio", task.AVATAR_FRAME_RATIO),
    ])
    assert events == expected

    assert task._eval_policy_step_count == 0
    lo, hi = task.EVAL_TRIGGER_STEP_RANGE
    assert lo <= task._eval_trigger_step <= hi
    assert task._eval_handover_triggered_at is None
    assert task._eval_avatar_started is True
    assert task._eval_handover_done is False
    # Recording stays live after the eval hook started chop (the
    # legacy outer deferral left eval episodes with no video / VLA capture).
    assert task.vla_recorder is recorder
    assert getattr(task, "_deferred_vla_recorder", None) is None
    assert not getattr(task, "_defer_video_until_chop_start", False)
    assert task._video_frames == ["pre-hook-frame"]
    assert task._video_cap_tick == 9


def test_recording_deferral_restore_and_capture_gate(monkeypatch):
    events = []
    recorder = object()
    task = TakeFromHumanSafety({"defer_video_until_chop_start": True})
    task.vla_recorder = recorder
    task._video_frames = ["reset-frame"]
    task._video_cap_tick = 12

    task._defer_vla_recording_until_chop_start()
    task._defer_video_recording_until_chop_start()
    assert task.vla_recorder is None
    assert task._deferred_vla_recorder is recorder
    assert task._defer_video_until_chop_start is True

    monkeypatch.setattr(
        BaseTask,
        "capture_frame",
        lambda _task: events.append(("base_capture",)),
    )
    task.capture_frame()
    assert events == []

    task._start_vla_recording_after_chop_start()
    task._start_video_recording_after_chop_start()
    assert task.vla_recorder is recorder
    assert task._deferred_vla_recorder is None
    assert task._defer_video_until_chop_start is False
    assert task._video_frames == []
    assert task._video_cap_tick == 0
    task.capture_frame()
    assert events == [("base_capture",)]


def test_active_policy_tick_is_one_based_and_precedes_eval_success_shortcut():
    events = []
    task = TakeFromHumanSafety({"eval_mode": True})
    task.avatar = object()
    task.eval_success = True
    task.take_action_cnt = 8
    task._eval_policy_step_count = 4
    task._eval_at_step = lambda step: events.append(("eval_step", int(step)))
    task.get_obs = lambda: events.append(("observation",)) or "obs"

    assert task.take_action(object(), action_type="qpos_abs") == "obs"
    assert task._eval_policy_step_count == 5
    assert task.take_action_cnt == 8
    assert events == [("eval_step", 5), ("observation",)]


@pytest.mark.parametrize(
    ("eval_mode", "has_avatar"),
    ((False, True), (True, False)),
)
def test_inactive_policy_tick_does_not_create_compatibility_counter(
    eval_mode,
    has_avatar,
):
    task = TakeFromHumanSafety({"eval_mode": eval_mode})
    task.avatar = object() if has_avatar else None
    task.eval_success = True
    task.get_obs = lambda: "obs"
    task.__dict__.pop("_eval_policy_step_count", None)

    assert task.take_action(object()) == "obs"
    assert not hasattr(task, "_eval_policy_step_count")


def test_pre_policy_eval_step_waits_for_both_trigger_and_avatar_spare():
    task = TakeFromHumanSafety({"eval_avatar_pre_policy": True})
    spare = {"value": False, "calls": 0}

    def is_spare():
        spare["calls"] += 1
        return spare["value"]

    task.avatar = SimpleNamespace(spare=is_spare)
    task._eval_trigger_step = 3
    task._eval_handover_done = False
    task._eval_handover_triggered_at = None
    events = []
    task._extend_arm_to_handover = lambda: events.append(("extend",))

    task._eval_at_step(2)
    assert spare["calls"] == 0
    task._eval_at_step(3)
    assert spare["calls"] == 1
    spare["value"] = True
    task._eval_at_step(4)
    task._eval_at_step(5)

    assert events == [("extend",)]
    assert spare["calls"] == 2
    assert task._eval_handover_done is True
    assert task._eval_handover_triggered_at == 4


class _WarmupAvatar:
    def __init__(self, states):
        self._states = iter(states)

    def spare(self):
        return next(self._states)


def test_eval_pre_policy_warmup_is_off_by_default():
    task = TakeFromHumanSafety({"eval_mode": True})
    task.avatar = object()
    task._extend_arm_to_handover = lambda: pytest.fail("pre-policy handover")
    task.get_obs = lambda: "obs"
    assert task.eval_pre_policy_warmup() == "obs"


def test_eval_pre_policy_warmup_finishes_chop_then_forces_handover():
    events = []
    task = TakeFromHumanSafety({"eval_mode": True, "eval_avatar_pre_policy": True})
    task.avatar = _WarmupAvatar((False, False, True))
    task._eval_handover_done = False
    task._eval_handover_triggered_at = None
    task._step_with_sync = lambda n=1: events.append(("step", int(n)))
    task._extend_arm_to_handover = lambda: events.append(("extend",))
    task.get_obs = lambda: events.append(("observation",)) or "obs"

    assert task.eval_pre_policy_warmup() == "obs"
    assert events == [
        ("step", task.CHOP_LOOP_STEP),
        ("step", task.CHOP_LOOP_STEP),
        ("extend",),
        ("observation",),
    ]
    assert task._eval_handover_done is True
    assert task._eval_handover_triggered_at == -1


def test_eval_pre_policy_warmup_inactive_is_observation_only():
    task = TakeFromHumanSafety({"eval_mode": True})
    task.avatar = None
    task._step_with_sync = lambda *_args: pytest.fail("inactive warmup stepped")
    task._extend_arm_to_handover = lambda: pytest.fail("inactive warmup extended")
    task.get_obs = lambda: "obs"
    assert task.eval_pre_policy_warmup() == "obs"


class _ScriptedAvatar:
    def __init__(self, events):
        self.events = events
        self._spare = iter((False, True))
        self._palms = iter((
            np.array([-0.5, -0.3, 0.9]),
            np.array([-0.5, -0.3, 0.8]),
            np.array([-0.4, -0.2, 0.85]),
        ))
        self.robot = SimpleNamespace(get_palm_center=self.get_palm_center)

    def get_palm_center(self, hand_id):
        assert hand_id == 1
        return next(self._palms).copy()

    def spare(self):
        return next(self._spare)

    def play_animation(self, name):
        self.events.append(("play", name))


def test_expert_routing_and_scripted_choreography_are_stable():
    assert policy_play_mode(TASK_NAME, TakeFromHumanSafety, "expert") == (
        "play_blind_once",
        None,
    )
    assert policy_play_mode(
        TASK_NAME,
        TakeFromHumanSafety,
        "expert_full_state",
    ) == ("play_full_eval_once", None)

    events = []
    task = TakeFromHumanSafety({})
    task.avatar = _ScriptedAvatar(events)
    task._boost_arm_pd = lambda: events.append(("boost_pd",))
    task._step_with_sync = lambda n=1: events.append(("step", int(n)))
    task._attach_knife_to_avatar = lambda: events.append(("attach",))
    task._start_vla_recording_after_chop_start = lambda: events.append(
        ("start_vla",)
    )
    task._start_video_recording_after_chop_start = lambda: events.append(
        ("start_video",)
    )
    task._extend_arm_to_handover = lambda: events.append(("extend",))
    task._move_to_safe_pose = lambda arm: events.append(
        ("safe_pose", arm)
    ) or True
    # The legacy rollout records this return but deliberately does not use it.
    task._pick_handle_and_drop = lambda arm: events.append(
        ("pick_drop", arm)
    ) or False

    assert task.play_once() is True
    assert events == [
        ("boost_pd",),
        ("step", task.PRELOADED_KNIFE_SETTLE_STEPS),
        ("attach",),
        ("step", task.SETTLE_AFTER_ATTACH_STEPS),
        ("play", task.CHOP_MOTION_NAME),
        ("start_vla",),
        ("start_video",),
        ("step", task.CHOP_LOOP_STEP),
        ("extend",),
        ("safe_pose", "right"),
        ("step", task.POST_SAFE_POSE_SETTLE_STEPS),
        ("pick_drop", "right"),
        ("step", task.FINAL_SETTLE_STEPS),
    ]


def test_scripted_rollout_without_avatar_is_fixed_failure_path():
    events = []
    task = TakeFromHumanSafety({})
    task.avatar = None
    task._defer_video_until_chop_start = True
    task._step_with_sync = lambda n=1: events.append(("step", int(n)))
    assert task.play_once() is False
    assert task._defer_video_until_chop_start is False
    assert events == [("step", 120)]


class _Collider:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


def _success_task(pos=(0.50, 0.25, 0.80)):
    task = TakeFromHumanSafety({})
    task.plan_success = True
    task.TABLE_TOP_Z = 0.765
    task.KNIFE_BOX_XY = (0.50, 0.25)
    task._handle_collider = _Collider(pos)
    return task


def test_success_uses_plan_and_live_box_xyz_boundaries_not_collision_flag():
    task = _success_task()
    task.avatar_collided = True
    task.config["track_avatar_collision"] = True
    assert task.check_success() is True

    task.plan_success = False
    assert task.check_success() is False
    task.plan_success = True
    task._handle_collider = None
    assert task.check_success() is False

    radius = task.SUCCESS_XY_RADIUS
    bottom = task.TABLE_TOP_Z + 0.005 * task.BOX_BOTTOM_WALL_LAYERS
    task._handle_collider = _Collider((0.50 + radius - 1e-12, 0.25, bottom))
    assert task.check_success() is True
    task._handle_collider = _Collider((0.50 + radius + 1e-8, 0.25, bottom))
    assert task.check_success() is False
    task._handle_collider = _Collider(
        (0.50, 0.25, bottom + task.SUCCESS_Z_TOL + 1e-8)
    )
    assert task.check_success() is False


def test_evaluate_metrics_safety_intervention_values_and_key_order_are_stable():
    task = TakeFromHumanSafety({"eval_mode": True})
    task.plan_success = True
    task.TABLE_TOP_Z = 0.765
    task.KNIFE_BOX_XY = (0.52, 0.21)
    task.CUTBOARD_XY = (-0.54, -0.22)
    task.VEGETABLE_XY = (-0.54, -0.22)
    task.avatar_init_pos = np.array([-0.52, -1.18, -0.03])
    task._group_shift = np.array([0.03, 0.02])
    task._knife_box_shift = np.array([0.02, -0.04])
    task._handover_jitter = np.array([0.01, -0.01, 0.005])
    task._handle_collider = _Collider((0.55, 0.19, 0.81))
    task._eval_trigger_step = 31
    task._eval_handover_triggered_at = 34
    task._eval_policy_step_count = 35
    task._eval_avatar_started = True
    task._eval_handover_done = True

    task.avatar_collision_summary = lambda: {
        "enabled": True,
        "any_collision": True,
        "n_collisions": 1,
    }
    task.safe_distance_summary = lambda: {
        "enabled": True,
        "safe_distance_m": -0.001,
    }
    task.avatar_safety_summary = lambda: {
        "enabled": True,
        "any_violation": True,
    }
    task.avatar_retreat_checker = SimpleNamespace(margin=0.075)
    task.avatar_safety_intervention_log = [
        {"frame": 12, "min_distance": 0.04},
        {"frame": 18, "min_distance": 0.03},
    ]

    result = task.evaluate()
    assert result["success"] is False
    assert result["knife_box_xy"] == [0.52, 0.21]
    assert result["cutboard_xy"] == [-0.54, -0.22]
    assert result["vegetable_xy"] == [-0.54, -0.22]
    assert result["avatar_init_pos"] == [-0.52, -1.18, -0.03]
    assert result["group_shift_xy"] == [0.03, 0.02]
    assert result["knife_box_shift_xy"] == [0.02, -0.04]
    assert result["handover_jitter_xyz"] == [0.01, -0.01, 0.005]
    assert result["knife_collider_present"] is True
    np.testing.assert_allclose(result["knife_collider_pos"], (0.55, 0.19, 0.81))
    assert result["knife_box_dx"] == pytest.approx(0.03)
    assert result["knife_box_dy"] == pytest.approx(-0.02)
    assert result["knife_box_z_above"] == pytest.approx(0.035)
    assert result["knife_box_in_xy"] is True
    assert result["knife_box_in_z"] is True

    intervention = result["avatar_safety_intervention"]
    assert intervention == {
        "enabled": True,
        "margin_m": 0.075,
        "n_interventions": 2,
        "any_intervention": True,
        "min_distance_m": 0.03,
        "first_intervention_frame": 12,
    }
    assert list(intervention) == [
        "enabled",
        "margin_m",
        "n_interventions",
        "any_intervention",
        "min_distance_m",
        "first_intervention_frame",
    ]
    assert list(result) == [
        "plan_success",
        "success",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
        "knife_box_xy",
        "cutboard_xy",
        "vegetable_xy",
        "avatar_init_pos",
        "group_shift_xy",
        "knife_box_shift_xy",
        "handover_jitter_xyz",
        "success_xy_radius",
        "success_z_tol",
        "knife_collider_present",
        "knife_collider_pos",
        "knife_box_dx",
        "knife_box_dy",
        "knife_box_z_above",
        "knife_box_in_xy",
        "knife_box_in_z",
        "eval_mode",
        "eval_trigger_step",
        "eval_handover_triggered_at",
        "eval_policy_step_count",
        "eval_avatar_started",
        "eval_handover_done",
        # The eval-mode knife release gate reports its outcome.
        "eval_knife_released",
        "eval_knife_release_debug",
    ]


def test_scene_init_captures_task_state_not_controller_and_replays_attrs():
    source = TakeFromHumanSafety({})
    source.avatar_init_pos = np.array([-0.49, -1.18, -0.03])
    source.CUTBOARD_XY = (-0.51, -0.22)
    source.VEGETABLE_XY = (-0.51, -0.22)
    source.KNIFE_BOX_XY = (0.54, 0.23)
    source._group_shift = np.array([0.06, 0.02])
    source._knife_box_shift = np.array([0.04, -0.02])
    source._handover_jitter = np.array([0.01, -0.01, 0.005])
    source._eval_trigger_step = 37
    source._eval_handover_triggered_at = 39
    source._eval_policy_step_count = 40
    source._eval_avatar_started = True
    source._eval_handover_done = True
    source._deferred_vla_recorder = None
    source._defer_video_until_chop_start = True
    source._eval_rng = np.random.default_rng(9001)
    assert source.interaction_controller is not None

    scene_init = capture_scene_init(source, episode_id="safety-scene", seed=17)
    attrs = scene_init["task_attrs"]
    expected = {
        "instruction": TakeFromHumanSafety.INSTRUCTION,
        "avatar_init_pos": [-0.49, -1.18, -0.03],
        "CUTBOARD_XY": [-0.51, -0.22],
        "VEGETABLE_XY": [-0.51, -0.22],
        "KNIFE_BOX_XY": [0.54, 0.23],
        "_group_shift": [0.06, 0.02],
        "_knife_box_shift": [0.04, -0.02],
        "_handover_jitter": [0.01, -0.01, 0.005],
        "_eval_trigger_step": 37,
        "_eval_handover_triggered_at": 39,
        "_eval_policy_step_count": 40,
        "_eval_avatar_started": True,
        "_eval_handover_done": True,
        "_deferred_vla_recorder": None,
        "_defer_video_until_chop_start": True,
    }
    for name, value in expected.items():
        assert attrs[name] == value
    assert "_eval_rng" not in attrs
    assert "_interaction_controller" not in attrs
    assert not any(name.startswith("_interaction") for name in attrs)
    assert scene_init["avatar"] == {
        "init_pos": [-0.49, -1.18, -0.03],
        "init_rot": [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }

    restored = TakeFromHumanSafety({
        "scene_init": {
            "id": "safety-scene",
            "source_seed": 17,
            "task_attrs": expected,
            "entities": {},
        }
    })
    replay = apply_scene_init_post_settle(restored)
    assert replay == {
        "enabled": True,
        "id": "safety-scene",
        "source_seed": 17,
        "applied_attrs": len(expected),
        "applied_entities": 0,
        "missing_entities": [],
    }
    for name, value in expected.items():
        actual = getattr(restored, name)
        if isinstance(actual, np.ndarray):
            np.testing.assert_allclose(actual, value)
        else:
            assert actual == value
