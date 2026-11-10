"""Scene-free release characterization for :mod:`envs.tasks.pour_water`.

The task currently receives its policy-eval lifecycle from
``EvalModeAvatarMixin``.  These contracts also admit its intended replacement:
``IntentEvalModeController`` driven through ``BaseTask``'s post-observation and
pre-action hooks.  No test constructs a Genesis scene, robot, planner,
renderer, or avatar asset.
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
from envs.interaction.eval_mode import IntentEvalModeController
from envs.scene_init import apply_scene_init_post_settle, capture_scene_init
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    PourWater as ExportedPourWater,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
import envs.tasks.pour_water as pour_module
from envs.tasks.pour_water import PourWater
from envs.utils import Pose
from scripts.eval_expert_set import policy_play_mode


TASK_NAME = "pour_water"


def test_public_identity_signature_constants_and_task_surface() -> None:
    assert PourWater.__module__ == "envs.tasks.pour_water"
    assert PourWater.__name__ == "PourWater"
    assert ExportedPourWater is PourWater
    assert issubclass(PourWater, BaseTask)
    assert PourWater.INSTRUCTION == "pour water from the mug into the cup"
    assert PourWater.default_instruction() == PourWater.INSTRUCTION
    assert PourWater.OBJECT_SET == ["039_mug"]
    assert PourWater.use_avatar is True
    assert pour_module._MUG_ID == "039_mug"
    assert pour_module._CUP_ID == "021_cup"
    np.testing.assert_array_equal(
        pour_module._Q_UPRIGHT,
        [0.70710678, 0.70710678, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        pour_module._Q_MUG_HANDLE_TOWARD_ROBOT,
        [0.0, 0.0, 0.70710678, 0.70710678],
    )
    assert pour_module._MUG_BODY_GRASP_ORDER == (
        "manual_body_edge_topdown_posx_posz",
        "manual_body_edge_topdown",
        "manual_body_edge_topdown_negx_posz",
        "manual_body_edge_topdown_posx",
        "manual_body_edge_topdown_negx",
    )

    signature = inspect.signature(PourWater)
    assert tuple(signature.parameters) == ("config",)
    assert signature.parameters["config"].default is None
    for method_name in (
        "load_actors",
        "play_once",
        "check_success",
        "evaluate",
        "reset",
        "take_action",
        "eval_pre_policy_warmup",
        "_eval_at_reset",
        "_eval_at_step",
        "_start_eval_drink1",
        "_start_eval_drink2",
    ):
        assert callable(getattr(PourWater, method_name))


def test_constructor_copies_config_and_keeps_release_defaults() -> None:
    assert isinstance(PourWater(), PourWater)
    config = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    original = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    task = PourWater(config)

    assert config == original
    assert task.config is not config
    assert task.config["avatar"] is not config["avatar"]
    assert task.instruction == PourWater.INSTRUCTION
    assert task.plan_success is True
    assert task.eval_success is False
    assert task.take_action_cnt == 0
    assert task._target_entry is None
    np.testing.assert_array_equal(task.table_offset, [0.0, -0.35])
    assert task.EVAL_TRIGGER_STEP_MIN == 100
    assert task.EVAL_TRIGGER_STEP_MAX == 200


def test_only_legacy_mixin_or_intended_intent_controller_graph_is_allowed() -> None:
    task = PourWater({})
    assert interaction_api.IntentEvalModeController is IntentEvalModeController

    if EvalModeAvatarMixin in PourWater.__mro__:
        assert PourWater.__bases__ == (EvalModeAvatarMixin, BaseTask)
        assert isinstance(task.interaction_controller, LegacyInteractionController)
        assert task.interaction_controller.mode is InteractionMode.LEGACY
        return

    assert PourWater.__bases__ == (BaseTask,)
    controller = task.interaction_controller
    assert isinstance(controller, IntentEvalModeController)
    assert controller.mode is InteractionMode.INTENT
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False


def test_registry_intent_identity_and_no_human_rejection_are_stable() -> None:
    descriptor = TASK_DESCRIPTOR_MAP[TASK_NAME]
    assert descriptor.task_cls is PourWater
    assert descriptor.family == "pour_water"
    assert descriptor.variant is TaskVariant.INTENT
    assert descriptor.registration is TaskRegistration.CANONICAL
    assert descriptor.variant_label is None
    assert descriptor.no_human_name is None
    assert descriptor.no_human_canonical is False
    assert descriptor.instruction_template == PourWater.INSTRUCTION
    assert descriptor.instruction == PourWater.INSTRUCTION

    assert TASK_MAP[TASK_NAME] is PourWater
    assert resolve_task_class(TASK_NAME) == (TASK_NAME, PourWater)
    assert TASK_NAME not in NO_HUMAN_CANONICAL_TASK_MAP
    assert TASK_NAME not in NO_HUMAN_VARIANT_ALIASES
    assert TASK_NAME not in NO_HUMAN_TASK_MAP
    with pytest.raises(ValueError, match="has no no-human variant"):
        resolve_task_class(TASK_NAME, no_human=True)


class _FakeEntity:
    def __init__(self, position, quaternion):
        self._position = np.asarray(position, dtype=np.float64)
        self._quaternion = np.asarray(quaternion, dtype=np.float64)

    def get_pos(self):
        return self._position

    def get_quat(self):
        return self._quaternion


class _FakeActor:
    def __init__(self, position, quaternion):
        self.entity = _FakeEntity(position, quaternion)


def _capture_seeded_layout(monkeypatch, config: dict, seed: int):
    events = []

    def load_object(_scene, pose, object_id, **kwargs):
        events.append((
            "load_object",
            object_id,
            tuple(float(value) for value in pose.p),
            tuple(float(value) for value in pose.q),
            dict(kwargs),
        ))
        return _FakeActor(pose.p, pose.q)

    def load_object_config(object_id):
        events.append(("load_object_config", object_id))
        return {"rim_radius": 1.25}

    def find_mesh(_task, directory, model_id=None):
        events.append(("find_mesh", str(directory), model_id))
        return "synthetic_mug.glb"

    def load_mesh(_scene, mesh, pose, **kwargs):
        events.append((
            "load_mesh",
            str(mesh),
            tuple(float(value) for value in pose.p),
            tuple(float(value) for value in pose.q),
            dict(kwargs),
        ))
        return _FakeEntity(pose.p, pose.q)

    monkeypatch.setattr(pour_module, "load_object", load_object)
    monkeypatch.setattr(pour_module, "load_object_config", load_object_config)
    monkeypatch.setattr(PourWater, "_find_mesh", find_mesh)
    monkeypatch.setattr(pour_module, "load_mesh", load_mesh)

    task = PourWater(config)
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed)
        task.load_actors()
        tail = np.random.random_sample(8)
    finally:
        np.random.set_state(previous_state)
    return task, events, tail


@pytest.mark.parametrize(
    (
        "config",
        "seed",
        "expected_layout",
        "expected_tail",
        "require_no_collision",
    ),
    (
        (
            {},
            0,
            (
                0.009762700785464956,
                -0.07023729921453505,
                0.2715189366372419,
                -0.08972366239283561,
            ),
            (
                0.5448831829968969,
                0.4236547993389047,
                0.6458941130666561,
                0.4375872112626925,
                0.8917730007820798,
                0.9636627605010293,
                0.3834415188257777,
                0.7917250380826646,
            ),
            False,
        ),
        (
            {"track_avatar_collision": True},
            1,
            (
                -0.016595599059485197,
                -0.0965955990594852,
                0.2720324493442158,
                -0.1499885625182655,
            ),
            (
                0.30233257263183977,
                0.14675589081711304,
                0.0923385947687978,
                0.1862602113776709,
                0.34556072704304774,
                0.39676747423066994,
                0.538816734003357,
                0.4191945144032948,
            ),
            True,
        ),
        (
            {"track_avatar_collision": True, "no_human": True},
            42,
            (
                -0.025091976230527502,
                -0.1050919762305275,
                0.2950714306409916,
                -0.07680060581885949,
            ),
            (
                0.5986584841970366,
                0.15601864044243652,
                0.15599452033620265,
                0.05808361216819946,
                0.8661761457749352,
                0.6011150117432088,
                0.7080725777960455,
                0.020584494295802447,
            ),
            False,
        ),
    ),
)
def test_seeded_actor_layout_target_and_global_numpy_tail(
    monkeypatch,
    config,
    seed,
    expected_layout,
    expected_tail,
    require_no_collision,
) -> None:
    task, events, tail = _capture_seeded_layout(monkeypatch, config, seed)
    shift, cup_x, mug_x, mug_y = expected_layout

    assert task._target_entry.key == "039_mug"
    assert task._target_entry.object_id == "039_mug"
    assert task.instruction == PourWater.INSTRUCTION
    assert task._avatar_shift_x == shift
    np.testing.assert_allclose(
        task.cup_pose.p,
        [cup_x, 0.0, 0.885],
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        task.mug_pose.p,
        [mug_x, mug_y, 0.885],
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_array_equal(task.cup_pose.q, pour_module._Q_UPRIGHT)
    np.testing.assert_array_equal(
        task.mug_pose.q,
        pour_module._Q_MUG_HANDLE_TOWARD_ROBOT,
    )
    assert task.cup_scale == 0.088
    assert task.cup_rim_height == 0.08949599999999999
    assert task.cup_rim_radius == 0.043472
    assert task.mug_scale == 0.08
    assert task.mug_rim_radius == 0.1

    target = task.target
    assert target.object is task.mug_entity
    assert target.label == "cup_rim"
    assert target.success_dist_xy == 0.20
    assert target.success_dz_min == 0.10
    assert target.success_tilt_min_deg == 60.0
    assert target.success_require_no_avatar_collision is require_no_collision
    np.testing.assert_allclose(
        target.position(),
        [cup_x, 0.0, 0.974496],
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        target.object_axis_fn(),
        [0.0, 0.0, 1.0],
        rtol=0.0,
        atol=5e-15,
    )

    assert events == [
        (
            "load_object",
            "021_cup",
            (cup_x, 0.0, 0.885),
            tuple(pour_module._Q_UPRIGHT),
            {"model_id": 0, "convex": True},
        ),
        ("load_object_config", "039_mug"),
        (
            "find_mesh",
            str(pour_module.ASSETS_PATH / "objects" / "039_mug" / "visual"),
            0,
        ),
        (
            "load_mesh",
            "synthetic_mug.glb",
            (mug_x, mug_y, 0.885),
            tuple(pour_module._Q_MUG_HANDLE_TOWARD_ROBOT),
            {
                "scale": (0.08, 0.08, 0.08),
                "convex": True,
                "friction": 2.0,
                "density": 100.0,
            },
        ),
    ]
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-15)


class _EvalAvatar:
    def __init__(self, events, spare_states=()):
        self.events = events
        self._spare_states = iter(spare_states)

    def reset(self, position, rotation):
        self.events.append((
            "avatar_reset",
            tuple(float(value) for value in position),
            tuple(tuple(float(value) for value in row) for row in rotation),
        ))

    def play_animation(self, name, **kwargs):
        self.events.append(("play_animation", name, dict(kwargs)))

    def play_animation_hold_reverse(self, name, **kwargs):
        self.events.append(("play_hold_reverse", name, dict(kwargs)))

    def spare(self):
        state = next(self._spare_states)
        self.events.append(("avatar_spare", state))
        return state


def _run_eval_reset_boundary(task, events, monkeypatch, seed=17):
    """Run the legacy wrapper or its sole composed-controller equivalent."""
    observation = {"phase": "initial"}
    if EvalModeAvatarMixin in PourWater.__mro__:
        monkeypatch.setattr(
            BaseTask,
            "reset",
            lambda _task, seed=0: events.append(
                ("initial_observation", seed)
            ) or observation,
        )
        return task.reset(seed=seed)

    events.append(("initial_observation", seed))
    task.interaction_controller.after_base_reset_observation(
        task,
        observation=observation,
    )
    return observation


def test_eval_reset_is_post_observation_and_default_warmup_choreography(
    monkeypatch,
) -> None:
    events = []
    cup = object()
    task = PourWater({
        "eval_mode": True,
        "eval_avatar_post_warmup_settle_steps": 2,
    })
    task.avatar = _EvalAvatar(events, (False, True, False, True))
    task.cup_actor = SimpleNamespace(entity=cup)
    task._avatar_shift_x = 0.03
    task._eval_policy_step_count = 99
    task.step_sim = lambda: events.append(("sim_step",))
    task.get_obs = lambda: events.append(("warm_observation",)) or {
        "phase": "warm"
    }

    initial = _run_eval_reset_boundary(task, events, monkeypatch)
    assert initial == {"phase": "initial"}
    assert task._eval_policy_step_count == 0
    assert task._eval_trigger_step is None
    assert task._eval_drink1_started is False
    assert task._eval_drink2_queued is False

    warmed = task.eval_pre_policy_warmup()
    assert warmed == {"phase": "warm"}
    assert task._eval_drink1_started is True
    assert task._eval_drink2_queued is True
    assert events == [
        ("initial_observation", 17),
        (
            "avatar_reset",
            (0.009999999999999998, 0.91, -0.1),
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        ("play_animation", "drink_water_1", {"attach_obj": cup, "hand_id": 1}),
        ("avatar_spare", False),
        ("sim_step",),
        ("avatar_spare", True),
        (
            "play_hold_reverse",
            "drink_water_2",
            {
                "hold_at_frame": 39,
                "hold_duration": 70,
                "reverse_to_frame": 1,
                "hold_end": 30,
            },
        ),
        ("avatar_spare", False),
        ("sim_step",),
        ("avatar_spare", True),
        ("sim_step",),
        ("sim_step",),
        ("warm_observation",),
    ]


def test_delayed_eval_reset_trigger_clamp_state_and_rng_tail(monkeypatch) -> None:
    events = []
    task = PourWater({
        "eval_mode": True,
        "eval_avatar_pre_policy": False,
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 9,
    })
    task.avatar = _EvalAvatar(events)
    task._avatar_shift_x = -0.04
    previous_state = np.random.get_state()
    try:
        np.random.seed(0)
        initial = _run_eval_reset_boundary(task, events, monkeypatch, seed=23)
        tail = np.random.random_sample(8)
    finally:
        np.random.set_state(previous_state)

    assert initial == {"phase": "initial"}
    assert task._eval_policy_step_count == 0
    assert task._eval_trigger_step == 7
    assert task._eval_drink1_started is False
    assert task._eval_drink2_queued is False
    assert events == [
        ("initial_observation", 23),
        (
            "avatar_reset",
            (-0.06, 0.91, -0.1),
            ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    ]
    np.testing.assert_allclose(
        tail,
        (
            0.5928446182250183,
            0.8442657485810173,
            0.8579456176227568,
            0.8472517387841254,
            0.6235636967859723,
            0.3843817072926998,
            0.2975346065444723,
            0.05671297731744318,
        ),
        rtol=0.0,
        atol=5e-15,
    )


@pytest.mark.parametrize(
    ("eval_mode", "has_avatar"),
    ((False, True), (True, False)),
)
def test_eval_reset_is_inactive_without_both_gates(
    eval_mode,
    has_avatar,
    monkeypatch,
) -> None:
    events = []
    task = PourWater({"eval_mode": eval_mode})
    task.avatar = _EvalAvatar(events) if has_avatar else None
    task._eval_policy_step_count = 91

    assert _run_eval_reset_boundary(task, events, monkeypatch) == {
        "phase": "initial"
    }
    assert task._eval_policy_step_count == 91
    assert events == [("initial_observation", 17)]


def test_policy_hook_is_one_based_and_precedes_eval_success_short_circuit() -> None:
    events = []
    cup = object()
    task = PourWater({"eval_mode": True})
    task.avatar = _EvalAvatar(events, (True,))
    task.cup_actor = SimpleNamespace(entity=cup)
    task._eval_policy_step_count = 0
    task._eval_trigger_step = 1
    task._eval_drink1_started = False
    task._eval_drink2_queued = False
    task.eval_success = True
    task.take_action_cnt = 0
    task.get_obs = lambda: events.append(("observation",)) or "obs"

    action = object()
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert task.take_action(action, action_type="qpos_abs") == "obs"
    assert events == [
        ("play_animation", "drink_water_1", {"attach_obj": cup, "hand_id": 1}),
        ("observation",),
        ("avatar_spare", True),
        (
            "play_hold_reverse",
            "drink_water_2",
            {
                "hold_at_frame": 39,
                "hold_duration": 70,
                "reverse_to_frame": 1,
                "hold_end": 30,
            },
        ),
        ("observation",),
    ]
    assert task._eval_policy_step_count == 2
    assert task._eval_drink1_started is True
    assert task._eval_drink2_queued is True
    assert task.take_action_cnt == 0
    assert task.eval_success is True


def test_inactive_policy_call_does_not_create_eval_counter() -> None:
    task = PourWater({"eval_mode": True})
    task.avatar = None
    task.eval_success = True
    task.get_obs = lambda: "obs"
    task.__dict__.pop("_eval_policy_step_count", None)

    assert task.take_action(object()) == "obs"
    assert not hasattr(task, "_eval_policy_step_count")
    assert task.take_action_cnt == 0


def test_expert_routing_stays_on_public_scripted_rollout() -> None:
    assert policy_play_mode(TASK_NAME, PourWater, "expert") == (
        "play_blind_once",
        None,
    )
    assert policy_play_mode(TASK_NAME, PourWater, "expert_full_state") == (
        "play_once",
        None,
    )


class _ScriptedAvatar:
    def __init__(self, events, step_count):
        self.events = events
        self.step_count = step_count
        self._states = iter((False, True, False, True))

    def play_animation(self, name, **kwargs):
        self.events.append(("play_animation", name, dict(kwargs), self.step_count["n"]))

    def play_animation_hold_reverse(self, name, **kwargs):
        self.events.append(("play_hold_reverse", name, dict(kwargs), self.step_count["n"]))

    def spare(self):
        value = next(self._states)
        self.events.append(("avatar_spare", value, self.step_count["n"]))
        return value


def test_scripted_expert_avatar_grasp_and_pour_choreography() -> None:
    task = PourWater({})
    events = []
    steps = {"n": 0}
    cup = object()
    mug = object()
    task.avatar = _ScriptedAvatar(events, steps)
    task.cup_actor = SimpleNamespace(entity=cup)
    task.mug_entity = mug
    task.mug_scale = 0.08
    task._mug_physics_pose = Pose(
        [0.25, -0.10, 0.885],
        pour_module._Q_MUG_HANDLE_TOWARD_ROBOT,
    )
    task.step_sim = lambda: steps.__setitem__("n", steps["n"] + 1)
    task._calibrate_avatar_pos = lambda: events.append(("calibrate", steps["n"]))

    arm = SimpleNamespace(
        get_ee_pose=lambda: events.append(("get_ee_pose", steps["n"]))
        or np.array([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
    )
    task.robot = SimpleNamespace(
        get_arm=lambda arm_tag: events.append(
            ("get_arm", arm_tag, steps["n"])
        ) or arm,
    )
    task.open_gripper = lambda arm_tag: events.append(
        ("open_gripper", arm_tag, steps["n"])
    )
    task.close_gripper = lambda arm_tag: events.append(
        ("close_gripper", arm_tag, steps["n"])
    )
    task._boost_gripper_force = lambda selected_arm: events.append(
        ("boost_gripper", selected_arm is arm, steps["n"])
    )
    mug_pose = Pose(
        [0.25, -0.10, 0.885],
        pour_module._Q_MUG_HANDLE_TOWARD_ROBOT,
    )
    task._get_mug_logical_pose = lambda: events.append(
        ("logical_mug_pose", steps["n"])
    ) or mug_pose
    task._get_object_pose = lambda entity, fallback: events.append(
        ("live_mug_pose", entity is mug, steps["n"])
    ) or mug_pose

    class _Grasp:
        def to_world(self, pose, scale):
            events.append(("grasp_to_world", pose is mug_pose, scale, steps["n"]))
            return pose

    attempts = []

    def try_grasp(object_id, pose, arm_tag, **kwargs):
        attempts.append(kwargs["grasp_name"])
        events.append((
            "try_grasp",
            object_id,
            pose is mug_pose,
            arm_tag,
            kwargs["grasp_name"],
            kwargs["robot_type"],
            kwargs["object_scale"],
            steps["n"],
        ))
        if len(attempts) == 2:
            return object(), object(), _Grasp()
        return None

    task.try_grasp_by_name = try_grasp
    task.select_and_execute_grasp = lambda *_args, **_kwargs: pytest.fail(
        "fallback grasp selector ran despite a named-grasp success"
    )

    sweep_results = iter((45, 90))

    def sweep(pose_fn, arm_tag, label, yaws=None):
        events.append((
            "sweep",
            pose_fn.__name__,
            arm_tag,
            label,
            yaws,
            steps["n"],
        ))
        return next(sweep_results)

    task._sweep_yaw_and_execute = sweep
    # The closed-loop steps (yaw ordering by grip hinge, pour
    # position correction) need a live arm/mug; this test freezes the
    # surrounding choreography, so they are identities here.
    task._order_tilt_yaws_by_grip_hinge = lambda arm_tag, yaws: yaws
    task._correct_pour_position = lambda arm_tag: None

    assert task.play_once() is True
    assert steps["n"] == 152
    assert attempts == list(pour_module._MUG_BODY_GRASP_ORDER[:2])
    assert events == [
        ("calibrate", 0),
        ("play_animation", "drink_water_1", {"attach_obj": cup, "hand_id": 1}, 0),
        ("avatar_spare", False, 0),
        ("avatar_spare", True, 1),
        (
            "play_hold_reverse",
            "drink_water_2",
            {
                "hold_at_frame": 39,
                "hold_duration": 70,
                "reverse_to_frame": 1,
                "hold_end": 30,
            },
            1,
        ),
        ("avatar_spare", False, 1),
        ("avatar_spare", True, 2),
        ("get_arm", "right", 2),
        ("open_gripper", "right", 2),
        ("logical_mug_pose", 2),
        (
            "try_grasp",
            "039_mug",
            True,
            "right",
            "manual_body_edge_topdown_posx_posz",
            "franka",
            0.08,
            2,
        ),
        (
            "try_grasp",
            "039_mug",
            True,
            "right",
            "manual_body_edge_topdown",
            "franka",
            0.08,
            2,
        ),
        ("boost_gripper", True, 2),
        ("close_gripper", "right", 102),
        ("boost_gripper", True, 152),
        ("get_ee_pose", 152),
        ("live_mug_pose", True, 152),
        ("grasp_to_world", True, 0.08, 152),
        ("sweep", "_compute_pre_pour_pose", "right", "pre-pour", None, 152),
        (
            "sweep",
            "_compute_pour_tilt_pose",
            "right",
            "pour-tilt",
            (45, 0, -45, -90, 90, -135, 135, 180),
            152,
        ),
    ]


def _configured_metric_task():
    task = PourWater({"eval_mode": True})
    task.plan_success = True
    task._eval_trigger_step = 11
    task._eval_drink1_started = True
    task._eval_drink2_queued = False
    task.target = TargetSpec(
        object=object(),
        position=np.array([0.0, 0.0, 0.9]),
        label="cup_rim",
        object_axis_fn=lambda: np.array([0.0, 1.0, 0.0]),
        success_dist_xy=0.20,
        success_dz_min=0.10,
        success_tilt_min_deg=60.0,
        success_require_no_avatar_collision=True,
    )
    metrics = {
        "object_pos": np.array([0.10, 0.0, 1.05]),
        "target_pos": np.array([0.0, 0.0, 0.9]),
        "dist_xy": 0.10,
        "dist_3d": 0.18027756377319948,
        "dz": 0.15,
        "tilt_deg": 70.0,
    }
    task._compute_target_metrics = lambda: metrics
    collision = {"enabled": True, "any_collision": False, "checks": 3}
    task.avatar_collision_summary = lambda: collision
    task.safe_distance_summary = lambda: {"enabled": False}
    task.avatar_safety_summary = lambda: {"enabled": False}
    task.avatar_safety_intervention_summary = lambda: {"enabled": False}
    return task, metrics, collision


def test_success_uses_plan_geometry_tilt_and_collision_gates() -> None:
    task, metrics, collision = _configured_metric_task()
    assert task.check_success() is True

    task.plan_success = False
    assert task.check_success() is False
    task.plan_success = True

    metrics["dist_xy"] = 0.200001
    assert task.check_success() is False
    metrics["dist_xy"] = 0.20

    metrics["dz"] = 0.099999
    assert task.check_success() is False
    metrics["dz"] = 0.10

    metrics["tilt_deg"] = 59.999
    assert task.check_success() is False
    metrics["tilt_deg"] = 60.0

    collision["enabled"] = False
    assert task.check_success() is False
    collision["enabled"] = True
    collision["any_collision"] = True
    assert task.check_success() is False
    collision["any_collision"] = False
    assert task.check_success() is True


def test_evaluate_metric_values_and_key_order_are_stable() -> None:
    task, _metrics, collision = _configured_metric_task()

    result = task.evaluate()

    assert result["success"] is True
    assert result["plan_success"] is True
    assert result["target_label"] == "cup_rim"
    assert result["target_dist_xy"] == 0.10
    assert result["target_dist_3d"] == 0.18027756377319948
    assert result["target_dz"] == 0.15
    assert result["target_tilt_deg"] == 70.0
    assert result["avatar_collision"] is collision
    assert result["eval_trigger_step"] == 11
    assert result["eval_drink1_started"] is True
    assert result["eval_drink2_queued"] is False
    assert list(result) == [
        "plan_success",
        "success",
        "target_label",
        "target_object_pos",
        "target_pos",
        "target_dist_xy",
        "target_dist_3d",
        "target_dz",
        "target_tilt_deg",
        "target_dist_xy_threshold",
        "target_dz_min_threshold",
        "target_tilt_min_deg_threshold",
        "target_require_no_avatar_collision",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
        "eval_mode",
        "eval_trigger_step",
        "eval_drink1_started",
        "eval_drink2_queued",
    ]


def test_scene_init_captures_task_facing_state_not_controller_state() -> None:
    source = PourWater({})
    source._avatar_shift_x = 0.075
    source.cup_scale = 0.088
    source.cup_rim_height = 0.089496
    source.cup_rim_radius = 0.043472
    source.mug_scale = 0.08
    source.mug_rim_radius = 0.10
    source._mug_logical_q = np.array(
        [0.0, 0.0, 0.70710678, 0.70710678],
        dtype=np.float64,
    )
    source._eval_trigger_step = 17
    source._eval_drink1_started = True
    source._eval_drink2_queued = False
    source._eval_policy_step_count = 6
    assert source.interaction_controller is not None

    scene_init = capture_scene_init(source, episode_id="pour-scene", seed=13)
    attrs = scene_init["task_attrs"]
    expected = {
        "instruction": PourWater.INSTRUCTION,
        "_avatar_shift_x": 0.075,
        "cup_scale": 0.088,
        "cup_rim_height": 0.089496,
        "cup_rim_radius": 0.043472,
        "mug_scale": 0.08,
        "mug_rim_radius": 0.10,
        "_mug_logical_q": [0.0, 0.0, 0.70710678, 0.70710678],
        "_eval_trigger_step": 17,
        "_eval_drink1_started": True,
        "_eval_drink2_queued": False,
        "_eval_policy_step_count": 6,
    }
    for name, value in expected.items():
        assert attrs[name] == value
    assert "_interaction_controller" not in attrs
    assert scene_init["avatar"] == {
        "init_pos": [-0.02, 0.91, -0.1],
        "init_rot": [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }

    restored = PourWater({
        "scene_init": {
            "id": "pour-scene",
            "source_seed": 13,
            "task_attrs": expected,
            "entities": {},
        }
    })
    replay = apply_scene_init_post_settle(restored)
    assert replay == {
        "enabled": True,
        "id": "pour-scene",
        "source_seed": 13,
        "applied_attrs": len(expected),
        "applied_entities": 0,
        "missing_entities": [],
    }
    for name, value in expected.items():
        actual = getattr(restored, name)
        if name == "_mug_logical_q":
            np.testing.assert_array_equal(actual, value)
        else:
            assert actual == value
