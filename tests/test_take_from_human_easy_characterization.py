"""Scene-free release characterization for ``take_from_human_easy``.

The legacy task receives its eval lifecycle from ``EvalModeAvatarMixin``.
These contracts deliberately admit one replacement graph: direct
``BaseTask`` inheritance with the composed ``IntentEvalModeController``.
Nothing in this module creates a Genesis scene, renderer, planner, robot, or
avatar asset.
"""

from __future__ import annotations

import importlib
import inspect
import pickle
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
    TakeFromHumanEasy as PackageTakeFromHumanEasy,
    TaskRegistration,
    TaskVariant,
    resolve_task_class,
)
import envs.tasks.take_from_human_easy as take_module
from envs.tasks.take_from_human_easy import TakeFromHumanEasy
from envs.utils import Pose
from scripts.eval_expert_set import policy_play_mode


TASK_NAME = "take_from_human_easy"


def _uses_legacy_eval_mixin() -> bool:
    return EvalModeAvatarMixin in TakeFromHumanEasy.__mro__


def test_public_identity_signature_import_and_pickle_surface():
    assert TakeFromHumanEasy.__module__ == "envs.tasks.take_from_human_easy"
    assert TakeFromHumanEasy.__name__ == "TakeFromHumanEasy"
    assert PackageTakeFromHumanEasy is TakeFromHumanEasy
    assert importlib.import_module(
        "envs.tasks.take_from_human_easy"
    ).TakeFromHumanEasy is TakeFromHumanEasy
    assert pickle.loads(pickle.dumps(TakeFromHumanEasy)) is TakeFromHumanEasy
    assert issubclass(TakeFromHumanEasy, BaseTask)

    assert TakeFromHumanEasy.INSTRUCTION == (
        "take the {object} from the person and place it on the table"
    )
    # The apple is not in human_transfer_pool; the representative
    # first-pool-object is the rubik's cube.
    assert TakeFromHumanEasy.default_instruction() == (
        "take the rubik's cube from the person and place it on the table"
    )
    assert TakeFromHumanEasy.OBJECT_SET == "human_transfer_pool"
    assert TakeFromHumanEasy.use_avatar is True

    signature = inspect.signature(TakeFromHumanEasy)
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
        "_place_eval_apple_support_and_detach",
    ):
        assert callable(getattr(TakeFromHumanEasy, method_name))


def test_constructor_copies_config_and_preserves_release_fields():
    assert TakeFromHumanEasy().config["collision_ignore_capsule_prefixes"] == (
        "r_palm",
        "r_thumb",
        "r_index",
        "r_middle",
        "r_ring",
        "r_pinky",
    )

    config = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    original = {
        "renderer": "rasterizer",
        "avatar": {"frame_ratio": 3.0},
    }
    task = TakeFromHumanEasy(config)
    assert config == original
    assert task.config is not config
    assert task.config["avatar"] is not config["avatar"]
    assert task.obj_name is None
    assert task.obj_model_id == 0
    assert task.instruction == TakeFromHumanEasy.INSTRUCTION


def test_only_legacy_mixin_or_target_intent_controller_graph_is_allowed():
    assert interaction_api.IntentEvalModeController is IntentEvalModeController
    task = TakeFromHumanEasy({})

    if _uses_legacy_eval_mixin():
        assert TakeFromHumanEasy.__bases__ == (EvalModeAvatarMixin, BaseTask)
        assert isinstance(task.interaction_controller, LegacyInteractionController)
        assert task.interaction_controller.mode is InteractionMode.LEGACY
        return

    # The carry legs use TopDownPickPlaceMixin's guarded
    # motion primitives, so the mixin is a base (interaction
    # controller graph unchanged).
    from envs.manipulation import TopDownPickPlaceMixin
    assert TakeFromHumanEasy.__bases__ == (TopDownPickPlaceMixin, BaseTask)
    assert EvalModeAvatarMixin not in TakeFromHumanEasy.__mro__
    assert "create_interaction_controller" in TakeFromHumanEasy.__dict__
    controller = task.interaction_controller
    assert isinstance(controller, IntentEvalModeController)
    assert controller.mode is InteractionMode.INTENT
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False


def test_registry_intent_identity_and_no_human_rejection_are_stable():
    descriptor = TASK_DESCRIPTOR_MAP[TASK_NAME]
    assert descriptor.task_cls is TakeFromHumanEasy
    assert descriptor.family == "take_from_human"
    assert descriptor.variant is TaskVariant.INTENT
    assert descriptor.registration is TaskRegistration.CANONICAL
    assert descriptor.variant_label is None
    assert descriptor.no_human_name is None
    assert descriptor.no_human_canonical is False
    assert descriptor.instruction_template == TakeFromHumanEasy.INSTRUCTION
    assert descriptor.instruction == TakeFromHumanEasy.default_instruction()

    assert TASK_MAP[TASK_NAME] is TakeFromHumanEasy
    assert resolve_task_class(TASK_NAME) == (TASK_NAME, TakeFromHumanEasy)
    assert TASK_NAME not in NO_HUMAN_CANONICAL_TASK_MAP
    assert TASK_NAME not in NO_HUMAN_VARIANT_ALIASES
    assert TASK_NAME not in NO_HUMAN_TASK_MAP
    with pytest.raises(ValueError, match="has no no-human variant"):
        resolve_task_class(TASK_NAME, no_human=True)


class _FakeEntity:
    def __init__(self, pos, quat, events=None, label="entity"):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)
        self._events = events
        self._label = label

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return self._quat.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)
        if self._events is not None:
            self._events.append((f"{self._label}_set_pos", tuple(self._pos)))


class _FakeActor:
    def __init__(self, pos, quat, *, name, events=None):
        self.entity = _FakeEntity(pos, quat, events, name)
        self.name = name
        self.config = {"center": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


class _FakeSupport:
    def __init__(self, events=None):
        self.events = events
        self.positions = []

    def set_pos(self, pos):
        value = np.asarray(pos, dtype=np.float64)
        self.positions.append(value)
        if self.events is not None:
            self.events.append(("support_set_pos", tuple(value)))


def _capture_layout(monkeypatch, config, seed):
    events = []

    def load_object(_scene, pose, object_name, **kwargs):
        events.append((
            "load_object",
            object_name,
            dict(kwargs),
            tuple(float(value) for value in pose.p),
            tuple(float(value) for value in pose.q),
        ))
        return _FakeActor(pose.p, pose.q, name=object_name)

    def create_primitive(_scene, primitive, pose, **kwargs):
        events.append((
            "create_primitive",
            primitive,
            dict(kwargs),
            tuple(float(value) for value in pose.p),
            tuple(float(value) for value in pose.q),
        ))
        return _FakeSupport()

    monkeypatch.setattr(take_module, "load_object", load_object)
    monkeypatch.setattr(take_module, "create_primitive", create_primitive)
    monkeypatch.setattr(
        TakeFromHumanEasy,
        "_get_object_radius",
        staticmethod(lambda _name, _model=0: 0.05),
    )

    task = TakeFromHumanEasy(dict(config))
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
        "expected_plate",
        "expected_avatar_pos",
        "expected_avatar_rot",
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
            (0.2872033759818312, -0.3569621267255161),
            (0.24266321025719453, 0.768, -0.053),
            (
                (0.04698425568002, 0.9988956300425959, 0.0),
                (-0.9988956300425959, 0.04698425568002, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                0.4236547993389047,
                0.6458941130666561,
                0.4375872112626925,
                0.8917730007820798,
                0.9636627605010293,
                0.3834415188257777,
                0.7917250380826646,
                0.5288949197529045,
            ),
        ),
        (
            {},
            1,
            "073_rubikscube",
            1,
            (0.2542555011756435, -0.35593510131156836),
            (0.09802744995616278, 0.768, -0.053),
            (
                (-0.20552178665871765, 0.9786525405927318, 0.0),
                (-0.9786525405927318, -0.20552178665871765, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                0.14675589081711304,
                0.0923385947687978,
                0.1862602113776709,
                0.34556072704304774,
                0.39676747423066994,
                0.538816734003357,
                0.4191945144032948,
                0.6852195003967595,
            ),
        ),
        (
            {"object_name": "073_rubikscube"},
            42,
            "073_rubikscube",
            1,
            (0.24363502971184062, -0.3098571387180168),
            (0.2736785460347372, 0.768, -0.053),
            (
                (0.10313122433426682, 0.9946677588860087, 0.0),
                (-0.9946677588860087, 0.10313122433426682, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                0.15601864044243652,
                0.15599452033620265,
                0.05808361216819946,
                0.8661761457749352,
                0.6011150117432088,
                0.7080725777960455,
                0.020584494295802447,
                0.9699098521619943,
            ),
        ),
        (
            {"object_name": "073_rubikscube", "model_id": 7},
            7,
            "073_rubikscube",
            7,
            (0.1690770723434893, -0.3440162415519771),
            (0.20321821554581443, 0.768, -0.053),
            (
                (0.23188220976068813, 0.972743872145438, 0.0),
                (-0.972743872145438, 0.23188220976068813, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                0.9779895119966027,
                0.5384958704104337,
                0.5011204636599379,
                0.07205113335976154,
                0.26843898010187117,
                0.49988250082555996,
                0.6792299961209405,
                0.8037390361043755,
            ),
        ),
    ),
)
def test_seeded_actor_layout_and_global_numpy_tail(
    monkeypatch,
    config,
    seed,
    expected_name,
    expected_model,
    expected_plate,
    expected_avatar_pos,
    expected_avatar_rot,
    expected_tail,
):
    task, events, tail = _capture_layout(monkeypatch, config, seed)

    assert task.obj_name == expected_name
    assert task.obj_model_id == expected_model
    display = "apple" if expected_name == "035_apple" else "rubik's cube"
    assert task.instruction == (
        f"take the {display} from the person and place it on the table"
    )
    np.testing.assert_allclose(task._plate_xy, expected_plate, rtol=0.0, atol=5e-15)
    np.testing.assert_allclose(
        task.avatar_init_pos, expected_avatar_pos, rtol=0.0, atol=5e-15
    )
    np.testing.assert_allclose(
        task.avatar_init_rot, expected_avatar_rot, rtol=0.0, atol=5e-15
    )
    np.testing.assert_allclose(
        task.place_pos,
        (*expected_plate, 0.845),
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(tail, expected_tail, rtol=0.0, atol=5e-15)

    assert task.target.object is task.apple_actor
    assert task.target.position is task.place_pos
    assert task.target.label == "plate"
    assert task.target.success_dist_xy == 0.12
    assert task.target.success_dist_3d is None
    assert task.target.success_require_no_avatar_collision is False
    np.testing.assert_allclose(
        task.target.object_pos_fn(), (0.0, 0.768, 0.69), rtol=0.0, atol=0.0
    )
    assert task._clutter_actors == []
    assert task._eval_apple_support_active is False
    np.testing.assert_allclose(
        task._eval_object_support_half_size,
        (0.04, 0.04, 0.003),
        rtol=0.0,
        atol=2e-17,
    )

    assert [event[0] for event in events] == [
        "load_object",
        "create_primitive",
        "load_object",
    ]
    apple, support, plate = events
    assert apple[1] == expected_name
    assert apple[2] == {
        "model_id": expected_model,
        "convex": True,
        "is_static": False,
        "friction": 4.0,
    }
    assert apple[3:] == ((0.0, 0.768, 0.69), (1.0, 0.0, 0.0, 0.0))
    assert support[1] == "box"
    assert support[2]["color"] == (0.0, 0.0, 0.0)
    assert support[2]["is_static"] is True
    assert support[2]["collision"] is True
    assert support[2]["visual"] is False
    np.testing.assert_allclose(
        support[2]["size"]["half_size"], (0.04, 0.04, 0.003), atol=2e-17
    )
    assert support[3:] == ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 0.0))
    assert plate[1] == "003_plate"
    assert plate[2] == {"model_id": 0, "convex": True, "is_static": False}
    np.testing.assert_allclose(plate[3], (*expected_plate, 0.785), atol=5e-15)
    np.testing.assert_allclose(
        plate[4],
        (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
        rtol=0.0,
        atol=0.0,
    )


class _ResetSkin:
    def __init__(self, events):
        self.events = events

    def get_global_translation(self, name):
        self.events.append(("skin_translation", name))
        values = {
            "RightHand": np.array([[1.0, 2.0, 3.0]]),
            "RightHandMiddle1": np.array([[3.0, 4.0, 5.0]]),
        }
        return values[name]


class _ResetAvatar:
    def __init__(self, events):
        self.events = events
        self.robot = SimpleNamespace(
            skin=_ResetSkin(events),
            _get_hand_frame=self._get_hand_frame,
        )
        self._spare = iter((False, True))

    def _get_hand_frame(self, hand_id):
        self.events.append(("hand_frame", hand_id))
        return np.zeros(3), np.eye(3)

    def attach_object_to_hand(self, entity, hand_id):
        self.events.append(("attach", entity, hand_id))

    def play_animation(self, name):
        self.events.append(("play", name))

    def spare(self):
        return next(self._spare)


def _run_eval_reset_boundary(task, events, monkeypatch):
    observation = {"phase": "initial"}
    if _uses_legacy_eval_mixin():
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


def test_eval_reset_is_post_observation_then_outer_warmup(monkeypatch):
    events = []
    task = TakeFromHumanEasy({
        "eval_mode": True,
        "eval_detach_apple_to_support": False,
        "eval_avatar_post_warmup_settle_steps": 2,
    })
    task.avatar = _ResetAvatar(events)
    task.apple_actor = _FakeActor(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        name="apple",
        events=events,
    )
    task._boost_arm_pd = lambda arm: events.append(("boost", arm))
    task._eval_policy_step_count = 99
    task.step_sim = lambda: events.append(("warmup_sim_step",))
    task.get_obs = lambda: events.append(("warmup_observation",)) or {
        "phase": "warm"
    }

    assert _run_eval_reset_boundary(task, events, monkeypatch) == {
        "phase": "initial"
    }
    assert task._eval_policy_step_count == 0
    assert task._eval_apple_attached_to_avatar is True
    assert task._eval_handoff_closed_steps == 0
    assert task._eval_handoff_debug is None
    np.testing.assert_allclose(
        task.apple_actor.entity.get_pos(),
        (2.0, 3.07, 4.03),
        rtol=0.0,
        atol=5e-15,
    )
    assert task.eval_pre_policy_warmup() == {"phase": "warm"}

    assert events == [
        ("initial_observation", 17),
        ("boost", "right"),
        ("skin_translation", "RightHand"),
        ("skin_translation", "RightHandMiddle1"),
        ("hand_frame", 1),
        ("apple_set_pos", (2.0, 3.07, 4.03)),
        ("attach", task.apple_actor.entity, 1),
        ("play", "deliver_to_human"),
        ("warmup_sim_step",),
        ("warmup_sim_step",),
        ("warmup_sim_step",),
        ("warmup_observation",),
    ]


@pytest.mark.parametrize(
    ("eval_mode", "has_avatar"),
    ((False, True), (True, False)),
)
def test_eval_reset_inactive_without_both_gates(
    eval_mode,
    has_avatar,
    monkeypatch,
):
    events = []
    task = TakeFromHumanEasy({"eval_mode": eval_mode})
    task.avatar = object() if has_avatar else None
    task._boost_arm_pd = lambda arm: events.append(("boost", arm))
    task._eval_policy_step_count = 91

    assert _run_eval_reset_boundary(task, events, monkeypatch) == {
        "phase": "initial"
    }
    assert task._eval_policy_step_count == 91
    assert events == [("initial_observation", 17)]


def test_warmup_places_support_after_animation_then_settles():
    events = []
    task = TakeFromHumanEasy({
        "eval_mode": True,
        "eval_avatar_post_warmup_settle_steps": 2,
        "eval_apple_support_settle_steps": 3,
        # Legacy opt-in path; the default keeps the object in hand.
        "eval_detach_apple_to_support": True,
    })
    states = iter((False, True))
    task.avatar = SimpleNamespace(spare=lambda: next(states))
    task.step_sim = lambda: events.append("sim")
    task._place_eval_apple_support_and_detach = lambda: events.append("support")
    task.get_obs = lambda: events.append("obs") or "observation"

    assert task.eval_pre_policy_warmup() == "observation"
    assert events == [
        "sim",
        "sim",
        "sim",
        "support",
        "sim",
        "sim",
        "sim",
        "obs",
    ]


def test_support_placement_detaches_once_and_publishes_state(monkeypatch):
    events = []
    task = TakeFromHumanEasy({})
    support = _FakeSupport(events)
    task._eval_apple_support = support
    task._eval_object_support_half_size = (0.04, 0.04, 0.003)
    task._eval_apple_support_active = False
    task._eval_apple_attached_to_avatar = True
    task.obj_name = "035_apple"
    task.obj_model_id = 1
    task.apple_actor = object()
    task.avatar = SimpleNamespace(
        robot=SimpleNamespace(
            detach_object=lambda hand: events.append(("detach", hand))
        )
    )
    monkeypatch.setattr(
        task,
        "_get_object_center_world",
        lambda _actor: np.array([0.25, -0.10, 0.90]),
    )
    monkeypatch.setattr(
        task,
        "_get_object_radius",
        lambda _name, _model: 0.05,
    )

    task._place_eval_apple_support_and_detach()
    task._place_eval_apple_support_and_detach()

    assert events == [
        ("support_set_pos", (0.25, -0.10, 0.847)),
        ("detach", 1),
    ]
    np.testing.assert_allclose(support.positions[0], (0.25, -0.10, 0.847))
    assert task._eval_apple_attached_to_avatar is False
    assert task._eval_apple_support_active is True


class _PolicyArm:
    def __init__(self, ee_pos, gripper_val):
        self.ee_pos = np.asarray(ee_pos, dtype=np.float64)
        self.gripper_val = float(gripper_val)

    def get_ee_pose(self):
        return self.ee_pos.copy()


def test_active_policy_handoff_is_one_based_and_precedes_early_success():
    events = []
    task = TakeFromHumanEasy({
        "eval_mode": True,
        "eval_handoff_closed_steps": 2,
    })
    arm = _PolicyArm((0.03, 0.04, 0.0), 0.45)
    task.robot = SimpleNamespace(get_arm=lambda name: arm)
    task.avatar = SimpleNamespace(
        robot=SimpleNamespace(
            detach_object=lambda hand: events.append(("detach", hand))
        )
    )
    task.apple_actor = object()
    task.obj_name = "035_apple"
    task.obj_model_id = 1
    task._get_object_center_world = lambda _actor: np.zeros(3)
    task._eval_apple_attached_to_avatar = True
    task._eval_handoff_closed_steps = 0
    task._eval_handoff_debug = None
    task._eval_policy_step_count = 0
    task.eval_success = True
    task.take_action_cnt = 0
    task.get_obs = lambda: events.append(("observation",)) or "obs"

    assert task.take_action(object(), action_type="qpos_abs") == "obs"
    assert task._eval_policy_step_count == 1
    assert task._eval_handoff_closed_steps == 1
    assert task._eval_apple_attached_to_avatar is True
    assert task.take_action(object(), action_type="qpos_abs") == "obs"

    assert task._eval_policy_step_count == 2
    assert task.take_action_cnt == 0
    assert task._eval_apple_attached_to_avatar is False
    assert events == [
        ("observation",),
        ("detach", 1),
        ("observation",),
    ]
    assert task._eval_handoff_debug == {
        "step_idx": 2,
        "gripper": 0.45,
        "ee_pos": [0.03, 0.04, 0.0],
        "apple_pos": [0.0, 0.0, 0.0],
        "ee_apple_dist": 0.05,
        "closed_steps": 2,
        "required_closed_steps": 2,
    }


def test_eval_step_far_resets_cadence_and_detached_state_is_noop():
    events = []
    task = TakeFromHumanEasy({})
    arm = _PolicyArm((1.0, 0.0, 0.0), 0.35)
    task.robot = SimpleNamespace(get_arm=lambda name: arm)
    task.avatar = SimpleNamespace(
        robot=SimpleNamespace(
            detach_object=lambda hand: events.append(("detach", hand))
        )
    )
    task.apple_actor = object()
    task.obj_name = "073_rubikscube"
    task._get_object_center_world = lambda _actor: np.zeros(3)
    task._eval_apple_attached_to_avatar = True
    task._eval_handoff_closed_steps = 3

    task._eval_at_step(11)
    assert task._eval_handoff_closed_steps == 0
    assert task._eval_apple_attached_to_avatar is True
    assert events == []

    task._eval_apple_attached_to_avatar = False
    task.robot = SimpleNamespace(
        get_arm=lambda _name: pytest.fail("detached handoff queried the arm")
    )
    task._eval_at_step(12)
    assert events == []


def test_inactive_policy_call_does_not_create_eval_counter():
    task = TakeFromHumanEasy({"eval_mode": True})
    task.avatar = None
    task.eval_success = True
    task.get_obs = lambda: "obs"
    task.__dict__.pop("_eval_policy_step_count", None)

    assert task.take_action(object()) == "obs"
    assert not hasattr(task, "_eval_policy_step_count")


def test_expert_routing_stays_on_public_scripted_rollout():
    assert policy_play_mode(TASK_NAME, TakeFromHumanEasy, "expert") == (
        "play_blind_once",
        None,
    )
    assert policy_play_mode(
        TASK_NAME,
        TakeFromHumanEasy,
        "expert_full_state",
    ) == ("play_once", None)


class _ExpertArmEntity:
    def __init__(self, events):
        self.events = events

    def get_qpos(self):
        return np.array([0.1, 0.2, 0.3])

    def control_dofs_position(self, qpos):
        self.events.append(("hold_arm", tuple(np.asarray(qpos))))


class _ExpertArm:
    def __init__(self, events):
        self.tcp_offset = object()
        self.origin_pose = SimpleNamespace(p=np.array([0.0, -0.3, 0.75]))
        self.gripper_val = 0.55
        self.entity = _ExpertArmEntity(events)

    def get_ee_pose(self):
        # The pre-release convergence wait reads the EE; report the
        # scenario's fixed release pose so the wait converges immediately
        # (no extra sim events in the choreography).
        return np.array([0.25, -0.40, 0.985, 1.0, 0.0, 0.0, 0.0])


class _ExpertAvatar:
    def __init__(self, events):
        self.events = events
        self._spare = iter((False, True, False, True))
        self.robot = SimpleNamespace(
            skin=_ResetSkin(events),
            _get_hand_frame=lambda hand: (
                events.append(("hand_frame", hand)) or np.zeros(3),
                np.eye(3),
            ),
            detach_object=lambda hand: events.append(("detach", hand)),
        )

    def attach_object_to_hand(self, entity, hand_id):
        self.events.append(("attach", entity, hand_id))

    def play_animation(self, name):
        self.events.append(("play", name))

    def play_animation_reverse(self, name):
        self.events.append(("reverse", name))

    def spare(self):
        return next(self._spare)


@pytest.mark.parametrize("settled", [True, False])
def test_scripted_expert_preserves_handoff_grasp_transport_release_choreography(
    monkeypatch, settled,
):
    events = []
    task = TakeFromHumanEasy({})
    arm = _ExpertArm(events)
    task.robot = SimpleNamespace(
        get_arm=lambda name: events.append(("get_arm", name)) or arm,
        set_gripper=lambda value, name: events.append(
            ("hold_gripper", float(value), name)
        ),
    )
    task.avatar = _ExpertAvatar(events)
    task.apple_actor = _FakeActor(
        (0.0, 0.768, 0.69),
        (1.0, 0.0, 0.0, 0.0),
        name="apple",
        events=events,
    )
    task.obj_name = "035_apple"
    task.obj_model_id = 1
    task.place_pos = np.array([0.25, -0.40, 0.845])
    task.vla_recorder = SimpleNamespace(
        mark_episode_end=lambda name: events.append(("mark", name))
    )
    task.TABLE_TOP_Z = 0.765  # the carry-z clamp reads it
    task.AVATAR_HAND_SETTLE_STEPS = 1
    task.GRASP_SETTLE_STEPS = 1
    # The fake scene uses symbolic far-away coordinates; disable the
    # handover reach guard so the choreography stays the single-delivery
    # happy path this test characterizes.
    task.HANDOVER_REACH_GUARD = float("inf")
    task.POST_DETACH_SETTLE_STEPS = 1
    task.POST_LIFT_GRIP_HOLD_STEPS = 1
    task.RELEASE_SETTLE_STEPS = 1
    task.FINAL_SETTLE_STEPS = 1
    task._boost_arm_pd = lambda name: events.append(("boost", name))
    task.open_gripper = lambda name: events.append(("open", name))
    task.set_gripper = lambda value, name, num_steps: events.append(
        ("close", float(value), name, num_steps)
    )
    task.step_sim = lambda: events.append(("sim",))
    task._settle_released_object = lambda: events.append(("settled_release",)) or settled

    class _LinkPose:
        def __init__(self, pose):
            self.pose = pose

        def to_pose7(self):
            return np.concatenate([self.pose.p, self.pose.q])

    monkeypatch.setattr(
        take_module,
        "tcp_to_link_pose",
        lambda pose, _offset: _LinkPose(pose),
    )
    task.move_and_execute = lambda pose7, name: events.append(
        ("plan", tuple(np.asarray(pose7)), name)
    ) or object()
    pre = Pose((0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))
    grasp = Pose((0.1, 0.2, 0.7), (1.0, 0.0, 0.0, 0.0))
    task._compute_grasp_poses = lambda name: events.append(
        ("compute_grasp", name)
    ) or (pre, grasp)
    cartesian_calls = []

    def move_cartesian(start, end, name, n_steps, sim_per_step):
        cartesian_calls.append((
            np.asarray(start, dtype=float).copy(),
            np.asarray(end, dtype=float).copy(),
            name,
            n_steps,
            sim_per_step,
        ))
        events.append(("cartesian", len(cartesian_calls)))
        if len(cartesian_calls) == 1:
            task.apple_actor.entity._pos[2] += 0.04

    task._move_cartesian = move_cartesian
    # The stalled-lift finish reads the live TCP; the fake
    # arm reports a fixed pose, so stub it to keep the frozen choreography.
    task._finish_stalled_lift = lambda *a, **k: None

    screw_calls = []

    def move_screw(pos, name, tcp_yaw=0.0):
        screw_calls.append(np.asarray(pos, dtype=float).copy())
        events.append(("screw", len(screw_calls)))
        return True

    task._move_screw = move_screw

    assert task.play_once() is settled

    names = [event[0] for event in events]
    assert names == [
        "boost",
        "get_arm",
        "skin_translation",
        "skin_translation",
        "hand_frame",
        "apple_set_pos",
        "attach",
        "play",
        "sim",
        # Post-delivery handover reach check re-measures the palm target.
        "skin_translation",
        "skin_translation",
        "hand_frame",
        "sim",
        "open",
        "plan",
        # Feasibility pre-flight: one extra grasp-pose
        # computation probes the reach clamp before the committed moves.
        "compute_grasp",
        "compute_grasp",
        "plan",
        "compute_grasp",
        "plan",
        "sim",
        "close",
        "detach",
        "sim",
        # Only the lift is Cartesian; the three carry legs
        # ride the mixin plan_screw (branch-continuous) with the guarded
        # seeded fallback.
        "cartesian",
        # Post-lift grip stabilization hold (1 step in the fake).
        "hold_arm",
        "hold_gripper",
        "sim",
        # The human retracts BEFORE the transport legs — close
        # handovers had the first carry leg sweep the object through the
        # avatar hand still parked at the exchange pose.
        "reverse",
        "sim",
        "screw",
        "screw",
        "screw",
        "open",
        "hold_gripper",
        "sim",
        "hold_arm",
        "hold_gripper",
        "sim",
        "settled_release",
        "mark",
    ]
    assert events[7] == ("play", "deliver_to_human")
    assert events[21] == ("close", 0.45, "right", 100)
    assert events[22] == ("detach", 1)
    assert events[28] == ("reverse", "deliver_to_human")
    assert events[-1] == (
        "mark", "take_from_human_easy_post_release" if settled
        else "take_from_human_easy_release_unsettled",
    )
    assert len(cartesian_calls) == 1
    assert cartesian_calls[0][3:] == (
        task.LIFT_CARTESIAN_STEPS, task.CARTESIAN_SIM_PER_STEP,
    )
    assert len(screw_calls) == 3
    # carry_z = min(lift z, TABLE_TOP_Z + 0.38) = 1.145 for the fake scene
    np.testing.assert_allclose(screw_calls[0], (0.0, 0.05, 1.145))
    np.testing.assert_allclose(screw_calls[1][:2], task.place_pos[:2])
    np.testing.assert_allclose(screw_calls[2], (0.25, -0.40, 0.985))


def _configured_metric_task():
    task = TakeFromHumanEasy({})
    task.obj_name = "035_apple"
    task.obj_model_id = 1
    task.plan_success = True
    task._eval_apple_attached_to_avatar = False
    task._eval_apple_support_active = True
    task._eval_handoff_debug = {
        "step_idx": 4,
        "closed_steps": 4,
        "required_closed_steps": 4,
    }
    task.target = TargetSpec(
        object=object(),
        position=np.array([0.25, -0.40, 0.845]),
        label="plate",
        success_dist_xy=0.12,
    )
    metrics = {
        "object_pos": np.array([0.20, -0.30, 0.86]),
        "target_pos": np.array([0.25, -0.40, 0.845]),
        "dist_xy": float(np.hypot(0.05, 0.10)),
        "dist_3d": float(np.linalg.norm([-0.05, 0.10, 0.015])),
        "dz": 0.015,
    }
    task._compute_target_metrics = lambda: metrics
    task.avatar_collision_summary = lambda: {"enabled": False}
    task.safe_distance_summary = lambda: {"enabled": False}
    task.avatar_safety_summary = lambda: {"enabled": False}
    task.avatar_safety_intervention_summary = lambda: {"enabled": False}
    return task, metrics


def test_success_uses_plan_and_live_plate_xy_threshold():
    task, metrics = _configured_metric_task()
    assert task.check_success() is True

    task.plan_success = False
    assert task.check_success() is False
    task.plan_success = True
    task._compute_target_metrics = lambda: None
    assert task.check_success() is False
    task._compute_target_metrics = lambda: metrics

    metrics["dist_xy"] = 0.120001
    assert task.check_success() is False
    metrics["dist_xy"] = 0.12
    assert task.check_success() is True


def test_evaluate_metric_values_and_key_order_are_stable():
    task, _metrics = _configured_metric_task()
    result = task.evaluate()

    assert result["success"] is True
    assert result["object_name"] == "035_apple"
    assert result["object_model_id"] == 1
    assert result["eval_handoff_debug"]["step_idx"] == 4
    assert result["eval_apple_attached_to_avatar"] is False
    assert result["eval_apple_support_active"] is True
    assert list(result) == [
        "plan_success",
        "success",
        "target_label",
        "target_object_pos",
        "target_pos",
        "target_dist_xy",
        "target_dist_3d",
        "target_dz",
        "target_dist_xy_threshold",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
        "object_name",
        "object_model_id",
        "eval_handoff_debug",
        "eval_apple_attached_to_avatar",
        "eval_apple_support_active",
    ]
    assert "eval_policy_step_count" not in result


def test_scene_init_captures_task_state_not_controller_and_replays_attrs():
    source = TakeFromHumanEasy({})
    source.instruction = (
        "take the rubik's cube from the person and place it on the table"
    )
    source.obj_name = "073_rubikscube"
    source.obj_model_id = 1
    source._plate_xy = np.array([0.31, -0.42])
    source.place_pos = np.array([0.31, -0.42, 0.845])
    source.avatar_init_pos = np.array([0.27, 0.768, -0.053])
    source.avatar_init_rot = np.array(
        [[0.1, 0.995, 0.0], [-0.995, 0.1, 0.0], [0.0, 0.0, 1.0]]
    )
    source._eval_object_support_half_size = (0.04, 0.04, 0.003)
    source._eval_apple_support_active = True
    source._eval_apple_attached_to_avatar = False
    source._eval_handoff_closed_steps = 4
    source._eval_handoff_debug = {"step_idx": 4, "closed_steps": 4}
    source._eval_policy_step_count = 7
    source._avatar_head_hand_aligned = True
    source._avatar_head_hand_yaw_deg = -12.5
    source._clutter_actors = []
    assert source.interaction_controller is not None

    scene_init = capture_scene_init(source, episode_id="take-scene", seed=17)
    attrs = scene_init["task_attrs"]
    expected = {
        "instruction": source.instruction,
        "obj_name": "073_rubikscube",
        "obj_model_id": 1,
        "_plate_xy": [0.31, -0.42],
        "place_pos": [0.31, -0.42, 0.845],
        "avatar_init_pos": [0.27, 0.768, -0.053],
        "avatar_init_rot": [
            [0.1, 0.995, 0.0],
            [-0.995, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "_eval_object_support_half_size": [0.04, 0.04, 0.003],
        "_eval_apple_support_active": True,
        "_eval_apple_attached_to_avatar": False,
        "_eval_handoff_closed_steps": 4,
        "_eval_handoff_debug": {"step_idx": 4, "closed_steps": 4},
        "_eval_policy_step_count": 7,
        "_avatar_head_hand_aligned": True,
        "_avatar_head_hand_yaw_deg": -12.5,
        "_clutter_actors": [],
    }
    for name, value in expected.items():
        assert attrs[name] == value
    assert "_interaction_controller" not in attrs
    assert scene_init["avatar"] == {
        "init_pos": [0.27, 0.768, -0.053],
        "init_rot": [
            [0.1, 0.995, 0.0],
            [-0.995, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }

    restored = TakeFromHumanEasy({
        "scene_init": {
            "id": "take-scene",
            "source_seed": 17,
            "task_attrs": expected,
            "entities": {},
        }
    })
    replay = apply_scene_init_post_settle(restored)
    assert replay == {
        "enabled": True,
        "id": "take-scene",
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
