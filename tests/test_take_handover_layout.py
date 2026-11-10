"""Scene-free contracts for the original-table coordinated handover layout."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.interaction.eval_mode import IntentEvalModeController
from envs.scene_init import apply_scene_init_post_settle, capture_scene_init
from envs.tasks.take_from_human_easy import TakeFromHumanEasy
from envs.utils import Pose


class _Entity:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.quaternion = np.array([1., 0., 0., 0.])
        self.entity = self

    def get_pos(self):
        return self.position.copy()

    def set_pos(self, position):
        self.position = np.array(position, dtype=float)

    def get_quat(self):
        return self.quaternion.copy()

    def set_quat(self, quaternion):
        self.quaternion = np.array(quaternion, dtype=float)


class _Avatar:
    def __init__(self, *, far=False):
        self.robot = SimpleNamespace(global_trans=np.array([0., .8, .9]),
                                     global_rot=np.eye(3))
        self.robot.skin = SimpleNamespace(get_global_translation=self._joint)
        self.robot._get_hand_frame = lambda _: (None, np.eye(3))
        self.frame = 0
        self.plays = 0
        self.resets = []
        self.far = far

    def _joint(self, name):
        # Exact midpoint offset cancels the task's configured local offset.
        midpoint = self.robot.global_trans + [0., .3 if self.far else -.4, 0.]
        midpoint -= TakeFromHumanEasy.HAND_APPLE_OFFSET_LOCAL
        return np.array([midpoint + ([.02, 0., 0.] if name == "RightHand"
                                    else [-.02, 0., 0.])])

    def play_animation(self, name):
        assert name == "deliver_to_human"
        self.plays += 1
        self.frame = 0

    def spare(self):
        return self.frame == 2

    def step(self):
        self.frame += 1

    def reset(self, position, rotation):
        self.robot.global_trans = np.array(position, dtype=float)
        self.robot.global_rot = np.array(rotation, dtype=float)
        self.resets.append((self.robot.global_trans.copy(), self.robot.global_rot.copy()))
        self.frame = 0


def _task(*, far=False, body_bias=0.):
    task = TakeFromHumanEasy.__new__(TakeFromHumanEasy)
    task.config = {}
    task.avatar = _Avatar(far=far)
    task.avatar_init_pos = task.avatar.robot.global_trans.copy()
    task.avatar_init_rot = np.eye(3)
    task._table_center_xy = np.array([0., -.1])
    task._table_half_size_xy = np.array([.494, .654])
    task.TABLE_TOP_Z = .765
    task.table = _Entity([0., -.1, .74])
    task.apple_actor = _Entity([.1, .1, .9])
    task.plate_actor = _Entity([.1, .2, .8])
    task.apple_pose = Pose(task.apple_actor.get_pos())
    task.place_pos = np.array([.1, .2, .85])
    task._plate_xy = task.place_pos[:2].copy()
    task.target = SimpleNamespace(position=task.place_pos.copy())
    observed_fronts = []
    updates = []

    def capsules():
        # Minimum occurs mid-animation, not at the final handover frame.
        front = ([.6, .3, .5][task.avatar.frame] + body_bias
                 + task.avatar.robot.global_trans[1] - .8)
        observed_fronts.append(front)
        return [("torso_lower", [0., front + .1, .5], [0., front + .2, .8], .1),
                ("r_forearm", [0., -10., .8], [0., -9., .8], .2)]

    task.avatar_collider = SimpleNamespace(current_capsules=capsules,
                                           update=lambda: updates.append(True))
    bases = []
    planner = SimpleNamespace(_base_pose=np.array([0., 0., .8, 1., 0., 0., 0.]),
                              _mplib=SimpleNamespace(set_base_pose=lambda p: bases.append(p.copy())),
                              _to_mplib_pose=lambda p: p)
    arm = SimpleNamespace(entity=_Entity([0., 0., .8]), origin_pose=Pose([0., 0., .8]),
                          planner=planner, joints=np.array([.1, .2, .3]))
    origins = []
    task.robot = SimpleNamespace(get_arm=lambda tag: arm if tag == "right" else None,
                                 set_origin_endpose=lambda: origins.append(True))
    return task, SimpleNamespace(arm=arm, bases=bases, origins=origins,
                                  updates=updates, observed_fronts=observed_fronts)


def test_original_table_geometry_is_restored():
    task, _ = _task()
    task._create_rectangular_table = lambda **kwargs: kwargs
    result = task._create_table()
    assert result["half_size"] == pytest.approx((.494, .654))
    assert result["center_xy"] == pytest.approx((0., -.1))
    assert result["leg_inset"] == pytest.approx((.044, .054))
    assert result["table_height"] == .74


def test_reset_controller_keeps_intent_lifecycle_and_initializes_layout():
    task, _ = _task()
    controller = task.create_interaction_controller()
    assert isinstance(controller, IntentEvalModeController)
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False
    controller.on_base_reset_ready(task)
    assert task._handover_layout_offset[1] == pytest.approx(.274)


def test_hand_target_uses_rotated_offset_and_joint_midpoint():
    task, _ = _task()
    axes = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    task.avatar.robot._get_hand_frame = lambda _: (None, axes)
    wrist = task.avatar._joint("RightHand")[0]
    middle = task.avatar._joint("RightHandMiddle1")[0]
    np.testing.assert_allclose(task._hand_object_target(),
                               (wrist + middle) / 2 + axes @ task.HAND_APPLE_OFFSET_LOCAL)


@pytest.mark.parametrize("name", ["torso_upper", "l_thigh", "r_shin"])
def test_body_front_includes_capsule_radius_and_excludes_arms(name):
    task, _ = _task()
    task.avatar_collider.current_capsules = lambda: [
        (name, [0., .5, 0.], [0., .4, 0.], .1),
        ("r_forearm", [0., -10., 0.], [0., -10., 0.], .2),
        ("r_palm", [0., -20., 0.], [0., -20., 0.], .2)]
    assert task._body_min_y() == pytest.approx(.3)


@pytest.mark.parametrize("far", [False, True])
@pytest.mark.parametrize("body_bias", [0., .4])
def test_all_layout_members_shift_together_from_whole_animation_minimum(far, body_bias):
    task, state = _task(far=far, body_bias=body_bias)
    initial = {name: getattr(task, name).get_pos()
               for name in ("table", "apple_actor", "plate_actor")}
    place = task.place_pos.copy()
    avatar_position = task.avatar.robot.global_trans.copy()
    base = state.arm.origin_pose.p.copy()
    joints = state.arm.joints.copy()
    task._initialize_handover_layout()
    offset = task._handover_layout_offset
    expected_y = max(0., .554 + .02 - min(state.observed_fronts))
    np.testing.assert_allclose(offset, [0., expected_y, 0.])
    assert task._handover_body_table_clearance >= .02 - 1e-12
    assert task.avatar.plays == (2 if far else 1)
    for name in ("apple_actor", "plate_actor"):
        np.testing.assert_allclose(getattr(task, name).get_pos(), initial[name] + offset)
    np.testing.assert_allclose(task.table.get_pos(), initial["table"])
    np.testing.assert_allclose(task.apple_pose.p, initial["apple_actor"] + offset)
    np.testing.assert_allclose(task.place_pos, place + offset)
    np.testing.assert_allclose(task.target.position, place + offset)
    np.testing.assert_allclose(task._plate_xy, (place + offset)[:2])
    np.testing.assert_allclose(task.avatar_init_pos, avatar_position + offset)
    np.testing.assert_allclose(task.avatar.robot.global_trans, avatar_position + offset)
    np.testing.assert_allclose(task.avatar.resets[-2][0], avatar_position)
    np.testing.assert_allclose(state.arm.origin_pose.p, base + offset)
    np.testing.assert_allclose(state.arm.entity.get_pos(), base + offset)
    np.testing.assert_allclose(state.arm.planner._base_pose[:3], base + offset)
    np.testing.assert_allclose(state.bases[-1], state.arm.origin_pose.to_pose7())
    np.testing.assert_array_equal(state.arm.joints, joints)
    assert state.origins == state.updates == [True]
    # No physics, RNG, or joint-position APIs are supplied by this fixture.


def test_preview_error_restores_avatar_before_any_layout_member_is_moved():
    task, state = _task(far=True)
    position = task.avatar.robot.global_trans.copy()
    apple = task.apple_actor.get_pos()

    def broken_step():
        task.avatar.robot.global_trans += [0., -.1, 0.]
        raise RuntimeError("preview failed")

    task.avatar.step = broken_step
    with pytest.raises(RuntimeError, match="preview failed"):
        task._initialize_handover_layout()
    np.testing.assert_allclose(task.avatar.robot.global_trans, position)
    np.testing.assert_allclose(task.avatar.robot.global_rot, np.eye(3))
    np.testing.assert_allclose(task.apple_actor.get_pos(), apple)
    assert state.bases == state.origins == state.updates == []


@pytest.mark.parametrize("missing", ["avatar", "avatar_collider"])
def test_missing_avatar_or_collider_is_noop(missing):
    task, state = _task()
    setattr(task, missing, None)
    task._initialize_handover_layout()
    assert not hasattr(task, "_handover_layout_offset")
    assert state.bases == state.origins == state.updates == []


def test_scene_init_roundtrip_does_not_double_translate_objects_or_avatar():
    original, original_state = _task()
    original._initialize_handover_layout()
    spec = capture_scene_init(original, seed=0)
    assert spec["task_attrs"]["_handover_layout_offset"] == pytest.approx([0., .274, 0.])
    assert not any(name.startswith("robot") for name in spec["entities"])
    replay, state = _task()
    replay.config = {"scene_init": spec}
    result = apply_scene_init_post_settle(replay)
    assert result["missing_entities"] == []
    replay._initialize_handover_layout()
    for name in ("apple_actor", "plate_actor", "table"):
        np.testing.assert_allclose(getattr(replay, name).get_pos(), getattr(original, name).get_pos())
    np.testing.assert_allclose(replay.avatar.robot.global_trans, original.avatar.robot.global_trans)
    np.testing.assert_allclose(replay.target.position, original.target.position)
    np.testing.assert_allclose(state.arm.origin_pose.p, original_state.arm.origin_pose.p)
    assert replay.avatar.plays == 0
    assert len(state.bases) == 1
