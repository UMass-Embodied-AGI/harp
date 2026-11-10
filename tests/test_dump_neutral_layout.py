"""Dump-neutral candidate checks preserve physical and collision contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.collision_checker import AvatarCollisionChecker
from envs.avatar.utils import ActionStatus, AvatarState
from envs.task_bases.dump_neutral_layout import initial_avatar_pose_is_clear
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.tasks.dump_bin_copresence import DumpBinCopresence


FIELDS = ('global_trans', 'global_rot', 'pose', 'node_trans', 'global_mat',
          'global_mat_inv', 'base_state', 'action_state', 'action_status')


def rig(point=(0., 0., .5), margin=0., ignored=(), refined=False):
    motion = SimpleNamespace(
        data=[np.r_[np.zeros(3), 1., np.zeros(3)]],
        node_data=[np.eye(4)[None]], global_mat=np.eye(3)[None],
        global_mat_inv=np.eye(3)[None], at_frame=37,
    )
    robot = SimpleNamespace(
        global_trans=np.array([4., 5., 6.]), global_rot=np.eye(3),
        pose=np.r_[np.ones(3), 1., np.zeros(3)], node_trans=np.eye(4)[None],
        global_mat=np.eye(3)[None], global_mat_inv=np.eye(3)[None],
        base_state=AvatarState.SITTING, action_state=AvatarState.KEEPING,
        action_status=ActionStatus.ONGOING, attached_object=[None, None],
        two_hand_attached_object=None, skin=SimpleNamespace(),
    )
    updates = []

    def update():
        updates.append({name: getattr(robot, name) for name in FIELDS})
        robot.skin.position = robot.global_trans.copy()

    robot.update = update
    collider = SimpleNamespace(
        current_capsules=lambda: [('r_forearm', robot.skin.position,
                                   robot.skin.position + [0., 0., 1.], .1)],
        current_boxes=lambda: [], distal_clipped_capsules=(),
    )
    if refined:
        # Original skin refinement excludes the otherwise colliding point.
        collider.capsule_mesh_halfspaces = lambda *args: np.array([[1., 0., 0., .2]])
    vertices = np.asarray([point], dtype=float)
    geom = SimpleNamespace(init_verts=vertices, get_verts=lambda: vertices)
    entity = SimpleNamespace(links=[SimpleNamespace(name='hand', _geoms=[geom])])
    checker = AvatarCollisionChecker(collider, entity, margin=margin,
                                     ignored_capsules=ignored)
    avatar = SimpleNamespace(robot=robot, motion_modules={'motion': motion},
                             frame_ratio=7., _keep_motion_name='untouched')
    task = SimpleNamespace(
        avatar=avatar, avatar_collider=collider, avatar_collision_checker=checker,
        _neutral_avatar_body_rot_for_work_xy=lambda job, xy: np.eye(3),
        avatar_collided=True, avatar_collision_log=[{'existing': True}],
        _avatar_collision_tick=11,
    )
    return task, robot, motion, updates


def query(task, position=(0., 0., 0.)):
    return initial_avatar_pose_is_clear(task, {'motion': 'motion', 'kind': 'mouse'},
                                        [0., 0.], position)


def test_overlap_rejected_and_alternative_pose_accepted():
    task, _, _, _ = rig()
    assert not query(task)
    assert query(task, (1., 0., 0.))


@pytest.mark.parametrize('kwargs,expected', [
    ({'margin': -.11}, True),
    ({'ignored': ('r_forearm',)}, True),
    ({'refined': True}, True),
    ({'point': (.105, 0., .5), 'margin': .01}, False),
    ({'point': (.105, 0., .5)}, True),
])
def test_original_checker_margin_exemptions_and_skin_refinement(kwargs, expected):
    task, _, _, _ = rig(**kwargs)
    assert query(task) is expected


def test_original_distal_clip_and_palm_box_semantics():
    task, _, _, _ = rig(point=(0., 0., 1.05))
    assert not query(task)
    task.avatar_collider.distal_clipped_capsules = ('r_forearm',)
    assert query(task)
    task.avatar_collider.current_boxes = lambda: [
        ('r_palm', np.array([0., 0., 1.05]), np.eye(3), np.array([.03, .02, .01]))]
    assert not query(task)


@pytest.mark.parametrize('raise_check', [False, True])
def test_pose_references_state_history_and_motion_clock_restored(raise_check):
    task, robot, motion, updates = rig()
    saved = {name: getattr(robot, name) for name in FIELDS}
    log = task.avatar_collision_log
    attachments = robot.attached_object
    motion_data = motion.data[0].copy()
    if raise_check:
        def broken():
            raise ValueError('checker failed')
        task.avatar_collision_checker.check = broken
        with pytest.raises(ValueError, match='checker failed'):
            query(task)
    else:
        assert not query(task)
    assert len(updates) == 2
    assert all(getattr(robot, name) is value for name, value in saved.items())
    np.testing.assert_array_equal(robot.skin.position, saved['global_trans'])
    assert updates[0]['base_state'] is AvatarState.STANDING
    assert updates[0]['action_state'] is AvatarState.NO_ACTION
    assert updates[0]['action_status'] is ActionStatus.INIT
    assert robot.attached_object is attachments
    assert task.avatar_collision_log is log and log == [{'existing': True}]
    assert task.avatar_collided and task._avatar_collision_tick == 11
    assert motion.at_frame == 37
    np.testing.assert_array_equal(motion.data[0], motion_data)
    assert task.avatar.frame_ratio == 7.
    assert task.avatar._keep_motion_name == 'untouched'


@pytest.mark.parametrize('missing', ['checker', 'collider', 'motion', 'skin', 'attached'])
def test_missing_geometry_or_attached_props_fail_without_preview(missing):
    task, robot, _, updates = rig()
    if missing == 'checker':
        task.avatar_collision_checker = None
    elif missing == 'collider':
        task.avatar_collider = None
    elif missing == 'motion':
        task.avatar.motion_modules.clear()
    elif missing == 'skin':
        robot.skin = None
    else:
        robot.attached_object[0] = object()
    with pytest.raises(RuntimeError, match='dump neutral initial clearance requires'):
        query(task)
    assert not updates


def test_invalid_candidate_rejected_before_mutation():
    task, _, _, updates = rig()
    with pytest.raises(RuntimeError, match='invalid dump neutral candidate'):
        query(task, (np.nan, 0., 0.))
    assert not updates


def test_layout_candidate_heading_is_checked_and_restored():
    task, robot, _, updates = rig()
    original_rotation = robot.global_rot
    candidate = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    task._neutral_avatar_measure_body_rot = candidate
    query(task)
    np.testing.assert_array_equal(updates[0]['global_rot'], candidate)
    assert robot.global_rot is original_rotation
    assert task._neutral_avatar_measure_body_rot is candidate


def test_dump_layout_recovery_is_default_off():
    task = DumpBinCopresence.__new__(DumpBinCopresence)
    task.config = {}
    assert not task._neutral_avatar_use_layout_recovery({'kind': 'sponge'})
    assert task.NEUTRAL_LAYOUT_RECOVERY_DEGREES == (0,)


@pytest.mark.parametrize('enabled,parent_ok,expected_calls,expected', [
    (False, True, 0, True), (True, False, 0, False), (True, True, 1, False),
])
def test_task_predicate_preserves_parent_and_is_default_off(
        monkeypatch, enabled, parent_ok, expected_calls, expected):
    task = DumpBinCopresence.__new__(DumpBinCopresence)
    task.config = {'dump_neutral_initial_clearance_v1': True} if enabled else {}
    task._dump_neutral_initial_clearance_counts = {'checked': 0, 'accepted': 0, 'rejected': 0}
    monkeypatch.setattr(NeutralAvatarTableWorkMixin, '_neutral_avatar_body_pose_is_valid',
                        lambda *args: parent_ok)
    calls = []

    def check(*args):
        calls.append(args)
        return False

    monkeypatch.setattr('envs.tasks.dump_bin_copresence.initial_avatar_pose_is_clear', check)
    assert task._neutral_avatar_body_pose_is_valid({}, [0, 0], [0, 0, 0]) is expected
    assert len(calls) == expected_calls
    assert task._dump_neutral_initial_clearance_counts == {
        'checked': expected_calls, 'accepted': 0, 'rejected': expected_calls}


@pytest.mark.parametrize('value', ['false', 'true', 1, None, False])
def test_candidate_requires_literal_true(monkeypatch, value):
    task = DumpBinCopresence.__new__(DumpBinCopresence)
    task.config = {'dump_neutral_initial_clearance_v1': value}
    monkeypatch.setattr(NeutralAvatarTableWorkMixin, '_neutral_avatar_body_pose_is_valid',
                        lambda *args: True)

    def unexpected(*args):
        pytest.fail('nonliteral opt-in must not query the candidate')

    monkeypatch.setattr('envs.tasks.dump_bin_copresence.initial_avatar_pose_is_clear', unexpected)
    assert task._neutral_avatar_body_pose_is_valid({}, [0, 0], [0, 0, 0])
