from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases.categorize import CategorizeTaskCore
from envs.task_bases.categorize_support.cube import cube_candidates, pick_and_place_cube
from envs.task_bases.categorize_support.radius import radius_candidates, pick_and_place_radius
from envs.task_bases.categorize_support.motion import CategorizeMotionRejected
from envs.utils import Pose


def setup_case():
    core = CategorizeTaskCore({})
    center = np.array([-.2, .1, .785])
    pose = Pose(center.copy(), [np.cos(.2), 0., 0., np.sin(.2)])
    pick = NS(get_center=lambda: center.copy(), radius=.02, grasp_z_offset=0., close_value=.35)
    place = NS(pos=np.array([-.2, -.2, .821]), transport_z=None, drop_from_z=1.001)
    target = core._top_down_tcp([-.2, .1, .779], yaw=np.pi)
    current = [target]
    events = []
    arm = NS(tcp_offset=Pose([0., 0., .1034]), get_ee_pose=lambda: current[0].to_pose7())
    def select(*args, **kwargs):
        events.append(('selector', args, kwargs))
        return target * arm.tcp_offset.inv(), None, NS(name='cube_radius_pi')
    def move(next_pose, tag):
        events.append(('move', next_pose))
        center[:] += next_pose.p - current[0].p
        current[0] = next_pose
    def reject(reason):
        raise CategorizeMotionRejected(reason)
    task = NS(TABLE_TOP_Z=.765, robot=NS(get_arm=lambda _: arm),
        _top_down_tcp=core._top_down_tcp,
        open_gripper=lambda _: events.append(('open',)),
        set_gripper=lambda value, _: events.append(('close', value)),
        select_and_execute_grasp=select, step_sim=lambda: None,
        _categorize_arrival=lambda *a: None,
        _categorize_begin_hold=lambda _: events.append(('hold',)),
        _move_tcp_screw=move, _reject_categorize_motion=reject)
    return task, pick, place, NS(get_pose=lambda: pose), events


def test_cube_symmetry_preserves_world_depth_and_approach_on_rotated_object():
    task, pick, _, actor, _ = setup_case()
    candidates = cube_candidates(task, pick, actor.get_pose())
    world = [g.to_world(actor.get_pose(), 1.) for g in candidates]
    for pose in world:
        np.testing.assert_allclose(pose.p, [-.2, .1, .779])
        np.testing.assert_allclose(pose.to_matrix()[:3, 2], [0., 0., -1.], atol=1e-12)
    np.testing.assert_allclose(world[0].to_matrix()[:3, 0], -world[1].to_matrix()[:3, 0], atol=1e-12)
    assert all(g.scale_frame == 'runtime' and g.pre_distance == .15 for g in candidates)


def test_failed_selector_never_closes_or_moves():
    task, pick, place, actor, events = setup_case()
    task.select_and_execute_grasp = lambda *a, **kw: None
    assert pick_and_place_cube(task, pick, place, actor, 'right') is False
    assert events == [('open',)]


def test_cube_uses_shared_selector_then_original_closure_and_motion_heights():
    task, pick, place, actor, events = setup_case()
    assert pick_and_place_cube(task, pick, place, actor, 'right')
    selector = next(e for e in events if e[0] == 'selector')
    assert selector[1][0] == 'cube'
    assert selector[2]['object_scale'] == 1.
    assert len(selector[2]['candidate_grasps']) == 2
    assert ('close', .35) in events
    assert [e[0] for e in events].count('open') == 2
    moves = [e[1] for e in events if e[0] == 'move']
    np.testing.assert_allclose([m.p[2] for m in moves], [.805, 1.145, 1.145, 1.001])
    for move in moves:
        np.testing.assert_allclose(move.q, moves[0].q)


def test_motion_rejection_never_opens_held_cube():
    task, pick, place, actor, events = setup_case()
    task._move_tcp_screw = lambda *a: task._reject_categorize_motion('native path')
    with pytest.raises(CategorizeMotionRejected, match='native path'):
        pick_and_place_cube(task, pick, place, actor, 'right')
    assert ('close', .35) in events
    assert [e[0] for e in events].count('open') == 1


def test_cube_must_actually_lift_before_transport_or_release():
    task, pick, place, actor, events = setup_case()
    task._move_tcp_screw = lambda *a: events.append(('unretained',))
    with pytest.raises(CategorizeMotionRejected, match='vertical lift'):
        pick_and_place_cube(task, pick, place, actor, 'right')
    assert [e[0] for e in events].count('open') == 1
    assert [e[0] for e in events].count('unretained') == 2


@pytest.mark.parametrize('name,scale,close', [('086_woodenblock', 1., .35), ('035_apple', .7, .58)])
def test_mesh_radius_preserves_model_scale_and_original_closure(name, scale, close):
    task, pick, place, actor, events = setup_case()
    pick.close_value = close
    assert pick_and_place_radius(task, pick, place, actor, 'right',
                                 object_name=name, model_id=1, object_scale=scale)
    selector = next(e for e in events if e[0] == 'selector')
    assert selector[1][0] == name
    assert selector[2]['model_id'] == 1
    assert selector[2]['object_scale'] == scale
    assert ('close', close) in events
    for candidate in selector[2]['candidate_grasps']:
        np.testing.assert_allclose(candidate.to_world(actor.get_pose(), scale).p,
                                   [-.2, .1, .779])


def test_radius_symmetry_preserves_anisotropic_jaw_axis():
    task, pick, _, actor, _ = setup_case()
    pick.get_grasp_yaw = lambda: np.pi / 2
    poses = [g.to_world(actor.get_pose(), .7) for g in radius_candidates(task, pick, actor.get_pose())]
    reference = task._top_down_tcp([-.2, .1, .779], yaw=np.pi/2).to_matrix()[:3, 0]
    np.testing.assert_allclose(poses[0].to_matrix()[:3, 0], -reference, atol=1e-12)
    np.testing.assert_allclose(poses[1].to_matrix()[:3, 0], reference, atol=1e-12)
