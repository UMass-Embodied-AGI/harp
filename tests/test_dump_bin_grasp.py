"""Public-helper, timing, and physical release contracts for dump."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.dump_bin_motion import retime_plan, DumpMotionRejected
from envs.task_bases.dump_bin import DumpBin
from envs.task_bases import dump_bin as dump, dump_bin_grasp as grasp
from envs.manipulation import PickSpec, PlaceSpec
from envs.utils import Pose


def test_timing_preserves_original_waypoints_and_duration():
    positions = np.arange(21, dtype=float).reshape(3, 7)
    original = PlanResult(True, positions.copy(), np.ones_like(positions))
    timed, fingers = retime_plan(original, .002, np.array([1., .5, 0.]))
    assert timed.position.shape == (5, 7)
    np.testing.assert_array_equal(timed.position[::2], positions)
    np.testing.assert_array_equal(timed.velocity, np.ones((5, 7)))
    np.testing.assert_array_equal(fingers, [1., .75, .5, .25, 0.])
    np.testing.assert_array_equal(original.position, positions)
    assert (len(timed.position)-1)*.002 == (len(positions)-1)*.004


@pytest.mark.parametrize('dt', [np.nan, 0., -.001, .005])
def test_invalid_physical_timing_is_rejected(dt):
    with pytest.raises(ValueError):
        retime_plan(PlanResult(True, np.zeros((2, 7))), dt)


def test_retained_bin_centers_keep_whole_object_inside():
    bin_xy = np.array([.65, -.3])
    half, object_half = np.array([.132, .195]), np.array([.05, .03])
    centers = list(grasp.release_centers(bin_xy, half, object_half))
    assert len(centers) > 1
    for center in centers:
        assert np.all(abs(center-bin_xy)+object_half+.008 < half)
    assert all(center[0] > .57 for center in centers)  # wide jar cannot use this edge


def test_oversize_object_has_no_release_candidate():
    assert list(grasp.release_centers([.65,-.3], [.13,.19], [.14,.2])) == []


def test_unsettled_jar_is_rejected_after_bounded_wait():
    task = DumpBin({})
    count = [0]
    task.step_sim = lambda: count.__setitem__(0, count[0]+1)
    pick = PickSpec(lambda: np.array([count[0]*.00001,0.,.8]), .03)
    with pytest.raises(DumpMotionRejected, match='did not settle'):
        grasp.wait_for_jar_rest(task, pick)
    assert count[0] == 3000


@pytest.mark.parametrize('moving_steps', [200, 1500])
def test_jar_rest_requires_two_quiet_windows(moving_steps):
    task = DumpBin({})
    count = [0]
    task.step_sim = lambda: count.__setitem__(0, count[0]+1)
    pick = PickSpec(lambda: np.array([min(count[0],moving_steps)*.00001,0.,.8]), .03)
    grasp.wait_for_jar_rest(task, pick)
    assert count[0] == moving_steps + 200


def test_apple_public_grasp_clears_live_top_without_changing_aperture(monkeypatch):
    task = DumpBin({'robot_type':'franka'})
    task.TABLE_TOP_Z = .765
    center = np.array([.1,-.2,.8])
    actor = SimpleNamespace(get_pose=lambda:Pose(center),
        get_AABB=lambda:np.array([center-.027,center+.027]))
    task._dump_active_actor = actor
    task._object_model_ids = {id(actor):1}
    task.robot = SimpleNamespace(get_arm=lambda tag:SimpleNamespace(tcp_offset=Pose()))
    task.open_gripper = lambda tag:None
    task.step_sim = lambda:None
    task._refresh_dump_obstacles = lambda tag:None
    task._dump_arrival = lambda *a, **kw: None
    monkeypatch.setattr(dump,'_read_scale',lambda *a:.7)
    def select(**kwargs):
        candidate = kwargs['candidate_grasps'][0]
        assert candidate.pre_distance == .10
        world = candidate.to_world(kwargs['object_pose'],kwargs['object_scale'])
        return world,world,candidate
    task.select_and_execute_grasp = select
    pick = PickSpec(lambda:center.copy(),.026,label='035_apple',close_value=.5)
    tcp = grasp.select_topdown(task,pick,PlaceSpec(np.zeros(3),transport_z=.945),'right')
    np.testing.assert_allclose(tcp.p,[.1,-.2,.799])
    np.testing.assert_allclose(np.abs(tcp.to_matrix()[:3,1]),[0.,1.,0.],atol=1e-12)
    assert pick.close_value == .5


@pytest.mark.parametrize('object_top,expected_z', [(.83, .821), (.86, .833)])
def test_topdown_uses_public_selector_and_runtime_scale_once(monkeypatch, object_top, expected_z):
    task = DumpBin({'robot_type':'franka'})
    task.TABLE_TOP_Z = .765
    actor_pose = Pose([.1, -.2, .8], [0.70710678, 0., 0., .70710678])
    actor = SimpleNamespace(get_pose=lambda: actor_pose,
                            get_AABB=lambda: np.array([[.09,-.21,.79],[.13,-.17,object_top]]))
    task._dump_active_actor = actor
    task._object_model_ids = {id(actor): 1}
    offset = Pose([0., 0., .1034])
    arm = SimpleNamespace(tcp_offset=offset, get_arm_qpos=lambda: np.zeros(7))
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.open_gripper = lambda tag: None
    task.step_sim = lambda: None
    task._refresh_dump_obstacles = lambda tag: None
    task._dump_arrival = lambda *args, **kw: None
    task._dump_plan_valid = lambda *args: True
    monkeypatch.setattr(dump, '_read_scale', lambda name, mid: .3 if mid == 1 else pytest.fail('wrong model'))
    center = np.array([.11, -.19, .81])
    pick = PickSpec(lambda: center.copy(), .03, label='031_jam-jar', grasp_z_offset=.02,
                    approach_xy_offset=(0., .01))
    place = PlaceSpec(np.array([.65,-.3,.43]), transport_z=.945, drop_from_z=.43)
    observed = {}
    def select(**kwargs):
        observed.update(kwargs)
        annotation = kwargs['candidate_grasps'][0]
        actual = annotation.to_world(kwargs['object_pose'], kwargs['object_scale'])
        pre = PlanResult(True, np.zeros((2, 7)))
        descent = PlanResult(True, np.zeros((2, 7)))
        from envs.grasp import tcp_to_link_pose
        link = tcp_to_link_pose(actual, offset)
        assert kwargs['post_plan_check']((pre, descent, link, link))
        return link, link, annotation
    task.select_and_execute_grasp = select
    tcp = grasp.select_topdown(task, pick, place, 'right')
    assert observed['object_scale'] == .3
    assert observed['model_id'] == 1
    annotation = observed['candidate_grasps'][0]
    assert annotation.scale_frame == 'runtime'
    assert observed['max_candidates'] == 2
    # Jar candidates retain height/closing width and rotate the closing axis.
    for candidate in observed['candidate_grasps']:
        world = candidate.to_world(actor_pose, .3)
        np.testing.assert_allclose(abs(world.to_matrix()[:3, 1]), [1., 0., 0.], atol=1e-7)
    np.testing.assert_allclose(annotation.to_world(actor_pose, .3).p, [.11,-.18,expected_z])
    np.testing.assert_allclose(tcp.p, [.11,-.18,expected_z])
    assert task._dump_selector_targets == {}


def test_no_acquisition_means_no_close_lift_carry_or_release(monkeypatch):
    task = DumpBin({})
    task.robot = SimpleNamespace(get_arm=lambda tag: object())
    task.set_gripper = lambda *args: pytest.fail('closed after rejected grasp')
    monkeypatch.setattr(grasp, 'select_topdown', lambda *args, **kw: None)
    assert grasp.pick_and_drop_topdown(task, PickSpec(lambda: np.zeros(3), .02),
                                       PlaceSpec(np.zeros(3)), 'right') is False


@pytest.mark.parametrize('label', ['048_stapler', '073_rubikscube', 'cube'])
def test_symmetric_selector_returns_selected_flipped_grasp(monkeypatch, label):
    task=DumpBin({'robot_type':'franka'})
    task.TABLE_TOP_Z=.765
    actor=SimpleNamespace(get_pose=lambda:Pose([.1,-.2,.79]))
    task._dump_active_actor=actor
    task.robot=SimpleNamespace(get_arm=lambda tag:SimpleNamespace(tcp_offset=Pose()))
    task.open_gripper=lambda tag:None
    task.step_sim=lambda:None
    task._refresh_dump_obstacles=lambda tag:None
    task._dump_arrival=lambda *args, **kw: None
    monkeypatch.setattr(dump,'_read_scale',lambda *args:.05)
    def select(**kwargs):
        assert kwargs['max_candidates']==2
        assert kwargs['screw_qpos_step']==.005
        a,b=kwargs['candidate_grasps']
        expected_pre=.10 if label=='048_stapler' else max(.15,.945-(.79-.3*.0115+(.01 if label=='cube' else 0.)))
        assert a.pre_distance==b.pre_distance==pytest.approx(expected_pre)
        first=a.to_world(kwargs['object_pose'],kwargs['object_scale'])
        flipped=b.to_world(kwargs['object_pose'],kwargs['object_scale'])
        np.testing.assert_allclose(first.p,flipped.p)
        np.testing.assert_allclose(first.to_matrix()[:3,2],flipped.to_matrix()[:3,2],atol=1e-7)
        return flipped,flipped,b
    task.select_and_execute_grasp=select
    tcp=grasp.select_topdown(task,PickSpec(lambda:np.array([.1,-.2,.79]),.0115,label=label),
                            PlaceSpec(np.array([.65,-.3,.43]),transport_z=.945),'right')
    np.testing.assert_allclose(tcp.to_matrix()[:3,:3],task._top_down_tcp(tcp.p,yaw=np.pi).to_matrix()[:3,:3])
    if label=='cube':
        assert tcp.p[2]==pytest.approx(.79-.3*.0115+.01)


def test_selected_yaw_is_preserved_until_release_preflight(monkeypatch):
    task=DumpBin({})
    center=np.array([.1,-.2,.8])
    tcp=task._top_down_tcp(center.copy(),yaw=np.pi)
    task.robot=SimpleNamespace(get_arm=lambda tag:SimpleNamespace(get_ee_pose=lambda:tcp.to_pose7()))
    task.set_gripper=lambda *args, **kw: None
    task.step_sim=lambda:None
    seen=[]
    task._move_cartesian=lambda *a,**kw:seen.append(kw['tcp_yaw'])
    def lift(*a,**kw):
        seen.append(kw['tcp_yaw'])
        center[2]=.94
    task._move_screw=lift
    monkeypatch.setattr(grasp,'select_topdown',lambda *args:tcp)
    def preflight(task,pick,place,tag,quaternion):
        np.testing.assert_allclose(abs(np.dot(quaternion,tcp.q)),1.)
        return None
    monkeypatch.setattr(grasp,'plan_release',preflight)
    with pytest.raises(DumpMotionRejected,match='controlled descent'):
        grasp.pick_and_drop_topdown(task,PickSpec(lambda:center.copy(),.02),
                                    PlaceSpec(np.array([.65,-.3,.43]),transport_z=.945),'right')
    np.testing.assert_allclose(np.abs(seen),[np.pi,np.pi])


def test_failed_release_preflight_does_not_open_or_carry(monkeypatch):
    task = DumpBin({})
    center = np.array([0., 0., .80])
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(get_ee_pose=lambda: [0,0,.96,1,0,0,0]))
    task.set_gripper = lambda *a, **kw: None
    task.step_sim = lambda: None
    task._move_cartesian = lambda *a, **k: None
    task._move_screw = lambda *a, **k: center.__setitem__(2, .94)
    task.open_gripper = lambda *a: pytest.fail('release after rejected preflight')
    monkeypatch.setattr(grasp, 'select_topdown', lambda *a: Pose([0,0,.80]))
    monkeypatch.setattr(grasp, 'plan_release', lambda *a, **kw: None)
    with pytest.raises(DumpMotionRejected, match='controlled descent'):
        grasp.pick_and_drop_topdown(task, PickSpec(lambda: center.copy(), .02),
                                   PlaceSpec(np.array([.65,-.3,.43]), transport_z=.945, drop_from_z=.43), 'right')


@pytest.mark.parametrize('label', ['', 'cube'])
@pytest.mark.parametrize('release_state, expected', [('floor', True), ('held', False), ('outside', False), ('moving', False), ('settling', True), ('late_settling', True), ('delayed_settling', None), ('rocking', False), ('small_rocking', False)])
def test_release_requires_stable_floor_support(monkeypatch, release_state, expected, label):
    if expected is None:
        expected = False
    task = DumpBin({})
    center = np.array([.1, -.2, .80])
    actor = SimpleNamespace(get_AABB=lambda: np.array([center-.02, center+.02]))
    task._dump_active_actor = actor
    task.big_bin_pose = Pose([.65,-.3,.01])
    task._big_bin_xy_half = np.array([.132,.195])
    task._dump_hold_offset = None
    arm = SimpleNamespace(get_ee_pose=lambda: [center[0],center[1],center[2]+.02,1,0,0,0])
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    released = []
    def open_gripper(arm_tag):
        released.append(True)
        center[:] = [.9 if release_state == 'outside' else .65, -.3,
                     .80 if release_state == 'held' else .03]
    task.open_gripper = open_gripper
    task.set_gripper = lambda *a, **kw: None
    release_steps = 0
    def step():
        nonlocal release_steps
        if released:
            release_steps += 1
        if released and release_state == 'moving':
            center[0] += .0002
        moving_until = {'settling': 350, 'late_settling': 1500, 'delayed_settling': 3500}
        if released and release_steps <= moving_until.get(release_state, 0):
            center[0] = .65 + .02 * np.sin(release_steps * np.pi / 200)
        if released and release_state in ('rocking', 'small_rocking'):
            # Returns to the same center every observation window while moving.
            amplitude = .001 if release_state == 'small_rocking' else .02
            center[0] = .65 + amplitude * np.sin(release_steps * 2 * np.pi / 100)
    task.step_sim = step
    task._move_cartesian = lambda *a, **k: center.__setitem__(2, .84)
    task._move_screw = lambda *a, **k: center.__setitem__(2, .925)
    task._dump_arrival = lambda *a, **kw: None
    task._dump_check_retention = lambda *a, **kw: None
    task._reseed_cached_arm_target = lambda *a, **kw: None
    def execute(plan, arm_tag):
        center[:] = [.65,-.3,.925 if plan == 'carry' else .40]
    task.execute_plan = execute
    monkeypatch.setattr(grasp,'select_topdown',lambda *a: Pose([.1,-.2,.8]))
    monkeypatch.setattr(grasp,'plan_release',lambda *a: [('carry',np.zeros(7)),('descent',np.zeros(7))])
    pick = PickSpec(lambda: center.copy(), .02, label=label)
    place = PlaceSpec(np.array([.65,-.3,.43]),transport_z=.945,drop_from_z=.43)
    result = (grasp.finish_drop(task,pick,[('carry',np.zeros(7)),('descent',np.zeros(7))],'right')
              if label else grasp.pick_and_drop_topdown(task,pick,place,'right'))
    assert result is expected
    assert released == [True]
    assert release_steps <= 2200
    if not expected:
        assert release_steps == 2200
    if expected:
        assert release_steps >= 500
    if release_state == 'settling':
        assert release_steps > 300
    if release_state == 'late_settling':
        assert release_steps > 1500
    if release_state == 'delayed_settling' and expected:
        assert release_steps >= 3800
