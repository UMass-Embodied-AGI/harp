"""Ordinary cabinet native planning, failure propagation and release checks."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.manipulation import PickSpec, PlaceSpec
from envs.planning.base import PlanResult
from envs.task_bases.cabinet_ordinary_motion import CabinetOrdinaryMotionMixin, _OrdinaryCabinetRejected
from envs.task_bases.put_object_cabinet import PutObjectCabinet
from envs.utils import Pose


@pytest.fixture
def motion():
    state = {'q':np.zeros(7),'target':Pose([.1,-.2,.9]),'lag':0.,'fk_error':0.,
             'collision':False,'executed':[],'ik_goal':.01,'rrt_jump':False,'ik_calls':0}
    def forbidden(*_a,**_kw):
        pytest.fail('Genesis IK, pose setters, or relaxed planner fallback used')
    def solve(seed,pose,**kwargs):
        state['ik_calls'] += 1
        state['target'] = Pose.from_pose7(pose)
        return PlanResult(True,np.stack([seed,np.full(7,state['ik_goal'])]))
    def rrt(**kwargs):
        goal = kwargs['goal_qposes'][0][:7]
        path = np.stack([state['q'],goal]) if state['rrt_jump'] else np.linspace(state['q'],goal,30)
        return {'status':'Success','position':path,'velocity':np.zeros_like(path)}
    backend = SimpleNamespace(check_for_env_collision=lambda _q:['fixture'] if state['collision'] else [],plan_qpos=rrt)
    planner = SimpleNamespace(_mplib=backend,_pad_qpos=lambda q:np.r_[q,0.,0.],solve_ik=solve,
        plan_path=forbidden, plan_screw_path=lambda _q,_p,**_kw:PlanResult(True,np.zeros((2,7)),np.zeros((2,7))))
    arm = SimpleNamespace(n_arm=7,planner=planner,tcp_offset=Pose(),gripper_val=1.,_cached_target=np.r_[np.zeros(7),.04,.04],
        get_arm_qpos=lambda:state['q'].copy(),
        get_ee_pose=lambda:Pose(state['target'].p+[0,0,state['lag']],state['target'].q).to_pose7(),
        entity=SimpleNamespace(inverse_kinematics=forbidden,set_qpos=forbidden))
    task = PutObjectCabinet.__new__(PutObjectCabinet)
    task.config = {'robot_type':'franka'}
    task.TABLE_TOP_Z = .765
    task.robot = SimpleNamespace(get_arm=lambda _:arm)
    task.plan_success = True
    task._ordinary_pick_active = True
    task._refresh_avatar_planner_obstacles = lambda _:None
    task._drawer_plan_endpoint_tcp = lambda *_:Pose(state['target'].p+[0,0,state['fk_error']],state['target'].q)
    task.step_sim = lambda:None
    task._reseed_cached_arm_target = lambda _:None
    def execute(plan,_tag):
        state['executed'].append(plan)
        state['q'] = plan.position[-1].copy()
    task.execute_plan = execute
    return task,arm,state


def test_large_ik_endpoint_requires_actual_continuous_rrt(motion):
    task,arm,state = motion
    state['ik_goal'] = .8  # <=3.5rad L2, but cannot be executed in one .30rad step.
    assert task._move_seeded(state['target'].to_pose7(),'right')
    assert len(state['executed']) == 1 and state['ik_calls'] == 1
    assert np.max(abs(np.diff(state['executed'][0].position,axis=0))) < .30


@pytest.mark.parametrize('fault',['jump','branch','collision','fk','unavailable','nan'])
def test_rejected_seeded_path_never_executes(motion,fault):
    task,arm,state = motion
    if fault=='jump':state.update(ik_goal=.8,rrt_jump=True)
    if fault=='branch':state['ik_goal']=1.4
    if fault=='collision':state['collision']=True
    if fault=='fk':state['fk_error']=.016
    if fault=='unavailable':arm.planner._mplib.plan_qpos=lambda **_: {'status':'Failure'}
    if fault=='nan':state['ik_goal']=float('nan')
    with pytest.raises(_OrdinaryCabinetRejected):
        task._move_seeded(state['target'].to_pose7(),'right')
    assert not state['executed'] and not task.plan_success


def test_measured_vertical_lag_stops_after_valid_plan(motion):
    task,arm,state = motion
    state['lag']=.047
    with pytest.raises(_OrdinaryCabinetRejected,match='arrival'):
        task._move_seeded(state['target'].to_pose7(),'right')
    assert len(state['executed'])==1


def test_tiny_lift_uses_native_ik_and_tighter_fk(motion):
    task,arm,state = motion
    q = task._solve_ik(state['target'],'right')
    np.testing.assert_allclose(q,np.full(7,.01))
    assert state['ik_calls']==1 and not state['executed']
    state['fk_error']=.012
    with pytest.raises(_OrdinaryCabinetRejected,match='FK'):
        task._solve_ik(state['target'],'right')


def test_tiny_lift_branch_jump_is_not_an_rrt_goal(motion):
    task,arm,state = motion
    state['ik_goal']=.31
    with pytest.raises(_OrdinaryCabinetRejected,match='continuity'):
        task._solve_ik(state['target'],'right')


def test_public_rejection_cannot_continue_to_release(monkeypatch,motion):
    task,arm,state = motion
    task._ordinary_pick_active = False
    task._ordinary_fixtures = lambda _:nullcontext()
    events=[]
    def shared(self,*_):
        self._ordinary_reject('carry route failed')
        events.append('release')
    monkeypatch.setattr(CabinetOrdinaryMotionMixin,'_ordinary_public_pick_and_place',shared)
    pick=PickSpec(get_center=lambda:np.array([0.,0.,.8]),radius=.02)
    assert not task.pick_and_place(pick,PlaceSpec(pos=np.array([0.,0.,1.])), 'right')
    assert not events and not task._ordinary_pick_active and not task.plan_success


def test_closing_checks_latest_measured_target(monkeypatch,motion):
    task,arm,state = motion
    task._ordinary_last_target = state['target']
    state['lag']=.047
    events=[]
    monkeypatch.setattr(BaseTask,'set_gripper',lambda *_a,**_k:events.append('close'))
    with pytest.raises(_OrdinaryCabinetRejected,match='arrival'):
        task.set_gripper(.5,'right')
    assert not events


def test_native_clock_retains_four_ms_duration(monkeypatch):
    task=PutObjectCabinet.__new__(PutObjectCabinet)
    task._ordinary_pick_active=True
    task.scene=SimpleNamespace(sim=SimpleNamespace(dt=.002))
    plans=[]
    monkeypatch.setattr(BaseTask,'execute_plan',lambda _self,plan,_tag,gripper_result=None:plans.append((plan,gripper_result)))
    pos=np.zeros((3,7));grip=np.array([1.,.5,.3])
    task.execute_plan(PlanResult(True,pos,np.ones_like(pos)),'right',grip)
    assert len(plans[0][0].position)*.002==len(pos)*.004
    np.testing.assert_array_equal(plans[0][1],np.repeat(grip,2))


def test_held_payload_after_open_withdrawal_is_failure(motion):
    task,arm,state=motion
    task._ordinary_move_pose=lambda *_:state['target']
    pick=PickSpec(get_center=lambda:state['target'].p.copy(),radius=.02)
    place=PlaceSpec(pos=np.array([.1,-.2,.85]),transport_z=1.1)
    with pytest.raises(_OrdinaryCabinetRejected,match='still follows'):
        task._after_pick_place_release(pick,place,'right',place.pos)


def test_offset_payload_still_held_above_support_is_failure(motion):
    task,arm,state=motion
    task._ordinary_move_pose=lambda *_:state['target']
    task.object=SimpleNamespace(entity=SimpleNamespace(get_AABB=lambda:np.array([[.08,-.22,.82],[.12,-.18,.84]])))
    pick=PickSpec(get_center=lambda:state['target'].p-[0,0,.07],radius=.02)
    place=PlaceSpec(pos=np.array([.1,-.2,.8]),transport_z=1.1,support_z=.765)
    with pytest.raises(_OrdinaryCabinetRejected,match='above support'):
        task._after_pick_place_release(pick,place,'right',place.pos)


def test_cavity_envelope_removed_only_after_tiny_and_full_lift(motion):
    task,arm,state=motion
    removed=[]
    names=['cabinet_ordinary_table','cabinet_ordinary_cabinet']
    backend=arm.planner._mplib
    backend.planning_world=SimpleNamespace(remove_object=lambda name:removed.append(name))
    backend.acm=SimpleNamespace(remove_entry=lambda _:None)
    task._ordinary_fixture_state=(backend,names)
    task._ordinary_transport_z=1.1
    def screw(_seed,pose,**_kw):
        state['target']=Pose.from_pose7(pose)
        return PlanResult(True,np.zeros((2,7)),np.zeros((2,7)))
    arm.planner.plan_screw_path=screw
    task._ordinary_tiny_lift_complete=False
    task._move_screw([.1,-.2,1.1],'right')
    assert not removed
    task._ordinary_tiny_lift_complete=True
    task._move_screw([.1,-.2,1.1],'right')
    assert removed==['cabinet_ordinary_cabinet'] and names==['cabinet_ordinary_table']


@pytest.mark.parametrize('scale', [.047, .7, 1.])
@pytest.mark.parametrize('yaw', [0., .7])
def test_public_ordinary_candidate_preserves_pick_geometry_at_actual_scale(motion, scale, yaw):
    task, arm, state = motion
    task._object_scale = scale
    obj = Pose([.1, -.2, .8], [np.cos(.3), 0, 0, np.sin(.3)])
    pick = PickSpec(get_center=lambda:np.array([.1,-.2,.82]), radius=.025,
                    grasp_z_offset=.005, approach_xy_offset=(.002,-.003), get_grasp_yaw=lambda:yaw)
    candidate = task._ordinary_grasp_candidate(pick, obj)
    actual = candidate.to_world(obj, scale)
    expected = task._top_down_tcp([.102, -.203, .8175], yaw=yaw)
    np.testing.assert_allclose(actual.p, expected.p, atol=1e-12)
    assert abs(np.dot(actual.q,expected.q)) == pytest.approx(1.)
    assert candidate.pre_distance == .15


@pytest.mark.parametrize('fault', ['branch', 'jump', 'fk', 'collision'])
def test_public_preflight_rejects_invalid_second_leg(motion, fault):
    task, arm, state = motion
    path = np.zeros((3,7))
    if fault == 'branch': path[-1] = 1.4
    if fault == 'jump': path[-1,0] = .31
    if fault == 'fk': state['fk_error'] = .016
    if fault == 'collision': state['collision'] = True
    first = PlanResult(True,np.zeros((2,7)),np.zeros((2,7)))
    second = PlanResult(True,path,np.zeros_like(path))
    assert not task._ordinary_grasp_plans_valid((first,second,state['target'],state['target']),'right')
    assert not state['executed']


@pytest.mark.parametrize('failure', ['none', 'unavailable', 'arrival', 'carry_drop', 'descent_drop', 'palm_block'])
def test_public_pick_controls_release_and_preserves_settings(motion, failure):
    task, arm, state = motion
    task._ordinary_pick_active = False
    task._ordinary_fixtures = lambda _:nullcontext()
    task._object_scale = .7
    task._object_id = '035_apple'
    task._object_model_id = 1
    task.object = SimpleNamespace(get_pose=lambda:Pose([.1,-.2,.8]))
    task._drawer_rim_z = 1.08
    arm.tcp_offset = Pose([0, 0, .1034])
    events = []
    moved = []
    drift = np.zeros(3)
    def center(): return state['target'].p + [0,0,.01] + drift
    pick = PickSpec(get_center=center,radius=.025,close_value=.5,max_grasp_offset=.1,
                    transport_retention_tolerance=.08,after_close=lambda:events.append('after_close'),
                    get_half_height=lambda:.025,require_lift=True)
    place = PlaceSpec(pos=np.array([.1,.1,1.]),transport_z=1.15,support_z=.99,support_clearance=.01)
    def public(**kwargs):
        events.append('public')
        assert kwargs['object_scale']==.7 and kwargs['max_candidates']==1
        assert kwargs['table_z']==.765 and callable(kwargs['post_plan_check'])
        candidate = kwargs['candidate_grasps'][0]
        target = candidate.to_world(kwargs['object_pose'],kwargs['object_scale'])
        state['target'] = target
        if failure=='unavailable': return None
        if failure=='arrival': state['lag']=.016
        link = target * arm.tcp_offset.inv()
        return link, link, candidate
    def move(target, _tag, **kwargs):
        moved.append((target,kwargs))
        state['target']=target
        if (failure=='carry_drop' and len(moved)==3) or (failure=='descent_drop' and len(moved)==4):
            drift[2]=-.085
        if failure=='palm_block' and len(moved)==4:
            state['target'] = Pose(target.p-[0,0,.007], target.q)
        return target
    task.select_and_execute_grasp = public
    task._ordinary_move_pose = move
    task.open_gripper = lambda _:events.append('open')
    task.set_gripper = lambda value,_:events.append(('close',value))
    task._after_pick_place_release = lambda *_:events.append('withdraw')
    ok = task.pick_and_place(pick,place,'right')
    assert ok == (failure=='none')
    assert not task._ordinary_pick_active
    if failure in ('unavailable','arrival'):
        # An unplanned nominal grasp is followed by its half-turned twin.
        expected = ['open','public','public'] if failure == 'unavailable' else ['open','public']
        assert events == expected and not moved
    else:
        assert events[:4] == ['open','public',('close',.5),'after_close']
        assert moved[0][1] == {'tiny':True}
        assert all(abs(np.dot(t.q,moved[0][0].q))==pytest.approx(1.) for t,_ in moved)
        assert events.count('open') == (2 if failure=='none' else 1)
        assert ('withdraw' in events) == (failure=='none')


@pytest.mark.parametrize('tilt', [0., -.03, .03, -.10, .10])
def test_release_height_keeps_native_and_physical_palm_above_rim(motion, tilt):
    import trimesh
    from pathlib import Path
    task, arm, state = motion
    task._drawer_rim_z = 1.0883572101593018
    arm.tcp_offset = Pose([0, 0, .1034])
    down = Pose([.28, -.50, 1.18], [0, 1, 0, 0])
    carried = down * Pose(q=[np.cos(tilt/2), np.sin(tilt/2), 0, 0])
    height = task._ordinary_release_height(1.010586, 1.18, carried, 'right')
    release = Pose([*carried.p[:2], height], carried.q)
    assert task._ordinary_palm_bottom(release, 'right') >= task._drawer_rim_z+.008-1e-12
    mesh = trimesh.load(Path(__file__).resolve().parents[1] /
        'assets/embodiments/franka-panda/franka_description/meshes/collision/hand.stl',
        force='mesh', process=False)
    hand = release * arm.tcp_offset.inv()
    vertices = mesh.vertices @ hand.to_matrix()[:3,:3].T + hand.p
    assert vertices[:,2].min() >= task._drawer_rim_z+.008-1e-12
    if tilt == 0:
        assert height == pytest.approx(1.0589572101593018)


@pytest.mark.parametrize('support,transport,rim', [
    (np.nan,1.18,1.088), (1.02,np.inf,1.088), (1.02,1.18,np.nan),
    (1.02,1.03,1.088),
])
def test_invalid_or_unreachable_palm_release_height_is_rejected(motion, support, transport, rim):
    task, arm, state = motion
    task._drawer_rim_z = rim
    arm.tcp_offset = Pose([0,0,.1034])
    with pytest.raises(_OrdinaryCabinetRejected):
        task._ordinary_release_height(support,transport,Pose([0,0,1.18],[0,1,0,0]),'right')
    assert not state['executed'] and not task.plan_success


def test_palm_clearance_does_not_lower_an_already_clear_release(motion):
    task, arm, state = motion
    task._drawer_rim_z = 1.0883572101593018
    arm.tcp_offset = Pose([0,0,.1034])
    assert task._ordinary_release_height(1.086,1.18,Pose([0,0,1.18],[0,1,0,0]),'right') == 1.086


@pytest.mark.parametrize('object_id,robot,active', [
    ('cube','franka',True), ('073_rubikscube','franka',True),
    ('086_woodenblock','franka',True), ('031_jam-jar','franka',True),
    ('100_seal','franka',False), ('cube','other',False),
])
def test_palm_clearance_scope_preserves_parked_seal(motion, object_id, robot, active):
    task, arm, state = motion
    task._object_id = object_id
    task.config['robot_type'] = robot
    assert task._ordinary_palm_clearance_applies() == active
