from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases.categorize_support import release as m
from envs.task_bases.categorize_support.motion import CategorizeMotionRejected
from envs.utils import Pose


@pytest.mark.parametrize('column,label,failure', [
    (True,'073_rubikscube',None), (False,'073_rubikscube',None),
    (True,'038_milk-box',None), (True,'073_rubikscube','release'),
    (True,'073_rubikscube','carry'),
])
def test_actual_yaml_dispatch_keeps_checked_release_and_failure_order(monkeypatch,column,label,failure):
    from envs.task_bases import categorize as core
    from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
    task=CategorizeObjectsInterruption({})
    task._CATEGORIZE_COLUMN_LAYOUT=column
    task._USE_OBSTACLE_INFLATION=False
    task.TABLE_TOP_Z=.765
    tcp=np.array([-.25,.15,.795,0.,0.,1.,0.])
    center=tcp[:3].copy()
    actor=NS(get_pose=lambda:Pose(center.copy()))
    arm=NS(tcp_offset=Pose(),entity=NS(get_qpos=lambda:np.zeros(9)),
           get_ee_pose=lambda:tcp.copy(),finger_joints=[])
    task.robot=NS(get_arm=lambda _:arm)
    task._sort_targets=[(actor,label,1,0)]
    task._categorize_hold_local=None
    events=[]
    moves=[]
    def move(pose,tag):
        moves.append(pose)
        if failure=='carry' and len(moves)==3:
            raise CategorizeMotionRejected('carry rejected')
        center[:]+=pose.p-tcp[:3]
        tcp[:3]=pose.p
        return True
    def begin_hold(tag):
        task._categorize_hold_local=np.zeros(3)
    def opening(tag):
        events.append(('open',task._categorize_hold_local is not None))
        task._categorize_hold_local=None
    def release(owner,payload,name,tag):
        assert owner is task and payload is actor and name==label and tag=='right'
        assert task._categorize_hold_local is not None and len(moves)==3
        events.append(('checked_release',))
        if failure=='release':raise CategorizeMotionRejected('release rejected')
        return True
    monkeypatch.setattr(task,'_get_object_scale',lambda *_:1.)
    monkeypatch.setattr(task,'_select_and_execute_task_grasp',lambda *a,**k:(None,None,NS(name='annotation')))
    monkeypatch.setattr(task,'open_gripper',opening)
    monkeypatch.setattr(task,'set_gripper',lambda *a:None)
    monkeypatch.setattr(task,'step_sim',lambda:None)
    monkeypatch.setattr(task,'_categorize_begin_hold',begin_hold)
    monkeypatch.setattr(task,'_move_tcp_screw',move)
    monkeypatch.setattr(core,'lower_release',release)
    pick=NS(label=label,actor=actor,radius=.029,get_center=lambda:center.copy())
    place=NS(pos=np.array([-.25,-.2,.82]),drop_from_z=1.001,basket_idx=0)
    if failure:
        with pytest.raises(CategorizeMotionRejected,match=failure):
            task._pick_and_place_checked(pick,place,'right')
        assert not any(e==('open',True) for e in events)
    else:
        assert task._pick_and_place_checked(pick,place,'right')
    assert events[0]==('open',False)
    expected=column and label=='073_rubikscube' and failure!='carry'
    assert (('checked_release',) in events)==expected
    if not failure and not expected:assert events[-1]==('open',True)


@pytest.mark.parametrize('column,label,expected', [
    (True,'cube',True), (True,'086_woodenblock',True), (True,'073_rubikscube',True),
    (True,'038_milk-box',False), (True,'100_seal',False), (False,'cube',False),
])
def test_release_scope_preserves_other_objects_and_layouts(column,label,expected):
    assert m.uses_lower_release(NS(_CATEGORIZE_COLUMN_LAYOUT=column),label)==expected


def make_case(monkeypatch,fail=None):
    events=[]
    aabb=np.array([[-.27,-.22,.940],[-.23,-.18,.980]])
    actor=NS(entity=NS(get_AABB=lambda:aabb.copy()))
    anchor=np.array([.01,.02,.03])
    world=NS(remove_object=lambda name:events.append(('remove',name)))
    arm=NS(get_ee_pose=lambda:np.array([-.25,-.2,1.001,0.,0.,1.,0.]),
           tcp_offset=Pose([0,0,.1034]),planner=NS(_mplib=NS(planning_world=world)))
    def reject(reason):raise CategorizeMotionRejected(reason)
    task=NS(_CATEGORIZE_COLUMN_LAYOUT=True,_categorize_hold_local=anchor,
        _sort_targets=[(actor,'086_woodenblock',1,0)],basket_poses=[Pose([-.25,-.2,.821])],
        _BASKET_HALF_X=.096,_BASKET_HALF_Y=.096,_BASKET_HALF_Z=.056,
        _OBSTACLE_MARGIN_XY=.02,_OBSTACLE_MARGIN_Z=.01,TABLE_TOP_Z=.765,
        _categorize_skip_basket=None,robot=NS(get_arm=lambda _:arm),
        _reject_categorize_motion=reject,
        _categorize_inputs=lambda goal,tag:(arm,np.zeros(7),goal),
        _categorize_refresh_obstacles=lambda tag:events.append(('restore',task._categorize_skip_basket)),
        step_sim=lambda:events.append(('step',)))
    def opening(tag):
        assert task._categorize_hold_local is anchor
        events.append(('open',));task._categorize_hold_local=None
    task.open_gripper=opening
    calls=[]
    def plan(*args):
        phase='descent' if not calls else 'retreat'
        calls.append(phase);events.append(('plan',phase))
        if fail==phase:return None
        return NS(position=np.array([np.zeros(7),np.ones(7)*.1]),phase=phase)
    def execute(plan,tag):
        events.append(('execute',plan.phase))
        if fail=='exception':raise ValueError('unexpected executor failure')
        if plan.phase=='descent' and fail!='gap':aabb[:,2]-=.115
    task._execute_categorize_plan=execute
    monkeypatch.setattr(m,'checked_plan',plan)
    monkeypatch.setattr(m,'install_solids',lambda world,boxes:events.append(('install',len(boxes))) or ['floor','wall1','wall2','wall3','wall4'])
    monkeypatch.setattr(m,'payload_obstacles',lambda *args:[np.array([[-1,-1,0],[1,1,1]])])
    return task,actor,events,anchor


def test_all_five_basket_solids_keep_original_inflation(monkeypatch):
    task,actor,events,anchor=make_case(monkeypatch)
    boxes=m.basket_solids(task,0)
    assert len(boxes)==5
    np.testing.assert_allclose(boxes[0],[[-.366,-.316,.755],[-.134,-.084,.785]])
    np.testing.assert_allclose(boxes[1],[[-.366,-.316,.755],[-.316,-.084,.887]])
    np.testing.assert_allclose(boxes[2],[[-.184,-.316,.755],[-.134,-.084,.887]])


@pytest.mark.parametrize('failure',['descent','retreat','gap'])
def test_rejected_release_never_opens_and_restores_obstacles(monkeypatch,failure):
    task,actor,events,anchor=make_case(monkeypatch,failure)
    with pytest.raises(CategorizeMotionRejected):m.lower_release(task,actor,'086_woodenblock','right')
    assert ('open',) not in events
    assert task._categorize_hold_local is anchor
    assert task._categorize_skip_basket is None
    assert [e[0] for e in events].count('remove')==5
    assert events[-1]==('restore',None)
    if failure in ('descent','retreat'):
        assert not any(e[0]=='execute' for e in events)


def test_open_only_after_both_plans_then_retreat_before_restoring_proxy(monkeypatch):
    task,actor,events,anchor=make_case(monkeypatch)
    assert m.lower_release(task,actor,'086_woodenblock','right')
    assert events.index(('plan','retreat'))<events.index(('execute','descent'))<events.index(('open',))
    assert events.index(('open',))<events.index(('execute','retreat'))<events.index(('remove','floor'))
    assert events.count(('step',))==250
    assert task._categorize_hold_local is None
    assert task._categorize_skip_basket is None


def test_unexpected_execution_error_propagates_with_obstacle_cleanup(monkeypatch):
    task,actor,events,anchor=make_case(monkeypatch,'exception')
    with pytest.raises(ValueError,match='unexpected executor'):
        m.lower_release(task,actor,'086_woodenblock','right')
    assert ('open',) not in events and task._categorize_hold_local is anchor
    assert events[-1]==('restore',None)


def test_missing_hold_or_wrong_basket_never_changes_planner(monkeypatch):
    task,actor,events,_=make_case(monkeypatch)
    task._categorize_hold_local=None
    with pytest.raises(CategorizeMotionRejected,match='retained'):
        m.lower_release(task,actor,'086_woodenblock','right')
    assert events==[]
    task._categorize_hold_local=np.zeros(3);task.basket_poses=[Pose([.5,.5,.821])]
    with pytest.raises(CategorizeMotionRejected,match='above assigned basket'):
        m.lower_release(task,actor,'086_woodenblock','right')
    assert events==[]


@pytest.mark.parametrize('hazard',['finger','payload',None])
def test_dense_path_checks_full_opening_and_payload_before_acceptance(monkeypatch,hazard):
    widths=[];precisions=[]
    def collision(q):
        widths.append(q[-1]);return ['wall'] if hazard=='finger' and q[-1]>.039 else []
    backend=NS(robot=NS(get_user_joint_names=lambda:[f'joint{i}' for i in range(7)]+['finger1','finger2']),
        check_for_self_collision=lambda q:[],check_for_env_collision=collision)
    def plan(q,target,qpos_step):
        precisions.append(qpos_step)
        return NS(success=qpos_step!=.005,position=np.array([np.zeros(7),np.ones(7)*.1]))
    arm=NS(planner=NS(_mplib=backend,_pad_qpos=lambda q:np.r_[q,0.,0.],plan_screw_path=plan),tcp_offset=Pose())
    task=NS(_categorize_plan_valid=lambda arm,p,*args:p.success,TABLE_TOP_Z=.765,_OBSTACLE_MARGIN_Z=.01)
    monkeypatch.setattr(m,'mplib_world_link_pose',lambda *args:Pose([0,0,1.]))
    monkeypatch.setattr(m,'payload_clear',lambda *args:hazard!='payload')
    result=m.checked_plan(task,arm,np.zeros(7),np.r_[np.zeros(3),1,0,0,0],np.zeros((8,3)),[])
    assert (result is not None)==(hazard is None)
    assert max(widths)==.04
    if hazard is None:assert precisions==[.005,.004]


def test_failed_descent_releases_from_hover_when_shallow_release_is_enabled(monkeypatch):
    task,actor,events,anchor=make_case(monkeypatch,'descent')
    task.CATEGORIZE_SHALLOW_RELEASE=True
    monkeypatch.setattr(m,'checked_plan',lambda *a:events.append(('plan',)) or None)
    assert m.lower_release(task,actor,'086_woodenblock','right') is True
    assert ('open',) in events and not any(e[0]=='execute' for e in events)
    assert [e for e in events if e==('plan',)]==[('plan',)]*3   # original + two shallow heights
    assert task._categorize_hold_local is None
    assert task._categorize_skip_basket is None
    assert events[-1]==('restore',None)
