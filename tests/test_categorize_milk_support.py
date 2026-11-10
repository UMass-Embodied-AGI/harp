from types import SimpleNamespace as NS

import numpy as np
import pytest
from transforms3d.quaternions import qinverse, qmult

from envs.task_bases.categorize_support import milk_support as m
from envs.task_bases.categorize_support.motion import CategorizeMotionRejected
from envs.utils import Pose


@pytest.mark.parametrize('bounds', [np.zeros((3,2)), [[0,0,0],[1,np.nan,1]], [[1,0,0],[0,1,1]]])
def test_invalid_geometry_rejected(bounds):
    with pytest.raises(ValueError):
        m.bounds_corners(bounds)


def test_payload_guard_covers_solid_interior_table_and_touching_boundaries():
    corners=m.bounds_corners([[-.02,-.02,-.02],[.02,.02,.02]])
    tcp=Pose([0,0,1.])
    assert m.payload_clear(corners,tcp,[],.9)
    assert not m.payload_clear(corners,tcp,[],.99)
    assert not m.payload_clear(corners,tcp,[np.array([[-.001,-.001,.999],[.001,.001,1.001]])],.9)
    assert not m.payload_clear(corners,tcp,[np.array([[.02,-.01,.99],[.03,.01,1.01]])],.9)


def test_payload_obstacles_retain_basket_capsule_and_box_padding():
    task=NS(_OBSTACLE_MARGIN_XY=.02,_OBSTACLE_MARGIN_Z=.01,
        _BASKET_HALF_X=.096,_BASKET_HALF_Y=.096,_BASKET_HALF_Z=.056,
        basket_poses=[Pose([0,0,1.])],_sort_targets=[],
        avatar_collider=NS(current_capsules=lambda:[('arm',[1,0,0],[1,0,1],.1)],
            current_boxes=lambda:[('hand',[2,0,0],np.eye(3),[.1,.2,.3])]),
        _neutral_object_obstacle_points=lambda:None)
    boxes=m.payload_obstacles(task,None)
    np.testing.assert_allclose(boxes[0],[[-.116,-.116,.934],[.116,.116,1.066]])
    np.testing.assert_allclose(boxes[1],[[.89,-.11,-.11],[1.11,.11,1.11]])
    np.testing.assert_allclose(boxes[2],[[1.895,-.205,-.305],[2.105,.205,.305]])


def setup_case(monkeypatch, *, native_ok=True, collision=False, follows=True):
    pose=[Pose([-.2,.1,1.],[0,np.sqrt(.5),0,-np.sqrt(.5)])]
    object_q=[np.array([1.,0,0,0])]
    target=[None]
    calls=[]
    entity=NS(get_AABB=lambda:np.array([[-.23,.07,.97],[-.17,.13,1.03]]),
              get_qpos=lambda:np.r_[np.zeros(7),.025,.026])
    actor=NS(entity=entity,get_pose=lambda:Pose([-.2,.1,1.],object_q[0]))
    def collide(full):
        calls.append(('fingers',full[-2:].copy()))
        return ['collision'] if collision else []
    backend=NS(robot=NS(get_user_joint_names=lambda:[f'joint{i}' for i in range(7)]+['finger1','finger2']),
               check_for_self_collision=lambda q:[],check_for_env_collision=collide)
    planner=NS(_mplib=backend,_pad_qpos=lambda q:np.r_[q,0.,0.],
        plan_screw_path=lambda *a,**kw:NS(success=native_ok,position=np.array([np.zeros(7),np.ones(7)*.1])))
    arm=NS(entity=entity,planner=planner,_finger_dof_indices=[7,8],tcp_offset=Pose([0,0,.1034]),
        get_ee_pose=lambda:pose[0].to_pose7())
    anchor=np.array([.01,.02,.03])
    def inputs(link,tag):
        target[0]=Pose.from_pose7(link)
        calls.append(('obstacles',))
        return arm,np.zeros(7),link
    def execute(plan,tag):
        calls.append(('execute',))
        relative=qmult(qinverse(pose[0].q),object_q[0])
        pose[0]=target[0]*arm.tcp_offset
        if follows:object_q[0]=qmult(pose[0].q,relative)
    def reject(reason):raise CategorizeMotionRejected(reason)
    task=NS(robot=NS(get_arm=lambda _:arm),_categorize_hold_local=anchor,
        _categorize_inputs=inputs,_categorize_plan_valid=lambda *args:native_ok,
        _execute_categorize_plan=execute,_categorize_arrival=lambda *args:None,
        _reject_categorize_motion=reject,TABLE_TOP_Z=.765,_OBSTACLE_MARGIN_Z=.01)
    monkeypatch.setattr(m,'payload_obstacles',lambda *args:[])
    monkeypatch.setattr(m,'mplib_world_link_pose',lambda *args:target[0])
    return task,NS(actor=actor),calls,anchor


def test_turn_checks_actual_finger_opening_and_keeps_hold_anchor(monkeypatch):
    task,pick,calls,anchor=setup_case(monkeypatch)
    result=m.orient_milk_for_support(task,pick,'right')
    assert abs(m.quat2mat(result)[2,1])>.999
    assert sum(c[0]=='execute' for c in calls)==1
    for call in calls:
        if call[0]=='fingers':np.testing.assert_allclose(call[1],[.025,.026])
    assert task._categorize_hold_local is anchor
    np.testing.assert_allclose(anchor,[.01,.02,.03])


@pytest.mark.parametrize('reason',['native','finger','payload'])
def test_rejected_turn_never_executes_or_opens(monkeypatch,reason):
    task,pick,calls,_=setup_case(monkeypatch,native_ok=reason!='native',collision=reason=='finger')
    if reason=='payload':monkeypatch.setattr(m,'payload_clear',lambda *args:False)
    with pytest.raises(CategorizeMotionRejected,match='no checked native path'):
        m.orient_milk_for_support(task,pick,'right')
    assert not any(c[0]=='execute' for c in calls)


def test_object_must_follow_turn_before_carry(monkeypatch):
    task,pick,calls,_=setup_case(monkeypatch,follows=False)
    with pytest.raises(CategorizeMotionRejected,match='angular slip'):
        m.orient_milk_for_support(task,pick,'right')
    assert sum(c[0]=='execute' for c in calls)==1


def test_missing_hold_is_not_reanchored(monkeypatch):
    task,pick,calls,_=setup_case(monkeypatch)
    task._categorize_hold_local=None
    with pytest.raises(CategorizeMotionRejected,match='retained grasp'):
        m.orient_milk_for_support(task,pick,'right')
    assert calls==[]


def test_already_vertical_jaws_need_no_additional_rotation(monkeypatch):
    task,pick,calls,anchor=setup_case(monkeypatch)
    arm=task.robot.get_arm('right')
    vertical=Pose([-.2,.1,1.],[np.sqrt(.5),np.sqrt(.5),0,0])
    arm.get_ee_pose=lambda:vertical.to_pose7()
    np.testing.assert_allclose(m.orient_milk_for_support(task,pick,'right'),vertical.q)
    assert calls==[] and task._categorize_hold_local is anchor
