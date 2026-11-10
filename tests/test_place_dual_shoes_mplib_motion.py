"""Failure-specific guards for native physical shoe placement."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.planning.base import PlanResult
from envs.task_bases.place_dual_shoes_motion import _ShoeMotionRejected, _pose_matches
from envs.task_bases.place_dual_shoes import DualShoesTaskCore
from envs.utils import Pose


@pytest.fixture
def native():
    state = {"tcp": Pose([0., 0., .9]), "collision": False, "executed": []}
    backend = SimpleNamespace(check_for_env_collision=lambda _q: ["table"] if state["collision"] else [])
    arm = SimpleNamespace(n_arm=7, tcp_offset=Pose([0, 0, .1034]),
                          get_arm_qpos=lambda: np.zeros(7),
                          get_ee_pose=lambda: state["tcp"].to_pose7(),
                          planner=SimpleNamespace(_mplib=backend, _pad_qpos=lambda q: np.r_[q, 0., 0.]))
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.plan_success = True
    task._shoe_fk_tcp = lambda *_a: state["tcp"]
    task._shoe_refresh_avatar = lambda _tag: None
    task.execute_plan = lambda plan, _tag: state["executed"].append(plan)
    task.step_sim = lambda: None
    return task, arm, state


@pytest.mark.parametrize("fault", ["nan", "jump", "branch", "velocity", "collision", "fk", "orientation"])
def test_unsafe_native_path_never_executes(native, fault):
    task, arm, state = native
    target = Pose([0., 0., .9])
    pos = np.stack([np.zeros(7), np.full(7, .01)])
    velocity = np.zeros_like(pos)
    if fault == "nan": pos[1, 0] = np.nan
    if fault == "jump": pos[1, 0] = .31
    if fault == "branch": pos = np.linspace(np.zeros(7), np.full(7, 1.4), 20); velocity = np.zeros_like(pos)
    if fault == "velocity": velocity = np.zeros((1, 7))
    if fault == "collision": state["collision"] = True
    if fault == "fk": state["tcp"] = Pose([0, 0, .92])
    if fault == "orientation": state["tcp"] = Pose([0, 0, .9], [0, 1, 0, 0])
    plan = PlanResult(True, pos, velocity)
    arm.planner.plan_screw_path = lambda *_a, **_k: plan
    with pytest.raises(_ShoeMotionRejected):
        task._shoe_move(target, "right", "test")
    assert not state["executed"] and not task.plan_success


def test_measured_lag_rejected_after_valid_native_execution(native):
    task, arm, state = native
    target = Pose([0, 0, .9])
    task._shoe_fk_tcp = lambda *_a: target
    state["tcp"] = Pose([0, 0, .87])
    arm.planner.plan_screw_path = lambda *_a, **_k: PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7)))
    with pytest.raises(_ShoeMotionRejected, match="arrival"):
        task._shoe_move(target, "right", "lift")
    assert len(state["executed"]) == 1


def test_tiny_lift_has_tighter_fk_gate():
    target = Pose([0, 0, .9])
    actual = Pose([0, 0, .912])
    assert _pose_matches(actual, target)
    assert not _pose_matches(actual, target, tiny=True)


@pytest.mark.parametrize('fault', ['converging', 'stalled', 'arrived', 'invalid'])
def test_open_correction_waits_boundedly_without_relaxing_arrival(native, fault):
    task, arm, state = native
    target = Pose([0., 0., .9])
    task._shoe_fk_tcp = lambda *_: target
    arm.planner.plan_screw_path = lambda *_a, **_k: PlanResult(
        True, np.zeros((2, 7)), np.zeros((2, 7)))
    state['steps'] = 0
    def step():
        state['steps'] += 1
        # Recorded failure is just outside10mm after60 steps, then converges.
        error = .0095 if fault == 'arrived' or (fault == 'converging' and state['steps'] >= 90) else .0103
        state['tcp'] = Pose(target.p-[0., 0., error], target.q)
    task.step_sim = step
    if fault == 'invalid':
        arm.get_ee_pose = lambda: np.full(7, np.nan)
    if fault in ('stalled', 'invalid'):
        with pytest.raises(_ShoeMotionRejected, match='arrival|invalid measured'):
            task._shoe_move(target, 'right', 'open correction', tiny=True, settle_open=True)
        assert not task.plan_success
    else:
        actual = task._shoe_move(target, 'right', 'open correction', tiny=True, settle_open=True)
        assert _pose_matches(actual, target, tiny=True)
    assert len(state['executed']) == 1
    assert state['steps'] == {'converging': 90, 'stalled': 210, 'arrived': 60, 'invalid': 60}[fault]


def test_held_motion_does_not_get_open_correction_wait(native):
    task, arm, state = native
    target = Pose([0., 0., .9])
    task._shoe_fk_tcp = lambda *_: target
    arm.planner.plan_screw_path = lambda *_a, **_k: PlanResult(
        True, np.zeros((2, 7)), np.zeros((2, 7)))
    state['tcp'] = Pose([0., 0., .8897])
    calls = []
    task.step_sim = lambda: calls.append(1)
    with pytest.raises(_ShoeMotionRejected, match='arrival'):
        task._shoe_move(target, 'right', 'tiny lift', tiny=True)
    assert len(calls) == 60 and len(state['executed']) == 1


def test_tiny_lift_preserves_existing_bias_but_requests_measured_30mm(native):
    task, arm, state = native
    arm._cached_target = np.r_[np.full(7,.1), .012, .012]
    start = state['tcp']
    offset = np.array([-.0016,.0045,.0087])
    task._shoe_fk_tcp = lambda tag,q: Pose(start.p+offset,start.q)
    calls = []
    def move(target, tag, label, **kwargs):
        calls.append((target,tag,label,kwargs))
        return target
    task._shoe_move = move
    task._shoe_tiny_lift(start,'right')
    target,tag,label,kwargs = calls[0]
    np.testing.assert_allclose(target.p,start.p+[0,0,.030])
    np.testing.assert_allclose(kwargs['tracking_offset'],offset)
    assert tag=='right' and label=='tiny lift' and kwargs['tiny']


@pytest.mark.parametrize('fault',['missing','shape','nan','fk','large_bias','rotation'])
def test_tiny_bias_rejects_unreliable_command_before_execution(native,fault):
    task, arm, state = native
    arm._cached_target = np.zeros(9)
    command = Pose(state['tcp'].p+[0,0,.009])
    if fault=='missing': arm._cached_target=None
    if fault=='shape': arm._cached_target=np.zeros(7)
    if fault=='nan': arm._cached_target[1]=np.nan
    if fault=='fk': command=None
    if fault=='large_bias': command=Pose(state['tcp'].p+[0,0,.0121])
    if fault=='rotation': command=Pose(command.p,[np.cos(.03),0,0,np.sin(.03)])
    task._shoe_fk_tcp=lambda *_: command
    calls=[]
    task._shoe_move=lambda *_a,**_k: calls.append(1)
    with pytest.raises(_ShoeMotionRejected):
        task._shoe_tiny_lift(state['tcp'],'right')
    assert not calls and not task.plan_success


@pytest.mark.parametrize('fault',['non_tiny','open','shape','nan','large'])
def test_motion_rejects_invalid_offset_before_planning(native,fault):
    task,arm,state=native
    offset=np.array([0.,0.,.009])
    if fault=='shape': offset=np.zeros(4)
    if fault=='nan': offset[0]=np.nan
    if fault=='large': offset[2]=.0121
    with pytest.raises(_ShoeMotionRejected,match='bounded tracking'):
        task._shoe_move(state['tcp'],'right','test',tiny=fault!='non_tiny',
                        settle_open=fault=='open',tracking_offset=offset)
    assert not state['executed']


@pytest.mark.parametrize('arrives_at_command',[False,True])
def test_corrected_plan_checks_command_fk_and_original_measured_arrival(native,arrives_at_command):
    task,arm,state=native
    target=Pose([0,0,.93])
    offset=np.array([0.,0.,.0115])
    command=Pose(target.p+offset,target.q)
    def plan(_seed,link,**kwargs):
        np.testing.assert_allclose(Pose.from_pose7(link).p+arm.tcp_offset.p,command.p)
        return PlanResult(True,np.zeros((2,7)),np.zeros((2,7)))
    arm.planner.plan_screw_path=plan
    task._shoe_fk_tcp=lambda *_:command
    def execute(plan,_tag):
        state['executed'].append(plan)
        state['tcp']=command if arrives_at_command else target
    task.execute_plan=execute
    if arrives_at_command:
        with pytest.raises(_ShoeMotionRejected,match='arrival'):
            task._shoe_move(target,'right','tiny lift',tiny=True,tracking_offset=offset)
    else:
        actual=task._shoe_move(target,'right','tiny lift',tiny=True,tracking_offset=offset)
        np.testing.assert_allclose(actual.p,target.p)
    assert len(state['executed'])==1


@pytest.mark.parametrize('yaw', [-.7, 0., .7])
def test_midbody_grasp_tracks_actual_mesh_frame_and_closes_across_width(yaw):
    from envs.robot.franka_robot import FrankaArm
    from envs.task_bases.place_dual_shoes_motion import _shoe_vertices
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    pose = Pose([.1,-.2,.765], [np.cos(yaw/2),0,0,np.sin(yaw/2)])*Pose(
        [0,0,0], [2**-.5,2**-.5,0,0])
    arm = FrankaArm(None,Pose())
    task.robot = SimpleNamespace(get_arm=lambda _:arm)
    task.TABLE_TOP_Z = .765
    task._shoe_scale = np.array([.11,.11,.11])
    task._shoe_pos_quat = lambda _: (pose.p,pose.q)
    task._choose_grip_end = lambda _:1
    task._object_grip_yaw = lambda _:0.
    task._shoe_grip_xy = lambda *_:pose.p[:2]
    task._grip_yaw_deg = 37.
    _,scale,candidates = task._shoe_candidates(object(),'right')
    center = [g for g in candidates if g.name.startswith('shoe_midbody_')]
    assert len(center) == 2 and candidates[:2] == center
    vertices = _shoe_vertices()*.11
    expected = pose*Pose([0,vertices[:,1].min()+.060,0])
    for grasp in center:
        world = grasp.to_world(pose,scale)
        np.testing.assert_allclose(world.p,expected.p,atol=1e-12)
        closing = world.to_matrix()[:3,1]
        width = pose.to_matrix()[:3,0]
        assert abs(np.dot(closing,width)) == pytest.approx(1.)
        np.testing.assert_allclose(world.to_matrix()[:3,2],-pose.to_matrix()[:3,1],atol=1e-12)
    assert task._grip_yaw_deg == 37.

    import trimesh
    from envs.utils import ASSETS_PATH
    oblique = [g for g in candidates if g.name.startswith('shoe_oblique_')]
    assert len(oblique) == 4
    mesh = trimesh.load(ASSETS_PATH/'objects/041_shoe/visual/base0.glb',force='mesh',process=False)
    mesh.vertices *= scale
    yawed = [g for g in candidates if g.name.startswith('shoe_yaw_')]
    assert len(yawed) == 4
    for grasp in yawed:
        local = grasp.to_world(Pose(),scale)
        world = grasp.to_world(pose,scale)
        np.testing.assert_allclose(local.p,[0,vertices[:,1].min()+.062,0],atol=1e-12)
        np.testing.assert_allclose(world.to_matrix()[:3,2],-pose.to_matrix()[:3,1],atol=1e-12)
        closing = local.to_matrix()[:3,1]
        assert abs(closing[0]) == pytest.approx(np.cos(np.deg2rad(30)))
        assert closing[1] == pytest.approx(0.,abs=1e-12)
        hits,_,_ = mesh.ray.intersects_location(np.array([local.p-.12*closing]),
                                              np.array([closing]),multiple_hits=True)
        crossings = [float(np.dot(p-local.p,closing)) for p in hits]
        assert len(crossings)==2 and max(abs(x) for x in crossings)<.04
    for grasp in oblique:
        local = grasp.to_world(Pose(),scale)
        world = grasp.to_world(pose,scale)
        np.testing.assert_allclose(local.p,[0,vertices[:,1].min()+.060,0],atol=1e-12)
        approach = world.to_matrix()[:3,2]
        assert np.dot(approach,pose.to_matrix()[:3,1]) == pytest.approx(-np.cos(np.deg2rad(15)))
        assert abs(np.dot(approach,pose.to_matrix()[:3,0])) == pytest.approx(np.sin(np.deg2rad(15)))
        closing = local.to_matrix()[:3,1]
        assert abs(closing[0]) == pytest.approx(np.cos(np.deg2rad(15)))
        hits,_,_ = mesh.ray.intersects_location(np.array([local.p-.12*closing]),
                                              np.array([closing]),multiple_hits=True)
        crossings = [float(np.dot(p-local.p,closing)) for p in hits]
        assert len(crossings)==2 and max(abs(x) for x in crossings)<.04


@pytest.mark.parametrize('yaw', [-.7,0.,.7])
@pytest.mark.parametrize('side', [-1.,1.])
def test_offset_lean_clears_shoe_and_brackets_actual_mesh_after_world_shift(yaw,side):
    from envs.robot.franka_robot import FrankaArm
    from envs.grasp import tcp_to_link_pose
    from envs.task_bases.place_dual_shoes_motion import _shoe_mesh, _shoe_vertices, _PALM_LO, _PALM_HI
    task=DualShoesTaskCore.__new__(DualShoesTaskCore)
    pose=Pose([.1,-.2,.765],[np.cos(yaw/2),0,0,np.sin(yaw/2)])*Pose([0,0,0],[2**-.5,2**-.5,0,0])
    arm=FrankaArm(None,Pose())
    task.robot=SimpleNamespace(get_arm=lambda _:arm)
    task.TABLE_TOP_Z=.765;task._shoe_scale=np.array([.11]*3)
    task._shoe_pos_quat=lambda _:(pose.p,pose.q)
    task._choose_grip_end=lambda _:1;task._object_grip_yaw=lambda _:0.
    task._shoe_grip_xy=lambda *_:pose.p[:2];task._grip_yaw_deg=0.
    away=side*pose.to_matrix()[:3,0]
    task.shoebox=SimpleNamespace(entity=SimpleNamespace(get_pos=lambda:pose.p-.2*away))
    _,scale,candidates=task._shoe_candidates(object(),'right')
    shifted=[g for g in candidates if g.name.startswith('shoe_offset_lean_')]
    assert len(shifted)==2  # Opposite lean exceeds one side of the opening.
    vertices=_shoe_vertices()*scale
    center=pose*Pose([0,vertices[:,1].min()+.060,0])
    for grasp in shifted:
        local=grasp.to_world(Pose(),scale);world=grasp.to_world(pose,scale)
        np.testing.assert_allclose(world.p-center.p,.010*away,atol=1e-12)
        closing=local.to_matrix()[:3,1]
        hits,_,_=_shoe_mesh().ray.intersects_location(np.array([(local.p-.12*closing)/scale]),np.array([closing]),multiple_hits=True)
        crossings=sorted(float(np.dot(hit*scale-local.p,closing)) for hit in hits)
        assert len(crossings)==2 and -.04<crossings[0]<0<crossings[1]<.04
        link=tcp_to_link_pose(world,arm.tcp_offset)
        hand_vertices=(vertices@pose.to_matrix()[:3,:3].T+pose.p-link.p)@link.to_matrix()[:3,:3]
        mask=np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
        assert mask.any() and hand_vertices[mask,2].min()-_PALM_HI[2]>=.008


def test_execution_preserves_native_duration_and_gripper_timeline(monkeypatch):
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    task._shoe_motion_active = True
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    captured = []
    monkeypatch.setattr(BaseTask, "execute_plan", lambda _self, result, tag, gripper_result=None:
                        captured.append((result, gripper_result)))
    pos = np.arange(21).reshape(3, 7) * .001
    vel = np.ones_like(pos)
    grip = np.array([1., .6, .025])
    task.execute_plan(PlanResult(True, pos, vel), "right", grip)
    np.testing.assert_array_equal(captured[0][0].position, np.repeat(pos, 2, axis=0))
    np.testing.assert_array_equal(captured[0][1], np.repeat(grip, 2))
    assert len(captured[0][0].position) * .002 == len(pos) * .004


def _physical_stub(*, miss=False, carry_drop=False, held_release=False, wait=False):
    from envs.grasp import GraspPose
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    state = {'tcp': Pose([0., 0., .82]), 'shoe': np.array([0., 0., .80]),
             'closed': False, 'calls': [], 'public': [], 'steps': 0}
    arm = SimpleNamespace(n_arm=7, _cached_target=np.zeros(9),
                          tcp_offset=Pose(), get_ee_pose=lambda: state['tcp'].to_pose7())
    shoe = object()  # No actor setters or attachment API are available.
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    task.plan_success = True
    task._shoe_motion_active = True
    task.TABLE_TOP_Z = .765
    task._shoe_refresh_avatar = lambda _: None
    task._shoe_fk_tcp = lambda *_: state['tcp']
    task._shoe_pos_quat = lambda _: (state['shoe'].copy(), np.array([1., 0., 0., 0.]))
    task._shoe_world_aabb = lambda _: np.array([state['shoe']-[.02,.02,.02], state['shoe']+[.02,.02,.02]])
    task._box_rim_z = .87
    task._box_slot_xy = lambda _: (.15, -.1)
    task.shoebox = SimpleNamespace(entity=SimpleNamespace(get_AABB=lambda: np.array([[-.1,-.2,.766],[.2,.1,.87]])))
    task.RELEASE_FORCE_N = None
    task._shoe_candidates = lambda *_: (Pose(state['shoe']), .11,
        [GraspPose('test', [0,0,.02/.11], [1,0,0,0], scale_frame='mesh_unit')])
    task.step_sim = lambda: state.update(steps=state['steps']+1)
    task.avatar = SimpleNamespace(spare=lambda: state['steps']>=3) if wait else None
    def public(**kwargs):
        state['public'].append(kwargs)
        if wait and len(state['public']) == 1:
            return None
        target = state['tcp']
        if miss:
            state['tcp'] = Pose(target.p+[0,0,.047], target.q)
        return target, target, kwargs['candidate_grasps'][0]
    def close(*_):
        state['closed'] = True
        state['calls'].append(('close', None))
    def move(target, _tag, label, **_kwargs):
        state['calls'].append(('move', label))
        if state['closed'] or held_release:
            state['shoe'] += target.p-state['tcp'].p
        state['tcp'] = target
        if carry_drop and label == 'carry':
            state['shoe'][2] -= .10
        return target
    def release(*_args, **_kwargs):
        state['calls'].append(('open', None))
        state['closed'] = False
        if not held_release:
            state['shoe'][2] = .806
    task.select_and_execute_grasp = public
    task._close_until_contact = close
    task._shoe_move = move
    task.open_gripper = release
    return task, shoe, state


def test_public_actual_scale_and_complete_physical_release():
    task, shoe, state = _physical_stub()
    assert task._shoe_native_pick(shoe, 'right', 'shoe')
    assert task._shoe_native_place(shoe, 'right', 'shoe', 0)
    kwargs = state['public'][0]
    assert kwargs['object_scale'] == .11 and kwargs['table_z'] == .765
    assert callable(kwargs['post_plan_check']) and kwargs['plan_attempts'] == 3
    assert [v for k,v in state['calls'] if k=='move'] == ['tiny lift','lift','carry','descent','release retreat']
    np.testing.assert_allclose(task._shoe_grasp_local, [0,0,.02])


@pytest.mark.parametrize('center_plans', [True, False])
def test_center_pool_precedes_higher_approach_score_end_grasp(center_plans):
    from envs.grasp import GraspPose
    task, shoe, state = _physical_stub()
    _, scale, original = task._shoe_candidates(shoe, 'right')
    g = original[0]
    end = GraspPose('shoe_heel_side_0', g.pose.p, g.pose.q, scale_frame='mesh_unit')
    centers = [GraspPose(f'shoe_midbody_{flip}', g.pose.p, g.pose.q,
                        scale_frame='mesh_unit') for flip in (0, 180)]
    task._shoe_candidates = lambda *_: (Pose(state['shoe']), scale, [end, *centers])
    public = task.select_and_execute_grasp
    pools = []
    def ranked_public(**kwargs):
        pool = kwargs['candidate_grasps']
        pools.append(pool)
        assert kwargs['max_candidates'] == len(pool)
        assert kwargs['screw_qpos_step'] == .02 and kwargs['plan_attempts'] == 3
        assert callable(kwargs['post_plan_check'])
        if pool == centers and not center_plans:
            assert not state['closed'] and not state['calls']
            return None
        return public(**kwargs)
    task.select_and_execute_grasp = ranked_public
    assert task._shoe_native_pick(shoe, 'right', 'shoe')
    assert pools == ([centers] if center_plans else [centers, [end]])
    assert sum(k == 'close' for k, _ in state['calls']) == 1


@pytest.mark.parametrize('selected_pool',['center','yawed','shifted','oblique','ends',None])
def test_oblique_pool_only_after_upright_planning_failure(selected_pool):
    from envs.grasp import GraspPose
    task,shoe,state=_physical_stub()
    _,scale,original=task._shoe_candidates(shoe,'right')
    g=original[0]
    groups={key:[GraspPose(name,g.pose.p,g.pose.q,scale_frame='mesh_unit')]
        for key,name in (('center','shoe_midbody_0'),('yawed','shoe_yaw_-30_180'),('shifted','shoe_offset_lean_-22.5_0'),('oblique','shoe_oblique_-15_180'),('ends','shoe_heel_side_0'))}
    task._shoe_candidates=lambda *_:(Pose(state['shoe']),scale,[*groups['ends'],*groups['oblique'],*groups['shifted'],*groups['yawed'],*groups['center']])
    public=task.select_and_execute_grasp
    attempted=[]
    def choose(**kwargs):
        assert not state['closed'] and not state['calls']
        key=next(k for k,v in groups.items() if v==kwargs['candidate_grasps'])
        attempted.append(key)
        return public(**kwargs) if key==selected_pool else None
    task.select_and_execute_grasp=choose
    if selected_pool is None:
        with pytest.raises(_ShoeMotionRejected,match='no PUBLIC acquisition'):
            task._shoe_native_pick(shoe,'right','shoe')
        assert not state['closed']
    else:
        assert task._shoe_native_pick(shoe,'right','shoe')
        assert sum(k=='close' for k,_ in state['calls'])==1
    expected=['center','yawed','shifted','oblique','ends']
    assert attempted==(expected if selected_pool is None else expected[:expected.index(selected_pool)+1])


@pytest.mark.parametrize('selected',['shoe_yaw_-30_180','shoe_offset_lean_-22.5_0','shoe_oblique_-15_180'])
def test_failed_executed_oblique_grasp_never_falls_back_to_end_pinch(selected):
    from envs.grasp import GraspPose
    task,shoe,state=_physical_stub()
    _,scale,original=task._shoe_candidates(shoe,'right')
    g=original[0]
    candidates=[GraspPose(name,g.pose.p,g.pose.q,scale_frame='mesh_unit')
        for name in ('shoe_midbody_0','shoe_yaw_-30_180','shoe_offset_lean_-22.5_0','shoe_oblique_-15_180','shoe_heel_side_0')]
    task._shoe_candidates=lambda *_:(Pose(state['shoe']),scale,candidates)
    public=task.select_and_execute_grasp
    attempted=[]
    def choose(**kwargs):
        name=kwargs['candidate_grasps'][0].name
        attempted.append(name)
        return public(**kwargs) if name==selected else None
    task.select_and_execute_grasp=choose
    task._shoe_retained=lambda *_:task._shoe_reject('tiny lift: lost purchase')
    with pytest.raises(_ShoeMotionRejected,match='lost purchase'):
        task._shoe_native_pick(shoe,'right','shoe')
    order=['shoe_midbody_0','shoe_yaw_-30_180','shoe_offset_lean_-22.5_0','shoe_oblique_-15_180']
    assert attempted==order[:order.index(selected)+1]
    assert sum(k=='close' for k,_ in state['calls'])==1


@pytest.mark.parametrize('fault', ['arrival', 'tiny_lift'])
def test_center_acquisition_failure_never_retries_end_grasps(fault):
    from envs.grasp import GraspPose
    task, shoe, state = _physical_stub(miss=fault == 'arrival')
    _, scale, original = task._shoe_candidates(shoe, 'right')
    g = original[0]
    candidates = [GraspPose(name, g.pose.p, g.pose.q, scale_frame='mesh_unit')
                  for name in ('shoe_midbody_0', 'shoe_midbody_180', 'shoe_heel_side_0')]
    task._shoe_candidates = lambda *_: (Pose(state['shoe']), scale, candidates)
    if fault == 'tiny_lift':
        task._shoe_retained = lambda *_: task._shoe_reject('tiny lift: lost purchase')
    with pytest.raises(_ShoeMotionRejected):
        task._shoe_native_pick(shoe, 'right', 'shoe')
    assert len(state['public']) == 1
    assert [g.name for g in state['public'][0]['candidate_grasps']] == [
        'shoe_midbody_0', 'shoe_midbody_180']
    assert sum(k == 'close' for k, _ in state['calls']) == (fault == 'tiny_lift')


def test_empty_grasp_pools_do_not_call_public_or_close():
    task, shoe, state = _physical_stub()
    task._shoe_candidates = lambda *_: (Pose(state['shoe']), .11, [])
    with pytest.raises(_ShoeMotionRejected, match='no PUBLIC acquisition'):
        task._shoe_native_pick(shoe, 'right', 'shoe')
    assert not state['public'] and not state['calls']


def test_missed_public_arrival_never_closes():
    task, shoe, state = _physical_stub(miss=True)
    with pytest.raises(_ShoeMotionRejected, match='arrival'):
        task._shoe_native_pick(shoe, 'right', 'shoe')
    assert not state['closed'] and not state['calls']


def test_open_grasp_refinement_corrects_recorded_shoe_bias():
    task, shoe, state = _physical_stub()
    target = state['tcp']
    bias = np.array([-.002185,.005627,.007274])
    state['tcp'] = Pose(target.p-bias,target.q)
    commands = []
    def move(command, tag, label, *, tiny=False, settle_open=False):
        assert tiny and settle_open and tag=='right' and not state['closed']
        commands.append(command)
        state['tcp'] = Pose(command.p-bias,command.q)
        return state['tcp']
    task._shoe_move = move
    actual = task._shoe_refine_open_grasp(target,shoe,'right')
    assert len(commands)==1 and not state['calls']
    np.testing.assert_allclose(actual.p,target.p,atol=1e-12)


@pytest.mark.parametrize('fault',['large_bias','rotation','body_motion','nonfinite_body','stalled','total_bound'])
def test_open_shoe_refinement_rejects_unreliable_feedback(fault):
    task, shoe, state = _physical_stub()
    target = state['tcp']
    distance = .0101 if fault=='large_bias' else .007 if fault=='total_bound' else .003
    state['tcp'] = Pose(target.p-[distance,0,0],target.q)
    if fault=='rotation':
        state['tcp'] = Pose(state['tcp'].p,[np.cos(.03),0,0,np.sin(.03)])
    commands = []
    def move(command, tag, label, *, tiny=False, settle_open=False):
        assert tiny and settle_open and not state['closed']
        commands.append(command)
        if fault=='body_motion': state['shoe'][0]+=.0031
        if fault=='nonfinite_body': state['shoe'][0]=np.nan
        return state['tcp']
    task._shoe_move = move
    with pytest.raises(_ShoeMotionRejected):
        task._shoe_refine_open_grasp(target,shoe,'right')
    assert not task.plan_success and not state['closed'] and len(commands)<=3
    if fault in ('large_bias','rotation'): assert not commands


def test_unresolved_refinement_aborts_public_pick_before_closing():
    task, shoe, state = _physical_stub()
    public = task.select_and_execute_grasp
    def biased_public(**kwargs):
        result = public(**kwargs)
        state['tcp'] = Pose(result[0].p-[.0101,0,0],result[0].q)
        return result
    task.select_and_execute_grasp = biased_public
    with pytest.raises(_ShoeMotionRejected,match='alignment'):
        task._shoe_native_pick(shoe,'right','shoe')
    assert not state['closed'] and not state['calls']


def test_planning_waits_for_avatar_before_retrying_without_grasp():
    task, shoe, state = _physical_stub(wait=True)
    assert task._shoe_native_pick(shoe, 'right', 'shoe')
    assert len(state['public']) == 2 and state['steps'] >= 3
    assert sum(k=='close' for k,_ in state['calls']) == 1


def test_carried_shoe_drop_aborts_before_open():
    task, shoe, state = _physical_stub(carry_drop=True)
    task._shoe_native_pick(shoe, 'right', 'shoe')
    with pytest.raises(_ShoeMotionRejected, match='lost measured grasp'):
        task._shoe_native_place(shoe, 'right', 'shoe', 0)
    assert ('open',None) not in state['calls'] and ('move','descent') not in state['calls']


def test_held_shoe_after_withdrawal_is_failure():
    task, shoe, state = _physical_stub(held_release=True)
    task._shoe_native_pick(shoe, 'right', 'shoe')
    with pytest.raises(_ShoeMotionRejected, match='followed'):
        task._shoe_native_place(shoe, 'right', 'shoe', 0)
    assert ('move','release retreat') in state['calls']


@pytest.mark.parametrize('scale', [0., -.1, [.1,.11,.1], [float('nan')]*3])
def test_invalid_scale_rejected_before_public_grasp(scale):
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    task._shoe_scale = scale
    with pytest.raises(_ShoeMotionRejected, match='scale'):
        task._shoe_scale_scalar()


def test_fixture_poses_use_solver_world_frame():
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    calls = []
    def pos(*, relative):
        calls.append(relative)
        return np.array([.1,.2,.3])
    def quat(*, relative):
        calls.append(relative)
        return np.array([1.,0,0,0])
    pose = task._shoe_geom_pose(SimpleNamespace(get_pos=pos, get_quat=quat))
    assert calls == [False,False]
    np.testing.assert_allclose(pose.p,[.1,.2,.3])


def test_native_convex_fixture_updates_and_cleanup_preserve_foreign_objects():
    import genesis as gs
    import trimesh
    task = DualShoesTaskCore.__new__(DualShoesTaskCore)
    objects = {'foreign_avatar': object()}
    entries = set()
    world = SimpleNamespace(has_object=lambda name: name in objects,
        add_object=lambda name,obj: objects.update({name:obj}),
        remove_object=lambda name: objects.pop(name))
    acm = SimpleNamespace(has_entry=lambda name: name in entries,
        set_entry=lambda _base,name,_allow: entries.add(name),
        remove_entry=lambda name: entries.discard(name))
    backend = SimpleNamespace(planning_world=world, acm=acm)
    arm = SimpleNamespace(planner=SimpleNamespace(_mplib=backend))
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    task.config = {'robot_type':'franka'}
    task.table = SimpleNamespace(get_AABB=lambda: np.array([[-.6,-.9,.715],[.6,.2,.765]]))
    position = np.array([.1,.2,.8])
    def geom(kind):
        return SimpleNamespace(type=kind, data=np.array([.1,.01,.2]),
            get_pos=lambda *,relative: position.copy() if relative is False else None,
            get_quat=lambda *,relative: np.array([1.,0,0,0]) if relative is False else None)
    task.shoebox = SimpleNamespace(entity=SimpleNamespace(links=[SimpleNamespace(
        geoms=[geom(gs.GEOM_TYPE.BOX) for _ in range(5)])]))
    convex = geom(gs.GEOM_TYPE.MESH)
    mesh = trimesh.creation.icosphere(subdivisions=0, radius=.03)
    convex.init_verts, convex.init_faces, convex.is_convex = mesh.vertices, mesh.faces, True
    target = object()
    task.shoes = [target, SimpleNamespace(entity=SimpleNamespace(links=[SimpleNamespace(geoms=[convex])]))]
    with pytest.raises(RuntimeError, match='abort body'):
        with task._shoe_motion_scene(target,'right'):
            assert len(objects) == 8 and task._shoe_motion_active
            assert entries == {'place_dual_shoes_table'}
            previous = objects['place_dual_shoes_other_1_0']
            position[0] += .15
            task._shoe_refresh_fixtures()
            assert objects['place_dual_shoes_other_1_0'] is not previous
            raise RuntimeError('abort body')
    assert list(objects) == ['foreign_avatar'] and not entries
    assert not task._shoe_motion_active and task._shoe_scene_state is None
