"""Dump milk lift feasibility and retained-rotation contracts."""
from types import SimpleNamespace
import numpy as np
import pytest

from envs.manipulation import PickSpec, PlaceSpec
from envs.planning.base import PlanResult
from envs.task_bases.dump_bin import DumpBin
from envs.task_bases import dump_bin as dump, dump_bin_milk as milk
from envs.task_bases.dump_bin_motion import DumpMotionRejected
from envs.utils import Pose
from envs.grasp import GraspPose


@pytest.mark.parametrize('outcome', ['clear', 'collision', 'exception'])
@pytest.mark.parametrize('margin', [None, .003])
def test_milk_preplan_uses_open_fingers_and_cleans_obstacle_and_method(outcome, margin):
    task = DumpBin({'robot_type': 'franka'})
    bounds = np.array([[.05,-.2,.765], [.12,-.13,.89]])
    expected_size = bounds[1]-bounds[0]+2*(.008 if margin is None else margin)
    task._dump_active_actor = SimpleNamespace(get_AABB=lambda: bounds.copy())
    objects, checked = {}, []
    good = PlanResult(True, np.full((2, 7), .1), np.zeros((2, 7)))
    def collision(q):
        checked.append(q.copy())
        np.testing.assert_allclose(objects['dump_milk_pre_target'].get_collision_geometry().side,
                                   expected_size)
        return [object()] if outcome == 'collision' else []
    world = SimpleNamespace(has_object=lambda name: name in objects,
                            add_object=lambda name, box: objects.__setitem__(name, box),
                            remove_object=lambda name: objects.pop(name))
    def original(current, target, **kwargs):
        np.testing.assert_array_equal(current[:7], np.zeros(7))
        np.testing.assert_allclose(current[7:], [.04, .04])
        assert kwargs == {'log': False}
        box = objects['dump_milk_pre_target']
        np.testing.assert_allclose(box.get_collision_geometry().side, expected_size)
        np.testing.assert_allclose(box.get_pose().p, bounds.mean(axis=0))
        planner._mplib.TOPP(np.array([np.zeros(7), np.ones(7)*.1]), .004)
        if outcome == 'exception':
            raise OSError('native planner unavailable')
        return good
    pad = lambda q: np.r_[q, np.zeros(9-len(q))]
    topp_calls = []
    def topp(path, step):
        topp_calls.append(path.copy())
        assert step == .004
        assert len(path) == 5
        np.testing.assert_allclose(path[[0,-1]], [np.zeros(7), np.ones(7)*.1])
    planner = SimpleNamespace(plan_path=original, _pad_qpos=pad,
                              _mplib=SimpleNamespace(planning_world=world, check_for_env_collision=collision, TOPP=topp))
    arm = SimpleNamespace(planner=planner, n_arm=7, gripper_val=1.)
    def run():
        options = {} if margin is None else {'target_margin': margin}
        with milk.milk_approach_planning(task, arm, **options):
            result = planner.plan_path(np.zeros(7), np.zeros(7), log=False)
            assert result.success == (outcome == 'clear')
            assert not objects  # Removed before public straight descent.
    if outcome == 'exception':
        with pytest.raises(OSError, match='native planner unavailable'):
            run()
    else:
        run()
        assert len(checked) == (2 if outcome == 'clear' else 1)
        for q in checked:
            np.testing.assert_allclose(q[7:], [.04, .04])
    assert not objects
    assert planner.plan_path is original and planner._pad_qpos is pad
    assert planner._mplib.TOPP is topp and len(topp_calls) == 1


@pytest.mark.parametrize('margin', [-.001, float('nan'), float('inf')])
def test_body_preplan_rejects_invalid_margin_before_touching_planner(margin):
    task = DumpBin({'robot_type': 'franka'})
    with pytest.raises(DumpMotionRejected, match='obstacle margin'):
        with milk.milk_approach_planning(task, SimpleNamespace(), target_margin=margin):
            pytest.fail('invalid margin must be rejected')


def test_milk_prepath_densification_preserves_vertices_and_segments():
    vertices = np.array([np.zeros(7), [.11,0,0,0,0,0,0], [.11,.07,0,0,0,0,0]])
    dense = milk._densify_milk_pre_path(vertices)
    np.testing.assert_allclose(dense[[0,5,-1]], vertices)
    assert np.max(np.abs(np.diff(dense, axis=0))) <= .025
    np.testing.assert_allclose(dense[:6,1:], 0)
    np.testing.assert_allclose(dense[5:,0], .11)


@pytest.mark.parametrize('bad', [np.zeros((2,6)), np.zeros((1,7)),
    np.full((2,7), np.nan), np.array([np.zeros(7),np.ones(7)*8]),
    np.tile([np.zeros(7),np.ones(7)*6],(100,1))])
def test_milk_prepath_rejects_malformed_or_unbounded_vertices(bad):
    with pytest.raises(ValueError, match='milk pre-approach'):
        milk._densify_milk_pre_path(bad)


def test_milk_lift_retracts_before_high_lift_and_rotation():
    task = DumpBin({})
    tcp = Pose([.1, -.1, .85], [.5, -.5, .5, .5])
    targets = milk.transport_targets(task, tcp, PlaceSpec(np.zeros(3), transport_z=.945))
    np.testing.assert_allclose([p.p for p in targets],
                               [[.1,-.1,.90], [.1,-.15,.90], [.1,-.15,.945], [.1,-.15,.945]])
    for target in targets[:3]:
        np.testing.assert_allclose(target.q, tcp.q)
    np.testing.assert_allclose(targets[-1].to_matrix()[:3,:3], task._R_DOWN)


def test_milk_side_carry_reaches_bin_before_turning_with_measured_offset():
    task = DumpBin({})
    task.big_bin_pose = Pose([.65,-.3,.01])
    tcp = Pose([.1,-.1,.85], [.5,-.5,.5,.5])
    targets = milk.transport_targets(task,tcp,PlaceSpec(np.zeros(3),transport_z=.945),
                                     carry_center=np.array([.11,-.08,.83]))
    assert len(targets) == 5
    np.testing.assert_allclose(targets[2].p, [.15,-.15,.945])
    np.testing.assert_allclose([value.p for value in targets[-2:]],
                               [[.64,-.32,.945],[.64,-.32,.945]])
    for value in targets[:-1]:
        np.testing.assert_allclose(value.q,tcp.q)
    final_q=targets[-1].q
    down_q=task._top_down_tcp(np.zeros(3)).q
    angle=lambda a,b: 2*np.arccos(np.clip(abs(np.dot(a,b)),0,1))
    np.testing.assert_allclose(np.linalg.norm(final_q),1.)
    np.testing.assert_allclose(angle(tcp.q,final_q),angle(tcp.q,down_q)/8)
    np.testing.assert_allclose(angle(final_q,down_q),7*angle(tcp.q,down_q)/8)


@pytest.mark.parametrize('center', [[0,0], [0,0,float('nan')]])
def test_milk_side_carry_rejects_invalid_center(center):
    task=DumpBin({});task.big_bin_pose=Pose([.65,-.3,.01])
    with pytest.raises(DumpMotionRejected,match='side-carry target'):
        milk.transport_targets(task,Pose([.1,-.1,.85]),PlaceSpec(np.zeros(3),transport_z=.945),
                               carry_center=center)


def test_close_to_mount_body_lift_moves_into_high_transport_workspace():
    task=DumpBin({})
    tcp=Pose([.09,-.22,.82],[.5,-.5,.5,.5])
    targets=milk.transport_targets(task,tcp,PlaceSpec(np.zeros(3),transport_z=.945))
    np.testing.assert_allclose(targets[0].p,[.09,-.22,.87])
    np.testing.assert_allclose(targets[1].p,[.09,-.15,.87])
    np.testing.assert_allclose(targets[2].p,[.09,-.15,.945])
    for target in targets[:3]:
        np.testing.assert_allclose(target.q,tcp.q)


@pytest.mark.parametrize('scale',[.05,.7])
@pytest.mark.parametrize('yaw',[0.,np.pi/2])
def test_lower_milk_annotation_scales_catalog_once_and_shifts_world_height(monkeypatch,scale,yaw):
    task=DumpBin({})
    obj=Pose([.1,-.2,.8],[np.cos(yaw/2),0,0,np.sin(yaw/2)])
    catalog=GraspPose('catalog',[.1,.2,.3],[.5,-.5,.5,.5],pre_distance=.12,scale_frame='mesh_unit')
    monkeypatch.setattr(milk,'get_grasp_pose',lambda *a,**kw:catalog)
    result=milk.milk_body_grasp(task,obj,scale,0)
    original=catalog.to_world(obj,scale)
    world=result.to_world(obj,scale)
    np.testing.assert_allclose(world.p,original.p-[0,0,.02])
    np.testing.assert_allclose(world.to_matrix()[:3,:3],original.to_matrix()[:3,:3],atol=1e-12)
    assert result.pre_distance==.060 and result.scale_frame=='runtime'


@pytest.mark.parametrize('screw_valid',[False,True])
def test_milk_rotation_uses_high_rrt_only_after_checked_screw_failure(monkeypatch,screw_valid):
    task=DumpBin({});arm=SimpleNamespace(tcp_offset=Pose())
    screw=PlanResult(True,np.zeros((2,7)),np.zeros((2,7))) if screw_valid else None
    fallback=PlanResult(False);calls=[]
    monkeypatch.setattr(milk,'plan_body_screw',lambda *a:calls.append('screw') or screw)
    monkeypatch.setattr(milk,'plan_high_rotation',lambda *a:calls.append('high_rrt') or fallback)
    result=milk.plan_body_rotation(task,arm,np.zeros(7),Pose([0,0,.945]),screw_first=True)
    assert result is (screw if screw_valid else fallback)
    assert calls==(['screw'] if screw_valid else ['screw','high_rrt'])


@pytest.mark.parametrize('label,near', [('038_milk-box',False), ('031_jam-jar',False), ('031_jam-jar',True)])
def test_unreachable_body_continuation_rejects_before_acquisition(monkeypatch, label, near):
    task = DumpBin({'robot_type':'franka'})
    task.big_bin_pose = Pose([.65,-.3,.01])
    center = np.array([0.,-.396,.82] if near else [.1,-.2,.82])
    actor = SimpleNamespace(get_pose=lambda: Pose(center),
        get_AABB=lambda:np.array([center-[.035,.035,.045],center+[.035,.035,.05]]))
    task._dump_active_actor = actor
    task._object_model_ids = {id(actor): 0}
    good = PlanResult(True, np.zeros((2,7)), np.zeros((2,7)))
    failed = PlanResult(False)
    planned = []
    arm = SimpleNamespace(tcp_offset=Pose(), origin_pose=Pose([0,-.65,.75]), get_arm_qpos=lambda: np.zeros(7),
        planner=SimpleNamespace(_mplib=SimpleNamespace(), plan_path=lambda *a, **kw: good,
                                plan_screw_path=lambda *a, **kw: (planned.append('lift'), failed)[1]))
    original_approach = arm.planner.plan_path
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.open_gripper = lambda tag: None
    task.set_gripper = lambda *a: pytest.fail('closed after failed continuation')
    task.step_sim = lambda: None
    task._refresh_dump_obstacles = lambda tag: None
    task._dump_plan_valid = lambda arm, path, *a: path is not None and path.success
    grasp = SimpleNamespace(to_world=lambda pose, scale: Pose([.1,-.2,.85]))
    monkeypatch.setattr(milk, 'get_grasp_pose', lambda *a, **kw: grasp)
    monkeypatch.setattr(dump, '_read_scale', lambda *a: .7)
    names = []
    def select(**kwargs):
        assert arm.planner.plan_path is not original_approach
        assert kwargs['object_scale'] == .7
        candidate = kwargs['candidate_grasps'][0]
        names.append(candidate.name)
        base = 'dump_milk_lower_body' if label == '038_milk-box' else 'dump_jar_body_oblique'
        # The original side first; the other approach sides only after it fails.
        assert candidate.name == (base if len(names) == 1 else base + f'_side{int(round(np.degrees(milk.SIDE_TURNS[len(names)-1])))}')
        if label == '038_milk-box' and len(names) == 1:
            np.testing.assert_allclose(candidate.to_world(Pose(center),.7).p,[.1,-.2,.83])
        assert kwargs['object_name'] == label
        assert kwargs['screw_qpos_step'] == (.03 if label == '031_jam-jar' else .005)
        assert kwargs['max_candidates'] == (2 if label == '031_jam-jar' and not near else 1)
        assert not kwargs['post_plan_check']((good, good, Pose(), Pose()))
        return None
    task.select_and_execute_grasp = select
    assert milk.pick_and_drop_side(task, PickSpec(lambda: center.copy(), .03, label=label, close_value=.55),
                                   PlaceSpec(np.zeros(3), transport_z=.945), 'right') is False
    assert names[0] in ('dump_milk_lower_body', 'dump_jar_body_oblique') and len(names) == len(milk.SIDE_TURNS)
    assert planned == ['lift'] * (6 if label == '031_jam-jar' and not near else 3) * len(milk.SIDE_TURNS)
    assert arm.planner.plan_path is original_approach
    assert task._dump_selector_targets == {}


@pytest.mark.parametrize('yaw', [0., np.pi/2])
@pytest.mark.parametrize('scale', [.05, .7])
def test_jar_body_runtime_grasp_preserves_world_geometry(yaw, scale):
    task=DumpBin({})
    center=np.array([.1,-.2,.82])
    bounds=np.array([[.065,-.235,.775],[.135,-.165,.87]])
    actor=SimpleNamespace(get_AABB=lambda:bounds.copy())
    obj=Pose([.08,-.19,.78],[np.cos(yaw/2),0,0,np.sin(yaw/2)])
    grasp=milk.jar_body_grasp(task,actor,obj,center)
    tcp=grasp.to_world(obj,scale)
    np.testing.assert_allclose(tcp.p,[.1,-.206,.83])
    np.testing.assert_allclose(np.abs(tcp.to_matrix()[:3,1]),[1.,0.,0.],atol=1e-12)
    np.testing.assert_allclose(grasp.pre_grasp_world(obj,scale).p-tcp.p,
                               [0.,-.05*np.sin(np.deg2rad(80)),.05*np.cos(np.deg2rad(80))],atol=1e-12)
    targets=milk.transport_targets(task,tcp,PlaceSpec(np.zeros(3),transport_z=.945),final_yaw=np.pi/2)
    # The jaws are symmetric; their undirected closing axis stays parallel.
    np.testing.assert_allclose(np.abs(targets[-1].to_matrix()[:3,1]),
                               np.abs(tcp.to_matrix()[:3,1]),atol=1e-12)
    expected=Pose(np.zeros(3),[np.cos(np.deg2rad(40)),np.sin(np.deg2rad(40)),0.,0.])
    expected=expected*task._top_down_tcp(np.zeros(3),yaw=np.pi/2)
    np.testing.assert_allclose(tcp.to_matrix()[:3,:3],expected.to_matrix()[:3,:3],atol=1e-12)
    assert np.rad2deg(2*np.arccos(abs(np.dot(tcp.q,targets[-1].q)))) == pytest.approx(80.)


def test_jar_body_rejects_invalid_bounds():
    task=DumpBin({})
    actor=SimpleNamespace(get_AABB=lambda:np.array([[0.,0.,0.],[1.,float('nan'),1.]]))
    with pytest.raises(DumpMotionRejected,match='bounds'):
        milk.jar_body_grasp(task,actor,Pose(),np.zeros(3))


@pytest.mark.parametrize('offset,expected', [([0.,.25],True),([0.,.33],False),([.3,.2],False)])
def test_jar_near_mount_uses_measured_relative_distance(offset, expected):
    base=np.array([.2,-.5,.75])
    arm=SimpleNamespace(origin_pose=Pose(base))
    center=base+np.r_[offset,.07]
    assert milk.jar_near_mount(arm,center) is expected
    shift=np.array([.3,-.2,0.])
    arm.origin_pose=Pose(base+shift)
    assert milk.jar_near_mount(arm,center+shift) is expected


def test_near_mount_jar_centers_contact_lever_and_shortens_retraction_without_changing_height():
    task=DumpBin({});center=np.array([0.,-.396,.81])
    bounds=np.array([center-[.035,.035,.045],center+[.035,.035,.05]])
    actor=SimpleNamespace(get_AABB=lambda:bounds.copy())
    obj=Pose(center)
    normal=milk.jar_body_grasp(task,actor,obj,center).to_world(obj,.7)
    grip=milk.jar_body_grasp(task,actor,obj,center,away_from_mount=True)
    tcp=grip.to_world(obj,.7)
    np.testing.assert_allclose(tcp.p,[0.,-.392,.820])
    np.testing.assert_allclose((tcp.p-center)[[0,2]],(normal.p-center)[[0,2]]*[-1.,1.])
    assert tcp.p[1]-center[1] == pytest.approx(.004)
    delta=grip.pre_grasp_world(obj,.7).p-tcp.p
    np.testing.assert_allclose(delta,[0.,.05*np.sin(np.deg2rad(80)),.05*np.cos(np.deg2rad(80))],atol=1e-12)
    targets=milk.transport_targets(task,tcp,PlaceSpec(np.zeros(3),transport_z=.945),
                                  final_yaw=np.pi/2,retract_y=-.30)
    np.testing.assert_allclose([p.p for p in targets],
                              [[0.,-.392,.870],[0.,-.30,.870],[0.,-.30,.945],[0.,-.30,.945]])


def test_jar_near_mount_rejects_nonfinite_geometry():
    arm=SimpleNamespace(origin_pose=Pose([0.,-.65,.75]))
    with pytest.raises(ValueError,match='mount center'):
        milk.jar_near_mount(arm,[np.nan,0.,.8])


@pytest.mark.parametrize('second_valid', [False, True])
def test_body_transport_retries_complete_station_from_same_start(monkeypatch, second_valid):
    task = DumpBin({})
    task._dump_plan_valid = lambda arm, path, *a: path is not None and path.success
    arm = SimpleNamespace(tcp_offset=Pose())
    start = np.arange(7)/10
    choices = [tuple(Pose([0,y,z]) for z in [.8,.9,1.]) for y in [-.15,-.30]]
    calls = []
    def screw(task, arm, current, target):
        calls.append((current.copy(), target.copy()))
        passed = target[2] == .8 or (target[1] == -.30 and second_valid)
        return PlanResult(passed, np.array([current,current+.1]), np.zeros((2,7)))
    rotations = []
    def rotate(task, arm, current, tcp, **kw):
        rotations.append((current.copy(), tcp, kw))
        return PlanResult(True, np.array([current,current+.1]), np.zeros((2,7)))
    monkeypatch.setattr(milk,'plan_body_screw',screw)
    monkeypatch.setattr(milk,'plan_body_rotation',rotate)
    callback = object()
    result = milk.select_body_transport(task,arm,start,choices,screw_first=False,
                                        rotation_options={'endpoint_check':callback})
    assert result is (choices[1] if second_valid else None)
    np.testing.assert_array_equal(start,np.arange(7)/10)
    np.testing.assert_array_equal(calls[0][0],start)
    np.testing.assert_array_equal(calls[2][0],start)
    assert len(calls) == 4 and len(rotations) == int(second_valid)
    if second_valid:
        assert rotations[0][2] == {'screw_first':False,'endpoint_check':callback}


@pytest.mark.parametrize('count', [0,3])
def test_body_transport_rejects_unbounded_station_search(count):
    task = DumpBin({})
    with pytest.raises(DumpMotionRejected,match='option count'):
        milk.select_body_transport(task,None,np.zeros(7),[()] * count,
                                   screw_first=False,rotation_options={})


def test_jar_centering_failure_stops_before_closure(monkeypatch):
    task = DumpBin({'robot_type': 'franka'})
    center = np.array([.1,-.2,.82])
    actor = SimpleNamespace(get_pose=lambda: Pose(center),
        get_AABB=lambda: np.array([center-[.035,.035,.045],center+[.035,.035,.05]]))
    task._dump_active_actor = actor
    task._object_model_ids = {id(actor): 0}
    original_plan = lambda *a, **kw: pytest.fail('centering fixture must not plan acquisition')
    planner = SimpleNamespace(_mplib=SimpleNamespace(), plan_path=original_plan)
    arm = SimpleNamespace(tcp_offset=Pose(), origin_pose=Pose([0,-.65,.75]), planner=planner)
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.open_gripper = lambda tag: None
    task.step_sim = lambda: None
    task._refresh_dump_obstacles = lambda tag: None
    task.select_and_execute_grasp = lambda **kw: object()
    task.set_gripper = lambda *a: pytest.fail('closed after rejected centering')
    arrivals = []
    def arrive(target, tag, *, center_grasp=False):
        arrivals.append((np.asarray(target), center_grasp))
        if center_grasp:
            raise DumpMotionRejected('centering cannot reach 2mm')
    task._dump_arrival = arrive
    monkeypatch.setattr(dump, '_read_scale', lambda *a: .7)
    pick = PickSpec(lambda: center.copy(), .035, label='031_jam-jar', close_value=.625)
    with pytest.raises(DumpMotionRejected, match='centering'):
        milk.pick_and_drop_side(task, pick, PlaceSpec(np.zeros(3), transport_z=.945), 'right')
    assert [centered for _, centered in arrivals] == [False, True]
    np.testing.assert_array_equal(arrivals[0][0], arrivals[1][0])
    assert planner.plan_path is original_plan


def test_jar_zero_dispatches_through_checked_body_path(monkeypatch):
    from envs.task_bases import dump_bin_grasp
    task=DumpBin({})
    actor=object();task._dump_active_actor=actor;task._object_model_ids={id(actor):0}
    pick=PickSpec(lambda:np.zeros(3),.035,label='031_jam-jar')
    place=PlaceSpec(np.zeros(3))
    calls=[]
    monkeypatch.setattr(milk,'pick_and_drop_side',lambda *args:calls.append(args) or False)
    assert dump_bin_grasp.pick_and_drop_topdown(task,pick,place,'right') is False
    assert calls==[(task,pick,place,'right')]


def test_retained_rotation_preserves_purchase_but_slip_rejects():
    task = DumpBin({})
    tcp = Pose([.1,-.2,.94], [np.sqrt(.5),0,0,np.sqrt(.5)])
    local = np.array([0.,.04,0.])
    center = tcp.p + tcp.to_matrix()[:3,:3] @ local
    task._dump_active_pick = PickSpec(lambda: center.copy(), .03)
    task._dump_hold_offset = local.copy()
    task._dump_hold_local = local.copy()
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(get_ee_pose=lambda: tcp.to_pose7()))
    task._dump_check_retention('right')
    center[2] -= .031
    with pytest.raises(DumpMotionRejected, match='drift'):
        task._dump_check_retention('right')


def rotation_fixture(monkeypatch, *, goals=None, valid=True, positions=None):
    """Native planner boundary double; use independent FK corridor samples."""
    calls = []
    backend = SimpleNamespace(
        robot=SimpleNamespace(get_base_pose=lambda: Pose()),
        move_group_joint_indices=np.arange(7),
        IK=lambda *a, **kw: ('Success', np.array(goals if goals is not None else [np.ones(7)*.1])),
    )
    def plan(**kw):
        calls.append(kw)
        return {'status': 'Success', 'position': np.array([np.zeros(7), kw['goal_qposes'][0]]),
                'velocity': np.zeros((2, 7))}
    backend.plan_qpos = plan
    planner = SimpleNamespace(_mplib=backend, _pad_qpos=lambda q: q,
                              _to_mplib_pose=Pose.from_pose7)
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose())
    task = SimpleNamespace(TABLE_TOP_Z=.765, _dump_plan_valid=lambda *a: valid)
    samples = iter(positions) if positions is not None else None
    monkeypatch.setattr(milk, 'mplib_world_link_pose',
                        lambda *a: next(samples) if samples is not None else Pose([.1, -.2, .945]))
    return task, arm, calls


@pytest.mark.parametrize('passing_step', [.002, None])
def test_body_screw_retry_checks_every_path_and_keeps_original_start(passing_step):
    calls = []
    start = np.arange(7)/10
    target = Pose([.1,-.2,.9]).to_pose7()
    def plan(q, goal, *, qpos_step, log):
        np.testing.assert_array_equal(q, start)
        np.testing.assert_array_equal(goal, target)
        calls.append(qpos_step)
        q[:] = 99  # A planner must not mutate the retry's measured start.
        return SimpleNamespace(success=True, accepted=qpos_step == passing_step)
    arm = SimpleNamespace(planner=SimpleNamespace(plan_screw_path=plan))
    task = SimpleNamespace(_dump_plan_valid=lambda arm, path, q, goal: path.accepted)
    result = milk.plan_body_screw(task, arm, start, target)
    assert calls == ([.005,.002] if passing_step else [.005,.002,.001])
    assert bool(result) == bool(passing_step)
    np.testing.assert_array_equal(start, np.arange(7)/10)


def test_high_rotation_ranks_native_goals_and_preserves_path_checks(monkeypatch):
    task, arm, calls = rotation_fixture(monkeypatch, goals=[np.ones(7), np.ones(7)*.1])
    path = milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .945]))
    assert path.success and len(calls) == 1
    np.testing.assert_allclose(calls[0]['goal_qposes'][0], .1)
    assert calls[0]['time_step'] == .004 and calls[0]['planning_time'] == 5
    task._dump_plan_valid = lambda *a: False
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .945])) is None


@pytest.mark.parametrize('bad', [3.501, float('nan'), float('inf')])
def test_high_rotation_rejects_invalid_native_goal_before_planning(monkeypatch, bad):
    task, arm, calls = rotation_fixture(monkeypatch, goals=[np.full(7, bad)])
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .945])) is None
    assert calls == []


def test_high_rotation_bounds_native_attempts(monkeypatch):
    task, arm, calls = rotation_fixture(monkeypatch, goals=[np.ones(7)*n/10 for n in range(1, 11)], valid=False)
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .945])) is None
    assert len(calls) == 6


def test_high_rotation_rejects_dead_end_and_preserves_preferred_goal_bound(monkeypatch):
    task, arm, calls = rotation_fixture(monkeypatch, goals=[np.ones(7)*n/10 for n in range(1, 11)])
    checked = []
    preferred = np.full(7, .7)
    def continuation(current, goal):
        checked.append(goal.copy())
        return np.allclose(goal, .2)
    result = milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1,-.2,.945]),
                                    preferred_goal=preferred, endpoint_check=continuation)
    assert result.success and len(calls) == 3
    np.testing.assert_allclose(checked[0], .7)
    np.testing.assert_allclose(checked[-1], .2)
    calls.clear()
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1,-.2,.945]),
                                  preferred_goal=preferred, endpoint_check=lambda *a: False) is None
    assert len(calls) == 6
    np.testing.assert_allclose(preferred, .7)


def test_preferred_rotation_still_validated_when_native_ik_returns_no_goals(monkeypatch):
    task, arm, calls = rotation_fixture(monkeypatch)
    arm.planner._mplib.IK = lambda *a, **kw: ('Failure', None)
    preferred = np.full(7, .2)
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1,-.2,.945]),
                                  preferred_goal=preferred).success
    task._dump_plan_valid = lambda *a: False
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1,-.2,.945]),
                                  preferred_goal=preferred) is None


def test_jar_rotation_prediction_preserves_asymmetric_bounds_and_live_state(monkeypatch):
    from envs.task_bases import dump_bin_grasp
    center = np.array([1., 2., 3.])
    bounds = np.array([[.9,1.8,2.7],[1.2,2.1,3.4]])
    original_bounds = bounds.copy()
    reference = Pose([1.,2.,2.8])
    end = Pose([.5,-.3,.95], [np.sqrt(.5),0,0,np.sqrt(.5)])
    arm = SimpleNamespace(planner=object(), tcp_offset=Pose())
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda tag: arm))
    monkeypatch.setattr(milk,'mplib_world_link_pose',lambda *a: end)
    accepted, observed = [], []
    def release(task,place,tag,current,tcp,predicted,box,quat):
        observed.append((predicted.copy(),box.copy()))
        np.testing.assert_allclose(predicted,[.5,-.3,1.15])
        np.testing.assert_allclose(box,[[.4,-.4,.85],[.7,-.1,1.55]])
        return ('carry','descent')
    monkeypatch.setattr(dump_bin_grasp,'plan_release_from_state',release)
    check = milk.jar_rotation_release_check(task,object(),'right',reference,center,bounds,accepted.append)
    assert check(np.zeros(7),np.ones(9))
    assert len(accepted)==1 and len(observed)==1
    np.testing.assert_array_equal(bounds,original_bounds)
    np.testing.assert_array_equal(center,[1.,2.,3.])
    monkeypatch.setattr(dump_bin_grasp,'plan_release_from_state',lambda *a: None)
    check = milk.jar_rotation_release_check(task,object(),'right',reference,center,bounds,accepted.append)
    assert not check(np.zeros(7),np.ones(9))
    assert len(accepted)==1


@pytest.mark.parametrize('bad_pose', [None, Pose([.1,-.2,.914]), Pose([.201,-.2,.945]),
                                     Pose([float('nan'),-.2,.945])])
def test_high_rotation_rejects_missing_fk_or_corridor_departure(monkeypatch, bad_pose):
    task, arm, calls = rotation_fixture(monkeypatch, positions=[Pose([.1,-.2,.945]), bad_pose])
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .945])) is None
    assert len(calls) == 1


def test_high_rotation_corridor_respects_table_when_target_is_lower(monkeypatch):
    task, arm, _ = rotation_fixture(monkeypatch, positions=[Pose([.1,-.2,.90]), Pose([.1,-.2,.90])])
    assert milk.plan_high_rotation(task, arm, np.zeros(7), Pose([.1, -.2, .92])) is None
