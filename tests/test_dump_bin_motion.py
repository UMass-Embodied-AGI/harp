"""Dump-family rejection contracts, independent of simulator construction."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.manipulation import PickSpec, PlaceSpec
from envs.planning.base import PlanResult
from envs.task_bases.dump_bin import DumpBin
from envs.task_bases.dump_bin_motion import DumpMotionRejected, pose_errors
from envs.task_bases import dump_bin_motion as motion, dump_bin_grasp as grasp
from envs.tasks.dump_bin_contribution import DumpBinContribution
from envs.utils import Pose


@pytest.fixture
def rig(monkeypatch):
    task = DumpBin({'robot_type': 'franka'})
    target = np.array([.1, .2, 1.0, 1., 0., 0., 0.])
    calls = []
    backend = SimpleNamespace(joint_limits=np.tile([-4., 4.], (7, 1)),
                              move_group_joint_indices=np.arange(7))
    good = PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7)))
    planner = SimpleNamespace(
        _mplib=backend,
        update_obstacles=lambda *a, **k: calls.append('obstacles'),
        plan_path=lambda *a, **k: (calls.append('rrt'), good)[1],
        plan_screw_path=lambda *a, **k: (calls.append('screw'), good)[1],
        solve_ik=lambda *a, **k: (calls.append('mplib_ik'), good)[1],
    )
    arm = SimpleNamespace(planner=planner, n_arm=7, tcp_offset=Pose(),
                          get_arm_qpos=lambda: np.zeros(7),
                          get_ee_pose=lambda: target.copy(), _cached_target=None,
                          entity=SimpleNamespace(inverse_kinematics=lambda **k: pytest.fail('Genesis IK called')))
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task._install_dump_table = lambda tag: None
    task.execute_plan = lambda *a, **k: calls.append('execute')
    task._reseed_cached_arm_target = lambda a: calls.append('reseed')
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose.from_pose7(target))
    return task, arm, target, good, calls


@pytest.mark.parametrize('table_top', [.765, float(np.float32(.74))+.025])
def test_table_obstacle_persists_and_exempts_only_fixed_mount(table_top):
    task = DumpBin({'robot_type': 'franka'})
    task.TABLE_TOP_Z = table_top
    clouds, allowed, refreshed, removed = [], [], [], []
    backend = SimpleNamespace(
        link_name_2_idx={'panda_link0': 0, 'panda_link5': 5, 'panda_hand': 8},
        planning_world=SimpleNamespace(
            add_object=lambda name, box: clouds.append((name, box)),
            remove_object=lambda name: removed.append(name),
            get_allowed_collision_matrix=lambda: SimpleNamespace(
            set_entry=lambda *args: allowed.append(args))))
    planner = SimpleNamespace(_mplib=backend, update_obstacles=lambda points, **kw: refreshed.append(points))
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(planner=planner))
    task._neutral_object_obstacle_points = lambda: np.array([[.2, -.3, .9]])
    task._refresh_dump_obstacles('right')
    task._refresh_dump_obstacles('right')
    assert len(clouds) == 1
    def bounds(entry):
        name, box = entry
        half = np.asarray(box.get_collision_geometry().side).ravel()/2
        center = np.asarray(box.get_pose().p).ravel()
        return np.array([center-half, center+half])
    np.testing.assert_allclose(bounds(clouds[0]),
                               [[-.5, -.8, table_top-.05], [.5, .1, table_top]], atol=1e-14, rtol=0)
    assert clouds[0][0] == 'dump_tabletop'
    assert allowed == [('panda_link0', 'dump_tabletop', True)]
    assert len(refreshed) == 2
    np.testing.assert_array_equal(refreshed[0], [[.2, -.3, .9]])
    task._set_dump_release_clearance('right')
    task._set_dump_release_clearance('right')
    assert len(clouds) == 2
    np.testing.assert_allclose(bounds(clouds[1]),
                               [[-.515, -.815, table_top-.065], [.515, .115, table_top+.015]], atol=1e-14, rtol=0)
    assert allowed[-1] == ('panda_link0', 'dump_release_table_margin', True)
    task._clear_dump_release_clearance()
    task._refresh_dump_obstacles('right')
    assert removed == ['dump_release_table_margin']
    assert len(clouds) == 2  # The real table persists for the next acquisition.


def test_seeded_dispatch_is_mplib_only(rig):
    task, arm, target, good, calls = rig
    assert task._move_seeded(target, 'right') is good
    assert calls == ['obstacles', 'rrt', 'execute', 'reseed']


def test_endpoint_hold_preserves_planned_joints_and_gripper(rig):
    task, arm, target, good, calls = rig
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    arm.gripper_val = .37
    commanded = []
    task.robot.set_arm_joints = lambda q, tag: commanded.append(q.copy())
    task.robot.set_gripper = lambda value, tag: calls.append(('gripper', value))
    task.step_sim = lambda: calls.append('step')
    task._dump_settle_endpoint(good, 'right')
    assert len(commanded) == 50
    assert calls.count('step') == 50
    assert calls.count(('gripper', .37)) == 50
    for q in commanded:
        np.testing.assert_array_equal(q, good.position[-1])


def test_endpoint_hold_stops_on_retention_loss(rig):
    task, arm, target, good, calls = rig
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    arm.gripper_val = .37
    task.robot.set_arm_joints = lambda q, tag: calls.append('arm')
    task.robot.set_gripper = lambda value, tag: calls.append('gripper')
    task.step_sim = lambda: calls.append('step')
    task._dump_check_retention = lambda tag: task._reject_dump_motion('retention')
    with pytest.raises(DumpMotionRejected, match='retention'):
        task._dump_settle_endpoint(good, 'right')
    assert calls == ['arm', 'gripper', 'step']


def test_endpoint_hold_requires_finite_fk_before_motion(rig, monkeypatch):
    task, arm, target, good, calls = rig
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *args: None)
    with pytest.raises(DumpMotionRejected, match='finite FK'):
        task._dump_settle_endpoint(good, 'right')
    assert calls == []


def test_retention_monitor_stops_plan_on_first_lost_step(rig, monkeypatch):
    from envs.base_task import BaseTask
    task, arm, target, good, calls = rig
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    task._dump_retention_arm = 'right'
    arm.gripper_val = .3
    task._apply_avatar_retreat_if_needed = lambda tags: False
    task.robot.set_arm_joints = lambda q, tag: calls.append('command')
    task.robot.set_gripper = lambda value, tag: None
    monkeypatch.setattr(BaseTask, 'step_sim', lambda self: calls.append('physics'))
    task._dump_check_retention = lambda tag: task._reject_dump_motion('retention')
    with pytest.raises(DumpMotionRejected, match='retention'):
        motion.DumpBinMotionMixin.execute_plan(task, good, 'right')
    assert calls == ['command', 'physics']


@pytest.mark.parametrize('stage', ['approach', 'lift', 'carry', 'descent'])
def test_typed_rejection_stops_shared_sequence_before_release(monkeypatch, stage):
    task = DumpBin({})
    events = []
    def sequence(self, pick, place, arm):
        for name in ('approach', 'lift', 'carry', 'descent', 'release'):
            if name == stage:
                self._reject_dump_motion(name)
            events.append(name)
        return True
    monkeypatch.setattr(grasp, 'pick_and_drop_topdown', sequence)
    pick = PickSpec(lambda: np.zeros(3), .02)
    assert task.pick_and_place(pick, PlaceSpec(np.zeros(3)), 'right') is False
    assert 'release' not in events
    assert task.plan_success is False
    assert task._dump_active_pick is None
    assert task._dump_hold_offset is None
    assert task._dump_retention_arm is None


def test_unexpected_pick_error_remains_visible(monkeypatch):
    task = DumpBin({})
    def crash(*args):
        raise OSError('simulation read failed')
    monkeypatch.setattr(grasp, 'pick_and_drop_topdown', crash)
    with pytest.raises(OSError, match='simulation read failed'):
        task.pick_and_place(PickSpec(lambda: np.zeros(3), .02), PlaceSpec(np.zeros(3)), 'right')
    assert task._dump_active_pick is None


@pytest.mark.parametrize('bad', ['failed', 'nan', 'velocity', 'empty', 'jump', 'limit', 'fk'])
def test_invalid_plan_never_executes(rig, monkeypatch, bad):
    task, arm, target, good, calls = rig
    if bad == 'failed':
        good.success = False
    elif bad == 'nan':
        good.position[0, 0] = np.nan
    elif bad == 'velocity':
        good.velocity = np.zeros((1, 7))
    elif bad == 'empty':
        good.position = good.velocity = np.zeros((0, 7))
    elif bad == 'jump':
        good.position[-1, 0] = .31
    elif bad == 'limit':
        arm.planner._mplib.joint_limits[0] = [.01, 4.]
    elif bad == 'fk':
        monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose([np.nan, 0, 0]))
    with pytest.raises(DumpMotionRejected):
        task._move_seeded(target, 'right')
    assert 'execute' not in calls


@pytest.mark.parametrize('bad', [np.nan, .02])
def test_arrival_failure_is_not_success(rig, bad):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[0] += bad
    arm.get_ee_pose = lambda: measured
    with pytest.raises(DumpMotionRejected):
        task._move_seeded(target, 'right')
    assert calls.count('execute') == 1


def test_loaded_arrival_correction_reaches_original_gate(rig, monkeypatch):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[0] -= .012
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_hold_offset = np.zeros(3)
    commands = []
    def plan(current, commanded, **kwargs):
        assert kwargs['qpos_step'] == .002
        commands.append(commanded.copy())
        return good
    arm.planner.plan_screw_path = plan
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *args: Pose.from_pose7(commands[-1]))
    task.execute_plan = lambda *args: measured.__setitem__(0, measured[0]+.008)
    task._dump_arrival(target, 'right')
    assert len(commands) == 1
    np.testing.assert_allclose(commands[0][:3]-target[:3], [.008, 0., 0.])
    assert np.linalg.norm(measured[:3]-target[:3]) < .01


def test_loaded_arrival_correction_rejects_blocked_path(rig):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[0] -= .012
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_hold_offset = np.zeros(3)
    good.success = False
    with pytest.raises(DumpMotionRejected, match='checked straight path'):
        task._dump_arrival(target, 'right')
    assert 'execute' not in calls


def test_loaded_arrival_correction_has_finite_budget(rig, monkeypatch):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[0] -= .012
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_hold_offset = np.zeros(3)
    commands = []
    def plan(current, commanded, **kwargs):
        commands.append(commanded.copy())
        return good
    arm.planner.plan_screw_path = plan
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *args: Pose.from_pose7(commands[-1]))
    with pytest.raises(DumpMotionRejected, match='20mm budget'):
        task._dump_arrival(target, 'right')
    assert calls.count('execute') == 2
    assert all(np.linalg.norm(p[:3]-target[:3]) <= .020 for p in commands)


def test_unbounded_preclose_correction_never_plans_or_moves(rig):
    task, arm, target, good, calls = rig
    task._dump_active_pick = PickSpec(lambda: np.array([.284, .2, 1.]), .03)
    task._dump_hold_offset = None
    with pytest.raises(DumpMotionRejected, match='pre-close lateral correction'):
        task._move_screw(np.array([.284, .2, 1.]), 'right')
    assert calls == ['obstacles']


def test_failed_screw_has_no_rrt_fallback(rig):
    task, arm, target, good, calls = rig
    good.success = False
    with pytest.raises(DumpMotionRejected):
        task._move_screw(target[:3], 'right')
    # Three screw precisions, then only the checked straight joint-space
    # fallback (MPLib IK goal); never an RRT path or Genesis IK.
    assert calls == ['obstacles', 'screw', 'screw', 'screw', 'mplib_ik']
    assert 'rrt' not in calls


def test_tiny_lift_uses_mplib_ik(rig):
    task, arm, target, good, calls = rig
    np.testing.assert_array_equal(task._solve_ik(Pose.from_pose7(target), 'right'), np.zeros(7))
    assert calls == ['obstacles', 'mplib_ik']


def test_retention_loss_rejects_before_release(rig):
    task, arm, target, good, calls = rig
    task._dump_active_pick = PickSpec(lambda: np.array([.1, .2, .05]), .03)
    task._dump_hold_offset = np.array([0., 0., -.02])
    with pytest.raises(DumpMotionRejected, match='held object/TCP drift'):
        task._dump_check_retention('right')


@pytest.mark.parametrize('result', [True, False])
def test_assist_propagates_parallel_result(result):
    task = DumpBinContribution({})
    task.avatar = object()
    task._boost_finger_pd = lambda arm: None
    task._prepare_avatar_assist_geometry = lambda: np.zeros(3)
    task._run_parallel = lambda *a: result
    assert task.play_once() is result


def test_assist_parallel_failure_restores_step_and_never_releases(monkeypatch):
    task = DumpBinContribution({})
    task.avatar = SimpleNamespace(frame_ratio=1., spare=lambda: True)
    task.objects = [object(), object()]
    task.ROBOT_START_AFTER_AVATAR = False
    task._assist_controller = lambda: SimpleNamespace(scripted_motion_frames=lambda t: (1, 2))
    calls = []
    task._pick_and_drop_cube = lambda *a: (calls.append('pick'), False)[1]
    task.open_gripper = lambda *a: pytest.fail('release after failed motion')
    assert task._run_parallel(np.zeros(3), 'right') is False
    assert calls == ['pick']
    assert 'step_sim' not in task.__dict__
    assert task.avatar.frame_ratio == 1.


def test_assist_parallel_unexpected_error_propagates_and_cleans_up():
    task = DumpBinContribution({})
    task.avatar = SimpleNamespace(frame_ratio=1., spare=lambda: True)
    task.objects = [object()]
    task.ROBOT_START_AFTER_AVATAR = False
    task._assist_controller = lambda: SimpleNamespace(scripted_motion_frames=lambda t: (1, 2))
    def fail(*a):
        raise OSError('unavailable simulator')
    task._pick_and_drop_cube = fail
    with pytest.raises(OSError):
        task._run_parallel(np.zeros(3), 'right')
    assert 'step_sim' not in task.__dict__
    assert task.avatar.frame_ratio == 1.


def test_pose_nan_and_rotation_rejected():
    target = np.array([0., 0., 0., 1., 0., 0., 0.])
    assert pose_errors(np.full(7, np.nan), target) == (float('inf'), float('inf'))
    actual = np.array([0., 0., 0., 0., 1., 0., 0.])
    assert pose_errors(actual, target)[1] == pytest.approx(np.pi)


def test_centered_grasp_corrects_small_unloaded_bias(rig, monkeypatch):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[1] -= .0045
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_hold_offset = None
    task._dump_active_actor = object()
    task._dump_selector_center = np.array([.1, .2, .8])
    task._dump_object_center = lambda actor: task._dump_selector_center.copy()
    commands = []
    def plan(current, commanded, **kwargs):
        assert kwargs['qpos_step'] == .002
        commands.append(commanded.copy())
        return good
    arm.planner.plan_screw_path = plan
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose.from_pose7(commands[-1]))
    task.execute_plan = lambda *a: measured.__setitem__(1, measured[1]+.004)
    task._dump_arrival(target, 'right', center_grasp=True)
    assert len(commands) == 1
    assert np.linalg.norm(measured[:3]-target[:3]) < .002
    np.testing.assert_allclose(commands[0][:3]-target[:3], [0, .004, 0])


def test_centered_arrival_preserves_compensated_arm_target_during_closure(rig, monkeypatch):
    from envs.robot.franka_robot import FrankaArm

    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[1] -= .004
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_active_actor = object()
    task._dump_selector_center = np.array([.1, .2, .8])
    task._dump_object_center = lambda actor: task._dump_selector_center.copy()
    pd_commands = []
    control = FrankaArm(SimpleNamespace(
        control_dofs_position=lambda q: pd_commands.append(q.copy())), Pose())
    control._cached_target = np.r_[np.zeros(7), [.04, .04]]
    control.finger_joints = [SimpleNamespace(dof_idx_local=7), SimpleNamespace(dof_idx_local=8)]
    # Distinct commanded/measured states model the unloaded PD residual.
    compensated = np.full(7, .02)
    def execute(*args):
        control._cached_target[:7] = compensated
        measured[:] = target
    def reseed(*args):
        control._cached_target[:7] = 0.
    task.execute_plan = execute
    task._reseed_cached_arm_target = reseed
    monkeypatch.setattr(motion, 'mplib_world_link_pose',
                        lambda *a: Pose.from_pose7(target+[0,.004,0,0,0,0,0]))
    task._dump_arrival(target, 'right', center_grasp=True)
    control.set_gripper(.5)
    np.testing.assert_array_equal(pd_commands[-1][:7], compensated)
    np.testing.assert_allclose(pd_commands[-1][7:], [.02, .02])


@pytest.mark.parametrize('residual', [.013, .02, np.nan])
def test_centered_grasp_does_not_rescue_large_or_invalid_approach(rig, residual):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[0] += residual
    arm.get_ee_pose = lambda: measured.copy()
    with pytest.raises(DumpMotionRejected):
        task._dump_arrival(target, 'right', center_grasp=True)
    assert 'execute' not in calls


@pytest.mark.parametrize('shift', [.011, np.nan])
def test_centered_grasp_rejects_object_motion_during_correction(rig, monkeypatch, shift):
    task, arm, target, good, calls = rig
    measured = target.copy()
    measured[1] -= .004
    arm.get_ee_pose = lambda: measured.copy()
    task._dump_active_actor = object()
    task._dump_selector_center = np.array([.1, .2, .8])
    task._dump_object_center = lambda actor: task._dump_selector_center+[shift, 0, 0]
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose.from_pose7(target+[0,.004,0,0,0,0,0]))
    with pytest.raises(DumpMotionRejected, match='object moved'):
        task._dump_arrival(target, 'right', center_grasp=True)
    assert calls.count('execute') == 1


def test_release_chord_removes_terminal_parameterization_jump_without_moving_endpoints():
    vertices = np.zeros((5, 7))
    vertices[:, 0] = [0., .005, .010, .010, .0101]
    original = vertices.copy()
    result = motion.release_chord_vertices(vertices)
    np.testing.assert_allclose(result[:, 0], np.linspace(0., .0101, 5))
    np.testing.assert_array_equal(result[[0, -1]], vertices[[0, -1]])
    np.testing.assert_array_equal(vertices, original)


@pytest.mark.parametrize('vertices', [np.zeros((1, 7)), np.zeros((3, 6)),
    np.full((3, 7), np.nan), np.array([[0.]*7, [.1]*7]), np.zeros((4097, 7))])
def test_release_chord_rejects_invalid_or_sparse_inputs(vertices):
    with pytest.raises(ValueError):
        motion.release_chord_vertices(vertices)


@pytest.mark.parametrize('collision_kind', [None, 'self', 'environment'])
def test_release_screw_preserves_native_timeline_and_checks_interior_waypoints(collision_kind):
    seen = []
    vertices = np.zeros((4, 7))
    vertices[:, 0] = [0., .005, .010, .0101]
    native_position = np.zeros((3, 7))
    native_position[:, 0] = [0., .007, .0101]
    native_velocity = np.ones_like(native_position)*.2
    def original(path, interval):
        np.testing.assert_allclose(path[:, 0], np.linspace(0., .0101, 4))
        assert interval == .004
        return np.array([0., .004, .008]), native_position, native_velocity, None, .008
    backend = SimpleNamespace(TOPP=original)
    def collision(kind, q):
        seen.append((kind, float(q[0])))
        return [object()] if kind == collision_kind and q[0] == .007 else []
    backend.check_for_self_collision = lambda q: collision('self', q)
    backend.check_for_env_collision = lambda q: collision('environment', q)
    def screw(current, target, **kwargs):
        assert kwargs == dict(qpos_step=.005, log=False)
        _, pos, vel, _, _ = backend.TOPP(vertices, .004)
        return PlanResult(True, pos, vel)
    planner = SimpleNamespace(_mplib=backend, plan_screw_path=screw,
                              _pad_qpos=lambda q: np.r_[q, 0., 0.])
    result = motion.plan_release_screw(SimpleNamespace(planner=planner), np.zeros(7), np.zeros(7))
    assert backend.TOPP is original
    assert result.success == (collision_kind is None)
    assert ('self', .007) in seen
    if collision_kind is None:
        np.testing.assert_array_equal(result.position, native_position)
        np.testing.assert_array_equal(result.velocity, native_velocity)
        assert ('environment', .0101) in seen


def test_release_screw_restores_class_method_and_exposes_unexpected_failure():
    class Backend:
        def TOPP(self, *args):
            raise AssertionError('unused')
    backend = Backend()
    def screw(*args, **kwargs):
        assert 'TOPP' in vars(backend)
        raise RuntimeError('unexpected native failure')
    planner = SimpleNamespace(_mplib=backend, plan_screw_path=screw)
    with pytest.raises(RuntimeError, match='unexpected native failure'):
        motion.plan_release_screw(SimpleNamespace(planner=planner), np.zeros(7), np.zeros(7))
    assert 'TOPP' not in vars(backend)
