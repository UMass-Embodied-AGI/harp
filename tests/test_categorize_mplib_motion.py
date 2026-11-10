"""Failure, timing and collision contracts for the categorize adapter."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.categorize_support import motion
from envs.task_bases.categorize import CategorizeTaskCore
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence
from envs.utils import Pose


def plan(pos=None):
    pos = np.zeros((3, 7)) if pos is None else np.asarray(pos, dtype=float)
    return PlanResult(True, pos, np.zeros_like(pos))


def fixture_task(monkeypatch):
    calls = []
    backend = NS(joint_limits=np.tile([-3., 3.], (7, 1)), move_group_joint_indices=np.arange(7),
                 check_for_self_collision=lambda q: False, check_for_env_collision=lambda q: False)
    planner = NS(_mplib=backend, _pad_qpos=lambda q: q,
                 plan_path=lambda *a, **k: calls.append('path') or plan(),
                 plan_screw_path=lambda *a, **k: calls.append('screw') or plan(),
                 solve_ik=lambda *a, **k: calls.append('ik') or plan())
    arm = NS(planner=planner, n_arm=7, get_arm_qpos=lambda: np.zeros(7), tcp_offset=Pose([0, 0, 0]),
             get_ee_pose=lambda: np.array([0., 0., 0., 1., 0., 0., 0.]))
    task = CategorizeTaskCore({})
    task.robot = NS(get_arm=lambda _: arm)
    task._categorize_refresh_obstacles = lambda tag: calls.append('obstacles')
    task._execute_categorize_plan = lambda *a, **k: calls.append('execute')
    monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose([0, 0, 0]))
    return task, arm, calls


@pytest.mark.parametrize('method', ['_move_seeded', 'move_and_execute', '_joint_interp_to'])
def test_link_motion_is_mplib_only(monkeypatch, method):
    task, arm, calls = fixture_task(monkeypatch)
    getattr(task, method)([0, 0, 0, 1, 0, 0, 0], 'right')
    assert calls[0] == 'obstacles'
    assert calls[-1] == 'execute'
    assert ('screw' if method == '_joint_interp_to' else 'path') in calls


def test_screw_rejection_never_falls_back_or_executes(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    arm.planner.plan_screw_path = lambda *a, **k: calls.append('screw') or PlanResult(False)
    with pytest.raises(motion.CategorizeMotionRejected):
        task._move_tcp_screw(Pose([0, 0, 0]), 'right')
    assert calls == ['obstacles', 'screw', 'screw', 'screw']
    assert task.plan_success is False


def test_ik_uses_mplib_and_checks_branch(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    np.testing.assert_array_equal(task._solve_ik(Pose([0, 0, 0]), 'right'), np.zeros(7))
    assert calls == ['obstacles', 'ik']
    arm.planner.solve_ik = lambda *a, **k: plan(np.ones((2, 7)))
    with pytest.raises(motion.CategorizeMotionRejected):
        task._solve_ik(Pose([0, 0, 0]), 'right')


@pytest.mark.parametrize('defect', ['nan', 'velocity', 'limits', 'jump', 'collision', 'fk', 'angle'])
def test_plan_defects_are_rejected_before_execution(monkeypatch, defect):
    task, arm, calls = fixture_task(monkeypatch)
    result = plan()
    if defect == 'nan': result.position[-1, 0] = np.nan
    if defect == 'velocity': result.velocity[-1, 0] = np.inf
    if defect == 'limits': arm.planner._mplib.joint_limits[:, 0] = .1
    if defect == 'jump': result.position[-1, 0] = .31
    if defect == 'collision': arm.planner._mplib.check_for_env_collision = lambda q: True
    if defect == 'fk': monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose([.02, 0, 0]))
    if defect == 'angle': monkeypatch.setattr(motion, 'mplib_world_link_pose', lambda *a: Pose([0, 0, 0], [0, 1, 0, 0]))
    arm.planner.plan_path = lambda *a, **k: result
    with pytest.raises(motion.CategorizeMotionRejected):
        task._move_seeded([0, 0, 0, 1, 0, 0, 0], 'right')
    assert 'execute' not in calls


def test_measured_arrival_rejects_tracking_error(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    arm.get_ee_pose = lambda: np.array([.02, 0, 0, 1, 0, 0, 0])
    with pytest.raises(motion.CategorizeMotionRejected, match='arrival'):
        task._move_seeded([0, 0, 0, 1, 0, 0, 0], 'right')
    assert 'execute' in calls


@pytest.mark.parametrize('dt,slowdown', [(.002, 1), (.004, 1), (.002, 3)])
def test_physical_clock_preserves_duration_and_gripper_alignment(dt, slowdown):
    raw = plan(np.tile(np.arange(3)[:, None], (1, 7)))
    raw.velocity[:] = 2.
    result, grip = motion.timed_plan(raw, dt, slowdown, [1, .5, 0])
    assert (len(result.position)-1)*dt == pytest.approx(.008*slowdown)
    np.testing.assert_array_equal(result.position[[0, -1]], raw.position[[0, -1]])
    np.testing.assert_allclose(result.velocity, 2./slowdown)
    assert len(grip) == len(result.position)
    np.testing.assert_allclose(grip[[0, -1]], [1, 0])


@pytest.mark.parametrize('dt', [0, np.nan, .01])
def test_bad_clock_is_visible(dt):
    with pytest.raises(ValueError): motion.timed_plan(plan(), dt)


@pytest.mark.parametrize('task_cls', [CategorizeObjectsInterruption, CategorizeObjectsContribution, CategorizeObjectsCopresence])
def test_episode_stops_on_expected_rejection_without_later_work(task_cls):
    task = task_cls({})
    events = []
    def fail():
        events.append('failed')
        raise motion.CategorizeMotionRejected('approach rejected')
    task._boost_arm_pd = fail
    task._open_basket_lids = lambda: events.append('later')
    assert task.play_once() is False
    assert events == ['failed']
    assert task.plan_success is False


@pytest.mark.parametrize('task_cls', [CategorizeObjectsInterruption, CategorizeObjectsContribution, CategorizeObjectsCopresence])
def test_unexpected_exception_remains_visible(task_cls):
    task = task_cls({})
    task._boost_arm_pd = lambda: (_ for _ in ()).throw(ValueError('unexpected'))
    with pytest.raises(ValueError, match='unexpected'): task.play_once()


def test_incomplete_pick_cannot_be_followed_by_release(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    pick = NS(actor=object(), get_center=lambda: np.zeros(3), label='object')
    task._pick_and_place_checked = lambda *a: False
    with pytest.raises(motion.CategorizeMotionRejected, match='incomplete'):
        task.pick_and_place(pick, None, 'right')
    assert task._categorize_active_pick is None
    assert task._categorize_hold_local is None


def test_radius_yaw_context_is_preserved_without_shared_motion_bypass():
    task = CategorizeTaskCore({})
    task._categorize_grasp_yaw = np.pi / 2
    contextual = task._top_down_tcp([0, 0, 0])
    explicit = task._top_down_tcp([0, 0, 0], yaw=np.pi/2)
    np.testing.assert_allclose(contextual.q, explicit.q)


def test_parked_experiments_stay_off():
    assert CategorizeTaskCore.SEAL_MEASURED_GRASP is False
    assert CategorizeTaskCore.SEAL_HOLD_FORCE_N == 7.5
    assert CategorizeTaskCore.MILK_CONTINUATION_PREFLIGHT is False


class _PhysicalBase:
    def step_sim(self): return 'stepped'
    def open_gripper(self, tag, num_steps=200): self.opened = True
    def set_gripper(self, value, tag, num_steps=200): self.closed = value


class _RetentionTask(motion.CategorizeMotionMixin, _PhysicalBase):
    pass


def test_retention_guard_checks_each_physical_step_and_clears_on_release():
    task = _RetentionTask()
    center = np.array([0., 0., .02])
    tcp = np.array([0., 0., 0., 1., 0., 0., 0.])
    task.robot = NS(get_arm=lambda _: NS(get_ee_pose=lambda: tcp))
    task._categorize_active_pick = NS(get_center=lambda: center)
    task._categorize_hold_arm = 'right'
    task._categorize_begin_hold('right')
    assert task.step_sim() == 'stepped'
    center[0] = .031
    with pytest.raises(motion.CategorizeMotionRejected, match='drift'):
        task.step_sim()
    assert task.plan_success is False
    task.open_gripper('right')
    assert task._categorize_hold_local is None
    assert task.step_sim() == 'stepped'


def test_retention_anchor_rotates_with_tcp_without_reanchoring():
    task = _RetentionTask()
    center = np.array([.02, 0., 0.])
    tcp = np.array([0., 0., 0., 1., 0., 0., 0.])
    task.robot = NS(get_arm=lambda _: NS(get_ee_pose=lambda: tcp))
    task._categorize_active_pick = NS(get_center=lambda: center)
    task._categorize_hold_arm = 'right'
    task._categorize_begin_hold('right')
    original = task._categorize_hold_local.copy()
    tcp[3:] = [0., 0., 0., 1.]
    center[:] = [-.02, 0, 0]
    assert task.step_sim() == 'stepped'
    np.testing.assert_array_equal(task._categorize_hold_local, original)


def test_preclosure_knock_cannot_be_hidden_by_a_new_center():
    task = _RetentionTask()
    task._categorize_pick_center = np.zeros(3)
    task._categorize_active_pick = NS(get_center=lambda: np.array([.011, 0, 0]))
    with pytest.raises(motion.CategorizeMotionRejected, match='before closure'):
        task.set_gripper(.5, 'right')
    assert not hasattr(task, 'closed')


def test_avatar_and_neutral_obstacles_survive_family_cloud_refresh(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    del task._categorize_refresh_obstacles
    task._OBSTACLE_RESOLUTION = .015
    task._categorize_scene_points = lambda **kwargs: np.array([[1., 0., 0.]])
    task.avatar_collider = NS(current_capsules=lambda: ['capsule'], current_boxes=lambda: [])
    monkeypatch.setattr(motion, 'sample_capsule_planner_points', lambda *a, **k: np.array([[2., 0., 0.]]))
    task._neutral_object_obstacle_points = lambda: np.array([[3., 0., 0.]])
    captured = []
    arm.planner.update_obstacles = lambda points, **kwargs: captured.append(points.copy())
    task._categorize_install_table = lambda backend: None
    task._categorize_refresh_obstacles('right')
    np.testing.assert_array_equal(captured[0][:, 0], [1, 2, 3])


def test_named_grasps_use_standard_selector_preserving_allowlist_and_scale(monkeypatch):
    task = CategorizeTaskCore({})
    task.TABLE_TOP_Z = .765
    calls = []
    monkeypatch.setattr(task, 'select_and_execute_grasp', lambda *a, **k: calls.append(k) or None)
    monkeypatch.setattr(task, 'try_grasp_by_name', lambda *a, **k: pytest.fail('debug grasp helper'))
    assert task._select_and_execute_task_grasp('038_milk-box', Pose([0, 0, .8]), 'right',
                                              model_id=0, object_scale=.04) is None
    assert [c['grasp_names'][0] for c in calls] == list(task.EXCLUSIVE_GRASP_NAMES['038_milk-box'])
    assert all(c['object_scale'] == .04 for c in calls)


def test_checked_executor_retimes_before_existing_pd_hook(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    del task._execute_categorize_plan
    task.scene = NS(sim=NS(dt=.002))
    arm.gripper_val = .6
    task.robot.set_arm_joints = lambda *a: None
    task.robot.set_gripper = lambda *a: None
    task.step_sim = lambda: calls.append('step')
    raw = plan()
    task.execute_plan = lambda result, tag, **kwargs: calls.append(('delegate', len(result.position), tag))
    task._execute_categorize_plan(raw, 'right')
    assert ('delegate', 5, 'right') in calls
    assert calls.count('step') == 50


def test_selector_routes_original_plans_through_checked_executor(monkeypatch):
    task, arm, calls = fixture_task(monkeypatch)
    raw = plan()
    task._categorize_grasp_targets = {id(raw): np.array([0,0,0,1,0,0,0])}
    task.execute_plan(raw, 'right')
    assert calls == ['execute']


@pytest.mark.parametrize('center', [[-.2363,.1843,.8231],[-.2673,.1899,.7996],[-.2375,.1925,.7871]])
def test_intent_hover_clears_prehover_without_unreachable_fixed_height(center):
    target = CategorizeObjectsInterruption._intent_above_position(center)
    np.testing.assert_array_equal(target[:2], center[:2])
    assert target[2] == pytest.approx(center[2]+.22)
    assert target[2] - (center[2]+.12) == pytest.approx(.10)
    assert target[2] < 1.215


@pytest.mark.parametrize('center', [[0, 0], [0, 0, np.nan]])
def test_intent_rejects_invalid_geometry(center):
    with pytest.raises(ValueError): CategorizeObjectsInterruption._intent_above_position(center)
