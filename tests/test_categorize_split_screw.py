"""Focused tests for the opt-in categorize midpoint split.

Fake native model: 9 user joints; joint 0 encodes TCP z directly (FK returns
Pose([0, 0, q0]) with a zero TCP offset), so screw targets are pure vertical
moves. The fake plan_screw interpolates joint 0 and can be scripted per
target to return native failure, raise, or return malformed data.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.categorize_support import split as m
from envs.task_bases.categorize_support.motion import CategorizeMotionRejected
from envs.utils import Pose

NAMES = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']


def fake_arm(monkeypatch, *, fail_targets=(), raise_targets=(), malformed_targets=(), fingers=(.0137, .0141), start_z=.80):
    """Return (arm, calls). ``*_targets`` are TCP z values that trigger outcomes."""
    calls = []
    qpos_store = {'q': np.r_[start_z, np.zeros(6), fingers]}
    restored = []

    class Robot:
        def get_user_joint_names(self): return list(NAMES)
        def get_qpos(self): return qpos_store['q'].copy()
        def set_qpos(self, q, full=True): restored.append(np.array(q)); qpos_store['q'] = np.array(q)

    class World:
        def set_qpos_all(self, q): restored.append(('world', np.array(q)))

    def plan_screw(goal_pose, current_qpos, *, qpos_step, time_step, wrt_world, verbose):
        z_goal = round(float(goal_pose.p[2]), 9)  # rounded only for bookkeeping/asserts
        calls.append(dict(z_goal=z_goal, start=np.array(current_qpos), step=qpos_step))
        if any(abs(z_goal - z) < 1e-9 for z in raise_targets):
            raise RuntimeError('native boom')
        if any(abs(z_goal - z) < 1e-9 for z in malformed_targets):
            return {'status': 'Success', 'position': [[float('nan')] * 7]}
        # Native failure is scripted for the ORIGINAL start only, so a leg from the
        # midpoint to the same final z can succeed (the split hypothesis).
        if any(abs(z_goal - z) < 1e-9 for z in fail_targets) and abs(float(current_qpos[0]) - start_z) < 1e-9:
            return {'status': 'screw plan failed'}
        z0 = float(current_qpos[0])
        n = max(2, int(abs(z_goal - z0) / .01) + 1)
        pos = np.zeros((n, 7)); pos[:, 0] = np.linspace(z0, z_goal, n); pos[0] = current_qpos[:7]
        return {'status': 'Success', 'position': pos, 'velocity': np.zeros_like(pos), 'duration': (n - 1) * time_step}

    backend = NS(robot=Robot(), planning_world=World(), plan_screw=plan_screw,
                 joint_limits=np.array([[-3., 3.]] * 7 + [[0., .04]] * 2), move_group_joint_indices=list(range(7)),
                 check_for_self_collision=lambda q: [], check_for_env_collision=lambda q: [])
    planner = NS(_mplib=backend, _pad_qpos=lambda q: np.r_[q, np.zeros(9 - len(q))] if len(q) < 9 else np.asarray(q, dtype=float),
                 _to_mplib_pose=lambda pose7: NS(p=np.asarray(pose7)[:3], q=np.asarray(pose7)[3:]))
    joints = [NS(name=f'joint{i}', dof=i - 1) for i in range(1, 8)]
    fjoints = [NS(name='finger_joint1', dof=7), NS(name='finger_joint2', dof=8)]
    entity = NS(get_qpos=lambda: qpos_store['q'].copy())
    arm = NS(planner=planner, entity=entity, arm_joints=joints, finger_joints=fjoints, n_arm=7, tcp_offset=Pose(),
             get_ee_pose=lambda: np.r_[0., 0., qpos_store['q'][0], 1., 0., 0., 0.])
    monkeypatch.setattr(m, '_get_dof_idx', lambda joint: joint.dof)
    monkeypatch.setattr(m, 'to_numpy', lambda v: np.asarray(v, dtype=float))
    monkeypatch.setattr(m, 'mplib_world_link_pose', lambda planner_, q: Pose([0., 0., float(np.asarray(q)[0])]))
    return arm, calls, qpos_store, restored


def fake_task(arm, opt_in=True, column=True, label='038_milk-box', model_id=0, held=True):
    actor = NS(entity=NS(get_AABB=lambda: np.array([[-.03, -.03, arm.get_ee_pose()[2] - .04], [.03, .03, arm.get_ee_pose()[2] + .06]])))
    events = []
    def reject(reason): raise CategorizeMotionRejected(reason)
    task = NS(config={m.OPT_IN: True} if opt_in else {}, _CATEGORIZE_COLUMN_LAYOUT=column,
              _sort_targets=[(actor, label, model_id, 0)], _categorize_active_pick=NS(actor=actor),
              _categorize_hold_local=np.zeros(3) if held else None, robot=NS(get_arm=lambda _: arm),
              TABLE_TOP_Z=.765, _OBSTACLE_MARGIN_Z=.01, scene=NS(sim=NS(dt=.001)), _categorize_last_cloud_identity={'count': 3, 'sha256': 'abc'},
              _reject_categorize_motion=reject,
              _categorize_plan_valid=lambda arm_, plan, current, target: bool(plan is not None and plan.success),
              _categorize_inputs=lambda target, tag: (arm, arm.entity.get_qpos()[:7], np.asarray(target, dtype=float)),
              _categorize_arrival=lambda target, tag: events.append(('arrival', round(float(np.asarray(target)[2]), 9))))
    def execute(plan, tag):
        events.append(('execute', round(float(np.asarray(plan.position)[-1][0]), 9)))
        q = arm.entity.get_qpos(); q[0] = float(np.asarray(plan.position)[-1][0]); arm.planner._mplib.robot.set_qpos(q, True)
    task._execute_categorize_plan = execute
    return task, actor, events


# ---------------------------------------------------------------- scope ----

def test_opt_in_is_literal_and_column_scoped():
    assert m.enabled(NS(config={m.OPT_IN: True}, _CATEGORIZE_COLUMN_LAYOUT=True))
    for bad in ('True', 1, 1.0, 'true', [True]):
        assert not m.enabled(NS(config={m.OPT_IN: bad}, _CATEGORIZE_COLUMN_LAYOUT=True))
    assert not m.enabled(NS(config={m.OPT_IN: True}, _CATEGORIZE_COLUMN_LAYOUT=False))
    assert not m.enabled(NS(config={}, _CATEGORIZE_COLUMN_LAYOUT=True))


@pytest.mark.parametrize('label,mid,phase', [('038_milk-box', 0, 'small_lift'), ('086_woodenblock', 1, 'descent'),
                                             ('038_milk-box', 1, None), ('086_woodenblock', 0, None), ('035_apple', 1, None), ('cube', 0, None)])
def test_scope_is_exactly_two_model_pairs(label, mid, phase):
    actor = object()
    assert m.scoped_phase(NS(_sort_targets=[(actor, label, mid, 0)]), actor) == phase
    assert m.SCOPE == {('038_milk-box', 0): 'small_lift', ('086_woodenblock', 1): 'descent'}


# ------------------------------------------------------------- full9 ----

def test_read_full9_preserves_asymmetric_measured_fingers(monkeypatch):
    arm, _, _, _ = fake_arm(monkeypatch, fingers=(.0137, .0141))
    full, move_idx, finger_idx = m.read_full9(arm)
    assert full.shape == (9,) and list(move_idx) == list(range(7)) and list(finger_idx) == [7, 8]
    assert full[7] == .0137 and full[8] == .0141


@pytest.mark.parametrize('fault', ['unmapped', 'nonfinite', 'limit'])
def test_read_full9_rejects_bad_states(monkeypatch, fault):
    arm, _, store, _ = fake_arm(monkeypatch)
    if fault == 'unmapped': arm.finger_joints[1] = NS(name='other', dof=8)
    if fault == 'nonfinite': store['q'][3] = np.nan
    if fault == 'limit': store['q'][2] = 3.5
    with pytest.raises(ValueError):
        m.read_full9(arm)


# -------------------------------------------------------------- seam ----

def test_raw_screw_classifies_outcomes_and_restores_state(monkeypatch):
    arm, calls, store, restored = fake_arm(monkeypatch, fail_targets=(.90,), raise_targets=(.91,), malformed_targets=(.92,))
    full = store['q'].copy()
    target = lambda z: np.r_[0., 0., z, 1., 0., 0., 0.]
    ok = m.raw_screw(arm, full, target(.85), .005)
    assert ok['kind'] == 'success' and ok['plan'].success and ok['samples'] >= 2 and abs(ok['native_duration_s'] - (ok['samples'] - 1) * .004) < 1e-12
    assert m.raw_screw(arm, full, target(.90), .005)['kind'] == 'native_failure'
    assert m.raw_screw(arm, full, target(.91), .005)['kind'] == 'exception'
    assert m.raw_screw(arm, full, target(.92), .005)['kind'] == 'malformed'
    assert len(restored) >= 8  # robot + world restored after every call
    # returned start must equal the native start
    monkeypatch.setattr(arm.planner._mplib, 'plan_screw', lambda **k: {'status': 'Success', 'position': np.ones((3, 7)), 'velocity': np.zeros((3, 7))})
    assert m.raw_screw(arm, full, target(.85), .005)['kind'] == 'malformed'


# ----------------------------------------------------------- geometry ----

def test_midpoint_is_exact_and_final_is_unchanged():
    start = np.r_[.1, .2, .80, 0., 0., 1., 0.]; final = np.r_[.1, .2, .86, 0., 0., 1., 0.]
    mid, fin = m.midpoint_geometry(start, final)
    assert np.array_equal(mid.p, (start[:3] + final[:3]) / 2.) and np.array_equal(mid.q, final[3:])
    assert abs(mid.p[2] - .83) < 1e-12
    assert np.array_equal(fin.p, final[:3]) and np.array_equal(fin.q, final[3:])


def test_non_vertical_or_rotated_routes_are_out_of_scope():
    start = np.r_[.1, .2, .80, 0., 0., 1., 0.]
    assert m.midpoint_geometry(start, np.r_[.1, .21, .86, 0., 0., 1., 0.]) is None
    assert m.midpoint_geometry(start, np.r_[.1, .2, .86, np.cos(.1), 0., np.sin(.1), 0.]) is None


# ----------------------------------------------------------- proposal ----

def test_original_target_success_through_seam_needs_no_split(monkeypatch):
    arm, calls, _, _ = fake_arm(monkeypatch)
    task, actor, events = fake_task(arm)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005, .002, .001), 'small_lift')
    assert prop['plan'] is not None and prop['leg1'] is None
    assert [c['z_goal'] for c in calls] == [.86]
    assert task._categorize_split_log[-1]['decision'] == 'original_target_accepted_through_seam'


def test_split_only_after_all_original_precisions_fail_natively(monkeypatch):
    arm, calls, _, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005, .002, .001), 'small_lift')
    assert prop is not None and prop['plan'] is None
    assert [c['z_goal'] for c in calls] == [.86, .86, .86, .83, .86]
    assert np.allclose(prop['mid_target'][:3], [0., 0., .83]) and np.allclose(prop['final_target'][:3], [0., 0., .86])
    # leg2 planned from the hypothetical leg-1 endpoint with the same measured fingers
    assert abs(calls[-1]['start'][0] - .83) < 1e-12 and calls[-1]['start'][7] == .0137 and calls[-1]['start'][8] == .0141
    rec = task._categorize_split_log[-1]
    assert rec['decision'] == 'split_preflighted' and rec['leg2_start']['hypothetical'] is True
    assert rec['original']['decision'] == 'native_failure_all_precisions' and len(rec['original']['attempts']) == 3


@pytest.mark.parametrize('fault', ['exception', 'malformed', 'rejected_success'])
def test_original_non_native_outcomes_fail_closed_without_split(monkeypatch, fault):
    kw = dict(raise_targets=(.86,)) if fault == 'exception' else dict(malformed_targets=(.86,)) if fault == 'malformed' else {}
    arm, calls, _, _ = fake_arm(monkeypatch, **kw)
    task, actor, events = fake_task(arm)
    if fault == 'rejected_success':
        task._categorize_plan_valid = lambda *a: False
    assert m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005, .002), 'small_lift') is None
    assert all(c['z_goal'] == .86 for c in calls)  # never proposed the midpoint
    assert task._categorize_split_log[-1]['decision'].startswith('fail_closed_original')


@pytest.mark.parametrize('which', ['leg1', 'leg2'])
def test_split_leg_failures_fail_closed(monkeypatch, which):
    fails = (.86, .83) if which == 'leg1' else (.86,)
    arm, calls, _, _ = fake_arm(monkeypatch, fail_targets=fails)
    task, actor, events = fake_task(arm)
    if which == 'leg2':
        real = arm.planner._mplib.plan_screw
        def plan(goal_pose, current_qpos, **k):
            if abs(goal_pose.p[2] - .86) < 1e-9 and abs(current_qpos[0] - .83) < 1e-9:
                return {'status': 'screw plan failed'}
            return real(goal_pose, current_qpos, **k)
        arm.planner._mplib.plan_screw = plan
    assert m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005,), 'small_lift') is None
    assert events == []


def test_full_path_checks_cover_every_sample_measured_and_swept_fingers(monkeypatch):
    arm, _, store, _ = fake_arm(monkeypatch)
    seen = []
    arm.planner._mplib.check_for_env_collision = lambda q: seen.append(np.array(q)) or []
    plan = PlanResult(True, np.array([[.80] + [0.] * 6, [.83] + [0.] * 6, [.86] + [0.] * 6]), np.zeros((3, 7)))
    task, _, _ = fake_task(arm)
    full, move_idx, finger_idx = m.read_full9(arm)
    assert m.full_path_checks(task, arm, plan, full, finger_idx, None, [])['ok']
    assert len(seen) == 3 * 12  # measured pair + 11 widths per sample, both endpoints included
    assert any(abs(q[7] - .0137) < 1e-12 and abs(q[8] - .0141) < 1e-12 for q in seen)
    assert any(abs(q[7] - .04) < 1e-12 for q in seen) and any(q[7] == 0. for q in seen)
    arm.planner._mplib.check_for_env_collision = lambda q: ['hit'] if abs(q[0] - .86) < 1e-9 and q[7] == 0. else []
    out = m.full_path_checks(task, arm, plan, full, finger_idx, None, [])
    assert not out['ok'] and out['sample'] == 2


def test_payload_clearance_is_checked_on_every_sample(monkeypatch):
    arm, _, store, _ = fake_arm(monkeypatch)
    task, _, _ = fake_task(arm)
    plan = PlanResult(True, np.array([[.80] + [0.] * 6, [.86] + [0.] * 6]), np.zeros((2, 7)))
    full, _, finger_idx = m.read_full9(arm)
    local = np.array([[0., 0., -.02]] * 8)  # payload bottom 20 mm below TCP (above the table margin at both samples)
    assert m.full_path_checks(task, arm, plan, full, finger_idx, local, [])['ok']
    block = [np.array([[-1., -1., .835], [1., 1., .85]])]  # obstacle slab overlapping the final sample's payload bottom (.84)
    out = m.full_path_checks(task, arm, plan, full, finger_idx, local, block)
    assert not out['ok'] and out['reason'] == 'payload clearance'


# ---------------------------------------------------------- execution ----

def test_execute_split_orders_legs_and_remeasures_before_leg2(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005,), 'small_lift')
    n_before = len(calls)
    # first-precision leg2 fails once for the ORIGINAL target from the measured midpoint, then the second precision passes:
    assert m.execute_split(task, 'right', prop, actor, lambda tcp: None, lambda: [], 'small_lift')
    assert events == [('execute', .83), ('arrival', .83), ('execute', .86), ('arrival', .86)]
    assert len(calls) == n_before + 1 and abs(calls[-1]['start'][0] - .83) < 1e-12  # leg2 re-planned from measured state
    tail = task._categorize_split_log[-1]
    assert tail['event'] == 'midpoint_replan' and abs(tail['extra_hold_s'] - .1) < 1e-9


def test_leg2_loss_after_midpoint_rejects_without_further_motion(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005,), 'small_lift')
    real = arm.planner._mplib.plan_screw
    arm.planner._mplib.plan_screw = lambda goal_pose, current_qpos, **k: {'status': 'screw plan failed'}
    with pytest.raises(CategorizeMotionRejected, match='remaining leg'):
        m.execute_split(task, 'right', prop, actor, lambda tcp: None, lambda: [], 'small_lift')
    assert events == [('execute', .83), ('arrival', .83)]


def test_midpoint_arrival_failure_stops_before_leg2(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], None, [], (.005,), 'small_lift')
    task._categorize_arrival = lambda target, tag: task._reject_categorize_motion('arrival')
    with pytest.raises(CategorizeMotionRejected, match='arrival'):
        m.execute_split(task, 'right', prop, actor, lambda tcp: None, lambda: [], 'small_lift')
    assert events == [('execute', .83)]


# -------------------------------------------------------------- hooks ----

def test_descent_split_preflights_retreat_before_any_motion(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.70,))
    task, actor, events = fake_task(arm, label='086_woodenblock', model_id=1)
    seen = []
    monkeypatch.setattr('envs.task_bases.categorize_support.release.checked_plan',
                        lambda task_, arm_, start, target, local, obstacles: seen.append(('retreat', float(start[0]), float(target[2]))) or PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7))))
    target_link = np.r_[0., 0., .70, 1., 0., 0., 0.]
    prop = m.descent_split(task, arm, 'right', actor, store['q'][:7], target_link, None, [], np.r_[0., 0., .80, 1., 0., 0., 0.])
    assert prop is not None and prop['retreat'] is not None and seen == [('retreat', .70, .80)]
    assert events == []  # nothing moved during preflight


def test_descent_split_out_of_scope_returns_none_without_native_calls(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.70,))
    task, actor, events = fake_task(arm, label='035_apple', model_id=1)
    assert m.descent_split(task, arm, 'right', actor, store['q'][:7], np.r_[0., 0., .70, 1., 0., 0., 0.], None, [], np.r_[0., 0., .80, 1., 0., 0., 0.]) is None
    assert calls == []


def test_small_lift_split_requires_scope_and_hold(monkeypatch):
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm, held=False)
    assert m.small_lift_split(task, Pose([0., 0., .86]), 'right') is False and calls == []
    task, actor, events = fake_task(arm, label='035_apple', model_id=1)
    assert m.small_lift_split(task, Pose([0., 0., .86]), 'right') is False and calls == []


def test_actual_move_tcp_screw_hook_only_first_post_close_lift(monkeypatch):
    from envs.task_bases.categorize_support import motion
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))

    class Task(motion.CategorizeMotionMixin):
        def __init__(self):
            self.config = {m.OPT_IN: True}; self._CATEGORIZE_COLUMN_LAYOUT = True
            self.robot = NS(get_arm=lambda _: arm)
            self._categorize_active_pick = NS(actor=NS(entity=NS(get_AABB=lambda: np.array([[0, 0, .76], [.06, .06, .86]]))), get_center=lambda: np.array([.03, .03, .81]), radius=.03)
            self._sort_targets = [(self._categorize_active_pick.actor, '038_milk-box', 0, 0)]
            self._categorize_hold_local = None; self.scene = NS(sim=NS(dt=.001)); self.TABLE_TOP_Z = .765; self._OBSTACLE_MARGIN_Z = .01
            self.plan_success = True; self.splits = []
        def _categorize_inputs(self, target, tag): return arm, arm.entity.get_qpos()[:7], np.asarray(target, dtype=float)
        def _categorize_plan_valid(self, arm_, plan, current, target): return bool(plan is not None and plan.success)
        def _execute_categorize_plan(self, plan, tag): pass
        def _categorize_arrival(self, target, tag): pass
    task = Task()
    arm.planner.plan_screw_path = lambda current, target, qpos_step: PlanResult(False)
    monkeypatch.setattr(m, 'small_lift_split', lambda task_, target_tcp, tag: task_.splits.append(float(target_tcp.p[2])) or True)
    monkeypatch.setattr(m, 'payload_obstacles', lambda *a: [])
    # not held: original rejection, no split
    with pytest.raises(CategorizeMotionRejected):
        task._move_tcp_screw(Pose([0., 0., .86]), 'right')
    assert task.splits == []
    # held: first screw after closure is armed
    task._categorize_hold_local = np.zeros(3); task._categorize_split_lift_armed = True
    assert task._move_tcp_screw(Pose([0., 0., .86]), 'right') is True and task.splits == [.86]
    # second screw while held is not the first lift: original rejection stands
    with pytest.raises(CategorizeMotionRejected):
        task._move_tcp_screw(Pose([0., 0., .90]), 'right')
    assert task.splits == [.86]


def test_begin_hold_arms_the_first_lift(monkeypatch):
    from envs.task_bases.categorize_support import motion
    task = NS(_categorize_active_pick=NS(get_center=lambda: np.array([0., 0., .8])),
              robot=NS(get_arm=lambda _: NS(get_ee_pose=lambda: np.r_[0., 0., .79, 1., 0., 0., 0.])),
              _reject_categorize_motion=lambda r: (_ for _ in ()).throw(CategorizeMotionRejected(r)))
    motion.CategorizeMotionMixin._categorize_begin_hold(task, 'right')
    assert task._categorize_split_lift_armed is True


def test_default_off_never_touches_the_split_module(monkeypatch):
    from envs.task_bases.categorize_support import motion
    arm, calls, store, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    called = []
    monkeypatch.setattr(m, 'small_lift_split', lambda *a: called.append(1) or True)

    class Task(motion.CategorizeMotionMixin):
        def __init__(self):
            self.config = {}; self._CATEGORIZE_COLUMN_LAYOUT = True; self.robot = NS(get_arm=lambda _: arm)
            self._categorize_active_pick = NS(actor=None, get_center=lambda: np.array([.03, .03, .81]), radius=.03)
            self._categorize_hold_local = np.zeros(3); self._categorize_split_lift_armed = True; self.plan_success = True
        def _categorize_inputs(self, target, tag): return arm, arm.entity.get_qpos()[:7], np.asarray(target, dtype=float)
        def _categorize_plan_valid(self, *a): return False
    task = Task()
    arm.planner.plan_screw_path = lambda current, target, qpos_step: PlanResult(False)
    with pytest.raises(CategorizeMotionRejected):
        task._move_tcp_screw(Pose([0., 0., .86]), 'right')
    assert called == []


def test_lower_release_hook_uses_split_only_after_original_and_never_opens_on_failure(monkeypatch):
    from envs.task_bases.categorize_support import release as rel
    arm, calls, store, _ = fake_arm(monkeypatch)
    events = []
    aabb = np.array([[-.27, -.22, .940], [-.23, -.18, .980]])
    actor = NS(entity=NS(get_AABB=lambda: aabb.copy()))
    world = NS(remove_object=lambda name: events.append(('remove', name)))
    arm.get_ee_pose = lambda: np.array([-.25, -.2, 1.001, 0., 0., 1., 0.])
    arm.planner._mplib.planning_world = world
    arm.tcp_offset = Pose([0, 0, .1034])
    def reject(reason): raise CategorizeMotionRejected(reason)
    task = NS(config={m.OPT_IN: True}, _CATEGORIZE_COLUMN_LAYOUT=True, _categorize_hold_local=np.zeros(3),
              _sort_targets=[(actor, '086_woodenblock', 1, 0)], basket_poses=[Pose([-.25, -.2, .821])],
              _BASKET_HALF_X=.096, _BASKET_HALF_Y=.096, _BASKET_HALF_Z=.056, _OBSTACLE_MARGIN_XY=.02, _OBSTACLE_MARGIN_Z=.01,
              TABLE_TOP_Z=.765, _categorize_skip_basket=None, robot=NS(get_arm=lambda _: arm), _reject_categorize_motion=reject,
              _categorize_inputs=lambda goal, tag: (arm, np.zeros(7), np.asarray(goal, dtype=float)),
              _categorize_refresh_obstacles=lambda tag: events.append(('restore',)), step_sim=lambda: events.append(('step',)),
              open_gripper=lambda tag: events.append(('open',)), scene=NS(sim=NS(dt=.001)))
    task._execute_categorize_plan = lambda plan, tag: events.append(('execute',))
    task._categorize_arrival = lambda target, tag: events.append(('arrival',))
    monkeypatch.setattr(rel, 'checked_plan', lambda *a: None)  # original descent attempts fail
    monkeypatch.setattr(rel, 'install_solids', lambda world_, boxes: ['s0', 's1', 's2', 's3', 's4'])
    monkeypatch.setattr(rel, 'payload_obstacles', lambda *a: [np.array([[-1, -1, 0], [1, 1, 1]])])
    monkeypatch.setattr(m, 'descent_split', lambda *a: None)  # split fails closed
    with pytest.raises(CategorizeMotionRejected, match='no checked descent'):
        rel.lower_release(task, actor, '086_woodenblock', 'right')
    assert ('open',) not in events and ('execute',) not in events and events[-1] == ('restore',)
    # opt-in off: descent_split never consulted
    task.config = {}
    seen = []
    monkeypatch.setattr(m, 'descent_split', lambda *a: seen.append(1))
    with pytest.raises(CategorizeMotionRejected):
        rel.lower_release(task, actor, '086_woodenblock', 'right')
    assert seen == []


def test_cloud_identity_records_exact_planner_input():
    a = m.identity_of_cloud(np.zeros((5, 3))); b = m.identity_of_cloud(np.zeros((5, 3))); c = m.identity_of_cloud(np.ones((5, 3)))
    assert a == b and a != c and a['count'] == 5


def test_small_lift_payload_floor_is_its_own_start_and_descent_keeps_table_margin(monkeypatch):
    arm, _, store, _ = fake_arm(monkeypatch)
    task, _, _ = fake_task(arm)
    start = np.r_[0., 0., .80, 1., 0., 0., 0.]
    resting = np.array([[0., 0., -.035]] * 8)  # payload bottom at .765 = on the table at the start of the lift
    assert abs(m.payload_floor(task, 'small_lift', start, resting) - (.765 - 1e-6)) < 1e-9
    assert m.payload_floor(task, 'descent', start, resting) == .775
    assert m.payload_floor(task, 'small_lift', start, None) == .775
    full, _, finger_idx = m.read_full9(arm)
    up = PlanResult(True, np.array([[.80] + [0.] * 6, [.86] + [0.] * 6]), np.zeros((2, 7)))
    floor = m.payload_floor(task, 'small_lift', start, resting)
    assert m.full_path_checks(task, arm, up, full, finger_idx, resting, [], floor)['ok']
    assert not m.full_path_checks(task, arm, up, full, finger_idx, resting, [])['ok']  # table margin would wrongly reject the lift
    down = PlanResult(True, np.array([[.80] + [0.] * 6, [.79] + [0.] * 6]), np.zeros((2, 7)))
    out = m.full_path_checks(task, arm, down, full, finger_idx, resting, [], floor)
    assert not out['ok'] and out['sample'] == 1  # never below the measured start


def test_propose_split_from_a_table_resting_start_uses_the_start_floor(monkeypatch):
    arm, calls, _, _ = fake_arm(monkeypatch, fail_targets=(.86,))
    task, actor, events = fake_task(arm)
    resting = np.array([[0., 0., -.035]] * 8)
    prop = m.propose_split(task, arm, 'right', np.r_[0., 0., .86, 1., 0., 0., 0.], resting, [], (.005,), 'small_lift')
    assert prop is not None and prop['plan'] is None and abs(prop['floor_z'] - (.765 - 1e-6)) < 1e-9
