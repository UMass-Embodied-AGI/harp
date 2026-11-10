from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases.categorize_support import carry
from envs.task_bases.categorize_support import radius
from envs.task_bases.categorize_support.motion import CategorizeMotionRejected
from envs.utils import Pose


@pytest.mark.parametrize('column,name,mid,drop,transport,expected', [
    (True, '035_apple', 1, 1.001, None, True),
    (False, '035_apple', 1, 1.001, None, False),
    (True, '035_apple', 0, 1.001, None, False),
    (True, '038_milk-box', 0, 1.001, None, False),
    (True, 'cube', 0, 1.001, None, False),
    (True, '035_apple', 1, None, None, False),
    (True, '035_apple', 1, 1.001, 1.2, False),
])
def test_scope(column, name, mid, drop, transport, expected):
    assert carry.uses_direct_carry(NS(_CATEGORIZE_COLUMN_LAYOUT=column), name, mid,
        NS(drop_from_z=drop, transport_z=transport)) is expected


def case(monkeypatch):
    center = np.array([-.2, .1, .815])
    tcp = np.array([-.2, .1, .809, 0., 0., 1., 0.])
    anchor = np.array([0., 0., -.006])
    events = []
    arm = NS(get_ee_pose=lambda: tcp.copy(), tcp_offset=Pose([0., 0., .1034]))
    actor = NS(entity=NS(get_AABB=lambda: np.array([[-.227, .073, .787], [-.173, .127, .84]])))
    pick = NS(get_center=lambda: center.copy(), radius=.025)
    place = NS(pos=np.array([-.23, -.2, .821]), drop_from_z=1.001, transport_z=None)
    def reject(reason): raise CategorizeMotionRejected(reason)
    def execute(plan, tag):
        events.append('execute')
        center[2] += .19
    task = NS(robot=NS(get_arm=lambda _: arm), _categorize_hold_local=anchor,
        TABLE_TOP_Z=.765, _OBSTACLE_MARGIN_Z=.010,
        _reject_categorize_motion=reject,
        _categorize_inputs=lambda target, tag: (arm, np.zeros(7), target),
        _execute_categorize_plan=execute,
        _categorize_arrival=lambda *a: events.append('arrival'),
        step_sim=lambda: None, open_gripper=lambda _: events.append('open'))
    def plan(*args):
        events.append('preflight')
        assert args[-2].shape == (8, 3)
        return NS()
    monkeypatch.setattr(carry, 'checked_plan', plan)
    monkeypatch.setattr(carry, 'payload_obstacles', lambda *a: [])
    return task, pick, place, actor, events, anchor


def test_success_preflights_then_checks_arrival_and_lift_before_open(monkeypatch):
    task, pick, place, actor, events, anchor = case(monkeypatch)
    assert carry.carry_and_release(task, pick, place, actor, 'right')
    assert events == ['preflight', 'execute', 'arrival', 'open']
    assert task._categorize_hold_local is anchor


@pytest.mark.parametrize('failure', ['no_hold', 'table', 'native', 'tracking', 'lift', 'settle'])
def test_rejections_never_open(monkeypatch, failure):
    task, pick, place, actor, events, _ = case(monkeypatch)
    if failure == 'no_hold': task._categorize_hold_local = None
    if failure == 'table':
        actor.entity.get_AABB = lambda: np.array([[-.23, .07, .774], [-.17, .13, .83]])
    if failure == 'native': monkeypatch.setattr(carry, 'checked_plan', lambda *a: None)
    if failure == 'tracking':
        task._execute_categorize_plan = lambda *a: task._reject_categorize_motion('tracking')
    if failure == 'lift': task._execute_categorize_plan = lambda *a: None
    if failure == 'settle': task.step_sim = lambda: task._reject_categorize_motion('retention')
    with pytest.raises(CategorizeMotionRejected):
        carry.carry_and_release(task, pick, place, actor, 'right')
    assert 'open' not in events


def test_actual_radius_dispatch_calls_direct_carry_only_after_small_lift(monkeypatch):
    # Use the established radius fixture to execute the real dispatch path.
    from test_categorize_cube import setup_case
    task, pick, place, actor, events = setup_case()
    task._CATEGORIZE_COLUMN_LAYOUT = True
    def finish(*args):
        assert [e[0] for e in events].count('move') == 1
        assert [e[0] for e in events].count('hold') == 1
        events.append(('direct',))
        return True
    monkeypatch.setattr(radius, 'carry_and_release', finish)
    assert radius.pick_and_place_radius(task, pick, place, actor, 'right',
        object_name='035_apple', model_id=1, object_scale=.7)
    assert events[-1] == ('direct',)


# --- milk corner route -------------------------------------------------------

@pytest.mark.parametrize('column,name,mid,drop,expected', [
    (True, '038_milk-box', 0, 1.001, True),
    (False, '038_milk-box', 0, 1.001, False),
    (True, '038_milk-box', 1, 1.001, False),
    (True, '035_apple', 0, 1.001, False),
    (True, '038_milk-box', 0, None, False),
])
def test_milk_corner_scope(column, name, mid, drop, expected):
    assert carry.uses_milk_corner(NS(_CATEGORIZE_COLUMN_LAYOUT=column), name, mid,
                                  NS(drop_from_z=drop)) is expected


SEED0_TCP = np.array([-0.2111325968146091, 0.1234653901005969, 0.8529640777114179,
                      0.0037052777629341008, 0.7117350763318744, 0.010759606155841261,
                      -0.7023558093380675])
SEED0_AABB = np.array([[-0.24127820134162903, 0.09227430820465088, 0.789585292339325],
                       [-0.17118626832962036, 0.15557235479354858, 0.9126424789428711]])
SEED0_CENTER = np.array([-0.20692059303395807, 0.1236325110617204, 0.8511143347478911])
SEED0_BASKET = np.array([-0.23630631681329076, -0.2, 0.821])
SEED0_PADDED = np.array([SEED0_BASKET - [.116, .116, .066], SEED0_BASKET + [.116, .116, .066]])


def test_milk_corner_geometry_reproduces_frozen_seed0_probe():
    target_xy = SEED0_BASKET[:2] - (SEED0_CENTER - SEED0_TCP[:3])[:2]
    fraction, required, corner = carry.milk_corner_geometry(SEED0_TCP, SEED0_AABB, target_xy, 1.001, SEED0_PADDED)
    assert abs(fraction - 0.54467429) < 1e-6
    assert abs(required - 0.960378785) < 1e-6
    assert abs(corner - 0.911786468) < 1e-6
    # The diagonal from the corner reaches the required height exactly at the crossing.
    assert abs(corner + fraction * (1.001 - corner) - required) < 1e-9


def test_milk_corner_geometry_rejects_degenerate_routes():
    target_xy = SEED0_BASKET[:2] - (SEED0_CENTER - SEED0_TCP[:3])[:2]
    # Already above the padded basket: no crossing inside the translation.
    above = SEED0_TCP.copy(); above[:2] = target_xy
    assert carry.milk_corner_geometry(above, SEED0_AABB + np.r_[target_xy - SEED0_TCP[:2], 0.], target_xy, 1.001, SEED0_PADDED) is None
    # Translation that never enters the footprint.
    assert carry.milk_corner_geometry(SEED0_TCP, SEED0_AABB, SEED0_TCP[:2] + [0., .2], 1.001, SEED0_PADDED) is None
    # Release height too low for any corner below it.
    assert carry.milk_corner_geometry(SEED0_TCP, SEED0_AABB, target_xy, .95, SEED0_PADDED) is None
    # Already high enough: corner collapses onto the current TCP height.
    high = SEED0_TCP.copy(); high[2] = .99
    fraction, required, corner = carry.milk_corner_geometry(high, SEED0_AABB + [0., 0., .99 - SEED0_TCP[2]], target_xy, 1.001, SEED0_PADDED)
    assert corner == high[2]


def milk_case(monkeypatch, *, rise=True):
    tcp = SEED0_TCP.copy()
    center = SEED0_CENTER.copy()
    aabb = SEED0_AABB.copy()
    anchor = np.array([0., 0., -.002])
    events = []
    arm = NS(get_ee_pose=lambda: tcp.copy(), tcp_offset=Pose([0., 0., .1034]))
    actor = NS(entity=NS(get_AABB=lambda: aabb.copy()))
    pick = NS(get_center=lambda: center.copy(), radius=.035)
    place = NS(pos=SEED0_BASKET.copy(), drop_from_z=1.001)
    def reject(reason): raise CategorizeMotionRejected(reason)
    def inputs(target, tag):
        events.append('inputs'); return arm, np.zeros(7), target
    def execute(plan, tag):
        events.append(('execute', plan.leg))
        dz = (Pose.from_pose7(plan.target) * arm.tcp_offset).p[2] - tcp[2]
        tcp[2] += dz; center[2] += dz; aabb[:, 2] += dz
    task = NS(robot=NS(get_arm=lambda _: arm), _categorize_hold_local=anchor,
        TABLE_TOP_Z=.765, _OBSTACLE_MARGIN_Z=.010, _sort_targets=[(actor, '038_milk-box', 0, 0)],
        _reject_categorize_motion=reject, _categorize_inputs=inputs,
        _execute_categorize_plan=execute,
        _categorize_arrival=lambda target, tag: events.append('arrival'))
    plans = []
    def plan(task_, arm_, current, target, local, obstacles):
        assert local.shape == (8, 3) and obstacles
        leg = 'rise' if len(plans) == 0 and rise else 'diagonal'
        plans.append(leg); events.append(('preflight', leg))
        return NS(position=np.array([current, current]), leg=leg, target=np.asarray(target, dtype=float))
    monkeypatch.setattr(carry, 'checked_plan', plan)
    monkeypatch.setattr(carry, 'payload_obstacles', lambda *a: [SEED0_PADDED.copy()])
    return task, pick, place, actor, events, anchor, tcp, center


def test_milk_corner_success_preflights_both_legs_then_replans_after_rise(monkeypatch):
    task, pick, place, actor, events, anchor, tcp, center = milk_case(monkeypatch)
    assert carry.milk_corner_carry(task, pick, place, actor, 'right', 1.001)
    assert events == ['inputs', ('preflight', 'rise'), ('preflight', 'diagonal'),
                      ('execute', 'rise'), 'arrival', 'inputs', ('preflight', 'diagonal'),
                      ('execute', 'diagonal'), 'arrival']
    assert abs(tcp[2] - 1.001) < 1e-9 and center[2] - SEED0_CENTER[2] > .14
    assert task._categorize_hold_local is anchor


@pytest.mark.parametrize('failure', ['no_hold', 'basket', 'table', 'geometry', 'rise', 'diagonal'])
def test_milk_corner_preflight_failures_fall_back_without_moving(monkeypatch, failure):
    task, pick, place, actor, events, anchor, tcp, center = milk_case(monkeypatch)
    if failure == 'no_hold': task._categorize_hold_local = None
    if failure == 'basket': task._sort_targets = []
    if failure == 'table': actor.entity.get_AABB = lambda: np.array([[-.24, .09, .770], [-.17, .16, .89]])
    if failure == 'geometry': place.pos = SEED0_TCP[:3] + [0., .2, 0.]
    if failure == 'rise': monkeypatch.setattr(carry, 'checked_plan', lambda *a: None)
    if failure == 'diagonal':
        calls = []
        def plan(*a):
            calls.append(1); events.append(('preflight', len(calls)))
            return NS(position=np.zeros((2, 7)), leg='rise', target=np.asarray(a[3], dtype=float)) if len(calls) == 1 else None
        monkeypatch.setattr(carry, 'checked_plan', plan)
    assert carry.milk_corner_carry(task, pick, place, actor, 'right', 1.001) is False
    assert not any(isinstance(e, tuple) and e[0] == 'execute' for e in events)
    assert abs(tcp[2] - SEED0_TCP[2]) < 1e-12


@pytest.mark.parametrize('failure', ['replan', 'lift'])
def test_milk_corner_execution_failures_reject_without_opening(monkeypatch, failure):
    task, pick, place, actor, events, anchor, tcp, center = milk_case(monkeypatch)
    if failure == 'replan':
        calls = []
        def plan(*a):
            calls.append(1)
            if len(calls) == 3: return None
            leg = 'rise' if len(calls) == 1 else 'diagonal'
            return NS(position=np.zeros((2, 7)), leg=leg, target=np.asarray(a[3], dtype=float))
        monkeypatch.setattr(carry, 'checked_plan', plan)
    if failure == 'lift':
        def execute(plan, tag):
            events.append(('execute', plan.leg))
            arm = task.robot.get_arm('right')
            tcp[2] = (Pose.from_pose7(plan.target) * arm.tcp_offset).p[2]  # object stays behind
        task._execute_categorize_plan = execute
    with pytest.raises(CategorizeMotionRejected):
        carry.milk_corner_carry(task, pick, place, actor, 'right', 1.001)
    assert task._categorize_hold_local is anchor


def test_milk_corner_without_rise_plans_single_diagonal(monkeypatch):
    task, pick, place, actor, events, anchor, tcp, center = milk_case(monkeypatch, rise=False)
    tcp[2] = .95; center[2] += .95 - SEED0_TCP[2]
    actor.entity.get_AABB = lambda: SEED0_AABB + [0., 0., .95 - SEED0_TCP[2]]
    assert carry.milk_corner_carry(task, pick, place, actor, 'right', 1.001)
    assert [e for e in events if isinstance(e, tuple)] == [('preflight', 'diagonal'), ('execute', 'diagonal')]


@pytest.mark.parametrize('column,mid,corner,expected_moves', [
    (True, 0, True, 1), (True, 0, False, 3), (True, 1, True, 3), (False, 0, True, 3),
])
def test_actual_yaml_dispatch_uses_corner_only_for_column_milk_model0(monkeypatch, column, mid, corner, expected_moves):
    from envs.task_bases import categorize as core
    from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
    task = CategorizeObjectsInterruption({})
    task._CATEGORIZE_COLUMN_LAYOUT = column
    task._USE_OBSTACLE_INFLATION = False
    task.TABLE_TOP_Z = .765
    tcp = np.array([-.25, .15, .795, 0., 0., 1., 0.])
    center = tcp[:3].copy()
    actor = NS(get_pose=lambda: Pose(center.copy()))
    arm = NS(tcp_offset=Pose(), entity=NS(get_qpos=lambda: np.zeros(9)),
             get_ee_pose=lambda: tcp.copy(), finger_joints=[])
    task.robot = NS(get_arm=lambda _: arm)
    task._sort_targets = [(actor, '038_milk-box', mid, 0)]
    task._categorize_hold_local = None
    events, moves = [], []
    def move(pose, tag):
        moves.append(pose); center[:] += pose.p - tcp[:3]; tcp[:3] = pose.p; return True
    def corner_carry(owner, pick, place, payload, tag, safe_z):
        assert owner is task and payload is actor and tag == 'right' and safe_z == 1.001
        assert task._categorize_hold_local is not None and len(moves) == 1
        events.append('corner')
        if corner:
            center[2] += .2; tcp[2] += .2
        return corner
    def opening(tag):
        events.append(('open', task._categorize_hold_local is not None)); task._categorize_hold_local = None
    monkeypatch.setattr(task, '_get_object_scale', lambda *_: 1.)
    monkeypatch.setattr(task, '_select_and_execute_task_grasp', lambda *a, **k: (None, None, NS(name='grasp_058')))
    monkeypatch.setattr(task, 'open_gripper', opening)
    monkeypatch.setattr(task, 'set_gripper', lambda *a: None)
    monkeypatch.setattr(task, 'step_sim', lambda: None)
    monkeypatch.setattr(task, '_categorize_begin_hold', lambda tag: setattr(task, '_categorize_hold_local', np.zeros(3)))
    monkeypatch.setattr(task, '_move_tcp_screw', move)
    monkeypatch.setattr(core, 'milk_corner_carry', corner_carry)
    pick = NS(label='038_milk-box', actor=actor, radius=.035, get_center=lambda: center.copy())
    place = NS(pos=np.array([-.25, -.2, .821]), drop_from_z=1.001, basket_idx=0)
    assert task._pick_and_place_checked(pick, place, 'right')
    assert len(moves) == expected_moves
    assert ('corner' in events) == (column and mid == 0)
    assert events[-1] == ('open', True)
