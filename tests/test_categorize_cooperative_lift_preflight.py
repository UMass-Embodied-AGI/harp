"""Non-executing cooperative lift checks preserve native guards and siblings."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases import categorize as core
from envs.task_bases.categorize_support.cooperative_lift import OPT_IN, cooperative_lift_check
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence
from envs.utils import Pose


def fixture_case(name='073_rubikscube', model_id=1):
    original = np.linspace(.1, .9, 9)
    state = dict(qpos=original.copy())
    calls, checks, restores = [], [], []
    def set_qpos(qpos, full):
        state['qpos'] = np.array(qpos)
        restores.append(('robot', np.array(qpos), full))
    backend = NS(
        robot=NS(get_qpos=lambda: state['qpos'].copy(), set_qpos=set_qpos),
        move_group_joint_indices=np.arange(7),
        planning_world=NS(set_qpos_all=lambda q: restores.append(('world', np.array(q)))),
        pinocchio_model=NS(compute_forward_kinematics=lambda q: restores.append(('fk', np.array(q)))),
    )
    def screw(current, target, **kwargs):
        plan = PlanResult(True, np.array([current, current + .01]), np.zeros((2, 7)))
        calls.append((current.copy(), target.copy(), kwargs, plan))
        state['qpos'][:] = 9.
        return plan
    arm = NS(n_arm=7, tcp_offset=Pose([0., 0., .1034]),
             planner=NS(_mplib=backend, plan_screw_path=screw))
    def validate(checked_arm, plan, current, target):
        assert checked_arm is arm
        checks.append((plan, current.copy(), target.copy()))
        return bool(plan is not None and plan.success)
    actor = object()
    pick = NS(actor=actor, label=name, radius=.026,
              get_center=lambda: np.array([.21, -.11, .8]))
    task = NS(config={OPT_IN: True}, _CATEGORIZE_COLUMN_LAYOUT=False,
              _sort_targets=[(actor, name, model_id, 0)], robot=NS(get_arm=lambda _: arm),
              _categorize_plan_valid=validate)
    grasp = PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7)))
    plans = (None, grasp, Pose([.2, -.1, .7]), None)
    return task, pick, plans, arm, calls, checks, restores, state, original


@pytest.mark.parametrize('value', [None, False, 0, 1, 'true', 1.0, np.bool_(True)])
def test_only_literal_opt_in_adds_a_check(value):
    task, pick, *_ = fixture_case()
    task.config = {} if value is None else {OPT_IN: value}
    marker = object()
    assert cooperative_lift_check(task, pick, 'right', 1.001, marker) is marker


@pytest.mark.parametrize('name,mid', [('038_milk-box', 1), ('073_rubikscube', 0),
                                      ('031_jam-jar', 0), ('048_stapler', 0), ('cube', 0)])
def test_unlisted_models_leave_existing_predicate_untouched(name, mid):
    task, pick, *_ = fixture_case(name, mid)
    marker = object()
    assert cooperative_lift_check(task, pick, 'right', 1.001, marker) is marker


@pytest.mark.parametrize('name,mid', [('038_milk-box', 0), ('073_rubikscube', 1)])
def test_both_original_lifts_use_proposed_endpoint_and_unchanged_native_validation(name, mid):
    task, pick, plans, arm, calls, checks, restores, state, original = fixture_case(name, mid)
    assert cooperative_lift_check(task, pick, 'right', 1.001)(plans)
    assert len(calls) == len(checks) == 2
    np.testing.assert_array_equal(calls[0][0], plans[1].position[-1])
    np.testing.assert_array_equal(calls[1][0], calls[0][3].position[-1])
    for call, z in zip(calls, [.826, 1.001]):
        tcp = Pose.from_pose7(call[1]) * arm.tcp_offset
        np.testing.assert_allclose(tcp.p, [.2, -.1, z])
        np.testing.assert_allclose(tcp.q, plans[2].q)
        assert call[2] == dict(qpos_step=.005, log=False)
    assert [entry[0] for entry in restores] == ['robot', 'world', 'fk']
    np.testing.assert_array_equal(state['qpos'], original)
    np.testing.assert_array_equal(restores[1][1], original[:7])
    np.testing.assert_array_equal(restores[2][1], original)


@pytest.mark.parametrize('phase', ['small', 'high'])
def test_guard_rejection_exhausts_only_existing_precisions_and_rejects_candidate(phase):
    task, pick, plans, arm, calls, _, _, state, original = fixture_case()
    task._categorize_plan_valid = lambda arm_, plan, current, target: (
        phase == 'high' and target[2] < .8
    )
    assert not cooperative_lift_check(task, pick, 'right', 1.001)(plans)
    steps = [call[2]['qpos_step'] for call in calls]
    assert steps == ([.005, .002, .001] if phase == 'small' else [.005, .005, .002, .001])
    assert all(call[1][2] < .8 for call in calls) if phase == 'small' else calls[-1][1][2] > .8
    np.testing.assert_array_equal(state['qpos'], original)


def test_existing_rejecting_callback_runs_first_and_never_plans():
    task, pick, plans, _, calls, _, restores, state, original = fixture_case()
    seen = []
    def old(value):
        seen.append(value)
        state['qpos'][:] = 9.
        return False
    assert not cooperative_lift_check(task, pick, 'right', 1.001, old)(plans)
    assert seen == [plans] and calls == []
    assert [entry[0] for entry in restores] == ['robot', 'world', 'fk']
    np.testing.assert_array_equal(state['qpos'], original)
    np.testing.assert_array_equal(restores[1][1], original[:7])
    np.testing.assert_array_equal(restores[2][1], original)


def test_existing_callback_errors_propagate_and_restore_planner_state():
    task, pick, plans, _, calls, _, restores, state, original = fixture_case()
    def old(_plans):
        state['qpos'][:] = 9.
        raise RuntimeError('existing callback failure')
    with pytest.raises(RuntimeError, match='existing callback failure'):
        cooperative_lift_check(task, pick, 'right', 1.001, old)(plans)
    assert calls == []
    assert [entry[0] for entry in restores] == ['robot', 'world', 'fk']
    np.testing.assert_array_equal(state['qpos'], original)
    np.testing.assert_array_equal(restores[1][1], original[:7])
    np.testing.assert_array_equal(restores[2][1], original)


def test_existing_accepting_callback_cannot_bypass_new_checks():
    task, pick, plans, _, calls, *_ = fixture_case()
    seen = []
    task._categorize_plan_valid = lambda *_: False
    assert not cooperative_lift_check(task, pick, 'right', 1.001,
                                      lambda value: seen.append(value) or True)(plans)
    assert seen == [plans] and len(calls) == 3


@pytest.mark.parametrize('where', ['planner', 'validator'])
def test_native_errors_propagate_and_restore_planner_state(where):
    task, pick, plans, arm, _, _, restores, state, original = fixture_case()
    def fail(*args, **kwargs):
        state['qpos'][:] = 9.
        raise RuntimeError('native failure')
    if where == 'planner':
        arm.planner.plan_screw_path = fail
    else:
        task._categorize_plan_valid = fail
    with pytest.raises(RuntimeError, match='native failure'):
        cooperative_lift_check(task, pick, 'right', 1.001)(plans)
    np.testing.assert_array_equal(state['qpos'], original)
    assert [entry[0] for entry in restores] == ['robot', 'world', 'fk']


@pytest.mark.parametrize('task_class', (core.CategorizeTaskCore, CategorizeObjectsInterruption, CategorizeObjectsCopresence))
def test_core_and_siblings_ignore_even_explicit_cooperative_flag(task_class):
    task = task_class({OPT_IN: True})
    pick = NS(label='073_rubikscube')
    assert task._categorize_yaml_preflight(pick, None, 'right', 1.001) is None


def test_default_cooperative_hook_preserves_existing_milk_callback(monkeypatch):
    task = CategorizeObjectsContribution({})
    task.MILK_CONTINUATION_PREFLIGHT = True
    marker = object()
    monkeypatch.setattr(core, 'milk_continuation_check', lambda *_: marker)
    pick = NS(label='038_milk-box')
    assert task._categorize_yaml_preflight(pick, None, 'right', 1.001) is marker
    assert OPT_IN not in task.config
