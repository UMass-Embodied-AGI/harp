"""Tests for the task-local, opt-in held-friction correction.

The override must be a literal ``True`` opt-in, cover exactly six (name, model) pairs, leave the
procedural cube branch untouched, and reproduce the default spawn friction otherwise.
"""
import numpy as np
import pytest

from envs.object_catalog import get_entry
from envs.task_bases import categorize as core
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption

ALLOWLIST = {('038_milk-box', 0), ('086_woodenblock', 1), ('035_apple', 1), ('073_rubikscube', 1),
             ('048_stapler', 0), ('031_jam-jar', 0)}
OTHERS = [('035_apple', 0), ('038_milk-box', 1), ('086_woodenblock', 0), ('073_rubikscube', 0),
          ('048_stapler', 1), ('031_jam-jar', 1), ('100_seal', 1)]
QUAT = np.array([1., 0., 0., 0.])


def v11_friction(name, model_id):
    """The default rule: (name, model) override, then name override, then catalog capped at 5."""
    table = core.CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE
    key = (name, model_id) if (name, model_id) in table else name
    if key in table:
        return float(table[key])
    return float(min(5.0, get_entry(name).friction))


def spawn(monkeypatch, config, name, model_id):
    calls = {}
    def fake_load(scene, pose, obj_name, **kwargs):
        calls['load'] = dict(obj_name=obj_name, **kwargs)
        return 'loaded'
    def fake_prim(scene, kind, pose, **kwargs):
        calls['prim'] = dict(kind=kind, pose=[list(map(float, pose.p)), list(map(float, pose.q))], **kwargs)
        return 'prim'
    monkeypatch.setattr(core, 'load_object', fake_load)
    monkeypatch.setattr(core, 'create_primitive', fake_prim)
    task = CategorizeObjectsInterruption(dict(config))
    task.scene = None
    task._spawn_categorize_object(name, model_id, .1, .2, .785, QUAT)
    return calls


def test_constants_are_exactly_the_authorized_boundary():
    assert core.CategorizeTaskCore.PAYLOAD_FRICTION_V1_MODELS == frozenset(ALLOWLIST)
    assert core.CategorizeTaskCore.PAYLOAD_FRICTION_V1_VALUE == 1.0
    assert core.CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE == {'086_woodenblock': 2.5}


@pytest.mark.parametrize('name,model_id', sorted(ALLOWLIST) + OTHERS)
def test_explicit_off_reproduces_v11(monkeypatch, name, model_id):
    calls = spawn(monkeypatch, {'categorize_payload_friction_v1': False}, name, model_id)
    assert calls['load']['friction'] == v11_friction(name, model_id)
    assert calls['load']['model_id'] == model_id and calls['load']['convex'] is True
    assert calls['load']['is_static'] is False


@pytest.mark.parametrize('name,model_id', sorted(ALLOWLIST))
def test_literal_opt_in_sets_allowlisted_pairs_to_one(monkeypatch, name, model_id):
    calls = spawn(monkeypatch, {'categorize_payload_friction_v1': True}, name, model_id)
    assert calls['load']['friction'] == 1.0
    assert v11_friction(name, model_id) != 1.0  # the treatment actually changes these pairs


@pytest.mark.parametrize('name,model_id', OTHERS)
def test_opt_in_leaves_other_models_and_objects_unchanged(monkeypatch, name, model_id):
    calls = spawn(monkeypatch, {'categorize_payload_friction_v1': True}, name, model_id)
    assert calls['load']['friction'] == v11_friction(name, model_id)


@pytest.mark.parametrize('value', ['True', 1, 'yes', 1.0, 'true', [True]])
def test_non_literal_values_do_not_opt_in(monkeypatch, value):
    for name, model_id in sorted(ALLOWLIST):
        calls = spawn(monkeypatch, {'categorize_payload_friction_v1': value}, name, model_id)
        assert calls['load']['friction'] == v11_friction(name, model_id)


def test_cube_branch_is_identical_in_both_modes_and_has_no_friction_argument(monkeypatch):
    off = spawn(monkeypatch, {'categorize_payload_friction_v1': False}, 'cube', 0)
    on = spawn(monkeypatch, {'categorize_payload_friction_v1': True}, 'cube', 0)
    assert 'load' not in off and 'load' not in on
    assert off['prim'] == on['prim']
    assert 'friction' not in off['prim']


def test_override_is_fixed_at_spawn_and_not_a_runtime_switch():
    # The overlay adds no attribute or method that could change friction after creation.
    names = [n for n in dir(core.CategorizeTaskCore) if 'friction' in n.lower()]
    assert sorted(names) == ['PAYLOAD_FRICTION_V1_MODELS', 'PAYLOAD_FRICTION_V1_VALUE', 'SPAWN_FRICTION_OVERRIDE']
