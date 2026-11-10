"""Cooperative radius acquisition keeps contact geometry and sibling behavior."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases import categorize as core
from envs.task_bases.categorize import PickSpec, PlaceSpec
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence


MODELS = (('048_stapler', 0), ('031_jam-jar', 0), ('100_seal', 1))


def radius_case(task_class, monkeypatch, name, model_id):
    task = task_class({})
    center = np.array([-.15, -.10, .810])
    bounds = np.array([[-.175, -.13, .765], [-.125, -.07, .855]])
    actor = NS(entity=NS(get_AABB=lambda: bounds.copy()))
    task._sort_targets = [(actor, name, model_id, 0)]
    task._get_object_scale = lambda *_: .7
    task._categorize_begin_hold = lambda _: None
    selected, legacy = [], []
    monkeypatch.setattr(core, 'pick_and_place_radius',
                        lambda *args, **kwargs: selected.append((args, kwargs)) or True)
    monkeypatch.setattr(core.TopDownPickPlaceMixin, 'pick_and_place',
                        lambda *args: legacy.append(args) or True)
    monkeypatch.setattr(core, 'uses_lower_release', lambda *_: False)
    pick = PickSpec(get_center=lambda: center.copy(), radius=.025,
                    label=name, actor=actor)
    place = PlaceSpec(pos=np.array([-.25, .15, .821]), label='basket 0',
                      basket_idx=0, drop_from_z=1.001)
    assert task._pick_and_place_checked(pick, place, 'right')
    return task, selected, legacy


@pytest.mark.parametrize('name,model_id', MODELS)
def test_cooperative_selects_equivalent_branches_with_original_contact_settings(
    monkeypatch, name, model_id,
):
    task, selected, legacy = radius_case(CategorizeObjectsContribution, monkeypatch, name, model_id)
    assert len(selected) == 1 and legacy == []
    args, kwargs = selected[0]
    radius_pick, radius_place = args[1:3]
    assert kwargs == dict(object_name=name, model_id=model_id, object_scale=.7)
    assert radius_pick.radius == pytest.approx(.025)
    expected_offset = task.OBJECT_GRASP_Z_OFFSET.get(name, core.get_entry(name).grasp_z_offset or 0.)
    assert radius_pick.grasp_z_offset == expected_offset
    expected_close = task._radius_close_for_live_width(core.get_entry(name).close_value, .025)
    assert radius_pick.close_value == expected_close
    assert radius_place.drop_from_z == 1.001
    assert radius_place.transport_z is None
    if name == '048_stapler':
        assert radius_pick.get_grasp_yaw() == np.pi / 2


@pytest.mark.parametrize('task_class', (CategorizeObjectsInterruption, CategorizeObjectsCopresence))
@pytest.mark.parametrize('name,model_id', MODELS)
def test_existing_sibling_acquisition_path_is_preserved(monkeypatch, task_class, name, model_id):
    _, selected, legacy = radius_case(task_class, monkeypatch, name, model_id)
    assert selected == [] and len(legacy) == 1


def test_unlisted_cooperative_models_keep_the_existing_path(monkeypatch):
    _, selected, legacy = radius_case(CategorizeObjectsContribution, monkeypatch, '100_seal', 2)
    assert selected == [] and len(legacy) == 1


def test_allowlist_does_not_change_existing_yaml_or_physics_defaults():
    task = CategorizeObjectsContribution({})
    assert task.RADIUS_SELECTOR_MODELS == frozenset(MODELS)
    assert '038_milk-box' in task.YAML_ONLY_GRASP_LABELS
    assert '073_rubikscube' in task.YAML_ONLY_GRASP_LABELS
    assert 'categorize_payload_friction_v1' not in task.config
    assert 'categorize_split_screw_v1' not in task.config
