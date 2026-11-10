"""Model-local material experiments must leave default scene physics intact."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import categorize
from envs.task_bases.categorize import CategorizeTaskCore
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence


def spawn(monkeypatch, overrides, name="100_seal", model=1, catalog_friction=4.0):
    calls = []
    actor = object()
    monkeypatch.setattr(categorize, "get_entry", lambda _: SimpleNamespace(friction=catalog_friction))

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return actor

    monkeypatch.setattr(categorize, "load_object", load)
    scene = object()
    task = SimpleNamespace(scene=scene, SPAWN_FRICTION_OVERRIDE=overrides)
    result = CategorizeTaskCore._spawn_categorize_object(task, name, model, .1, .2, .765, [1, 0, 0, 0])
    assert result is actor and len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is scene and args[2] == name
    np.testing.assert_allclose(args[1].to_pose7(), [.1, .2, .765, 1, 0, 0, 0])
    assert kwargs == dict(model_id=model, convex=True, is_static=False, friction=kwargs["friction"])
    return kwargs["friction"]


@pytest.mark.parametrize("model", [0, 1, 2])
def test_defaults_preserve_catalog_seal_friction(monkeypatch, model):
    assert spawn(monkeypatch, CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE, model=model) == 4.0
    assert ("100_seal", 1) not in CategorizeObjectsCopresence.SPAWN_FRICTION_OVERRIDE
    assert CategorizeObjectsCopresence.SEAL_MEASURED_GRASP is False


@pytest.mark.parametrize("model,expected", [(0, 4.0), (1, 1.0), (2, 4.0)])
def test_opt_in_is_exactly_model_scoped(monkeypatch, model, expected):
    before = dict(CategorizeObjectsCopresence.SPAWN_FRICTION_OVERRIDE)
    overrides = {**before, ("100_seal", 1): 1.0}
    assert spawn(monkeypatch, overrides, model=model) == expected
    assert CategorizeObjectsCopresence.SPAWN_FRICTION_OVERRIDE == before
    assert CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE == before


@pytest.mark.parametrize("model,expected", [(0, 2.5), (1, 1.0), (2, 2.5)])
def test_model_setting_precedes_name_setting(monkeypatch, model, expected):
    assert spawn(monkeypatch, {"100_seal": 2.5, ("100_seal", 1): 1.0}, model=model) == expected


def test_existing_wood_setting_and_other_catalog_values_are_preserved(monkeypatch):
    overrides = {**CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE, ("100_seal", 1): 1.0}
    assert spawn(monkeypatch, overrides, name="086_woodenblock", catalog_friction=5.0) == 2.5
    assert spawn(monkeypatch, overrides, name="035_apple", catalog_friction=4.0) == 4.0
    assert spawn(monkeypatch, overrides, name="071_can", catalog_friction=9.0) == 5.0


def test_unknown_catalog_entry_keeps_legacy_fallback(monkeypatch):
    assert spawn(monkeypatch, {}, catalog_friction=None) == 2.5


def test_procedural_cube_path_does_not_gain_material_override(monkeypatch):
    entity, scene = object(), object()
    monkeypatch.setattr(categorize, "get_entry", lambda _: SimpleNamespace(prim_half=.02))
    calls = []
    monkeypatch.setattr(categorize, "create_primitive", lambda *a, **kw: calls.append((a, kw)) or entity)
    task = SimpleNamespace(scene=scene, SPAWN_FRICTION_OVERRIDE={"cube": 1.0, ("cube", 0): .5})
    result = CategorizeTaskCore._spawn_categorize_object(task, "cube", 0, .1, .2, .765, [1, 0, 0, 0])
    assert result.entity is entity and len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is scene and args[1] == "box"
    np.testing.assert_allclose(args[2].p, [.1, .2, .785])
    assert "friction" not in kwargs
