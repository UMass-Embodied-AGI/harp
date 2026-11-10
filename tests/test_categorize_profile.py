"""Compatibility of shared categorize settings and scenario overrides."""

import pytest

from envs.base_task import BaseTask
from envs.manipulation import TopDownPickPlaceMixin
from envs.task_bases.categorize import CategorizeTaskCore
from envs.task_bases.categorize_support.motion import CategorizeMotionMixin
from envs.task_bases.categorize_support.profile import CategorizeTaskProfile
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence


def test_profile_keeps_existing_method_resolution_order():
    class OriginalBases(CategorizeMotionMixin, TopDownPickPlaceMixin, BaseTask):
        pass

    inherited_mro = tuple(
        base for base in CategorizeTaskCore.__mro__[1:]
        if base is not CategorizeTaskProfile
    )
    assert inherited_mro == OriginalBases.__mro__[1:]


@pytest.mark.parametrize(
    "task_class", (CategorizeObjectsContribution, CategorizeObjectsInterruption, CategorizeObjectsCopresence)
)
def test_core_settings_remain_patchable_without_overwriting_scenario_defaults(
    monkeypatch, task_class
):
    task = task_class.__new__(task_class)
    scenario_layout = task._CATEGORIZE_COLUMN_LAYOUT
    marker = object()
    monkeypatch.setattr(CategorizeTaskCore, "USE_RADIUS_GRASP", marker)
    monkeypatch.setattr(CategorizeTaskProfile, "_CATEGORIZE_COLUMN_LAYOUT", marker)

    assert task.USE_RADIUS_GRASP is marker
    if "_CATEGORIZE_COLUMN_LAYOUT" in task_class.__dict__:
        assert task._CATEGORIZE_COLUMN_LAYOUT is scenario_layout
    else:
        assert task._CATEGORIZE_COLUMN_LAYOUT is marker


def test_mutable_defaults_remain_shared_through_the_public_core(monkeypatch):
    marker = object()
    monkeypatch.setitem(CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE, "test-object", marker)

    for task_class in (
        CategorizeTaskProfile, CategorizeObjectsContribution,
        CategorizeObjectsInterruption, CategorizeObjectsCopresence,
    ):
        assert task_class.SPAWN_FRICTION_OVERRIDE is CategorizeTaskCore.SPAWN_FRICTION_OVERRIDE
        assert task_class.SPAWN_FRICTION_OVERRIDE["test-object"] is marker
