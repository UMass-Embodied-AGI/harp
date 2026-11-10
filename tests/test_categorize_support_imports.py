"""Private family organization must leave public task identities unchanged."""

import importlib
from pathlib import Path

import pytest


HELPERS = (
    "carry", "cooperative_lift", "cube", "milk_support", "motion", "profile",
    "radius", "release", "seal", "split", "stack", "transport",
)


@pytest.mark.parametrize("name", HELPERS)
def test_private_helpers_have_one_canonical_family_module(name):
    module_name = f"envs.task_bases.categorize_support.{name}"
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
    path = Path(module.__file__)
    assert path.parent.name == "categorize_support"
    assert not (path.parent.parent / f"categorize_{name}.py").exists()


@pytest.mark.parametrize(("module_name", "class_name"), [
    ("envs.task_bases.categorize", "CategorizeTaskCore"),
    ("envs.tasks.categorize_objects_contribution", "CategorizeObjectsContribution"),
    ("envs.tasks.categorize_objects_interruption", "CategorizeObjectsInterruption"),
    ("envs.tasks.categorize_objects_copresence", "CategorizeObjectsCopresence"),
])
def test_public_core_and_scenario_class_paths_are_unchanged(module_name, class_name):
    cls = getattr(importlib.import_module(module_name), class_name)
    assert cls.__module__ == module_name


def test_core_and_legacy_cube_names_share_canonical_helper_objects():
    from envs.task_bases import categorize
    from envs.task_bases.categorize_support import cube, motion, profile, radius

    assert categorize.CategorizeMotionMixin is motion.CategorizeMotionMixin
    assert categorize.CategorizeTaskProfile is profile.CategorizeTaskProfile
    assert categorize.pick_and_place_radius is radius.pick_and_place_radius
    assert cube.cube_candidates is radius.radius_candidates
    assert cube.pick_and_place_cube is radius.pick_and_place_radius
