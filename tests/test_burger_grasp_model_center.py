"""Actor-specific model geometry must survive the shared pick interface."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from envs.task_bases.place_burger_fries import PlaceBurgerFries, _BURGER_SPEC
from envs.utils import Pose


def test_center_uses_loaded_actor_model_after_human_rotates_it():
    task = PlaceBurgerFries.__new__(PlaceBurgerFries)
    actor = SimpleNamespace(get_pose=lambda: Pose(
        [1.0, 2.0, 3.0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
    ))
    task.food_actors = [(actor, replace(_BURGER_SPEC, model_id=4))]
    task._get_model_center = lambda name, model_id: np.array([model_id, 0.0, 0.0])
    np.testing.assert_allclose(
        task._get_object_world_center(actor, _BURGER_SPEC.asset_id, 0),
        [1.0, 6.0, 3.0],
    )


def test_two_burger_models_keep_independent_centers():
    task = PlaceBurgerFries.__new__(PlaceBurgerFries)
    first = SimpleNamespace(get_pose=lambda: Pose([0, 0, 0], [1, 0, 0, 0]))
    second = SimpleNamespace(get_pose=lambda: Pose([0, 0, 0], [1, 0, 0, 0]))
    task.food_actors = [(first, replace(_BURGER_SPEC, model_id=4)),
                        (second, replace(_BURGER_SPEC, model_id=5))]
    task._get_model_center = lambda name, model_id: np.array([model_id, 0.0, 0.0])
    assert task._get_object_world_center(first, _BURGER_SPEC.asset_id)[0] == 4
    assert task._get_object_world_center(second, _BURGER_SPEC.asset_id)[0] == 5
