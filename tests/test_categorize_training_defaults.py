"""The successful training-pilot options are defaults only for two tasks."""
from copy import deepcopy

import pytest

from envs.task_bases.categorize import CategorizeTaskCore
from envs.task_bases.categorize_support.split import enabled
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence


TASKS = (CategorizeObjectsInterruption, CategorizeObjectsCopresence)
FLAGS = ("categorize_payload_friction_v1", "categorize_split_screw_v1")


@pytest.mark.parametrize("task_class", TASKS)
@pytest.mark.parametrize("config", [None, {}])
def test_task_defaults_enable_both_tested_options(task_class, config):
    task = task_class(config)
    assert all(task.config[flag] is True for flag in FLAGS)
    assert enabled(task)


@pytest.mark.parametrize("task_class", TASKS)
@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [False, None, 0, "true"])
def test_explicit_values_are_preserved_without_enabling_nonliteral_values(task_class, flag, value):
    task = task_class({flag: value})
    assert task.config[flag] is value
    assert task.config[next(key for key in FLAGS if key != flag)] is True
    if flag == "categorize_split_screw_v1":
        assert not enabled(task)


@pytest.mark.parametrize("task_class", TASKS)
def test_defaults_equal_the_explicit_pilot_configuration(task_class):
    config = dict(randomize_avatar=True, track_avatar_collision=True, eval_mode=False,
                  randomize_table=False, renderer="nyx", avatar={"frame_ratio": 2.0})
    before = deepcopy(config)
    default_task = task_class(config)
    explicit_task = task_class(dict(config, **{flag: True for flag in FLAGS}))
    assert default_task.config == explicit_task.config
    assert config == before and all(flag not in config for flag in FLAGS)


@pytest.mark.parametrize("task_class", (CategorizeTaskCore, CategorizeObjectsContribution))
def test_shared_core_and_cooperative_defaults_are_unchanged(task_class):
    task = task_class({})
    assert all(flag not in task.config for flag in FLAGS)
    assert not enabled(task)


@pytest.mark.parametrize("task_class", TASKS)
def test_explicit_off_keeps_both_disabled(task_class):
    task = task_class({flag: False for flag in FLAGS})
    assert all(task.config[flag] is False for flag in FLAGS)
    assert not enabled(task)
