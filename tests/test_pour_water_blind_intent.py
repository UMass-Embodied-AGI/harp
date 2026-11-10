"""pour_water blind ("no intent") expert: no human cue, no waiting, live targets."""

import inspect
from types import SimpleNamespace

import numpy as np

from envs.interaction.eval_mode import IntentEvalModeController
from envs.tasks.pour_water import PourWater
from scripts.eval_expert_set import (
    BLIND_INTENT_TASKS,
    configure_parent_rollout_eval,
    run_with_eval_step_driver,
)


class _FakeAvatar:
    """Animation that takes ``n`` avatar steps to finish."""

    def __init__(self, n=3):
        self.n = n
        self.left = 0
        self.played = []

    def reset(self, *a, **k):
        self.left = 0

    def play_animation(self, name, **kw):
        self.played.append(name)
        self.left = self.n

    def play_animation_hold_reverse(self, name, **kw):
        self.played.append(name + "_hold_rev")
        self.left = self.n

    def step(self):
        self.left = max(0, self.left - 1)

    def spare(self):
        return self.left == 0


def _fake_task(**config):
    task = PourWater.__new__(PourWater)
    task.config = {"eval_mode": True, "eval_avatar_pre_policy": False, **config}
    task.avatar = _FakeAvatar()
    task.cup_actor = SimpleNamespace(entity=object())
    task._avatar_shift_x = 0.0
    task._interaction_sim_step = 0
    task._interaction_policy_step = 0
    task._interaction_controller = None
    return task


def test_harness_configures_blind_rollout():
    assert "pour_water" in BLIND_INTENT_TASKS
    cfg = {}
    configure_parent_rollout_eval(cfg, "pour_water", "play_blind_once")
    assert cfg == {"eval_avatar_pre_policy": False, "eval_force_policy_ticks": True,
                   "eval_parent_rollout_tick_interval": 25}


def test_play_blind_once_never_reads_or_drives_the_human():
    src = inspect.getsource(PourWater.play_blind_once)
    for banned in ("_calibrate_avatar_pos", "play_animation", "spare()",
                   "_start_eval_drink", "_eval_trigger_step", "while "):
        assert banned not in src, banned
    # Live targets: mug at plan time, cup inside the pose functions.
    assert "_get_mug_logical_pose()" in src
    assert "_refresh_avatar_planner_obstacles" in src
    assert "_get_object_pose(" in inspect.getsource(PourWater._cup_mug_geometry)


def test_trigger_window_is_seeded_and_bounded():
    steps = set()
    for seed in range(50):
        np.random.seed(seed)
        task = _fake_task()
        task._eval_at_reset()
        steps.add(task._eval_trigger_step)
    assert min(steps) >= PourWater.EVAL_TRIGGER_STEP_MIN
    assert max(steps) <= PourWater.EVAL_TRIGGER_STEP_MAX
    assert len(steps) > 10
    task = _fake_task(eval_trigger_step_min=4, eval_trigger_step_max=4)
    task._eval_at_reset()
    assert task._eval_trigger_step == 4


def test_policy_tick_driver_runs_drinking_and_records_metrics():
    task = _fake_task(eval_trigger_step_min=2, eval_trigger_step_max=2)
    assert isinstance(task.interaction_controller, IntentEvalModeController)
    task.interaction_controller.after_base_reset_observation(task, observation=None)

    def step_sim():  # stands in for BaseTask.step_sim
        task.avatar.step()
        task._interaction_sim_step += 1

    task.step_sim = step_sim
    readiness = []

    def robot_script():
        for _ in range(60):
            task.step_sim()
            readiness.append(task._eval_human_drinking_done())
        return True

    assert run_with_eval_step_driver(
        task, robot_script,
        {"eval_force_policy_ticks": True, "eval_parent_rollout_tick_interval": 5},
    )
    assert task.avatar.played == ["drink_water_1", "drink_water_2_hold_rev"]
    # Policy step 2 = sim step 10 at the driver's 5 sim steps per tick.
    assert task._eval_human_started_sim_step == 10
    assert task._eval_human_done_sim_step is not None
    assert task._eval_human_done_sim_step > task._eval_human_started_sim_step
    # Not ready before drinking finished; ready afterwards.
    assert readiness[0] is False and readiness[-1] is True
