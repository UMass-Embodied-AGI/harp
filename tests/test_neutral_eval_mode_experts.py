"""Neutral rows: both experts face the policies' eval-mode human."""

from types import SimpleNamespace

from envs.interaction.neutral_table_work import NeutralTableWorkController
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from scripts.eval_expert_set import build_config, is_neutral_eval_row


ROW = {"task": "stack_three_bowls_copresence", "category": "copresence", "seed": 9000004,
       "action_substeps": 25, "config_overrides": {"eval_mode": True, "renderer": "nyx"}}


def test_neutral_rows_run_both_experts_in_eval_mode():
    assert is_neutral_eval_row(ROW)
    for policy in ("expert_full_state", "expert"):
        cfg = build_config(ROW, policy)
        assert cfg["eval_mode"] is True
        assert cfg["neutral_eval_human_from_ticks"] is True
    assert build_config(ROW, "expert")["random_object"] is False
    pinned = dict(ROW, config_overrides={"random_object": True})
    assert build_config(pinned, "expert")["random_object"] is True
    other = dict(ROW, task="place_burger_and_fries_contribution", category="contribution")
    assert not is_neutral_eval_row(other)
    assert "neutral_eval_human_from_ticks" not in build_config(other, "expert_full_state")


def _stub(**config):
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = dict(config)
    task.avatar = SimpleNamespace()
    task._neutral_avatar_started = False
    task._neutral_avatar_job = {"kind": "sponge"}
    return task


def test_scripted_start_is_deferred_to_the_eval_tick():
    task = _stub(eval_mode=True, neutral_eval_human_from_ticks=True)
    assert task._start_neutral_avatar_table_work(add_start_delay=True) is False
    assert task._neutral_avatar_started is False


def test_neutral_eval_driver_starts_the_human_at_the_trigger_step():
    from scripts.eval_expert_set import run_neutral_eval

    started = []

    class Task:
        avatar = object()
        config = {"eval_mode": True}
        _neutral_eval_policy_step_count = 0
        _neutral_eval_trigger_step = 2
        _neutral_avatar_started = False
        _neutral_eval_avatar_failed = False

        def _neutral_controller(self):
            return NeutralTableWorkController()

        def _start_neutral_avatar_table_work(self, add_start_delay):
            started.append((add_start_delay, self._neutral_eval_policy_step_count, self.sim))
            self._neutral_avatar_started = True
            return True

        sim = 0

        def step_sim(self):
            self.sim += 1

        def play_once(self):
            for _ in range(100):
                self.step_sim()
            return True

    task = Task()
    orig = task.step_sim
    assert run_neutral_eval(task, {"action_substeps": 25}) is True
    # tick 1 before any sim step, tick 2 after 25 sim steps -> trigger 2 fires there
    assert started == [(False, 2, 25)]
    assert task._neutral_eval_policy_step_count == 5
    assert task.step_sim == orig


def test_eval_trigger_draw_leaves_the_global_stream_untouched():
    import numpy as np
    from envs.interaction.neutral_table_work import draw_neutral_eval_trigger_step

    task = SimpleNamespace(config={}, _neutral_avatar_reset_seed=9000007)
    np.random.seed(123)
    expected = np.random.RandomState(123).randint(0, 1000, size=3)
    steps = {draw_neutral_eval_trigger_step(task, 0, 5) for _ in range(3)}
    assert len(steps) == 1 and 0 <= steps.pop() <= 5        # deterministic per seed
    assert list(np.random.randint(0, 1000, size=3)) == list(expected)
    seen = {draw_neutral_eval_trigger_step(SimpleNamespace(config={}, _neutral_avatar_reset_seed=s), 0, 5)
            for s in range(9000000, 9000200)}
    assert seen == set(range(6))
    legacy = SimpleNamespace(config={"neutral_eval_trigger_rng": "global"}, _neutral_avatar_reset_seed=1)
    np.random.seed(5)
    ref = np.random.RandomState(5).randint(0, 6)
    assert draw_neutral_eval_trigger_step(legacy, 0, 5) == ref


def _patching_task(trigger):
    """The human's start wraps the CURRENT step_sim (like
    _patch_neutral_avatar_step_sim); count how many sim steps it sees."""
    from scripts.eval_expert_set import run_neutral_eval  # noqa: F401

    class Task:
        avatar = object()
        config = {"eval_mode": True}
        _neutral_eval_policy_step_count = 0
        _neutral_avatar_started = False
        _neutral_eval_avatar_failed = False
        sim = 0
        patched_steps = 0

        def __init__(self):
            self._neutral_eval_trigger_step = trigger

        def _neutral_controller(self):
            return NeutralTableWorkController()

        def _start_neutral_avatar_table_work(self, add_start_delay):
            inner = self.step_sim

            def patched(*a, **k):
                self.patched_steps += 1
                return inner(*a, **k)
            self._neutral_original_step_sim = inner
            self.step_sim = patched
            self._neutral_avatar_started = True
            return True

        def step_sim(self):
            self.sim += 1

        def play_once(self):
            for _ in range(100):
                self.step_sim()
            return True

    return Task()


def test_human_patch_survives_when_the_human_starts_on_the_first_tick():
    from scripts.eval_expert_set import run_neutral_eval

    for trigger, expected in ((0, 100), (1, 100), (2, 75), (3, 50)):
        task = _patching_task(trigger)
        assert run_neutral_eval(task, {"action_substeps": 25}) is True
        assert task.sim == 100
        # every sim step after the start goes through the human's patch
        assert task.patched_steps == expected, (trigger, task.patched_steps)
