"""Blind ("no intent") expert contract for open_microwave."""

import dis
from types import SimpleNamespace

import numpy as np

from envs.interaction.eval_mode import TaskDrivenEvalTickController
from envs.tasks.open_microwave import OpenMicrowave


def _attr_names(fn):
    return {ins.argval for ins in dis.get_instructions(fn)
            if ins.opname in {"LOAD_ATTR", "LOAD_METHOD", "LOAD_GLOBAL"}}


def test_blind_rollout_never_plays_or_waits_on_the_human():
    names = _attr_names(OpenMicrowave.play_blind_once)
    for forbidden in ("play_animation", "spare", "_attach_plate_to_avatar",
                      "_avatar_plate_carried_ok", "_eval_trigger_step"):
        assert forbidden not in names, forbidden
    # Avatar + plate are put in the planner cloud at plan time.
    assert "_blind_refresh_planner_obstacles" in names


def test_full_state_play_once_still_waits_for_the_approach():
    names = _attr_names(OpenMicrowave.play_once)
    assert "play_animation" in names and "spare" in names
    assert "_blind_refresh_planner_obstacles" not in names


class _Avatar:
    def __init__(self):
        self.played = []
        self.frame_ratio = None
        self._spare = True

    def play_animation(self, name):
        self.played.append(name)
        self._spare = False

    def spare(self):
        return self._spare


def _fake_task(trigger_step):
    task = object.__new__(OpenMicrowave)
    task.avatar = _Avatar()
    task._plate_attached = True
    task._eval_trigger_step = trigger_step
    task._eval_avatar_started = False
    task._eval_avatar_failed = False
    task._policy_step_count = 0
    task._eval_human_started_sim_step = None
    task._eval_human_finished_sim_step = None
    task._interaction_sim_step = 0
    task._blind_sim_step0 = 0
    task._avatar_plate_carried_ok = lambda: True
    return task


def test_human_starts_at_trigger_policy_step_under_harness_driver():
    task = _fake_task(trigger_step=7)
    ctrl = TaskDrivenEvalTickController()
    for policy_step in range(1, 12):
        task._interaction_sim_step = 5 * policy_step  # driver: 1 tick / 5 sim
        ctrl.on_scripted_rollout_tick(task)
        if policy_step < 7:
            assert task.avatar.played == []
    assert task.avatar.played == ["microwave"]
    assert task._eval_human_started_sim_step == 35
    assert task._eval_human_finished_sim_step is None
    task.avatar._spare = True
    task._interaction_sim_step = 300
    ctrl.on_scripted_rollout_tick(task)
    assert task._eval_human_finished_sim_step == 300
    assert task.avatar.played == ["microwave"]


def test_trigger_window_is_the_eval_window():
    assert (OpenMicrowave.EVAL_TRIGGER_STEP_MIN,
            OpenMicrowave.EVAL_TRIGGER_STEP_MAX) == (20, 80)


def test_blind_intent_metric_helper_reads_spare_only_for_metrics():
    task = _fake_task(trigger_step=1)
    assert task._blind_metric_human_ready() is False
    task._eval_avatar_started = True
    assert task._blind_metric_human_ready() is True
    task.avatar._spare = False
    assert task._blind_metric_human_ready() is False


def test_planner_cloud_includes_avatar_capsules_and_plate():
    task = _fake_task(trigger_step=1)
    captured = {}
    planner = SimpleNamespace(
        update_obstacles=lambda pts, resolution: captured.update(pts=pts))
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(planner=planner))
    task.avatar_collider = SimpleNamespace(
        current_capsules=lambda: [("torso", (1.0, 0.0, 0.5), (1.0, 0.0, 1.2), 0.1)],
        current_boxes=lambda: [],
    )
    task.avatar_plate = SimpleNamespace(entity=SimpleNamespace(
        get_pos=lambda: np.array([0.0, 0.0, 1.0]),
        get_quat=lambda: np.array([0.70710678, 0.70710678, 0.0, 0.0]),  # local y -> world z
    ), config={"extents": [0.2, 0.02, 0.2], "scale": [1.0, 1.0, 1.0]})
    task._blind_refresh_planner_obstacles("right")
    pts = captured["pts"]
    assert pts.shape[1] == 3
    near_plate = pts[np.abs(pts[:, 2] - 1.0) < 1e-6]
    assert len(near_plate) > 10
    assert np.linalg.norm(near_plate[:, :2], axis=1).max() <= 0.1 + 1e-6
    assert (pts[:, 0] > 0.8).any()  # torso capsule samples
