"""Scene-free checks for the oil_bottle_recovery blind ("no intent") expert."""
import inspect
from types import SimpleNamespace

import numpy as np

from envs.tasks.oil_bottle_recovery import OilBottleRecovery


def _task(config=None):
    task = OilBottleRecovery.__new__(OilBottleRecovery)
    task.config = dict(config or {})
    return task


def test_blind_expert_never_waits_on_or_drives_the_human():
    src = inspect.getsource(OilBottleRecovery.play_blind_once)
    for forbidden in ("_wait_for_knock_trigger", "_knock_over_bottle",
                      "_start_frying_animation", "play_animation", "spare(",
                      "frame_ratio", "_KNOCK_SETTLE_STEPS",
                      "_attach_items_to_hands", "_eval_step_avatar",
                      "_stage_to_reference_pre", "_REFERENCE_PRE_QPOS"):
        assert forbidden not in src, forbidden
    # Every re-plan through the snapshot helper is a single snapshot with no
    # sim-time wait.
    assert src.count("_plan_path_with_fk_check(") == src.count("max_snap_retries=1")
    assert src.count("_plan_path_with_fk_check(") == src.count("snap_steps=0")


def test_eval_hook_records_human_start_and_knock_sim_steps():
    task = _task({"eval_knock_timeout_steps": 2})
    task.avatar = SimpleNamespace(play_animation=lambda *a, **k: None)
    task._eval_trigger_step = 2
    task._eval_frying_started = False
    task._eval_knock_done = False
    task._eval_knock_wait_steps = 0
    task._eval_policy_step_count = 0
    task._eval_human_started_sim_step = None
    task._eval_knock_sim_step = None
    task._right_hand_xy = lambda: np.array([5.0, 5.0])  # far: knock by timeout
    task.oil_entity = object()
    task._get_entity_pose = lambda *_: SimpleNamespace(p=np.zeros(3))
    knocks = []
    task._knock_over_bottle = lambda: knocks.append(task._interaction_sim_step)
    for k in range(1, 6):
        task._interaction_sim_step = 5 * k
        task._eval_step_avatar()
    assert task._eval_human_started_sim_step == 10
    assert task._eval_knock_sim_step == 15
    assert knocks == [15]


def test_human_ready_requires_knock_before_robot_and_bottle_at_rest():
    task = _task({"eval_mode": True})
    task._eval_trigger_step = 50
    task._eval_human_started_sim_step = 250
    task._eval_knock_sim_step = 300
    task._blind_robot_interaction_sim_step = 900
    task._blind_bottle_state_at_robot_step = {"pos": [0, 0, 0], "tilt_deg": 88.0,
                                              "speed_m_s": 0.001}
    m = task._blind_intent_metrics()
    assert m["human_ready_before_robot"] is True
    assert m["trigger_step"] == 50 and m["robot_interaction_sim_step"] == 900

    task._blind_bottle_state_at_robot_step = {"pos": [0, 0, 0], "tilt_deg": 2.0,
                                              "speed_m_s": 0.0}
    assert task._blind_intent_metrics()["human_ready_before_robot"] is False
    task._blind_bottle_state_at_robot_step = {"pos": [0, 0, 0], "tilt_deg": 70.0,
                                              "speed_m_s": 0.4}
    assert task._blind_intent_metrics()["human_ready_before_robot"] is False
    task._eval_knock_sim_step = None
    task._blind_bottle_state_at_robot_step = {"pos": [0, 0, 0], "tilt_deg": 88.0,
                                              "speed_m_s": 0.0}
    assert task._blind_intent_metrics()["human_ready_before_robot"] is False
