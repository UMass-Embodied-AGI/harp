"""Scene-free contracts for the deliver_to_human_easy blind (no-intent) expert."""

from types import SimpleNamespace

import numpy as np

from envs.tasks.deliver_to_human import DeliverToHuman


class _Avatar:
    def __init__(self, spare=True):
        self.played = []
        self._spare = spare

    def play_animation(self, name):
        self.played.append(name)

    def spare(self):
        return self._spare


def _blind_task(**cfg):
    config = {"eval_mode": True, "eval_avatar_pre_policy": False}
    config.update(cfg)
    task = DeliverToHuman(config)
    task.avatar = _Avatar()
    return task


def test_pre_policy_path_still_presents_at_reset():
    task = DeliverToHuman({"eval_mode": True, "eval_avatar_pre_policy": True})
    task.avatar = _Avatar()
    task._eval_at_reset()
    assert task.avatar.played == ["take_from_human"]
    assert task._eval_trigger_step is None
    task._eval_at_step(50)
    assert task.avatar.played == ["take_from_human"]


def test_default_eval_human_presents_during_the_policy_from_a_private_stream():
    lo, hi = DeliverToHuman.EVAL_TRIGGER_STEP_MIN, DeliverToHuman.EVAL_TRIGGER_STEP_MAX
    np.random.seed(0)
    state = np.random.get_state()
    seen = set()
    for seed in range(30):
        task = DeliverToHuman({"eval_mode": True})
        task.avatar = _Avatar()
        task._episode_seed = seed
        task._eval_at_reset()
        assert task.avatar.played == []
        assert lo <= task._eval_trigger_step <= hi
        seen.add(task._eval_trigger_step)
    assert len(seen) > 10
    assert np.array_equal(np.random.get_state()[1], state[1])


def test_blind_window_is_drawn_from_config_and_starts_on_policy_step():
    np.random.seed(3)
    task = _blind_task(eval_trigger_step_min=10, eval_trigger_step_max=12)
    task._eval_at_reset()
    trigger = task._eval_trigger_step
    assert 10 <= trigger <= 12
    assert task.avatar.played == []
    for step in range(1, trigger):
        task._eval_at_step(step)
    assert task.avatar.played == []
    task._blind_sim_step = 5 * trigger
    task._eval_at_step(trigger)
    assert task.avatar.played == ["take_from_human"]
    assert task._blind_human_started_sim_step == 5 * trigger
    task._eval_at_step(trigger + 1)
    assert task.avatar.played == ["take_from_human"]


def test_class_window_defaults_are_used_without_config():
    task = _blind_task()
    task._eval_at_reset()
    assert (
        DeliverToHuman.EVAL_TRIGGER_STEP_MIN
        <= task._eval_trigger_step
        <= DeliverToHuman.EVAL_TRIGGER_STEP_MAX
    )


def test_zero_trigger_presents_at_reset():
    task = _blind_task(eval_trigger_step_min=0, eval_trigger_step_max=0)
    task._eval_at_reset()
    assert task.avatar.played == ["take_from_human"]
    assert task._blind_human_started_sim_step == 0


def test_play_blind_once_never_plays_or_waits_on_the_avatar():
    task = _blind_task(grasp_method="annotated")
    task._eval_at_reset()
    task.avatar.spare = lambda: (_ for _ in ()).throw(AssertionError("waited"))
    steps = []
    task.step_sim = lambda: steps.append(1)
    task._play_once_annotated = lambda: True
    assert task.play_blind_once() is True
    assert len(steps) == 50
    assert task.avatar.played == []


def test_robot_interaction_marker_records_human_readiness():
    task = _blind_task(eval_trigger_step_min=5, eval_trigger_step_max=5)
    task._eval_at_reset()
    task._blind_sim_step = 400
    task._mark_robot_interaction()
    assert task._blind_robot_interaction_sim_step == 400
    assert task._blind_human_ready_before_robot is False  # not started

    task._eval_at_step(5)
    task.avatar._spare = False
    task._mark_robot_interaction()
    assert task._blind_human_ready_before_robot is False  # mid-animation
    task.avatar._spare = True
    task._mark_robot_interaction()
    assert task._blind_human_ready_before_robot is True


def test_blind_primitive_reads_palm_when_planning_delivery():
    task = _blind_task(grasp_method="primitive")
    task._eval_at_reset()
    task.obj_name = "073_rubikscube"
    task.obj_model_id = 1
    task.TABLE_TOP_Z = 0.74
    task.obj_actor = SimpleNamespace(
        get_pose=lambda: SimpleNamespace(p=np.array([0.0, 0.0, 0.76])),
    )
    task._get_object_cross_section_radius = lambda *a: 0.03
    task._get_object_world_center = lambda *a: np.zeros(3)
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        get_ee_pose=lambda: np.zeros(7), get_arm_qpos=lambda: np.zeros(7)))
    palms = iter([np.array([0.1, 0.5, 0.95]), np.array([0.2, 0.6, 1.10])])
    task._deliver_pos = lambda arm_tag: next(palms)
    calls = []

    def pick_and_place(pick, place, arm_tag):
        calls.append(("pnp", place.stop_after_lift, place.release))
        return True

    task.pick_and_place = pick_and_place
    task._move_screw = lambda pos, arm_tag: calls.append(("screw", tuple(pos)))
    task.step_sim = lambda: None
    task._object_still_held = lambda: True
    task._deliver_pose_has_ik = lambda pos, arm_tag, seed=None: True
    # Eval mode: from above the delivery point straight to the handover
    # (release over the live palm).
    task._handover_tcp_target = lambda arm_tag, standoff=None, dz=None: np.array([0.2, 0.8, 1.2])
    task._handover_release = lambda arm_tag, top_down, target=None: calls.append(
        ("handover", top_down)) or True
    assert task._play_blind_primitive() is True
    # The above-delivery pose is raised to the handover height (1.2 m).
    assert calls == [
        ("pnp", True, False),
        ("screw", (0.2, 0.6, 1.2)),
        ("handover", True),
    ]
    assert task._blind_robot_interaction_sim_step is not None


def test_blind_primitive_fails_without_moving_when_palm_has_no_ik():
    task = _blind_task(grasp_method="primitive")
    task._eval_at_reset()
    task.obj_name = "073_rubikscube"
    task.obj_model_id = 1
    task.TABLE_TOP_Z = 0.74
    task.plan_success = True
    task.obj_actor = SimpleNamespace(
        get_pose=lambda: SimpleNamespace(p=np.array([0.0, 0.0, 0.76])),
    )
    task._get_object_cross_section_radius = lambda *a: 0.03
    task._get_object_world_center = lambda *a: np.zeros(3)
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        get_arm_qpos=lambda: np.zeros(7)))
    task._deliver_pos = lambda arm_tag: np.array([-0.2, 1.3, 0.54])
    task._handover_tcp_target = lambda arm_tag, standoff=None, dz=None: np.array([-0.2, 1.5, 0.66])
    task.pick_and_place = lambda pick, place, arm_tag: True
    moves = []
    task._move_screw = lambda pos, arm_tag: moves.append(pos)
    task._deliver_pose_has_ik = lambda pos, arm_tag, seed=None: False
    assert task._play_blind_primitive() is False
    assert moves == []
    assert task.plan_success is False
    assert task._stage_failed == "deliver_ik"
