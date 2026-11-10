"""Scene-free contracts for the take_from_human_easy blind ("no intent") expert."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import envs.tasks.take_from_human_easy as take_module
from envs.tasks.take_from_human_easy import TakeFromHumanEasy
from envs.base_task import TargetSpec
from scripts.eval_expert_set import configure_parent_rollout_eval


TASK_NAME = "take_from_human_easy"


def test_blind_rollout_disables_pre_policy_offer():
    cfg = {}
    configure_parent_rollout_eval(cfg, TASK_NAME, "play_blind_once")
    assert cfg["eval_avatar_pre_policy"] is False
    assert cfg["eval_force_policy_ticks"] is True


class _Skin:
    def get_global_translation(self, name):
        return np.array([[0.0, 0.1 if name.endswith("Middle1") else 0.0, 1.0]])


class _HookAvatar:
    def __init__(self):
        self.events = []
        self.spare_value = True
        self.robot = SimpleNamespace(
            skin=_Skin(),
            _get_hand_frame=lambda _hand: (np.zeros(3), np.eye(3)),
            detach_object=lambda hand: self.events.append(("detach", hand)),
        )

    def attach_object_to_hand(self, _entity, hand_id):
        self.events.append(("attach", hand_id))

    def play_animation(self, name):
        self.events.append(("play", name))
        self.spare_value = False

    def spare(self):
        return self.spare_value

    def play_animation_reverse(self, name):
        self.events.append(("reverse", name))
        self.spare_value = False


def _reset_task(config, seed=3):
    task = TakeFromHumanEasy(dict(config))
    task._episode_seed = seed
    task.avatar = _HookAvatar()
    task.apple_actor = SimpleNamespace(entity=SimpleNamespace(set_pos=lambda _p: None))
    task._boost_arm_pd = lambda _arm: None
    task._interaction_sim_step = 0
    task._eval_place_human_in_reach = lambda: task.avatar.events.append(("place",))
    task._eval_at_reset()
    return task


def test_pre_policy_default_still_plays_offer_at_reset():
    task = _reset_task({"eval_mode": True})
    assert task._eval_trigger_step is None
    assert task.avatar.events == [("attach", 1), ("play", "deliver_to_human")]


def test_delayed_offer_starts_at_seeded_trigger_and_records_timing():
    config = {"eval_mode": True, "eval_avatar_pre_policy": False,
              "eval_trigger_step_min": 4, "eval_trigger_step_max": 9}
    task = _reset_task(config, seed=11)
    trigger = task._eval_trigger_step
    assert 4 <= trigger <= 9
    assert _reset_task(config, seed=11)._eval_trigger_step == trigger
    assert task.avatar.events == [("place",), ("attach", 1)]
    # Keep the handoff check far from the object so only the trigger acts.
    task._eval_apple_attached_to_avatar = False
    for step in range(1, trigger):
        task._interaction_sim_step = 5 * step
        task._eval_at_step(step)
    assert task.avatar.events == [("place",), ("attach", 1)]
    task._interaction_sim_step = 5 * trigger
    task._eval_at_step(trigger)
    assert task.avatar.events[-1] == ("play", "deliver_to_human")
    assert task._eval_human_started_sim_step == 5 * trigger
    task._interaction_sim_step += 5
    task._eval_at_step(trigger + 1)
    assert task._eval_human_finished_sim_step is None
    task.avatar.spare_value = True
    task._interaction_sim_step += 5
    task._eval_at_step(trigger + 2)
    assert task._eval_human_finished_sim_step == 5 * trigger + 10
    assert task.avatar.events.count(("play", "deliver_to_human")) == 1
    # After the handoff release the human withdraws the arm, once.
    task._eval_handoff_debug = {"step_idx": trigger + 3}
    task._interaction_sim_step += 5
    task._eval_at_step(trigger + 3)
    assert task.avatar.events[-1] == ("reverse", "deliver_to_human")
    assert task._eval_human_retract_sim_step == 5 * trigger + 15
    task.avatar.spare_value = True
    task._eval_at_step(trigger + 4)
    assert task.avatar.events.count(("reverse", "deliver_to_human")) == 1


def test_blind_handoff_release_works_with_policy_tick_cadence():
    """Under the 5-sim-step driver the closed-gripper release still fires."""
    task = TakeFromHumanEasy({"eval_mode": True, "eval_avatar_pre_policy": False})
    arm = SimpleNamespace(gripper_val=0.45, get_ee_pose=lambda: np.zeros(7))
    task.robot = SimpleNamespace(get_arm=lambda _n: arm)
    task.avatar = _HookAvatar()
    task.apple_actor = object()
    task.obj_name = "035_apple"
    task._get_object_center_world = lambda _a: np.array([0.0, 0.0, 0.03])
    task._eval_trigger_step = None
    task._eval_apple_attached_to_avatar = True
    for step in range(1, 5):
        task._eval_at_step(step)
    assert task._eval_apple_attached_to_avatar is False
    assert task.avatar.events == [("detach", 1)]


class _BlindAvatar:
    """Avatar whose behaviour the blind script must never drive or wait on."""

    def __init__(self):
        self.spare_calls = 0
        self.robot = SimpleNamespace(
            detach_object=lambda _h: pytest.fail("robot script detached"),
        )

    def spare(self):
        self.spare_calls += 1
        return True

    def play_animation(self, *_a, **_k):
        pytest.fail("robot script played the avatar")

    def play_animation_reverse(self, *_a, **_k):
        pytest.fail("robot script played the avatar")

    def attach_object_to_hand(self, *_a, **_k):
        pytest.fail("robot script touched the avatar hand")

    def reset(self, *_a, **_k):
        pytest.fail("robot script moved the avatar")


class _Entity:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self._pos.copy()

    def get_qpos(self):
        return np.zeros(9)

    def control_dofs_position(self, _q):
        pass


def _blind_task(monkeypatch, human_started):
    events = []
    task = TakeFromHumanEasy({"eval_mode": True, "eval_avatar_pre_policy": False})
    arm = SimpleNamespace(
        tcp_offset=None,
        origin_pose=SimpleNamespace(p=np.array([0.0, -0.3, 0.75])),
        gripper_val=0.45,
        entity=_Entity(np.zeros(3)),
        get_ee_pose=lambda: np.array([0.25, -0.40, 0.985, 1, 0, 0, 0]),
    )
    task.robot = SimpleNamespace(get_arm=lambda _n: arm,
                                 set_gripper=lambda *_a: None)
    task.avatar = _BlindAvatar()
    apple = _Entity((0.1, 0.2, 0.9))
    task.apple_actor = SimpleNamespace(entity=apple)
    observed = []

    def center(_actor):
        observed.append(apple.get_pos())
        return apple.get_pos()

    task._get_object_center_world = center
    task.obj_name = "035_apple"
    task.place_pos = np.array([0.25, -0.40, 0.845])
    task.TABLE_TOP_Z = 0.765
    task.vla_recorder = None
    task.plan_success = True
    task._interaction_sim_step = 600
    task._eval_human_started_sim_step = human_started
    for name in ("GRASP_SETTLE_STEPS", "POST_DETACH_SETTLE_STEPS",
                 "POST_LIFT_GRIP_HOLD_STEPS", "RELEASE_SETTLE_STEPS",
                 "FINAL_SETTLE_STEPS"):
        setattr(task, name, 1)
    task._boost_arm_pd = lambda _n: None
    task.open_gripper = lambda _n: events.append("open")
    task.set_gripper = lambda v, _n, num_steps: events.append(("close", v))
    task.step_sim = lambda: events.append("sim")
    task._settle_released_object = lambda: events.append("settle") or True

    class _Link:
        def __init__(self, pose):
            self.pose = pose

        def to_pose7(self):
            return np.concatenate([self.pose.p, self.pose.q])

    monkeypatch.setattr(take_module, "tcp_to_link_pose", lambda p, _o: _Link(p))
    task.move_and_execute = lambda pose7, _n: events.append(
        ("plan", tuple(np.round(np.asarray(pose7)[:3], 4)))) or object()

    def lift(_s, _e, _n, n_steps, sim_per_step):
        events.append("lift")
        apple._pos[2] += 0.05

    task._move_cartesian = lift
    task._move_screw = lambda pos, _n, tcp_yaw=0.0: events.append("screw") or True
    task._move_seeded = lambda *_a: events.append("seeded")
    return task, events, apple, observed


@pytest.mark.parametrize("human_started", [None, 100])
def test_blind_script_never_drives_or_waits_on_human(monkeypatch, human_started):
    task, events, _apple, _observed = _blind_task(monkeypatch, human_started)
    assert task.play_blind_once() is True
    assert task._blind_robot_interaction_sim_step == 600
    assert task._blind_human_ready_before_robot is (human_started is not None)
    assert task.avatar.spare_calls <= 1
    assert ("close", 0.45) in events
    assert events.count("screw") == 3
    assert events[-1] == "settle"


def test_blind_grasp_targets_track_object_as_observed_at_plan_time(monkeypatch):
    task, events, apple, _observed = _blind_task(monkeypatch, None)
    plans = []
    original = task.move_and_execute

    def move(pose7, name):
        plans.append(np.asarray(pose7)[:3].copy())
        # The human moves the object while the robot is travelling.
        apple._pos[0] += 0.05
        return original(pose7, name)

    task.move_and_execute = move
    assert task.play_blind_once() is True
    # plans: mid waypoint, pre-grasp, grasp.  The grasp is computed after
    # the pre-grasp move, so it follows the object's newer observed pose.
    assert len(plans) == 3
    assert plans[2][0] == pytest.approx(plans[1][0] + 0.05)


@pytest.mark.parametrize("eval_mode,pre_policy", [
    (True, False), (True, True), (False, False),
])
def test_evaluate_reports_blind_intent_only_in_delayed_eval(eval_mode, pre_policy):
    task = TakeFromHumanEasy({"eval_mode": eval_mode,
                              "eval_avatar_pre_policy": pre_policy})
    eval_mode = eval_mode and not pre_policy
    task.obj_name = "035_apple"
    task.obj_model_id = 0
    task.plan_success = True
    task._eval_trigger_step = 7
    task._eval_human_started_sim_step = 335
    task._blind_robot_interaction_sim_step = 600
    task._blind_human_ready_before_robot = False
    task.target = TargetSpec(object=object(), position=np.zeros(3),
                             label="plate", success_dist_xy=0.12)
    task._compute_target_metrics = lambda: {
        "object_pos": np.zeros(3), "target_pos": np.zeros(3),
        "dist_xy": 0.0, "dist_3d": 0.0, "dz": 0.0,
    }
    task.avatar_collision_summary = lambda: {"enabled": False}
    task.safe_distance_summary = lambda: {"enabled": False}
    task.avatar_safety_summary = lambda: {"enabled": False}
    task.avatar_safety_intervention_summary = lambda: {"enabled": False}
    out = task.evaluate()
    if eval_mode:
        assert out["blind_intent"]["trigger_step"] == 7
        assert out["blind_intent"]["human_started_sim_step"] == 335
        assert out["blind_intent"]["robot_interaction_sim_step"] == 600
        assert out["blind_intent"]["human_ready_before_robot"] is False
    else:
        assert "blind_intent" not in out


class _StanceAvatar:
    def __init__(self, trans):
        self.events = []
        self.robot = SimpleNamespace(global_trans=np.asarray(trans, dtype=float),
                                     global_rot=np.eye(3))
        self._frames = 0

    def play_animation(self, name):
        self.events.append(("play", name))
        self._frames = 2

    def spare(self):
        return self._frames == 0

    def step(self):
        self._frames -= 1

    def reset(self, trans, rot):
        self.events.append(("reset", tuple(np.round(trans, 4))))
        self.robot.global_trans = np.asarray(trans, dtype=float)


@pytest.mark.parametrize("start_y,plays", [(1.20, 2), (0.70, 1)])
def test_delayed_offer_stands_human_within_scripted_reach_guard(start_y, plays):
    task = TakeFromHumanEasy({"eval_mode": True, "eval_avatar_pre_policy": False})
    arm = SimpleNamespace(origin_pose=SimpleNamespace(p=np.array([0.0, 0.0, 0.75])))
    task.robot = SimpleNamespace(get_arm=lambda _n: arm)
    task.avatar = _StanceAvatar((0.0, start_y, 0.0))
    task.avatar_collider = None
    # The offered object sits 0.3 m in front of the human, at arm-base height.
    task._hand_object_target = lambda: (
        task.avatar.robot.global_trans + np.array([0.0, -0.3, 0.75]))
    task._eval_place_human_in_reach()
    assert [e for e in task.avatar.events if e[0] == "play"] == [
        ("play", "deliver_to_human")] * plays
    final = task._hand_object_target()
    assert np.linalg.norm(final - arm.origin_pose.p) <= task.HANDOVER_REACH_GUARD
    assert task.avatar.events[-1][0] == "reset"
    if plays == 1:
        assert task._eval_human_stance_shift_xy == [0.0, 0.0]
    else:
        assert task._eval_human_stance_shift_xy[1] == pytest.approx(
            task.HANDOVER_STEP_TARGET - 0.9)
