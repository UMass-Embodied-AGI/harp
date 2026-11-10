"""deliver_to_human_easy eval mode: success is a real handover (review clip 06).

The robot must open the gripper while the object is at the presented palm;
the human then holds the object, and success is "the human holds it".  A
cube merely held near the palm, or released away from it, fails.  The
presentation plays at physical speed instead of snapping.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.tasks.deliver_to_human import DeliverToHuman

PALM = np.array([0.0, 0.55, 1.05])


class _Rig:
    def __init__(self):
        self.attached_object = [None, None]
        self.two_hand_attached_object = None

    def get_palm_center(self, hand_id=1):
        return PALM.copy()

    def _get_hand_frame(self, hand_id=1):
        # Palm-down hand reaching toward the robot (-y): across-palm = x,
        # fingers = -y, normal e1 x e2 = -z (palm side down).
        R = np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        return PALM + [0.0, 0.08, 0.0], R


class _Cube:
    """Kinematically movable cube (entity origin = centre)."""

    def __init__(self, state):
        self.state = state

    def get_pos(self):
        return np.asarray(self.state["obj"], dtype=float).copy()

    def set_pos(self, p):
        self.state["obj"] = np.asarray(p, dtype=float)

    def set_dofs_velocity(self, v):
        pass

    def get_AABB(self):
        c = self.get_pos()
        return np.stack([c - 0.03, c + 0.03])


class _Avatar:
    def __init__(self):
        self.robot = _Rig()
        self._spare = True
        self.played = []
        self.frame_ratio = 1.0
        self.motion_data = {"take_from_human": {"trans": np.zeros((50, 3))}}
        self.ratios = []

    def spare(self):
        return self._spare

    def play_animation(self, name):
        self.played.append(name)
        self.ratios.append(self.frame_ratio)

    def attach_object_to_hand(self, entity, hand_id=1):
        self.robot.attached_object[hand_id] = (entity, np.zeros(3), np.eye(3))


def _task():
    task = DeliverToHuman({"eval_mode": True, "eval_avatar_pre_policy": False})
    task.avatar = _Avatar()
    task._episode_seed = 0
    task._eval_at_reset()
    task._eval_trigger_step = 1
    task._blind_human_started_sim_step = 0          # presentation started
    task.plan_success = True
    state = {"obj": PALM + [0.0, -0.05, 0.06], "tcp": PALM + [0.0, -0.05, 0.06], "grip": 0.5}
    task.obj_actor = SimpleNamespace(entity=_Cube(state))
    task._object_center = lambda: np.asarray(state["obj"], dtype=float)
    task._object_still_held = lambda: True
    arm = SimpleNamespace(get_ee_pose=lambda: np.concatenate([state["tcp"], [1, 0, 0, 0]]))
    arm.gripper_val = 0.5
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.avatar_collision_summary = lambda: {"any_collision": False}

    def set_state(**kw):
        state.update(kw)
        arm.gripper_val = state["grip"]

    return task, set_state


def _tick(task, n=1):
    for _ in range(n):
        task._eval_at_step(1000)


def _finish_glide(task, path=None):
    n = task.HANDOVER_GLIDE_DELAY_SIM_STEPS + task.HANDOVER_GLIDE_SIM_STEPS
    for _ in range(n):
        assert task._handover_done is False
        task._advance_handover_glide()
        if path is not None:
            path.append(task._object_center().copy())
    assert task._handover_glide is None


def test_release_at_the_presented_palm_hands_the_object_over():
    task, set_state = _task()
    _tick(task, 3)                                   # carried to the palm, closed
    assert task.check_success() is False             # held near the palm: no
    set_state(grip=1.0)                              # robot opens the gripper
    _tick(task)
    assert task._handover_taken is True
    assert task.check_success() is False             # not until the glide ends
    _finish_glide(task)
    assert task._handover_done is True
    assert task.avatar.robot.attached_object[1][0] is task.obj_actor.entity
    assert task.check_success() is True
    set_state(tcp=PALM + [0.0, -0.3, 0.3])           # robot lifts away
    _tick(task)
    assert task.check_success() is True


def test_holding_the_cube_near_the_palm_without_release_fails():
    task, _ = _task()
    _tick(task, 20)
    assert task._handover_done is False
    assert task.check_success() is False


def test_release_away_from_the_palm_fails():
    task, set_state = _task()
    far = PALM + [0.0, -0.20, 0.0]
    set_state(obj=far, tcp=far)
    _tick(task, 2)
    set_state(grip=1.0)
    _tick(task)
    assert task._handover_done is False
    assert task.check_success() is False
    # The free cube later falling past the palm is not a handover.
    set_state(obj=PALM + [0.0, 0.0, 0.05], tcp=far + [0, 0, 0.2])
    _tick(task, 3)
    assert task._handover_done is False


def test_release_before_the_presentation_finishes_fails():
    task, set_state = _task()
    task.avatar._spare = False                       # clip still playing
    _tick(task, 2)
    set_state(grip=1.0)
    _tick(task)
    assert task._handover_done is False
    assert task.check_success() is False


def test_gradual_opening_counts_as_release():
    task, set_state = _task()
    _tick(task, 2)
    for g in (0.55, 0.6, 0.65, 0.7):
        set_state(grip=g)
        _tick(task)
    assert task._handover_taken is True


def test_collision_still_fails_the_handover():
    task, set_state = _task()
    _tick(task, 2)
    set_state(grip=1.0)
    _tick(task)
    _finish_glide(task)
    task.avatar_collision_summary = lambda: {"any_collision": True}
    assert task.check_success() is False


def test_presentation_plays_at_physical_speed():
    task, _ = _task()
    task._blind_human_started_sim_step = None
    task._start_blind_presentation()
    assert task.avatar.played == ["take_from_human"]
    # 50 frames stretched to 1.5 s of 2 ms sim steps = 750 sim steps.
    assert abs(task.avatar.ratios[0] * 50 - 750) < 1e-6
    assert task.avatar.frame_ratio == 1.0            # restored afterwards


def test_cube_glides_round_the_fingertips_into_the_palm_down_hand():
    task, set_state = _task()
    _tick(task, 2)
    start = task._object_center().copy()
    set_state(grip=1.0)
    _tick(task)
    path = []
    _finish_glide(task, path)
    path = np.array(path)
    end = path[-1]
    # Held still during the delay, then moved smoothly (no jumps > 1.5 cm).
    assert np.allclose(path[:task.HANDOVER_GLIDE_DELAY_SIM_STEPS], start)
    assert np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) < 0.015
    # Ends under the palm (palm side = down), a little along the fingers.
    assert end[2] < PALM[2] - 0.03
    assert abs(end[1] - (PALM[1] - task.HANDOVER_GRASP_ALONG_FINGERS_M)) < 1e-6
    # Never passes through the hand: while below the start height it stays
    # beyond the fingertips or already under the fingers.
    tips_y = PALM[1] - task.HANDOVER_FINGER_LENGTH_M
    for p in path:
        if p[1] > tips_y and PALM[2] - 0.03 < p[2] < start[2] - 1e-6:
            raise AssertionError(f"glide crosses the hand at {p}")
    assert task.check_success() is True


def test_eval_episode_results_keep_the_handover_metrics():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "eval.py").read_text()
    block = src[src.index("detail_keys = ("):]
    block = block[:block.index(")\n")]
    assert '"handover"' in block and '"blind_intent"' in block
