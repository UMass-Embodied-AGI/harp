"""Scene-free contracts for the eval-mode fixes.

Covers the shared avatar-check helper (record tagging from every step_sim
override), the stamp_documents doc-1 trigger, the take_from_human_safety knife
release, the take_from_human_easy tolerant handoff + z gate, and the
open_microwave reset clearance gate.  No Genesis scene is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import transforms3d as t3d

import envs.tasks.take_from_human_easy as easy_module
from envs.base_task import BaseTask
from envs.tasks.open_microwave import (
    OpenMicrowave,
    SceneLayoutError,
    reset_clearance_verdict,
)
from envs.tasks.place_burger_and_fries_copresence import PlaceBurgerAndFriesCopresence
from envs.tasks.feed_documents import (
    FeedDocuments,
    doc_covers_point,
    second_stamp_ready,
)
from envs.tasks.take_from_human_easy import TakeFromHumanEasy, handoff_grip_ready
from envs.tasks.take_from_human_safety import (
    TakeFromHumanSafety,
    release_condition,
    segment_point_distance,
)


# ---------------------------------------------------------------------------
# Shared collision helper
# ---------------------------------------------------------------------------

class _Checker:
    def __init__(self, collided=False, margin=0.0):
        self.calls = 0
        self.collided = collided
        self.margin = margin

    def check(self):
        self.calls += 1
        return {"collided": self.collided, "min_distance": 0.1 - self.calls}


class _Scene:
    dt = 0.002

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _arm_checks(task, stride=2, collided=True):
    task.scene = _Scene()
    task.robot = None
    task.avatar = None
    task.avatar_collider = None
    task.avatar_collision_checker = _Checker(collided=collided)
    task.avatar_safety_checker = _Checker()
    task._avatar_collision_stride = stride
    task._avatar_collision_tick = 0
    task.SIM_STEP_IDX = 0
    task.avatar_collision_log = []
    task.avatar_safety_log = []
    task.avatar_collided = False
    task._record_stride = None
    task.vla_recorder = None
    task._video_path = None
    return task


def _assert_tagged(task, n_steps=4, stride=2):
    for _ in range(n_steps):
        task.step_sim()
    assert task.SIM_STEP_IDX == n_steps
    log = task.avatar_collision_log
    assert [r["sim_step"] for r in log] == list(range(stride, n_steps + 1, stride))
    for r in log:
        assert r["t"] == pytest.approx(r["sim_step"] * 0.002)
        assert r["pretask"] is False
        assert "frame" in r
    assert [r["sim_step"] for r in task.avatar_safety_log] == [
        r["sim_step"] for r in log
    ]
    assert task.avatar_collided is True


def test_base_helper_tags_records_and_respects_stride_and_pretask():
    task = _arm_checks(BaseTask({}), stride=3, collided=False)
    task.SIM_STEP_IDX = 7
    task._record_suppressed = True
    for _ in range(3):
        task._tick_avatar_checks()
    assert len(task.avatar_collision_log) == 1
    rec = task.avatar_collision_log[0]
    assert rec["sim_step"] == 7 and rec["pretask"] is True
    assert rec["t"] == pytest.approx(7 * 0.002)
    assert task.avatar_collided is False
    summary = task.avatar_collision_summary()
    assert summary["first_collision_sim_step"] is None


def test_base_helper_noop_without_checkers():
    task = BaseTask({})
    task.avatar_collider = None
    task.avatar_collision_checker = None
    task.avatar_safety_checker = None
    task._avatar_collision_tick = 0
    task._tick_avatar_checks()
    assert task._avatar_collision_tick == 0


def test_stamp_documents_step_sim_uses_shared_helper():
    task = _arm_checks(FeedDocuments({}))
    task._stamp_ready = False
    task.stamp_marks = []
    _assert_tagged(task)
    assert task.avatar_collision_summary()["first_collision_sim_step"] == 2


def test_open_microwave_step_sim_uses_shared_helper():
    task = _arm_checks(OpenMicrowave({}))
    task._plate_attached = False
    task._door_hold_active = False
    _assert_tagged(task)


def test_frying_scene_step_sim_uses_shared_helper():
    from envs.tasks.oil_bottle_recovery import OilBottleRecovery

    task = _arm_checks(OilBottleRecovery({}))
    task._tilt_track_active = False
    _assert_tagged(task)


# ---------------------------------------------------------------------------
# stamp_documents
# ---------------------------------------------------------------------------

def test_doc_covers_point_uses_doc_frame_and_margin():
    half = (0.05, 0.05, 0.0125)
    q0 = (1.0, 0.0, 0.0, 0.0)
    assert doc_covers_point((0.1, 0.1, 0.8), q0, (0.14, 0.1), half)
    assert not doc_covers_point((0.1, 0.1, 0.8), q0, (0.16, 0.1), half)
    assert doc_covers_point((0.1, 0.1, 0.8), q0, (0.16, 0.1), half, margin=0.02)
    q45 = t3d.euler.euler2quat(0.0, 0.0, np.pi / 4)
    # A corner direction is farther from the centre for a 45-degree doc.
    assert doc_covers_point((0.0, 0.0, 0.8), q45, (0.0, 0.065), half)
    assert not doc_covers_point((0.0, 0.0, 0.8), q0, (0.065, 0.065), half)


def test_second_stamp_ready_requires_every_gate():
    assert second_stamp_ready(True, False, 10, 10, True)
    assert not second_stamp_ready(False, False, 10, 10, True)
    assert not second_stamp_ready(True, True, 10, 10, True)
    assert not second_stamp_ready(True, False, 9, 10, True)
    assert not second_stamp_ready(True, False, 10, 10, False)


class _Doc:
    def __init__(self, xy):
        self.pos = np.array([xy[0], xy[1], 0.8])

    def get_pos(self):
        return self.pos.copy()

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


def _stamp_eval_task(tcp_xy, doc0_xy, doc1_xy, spare=True, contract=None):
    cfg = {"eval_mode": True, "stamp_eval_robot_clear_steps": 3}
    if contract is not None:
        cfg["stamp_eval_doc_contract"] = contract
    task = FeedDocuments(cfg)
    task._stamp_xy = np.array([0.10, 0.10])
    task._doc_half = (0.05, 0.05, 0.0125)
    task.document_actors = [_Doc(doc0_xy), _Doc(doc1_xy)]
    state = {"tcp": np.array([*tcp_xy, 0.9, 1, 0, 0, 0], dtype=float)}
    arm = SimpleNamespace(get_ee_pose=lambda: state["tcp"].copy())
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.avatar = SimpleNamespace(spare=lambda: spare)
    task._eval_avatar_warmup_done = True
    task._eval_doc1_stamp_fired = False
    task._eval_robot_clear_steps = 0
    task._eval_stamp_policy_steps = 0
    calls = []
    task._play_one_stamp_cycle = lambda **kw: calls.append(kw)
    task._eval_return_avatar_to_rest = lambda: calls.append("rest")
    return task, state, calls


def test_stamp_eval_hook_fires_once_after_robot_clear_steps():
    task, state, calls = _stamp_eval_task((0.10, -0.15), (-0.20, 0.10), (0.11, 0.09))
    task._eval_stamp_step()
    task._eval_stamp_step()
    assert calls == []
    task._eval_stamp_step()
    assert len(calls) == 2
    kw = calls[0]
    assert kw["doc_idx"] == 1 and kw["doc"] is task.document_actors[1]
    assert kw["pin_at_stamp_xy"] is True  # demo contract (play_once pin)
    assert calls[1] == "rest"
    assert task._eval_doc1_stamp_step == 3
    task._eval_stamp_step()
    assert len(calls) == 2


def test_stamp_eval_hook_resets_clear_count_when_robot_returns():
    task, state, calls = _stamp_eval_task((0.10, -0.15), (-0.20, 0.10), (0.10, 0.10))
    task._eval_stamp_step()
    task._eval_stamp_step()
    state["tcp"][:2] = (0.12, 0.12)
    task._eval_stamp_step()
    assert task._eval_robot_clear_steps == 0
    state["tcp"][:2] = (0.10, -0.15)
    task._eval_stamp_step()
    task._eval_stamp_step()
    assert calls == []
    task._eval_stamp_step()
    assert calls and calls[0]["doc_idx"] == 1


def test_stamp_demo_contract_accepts_expert_push_shortfall():
    # The scripted expert leaves doc 1 ~0.115 m short of the stamp point
    # in y and play_once pins it there; the demo contract does the same.
    task, _state, calls = _stamp_eval_task(
        (0.10, -0.15), (-0.03, 0.14), (0.123, -0.015))
    for _ in range(3):
        task._eval_stamp_step()
    assert calls and calls[0]["pin_at_stamp_xy"] is True


def test_stamp_strict_contract_needs_doc_under_stamp_and_no_snap():
    task, _state, calls = _stamp_eval_task(
        (0.10, -0.15), (-0.03, 0.14), (0.123, -0.015), contract="strict")
    for _ in range(5):
        task._eval_stamp_step()
    assert calls == []
    task.document_actors[1] = _Doc((0.12, 0.08))
    task._eval_stamp_step()
    assert calls and calls[0]["pin_at_stamp_xy"] is False


@pytest.mark.parametrize(
    ("doc0", "doc1", "spare"),
    [
        ((0.10, 0.10), (0.30, -0.15), True),   # doc 1 not delivered
        ((0.12, 0.10), (0.10, 0.10), True),    # doc 0 still in the zone
        ((0.02, 0.10), (0.10, 0.10), True),    # doc 0 would overlap the pinned doc 1
        ((-0.2, 0.10), (0.10, 0.10), False),   # avatar still moving
    ],
)
def test_stamp_eval_hook_gates(doc0, doc1, spare):
    task, _state, calls = _stamp_eval_task((0.10, -0.15), doc0, doc1, spare=spare)
    for _ in range(6):
        task._eval_stamp_step()
    assert calls == []


def test_stamp_eval_hook_inactive_outside_eval_or_before_warmup():
    task, _state, calls = _stamp_eval_task((0.10, -0.15), (-0.2, 0.1), (0.1, 0.1))
    task._eval_avatar_warmup_done = False
    for _ in range(5):
        task._eval_stamp_step()
    task.config["eval_mode"] = False
    task._eval_avatar_warmup_done = True
    for _ in range(5):
        task._eval_stamp_step()
    assert calls == []


# ---------------------------------------------------------------------------
# take_from_human_safety
# ---------------------------------------------------------------------------

def test_segment_point_distance_and_release_condition():
    c = np.zeros(3)
    axis = np.array([0.0, 1.0, 0.0])
    assert segment_point_distance([0.03, 0.05, 0.0], c, axis, 0.10) == pytest.approx(0.03)
    assert segment_point_distance([0.0, 0.14, 0.0], c, axis, 0.10) == pytest.approx(0.04)
    assert release_condition(0.3, 0.05, 0.5, 0.07)
    assert not release_condition(0.5, 0.05, 0.5, 0.07)
    assert not release_condition(0.2, 0.08, 0.5, 0.07)


class _Collider:
    def get_pos(self):
        return np.array([0.0, 0.0, 1.0])

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


def _safety_task(tcp, gripper):
    task = TakeFromHumanSafety({"eval_mode": True, "action_substeps": 50})
    state = {"tcp": np.array([*tcp, 1, 0, 0, 0], dtype=float), "g": gripper}

    class _Arm:
        @property
        def gripper_val(self):
            return state["g"]

        def get_ee_pose(self):
            return state["tcp"].copy()

    arm = _Arm()
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    events = []
    task.avatar = SimpleNamespace(
        spare=lambda: True,
        robot=SimpleNamespace(detach_object=lambda hand: events.append(("detach", hand))),
        play_return_to_idle=lambda frames: events.append(("idle", frames)),
    )
    task._handle_collider = _Collider()
    task._eval_handover_done = True
    task._eval_trigger_step = 0
    task._eval_knife_released = False
    task._eval_knife_release_closed_steps = 0
    task._lock_collider_quat = True
    task._locked_collider_quat = np.array([1.0, 0, 0, 0])
    return task, state, events


def test_safety_release_after_k_closed_steps_near_knife():
    task, state, events = _safety_task((0.02, 0.05, 1.0), 0.3)
    required = task._eval_release_required_steps()
    assert required == 3  # ceil(150 / 50)
    for step in range(required - 1):
        task._eval_at_step(10 + step)
    assert events == []
    task._eval_at_step(20)
    assert events == [("detach", 1), ("idle", task.AVATAR_RETURN_TO_IDLE_FRAMES)]
    assert task._eval_knife_released is True
    assert task._lock_collider_quat is False
    assert task._eval_knife_release_debug["step_idx"] == 20
    task._eval_at_step(21)
    assert len(events) == 2


def test_safety_release_counter_resets_when_open_or_far():
    task, state, events = _safety_task((0.02, 0.05, 1.0), 0.3)
    task._eval_at_step(1)
    task._eval_at_step(2)
    state["g"] = 0.8
    task._eval_at_step(3)
    assert task._eval_knife_release_closed_steps == 0
    state["g"] = 0.3
    state["tcp"][:3] = (0.0, 0.0, 1.2)
    task._eval_at_step(4)
    assert task._eval_knife_release_closed_steps == 0
    assert events == []


def test_safety_release_waits_for_handover():
    task, state, events = _safety_task((0.02, 0.05, 1.0), 0.3)
    task._eval_handover_done = False
    task._eval_trigger_step = 10**6
    for step in range(6):
        task._eval_at_step(step)
    assert events == []


def test_safety_reset_skips_deferral_after_eval_chop_start(monkeypatch):
    task = TakeFromHumanSafety({"eval_mode": True, "defer_video_until_chop_start": True})

    def fake_reset(current, seed=0):
        current.vla_recorder = "rec"
        current._eval_avatar_started = True  # _eval_at_reset already ran
        return "obs"

    monkeypatch.setattr(BaseTask, "reset", fake_reset)
    task.avatar = None
    task._defer_video_until_chop_start = False
    assert task.reset(seed=3) == "obs"
    assert task._defer_video_until_chop_start is False
    assert task.vla_recorder == "rec"

    def fake_reset_play(current, seed=0):
        current.vla_recorder = "rec"
        return "obs"

    monkeypatch.setattr(BaseTask, "reset", fake_reset_play)
    task.reset(seed=3)
    assert task._defer_video_until_chop_start is True
    assert task.vla_recorder is None


# ---------------------------------------------------------------------------
# take_from_human_easy
# ---------------------------------------------------------------------------

class _EasyArm:
    def __init__(self, ee, g):
        self.ee = np.asarray(ee, dtype=float)
        self.gripper_val = float(g)

    def get_ee_pose(self):
        return self.ee.copy()


def _easy_task(g, obj="073_rubikscube"):
    task = TakeFromHumanEasy({"eval_mode": True, "eval_handoff_closed_steps": 2})
    arm = _EasyArm((0.03, 0.0, 0.0), g)
    task.robot = SimpleNamespace(get_arm=lambda _n: arm)
    events = []
    task.avatar = SimpleNamespace(
        robot=SimpleNamespace(detach_object=lambda h: events.append(("detach", h)))
    )
    task.apple_actor = object()
    task.obj_name = obj
    task._get_object_center_world = lambda _a: np.zeros(3)
    task._eval_apple_attached_to_avatar = True
    task._eval_handoff_closed_steps = 0
    return task, events


@pytest.mark.parametrize(("g", "detached"), [(0.55, True), (0.35, True), (0.65, False)])
def test_easy_handoff_threshold_is_tolerant(g, detached):
    task, events = _easy_task(g)
    task._eval_at_step(1)
    task._eval_at_step(2)
    assert (events == [("detach", 1)]) is detached


def test_easy_warmup_keeps_object_in_hand_by_default():
    task = TakeFromHumanEasy({"eval_mode": True, "eval_avatar_post_warmup_settle_steps": 1})
    task.avatar = SimpleNamespace(spare=lambda: True)
    task.step_sim = lambda: None
    task._place_eval_apple_support_and_detach = lambda: pytest.fail("support used")
    task.get_obs = lambda: "obs"
    assert task.eval_pre_policy_warmup() == "obs"


def test_easy_target_has_z_gate(monkeypatch):
    class _Ent:
        def __init__(self, p):
            self.p = np.asarray(p, dtype=float)

        def get_pos(self):
            return self.p

        def get_quat(self):
            return np.array([1.0, 0.0, 0.0, 0.0])

    class _Actor:
        def __init__(self, p):
            self.entity = _Ent(p)
            self.config = {"center": [0, 0, 0], "scale": [1, 1, 1]}

    monkeypatch.setattr(easy_module, "load_object", lambda _s, pose, *_a, **_k: _Actor(pose.p))
    monkeypatch.setattr(easy_module, "create_primitive", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        TakeFromHumanEasy, "_get_object_radius",
        staticmethod(lambda _n, _m=0: 0.03),
    )
    task = TakeFromHumanEasy({})
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    np.random.seed(0)
    task.load_actors()
    assert task.target.success_dz_min == pytest.approx(TakeFromHumanEasy.SUCCESS_DZ_MIN)
    task.plan_success = True
    place = np.asarray(task.place_pos, dtype=float)
    for z, ok in ((place[2] - 0.02, True), (task.TABLE_TOP_Z + 0.028, True),
                  (0.03, False), (-200.0, False),
                  # held over the plate mid-carry / at the release height
                  (place[2] + 0.27, False), (place[2] + 0.14, False)):
        task.apple_actor.entity.p = np.array([place[0], place[1], z])
        assert task.check_success() is ok, z


# ---------------------------------------------------------------------------
# open_microwave / place_burger_fries_neutral
# ---------------------------------------------------------------------------

def test_reset_clearance_verdict():
    assert reset_clearance_verdict(0.08, 0.05, 0.0) == "ok"
    assert reset_clearance_verdict(0.03, 0.05, 0.0) == "warn"
    assert reset_clearance_verdict(-0.01, 0.05, 0.0) == "raise"


def test_open_microwave_reset_clearance_raises_scene_error():
    task = OpenMicrowave({})
    task._episode_seed = 9000036
    task.avatar_collider = None
    task.avatar_collision_checker = SimpleNamespace(
        check=lambda: {"min_distance": -0.01, "closest_pair": ("link7", "r_forearm")})
    with pytest.raises(SceneLayoutError):
        task._check_reset_clearance()
    assert task._reset_avatar_clearance["min_distance"] == -0.01
    task.avatar_collision_checker = SimpleNamespace(
        check=lambda: {"min_distance": 0.02, "closest_pair": None})
    assert task._check_reset_clearance() == pytest.approx(0.02)
    task.avatar_collision_checker = None
    assert task._check_reset_clearance() is None


def test_burger_fries_neutral_full_state_expert_sees_the_human():
    assert PlaceBurgerAndFriesCopresence.NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE is True


def test_easy_retracts_human_after_lift_once():
    task, events = _easy_task(0.35)
    calls = []
    z = {"v": 1.30}
    task._get_object_center_world = lambda _a: np.array([0.0, 0.0, z["v"]])
    task.avatar = SimpleNamespace(
        robot=SimpleNamespace(detach_object=lambda h: events.append(("detach", h))),
        spare=lambda: True,
        play_animation_reverse=lambda name: calls.append(name),
    )
    task.robot.get_arm("right").ee = np.array([0.0, 0.0, 1.32])
    task._eval_at_step(1)
    task._eval_at_step(2)
    assert events == [("detach", 1)]
    assert task._eval_handoff_obj_z == pytest.approx(1.30)
    task._eval_at_step(3)
    assert calls == []
    z["v"] = 1.34
    task._eval_at_step(4)
    task._eval_at_step(5)
    assert calls == ["deliver_to_human"]


# ---------------------------------------------------------------------------
# take_from_human_easy grip gate (intent3): the human lets go only once the
# jaws are closed ON the object.
# ---------------------------------------------------------------------------

def test_handoff_grip_ready_needs_blocked_stalled_centred_fingers():
    held = [0.028, 0.028]
    assert handoff_grip_ready(held, 0.014, held, 0.0)
    # optional block test: fingers reached their commanded target
    assert not handoff_grip_ready([0.014, 0.014], 0.014, [0.014, 0.014], 0.0,
                                  block_margin=0.002)
    assert handoff_grip_ready([0.014, 0.014], 0.014, [0.014, 0.014], 0.0)
    # still closing
    assert not handoff_grip_ready(held, 0.014, [0.032, 0.032], 0.0)
    # object centre outside the jaw pads along the closing axis
    assert not handoff_grip_ready(held, 0.014, held, 0.026)
    assert handoff_grip_ready(held, 0.014, held, -0.02)
    # first observation has no previous sample
    assert not handoff_grip_ready(held, 0.014, None, 0.0)


class _FingerEntity:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)

    def get_qpos(self):
        return self.q


class _FingerArm(_EasyArm):
    gripper_open = 0.04
    gripper_closed = 0.0

    def __init__(self, ee, g, fingers):
        super().__init__(np.r_[np.asarray(ee, float), [1.0, 0.0, 0.0, 0.0]], g)
        self._finger_dof_indices = [7, 8]
        self.entity = _FingerEntity(np.r_[np.zeros(7), fingers])


def _easy_finger_task(fingers, ee=(0.0, 0.0, 0.0), mode="grip"):
    task, events = _easy_task(0.35)
    task.config["eval_handoff_mode"] = mode
    arm = _FingerArm(ee, 0.35, fingers)
    task.robot = SimpleNamespace(get_arm=lambda _n: arm)
    task._eval_handoff_prev_finger_q = None
    return task, events, arm


def test_easy_handoff_waits_for_grip_on_object():
    task, events, arm = _easy_finger_task([0.028, 0.028])
    for i in range(1, 4):
        task._eval_at_step(i)
    # step 1 has no previous finger sample; steps 2-3 satisfy the gate twice
    assert events == [("detach", 1)]
    assert task._eval_handoff_debug["grip"]["finger_q"] == [0.028, 0.028]


def test_easy_handoff_keeps_object_when_jaws_miss():
    # jaws shut 2.3 cm beside the object centre (open-loop replay 9000000)
    task, events, arm = _easy_finger_task([0.014, 0.014], ee=(0.0, 0.023, 0.0))
    for i in range(1, 6):
        task._eval_at_step(i)
    assert events == []
    assert task._eval_apple_attached_to_avatar


def test_easy_handoff_keeps_object_when_off_centre():
    # TCP 2.3 cm off along the closing (y) axis: object outside the pads.
    task, events, arm = _easy_finger_task([0.016, 0.016], ee=(0.0, 0.023, 0.0))
    for i in range(1, 6):
        task._eval_at_step(i)
    assert events == []


def test_easy_handoff_grip_gate_can_be_disabled():
    task, events, arm = _easy_finger_task([0.014, 0.014])
    task.config["eval_handoff_require_grip"] = False
    task._eval_at_step(1)
    task._eval_at_step(2)
    assert events == [("detach", 1)]


# ---------------------------------------------------------------------------
# stamp_documents doc-0 stamp on the policy path (intent3): the demos start
# with doc 0 unstamped and the human stamps once the robot is at standby.
# ---------------------------------------------------------------------------

def _stamp_doc0_task(tcp_xyz):
    task, state, calls = _stamp_eval_task(tcp_xyz[:2], (0.10, 0.10), (0.30, -0.15))
    task.TABLE_TOP_Z = 0.68
    state["tcp"][2] = tcp_xyz[2]
    task._eval_doc0_stamp_fired = False
    task._eval_doc0_standby_steps = 0
    return task, state, calls


def test_stamp_warmup_defers_doc0_stamp_to_policy_for_legacy_human():
    task = FeedDocuments({"eval_mode": True, "stamp_eval_human": "robot_triggered"})
    task.avatar = object()
    task.document_actors = [object(), object()]
    task._attach_stamp_to_palm = lambda: None
    task._play_one_stamp_cycle = lambda **kw: pytest.fail("stamped in warmup")
    task.get_obs = lambda: "obs"
    assert task.eval_pre_policy_warmup() == "obs"
    assert task._eval_doc0_stamp_fired is False


def test_stamp_doc0_fires_at_standby_then_doc1_logic_runs():
    task, state, calls = _stamp_doc0_task((0.10, -0.10, 0.90))
    task._eval_stamp_step()
    task._eval_stamp_step()
    assert calls == []
    task._eval_stamp_step()
    assert calls[0]["doc_idx"] == 0 and calls[1] == "rest"
    assert task._eval_doc0_stamp_step == 3
    assert task._eval_doc1_stamp_fired is False


def test_stamp_doc0_waits_while_robot_away_from_standby():
    task, state, calls = _stamp_doc0_task((0.40, 0.40, 1.10))
    for _ in range(10):
        task._eval_stamp_step()
    assert calls == []
    task.config["stamp_eval_doc0_fallback_steps"] = 12
    task._eval_stamp_step()
    assert calls == []
    task._eval_stamp_step()
    assert calls[0]["doc_idx"] == 0


# ---------------------------------------------------------------------------
# take_from_human_safety grip gate (intent3): the human keeps the knife when
# the jaws close beside the slab.
# ---------------------------------------------------------------------------

def _safety_finger_task(tcp, fingers, gripper=0.25):
    task, state, events = _safety_task(tcp, gripper)
    arm = task.robot.get_arm("right")
    arm.gripper_open = 0.04
    arm.gripper_closed = 0.0
    arm._finger_dof_indices = [7, 8]
    state["fingers"] = np.asarray(fingers, dtype=float)
    arm.entity = SimpleNamespace(get_qpos=lambda: np.r_[np.zeros(7), state["fingers"]])
    task._eval_knife_prev_finger_q = None
    return task, state, events


def test_safety_release_needs_jaws_closed_on_slab():
    # Slab long axis = world y through (0, 0, 1); jaw axis = TCP y = world y
    # here, so a TCP on the slab line has zero lateral offset.
    task, state, events = _safety_finger_task((0.0, 0.03, 1.0), [0.015, 0.015])
    for step in range(1, 5):
        task._eval_at_step(step)
    assert events and events[0] == ("detach", 1)
    assert task._eval_knife_release_debug["grip"]["finger_q"] == [0.015, 0.015]


def test_safety_keeps_knife_when_jaws_shut_on_nothing():
    # act seed 9000003: fingers reached their 0.25 target (10 mm) with the
    # slab 5 cm across the jaw axis, inside the old 7 cm proximity gate.
    task, state, events = _safety_finger_task((0.0, 0.03, 1.0), [0.010, 0.010])
    state["tcp"][3:7] = t3d.quaternions.axangle2quat([0, 0, 1], -np.pi / 2)
    state["tcp"][0] = 0.05
    for step in range(1, 8):
        task._eval_at_step(step)
    assert events == []
    assert task._eval_knife_released is False


def test_safety_keeps_knife_when_slab_outside_pads():
    # Rotate the TCP so its jaw axis is world x; slab line is 5 cm away in x.
    task, state, events = _safety_finger_task((0.05, 0.03, 1.0), [0.015, 0.015])
    q = t3d.quaternions.axangle2quat([0, 0, 1], -np.pi / 2)
    state["tcp"][3:7] = q
    for step in range(1, 8):
        task._eval_at_step(step)
    assert events == []


# Contact-release handoff (intent3 default for take_from_human_easy).

def _contact_task(fingers, ee=(0.0, 0.0, 0.0), g=0.6):
    task, events, arm = _easy_finger_task(fingers, ee=ee, mode="contact")
    arm.gripper_val = g
    task.obj_model_id = 0
    task._get_object_radius = lambda _n, _m=0: 0.0285
    return task, events, arm


def test_easy_grip_mode_is_default():
    # The human releases once the gripper has closed on
    # the cube (demo timing), not at first jaw contact.
    assert TakeFromHumanEasy.EVAL_HANDOFF_MODE == "grip"
    assert TakeFromHumanEasy({}).config.get("eval_handoff_mode") is None


def test_easy_grip_default_holds_through_contact_until_jaws_stall():
    task, events, arm = _easy_finger_task([0.035, 0.035], mode=None)
    task.config.pop("eval_handoff_mode")
    task.obj_model_id = 0
    task._get_object_radius = lambda _n, _m=0: 0.0285
    task._eval_at_step(1)
    arm.entity.q[7:] = (0.030, 0.030)   # jaws reach the faces, still closing
    task._eval_at_step(2)
    assert events == []
    arm.entity.q[7:] = (0.0285, 0.0285)
    task._eval_at_step(3)
    assert events == []                 # still moving (1.5 mm)
    task._eval_at_step(4)
    task._eval_at_step(5)
    assert events == [("detach", 1)]
    assert "mode" not in task._eval_handoff_debug
    assert task._eval_handoff_debug["grip"]["finger_q"] == [0.0285, 0.0285]


def test_easy_grip_keeps_object_when_closed_too_far_away():
    # Stalled, centred jaws 5 cm above the cube: the 3.5 cm gate keeps it.
    task, events, arm = _easy_finger_task([0.014, 0.014], ee=(0.0, 0.0, 0.05))
    for i in range(1, 6):
        task._eval_at_step(i)
    assert events == []


def test_easy_contact_release_when_jaws_reach_surface():
    task, events, arm = _contact_task([0.035, 0.035])
    task._eval_at_step(1)
    assert events == []  # jaws still 6.5 mm off the faces
    arm.entity.q[7:] = (0.030, 0.030)
    task._eval_at_step(2)
    assert events == [("detach", 1)]
    assert task._eval_handoff_debug["mode"] == "contact"


def test_easy_contact_keeps_object_when_off_centre_or_open():
    task, events, arm = _contact_task([0.030, 0.030], ee=(0.0, 0.02, 0.0))
    task._eval_at_step(1)
    task, events2, arm = _contact_task([0.030, 0.030], g=1.0)
    task._eval_at_step(1)
    assert events == [] and events2 == []


def test_easy_contact_keeps_object_when_jaws_shut_on_air():
    # train37 diffusion 9000000: jaws closed 8 cm short of the cube.
    task, events, arm = _contact_task([0.0128, 0.0128])
    task._eval_at_step(1)
    task, events2, arm = _contact_task([0.029, 0.029], ee=(0.05, 0.0, 0.0))
    task._get_object_center_world = lambda _a: np.array([0.0, 0.0, 0.0])
    task._eval_at_step(1)
    assert events == [] and events2 == []


def test_safety_keeps_knife_when_jaws_close_above_slab():
    # train37 diffusion 9000002: TCP on the slab line but 2 cm high; the
    # fingers reached their target (8.0 mm) +0.3 mm -- nothing between them.
    task, state, events = _safety_finger_task((0.0, 0.03, 1.0), [0.0083, 0.0082], gripper=0.2)
    for step in range(1, 9):
        task._eval_at_step(step)
    assert events == []


def test_safety_collision_block_counts_checks_and_colliding_checks():
    from envs.safety_metrics import compute_safety_metrics

    clean = compute_safety_metrics([{"min_distance": 0.2, "collided": False}] * 4)
    assert clean["collision"]["n_collision_checks"] == 4
    assert clean["collision"]["n_colliding_checks"] == 0
    hit = compute_safety_metrics(
        [{"min_distance": 0.2, "collided": False}, {"min_distance": -0.01, "collided": True}])
    assert hit["collision"]["n_collision_checks"] == 2
    assert hit["collision"]["n_colliding_checks"] == 1
    assert hit["collision"]["collision_step_ratio"] == 0.5


def test_stamp_doc1_eases_back_to_clip_start_before_stamping():
    task, _state, calls = _stamp_eval_task((0.10, -0.15), (-0.20, 0.10), (0.11, 0.09))
    task._eval_ease_to_stamp_start = lambda: calls.append("ease")
    for _ in range(3):
        task._eval_stamp_step()
    assert calls[0] == "ease" and isinstance(calls[1], dict) and calls[2] == "rest"


def test_stamp_rest_between_stamps_is_the_retreat_pose_by_default():
    task = FeedDocuments({"eval_mode": True})
    assert task._eval_rest_mode() == "retreat"
    targets = []
    task.avatar = SimpleNamespace(
        play_transition=lambda **kw: targets.append(kw["target_pose"]),
        play_return_to_idle=lambda **kw: targets.append("idle"),
    )
    task._stamp_rest_pose = {"pose": "frame0", "node_trans": 0, "global_mat": 0,
                             "global_mat_inv": 0}
    task._stamp_retreat_pose = {"pose": "retreat", "node_trans": 0, "global_mat": 0,
                                "global_mat_inv": 0}
    task._eval_return_avatar_to_rest()
    assert targets == ["retreat"]
    task.config["stamp_eval_rest_pose"] = "frame0"
    task._eval_return_avatar_to_rest()
    assert targets == ["retreat", "frame0"]
