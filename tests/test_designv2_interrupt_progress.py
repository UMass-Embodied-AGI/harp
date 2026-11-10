"""Unit tests for the interrupt contract helper.

No Genesis: the trigger, the human sequence and the scripted-expert driver
are exercised with small fakes.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.interaction.interrupt_progress import (
    DEFAULT_COMMIT_RADIUS_M,
    InterruptHumanSequence,
    InterruptYield,
    ProgressTrigger,
    ProgressTriggerParams,
    ScriptedRolloutDriver,
    burn_legacy_trigger_draw,
    interrupt_fired,
    interrupt_metrics,
    legacy_pace_block,
    new_interrupt_episode,
    robot_grasping,
    run_full_state_interrupt_expert,
)


OBJ = np.array([0.40, 0.00, 0.80])


def _armed(start_tcp=(0.40, -0.40, 1.10), **params):
    trig = ProgressTrigger(ProgressTriggerParams(**params))
    trig.arm(np.asarray(start_tcp), {0: OBJ})
    return trig


def test_no_fire_before_min_policy_steps_even_at_the_object():
    trig = _armed()
    for step in range(1, 20):
        assert trig.update(step, OBJ + [0, 0, 0.05], {0: OBJ}) is None
    assert trig.block_reason == "min_policy_steps"
    assert trig.update(20, OBJ + [0, 0, 0.05], {0: OBJ}) == 0


def test_start_pose_near_object_never_fires_without_moving():
    # Reset TCP already within 0.12 m (xy) of the object: the old force_dist
    # trigger fired at step 1 here.  A robot that does not move never fires.
    start = OBJ + np.array([0.05, 0.05, 0.10])
    trig = _armed(start_tcp=start)
    for step in range(1, 400):
        assert trig.update(step, start, {0: OBJ}) is None
    assert trig.block_reason == "robot_not_moved"
    m = trig.metrics()
    assert m["fired"] is False and m["not_fired_reason"] == "robot_not_moved"


def test_start_pose_near_object_also_needs_progress_not_just_motion():
    # d0 is floored at 0.30 m, so a robot that starts 0.12 m away and wiggles
    # sideways keeps progress < 0.5 unless it actually closes in.
    start = OBJ + np.array([0.0, 0.0, 0.20])
    trig = _armed(start_tcp=start)
    wiggle = start + np.array([0.06, 0.0, 0.0])
    assert trig.update(30, wiggle, {0: OBJ}) is None
    assert trig.block_reason == "no_progress_within_commit_radius"
    assert trig.update(31, OBJ + [0, 0, 0.12], {0: OBJ}) == 0


def test_fires_on_progress_within_commit_radius_and_records_metrics():
    trig = _armed()
    far = OBJ + np.array([0.0, -0.25, 0.20])  # progress ~0.36, outside radius
    assert trig.update(25, far, {0: OBJ}) is None
    near = OBJ + np.array([0.0, -0.05, 0.15])
    assert trig.update(26, near, {0: OBJ}) == 0
    m = trig.metrics()
    assert m["fired"] and m["trigger_step"] == 26 and m["trigger_reason"] == "progress"
    assert m["trigger_target"] == 0
    assert m["trigger_dist_m"] == pytest.approx(np.linalg.norm([0, 0.05, 0.15]))
    assert m["trigger_progress"] >= 0.5
    # Latched: never fires twice.
    assert trig.update(27, OBJ, {0: OBJ}) is None


def test_commit_radius_blocks_high_progress_far_away():
    trig = _armed(start_tcp=OBJ + [0, -1.0, 0.0])
    # progress 0.7 but 0.30 m away (> 0.22 commit radius)
    assert trig.update(30, OBJ + [0, -0.30, 0.0], {0: OBJ}) is None
    assert trig.update(31, OBJ + [0, -DEFAULT_COMMIT_RADIUS_M + 0.01, 0.0], {0: OBJ}) == 0


def test_phase_gate_and_empty_candidates_block_with_reasons():
    trig = _armed()
    near = OBJ + [0, 0, 0.1]
    assert trig.update(30, near, {0: OBJ}, gate_ok=False, gate_reason="drawer_closed") is None
    assert trig.block_reason == "drawer_closed"
    assert trig.update(31, near, {}) is None
    assert trig.block_reason == "no_eligible_object"


def test_target_is_the_approached_object_not_the_closest_at_reset():
    other = np.array([0.10, -0.30, 0.80])
    trig = ProgressTrigger()
    tcp0 = np.array([0.15, -0.35, 1.0])  # starts next to ``other``
    trig.arm(tcp0, {0: OBJ, 1: other})
    assert trig.update(30, OBJ + [0, 0, 0.15], {0: OBJ, 1: other}) == 0


def test_params_from_task_attrs_and_config_overrides():
    class T:
        INTERRUPT_COMMIT_RADIUS_M = 0.24
        config = {"interrupt_trigger_progress": 0.6}

    p = ProgressTriggerParams.for_task(T())
    assert p.commit_radius_m == 0.24 and p.trigger_progress == 0.6
    assert p.min_policy_steps == 20 and p.min_moved_m == 0.05


def test_trigger_state_round_trip_is_strict():
    trig = _armed()
    trig.update(30, OBJ + [0, 0, 0.1], {0: OBJ})
    clone = ProgressTrigger()
    clone.load_state_dict(trig.state_dict())
    assert clone.metrics() == trig.metrics()
    bad = trig.state_dict()
    bad["unknown"] = 1
    with pytest.raises(ValueError):
        clone.load_state_dict(bad)


# ----------------------------------------------------------------------
# Human sequence
# ----------------------------------------------------------------------


class _Motion:
    def __init__(self, attach_frame):
        self._attach_obj = object()
        self._attach_frame = attach_frame
        self._detach_frame = attach_frame + 50
        self.at_frame = 0


class _AvatarRobot:
    def __init__(self):
        self.global_trans = np.zeros(3)
        self.reaching = False

    def get_palm_center(self, hand_id):
        # Idle: hand hangs by the hip.  Reaching (clip): palm at the target.
        if self.reaching:
            return self.global_trans + np.array([0.0, -0.5, 0.9])
        return self.global_trans + np.array([0.0, 0.0, 0.5])


class _Avatar:
    def __init__(self):
        self.robot = _AvatarRobot()
        self.resets = []
        self.busy = False
        self.motion_modules = {}

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.resets.append(self.robot.global_trans.copy())

    def spare(self):
        return not self.busy


class _Task:
    def __init__(self):
        self.avatar = _Avatar()
        self.config = {"eval_mode": True}


def test_sequence_glides_without_teleport_then_returns_to_idle_root():
    task = _Task()
    seq = InterruptHumanSequence(glide_steps=30, glide_out_steps=10)
    home = np.array([0.0, 1.0, -0.3])
    target = np.array([0.3, 0.6, -0.3])
    task.avatar.reset(home, np.eye(3))
    clip_calls = []

    def start_clip():
        clip_calls.append(True)
        task.avatar.busy = True
        task.avatar.robot.reaching = True
        return True

    seq.start(step=40, start_root=home, target_root=target, rot=np.eye(3),
              target_key=0, target_pos=target + [0, -0.5, 0.9], start_clip=start_clip)
    for i in range(30):
        seq.tick(task, 41 + i)
    steps = np.diff(np.asarray(task.avatar.resets), axis=0)
    max_jump = float(np.max(np.linalg.norm(steps, axis=1)))
    assert max_jump <= np.linalg.norm(target - home) / 30 + 1e-9  # no teleport
    assert clip_calls == [True] and seq.phase == "clip"
    seq.tick(task, 71)
    assert seq.phase == "clip"
    task.avatar.busy = False  # clip + idle blend finished
    task.avatar.robot.reaching = False
    seq.tick(task, 72)
    assert seq.phase == "glide_out"
    for i in range(10):
        seq.tick(task, 73 + i)
    assert seq.done
    assert np.allclose(task.avatar.robot.global_trans, home)
    m = seq.metrics()
    # The palm reaches the target only once the clip runs, after the 1.5 s
    # glide.
    assert m["hand_reach_step"] - m["trigger_step"] >= 30
    assert m["warning_s"] >= 1.5 - 1e-9


def test_attach_guard_cancels_when_robot_grasps_or_object_moved():
    for grasped, pos, reason in (
        (True, OBJ, "robot_grasping"),
        (False, OBJ + [0.05, 0, 0], "object_moved"),
    ):
        task = _Task()
        seq = InterruptHumanSequence(glide_steps=1)
        seq.start(step=1, start_root=np.zeros(3), target_root=np.zeros(3),
                  rot=np.eye(3), target_key=0, target_pos=OBJ,
                  start_clip=lambda: True)
        seq.tick(task, 2)
        motion = _Motion(attach_frame=100)
        task.avatar.motion_modules["Inspect1"] = motion
        seq.set_attach_guard(motion_name="Inspect1", object_pos=lambda p=pos: p,
                             grasped=lambda g=grasped: g)
        motion.at_frame = 50
        seq.on_sim_step(task)
        assert motion._attach_obj is not None  # not yet imminent
        motion.at_frame = 99
        seq.on_sim_step(task)
        assert motion._attach_obj is None and motion._detach_frame is None
        assert seq.attach_cancelled_reason == reason


def test_attach_guard_keeps_attach_for_a_free_object():
    task = _Task()
    seq = InterruptHumanSequence(glide_steps=1)
    seq.start(step=1, start_root=np.zeros(3), target_root=np.zeros(3),
              rot=np.eye(3), target_key=0, target_pos=OBJ, start_clip=lambda: True)
    seq.tick(task, 2)
    motion = _Motion(attach_frame=10)
    task.avatar.motion_modules["Inspect1"] = motion
    seq.set_attach_guard(motion_name="Inspect1", object_pos=lambda: OBJ + [0.01, 0, 0],
                         grasped=lambda: False)
    motion.at_frame = 9
    seq.on_sim_step(task)
    assert motion._attach_obj is not None and seq.attach_cancelled_reason is None


def test_failed_clip_start_still_returns_to_idle():
    task = _Task()
    seq = InterruptHumanSequence(glide_steps=2, glide_out_steps=2)
    seq.start(step=1, start_root=np.zeros(3), target_root=np.ones(3),
              rot=np.eye(3), target_key=0, target_pos=OBJ, start_clip=lambda: False)
    for s in range(2, 8):
        seq.tick(task, s)
    assert seq.done and not seq.clip_started
    # Glided back to the pre-trigger (home) root.
    assert np.allclose(task.avatar.robot.global_trans, np.zeros(3))


def test_sequence_state_round_trip():
    task = _Task()
    seq = InterruptHumanSequence(glide_steps=3)
    seq.start(step=1, start_root=np.zeros(3), target_root=np.ones(3),
              rot=np.eye(3), target_key=2, target_pos=OBJ, start_clip=lambda: True)
    seq.tick(task, 2)
    clone = InterruptHumanSequence()
    clone.load_state_dict(seq.state_dict())
    assert clone.state_dict() == seq.state_dict()


# ----------------------------------------------------------------------
# Robot helpers, metrics and RNG burn
# ----------------------------------------------------------------------


class _Arm:
    def __init__(self, tcp, gripper):
        self._tcp = list(tcp) + [1, 0, 0, 0]
        self.gripper_val = gripper

    def get_ee_pose(self):
        return self._tcp


class _Robot:
    def __init__(self, arm):
        self._arm = arm

    def get_arm(self, tag):
        return self._arm


def test_robot_grasping_needs_closed_gripper_and_proximity():
    task = _Task()
    task.robot = _Robot(_Arm(OBJ + [0, 0, 0.03], gripper=0.2))
    assert robot_grasping(task, OBJ)
    task.robot = _Robot(_Arm(OBJ + [0, 0, 0.03], gripper=1.0))
    assert not robot_grasping(task, OBJ)
    task.robot = _Robot(_Arm(OBJ + [0, 0, 0.20], gripper=0.0))
    assert not robot_grasping(task, OBJ)


def test_metrics_report_not_fired_for_inactive_contract():
    task = _Task()
    m = interrupt_metrics(task)
    assert m["fired"] is False and m["not_fired_reason"] == "interrupt_contract_inactive"
    episode = new_interrupt_episode(task)
    episode.trigger.arm(np.zeros(3), {0: OBJ})
    block = legacy_pace_block(task)
    assert block["trigger_mode"] == "progress" and block["fired"] is False


def test_burned_draw_matches_legacy_randint_consumption():
    class T:
        EVAL_TRIGGER_STEP_MIN = 100
        EVAL_TRIGGER_STEP_MAX = 200
        config = {}

    np.random.seed(7)
    expected = int(np.random.randint(100, 201))
    tail = np.random.rand()
    np.random.seed(7)
    assert burn_legacy_trigger_draw(T()) == expected
    assert np.random.rand() == tail


# ----------------------------------------------------------------------
# Scripted rollout driver and full-state expert yield
# ----------------------------------------------------------------------


class _Controller:
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    def __init__(self):
        self.ticks = 0

    def on_scripted_rollout_tick(self, task):
        self.ticks += 1
        ep = task._interrupt
        ep.policy_steps += 1
        if ep.policy_steps == task.fire_at:
            ep.trigger.fired = True
            ep.human.phase = "glide_in"
        elif ep.human.phase == "glide_in" and ep.policy_steps >= task.fire_at + 3:
            ep.human.phase = "done"


class _ExpertTask(_Task):
    def __init__(self, fire_at):
        super().__init__()
        self.interaction_controller = _Controller()
        self.fire_at = fire_at
        self.sim = 0
        self.calls = []
        new_interrupt_episode(self)

    def step_sim(self):
        self.sim += 1

    def interrupt_expert_attempt(self):
        self.calls.append(("attempt", self.sim))
        for _ in range(500):
            self.step_sim()
        return True

    def interrupt_expert_retreat(self):
        self.calls.append(("retreat", self.sim))


def test_driver_ticks_at_policy_cadence():
    task = _ExpertTask(fire_at=10**9)
    with ScriptedRolloutDriver(task, tick_interval=25):
        for _ in range(250):
            task.step_sim()
    assert task.interaction_controller.ticks == 10
    # Restored afterwards.
    task.step_sim()
    assert task.interaction_controller.ticks == 10


def test_full_state_expert_yields_once_then_resumes():
    task = _ExpertTask(fire_at=8)
    driver = ScriptedRolloutDriver(task, tick_interval=25)
    with driver:
        ok = run_full_state_interrupt_expert(task, driver)
    assert ok
    kinds = [c[0] for c in task.calls]
    assert kinds == ["attempt", "retreat", "attempt"]
    assert task.calls[1][1] == 8 * 25  # yield raised at the firing tick
    assert interrupt_fired(task)
    assert task._interrupt.extra["expert_yielded_at_tick"] == 8


def test_full_state_expert_without_interrupt_runs_once():
    task = _ExpertTask(fire_at=10**9)
    driver = ScriptedRolloutDriver(task, tick_interval=25)
    with driver:
        assert run_full_state_interrupt_expert(task, driver)
    assert [c[0] for c in task.calls] == ["attempt"]


def test_interrupt_yield_is_not_swallowed_by_except_exception():
    def scripted():
        try:
            raise InterruptYield()
        except Exception:  # noqa: BLE001 - mirrors planner helpers
            return "swallowed"

    with pytest.raises(InterruptYield):
        scripted()


def test_glide_length_is_at_least_one_and_a_half_seconds():
    from envs.interaction.interrupt_progress import glide_steps_for_task

    class T20:
        config = {"action_substeps": 25}

        def _sim_dt(self):
            return 0.002

    class T40(T20):
        def _sim_dt(self):
            return 0.001

    assert glide_steps_for_task(T20()) == 30
    assert glide_steps_for_task(T40()) == 60
    t = T20()
    t.config = {"action_substeps": 25, "interrupt_glide_policy_steps": 12}
    assert glide_steps_for_task(t) == 12


def test_sequence_records_robot_diagnostics_at_trigger_glide_end_and_attach():
    # Metrics-only diagnostics: TCP-object distance and gripper command one
    # policy step after the trigger, when the glide ends, and at the attach
    # check; they never change the sequence's behaviour.
    task = _Task()
    arm = _Arm(OBJ + np.array([0.0, 0.0, 0.20]), gripper=1.0)
    task.robot = _Robot(arm)
    seq = InterruptHumanSequence(glide_steps=2)
    seq.start(step=5, start_root=np.zeros(3), target_root=np.zeros(3),
              rot=np.eye(3), target_key=0, target_pos=OBJ, start_clip=lambda: True)
    seq.tick(task, 6)
    arm._tcp[:3] = list(OBJ + np.array([0.0, 0.0, 0.05]))
    arm.gripper_val = 0.4
    seq.tick(task, 7)  # glide ends, clip starts
    motion = _Motion(attach_frame=10)
    task.avatar.motion_modules["Inspect1"] = motion
    seq.set_attach_guard(motion_name="Inspect1", object_pos=lambda: OBJ,
                         grasped=lambda: False)
    motion.at_frame = 9
    seq.on_sim_step(task)
    m = seq.metrics()
    diag = m["diag"]
    assert diag["after_trigger"]["step"] == 6
    assert np.isclose(diag["after_trigger"]["tcp_obj_dist_m"], 0.20)
    assert diag["after_trigger"]["gripper_cmd"] == 1.0
    assert diag["clip_start"]["step"] == 7
    assert np.isclose(diag["clip_start"]["tcp_obj_dist_m"], 0.05)
    assert np.isclose(diag["attach"]["tcp_obj_dist_m"], 0.05)
    assert diag["attach"]["gripper_cmd"] == 0.4
    assert np.isclose(m["diag_min_live_tcp_obj_dist_m"], 0.05)
    assert m["diag_near10cm_policy_steps"] == 1
    assert seq.attach_cancelled_reason is None
