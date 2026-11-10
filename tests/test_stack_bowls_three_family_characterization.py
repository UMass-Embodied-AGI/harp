"""Scene-free characterization contracts for the stack-bowls-three family.

Written before the family moves onto composed interaction controllers.  The
tests freeze seeded layouts and RNG tails, allocation attributes, hook
ordering, scripted choreography, the eval-mode yield machine, and the
metric surface at the public task level so the migration can be verified
without a simulator.  They deliberately avoid asserting which mixin or
controller owns the behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.interaction import LegacyInteractionController
from envs.interaction.stack_bowls_three import StackBowlsThreeAssistController
import envs.task_bases.stack_bowls_three as base_module
from envs.task_bases.stack_bowls_three import (
    StackBowlsThree,
    _MIN_SEP_M,
)
import envs.tasks.stack_three_bowls_contribution as assist_module
from envs.tasks.stack_three_bowls_contribution import StackThreeBowlsContribution
import envs.tasks.stack_three_bowls_interruption as interrupt_module
from envs.tasks.stack_three_bowls_interruption import (
    StackThreeBowlsInterruption,
    _BASE_X_HI,
    _BASE_X_LO,
    _BASE_Y_HI,
    _BASE_Y_LO,
    _INTERRUPT_FALLBACK_XY,
    _INTERRUPT_SAMPLE_ATTEMPTS,
    _LIFT_X_HI,
    _LIFT_X_LO,
    _LIFT_Y_HI,
    _LIFT_Y_LO,
    _REACH_X_HI,
    _REACH_X_LO,
    _REACH_Y_HI,
    _REACH_Y_LO,
)


SEEDS = (0, 8, 42)


class _FakeEntity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return self._quat.copy()


class _FakeActor:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.entity = _FakeEntity(pos, quat)


def _patch_actor_loading(monkeypatch, *modules):
    loads = []

    def load_object(_scene, pose, object_id, **kwargs):
        loads.append((object_id, np.asarray(pose.p, dtype=float).copy(), kwargs))
        return _FakeActor(pose.p, pose.q)

    for module in modules:
        monkeypatch.setattr(module, "load_object", load_object)
        if hasattr(module, "load_fixed_collision_bowl"):
            monkeypatch.setattr(module, "load_fixed_collision_bowl", load_object)
        monkeypatch.setattr(module, "_read_scale", lambda *_a, **_k: 0.5)
        monkeypatch.setattr(
            module, "_read_extents", lambda *_a, **_k: np.array([0.3, 0.1, 0.3])
        )
    return loads


def _legacy_interrupt_layout(rng, n_bowls):
    for _ in range(_INTERRUPT_SAMPLE_ATTEMPTS):
        pts = []
        ok = True
        for slot in range(n_bowls):
            if slot == 0:
                x = float(rng.uniform(_BASE_X_LO, _BASE_X_HI))
                y = float(rng.uniform(_BASE_Y_LO, _BASE_Y_HI))
            else:
                x = float(rng.uniform(_LIFT_X_LO, _LIFT_X_HI))
                y = float(rng.uniform(_LIFT_Y_LO, _LIFT_Y_HI))
            if any((x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2 for px, py in pts):
                ok = False
                break
            pts.append((x, y))
        if ok and len(pts) == n_bowls:
            return pts
    fallback = list(_INTERRUPT_FALLBACK_XY)
    fallback[1] = (_REACH_X_LO + 0.05, _REACH_Y_HI - 0.03)
    fallback[2] = (_REACH_X_HI - 0.05, _REACH_Y_LO + 0.03)
    return fallback[:n_bowls]


# ----------------------------------------------------------------------
# Assist (shared candidate pool; see also test_stack_bowls_three_assist_shared_pool)
# ----------------------------------------------------------------------


def _shared_assist_layout(task, rng):
    n_bowls = int(task._num_bowls)
    if task.simplified_mode_enabled():
        return [tuple(p) for p in task.SHARED_FIXED_XY[:n_bowls]]
    if n_bowls > 3:
        (x_lo, x_hi), (y_lo, y_hi) = task.HARD_SHARED_X, task.HARD_SHARED_Y
    else:
        (x_lo, x_hi), (y_lo, y_hi) = task.SHARED_X, task.SHARED_Y
    for _ in range(task.ASSIST_SAMPLE_ATTEMPTS):
        pts = []
        ok = True
        for i in range(n_bowls):
            if i == 0:
                x = float(rng.uniform(*task.BASE_X))
                y = float(rng.uniform(*task.BASE_Y))
            else:
                x = float(rng.uniform(x_lo, x_hi))
                y = float(rng.uniform(y_lo, y_hi))
                clearance = (task.HARD_CANDIDATE_HOME_CLEARANCE_M if n_bowls > 3
                             else task.CANDIDATE_HOME_CLEARANCE_M)
                if (np.hypot(x - task.ROBOT_MOUNT_XY[0], y - task.ROBOT_MOUNT_XY[1])
                        > task.ROBOT_REACH_M
                        or np.hypot(x - task.ROBOT_HOME_TCP_XY[0],
                                    y - task.ROBOT_HOME_TCP_XY[1]) < clearance):
                    ok = False
                    break
            if any((x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2 for px, py in pts):
                ok = False
                break
            pts.append((x, y))
        if ok and len(pts) == n_bowls:
            return pts
    fallback = task.HARD_FALLBACK_XY if n_bowls > 3 else task.SHARED_FIXED_XY
    return [tuple(p) for p in fallback[:n_bowls]]


@pytest.mark.parametrize("hard", (False, True))
@pytest.mark.parametrize("seed", SEEDS)
def test_assist_layout_sampler_is_one_shared_band_and_preserves_rng_tail(seed, hard):
    task = StackThreeBowlsContribution({"hard": hard})
    expected = _shared_assist_layout(task, np.random.RandomState(seed))
    expected_tail = np.random.RandomState(seed)
    _shared_assist_layout(task, expected_tail)
    rng = np.random.RandomState(seed)
    actual = task._sample_assist_xy(rng)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(rng.random_sample(4), expected_tail.random_sample(4))


@pytest.mark.parametrize("seed", SEEDS)
def test_assist_load_actors_shared_pool_and_global_rng_tail(seed, monkeypatch):
    loads = _patch_actor_loading(monkeypatch, assist_module)
    task = StackThreeBowlsContribution({"hard": True})
    task.scene = object()
    task.TABLE_TOP_Z = 0.765

    np.random.seed(seed)
    local_seed = int(np.random.randint(0, 2 ** 31 - 1))
    rng = np.random.RandomState(local_seed)
    positions = _shared_assist_layout(task, rng)
    yaws = [float(rng.uniform(-task.BOWL_YAW_LIMIT, task.BOWL_YAW_LIMIT)) for _ in range(4)]
    human = 1 + int(rng.randint(3))
    expected_tail = np.random.random_sample(6)

    np.random.seed(seed)
    task.load_actors()
    actual_tail = np.random.random_sample(6)

    assert len(task.bowls) == 4 and len(loads) == 4
    for (object_id, pos, _kwargs), (cx, cy) in zip(loads, positions):
        assert object_id == base_module._BOWL_ID
        np.testing.assert_allclose(pos[:2], [cx, cy])
        assert pos[2] == pytest.approx(0.765 + task.BOWL_SPAWN_Z_CLEARANCE)
    assert task._spawn_xy == [tuple(p) for p in positions]
    assert task._bowl_height == pytest.approx(0.1 * 0.5)
    assert task._stacked_indices == [1, 2, 3]
    assert task._shared_candidates == [1, 2, 3]
    assert task._human_target == human
    assert task._shared_pool == {"human_target": human, "robot_target": None}
    assert not hasattr(task, "_robot_bowl_indices")
    assert len(yaws) == 4
    np.testing.assert_array_equal(actual_tail, expected_tail)


class _FakeAssistAvatar:
    def __init__(self):
        self.calls = []

    def pick_and_place(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(_hand_id=1)

    def spare(self):
        return True


def _shared_task(positions, *, hard=False):
    task = StackThreeBowlsContribution({"hard": hard})
    task.TABLE_TOP_Z = 0.765
    task._bowl_height = 0.05
    task._bowl_scale = 1.0
    task._bowl_extents = np.array([0.114, 0.05, 0.114])
    task.bowls = [_FakeActor(p) for p in positions]
    task._num_bowls = len(positions)
    StackBowlsThreeAssistController.allocate(task)
    return task


def test_assist_start_avatar_assist_parameters_and_collision_reset():
    task = _shared_task([
        [0.0, -0.05, 0.77], [0.15, -0.10, 0.77], [-0.10, 0.0, 0.77],
    ])
    task._human_target = 2
    task.avatar = _FakeAssistAvatar()
    task.avatar_collided = True
    task.avatar_collision_log = ["stale"]

    assert task._start_avatar_assist() is True
    call = task.avatar.calls[0]
    np.testing.assert_allclose(call["pick_pos"], [-0.10, 0.0, 0.77 + task.AVATAR_PICK_LIFT])
    np.testing.assert_allclose(
        call["place_pos"],
        [0.0 + task.AVATAR_DROP_XY_OFFSET[0], -0.05 + task.AVATAR_DROP_XY_OFFSET[1],
         0.765 + 2.0 * 0.05 + task.AVATAR_DROP_CLEARANCE_M],
    )
    assert call["attach_obj"] is task.bowls[2].entity
    assert call["hand_id"] == 0
    assert call["approach_frames"] == task.AVATAR_APPROACH_FRAMES
    assert call["transport_frames"] == task.AVATAR_TRANSPORT_FRAMES
    assert call["retract_frames"] == task.AVATAR_RETRACT_FRAMES
    assert call["natural"] is True
    assert call["body_margin"] == task.AVATAR_BODY_MARGIN
    assert call["body_x_bounds"] == task.AVATAR_BODY_X_BOUNDS
    assert call["body_y_bounds"] == (task.AVATAR_MIN_BODY_Y, task.AVATAR_MAX_BODY_Y)
    assert call["yaw_limit_deg"] == task.AVATAR_YAW_LIMIT_DEG
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []
    assert task._human_target == 2

    task.avatar = None
    assert task._start_avatar_assist() is False


class _SpareSequenceAvatar:
    def __init__(self, values):
        self._values = iter(values)

    def spare(self):
        return next(self._values, True)


def test_assist_scripted_rollout_starts_avatar_waits_then_stacks_the_other_bowls():
    task = _shared_task([
        [0.0, 0.0, 0.77], [0.2, 0.0, 0.77], [-0.2, 0.0, 0.77], [0.0, 0.2, 0.77],
    ], hard=True)
    task.INITIAL_SETTLE_STEPS = 2
    task.FINAL_SETTLE_STEPS = 2
    task.avatar = _SpareSequenceAvatar([False, True])
    events = []

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    def start():
        events.append(("avatar",))
        task._human_target = 2
        return True

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim",))
    task._start_avatar_assist = start
    task._pick_and_stack_one = lambda idx, target: (
        events.append(("stack", idx, tuple(round(v, 6) for v in target))), True
    )[1]

    assert task.play_once() is True
    labels = [event[0] for event in events]
    # The full-state expert yields: it steps until the human is idle (one
    # non-spare check here), waits one rest window for the bowls (static
    # here), then stacks, then settles.
    assert labels == [
        "avatar", "suppress", "sim", "sim", "suppress", "sim",
        *["sim"] * task.BOWL_REST_WINDOW_STEPS, "stack", "stack",
        "sim", "sim",
    ]
    stacks = [event for event in events if event[0] == "stack"]
    assert stacks[0][1] == 1 and stacks[1][1] == 3
    # Full state releases one nest rise over the observed (final) stack top;
    # the stubbed stack never moves, so both targets sit over the base.
    nest = round(0.77 + task.FULL_STATE_NEST_RISE_M, 6)
    assert stacks[0][2] == (0.0, 0.0, nest)
    assert stacks[1][2] == (0.0, 0.0, nest)
    assert task._shared_pool["robot_target"] == 1


def test_assist_eval_hooks_fire_once_and_record_failure():
    task = StackThreeBowlsContribution({"eval_mode": True})
    starts = []
    task._start_avatar_assist = lambda: starts.append(1) or True

    task._eval_at_reset()
    assert task._eval_assist_trigger_step == 0
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is False
    assert starts == [1]
    task._eval_at_step(99)
    assert starts == [1]

    task = StackThreeBowlsContribution({"eval_mode": True})
    task._eval_assist_trigger_step = 5
    task._eval_assist_fired = False
    task._eval_assist_failed = False
    task._start_avatar_assist = lambda: False
    task._eval_at_step(4)
    assert task._eval_assist_fired is False
    task._eval_at_step(5)
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is True

    fresh = StackThreeBowlsContribution({"eval_mode": True})
    fresh._start_avatar_assist = lambda: pytest.fail("fired below default trigger")
    fresh._eval_at_step(fresh.EVAL_TRIGGER_STEP_MIN - 1)


def test_assist_success_gate_and_evaluate_surface(monkeypatch):
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": 1})
    monkeypatch.setattr(StackBowlsThree, "check_success", lambda _self: True)
    task = StackThreeBowlsContribution({"eval_mode": True})
    task._stacked_indices = [1, 2]
    task._bowl_height = 0.05
    heights = {0: 0.77, 1: 0.785, 2: 0.80}
    task._bowl_xyz = lambda i: np.array([0.0, 0.0, heights[i]])
    task.avatar_collided = True
    assert task.check_success() is True
    task.config["track_avatar_collision"] = True
    assert task.check_success() is False
    task.avatar_collided = False
    assert task.check_success() is True
    arm = SimpleNamespace(gripper_val=1.0,
                          get_ee_pose=lambda: np.array([0.0, 0.0, 0.84, 1.0, 0.0, 0.0, 0.0]))
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    assert task.check_success() is True        # open gripper near the stack
    arm.gripper_val = 0.0                      # bowl 2 still pinched
    assert task.check_success() is False

    task._num_bowls = 3
    StackBowlsThreeAssistController.allocate(task)
    task._eval_assist_trigger_step = 0
    task._eval_assist_fired = True
    task._eval_assist_failed = False
    task._eval_policy_step_count = 7
    task._human_target = 2
    task._shared_pool.update(human_target=2, robot_target=2)
    metrics = task.evaluate()
    pool = metrics.pop("shared_pool")
    assert metrics == {
        "base": 1,
        "eval_mode": True,
        "eval_assist_trigger_step": 0,
        "eval_assist_fired": True,
        "eval_assist_failed": False,
        "eval_policy_step_count": 7,
        "assist_avatar_bowl_idx": 2,
    }
    assert pool["same_target"] is True
    assert pool == {"human_target": 2, "robot_target": 2, "same_target": True}


# ----------------------------------------------------------------------
# Interrupt
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_interrupt_load_actors_seed_tied_choices_layout_and_rng_tail(seed, monkeypatch):
    loads = _patch_actor_loading(monkeypatch, interrupt_module)
    task = StackThreeBowlsInterruption({"hard": True})
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    task._episode_seed = seed

    rng_top = np.random.RandomState(seed + task.INTERRUPT_SEED_OFFSET)
    expected_idx = int(rng_top.choice([1, 2]))
    expected_layer = int(rng_top.choice([2, 3]))

    np.random.seed(seed)
    local_seed = int(np.random.randint(0, 2 ** 31 - 1))
    rng = np.random.RandomState(local_seed)
    positions = _legacy_interrupt_layout(rng, 3)
    expected_tail = np.random.random_sample(6)

    np.random.seed(seed)
    task.load_actors()
    actual_tail = np.random.random_sample(6)

    assert task._interrupted_idx == expected_idx
    assert task._interrupt_stack_layer == expected_layer
    assert len(loads) == 3
    for (_object_id, pos, _kwargs), (cx, cy) in zip(loads, positions):
        np.testing.assert_allclose(pos[:2], [cx, cy])
    assert task._stacked_indices == [1, 2]
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_interrupt_simplified_layout_fixes_target_and_layer(monkeypatch):
    _patch_actor_loading(monkeypatch, interrupt_module)
    task = StackThreeBowlsInterruption({})
    task.scene = object()
    task.TABLE_TOP_Z = 0.765
    np.random.seed(3)
    task.load_actors()
    assert task._interrupted_idx == 1
    assert task._interrupt_stack_layer == 2
    assert len(task.bowls) == task._num_bowls == 2


def _run_reset_flow(task, seed, monkeypatch, observation):
    """Drive the family reset boundary for both the legacy wrapper and the
    composed lifecycle hook."""

    def fake_base_reset(self, seed=0):
        np.random.seed(seed)
        self._episode_seed = int(seed)
        return observation

    if isinstance(task.interaction_controller, LegacyInteractionController):
        monkeypatch.setattr(StackBowlsThree, "reset", fake_base_reset)
        return task.reset(seed=seed)
    result = fake_base_reset(task, seed=seed)
    task.interaction_controller.on_episode_seeded(task, seed)
    task.interaction_controller.after_base_reset_observation(
        task, observation=result,
    )
    return result


class _EvalArm:
    gripper_val = 1.0

    def __init__(self):
        self.tcp = np.array([0.2, -0.4, 1.1])
        self.qpos = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def get_arm_qpos(self):
        return self.qpos.copy()

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0.0, 0.0, 0.0]])


class _EvalRobot:
    def __init__(self):
        self.arm = _EvalArm()
        self.joints = []
        self.grippers = []

    def get_arm(self, _tag):
        return self.arm

    def set_arm_joints(self, cmd, _tag):
        self.joints.append(np.asarray(cmd).copy())

    def set_gripper(self, value, _tag):
        self.grippers.append(float(value))


class _EvalAvatarRobot:
    def __init__(self):
        self.global_trans = np.array([0.0, 0.55, -0.18])

    def get_palm_center(self, _hand):
        return self.global_trans + np.array([0.0, 0.0, 1.0])


class _EvalAvatar:
    def __init__(self):
        self.motions = []
        self.resets = []
        self._spare = True
        self.frame_ratio = None
        self.robot = _EvalAvatarRobot()
        self.motion_modules = {}

    def play_bowl_motion(self, **kwargs):
        self.motions.append(kwargs)
        self._spare = False

    def reset(self, pos, rot):
        self.robot.global_trans = np.asarray(pos, dtype=float).copy()
        self.resets.append(self.robot.global_trans.copy())

    def spare(self):
        return self._spare


def _eval_scene(task):
    task.TABLE_TOP_Z = 0.765
    task._bowl_height = 0.05
    task._bowl_scale = 0.5
    task._bowl_extents = np.array([0.3, 0.1, 0.3])
    task.bowls = [
        _FakeActor([0.0, 0.0, 0.77]),
        _FakeActor([0.6, 0.0, 0.77]),
        _FakeActor([0.3, 0.1, 0.77]),
    ]
    task.robot = _EvalRobot()
    task.avatar = _EvalAvatar()
    task._dryrun_capture_bowl_frame = lambda: np.array([0.0, 0.3, 0.9])
    return task


@pytest.mark.parametrize("eval_mode", (False, True))
def test_interrupt_reset_calibrates_then_arms_eval_state(eval_mode, monkeypatch):
    # Reset still calibrates the idle human and keeps the retired
    # trigger draw (burned) so the RNG tail is unchanged; eval mode now arms
    # the shared progress trigger instead of the retreat/return machine.
    task = _eval_scene(StackThreeBowlsInterruption({
        "eval_mode": eval_mode,
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
    }))
    events = []
    task._calibrate_avatar_for_bowl_motion = lambda: events.append("calibrate")
    observation = {"obs": 1}

    returned = _run_reset_flow(task, 42, monkeypatch, observation)
    tail = np.random.random_sample(4)

    reference = np.random.RandomState(42)
    expected_trigger = int(reference.randint(7, 12)) if eval_mode else None
    expected_tail = reference.random_sample(4)

    assert returned is observation
    assert task._episode_seed == 42
    assert events == ["calibrate"]
    assert task.avatar.frame_ratio == task.AVATAR_MOTION_SLOW
    assert task._eval_policy_step_count == 0
    assert task._eval_interrupt_fired is False
    assert task._eval_interrupt_failed is False
    assert task._eval_trigger_step == expected_trigger
    np.testing.assert_array_equal(tail, expected_tail)
    if eval_mode:
        episode = task._interrupt
        assert episode.trigger.armed and not episode.fired
        assert set(episode.trigger.initial_dist) == {1, 2}
        assert episode.trigger.params.commit_radius_m == 0.24
    else:
        assert getattr(task, "_interrupt", None) is None


def test_interrupt_reset_without_avatar_skips_calibration(monkeypatch):
    # The pace_no_human_ablation path was removed.
    task = StackThreeBowlsInterruption({"eval_mode": True, "pace_no_human_ablation": True})
    task.avatar = None
    task._calibrate_avatar_for_bowl_motion = lambda: pytest.fail("calibrated without avatar")
    _run_reset_flow(task, 5, monkeypatch, "obs")
    assert not hasattr(task, "_eval_interrupt_fired")


def test_interrupt_take_action_ticks_then_applies_policy_action_unmodified(monkeypatch):
    # The policy action is never substituted.
    task = StackThreeBowlsInterruption({"eval_mode": True})
    task.avatar = object()
    events = []
    task._eval_step_avatar = lambda: events.append(("tick",))

    def base_action(_self, action, action_type="qpos"):
        events.append(("robot", np.asarray(action), action_type))
        return "obs"

    monkeypatch.setattr(StackBowlsThree, "take_action", base_action)
    assert task.take_action(np.ones(8), action_type="qpos") == "obs"
    assert [event[0] for event in events] == ["tick", "robot"]
    np.testing.assert_array_equal(events[-1][1], np.ones(8))
    assert events[-1][2] == "qpos"
    assert task.eval_skip_policy_action() is False
    assert task.consume_eval_policy_reset_requested() is False

    task.avatar = None
    events.clear()
    assert task.take_action(np.ones(8), action_type="qpos") == "obs"
    assert [event[0] for event in events] == ["robot"]


def _armed_interrupt_task(**cfg):
    task = _eval_scene(StackThreeBowlsInterruption({"eval_mode": True, **cfg}))
    task._calibrate_avatar_for_bowl_motion = lambda: None
    task._episode_seed = 0
    task.interaction_controller.after_base_reset_observation(task, observation=None)
    return task


def test_interrupt_eval_progress_trigger_glides_then_plays_bowl_clip():
    # Replaces the random-step/force_dist trigger and the env-forced
    # retreat -> hold -> return machine.
    task = _armed_interrupt_task()
    home_root = task.avatar.robot.global_trans.copy()
    for _ in range(25):  # robot never moves: no interrupt, ever
        task._eval_step_avatar()
    assert task._eval_interrupt_fired is False
    assert task._interrupt.trigger.block_reason == "robot_not_moved"

    # Robot commits to bowl 2 (closer than bowl 1 at that moment).
    task.robot.arm.tcp = task._interrupt_controller().bowl_grasp_center(task, 2) + [0, 0, 0.12]
    task._eval_step_avatar()
    assert task._eval_interrupt_fired is True
    assert task._interrupted_idx == 2
    assert task._interrupt.trigger.trigger_target == 2
    assert task.avatar.motions == []  # glide first
    glide = task._interrupt.human.glide_steps
    for _ in range(glide):
        task._eval_step_avatar()
    resets = np.asarray(task.avatar.resets)
    assert len(resets) == glide
    expected_target = (
        np.asarray(task.avatar_init_pos) + np.array([0.3, 0.1, 0.77]) - [0.0, 0.3, 0.9]
    )
    np.testing.assert_allclose(resets[-1], expected_target)
    jumps = np.linalg.norm(np.diff(np.vstack([home_root, resets]), axis=0), axis=1)
    assert jumps.max() <= np.linalg.norm(expected_target - home_root) / glide + 1e-9
    assert len(task.avatar.motions) == 1
    motion = task.avatar.motions[0]
    assert motion["attach_obj"] is task.bowls[2].entity
    assert motion["snap_to_hands"] is False
    assert motion["return_to_idle_after"] == task._interrupt.human.idle_return_frames

    task._eval_step_avatar()  # clip still running
    assert task._interrupt.human.phase == "clip"
    task.avatar._spare = True
    task._eval_step_avatar()
    assert task._interrupt.human.phase == "glide_out"
    for _ in range(task._interrupt.human.glide_out_steps):
        task._eval_step_avatar()
    assert task._interrupt.human.done
    np.testing.assert_allclose(task.avatar.robot.global_trans, home_root)
    assert task.robot.joints == [] and task.robot.grippers == []  # no override


def test_interrupt_eval_stacked_or_held_bowls_are_not_targets():
    task = _armed_interrupt_task()
    ctrl = task._interrupt_controller()
    task.bowls[1] = _FakeActor([0.0, 0.0, 0.82])  # stacked on the base
    task.robot.arm.gripper_val = 0.0               # closed on bowl 2
    task.robot.arm.tcp = ctrl.bowl_grasp_center(task, 2) + [0.0, 0.0, 0.03]
    assert ctrl.eligible_bowls(task) == {}
    task.robot.arm.gripper_val = 1.0
    assert set(ctrl.eligible_bowls(task)) == {2}


def _scripted_interrupt_task(layer):
    task = StackThreeBowlsInterruption({"hard": True})
    task.INITIAL_SETTLE_STEPS = 1
    task.FINAL_SETTLE_STEPS = 1
    task._bowl_height = 0.05
    task.bowls = [
        _FakeActor([0.0, 0.0, 0.77]),
        _FakeActor([0.2, 0.0, 0.77]),
        _FakeActor([-0.2, 0.0, 0.77]),
    ]
    task._interrupted_idx = 2
    task._interrupt_stack_layer = layer
    task.avatar = object()
    events = []

    @contextmanager
    def suppress_recording():
        events.append(("suppress",))
        yield

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim",))
    task._run_avatar_interrupt = lambda idx, arm: events.append(("interrupt", idx, arm)) or True
    task._pick_and_stack_one = lambda idx, target: (
        events.append(("stack", idx, tuple(round(v, 6) for v in target))), True
    )[1]
    return task, events


def test_interrupt_scripted_choreography_layer_two_interrupts_first_stack():
    task, events = _scripted_interrupt_task(2)
    assert task.play_once() is True
    assert [e for e in events if e[0] != "sim" and e[0] != "suppress"] == [
        ("interrupt", 2, "right"),
        ("stack", 2, (0.0, 0.0, round(0.77 + 0.05, 6))),
        ("stack", 1, (0.0, 0.0, round(0.77 + 0.05, 6))),
    ]


def test_interrupt_scripted_choreography_layer_three_interrupts_last_stack():
    task, events = _scripted_interrupt_task(3)
    assert task.play_once() is True
    assert [e for e in events if e[0] != "sim" and e[0] != "suppress"] == [
        ("stack", 1, (0.0, 0.0, round(0.77 + 0.05, 6))),
        ("interrupt", 2, "right"),
        ("stack", 2, (0.0, 0.0, round(0.77 + 0.05, 6))),
    ]


def test_interrupt_success_gates_and_evaluate_surface(monkeypatch):
    # The retreat/wait/return success gates are gone and the
    # metric surface reports the shared ``interrupt`` block plus a
    # progress-mode ``pace_interrupt`` compatibility block.
    monkeypatch.setattr(BaseTask, "evaluate", lambda _self: {"base": 1})
    monkeypatch.setattr(StackBowlsThree, "check_success", lambda _self: True)
    task = _armed_interrupt_task()
    task.avatar_collided = False
    assert task.check_success() is True
    task.avatar_collided = True
    assert task.check_success() is False

    task.avatar_collided = False
    metrics = task.evaluate()
    assert tuple(metrics) == (
        "base", "eval_mode", "eval_policy_step_count",
        "eval_interrupt_fired", "eval_interrupt_failed", "interrupted_bowl_idx",
        "interrupt", "pace_interrupt",
    )
    assert metrics["interrupt"]["fired"] is False
    assert metrics["interrupt"]["not_fired_reason"] == "not_armed" or \
        metrics["interrupt"]["not_fired_reason"] == "min_policy_steps"
    assert metrics["pace_interrupt"]["trigger_mode"] == "progress"
