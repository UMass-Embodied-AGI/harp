"""Scene-free contracts for observing a take-task release to completion.

Scripted readings model object motion only; no Genesis scene or robot is
created, and the production settling method is called without replacement.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.tasks.take_from_human_easy import TakeFromHumanEasy


TABLE_TOP = 0.765
HALF_OBJECT = np.array([0.02, 0.02, 0.02])
PLATE_AABB = np.array([[-0.10, -0.10, TABLE_TOP],
                       [0.10, 0.10, TABLE_TOP + 0.02]])


def _state(position=(0.20, 0.0, TABLE_TOP + 0.02), quaternion=(1.0, 0.0, 0.0, 0.0)):
    position = np.asarray(position, dtype=float)
    return {
        "position": position,
        "quaternion": np.asarray(quaternion, dtype=float),
        "aabb": np.stack([position - HALF_OBJECT, position + HALF_OBJECT]),
    }


class _ScriptedEntity:
    def __init__(self, clock, state_at_step):
        self.clock = clock
        self.state_at_step = state_at_step

    def _read(self, key):
        return np.asarray(self.state_at_step(self.clock.steps)[key], dtype=float).copy()

    def get_pos(self):
        return self._read("position")

    def get_quat(self):
        return self._read("quaternion")

    def get_AABB(self):
        return self._read("aabb")


def _task(state_at_step=None, *, dt=0.001, plate_aabb=None):
    task = TakeFromHumanEasy.__new__(TakeFromHumanEasy)
    clock = SimpleNamespace(steps=0)
    task.SIMULATION_DT = dt
    task.TABLE_TOP_Z = TABLE_TOP
    task._table_center_xy = np.array([0.0, 0.0])
    task._table_half_size_xy = np.array([0.5, 0.5])
    task.apple_actor = SimpleNamespace(
        entity=_ScriptedEntity(clock, state_at_step or (lambda _: _state()))
    )
    plate = PLATE_AABB if plate_aabb is None else plate_aabb
    task.plate_actor = SimpleNamespace(
        entity=SimpleNamespace(get_AABB=lambda: np.asarray(plate, dtype=float).copy())
    )
    task.plan_success = True
    task._release_settled = False
    task._release_settle_diagnostics = None

    def step_sim():
        clock.steps += 1

    task.step_sim = step_sim
    return task, clock


def _assert_failed(task, clock):
    assert task._settle_released_object() is False
    assert task._release_settled is False
    assert task.plan_success is False
    diagnostics = task._release_settle_diagnostics
    assert diagnostics["steps"] == clock.steps
    assert 0 < clock.steps <= diagnostics["budget"]
    return diagnostics


def test_stationary_airborne_object_does_not_count_as_settled():
    task, clock = _task(lambda _: _state((0.20, 0.0, TABLE_TOP + 0.32)))
    diagnostics = _assert_failed(task, clock)
    assert diagnostics["supported"] is False
    assert diagnostics["linear_speed"] == pytest.approx(0.0)
    assert diagnostics["angular_speed"] == pytest.approx(0.0)
    assert clock.steps * task.SIMULATION_DT > 0.75


@pytest.mark.parametrize("dt", [0.001, 0.002, 0.004])
@pytest.mark.parametrize("support", ["table", "plate"])
def test_quiet_supported_object_requires_five_consecutive_samples(dt, support):
    position = ((0.20, 0.0, TABLE_TOP + 0.02) if support == "table"
                else (0.0, 0.0, PLATE_AABB[1, 2] + 0.02))
    task, clock = _task(lambda _: _state(position), dt=dt)
    assert task._settle_released_object() is True
    assert clock.steps == 5 * max(1, round(0.02 / dt))
    assert clock.steps * dt == pytest.approx(0.1)
    assert task.plan_success is True
    assert task._release_settled is True
    assert task._release_settle_diagnostics["supported"] is True
    assert task._release_settle_diagnostics["support_z"] == pytest.approx(
        TABLE_TOP if support == "table" else PLATE_AABB[1, 2]
    )


def test_motion_resets_the_consecutive_quiet_sample_count():
    # Three quiet samples, then a 4 mm displacement at sample four. Only
    # five further quiet samples can complete the wait (at 180 ms).
    task, clock = _task(lambda step: _state(
        (0.20 + (0.004 if step >= 80 else 0.0), 0.0, TABLE_TOP + 0.02)
    ))
    assert task._settle_released_object() is True
    assert clock.steps == 180


def test_falling_object_is_observed_past_the_old_150_ms_tail():
    def falling(step):
        time = step * 0.001
        bottom = TABLE_TOP + max(0.0, 0.30 - 0.5 * 9.81 * time * time)
        return _state((0.20, 0.0, bottom + 0.02))

    task, clock = _task(falling)
    assert task._settle_released_object() is True
    assert clock.steps > 150
    assert clock.steps >= 340
    assert task._release_settle_diagnostics["bottom_z"] == pytest.approx(TABLE_TOP)
    assert task._release_settle_diagnostics["linear_speed"] == pytest.approx(0.0)


@pytest.mark.parametrize("dt", [0.001, 0.002, 0.004])
def test_post_landing_rotation_gets_time_to_complete_the_same_quiet_window(dt):
    # Failures hit a 0.75s deadline on the plate. Model
    # residual rotation lasting 0.9s, then require the original five quiet
    # 20ms samples; a single quiet endpoint must never suffice.
    def readings(step):
        angle = min(step * dt, 0.9)
        return _state(quaternion=(np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)))

    task, clock = _task(readings, dt=dt)
    assert task._settle_released_object() is True
    assert clock.steps * dt == pytest.approx(1.0)
    assert task._release_settle_diagnostics["budget"] * dt == pytest.approx(2.0)


def test_timeout_does_not_accept_an_incomplete_quiet_window():
    def readings(step):
        angle = min(step * 0.001, 1.94)
        return _state(quaternion=(np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)))

    task, clock = _task(readings)
    diagnostics = _assert_failed(task, clock)
    assert clock.steps == diagnostics["budget"] == 2000
    assert diagnostics["angular_speed"] == pytest.approx(0.0, abs=2e-6)


def test_successful_settling_never_clears_an_earlier_planning_failure():
    task, clock = _task()
    task.plan_success = False
    assert task._settle_released_object() is True
    assert task.plan_success is False
    assert clock.steps == 100


@pytest.mark.parametrize("condition", ["moving", "spinning", "floor", "off_table"])
def test_nonsettled_or_unsupported_objects_fail(condition):
    def readings(step):
        time = step * 0.001
        if condition == "moving":
            return _state((0.20 + 0.10 * time, 0.0, TABLE_TOP + 0.02))
        if condition == "spinning":
            return _state(quaternion=(np.cos(time / 2), 0.0, 0.0, np.sin(time / 2)))
        if condition == "floor":
            return _state((0.20, 0.0, 0.02))
        # Center lies on-table but the object's full footprint does not.
        return _state((0.49, 0.0, TABLE_TOP + 0.02))

    task, clock = _task(readings)
    diagnostics = _assert_failed(task, clock)
    if condition == "moving":
        assert diagnostics["linear_speed"] == pytest.approx(0.10)
    elif condition == "spinning":
        assert diagnostics["angular_speed"] == pytest.approx(1.0)
    else:
        assert diagnostics["supported"] is False


@pytest.mark.parametrize("scale_changes", [False, True])
def test_quaternion_sign_flips_do_not_look_like_rotation(scale_changes):
    quaternion = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])

    def readings(step):
        sample = step // 20
        sign = (-1.0) ** sample
        scale = (1.0 + sample % 3) if scale_changes else 1.0
        return _state(quaternion=quaternion * sign * scale)

    task, clock = _task(readings)
    assert task._settle_released_object() is True
    assert clock.steps == 100
    assert task._release_settle_diagnostics["angular_speed"] == pytest.approx(0.0, abs=2e-6)


@pytest.mark.parametrize("field", ["position", "quaternion", "aabb", "plate_aabb"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_observed_state_fails_closed(field, invalid):
    readings = _state()
    plate = PLATE_AABB.copy()
    if field == "plate_aabb":
        plate.flat[0] = invalid
    else:
        readings[field].flat[0] = invalid
    task, clock = _task(lambda _: readings, plate_aabb=plate)
    diagnostics = _assert_failed(task, clock)
    assert diagnostics["supported"] is False
    assert diagnostics["reason"] == "invalid_object_state"
    assert clock.steps == 20


def test_zero_length_quaternion_fails_closed():
    task, clock = _task(lambda _: _state(quaternion=(0.0, 0.0, 0.0, 0.0)))
    diagnostics = _assert_failed(task, clock)
    assert diagnostics["reason"] == "invalid_object_state"


@pytest.mark.parametrize("field", ["position", "quaternion", "aabb"])
def test_initial_invalid_reading_is_not_erased_by_a_later_valid_sample(field):
    def readings(step):
        state = _state()
        if step == 0:
            state[field].flat[0] = np.nan
        return state

    task, clock = _task(readings)
    assert _assert_failed(task, clock)["reason"] == "invalid_object_state"


def test_initial_inverted_aabb_is_not_erased_by_a_later_valid_sample():
    def readings(step):
        state = _state()
        if step == 0:
            state["aabb"][:, 2] = state["aabb"][::-1, 2].copy()
        return state

    task, clock = _task(readings)
    assert _assert_failed(task, clock)["reason"] == "invalid_object_state"
    assert clock.steps == 20


@pytest.mark.parametrize("entity_name", ["object", "plate"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_inverted_aabb_fails_closed(entity_name, axis):
    readings = _state()
    plate = PLATE_AABB.copy()
    aabb = readings["aabb"] if entity_name == "object" else plate
    aabb[:, axis] = aabb[::-1, axis].copy()
    task, clock = _task(lambda _: readings, plate_aabb=plate)
    assert _assert_failed(task, clock)["reason"] == "invalid_object_state"
    assert clock.steps == 20


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("half_size", [0.0, -0.5])
def test_nonpositive_table_half_size_fails_closed(axis, half_size):
    task, clock = _task()
    task._table_half_size_xy[axis] = half_size
    assert _assert_failed(task, clock)["reason"] == "invalid_object_state"
    assert clock.steps == 20


@pytest.mark.parametrize("field", ["_table_center_xy", "_table_half_size_xy"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_support_geometry_fails_closed(field, invalid):
    task, clock = _task()
    getattr(task, field)[0] = invalid
    assert _assert_failed(task, clock)["supported"] is False


def _evaluation_task(monkeypatch, *, eval_mode, recorder):
    # The task's wrapper must only append optional diagnostics; the shared
    # evaluation result and recorder must not be rewritten by this wrapper.
    base = {"plan_success": True, "success": True, "unchanged_metric": 7}
    monkeypatch.setattr(BaseTask, "evaluate", lambda self: dict(base))
    task = TakeFromHumanEasy.__new__(TakeFromHumanEasy)
    task.config = {"eval_mode": eval_mode}
    task.vla_recorder = recorder
    task.obj_name = "073_rubikscube"
    task.obj_model_id = 1
    task._eval_apple_attached_to_avatar = False
    task._eval_apple_support_active = True
    return task, base


@pytest.mark.parametrize("eval_mode", [False, True])
@pytest.mark.parametrize("reset_fields", [False, True])
def test_evaluate_omits_release_fields_without_an_attempt(monkeypatch, eval_mode, reset_fields):
    recorder = SimpleNamespace(mark_episode_end=lambda *_: pytest.fail("evaluate changed VLA end"))
    task, base = _evaluation_task(monkeypatch, eval_mode=eval_mode, recorder=recorder)
    if reset_fields:
        task._release_settled = False
        task._release_settle_diagnostics = None
    before_keys = set(vars(task))
    result = task.evaluate()
    assert result == {
        **base,
        "object_name": "073_rubikscube",
        "object_model_id": 1,
        "eval_apple_attached_to_avatar": False,
        "eval_apple_support_active": True,
    }
    assert set(vars(task)) == before_keys
    assert task.vla_recorder is recorder


@pytest.mark.parametrize("settled", [False, True])
def test_evaluate_adds_release_diagnostics_without_overriding_base_metrics(monkeypatch, settled):
    task, base = _evaluation_task(monkeypatch, eval_mode=True, recorder=None)
    diagnostics = {"steps": 100, "supported": settled}
    task._release_settled = settled
    task._release_settle_diagnostics = diagnostics
    result = task.evaluate()
    assert {key: result[key] for key in base} == base
    assert result["release_settled"] is settled
    assert result["release_settle_diagnostics"] == diagnostics
    assert task._release_settle_diagnostics == {"steps": 100, "supported": settled}
    assert task.vla_recorder is None
