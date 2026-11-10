"""Contracts for a post-exchange position-command experiment, not a force fix."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import take_hold


@pytest.fixture
def runtime(monkeypatch):
    clock = SimpleNamespace(steps=0, grips=[], arms=[], adjust=lambda s, n: s)
    q = np.arange(7) / 10
    arm = SimpleNamespace(gripper_val=.35, get_arm_qpos=lambda: q.copy(),
                          set_arm_joints=lambda v: clock.arms.append(v.copy()))

    def grip(value, tag):
        assert tag == "right"
        arm.gripper_val = value
        clock.grips.append(value)

    def step():
        clock.steps += 1

    task = SimpleNamespace(
        config={"take_post_handover_hold": True}, obj_name="073_rubikscube",
        plan_success=True, SIMULATION_DT=.001, step_sim=step,
        _avatar_retreat_enabled=lambda: False,
        robot=SimpleNamespace(get_arm=lambda _: arm, set_gripper=grip),
        avatar=SimpleNamespace(spare=lambda: True, robot=SimpleNamespace(
            attached_object=[None, None], two_hand_attached_object=None)),
    )

    def state(*_):
        return clock.adjust({"tcp": np.array([0., 0., 1., 1., 0., 0., 0.]),
                             "position": np.array([0., 0., .98]),
                             "quaternion": np.array([1., 0., 0., 0.])}, clock.steps)

    monkeypatch.setattr(take_hold, "_state", state)
    return task, arm, clock, q


def test_disabled_and_non_cube_are_no_ops():
    assert take_hold.establish_carry_hold(SimpleNamespace(config={}), "right")
    assert take_hold.establish_carry_hold(SimpleNamespace(
        config={"take_post_handover_hold": True}, obj_name="035_apple"), "right")


def test_only_position_targets_ramp_then_persist(runtime):
    task, arm, clock, q = runtime
    assert take_hold.establish_carry_hold(task, "right")
    assert task.plan_success
    assert clock.steps == 300
    assert len(clock.arms) == len(clock.grips) == 300
    assert all(np.array_equal(target, q) for target in clock.arms)
    assert np.all(np.diff(clock.grips) <= 0)
    assert clock.grips[0] == pytest.approx(.35 - .05 / 150)
    assert clock.grips[149:] == pytest.approx([.30] * 151)
    assert arm.gripper_val == pytest.approx(.30)


@pytest.mark.parametrize("dt,steps", [(.002, 150), (.005, 60), (.01, 30)])
def test_duration_is_physical_time(runtime, dt, steps):
    task, _, clock, _ = runtime
    task.SIMULATION_DT = dt
    assert take_hold.establish_carry_hold(task, "right")
    assert clock.steps == steps


@pytest.mark.parametrize("problem", ["prior_failure", "retreat", "checker", "busy_avatar",
                                    "attached", "two_hand", "wrong_command", "bad_q",
                                    "nan_q", "bad_dt", "robot_attachment"])
def test_invalid_entry_never_commands_or_steps(runtime, problem):
    task, arm, clock, _ = runtime
    if problem == "prior_failure":
        task.plan_success = False
    elif problem == "retreat":
        task._avatar_retreat_enabled = lambda: True
    elif problem == "checker":
        task.avatar_retreat_checker = object()
    elif problem == "busy_avatar":
        task.avatar.spare = lambda: False
    elif problem == "attached":
        task.avatar.robot.attached_object[1] = object()
    elif problem == "two_hand":
        task.avatar.robot.two_hand_attached_object = object()
    elif problem == "wrong_command":
        arm.gripper_val = .45
    elif problem == "bad_q":
        arm.get_arm_qpos = lambda: np.zeros(9)
    elif problem == "nan_q":
        arm.get_arm_qpos = lambda: np.full(7, np.nan)
    elif problem == "robot_attachment":
        task._gripper_attached = {"right": object()}
    else:
        task.SIMULATION_DT = np.nan
    assert not take_hold.establish_carry_hold(task, "right")
    assert not task.plan_success
    assert clock.steps == 0 and clock.grips == [] and clock.arms == []


@pytest.mark.parametrize("motion", ["slip", "object_rotation", "arm_translation", "arm_rotation"])
def test_measured_motion_aborts_without_opening(runtime, motion):
    task, _, clock, _ = runtime

    def adjust(state, step):
        if step >= 2:
            if motion == "slip":
                state["position"][2] -= .016
            elif motion == "object_rotation":
                state["quaternion"] = np.array([np.cos(.08), 0, 0, np.sin(.08)])
            elif motion == "arm_translation":
                state["tcp"][0] += .009
            else:
                state["tcp"][3:] = [np.cos(.026), 0, 0, np.sin(.026)]
        return state

    clock.adjust = adjust
    assert not take_hold.establish_carry_hold(task, "right")
    assert not task.plan_success and clock.steps == 2
    assert max(clock.grips) < .35


def test_existing_failure_latch_is_not_restored(runtime):
    task, _, clock, _ = runtime

    def step():
        clock.steps += 1
        task.plan_success = False

    task.step_sim = step
    assert not take_hold.establish_carry_hold(task, "right")
    assert clock.steps == 1 and not task.plan_success


def test_observation_error_stops_without_opening(runtime, monkeypatch):
    task, _, clock, _ = runtime

    def error(*_):
        raise ValueError("invalid observation")

    monkeypatch.setattr(take_hold, "_state", error)
    assert not take_hold.establish_carry_hold(task, "right")
    assert clock.steps == 0 and clock.grips == [] and not task.plan_success
