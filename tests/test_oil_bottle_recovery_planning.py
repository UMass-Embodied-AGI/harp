"""Scene-free tests for the proposed OBR endpoint gate."""
from types import SimpleNamespace

import numpy as np

from envs.planning.base import PlanResult
from envs.tasks.oil_bottle_recovery import OilBottleRecovery
from envs.task_bases.oil_recovery import OilFryingMotionClock
from envs.utils import Pose


def fixture(actual, *, plan_steps=2):
    calls = []
    plan = PlanResult(True, np.zeros((plan_steps, 7)), np.zeros((plan_steps, 7)))

    class NoSimulatorProbe:
        def __getattr__(self, name):
            raise AssertionError(f"FK must not access simulated entity.{name}")

    backend = SimpleNamespace(
        pad_move_group_qpos=lambda q: q,
        pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q: calls.append("fk"),
            get_link_pose=lambda _: actual,
        ),
        link_name_2_idx={"hand": 0}, move_group="hand",
        robot=SimpleNamespace(get_base_pose=lambda: Pose([0.0, 0.4, 0.0])),
    )
    planner = SimpleNamespace(
        _mplib=backend, _to_mplib_pose=Pose.from_pose7,
        plan_path=lambda *a, **kw: plan,
    )
    arm = SimpleNamespace(
        planner=planner, get_arm_qpos=lambda: np.zeros(7), entity=NoSimulatorProbe(),
    )
    task = OilBottleRecovery.__new__(OilBottleRecovery)
    task.config = {}
    task._refresh_planner_obstacles = lambda _: calls.append("refresh")
    task.step_sim = lambda: calls.append("step")
    return task, arm, plan, calls


def run(task, arm, **kwargs):
    return task._plan_path_with_fk_check(
        Pose([0.2, 0.4, 0.3]), "right", arm,
        max_fk_retries=1, max_snap_retries=1, snap_steps=1,
        **kwargs,
    )


def test_valid_endpoint_uses_pure_mplib_fk_in_robot_base_frame():
    task, arm, plan, calls = fixture(Pose([0.2, 0.0, 0.3]))
    result, error = run(task, arm)
    assert result is plan
    assert error < 1e-9
    assert calls == ["refresh", "fk"]


def test_exhausted_retries_never_return_best_but_inaccurate_plan():
    task, arm, _, calls = fixture(Pose([0.23, 0.0, 0.3]))
    result, error = run(task, arm, fk_tol=0.005)
    assert result is None
    assert np.isclose(error, 0.03)
    assert calls == ["refresh", "fk", "step"]


def test_correct_position_wrong_orientation_is_not_accepted():
    task, arm, _, calls = fixture(Pose([0.2, 0.0, 0.3], [0, 0, 0, 1]))
    result, error = run(task, arm)
    assert result is None
    assert error < 1e-9
    assert calls == ["refresh", "fk", "step"]


def test_held_object_branch_cap_rejects_without_fk_or_execution():
    task, arm, _, calls = fixture(Pose([0.2, 0.0, 0.3]), plan_steps=601)
    result, error = run(task, arm, max_plan_steps=600)
    assert result is None
    assert np.isinf(error)
    assert calls == ["refresh", "step"]


def test_scripted_frying_installs_clock_on_the_started_clip():
    task = OilBottleRecovery.__new__(OilBottleRecovery)
    motion = SimpleNamespace(at_frame=0)
    calls = []
    task.avatar = SimpleNamespace(frame_ratio=1, motion_modules={})
    def play(name, **kwargs):
        calls.append((name, task.avatar.frame_ratio, kwargs))
        task.avatar.motion_modules[name] = motion
    task.avatar.play_animation = play
    task._start_frying_animation()
    clock = task.avatar.motion_modules["frying"]
    assert isinstance(clock, OilFryingMotionClock)
    assert clock.motion is motion and clock.at_frame == 0
    assert calls == [("frying", 4, {"return_to_idle_after": 30})]
