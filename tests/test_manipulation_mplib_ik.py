"""The shared seeded IK helpers (envs/manipulation.py) use MPLib IK only."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mplib")

from envs.manipulation import TopDownPickPlaceMixin  # noqa: E402
from envs.planning.mplib_planner import MplibPlanner  # noqa: E402
from envs.utils import Pose  # noqa: E402

FRANKA = Path(__file__).resolve().parents[1] / "assets" / "embodiments" / "franka-panda"
BASE = np.array([0.6, -0.3, 0.765, 1.0, 0.0, 0.0, 0.0])


class _Task(TopDownPickPlaceMixin):
    def __init__(self, arm):
        self.robot = SimpleNamespace(get_arm=lambda _tag: arm)


@pytest.fixture(scope="module")
def task():
    if not (FRANKA / "panda.urdf").is_file():
        pytest.skip("franka assets not available")
    planner = MplibPlanner(str(FRANKA / "panda.urdf"), str(FRANKA / "panda.srdf"),
                           "panda_hand", BASE, entity=None, n_arm=7)
    q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])

    def explode(**_kw):  # Genesis IK must never be called
        raise AssertionError("Genesis inverse_kinematics called")

    arm = SimpleNamespace(planner=planner, n_arm=7, tcp_offset=Pose(),
                          arm_joints=[], _cached_target=None,
                          entity=SimpleNamespace(inverse_kinematics=explode))
    arm.q = q0.copy()
    arm.get_arm_qpos = lambda: arm.q.copy()
    return _Task(arm), arm


def _link_pose(arm, q):
    backend = arm.planner._mplib
    backend.pinocchio_model.compute_forward_kinematics(arm.planner._pad_qpos(q))
    pose = backend.robot.get_base_pose() * backend.pinocchio_model.get_link_pose(
        backend.link_name_2_idx["panda_hand"])
    return np.r_[np.asarray(pose.p), np.asarray(pose.q)]


def test_seeded_ik_stays_on_the_seed_branch(task):
    t, arm = task
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_goal = arm.q + rng.uniform(-0.05, 0.05, 7)
        target = _link_pose(arm, q_goal)
        q = t._mplib_seeded_ik(target, "right", seed=arm.q, max_joint_step=0.3)
        assert q is not None
        assert np.max(np.abs(q - arm.q)) <= 0.3
        np.testing.assert_allclose(_link_pose(arm, q)[:3], target[:3], atol=5e-3)


def test_joint_step_bound_rejects_far_goal(task):
    t, arm = task
    far = arm.q.copy()
    far[0] += 1.0
    assert t._mplib_seeded_ik(_link_pose(arm, far), "right", seed=arm.q,
                              max_joint_step=0.3) is None


def test_solve_ik_uses_mplib_and_returns_none_when_unreachable(task):
    t, arm = task
    target = Pose.from_pose7(_link_pose(arm, arm.q + 0.02))
    q = t._solve_ik(target, "right")   # identity TCP offset: link == TCP
    assert q is not None and q.shape == (7,)
    assert t._solve_ik(Pose(np.array([5.0, 5.0, 5.0])), "right") is None


def test_planner_cartesian_ik_uses_mplib(task):
    _t, arm = task
    planner = arm.planner
    target = _link_pose(arm, arm.q + np.array([0.05, 0.03, 0, -0.05, 0, 0.04, 0]))
    res = planner.solve_ik_cartesian(arm.q.copy(), target, num_waypoints=20, log=False)
    assert res.success and res.position.shape == (20, 7)
    assert np.max(np.abs(np.diff(np.vstack([arm.q, res.position]), axis=0))) <= 0.3
    np.testing.assert_allclose(_link_pose(arm, res.position[-1])[:3], target[:3], atol=5e-3)
    far = np.r_[5.0, 5.0, 5.0, target[3:]]
    assert not planner.solve_ik_cartesian(arm.q.copy(), far, num_waypoints=5, log=False).success
