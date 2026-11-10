"""The adapter's copy of MPLib's screw integrator (envs/planning/mplib_planner.py).

* Without the fixes, ``_screw_core`` must reproduce MPLib 0.2.1's
  ``Planner.plan_screw`` exactly (status and every waypoint).
* ``finish_on_reach`` may only change calls MPLib fails on its end stall,
  and the finished path must reach the goal.
"""

from pathlib import Path

import numpy as np
import pytest

mplib = pytest.importorskip("mplib")
import mplib.pymp as pymp  # noqa: E402

from envs.planning import mplib_planner as mp  # noqa: E402

FRANKA = Path(__file__).resolve().parents[1] / "assets" / "embodiments" / "franka-panda"


@pytest.fixture(scope="module")
def backend():
    if not (FRANKA / "panda.urdf").is_file():
        pytest.skip("franka assets not available")
    planner = mplib.Planner(urdf=str(FRANKA / "panda.urdf"), srdf=str(FRANKA / "panda.srdf"),
                            move_group="panda_hand", use_convex=False)
    planner.set_base_pose(pymp.Pose(p=np.array([0.6, -0.3, 0.765]), q=np.array([1.0, 0, 0, 0])))
    stock = planner.plan_screw
    mp._install_backend_hooks(planner)
    return planner, stock


def _cases(backend, n, seed):
    """(start qpos, world goal pose) pairs: random start, goal a few cm and
    degrees away from the start's hand pose."""
    planner, _ = backend
    rng = np.random.default_rng(seed)
    lim = planner.joint_limits[:7]
    base = planner.robot.get_base_pose()
    out = []
    for _ in range(n):
        q = rng.uniform(lim[:, 0] + 0.05, lim[:, 1] - 0.05)
        if rng.random() < 0.5:   # start near a limit on one joint
            j = int(rng.integers(0, 7))
            side = int(rng.integers(0, 2))
            q[j] = lim[j, side] + (1 if side == 0 else -1) * 0.02 * rng.random()
            q[j] = float(np.clip(q[j], lim[j, 0] + 1e-3, lim[j, 1] - 1e-3))
        q = planner.pad_move_group_qpos(q)
        planner.pinocchio_model.compute_forward_kinematics(q)
        pose = base * planner.pinocchio_model.get_link_pose(planner.link_name_2_idx["panda_hand"])
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0, 0.6)
        dq = np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]
        goal = pymp.Pose(p=np.asarray(pose.p) + rng.uniform(-0.12, 0.12, 3), q=np.asarray(pose.q))
        goal = pymp.Pose(p=goal.p, q=_qmul(dq, np.asarray(goal.q)))
        out.append((q, goal, float(rng.choice([0.1, 0.02, 0.005]))))
    return out


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def test_core_reproduces_mplib_plan_screw(backend):
    planner, stock = backend
    for q, goal, step in _cases(backend, 60, 0):
        ref = stock(goal, q.copy(), qpos_step=step, time_step=1 / 250, wrt_world=True)
        got, reason = mp._screw_core(planner, goal, q.copy(), qpos_step=step,
                                     time_step=1 / 250, wrt_world=True, verbose=False)
        assert got["status"] == ref["status"]
        assert (reason is None) == (ref["status"] == "Success")
        if ref["status"] == "Success":
            np.testing.assert_array_equal(got["position"], ref["position"])


def _endpoint_error(planner, q_arm, goal):
    q = planner.pad_move_group_qpos(np.asarray(q_arm, dtype=np.float64).copy())
    planner.pinocchio_model.compute_forward_kinematics(q)
    reached = planner.robot.get_base_pose() * planner.pinocchio_model.get_link_pose(
        planner.link_name_2_idx["panda_hand"])
    g, r = goal.to_transformation_matrix(), reached.to_transformation_matrix()
    cos = np.clip((np.trace(g[:3, :3].T @ r[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(g[:3, 3] - r[:3, 3])), float(np.arccos(cos))


def test_hooked_plan_screw_only_changes_end_stalls(backend):
    planner, stock = backend
    rescued = failed = 0
    for q, goal, step in _cases(backend, 200, 1):
        ref = stock(goal, q.copy(), qpos_step=step, time_step=1 / 250, wrt_world=True)
        got = planner.plan_screw(goal, q.copy(), qpos_step=step, time_step=1 / 250, wrt_world=True)
        if ref["status"] == "Success":
            np.testing.assert_array_equal(got["position"], ref["position"])
            continue
        failed += 1
        _, reason = mp._screw_core(planner, goal, q.copy(), qpos_step=step,
                                   time_step=1 / 250, wrt_world=True, verbose=False)
        if got["status"] != "Success":
            continue
        assert reason["kind"] == "end_stall"
        rescued += 1
        pos = np.asarray(got["position"])
        np.testing.assert_allclose(pos[0], q[:7], atol=1e-9)
        err_p, err_r = _endpoint_error(planner, pos[-1], goal)
        assert err_p < 2e-3 and err_r < 1e-2
    print(f"stock failed={failed} rescued={rescued}")


def test_finish_on_reach_completes_exact_multiple_step(backend):
    """A straight move whose length is an exact number of steps leaves no
    twist for MPLib's final step; MPLib then reports failure."""
    planner, stock = backend
    q = planner.pad_move_group_qpos(np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]))
    planner.pinocchio_model.compute_forward_kinematics(q)
    base = planner.robot.get_base_pose()
    pose = base * planner.pinocchio_model.get_link_pose(planner.link_name_2_idx["panda_hand"])
    found = 0
    for dz in np.linspace(0.01, 0.2, 400):
        goal = pymp.Pose(p=np.asarray(pose.p) + np.array([0, 0, dz]), q=np.asarray(pose.q))
        ref = stock(goal, q.copy(), qpos_step=0.005, time_step=1 / 250, wrt_world=True)
        if ref["status"] == "Success":
            continue
        _, reason = mp._screw_core(planner, goal, q.copy(), qpos_step=0.005,
                                   time_step=1 / 250, wrt_world=True, verbose=False)
        if reason["kind"] != "end_stall":
            continue
        got = planner.plan_screw(goal, q.copy(), qpos_step=0.005, time_step=1 / 250, wrt_world=True)
        assert got["status"] == "Success"
        err_p, err_r = _endpoint_error(planner, got["position"][-1], goal)
        assert err_p < 1e-3 and err_r < 1e-2
        found += 1
    print(f"end_stall cases={found}")
