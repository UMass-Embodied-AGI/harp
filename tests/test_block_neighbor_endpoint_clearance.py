"""Endpoint clearance must preserve valid departure states and planner state."""
from types import SimpleNamespace as NS

import mplib
import numpy as np
import pytest
from mplib.pymp import Pose as MplibPose

from envs.planning.base import PlanResult
from envs.task_bases import block_grasp
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.utils import ASSETS_PATH, Pose


HIGH_Q = np.array([.23641197470478742, -.023951185781570294, .16849704061425408,
                   -2.5201971308543385, .008023518273780037, 2.4966301369433537,
                   2.7542501999883187, 0., 0.])
LOW_Q = np.array([.27863751811657084, .40109174205911285, .09943874912201693,
                  -2.507991327250979, -.1443321645140765, 2.9046077125863605,
                  2.8638695850181586, 0., 0.])


def native_task(center, quat):
    urdf = ASSETS_PATH / "embodiments/franka-panda/panda.urdf"
    backend = mplib.Planner(urdf=str(urdf), srdf=str(urdf.with_suffix(".srdf")),
                            move_group="panda_hand", use_convex=False, new_package_keyword="")
    backend.set_base_pose(MplibPose(p=np.array([0., -.65, .75]),
                                    q=np.array([2 ** -.5, 0., 0., 2 ** -.5])))
    cube = NS(get_pos=lambda: np.array(center), get_quat=lambda: np.array(quat))
    planner = NS(_mplib=backend, _pad_qpos=lambda q: np.r_[q, 0., 0.],
        update_obstacles=lambda points, **kwargs: backend.update_point_cloud(points, **kwargs))
    task = NS(robot=NS(get_arm=lambda tag: NS(planner=planner, gripper_open=.04)),
              blocks={"target": cube, "large": cube}, block_half_size=lambda key: .03,
              avatar_collider=None)
    return task, planner, backend


def test_recorded_fragile_endpoint_rejected_and_planner_state_restored():
    task, planner, backend = native_task(
        [-.16910552978515625, -.09409117698669434, .7949724793434143],
        [.9990987181663513, 3.2409212735728943e-7, -3.5091497352368606e-7, -.04244830459356308])
    backend.robot.set_qpos(HIGH_Q, True)
    with block_grasp.block_neighbor_obstacles(task, "target", "right"):
        block_grasp.refresh_block_obstacles(task, "right")
        assert not backend.check_for_env_collision(LOW_Q)
        assert not block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, LOW_Q[None, :7]))
        np.testing.assert_array_equal(backend.robot.get_qpos(), HIGH_Q)
        assert block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, HIGH_Q[None, :7]))
        np.testing.assert_array_equal(backend.robot.get_qpos(), HIGH_Q)


def test_post_large_release_remains_valid_without_obstacle_expansion():
    # Original size-assist18,3.838mm finger clearance at the next pickup's start.
    task, planner, backend = native_task(
        [-.15743055939674377, -.08850213885307312, .7949725389480591],
        [.9999990463256836, -6.345083392034212e-8, 5.479842570821347e-7, .001419999054633081])
    released_q = np.array([.14949558675289154, .4839523136615753, .12379695475101471,
                          -2.079575538635254, -.0996820479631424, 2.5441975593566895,
                          2.696535587310791, 0., 0.])
    backend.robot.set_qpos(released_q, True)
    with block_grasp.block_neighbor_obstacles(task, "target", "right"):
        block_grasp.refresh_block_obstacles(task, "right")
        def neighbor_distance():
            world = backend.planning_world
            return backend.robot.get_fcl_model().distance_with(
                world.get_object("block_ranking_neighbor_large"),
                acm=world.get_allowed_collision_matrix()).min_distance
        assert .003 < neighbor_distance() < .005
        assert not backend.check_for_env_collision(released_q)
        assert block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, HIGH_Q[None, :7]))
        np.testing.assert_array_equal(backend.robot.get_qpos(), released_q)
        assert .003 < neighbor_distance() < .005
        assert not backend.check_for_env_collision(released_q)


def test_distance_exception_fails_closed_and_restores_joint_state():
    joints = np.arange(9.)
    original = joints.copy()
    def broken_distance(*args, **kwargs):
        raise ValueError("native distance unavailable")
    robot = NS(get_qpos=lambda: joints.copy(),
               set_qpos=lambda q, update: joints.__setitem__(slice(None), q),
               get_user_joint_names=lambda: [f"panda_joint{i}" for i in range(1, 8)] +
                   ["panda_finger_joint1", "panda_finger_joint2"],
               get_fcl_model=lambda: NS(distance_with=broken_distance))
    world = NS(get_allowed_collision_matrix=lambda: object(), get_object=lambda name: object())
    task = NS(_block_neighbor_scene={"names": ["large"], "world": world, "arm_tag": "right"},
              robot=NS(get_arm=lambda tag: NS(gripper_open=.04)))
    planner = NS(_mplib=NS(robot=robot, joint_limits=np.tile([0., .04], (9, 1))),
                 _pad_qpos=lambda q: np.r_[q, 0., 0.])
    assert not block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, np.zeros((1, 7))))
    np.testing.assert_array_equal(joints, original)


def test_without_neighbor_context_adds_no_backend_dependency():
    assert block_grasp._neighbor_endpoint_clearance_ok(NS(), object(), object())


def test_rejected_clearance_uses_existing_yaw_order_without_execution(monkeypatch):
    attempts = []
    def approach(q, target, **kwargs):
        attempts.append(target)
        return PlanResult(True, np.zeros((2, 7)))
    planner = NS(plan_path=approach, plan_screw_path=lambda *a, **kw: PlanResult(True, np.zeros((2, 7))))
    arm = NS(planner=planner, tcp_offset=Pose([0., 0., 0.]), get_arm_qpos=lambda: np.zeros(7))
    task = NS(robot=NS(get_arm=lambda tag: arm), TABLE_TOP_Z=.765, _R_DOWN=BlocksRankingRGB._R_DOWN,
              execute_plan=lambda *args: pytest.fail("clearance preflight executed motion"))
    monkeypatch.setattr(block_grasp, "refresh_block_obstacles", lambda *args: None)
    monkeypatch.setattr(block_grasp, "endpoint_matches", lambda planner, plan, target: plan.success)
    monkeypatch.setattr(block_grasp, "_neighbor_endpoint_clearance_ok", lambda *args: len(attempts) > 2)
    result = block_grasp.plan_block_grasp(task, [0., 0., .795], .03, .965, "right")
    assert result is not None and np.isclose(result.yaw, np.pi / 2)
    assert len(attempts) == 3 and len(result.legs) == 3
