"""Measured neighbor defects must be rejected with the commanded open jaws."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases import block_grasp
from test_block_neighbor_endpoint_clearance import HIGH_Q, native_task


def test_recorded_rgb10_open_finger_collision_rejected(capsys):
    task, planner, backend = native_task(
        [-.15512309968471527, -.08769979327917099, .7849729061126709],
        [.9999998211860657, -.000012575834261951968, -7.58698718072992e-7, -.0006331385229714215])
    task.block_half_size = lambda key: .02
    end = np.array([.39361370557931835, .7706036746185443, -.19117007935166733,
                    -1.7665937677766723, .22643640251783873, 2.515216294509759,
                    2.4305968343890534])
    backend.robot.set_qpos(HIGH_Q, True)
    with block_grasp.block_neighbor_obstacles(task, "target", "right"):
        block_grasp.refresh_block_obstacles(task, "right")
        backend.robot.set_qpos(np.r_[end, 0., 0.], True)
        world = backend.planning_world
        distance = backend.robot.get_fcl_model().distance_with(
            world.get_object("block_ranking_neighbor_large"), acm=world.get_allowed_collision_matrix())
        assert .011 < distance.min_distance < .012
        backend.robot.set_qpos(np.r_[end, .04, .04], True)
        collisions = backend.check_for_env_collision(np.r_[end, .04, .04])
        assert any(c.link_name1 == "panda_leftfinger" and
                   c.object_name2 == "block_ranking_neighbor_large" for c in collisions)
        backend.robot.set_qpos(HIGH_Q, True)
        assert not block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, end[None]))
        assert "fingers=open" in capsys.readouterr().out
        np.testing.assert_array_equal(backend.robot.get_qpos(), HIGH_Q)


def test_recorded_rgb14_near_neighbor_rejected():
    task, planner, backend = native_task(
        [.17292116582393646, -.10309174656867981, .7849724888801575],
        [.6601257920265198, .2534433603286743, .6601263284683228, -.25344356894493103])
    task.block_half_size = lambda key: .02
    end = np.array([-.03472444104825998, .6491234602204097, -.18735210003064617,
                    -2.0053209511296544, .230080349222494, 2.633906935317511, 1.9734265771736177])
    backend.robot.set_qpos(HIGH_Q, True)
    with block_grasp.block_neighbor_obstacles(task, "target", "right"):
        block_grasp.refresh_block_obstacles(task, "right")
        assert not block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, end[None]))
        np.testing.assert_array_equal(backend.robot.get_qpos(), HIGH_Q)


@pytest.mark.parametrize("aperture", [float("nan"), -.01, .041])
def test_invalid_open_aperture_fails_closed_and_restores_state(aperture):
    task, planner, backend = native_task([-.169, -.094, .795], [1., 0., 0., 0.])
    task.robot.get_arm = lambda tag: NS(planner=planner, gripper_open=aperture)
    backend.robot.set_qpos(HIGH_Q, True)
    with block_grasp.block_neighbor_obstacles(task, "target", "right"):
        block_grasp.refresh_block_obstacles(task, "right")
        assert not block_grasp._neighbor_endpoint_clearance_ok(task, planner, PlanResult(True, HIGH_Q[None, :7]))
        np.testing.assert_array_equal(backend.robot.get_qpos(), HIGH_Q)
