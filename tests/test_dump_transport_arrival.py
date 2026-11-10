"""Unloaded arrivals at the transport altitude may use the checked correction.

The grasp selector's first move goes to an "above" pose at the task's own
transport altitude, TABLE_TOP_Z + TRANSPORT_Z_ABOVE_TABLE. Across the original
runs of dump_bin_interruption the arm landed 8.5-10.4 mm off that pose after the
endpoint settle -- a steady-state PD sag, not a transient -- and every miss over
the 10 mm gate was rejected outright because nothing was held. The loaded path
already corrects residuals of exactly that size. These tests pin that the
transport-altitude case now gets the same bounded, collision-checked correction,
and that an unloaded miss LOWER down, near the objects, is still rejected.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.dump_bin_motion import DumpBinMotionMixin, DumpMotionRejected
from envs.utils import Pose

TABLE = .765
TRANSPORT = .18
TCP_OFFSET = Pose([0., 0., .1034])


class Task(DumpBinMotionMixin):
    TABLE_TOP_Z = TABLE
    TRANSPORT_Z_ABOVE_TABLE = TRANSPORT

    def __init__(self, bias, held=False, plan_ok=True):
        # Measured TCP = commanded TCP + bias, until a correction is executed.
        self.bias = np.asarray(bias, dtype=float)
        self.cmd_tcp = None
        self._dump_hold_offset = np.zeros(3) if held else None
        self.plan_ok = plan_ok
        self.corrections, self.rejections = [], []
        self.arm = NS(tcp_offset=TCP_OFFSET, n_arm=7,
                      get_ee_pose=lambda: np.r_[self.cmd_tcp + self.bias, 1., 0., 0., 0.],
                      planner=NS(plan_screw_path=lambda current, target, qpos_step:
                                 PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7))) if self.plan_ok else PlanResult(False)))
        self.robot = NS(get_arm=lambda tag: self.arm)

    def _dump_check_retention(self, arm_tag):
        pass

    def _reject_dump_motion(self, reason):
        self.rejections.append(reason)
        raise DumpMotionRejected(reason)

    def _dump_motion_inputs(self, target, arm_tag):
        self._pending = np.asarray(target, dtype=float)
        return self.arm, np.zeros(7), self._pending

    def _dump_plan_valid(self, arm, result, current, target):
        return bool(result is not None and result.success)

    def execute_plan(self, result, arm_tag):
        # The corrected link target lands the TCP exactly where it was aimed.
        commanded_tcp = (Pose.from_pose7(self._pending) * TCP_OFFSET).p
        self.corrections.append(commanded_tcp.copy())
        self.cmd_tcp = commanded_tcp

    def _reseed_cached_arm_target(self, arm):
        pass


def link_target_at(tcp_z):
    return (Pose([.1, .05, tcp_z]) * Pose([0., 0., -.1034])).to_pose7()


def run(task, tcp_z):
    target = link_target_at(tcp_z)
    task.cmd_tcp = (Pose.from_pose7(target) * TCP_OFFSET).p.copy()
    task._dump_arrival(target, 'right')


def test_transport_altitude_unloaded_miss_is_corrected_not_rejected():
    task = Task(bias=[0., 0., -.018])             # above the 15 mm empty-hand accept band
    run(task, TABLE + TRANSPORT)
    assert task.rejections == []
    assert len(task.corrections) >= 1


def test_unloaded_miss_below_transport_is_still_rejected_outright():
    task = Task(bias=[0., 0., -.018])
    with pytest.raises(DumpMotionRejected):
        run(task, TABLE + TRANSPORT - .05)          # 5 cm lower, near the objects
    assert task.corrections == []
    assert task.rejections and task.rejections[-1].startswith('arrival xyz=')


def test_within_tolerance_needs_no_correction_anywhere():
    for z in (TABLE + TRANSPORT, TABLE + .05):
        task = Task(bias=[0., 0., -.004])
        run(task, z)
        assert task.rejections == [] and task.corrections == []


def test_transport_correction_still_needs_a_checked_path():
    task = Task(bias=[0., 0., -.018], plan_ok=False)
    with pytest.raises(DumpMotionRejected, match='no checked straight path'):
        run(task, TABLE + TRANSPORT)


def test_transport_correction_keeps_the_existing_distance_ceiling():
    # 30 mm is beyond the unchanged 25 mm correctable ceiling: still rejected.
    task = Task(bias=[0., 0., -.030])
    with pytest.raises(DumpMotionRejected, match='arrival xyz='):
        run(task, TABLE + TRANSPORT)
    assert task.corrections == []


@pytest.mark.parametrize('tcp_z,expected', [(TABLE + TRANSPORT, True), (TABLE + TRANSPORT - .004, True),
                                            (TABLE + TRANSPORT - .006, False), (TABLE + .02, False)])
def test_transport_altitude_boundary(tcp_z, expected):
    task = Task(bias=[0., 0., 0.])
    assert task._dump_target_at_transport(link_target_at(tcp_z), task.arm) is expected


def test_loaded_behaviour_is_unchanged_below_transport():
    task = Task(bias=[0., 0., -.0102], held=True)
    run(task, TABLE + TRANSPORT - .05)
    assert task.rejections == [] and len(task.corrections) >= 1
