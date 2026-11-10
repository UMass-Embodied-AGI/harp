import numpy as np

from envs.task_bases.dump_bin_motion import DumpBinMotionMixin
from envs.utils import Pose


class _Arm:
    tcp_offset = Pose()

    def __init__(self, p):
        self._p = np.asarray(p, dtype=float)

    def get_ee_pose(self):
        return np.r_[self._p, 1., 0., 0., 0.]


class _Robot:
    def __init__(self, arm):
        self.arm = arm

    def get_arm(self, tag):
        return self.arm


class _Task(DumpBinMotionMixin):
    TABLE_TOP_Z, TRANSPORT_Z_ABOVE_TABLE = .74, .30

    def __init__(self, miss):
        self.robot = _Robot(_Arm([miss, 0., .80]))
        self._dump_hold_offset = None
        self.rejected = None

    def _dump_check_retention(self, arm_tag):
        pass

    def _reject_dump_motion(self, reason):
        self.rejected = reason
        raise RuntimeError(reason)


def _arrive(miss):
    task = _Task(miss)
    try:
        task._dump_arrival(np.r_[0., 0., .80, 1., 0., 0., 0.], 'right')
    except RuntimeError:
        pass
    return task.rejected


def test_empty_hand_miss_up_to_15mm_is_accepted_without_motion():
    assert DumpBinMotionMixin.UNLOADED_ARRIVAL_ACCEPT == .015
    assert _arrive(.0112) is None


def test_empty_hand_miss_beyond_15mm_is_still_rejected():
    assert _arrive(.018) is not None
