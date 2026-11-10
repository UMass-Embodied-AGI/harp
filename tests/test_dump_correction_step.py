import numpy as np
from types import SimpleNamespace as NS

from envs.planning.base import PlanResult
from envs.task_bases.dump_bin_motion import DumpBinMotionMixin, DumpMotionRejected
from envs.utils import Pose


class _Task(DumpBinMotionMixin):
    TABLE_TOP_Z, TRANSPORT_Z_ABOVE_TABLE = .765, .18

    def __init__(self, fine_ok, coarse_ok):
        self.steps = []
        self.measured = np.r_[.1, .05, .80, 1., 0., 0., 0.]
        self.measured[1] += .012
        def plan(current, target, qpos_step):
            self.steps.append(qpos_step)
            ok = fine_ok if qpos_step == .002 else coarse_ok
            return PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7))) if ok else PlanResult(False)
        self.arm = NS(tcp_offset=Pose(), n_arm=7, get_ee_pose=lambda: self.measured.copy(),
                      planner=NS(plan_screw_path=plan))
        self.robot = NS(get_arm=lambda tag: self.arm)
        self._dump_hold_offset = np.zeros(3)

    def _dump_check_retention(self, arm_tag):
        pass

    def _dump_motion_inputs(self, target, arm_tag):
        return self.arm, np.zeros(7), np.asarray(target, dtype=float)

    def _dump_plan_valid(self, arm, result, current, target):
        return bool(result is not None and result.success)

    def execute_plan(self, result, arm_tag):
        self.measured[1] -= .008

    def _reseed_cached_arm_target(self, arm):
        pass

    def _reject_dump_motion(self, reason):
        raise DumpMotionRejected(reason)


def _arrive(task):
    task._dump_arrival(np.r_[.1, .05, .80, 1., 0., 0., 0.], 'right')


def test_fine_step_is_used_first_and_alone_when_it_plans():
    task = _Task(True, True)
    _arrive(task)
    assert task.steps == [.002]


def test_coarse_step_only_after_the_fine_step_fails():
    task = _Task(False, True)
    _arrive(task)
    assert task.steps == [.002, .005]


def test_still_rejected_when_neither_step_plans():
    task = _Task(False, False)
    try:
        _arrive(task)
    except DumpMotionRejected as exc:
        assert 'no checked straight path' in str(exc)
    else:
        raise AssertionError('expected a rejection')
