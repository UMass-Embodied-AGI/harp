"""_move_screw reports a failed seeded fallback instead of returning True."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.manipulation import TopDownPickPlaceMixin
from envs.utils import Pose


class _Planner:
    def __init__(self):
        self._mplib = SimpleNamespace(plan_screw=lambda **_: {"status": "IK Failed"})

    def _to_mplib_pose(self, pose7):
        return pose7

    def _pad_qpos(self, qpos):
        return np.asarray(qpos, dtype=float)


class _Task(TopDownPickPlaceMixin):
    def __init__(self, seeded_result):
        arm = SimpleNamespace(tcp_offset=Pose([0, 0, 0.1], [1, 0, 0, 0]), planner=_Planner(),
                              get_arm_qpos=lambda: np.zeros(7))
        self.robot = SimpleNamespace(get_arm=lambda tag: arm)
        self._seeded_result = seeded_result
        self.executed = []

    def _move_seeded(self, link_pose7, arm_tag):
        return self._seeded_result

    def execute_plan(self, plan, arm_tag):
        self.executed.append(plan)


@pytest.mark.parametrize("seeded, expected", [(None, False), ({"status": "Success"}, True)])
def test_screw_failure_returns_seeded_fallback_outcome(seeded, expected):
    task = _Task(seeded)
    assert task._move_screw([0.1, 0.0, 0.9], "right") is expected
    assert task.executed == []
