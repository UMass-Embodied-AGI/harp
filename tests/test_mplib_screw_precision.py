from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.planning.mplib_planner import MplibPlanner


def planner_stub(debug=False):
    calls = []
    planner = object.__new__(MplibPlanner)
    planner.max_retry = 1
    planner._bounded_paths_debug_enabled = lambda: debug
    planner._to_mplib_pose = lambda target: target
    planner._pad_qpos = lambda q: q.copy()
    planner._mplib = SimpleNamespace(plan_screw=lambda **kw: calls.append(kw) or {
        "status": "Success", "position": np.ones((2, 7)),
    })
    planner._plan_bounded_screw = lambda *a, **kw: calls.append(kw) or PlanResult(True, np.ones((2, 7)))
    return planner, calls


@pytest.mark.parametrize("debug,requested,expected", [(False, None, .1), (False, .025, .025),
                                                       (True, None, .05), (True, .025, .025)])
def test_precision_is_opt_in_and_debug_default_is_preserved(debug, requested, expected):
    planner, calls = planner_stub(debug)
    kwargs = {} if requested is None else {"qpos_step": requested}
    result = planner.plan_screw_path(np.zeros(7), np.zeros(7), log=False, **kwargs)
    assert result.success and calls[0]["qpos_step"] == expected


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, True, [.025], "invalid"])
def test_invalid_precision_rejected_before_planner_call(invalid):
    planner, calls = planner_stub()
    with pytest.raises(ValueError):
        planner.plan_screw_path(np.zeros(7), np.zeros(7), qpos_step=invalid)
    assert calls == []
