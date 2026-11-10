from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases.dump_bin import DumpBin
from envs.utils import Pose


def test_apple_and_jar_close_from_measured_width():
    assert DumpBin.WIDTH_CLOSE_LABELS == ('035_apple', '031_jam-jar')


def _pick_from_bounds(label, bounds, robot_type='franka'):
    task = DumpBin.__new__(DumpBin)
    actor = SimpleNamespace(get_AABB=lambda: np.asarray(bounds, dtype=float))
    task.config = {'robot_type': robot_type}
    task.TABLE_TOP_Z = .765
    task.big_bin_pose = Pose([.4, .2, 0.])
    task._object_labels = {id(actor): label}
    task._object_close_values = {id(actor): .45}
    picks = []
    task.pick_and_place = lambda pick, place, arm_tag: picks.append(pick) or False
    assert task._pick_and_drop_cube(actor, 'right') is False
    return picks


@pytest.mark.parametrize('label', ['035_apple', '031_jam-jar'])
@pytest.mark.parametrize('width_x,width_y', [(.058, .064), (.074, .068)])
def test_franka_closure_uses_measured_narrower_horizontal_extent(label, width_x, width_y):
    picks = _pick_from_bounds(label, [[0., 0., .765], [width_x, width_y, .83]])
    assert len(picks) == 1
    assert picks[0].close_value == pytest.approx((min(width_x, width_y) - .008) / .08)


@pytest.mark.parametrize('bounds', [
    [[0., 0., .765], [np.nan, .06, .83]],
    [[0., 0., .765], [0., .06, .83]],
    [[0., 0., .765]],
])
def test_invalid_bounds_reject_before_pick_executes(bounds):
    assert _pick_from_bounds('035_apple', bounds) == []


def test_other_embodiment_keeps_its_existing_close_target():
    picks = _pick_from_bounds('035_apple', [[0., 0., .765], [.058, .064, .83]], 'piper')
    assert picks[0].close_value == .45
