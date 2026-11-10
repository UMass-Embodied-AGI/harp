"""Regression cases for side selection before neutral-layout acceptance."""
import numpy as np

from envs.task_bases.neutral_avatar_layout import _NeutralAvatarLayoutMixin


class Layout(_NeutralAvatarLayoutMixin):
    NEUTRAL_TABLE_XY_BOUNDS = (-1.0, 1.0, -1.0, 1.0)
    NEUTRAL_CLEARANCE_MARGIN = 0.01
    NEUTRAL_AVATAR_BLOCKED_SIDES = ()

    def __init__(self, blocked=()):
        self.config = {"neutral_avatar_region_candidates": 1,
                       "neutral_avatar_blocked_sides": blocked}
        self._neutral_avatar_job = {"kind": "sponge"}

    def neutral_task_regions(self):
        return []

    def _neutral_motion_table_margin(self, footprint):
        return 0.0


def test_default_side_cannot_reintroduce_blocked_cabinet_side():
    task = Layout(blocked=("right",))
    xy = np.array([0.9, 0.0])
    task._neutral_avatar_default_side_candidates = {(0.9, 0.0)}
    assert task._neutral_avatar_side_for_work_xy(task._neutral_avatar_job, xy) == "back"


def test_side_budget_is_applied_after_orientation_filter():
    task = Layout()
    task._neutral_avatar_default_side_work_regions = lambda: [
        ("right", (0.7, 1.0, -0.5, 0.5)),
        ("left", (-1.0, -0.7, -0.5, 0.5)),
    ]
    # The largest free cell is on the left. A one-candidate quota must
    # still examine the later right cell for the right-facing footprint.
    task._neutral_empty_anchor_rect_candidates = lambda *args: [
        np.array([-0.8, 0.0]), np.array([0.8, 0.0]),
    ]
    footprint = {"min_offset": np.array([-0.1, -0.1]),
                 "max_offset": np.array([0.1, 0.1])}
    candidates = task._neutral_avatar_motion_work_candidates(
        task._neutral_avatar_job, footprint,
    )
    np.testing.assert_allclose(candidates, [[0.8, 0.0], [-0.8, 0.0]])
    assert [task._neutral_avatar_side_for_work_xy(task._neutral_avatar_job, xy)
            for xy in candidates] == ["right", "left"]


def test_all_blocked_sides_have_no_default():
    task = Layout(blocked=("left", "right", "back"))
    assert task._neutral_avatar_default_table_side_for_xy(np.zeros(2)) == ""
