"""Failed neutral searches may add poses, never bypass acceptance gates."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases.neutral_avatar_layout import _NeutralAvatarLayoutMixin
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin


class Selected(Exception):
    pass


def acceptance_task(*, rejected_gate=None, primary_clear=False):
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    task.avatar = SimpleNamespace(frame_ratio=1)
    task.config = {"neutral_avatar_strict_layout": True, "neutral_use_region_placement": True}
    task._neutral_avatar_job = {"kind": "sponge", "base_motion": "wipe",
                               "events": [{"type": "attach", "clip_frame": 0, "object": "sponge"}]}
    task._neutral_avatar_layout_ready = False
    task._install_neutral_avatar_sponge_pingpong_subclip = lambda job: None
    task._neutral_avatar_motion_slow = lambda: 1
    task._neutral_avatar_work_candidates = lambda job: [np.array([0., 0.])]
    task.neutral_task_regions = lambda: []
    gripper_regions = [(np.array([.2, .1]), .1, "gripper:real")]
    task._neutral_avatar_prop_forbidden_regions = lambda regions: gripper_regions
    task._neutral_avatar_body_rot_for_work_xy = lambda job, xy: np.eye(3)
    task._neutral_avatar_measure_hold_points = lambda job: {0: np.zeros(3)}
    task._neutral_avatar_compute_motion_footprint = lambda job: {}
    task._neutral_avatar_body_pos_for_work_xy = lambda job, xy, hold: np.r_[xy, 0.]
    task._neutral_footprint_is_clear = lambda xy, fp, regions: False
    task._neutral_motion_region_is_clear = lambda xy, fp, regions: (
        (primary_clear or xy[0] != 0) and rejected_gate != "region")
    task._neutral_avatar_body_pose_is_valid = lambda *args: rejected_gate != "body"
    task._neutral_avatar_side_for_work_xy = lambda *args: "right"
    task._neutral_footprint_inside_side_region = lambda *args: rejected_gate != "side"
    task._neutral_footprint_inside_table = lambda *args: rejected_gate != "table"

    def prop_clearance(job, xy, body, holds, forbidden):
        assert forbidden is gripper_regions
        return -.001 if rejected_gate == "gripper" else .1

    task._neutral_initial_props_clearance = prop_clearance
    task._neutral_initial_props_table_clearance = lambda *args: -.001 if rejected_gate == "prop_table" else .1
    task._neutral_debug = lambda message: (_ for _ in ()).throw(Selected(message))
    calls = []
    turned = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])

    def fallback(job, cache, regions):
        calls.append("fallback")
        assert len(cache) == 1  # Original orientation was evaluated first.
        yield np.array([.3, .1]), turned

    task._neutral_avatar_fallback_work_candidates = fallback
    return task, calls, turned


def test_successful_primary_layout_never_enters_fallback():
    task, calls, _ = acceptance_task(primary_clear=True)
    with pytest.raises(Selected):
        task._prepare_neutral_avatar_start_layout()
    assert calls == []


def test_retry_uses_its_measured_rotation_at_acceptance():
    task, calls, turned = acceptance_task()
    with pytest.raises(Selected):
        task._prepare_neutral_avatar_start_layout()
    assert calls == ["fallback"]
    np.testing.assert_array_equal(task._neutral_avatar_measure_body_rot, turned)


@pytest.mark.parametrize("gate", ["region", "body", "side", "table", "gripper", "prop_table"])
def test_retry_preserves_every_strict_acceptance_gate(gate):
    task, calls, _ = acceptance_task(rejected_gate=gate)
    with pytest.raises(RuntimeError, match="no neutral layout candidate"):
        task._prepare_neutral_avatar_start_layout()
    assert calls == ["fallback"]
    assert len(task._neutral_avatar_region_debug_records) == 2
    assert not any(item["selected"] for item in task._neutral_avatar_region_debug_records)


class Search(_NeutralAvatarLayoutMixin):
    BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    NEUTRAL_TABLE_XY_BOUNDS = (-1., 1., -1., 1.)
    NEUTRAL_AVATAR_BLOCKED_SIDES = ("left", "back")
    NEUTRAL_USE_REGION_PLACEMENT = True

    def __init__(self):
        self.config = {}
        self._neutral_avatar_job = {"kind": "sponge"}
        self.measured = []
        self._neutral_avatar_measure_body_rot = None

    def _neutral_avatar_body_rot_for_side(self, side, job):
        assert side == "right"
        return np.eye(3)

    def _neutral_motion_table_margin(self, footprint):
        return .01

    def _neutral_avatar_measure_hold_points(self, job):
        self.measured.append(self._neutral_avatar_measure_body_rot.copy())
        return {0: np.zeros(3)}

    def _neutral_avatar_compute_motion_footprint(self, job):
        return self.footprint(self._neutral_avatar_measure_body_rot)

    @staticmethod
    def footprint(rotation):
        return {"min_offset": np.array([-.1, -.1]), "max_offset": np.array([.1, .1]),
                "measured_rotation": rotation.copy()}

    def _neutral_motion_region_is_clear(self, xy, footprint, regions):
        # The original heading is blocked; the measured turn opens a pocket.
        return footprint["measured_rotation"][0, 1] > .1 and xy[0] > .7


def test_search_reuses_original_measurement_then_measures_bounded_yaw():
    task = Search()
    cache = {tuple(np.eye(3).ravel()): ({0: np.zeros(3)}, task.footprint(np.eye(3)))}
    state = np.random.get_state()
    iterator = task._neutral_avatar_fallback_work_candidates(task._neutral_avatar_job, cache, [])
    xy, rotation = next(iterator)
    iterator.close()
    assert len(task.measured) == 1
    assert np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])) == pytest.approx(-30.)
    np.testing.assert_allclose(rotation, task.measured[0])
    assert task._neutral_avatar_default_table_side_for_xy(xy) == "right"
    assert task._neutral_avatar_measure_body_rot is None
    now = np.random.get_state()
    assert state[0] == now[0] and state[2:] == now[2:]
    np.testing.assert_array_equal(state[1], now[1])


@pytest.mark.parametrize("kind", ["mouse", "writing", "bottle"])
def test_unrelated_clips_are_not_changed(kind):
    task = Search()
    assert list(task._neutral_avatar_fallback_work_candidates({"kind": kind}, {}, [])) == []
    assert task.measured == []


def test_coarse_placement_and_explicit_side_contracts_are_unchanged():
    task = Search()
    task.config["neutral_use_region_placement"] = False
    assert list(task._neutral_avatar_fallback_work_candidates(task._neutral_avatar_job, {}, [])) == []
    task.config.clear()
    task._neutral_avatar_allowed_side_work_regions = lambda job: [("right", (0., 1., 0., 1.))]
    assert list(task._neutral_avatar_fallback_work_candidates(task._neutral_avatar_job, {}, [])) == []
    assert task.measured == []
