"""Can pose search retains every strict reset gate, including final fallback."""
import numpy as np
import pytest

from test_neutral_pose_fallback import Search, Selected, acceptance_task


def can_task(**kwargs):
    task, calls, turned = acceptance_task(**kwargs)
    task._neutral_avatar_job = {
        "kind": "put_objects_in_bowl", "base_motion": "put_objects_in_bowl",
        "events": [{"type": "pick", "clip_frame": 0, "object": "object_1"}],
    }
    task._neutral_avatar_motion_footprint = {}
    task._install_neutral_reverse_subclip = lambda job: None
    return task, calls, turned


def test_successful_can_search_preserves_original_path():
    task, calls, _ = can_task(primary_clear=True)
    with pytest.raises(Selected, match="selected work_xy"):
        task._prepare_neutral_avatar_start_layout()
    assert calls == []


def test_can_retry_executes_with_its_measured_orientation():
    task, calls, turned = can_task()
    with pytest.raises(Selected, match="selected work_xy"):
        task._prepare_neutral_avatar_start_layout()
    assert calls == ["fallback"]
    np.testing.assert_array_equal(task._neutral_avatar_measure_body_rot, turned)


@pytest.mark.parametrize("gate", ["region", "body", "side", "table", "gripper", "prop_table"])
def test_can_terminal_fallback_cannot_bypass_a_rejected_gate(gate):
    task, calls, _ = can_task(rejected_gate=gate)
    with pytest.raises(RuntimeError, match="no neutral can layout candidate"):
        task._prepare_neutral_avatar_start_layout()
    assert calls == ["fallback"]
    assert len(task._neutral_avatar_region_debug_records) == 2
    assert not any(r["selected"] for r in task._neutral_avatar_region_debug_records)


def test_can_search_measures_new_heading_and_preserves_random_state():
    task = Search()
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl"}
    cache = {tuple(np.eye(3).ravel()): ({0: np.zeros(3)}, task.footprint(np.eye(3)))}
    previous = np.random.get_state()
    iterator = task._neutral_avatar_fallback_work_candidates(task._neutral_avatar_job, cache, [])
    xy, rotation = next(iterator)
    iterator.close()
    assert len(task.measured) == 1
    assert np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])) == pytest.approx(-30.)
    np.testing.assert_allclose(rotation, task.measured[0])
    assert task._neutral_avatar_default_table_side_for_xy(xy) == "right"
    assert task._neutral_avatar_measure_body_rot is None
    current = np.random.get_state()
    np.testing.assert_array_equal(current[1], previous[1])
    assert current[0] == previous[0] and current[2:] == previous[2:]


def test_can_search_respects_coarse_and_explicit_side_contracts():
    task = Search()
    job = {"kind": "put_objects_in_bowl"}
    task.config["neutral_use_region_placement"] = False
    assert list(task._neutral_avatar_fallback_work_candidates(job, {}, [])) == []
    task.config.clear()
    task._neutral_avatar_allowed_side_work_regions = lambda job: [("right", (0., 1., 0., 1.))]
    assert list(task._neutral_avatar_fallback_work_candidates(job, {}, [])) == []
    assert task.measured == []
