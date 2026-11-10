"""An unchanged can side band contains valid poses missed by the broad grid."""
import json
from pathlib import Path
import numpy as np
from test_neutral_pose_fallback import Search
from envs.tasks import resolve_task_class


def test_recorded_seed10_recovers_six_poses_with_all_existing_gates():
    data = json.loads((Path(__file__).parent / "fixtures/neutral_can_band_seed10.json").read_text())
    _, cls = resolve_task_class("rank_blocks_by_color_copresence")
    task = cls(data["config"])
    task.TABLE_TOP_Z = data["table_top_z"]
    task._neutral_avatar_reset_seed = 10
    task._neutral_avatar_job = job = data["job"]
    task._neutral_table_xy_bounds = lambda: tuple(data["table_bounds"])
    task._neutral_avatar_candidate_sides = {}
    task._neutral_avatar_default_side_candidates = set()
    state = data["state"]
    reference, rotation, holds, fp, body = state["selected_tuple"]
    rotation = np.asarray(rotation)
    holds = {int(k): np.asarray(v) for k, v in holds.items()}
    regions = [(np.asarray(c), r, label) for c, r, label in data["task_regions"]]
    forbidden = [(np.asarray(c), r, label) for c, r, label in data["prop_forbidden_regions"]]
    assert any(label.startswith("gripper:") for _, _, label in forbidden)
    task._neutral_avatar_measure_body_rot = rotation
    band = dict(task._neutral_avatar_default_side_work_regions(job))["left"]
    poses = [("left", band, rotation, fp)]
    accepted = []
    for xy, measured_rotation in task._neutral_can_band_work_candidates(poses, regions):
        assert measured_rotation is rotation
        moved_body = np.asarray(body).copy()
        moved_body[:2] += xy - reference
        assert task._neutral_avatar_side_for_work_xy(job, xy) == "left"
        assert task._neutral_motion_region_is_clear(xy, fp, regions)
        assert task._neutral_footprint_inside_side_region(xy, fp, "left")
        assert task._neutral_footprint_inside_table(xy, fp)
        if (task._neutral_avatar_body_pose_is_valid(job, xy, moved_body)
            and task._neutral_initial_props_clearance(job, xy, moved_body, holds, forbidden) >= 0.
            and task._neutral_initial_props_table_clearance(job, xy, moved_body, holds) >= 0.):
            accepted.append(xy)
    assert len(accepted) == 6
    np.testing.assert_allclose(accepted[0], [-.4244023569742923, -.4280460769237149], atol=1e-12)
    # The original full-table 41-grid never visits this pocket's Y coordinates.
    bounds = np.asarray(data["table_bounds"])
    lo = bounds[[0, 2]] - fp["min_offset"]
    hi = bounds[[1, 3]] - fp["max_offset"]
    old_y = np.linspace(lo[1]+1e-7, hi[1]-1e-7, 41)
    assert min(abs(old_y - accepted[0][1])) > .001
    assert not state["selected"] and not state["side_region_ok"]


def test_existing_fallback_success_does_not_enter_band_pass():
    task = Search()
    job = {"kind": "put_objects_in_bowl"}
    def forbidden(*args):
        raise AssertionError("old selection must remain first")
    task._neutral_can_band_work_candidates = forbidden
    iterator = task._neutral_avatar_fallback_work_candidates(job, {}, [])
    next(iterator)
    iterator.close()


def test_band_pass_reuses_all_measurements_only_after_old_can_search_exhausts():
    task = Search()
    task._neutral_motion_region_is_clear = lambda *args: False
    captured = []
    def band(poses, regions):
        captured.extend(poses)
        return iter(())
    task._neutral_can_band_work_candidates = band
    cache = {}
    assert list(task._neutral_avatar_fallback_work_candidates({"kind": "put_objects_in_bowl"}, cache, [])) == []
    assert len(captured) == len(task.measured) == len(cache) == 5
    for side, region, rotation, fp in captured:
        assert side == "right"
        assert fp is cache[tuple(np.round(rotation.ravel(), 6))][1]
    assert task._neutral_avatar_measure_body_rot is None


def test_non_can_clip_adds_no_band_poses():
    task = Search()
    task._neutral_motion_region_is_clear = lambda *args: False
    captured = []
    task._neutral_can_band_work_candidates = lambda poses, regions: captured.extend(poses) or iter(())
    assert list(task._neutral_avatar_fallback_work_candidates({"kind": "sponge"}, {}, [])) == []
    assert captured == []


def test_disjoint_side_table_bounds_yield_nothing():
    task = Search()
    task.NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN = .03
    fp = {"min_offset": np.array([-.1, -.1]), "max_offset": np.array([.1, .1])}
    assert list(task._neutral_can_band_work_candidates([("right", (2., 3., 2., 3.), np.eye(3), fp)], [])) == []
