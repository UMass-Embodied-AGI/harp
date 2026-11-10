"""Measured can arm envelopes reject base intersections missed by prop checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from envs.task_bases.neutral_avatar_layout import _NeutralAvatarLayoutMixin as Layout


def footprint(points, radius=0.):
    return {"kind": "put_objects_in_bowl", "can_base_sweep": [
        {"name": "l_forearm", "vertices": Layout._neutral_xy_convex_hull(points), "radius": radius}]}


def clear(fp, center=(0., 0.), radius=.2, xy=(0., 0.), label="robot_base:right_arm"):
    return Layout._neutral_can_base_sweep_is_clear(xy, fp, [(center, radius, label)])


def test_segment_middle_crosses_base_although_both_endpoints_clear():
    assert not clear(footprint([[-1., 0.], [1., 0.]], .01))


def test_whole_motion_envelope_contains_base_without_vertex_overlap():
    assert not clear(footprint([[-1., -1.], [1., -1.], [0., 1.]]))


def test_capsule_radius_is_retained_and_translation_is_relative_to_anchor():
    fp = footprint([[0., .24], [1., .24]], .05)
    assert not clear(fp)
    assert clear(fp, xy=(0., .02))
    assert not clear(fp, center=(2., 3.), xy=(2., 3.))


def test_point_and_collinear_envelopes_are_supported():
    assert not clear(footprint([[0., .1]]))
    assert clear(footprint([[.4, .4]]))
    hull = Layout._neutral_xy_convex_hull([[1., 0.], [-1., 0.], [0., 0.], [1., 0.]])
    np.testing.assert_array_equal(hull, [[-1., 0.], [1., 0.]])


def test_gate_is_specific_to_existing_robot_base_regions():
    assert clear(footprint([[0., 0.]]), label="block:red")
    assert Layout._neutral_can_base_sweep_is_clear([0, 0], {"kind": "mouse"}, [])
    assert not clear({"can_base_sweep": []})
    assert not clear({"can_base_sweep": [{"vertices": [[float("nan"), 0]], "radius": 0}]})


def collider():
    return SimpleNamespace(
        current_capsules=lambda: [(n, [0., 0., 1.], [1., 0., 1.], .05)
                                 for n in ("l_forearm", "r_forearm", "l_index_dist", "r_index_dist", "l_shin")],
        current_boxes=lambda: [(n, [2., 3., 1.], np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), [.1, .2, .03])
                              for n in ("l_palm", "r_palm")])


def test_both_arms_fingers_and_rotated_palm_boxes_are_measured():
    task = Layout()
    task.BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    task.avatar_collider = collider()
    sweep = {}
    task._neutral_can_collect_arm_sweep(sweep)
    assert set(sweep) == {"l_forearm", "r_forearm", "l_index_dist", "r_index_dist", "l_palm", "r_palm"}
    radius, corners = sweep["l_palm"]
    assert radius == 0
    np.testing.assert_allclose(np.min(corners, axis=0), [1.8, 2.9])
    np.testing.assert_allclose(np.max(corners, axis=0), [2.2, 3.1])
    assert sweep["l_forearm"][0] == .05


@pytest.mark.parametrize("missing", ["all", "palms", "arms"])
def test_missing_geometry_fails_closed(missing):
    task = Layout()
    task.BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    task.avatar_collider = None if missing == "all" else collider()
    if missing == "palms":
        task.avatar_collider.current_boxes = lambda: []
    elif missing == "arms":
        task.avatar_collider.current_capsules = lambda: []
    with pytest.raises(RuntimeError, match="geometry"):
        task._neutral_can_collect_arm_sweep({})


def test_measurement_collects_every_clip_frame_and_offsets_from_first_anchor():
    task = Layout()
    task.BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    task.avatar_init_pos = np.zeros(3)
    task.avatar_init_rot = np.eye(3)
    task.avatar_collider = collider()
    calls = []
    avatar = SimpleNamespace(frame_ratio=.3, frame=-1)
    avatar.reset = lambda *args: setattr(avatar, "frame", -1)
    avatar.play_animation = lambda motion: None
    avatar.spare = lambda: avatar.frame >= 2
    avatar.step = lambda: setattr(avatar, "frame", avatar.frame + 1)
    task.avatar = avatar
    task.scene = SimpleNamespace(step=lambda: None)
    task._neutral_prop_radius = lambda name: .04
    task._neutral_avatar_event_anchor_pos = lambda *args: np.array([3., 4., 1.])
    task._neutral_bowl_event_type = lambda job: "unused"
    original = task._neutral_can_collect_arm_sweep
    def collect(sweep):
        calls.append(avatar.frame)
        original(sweep)
    task._neutral_can_collect_arm_sweep = collect
    fp = task._neutral_avatar_compute_motion_footprint({
        "kind": "put_objects_in_bowl", "motion": "clip", "base_motion": "clip",
        "clip": {"num_frames": 3},
        "events": [{"type": "pick", "clip_frame": 0, "hand_id": 0, "object": "object_1"}]})
    assert calls == [0, 1, 2]
    assert avatar.frame_ratio == .3
    arm = next(item for item in fp["can_base_sweep"] if item["name"] == "l_forearm")
    np.testing.assert_allclose(arm["vertices"], [[-3., -4.], [-2., -4.]])
    assert arm["radius"] == .05


def test_region_gate_cannot_accept_a_prop_safe_base_intersection():
    task = Layout()
    task.BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    task._neutral_table_xy_bounds = lambda: (-2., 2., -2., 2.)
    task._neutral_motion_table_margin = lambda fp: 0.
    task._neutral_task_region_is_physical = lambda label: True
    fp = footprint([[-1., 0.], [1., 0.]])
    fp["region_points"] = [{"offset": [1., 1.], "radius": .04, "label": "object_1"}]
    assert not task._neutral_motion_region_is_clear(np.zeros(2), fp, [(np.zeros(2), .2, "robot_base:right_arm")])
