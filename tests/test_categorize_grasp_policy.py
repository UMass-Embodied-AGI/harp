import numpy as np
import transforms3d as t3d
from types import SimpleNamespace

from envs.grasp import load_grasp_poses
from envs.planning.base import PlanResult
from envs.task_bases.categorize_support import transport as categorize_transport
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.utils import Pose


def test_milk_box_uses_four_side_grasps_only():
    names = CategorizeObjectsContribution.EXCLUSIVE_GRASP_NAMES["038_milk-box"]
    assert len(names) == 4
    assert "038_milk-box" in CategorizeObjectsContribution.YAML_ONLY_GRASP_LABELS

    by_name = {
        grasp.name: grasp
        for grasp in load_grasp_poses("038_milk-box", model_id=0, robot_type="franka")
    }
    for name in names:
        approach = t3d.quaternions.quat2mat(by_name[name].pose.q)[:, 2]
        # Authored +Y is world-up after the categorize spawn transform.
        assert abs(float(approach[1])) < 0.20


def test_rubiks_cube_uses_authored_top_down_face_grasps_only():
    names = CategorizeObjectsContribution.EXCLUSIVE_GRASP_NAMES["073_rubikscube"]
    assert names == (
        "manual_topdown_close_x",
        "manual_topdown_close_x_flipfinger",
        "manual_topdown_close_y",
        "manual_topdown_close_y_flipfinger",
    )
    assert "073_rubikscube" in CategorizeObjectsContribution.YAML_ONLY_GRASP_LABELS

    by_name = {
        grasp.name: grasp
        for grasp in load_grasp_poses("073_rubikscube", model_id=1, robot_type="franka")
    }
    assert set(names) <= set(by_name)


def test_apple_is_not_silently_skipped():
    assert "035_apple" not in CategorizeObjectsContribution.RADIUS_SKIP_LABELS
    assert np.isclose(CategorizeObjectsContribution.RADIUS_CLOSE_OVERRIDE["035_apple"], 0.58)
    assert CategorizeObjectsContribution.APPLE_YAML_MIN_X == 0.0
    assert CategorizeObjectsContribution.APPLE_CENTER_MIN_X == 0.20
    assert CategorizeObjectsContribution.APPLE_CENTER_GRASP_NAMES[0] == "manual_center_p0_yaw0"
    assert CategorizeObjectsContribution.APPLE_UNDERHAND_GRASP_NAMES[0] == "manual_underhand_p0_yaw0"


def test_stapler_bans_long_axis_yaw_zero():
    assert np.isclose(
        CategorizeObjectsContribution.RADIUS_GRASP_YAW["048_stapler"],
        np.pi / 2.0,
    )


def test_catalog_close_must_contact_the_selected_model():
    close = CategorizeObjectsContribution._radius_close_for_live_width
    # Woodenblock@0's 0.52 preset leaves a gap wider than the entire @1
    # cross section.  The corrected target must buy contact without closing
    # the fingers through its center.
    corrected = close(0.52, 0.0175)
    assert np.isclose(corrected * 0.04, 0.0135)
    assert 0.0 < corrected * 0.04 < 0.0175
    # Validated contacting presets and the normal adaptive formula survive.
    assert close(0.52, 0.0249) == 0.52
    assert close(0.58, 0.026) == 0.58
    assert close(None, 0.0175) is None


def test_stapler_height_offset_cannot_grasp_above_a_flat_body():
    offset_for_top = CategorizeObjectsContribution._radius_offset_below_live_top
    center_z, radius, top_z = 0.7824, 0.025, 0.7910
    offset = offset_for_top(0.020, center_z, radius, top_z)
    tcp_z = center_z - 0.3 * radius + offset
    assert np.isclose(tcp_z, top_z - 0.008)
    # A tall jar's mid-body preset remains valid.
    assert offset_for_top(0.020, 0.812, 0.030, 0.860) == 0.020


def test_side_grasp_places_object_center_in_basket():
    basket = np.array([-0.25, 0.15, 0.82])
    held_offset = np.array([-0.035, 0.045, -0.025])
    target = CategorizeObjectsContribution._yaml_transport_target(basket, 1.0, held_offset)
    np.testing.assert_allclose((target + held_offset)[:2], basket[:2])
    assert target[2] == 1.0


def test_milk_continuation_uses_the_standard_ranker_and_original_allowlist(monkeypatch):
    task = CategorizeObjectsContribution({"setting": "sim"})
    task.TABLE_TOP_Z = 0.765
    task.MILK_CONTINUATION_PREFLIGHT = True
    calls = []
    marker = object()
    monkeypatch.setattr(task, "select_and_execute_grasp", lambda *a, **k: calls.append((a, k)) or marker)
    monkeypatch.setattr(task, "try_grasp_by_name", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("production milk continuation used the debug-only helper"),
    ))
    predicate = lambda _: True
    result = task._select_and_execute_task_grasp(
        "038_milk-box", Pose([0, 0, 0.8]), "right", model_id=0,
        object_scale=1.0, post_plan_check=predicate,
    )
    assert result is marker
    assert calls[0][1]["grasp_names"] == task.EXCLUSIVE_GRASP_NAMES["038_milk-box"]
    assert calls[0][1]["post_plan_check"] is predicate


def test_milk_checks_lift_and_compensated_carry_without_execution(monkeypatch):
    calls = []

    def screw(qpos, target, **_kwargs):
        calls.append((np.array(qpos), np.array(target)))
        return PlanResult(True, np.full((2, 7), len(calls)))

    arm = SimpleNamespace(planner=SimpleNamespace(plan_screw_path=screw), tcp_offset=Pose([0, 0, 0]))
    task = SimpleNamespace(
        robot=SimpleNamespace(get_arm=lambda _: arm),
        _yaml_transport_target=CategorizeObjectsContribution._yaml_transport_target,
    )
    pick = SimpleNamespace(get_center=lambda: np.array([0.12, -0.12, 0.83]), radius=0.03)
    place = SimpleNamespace(pos=np.array([-0.25, 0.15, 0.82]))
    monkeypatch.setattr(categorize_transport, "endpoint_matches", lambda *_: True)
    check = categorize_transport.milk_continuation_check(task, pick, place, "right", 1.0)
    plans = (None, PlanResult(True, np.zeros((2, 7))), Pose([0.10, -0.15, 0.85]), None)
    assert check(plans)
    assert len(calls) == 3
    np.testing.assert_allclose(calls[0][1][:3], [0.10, -0.15, 0.86])
    np.testing.assert_allclose(calls[2][1][:3], [-0.27, 0.12, 1.0])
    np.testing.assert_array_equal(calls[2][0], np.full(7, 2))
