"""Mesh refinement preserves forearm skin and falls back conservatively."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.collider import AvatarCollider
from envs.avatar.collision_checker import AvatarCollisionChecker
from envs.avatar.forearm_geometry import ForearmMeshRefiner, convex_halfspaces, forearm_vertex_indices


def skin_fixture(side="Right"):
    vertices = np.array([
        [-.02, -.02, 0], [.02, -.02, 0], [.02, .02, 0], [-.02, .02, 0],
        [-.02, -.02, 1], [.02, -.02, 1], [.02, .02, 1], [-.02, .02, 1],
        [0, 0, 1.015], [0, 0, 1.3], [.2, 0, 1.3],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        [4, 5, 8], [8, 9, 10],
    ])
    names = [f"mixamorig:{side}ForeArm", f"{side}ForeArm2", f"{side}Hand", f"{side}Arm"]
    joints = np.tile(np.arange(4), (len(vertices), 1))
    weights = np.zeros(joints.shape)
    weights[:8, :3] = [.3, .3, .4]  # Forearm twist aggregation is essential.
    weights[8:10, 2] = 1.0
    weights[10] = [.4, 0, 0, .6]  # Must compare against upper arm, not just hand.
    return SimpleNamespace(
        _latest_render_vverts=vertices, _entity=SimpleNamespace(_nyx_custom_vfaces=faces),
        _nodes=(None, [SimpleNamespace(name=name) for name in names]),
        _skin_joints=np.arange(len(names)), _vert_joints_cuda=joints,
        _vert_weights_cuda=weights,
    )


def test_twist_aggregation_includes_complete_mixed_boundary_triangles():
    skin = skin_fixture()
    selected = forearm_vertex_indices(
        [node.name for node in skin._nodes[1]], skin._vert_joints_cuda,
        skin._vert_weights_cuda, skin._entity._nyx_custom_vfaces, "RightForeArm",
    )
    np.testing.assert_array_equal(selected, np.arange(9))


def test_hull_contains_entire_selected_surface_including_distal_seam():
    skin = skin_fixture()
    refiner = ForearmMeshRefiner(skin)
    equations = refiner.halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1])
    residual = skin._latest_render_vverts[:9] @ equations[:, :3].T + equations[:, 3]
    assert np.max(residual) <= 0
    # A point in the real, mixed-weight wrist seam must remain inside.
    assert np.max(np.array([0, -.002, 1.004]) @ equations[:, :3].T + equations[:, 3]) < 0
    # Empty space inside the old broad capsule must remain outside the hull.
    assert np.max(np.array([.035, 0, .8]) @ equations[:, :3].T + equations[:, 3]) > .014


def test_pose_cache_reuses_same_mesh_but_updates_for_new_world_vertices():
    skin = skin_fixture()
    refiner = ForearmMeshRefiner(skin)
    first = refiner.halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1])
    assert refiner.halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is first
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    translation = np.array([.2, -.4, 1.1])
    skin._latest_render_vverts = skin._latest_render_vverts @ rotation.T + translation
    second = refiner.halfspaces("RightForeArm", translation, np.array([0, 0, 1]) @ rotation.T + translation)
    assert second is not first
    residual = skin._latest_render_vverts[:9] @ second[:, :3].T + second[:, 3]
    assert np.max(residual) <= 1e-12


@pytest.mark.parametrize("broken", ["indices", "degenerate", "nonfinite", "faces"])
def test_invalid_skin_keeps_capsule_fallback(broken):
    skin = skin_fixture()
    if broken == "indices":
        skin._vert_joints_cuda[0, 0] = 1000
    elif broken == "faces":
        skin._entity._nyx_custom_vfaces[0, 0] = 1000
    elif broken == "nonfinite":
        skin._latest_render_vverts[0, 0] = np.nan
    else:
        skin._latest_render_vverts[:] = 0
    with pytest.warns(RuntimeWarning, match="retaining capsule"):
        assert ForearmMeshRefiner(skin).halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is None


def test_missing_unrendered_skin_keeps_capsule_fallback():
    assert ForearmMeshRefiner(SimpleNamespace()).halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is None


@pytest.mark.parametrize("field", ["joints", "faces", "palette"])
@pytest.mark.parametrize("invalid", [-1, .5, np.nan, np.inf, 1000, 2**64-1])
def test_raw_invalid_indices_fall_back_without_truncation_or_wrapping(field, invalid):
    skin = skin_fixture()
    attribute = {"joints": "_vert_joints_cuda", "faces": "_nyx_custom_vfaces",
                 "palette": "_skin_joints"}[field]
    owner = skin._entity if field == "faces" else skin
    # Unsigned overflow and fractional input must be checked before int64 casts.
    dtype = np.uint64 if invalid == 2**64-1 else np.float64
    values = np.asarray(getattr(owner, attribute), dtype=dtype)
    values.flat[0] = invalid
    setattr(owner, attribute, values)
    with pytest.warns(RuntimeWarning, match="retaining capsule"):
        assert ForearmMeshRefiner(skin).halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is None


@pytest.mark.parametrize("dtype", [bool, str])
def test_non_numeric_palette_indices_are_not_coerced(dtype):
    skin = skin_fixture()
    skin._skin_joints = skin._skin_joints.astype(dtype)
    with pytest.warns(RuntimeWarning, match="retaining capsule"):
        assert ForearmMeshRefiner(skin).halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is None


def test_integral_float_indices_remain_supported_without_changing_selection():
    skin = skin_fixture()
    skin._skin_joints = skin._skin_joints.astype(float)
    skin._vert_joints_cuda = skin._vert_joints_cuda.astype(float)
    skin._entity._nyx_custom_vfaces = skin._entity._nyx_custom_vfaces.astype(float)
    refiner = ForearmMeshRefiner(skin)
    assert refiner.halfspaces("RightForeArm", [0, 0, 0], [0, 0, 1]) is not None
    np.testing.assert_array_equal(refiner._indices["RightForeArm"], np.arange(9))


def query_with_skin(point, skin, *, side="Right", part="forearm", margin=0.0, ignored_capsules=()):
    bones = ("ForeArm", "Hand") if part == "forearm" else ("Arm", "ForeArm")
    positions = {f"{side}{bones[0]}": np.array([0, 0, 0]),
                 f"{side}{bones[1]}": np.array([0, 0, 1])}
    skin.get_global_translation = lambda name: (positions[name], None)
    avatar = SimpleNamespace(skin=skin)
    rig = AvatarCollider(None, avatar, visualization=False)
    points = np.array([point], dtype=float)
    geom = SimpleNamespace(init_verts=points, get_verts=lambda: points)
    robot = SimpleNamespace(links=[SimpleNamespace(name="link5", _geoms=[geom])])
    return AvatarCollisionChecker(rig, robot, margin=margin, ignored_capsules=ignored_capsules)


def test_checker_rejects_empty_forearm_proxy_space_but_keeps_actual_forearm():
    assert not query_with_skin([.035, 0, .8], skin_fixture()).check()["collided"]
    assert query_with_skin([.01, 0, .8], skin_fixture()).check()["collided"]
    assert query_with_skin([0, -.002, 1.004], skin_fixture()).check()["collided"]


def test_left_forearm_refinement_preserves_surface_and_mixed_wrist_seam():
    for point in ([.01, 0, .8], [0, -.002, 1.004]):
        result = query_with_skin(point, skin_fixture("Left"), side="Left").check()
        assert result["collided"]
        assert result["pairs"][0][1] == "l_forearm"
    assert not query_with_skin([.035, 0, .8], skin_fixture("Left"), side="Left").check()["collided"]


@pytest.mark.parametrize("point,margin,collided", [
    ([.025, 0, .8], .006, True),
    ([.025, 0, .8], .004, False),
    # At a corner, plane residual10 mm is a conservative bound on the true
    # hull distance sqrt(2)*10 mm. Positive margins must not under-detect.
    ([.03, .03, .8], .011, True),
    ([.03, .03, .8], .009, False),
    # Positive broad-phase capsule margin alone cannot imply a hull hit.
    ([.055, 0, .8], .006, False),
])
def test_positive_hull_margin_is_conservative(point, margin, collided):
    result = query_with_skin(point, skin_fixture(), margin=margin).check()
    assert result["collided"] is collided


def test_deeper_allowed_palm_cannot_mask_true_mesh_forearm_contact():
    query = query_with_skin([.01, 0, .8], skin_fixture(), ignored_capsules=("r_palm",))
    query.collider.current_boxes = lambda: [
        ("r_palm", [.01, 0, .8], np.eye(3), [.04, .04, .04]),
    ]
    result = query.check()
    assert result["collided"]
    assert result["pairs"][0][1] == "r_forearm"
    assert result["pairs"][0][2] == pytest.approx(-.01, abs=2e-9)
    assert result["ignored_pairs"][0][1] == "r_palm"
    assert result["ignored_pairs"][0][2] < result["pairs"][0][2]


def test_checker_retains_contact_if_mesh_is_unsupported():
    assert query_with_skin([.035, 0, .8], SimpleNamespace()).check()["collided"]


def test_custom_capsule_geometry_is_not_silently_refined():
    skin = skin_fixture()
    rig = AvatarCollider(None, SimpleNamespace(skin=skin), visualization=False,
                         segments=[("r_forearm", "RightForeArm", "RightHand", 1, .05)])
    assert rig.forearm_mesh_halfspaces("r_forearm", [0, 0, 0], [0, 0, 1]) is None


def test_hull_equations_have_unit_normals_and_only_numerical_padding():
    equations = convex_halfspaces(skin_fixture()._latest_render_vverts[:8])
    np.testing.assert_allclose(np.linalg.norm(equations[:, :3], axis=1), 1)
    assert np.max(np.array([.020001, 0, .5]) @ equations[:, :3].T + equations[:, 3]) > 9e-7


@pytest.mark.parametrize("side,prefix", [("Left", "l"), ("Right", "r")])
def test_upper_arm_refinement_preserves_surface_and_complete_elbow_seam(side, prefix):
    skin = skin_fixture(side)
    # Keep a closed shaft and the mixed elbow seam. The family labels are
    # intentionally still ForeArm/Hand: upper-arm spatial coverage ignores them.
    skin._entity._nyx_custom_vfaces = skin._entity._nyx_custom_vfaces[:-1]
    for point in ([.01, 0, .8], [0, -.002, 1.004]):
        query = query_with_skin(point, skin, side=side, part="upper_arm")
        result = query.check()
        assert result["collided"]
        assert result["pairs"][0][1] == f"{prefix}_upper_arm"
        assert query.collider._forearm_refiner._indices == {}
    assert not query_with_skin([.035, 0, .8], skin, side=side, part="upper_arm").check()["collided"]


def test_upper_arm_without_supported_skin_retains_original_capsule():
    assert query_with_skin([.035, 0, .8], SimpleNamespace(), part="upper_arm").check()["collided"]


def test_custom_upper_arm_capsule_is_not_silently_refined():
    rig = AvatarCollider(None, SimpleNamespace(skin=skin_fixture()), visualization=False,
                         segments=[("r_upper_arm", "RightArm", "RightForeArm", 1, .055)])
    assert rig.forearm_mesh_halfspaces("r_upper_arm", [0, 0, 0], [0, 0, 1]) is None
    assert rig.capsule_mesh_halfspaces("r_upper_arm", [0, 0, 0], [0, 0, 1], .055) is None


def test_upper_arm_includes_complete_neighbor_triangles_independent_of_weights():
    skin = skin_fixture()
    equations = ForearmMeshRefiner(skin).capsule_halfspaces([0, 0, 0], [0, 0, 1], .055)
    # Last triangle touches the capsule box only near the elbow; all of its
    # vertices, even distant shoulder/hand-dominant ones, must remain covered.
    residual = skin._latest_render_vverts @ equations[:, :3].T + equations[:, 3]
    assert np.max(residual) <= 1e-12


def test_upper_arm_includes_crossing_triangle_without_any_vertex_inside_box():
    skin = skin_fixture()
    skin._latest_render_vverts = np.vstack([
        skin._latest_render_vverts[:8], [-.1, -.1, .5], [.1, -.1, .5], [0, .1, .5],
    ])
    skin._entity._nyx_custom_vfaces = np.vstack([skin._entity._nyx_custom_vfaces[:12], [8, 9, 10]])
    equations = ForearmMeshRefiner(skin).capsule_halfspaces([0, 0, 0], [0, 0, 1], .055)
    residual = skin._latest_render_vverts[8:] @ equations[:, :3].T + equations[:, 3]
    assert np.max(residual) <= 1e-12


def test_upper_arm_positive_margin_expands_for_candidates_and_nearby_skin():
    skin = skin_fixture()
    # Skin at x=.066 is outside r+margin=.065 but within margin of a
    # candidate point x=.06. Selecting only the old-radius box would miss it.
    skin._latest_render_vverts = np.vstack([
        skin._latest_render_vverts[:8], [.066, -.01, .4], [.066, .01, .4], [.066, 0, .6],
    ])
    skin._entity._nyx_custom_vfaces = np.vstack([skin._entity._nyx_custom_vfaces[:12], [8, 9, 10]])
    query = query_with_skin([.06, 0, .5], skin, part="upper_arm", margin=.01)
    assert query.check()["collided"]
    refiner = query.collider._forearm_refiner
    expanded = refiner.capsule_halfspaces([0, 0, 0], [0, 0, 1], .055, .01)
    assert np.max(skin._latest_render_vverts[8:] @ expanded[:, :3].T + expanded[:, 3]) <= 1e-12
    strict = refiner.capsule_halfspaces([0, 0, 0], [0, 0, 1], .055, 0)
    assert np.max(np.array([.06, 0, .5]) @ strict[:, :3].T + strict[:, 3]) > .03


def test_upper_arm_spatial_pose_cache_updates_without_changing_family_cache():
    skin = skin_fixture()
    refiner = ForearmMeshRefiner(skin)
    first = refiner.capsule_halfspaces([0, 0, 0], [0, 0, 1], .055)
    assert refiner.capsule_halfspaces([0, 0, 0], [0, 0, 1], .055) is first
    skin._latest_render_vverts = skin._latest_render_vverts + [.4, .2, -.3]
    second = refiner.capsule_halfspaces([.4, .2, -.3], [.4, .2, .7], .055)
    assert second is not first
    assert np.max(skin._latest_render_vverts @ second[:, :3].T + second[:, 3]) <= 1e-12


@pytest.mark.parametrize("broken", ["indices", "fractional", "nonfinite", "degenerate", "absent_faces"])
def test_upper_arm_invalid_geometry_preserves_old_capsule(broken):
    skin = skin_fixture()
    if broken == "indices":
        skin._entity._nyx_custom_vfaces[0, 0] = 10000
    elif broken == "fractional":
        skin._entity._nyx_custom_vfaces = skin._entity._nyx_custom_vfaces.astype(float)
        skin._entity._nyx_custom_vfaces[0, 0] = .5
    elif broken == "nonfinite":
        skin._latest_render_vverts[0, 0] = np.nan
    elif broken == "degenerate":
        skin._latest_render_vverts[:] = 0
    else:
        del skin._entity._nyx_custom_vfaces
    with pytest.warns(RuntimeWarning, match="retaining capsule"):
        assert query_with_skin([.035, 0, .8], skin, part="upper_arm").check()["collided"]
