"""Joint-boundary geometry and intentional-contact collision regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.collision_checker import AvatarCollisionChecker, _capsule_distances
from envs.avatar.collider import AvatarCollider


def checker(points, *, clipped=True, palm=False, ignore_palm=False, margin=0.0):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    geom = SimpleNamespace(init_verts=points, get_verts=lambda: points)
    entity = SimpleNamespace(links=[SimpleNamespace(name="link6", _geoms=[geom])])
    collider = SimpleNamespace(
        current_capsules=lambda: [("r_forearm", [0, 0, 0], [0, 0, 4], 1.0)],
        current_boxes=lambda: ([("r_palm", [0, 0, 3.9], np.eye(3), [.5, .5, .5])]
                               if palm else []),
        distal_clipped_capsules={"r_forearm"} if clipped else set(),
    )
    return AvatarCollisionChecker(collider, entity, margin=margin,
                                  ignored_capsules=("r_palm",) if ignore_palm else ())


def test_distal_cap_does_not_claim_space_beyond_wrist():
    point = [[0.1, 0, 4.1]]
    assert checker(point, clipped=False).check()["collided"]
    result = checker(point).check()
    assert not result["collided"]
    assert result["min_distance"] == pytest.approx(.1)


def test_reviewed_take_seed1_point_is_beyond_wrist_not_in_forearm():
    # Frozen take seed1, presentation sim-step90/video-frame8. The original
    # full sphere reports22.2mm forearm penetration at this rendered point.
    start = [-.20574854335248857, .7292825662352965, 1.1524298916066469]
    wrist = [-.009570271749755863, .6211975632833592, 1.0313653944048582]
    point = [[-.01431694, .61967856, 1.00400209]]
    original = _capsule_distances(point, start, wrist, .05)
    corrected = _capsule_distances(point, start, wrist, .05, flat_distal=True)
    assert original[0] == pytest.approx(-.02218654, abs=1e-8)
    assert corrected[0] > .009


@pytest.mark.parametrize("point", [[.1, 0, 3.9], [.8, 0, 2], [0, 0, -.8]])
def test_true_forearm_shaft_and_proximal_contacts_remain(point):
    assert checker([point]).check()["collided"]


def test_distal_face_and_positive_contact_margin():
    assert checker([[0, 0, 4]]).check()["collided"]
    assert checker([[0, 0, 4.01]], margin=.02).check()["collided"]
    assert not checker([[0, 0, 4.03]], margin=.02).check()["collided"]


def test_flat_distal_distance_is_exact_at_outside_corner():
    distance = _capsule_distances([[1.1, 0, 4.2]], [0, 0, 0], [0, 0, 4], 1,
                                  flat_distal=True)
    assert distance[0] == pytest.approx(np.hypot(.1, .2))


def test_forearm_distance_is_rigid_transform_invariant():
    points = np.array([[.8, 0, 2], [.1, 0, 4.1], [0, 0, -.8]])
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    translation = np.array([.2, -.6, 1.1])
    expected = _capsule_distances(points, [0, 0, 0], [0, 0, 4], 1, flat_distal=True)
    actual = _capsule_distances(points @ rotation.T + translation, translation,
                                np.array([0, 0, 4]) @ rotation.T + translation,
                                1, flat_distal=True)
    np.testing.assert_allclose(actual, expected)


def test_clip_preserves_every_old_overlap_on_forearm_side_of_wrist():
    points = np.random.RandomState(19).uniform([-2, -2, -2], [2, 2, 4], (2000, 3))
    original = _capsule_distances(points, [0, 0, 0], [0, 0, 4], 1)
    corrected = _capsule_distances(points, [0, 0, 0], [0, 0, 4], 1, flat_distal=True)
    np.testing.assert_array_equal(original <= 0, corrected <= 0)


def test_hand_contact_is_still_detected_by_hand_shape():
    result = checker([[0, 0, 4.1]], palm=True).check()
    assert result["collided"]
    assert result["pairs"][0][1] == "r_palm"


def test_configured_hand_contact_can_be_ignored_without_forearm_false_positive():
    result = checker([[0, 0, 4.1]], palm=True, ignore_palm=True).check()
    assert not result["collided"]
    assert result["ignored_pairs"][0][1] == "r_palm"


def test_ignored_deeper_palm_cannot_hide_true_forearm_overlap():
    result = checker([[0, 0, 3.9]], palm=True, ignore_palm=True).check()
    assert result["collided"]
    assert result["pairs"][0][1] == "r_forearm"
    assert result["ignored_pairs"][0][1] == "r_palm"


def test_clipping_is_opt_in_for_colliders_without_joint_metadata():
    query = checker([[0, 0, 4.1]], clipped=False)
    del query.collider.distal_clipped_capsules
    assert query.check()["collided"]


def test_default_avatar_does_not_remove_unvalidated_wrist_seam_coverage():
    rig = AvatarCollider(None, SimpleNamespace(skin=None), visualization=False)
    assert rig.distal_clipped_capsules == set()
    forearms = [r for name, _a, _b, r, _length, _entity in rig.entities
                if name in {"l_forearm", "r_forearm"}]
    assert forearms == [.05, .05]


def test_clipping_requires_explicit_existing_segment_names():
    segments = [
        ("custom_arm", "RightForeArm", "RightHand", .25, .06),
        ("r_forearm", "Spine1", "Neck", .27, .14),
    ]
    rig = AvatarCollider(None, SimpleNamespace(skin=None), visualization=False,
                         segments=segments, distal_clipped_capsules=("custom_arm",))
    assert rig.distal_clipped_capsules == {"custom_arm"}
    with pytest.raises(ValueError, match="Unknown clipped avatar segments"):
        AvatarCollider(None, SimpleNamespace(skin=None), visualization=False,
                       segments=segments, distal_clipped_capsules=("typo",))
