"""Cooperative milk stacking semantics, with no simulator or expert execution.

The coordinates are retained from user-reviewed seed 107094835. Contacts are
synthetic evidence, not a claim that the retained trace measured those forces.
Native replay separately verifies the actual placement's physical support.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.task_bases.categorize_support import stack
from envs.task_bases.categorize import CategorizeTaskCore
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.categorize_objects_copresence import CategorizeObjectsCopresence
from envs.utils import Pose


class _Entity:
    def __init__(self, center, aabb, geom_start):
        self.center = np.array(center, dtype=float)
        self.aabb = np.array(aabb, dtype=float)
        self.geom_start = geom_start
        self.geom_end = geom_start + 3
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def get_AABB(self):
        return self.aabb.copy()

    def get_vel(self):
        return self.velocity.copy()

    def get_ang(self):
        return self.angular_velocity.copy()


def _contact_rows(pairs):
    """Pairs contain (geom_a, geom_b, force_on_a); action/reaction is explicit."""
    forces = np.asarray([p[2] for p in pairs], dtype=float).reshape(-1, 3)
    return {
        "geom_a": np.asarray([p[0] for p in pairs], dtype=int),
        "geom_b": np.asarray([p[1] for p in pairs], dtype=int),
        "force_a": forces,
        "force_b": -forces,
    }


def _install_contacts(monkeypatch, case, pairs):
    rows = _contact_rows(pairs)
    case.contact_reads = []

    def contacts(entity):
        case.contact_reads.append(entity)
        include = (
            ((rows["geom_a"] >= entity.geom_start)
             & (rows["geom_a"] < entity.geom_end))
            | ((rows["geom_b"] >= entity.geom_start)
               & (rows["geom_b"] < entity.geom_end))
        )
        return {name: values[include].copy() for name, values in rows.items()}

    monkeypatch.setattr(stack, "entity_contacts", contacts)
    return rows


def _task_case(monkeypatch, task_cls=CategorizeObjectsContribution, config=None):
    task = task_cls(config or {})
    task.plan_success = True
    task.avatar_collision_checker = None
    task.avatar_collided = False
    task.basket_poses = [
        Pose((-.2548714092, .15, .821)),
        Pose((.25, .15, .821)),
    ]
    upper = NS(entity=_Entity(
        [-.2518689783, .2167503306, .8976211218],
        [[-.2898289561, .1482987851, .8458285332],
         [-.2154164016, .2741203904, .9351179004]],
        10,
    ))
    lower = NS(entity=_Entity(
        [-.2500621222, .1431849568, .8322831772],
        [[-.2843833864, .1103674173, .7723207474],
         [-.2157211602, .1763407141, .8925635815]],
        20,
    ))
    task._sort_targets = [(upper, *stack.MILK_MODEL, 0)]
    task._avatar_sort_targets = [(lower, *stack.MILK_MODEL, 0)]
    task._get_object_world_center = (
        lambda actor, _name, _model: actor.entity.center.copy()
    )
    task.robot = NS(
        left_arm=NS(entity=NS(geom_start=100, geom_end=110)),
        right_arm=NS(entity=NS(geom_start=110, geom_end=120)),
    )
    task.avatar = NS(robot=NS(
        attached_object=[None, None], two_hand_attached_object=None,
    ))
    case = NS(task=task, upper=upper, lower=lower)
    _install_contacts(monkeypatch, case, [(10, 20, [0., 0., .2])])
    return case


@pytest.fixture
def case(monkeypatch):
    return _task_case(monkeypatch)


def _upper_metrics(case):
    metrics = case.task._basket_metrics()
    return metrics, metrics["basket_objects"][0]


def _assert_rejected(case, reason=None):
    metrics, upper = _upper_metrics(case)
    assert upper["stack_success"] is False
    assert upper["success"] is False
    assert upper["footprint_success"] is False
    assert metrics["basket_all_correct"] is False
    assert case.task.check_success() is False
    if reason is not None:
        assert upper["stack_support"]["reason"] == reason
    return upper


def test_user_reviewed_tilted_stack_preserves_actual_footprint(case):
    metrics, upper = _upper_metrics(case)
    lower = metrics["basket_objects"][1]
    assert upper["dist_xy"] == pytest.approx(.0668178, abs=1e-6)
    assert upper["rel_aabb_hi_xy"][1] > .124
    assert metrics["basket_footprint_half_xy"] == pytest.approx(.088)
    # Tilt makes world AABBs overlap in height; an axis-aligned z-gap test
    # would reject a placement that is acceptable.
    assert case.upper.entity.aabb[0, 2] < case.lower.entity.aabb[1, 2]
    assert upper["center_success"] is True
    assert upper["footprint_success"] is False
    assert upper["stack_success"] is True
    assert upper["success"] is True
    assert upper["stack_support"] == {
        "success": True,
        "reason": "supported",
        "lower_floor_gap_m": pytest.approx(-.0026792526),
        "linear_speeds_m_s": [0., 0.],
        "angular_speeds_rad_s": [0., 0.],
        "upward_support_force_n": pytest.approx(.2),
        "side": "avatar",
    }
    assert lower["footprint_success"] is True
    assert lower["success"] is True
    assert lower["stack_success"] is False
    assert metrics["basket_scoring_version"] == stack.SCORING_VERSION
    assert metrics["basket_all_correct"] is True
    assert case.task.check_success() is True


@pytest.mark.parametrize("pairs, force", [
    ([(10, 20, [0., 0., .2])], .2),
    ([(20, 10, [0., 0., -.2])], .2),
    ([(11, 21, [0., 0., .13]), (22, 12, [0., 0., -.08])], .21),
    ([(10, 20, [0., 0., .3]), (20, 10, [0., 0., .1])], .2),
])
def test_support_uses_force_on_upper_in_either_pair_orientation(
    monkeypatch, case, pairs, force,
):
    _install_contacts(monkeypatch, case, pairs)
    _, upper = _upper_metrics(case)
    assert upper["stack_success"] is True
    assert upper["stack_support"]["upward_support_force_n"] == pytest.approx(force)


@pytest.mark.parametrize("pairs", [
    [],  # unsupported floating carton
    [(10, 200, [0., 0., 1.])],  # rim or tabletop support only
    [(10, 20, [0., 0., 0.])],  # collision proximity, no load
    [(10, 20, [0., 0., .01])],  # support must exceed the force threshold
    [(10, 20, [1., 0., 0.])],  # horizontal milk contact, rim bears load
    [(10, 20, [0., 0., -.2])],  # the carton is not supported from below
    [(20, 10, [0., 0., .2])],
    [(10, 20, [0., 0., np.nan])],
    [(10, 20, [0., 0., np.inf])],
    [(10, 20, [0., 0., -np.inf])],
])
def test_absent_or_invalid_upward_milk_support_rejected(monkeypatch, case, pairs):
    _install_contacts(monkeypatch, case, pairs)
    _assert_rejected(case, "no_upward_milk_support")


def test_valid_milk_support_may_coexist_with_rim_contact(monkeypatch, case):
    _install_contacts(monkeypatch, case, [
        (10, 20, [0., 0., .2]),
        (11, 200, [0., 0., 1.]),
    ])
    assert case.task.check_success() is True


@pytest.mark.parametrize("side, orientation, arm_geom", [
    ("upper", "a", 100), ("upper", "b", 119),
    ("lower", "a", 109), ("lower", "b", 110),
])
def test_either_carton_touching_either_robot_arm_rejected(
    monkeypatch, case, side, orientation, arm_geom,
):
    entity_geom = getattr(case, side).entity.geom_start
    pair = (entity_geom, arm_geom, [0., 0., 0.])
    if orientation == "b":
        pair = (arm_geom, entity_geom, [0., 0., 0.])
    _install_contacts(monkeypatch, case, [(10, 20, [0., 0., .2]), pair])
    _assert_rejected(case, "robot_contact")


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize("attachment", ["left", "right", "two_hand"])
def test_either_carton_avatar_attached_rejected(case, side, attachment):
    attached = (getattr(case, side).entity, object())
    rig = case.task.avatar.robot
    if attachment == "two_hand":
        rig.two_hand_attached_object = attached
    else:
        rig.attached_object[int(attachment == "right")] = attached
    _assert_rejected(case, "avatar_attached")
    assert not case.contact_reads


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize("attribute, velocity", [
    ("velocity", [.03001, 0., 0.]),
    ("velocity", [.022, .022, 0.]),  # threshold is vector norm, not each axis
    ("angular_velocity", [0., .50001, 0.]),
    ("angular_velocity", [.36, .36, 0.]),
])
def test_both_cartons_must_be_settled(case, side, attribute, velocity):
    setattr(getattr(case, side).entity, attribute, np.asarray(velocity))
    _assert_rejected(case, "not_settled")


def test_speed_thresholds_are_inclusive(case):
    case.upper.entity.velocity = np.array([stack.MAX_LINEAR_SPEED, 0., 0.])
    case.lower.entity.angular_velocity = np.array([0., stack.MAX_ANGULAR_SPEED, 0.])
    assert case.task.check_success() is True


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize("attribute", ["velocity", "angular_velocity"])
@pytest.mark.parametrize("value", [[np.nan, 0., 0.], [np.inf, 0., 0.], [[0., 0., 0.]]])
def test_bad_velocity_evidence_fails_closed(case, side, attribute, value):
    setattr(getattr(case, side).entity, attribute, np.asarray(value))
    row = _assert_rejected(case, "support_unavailable")
    assert "ValueError" in row["stack_support"]["error"]


@pytest.mark.parametrize("error", [
    AttributeError("no native velocity"), KeyError("force_a"),
    TypeError("bad observation"), ValueError("invalid contacts"),
    RuntimeError("native contact read failed"),
])
def test_contact_read_failure_is_diagnostic_not_an_episode_crash(monkeypatch, case, error):
    def unavailable(_entity):
        raise error

    monkeypatch.setattr(stack, "entity_contacts", unavailable)
    row = _assert_rejected(case, "support_unavailable")
    assert type(error).__name__ in row["stack_support"]["error"]


@pytest.mark.parametrize("missing", ["avatar_rig", "attachments", "arm_entities", "velocity"])
def test_missing_release_observations_fail_closed(case, missing):
    if missing == "avatar_rig":
        case.task.avatar = object()
    elif missing == "attachments":
        del case.task.avatar.robot.attached_object
    elif missing == "arm_entities":
        case.task.robot.left_arm = case.task.robot.right_arm = None
    else:
        case.upper.entity.get_vel = None
    row = _assert_rejected(case, "support_unavailable")
    assert row["stack_support"]["error"]


@pytest.mark.parametrize("upper_z", [.8322831772, .80])
def test_upper_must_be_above_lower_carton(case, upper_z):
    case.upper.entity.center[2] = upper_z
    _assert_rejected(case, "no_valid_lower_carton")
    assert not case.contact_reads


@pytest.mark.parametrize("bottom", [.765, .78001, .70, .86])
def test_lower_carton_must_rest_on_basket_floor(case, bottom):
    # The new alternative must not certify a tabletop, floating, below-table,
    # or elevated rim support merely because that lower actor's XY passes.
    case.lower.entity.aabb[0, 2] = bottom
    _assert_rejected(case, "lower_not_on_floor")
    assert not case.contact_reads


@pytest.mark.parametrize("bad_z", [np.nan, np.inf, .90])
def test_invalid_lower_carton_vertical_bounds_fail_closed(case, bad_z):
    case.lower.entity.aabb[0, 2] = bad_z
    row = _assert_rejected(case, "support_unavailable")
    assert "invalid lower carton bounds" in row["stack_support"]["error"]


@pytest.mark.parametrize("change", ["wrong_basket", "outside_center", "overhanging", "not_milk", "other_milk_model"])
def test_support_carton_requires_original_correct_basket_rule(case, change):
    if change == "wrong_basket":
        # Even coincident basket coordinates do not make a different category
        # assignment eligible as the lower support.
        case.task.basket_poses[1] = case.task.basket_poses[0]
        case.task._avatar_sort_targets = [(case.lower, *stack.MILK_MODEL, 1)]
    elif change == "outside_center":
        case.lower.entity.center[0] += .2
    elif change == "overhanging":
        case.lower.entity.aabb[1, 1] += .2
    elif change == "not_milk":
        case.task._avatar_sort_targets = [(case.lower, "100_seal", 1, 0)]
    else:
        case.task._avatar_sort_targets = [(case.lower, "038_milk-box", 1, 0)]
    _assert_rejected(case, "no_valid_lower_carton")
    assert not case.contact_reads


def test_upper_center_tolerance_not_widened(case):
    case.upper.entity.center[1] = case.task.basket_poses[0].p[1] + .1001
    _assert_rejected(case, "center_outside")
    assert not case.contact_reads


@pytest.mark.parametrize("side, height", [
    ("upper", np.inf), ("upper", np.nan),
    ("lower", -np.inf), ("lower", np.nan),
])
def test_nonfinite_carton_center_cannot_certify_stack(case, side, height):
    getattr(case, side).entity.center[2] = height
    _assert_rejected(case)


@pytest.mark.parametrize("name, model", [("100_seal", 1), ("038_milk-box", 1), ("031_jam-jar", 0)])
def test_alternative_rule_not_applied_to_other_objects(case, name, model):
    case.task._sort_targets = [(case.upper, name, model, 0)]
    metrics, upper = _upper_metrics(case)
    assert upper["success"] is False
    assert "stack_success" not in upper
    assert "stack_support" not in upper
    assert metrics["basket_all_correct"] is False
    assert case.task.check_success() is False
    assert not case.contact_reads


def test_ordinary_milk_footprint_pass_does_not_require_new_contact_api(case, monkeypatch):
    case.upper.entity.aabb = case.lower.entity.aabb.copy()
    case.upper.entity.center = case.lower.entity.center.copy()

    def forbidden(_entity):
        pytest.fail("ordinary placement should not need stacking evidence")

    monkeypatch.setattr(stack, "entity_contacts", forbidden)
    metrics, upper = _upper_metrics(case)
    assert upper["footprint_success"] is True
    assert upper["success"] is True
    assert upper["stack_success"] is False
    assert "stack_support" not in upper
    assert metrics["basket_all_correct"] is True
    assert case.task.check_success() is True


@pytest.mark.parametrize("task_cls", [CategorizeTaskCore, CategorizeObjectsCopresence, CategorizeObjectsInterruption])
def test_other_family_scenarios_keep_original_footprint_rule(monkeypatch, task_cls):
    case = _task_case(monkeypatch, task_cls)
    metrics = (case.task._robot_basket_metrics() if task_cls is CategorizeObjectsCopresence
               else case.task._basket_metrics())
    assert "basket_scoring_version" not in metrics
    assert metrics["basket_objects"][0]["footprint_success"] is False
    assert "stack_success" not in metrics["basket_objects"][0]
    assert metrics["basket_all_correct"] is False
    assert case.task.check_success() is False
    assert not case.contact_reads


@pytest.mark.parametrize("gate", ["plan_failed", "avatar_collision"])
def test_success_preserves_global_plan_and_collision_gates(case, gate):
    assert case.task._basket_metrics()["basket_all_correct"] is True
    if gate == "plan_failed":
        case.task.plan_success = False
    else:
        case.task.avatar_collision_checker = object()
        case.task.avatar_collided = True
    assert case.task.check_success() is False


def test_policy_state_does_not_depend_on_expert_active_pick(case):
    assert not hasattr(case.task, "_categorize_active_pick")
    assert not hasattr(case.task, "_categorize_hold_local")
    assert case.task.check_success() is True
    # A stale expert-only annotation cannot override observed contact release.
    case.task._categorize_active_pick = NS(actor=case.upper)
    case.task._categorize_hold_local = np.array([0., 0., 0.])
    assert case.task.check_success() is True


def test_absent_avatar_does_not_require_attachment_observation(case):
    case.task.avatar = None
    assert case.task.check_success() is True


def test_no_human_mode_does_not_use_ungraded_avatar_carton_as_support(case):
    case.task.config["no_human"] = True
    _assert_rejected(case, "no_valid_lower_carton")


def test_scoring_never_advances_simulation_or_mutates_contact_observations(case, monkeypatch):
    original = _install_contacts(monkeypatch, case, [(10, 20, [0., 0., .2])])
    copied = {key: value.copy() for key, value in original.items()}

    def forbidden(*_args, **_kwargs):
        pytest.fail("scoring must only observe the simulation")

    case.task.step_sim = forbidden
    case.task.scene = NS(step=forbidden)
    assert case.task.check_success() is True
    assert case.task.check_success() is True
    for key in original:
        np.testing.assert_array_equal(original[key], copied[key])
