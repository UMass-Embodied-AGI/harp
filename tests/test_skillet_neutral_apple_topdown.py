"""Neutral skillet apple routes through the family top-down primitive.

Scene-free: no Genesis scene, planner, robot or avatar asset is built.  The
neutral class changes exactly one field of the sampled apple spec; sibling
variants, the other foods, the food sampling and its RNG use are unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from types import SimpleNamespace

from envs.utils import Pose

from envs.task_bases.place_food_in_skillet import (
    FoodSpec,
    PlaceFoodInSkillet,
    SkilletTaskCore,
)
from envs.tasks.place_food_in_skillet_contribution import PlaceFoodInSkilletContribution
from envs.tasks.place_food_in_skillet_interruption import (
    PlaceFoodInSkilletInterruption,
)
from envs.tasks.place_food_in_skillet_copresence import PlaceFoodInSkilletCopresence


ALL_FOODS = ["apple", "burger", "bread"]


def specs_by_name(task):
    return {spec.name: spec for spec in task._enabled_food_specs()}


def assert_same_except(actual: FoodSpec, expected: FoodSpec, **overrides):
    for field in dataclasses.fields(FoodSpec):
        left = getattr(actual, field.name)
        right = overrides.get(field.name, getattr(expected, field.name))
        if isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right, field.name


def test_neutral_apple_is_top_down_and_every_other_field_is_the_base_spec():
    base = specs_by_name(PlaceFoodInSkillet({"foods": ALL_FOODS}))
    neutral = specs_by_name(PlaceFoodInSkilletCopresence({"foods": ALL_FOODS}))
    assert base["apple"].side_grasp is True
    assert neutral["apple"].side_grasp is False
    assert neutral["apple"].use_grasp_helper is False
    assert base["apple"].close_value == 0.30 and neutral["apple"].close_value == 0.50
    assert_same_except(neutral["apple"], base["apple"], side_grasp=False, close_value=0.50)
    for name in ("burger", "bread"):
        assert_same_except(neutral[name], base[name])
    assert [spec.name for spec in PlaceFoodInSkilletCopresence(
        {"foods": ALL_FOODS})._enabled_food_specs()] == ALL_FOODS


def test_top_down_route_is_the_burger_route_in_the_family_dispatch():
    neutral = specs_by_name(PlaceFoodInSkilletCopresence({"foods": ALL_FOODS}))
    # SkilletTaskCore._pick_and_drop_food dispatches side_grasp, then
    # use_grasp_helper, then the inline top-down primitive.
    for name in ALL_FOODS:
        assert (neutral[name].side_grasp, neutral[name].use_grasp_helper) == (False, False)
    assert PlaceFoodInSkilletCopresence._pick_and_drop_food is SkilletTaskCore._pick_and_drop_food


def test_close_value_opt_out_keeps_the_spec_close(monkeypatch):
    monkeypatch.setattr(PlaceFoodInSkilletCopresence, "NEUTRAL_APPLE_CLOSE_VALUE", None)
    base = specs_by_name(PlaceFoodInSkillet({"foods": ALL_FOODS}))
    neutral = specs_by_name(PlaceFoodInSkilletCopresence({"foods": ALL_FOODS}))
    assert_same_except(neutral["apple"], base["apple"], side_grasp=False)


def _pick_trace(task_cls):
    """Drive SkilletTaskCore._pick_and_drop_food with stubbed motion; return gripper commands."""
    calls = []
    apple = specs_by_name(task_cls({"foods": ["apple"]}))["apple"]
    assert apple.side_grasp is False and apple.use_grasp_helper is False

    class Driver(task_cls):
        def __init__(self):
            self.config = {"foods": ["apple"]}
            self.TABLE_TOP_Z = 0.765
            self.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
                tcp_offset=Pose([0., 0., .1034], [1., 0., 0., 0.])))
            self.z = 0.786
        def _actor_center(self, actor, asset_id, model_id=0):
            return np.array([0.2, 0.0, self.z])
        def open_gripper(self, tag, num_steps=0): calls.append(("open", tag))
        def _move_seeded(self, pose7, tag): return True
        def _move_screw(self, pos, tag):
            self.z = float(pos[2]) + 0.01
            return True
        def step_sim(self): pass
        def set_gripper(self, value, tag, num_steps=0): calls.append(("close", float(value)))
        def _pan_bowl_center(self): return np.array([-0.12, 0.08, 0.81])
        def _rotated_top_down_tcp(self, pos, yaw): return Pose(np.asarray(pos, float), [1., 0., 0., 0.])
        def _move_tcp_cartesian(self, a, b, tag, **kw): return True
        def _gentle_place_food(self, *a, **kw): return Pose([0., 0., 0.9], [1., 0., 0., 0.])
        def _retreat_after_food_release(self, *a, **kw): return None

    driver = Driver()
    assert driver._pick_and_drop_food(object(), apple, "right") is True
    return calls


def test_base_top_down_apple_keeps_its_post_lift_regrip():
    calls = _pick_trace(_BaseTopDownApple)
    assert [c for c in calls if c[0] == "close"] == [("close", 0.30), ("close", 0.20)]


def test_neutral_top_down_apple_has_no_post_lift_regrip():
    calls = _pick_trace(PlaceFoodInSkilletCopresence)
    assert [c for c in calls if c[0] == "close"] == [("close", 0.50)]
    assert PlaceFoodInSkillet.APPLE_POST_LIFT_CLOSE_VALUE == 0.20
    assert PlaceFoodInSkilletCopresence.APPLE_POST_LIFT_CLOSE_VALUE is None


class _BaseTopDownApple(PlaceFoodInSkillet):
    """Base task with the apple routed top-down only for the trace (specs otherwise base)."""

    def _enabled_food_specs(self):
        return [dataclasses.replace(s, side_grasp=False) if s.name == "apple" else s
                for s in super()._enabled_food_specs()]


def test_opt_out_restores_the_side_grasp(monkeypatch):
    monkeypatch.setattr(PlaceFoodInSkilletCopresence, "NEUTRAL_APPLE_TOP_DOWN", False)
    base = specs_by_name(PlaceFoodInSkillet({"foods": ALL_FOODS}))
    neutral = specs_by_name(PlaceFoodInSkilletCopresence({"foods": ALL_FOODS}))
    assert neutral["apple"].side_grasp is True
    assert_same_except(neutral["apple"], base["apple"])


def test_sibling_variants_keep_their_own_apple_specs():
    for task_cls in (PlaceFoodInSkillet, PlaceFoodInSkilletContribution, PlaceFoodInSkilletInterruption):
        assert "NEUTRAL_APPLE_TOP_DOWN" not in vars(task_cls)
        assert "_enabled_food_specs" not in vars(task_cls) or task_cls is PlaceFoodInSkilletContribution
    interrupt = specs_by_name(PlaceFoodInSkilletInterruption({"foods": ALL_FOODS}))
    assert interrupt["apple"].side_grasp is True
    seen = False
    for seed in range(12):
        np.random.seed(seed)
        specs = specs_by_name(PlaceFoodInSkilletContribution({}))
        if "apple" in specs:
            # The assist robot's apple uses the top-down
            # primitive too, with the neutral variant's 0.50 close.
            assert specs["apple"].side_grasp is False
            assert specs["apple"].close_value == pytest.approx(0.50)
            seen = True
    assert seen


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 7, 9))
def test_sampling_and_rng_tail_match_the_base_random_object_draw(seed):
    np.random.seed(seed)
    base_specs = PlaceFoodInSkillet({"random_object": True})._enabled_food_specs()
    base_tail = np.random.random_sample(4)
    np.random.seed(seed)
    neutral_specs = PlaceFoodInSkilletCopresence({})._enabled_food_specs()
    neutral_tail = np.random.random_sample(4)
    assert [spec.name for spec in neutral_specs] == [spec.name for spec in base_specs]
    np.testing.assert_array_equal(neutral_tail, base_tail)
    for actual, expected in zip(neutral_specs, base_specs):
        if expected.name == "apple":
            assert_same_except(actual, expected, side_grasp=False, close_value=0.50)
        else:
            assert_same_except(actual, expected)
