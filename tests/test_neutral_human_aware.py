"""Human-aware neutral full-state expert: clearance math and the wait gate."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import neutral_human_aware as ha
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin


def test_point_capsule_distance_and_synchronous_clearance():
    cap = ("r_forearm", np.array([0.0, 0.0, 0.8]), np.array([0.0, 0.2, 0.8]), 0.04)
    assert ha.point_capsule_distance([0.1, 0.1, 0.8], *cap[1:]) == pytest.approx(0.06)
    far = ("r_forearm", np.array([1.0, 0.0, 0.8]), np.array([1.0, 0.2, 0.8]), 0.04)
    pts = np.array([[0.1, 0.1, 0.8], [0.5, 0.1, 0.8]])
    # the human is near only at t=0 while the link is far at t=1: synchronous
    assert ha.path_human_clearance([(pts, 0.0)], [[cap], [far]]) == pytest.approx(0.06)
    assert ha.path_human_clearance([(pts, 0.05)], [[cap], [far]]) == pytest.approx(0.01)
    assert ha.path_human_clearance([(pts, 0.0)], [[far], [far]]) == pytest.approx(0.46)


class _Gate(ha._NeutralHumanAwareMixin):
    def __init__(self, clearances, **cfg):
        self.config = {"neutral_human_aware_expert": True, "action_substeps": 25, **cfg}
        self.avatar = object()
        self._neutral_avatar_reset_seed = 1
        self._clear = list(clearances)
        self.sim = 0
        self.calls = []

    replan_clearance = None     # None: re-plan fails
    threat = float("inf")       # robot-where-it-stands clearance

    def _neutral_aware_robot_threat(self, arm_tag, prm):
        return self.threat

    def _neutral_aware_yield(self, arm_tag, prm):
        self.yielded = getattr(self, "yielded", 0) + 1
        return True

    def _neutral_aware_clearance(self, positions, arm_tag, prm):
        if getattr(positions, "shape", (0,))[0] == 99:       # the re-plan
            return self.replan_clearance
        self.calls.append(self.sim)
        return self._clear.pop(0) if len(self._clear) > 1 else self._clear[0]

    detour = None               # detour re-plan result (None: no detour clears)

    def _neutral_aware_detour(self, result, arm_tag, prm):
        self.detours = getattr(self, "detours", 0) + 1
        return self.detour

    def _neutral_aware_replan(self, result, arm_tag):
        self.replans = getattr(self, "replans", 0) + 1
        if self.replan_clearance is None:
            return None
        return SimpleNamespace(position=np.zeros((99, 7)))

    def step_sim(self):
        self.sim += 1


PLAN = SimpleNamespace(position=np.zeros((40, 7)))


def test_gate_holds_until_clear_rechecking_every_policy_tick():
    g = _Gate([0.01, 0.02, 0.2])
    assert g._neutral_aware_gate(PLAN, "right") is PLAN    # re-plan failed: keep validated plan
    assert g.calls == [0, 25, 50]          # re-check every policy tick
    assert g.sim == 50
    st = g._neutral_aware_stats()
    assert st["waits"] == 1 and st["wait_sim_steps"] == 50 and st["budget_exhausted"] == 0
    assert st["min_clearance_m"] == pytest.approx(0.01)


def test_resume_uses_a_clear_replan():
    g = _Gate([0.01, 0.2])
    g.replan_clearance = 0.3
    out = g._neutral_aware_gate(PLAN, "right")
    assert out is not PLAN and out.position.shape[0] == 99
    assert g._neutral_aware_stats()["replans_used"] == 1


def test_out_of_budget_aborts_instead_of_driving_into_the_human():
    g = _Gate([0.0], neutral_aware_max_wait_segment=100)
    g.plan_success = True
    g.replan_clearance = 0.01                              # re-plan still meets the human
    assert g._neutral_aware_gate(PLAN, "right") is None
    assert g.plan_success is False and g._neutral_aware_stats()["aborted"] == 1
    g2 = _Gate([0.0], neutral_aware_max_wait_segment=100)
    g2.replan_clearance = 0.2                              # a clear re-plan exists
    assert g2._neutral_aware_gate(PLAN, "right").position.shape[0] == 99


def test_gate_is_bounded_and_skips_empty_plans():
    g = _Gate([0.0], neutral_aware_max_wait_segment=100)
    g.plan_success = True
    g._neutral_aware_gate(PLAN, "right")
    assert g.sim == 100 and g._neutral_aware_stats()["budget_exhausted"] == 1
    g2 = _Gate([0.0], neutral_aware_max_wait_episode=60)
    g2.plan_success = True
    g2._neutral_aware_gate(PLAN, "right")
    g2._neutral_aware_gate(PLAN, "right")
    assert g2.sim == 50 and g2._neutral_aware_stats()["budget_exhausted"] == 2
    g3 = _Gate([0.0])
    empty = SimpleNamespace(position=np.zeros((0, 7)))
    assert g3._neutral_aware_gate(empty, "right") is empty
    assert g3.sim == 0 and g3.calls == []


def test_clear_path_never_waits_or_replans():
    g = _Gate([0.3])
    assert g._neutral_aware_gate(PLAN, "right") is PLAN
    assert g.sim == 0 and g._neutral_aware_stats()["waits"] == 0 and not hasattr(g, "replans")


def test_disabled_for_blind_and_policies():
    from scripts.eval_expert_set import build_config
    row = {"task": "dump_bin_copresence", "category": "copresence", "seed": 1, "config_overrides": {}}
    assert build_config(row, "expert_full_state")["neutral_human_aware_expert"] is True
    assert build_config(row, "expert")["neutral_human_aware_expert"] is False
    off = dict(row, config_overrides={"neutral_human_aware_expert": False})
    assert build_config(off, "expert_full_state")["neutral_human_aware_expert"] is False
    on = dict(row, config_overrides={"neutral_human_aware_expert": True})
    assert build_config(on, "expert")["neutral_human_aware_expert"] is False
    g = _Gate([0.0])
    g.config["neutral_human_aware_expert"] = False
    assert not g._neutral_aware_enabled() and g.neutral_aware_metrics() == {}


def test_every_neutral_task_routes_execute_plan_through_the_gate():
    from envs.tasks import resolve_task_class
    for name in ("rank_blocks_by_color_copresence", "rank_blocks_by_size_copresence", "categorize_objects_copresence",
                 "dump_bin_copresence", "place_bread_in_basket_copresence", "place_burger_and_fries_copresence",
                 "place_pair_of_shoes_copresence", "place_food_in_skillet_copresence",
                 "put_object_in_cabinet_copresence", "stack_three_bowls_copresence"):
        _, cls = resolve_task_class(name)
        owners = [k for k in cls.__mro__ if "execute_plan" in k.__dict__]
        i = owners.index(ha._NeutralHumanAwareMixin)
        # Only wrappers that delegate to super().execute_plan may come first.
        assert all(k.__name__ == "BlockNeutralPublicGraspMixin" for k in owners[:i]), (name, owners)
        assert issubclass(cls, NeutralAvatarTableWorkMixin)


def test_robot_yields_when_the_human_would_reach_it_and_then_replans():
    g = _Gate([0.0, 0.0, 0.3])
    g.threat = 0.01
    g.replan_clearance = 0.02            # even a re-plan below clearance is used: the old plan is stale
    out = g._neutral_aware_gate(PLAN, "right")
    st = g._neutral_aware_stats()
    assert g.yielded == 2 and st["yields"] == 2            # capped per segment
    assert out is not PLAN and out.position.shape[0] == 99
    g2 = _Gate([0.0, 0.3])
    g2.threat = 0.01
    g2.plan_success = True
    assert g2._neutral_aware_gate(PLAN, "right") is None   # moved, and no re-plan exists
    assert g2.plan_success is False


def test_conflict_tries_a_detour_before_waiting():
    g = _Gate([0.0])
    g.detour = SimpleNamespace(position=np.zeros((77, 7)))
    assert g._neutral_aware_gate(PLAN, "right") is g.detour
    assert g.sim == 0 and g.detours == 1                   # no wait at all
    g2 = _Gate([0.01, 0.3])
    assert g2._neutral_aware_gate(PLAN, "right") is PLAN   # no detour: wait as before
    assert g2.detours == 1 and g2.sim == 25


def test_out_of_budget_uses_a_detour_instead_of_aborting():
    g = _Gate([0.0], neutral_aware_max_wait_segment=100)
    g.plan_success = True
    g.replan_clearance = 0.01
    first = SimpleNamespace(position=np.zeros((77, 7)))
    calls = []

    def detour(result, arm_tag, prm):
        calls.append(result)
        return None if len(calls) == 1 else first           # human moved: second try clears
    g._neutral_aware_detour = detour
    assert g._neutral_aware_gate(PLAN, "right") is first
    assert g.plan_success is True and len(calls) == 2


def test_dense_capsule_points_cover_the_grown_surface():
    cap = ("r_forearm", np.array([0.0, 0.0, 0.8]), np.array([0.0, 0.3, 0.8]), 0.04)
    pts = ha.dense_capsule_points([cap], 0.06)
    d = np.array([ha.point_capsule_distance(p, *cap[1:]) for p in pts])
    assert np.allclose(d, 0.06, atol=1e-6)
    # no gap wider than the spacing on a shaft ring
    ring = pts[np.abs(pts[:, 1] - 0.15) < 0.016]
    assert len(ring) >= int(2 * np.pi * 0.10 / ha.DETOUR_POINT_SPACING_M)
