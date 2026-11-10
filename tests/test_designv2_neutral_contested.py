"""Neutral co-presence + easy-task randomization contracts.

Pure-python checks (no Genesis scene): contested selector geometry and
determinism, private-RNG isolation, eval trigger timing, the collision-log
start fix, alias resolution, bowl jitter and skillet jitter constants.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import neutral_avatar_contested as nc
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin


@contextmanager
def _global_rng_preserved():
    state = np.random.get_state()
    probe = np.random.RandomState()
    probe.set_state(state)
    expected = probe.randint(0, 2**31 - 1, size=4)
    np.random.set_state(state)
    try:
        yield
        actual = np.random.randint(0, 2**31 - 1, size=4)
        assert np.array_equal(actual, expected), "global np.random stream was consumed"
    finally:
        np.random.set_state(state)


# -- decision -------------------------------------------------------------

def test_contested_draw_is_private_and_deterministic():
    with _global_rng_preserved():
        a = nc.contested_draw(9_000_004, 0.4)
        b = nc.contested_draw(9_000_004, 0.4)
    assert a == b
    assert a[1] == pytest.approx(
        float(np.random.default_rng(9_000_004 + nc.CONTESTED_RNG_SALT).random())
    )
    assert nc.contested_draw(9_000_004, 0.0)[0] is False
    assert nc.contested_draw(9_000_004, 1.0)[0] is True


def test_contested_fraction_is_roughly_honoured():
    hits = sum(nc.contested_draw(seed, 0.4)[0] for seed in range(9_000_000, 9_002_000))
    assert 0.36 < hits / 2000 < 0.44


# -- geometry ---------------------------------------------------------------

def test_point_segment_distance():
    assert nc.point_segment_distance((0, 1), (-1, 0), (1, 0)) == pytest.approx(1.0)
    assert nc.point_segment_distance((2, 0), (-1, 0), (1, 0)) == pytest.approx(1.0)
    assert nc.point_segment_distance((0, 0), (0, 0), (0, 0)) == pytest.approx(0.0)


def test_footprint_gap_uses_region_points_and_corridor_radius():
    corridors = [((0.0, 0.0), (1.0, 0.0))]
    footprint = {"region_points": [{"offset": [0.0, 0.0], "radius": 0.05, "label": "sponge"}]}
    # disc at y=0.25, r=0.05 -> 0.20 from the centreline -> gap 0.10 vs 0.10 band
    gap = nc.footprint_corridor_gap((0.5, 0.25), footprint, corridors, 0.10)
    assert gap == pytest.approx(0.10)
    gap = nc.footprint_corridor_gap((0.5, 0.18), footprint, corridors, 0.10)
    assert gap == pytest.approx(0.03)
    assert nc.footprint_corridor_gap((0.5, 0.18), footprint, [], 0.10) == float("inf")


def test_footprint_gap_falls_back_to_aabb():
    footprint = {"min_offset": [-0.1, -0.05], "max_offset": [0.1, 0.05], "region_points": []}
    gap = nc.footprint_corridor_gap((0.5, 0.30), footprint, [((0.0, 0.0), (1.0, 0.0))], 0.10)
    assert gap == pytest.approx(0.25 - 0.10, abs=1e-6)


def test_choose_contested_requires_entering_the_band():
    # only min_gap <= gap <= max_gap qualifies (default [-0.04, 0.0]); deepest wins
    assert nc.DEFAULT_MAX_GAP == 0.0 and nc.DEFAULT_MIN_GAP == -0.04
    assert nc.choose_contested([0.3, 0.04, 0.01, -0.02]) == 3
    assert nc.choose_contested([0.3, -0.01, -0.03, -0.05]) == 2
    assert nc.choose_contested([0.0, 0.2]) == 0          # touching the band counts
    assert nc.choose_contested([0.3, 0.04, 0.001]) is None  # near but outside
    assert nc.choose_contested([0.3, -0.057]) is None     # deeper than min_gap
    assert nc.choose_contested([]) is None
    assert nc.choose_contested([-0.02, -0.02]) == 0       # ties keep order
    assert nc.choose_contested([0.3, -0.06], 0.0, -0.10) == 1


def test_failure_reason_codes():
    assert nc.contest_failure_reason([0.1], 0) == "no_corridor"
    assert nc.contest_failure_reason([], 3) == "no_valid_candidate"
    assert nc.contest_failure_reason([float("inf")], 3) == "no_valid_candidate"
    assert nc.contest_failure_reason([0.2, 0.05], 3) == "no_candidate_in_band"
    assert nc.contest_failure_reason([0.2, -0.08], 3) == "only_too_deep_candidates"


def test_corridors_are_pick_to_place_only():
    regions = [
        (np.array([0.0, -0.3]), 0.06, "breads:0"),
        (np.array([0.1, -0.1]), 0.105, "basket"),
        (np.array([0.0, -0.65]), 0.20, "robot_base:right_arm"),
        (np.array([0.0, -0.2]), 0.035, "gripper:right_arm:3:hand"),
    ]
    corridors = nc.corridors_from_regions(regions)
    pairs = {(tuple(np.round(a, 3)), tuple(np.round(b, 3))) for a, b in corridors}
    assert pairs == {((0.0, -0.3), (0.1, -0.1))}

    bowls = nc.corridors_from_regions(
        [(np.array([-0.08, -0.06]), 0.1, "bowls:0"), (np.array([0.14, -0.16]), 0.1, "bowls:1")]
    )
    assert len(bowls) == 1
    assert np.allclose(bowls[0][0], [0.14, -0.16]) and np.allclose(bowls[0][1], [-0.08, -0.06])

    cab = nc.corridors_from_regions(
        [(np.array([-0.1, -0.3]), 0.07, "cabinet_object"), (np.array([0.3, -0.65]), 0.14, "cabinet")],
        extra_places=[(0.65, -0.30)],
    )
    assert len(cab) == 2


def test_corridors_pair_each_pick_with_its_own_place():
    blocks = nc.corridors_from_regions([
        (np.array([0.0, 0.0]), 0.05, "block:red"),
        (np.array([0.1, 0.0]), 0.05, "block:blue"),
        (np.array([0.0, 0.3]), 0.06, "slot:red"),
        (np.array([0.1, 0.3]), 0.06, "slot:blue"),
    ])
    assert len(blocks) == 2
    for a, b in blocks:
        assert np.isclose(a[0], b[0])  # block K goes straight to slot K
    cat = nc.corridors_from_regions(
        [
            (np.array([0.0, 0.0]), 0.04, "categorize_obj:0"),
            (np.array([0.2, 0.0]), 0.04, "categorize_obj:1"),
            (np.array([-0.3, 0.3]), 0.1, "categorize_basket:0"),
            (np.array([0.3, 0.3]), 0.1, "categorize_basket:1"),
        ],
        place_map={"categorize_obj:0": ["categorize_basket:1"],
                   "categorize_obj:1": ["categorize_basket:0"]},
    )
    got = {(tuple(np.round(a, 2)), tuple(np.round(b, 2))) for a, b in cat}
    assert got == {((0.0, 0.0), (0.3, 0.3)), ((0.2, 0.0), (-0.3, 0.3))}


def _fake_task(seed, **config):
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = dict(config)
    task._neutral_avatar_reset_seed = seed
    task._neutral_avatar_job = {"kind": "sponge", "base_motion": "wipe_table_2"}
    task._neutral_contested_state = None
    task._neutral_robot_gripper_forbidden_regions = lambda: [
        (np.array([0.0, -0.40]), 0.035, "gripper:right_arm:tcp"),
    ]
    return task


def test_selector_picks_nearest_corridor_candidate_and_records_state():
    task = _fake_task(9_000_004, neutral_contested_force=True)
    assert task._neutral_contested_requested() is True
    regions = [
        (np.array([-0.08, -0.06]), 0.10, "bowls:0"),
        (np.array([0.14, -0.16]), 0.10, "bowls:1"),
    ]
    fp = {"region_points": [{"offset": [0.0, 0.0], "radius": 0.05, "label": "sponge"}]}
    # corridor bowls:1 (0.14,-0.16) -> bowls:0 (-0.08,-0.06)
    far = {"xy": np.array([0.45, 0.10]), "footprint": fp, "side": "right"}
    outside = {"xy": np.array([0.05, 0.10]), "footprint": fp, "side": "left"}   # gap > 0
    near = {"xy": np.array([0.03, 0.04]), "footprint": fp, "side": "back"}      # enters band
    chosen = task._neutral_contested_select([far, outside, near], regions)
    assert chosen is near
    assert outside["contest_gap"] > 0.0
    state = task._neutral_contested_state
    assert state["applied"] is True and state["failed"] is False
    assert state["reason"] == "applied"
    assert state["selected_xy"] == [0.03, 0.04]
    assert -0.04 <= state["gap_m"] <= 0.0
    assert near["contest_gap"] < far["contest_gap"]
    metrics = task.neutral_layout_metrics()
    assert metrics["neutral_contested"] is True
    assert metrics["neutral_layout"]["contested"]["selected_side"] == "back"


def test_selector_reports_failure_when_nothing_reaches_the_band():
    task = _fake_task(9_000_004, neutral_contested_force=True)
    task._neutral_contested_requested()
    fp = {"region_points": [{"offset": [0.0, 0.0], "radius": 0.05, "label": "sponge"}]}
    far = {"xy": np.array([0.55, 0.15]), "footprint": fp}
    regions = [
        (np.array([-0.08, -0.06]), 0.10, "bowls:0"),
        (np.array([-0.10, -0.16]), 0.10, "bowls:1"),
    ]
    assert task._neutral_contested_select([far], regions) is None
    state = task._neutral_contested_state
    assert state["applied"] is False and state["failed"] is True
    assert state["reason"] == "no_candidate_in_band"
    assert task.neutral_layout_metrics()["neutral_contested"] is False


def test_contested_decision_honours_fraction_and_force():
    assert _fake_task(9_000_004)._neutral_contested_requested() is True   # u=0.293
    assert _fake_task(9_000_000)._neutral_contested_requested() is False  # u=0.596
    assert _fake_task(9_000_000, neutral_contested_fraction=1.0)._neutral_contested_requested()
    assert not _fake_task(9_000_004, neutral_contested_fraction=0.0)._neutral_contested_requested()
    assert not _fake_task(9_000_004, neutral_contested_force=False)._neutral_contested_requested()
    with _global_rng_preserved():
        _fake_task(123)._neutral_contested_requested()


def test_only_table_reach_kinds_are_contested():
    assert nc.CONTESTABLE_KINDS == ("mouse", "sponge", "bottle")
    for kind in ("sitting_clap", "put_objects_in_bowl", "writing", "laptop", "typing", "phone"):
        task = _fake_task(9_000_004, neutral_contested_force=True)
        task._neutral_avatar_job = {"kind": kind}
        assert task._neutral_contested_requested() is False
        m = task.neutral_layout_metrics()
        assert m["neutral_contested"] is False
        assert m["neutral_contested_reason"] == f"kind_not_contestable:{kind}"
        assert m["neutral_contested_requested"] is True
    for kind in nc.CONTESTABLE_KINDS:
        task = _fake_task(9_000_004, neutral_contested_force=True)
        task._neutral_avatar_job = {"kind": kind}
        assert task._neutral_contested_requested() is True


def test_every_outcome_records_a_reason():
    not_drawn = _fake_task(9_000_000)
    not_drawn._neutral_contested_requested()
    assert not_drawn.neutral_layout_metrics()["neutral_contested_reason"] == "not_drawn"
    # requested, never reached a selector (e.g. legacy fallback path)
    silent = _fake_task(9_000_004, neutral_contested_force=True)
    silent._neutral_contested_requested()
    m = silent.neutral_layout_metrics()
    assert m["neutral_contested"] is False
    assert m["neutral_contested_reason"] == "selector_not_reached"
    assert m["neutral_layout"]["contested"]["failed"] is True
    # a recovery-candidate layout is reported, not silent
    rec = _fake_task(9_000_004, neutral_contested_force=True)
    rec._neutral_contested_requested()
    rec._neutral_contested_mark_unsupported("recovery_candidate_only")
    assert rec.neutral_layout_metrics()["neutral_contested_reason"] == "recovery_candidate_only"
    # metrics before any layout call still decide and report
    fresh = _fake_task(9_000_000)
    assert fresh.neutral_layout_metrics()["neutral_contested_reason"] == "not_drawn"


def test_corridors_use_presettle_spawn_regions():
    task = _fake_task(9_000_004, neutral_contested_force=True)
    fp = {"region_points": [{"offset": [0.0, 0.0], "radius": 0.05, "label": "sponge"}]}
    settled = [(np.array([-0.08, -0.06]), 0.10, "bowls:0"), (np.array([0.14, -0.16]), 0.10, "bowls:1")]
    # spawn poses put the corridor far from this candidate; settle drift moved it close
    task._neutral_presettle_regions = [(np.array([-0.08, -0.60]), 0.10, "bowls:0"),
                                       (np.array([0.14, -0.70]), 0.10, "bowls:1")]
    near = {"xy": np.array([0.03, 0.04]), "footprint": fp}
    assert task._neutral_contested_select([near], settled) is None
    assert task._neutral_contested_state["corridors"][0][0] == [0.14, -0.7]
    task.config["neutral_contested_presettle"] = False
    assert task._neutral_contested_select([near], settled) is near


def test_reselect_after_success_clears_stale_failure():
    task = _fake_task(9_000_004, neutral_contested_force=True)
    fp = {"region_points": [{"offset": [0.0, 0.0], "radius": 0.05, "label": "sponge"}]}
    regions = [(np.array([-0.08, -0.06]), 0.10, "bowls:0"), (np.array([0.14, -0.16]), 0.10, "bowls:1")]
    assert task._neutral_contested_select([{"xy": np.array([0.6, 0.3]), "footprint": fp}], regions) is None
    assert task._neutral_contested_state["failed"] is True
    near = {"xy": np.array([0.03, 0.04]), "footprint": fp}
    assert task._neutral_contested_select([near], regions) is near
    st = task._neutral_contested_state
    assert st["applied"] is True and st["failed"] is False and st["reason"] == "applied"


# -- timing + collision log --------------------------------------------------

def test_eval_trigger_defaults_start_within_five_policy_steps():
    assert NeutralAvatarTableWorkMixin.NEUTRAL_EVAL_TRIGGER_STEP_MIN == 0
    assert NeutralAvatarTableWorkMixin.NEUTRAL_EVAL_TRIGGER_STEP_MAX == 5


def _start_stub(log, collided):
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = {}
    task.avatar = SimpleNamespace(play_animation=lambda *_a, **_k: None)
    task._neutral_avatar_started = False
    task._neutral_avatar_job = {"motion": "m", "kind": "sponge", "events": []}
    task._neutral_avatar_motion_footprint = {"min_offset": np.zeros(2), "max_offset": np.zeros(2)}
    task.neutral_avatar_target = np.zeros(3)
    task.avatar_init_pos = np.zeros(3)
    task.avatar_collision_log = list(log)
    task.avatar_collided = collided

    @contextmanager
    def suppress():
        yield
    task.suppress_recording = suppress
    task._prepare_neutral_avatar_start_layout = lambda: False
    task._install_neutral_loop_subclip = lambda job: None
    task._patch_neutral_avatar_step_sim = lambda job, event_frame_offset=0: None
    task._neutral_avatar_apply_motion_frame = lambda *_a: None
    task._neutral_avatar_motion_slow = lambda: 1.0
    task._neutral_debug = lambda *_a: None
    return task


def test_start_keeps_task_phase_contacts_and_drops_pretask_records():
    log = [
        {"collided": True, "pretask": True, "min_distance": -0.01},
        {"collided": True, "pretask": False, "min_distance": -0.002},
        {"collided": False, "pretask": False, "min_distance": 0.08},
    ]
    task = _start_stub(log, collided=True)
    assert task._start_neutral_avatar_table_work(add_start_delay=False) is True
    assert [r["min_distance"] for r in task.avatar_collision_log] == [-0.002, 0.08]
    assert task.avatar_collided is True


def test_start_clears_only_pretask_collision_flag():
    log = [{"collided": True, "pretask": True}, {"collided": False, "pretask": False}]
    task = _start_stub(log, collided=True)
    task._start_neutral_avatar_table_work(add_start_delay=False)
    assert task.avatar_collided is False
    assert len(task.avatar_collision_log) == 1


# -- motion aliases ----------------------------------------------------------

def test_unforced_choices_resolve_aliases(monkeypatch):
    from envs.tasks.place_burger_and_fries_copresence import PlaceBurgerAndFriesCopresence

    task = PlaceBurgerAndFriesCopresence.__new__(PlaceBurgerAndFriesCopresence)
    task.config = {}
    task._neutral_avatar_job = None
    task._neutral_avatar_reset_seed = 0
    seen = []

    def fake_randint(n):
        seen.append(n)
        return 0
    monkeypatch.setattr(np.random, "randint", fake_randint)
    task._resolve_neutral_avatar_job()
    assert seen == [4]  # wipe_table_1 is no longer silently dropped
    assert task._neutral_avatar_job["motion"] == "wipe_table_2_reverse"


# -- bowl + skillet jitter ---------------------------------------------------

def test_easy_bowl_jitter_bounds_minsep_and_yaw_stream():
    from envs.task_bases import stack_bowls_three as sb

    for seed in range(200):
        rng = np.random.RandomState(seed)
        nominal = sb._sample_bowl_xy(rng, 2)
        yaws = [rng.uniform(-np.pi / 4, np.pi / 4) for _ in range(2)]
        jittered = sb._jitter_easy_bowl_xy(rng, nominal, 0.03)

        ref = np.random.RandomState(seed)
        assert sb._sample_bowl_xy(ref, 2) == nominal
        assert [ref.uniform(-np.pi / 4, np.pi / 4) for _ in range(2)] == yaws

        for (x0, y0), (x1, y1) in zip(nominal, jittered):
            assert abs(x1 - x0) <= 0.03 + 1e-12 and abs(y1 - y0) <= 0.03 + 1e-12
        (ax, ay), (bx, by) = jittered
        assert np.hypot(ax - bx, ay - by) >= sb._MIN_SEP_M
    assert nominal == [(-0.08, -0.06), (0.14, -0.16)]
    assert sb._jitter_easy_bowl_xy(np.random.RandomState(0), nominal, 0.0) == nominal
    assert sb.StackBowlsThree.EASY_BOWL_XY_JITTER_M == 0.03


def test_skillet_food_jitter_constants():
    from envs.task_bases.place_food_in_skillet import SkilletTaskCore
    from envs.tasks.place_food_in_skillet_contribution import PlaceFoodInSkilletContribution
    from envs.tasks.place_food_in_skillet_copresence import PlaceFoodInSkilletCopresence

    assert SkilletTaskCore.FOOD_SPAWN_JITTER_XY == 0.035
    assert PlaceFoodInSkilletCopresence.FOOD_SPAWN_JITTER_XY == 0.035
    assert PlaceFoodInSkilletContribution.FOOD_SPAWN_JITTER_XY == 0.030


# -- tipped human can is a scored failure, not an exception --------------------

def test_tipped_can_scores_failure_instead_of_raising():
    from envs.interaction.neutral_table_work import NeutralTableWorkController

    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = {}
    task.plan_success = True
    task.avatar_collided = False
    task._neutral_prop_disturbed = None
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl"}
    task._neutral_put_objects_asset = lambda: "071_can"
    task._neutral_can_prop_is_upright = lambda *_a: False
    task._neutral_debug = lambda *_a: None
    can = SimpleNamespace(get_pos=lambda: np.array([0.1, -0.4, 0.78]),
                          get_quat=lambda: np.array([0.7, 0.1, 0.7, 0.0]))
    assert NeutralTableWorkController.avatar_success_gate(task) is True
    task._neutral_assert_can_upright_before_pick("object_1", can)   # no raise
    assert task.plan_success is False
    assert task._neutral_prop_disturbed["object"] == "object_1"
    assert task._neutral_prop_disturbed["reason"] == "can_not_upright_before_avatar_pick"
    assert NeutralTableWorkController.avatar_success_gate(task) is False
