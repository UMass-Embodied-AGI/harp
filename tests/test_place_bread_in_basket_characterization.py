"""Scene-free characterization contracts for the bread/basket family.

These tests freeze allocation, choreography, and metrics at the public task
surface.  They deliberately avoid assertions about which mixin or controller
owns the behavior.
"""

from __future__ import annotations

from contextlib import contextmanager

from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask, TargetSpec
from envs.task_bases.place_bread_in_basket import (
    PlaceBreadInBasket,
    _MIN_SEP_M,
    _REGION_X,
    _REGION_Y,
    _SAMPLE_BATCH_ATTEMPTS,
    _SAMPLE_FALLBACK_MARGIN,
    _SAMPLE_POINT_ATTEMPTS,
)
from envs.tasks.place_bread_in_basket_contribution import PlaceBreadInBasketContribution
from envs.tasks.place_bread_in_basket_interruption import PlaceBreadInBasketInterruption


SEEDS = (0, 8, 42)


class _FakeEntity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=np.float64)
        self._quat = np.asarray(quat, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self):
        return self._quat.copy()


class _FakeActor:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.entity = _FakeEntity(pos, quat)


def _legacy_base_positions(task, rng):
    n = 1 + int(task._num_breads)
    for outer in range(_SAMPLE_BATCH_ATTEMPTS):
        positions = []
        valid = True
        for _ in range(n):
            placed = False
            for _ in range(_SAMPLE_POINT_ATTEMPTS):
                candidate = (
                    float(rng.uniform(*_REGION_X)),
                    float(rng.uniform(*_REGION_Y)),
                )
                if all(
                    (candidate[0] - previous[0]) ** 2
                    + (candidate[1] - previous[1]) ** 2
                    >= _MIN_SEP_M ** 2
                    for previous in positions
                ):
                    positions.append(candidate)
                    placed = True
                    break
            if not placed:
                valid = False
                break
        if valid:
            return positions, {"attempts": outer + 1}
    xs = np.linspace(
        _REGION_X[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_X[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    ys = np.linspace(
        _REGION_Y[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_Y[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    return [
        (float(xs[index]), float(ys[index])) for index in range(n)
    ], {"attempts": _SAMPLE_BATCH_ATTEMPTS, "fallback": True}


def _legacy_assist_positions(task, rng):
    """Legacy draw count/order (loaf index, then 2 draws per object per
    attempt) over the structured shared-pool layout, accepted only inside
    the robot reach band / parked-hand keep-out / per-loaf human reach."""
    from envs.tasks.place_bread_in_basket_contribution import (
        _ASSIST_FALLBACK_POSITIONS,
        _ASSIST_SAMPLE_ATTEMPTS,
        _HUMAN_REACH,
        _POOL_LAYOUT,
        _POOL_LOAF_Y_MAX,
        _POOL_MIN_SEP_M,
        _POOL_STANCE_Y,
        _ROBOT_BASE_XY,
        _ROBOT_HOME_HAND_XY,
        _ROBOT_HOME_KEEPOUT_M,
        _ROBOT_REACH_ANNULUS,
    )

    def reaches(pair):
        lo, hi = -np.inf, np.inf
        for k, (x, y) in enumerate(pair):
            z = _HUMAN_REACH["basket_z"] if k == 0 else _HUMAN_REACH["loaf_z"]
            r2 = (_HUMAN_REACH["radius"] ** 2 - (_POOL_STANCE_Y - y) ** 2
                  - (_HUMAN_REACH["shoulder_z"] - z) ** 2)
            if r2 < 0:
                return False
            lo, hi = max(lo, x - np.sqrt(r2)), min(hi, x + np.sqrt(r2))
        return lo <= hi

    n_breads = max(2, int(task._num_breads))
    avatar_idx = int(rng.randint(0, n_breads))
    robot_indices = [index for index in range(n_breads) if index != avatar_idx]
    for outer in range(_ASSIST_SAMPLE_ATTEMPTS):
        basket = (
            float(rng.uniform(*_POOL_LAYOUT["basket_x"])),
            float(rng.uniform(*_POOL_LAYOUT["basket_y"])),
        )
        breads = []
        for k in range(n_breads):
            offset = float(rng.uniform(*_POOL_LAYOUT["offset"]))
            y = float(rng.uniform(*_POOL_LAYOUT["loaf_y"]))
            breads.append((basket[0] + (-offset if k % 2 == 0 else offset), y))
        positions = [basket, *breads]
        ok = all(
            _ROBOT_REACH_ANNULUS[0]
            <= float(np.hypot(x - _ROBOT_BASE_XY[0], y - _ROBOT_BASE_XY[1]))
            <= _ROBOT_REACH_ANNULUS[1]
            and float(np.hypot(x - _ROBOT_HOME_HAND_XY[0], y - _ROBOT_HOME_HAND_XY[1]))
            >= _ROBOT_HOME_KEEPOUT_M
            for x, y in positions
        )
        ok = ok and all(reaches([basket, loaf]) for loaf in breads)
        ok = ok and all(y <= _POOL_LOAF_Y_MAX for _x, y in breads)
        ok = ok and all(
            (positions[i][0] - positions[j][0]) ** 2
            + (positions[i][1] - positions[j][1]) ** 2
            >= _POOL_MIN_SEP_M ** 2
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        if ok:
            return positions, {
                "avatar_idx": avatar_idx,
                "robot_indices": robot_indices,
                "attempts": outer + 1,
            }
    return list(_ASSIST_FALLBACK_POSITIONS[: 1 + n_breads]), {
        "avatar_idx": avatar_idx,
        "robot_indices": robot_indices,
        "attempts": _ASSIST_SAMPLE_ATTEMPTS,
        "fallback": True,
    }


def _legacy_interrupt_positions(task, rng):
    from envs.tasks.place_bread_in_basket_interruption import (
        _INTERRUPT_SAMPLE_ATTEMPTS,
    )

    n_breads = max(1, int(task._num_breads))
    if "interrupt_pick_ordinal" in task.config:
        ordinal = int(task.config["interrupt_pick_ordinal"]) % n_breads
    elif task.config.get("random_interrupt_target", True):
        ordinal = int(rng.randint(0, n_breads))
    else:
        ordinal = 0

    for outer in range(_INTERRUPT_SAMPLE_ATTEMPTS):
        basket = (
            float(rng.uniform(*_REGION_X)),
            float(rng.uniform(*_REGION_Y)),
        )
        breads = [
            (
                float(rng.uniform(task.INT_X_LO, task.INT_X_HI)),
                float(rng.uniform(task.INT_Y_LO, task.INT_Y_HI)),
            )
            for _ in range(n_breads)
        ]
        positions = [basket, *breads]
        if all(
            (positions[i][0] - positions[j][0]) ** 2
            + (positions[i][1] - positions[j][1]) ** 2
            >= _MIN_SEP_M ** 2
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        ):
            order = sorted(
                range(len(breads)),
                key=lambda index: float(np.linalg.norm(
                    np.asarray(breads[index]) - np.asarray(basket)
                )),
            )
            return positions, {
                "ordinal": ordinal,
                "interrupted_idx": int(order[ordinal]),
                "attempts": outer + 1,
            }

    basket = (
        float(_REGION_X[0] + _SAMPLE_FALLBACK_MARGIN),
        float(_REGION_Y[0] + _SAMPLE_FALLBACK_MARGIN),
    )
    breads = [(
        float(task.INT_X_LO + _SAMPLE_FALLBACK_MARGIN),
        float(task.INT_Y_LO + _SAMPLE_FALLBACK_MARGIN),
    )]
    if n_breads > 1:
        breads.append((
            float(task.INT_X_HI - _SAMPLE_FALLBACK_MARGIN),
            float(task.INT_Y_HI - _SAMPLE_FALLBACK_MARGIN),
        ))
    order = sorted(
        range(len(breads)),
        key=lambda index: float(np.linalg.norm(
            np.asarray(breads[index]) - np.asarray(basket)
        )),
    )
    return [basket, *breads], {
        "ordinal": ordinal,
        "interrupted_idx": int(order[ordinal]),
        "attempts": _INTERRUPT_SAMPLE_ATTEMPTS,
        "fallback": True,
    }


def _assert_seeded_layout(task, seed, legacy_sampler):
    np.random.seed(seed)
    expected_local_seed = int(np.random.randint(0, 2 ** 31 - 1))
    expected_rng = np.random.RandomState(expected_local_seed)
    expected_positions, expected_state = legacy_sampler(task, expected_rng)
    expected_local_tail = expected_rng.random_sample(8)
    expected_global_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual_local_seed = int(np.random.randint(0, 2 ** 31 - 1))
    actual_rng = np.random.RandomState(actual_local_seed)
    actual_positions = task._sample_positions(actual_rng)
    actual_local_tail = actual_rng.random_sample(8)
    actual_global_tail = np.random.random_sample(8)

    assert actual_local_seed == expected_local_seed
    np.testing.assert_array_equal(actual_positions, expected_positions)
    np.testing.assert_array_equal(actual_local_tail, expected_local_tail)
    np.testing.assert_array_equal(actual_global_tail, expected_global_tail)
    return expected_state


@pytest.mark.parametrize("seed", SEEDS)
def test_base_positions_preserve_local_layout_and_global_seed_stream(seed):
    task = PlaceBreadInBasket({"hard": True})
    state = _assert_seeded_layout(task, seed, _legacy_base_positions)
    assert state["attempts"] >= 1


@pytest.mark.parametrize("seed", SEEDS)
def test_assist_allocation_positions_and_rng_tails_match_legacy(seed):
    task = PlaceBreadInBasketContribution({})
    state = _assert_seeded_layout(task, seed, _legacy_assist_positions)

    assert task._avatar_bread_idx == state["avatar_idx"]
    assert task._robot_bread_indices == state["robot_indices"]
    assert task._robot_bread_idx == state["robot_indices"][0]


def test_hard_assist_preserves_three_bread_allocation_and_fallback_stream():
    task = PlaceBreadInBasketContribution({"hard": True})
    state = _assert_seeded_layout(task, 7, _legacy_assist_positions)

    assert state.get("fallback") is True
    assert task._avatar_bread_idx == state["avatar_idx"]
    assert task._robot_bread_indices == state["robot_indices"]
    assert len(task._robot_bread_indices) == 2


@pytest.mark.parametrize("seed", SEEDS)
def test_interrupt_ordinal_target_mapping_positions_and_rng_tails(seed):
    task = PlaceBreadInBasketInterruption({"hard": True})
    state = _assert_seeded_layout(task, seed, _legacy_interrupt_positions)

    assert task._interrupted_pick_ordinal == state["ordinal"]
    assert task._interrupted_idx == state["interrupted_idx"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    ((0, 0), (1, 1), (2, 0), (3, 1)),
)
def test_interrupt_pick_ordinal_config_remains_zero_based_modulo(
    configured, expected
):
    task = PlaceBreadInBasketInterruption({
        "hard": True,
        "interrupt_pick_ordinal": configured,
    })
    assert task._choose_interrupt_pick_ordinal(np.random.RandomState(5)) == expected


def test_interrupt_eval_trigger_draw_and_compatibility_state_preserve_rng_tail():
    # The random-step trigger is gone, but its draw is burned at
    # the same position with the same bounds, so the RNG tail is unchanged.
    # The legacy glide/pace compatibility fields are replaced by the shared
    # progress trigger (task._interrupt).
    task = PlaceBreadInBasketInterruption({
        "eval_mode": True,
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
    })
    task._dryrun_capture_palm = lambda: np.array([0.1, -0.2, 0.9])
    task.TABLE_TOP_Z = 0.75
    task._basket_rim_z = 0.82
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [_FakeActor([0.10, 0.3, 0.79])]
    task._bread_center_mesh = np.zeros(3)
    task._bread_scale = 1.0
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        get_ee_pose=lambda: [0.0, -0.3, 1.0, 1, 0, 0, 0], gripper_val=1.0,
    ))

    np.random.seed(42)
    expected_trigger = int(np.random.randint(7, 12))
    expected_tail = np.random.random_sample(8)

    np.random.seed(42)
    task._init_eval_interrupt_state()
    actual_tail = np.random.random_sample(8)

    assert task._eval_trigger_step == expected_trigger
    assert task._eval_interrupt_fired is False
    assert task._eval_target_bread is None
    assert task._policy_step_count == 0
    assert task._interrupt.trigger.armed is True
    assert task._interrupt.fired is False
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_base_scripted_expert_sorts_breads_and_returns_any_success():
    task = PlaceBreadInBasket({"hard": True})
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [
        _FakeActor([0.30, 0.0, 0.79]),
        _FakeActor([0.10, 0.0, 0.79]),
    ]
    task.INITIAL_SETTLE_STEPS = 2
    events = []

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim", None))
    task._refresh_basket_geom = lambda: events.append(("refresh", None))
    task._pick_and_place_bread = lambda bread, arm, label: (
        events.append(("pick", task.breads.index(bread), arm, label)),
        bread is task.breads[0],
    )[1]

    assert task.play_once() is True
    assert events == [
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        ("refresh", None),
        ("pick", 1, "right", "B"),
        ("pick", 0, "right", "A"),
    ]


class _FakePickPlaceMotion:
    def __init__(self):
        self.first = object()
        self.second = object()
        self.frames = [self.first, self.second]
        self.attach_frame = 17
        self.detach_frame = 31


class _FakeAssistAvatar:
    def __init__(self):
        self.calls = []
        self.motion = _FakePickPlaceMotion()
        self.motion_modules = {"pick_and_place": self.motion}

    def pick_and_place(self, **kwargs):
        self.calls.append(kwargs)


def test_assist_avatar_motion_parameters_delay_and_collision_reset():
    task = PlaceBreadInBasketContribution({})
    task.breads = [_FakeActor([0.10, -0.05, 0.79]), _FakeActor([-0.1, 0.1, 0.79])]
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task._avatar_bread_idx = 1
    task._bread_center_mesh = np.array([0.01, 0.02, 0.03])
    task._bread_scale = 0.5
    task._basket_rim_z = 0.82
    task._refresh_basket_geom = lambda: None
    task.TABLE_TOP_Z = 0.75
    task.avatar = _FakeAssistAvatar()
    task.avatar_collided = True
    task.avatar_collision_log = ["old"]

    original_attach = task.avatar.motion.attach_frame
    original_detach = task.avatar.motion.detach_frame
    task._start_avatar_assist(add_start_delay=True)

    assert len(task.avatar.calls) == 1
    call = task.avatar.calls[0]
    np.testing.assert_allclose(call["pick_pos"], [-0.095, 0.11, 0.805])
    np.testing.assert_array_equal(
        call["place_pos"], [0.0, 0.0, task._basket_rim_z + task.AVATAR_DROP_ABOVE_RIM]
    )
    assert call["attach_obj"] is task.breads[1].entity
    assert call["hand_id"] is None
    assert call["approach_frames"] == task.AVATAR_APPROACH_FRAMES
    assert call["transport_frames"] == task.AVATAR_TRANSPORT_FRAMES
    assert call["retract_frames"] == task.AVATAR_RETRACT_FRAMES
    assert call["natural"] is True
    assert call["body_margin"] == task.AVATAR_NATURAL_BODY_MARGIN
    assert call["yaw_limit_deg"] == task.AVATAR_NATURAL_YAW_LIMIT_DEG
    assert call["body_y_bounds"] == task.AVATAR_NATURAL_BODY_Y_BOUNDS
    assert call["settle_steps"] == task.AVATAR_NATURAL_SETTLE_STEPS
    # Shared pool: the human starts immediately (no extra start delay).
    delay = task.AVATAR_START_DELAY_STEPS
    assert delay == 0
    assert task.avatar.motion.frames[: delay + 1] == [task.avatar.motion.first] * (
        delay + 1
    )
    assert task.avatar.motion.frames[-1] is task.avatar.motion.second
    assert task.avatar.motion.attach_frame == original_attach + delay
    assert task.avatar.motion.detach_frame == original_detach + delay
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []


@contextmanager
def _null_context():
    yield


class _SpareSequenceAvatar:
    def __init__(self, sequence):
        self._sequence = iter(sequence)

    def spare(self):
        return next(self._sequence, True)


def test_assist_scripted_expert_routes_only_robot_allocation_in_distance_order():
    task = PlaceBreadInBasketContribution({"hard": True})
    task.INITIAL_SETTLE_STEPS = 2
    task.AVATAR_FINISH_SETTLE_STEPS = 2
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [
        _FakeActor([0.30, 0.0, 0.79]),
        _FakeActor([-0.30, 0.0, 0.79]),
        _FakeActor([0.10, 0.0, 0.79]),
    ]
    task._avatar_bread_idx = 1
    task._robot_bread_idx = 0
    task._robot_bread_indices = [0, 2]
    task.avatar = _SpareSequenceAvatar([False, False, True])
    task._avatar_in_planner = lambda arm: _null_context()
    events = []

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim", None))
    task._start_avatar_assist = lambda add_start_delay=True: events.append(
        ("avatar", add_start_delay)
    )
    task._pick_and_place_bread = lambda bread, arm, label: (
        events.append(("pick", task.breads.index(bread), arm, label)),
        bread is task.breads[2],
    )[1]

    assert task.play_once() is False
    assert [event for event in events if event[0] == "pick"] == [
        ("pick", 2, "right", "C"),
        ("pick", 0, "right", "A"),
    ]
    labels = [event[0] for event in events]
    assert labels.index("avatar") < labels.index("pick")
    assert ("avatar", False) in events  # no extra avatar start delay
    assert labels.count("sim") == 6  # 2 initial + 2 bounded wait + 2 final settle


class _FakeArm:
    tcp_offset = object()


class _FakeRobot:
    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return _FakeArm()


class _FakeLinkPose:
    def __init__(self, value):
        self.value = value

    def to_pose7(self):
        return ("pose7", self.value)


class _AlwaysSpareAvatar:
    motion_modules = {}

    def spare(self):
        return True


def test_interrupt_scripted_expert_preserves_phase_and_pick_order(monkeypatch):
    from envs.tasks import place_bread_in_basket_interruption as interrupt_module

    task = PlaceBreadInBasketInterruption({"hard": True})
    task.INITIAL_SETTLE_STEPS = 2
    task.PRE_ATTACH_WAIT_MARGIN_STEPS = 1
    task.POST_RELEASE_SETTLE_STEPS = 1
    task.TABLE_TOP_Z = 0.75
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [
        _FakeActor([0.10, 0.0, 0.79]),
        _FakeActor([0.30, 0.0, 0.79]),
    ]
    task._interrupted_pick_ordinal = 0
    task.robot = _FakeRobot()
    task.avatar = _AlwaysSpareAvatar()
    events = []

    monkeypatch.setattr(
        interrupt_module,
        "tcp_to_link_pose",
        lambda pose, _offset: _FakeLinkPose(pose),
    )

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim", None))
    task._refresh_basket_geom = lambda: events.append(("refresh", None))
    task.open_gripper = lambda arm: events.append(("open", arm))
    task.move_and_execute = lambda pose, arm: events.append(("intent", pose, arm))
    task._play_inspect_motion = lambda **kwargs: events.append(("inspect", kwargs))
    task._retreat_safe = lambda arm: events.append(("retreat", arm))
    task._wait_until_avatar_attach = lambda: events.append(("wait_attach", None))
    task._pick_and_place_bread = lambda bread, arm, label: (
        events.append(("pick", task.breads.index(bread), arm, label)), True
    )[1]

    assert task.play_once() is True
    assert task._interrupted_idx == 0
    assert [event for event in events if event[0] == "pick"] == [
        ("pick", 1, "right", "B"),
        ("pick", 0, "right", "A"),
    ]
    labels = [event[0] for event in events]
    assert labels == [
        "suppress", "sim", "sim", "suppress", "refresh", "open", "intent",
        "inspect", "retreat", "wait_attach", "sim", "pick", "sim", "pick",
    ]
    inspect = next(event[1] for event in events if event[0] == "inspect")
    assert inspect["attach_obj"] is task.breads[0].entity
    assert inspect["hand_id"] == task.AVATAR_HAND_ID
    assert inspect["attach_frame"] == task._attach_frame_scaled
    assert inspect["detach_frame"] == task._detach_frame_scaled
    assert inspect["return_to_idle_after"] is None


def _configure_bread_scoring(task, positions):
    task.TABLE_TOP_Z = 0.75
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [_FakeActor(pos) for pos in positions]
    task.plan_success = True


def test_base_success_requires_plan_and_every_bread_inside_above_table():
    task = PlaceBreadInBasket({"hard": True})
    _configure_bread_scoring(task, [[0.08, 0.0, 0.74], [-0.10, 0.0, 0.76]])
    assert task.check_success() is True

    task.plan_success = False
    assert task.check_success() is False
    task.plan_success = True
    task.breads[1].entity.set_pos([-0.1001, 0.0, 0.76])
    assert task.check_success() is False
    task.breads[1].entity.set_pos([0.0, 0.0, 0.729])
    assert task.check_success() is False


def test_base_evaluate_exposes_standard_target_success_and_safety_surface():
    task = PlaceBreadInBasket({"hard": True})
    _configure_bread_scoring(task, [[0.05, 0.0, 0.76], [-0.05, 0.0, 0.76]])
    task.target = TargetSpec(
        object=task.breads[0].entity,
        position=lambda: task.basket.entity.get_pos(),
        label="basket",
    )

    metrics = task.evaluate()

    assert {
        "plan_success",
        "success",
        "target_label",
        "target_object_pos",
        "target_pos",
        "target_dist_xy",
        "target_dist_3d",
        "target_dz",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
    } <= set(metrics)
    assert metrics["plan_success"] is True
    assert metrics["success"] is True
    assert metrics["target_label"] == "basket"
    assert metrics["target_object_pos"] == [0.05, 0.0, 0.76]
    assert metrics["target_pos"] == [0.0, 0.0, 0.75]
    assert metrics["target_dist_xy"] == pytest.approx(0.05)


def test_assist_success_requires_no_avatar_collision_by_default():
    task = PlaceBreadInBasketContribution({})
    _configure_bread_scoring(task, [[0.05, 0.0, 0.76], [-0.05, 0.0, 0.76]])
    task.avatar_collision_checker = object()
    task.avatar_collided = True
    assert task.config["success_require_no_avatar_collision"] is True
    assert task.check_success() is False

    task.config["success_require_no_avatar_collision"] = False
    assert task.check_success() is True


def test_assist_evaluate_preserves_per_bread_and_allocation_metrics(monkeypatch):
    monkeypatch.setattr(
        BaseTask,
        "evaluate",
        lambda _self: {"base": 1, "success": False},
    )
    task = PlaceBreadInBasketContribution({"eval_mode": True})
    _configure_bread_scoring(task, [[0.05, 0.0, 0.76], [-0.05, 0.0, 0.76]])
    task._avatar_bread_idx = 1
    task._robot_bread_idx = 0
    task._robot_bread_indices = [0]
    task._eval_assist_trigger_step = 0
    task._eval_assist_fired = True
    task._eval_assist_failed = False
    task._eval_policy_step_count = 7

    metrics = task.evaluate()
    assert metrics["base"] == 1
    assert metrics["success"] is True
    assert metrics["bread_0_pos"] == [0.05, 0.0, 0.76]
    assert metrics["bread_0_dist_xy"] == pytest.approx(0.05)
    assert metrics["bread_0_above_table"] is True
    assert metrics["bread_0_in_basket"] is True
    assert metrics["bread_1_pos"] == [-0.05, 0.0, 0.76]
    assert metrics["all_breads_in_basket"] is True
    assert metrics["assist_avatar_bread_idx"] == 1
    assert metrics["assist_robot_bread_idx"] == 0
    assert metrics["assist_robot_bread_indices"] == [0]
    assert metrics["eval_mode"] is True
    assert metrics["eval_assist_trigger_step"] == 0
    assert metrics["eval_assist_fired"] is True
    assert metrics["eval_assist_failed"] is False
    assert metrics["eval_policy_step_count"] == 7


def test_interrupt_success_adds_avatar_collision_gate():
    task = PlaceBreadInBasketInterruption({"hard": True})
    _configure_bread_scoring(task, [[0.05, 0.0, 0.76], [-0.05, 0.0, 0.76]])
    assert task.config["track_avatar_collision"] is True
    task.avatar_collided = False
    assert task.check_success() is True
    task.avatar_collided = True
    assert task.check_success() is False


def test_interrupt_evaluate_preserves_pace_metric_surface(monkeypatch):
    # Evaluate() reports the uniform ``interrupt`` block and a
    # ``pace_interrupt`` block with the legacy key names filled from the
    # progress trigger (trigger_mode is always "progress", distances 3-D).
    from envs.interaction.interrupt_progress import new_interrupt_episode

    monkeypatch.setattr(
        BaseTask,
        "evaluate",
        lambda _self: {"success": True},
    )
    task = PlaceBreadInBasketInterruption({"eval_mode": True})
    episode = new_interrupt_episode(task)
    target = np.array([0.1, 0.0, 0.8])
    episode.trigger.arm(np.array([0.1, -0.4, 1.1]), {1: target})
    assert episode.trigger.update(25, target + [0, 0, 0.12], {1: target}) == 1
    task._policy_step_count = 12

    metrics = task.evaluate()
    assert metrics["success"] is True
    assert metrics["interrupt"]["fired"] is True
    assert metrics["interrupt"]["trigger_step"] == 25
    block = metrics["pace_interrupt"]
    assert block["trigger_mode"] == "progress"
    assert block["fired"] is True
    assert block["trigger_step"] == 25
    assert block["trigger_reason"] == "progress"
    assert block["target_idx"] == 1
    assert block["trigger_dist_m"] == pytest.approx(0.12)
    assert block["policy_steps"] == 12




class _HeldArm:
    def __init__(self, tcp, gripper):
        self.tcp = np.asarray(tcp, dtype=float)
        self.gripper_val = gripper

    def get_ee_pose(self):
        return list(self.tcp) + [1.0, 0.0, 0.0, 0.0]


def _bread_task_classes():
    from envs.tasks.place_bread_in_basket_copresence import PlaceBreadInBasketCopresence

    return [PlaceBreadInBasket, PlaceBreadInBasketContribution,
            PlaceBreadInBasketInterruption, PlaceBreadInBasketCopresence]


@pytest.mark.parametrize("cls_idx", range(4))
def test_bread_held_or_floating_over_the_basket_is_not_placed(cls_idx):
    # A loaf the human holds over the basket, a loaf in
    # the closed gripper, or a loaf hanging above the rim must not score success.
    task_cls = _bread_task_classes()[cls_idx]
    task = task_cls({})
    positions = [[0.05, 0.0, 0.76], [-0.05, 0.0, 0.76]]
    _configure_bread_scoring(task, positions)
    task._basket_rim_z = 0.814
    task.avatar_collided = False
    task.avatar_collision_checker = None
    rig = SimpleNamespace(attached_object=[None, None], two_hand_attached_object=None)
    task.avatar = SimpleNamespace(robot=rig)
    held = task.breads[1]
    assert task.check_success() is True

    # Human holds the loaf over the basket (one- or two-handed).
    rig.attached_object[0] = (held.entity, np.zeros(3), np.eye(3))
    assert task.check_success() is False
    rig.attached_object[0] = None
    rig.two_hand_attached_object = (held.entity, np.zeros(3))
    assert task.check_success() is False
    rig.two_hand_attached_object = None
    assert task.check_success() is True

    # Loaf floating 10 cm above the rim; a loaf stacked just above the rim
    # (on the other loaf) still counts.
    held.entity.set_pos([-0.05, 0.0, 0.814 + 0.10])
    assert task.check_success() is False
    held.entity.set_pos([-0.05, 0.0, 0.814 + 0.03])
    assert task.check_success() is True
    held.entity.set_pos(positions[1])

    # Loaf still in the closed gripper is not placed; an open gripper is fine.
    arm = _HeldArm(positions[1], gripper=0.3)
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    assert task.check_success() is False
    arm.gripper_val = 1.0
    assert task.check_success() is True
    # Kinematically attached loaf is held regardless of the gripper value.
    task._gripper_attached = {"right": (held.entity, None)}
    assert task.check_success() is False
