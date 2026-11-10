"""Scene-free parity tests for shared avatar-capsule planner sampling."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.avatar.collision_checker import sample_capsule_planner_points
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.tasks.dump_bin_interruption import DumpBinInterruption
from envs.tasks.factory_inspect_pack import FactoryInspectPack
from envs.tasks.factory_kit_packing import FactoryKitPacking
from envs.tasks.factory_line_feeding import FactoryLineFeeding


@dataclass(frozen=True)
class _WrapperSpec:
    name: str
    task_cls: type
    module_name: str
    inflate_factor: float
    signature: str
    annotations: dict
    doc: str | None = None


_LINE_DOC = (
    "Sampled point cloud over the avatar's current collision capsules\n"
    "        (inflated), for the mplib planner obstacle set."
)

_WRAPPERS = (
    _WrapperSpec(
        "neutral",
        NeutralAvatarTableWorkMixin,
        "envs.task_bases.neutral_avatar_table_work",
        1.6,
        "(self) -> 'np.ndarray | None'",
        {"return": "np.ndarray | None"},
    ),
    _WrapperSpec(
        "dump_interrupt",
        DumpBinInterruption,
        "envs.tasks.dump_bin_interruption",
        1.8,
        "(self) -> numpy.ndarray | None",
        {"return": np.ndarray | None},
    ),
    _WrapperSpec(
        "factory_kit",
        FactoryKitPacking,
        "envs.tasks.factory_kit_packing",
        1.25,
        "(self) -> 'np.ndarray | None'",
        {"return": "np.ndarray | None"},
    ),
    _WrapperSpec(
        "factory_inspect",
        FactoryInspectPack,
        "envs.tasks.factory_inspect_pack",
        1.8,
        "(self)",
        {},
    ),
    _WrapperSpec(
        "factory_line",
        FactoryLineFeeding,
        "envs.tasks.factory_line_feeding",
        1.5,
        "(self)",
        {},
        _LINE_DOC,
    ),
)


def _expected_shell(centers, directions, radius):
    rows = []
    for center in np.asarray(centers, dtype=np.float64):
        rows.append(center)
        rows.extend(center + radius * directions)
    return np.asarray(rows, dtype=np.float64)


_SQRT_HALF = np.sqrt(0.5)
_HORIZONTAL_DIRECTIONS = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, -_SQRT_HALF, -_SQRT_HALF],
        [0.0, 0.0, -1.0],
        [0.0, _SQRT_HALF, -_SQRT_HALF],
        [0.0, 1.0, 0.0],
        [0.0, _SQRT_HALF, _SQRT_HALF],
        [0.0, 0.0, 1.0],
        [0.0, -_SQRT_HALF, _SQRT_HALF],
    ],
    dtype=np.float64,
)
_VERTICAL_DIRECTIONS = np.array(
    [
        [0.0, 1.0, 0.0],
        [-_SQRT_HALF, _SQRT_HALF, 0.0],
        [-1.0, 0.0, 0.0],
        [-_SQRT_HALF, -_SQRT_HALF, 0.0],
        [0.0, -1.0, 0.0],
        [_SQRT_HALF, -_SQRT_HALF, 0.0],
        [1.0, 0.0, 0.0],
        [_SQRT_HALF, _SQRT_HALF, 0.0],
    ],
    dtype=np.float64,
)


def test_helper_public_signature_and_wrapper_compatibility_surfaces():
    assert sample_capsule_planner_points.__module__ == "envs.avatar.collision_checker"
    assert str(inspect.signature(sample_capsule_planner_points)) == (
        "(capsules, *, inflate_factor: float) -> numpy.ndarray | None"
    )
    assert sample_capsule_planner_points.__annotations__ == {
        "inflate_factor": float,
        "return": np.ndarray | None,
    }
    assert sample_capsule_planner_points.__doc__ == (
        "Sample ordered planner points from world-space avatar capsules."
    )

    for spec in _WRAPPERS:
        method = spec.task_cls.__dict__["_avatar_obstacle_points"]
        assert str(inspect.signature(method)) == spec.signature
        assert method.__annotations__ == spec.annotations
        assert method.__doc__ == spec.doc


@pytest.mark.parametrize(
    ("name", "pa", "pb", "directions"),
    (
        (
            "abs-axis-z-below-0.9",
            np.array([0.10, -0.20, 0.30]),
            np.array([0.14, -0.20, 0.30]),
            _HORIZONTAL_DIRECTIONS,
        ),
        (
            "abs-axis-z-at-least-0.9",
            np.array([0.10, -0.20, 0.30]),
            np.array([0.10, -0.20, 0.34]),
            _VERTICAL_DIRECTIONS,
        ),
    ),
    ids=("abs-axis-z-below-0.9", "abs-axis-z-at-least-0.9"),
)
def test_sampler_freezes_point_order_geometry_and_both_basis_branches(
    name, pa, pb, directions
):
    del name
    marker = np.array([9.0, 8.0, 7.0])
    capsules = (
        ("shell", pa, pb, 0.20),
        ("degenerate-after-shell", marker, marker + 0.5e-6, 99.0),
    )
    before = [(a.copy(), b.copy()) for _n, a, b, _r in capsules]

    actual = sample_capsule_planner_points(capsules, inflate_factor=1.5)
    expected = np.vstack(
        [
            _expected_shell((pa, pb), directions, radius=0.30),
            marker,
        ]
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    for (_name, a, b, _radius), (old_a, old_b) in zip(
        capsules, before, strict=True
    ):
        np.testing.assert_array_equal(a, old_a)
        np.testing.assert_array_equal(b, old_b)


@pytest.mark.parametrize(
    ("length", "n_axis"),
    (
        (0.04 - 1e-12, 2),
        (0.04, 2),
        (0.04 + 1e-12, 2),
        (0.08 - 1e-12, 2),
        (0.08, 3),
        (0.08 + 1e-12, 3),
    ),
)
def test_sampler_freezes_axis_count_thresholds_near_004_and_008(length, n_axis):
    pa = np.zeros(3, dtype=np.float64)
    pb = np.array([length, 0.0, 0.0], dtype=np.float64)

    points = sample_capsule_planner_points(
        (("threshold", pa, pb, 0.0),), inflate_factor=2.0
    )

    assert points.shape == (9 * n_axis, 3)
    np.testing.assert_allclose(
        points[::9],
        pa[None, :] + np.linspace(0.0, 1.0, n_axis)[:, None] * (pb - pa),
        rtol=0.0,
        atol=0.0,
    )


def test_sampler_degenerate_boundary_empty_float64_and_no_input_mutation():
    assert sample_capsule_planner_points([], inflate_factor=1.0) is None
    assert sample_capsule_planner_points(iter(()), inflate_factor=1.0) is None

    pa = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    pa_before = pa.copy()
    below = np.array([0.5e-6, 2.0, 3.0], dtype=np.float64)
    at_boundary = np.array([1e-6, 2.0, 3.0], dtype=np.float64)

    degenerate = sample_capsule_planner_points(
        (("below", pa, below, np.float32(0.2)),), inflate_factor=np.float32(1.5)
    )
    nondegenerate = sample_capsule_planner_points(
        (("boundary", pa.astype(np.float64), at_boundary, 0.2),),
        inflate_factor=1.5,
    )

    assert degenerate.dtype == np.float64
    assert degenerate.shape == (1, 3)
    np.testing.assert_array_equal(degenerate[0], pa.astype(np.float64))
    assert nondegenerate.shape == (18, 3)
    np.testing.assert_array_equal(pa, pa_before)
    assert pa.dtype == np.float32


def test_sampler_propagates_capsule_source_errors():
    class CapsuleSourceError(RuntimeError):
        pass

    def broken_source():
        yield ("first", np.zeros(3), np.zeros(3), 0.1)
        raise CapsuleSourceError("capsule source failed")

    with pytest.raises(CapsuleSourceError, match="capsule source failed"):
        sample_capsule_planner_points(broken_source(), inflate_factor=1.0)


class _Collider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def current_capsules(self):
        self.calls += 1
        return self.payload


@pytest.mark.parametrize("spec", _WRAPPERS, ids=lambda spec: spec.name)
def test_thin_wrappers_forward_once_with_exact_task_inflate_factor(
    monkeypatch, spec
):
    task = spec.task_cls.__new__(spec.task_cls)
    module = importlib.import_module(spec.module_name)
    sampler_calls = []
    sampler_result = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)

    def fake_sampler(capsules, *, inflate_factor):
        sampler_calls.append((capsules, inflate_factor))
        return sampler_result

    monkeypatch.setattr(module, "sample_capsule_planner_points", fake_sampler)
    task.avatar_collider = None
    assert task._avatar_obstacle_points() is None
    assert sampler_calls == []

    payload = object()
    collider = _Collider(payload)
    task.avatar_collider = collider
    result = task._avatar_obstacle_points()

    assert result is sampler_result
    assert collider.calls == 1
    assert len(sampler_calls) == 1
    assert sampler_calls[0][0] is payload
    assert sampler_calls[0][1] == pytest.approx(spec.inflate_factor)
    assert task._AVATAR_INFLATE_FACTOR == pytest.approx(spec.inflate_factor)


@pytest.mark.parametrize("spec", _WRAPPERS, ids=lambda spec: spec.name)
def test_thin_wrappers_call_current_capsules_once_and_preserve_empty_none(spec):
    task = spec.task_cls.__new__(spec.task_cls)
    collider = _Collider([])
    task.avatar_collider = collider

    assert task._avatar_obstacle_points() is None
    assert collider.calls == 1


@pytest.mark.parametrize("spec", _WRAPPERS, ids=lambda spec: spec.name)
def test_thin_wrappers_propagate_collider_and_sampler_errors(monkeypatch, spec):
    class WrapperSourceError(RuntimeError):
        pass

    class BrokenCollider:
        calls = 0

        def current_capsules(self):
            self.calls += 1
            raise WrapperSourceError("current_capsules failed")

    task = spec.task_cls.__new__(spec.task_cls)
    broken = BrokenCollider()
    task.avatar_collider = broken
    with pytest.raises(WrapperSourceError, match="current_capsules failed"):
        task._avatar_obstacle_points()
    assert broken.calls == 1

    collider = _Collider(object())
    task.avatar_collider = collider
    module = importlib.import_module(spec.module_name)

    def broken_sampler(capsules, *, inflate_factor):
        raise WrapperSourceError("sampler failed")

    monkeypatch.setattr(module, "sample_capsule_planner_points", broken_sampler)
    with pytest.raises(WrapperSourceError, match="sampler failed"):
        task._avatar_obstacle_points()
    assert collider.calls == 1
