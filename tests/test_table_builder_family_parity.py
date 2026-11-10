"""Scene-free parity contracts for the shared rectangular-table builder."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.base_task import BaseTask
from envs.scene_init import _capture_task_attrs
from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB
from envs.task_bases.dump_bin import DumpBin
from envs.task_bases.place_bread_in_basket import PlaceBreadInBasket
from envs.task_bases.place_dual_shoes import PlaceDualShoes
from envs.task_bases.put_object_cabinet import PutObjectCabinet
from envs.task_bases.stack_bowls_three import StackBowlsThree
from envs.tasks.place_pair_of_shoes_contribution import PlacePairOfShoesContribution
from envs.tasks.put_object_in_cabinet_contribution import PutObjectInCabinetContribution
from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption


_TOP_COLOR = (0.8, 0.75, 0.65)
_LEG_COLOR = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class _TableSpec:
    name: str
    task_cls: type
    primitive_module: str
    half_size: tuple[float, float]
    center_xy: tuple[float, float]
    leg_xy: tuple[tuple[float, float], ...]
    table_thickness: float = 0.05
    leg_radius: float = 0.025


_DUMP_LEGS = (
    (-0.45, -0.75),
    (0.45, -0.75),
    (-0.45, 0.05),
    (0.45, 0.05),
)
_WIDE_LEGS = (
    (-0.55, -0.85),
    (0.55, -0.85),
    (-0.55, 0.15),
    (0.55, 0.15),
)
_BLOCK_LEGS = (
    (-0.50, -0.75),
    (0.50, -0.75),
    (-0.50, 0.15),
    (0.50, 0.15),
)


_FAMILY_SPECS = (
    _TableSpec(
        "dump_bin",
        DumpBin,
        "envs.task_bases.dump_bin",
        (0.50, 0.45),
        (0.0, -0.35),
        _DUMP_LEGS,
    ),
    _TableSpec(
        "place_bread_in_basket",
        PlaceBreadInBasket,
        "envs.task_bases.place_bread_in_basket",
        (0.60, 0.55),
        (0.0, -0.35),
        _WIDE_LEGS,
    ),
    _TableSpec(
        "blocks_ranking_rgb",
        BlocksRankingRGB,
        "envs.task_bases.blocks_ranking_rgb",
        (0.55, 0.50),
        (0.0, -0.30),
        _BLOCK_LEGS,
    ),
    _TableSpec(
        "stack_bowls_three",
        StackBowlsThree,
        "envs.task_bases.stack_bowls_three",
        (0.60, 0.55),
        (0.0, -0.35),
        _WIDE_LEGS,
    ),
    _TableSpec(
        "place_dual_shoes",
        PlaceDualShoes,
        "envs.task_bases.place_dual_shoes",
        (0.60, 0.55),
        (0.0, -0.35),
        _WIDE_LEGS,
    ),
    _TableSpec(
        "put_object_cabinet",
        PutObjectCabinet,
        "envs.task_bases.put_object_cabinet",
        (0.50, 0.45),
        (0.0, -0.35),
        _DUMP_LEGS,
    ),
)

_FIXED_NO_VARIANT_TABLE_SPECS = (
    _TableSpec(
        "put_object_in_cabinet_contribution",
        PutObjectInCabinetContribution,
        "envs.tasks.put_object_in_cabinet_contribution",
        (0.50, 0.45),
        (0.0, -0.35),
        _DUMP_LEGS,
    ),
    _TableSpec(
        "put_object_in_cabinet_interruption",
        PutObjectInCabinetInterruption,
        "envs.tasks.put_object_in_cabinet_interruption",
        (0.50, 0.45),
        (0.0, -0.35),
        _DUMP_LEGS,
    ),
    _TableSpec(
        "place_pair_of_shoes_contribution",
        PlacePairOfShoesContribution,
        "envs.task_bases.place_dual_shoes",
        (0.60, 0.55),
        (0.0, -0.35),
        _WIDE_LEGS,
    ),
)


@dataclass(frozen=True)
class _PrimitiveCall:
    kind: str
    pos: tuple[float, ...]
    size: dict
    kwargs: dict
    entity: object


def _bare_task(task_cls):
    task = task_cls.__new__(task_cls)
    task.scene = object()
    task.config = {}
    task.table_offset = np.array([0.0, 0.0], dtype=np.float64)
    return task


def _install_primitive_recorder(monkeypatch, *module_names):
    calls: list[_PrimitiveCall] = []

    def fake_create_primitive(scene, kind, pose, *, size, **kwargs):
        entity = object()
        calls.append(
            _PrimitiveCall(
                kind=str(kind),
                pos=tuple(float(v) for v in np.asarray(pose.p).ravel()),
                size=dict(size),
                kwargs=dict(kwargs),
                entity=entity,
            )
        )
        return entity

    # Patch all locations so the same test describes the legacy family-local
    # implementations and the migrated BaseTask implementation (whose table
    # builder lives in envs.base_task_parts.scene since the base_task split).
    for module_name in {"envs.base_task", "envs.base_task_parts.scene", *module_names}:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "create_primitive", fake_create_primitive, raising=False)
    return calls


def _assert_table_calls(
    calls,
    *,
    table_height,
    half_size,
    center_xy,
    table_thickness,
    leg_radius,
    leg_xy,
    top_color=_TOP_COLOR,
):
    assert len(calls) == 1 + len(leg_xy)

    top = calls[0]
    assert top.kind == "box"
    assert top.pos == pytest.approx((*center_xy, table_height))
    assert top.size["half_size"] == pytest.approx(
        (*half_size, table_thickness / 2.0)
    )
    assert top.kwargs == {"color": top_color, "is_static": True}

    leg_height = table_height - table_thickness / 2.0
    for call, expected_xy in zip(calls[1:], leg_xy, strict=True):
        assert call.kind == "cylinder"
        assert call.pos == pytest.approx((*expected_xy, leg_height / 2.0))
        assert call.size["radius"] == pytest.approx(leg_radius)
        assert call.size["half_length"] == pytest.approx(leg_height / 2.0)
        assert call.kwargs == {"color": _LEG_COLOR, "is_static": True}


def _assert_table_state(
    task,
    *,
    table_height,
    half_size,
    center_xy,
    table_thickness,
):
    assert task.TABLE_HEIGHT == pytest.approx(table_height)
    assert task.TABLE_THICKNESS == pytest.approx(table_thickness)
    assert task.TABLE_TOP_Z == pytest.approx(
        table_height + table_thickness / 2.0
    )
    assert task._table_top_z_fixed is False
    np.testing.assert_array_equal(task._table_center_xy, center_xy)
    np.testing.assert_array_equal(task._table_half_size_xy, half_size)
    assert task.table_random_object_bounds() == pytest.approx(
        (
            center_xy[0] - half_size[0],
            center_xy[0] + half_size[0],
            center_xy[1] - half_size[1],
            center_xy[1] + half_size[1],
        )
    )


def _assert_serialized_table_state(
    task,
    *,
    table_height,
    half_size,
    center_xy,
    table_thickness,
):
    attrs = _capture_task_attrs(task)
    assert attrs["_table_center_xy"] == pytest.approx(center_xy)
    assert attrs["_table_half_size_xy"] == pytest.approx(half_size)
    assert attrs["TABLE_HEIGHT"] == pytest.approx(table_height)
    assert attrs["TABLE_THICKNESS"] == pytest.approx(table_thickness)
    assert attrs["TABLE_TOP_Z"] == pytest.approx(
        table_height + table_thickness / 2.0
    )
    assert attrs["_table_top_z_fixed"] is False


def test_rectangular_helper_honors_explicit_thickness_and_ordered_leg_xy(
    monkeypatch,
):
    task = _bare_task(BaseTask)
    task._try_create_table_variant = lambda *args, **kwargs: False
    calls = _install_primitive_recorder(monkeypatch)
    leg_xy = (
        (0.31, -0.22),
        (-0.27, 0.18),
        (0.44, 0.12),
        (-0.36, -0.19),
    )

    result = task._create_rectangular_table(
        table_height=0.81,
        half_size=(0.42, 0.31),
        center_xy=(0.07, -0.09),
        table_thickness=0.08,
        color=(0.2, 0.3, 0.4),
        leg_radius=0.037,
        leg_xy=leg_xy,
    )

    assert result is calls[0].entity
    assert task.table is result
    _assert_table_calls(
        calls,
        table_height=0.81,
        half_size=(0.42, 0.31),
        center_xy=(0.07, -0.09),
        table_thickness=0.08,
        leg_radius=0.037,
        leg_xy=leg_xy,
        top_color=(0.2, 0.3, 0.4),
    )
    _assert_table_state(
        task,
        table_height=0.81,
        half_size=(0.42, 0.31),
        center_xy=(0.07, -0.09),
        table_thickness=0.08,
    )


def test_family_helper_path_records_table_bounds_exactly_once(monkeypatch):
    task = _bare_task(DumpBin)
    calls = _install_primitive_recorder(monkeypatch, "envs.task_bases.dump_bin")
    real_remember = task._remember_table_bounds
    remember_calls = []

    def remember_spy(*, half_size, center_xy):
        remember_calls.append((tuple(half_size), tuple(center_xy)))
        return real_remember(half_size=half_size, center_xy=center_xy)

    task._remember_table_bounds = remember_spy

    task._create_table(table_height=0.713)

    assert len(calls) == 5
    assert remember_calls == [((0.50, 0.45), (0.0, -0.35))]


@pytest.mark.parametrize("spec", _FAMILY_SPECS, ids=lambda spec: spec.name)
def test_family_table_geometry_state_bounds_and_serialization(monkeypatch, spec):
    task = _bare_task(spec.task_cls)
    calls = _install_primitive_recorder(monkeypatch, spec.primitive_module)
    table_height = 0.713

    result = task._create_table(table_height=table_height)

    # BaseTask.reset ignores this private hook's return value. Keep the family
    # wrappers' legacy implicit-None surface while the shared helper returns
    # the table entity for direct callers.
    assert result is None
    assert task.table is calls[0].entity
    _assert_table_calls(
        calls,
        table_height=table_height,
        half_size=spec.half_size,
        center_xy=spec.center_xy,
        table_thickness=spec.table_thickness,
        leg_radius=spec.leg_radius,
        leg_xy=spec.leg_xy,
    )
    _assert_table_state(
        task,
        table_height=table_height,
        half_size=spec.half_size,
        center_xy=spec.center_xy,
        table_thickness=spec.table_thickness,
    )
    _assert_serialized_table_state(
        task,
        table_height=table_height,
        half_size=spec.half_size,
        center_xy=spec.center_xy,
        table_thickness=spec.table_thickness,
    )


@pytest.mark.parametrize(
    "spec", _FIXED_NO_VARIANT_TABLE_SPECS, ids=lambda spec: spec.name
)
def test_fixed_family_tables_preserve_geometry_and_skip_variants(
    monkeypatch, spec
):
    # Fixed variants share their family core and disable table variants through
    # the core's policy profile.
    assert "_create_table" not in spec.task_cls.__dict__
    task = _bare_task(spec.task_cls)
    # BaseTask.__init__ establishes this before reset; the class-local copies
    # intentionally leave it untouched while building their fixed tables.
    task._table_top_z_fixed = False
    variant_calls = []

    def unexpected_variant(*args, **kwargs):
        variant_calls.append((args, kwargs))
        return False

    task._try_create_table_variant = unexpected_variant
    calls = _install_primitive_recorder(monkeypatch, spec.primitive_module)
    table_height = 0.713

    result = task._create_table(table_height=table_height)

    assert result is None
    assert variant_calls == []
    assert task.table is calls[0].entity
    _assert_table_calls(
        calls,
        table_height=table_height,
        half_size=spec.half_size,
        center_xy=spec.center_xy,
        table_thickness=spec.table_thickness,
        leg_radius=spec.leg_radius,
        leg_xy=spec.leg_xy,
    )
    _assert_table_state(
        task,
        table_height=table_height,
        half_size=spec.half_size,
        center_xy=spec.center_xy,
        table_thickness=spec.table_thickness,
    )


def test_blocks_real_layout_preserves_rotated_table_and_leg_order(monkeypatch):
    task = _bare_task(BlocksRankingRGB)
    task._collection_setting = "real"
    task._configure_collection_layout()
    calls = _install_primitive_recorder(
        monkeypatch, "envs.task_bases.blocks_ranking_rgb"
    )
    table_height = 0.727

    result = task._create_table(table_height=table_height)

    half_size = tuple(float(v) for v in task.TABLE_HALF_SIZE)
    center_xy = tuple(float(v) for v in task.TABLE_CENTER_XY)
    leg_xy = tuple(tuple(float(v) for v in xy) for xy in task.TABLE_LEG_XY)
    assert result is None
    assert half_size == pytest.approx((0.50, 0.55))
    assert center_xy == pytest.approx((-0.30, 0.0))
    np.testing.assert_allclose(
        leg_xy,
        (
            (-0.75, 0.50),
            (-0.75, -0.50),
            (0.15, 0.50),
            (0.15, -0.50),
        ),
    )
    _assert_table_calls(
        calls,
        table_height=table_height,
        half_size=half_size,
        center_xy=center_xy,
        table_thickness=task.TABLE_THICKNESS_VALUE,
        leg_radius=task.TABLE_LEG_RADIUS,
        leg_xy=leg_xy,
    )
    _assert_table_state(
        task,
        table_height=table_height,
        half_size=half_size,
        center_xy=center_xy,
        table_thickness=task.TABLE_THICKNESS_VALUE,
    )


@pytest.mark.parametrize("spec", _FAMILY_SPECS, ids=lambda spec: spec.name)
def test_family_table_variant_short_circuits_all_procedural_primitives(
    monkeypatch, spec
):
    task = _bare_task(spec.task_cls)
    calls = _install_primitive_recorder(monkeypatch, spec.primitive_module)
    table_entity = object()
    task.table = table_entity
    variant_calls = []

    def fake_variant(height, *, half_size, center_xy):
        # Reproduce BaseTask._try_create_table_variant's authoritative bounds
        # recording before replacing its scene-building side effect.
        task._remember_table_bounds(half_size=half_size, center_xy=center_xy)
        variant_calls.append(
            {
                "height": height,
                "half_size": half_size,
                "center_xy": center_xy,
                "TABLE_HEIGHT": task.TABLE_HEIGHT,
                "TABLE_THICKNESS": task.TABLE_THICKNESS,
                "TABLE_TOP_Z": task.TABLE_TOP_Z,
                "fixed": task._table_top_z_fixed,
                "bounds": task.table_random_object_bounds(),
            }
        )
        return True

    task._try_create_table_variant = fake_variant
    table_height = 0.719

    result = task._create_table(table_height=table_height)

    assert result is None
    assert task.table is table_entity
    assert calls == []
    assert variant_calls == [
        {
            "height": table_height,
            "half_size": spec.half_size,
            "center_xy": spec.center_xy,
            "TABLE_HEIGHT": table_height,
            "TABLE_THICKNESS": spec.table_thickness,
            "TABLE_TOP_Z": table_height + spec.table_thickness / 2.0,
            "fixed": False,
            "bounds": pytest.approx(
                (
                    spec.center_xy[0] - spec.half_size[0],
                    spec.center_xy[0] + spec.half_size[0],
                    spec.center_xy[1] - spec.half_size[1],
                    spec.center_xy[1] + spec.half_size[1],
                )
            ),
        }
    ]
