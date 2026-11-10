"""Declarative configuration and portable asset paths for neutral tasks.

This module is deliberately independent of task lifecycle and simulator code.
The historical names remain re-exported by ``neutral_avatar_table_work`` so
existing task modules, configs, and external imports keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_PATH = Path(__file__).resolve().parents[2]
NEUTRAL_MOTION_PKL = str(
    ROOT_PATH / "assets" / "neutral_motions" / "raw_blender_interaction_clips.pkl"
)
NEUTRAL_MOTION_META = str(
    ROOT_PATH / "assets" / "neutral_motions" / "raw_blender_interaction_clips.json"
)


def _resolve_repo_path(p) -> str:
    """Return an absolute config path without depending on the caller's CWD."""
    p = Path(str(p))
    return str(p if p.is_absolute() else ROOT_PATH / p)


# Historical private names are part of the compatibility surface: configs,
# downstream extensions, and characterization tests imported them from the
# original monolith before this bounded extraction.
_NEUTRAL_MOTION_PKL = NEUTRAL_MOTION_PKL
_NEUTRAL_MOTION_META = NEUTRAL_MOTION_META
resolve_repo_path = _resolve_repo_path


@dataclass(frozen=True)
class NeutralTaskSpec:
    """Declarative neutral-avatar placement configuration for one task.

    Runtime layout code still consumes the historical ``NEUTRAL_*`` class
    attributes. ``apply_to_class`` hydrates those attributes from one
    structured spec while keeping existing rollout behavior stable.
    """

    work_xy: tuple[float, float] | None = None
    mouse_work_xy: tuple[float, float] | None = None
    sponge_work_xy: tuple[float, float] | None = None
    table_xy_bounds: tuple[float, float, float, float] | None = None
    max_table_edge_gap: float | None = None
    min_table_edge_gap: float | None = None
    avatar_as_planner_obstacle: bool | None = None
    use_region_placement: bool | None = None
    blocked_avatar_sides: tuple[str, ...] | None = None
    work_candidates: tuple[tuple[float, float], ...] | None = None
    mouse_work_candidates: tuple[tuple[float, float], ...] | None = None
    sponge_work_candidates: tuple[tuple[float, float], ...] | None = None
    work_region: tuple[float, float, float, float] | None = None
    mouse_work_region: tuple[float, float, float, float] | None = None
    sponge_work_region: tuple[float, float, float, float] | None = None
    sponge_side_samples: int | None = None

    @staticmethod
    def _xy(value: tuple[float, float]) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _bounds(
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return tuple(float(component) for component in value)

    @staticmethod
    def _candidates(
        value: tuple[tuple[float, float], ...],
    ) -> list[np.ndarray]:
        return [np.asarray(xy, dtype=np.float64) for xy in value]

    def apply_to_class(self, cls: type) -> None:
        mapping = {
            "work_xy": ("NEUTRAL_WORK_XY", self._xy),
            "mouse_work_xy": ("NEUTRAL_MOUSE_WORK_XY", self._xy),
            "sponge_work_xy": ("NEUTRAL_SPONGE_WORK_XY", self._xy),
            "table_xy_bounds": ("NEUTRAL_TABLE_XY_BOUNDS", self._bounds),
            "max_table_edge_gap": (
                "NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP",
                float,
            ),
            "min_table_edge_gap": (
                "NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP",
                float,
            ),
            "avatar_as_planner_obstacle": (
                "NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE",
                bool,
            ),
            "use_region_placement": ("NEUTRAL_USE_REGION_PLACEMENT", bool),
            "blocked_avatar_sides": (
                "NEUTRAL_AVATAR_BLOCKED_SIDES",
                lambda value: tuple(str(side) for side in value),
            ),
            "work_candidates": ("NEUTRAL_WORK_CANDIDATES", self._candidates),
            "mouse_work_candidates": (
                "NEUTRAL_MOUSE_WORK_CANDIDATES",
                self._candidates,
            ),
            "sponge_work_candidates": (
                "NEUTRAL_SPONGE_WORK_CANDIDATES",
                self._candidates,
            ),
            "work_region": ("NEUTRAL_WORK_REGION", self._bounds),
            "mouse_work_region": ("NEUTRAL_MOUSE_WORK_REGION", self._bounds),
            "sponge_work_region": ("NEUTRAL_SPONGE_WORK_REGION", self._bounds),
            "sponge_side_samples": ("NEUTRAL_SPONGE_SIDE_SAMPLES", int),
        }
        for field_name, (attribute_name, converter) in mapping.items():
            value = getattr(self, field_name)
            if value is not None:
                setattr(cls, attribute_name, converter(value))


__all__ = (
    "NEUTRAL_MOTION_META",
    "NEUTRAL_MOTION_PKL",
    "NeutralTaskSpec",
    "ROOT_PATH",
    "resolve_repo_path",
)
