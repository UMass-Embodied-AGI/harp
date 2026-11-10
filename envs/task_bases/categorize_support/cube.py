"""Compatibility names for the first cube-only selector candidate."""
from .radius import radius_candidates as cube_candidates
from .radius import pick_and_place_radius as pick_and_place_cube

__all__ = ["cube_candidates", "pick_and_place_cube"]
