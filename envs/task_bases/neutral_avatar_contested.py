"""Contested Co-presence layout for neutral-avatar tasks.

The ordinary neutral layout keeps the human's table work out of the robot's
way: it takes the first (or the farthest-from-robot) candidate anchor that
clears every task region.  On a configurable fraction of seeds this module
changes only *which already-accepted candidate is chosen*: one whose hand /
prop sweep enters the band around the robot's pick->place corridor.  Clips,
props, candidate generators and every hard keep-out (task-object disks,
targets, slots, robot base, table bounds) are unchanged, so contested layouts
stay inside the training image distribution.

Rules (without them, many "contested" seeds land 0.2-0.8 m from the robot
or park a forearm on the robot's home gripper):

* corridors are only the episode's pick -> place segments (each pick to its
  own place when the task pairs them: block K -> slot K, categorize object
  -> its basket); the robot's home TCP is not part of any corridor;
* only table-reach kinds (``mouse``, ``sponge``, ``bottle``) are contested;
  every other kind falls back to the normal layout
  (``kind_not_contestable:<kind>``);
* the hand sweep must ENTER the band: ``min_gap <= gap <= max_gap`` with
  defaults ``[-0.04, 0.0]``; the deepest such candidate is chosen;
* corridors use the task-object regions at their pre-settle spawn poses
  (seeded, backend independent), so the decision does not flip between
  CPU, GPU and NYX near the band edge;
* every episode records a reason (``not_drawn``, ``applied``,
  ``kind_not_contestable:*``, ``no_corridor``, ``no_valid_candidate``,
  ``no_candidate_in_band``, ``only_too_deep_candidates``,
  ``recovery_candidate_only``, ``selector_not_reached``); a seed without a
  qualifying candidate keeps the non-contested layout.

Determinism contract
--------------------
* The contested decision is drawn from a private generator seeded by the
  episode seed (``default_rng(seed + CONTESTED_RNG_SALT)``), never from the
  global ``np.random`` stream, so object spawn draws are identical whether the
  feature is on or off.
* Candidate generation (the only layout step that consumes ``np.random``)
  runs before selection; selection itself is RNG-free.

Config keys (all optional)
--------------------------
``neutral_contested_fraction``   probability a seed is contested (default 0.4)
``neutral_contested_force``      True/False forces the decision (diagnostics)
``neutral_contested_max_gap``    upper end of the qualifying gap (default 0.0 m:
                                 the sweep must reach the band)
``neutral_contested_min_gap``    deepest accepted intrusion (default -0.04 m)
``neutral_contested_corridor_radius``  corridor half-width (0.10 m)
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


CONTESTED_RNG_SALT = 5_310_771
DEFAULT_CONTESTED_FRACTION = 0.4
DEFAULT_CORRIDOR_RADIUS = 0.10
DEFAULT_MAX_GAP = 0.0
# Deepest intrusion into the corridor band accepted when no candidate merely
# touches it.  A gap of -0.057 (blocks) put the human palm on the
# block->slot line and even the full-state expert collided; -0.045 / -0.016
# (dump, blocks size) were solved with >= 12 cm clearance.
DEFAULT_MIN_GAP = -0.04

# Task-region label prefixes: objects the robot picks, and places it puts them.
# Kinds whose hand works on the table top next to the task objects; the
# others (seated clap/writing, the far-reaching can-to-bowl clip, laptop,
# typing, phone) never bring the arm into the robot's corridor.
CONTESTABLE_KINDS = ("mouse", "sponge", "bottle")

PICK_LABEL_PREFIXES = (
    "block:", "objects:", "breads:", "shoes:", "food:", "cabinet_object",
    "categorize_obj:",
)
PLACE_LABEL_PREFIXES = (
    "slot:", "basket", "shoebox", "target", "cooktop", "pan", "tray",
    "cabinet", "categorize_basket:",
)


def contested_draw(seed: int, fraction: float) -> tuple[bool, float]:
    """Seed-tied private draw; never touches the global ``np.random`` state."""
    rng = np.random.default_rng(int(seed) + CONTESTED_RNG_SALT)
    u = float(rng.random())
    return bool(u < float(fraction)), u


def point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()[:2]
    a = np.asarray(a, dtype=np.float64).ravel()[:2]
    b = np.asarray(b, dtype=np.float64).ravel()[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_rect_distance(a, b, lo, hi, samples_per_m: float = 100.0) -> float:
    """Distance between a segment and an axis-aligned rectangle (0 if touching)."""
    a = np.asarray(a, dtype=np.float64).ravel()[:2]
    b = np.asarray(b, dtype=np.float64).ravel()[:2]
    lo = np.asarray(lo, dtype=np.float64).ravel()[:2]
    hi = np.asarray(hi, dtype=np.float64).ravel()[:2]
    n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) * samples_per_m)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = a[None, :] + t * (b - a)[None, :]
    clamped = np.minimum(np.maximum(pts, lo[None, :]), hi[None, :])
    return float(np.min(np.linalg.norm(pts - clamped, axis=1)))


def region_discs(anchor, footprint: dict) -> list[tuple[np.ndarray, float]]:
    """World-XY swept discs of a footprint placed at ``anchor``."""
    anchor = np.asarray(anchor, dtype=np.float64).ravel()[:2]
    out = []
    for item in footprint.get("region_points") or []:
        if isinstance(item, dict):
            offset = np.asarray(item.get("offset", [0.0, 0.0]), dtype=np.float64).ravel()[:2]
            radius = float(item.get("radius", 0.05))
        else:
            offset = np.asarray(item[:2], dtype=np.float64).ravel()[:2]
            radius = float(item[2])
        out.append((anchor + offset, radius))
    return out


def footprint_corridor_gap(
    anchor,
    footprint: dict,
    corridors: Sequence[tuple[Any, Any]],
    corridor_radius: float = DEFAULT_CORRIDOR_RADIUS,
) -> float:
    """Signed gap between the human sweep and the corridor band.

    ``gap = dist(sweep, corridor centreline) - corridor_radius``; the sweep is
    the footprint's ``region_points`` discs, or its AABB when it has none.
    Negative means the sweep enters the band.  ``inf`` without corridors.
    """
    if not corridors:
        return float("inf")
    best = float("inf")
    discs = region_discs(anchor, footprint)
    if discs:
        for a, b in corridors:
            for center, radius in discs:
                best = min(best, point_segment_distance(center, a, b) - radius)
    else:
        anchor = np.asarray(anchor, dtype=np.float64).ravel()[:2]
        lo = anchor + np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
        hi = anchor + np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
        for a, b in corridors:
            best = min(best, segment_rect_distance(a, b, lo, hi))
    return float(best - float(corridor_radius))


def choose_contested(
    gaps: Sequence[float],
    max_gap: float = DEFAULT_MAX_GAP,
    min_gap: float = DEFAULT_MIN_GAP,
) -> int | None:
    """Index of the contested choice among VALID candidates, or None.

    Qualifying candidates have ``min_gap <= gap <= max_gap`` (default
    ``[-0.04, 0.0]``: the hand sweep enters the corridor band, but not deeper
    than 4 cm, where the palm lands on the block->slot line and even
    the full-state expert hits it).  The deepest qualifying candidate wins;
    ties keep candidate order, so the choice is deterministic.
    """
    finite = [(float(g), i) for i, g in enumerate(gaps) if np.isfinite(g)]
    band = [(g, i) for g, i in finite if float(min_gap) <= g <= float(max_gap)]
    if not band:
        return None
    return min(band)[1]


def contest_failure_reason(gaps: Sequence[float], n_corridors: int,
                           max_gap: float = DEFAULT_MAX_GAP,
                           min_gap: float = DEFAULT_MIN_GAP) -> str:
    """Reason code when :func:`choose_contested` returns None."""
    if not n_corridors:
        return "no_corridor"
    finite = [float(g) for g in gaps if np.isfinite(g)]
    if not finite:
        return "no_valid_candidate"
    if any(g < float(min_gap) for g in finite):
        return "only_too_deep_candidates"
    return "no_candidate_in_band"


def _label_key(label: str) -> str:
    return str(label).split(":", 1)[1] if ":" in str(label) else ""


def corridors_from_regions(
    regions: Iterable[tuple[Any, float, str]],
    tcp_xy: Sequence[Any] = (),
    extra_places: Sequence[Any] = (),
    place_map: dict[str, Sequence[str]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Robot pick->place corridors from labelled task regions.

    Segments: every pick object -> its place.  A pick is paired with only its
    own place when the pairing is known: ``place_map[pick_label]`` (e.g.
    categorize object -> its basket), or ``block:K -> slot:K``.  Otherwise it
    is paired with every place (``extra_places`` adds off-table goals such as
    a bin).  Base bowl ``bowls:0`` is the place; other bowls are picks.
    ``tcp_xy`` (gripper TCP -> pick) is accepted for callers that want it;
    the contested layout passes none (a human on the home TCP is an
    unavoidable collision, not a contest).
    """
    picks, places = [], []
    for center, _radius, label in regions:
        label = str(label)
        xy = np.asarray(center, dtype=np.float64).ravel()[:2]
        if label.startswith("bowls:"):
            (places if label == "bowls:0" else picks).append((label, xy))
        elif label.startswith(PICK_LABEL_PREFIXES):
            picks.append((label, xy))
        elif label.startswith(PLACE_LABEL_PREFIXES):
            places.append((label, xy))
    places.extend(("extra", np.asarray(p, dtype=np.float64).ravel()[:2]) for p in extra_places)
    place_map = dict(place_map or {})
    by_label = {lab: xy for lab, xy in places}
    out = []
    for tcp in tcp_xy:
        tcp = np.asarray(tcp, dtype=np.float64).ravel()[:2]
        out.extend((tcp, p) for _lab, p in picks)
    for lab, p in picks:
        targets = None
        if lab in place_map:
            targets = [by_label[q] for q in place_map[lab] if q in by_label]
        elif lab.startswith("block:") and f"slot:{_label_key(lab)}" in by_label:
            targets = [by_label[f"slot:{_label_key(lab)}"]]
        if not targets:
            targets = [q for _l, q in places]
        out.extend((p, q) for q in targets if float(np.linalg.norm(p - q)) > 1e-6)
    return out


class _NeutralAvatarContestedMixin:
    """Per-episode contested decision, corridors and metrics."""

    NEUTRAL_CONTESTED_FRACTION = DEFAULT_CONTESTED_FRACTION
    NEUTRAL_CONTESTED_CORRIDOR_RADIUS = DEFAULT_CORRIDOR_RADIUS
    NEUTRAL_CONTESTED_MAX_GAP = DEFAULT_MAX_GAP
    # Kinds that may be contested (table-reach work beside the task objects).
    NEUTRAL_CONTESTED_KINDS = CONTESTABLE_KINDS

    # -- config ----------------------------------------------------------

    def _neutral_contested_params(self) -> dict:
        cfg = self.config or {}
        radius = float(cfg.get(
            "neutral_contested_corridor_radius", self.NEUTRAL_CONTESTED_CORRIDOR_RADIUS,
        ))
        return {
            "fraction": float(cfg.get(
                "neutral_contested_fraction", self.NEUTRAL_CONTESTED_FRACTION,
            )),
            "corridor_radius": radius,
            "max_gap": float(cfg.get("neutral_contested_max_gap", self.NEUTRAL_CONTESTED_MAX_GAP)),
            "min_gap": float(cfg.get("neutral_contested_min_gap", DEFAULT_MIN_GAP)),
        }

    def _neutral_contested_kind(self) -> str:
        job = getattr(self, "_neutral_avatar_job", None) or {}
        return str(job.get("kind", "") or "")

    def _neutral_contested_requested(self) -> bool:
        """Whether this episode's layout should be contested.

        The seed draw is made once per episode (private RNG).  A drawn seed
        whose motion kind is not table-reach falls back to the normal layout
        with reason ``kind_not_contestable:<kind>``.
        """
        state = getattr(self, "_neutral_contested_state", None)
        seed = int(getattr(self, "_neutral_avatar_reset_seed", 0))
        if state is None or state.get("seed") != seed:
            params = self._neutral_contested_params()
            forced = (self.config or {}).get("neutral_contested_force")
            requested, u = contested_draw(seed, params["fraction"])
            if forced is not None:
                requested = bool(forced)
            state = {
                "seed": seed,
                "draw": u,
                "requested": bool(requested),
                "forced": None if forced is None else bool(forced),
                "fraction": params["fraction"],
                "kind": "",
                "supported": False,
                "applied": False,
                "failed": False,
                "reason": None if requested else "not_drawn",
                "gap_m": None,
                "n_valid_candidates": None,
                "selected_xy": None,
                "selected_side": None,
                "corridors": [],
            }
            self._neutral_contested_state = state
        kind = self._neutral_contested_kind()
        if kind:
            state["kind"] = kind
            state["supported"] = kind in self.NEUTRAL_CONTESTED_KINDS
        if not state["requested"]:
            return False
        if kind and not state["supported"]:
            state["applied"] = False
            state["failed"] = True
            state["reason"] = f"kind_not_contestable:{kind}"
            return False
        return True

    # -- geometry --------------------------------------------------------

    def _neutral_contested_extra_places(self) -> list:
        """Off-table goals (e.g. a floor bin) that task regions do not list."""
        return []

    def _neutral_contested_place_map(self) -> dict[str, list[str]]:
        """Pick label -> its own place labels, when the task pairs them.

        Categorize: each object goes to one basket (``_sort_targets``).
        Blocks pair ``block:K -> slot:K`` inside ``corridors_from_regions``.
        """
        out: dict[str, list[str]] = {}
        for i, item in enumerate(getattr(self, "_sort_targets", None) or []):
            try:
                out[f"categorize_obj:{i}"] = [f"categorize_basket:{int(item[3])}"]
            except Exception:
                continue
        return out

    def _settle_scene(self, *args, **kwargs):
        # Snapshot the task-object regions at their SPAWN poses (seeded,
        # renderer/physics-backend independent) before the reset settle.  The
        # contested corridors are built from this snapshot, so the contested
        # decision does not flip with mm-level settle drift between CPU, GPU
        # and NYX runs near the band edge.
        try:
            self._neutral_presettle_regions = list(self.neutral_task_regions())
        except Exception:
            self._neutral_presettle_regions = None
        return super()._settle_scene(*args, **kwargs)

    def _neutral_contested_corridor_regions(self, task_regions):
        """Regions the corridors are built from: the pre-settle snapshot when
        available (``neutral_contested_presettle``, default on)."""
        pre = getattr(self, "_neutral_presettle_regions", None)
        if pre and bool((self.config or {}).get("neutral_contested_presettle", True)):
            return pre
        return task_regions

    def _neutral_contested_corridors(self, task_regions) -> list[tuple[np.ndarray, np.ndarray]]:
        return corridors_from_regions(
            self._neutral_contested_corridor_regions(task_regions),
            extra_places=self._neutral_contested_extra_places(),
            place_map=self._neutral_contested_place_map(),
        )

    def _neutral_contested_select(self, records: list[dict], task_regions) -> dict | None:
        """Pick the contested record among VALID ``records`` (each has
        ``xy`` and ``footprint``).  Updates the per-episode state; returns
        None when no candidate qualifies (caller keeps its normal choice)."""
        self._neutral_contested_requested()
        state = getattr(self, "_neutral_contested_state", None) or {}
        params = self._neutral_contested_params()
        corridors = self._neutral_contested_corridors(task_regions)
        state["corridors"] = [
            [np.round(a, 4).tolist(), np.round(b, 4).tolist()] for a, b in corridors
        ]
        state["n_valid_candidates"] = len(records)
        state["band_m"] = [params["min_gap"], params["max_gap"]]
        gaps = [
            footprint_corridor_gap(rec["xy"], rec["footprint"], corridors, params["corridor_radius"])
            for rec in records
        ]
        for rec, gap in zip(records, gaps):
            rec["contest_gap"] = float(gap)
        finite = [g for g in gaps if np.isfinite(g)]
        state["candidate_gap_range_m"] = [float(min(finite)), float(max(finite))] if finite else None
        idx = choose_contested(gaps, params["max_gap"], params["min_gap"]) if corridors else None
        if idx is None:
            state["applied"] = False
            state["failed"] = True
            state["reason"] = contest_failure_reason(
                gaps, len(corridors), params["max_gap"], params["min_gap"])
            state["gap_m"] = None
            state["selected_xy"] = None
            state["selected_side"] = None
            self._neutral_contested_state = state
            return None
        chosen = records[idx]
        state["applied"] = True
        state["failed"] = False
        state["reason"] = "applied"
        state["gap_m"] = float(gaps[idx])
        state["selected_xy"] = np.asarray(chosen["xy"], dtype=np.float64).ravel()[:2].round(4).tolist()
        state["selected_side"] = chosen.get("side")
        self._neutral_contested_state = state
        return chosen

    def _neutral_contested_mark_unsupported(self, reason: str) -> None:
        state = getattr(self, "_neutral_contested_state", None)
        if state is not None and state.get("requested") and not state.get("applied"):
            state["failed"] = True
            if not str(state.get("reason") or "").startswith("kind_not_contestable"):
                state["reason"] = str(reason)

    def _neutral_contested_finalize(self) -> dict:
        """Every outcome carries a reason (requested but never decided ->
        ``selector_not_reached``)."""
        state = getattr(self, "_neutral_contested_state", None)
        if state is None and getattr(self, "_neutral_avatar_reset_seed", None) is not None:
            self._neutral_contested_requested()
            state = getattr(self, "_neutral_contested_state", None)
        if state is None:
            return {}
        if state.get("requested") and not state.get("applied") and not state.get("reason"):
            state["failed"] = True
            state["reason"] = "selector_not_reached"
        return state

    # -- metrics ---------------------------------------------------------

    def neutral_layout_metrics(self) -> dict:
        state = dict(self._neutral_contested_finalize() or {})
        job = getattr(self, "_neutral_avatar_job", None) or {}
        work_xy = getattr(self, "_neutral_avatar_selected_work_xy", None)
        body = getattr(self, "_neutral_avatar_body_pos", None)
        return {
            "neutral_contested": bool(state.get("applied", False)),
            "neutral_contested_requested": bool(state.get("requested", False)),
            "neutral_contested_reason": state.get("reason"),
            "neutral_layout": {
                "motion": job.get("base_motion"),
                "kind": job.get("kind"),
                "work_xy": None if work_xy is None else np.asarray(work_xy, dtype=float).ravel()[:2].round(4).tolist(),
                "body_pos": None if body is None else np.asarray(body, dtype=float).ravel()[:3].round(4).tolist(),
                "contested": state,
                "eval_trigger_step": getattr(self, "_neutral_eval_trigger_step", None),
            },
        }
