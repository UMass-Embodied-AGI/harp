"""Post-hoc human-robot safety metrics from the per-check distance log.

Pure numpy — no Genesis imports — so everything here is unit-testable on
a login node.  Input is `BaseTask.avatar_collision_log`: one dict per
stride-sampled `AvatarCollisionChecker.check()` call, with keys
`min_distance` (signed metres, non-ignored shapes only), `collided`,
`closest_pair`, and (since the sim-step counter fix) `sim_step` / `t`.

Metric naming follows the HRI/HRC literature:

- **minimum separation distance** — the core speed-and-separation
  monitoring quantity of ISO/TS 15066 (episode min + distribution
  stats: mean / p5 / p25 / median).
- **intrusion time ratio** — fraction of checked time spent below a
  proximity threshold (CrowdNav-style "discomfort frequency"; HRC
  papers report "% of trajectory with separation > X cm").
- **collision rate** — episode-level (any contact) and step-level
  (fraction of checks in contact) rates.
- **near miss** — separation threshold violated without any contact.
- **approach speed** — finite-difference of the separation series
  (velocity-aware signal in the spirit of ISO/TS 15066's protective
  distance and the Kulic-Croft danger index).

Sampling caveat: checks run every `collision_check_stride` sim steps
(default 30 ≈ 60 ms at dt=0.002), so brief contacts between checks can
be missed — `collision_step_ratio` and `time_below_s` are lower-bound
estimates.  Each check costs <1 ms; use `--collision-check-stride 5`-ish
for metric-quality runs.
"""
import math

import numpy as np

DEFAULT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.20)


def _threshold_key(tau: float) -> str:
    return f"{float(tau):.2f}"


def _record_time(record, idx, check_interval_s):
    """Best-effort timestamp (s) for a check record."""
    t = record.get("t")
    if t is not None:
        return float(t)
    return float((idx + 1) * check_interval_s)


def compute_safety_metrics(records, thresholds=DEFAULT_DISTANCE_THRESHOLDS,
                           check_interval_s=0.06):
    """Per-episode safety metrics from a list of check() result dicts.

    Parameters
    ----------
    records : list of dict
        `BaseTask.avatar_collision_log` (chronological).
    thresholds : iterable of float
        Separation thresholds (m) for intrusion/violation stats.
    check_interval_s : float
        Wall-sim time between consecutive checks (stride * dt).

    Returns a JSON-safe dict (native python types only); distance stats
    are None when no valid distance samples exist, ratios are None when
    there were no checks at all.
    """
    thresholds = sorted(float(t) for t in thresholds)
    records = list(records or [])
    n_checks = len(records)

    valid = [(i, r, float(r["min_distance"])) for i, r in enumerate(records)
             if r.get("min_distance") is not None
             and math.isfinite(float(r["min_distance"]))]
    dists = np.array([d for _, _, d in valid], dtype=np.float64)
    n_valid = len(valid)

    out = {
        "enabled": True,
        "n_checks": int(n_checks),
        "n_valid_distance_checks": int(n_valid),
        "check_interval_s": float(check_interval_s),
        "checked_duration_s": float(n_checks * check_interval_s),
        "min_distance_m": None,
        "mean_distance_m": None,
        "p5_distance_m": None,
        "p25_distance_m": None,
        "median_distance_m": None,
        "closest_pair": None,
        "closest_sim_step": None,
        "closest_time_s": None,
    }

    if n_valid > 0:
        arg_min = int(np.argmin(dists))
        idx_min, rec_min, _ = valid[arg_min]
        out.update({
            "min_distance_m": float(dists.min()),
            "mean_distance_m": float(dists.mean()),
            "p5_distance_m": float(np.percentile(dists, 5)),
            "p25_distance_m": float(np.percentile(dists, 25)),
            "median_distance_m": float(np.percentile(dists, 50)),
            "closest_pair": rec_min.get("closest_pair"),
            "closest_sim_step": rec_min.get("sim_step"),
            "closest_time_s": _record_time(rec_min, idx_min, check_interval_s),
        })

    # --- collision (contact) ---------------------------------------
    collision_idxs = [i for i, r in enumerate(records) if r.get("collided")]
    any_collision = bool(collision_idxs)
    collision = {
        "any_collision": any_collision,
        # Checks run over the scored (task-phase) log, and how many of them
        # found contact.  ``n_collision_checks`` used to hold the colliding
        # count, which read as "0 checks" on every collision-free episode.
        "n_collision_checks": int(n_checks),
        "n_colliding_checks": int(len(collision_idxs)),
        "collision_step_ratio": (float(len(collision_idxs) / n_checks)
                                 if n_checks else None),
        "first_collision_sim_step": None,
        "first_collision_time_s": None,
    }
    if any_collision:
        i0 = collision_idxs[0]
        collision["first_collision_sim_step"] = records[i0].get("sim_step")
        collision["first_collision_time_s"] = _record_time(
            records[i0], i0, check_interval_s)
    out["collision"] = collision

    # --- intrusion (threshold violations) --------------------------
    intrusion = {}
    for tau in thresholds:
        entry = {
            "threshold_m": float(tau),
            "n_violations": 0,
            "n_checked": int(n_valid),
            "intrusion_time_ratio": None,
            "time_below_s": None,
            "any_violation": False,
            "near_miss": False,
            "first_violation_sim_step": None,
            "first_violation_time_s": None,
        }
        if n_valid > 0:
            viol = [(i, r) for (i, r, d) in valid if d < tau]
            n_viol = len(viol)
            entry.update({
                "n_violations": int(n_viol),
                "intrusion_time_ratio": float(n_viol / n_valid),
                "time_below_s": float(n_viol * check_interval_s),
                "any_violation": bool(n_viol > 0),
                "near_miss": bool(n_viol > 0 and not any_collision),
            })
            if n_viol > 0:
                i0, r0 = viol[0]
                entry["first_violation_sim_step"] = r0.get("sim_step")
                entry["first_violation_time_s"] = _record_time(
                    r0, i0, check_interval_s)
        intrusion[_threshold_key(tau)] = entry
    out["intrusion"] = intrusion

    # --- approach speed (finite difference of the separation series) -
    approach = {"max_approach_speed_mps": None,
                "approach_speed_at_min_mps": None}
    if n_valid >= 2 and check_interval_s > 0:
        # Positive = robot and human getting closer.  Use the interval
        # between consecutive checks (index gap accounted for so missing
        # samples don't inflate speeds).
        idxs = np.array([i for i, _, _ in valid], dtype=np.float64)
        gaps = np.diff(idxs) * check_interval_s
        speeds = -(np.diff(dists)) / gaps
        arg_min = int(np.argmin(dists))
        approach["max_approach_speed_mps"] = float(speeds.max())
        if arg_min > 0:
            approach["approach_speed_at_min_mps"] = float(speeds[arg_min - 1])
    out["approach"] = approach

    return out


def aggregate_safety(safety_list):
    """Aggregate per-episode `safety` dicts across episodes.

    Entries that are not dicts or have `enabled != True` are skipped
    (episodes without an avatar / with tracking off).  None-valued
    fields are skipped from means, and episodes with zero checks are
    excluded from rate denominators (no data ≠ no violation).
    """
    tracked = [s for s in (safety_list or [])
               if isinstance(s, dict) and s.get("enabled") is True]
    out = {"episodes_tracked": len(tracked)}
    if not tracked:
        return out

    def _mean(values):
        vals = [float(v) for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    checked = [s for s in tracked if s.get("n_checks")]
    out["episode_collision_rate"] = _mean(
        [1.0 if s.get("collision", {}).get("any_collision") else 0.0
         for s in checked])
    out["mean_collision_step_ratio"] = _mean(
        s.get("collision", {}).get("collision_step_ratio") for s in checked)

    min_ds = [s.get("min_distance_m") for s in tracked
              if s.get("min_distance_m") is not None]
    out["mean_min_distance_m"] = _mean(min_ds)
    out["min_min_distance_m"] = float(min(min_ds)) if min_ds else None
    out["mean_distance_m"] = _mean(s.get("mean_distance_m") for s in tracked)

    keys = sorted({k for s in tracked for k in (s.get("intrusion") or {})})
    out["thresholds"] = {}
    for k in keys:
        entries = [s["intrusion"][k] for s in tracked
                   if k in (s.get("intrusion") or {})
                   and s["intrusion"][k].get("intrusion_time_ratio") is not None]
        out["thresholds"][k] = {
            "episode_violation_rate": _mean(
                [1.0 if e.get("any_violation") else 0.0 for e in entries]),
            "mean_intrusion_time_ratio": _mean(
                e.get("intrusion_time_ratio") for e in entries),
            "near_miss_rate": _mean(
                [1.0 if e.get("near_miss") else 0.0 for e in entries]),
        }
    return out


# ----------------------------------------------------------------------
# CSV flattening (for eval_expert_set.py results.csv)
# ----------------------------------------------------------------------

_CSV_FIXED_FIELDS = (
    "any_collision",
    "collision_step_ratio",
    "min_distance_m",
    "mean_distance_m",
    "p5_distance_m",
)


def safety_csv_fields(safety_list):
    """Column names for the safety block, incl. per-threshold columns."""
    keys = sorted({k for s in (safety_list or [])
                   if isinstance(s, dict)
                   for k in (s.get("intrusion") or {})})
    return list(_CSV_FIXED_FIELDS) + [f"intrusion_ratio_{k}" for k in keys]


def safety_csv_row(safety):
    """Flatten one episode's `safety` dict into {column: value}.

    Missing / disabled → empty strings so the CSV stays rectangular.
    """
    if not isinstance(safety, dict) or safety.get("enabled") is not True:
        return {}
    row = {
        "any_collision": safety.get("collision", {}).get("any_collision"),
        "collision_step_ratio":
            safety.get("collision", {}).get("collision_step_ratio"),
        "min_distance_m": safety.get("min_distance_m"),
        "mean_distance_m": safety.get("mean_distance_m"),
        "p5_distance_m": safety.get("p5_distance_m"),
    }
    for k, entry in (safety.get("intrusion") or {}).items():
        row[f"intrusion_ratio_{k}"] = entry.get("intrusion_time_ratio")
    return {k: ("" if v is None else v) for k, v in row.items()}
