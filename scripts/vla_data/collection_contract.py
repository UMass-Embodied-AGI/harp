"""Metadata and acceptance rules for VLA collection products.

These helpers do not import the simulator or run episodes. Collection startup,
playback, renderer verification, and file quarantine stay in ``collect_vla``;
the NYX validator additionally checks the recorded image payloads.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_ACT_H5_KEYS = (
    "t",
    "qpos",
    "qpos_target",
    "qpos_delta_action",
    "qpos_target_action",
    "gripper_action",
    "gripper_target_action",
    "images",
)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bytes,)):
        return None
    return str(o)


def _neutral_meta(task, config: dict) -> dict:
    job = getattr(task, "_neutral_avatar_job", None)
    if not job:
        meta = {
            "neutral_avatar_motion": config.get("neutral_avatar_motion"),
            "neutral_avatar_motion_kind": None,
        }
    else:
        meta = {
            "neutral_avatar_motion": job.get("base_motion") or job.get("motion"),
            "neutral_avatar_motion_resolved": job.get("motion"),
            "neutral_avatar_motion_kind": job.get("kind"),
            "neutral_avatar_motion_objects": list(job.get("objects") or []),
        }
    selected_foods = getattr(task, "selected_food_names", None)
    if selected_foods:
        meta["selected_foods"] = list(selected_foods)
    return meta


def _collection_avatar_meta(task, config):
    """Record the selected model, not merely the requested randomization flag."""
    return {
        "randomize_avatar": config.get("randomize_avatar") is True,
        "avatar_randomized": getattr(task, "_vla_avatar_randomized", False) is True,
        "avatar_model": getattr(task, "_vla_avatar_model", None),
        "avatar_initialized": getattr(task, "avatar", None) is not None,
    }


def _success_from_evaluation(task, evaluate_dict: dict, fallback: bool = False) -> bool:
    """Apply the collector collision gate with an explicit task exemption."""
    success = bool(evaluate_dict.get("success", fallback))
    collided = bool(
        (evaluate_dict.get("avatar_collision") or {}).get("any_collision")
    )
    allow_collision = bool(
        getattr(task, "VLA_ALLOW_AVATAR_COLLISION", False)
    )
    if collided and not allow_collision:
        return False
    return success


def _require_reusable_collection_contract(record, config, *, require_play_success):
    """Only reuse outputs that prove this invocation's collection contract.

    Standalone readers may inspect legacy datasets; skipping a new collection
    request is stricter and must not silently reuse fixed-avatar or incomplete
    playback data from an earlier collector.
    """
    version = record.get("collection_contract_version")
    if type(version) is not int or version < 2:
        raise ValueError("existing output lacks the current collection contract")
    play_success = record.get("play_success")
    if (type(play_success) is not bool
            or ((require_play_success or record.get("success") is True) and not play_success)):
        raise ValueError("existing output lacks the required expert playback result")
    requested_randomization = config.get("randomize_avatar") is True
    if record.get("randomize_avatar") is not requested_randomization:
        raise ValueError("existing output does not match requested avatar randomization")
    if record.get("avatar_randomized") is not requested_randomization:
        raise ValueError("existing output does not prove actual avatar randomization")
    if requested_randomization and (
        record.get("avatar_initialized") is not True
        or not isinstance(record.get("avatar_model"), str)
        or not record["avatar_model"]
    ):
        raise ValueError("existing output lacks verified randomized avatar identity")


def validate_recorded_h5(episode_dir: Path, require_qpos_target: bool = True) -> dict:
    """Fail fast when a recorded episode is unusable for ACT/RLDS conversion."""
    import h5py
    import numpy as np

    episode_dir = Path(episode_dir)
    h5_path = episode_dir / "steps.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"missing recorded H5: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        required = list(REQUIRED_ACT_H5_KEYS)
        if require_qpos_target:
            required.extend(["qpos_target", "qpos_target_action"])
        missing = [key for key in dict.fromkeys(required) if key not in keys]
        if missing:
            raise KeyError(f"{h5_path} missing required VLA keys: {missing}")

        n_steps = int(f["qpos"].shape[0])
        if n_steps <= 0:
            raise ValueError(f"{h5_path} has no recorded training rows")

        for key in (
            "t",
            "qpos_target",
            "qpos_delta_action",
            "qpos_target_action",
            "gripper_action",
            "gripper_target_action",
        ):
            if int(f[key].shape[0]) != n_steps:
                raise ValueError(
                    f"{h5_path} dataset {key!r} length {f[key].shape[0]} "
                    f"does not match qpos length {n_steps}"
                )

        if "images" not in f or len(f["images"].keys()) == 0:
            raise ValueError(f"{h5_path} has no image camera datasets")

        invalid_targets = None
        if "qpos_target_action_valid" in f:
            valid = np.asarray(f["qpos_target_action_valid"][:], dtype=bool)
            invalid_targets = int((~valid).sum())
            if invalid_targets:
                raise ValueError(
                    f"{h5_path} has {invalid_targets}/{n_steps} invalid qpos_target_action labels"
                )

        schema = f.attrs.get("schema_version", "unknown")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8", errors="replace")

        return {
            "path": str(h5_path),
            "n_steps": n_steps,
            "schema_version": str(schema),
            "keys": sorted(keys),
            "cams": sorted(f["images"].keys()),
            "invalid_qpos_target_actions": invalid_targets,
        }
