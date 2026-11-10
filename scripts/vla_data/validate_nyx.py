"""Validate completed NYX training episodes without importing the simulator.

The marker is a commit record for an accepted episode, not a replacement for
validation. Consumers must match its file fingerprints before trusting it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile


SCHEMA_VERSION = "vla_steps_v2_qpos_target"
REQUIRED_CAMERAS = ("head_camera", "right_wrist")
SOURCE_FILES = ("steps.h5", "meta.json", "verdict.json")
VISIBLE_FRAME_MAX_RGB_THRESHOLD = 3


def _read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def episode_fingerprints(episode_dir):
    result = {}
    for name in SOURCE_FILES:
        stat = (Path(episode_dir) / name).stat()
        result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _atomic_marker(path, summary):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".validated-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_episode(episode_dir, *, expected_task=None, expected_seed=None,
                     decode_all=False, write_marker=False, require_visible_frames=False):
    """Return an accepted-episode summary or raise on missing/corrupt data.

    Every JPEG is checked for a nonempty payload. By default first/middle/last
    frames of every camera are decoded; ``decode_all`` decodes every frame.
    ``require_visible_frames`` also forces full decoding and rejects frames
    whose RGB channel maxima are all at most 3 (on the uint8 scale).
    Numeric datasets and both commanded-target validity masks are checked in
    full. No simulation, CUDA, or Genesis imports are performed.
    """
    import h5py
    import numpy as np
    from PIL import Image

    decode_all = decode_all or require_visible_frames
    episode_dir = Path(episode_dir)
    marker = episode_dir / "validated.json"
    # A failed revalidation must not leave an obsolete acceptance marker.
    if write_marker:
        marker.unlink(missing_ok=True)
    before = episode_fingerprints(episode_dir)
    meta = _read_json(episode_dir / "meta.json")
    verdict = _read_json(episode_dir / "verdict.json")
    for name, record in (("meta", meta), ("verdict", verdict)):
        if record.get("success") is not True or record.get("plan_success") is not True:
            raise ValueError(f"{name}: episode or plan was not successful")
        # New recordings must prove complete playback. Keep legacy schema
        # readable, but never accept an explicitly failed playback.
        if (record.get("play_success") is False
                or (record.get("collection_contract_version", 1) >= 2
                    and record.get("play_success") is not True)):
            raise ValueError(f"{name}: expert playback was not successful")
        if record.get("infra_failure_reason") or record.get("error"):
            raise ValueError(f"{name}: episode has a recorded error")
        if (record.get("renderer") != "nyx"
                or record.get("renderer_verified") is not True
                or record.get("camera_manager") != "NyxCameraManager"):
            raise ValueError(f"{name}: actual NYX renderer was not verified")
        if not isinstance(record.get("task"), str) or not record["task"]:
            raise ValueError(f"{name}: missing task identity")
        if type(record.get("seed")) is not int or record["seed"] < 0:
            raise ValueError(f"{name}: invalid seed")
    for key in ("task", "seed"):
        if meta[key] != verdict[key]:
            raise ValueError(f"meta/verdict {key} mismatch")
    randomized = (meta.get("config_snapshot", {}).get("randomize_avatar") is True
                  or meta.get("randomize_avatar") is True
                  or verdict.get("randomize_avatar") is True)
    if randomized:
        for name, record in (("meta", meta), ("verdict", verdict)):
            if (record.get("randomize_avatar") is not True
                    or record.get("avatar_randomized") is not True
                    or record.get("avatar_initialized") is not True
                    or not isinstance(record.get("avatar_model"), str)
                    or not record["avatar_model"]):
                raise ValueError(f"{name}: randomized avatar identity was not verified")
        if meta["avatar_model"] != verdict["avatar_model"]:
            raise ValueError("meta/verdict avatar_model mismatch")
        if meta.get("config_snapshot", {}).get("randomize_avatar") is not True:
            raise ValueError("meta: randomize_avatar config must be true")
    if expected_task is not None and meta["task"] != expected_task:
        raise ValueError("recording does not match the requested task")
    if expected_seed is not None and meta["seed"] != expected_seed:
        raise ValueError("recording does not match the requested seed")
    if not isinstance(meta.get("instruction"), str) or not meta["instruction"].strip():
        raise ValueError("meta: missing episode instruction")
    if verdict.get("mode") != "record":
        raise ValueError("verdict: expected record mode")
    if meta.get("recorder", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("meta: unsupported recorder schema")

    decoded = {}
    resolutions = {}
    with h5py.File(episode_dir / "steps.h5", "r") as handle:
        schema = handle.attrs.get("schema_version")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != SCHEMA_VERSION:
            raise ValueError("H5: unsupported recorder schema")
        if "qpos" not in handle or handle["qpos"].ndim != 2:
            raise ValueError("H5: missing or malformed qpos")
        n_steps, n_arm = handle["qpos"].shape
        if n_steps < 1 or n_arm < 1:
            raise ValueError("H5: no training rows or arm joints")
        if meta.get("n_steps") != n_steps:
            raise ValueError("meta/H5 row-count mismatch")
        if handle.attrs.get("n_arm") != n_arm:
            raise ValueError("H5: n_arm attribute mismatch")
        shapes = {"t": (n_steps,), "qpos": (n_steps, n_arm),
                  "qpos_target": (n_steps, n_arm),
                  "qpos_delta_action": (n_steps, n_arm),
                  "qpos_target_action": (n_steps, n_arm),
                  "ee": (n_steps, 7), "ee_delta_action": (n_steps, 6),
                  "gripper_meas": (n_steps,), "gripper_target": (n_steps,),
                  "gripper_action": (n_steps,), "gripper_target_action": (n_steps,)}
        for key, shape in shapes.items():
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise ValueError(f"H5: missing dataset {key}")
            dataset = handle[key]
            if dataset.shape != shape or dataset.dtype.kind not in "iuf":
                raise ValueError(f"H5: invalid shape/type for {key}")
            for offset in range(0, n_steps, 1024):
                if not np.isfinite(dataset[offset:offset + 1024]).all():
                    raise ValueError(f"H5: nonfinite values in {key}")
        times = handle["t"][:]
        if (handle["t"].dtype.kind not in "iu" or np.any(times < 0)
                or np.any(times[1:] <= times[:-1])):
            raise ValueError("H5: time steps must be increasing nonnegative integers")
        for key in ("qpos_target_valid", "qpos_target_action_valid"):
            if key not in handle or handle[key].shape != (n_steps,):
                raise ValueError(f"H5: missing or malformed validity mask {key}")
            values = handle[key][:]
            if values.dtype.kind not in "biu" or not np.all(values == 1):
                raise ValueError(f"H5: invalid commanded-target labels in {key}")
        if "images" not in handle or not isinstance(handle["images"], h5py.Group):
            raise ValueError("H5: missing image cameras")
        cameras = sorted(handle["images"].keys())
        if not set(REQUIRED_CAMERAS).issubset(cameras):
            raise ValueError(f"H5: required cameras missing: {REQUIRED_CAMERAS}")
        for camera in cameras:
            dataset = handle["images"][camera]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != (n_steps,):
                raise ValueError(f"H5: image length mismatch for {camera}")
            sample = set(range(n_steps)) if decode_all else {0, n_steps // 2, n_steps - 1}
            decoded[camera] = len(sample)
            for index in range(n_steps):
                raw = np.asarray(dataset[index])
                if raw.ndim != 1 or raw.dtype != np.uint8 or len(raw) < 4:
                    raise ValueError(f"H5: empty/invalid JPEG for {camera}[{index}]")
                if raw[0] != 255 or raw[1] != 216 or raw[-2] != 255 or raw[-1] != 217:
                    raise ValueError(f"H5: malformed JPEG for {camera}[{index}]")
                if index in sample:
                    try:
                        with Image.open(io.BytesIO(raw.tobytes())) as frame:
                            frame.load()
                            if frame.format != "JPEG" or frame.mode != "RGB":
                                raise ValueError("expected RGB JPEG")
                            if frame.width < 1 or frame.height < 1:
                                raise ValueError("empty frame")
                            size = list(frame.size)
                            channel_maxima = [maximum for _minimum, maximum in frame.getextrema()]
                    except Exception as exc:
                        raise ValueError(f"H5: JPEG decode failed for {camera}[{index}]") from exc
                    if require_visible_frames and max(channel_maxima) <= VISIBLE_FRAME_MAX_RGB_THRESHOLD:
                        raise ValueError(
                            f"H5: near-black RGB frame for {camera}[{index}]: "
                            f"channel maxima {channel_maxima} are all <= "
                            f"{VISIBLE_FRAME_MAX_RGB_THRESHOLD}"
                        )
                    if camera in resolutions and resolutions[camera] != size:
                        raise ValueError(f"H5: inconsistent image dimensions for {camera}")
                    resolutions[camera] = size
        keys = sorted(handle.keys())

    after = episode_fingerprints(episode_dir)
    if after != before:
        raise ValueError("episode files changed during validation")
    summary = {"validated": True, "validation_version": 1,
               "task": meta["task"], "seed": meta["seed"], "renderer": "nyx",
               "n_steps": n_steps, "n_arm": n_arm, "schema_version": SCHEMA_VERSION,
               "cams": cameras, "keys": keys, "resolutions": resolutions,
               "jpeg_validation": "all" if decode_all else "representative",
               "decoded_frames": decoded, "files": after}
    if require_visible_frames:
        summary.update(visible_frames_checked=True,
                       visible_frames_max_rgb_threshold=VISIBLE_FRAME_MAX_RGB_THRESHOLD)
    if randomized:
        summary.update(randomize_avatar=True, avatar_randomized=True,
                       avatar_model=meta["avatar_model"], avatar_initialized=True)
    if meta.get("play_success") is True and verdict.get("play_success") is True:
        summary["play_success"] = True
    if write_marker:
        _atomic_marker(marker, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decode-all", action="store_true")
    parser.add_argument("--write-marker", action="store_true")
    parser.add_argument("--require-visible-frames", action="store_true",
                        help="Decode every frame and reject entirely near-black RGB images")
    args = parser.parse_args(argv)
    try:
        summary = validate_episode(args.episode_dir, expected_task=args.task,
                                   expected_seed=args.seed, decode_all=args.decode_all,
                                   write_marker=args.write_marker,
                                   require_visible_frames=args.require_visible_frames)
    except Exception as exc:
        print(json.dumps({"validated": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
