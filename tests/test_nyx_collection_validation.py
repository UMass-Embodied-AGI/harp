"""Production acceptance checks using synthetic recordings, without Genesis."""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest

from scripts.vla_data.validate_nyx import SCHEMA_VERSION, validate_episode, main as validate_main


def write_episode(path, *, meta_overrides=None, verdict_overrides=None):
    path.mkdir(parents=True, exist_ok=True)
    n = 5
    common = dict(task="take_from_human_easy", seed=17, success=True,
                  plan_success=True, renderer="nyx", renderer_verified=True,
                  camera_manager="NyxCameraManager", error=None,
                  infra_failure_reason=None)
    meta = dict(common, instruction="Take the orange cup.", n_steps=n,
                recorder={"schema_version": SCHEMA_VERSION})
    meta.update(meta_overrides or {})
    verdict = dict(common, mode="record")
    verdict.update(verdict_overrides or {})
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "verdict.json").write_text(json.dumps(verdict))
    with h5py.File(path / "steps.h5", "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["n_arm"] = 7
        f["t"] = np.arange(1, n + 1, dtype=np.int64) * 25
        for key in ("qpos", "qpos_target", "qpos_target_action", "qpos_delta_action"):
            f[key] = np.zeros((n, 7), dtype=np.float32)
        f["ee"] = np.tile([0., 0., 1., 1., 0., 0., 0.], (n, 1))
        f["ee_delta_action"] = np.zeros((n, 6))
        for key in ("gripper_meas", "gripper_target", "gripper_action", "gripper_target_action"):
            f[key] = np.zeros(n)
        for key in ("qpos_target_valid", "qpos_target_action_valid"):
            f[key] = np.ones(n, dtype=bool)
        images = f.create_group("images")
        for cam in ("head_camera", "right_wrist"):
            blobs = np.empty(n, dtype=object)
            for index in range(n):
                stream = io.BytesIO()
                Image.new("RGB", (12, 8), (index * 40, 30, 80)).save(stream, format="JPEG")
                blobs[index] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            images.create_dataset(cam, data=blobs, dtype=h5py.vlen_dtype(np.dtype("uint8")))
    return path


@pytest.fixture
def episode(tmp_path):
    return write_episode(tmp_path / "seed_17")


def test_successful_recording_gets_atomic_fingerprinted_marker(episode):
    result = validate_episode(episode, expected_task="take_from_human_easy",
                              expected_seed=17, write_marker=True)
    assert result["validated"] is True
    assert result["n_steps"] == 5
    assert result["cams"] == ["head_camera", "right_wrist"]
    assert result["decoded_frames"] == {"head_camera": 3, "right_wrist": 3}
    assert json.loads((episode / "validated.json").read_text()) == result
    assert list(episode.glob(".validated-*.tmp")) == []
    assert result["files"]["steps.h5"]["size"] == (episode / "steps.h5").stat().st_size


@pytest.mark.parametrize("record,key,value", [
    ("meta", "success", False), ("verdict", "plan_success", False),
    ("meta", "renderer", "rasterizer"), ("verdict", "renderer_verified", False),
    ("meta", "camera_manager", "Camera"), ("verdict", "error", "interrupted"),
    ("verdict", "infra_failure_reason", "cuda"), ("meta", "n_steps", 0),
    ("meta", "instruction", ""), ("meta", "task", "other_task"),
    ("verdict", "seed", 18), ("verdict", "mode", "verdict"),
])
def test_rejects_failed_or_inconsistent_metadata_and_removes_old_marker(episode, record, key, value):
    validate_episode(episode, write_marker=True)
    path = episode / f"{record}.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_episode(episode, write_marker=True)
    assert not (episode / "validated.json").exists()


@pytest.mark.parametrize("key", ["qpos", "qpos_target", "qpos_target_action", "ee", "gripper_meas"])
def test_rejects_nonfinite_state_and_targets(episode, key):
    with h5py.File(episode / "steps.h5", "r+") as f:
        index = (2, 0) if f[key].ndim == 2 else 2
        f[key][index] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_episode(episode)


@pytest.mark.parametrize("key", ["qpos_target_valid", "qpos_target_action_valid"])
@pytest.mark.parametrize("mutation", ["missing", "false", "length"])
def test_requires_complete_commanded_target_masks(episode, key, mutation):
    with h5py.File(episode / "steps.h5", "r+") as f:
        if mutation == "false":
            f[key][2] = False
        else:
            del f[key]
            if mutation == "length":
                f[key] = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="validity mask|commanded-target"):
        validate_episode(episode)


@pytest.mark.parametrize("mutation", ["missing_camera", "image_length", "numeric_length", "time", "schema"])
def test_rejects_incomplete_or_wrong_schema_h5(episode, mutation):
    with h5py.File(episode / "steps.h5", "r+") as f:
        if mutation == "missing_camera":
            del f["images/right_wrist"]
        elif mutation == "image_length":
            del f["images/right_wrist"]
            f["images/right_wrist"] = np.zeros(4)
        elif mutation == "numeric_length":
            del f["ee"]
            f["ee"] = np.zeros((4, 7))
        elif mutation == "time":
            f["t"][2] = 1
        else:
            f.attrs["schema_version"] = "legacy"
    with pytest.raises(ValueError):
        validate_episode(episode)


def test_every_image_payload_is_checked_even_without_full_decode(episode):
    with h5py.File(episode / "steps.h5", "r+") as f:
        f["images/head_camera"][1] = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="empty/invalid JPEG"):
        validate_episode(episode)


def test_full_decode_finds_unsampled_corruption(episode):
    with h5py.File(episode / "steps.h5", "r+") as f:
        f["images/head_camera"][1] = np.array([255, 216, 0, 1, 255, 217], dtype=np.uint8)
    assert validate_episode(episode)["jpeg_validation"] == "representative"
    with pytest.raises(ValueError, match="JPEG decode failed"):
        validate_episode(episode, decode_all=True)


@pytest.mark.parametrize("value", [0, 3])
@pytest.mark.parametrize("write_marker", [False, True])
def test_visible_frame_check_finds_unsampled_near_black_and_respects_marker_writes(episode, value, write_marker):
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), (value, value, value)).save(stream, format="JPEG")
    with h5py.File(episode / "steps.h5", "r+") as f:
        f["images/right_wrist"][1] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    # The default retains historical behavior, including representative decoding.
    result = validate_episode(episode, write_marker=True)
    assert result["jpeg_validation"] == "representative"
    assert "visible_frames_checked" not in result
    with pytest.raises(ValueError, match=r"near-black RGB frame for right_wrist\[1\].*<= 3"):
        validate_episode(episode, require_visible_frames=True, write_marker=write_marker)
    if write_marker:
        assert not (episode / "validated.json").exists()
    else:
        assert json.loads((episode / "validated.json").read_text()) == result


def test_visible_frame_check_accepts_dark_constant_image_above_threshold(episode):
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), (4, 4, 4)).save(stream, format="JPEG")
    with h5py.File(episode / "steps.h5", "r+") as f:
        f["images/right_wrist"][1] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    result = validate_episode(episode, require_visible_frames=True, write_marker=True)
    assert result["visible_frames_checked"] is True
    assert result["visible_frames_max_rgb_threshold"] == 3
    assert result["jpeg_validation"] == "all"
    assert result["decoded_frames"] == {"head_camera": 5, "right_wrist": 5}
    assert json.loads((episode / "validated.json").read_text()) == result


def test_cli_visible_frame_flag_forces_complete_decoding(episode, capsys):
    assert validate_main([str(episode), "--require-visible-frames", "--write-marker"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["visible_frames_checked"] is True
    assert result["visible_frames_max_rgb_threshold"] == 3
    assert result["decoded_frames"] == {"head_camera": 5, "right_wrist": 5}


def test_identity_and_missing_h5_are_not_accepted(episode):
    with pytest.raises(ValueError, match="requested task"):
        validate_episode(episode, expected_task="pour_water")
    with pytest.raises(ValueError, match="requested seed"):
        validate_episode(episode, expected_seed=18)
    (episode / "steps.h5").unlink()
    with pytest.raises(FileNotFoundError):
        validate_episode(episode)


def test_cli_validates_all_jpegs_and_returns_machine_readable_summary(episode, capsys):
    assert validate_main([str(episode), "--decode-all", "--write-marker"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decoded_frames"] == {"head_camera": 5, "right_wrist": 5}


def collector_namespace(monkeypatch, tmp_path, *, renderer="nyx", real_camera=True,
                        playback_error=False, playback_success=True):
    """Execute the collector against an in-memory task, bypassing env imports."""
    source = Path(__file__).parents[1] / "scripts/vla_data/collect_vla.py"
    tree = ast.parse(source.read_text())
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            or (isinstance(node, ast.ImportFrom)
                and node.module == "scripts.vla_data.collection_contract")]
    fake_camera_module = ModuleType("envs.camera")
    class NyxCameraManager:
        pass
    fake_camera_module.NyxCameraManager = NyxCameraManager
    monkeypatch.setitem(sys.modules, "envs.camera", fake_camera_module)
    calls = []
    class Task:
        @staticmethod
        def default_instruction():
            return "Take the {object}."
        def __init__(self, config):
            self.config = dict(config)
            self.config["renderer"] = renderer
            self.cameras = NyxCameraManager() if real_camera else SimpleNamespace()
            self.plan_success = True
            self.instruction = "Take the orange cup."
            directory = Path(config["vla_recording"]["out_dir"])
            def close(meta):
                write_episode(directory, meta_overrides=meta,
                              verdict_overrides=json.loads((directory / "verdict.json").read_text()))
                return {"n_steps": 5}
            self.vla_recorder = SimpleNamespace(close=close, discard=lambda: {"n_captures": 0})
        def reset(self, seed):
            calls.append("reset")
        def play_once(self):
            calls.append("play")
            if playback_error:
                raise RuntimeError("task execution interrupted")
            return playback_success
        def evaluate(self):
            return {"success": True}
    namespace = dict(argparse=argparse, json=json, os=os, shutil=shutil, sys=sys,
                     time=time, traceback=traceback, Path=Path, np=np,
                     validate_episode=validate_episode,
                     resolve_task_class=lambda *a, **k: ("take_from_human_easy", Task),
                     load_config=lambda path: {"renderer": "nyx"},
                     get_task_instruction=lambda task: "Take the object.")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    root = tmp_path / "raw" / "take_from_human_easy" / "nyx"
    monkeypatch.setattr(sys, "argv", ["collect_vla", "--task", "take_from_human_easy",
                       "--seed", "17", "--mode", "record", "--renderer", "nyx",
                       "--out-root", str(root), "--failure-root", str(tmp_path / "rejected")])
    return namespace, calls, root / "seed_17"


def test_collector_saves_resolved_instruction_and_validates_before_skip(monkeypatch, tmp_path):
    module, calls, path = collector_namespace(monkeypatch, tmp_path)
    assert module["main"]() == 0
    assert calls == ["reset", "play"]
    assert json.loads((path / "meta.json").read_text())["instruction"] == "Take the orange cup."
    assert (path / "validated.json").exists()
    assert module["main"]() == 0
    assert calls == ["reset", "play"]
    # A completed metadata file alone cannot mask a damaged H5 on restart.
    (path / "steps.h5").write_bytes(b"partial write")
    assert module["main"]() == 0
    assert calls == ["reset", "play", "reset", "play"]
    assert (tmp_path / "rejected/take_from_human_easy/nyx/seed_17/steps.h5").exists()
    assert validate_episode(path)["validated"]


@pytest.mark.parametrize("renderer,real_camera", [("rasterizer", True), ("nyx", False)])
def test_collector_rejects_renderer_fallback_before_play(monkeypatch, tmp_path, renderer, real_camera):
    module, calls, path = collector_namespace(monkeypatch, tmp_path,
                                               renderer=renderer, real_camera=real_camera)
    assert module["main"]() == 1
    assert calls == ["reset"]
    rejected = tmp_path / "rejected/take_from_human_easy/nyx/seed_17"
    verdict = json.loads((rejected / "verdict.json").read_text())
    assert verdict["renderer"] == renderer
    assert verdict["renderer_verified"] is False
    assert verdict["infra_failure_reason"] == "renderer_mismatch"
    assert not (path / "validated.json").exists()


def test_later_evaluation_cannot_turn_execution_error_into_accepted_episode(monkeypatch, tmp_path):
    module, calls, path = collector_namespace(monkeypatch, tmp_path, playback_error=True)
    assert module["main"]() == 1
    assert not (path / "validated.json").exists()
    rejected = tmp_path / "rejected/take_from_human_easy/nyx/seed_17"
    assert json.loads((rejected / "verdict.json").read_text())["success"] is False


def test_exact_config_keeps_authored_motion_timing_and_sets_eval_seed(monkeypatch, tmp_path):
    module, calls, path = collector_namespace(monkeypatch, tmp_path)
    config = tmp_path / "exact.yml"
    config.write_text("eval_mode: false\nsetting: sim\naction_substeps: 25\n")
    module["load_config"] = lambda path: pytest.fail("exact config must not load default.yml")
    captured = {}
    original = module["_build_config"]
    def capture(args):
        result = original(args)
        captured.update(result)
        return result
    module["_build_config"] = capture
    monkeypatch.setattr(sys, "argv", sys.argv + ["--exact-config", "--config", str(config)])
    assert module["main"]() == 0
    assert "avatar" not in captured
    assert captured["eval_seed"] == 17
    assert captured["eval_mode"] is False
    assert captured["renderer"] == "nyx"
