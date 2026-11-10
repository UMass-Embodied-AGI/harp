"""Offline contract helpers retain the collector's existing H5 acceptance rules."""

import builtins
import runpy

import h5py
import numpy as np
import pytest

from scripts.vla_data import collection_contract
from scripts.vla_data.collection_contract import validate_recorded_h5
from test_nyx_collection_validation import collector_namespace, write_episode


def test_contract_import_does_not_load_simulator_or_recording_dependencies(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"envs", "genesis", "h5py", "numpy"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runpy.run_path(collection_contract.__file__)


def test_collector_reexports_contract_helpers(monkeypatch, tmp_path):
    namespace, _, _ = collector_namespace(monkeypatch, tmp_path)
    for name in (
        "REQUIRED_ACT_H5_KEYS", "_json_default", "_neutral_meta",
        "_collection_avatar_meta", "_success_from_evaluation",
        "_require_reusable_collection_contract", "validate_recorded_h5",
    ):
        assert namespace[name] is getattr(collection_contract, name)


def test_recorded_h5_returns_existing_training_summary(tmp_path):
    path = write_episode(tmp_path / "seed_17")
    result = validate_recorded_h5(path)
    with h5py.File(path / "steps.h5", "r") as stream:
        keys = sorted(stream.keys())
    assert result == {
        "path": str(path / "steps.h5"),
        "n_steps": 5,
        "schema_version": "vla_steps_v2_qpos_target",
        "keys": keys,
        "cams": ["head_camera", "right_wrist"],
        "invalid_qpos_target_actions": 0,
    }


@pytest.mark.parametrize("mutation,error,match", [
    ("missing_key", KeyError, "missing required VLA keys"),
    ("empty_rows", ValueError, "no recorded training rows"),
    ("mismatched_rows", ValueError, "does not match qpos length"),
    ("no_cameras", ValueError, "no image camera datasets"),
    ("invalid_targets", ValueError, "invalid qpos_target_action labels"),
])
def test_recorded_h5_rejects_incomplete_training_labels(tmp_path, mutation, error, match):
    path = write_episode(tmp_path / "seed_17")
    with h5py.File(path / "steps.h5", "a") as stream:
        if mutation == "missing_key":
            del stream["qpos_target_action"]
        elif mutation == "empty_rows":
            del stream["qpos"]
            stream["qpos"] = np.empty((0, 7))
        elif mutation == "mismatched_rows":
            del stream["t"]
            stream["t"] = np.arange(4)
        elif mutation == "no_cameras":
            del stream["images"]
            stream.create_group("images")
        else:
            stream["qpos_target_action_valid"][2] = False
    with pytest.raises(error, match=match):
        validate_recorded_h5(path)


def test_recorded_h5_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing recorded H5"):
        validate_recorded_h5(tmp_path)


def test_recorded_h5_legacy_optional_mask_and_schema_remain_readable(tmp_path):
    path = write_episode(tmp_path / "seed_17")
    with h5py.File(path / "steps.h5", "a") as stream:
        del stream["qpos_target_action_valid"]
        del stream.attrs["schema_version"]
    result = validate_recorded_h5(path)
    assert result["invalid_qpos_target_actions"] is None
    assert result["schema_version"] == "unknown"


@pytest.mark.parametrize("renderer,validator", [
    ("nyx", "validate_episode"), ("rasterizer", "validate_recorded_h5"),
])
def test_resume_uses_validator_from_collector_namespace(monkeypatch, tmp_path, renderer, validator):
    namespace, _, path = collector_namespace(monkeypatch, tmp_path)
    fields = dict(collection_contract_version=2, play_success=True,
                  randomize_avatar=False, avatar_randomized=False, renderer=renderer)
    write_episode(path, meta_overrides=fields, verdict_overrides=fields)
    calls = []
    result = object()

    def validate(directory, **kwargs):
        calls.append((directory, kwargs))
        return result

    namespace[validator] = validate
    assert namespace["_existing_record_is_complete"](
        path, "take_from_human_easy", 17, renderer, {},
    ) is result
    expected_kwargs = dict(expected_task="take_from_human_easy", expected_seed=17,
                           write_marker=True) if renderer == "nyx" else {}
    assert calls == [(path, expected_kwargs)]
