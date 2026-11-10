"""Collection acceptance must reflect actual playback and avatar selection."""
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.vla_data.validate_nyx import validate_episode
from test_nyx_collection_validation import collector_namespace, write_episode


def test_failed_playback_is_not_rescued_by_successful_evaluation(monkeypatch, tmp_path):
    module, calls, path = collector_namespace(monkeypatch, tmp_path, playback_success=False)
    assert module["main"]() == 1
    assert calls == ["reset", "play"]
    verdict = json.loads((tmp_path / "rejected/take_from_human_easy/nyx/seed_17/verdict.json").read_text())
    assert verdict["success"] is False
    assert verdict["play_success"] is False
    assert verdict["evaluate"]["success"] is True
    assert not (path / "validated.json").exists()


def test_collector_rejects_requested_but_unverified_avatar(monkeypatch, tmp_path):
    module, calls, path = collector_namespace(monkeypatch, tmp_path)
    module["load_config"] = lambda path: {"renderer": "nyx", "randomize_avatar": True}
    assert module["main"]() == 1
    assert calls == ["reset"]
    assert not (path / "validated.json").exists()


def randomized_episode(tmp_path):
    avatar = dict(randomize_avatar=True, avatar_randomized=True,
                  avatar_initialized=True, avatar_model="avatars/models/test.glb",
                  collection_contract_version=2, play_success=True)
    return write_episode(tmp_path / "seed_17", verdict_overrides=avatar,
                         meta_overrides={**avatar, "config_snapshot": {"randomize_avatar": True}})


def test_new_recording_marker_confirms_avatar_and_playback(tmp_path):
    result = validate_episode(randomized_episode(tmp_path))
    assert result["avatar_model"] == "avatars/models/test.glb"
    assert result["avatar_randomized"] is True
    assert result["play_success"] is True


@pytest.mark.parametrize("name", ["meta", "verdict"])
@pytest.mark.parametrize("field,value", [
    ("play_success", False), ("play_success", None),
    ("avatar_model", None), ("avatar_model", ""),
    ("avatar_randomized", False), ("avatar_initialized", False),
    ("randomize_avatar", False),
])
def test_rejects_missing_or_failed_new_provenance(tmp_path, name, field, value):
    episode = randomized_episode(tmp_path)
    path = episode / f"{name}.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_episode(episode)


def test_rejects_avatar_identity_mismatch(tmp_path):
    episode = randomized_episode(tmp_path)
    path = episode / "verdict.json"
    record = json.loads(path.read_text())
    record["avatar_model"] = "avatars/models/different.glb"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="avatar_model mismatch"):
        validate_episode(episode)


def test_fixed_avatar_legacy_recordings_remain_readable(tmp_path):
    assert validate_episode(write_episode(tmp_path / "seed_17"))["validated"]


@pytest.mark.parametrize("randomize", [False, True])
def test_skin_resolver_records_actual_selection_without_changing_rng_calls(randomize):
    source = Path(__file__).parents[1] / "envs/base_task_parts/scene.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_resolve_avatar_skin")
    namespace = {"os": os, "DEFAULT_AVATAR_POOL": ["chosen.glb"]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    calls = []
    def rng(label):
        calls.append(label)
        return SimpleNamespace(choice=lambda pool: pool[0])
    task = SimpleNamespace(config={"randomize_avatar": randomize}, AVATAR_POOL=None,
                           DEFAULT_AVATAR_SKIN={"glb_path": "fixed.glb"}, episode_rng=rng)
    result = namespace["_resolve_avatar_skin"](task)
    assert result["glb_path"] == ("chosen.glb" if randomize else "fixed.glb")
    assert task._vla_avatar_model == result["glb_path"]
    assert task._vla_avatar_randomized is randomize
    assert calls == (["avatar_skin"] if randomize else [])


def _resume_collector(monkeypatch, tmp_path, *, randomize=False, mode="record"):
    module, calls, path = collector_namespace(monkeypatch, tmp_path)
    module["load_config"] = lambda unused: {
        "renderer": "nyx", "randomize_avatar": randomize,
        # The fake recorder requires its directory even in verdict-only tests.
        "vla_recording": {"out_dir": str(path)},
    }
    task_name, parent = module["resolve_task_class"]("take_from_human_easy")
    class Task(parent):
        def __init__(self, config):
            super().__init__(config)
            self.avatar = SimpleNamespace()
            self._vla_avatar_randomized = config.get("randomize_avatar") is True
            self._vla_avatar_model = "random.glb" if self._vla_avatar_randomized else "fixed.glb"
    module["resolve_task_class"] = lambda *args, **kwargs: (task_name, Task)
    argv = list(sys.argv)
    argv[argv.index("--mode") + 1] = mode
    monkeypatch.setattr(sys, "argv", argv)
    return module, calls, path


def _reuse_fields(randomize=False):
    return dict(collection_contract_version=2, play_success=True,
                randomize_avatar=randomize, avatar_randomized=randomize,
                avatar_initialized=True,
                avatar_model="random.glb" if randomize else "fixed.glb")


def _write_reusable_record(path, randomize=False):
    fields = _reuse_fields(randomize)
    return write_episode(path, verdict_overrides=fields,
                         meta_overrides={**fields, "config_snapshot": {"randomize_avatar": randomize}})


@pytest.mark.parametrize("randomize", [False, True])
def test_matching_current_record_is_reused_without_play(monkeypatch, tmp_path, randomize):
    module, calls, path = _resume_collector(monkeypatch, tmp_path, randomize=randomize)
    _write_reusable_record(path, randomize)
    assert module["main"]() == 0
    assert calls == []


def test_legacy_record_remains_readable_but_is_quarantined_before_new_collection(monkeypatch, tmp_path):
    module, calls, path = _resume_collector(monkeypatch, tmp_path)
    write_episode(path)
    assert validate_episode(path, write_marker=True)["validated"] is True
    assert module["main"]() == 0
    assert calls == ["reset", "play"]
    rejected = tmp_path / "rejected/take_from_human_easy/nyx/seed_17"
    assert (rejected / "steps.h5").exists()
    assert "play_success" not in json.loads((rejected / "meta.json").read_text())
    assert json.loads((path / "meta.json").read_text())["play_success"] is True


@pytest.mark.parametrize("randomize", [False, True])
def test_record_from_other_randomization_setting_is_quarantined_and_recollected(monkeypatch, tmp_path, randomize):
    module, calls, path = _resume_collector(monkeypatch, tmp_path, randomize=randomize)
    _write_reusable_record(path, not randomize)
    assert module["main"]() == 0
    assert calls == ["reset", "play"]
    rejected = tmp_path / "rejected/take_from_human_easy/nyx/seed_17/meta.json"
    assert json.loads(rejected.read_text())["randomize_avatar"] is not randomize
    assert json.loads((path / "meta.json").read_text())["randomize_avatar"] is randomize


@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("play_success", [False, True])
def test_current_completed_verdict_is_reused_including_legitimate_task_failure(monkeypatch, tmp_path,
                                                                            randomize, play_success):
    module, calls, path = _resume_collector(monkeypatch, tmp_path, randomize=randomize, mode="verdict")
    fields = dict(_reuse_fields(randomize), mode="verdict", play_success=play_success,
                  success=play_success, plan_success=play_success)
    write_episode(path, verdict_overrides=fields)
    assert module["main"]() == 0
    assert calls == []


@pytest.mark.parametrize("mutation", ["legacy", "missing_playback", "failed_playback_pass",
                                      "other_randomization", "unverified_avatar"])
def test_incompatible_verdict_is_preserved_and_recomputed(monkeypatch, tmp_path, mutation):
    module, calls, path = _resume_collector(monkeypatch, tmp_path, randomize=True, mode="verdict")
    write_episode(path, verdict_overrides={**_reuse_fields(True), "mode": "verdict"})
    prior_path = path / "verdict.json"
    prior = json.loads(prior_path.read_text())
    if mutation == "legacy":
        prior.pop("collection_contract_version")
    elif mutation == "missing_playback":
        prior.pop("play_success")
    elif mutation == "failed_playback_pass":
        prior["play_success"] = False  # Cannot justify an old success=True verdict.
    elif mutation == "other_randomization":
        prior["randomize_avatar"] = False
        prior["avatar_randomized"] = False
    else:
        prior["avatar_model"] = None
    prior_path.write_text(json.dumps(prior))
    assert module["main"]() == 0
    assert calls == ["reset", "play"]
    rejected = tmp_path / "rejected/take_from_human_easy/nyx/seed_17/verdict.json"
    assert json.loads(rejected.read_text()) == prior
    assert json.loads(prior_path.read_text())["collection_contract_version"] == 2
