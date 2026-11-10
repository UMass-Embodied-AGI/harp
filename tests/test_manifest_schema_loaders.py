from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from envs.manifest_io import (
    EVAL_SET_REQUIRED_KEYS,
    EVAL_SET_SCHEMA_VERSION,
    SCENE_INIT_REQUIRED_KEYS,
    SCENE_INIT_SCHEMA_VERSION,
    SCENE_INIT_SET_REQUIRED_KEYS,
    SCENE_INIT_SET_SCHEMA_VERSION,
    ManifestSchemaError,
    load_eval_set_manifest,
    load_scene_init_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def eval_episode() -> dict:
    return {
        "id": "eval_test__intent__pour_water__seed_3",
        "category": "intent",
        "task": "pour_water",
        "seed": 3,
        "renderer": "rasterizer",
        "setting": "sim",
        "action_type": "ee",
        "action_substeps": 25,
        "max_steps": 300,
        "config_overrides": {},
    }


@pytest.fixture
def eval_row(eval_episode: dict) -> dict:
    return {
        "schema_version": EVAL_SET_SCHEMA_VERSION,
        "eval_set": "eval_test",
        **eval_episode,
    }


@pytest.fixture
def scene_init() -> dict:
    return {
        "schema_version": SCENE_INIT_SCHEMA_VERSION,
        "captured_at_utc": "2026-08-13T00:00:00+00:00",
        "table": {},
        "entities": {},
        "task_attrs": {},
    }


def _without_key(value: dict, key: str) -> dict:
    result = deepcopy(value)
    result.pop(key, None)
    return result


def _scene_init_set(episodes: list[dict]) -> dict:
    return {
        "schema_version": SCENE_INIT_SET_SCHEMA_VERSION,
        "scene_init_schema_version": SCENE_INIT_SCHEMA_VERSION,
        "generated_at_utc": "2026-08-13T00:00:00+00:00",
        "source_manifest": "eval_sets/eval_test/manifest.jsonl",
        "episodes": episodes,
    }


def test_loader_module_import_does_not_require_genesis():
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "genesis" or name.startswith("genesis."):
        raise AssertionError("manifest loading imported Genesis")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from envs.manifest_io import EVAL_SET_SCHEMA_VERSION
assert EVAL_SET_SCHEMA_VERSION == "genesis_hr_bench.eval_set.v1"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_eval_jsonl_v1_loads_rows_unchanged(tmp_path, eval_row):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    assert load_eval_set_manifest(path) == [eval_row]
    assert load_eval_set_manifest(tmp_path) == [eval_row]


def test_eval_yaml_v1_normalizes_to_jsonl_row_shape(tmp_path, eval_episode):
    document = {
        "schema_version": EVAL_SET_SCHEMA_VERSION,
        "name": "eval_test",
        "defaults": {"renderer": "rasterizer"},
        "episodes": [eval_episode],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    expected = {
        "schema_version": EVAL_SET_SCHEMA_VERSION,
        "eval_set": "eval_test",
        **eval_episode,
        "renderer": "rasterizer",
    }
    assert load_eval_set_manifest(path) == [expected]
    assert load_eval_set_manifest(tmp_path) == [expected]


def test_eval_yaml_normalization_matches_jsonl_collision_precedence(
    tmp_path, eval_episode
):
    episode = {
        **eval_episode,
        "eval_set": "episode_eval_set",
        "renderer": "episode_renderer",
        "config_overrides": {"episode_only": True},
    }
    defaults = {
        "renderer": "default_renderer",
        "config_overrides": {"default_only": True},
    }
    document = {
        "schema_version": EVAL_SET_SCHEMA_VERSION,
        "name": "document_eval_set",
        "defaults": defaults,
        "episodes": [episode],
    }
    yaml_path = tmp_path / "eval.yaml"
    jsonl_path = tmp_path / "eval.jsonl"
    expected = {
        "schema_version": EVAL_SET_SCHEMA_VERSION,
        "eval_set": document["name"],
        **episode,
        **defaults,
    }
    yaml_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    jsonl_path.write_text(json.dumps(expected) + "\n", encoding="utf-8")

    assert expected["eval_set"] == "episode_eval_set"
    assert expected["renderer"] == "default_renderer"
    assert expected["config_overrides"] == {"default_only": True}
    assert load_eval_set_manifest(yaml_path) == load_eval_set_manifest(jsonl_path)


@pytest.mark.parametrize("observed", [None, "genesis_hr_bench.eval_set.v2"])
def test_eval_jsonl_rejects_missing_and_unknown_schemas(
    tmp_path, eval_row, observed
):
    row = deepcopy(eval_row)
    if observed is None:
        row.pop("schema_version")
    else:
        row["schema_version"] = observed
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(
        ManifestSchemaError, match="unsupported or missing schema_version"
    ):
        load_eval_set_manifest(path)


@pytest.mark.parametrize("observed", [None, "genesis_hr_bench.eval_set.v2"])
def test_eval_document_rejects_missing_and_unknown_schemas(
    tmp_path, eval_episode, observed
):
    document = {"episodes": [eval_episode]}
    if observed is not None:
        document["schema_version"] = observed
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(
        ManifestSchemaError, match="unsupported or missing schema_version"
    ):
        load_eval_set_manifest(path)


def test_eval_jsonl_rejects_empty_and_non_mapping_records(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ManifestSchemaError, match="no versioned records"):
        load_eval_set_manifest(empty)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ManifestSchemaError, match="expected a mapping"):
        load_eval_set_manifest(invalid)

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text(
        yaml.safe_dump(
            {
                "schema_version": EVAL_SET_SCHEMA_VERSION,
                "name": "eval_empty",
                "defaults": {},
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ManifestSchemaError, match="no episodes"):
        load_eval_set_manifest(empty_yaml)


@pytest.mark.parametrize(
    "missing_key",
    [key for key in EVAL_SET_REQUIRED_KEYS if key != "schema_version"],
)
def test_eval_jsonl_rejects_missing_required_fields(
    tmp_path, eval_row, missing_key
):
    row = _without_key(eval_row, missing_key)
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ManifestSchemaError, match="missing required keys"):
        load_eval_set_manifest(path)


@pytest.mark.parametrize(
    "field,value",
    (
        ("task", ""),
        ("seed", True),
        ("action_substeps", 1.5),
        ("config_overrides", []),
    ),
)
def test_eval_jsonl_rejects_invalid_required_field_types(
    tmp_path, eval_row, field, value
):
    row = {**eval_row, field: value}
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ManifestSchemaError):
        load_eval_set_manifest(path)


def test_scene_init_set_v1_loads_episode_wrappers_unchanged(tmp_path, scene_init):
    episode = {"id": "scene-3", "task": "pour_water", "seed": 3, "init": scene_init}
    document = _scene_init_set([episode])
    path = tmp_path / "scene_init.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    assert load_scene_init_manifest(path) == [episode]


def test_scene_init_jsonl_and_bare_v1_documents_load_unchanged(
    tmp_path, scene_init
):
    episode = {"id": "scene-3", "seed": 3, "init": scene_init}
    jsonl_path = tmp_path / "scene_init.jsonl"
    jsonl_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    assert load_scene_init_manifest(jsonl_path) == [episode]

    json_path = tmp_path / "scene_init.json"
    json_path.write_text(json.dumps(scene_init), encoding="utf-8")
    assert load_scene_init_manifest(json_path) == [scene_init]


@pytest.mark.parametrize(
    "missing_key",
    [key for key in SCENE_INIT_REQUIRED_KEYS if key != "schema_version"],
)
def test_scene_init_rejects_missing_required_payload_fields(
    tmp_path, scene_init, missing_key
):
    path = tmp_path / "scene_init.json"
    path.write_text(
        json.dumps(_without_key(scene_init, missing_key)), encoding="utf-8"
    )

    with pytest.raises(ManifestSchemaError, match="missing required keys"):
        load_scene_init_manifest(path)


@pytest.mark.parametrize(
    ("field", "observed"),
    [
        ("schema_version", None),
        ("schema_version", "genesis_hr_bench.scene_init_set.v2"),
        ("scene_init_schema_version", None),
        ("scene_init_schema_version", "genesis_hr_bench.scene_init.v2"),
    ],
)
def test_scene_init_set_rejects_missing_and_unknown_declared_schemas(
    tmp_path, scene_init, field, observed
):
    document = _scene_init_set([{"id": "scene-3", "init": scene_init}])
    if observed is None:
        document.pop(field)
    else:
        document[field] = observed
    path = tmp_path / "scene_init.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(
        ManifestSchemaError, match=f"unsupported or missing {field}"
    ):
        load_scene_init_manifest(path)


@pytest.mark.parametrize(
    "missing_key",
    [
        key
        for key in SCENE_INIT_SET_REQUIRED_KEYS
        if key not in {"schema_version", "scene_init_schema_version"}
    ],
)
def test_scene_init_set_rejects_missing_required_fields(
    tmp_path, scene_init, missing_key
):
    document = _without_key(
        _scene_init_set([{"id": "scene-3", "init": scene_init}]),
        missing_key,
    )
    path = tmp_path / "scene_init.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ManifestSchemaError, match="missing required keys"):
        load_scene_init_manifest(path)


@pytest.mark.parametrize("observed", [None, "genesis_hr_bench.scene_init.v2"])
def test_scene_init_rejects_missing_and_unknown_nested_schemas(
    tmp_path, scene_init, observed
):
    init = _without_key(scene_init, "schema_version")
    if observed is not None:
        init["schema_version"] = observed
    episode = {"id": "scene-3", "init": init}
    path = tmp_path / "scene_init.jsonl"
    path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    with pytest.raises(
        ManifestSchemaError, match="unsupported or missing schema_version"
    ):
        load_scene_init_manifest(path)


def test_scene_init_rejects_unversioned_episode_but_allows_versioned_empty_set(
    tmp_path,
):
    unversioned = tmp_path / "unversioned.jsonl"
    unversioned.write_text(json.dumps({"id": "scene-3", "seed": 3}) + "\n")
    with pytest.raises(
        ManifestSchemaError, match="unsupported or missing schema_version"
    ):
        load_scene_init_manifest(unversioned)

    empty_set = tmp_path / "empty.yaml"
    empty_set.write_text(
        yaml.safe_dump(
            _scene_init_set([])
        ),
        encoding="utf-8",
    )
    assert load_scene_init_manifest(empty_set) == []


def test_scene_init_generation_failures_are_finalizer_only(tmp_path):
    failure = {
        "id": "scene-failed",
        "task": "pour_water",
        "init_error": "planner could not initialize",
    }
    path = tmp_path / "scene_init.yaml"
    path.write_text(
        yaml.safe_dump(
            _scene_init_set([failure])
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestSchemaError, match="generation failed"):
        load_scene_init_manifest(path)
    assert load_scene_init_manifest(path, allow_failures=True) == [failure]


@pytest.mark.parametrize("init_error", [None, "", "   "])
def test_scene_init_finalizer_rejects_malformed_failure_rows(
    tmp_path, init_error
):
    path = tmp_path / "scene_init.jsonl"
    path.write_text(
        json.dumps({"id": "scene-failed", "init_error": init_error}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestSchemaError, match="non-empty string"):
        load_scene_init_manifest(path, allow_failures=True)


def test_scene_init_failure_row_cannot_also_contain_init(tmp_path, scene_init):
    path = tmp_path / "scene_init.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "scene-failed",
                "init": scene_init,
                "init_error": "partial generation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestSchemaError, match="both init and init_error"):
        load_scene_init_manifest(path, allow_failures=True)
