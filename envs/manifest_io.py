"""Versioned eval-set and scene-init manifest loading.

This module intentionally has no Genesis dependency.  Manifest consumers need
to reject incompatible data before importing or constructing a simulator task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


EVAL_SET_SCHEMA_VERSION = "genesis_hr_bench.eval_set.v1"
SCENE_INIT_SCHEMA_VERSION = "genesis_hr_bench.scene_init.v1"
SCENE_INIT_SET_SCHEMA_VERSION = "genesis_hr_bench.scene_init_set.v1"
EVAL_SET_REQUIRED_KEYS = (
    "schema_version",
    "eval_set",
    "id",
    "category",
    "task",
    "seed",
    "renderer",
    "setting",
    "action_type",
    "action_substeps",
    "max_steps",
    "config_overrides",
)
SCENE_INIT_REQUIRED_KEYS = (
    "schema_version",
    "captured_at_utc",
    "table",
    "entities",
    "task_attrs",
)
SCENE_INIT_SET_REQUIRED_KEYS = (
    "schema_version",
    "scene_init_schema_version",
    "generated_at_utc",
    "source_manifest",
    "episodes",
)


class ManifestSchemaError(ValueError):
    """Raised when a manifest schema or versioned record shape is unsupported."""


def _display_version(value: Any) -> str:
    return "<missing>" if value is None else repr(value)


def _require_mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestSchemaError(
            f"{location}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, *, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestSchemaError(
            f"{location}: expected a list, got {type(value).__name__}"
        )
    return value


def _require_keys(
    value: dict[str, Any], required: tuple[str, ...], *, location: str
) -> None:
    missing = [key for key in required if key not in value]
    if missing:
        raise ManifestSchemaError(
            f"{location}: missing required keys: {', '.join(missing)}"
        )


def _require_nonempty_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestSchemaError(f"{location}: expected a non-empty string")
    return value


def _require_integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestSchemaError(f"{location}: expected an integer")
    return value


def _require_schema(
    value: dict[str, Any],
    expected: str,
    *,
    location: str,
    field: str = "schema_version",
) -> None:
    observed = value.get(field)
    if observed != expected:
        raise ManifestSchemaError(
            f"{location}: unsupported or missing {field}: "
            f"expected {expected!r}, got {_display_version(observed)}"
        )


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    if not resolved.is_dir():
        return resolved
    for name in ("manifest.jsonl", "manifest.yaml", "manifest.yml"):
        candidate = resolved / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No manifest.jsonl, manifest.yaml, or manifest.yml in {resolved}"
    )


def _validate_eval_set_row(
    row: dict[str, Any], *, location: str
) -> dict[str, Any]:
    _require_schema(row, EVAL_SET_SCHEMA_VERSION, location=location)
    _require_keys(row, EVAL_SET_REQUIRED_KEYS, location=location)
    for key in (
        "eval_set",
        "id",
        "category",
        "task",
        "renderer",
        "setting",
        "action_type",
    ):
        _require_nonempty_string(row[key], location=f"{location}.{key}")
    for key in ("seed", "action_substeps", "max_steps"):
        _require_integer(row[key], location=f"{location}.{key}")
    _require_mapping(
        row["config_overrides"], location=f"{location}.config_overrides"
    )
    return row


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            rows.append(
                _require_mapping(value, location=f"{path}:line {line_number}")
            )
    if not rows:
        raise ManifestSchemaError(f"{path}: manifest contains no versioned records")
    return rows


def _load_document(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if path.suffix.lower() == ".json":
            return json.load(handle)
    raise ValueError(
        f"Unsupported manifest extension {path.suffix!r}; expected "
        ".jsonl, .json, .yaml, or .yml"
    )


def load_eval_set_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load eval-set v1 rows, normalizing document manifests to JSONL shape.

    JSONL rows are returned unchanged.  YAML/JSON episodes receive the
    document schema and name, then the episode fields, then the top-level
    defaults.  This is the same shallow precedence used by the canonical
    eval-set generator.
    """
    manifest_path = _manifest_path(path)
    if manifest_path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(manifest_path)
        for index, row in enumerate(rows, start=1):
            _validate_eval_set_row(
                row,
                location=f"{manifest_path}:record {index}",
            )
        return rows

    data = _require_mapping(
        _load_document(manifest_path), location=str(manifest_path)
    )
    _require_schema(data, EVAL_SET_SCHEMA_VERSION, location=str(manifest_path))
    defaults = _require_mapping(
        data.get("defaults", {}), location=f"{manifest_path}:defaults"
    )
    episodes = _require_list(
        data.get("episodes"), location=f"{manifest_path}:episodes"
    )
    if not episodes:
        raise ManifestSchemaError(f"{manifest_path}: manifest contains no episodes")
    validated = []
    for index, row in enumerate(episodes):
        episode = _require_mapping(
            row, location=f"{manifest_path}:episodes[{index}]"
        )
        if "schema_version" in episode:
            _require_schema(
                episode,
                EVAL_SET_SCHEMA_VERSION,
                location=f"{manifest_path}:episodes[{index}]",
            )
        normalized = {
            "schema_version": data["schema_version"],
            "eval_set": data.get("name"),
            **episode,
            **defaults,
        }
        validated.append(
            _validate_eval_set_row(
                normalized,
                location=f"{manifest_path}:episodes[{index}]",
            )
        )
    return validated


def _validate_scene_init_entry(
    entry: Any,
    *,
    location: str,
    allow_failures: bool = False,
) -> dict[str, Any]:
    record = _require_mapping(entry, location=location)
    if "init_error" in record:
        if "init" in record:
            raise ManifestSchemaError(
                f"{location}: scene-init record cannot contain both init and init_error"
            )
        error = record["init_error"]
        if not allow_failures:
            raise ManifestSchemaError(
                f"{location}: scene-init generation failed: {error!r}"
            )
        if not isinstance(error, str) or not error.strip():
            raise ManifestSchemaError(
                f"{location}: init_error must be a non-empty string"
            )
        return record
    if "init" in record:
        init = _require_mapping(record["init"], location=f"{location}.init")
        _require_schema(init, SCENE_INIT_SCHEMA_VERSION, location=f"{location}.init")
        _validate_scene_init_payload(init, location=f"{location}.init")
        return record
    _require_schema(record, SCENE_INIT_SCHEMA_VERSION, location=location)
    _validate_scene_init_payload(record, location=location)
    return record


def _validate_scene_init_payload(
    value: dict[str, Any], *, location: str
) -> None:
    _require_keys(value, SCENE_INIT_REQUIRED_KEYS, location=location)
    _require_nonempty_string(
        value["captured_at_utc"], location=f"{location}.captured_at_utc"
    )
    for key in ("table", "entities", "task_attrs"):
        _require_mapping(value[key], location=f"{location}.{key}")


def load_scene_init_manifest(
    path: str | os.PathLike[str],
    *,
    allow_failures: bool = False,
) -> list[dict[str, Any]]:
    """Load scene-init v1 records, rejecting missing or unknown schemas.

    A JSONL file stores episode wrappers directly, so each row is validated by
    its nested ``init.schema_version`` (or as a bare scene-init record).  A
    YAML/JSON scene-init set additionally declares both the set and nested
    scene-init schema versions.
    """
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(manifest_path)
        return [
            _validate_scene_init_entry(
                row,
                location=f"{manifest_path}:record {index}",
                allow_failures=allow_failures,
            )
            for index, row in enumerate(rows, start=1)
        ]

    data = _load_document(manifest_path)
    if isinstance(data, list):
        if not data:
            raise ManifestSchemaError(
                f"{manifest_path}: manifest contains no versioned records"
            )
        return [
            _validate_scene_init_entry(
                row,
                location=f"{manifest_path}:episodes[{index}]",
                allow_failures=allow_failures,
            )
            for index, row in enumerate(data)
        ]

    record = _require_mapping(data, location=str(manifest_path))
    is_scene_init_set = (
        record.get("schema_version") == SCENE_INIT_SET_SCHEMA_VERSION
        or "scene_init_schema_version" in record
        or "episodes" in record
    )
    if not is_scene_init_set:
        return [
            _validate_scene_init_entry(
                record,
                location=str(manifest_path),
                allow_failures=allow_failures,
            )
        ]

    _require_schema(
        record, SCENE_INIT_SET_SCHEMA_VERSION, location=str(manifest_path)
    )
    _require_schema(
        record,
        SCENE_INIT_SCHEMA_VERSION,
        location=str(manifest_path),
        field="scene_init_schema_version",
    )
    _require_keys(
        record, SCENE_INIT_SET_REQUIRED_KEYS, location=str(manifest_path)
    )
    _require_nonempty_string(
        record["generated_at_utc"],
        location=f"{manifest_path}.generated_at_utc",
    )
    source_manifest = record["source_manifest"]
    if source_manifest is not None:
        _require_nonempty_string(
            source_manifest, location=f"{manifest_path}.source_manifest"
        )
    episodes = _require_list(
        record.get("episodes"), location=f"{manifest_path}:episodes"
    )
    return [
        _validate_scene_init_entry(
            row,
            location=f"{manifest_path}:episodes[{index}]",
            allow_failures=allow_failures,
        )
        for index, row in enumerate(episodes)
    ]
