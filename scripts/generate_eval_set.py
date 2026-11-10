#!/usr/bin/env python3
"""Generate initialization-only eval-set manifests.

The eval set intentionally stores task initialization specs only: task name,
seed, category, and deterministic environment defaults. It does not store
rollouts, observations, actions, videos, or success labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "genesis_hr_bench.eval_set.v1"
DEFAULT_NAME = "eval_custom"

DEFAULTS: dict[str, Any] = {
    "renderer": "rasterizer",
    "setting": "sim",
    "action_type": "ee",
    "action_substeps": 25,
    "max_steps": 300,
    "config_overrides": {
        "eval_mode": True,
        "randomize_avatar": False,
        "randomize_table": False,
        "track_avatar_collision": True,
        "save_video": False,
        "vla_recording": {"enabled": False},
        "neutral_avatar_strict_layout": True,
        "neutral_avatar_region_candidates": 32,
    },
}

BASE_EPISODES: list[dict[str, Any]] = [
    {"category": "cueing", "task": "take_from_human_easy", "seed": 1},
    {"category": "cueing", "task": "deliver_to_human", "seed": 135},
    {"category": "cueing", "task": "take_from_human_safety", "seed": 0},
    {"category": "cueing", "task": "feed_documents", "seed": 1},
    {"category": "cueing", "task": "pour_water", "seed": 0},
    {"category": "cueing", "task": "oil_bottle_recovery", "seed": 38},
    {"category": "cueing", "task": "open_microwave", "seed": 30},
    {"category": "cueing", "task": "factory_inspect_pack", "seed": 1},
    {"category": "contribution", "task": "categorize_objects_contribution", "seed": 21},
    {"category": "contribution", "task": "put_object_in_cabinet_contribution", "seed": 5},
    {"category": "contribution", "task": "stack_three_bowls_contribution", "seed": 3},
    {"category": "contribution", "task": "place_bread_in_basket_contribution", "seed": 9},
    {"category": "contribution", "task": "dump_bin_contribution", "seed": 0},
    {"category": "contribution", "task": "place_burger_and_fries_contribution", "seed": 9},
    {"category": "contribution", "task": "place_pair_of_shoes_contribution", "seed": 0},
    {"category": "contribution", "task": "place_food_in_skillet_contribution", "seed": 8},
    {"category": "contribution", "task": "rank_blocks_by_color_contribution", "seed": 2},
    {"category": "contribution", "task": "rank_blocks_by_size_contribution", "seed": 10},
    {"category": "interruption", "task": "categorize_objects_interruption", "seed": 2},
    {"category": "interruption", "task": "put_object_in_cabinet_interruption", "seed": 2},
    {"category": "interruption", "task": "stack_three_bowls_interruption", "seed": 5},
    {"category": "interruption", "task": "place_bread_in_basket_interruption", "seed": 0},
    {"category": "interruption", "task": "dump_bin_interruption", "seed": 7},
    {"category": "interruption", "task": "place_burger_and_fries_interruption", "seed": 1},
    {"category": "interruption", "task": "place_pair_of_shoes_interruption", "seed": 1},
    {"category": "interruption", "task": "place_food_in_skillet_interruption", "seed": 2},
    {"category": "interruption", "task": "rank_blocks_by_color_interruption", "seed": 9},
    {"category": "interruption", "task": "rank_blocks_by_size_interruption", "seed": 5},
    {"category": "copresence", "task": "categorize_objects_copresence", "seed": 0},
    {"category": "copresence", "task": "put_object_in_cabinet_copresence", "seed": 11},
    {"category": "copresence", "task": "stack_three_bowls_copresence", "seed": 1},
    {"category": "copresence", "task": "place_bread_in_basket_copresence", "seed": 15},
    {"category": "copresence", "task": "dump_bin_copresence", "seed": 5},
    {"category": "copresence", "task": "place_burger_and_fries_copresence", "seed": 28},
    {"category": "copresence", "task": "place_pair_of_shoes_copresence", "seed": 4},
    {"category": "copresence", "task": "place_food_in_skillet_copresence", "seed": 38},
    {"category": "copresence", "task": "rank_blocks_by_color_copresence", "seed": 20},
    {"category": "copresence", "task": "rank_blocks_by_size_copresence", "seed": 61},
]


def episode_id(eval_set_name: str, episode: dict[str, Any]) -> str:
    if "episode_index" in episode:
        return (
            f"{eval_set_name}__{episode['category']}__{episode['task']}"
            f"__ep_{episode['episode_index']:03d}__seed_{episode['seed']}"
        )
    return f"{eval_set_name}__{episode['category']}__{episode['task']}__seed_{episode['seed']}"


def expand_episodes(episodes_per_task: int, seed_offset: int) -> list[dict[str, Any]]:
    if episodes_per_task == 1 and seed_offset == 0:
        return [dict(episode) for episode in BASE_EPISODES]

    episodes = []
    for base in BASE_EPISODES:
        for episode_index in range(episodes_per_task):
            episode = dict(base)
            episode["episode_index"] = episode_index
            episode["seed"] = seed_offset + episode_index
            episodes.append(episode)
    return episodes


def with_ids(
    eval_set_name: str,
    episodes_per_task: int,
    seed_offset: int,
) -> list[dict[str, Any]]:
    return [
        {"id": episode_id(eval_set_name, episode), **episode}
        for episode in expand_episodes(episodes_per_task, seed_offset)
    ]


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def dump_yaml_mapping(lines: list[str], mapping: dict[str, Any], indent: int) -> None:
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            dump_yaml_mapping(lines, value, indent + 2)
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")


def build_yaml(eval_set_name: str, episodes_per_task: int, seed_offset: int) -> str:
    description = (
        "One deterministic initialization per task in the curated benchmark set, "
        "for quick policy evaluation smoke tests."
        if episodes_per_task == 1
        else (
            f"{episodes_per_task} deterministic initializations per task in the "
            "curated benchmark set."
        )
    )
    lines: list[str] = [
        f"schema_version: {SCHEMA_VERSION}",
        f"name: {eval_set_name}",
        f"description: {description}",
        "contains_rollouts: false",
        f"episodes_per_task: {episodes_per_task}",
        f"seed_offset: {seed_offset}",
        "defaults:",
    ]
    dump_yaml_mapping(lines, DEFAULTS, indent=2)
    lines.append("episodes:")
    for episode in with_ids(eval_set_name, episodes_per_task, seed_offset):
        lines.extend(
            [
                f"  - id: {episode['id']}",
                f"    category: {episode['category']}",
                f"    task: {episode['task']}",
                f"    seed: {episode['seed']}",
            ]
        )
        if "episode_index" in episode:
            lines.append(f"    episode_index: {episode['episode_index']}")
    return "\n".join(lines) + "\n"


def build_jsonl(eval_set_name: str, episodes_per_task: int, seed_offset: int) -> str:
    records = []
    for episode in with_ids(eval_set_name, episodes_per_task, seed_offset):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_set": eval_set_name,
                **episode,
                **DEFAULTS,
            }
        )
    return "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


def build_readme(eval_set_name: str, episodes_per_task: int) -> str:
    spec_noun = "spec" if episodes_per_task == 1 else "specs"
    return f"""# {eval_set_name}

This eval set stores deterministic episode initializations only. It does not
contain expert rollouts, videos, actions, observations, or success labels.

It contains {episodes_per_task} initialization {spec_noun} for each of
{len(BASE_EPISODES)} curated benchmark task names. This versioned list is not
automatically expanded when the runtime task registry changes.

Each episode is defined by:

- `task`: registered task name resolved by `envs.tasks.resolve_task_class`
- `seed`: seed passed to `task.reset(seed=seed)`
- `category`: high-level grouping for reporting
- defaults plus any episode-local fields: deterministic environment knobs that
  should be applied during inference

The policy under evaluation should run from these initial states and produce
its own trajectory. Any reference videos outside this directory are visual
debug artifacts, not part of the eval set.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME, help="Eval set name.")
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Number of initialization specs to generate for each task.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="First seed to use when --episodes-per-task is greater than 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write README.md, manifest.yaml, and manifest.jsonl "
            "(default: eval_sets/<name>)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow generated files to be written into a nonempty output directory.",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("eval_sets") / args.name
    return args


def prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    """Create ``output_dir`` or reject an unsafe overwrite.

    Eval-set directories are release artifacts, so a plain generator invocation
    must never silently replace one. ``--force`` permits updating the generated
    files in an existing directory but deliberately leaves unrelated files alone.
    """

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose a new directory or pass --force to overwrite generated files."
            )
        return
    output_dir.mkdir(parents=True)


def write_eval_set(
    eval_set_name: str,
    episodes_per_task: int,
    seed_offset: int,
    output_dir: Path,
    *,
    force: bool = False,
) -> int:
    """Write one eval set and return its episode count."""

    if episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    prepare_output_dir(output_dir, force=force)
    episodes = expand_episodes(episodes_per_task, seed_offset)
    (output_dir / "README.md").write_text(
        build_readme(eval_set_name, episodes_per_task),
        encoding="utf-8",
    )
    (output_dir / "manifest.yaml").write_text(
        build_yaml(eval_set_name, episodes_per_task, seed_offset),
        encoding="utf-8",
    )
    (output_dir / "manifest.jsonl").write_text(
        build_jsonl(eval_set_name, episodes_per_task, seed_offset),
        encoding="utf-8",
    )
    return len(episodes)


def main() -> None:
    args = parse_args()
    episode_count = write_eval_set(
        args.name,
        args.episodes_per_task,
        args.seed_offset,
        args.output_dir,
        force=args.force,
    )
    print(f"Wrote {episode_count} initialization specs to {args.output_dir}")


if __name__ == "__main__":
    main()
