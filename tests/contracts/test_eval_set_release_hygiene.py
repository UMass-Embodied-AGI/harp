import json
import re
from collections import Counter
from pathlib import Path

import pytest

from scripts import generate_eval_set


REPO = Path(__file__).resolve().parents[2]
EXPECTED_AUTHORED_SEEDS = {
    "take_from_human_easy": 1,
    "deliver_to_human": 135,
    "take_from_human_safety": 0,
    "feed_documents": 1,
    "pour_water": 0,
    "oil_bottle_recovery": 38,
    "open_microwave": 30,
    "factory_inspect_pack": 1,
    "categorize_objects_contribution": 21,
    "put_object_in_cabinet_contribution": 5,
    "stack_three_bowls_contribution": 3,
    "place_bread_in_basket_contribution": 9,
    "dump_bin_contribution": 0,
    "place_burger_and_fries_contribution": 9,
    "place_pair_of_shoes_contribution": 0,
    "place_food_in_skillet_contribution": 8,
    "rank_blocks_by_color_contribution": 2,
    "rank_blocks_by_size_contribution": 10,
    "categorize_objects_interruption": 2,
    "put_object_in_cabinet_interruption": 2,
    "stack_three_bowls_interruption": 5,
    "place_bread_in_basket_interruption": 0,
    "dump_bin_interruption": 7,
    "place_burger_and_fries_interruption": 1,
    "place_pair_of_shoes_interruption": 1,
    "place_food_in_skillet_interruption": 2,
    "rank_blocks_by_color_interruption": 9,
    "rank_blocks_by_size_interruption": 5,
    "categorize_objects_copresence": 0,
    "put_object_in_cabinet_copresence": 11,
    "stack_three_bowls_copresence": 1,
    "place_bread_in_basket_copresence": 15,
    "dump_bin_copresence": 5,
    "place_burger_and_fries_copresence": 28,
    "place_pair_of_shoes_copresence": 4,
    "place_food_in_skillet_copresence": 38,
    "rank_blocks_by_color_copresence": 20,
    "rank_blocks_by_size_copresence": 61,
}


def _manifest_unique_tasks(path: Path) -> set[str]:
    return {
        line.split(":", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("    task:")
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_yaml(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flatten_yaml_manifest(manifest: dict) -> list[dict]:
    return [
        {
            "schema_version": manifest["schema_version"],
            "eval_set": manifest["name"],
            **episode,
            **manifest["defaults"],
        }
        for episode in manifest["episodes"]
    ]


def test_generator_describes_its_curated_subset_truthfully():
    task_names = [episode["task"] for episode in generate_eval_set.BASE_EPISODES]
    assert len(task_names) == len(set(task_names))
    assert len(task_names) == 38

    yaml_text = generate_eval_set.build_yaml("release_test", 1, 0)
    readme = generate_eval_set.build_readme("release_test", 1)
    assert "curated benchmark set" in yaml_text
    assert "38 curated benchmark task names" in readme
    assert "per registered task" not in yaml_text
    assert "per registered task" not in readme


def test_generator_defaults_cannot_target_the_frozen_smoke_set():
    args = generate_eval_set.parse_args([])
    assert args.name == "eval_custom"
    assert args.output_dir == Path("eval_sets/eval_custom")
    assert args.force is False
    assert "eval_v0_smoke" not in str(args.output_dir)
    assert generate_eval_set.parse_args(["--force"]).force is True
    renamed = generate_eval_set.parse_args(["--name", "eval_release_candidate"])
    assert renamed.output_dir == Path("eval_sets/eval_release_candidate")
    explicit = generate_eval_set.parse_args(
        ["--name", "logical_name", "--output-dir", "elsewhere"]
    )
    assert explicit.output_dir == Path("elsewhere")


def test_generator_preserves_authored_smoke_seeds_without_episode_indices():
    episodes = generate_eval_set.expand_episodes(1, 0)
    assert episodes == generate_eval_set.BASE_EPISODES
    assert all("episode_index" not in episode for episode in episodes)
    assert {episode["task"]: episode["seed"] for episode in episodes} == (
        EXPECTED_AUTHORED_SEEDS
    )

    rows = generate_eval_set.with_ids("release_test", 1, 0)
    assert [row["seed"] for row in rows] == [
        episode["seed"] for episode in generate_eval_set.BASE_EPISODES
    ]
    assert len({row["id"] for row in rows}) == len(rows)
    assert all("__ep_" not in row["id"] for row in rows)


def test_generator_expands_each_task_with_offset_seeds_and_unique_ids():
    rows = generate_eval_set.with_ids("release_test", 3, 17)
    assert len(rows) == 3 * len(generate_eval_set.BASE_EPISODES)
    assert len({row["id"] for row in rows}) == len(rows)

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task_rows in by_task.values():
        assert [row["episode_index"] for row in task_rows] == [0, 1, 2]
        assert [row["seed"] for row in task_rows] == [17, 18, 19]
        assert all("__ep_" in row["id"] for row in task_rows)

    offset_singletons = generate_eval_set.with_ids("release_test", 1, 9)
    assert all(row["episode_index"] == 0 for row in offset_singletons)
    assert all(row["seed"] == 9 for row in offset_singletons)


def test_generated_yaml_and_jsonl_are_semantically_equal():
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load(generate_eval_set.build_yaml("release_test", 2, 4))
    records = [
        json.loads(line)
        for line in generate_eval_set.build_jsonl("release_test", 2, 4).splitlines()
    ]
    assert _flatten_yaml_manifest(manifest) == records


def test_generator_readme_uses_grammatical_counts():
    singular = re.sub(r"\s+", " ", generate_eval_set.build_readme("one", 1))
    plural = re.sub(r"\s+", " ", generate_eval_set.build_readme("many", 3))
    task_count = len(generate_eval_set.BASE_EPISODES)
    assert f"1 initialization spec for each of {task_count}" in singular
    assert f"3 initialization specs for each of {task_count}" in plural
    assert "1 initialization specs" not in singular


def test_generator_refuses_nonempty_output_without_force(tmp_path: Path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        generate_eval_set.write_eval_set("release_test", 1, 0, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert not (output_dir / "manifest.jsonl").exists()

    count = generate_eval_set.write_eval_set(
        "release_test", 1, 0, output_dir, force=True
    )
    assert count == len(generate_eval_set.BASE_EPISODES)
    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert len(_read_jsonl(output_dir / "manifest.jsonl")) == count


def test_generator_writes_a_new_or_empty_output_directory(tmp_path: Path):
    new_dir = tmp_path / "new"
    count = generate_eval_set.write_eval_set("release_test", 2, 7, new_dir)
    assert count == 2 * len(generate_eval_set.BASE_EPISODES)
    assert {path.name for path in new_dir.iterdir()} == {
        "README.md",
        "manifest.yaml",
        "manifest.jsonl",
    }

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    generate_eval_set.write_eval_set("empty_test", 1, 0, empty_dir)
    assert (empty_dir / "manifest.jsonl").is_file()


def test_checked_in_eval_sets_state_their_frozen_37_name_scope():
    for name in ("eval_v0_smoke",):
        root = REPO / "eval_sets" / name
        tasks = _manifest_unique_tasks(root / "manifest.yaml")
        assert len(tasks) == 37
        manifest_header = "\n".join(
            (root / "manifest.yaml").read_text(encoding="utf-8").splitlines()[:4]
        )
        assert "frozen 37-name benchmark set" in manifest_header


def test_checked_in_smoke_set_preserves_its_authored_seeds():
    records = _read_jsonl(REPO / "eval_sets/eval_v0_smoke/manifest.jsonl")
    expected = {
        task: seed
        for task, seed in EXPECTED_AUTHORED_SEEDS.items()
        if task != "factory_inspect_pack"
    }
    assert {record["task"]: record["seed"] for record in records} == expected


@pytest.mark.parametrize(
    ("name", "episodes_per_task"),
    [
        ("eval_v0_smoke", 1),
    ],
)
def test_checked_in_eval_set_rows_are_complete_and_unique(
    name: str, episodes_per_task: int
):
    records = _read_jsonl(REPO / "eval_sets" / name / "manifest.jsonl")
    assert len(records) == 37 * episodes_per_task
    assert len({record["id"] for record in records}) == len(records)
    assert Counter(record["task"] for record in records) == {
        task: episodes_per_task for task in {record["task"] for record in records}
    }
    assert len({record["task"] for record in records}) == 37
    assert all(record["eval_set"] == name for record in records)


@pytest.mark.parametrize("name", ["eval_v0_smoke"])
def test_checked_in_yaml_and_jsonl_manifests_are_semantically_equal(name: str):
    root = REPO / "eval_sets" / name
    manifest = _read_yaml(root / "manifest.yaml")
    records = _read_jsonl(root / "manifest.jsonl")
    assert _flatten_yaml_manifest(manifest) == records


def test_checked_in_readmes_describe_exact_frozen_scope():
    expected_phrases = {
        "eval_v0_smoke": (
            "1 initialization spec for each of 37 curated benchmark task names"
        ),
    }
    for name, phrase in expected_phrases.items():
        readme = re.sub(
            r"\s+",
            " ",
            (REPO / "eval_sets" / name / "README.md").read_text(encoding="utf-8"),
        )
        assert phrase in readme
        assert "per registered task" not in readme
        assert "each formal task" not in readme
