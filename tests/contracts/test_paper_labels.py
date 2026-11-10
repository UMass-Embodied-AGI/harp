"""HARP-Bench names: the 37 eval_v3_sampled tasks, conditions, and legacy names."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from envs.tasks import (
    BENCHMARK_TASKS,
    CATEGORY_DISPLAY,
    LEGACY_CATEGORY_VALUES,
    LEGACY_TASK_NAMES,
    PAPER_CATEGORY_ORDER,
    TASK_DESCRIPTOR_MAP,
    TASK_MAP,
    TaskVariant,
    canonical_category,
    category_display,
    paper_label,
    paper_task_label,
    task_category,
)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "eval_sets" / "eval_v3_sampled" / "manifest.jsonl"

EXPECTED_LABELS = {
    "deliver_to_human": ("Cueing", "Deliver to Human"),
    "pour_water": ("Cueing", "Pour Water"),
    "oil_bottle_recovery": ("Cueing", "Oil Bottle Recovery"),
    "open_microwave": ("Cueing", "Open Microwave"),
    "feed_documents": ("Cueing", "Feed Documents"),
    "take_from_human_easy": ("Cueing", "Take from Human (Easy)"),
    "take_from_human_safety": ("Cueing", "Take from Human (Safety)"),
}
_PAPER_TASKS = {
    "categorize_objects": "Categorize Objects",
    "put_object_in_cabinet": "Put Object in Cabinet",
    "stack_three_bowls": "Stack Three Bowls",
    "place_bread_in_basket": "Place Bread in Basket",
    "dump_bin": "Dump Bin",
    "place_burger_and_fries": "Place Burger and Fries",
    "place_pair_of_shoes": "Place Pair of Shoes",
    "place_food_in_skillet": "Place Food in Skillet",
    "rank_blocks_by_color": "Rank Blocks by Color",
    "rank_blocks_by_size": "Rank Blocks by Size",
}
for _suffix, _category in (
    ("contribution", "Contribution"),
    ("interruption", "Interruption"),
    ("copresence", "Co-presence"),
):
    for _stem, _display in _PAPER_TASKS.items():
        EXPECTED_LABELS[f"{_stem}_{_suffix}"] = (_category, _display)


def _manifest_tasks() -> dict[str, str]:
    tasks: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert tasks.setdefault(row["task"], row["category"]) == row["category"]
            assert row["id"].startswith(f"eval_v3_sampled__{row['category']}__{row['task']}__")
    return tasks


def test_condition_values_and_display_names_match_paper():
    assert CATEGORY_DISPLAY == {
        TaskVariant.CUEING.value: "Cueing",
        TaskVariant.CONTRIBUTION.value: "Contribution",
        TaskVariant.INTERRUPTION.value: "Interruption",
        TaskVariant.COPRESENCE.value: "Co-presence",
    }
    assert [CATEGORY_DISPLAY[v] for v in PAPER_CATEGORY_ORDER] == [
        "Cueing", "Contribution", "Interruption", "Co-presence",
    ]
    assert LEGACY_CATEGORY_VALUES == {
        "intent": "cueing", "assist": "contribution",
        "interrupt": "interruption", "neutral": "copresence",
    }
    assert category_display(TaskVariant.COPRESENCE) == "Co-presence"
    assert category_display("neutral") == "Co-presence"
    assert canonical_category("assist") == "contribution"
    assert category_display("base") == "base"


def test_eval_v3_sampled_tasks_use_harp_names_and_paper_labels():
    manifest = _manifest_tasks()
    assert set(manifest) == set(BENCHMARK_TASKS) == set(EXPECTED_LABELS)
    assert list(BENCHMARK_TASKS) == list(EXPECTED_LABELS)
    labels = {task: paper_label(task) for task in manifest}
    assert labels == EXPECTED_LABELS
    assert Counter(category for category, _ in labels.values()) == {
        "Cueing": 7, "Contribution": 10, "Interruption": 10, "Co-presence": 10,
    }
    for task, category_value in manifest.items():
        assert task in TASK_MAP
        assert task_category(task) == category_value
        assert CATEGORY_DISPLAY[category_value] == labels[task][0]


def test_benchmark_table_agrees_with_registry_descriptors():
    for task, (family, category) in BENCHMARK_TASKS.items():
        descriptor = TASK_DESCRIPTOR_MAP[task]
        assert descriptor.family == family
        assert descriptor.variant.value == category
        assert descriptor.paper_label == EXPECTED_LABELS[task]
    order = [category for _, category in BENCHMARK_TASKS.values()]
    assert order == sorted(order, key=PAPER_CATEGORY_ORDER.index)


def test_legacy_names_map_onto_benchmark_tasks_with_the_same_labels():
    assert len(LEGACY_TASK_NAMES) == 32
    assert LEGACY_TASK_NAMES["stamp_documents"] == "feed_documents"
    assert set(LEGACY_TASK_NAMES.values()) <= set(BENCHMARK_TASKS)
    for old, new in LEGACY_TASK_NAMES.items():
        assert paper_label(old) == paper_label(new)
        assert TASK_DESCRIPTOR_MAP[old].paper_label == paper_label(new)
    assert paper_task_label("categorize_cooperative") == "Contribution: Categorize Objects"


@pytest.mark.parametrize(
    "task", ["factory_kit_packing", "factory_inspect_pack", "put_object_cabinet", "categorize"]
)
def test_non_benchmark_tasks_have_no_paper_label(task):
    with pytest.raises(KeyError):
        paper_label(task)
    assert paper_task_label(task) == task
    assert task_category(task) is None
    if task in TASK_DESCRIPTOR_MAP:
        assert TASK_DESCRIPTOR_MAP[task].paper_label is None


def test_eval_expert_set_console_summary_uses_paper_condition_names():
    from scripts.eval_expert_set import format_category_summary, summarize

    results = [
        {"category": "copresence", "task": "dump_bin_copresence", "success": True},
        {"category": "cueing", "task": "pour_water", "success": False},
        {"category": None, "task": "factory_kit_packing", "success": True},
    ]
    summary = summarize(results)
    assert set(summary["by_category"]) == {"copresence", "cueing", "unknown"}
    assert format_category_summary(summary) == [
        "  Cueing (cueing): 0/1 = 0.000",
        "  Co-presence (copresence): 1/1 = 1.000",
        "  unknown (unknown): 1/1 = 1.000",
    ]


def test_eval_expert_set_routes_by_registry_variant_for_new_and_legacy_names():
    from scripts.eval_expert_set import (
        PARENT_PLAY_FALLBACK_TASKS,
        is_interrupt_task,
        is_neutral_eval_row,
    )

    for name in ("dump_bin_interruption", "dump_bin_interrupt",
                 "categorize_objects_interruption", "categorize_interrupt"):
        assert is_interrupt_task(name)
    for name in ("dump_bin_contribution", "pour_water", "dump_bin", "factory_kit_packing"):
        assert not is_interrupt_task(name)
    assert is_neutral_eval_row({"category": "copresence"})
    assert is_neutral_eval_row({"category": "neutral"})
    assert not is_neutral_eval_row({"category": "contribution"})
    # Same membership as the pre-HARP ``*_interrupt`` / ``*_assist`` suffix test.
    legacy = {old for old in LEGACY_TASK_NAMES if old.endswith(("_interrupt", "_assist"))}
    assert PARENT_PLAY_FALLBACK_TASKS == {LEGACY_TASK_NAMES[old] for old in legacy}
