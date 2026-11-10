"""HARP-Bench paper names, benchmark task table, and legacy-name mapping.

Public task names follow the paper: Cueing tasks use the paper task name
(``deliver_to_human``, ``feed_documents``, ``pour_water``, ...), and every other benchmark task is
``<paper task>_<condition>`` with condition ``contribution``,
``interruption`` or ``copresence``.  The pre-HARP names (``*_assist``,
``*_interrupt``, ``*_neutral``, ``categorize_cooperative``,
``deliver_to_human_easy``, ``stamp_documents``) remain accepted compatibility names; see
``LEGACY_TASK_NAMES``.

This module is dependency-free so lightweight report scripts can load it by
file path without importing the simulator stack.  ``envs/tasks/registry.py``
builds its descriptors from these tables and re-exports them.
"""

from __future__ import annotations

PAPER_NAME = "HARP-Bench"

# Benchmark condition machine value (TaskVariant value, eval-set manifest
# ``category``) -> paper display name, in the paper's order.
CATEGORY_DISPLAY: dict[str, str] = {
    "cueing": "Cueing",
    "contribution": "Contribution",
    "interruption": "Interruption",
    "copresence": "Co-presence",
}
PAPER_CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_DISPLAY)

# Pre-HARP condition values, still accepted in old manifests and results.
LEGACY_CATEGORY_VALUES: dict[str, str] = {
    "intent": "cueing",
    "assist": "contribution",
    "interrupt": "interruption",
    "neutral": "copresence",
}

# Registry ``family`` -> paper task name.  Families keep their historical
# identifiers (they are reporting groups shared with the no-human base task
# names).  ``take_from_human`` groups two tasks the paper names separately.
FAMILY_DISPLAY: dict[str, str] = {
    "deliver_to_human": "Deliver to Human",
    "pour_water": "Pour Water",
    "oil_bottle_recovery": "Oil Bottle Recovery",
    "open_microwave": "Open Microwave",
    "stamp_documents": "Feed Documents",
    "categorize": "Categorize Objects",
    "put_object_cabinet": "Put Object in Cabinet",
    "stack_bowls_three": "Stack Three Bowls",
    "place_bread_in_basket": "Place Bread in Basket",
    "dump_bin": "Dump Bin",
    "place_burger_fries": "Place Burger and Fries",
    "place_dual_shoes": "Place Pair of Shoes",
    "place_food_in_skillet": "Place Food in Skillet",
    "blocks_ranking_rgb": "Rank Blocks by Color",
    "blocks_ranking_size": "Rank Blocks by Size",
}
TASK_DISPLAY_OVERRIDES: dict[str, str] = {
    "take_from_human_easy": "Take from Human (Easy)",
    "take_from_human_safety": "Take from Human (Safety)",
}

# The 37 benchmark tasks (rows of eval_sets/eval_v3_sampled) in paper order:
# public task name -> (registry family, condition value).
BENCHMARK_TASKS: dict[str, tuple[str, str]] = {
    "deliver_to_human": ("deliver_to_human", "cueing"),
    "pour_water": ("pour_water", "cueing"),
    "oil_bottle_recovery": ("oil_bottle_recovery", "cueing"),
    "open_microwave": ("open_microwave", "cueing"),
    "feed_documents": ("stamp_documents", "cueing"),
    "take_from_human_easy": ("take_from_human", "cueing"),
    "take_from_human_safety": ("take_from_human", "cueing"),
    "categorize_objects_contribution": ("categorize", "contribution"),
    "put_object_in_cabinet_contribution": ("put_object_cabinet", "contribution"),
    "stack_three_bowls_contribution": ("stack_bowls_three", "contribution"),
    "place_bread_in_basket_contribution": ("place_bread_in_basket", "contribution"),
    "dump_bin_contribution": ("dump_bin", "contribution"),
    "place_burger_and_fries_contribution": ("place_burger_fries", "contribution"),
    "place_pair_of_shoes_contribution": ("place_dual_shoes", "contribution"),
    "place_food_in_skillet_contribution": ("place_food_in_skillet", "contribution"),
    "rank_blocks_by_color_contribution": ("blocks_ranking_rgb", "contribution"),
    "rank_blocks_by_size_contribution": ("blocks_ranking_size", "contribution"),
    "categorize_objects_interruption": ("categorize", "interruption"),
    "put_object_in_cabinet_interruption": ("put_object_cabinet", "interruption"),
    "stack_three_bowls_interruption": ("stack_bowls_three", "interruption"),
    "place_bread_in_basket_interruption": ("place_bread_in_basket", "interruption"),
    "dump_bin_interruption": ("dump_bin", "interruption"),
    "place_burger_and_fries_interruption": ("place_burger_fries", "interruption"),
    "place_pair_of_shoes_interruption": ("place_dual_shoes", "interruption"),
    "place_food_in_skillet_interruption": ("place_food_in_skillet", "interruption"),
    "rank_blocks_by_color_interruption": ("blocks_ranking_rgb", "interruption"),
    "rank_blocks_by_size_interruption": ("blocks_ranking_size", "interruption"),
    "categorize_objects_copresence": ("categorize", "copresence"),
    "put_object_in_cabinet_copresence": ("put_object_cabinet", "copresence"),
    "stack_three_bowls_copresence": ("stack_bowls_three", "copresence"),
    "place_bread_in_basket_copresence": ("place_bread_in_basket", "copresence"),
    "dump_bin_copresence": ("dump_bin", "copresence"),
    "place_burger_and_fries_copresence": ("place_burger_fries", "copresence"),
    "place_pair_of_shoes_copresence": ("place_dual_shoes", "copresence"),
    "place_food_in_skillet_copresence": ("place_food_in_skillet", "copresence"),
    "rank_blocks_by_color_copresence": ("blocks_ranking_rgb", "copresence"),
    "rank_blocks_by_size_copresence": ("blocks_ranking_size", "copresence"),
}

# Pre-HARP public task name -> current public task name.  Every key is a
# registered compatibility name that resolves to the same task class, so old
# commands, manifests, checkpoints, and datasets keyed by these names work.
LEGACY_TASK_NAMES: dict[str, str] = {
    "deliver_to_human_easy": "deliver_to_human",
    "stamp_documents": "feed_documents",
    "categorize_cooperative": "categorize_objects_contribution",
    "categorize_interrupt": "categorize_objects_interruption",
    "categorize_neutral": "categorize_objects_copresence",
    "put_object_cabinet_assist": "put_object_in_cabinet_contribution",
    "put_object_cabinet_interrupt": "put_object_in_cabinet_interruption",
    "put_object_cabinet_neutral": "put_object_in_cabinet_copresence",
    "stack_bowls_three_assist": "stack_three_bowls_contribution",
    "stack_bowls_three_interrupt": "stack_three_bowls_interruption",
    "stack_bowls_three_neutral": "stack_three_bowls_copresence",
    "place_bread_in_basket_assist": "place_bread_in_basket_contribution",
    "place_bread_in_basket_interrupt": "place_bread_in_basket_interruption",
    "place_bread_in_basket_neutral": "place_bread_in_basket_copresence",
    "dump_bin_assist": "dump_bin_contribution",
    "dump_bin_interrupt": "dump_bin_interruption",
    "dump_bin_neutral": "dump_bin_copresence",
    "place_burger_fries_assist": "place_burger_and_fries_contribution",
    "place_burger_fries_interrupt": "place_burger_and_fries_interruption",
    "place_burger_fries_neutral": "place_burger_and_fries_copresence",
    "place_dual_shoes_assist": "place_pair_of_shoes_contribution",
    "place_dual_shoes_interrupt": "place_pair_of_shoes_interruption",
    "place_dual_shoes_neutral": "place_pair_of_shoes_copresence",
    "place_food_in_skillet_assist": "place_food_in_skillet_contribution",
    "place_food_in_skillet_interrupt": "place_food_in_skillet_interruption",
    "place_food_in_skillet_neutral": "place_food_in_skillet_copresence",
    "blocks_ranking_rgb_assist": "rank_blocks_by_color_contribution",
    "blocks_ranking_rgb_interrupt": "rank_blocks_by_color_interruption",
    "blocks_ranking_rgb_neutral": "rank_blocks_by_color_copresence",
    "blocks_ranking_size_assist": "rank_blocks_by_size_contribution",
    "blocks_ranking_size_interrupt": "rank_blocks_by_size_interruption",
    "blocks_ranking_size_neutral": "rank_blocks_by_size_copresence",
}


def canonical_task_name(task_name: str) -> str:
    """Map a pre-HARP task name to its current name; other names pass through."""

    return LEGACY_TASK_NAMES.get(task_name, task_name)


def canonical_category(value) -> str:
    """Map a pre-HARP condition value (``neutral``...) to the current value."""

    text = str(getattr(value, "value", value))
    return LEGACY_CATEGORY_VALUES.get(text, text)


def task_category(task_name: str) -> str | None:
    """Condition value of a benchmark task (legacy names accepted), else None."""

    entry = BENCHMARK_TASKS.get(canonical_task_name(task_name))
    return entry[1] if entry else None


def category_display(value) -> str:
    """Paper name for a condition value; unknown values pass through."""

    text = canonical_category(value)
    return CATEGORY_DISPLAY.get(text, text)


def paper_label(task_name: str) -> tuple[str, str]:
    """Return ``(paper condition, paper task name)`` for a benchmark task.

    Legacy names are accepted.  Raises ``KeyError`` for registered tasks
    outside the 37-task benchmark (no-human base and factory scenarios).
    """

    name = canonical_task_name(task_name)
    try:
        family, category = BENCHMARK_TASKS[name]
    except KeyError:
        raise KeyError(
            f"{task_name!r} is not one of the {len(BENCHMARK_TASKS)} "
            f"{PAPER_NAME} benchmark tasks"
        ) from None
    display = TASK_DISPLAY_OVERRIDES.get(name) or FAMILY_DISPLAY[family]
    return CATEGORY_DISPLAY[category], display


def paper_task_label(task_name: str) -> str:
    """Return ``"<condition>: <task>"`` for benchmark tasks, else the name."""

    try:
        category, display = paper_label(task_name)
    except KeyError:
        return task_name
    return f"{category}: {display}"
