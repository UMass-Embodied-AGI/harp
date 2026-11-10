"""Public descriptor, alias, and no-human registry contracts."""

from dataclasses import replace

import pytest

import envs.tasks as tasks_package
from envs.tasks import (
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    TASK_ALIASES,
    TASK_DESCRIPTOR_MAP,
    TASK_DESCRIPTORS,
    TASK_MAP,
    TASK_REGISTRY,
    LEGACY_CLASS_NAMES,
    LEGACY_TASK_NAMES,
    CategorizeObjectsContribution,
    CategorizeObjectsInterruption,
    TaskDescriptor,
    TaskRegistration,
    TaskVariant,
    build_task_registry,
    canonical_task_name,
    resolve_task_class,
)


EXPECTED_TASKS = {
    "take_from_human_easy": ("envs.tasks.take_from_human_easy", "TakeFromHumanEasy"),
    "deliver_to_human": ("envs.tasks.deliver_to_human", "DeliverToHuman"),
    "take_from_human_safety": ("envs.tasks.take_from_human_safety", "TakeFromHumanSafety"),
    "feed_documents": ("envs.tasks.feed_documents", "FeedDocuments"),
    "pour_water": ("envs.tasks.pour_water", "PourWater"),
    "oil_bottle_recovery": ("envs.tasks.oil_bottle_recovery", "OilBottleRecovery"),
    "open_microwave": ("envs.tasks.open_microwave", "OpenMicrowave"),
    "categorize_objects_contribution": ("envs.tasks.categorize_objects_contribution", "CategorizeObjectsContribution"),
    "categorize_objects_interruption": ("envs.tasks.categorize_objects_interruption", "CategorizeObjectsInterruption"),
    "categorize_objects_copresence": ("envs.tasks.categorize_objects_copresence", "CategorizeObjectsCopresence"),
    "put_object_cabinet": ("envs.task_bases.put_object_cabinet", "PutObjectCabinet"),
    "stack_bowls_three": ("envs.task_bases.stack_bowls_three", "StackBowlsThree"),
    "place_bread_in_basket": ("envs.task_bases.place_bread_in_basket", "PlaceBreadInBasket"),
    "dump_bin": ("envs.task_bases.dump_bin", "DumpBin"),
    "place_burger_fries": ("envs.task_bases.place_burger_fries", "PlaceBurgerFries"),
    "place_dual_shoes": ("envs.task_bases.place_dual_shoes", "PlaceDualShoes"),
    "place_food_in_skillet": ("envs.task_bases.place_food_in_skillet", "PlaceFoodInSkillet"),
    "blocks_ranking_rgb": ("envs.task_bases.blocks_ranking_rgb", "BlocksRankingRGB"),
    "blocks_ranking_size": ("envs.task_bases.blocks_ranking_size", "BlocksRankingSize"),
    "put_object_in_cabinet_contribution": ("envs.tasks.put_object_in_cabinet_contribution", "PutObjectInCabinetContribution"),
    "put_object_in_cabinet_interruption": ("envs.tasks.put_object_in_cabinet_interruption", "PutObjectInCabinetInterruption"),
    "put_object_in_cabinet_copresence": ("envs.tasks.put_object_in_cabinet_copresence", "PutObjectInCabinetCopresence"),
    "stack_three_bowls_contribution": ("envs.tasks.stack_three_bowls_contribution", "StackThreeBowlsContribution"),
    "stack_three_bowls_interruption": ("envs.tasks.stack_three_bowls_interruption", "StackThreeBowlsInterruption"),
    "stack_three_bowls_copresence": ("envs.tasks.stack_three_bowls_copresence", "StackThreeBowlsCopresence"),
    "place_bread_in_basket_contribution": ("envs.tasks.place_bread_in_basket_contribution", "PlaceBreadInBasketContribution"),
    "place_bread_in_basket_interruption": ("envs.tasks.place_bread_in_basket_interruption", "PlaceBreadInBasketInterruption"),
    "place_bread_in_basket_copresence": ("envs.tasks.place_bread_in_basket_copresence", "PlaceBreadInBasketCopresence"),
    "dump_bin_contribution": ("envs.tasks.dump_bin_contribution", "DumpBinContribution"),
    "dump_bin_interruption": ("envs.tasks.dump_bin_interruption", "DumpBinInterruption"),
    "dump_bin_copresence": ("envs.tasks.dump_bin_copresence", "DumpBinCopresence"),
    "place_burger_and_fries_contribution": ("envs.tasks.place_burger_and_fries_contribution", "PlaceBurgerAndFriesContribution"),
    "place_burger_and_fries_interruption": ("envs.tasks.place_burger_and_fries_interruption", "PlaceBurgerAndFriesInterruption"),
    "place_burger_and_fries_copresence": ("envs.tasks.place_burger_and_fries_copresence", "PlaceBurgerAndFriesCopresence"),
    "place_pair_of_shoes_contribution": ("envs.tasks.place_pair_of_shoes_contribution", "PlacePairOfShoesContribution"),
    "place_pair_of_shoes_interruption": ("envs.tasks.place_pair_of_shoes_interruption", "PlacePairOfShoesInterruption"),
    "place_pair_of_shoes_copresence": ("envs.tasks.place_pair_of_shoes_copresence", "PlacePairOfShoesCopresence"),
    "place_food_in_skillet_contribution": ("envs.tasks.place_food_in_skillet_contribution", "PlaceFoodInSkilletContribution"),
    "place_food_in_skillet_interruption": ("envs.tasks.place_food_in_skillet_interruption", "PlaceFoodInSkilletInterruption"),
    "place_food_in_skillet_copresence": ("envs.tasks.place_food_in_skillet_copresence", "PlaceFoodInSkilletCopresence"),
    "rank_blocks_by_color_contribution": ("envs.tasks.rank_blocks_by_color_contribution", "RankBlocksByColorContribution"),
    "rank_blocks_by_color_interruption": ("envs.tasks.rank_blocks_by_color_interruption", "RankBlocksByColorInterruption"),
    "rank_blocks_by_color_copresence": ("envs.tasks.rank_blocks_by_color_copresence", "RankBlocksByColorCopresence"),
    "rank_blocks_by_size_contribution": ("envs.tasks.rank_blocks_by_size_contribution", "RankBlocksBySizeContribution"),
    "rank_blocks_by_size_interruption": ("envs.tasks.rank_blocks_by_size_interruption", "RankBlocksBySizeInterruption"),
    "rank_blocks_by_size_copresence": ("envs.tasks.rank_blocks_by_size_copresence", "RankBlocksBySizeCopresence"),
    "factory_inspect_pack": ("envs.tasks.factory_inspect_pack", "FactoryInspectPack"),
    "factory_kit_packing": ("envs.tasks.factory_kit_packing", "FactoryKitPacking"),
    "factory_line_feeding": ("envs.tasks.factory_line_feeding", "FactoryLineFeeding"),
}

EXPECTED_ALIASES = {
    "deliver_to_human_easy": ("envs.tasks.deliver_to_human", "DeliverToHuman"),
    "stamp_documents": ("envs.tasks.feed_documents", "FeedDocuments"),
    "categorize_cooperative": ("envs.tasks.categorize_objects_contribution", "CategorizeObjectsContribution"),
    "categorize_interrupt": ("envs.tasks.categorize_objects_interruption", "CategorizeObjectsInterruption"),
    "categorize_neutral": ("envs.tasks.categorize_objects_copresence", "CategorizeObjectsCopresence"),
    "put_object_cabinet_assist": ("envs.tasks.put_object_in_cabinet_contribution", "PutObjectInCabinetContribution"),
    "put_object_cabinet_interrupt": ("envs.tasks.put_object_in_cabinet_interruption", "PutObjectInCabinetInterruption"),
    "put_object_cabinet_neutral": ("envs.tasks.put_object_in_cabinet_copresence", "PutObjectInCabinetCopresence"),
    "stack_bowls_three_assist": ("envs.tasks.stack_three_bowls_contribution", "StackThreeBowlsContribution"),
    "stack_bowls_three_interrupt": ("envs.tasks.stack_three_bowls_interruption", "StackThreeBowlsInterruption"),
    "stack_bowls_three_neutral": ("envs.tasks.stack_three_bowls_copresence", "StackThreeBowlsCopresence"),
    "place_bread_in_basket_assist": ("envs.tasks.place_bread_in_basket_contribution", "PlaceBreadInBasketContribution"),
    "place_bread_in_basket_interrupt": ("envs.tasks.place_bread_in_basket_interruption", "PlaceBreadInBasketInterruption"),
    "place_bread_in_basket_neutral": ("envs.tasks.place_bread_in_basket_copresence", "PlaceBreadInBasketCopresence"),
    "dump_bin_assist": ("envs.tasks.dump_bin_contribution", "DumpBinContribution"),
    "dump_bin_interrupt": ("envs.tasks.dump_bin_interruption", "DumpBinInterruption"),
    "dump_bin_neutral": ("envs.tasks.dump_bin_copresence", "DumpBinCopresence"),
    "place_burger_fries_assist": ("envs.tasks.place_burger_and_fries_contribution", "PlaceBurgerAndFriesContribution"),
    "place_burger_fries_interrupt": ("envs.tasks.place_burger_and_fries_interruption", "PlaceBurgerAndFriesInterruption"),
    "place_burger_fries_neutral": ("envs.tasks.place_burger_and_fries_copresence", "PlaceBurgerAndFriesCopresence"),
    "place_dual_shoes_assist": ("envs.tasks.place_pair_of_shoes_contribution", "PlacePairOfShoesContribution"),
    "place_dual_shoes_interrupt": ("envs.tasks.place_pair_of_shoes_interruption", "PlacePairOfShoesInterruption"),
    "place_dual_shoes_neutral": ("envs.tasks.place_pair_of_shoes_copresence", "PlacePairOfShoesCopresence"),
    "place_food_in_skillet_assist": ("envs.tasks.place_food_in_skillet_contribution", "PlaceFoodInSkilletContribution"),
    "place_food_in_skillet_interrupt": ("envs.tasks.place_food_in_skillet_interruption", "PlaceFoodInSkilletInterruption"),
    "place_food_in_skillet_neutral": ("envs.tasks.place_food_in_skillet_copresence", "PlaceFoodInSkilletCopresence"),
    "blocks_ranking_rgb_assist": ("envs.tasks.rank_blocks_by_color_contribution", "RankBlocksByColorContribution"),
    "blocks_ranking_rgb_interrupt": ("envs.tasks.rank_blocks_by_color_interruption", "RankBlocksByColorInterruption"),
    "blocks_ranking_rgb_neutral": ("envs.tasks.rank_blocks_by_color_copresence", "RankBlocksByColorCopresence"),
    "blocks_ranking_size_assist": ("envs.tasks.rank_blocks_by_size_contribution", "RankBlocksBySizeContribution"),
    "blocks_ranking_size_interrupt": ("envs.tasks.rank_blocks_by_size_interruption", "RankBlocksBySizeInterruption"),
    "blocks_ranking_size_neutral": ("envs.tasks.rank_blocks_by_size_copresence", "RankBlocksBySizeCopresence"),
}

EXPECTED_NO_HUMAN_CANONICAL = {
    "categorize": ("envs.tasks.categorize_objects_contribution", "CategorizeObjectsContribution"),
    "put_object_cabinet": ("envs.task_bases.put_object_cabinet", "PutObjectCabinet"),
    "stack_bowls_three": ("envs.task_bases.stack_bowls_three", "StackBowlsThree"),
    "place_bread_in_basket": ("envs.task_bases.place_bread_in_basket", "PlaceBreadInBasket"),
    "dump_bin": ("envs.task_bases.dump_bin", "DumpBin"),
    "place_burger_fries": ("envs.task_bases.place_burger_fries", "PlaceBurgerFries"),
    "place_dual_shoes": ("envs.task_bases.place_dual_shoes", "PlaceDualShoes"),
    "place_food_in_skillet": ("envs.task_bases.place_food_in_skillet", "PlaceFoodInSkillet"),
    "blocks_ranking_rgb": ("envs.task_bases.blocks_ranking_rgb", "BlocksRankingRGB"),
    "blocks_ranking_size": ("envs.task_bases.blocks_ranking_size", "BlocksRankingSize"),
}

EXPECTED_NO_HUMAN_ALIASES = {
    "categorize_objects_contribution": "categorize",
    "categorize_cooperative": "categorize",
    "categorize_objects_interruption": "categorize",
    "categorize_interrupt": "categorize",
    "categorize_objects_copresence": "categorize",
    "categorize_neutral": "categorize",
    "put_object_in_cabinet_contribution": "put_object_cabinet",
    "put_object_cabinet_assist": "put_object_cabinet",
    "put_object_in_cabinet_interruption": "put_object_cabinet",
    "put_object_cabinet_interrupt": "put_object_cabinet",
    "put_object_in_cabinet_copresence": "put_object_cabinet",
    "put_object_cabinet_neutral": "put_object_cabinet",
    "stack_three_bowls_contribution": "stack_bowls_three",
    "stack_bowls_three_assist": "stack_bowls_three",
    "stack_three_bowls_interruption": "stack_bowls_three",
    "stack_bowls_three_interrupt": "stack_bowls_three",
    "stack_three_bowls_copresence": "stack_bowls_three",
    "stack_bowls_three_neutral": "stack_bowls_three",
    "place_bread_in_basket_contribution": "place_bread_in_basket",
    "place_bread_in_basket_assist": "place_bread_in_basket",
    "place_bread_in_basket_interruption": "place_bread_in_basket",
    "place_bread_in_basket_interrupt": "place_bread_in_basket",
    "place_bread_in_basket_copresence": "place_bread_in_basket",
    "place_bread_in_basket_neutral": "place_bread_in_basket",
    "dump_bin_contribution": "dump_bin",
    "dump_bin_assist": "dump_bin",
    "dump_bin_interruption": "dump_bin",
    "dump_bin_interrupt": "dump_bin",
    "dump_bin_copresence": "dump_bin",
    "dump_bin_neutral": "dump_bin",
    "place_burger_and_fries_contribution": "place_burger_fries",
    "place_burger_fries_assist": "place_burger_fries",
    "place_burger_and_fries_interruption": "place_burger_fries",
    "place_burger_fries_interrupt": "place_burger_fries",
    "place_burger_and_fries_copresence": "place_burger_fries",
    "place_burger_fries_neutral": "place_burger_fries",
    "place_pair_of_shoes_contribution": "place_dual_shoes",
    "place_dual_shoes_assist": "place_dual_shoes",
    "place_pair_of_shoes_interruption": "place_dual_shoes",
    "place_dual_shoes_interrupt": "place_dual_shoes",
    "place_pair_of_shoes_copresence": "place_dual_shoes",
    "place_dual_shoes_neutral": "place_dual_shoes",
    "place_food_in_skillet_contribution": "place_food_in_skillet",
    "place_food_in_skillet_assist": "place_food_in_skillet",
    "place_food_in_skillet_interruption": "place_food_in_skillet",
    "place_food_in_skillet_interrupt": "place_food_in_skillet",
    "place_food_in_skillet_copresence": "place_food_in_skillet",
    "place_food_in_skillet_neutral": "place_food_in_skillet",
    "rank_blocks_by_color_contribution": "blocks_ranking_rgb",
    "blocks_ranking_rgb_assist": "blocks_ranking_rgb",
    "rank_blocks_by_color_interruption": "blocks_ranking_rgb",
    "blocks_ranking_rgb_interrupt": "blocks_ranking_rgb",
    "rank_blocks_by_color_copresence": "blocks_ranking_rgb",
    "blocks_ranking_rgb_neutral": "blocks_ranking_rgb",
    "rank_blocks_by_size_contribution": "blocks_ranking_size",
    "blocks_ranking_size_assist": "blocks_ranking_size",
    "rank_blocks_by_size_interruption": "blocks_ranking_size",
    "blocks_ranking_size_interrupt": "blocks_ranking_size",
    "rank_blocks_by_size_copresence": "blocks_ranking_size",
    "blocks_ranking_size_neutral": "blocks_ranking_size",
}


def _class_id(task_cls):
    return task_cls.__module__, task_cls.__qualname__


def test_descriptor_projections_match_frozen_public_registry():
    assert isinstance(TASK_MAP, dict)
    assert isinstance(TASK_ALIASES, dict)
    assert list(TASK_MAP) == list(EXPECTED_TASKS)
    assert list(TASK_ALIASES) == list(EXPECTED_ALIASES)
    assert {name: _class_id(cls) for name, cls in TASK_MAP.items()} == EXPECTED_TASKS
    assert {name: _class_id(cls) for name, cls in TASK_ALIASES.items()} == EXPECTED_ALIASES
    assert TASK_REGISTRY.task_map is TASK_MAP
    assert TASK_REGISTRY.task_aliases is TASK_ALIASES
    assert TASK_REGISTRY.descriptors is TASK_DESCRIPTORS
    assert TASK_REGISTRY.descriptor_map is TASK_DESCRIPTOR_MAP


def test_every_legacy_task_class_remains_exported_from_package():
    for descriptor in TASK_DESCRIPTORS:
        assert getattr(tasks_package, descriptor.task_cls.__name__) is descriptor.task_cls


def test_categorize_metadata_and_legacy_compatibility_names():
    contribution = TASK_DESCRIPTOR_MAP["categorize_objects_contribution"]
    interruption = TASK_DESCRIPTOR_MAP["categorize_objects_interruption"]

    assert contribution.family == "categorize"
    assert contribution.variant is TaskVariant.CONTRIBUTION
    assert contribution.variant_label == "cooperative"
    assert contribution.registration is TaskRegistration.CANONICAL
    assert contribution.task_cls is CategorizeObjectsContribution

    assert interruption.family == "categorize"
    assert interruption.variant is TaskVariant.INTERRUPTION
    assert interruption.registration is TaskRegistration.CANONICAL
    assert interruption.task_cls is CategorizeObjectsInterruption
    assert interruption.task_cls is not contribution.task_cls

    # The pre-HARP names stay accepted and resolve to the same classes.
    assert "categorize_interrupt" not in TASK_MAP
    assert TASK_ALIASES["categorize_interrupt"] is CategorizeObjectsInterruption
    assert TASK_ALIASES["categorize_cooperative"] is CategorizeObjectsContribution
    assert resolve_task_class("categorize_interrupt") == (
        "categorize_interrupt",
        CategorizeObjectsInterruption,
    )


def test_every_legacy_name_is_a_compatibility_alias_of_its_new_task():
    assert set(TASK_ALIASES) == set(LEGACY_TASK_NAMES)
    for old, new in LEGACY_TASK_NAMES.items():
        assert new in TASK_MAP
        assert TASK_ALIASES[old] is TASK_MAP[new]
        old_d, new_d = TASK_DESCRIPTOR_MAP[old], TASK_DESCRIPTOR_MAP[new]
        assert old_d.registration is TaskRegistration.COMPATIBILITY
        assert (old_d.family, old_d.variant, old_d.no_human_name) == (
            new_d.family, new_d.variant, new_d.no_human_name,
        )
        assert old_d.instruction == new_d.instruction
        assert canonical_task_name(old) == new
    assert len(LEGACY_CLASS_NAMES) == len(LEGACY_TASK_NAMES)
    assert LEGACY_CLASS_NAMES["StampDocuments"] == "FeedDocuments"
    for old_name, new_cls in LEGACY_CLASS_NAMES.items():
        assert old_name != new_cls
        assert getattr(tasks_package, old_name) is getattr(tasks_package, new_cls)


def test_legacy_variant_member_names_alias_the_harp_values():
    assert TaskVariant.INTENT is TaskVariant.CUEING
    assert TaskVariant.ASSIST is TaskVariant.CONTRIBUTION
    assert TaskVariant.INTERRUPT is TaskVariant.INTERRUPTION
    assert TaskVariant.NEUTRAL is TaskVariant.COPRESENCE
    assert [v.value for v in TaskVariant] == [
        "base", "cueing", "contribution", "interruption", "copresence",
    ]


def test_factory_scene_is_not_mistaken_for_an_interaction_variant():
    assert TASK_DESCRIPTOR_MAP["factory_inspect_pack"].variant is TaskVariant.CUEING
    assert TASK_DESCRIPTOR_MAP["factory_kit_packing"].variant is TaskVariant.CONTRIBUTION
    assert TASK_DESCRIPTOR_MAP["factory_line_feeding"].variant is TaskVariant.CUEING


def test_descriptor_exposes_resolved_instruction_and_raw_template():
    for descriptor in TASK_DESCRIPTORS:
        assert descriptor.instruction_template == descriptor.task_cls.INSTRUCTION
        assert descriptor.instruction == descriptor.task_cls.default_instruction()

    descriptor = TASK_DESCRIPTOR_MAP["take_from_human_easy"]

    assert descriptor.instruction_template == (
        "take the {object} from the person and place it on the table"
    )
    assert descriptor.instruction == descriptor.task_cls.default_instruction()
    # The apple is not in human_transfer_pool; the representative
    # first-pool-object is the rubik's cube.
    assert descriptor.instruction == (
        "take the rubik's cube from the person and place it on the table"
    )


def test_normal_resolver_matches_frozen_registry_for_every_name():
    for name, expected_class_id in {**EXPECTED_TASKS, **EXPECTED_ALIASES}.items():
        resolved_name, task_cls = resolve_task_class(name)
        assert resolved_name == name
        assert _class_id(task_cls) == expected_class_id

    for unavailable in ("categorize", "frying_with_robot_pour"):
        with pytest.raises(ValueError, match=f"^Unknown task: {unavailable}\\."):
            resolve_task_class(unavailable)


def test_no_human_projections_and_resolver_match_frozen_registry():
    assert list(NO_HUMAN_CANONICAL_TASK_MAP) == list(EXPECTED_NO_HUMAN_CANONICAL)
    assert {
        name: _class_id(cls) for name, cls in NO_HUMAN_CANONICAL_TASK_MAP.items()
    } == EXPECTED_NO_HUMAN_CANONICAL
    assert NO_HUMAN_VARIANT_ALIASES == EXPECTED_NO_HUMAN_ALIASES
    assert list(NO_HUMAN_TASK_MAP) == [
        *EXPECTED_NO_HUMAN_CANONICAL,
        *EXPECTED_NO_HUMAN_ALIASES,
    ]

    for name in NO_HUMAN_TASK_MAP:
        canonical = EXPECTED_NO_HUMAN_ALIASES.get(name, name)
        resolved_name, task_cls = resolve_task_class(name, no_human=True)
        assert resolved_name == canonical
        assert task_cls is NO_HUMAN_CANONICAL_TASK_MAP[canonical]
        assert task_cls is NO_HUMAN_TASK_MAP[name]

    assert resolve_task_class("categorize", no_human=True) == (
        "categorize",
        CategorizeObjectsContribution,
    )


@pytest.mark.parametrize(
    "task_name",
    [
        "take_from_human_easy",
        "deliver_to_human",
        "deliver_to_human_easy",
        "take_from_human_safety",
        "feed_documents",
        "stamp_documents",
        "pour_water",
        "oil_bottle_recovery",
        "open_microwave",
        "factory_inspect_pack",
        "factory_kit_packing",
        "factory_line_feeding",
    ],
)
def test_normal_tasks_without_no_human_variants_still_reject_collapse(task_name):
    with pytest.raises(ValueError, match="has no no-human variant"):
        resolve_task_class(task_name, no_human=True)


class _DummyTask:
    INSTRUCTION = "dummy"


def test_registry_builder_rejects_duplicate_and_dangling_metadata():
    base = TaskDescriptor(
        "base",
        _DummyTask,
        "family",
        TaskVariant.BASE,
        no_human_name="base",
        no_human_canonical=True,
    )
    with pytest.raises(ValueError, match="Duplicate task descriptor name"):
        build_task_registry((base, replace(base)))

    dangling = TaskDescriptor(
        "assist",
        _DummyTask,
        "family",
        TaskVariant.ASSIST,
        no_human_name="missing",
    )
    with pytest.raises(ValueError, match="dangling no-human target"):
        build_task_registry((dangling,))

    duplicate_provider = replace(
        base,
        name="other_base",
        registration=TaskRegistration.CANONICAL,
    )
    with pytest.raises(ValueError, match="Duplicate no-human canonical name"):
        build_task_registry((base, duplicate_provider))


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"family": "  "}, ValueError, "non-empty family"),
        ({"variant": "base"}, TypeError, "invalid variant"),
        ({"registration": "canonical"}, TypeError, "invalid registration"),
        ({"variant_label": ""}, ValueError, "invalid variant_label"),
        ({"no_human_name": ""}, ValueError, "invalid no-human name"),
        ({"no_human_canonical": 1}, TypeError, "must be a bool"),
        (
            {"registration": TaskRegistration.COMPATIBILITY},
            ValueError,
            "must be canonical",
        ),
        (
            {"variant": TaskVariant.INTERRUPT},
            ValueError,
            "must be a base or assist task",
        ),
        ({"no_human_name": None}, ValueError, "has no target name"),
    ],
)
def test_registry_builder_rejects_malformed_descriptor_fields(
    changes, error_type, match
):
    descriptor = TaskDescriptor(
        "base",
        _DummyTask,
        "family",
        TaskVariant.BASE,
        no_human_name="base",
        no_human_canonical=True,
    )

    with pytest.raises(error_type, match=match):
        build_task_registry((replace(descriptor, **changes),))


def test_registry_builder_rejects_malformed_provider_relationships():
    base = TaskDescriptor(
        "base",
        _DummyTask,
        "family",
        TaskVariant.BASE,
        no_human_name="base",
        no_human_canonical=True,
    )
    non_provider_base = replace(base, no_human_canonical=False)
    with pytest.raises(ValueError, match="must be provided by that descriptor"):
        build_task_registry((non_provider_base,))

    conflicting_provider = replace(base, name="provider")
    with pytest.raises(ValueError, match="conflicts with registered task"):
        build_task_registry((replace(base, no_human_canonical=False), conflicting_provider))

    cross_family = TaskDescriptor(
        "assist",
        _DummyTask,
        "other_family",
        TaskVariant.ASSIST,
        no_human_name="base",
    )
    with pytest.raises(ValueError, match="cannot collapse across families"):
        build_task_registry((base, cross_family))

    intent_alias = replace(
        cross_family,
        family="family",
        variant=TaskVariant.INTENT,
    )
    with pytest.raises(ValueError, match="Intent task"):
        build_task_registry((base, intent_alias))

    with pytest.raises(TypeError, match="TaskDescriptor instances"):
        build_task_registry(("not-a-descriptor",))
