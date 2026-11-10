"""Structured task registry and legacy mapping projections.

``TASK_DESCRIPTORS`` is the authoritative list of public task names.  The
mutable dictionaries historically exported by :mod:`envs.tasks` are generated
from that list so existing callers can keep using the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .categorize_objects_contribution import CategorizeObjectsContribution
from .categorize_objects_copresence import CategorizeObjectsCopresence
from .categorize_objects_interruption import CategorizeObjectsInterruption
from .deliver_to_human import DeliverToHuman
from .dump_bin_contribution import DumpBinContribution
from .dump_bin_copresence import DumpBinCopresence
from .dump_bin_interruption import DumpBinInterruption
from .feed_documents import FeedDocuments
from .factory_inspect_pack import FactoryInspectPack
from .factory_kit_packing import FactoryKitPacking
from .factory_line_feeding import FactoryLineFeeding
from .oil_bottle_recovery import OilBottleRecovery
from .open_microwave import OpenMicrowave
# Paper names and the legacy-name table, re-exported as part of the registry.
from .paper_labels import (
    BENCHMARK_TASKS as BENCHMARK_TASKS,
    CATEGORY_DISPLAY as CATEGORY_DISPLAY,
    FAMILY_DISPLAY as FAMILY_DISPLAY,
    LEGACY_CATEGORY_VALUES as LEGACY_CATEGORY_VALUES,
    LEGACY_TASK_NAMES as LEGACY_TASK_NAMES,
    PAPER_CATEGORY_ORDER as PAPER_CATEGORY_ORDER,
    PAPER_NAME as PAPER_NAME,
    TASK_DISPLAY_OVERRIDES as TASK_DISPLAY_OVERRIDES,
    canonical_category as canonical_category,
    canonical_task_name as canonical_task_name,
    category_display as category_display,
    paper_label as paper_label,
    paper_task_label as paper_task_label,
    task_category as task_category,
)
from .place_bread_in_basket_contribution import PlaceBreadInBasketContribution
from .place_bread_in_basket_copresence import PlaceBreadInBasketCopresence
from .place_bread_in_basket_interruption import PlaceBreadInBasketInterruption
from .place_burger_and_fries_contribution import PlaceBurgerAndFriesContribution
from .place_burger_and_fries_copresence import PlaceBurgerAndFriesCopresence
from .place_burger_and_fries_interruption import PlaceBurgerAndFriesInterruption
from .place_food_in_skillet_contribution import PlaceFoodInSkilletContribution
from .place_food_in_skillet_copresence import PlaceFoodInSkilletCopresence
from .place_food_in_skillet_interruption import PlaceFoodInSkilletInterruption
from .place_pair_of_shoes_contribution import PlacePairOfShoesContribution
from .place_pair_of_shoes_copresence import PlacePairOfShoesCopresence
from .place_pair_of_shoes_interruption import PlacePairOfShoesInterruption
from .pour_water import PourWater
from .put_object_in_cabinet_contribution import PutObjectInCabinetContribution
from .put_object_in_cabinet_copresence import PutObjectInCabinetCopresence
from .put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
from .rank_blocks_by_color_contribution import RankBlocksByColorContribution
from .rank_blocks_by_color_copresence import RankBlocksByColorCopresence
from .rank_blocks_by_color_interruption import RankBlocksByColorInterruption
from .rank_blocks_by_size_contribution import RankBlocksBySizeContribution
from .rank_blocks_by_size_copresence import RankBlocksBySizeCopresence
from .rank_blocks_by_size_interruption import RankBlocksBySizeInterruption
from .stack_three_bowls_contribution import StackThreeBowlsContribution
from .stack_three_bowls_copresence import StackThreeBowlsCopresence
from .stack_three_bowls_interruption import StackThreeBowlsInterruption
from .take_from_human_easy import TakeFromHumanEasy
from .take_from_human_safety import TakeFromHumanSafety
from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB
from ..task_bases.blocks_ranking_size import BlocksRankingSize
from ..task_bases.dump_bin import DumpBin
from ..task_bases.place_bread_in_basket import PlaceBreadInBasket
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ..task_bases.place_dual_shoes import PlaceDualShoes
from ..task_bases.place_food_in_skillet import PlaceFoodInSkillet
from ..task_bases.put_object_cabinet import PutObjectCabinet
from ..task_bases.stack_bowls_three import StackBowlsThree


class TaskVariant(str, Enum):
    """Scenario role used by the benchmark's task taxonomy.

    ``CUEING``/``CONTRIBUTION``/``INTERRUPTION``/``COPRESENCE`` are the four
    HARP-Bench conditions; ``BASE`` is a no-human family task.  Factory
    work-cell scenarios reuse ``CUEING``/``CONTRIBUTION`` but are not
    benchmark tasks.  ``INTENT``/``ASSIST``/``INTERRUPT``/``NEUTRAL`` are
    enum aliases kept for pre-HARP callers; their values are the new ones,
    and ``LEGACY_CATEGORY_VALUES`` maps the old stored strings.
    """

    BASE = "base"
    CUEING = "cueing"
    CONTRIBUTION = "contribution"
    INTERRUPTION = "interruption"
    COPRESENCE = "copresence"
    # Pre-HARP member names (aliases of the members above).
    INTENT = "cueing"
    ASSIST = "contribution"
    INTERRUPT = "interruption"
    NEUTRAL = "copresence"


class TaskRegistration(str, Enum):
    """Whether a name is canonical or retained for compatibility."""

    CANONICAL = "canonical"
    COMPATIBILITY = "compatibility"


@dataclass(frozen=True)
class TaskDescriptor:
    """One accepted normal-mode task name and its registry metadata.

    ``family`` is a reporting group, not necessarily a unique implementation
    core: factory scenarios share a setting, and xArm dump-bin scenarios share
    an objective with their Franka counterparts. Consumers must use ``name``
    as identity and must not assume ``(family, variant)`` is unique.

    ``no_human_name`` is set only for the ten families that support no-human
    collapse.  Exactly one descriptor per such family is marked
    ``no_human_canonical`` and supplies the class for that collapsed name.
    """

    name: str
    task_cls: type
    family: str
    variant: TaskVariant
    registration: TaskRegistration = TaskRegistration.CANONICAL
    variant_label: str | None = None
    no_human_name: str | None = None
    no_human_canonical: bool = False

    @property
    def instruction(self) -> str | None:
        """Return the resolved static instruction used by registry consumers."""

        resolver = getattr(self.task_cls, "default_instruction", None)
        if callable(resolver):
            return resolver()
        return self.instruction_template

    @property
    def instruction_template(self) -> str | None:
        """Return the raw class template, including any ``{object}`` slot."""

        return getattr(self.task_cls, "INSTRUCTION", None)

    @property
    def paper_label(self) -> tuple[str, str] | None:
        """Return ``(paper category, paper task name)``, or ``None``.

        Only the 37 benchmark tasks carry a paper label; no-human base tasks
        and factory scenarios return ``None``.
        """

        try:
            return paper_label(self.name)
        except KeyError:
            return None


@dataclass(frozen=True)
class TaskRegistry:
    """Validated descriptors and their legacy dictionary projections."""

    descriptors: tuple[TaskDescriptor, ...]
    descriptor_map: dict[str, TaskDescriptor]
    task_map: dict[str, type]
    task_aliases: dict[str, type]
    no_human_canonical_task_map: dict[str, type]
    no_human_variant_aliases: dict[str, str]
    no_human_task_map: dict[str, type]


def build_task_registry(descriptors: Iterable[TaskDescriptor]) -> TaskRegistry:
    """Validate descriptors and build the historical registry dictionaries."""

    descriptor_tuple = tuple(descriptors)
    descriptor_map: dict[str, TaskDescriptor] = {}
    task_map: dict[str, type] = {}
    task_aliases: dict[str, type] = {}

    for descriptor in descriptor_tuple:
        if not isinstance(descriptor, TaskDescriptor):
            raise TypeError(
                "Task registry entries must be TaskDescriptor instances, got "
                f"{type(descriptor).__name__}"
            )
        if not isinstance(descriptor.name, str) or not descriptor.name.strip():
            raise ValueError("Task descriptor names must not be empty")
        if not isinstance(descriptor.family, str) or not descriptor.family.strip():
            raise ValueError(
                f"Task descriptor {descriptor.name!r} must have a non-empty family"
            )
        if not isinstance(descriptor.variant, TaskVariant):
            raise TypeError(
                f"Task descriptor {descriptor.name!r} has invalid variant "
                f"{descriptor.variant!r}"
            )
        if not isinstance(descriptor.registration, TaskRegistration):
            raise TypeError(
                f"Task descriptor {descriptor.name!r} has invalid registration "
                f"{descriptor.registration!r}"
            )
        if descriptor.variant_label is not None and (
            not isinstance(descriptor.variant_label, str)
            or not descriptor.variant_label.strip()
        ):
            raise ValueError(
                f"Task descriptor {descriptor.name!r} has invalid variant_label"
            )
        if descriptor.no_human_name is not None and (
            not isinstance(descriptor.no_human_name, str)
            or not descriptor.no_human_name.strip()
        ):
            raise ValueError(
                f"Task descriptor {descriptor.name!r} has invalid no-human name"
            )
        if type(descriptor.no_human_canonical) is not bool:
            raise TypeError(
                f"Task descriptor {descriptor.name!r} no_human_canonical "
                "must be a bool"
            )
        if descriptor.name in descriptor_map:
            raise ValueError(f"Duplicate task descriptor name: {descriptor.name!r}")
        if not isinstance(descriptor.task_cls, type):
            raise TypeError(
                f"Task descriptor {descriptor.name!r} must reference a class"
            )
        descriptor_map[descriptor.name] = descriptor
        destination = (
            task_map
            if descriptor.registration is TaskRegistration.CANONICAL
            else task_aliases
        )
        destination[descriptor.name] = descriptor.task_cls

    no_human_canonical_task_map: dict[str, type] = {}
    for descriptor in descriptor_tuple:
        if not descriptor.no_human_canonical:
            continue
        if descriptor.registration is not TaskRegistration.CANONICAL:
            raise ValueError(
                f"No-human canonical provider {descriptor.name!r} must be canonical"
            )
        if descriptor.variant not in (TaskVariant.BASE, TaskVariant.ASSIST):
            raise ValueError(
                f"No-human canonical provider {descriptor.name!r} must be a "
                "base or assist task"
            )
        if descriptor.no_human_name is None:
            raise ValueError(
                f"No-human canonical provider {descriptor.name!r} has no target name"
            )
        if descriptor.no_human_name in no_human_canonical_task_map:
            raise ValueError(
                "Duplicate no-human canonical name: "
                f"{descriptor.no_human_name!r}"
            )
        named_descriptor = descriptor_map.get(descriptor.no_human_name)
        if named_descriptor is not None and named_descriptor is not descriptor:
            raise ValueError(
                f"No-human canonical provider {descriptor.name!r} conflicts "
                f"with registered task {descriptor.no_human_name!r}"
            )
        no_human_canonical_task_map[descriptor.no_human_name] = descriptor.task_cls

    no_human_variant_aliases: dict[str, str] = {}
    for descriptor in descriptor_tuple:
        canonical = descriptor.no_human_name
        if canonical is None:
            if descriptor.no_human_canonical:
                raise ValueError(
                    f"No-human canonical provider {descriptor.name!r} has no target name"
                )
            continue
        if descriptor.name == canonical and not descriptor.no_human_canonical:
            raise ValueError(
                f"No-human canonical name {descriptor.name!r} must be "
                "provided by that descriptor"
            )
        if canonical not in no_human_canonical_task_map:
            raise ValueError(
                f"Task descriptor {descriptor.name!r} has dangling no-human "
                f"target {canonical!r}"
            )
        provider = next(
            item
            for item in descriptor_tuple
            if item.no_human_canonical and item.no_human_name == canonical
        )
        if descriptor.family != provider.family:
            raise ValueError(
                f"Task descriptor {descriptor.name!r} cannot collapse across "
                f"families: {descriptor.family!r} -> {provider.family!r}"
            )
        if descriptor.variant is TaskVariant.INTENT:
            raise ValueError(
                f"Intent task {descriptor.name!r} cannot be a no-human variant"
            )
        if descriptor.name == canonical:
            if descriptor.task_cls is not no_human_canonical_task_map[canonical]:
                raise ValueError(
                    f"No-human base {descriptor.name!r} does not supply its "
                    "canonical class"
                )
            continue
        no_human_variant_aliases[descriptor.name] = canonical

    no_human_task_map = {
        **no_human_canonical_task_map,
        **{
            name: no_human_canonical_task_map[canonical]
            for name, canonical in no_human_variant_aliases.items()
        },
    }
    return TaskRegistry(
        descriptors=descriptor_tuple,
        descriptor_map=descriptor_map,
        task_map=task_map,
        task_aliases=task_aliases,
        no_human_canonical_task_map=no_human_canonical_task_map,
        no_human_variant_aliases=no_human_variant_aliases,
        no_human_task_map=no_human_task_map,
    )


def _descriptor(
    name: str,
    task_cls: type,
    family: str,
    variant: TaskVariant,
    *,
    registration: TaskRegistration = TaskRegistration.CANONICAL,
    variant_label: str | None = None,
    no_human_name: str | None = None,
    no_human_canonical: bool = False,
) -> TaskDescriptor:
    return TaskDescriptor(
        name=name,
        task_cls=task_cls,
        family=family,
        variant=variant,
        registration=registration,
        variant_label=variant_label,
        no_human_name=no_human_name,
        no_human_canonical=no_human_canonical,
    )


# Canonical order follows the historical TASK_MAP with the HARP-Bench names
# substituted (``categorize_objects_interruption``, formerly the
# ``categorize_interrupt`` compatibility name, is now canonical).  Each
# pre-HARP name follows its canonical descriptor as a compatibility
# registration of the same class (see ``LEGACY_TASK_NAMES``).  Filtering
# compatibility descriptors out leaves the canonical order.
TASK_DESCRIPTORS = (
    # Cueing / singleton tasks.
    _descriptor(
        "take_from_human_easy",
        TakeFromHumanEasy,
        "take_from_human",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "deliver_to_human",
        DeliverToHuman,
        "deliver_to_human",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "deliver_to_human_easy",
        DeliverToHuman,
        "deliver_to_human",
        TaskVariant.CUEING,
        registration=TaskRegistration.COMPATIBILITY,
    ),
    _descriptor(
        "take_from_human_safety",
        TakeFromHumanSafety,
        "take_from_human",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "feed_documents",
        FeedDocuments,
        "stamp_documents",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "stamp_documents",
        FeedDocuments,
        "stamp_documents",
        TaskVariant.CUEING,
        registration=TaskRegistration.COMPATIBILITY,
    ),
    _descriptor(
        "pour_water",
        PourWater,
        "pour_water",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "oil_bottle_recovery",
        OilBottleRecovery,
        "oil_bottle_recovery",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "open_microwave",
        OpenMicrowave,
        "open_microwave",
        TaskVariant.CUEING,
    ),

    # Categorize Objects.  The contribution task keeps its stored
    # ``cooperative`` variant label and supplies the no-human class.
    _descriptor(
        "categorize_objects_contribution",
        CategorizeObjectsContribution,
        "categorize",
        TaskVariant.CONTRIBUTION,
        variant_label="cooperative",
        no_human_name="categorize",
        no_human_canonical=True,
    ),
    _descriptor(
        "categorize_cooperative",
        CategorizeObjectsContribution,
        "categorize",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        variant_label="cooperative",
        no_human_name="categorize",
    ),
    _descriptor(
        "categorize_objects_interruption",
        CategorizeObjectsInterruption,
        "categorize",
        TaskVariant.INTERRUPTION,
        no_human_name="categorize",
    ),
    _descriptor(
        "categorize_interrupt",
        CategorizeObjectsInterruption,
        "categorize",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="categorize",
    ),
    _descriptor(
        "categorize_objects_copresence",
        CategorizeObjectsCopresence,
        "categorize",
        TaskVariant.COPRESENCE,
        no_human_name="categorize",
    ),
    _descriptor(
        "categorize_neutral",
        CategorizeObjectsCopresence,
        "categorize",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="categorize",
    ),

    # Base/no-human tasks present as distinct LeRobot instructions.
    _descriptor(
        "put_object_cabinet",
        PutObjectCabinet,
        "put_object_cabinet",
        TaskVariant.BASE,
        no_human_name="put_object_cabinet",
        no_human_canonical=True,
    ),
    _descriptor(
        "stack_bowls_three",
        StackBowlsThree,
        "stack_bowls_three",
        TaskVariant.BASE,
        no_human_name="stack_bowls_three",
        no_human_canonical=True,
    ),
    _descriptor(
        "place_bread_in_basket",
        PlaceBreadInBasket,
        "place_bread_in_basket",
        TaskVariant.BASE,
        no_human_name="place_bread_in_basket",
        no_human_canonical=True,
    ),
    _descriptor(
        "dump_bin",
        DumpBin,
        "dump_bin",
        TaskVariant.BASE,
        no_human_name="dump_bin",
        no_human_canonical=True,
    ),
    _descriptor(
        "place_burger_fries",
        PlaceBurgerFries,
        "place_burger_fries",
        TaskVariant.BASE,
        no_human_name="place_burger_fries",
        no_human_canonical=True,
    ),
    _descriptor(
        "place_dual_shoes",
        PlaceDualShoes,
        "place_dual_shoes",
        TaskVariant.BASE,
        no_human_name="place_dual_shoes",
        no_human_canonical=True,
    ),
    _descriptor(
        "place_food_in_skillet",
        PlaceFoodInSkillet,
        "place_food_in_skillet",
        TaskVariant.BASE,
        no_human_name="place_food_in_skillet",
        no_human_canonical=True,
    ),
    _descriptor(
        "blocks_ranking_rgb",
        BlocksRankingRGB,
        "blocks_ranking_rgb",
        TaskVariant.BASE,
        no_human_name="blocks_ranking_rgb",
        no_human_canonical=True,
    ),
    _descriptor(
        "blocks_ranking_size",
        BlocksRankingSize,
        "blocks_ranking_size",
        TaskVariant.BASE,
        no_human_name="blocks_ranking_size",
        no_human_canonical=True,
    ),

    # Contribution / interruption / co-presence task families.
    _descriptor(
        "put_object_in_cabinet_contribution",
        PutObjectInCabinetContribution,
        "put_object_cabinet",
        TaskVariant.CONTRIBUTION,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "put_object_cabinet_assist",
        PutObjectInCabinetContribution,
        "put_object_cabinet",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "put_object_in_cabinet_interruption",
        PutObjectInCabinetInterruption,
        "put_object_cabinet",
        TaskVariant.INTERRUPTION,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "put_object_cabinet_interrupt",
        PutObjectInCabinetInterruption,
        "put_object_cabinet",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "put_object_in_cabinet_copresence",
        PutObjectInCabinetCopresence,
        "put_object_cabinet",
        TaskVariant.COPRESENCE,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "put_object_cabinet_neutral",
        PutObjectInCabinetCopresence,
        "put_object_cabinet",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="put_object_cabinet",
    ),
    _descriptor(
        "stack_three_bowls_contribution",
        StackThreeBowlsContribution,
        "stack_bowls_three",
        TaskVariant.CONTRIBUTION,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "stack_bowls_three_assist",
        StackThreeBowlsContribution,
        "stack_bowls_three",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "stack_three_bowls_interruption",
        StackThreeBowlsInterruption,
        "stack_bowls_three",
        TaskVariant.INTERRUPTION,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "stack_bowls_three_interrupt",
        StackThreeBowlsInterruption,
        "stack_bowls_three",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "stack_three_bowls_copresence",
        StackThreeBowlsCopresence,
        "stack_bowls_three",
        TaskVariant.COPRESENCE,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "stack_bowls_three_neutral",
        StackThreeBowlsCopresence,
        "stack_bowls_three",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="stack_bowls_three",
    ),
    _descriptor(
        "place_bread_in_basket_contribution",
        PlaceBreadInBasketContribution,
        "place_bread_in_basket",
        TaskVariant.CONTRIBUTION,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "place_bread_in_basket_assist",
        PlaceBreadInBasketContribution,
        "place_bread_in_basket",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "place_bread_in_basket_interruption",
        PlaceBreadInBasketInterruption,
        "place_bread_in_basket",
        TaskVariant.INTERRUPTION,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "place_bread_in_basket_interrupt",
        PlaceBreadInBasketInterruption,
        "place_bread_in_basket",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "place_bread_in_basket_copresence",
        PlaceBreadInBasketCopresence,
        "place_bread_in_basket",
        TaskVariant.COPRESENCE,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "place_bread_in_basket_neutral",
        PlaceBreadInBasketCopresence,
        "place_bread_in_basket",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_bread_in_basket",
    ),
    _descriptor(
        "dump_bin_contribution",
        DumpBinContribution,
        "dump_bin",
        TaskVariant.CONTRIBUTION,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "dump_bin_assist",
        DumpBinContribution,
        "dump_bin",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "dump_bin_interruption",
        DumpBinInterruption,
        "dump_bin",
        TaskVariant.INTERRUPTION,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "dump_bin_interrupt",
        DumpBinInterruption,
        "dump_bin",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "dump_bin_copresence",
        DumpBinCopresence,
        "dump_bin",
        TaskVariant.COPRESENCE,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "dump_bin_neutral",
        DumpBinCopresence,
        "dump_bin",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="dump_bin",
    ),
    _descriptor(
        "place_burger_and_fries_contribution",
        PlaceBurgerAndFriesContribution,
        "place_burger_fries",
        TaskVariant.CONTRIBUTION,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_burger_fries_assist",
        PlaceBurgerAndFriesContribution,
        "place_burger_fries",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_burger_and_fries_interruption",
        PlaceBurgerAndFriesInterruption,
        "place_burger_fries",
        TaskVariant.INTERRUPTION,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_burger_fries_interrupt",
        PlaceBurgerAndFriesInterruption,
        "place_burger_fries",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_burger_and_fries_copresence",
        PlaceBurgerAndFriesCopresence,
        "place_burger_fries",
        TaskVariant.COPRESENCE,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_burger_fries_neutral",
        PlaceBurgerAndFriesCopresence,
        "place_burger_fries",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_burger_fries",
    ),
    _descriptor(
        "place_pair_of_shoes_contribution",
        PlacePairOfShoesContribution,
        "place_dual_shoes",
        TaskVariant.CONTRIBUTION,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_dual_shoes_assist",
        PlacePairOfShoesContribution,
        "place_dual_shoes",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_pair_of_shoes_interruption",
        PlacePairOfShoesInterruption,
        "place_dual_shoes",
        TaskVariant.INTERRUPTION,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_dual_shoes_interrupt",
        PlacePairOfShoesInterruption,
        "place_dual_shoes",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_pair_of_shoes_copresence",
        PlacePairOfShoesCopresence,
        "place_dual_shoes",
        TaskVariant.COPRESENCE,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_dual_shoes_neutral",
        PlacePairOfShoesCopresence,
        "place_dual_shoes",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_dual_shoes",
    ),
    _descriptor(
        "place_food_in_skillet_contribution",
        PlaceFoodInSkilletContribution,
        "place_food_in_skillet",
        TaskVariant.CONTRIBUTION,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "place_food_in_skillet_assist",
        PlaceFoodInSkilletContribution,
        "place_food_in_skillet",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "place_food_in_skillet_interruption",
        PlaceFoodInSkilletInterruption,
        "place_food_in_skillet",
        TaskVariant.INTERRUPTION,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "place_food_in_skillet_interrupt",
        PlaceFoodInSkilletInterruption,
        "place_food_in_skillet",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "place_food_in_skillet_copresence",
        PlaceFoodInSkilletCopresence,
        "place_food_in_skillet",
        TaskVariant.COPRESENCE,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "place_food_in_skillet_neutral",
        PlaceFoodInSkilletCopresence,
        "place_food_in_skillet",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="place_food_in_skillet",
    ),
    _descriptor(
        "rank_blocks_by_color_contribution",
        RankBlocksByColorContribution,
        "blocks_ranking_rgb",
        TaskVariant.CONTRIBUTION,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "blocks_ranking_rgb_assist",
        RankBlocksByColorContribution,
        "blocks_ranking_rgb",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "rank_blocks_by_color_interruption",
        RankBlocksByColorInterruption,
        "blocks_ranking_rgb",
        TaskVariant.INTERRUPTION,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "blocks_ranking_rgb_interrupt",
        RankBlocksByColorInterruption,
        "blocks_ranking_rgb",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "rank_blocks_by_color_copresence",
        RankBlocksByColorCopresence,
        "blocks_ranking_rgb",
        TaskVariant.COPRESENCE,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "blocks_ranking_rgb_neutral",
        RankBlocksByColorCopresence,
        "blocks_ranking_rgb",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_rgb",
    ),
    _descriptor(
        "rank_blocks_by_size_contribution",
        RankBlocksBySizeContribution,
        "blocks_ranking_size",
        TaskVariant.CONTRIBUTION,
        no_human_name="blocks_ranking_size",
    ),
    _descriptor(
        "blocks_ranking_size_assist",
        RankBlocksBySizeContribution,
        "blocks_ranking_size",
        TaskVariant.CONTRIBUTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_size",
    ),
    _descriptor(
        "rank_blocks_by_size_interruption",
        RankBlocksBySizeInterruption,
        "blocks_ranking_size",
        TaskVariant.INTERRUPTION,
        no_human_name="blocks_ranking_size",
    ),
    _descriptor(
        "blocks_ranking_size_interrupt",
        RankBlocksBySizeInterruption,
        "blocks_ranking_size",
        TaskVariant.INTERRUPTION,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_size",
    ),
    _descriptor(
        "rank_blocks_by_size_copresence",
        RankBlocksBySizeCopresence,
        "blocks_ranking_size",
        TaskVariant.COPRESENCE,
        no_human_name="blocks_ranking_size",
    ),
    _descriptor(
        "blocks_ranking_size_neutral",
        RankBlocksBySizeCopresence,
        "blocks_ranking_size",
        TaskVariant.COPRESENCE,
        registration=TaskRegistration.COMPATIBILITY,
        no_human_name="blocks_ranking_size",
    ),

    # Factory work-cell HRC family (docs/factory_task_inventory.md).
    _descriptor(
        "factory_inspect_pack",
        FactoryInspectPack,
        "factory",
        TaskVariant.CUEING,
    ),
    _descriptor(
        "factory_kit_packing",
        FactoryKitPacking,
        "factory",
        TaskVariant.CONTRIBUTION,
    ),
    _descriptor(
        "factory_line_feeding",
        FactoryLineFeeding,
        "factory",
        TaskVariant.CUEING,
    ),
)


TASK_REGISTRY = build_task_registry(TASK_DESCRIPTORS)
TASK_DESCRIPTOR_MAP = TASK_REGISTRY.descriptor_map
TASK_MAP = TASK_REGISTRY.task_map
TASK_ALIASES = TASK_REGISTRY.task_aliases
NO_HUMAN_CANONICAL_TASK_MAP = TASK_REGISTRY.no_human_canonical_task_map
NO_HUMAN_VARIANT_ALIASES = TASK_REGISTRY.no_human_variant_aliases
NO_HUMAN_TASK_MAP = TASK_REGISTRY.no_human_task_map
