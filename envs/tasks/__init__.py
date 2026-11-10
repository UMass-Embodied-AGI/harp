"""Task registry for the HARP-Bench tasks described in ``README.md``."""

from .registry import (
    BENCHMARK_TASKS,
    BlocksRankingRGB,
    BlocksRankingSize,
    build_task_registry,
    canonical_category,
    canonical_task_name,
    CategorizeObjectsContribution,
    CategorizeObjectsCopresence,
    CategorizeObjectsInterruption,
    CATEGORY_DISPLAY,
    category_display,
    DeliverToHuman,
    DumpBin,
    DumpBinContribution,
    DumpBinCopresence,
    DumpBinInterruption,
    FactoryInspectPack,
    FeedDocuments,
    FactoryKitPacking,
    FactoryLineFeeding,
    FAMILY_DISPLAY,
    LEGACY_CATEGORY_VALUES,
    LEGACY_TASK_NAMES,
    NO_HUMAN_CANONICAL_TASK_MAP,
    NO_HUMAN_TASK_MAP,
    NO_HUMAN_VARIANT_ALIASES,
    OilBottleRecovery,
    OpenMicrowave,
    PAPER_CATEGORY_ORDER,
    paper_label,
    PAPER_NAME,
    paper_task_label,
    PlaceBreadInBasket,
    PlaceBreadInBasketContribution,
    PlaceBreadInBasketCopresence,
    PlaceBreadInBasketInterruption,
    PlaceBurgerAndFriesContribution,
    PlaceBurgerAndFriesCopresence,
    PlaceBurgerAndFriesInterruption,
    PlaceBurgerFries,
    PlaceDualShoes,
    PlaceFoodInSkillet,
    PlaceFoodInSkilletContribution,
    PlaceFoodInSkilletCopresence,
    PlaceFoodInSkilletInterruption,
    PlacePairOfShoesContribution,
    PlacePairOfShoesCopresence,
    PlacePairOfShoesInterruption,
    PourWater,
    PutObjectCabinet,
    PutObjectInCabinetContribution,
    PutObjectInCabinetCopresence,
    PutObjectInCabinetInterruption,
    RankBlocksByColorContribution,
    RankBlocksByColorCopresence,
    RankBlocksByColorInterruption,
    RankBlocksBySizeContribution,
    RankBlocksBySizeCopresence,
    RankBlocksBySizeInterruption,
    StackBowlsThree,
    StackThreeBowlsContribution,
    StackThreeBowlsCopresence,
    StackThreeBowlsInterruption,
    TakeFromHumanEasy,
    TakeFromHumanSafety,
    TASK_ALIASES,
    task_category,
    TASK_DESCRIPTOR_MAP,
    TASK_DESCRIPTORS,
    TASK_DISPLAY_OVERRIDES,
    TASK_MAP,
    TASK_REGISTRY,
    TaskDescriptor,
    TaskRegistration,
    TaskRegistry,
    TaskVariant,
)


# Pre-HARP class names, kept importable for external code.
DeliverToHumanEasy = DeliverToHuman
StampDocuments = FeedDocuments
CategorizeCooperative = CategorizeObjectsContribution
CategorizeInterrupt = CategorizeObjectsInterruption
CategorizeNeutral = CategorizeObjectsCopresence
PutObjectCabinetAssist = PutObjectInCabinetContribution
PutObjectCabinetInterrupt = PutObjectInCabinetInterruption
PutObjectCabinetNeutral = PutObjectInCabinetCopresence
StackBowlsThreeAssist = StackThreeBowlsContribution
StackBowlsThreeInterrupt = StackThreeBowlsInterruption
StackBowlsThreeNeutral = StackThreeBowlsCopresence
PlaceBreadInBasketAssist = PlaceBreadInBasketContribution
PlaceBreadInBasketInterrupt = PlaceBreadInBasketInterruption
PlaceBreadInBasketNeutral = PlaceBreadInBasketCopresence
DumpBinAssist = DumpBinContribution
DumpBinInterrupt = DumpBinInterruption
DumpBinNeutral = DumpBinCopresence
PlaceBurgerFriesAssist = PlaceBurgerAndFriesContribution
PlaceBurgerFriesInterrupt = PlaceBurgerAndFriesInterruption
PlaceBurgerFriesNeutral = PlaceBurgerAndFriesCopresence
PlaceDualShoesAssist = PlacePairOfShoesContribution
PlaceDualShoesInterrupt = PlacePairOfShoesInterruption
PlaceDualShoesNeutral = PlacePairOfShoesCopresence
PlaceFoodInSkilletAssist = PlaceFoodInSkilletContribution
PlaceFoodInSkilletInterrupt = PlaceFoodInSkilletInterruption
PlaceFoodInSkilletNeutral = PlaceFoodInSkilletCopresence
BlocksRankingRGBAssist = RankBlocksByColorContribution
BlocksRankingRGBInterrupt = RankBlocksByColorInterruption
BlocksRankingRGBNeutral = RankBlocksByColorCopresence
BlocksRankingSizeAssist = RankBlocksBySizeContribution
BlocksRankingSizeInterrupt = RankBlocksBySizeInterruption
BlocksRankingSizeNeutral = RankBlocksBySizeCopresence
LEGACY_CLASS_NAMES = {
    "DeliverToHumanEasy": "DeliverToHuman",
    "StampDocuments": "FeedDocuments",
    "CategorizeCooperative": "CategorizeObjectsContribution",
    "CategorizeInterrupt": "CategorizeObjectsInterruption",
    "CategorizeNeutral": "CategorizeObjectsCopresence",
    "PutObjectCabinetAssist": "PutObjectInCabinetContribution",
    "PutObjectCabinetInterrupt": "PutObjectInCabinetInterruption",
    "PutObjectCabinetNeutral": "PutObjectInCabinetCopresence",
    "StackBowlsThreeAssist": "StackThreeBowlsContribution",
    "StackBowlsThreeInterrupt": "StackThreeBowlsInterruption",
    "StackBowlsThreeNeutral": "StackThreeBowlsCopresence",
    "PlaceBreadInBasketAssist": "PlaceBreadInBasketContribution",
    "PlaceBreadInBasketInterrupt": "PlaceBreadInBasketInterruption",
    "PlaceBreadInBasketNeutral": "PlaceBreadInBasketCopresence",
    "DumpBinAssist": "DumpBinContribution",
    "DumpBinInterrupt": "DumpBinInterruption",
    "DumpBinNeutral": "DumpBinCopresence",
    "PlaceBurgerFriesAssist": "PlaceBurgerAndFriesContribution",
    "PlaceBurgerFriesInterrupt": "PlaceBurgerAndFriesInterruption",
    "PlaceBurgerFriesNeutral": "PlaceBurgerAndFriesCopresence",
    "PlaceDualShoesAssist": "PlacePairOfShoesContribution",
    "PlaceDualShoesInterrupt": "PlacePairOfShoesInterruption",
    "PlaceDualShoesNeutral": "PlacePairOfShoesCopresence",
    "PlaceFoodInSkilletAssist": "PlaceFoodInSkilletContribution",
    "PlaceFoodInSkilletInterrupt": "PlaceFoodInSkilletInterruption",
    "PlaceFoodInSkilletNeutral": "PlaceFoodInSkilletCopresence",
    "BlocksRankingRGBAssist": "RankBlocksByColorContribution",
    "BlocksRankingRGBInterrupt": "RankBlocksByColorInterruption",
    "BlocksRankingRGBNeutral": "RankBlocksByColorCopresence",
    "BlocksRankingSizeAssist": "RankBlocksBySizeContribution",
    "BlocksRankingSizeInterrupt": "RankBlocksBySizeInterruption",
    "BlocksRankingSizeNeutral": "RankBlocksBySizeCopresence",
}


def resolve_task_class(task_name: str, no_human: bool = False):
    """Return (resolved_task_name, TaskClass), applying no-human collapse."""
    if no_human:
        canonical = NO_HUMAN_VARIANT_ALIASES.get(task_name, task_name)
        if canonical not in NO_HUMAN_CANONICAL_TASK_MAP:
            supported = sorted(NO_HUMAN_CANONICAL_TASK_MAP)
            raise ValueError(
                f"Task {task_name!r} has no no-human variant. "
                f"No-human is only defined for the contribution/interruption/"
                f"co-presence families: {supported}"
            )
        return canonical, NO_HUMAN_CANONICAL_TASK_MAP[canonical]

    if task_name in TASK_MAP:
        return task_name, TASK_MAP[task_name]
    if task_name in TASK_ALIASES:
        return task_name, TASK_ALIASES[task_name]
    available = sorted([*TASK_MAP.keys(), *TASK_ALIASES.keys()])
    raise ValueError(f"Unknown task: {task_name}. Available: {available}")
