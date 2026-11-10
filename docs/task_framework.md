# Task framework

The target release architecture separates a task's robot objective from its
human interaction mode. The descriptor and lifecycle foundation are in
place. Several task families now use shared objective cores and composed
controllers; remaining registered interaction behavior still lives in legacy
classes and mixins:

```text
registered scenario
  = task-family core
  + interaction controller
  + robot embodiment/calibration
  + compatibility descriptor
```

The migration reserves inheritance for the robot task family and composes
human behavior through a controller. This will avoid using an interrupt class
as the parent of a neutral task, or copying an entire family implementation
just to change how a human participates.

## Public task model

`envs.tasks.TaskDescriptor` is the authoritative registry record. A descriptor
binds the public name, reporting family, semantic variant, implementation
class, and no-human compatibility mapping.

`family` is deliberately a reporting group, not a unique implementation key:
factory scenarios share the `factory` setting and calibrated xArm dump-bin
scenarios share the `dump_bin` objective. Public identity remains `name`; code
must not assume `(family, variant)` is unique. A later family-core extraction
can add an explicit implementation-core/embodiment field without changing
accepted task names.

The historical exports remain available:

- `TASK_MAP`: canonical runtime names
- `TASK_ALIASES`: accepted compatibility names
- `NO_HUMAN_TASK_MAP` and `NO_HUMAN_VARIANT_ALIASES`: collapsed no-human names
- `resolve_task_class(name, no_human=False)`: stable resolver entry point

`categorize_objects_contribution` (pre-HARP `categorize_cooperative`) is the
Contribution member of the Categorize Objects family. Its descriptor keeps
`variant_label="cooperative"` because that label is already stored in
datasets, manifests, and checkpoints. `categorize_objects_interruption`
(formerly the compatibility name `categorize_interrupt`) is now an ordinary
canonical task. All three categorization scenarios are siblings over
`CategorizeTaskCore`: contribution uses `AssistEvalModeController`,
interruption uses `CategorizeInterruptController`, and co-presence keeps the
neutral-table-work mixin over the same robot objective. Co-presence no
longer inherits or executes the interruption scenario. A versioned
compatibility controller consumes the two historical, behavior-irrelevant
reset draws so existing seeded scene layouts remain reproducible without
running Inspect motion.

## Task names

Public task names follow the HARP-Bench paper. A Cueing task uses the paper
task name; every other benchmark task is `<paper task>_<condition>` with
condition `contribution`, `interruption`, or `copresence`. The same strings
are the `TaskVariant` values and the eval-set manifest `category` values
(`cueing`, `contribution`, `interruption`, `copresence`). No-human base
tasks (`put_object_cabinet`, `blocks_ranking_rgb`, ..., and the collapsed
`categorize`), their registry `family` values, and the factory scenarios keep
their historical names; factory tasks reuse the `cueing`/`contribution`
variant values but are not benchmark tasks.

[`envs/tasks/paper_labels.py`](../envs/tasks/paper_labels.py) holds the
tables, and `envs.tasks` re-exports them:

- `BENCHMARK_TASKS`: the 37 benchmark tasks (the rows of
  `eval_sets/eval_v3_sampled`) in paper order, each with its registry family
  and condition.
- `CATEGORY_DISPLAY` (condition value → paper name, in `PAPER_CATEGORY_ORDER`)
  and `FAMILY_DISPLAY` / `TASK_DISPLAY_OVERRIDES` (paper task names).
- `paper_label(task_name) -> (condition, paper task name)`; it accepts legacy
  names and raises `KeyError` for no-human base and factory tasks.
  `TaskDescriptor.paper_label` returns `None` for those.
- `LEGACY_TASK_NAMES` (pre-HARP name → current name), `canonical_task_name`,
  `LEGACY_CATEGORY_VALUES` (`intent`/`assist`/`interrupt`/`neutral` → new
  value), and `canonical_category`.

| Paper condition | Paper task | Task name |
|---|---|---|
| Cueing | Deliver to Human | `deliver_to_human` |
| Cueing | Pour Water | `pour_water` |
| Cueing | Oil Bottle Recovery | `oil_bottle_recovery` |
| Cueing | Open Microwave | `open_microwave` |
| Cueing | Feed Documents | `feed_documents` |
| Cueing | Take from Human (Easy) | `take_from_human_easy` |
| Cueing | Take from Human (Safety) | `take_from_human_safety` |
| Contribution | Categorize Objects | `categorize_objects_contribution` |
| Contribution | Put Object in Cabinet | `put_object_in_cabinet_contribution` |
| Contribution | Stack Three Bowls | `stack_three_bowls_contribution` |
| Contribution | Place Bread in Basket | `place_bread_in_basket_contribution` |
| Contribution | Dump Bin | `dump_bin_contribution` |
| Contribution | Place Burger and Fries | `place_burger_and_fries_contribution` |
| Contribution | Place Pair of Shoes | `place_pair_of_shoes_contribution` |
| Contribution | Place Food in Skillet | `place_food_in_skillet_contribution` |
| Contribution | Rank Blocks by Color | `rank_blocks_by_color_contribution` |
| Contribution | Rank Blocks by Size | `rank_blocks_by_size_contribution` |
| Interruption | Categorize Objects | `categorize_objects_interruption` |
| Interruption | Put Object in Cabinet | `put_object_in_cabinet_interruption` |
| Interruption | Stack Three Bowls | `stack_three_bowls_interruption` |
| Interruption | Place Bread in Basket | `place_bread_in_basket_interruption` |
| Interruption | Dump Bin | `dump_bin_interruption` |
| Interruption | Place Burger and Fries | `place_burger_and_fries_interruption` |
| Interruption | Place Pair of Shoes | `place_pair_of_shoes_interruption` |
| Interruption | Place Food in Skillet | `place_food_in_skillet_interruption` |
| Interruption | Rank Blocks by Color | `rank_blocks_by_color_interruption` |
| Interruption | Rank Blocks by Size | `rank_blocks_by_size_interruption` |
| Co-presence | Categorize Objects | `categorize_objects_copresence` |
| Co-presence | Put Object in Cabinet | `put_object_in_cabinet_copresence` |
| Co-presence | Stack Three Bowls | `stack_three_bowls_copresence` |
| Co-presence | Place Bread in Basket | `place_bread_in_basket_copresence` |
| Co-presence | Dump Bin | `dump_bin_copresence` |
| Co-presence | Place Burger and Fries | `place_burger_and_fries_copresence` |
| Co-presence | Place Pair of Shoes | `place_pair_of_shoes_copresence` |
| Co-presence | Place Food in Skillet | `place_food_in_skillet_copresence` |
| Co-presence | Rank Blocks by Color | `rank_blocks_by_color_copresence` |
| Co-presence | Rank Blocks by Size | `rank_blocks_by_size_copresence` |

### Pre-HARP names

Every pre-HARP name is a `TaskRegistration.COMPATIBILITY` descriptor of the
same class (`TASK_ALIASES`), with the same instruction and no-human
collapse, so `resolve_task_class(old_name)` keeps working for old commands,
manifests, run directories, checkpoints, and LeRobot datasets. Scripts that
route by task or condition (`scripts/eval_expert_set.py`,
`scripts/collect.py`, `scripts/eval_vla.py`, `scripts/vla_train37/`) look up
the registry variant or canonicalize the name first, so old and new names
behave identically. `TaskVariant.INTENT`/`ASSIST`/`INTERRUPT`/`NEUTRAL`
remain enum aliases of the new members.

| Pre-HARP task name | Current task name |
|---|---|
| `deliver_to_human_easy` | `deliver_to_human` |
| `stamp_documents` | `feed_documents` |
| `categorize_cooperative` | `categorize_objects_contribution` |
| `put_object_cabinet_assist` | `put_object_in_cabinet_contribution` |
| `stack_bowls_three_assist` | `stack_three_bowls_contribution` |
| `place_bread_in_basket_assist` | `place_bread_in_basket_contribution` |
| `dump_bin_assist` | `dump_bin_contribution` |
| `place_burger_fries_assist` | `place_burger_and_fries_contribution` |
| `place_dual_shoes_assist` | `place_pair_of_shoes_contribution` |
| `place_food_in_skillet_assist` | `place_food_in_skillet_contribution` |
| `blocks_ranking_rgb_assist` | `rank_blocks_by_color_contribution` |
| `blocks_ranking_size_assist` | `rank_blocks_by_size_contribution` |
| `categorize_interrupt` | `categorize_objects_interruption` |
| `put_object_cabinet_interrupt` | `put_object_in_cabinet_interruption` |
| `stack_bowls_three_interrupt` | `stack_three_bowls_interruption` |
| `place_bread_in_basket_interrupt` | `place_bread_in_basket_interruption` |
| `dump_bin_interrupt` | `dump_bin_interruption` |
| `place_burger_fries_interrupt` | `place_burger_and_fries_interruption` |
| `place_dual_shoes_interrupt` | `place_pair_of_shoes_interruption` |
| `place_food_in_skillet_interrupt` | `place_food_in_skillet_interruption` |
| `blocks_ranking_rgb_interrupt` | `rank_blocks_by_color_interruption` |
| `blocks_ranking_size_interrupt` | `rank_blocks_by_size_interruption` |
| `categorize_neutral` | `categorize_objects_copresence` |
| `put_object_cabinet_neutral` | `put_object_in_cabinet_copresence` |
| `stack_bowls_three_neutral` | `stack_three_bowls_copresence` |
| `place_bread_in_basket_neutral` | `place_bread_in_basket_copresence` |
| `dump_bin_neutral` | `dump_bin_copresence` |
| `place_burger_fries_neutral` | `place_burger_and_fries_copresence` |
| `place_dual_shoes_neutral` | `place_pair_of_shoes_copresence` |
| `place_food_in_skillet_neutral` | `place_food_in_skillet_copresence` |
| `blocks_ranking_rgb_neutral` | `rank_blocks_by_color_copresence` |
| `blocks_ranking_size_neutral` | `rank_blocks_by_size_copresence` |

Task modules and classes were renamed to match (`envs/tasks/<task name>.py`).
The old class names stay importable from `envs.tasks` (`LEGACY_CLASS_NAMES`);
the old module paths do not.

| Pre-HARP class | Current class |
|---|---|
| `DeliverToHumanEasy` | `DeliverToHuman` |
| `StampDocuments` | `FeedDocuments` |
| `CategorizeCooperative` | `CategorizeObjectsContribution` |
| `CategorizeInterrupt` | `CategorizeObjectsInterruption` |
| `CategorizeNeutral` | `CategorizeObjectsCopresence` |
| `PutObjectCabinetAssist` | `PutObjectInCabinetContribution` |
| `PutObjectCabinetInterrupt` | `PutObjectInCabinetInterruption` |
| `PutObjectCabinetNeutral` | `PutObjectInCabinetCopresence` |
| `StackBowlsThreeAssist` | `StackThreeBowlsContribution` |
| `StackBowlsThreeInterrupt` | `StackThreeBowlsInterruption` |
| `StackBowlsThreeNeutral` | `StackThreeBowlsCopresence` |
| `PlaceBreadInBasketAssist` | `PlaceBreadInBasketContribution` |
| `PlaceBreadInBasketInterrupt` | `PlaceBreadInBasketInterruption` |
| `PlaceBreadInBasketNeutral` | `PlaceBreadInBasketCopresence` |
| `DumpBinAssist` | `DumpBinContribution` |
| `DumpBinInterrupt` | `DumpBinInterruption` |
| `DumpBinNeutral` | `DumpBinCopresence` |
| `PlaceBurgerFriesAssist` | `PlaceBurgerAndFriesContribution` |
| `PlaceBurgerFriesInterrupt` | `PlaceBurgerAndFriesInterruption` |
| `PlaceBurgerFriesNeutral` | `PlaceBurgerAndFriesCopresence` |
| `PlaceDualShoesAssist` | `PlacePairOfShoesContribution` |
| `PlaceDualShoesInterrupt` | `PlacePairOfShoesInterruption` |
| `PlaceDualShoesNeutral` | `PlacePairOfShoesCopresence` |
| `PlaceFoodInSkilletAssist` | `PlaceFoodInSkilletContribution` |
| `PlaceFoodInSkilletInterrupt` | `PlaceFoodInSkilletInterruption` |
| `PlaceFoodInSkilletNeutral` | `PlaceFoodInSkilletCopresence` |
| `BlocksRankingRGBAssist` | `RankBlocksByColorContribution` |
| `BlocksRankingRGBInterrupt` | `RankBlocksByColorInterruption` |
| `BlocksRankingRGBNeutral` | `RankBlocksByColorCopresence` |
| `BlocksRankingSizeAssist` | `RankBlocksBySizeContribution` |
| `BlocksRankingSizeInterrupt` | `RankBlocksBySizeInterruption` |
| `BlocksRankingSizeNeutral` | `RankBlocksBySizeCopresence` |

## Interaction migration seam

Controllers in `envs.interaction` currently expose a small,
simulator-independent phase-0 lifecycle:

1. `on_episode_seeded`
2. `augment_scene_before_build`
3. `on_avatar_created_before_build`
4. `on_base_reset_ready`
5. `after_base_reset_observation`
6. `on_policy_tick`
7. `before_base_policy_action` / `after_base_policy_action`
8. `on_base_sim_step`
9. `on_scripted_rollout_tick` (expert-eval parent rollouts; replaces the
   runner's former probing of private `_eval_*` task hooks)

The names deliberately describe the `BaseTask` core boundary. Legacy MRO
wrappers can still execute before or after these hooks; for example,
`on_base_reset_ready` runs before an outer legacy mixin's post-`super()` reset
work. A controller is created lazily on first reset/access, after subclass
construction. New randomness uses the namespaced `episode_rng` helper so
family layout sampling remains unchanged. A compatibility controller may
explicitly retain the legacy global NumPy stream when changing it would alter
already-published seeded scenes; that exception is versioned and covered by
RNG-parity tests. Every randomized controller declares a stable namespace;
class or module names are not used because cleanup renames must not change
seeded episodes.

`before_base_policy_action` returns a versioned, JSON-friendly
`PolicyDirective` with a semantic disposition: proceed, yield, retreat, or
resume. During phase 0 only `proceed` is executable. Any other disposition
fails closed instead of silently executing the original action. Directive
enforcement is active. Migrated families may delegate their legacy metric
fields to a controller, but outermost metric/planner-obstacle composition and
general scripted-expert phase hooks are still added family-by-family with their
runner tests; they are not claimed as universal framework behavior yet.

Unmigrated task classes receive `LegacyInteractionController`, a no-op adapter.
The migrated RGB and size block-ranking variants inherit their family core
directly. Assist scenarios return one shared `BlockRankingAssistController`;
interrupt scenarios return one shared `BlockRankingInterruptController` and
use the stateless `InspectMotionRuntime`. The dump-bin, bread, burger/fries,
stack-bowls, and skillet assist variants now use `AssistEvalModeController`,
which owns their former reset/action mixin boundary while their distinct
geometry and scripted choreography remain family-specific. Each inherits its
family core directly. Their public names, seeded
global-RNG order, initial-observation timing, expert choreography, and
evaluation fields remain compatible. The old `EvalModeAvatarMixin` and
`InspectMotionMixin` remain compatibility adapters for external/unmigrated
extensions. No registered task now inherits `InspectMotionMixin`: dump-bin,
bread-basket, and burger/fries interrupts return family-specific controllers
and reach the same stateless `InspectMotionRuntime` through a thin
compatibility facade. In this narrow slice their task classes still own all
reset follow-up, policy-action, scripted-play, evaluation, RNG, and task-facing
state; the controllers intentionally leave policy and scripted-rollout ticks
disabled.
`DeliverToHuman`, `TakeFromHumanEasy`, `PourWater`, and
`TakeFromHumanSafety` are migrated from `EvalModeAvatarMixin` to
`IntentEvalModeController`. The controller preserves the legacy
post-observation avatar start and the 1-based policy counter that advances
before BaseTask's success short-circuit or robot command. Each task still owns
its handoff/drinking logic, RNG, expert choreography, state fields, and
evaluation metrics. No registered task now inherits `EvalModeAvatarMixin`;
the class remains a thin compatibility adapter for external extensions.
Registered classes with a direct `step_sim` implementation call the same
monotonic notification helper; the counter is independent of video recording.

The block-ranking, dump-bin, bread-basket, base dual-shoes, stack-bowls, and
cabinet family cores also share `BaseTask`'s rectangular-table builder. Each
family passes its authored thickness and ordered world-space leg coordinates,
preserving calibrated real/xArm layouts while keeping procedural and debug
mesh-variant footprints on one geometry contract. Cabinet base, assist, and
interrupt scenarios are sibling classes over `CabinetTaskCore`; the assist and
interrupt scenarios provide family-specific interaction controllers, while the
neutral scenario layers its avatar behavior over the public base task. The
fixed cabinet assist/interrupt table configurations explicitly disable debug
table variants. Dual-shoes base, assist, and interrupt scenarios likewise are
sibling classes over `DualShoesTaskCore`. `DualShoesAssistController` owns the
assist reset follow-up and policy ticks. The interrupt scenario reaches inspect
metadata through `InspectMotionInterruptFacade` and
`DualShoesInterruptController`, while retaining its task-local policy-action
replacement wrapper; its neutral scenario still layers avatar behavior over
the public base task. The fixed assist table profile disables debug variants.
Skillet scenarios are siblings over `SkilletTaskCore`: the assist scenario
keeps its authored food allocation and Cartesian release retreat while sharing
the robot mechanics, the interrupt scenario reaches Inspect through
`SkilletInterruptController`, and neutral table work layers over the same
objective core. The interrupt's task-local action wrapper remains responsible
for retreat command replacement before the robot action is executed.

The neutral-avatar core, dump-bin interrupt, and three factory tasks retain
their task-facing avatar-obstacle wrappers and planner refresh cadence while
sharing one pure capsule-to-point sampler. Its capsule traversal, axial and
azimuthal sample order, degenerate-segment handling, inflation, and float64
return contract remain the legacy behavior.

The first bounded neutral-monolith slice separates `NeutralTaskSpec`, portable
motion-asset paths, and deterministic clip-array transforms into small modules.
`NeutralAvatarTableWorkMixin` re-exports the historical spec/path names and
keeps every clip-install method as a thin delegate. Config and subclass
overrides, destination names, event-frame remapping, in-place job mutations,
and NumPy RNG state therefore remain compatible; layout, lifecycle, simulator,
and step-loop ownership are unchanged.

## Composition status and compatibility

Every assist and interrupt variant composes its human-side behavior through a
family controller under `envs/interaction/` (`place_bread_in_basket`,
`dump_bin`, `place_burger_fries`, `place_food_in_skillet`, `stack_bowls_three`,
`place_dual_shoes`, `put_object_cabinet`, `categorize`, `block_ranking`). A
controller owns the layout and allocation draws, avatar motion starts and
calibration, the reset-order hook, and the eval-mode trigger/retreat/glide/
return machine. The task class keeps the robot choreography, the
`take_action` boundary for a yield command, and thin `_`-prefixed delegators.
`NeutralTableWorkController` owns the neutral lifecycle;
`NeutralAvatarTableWorkMixin` composes the job catalog, clip transforms, prop
layout, and planner obstacles. Task-facing attributes (`_eval_*`,
`_interrupted_idx`, ...) remain the single state owner, so metrics,
scene-init replay, and `scripts/eval.py` read the same surface.
`scripts/eval_expert_set.py` advances eval-mode behavior through
`InteractionController.on_scripted_rollout_tick`.

Changes to this framework must preserve public module/class paths,
constructor shape, instructions, seeded initialization order, task metrics,
serialized schemas, and `BaseTask` runner methods. Renames are additive
aliases; serialized schema changes require a new version and a deterministic
migrator. Before moving behavior, make sure the family's contract and
characterization tests pass unchanged (they freeze seeded layouts, RNG tails,
hook order, choreography, and metric surfaces), then compare per-seed episode
records against the previous version with `scripts/eval_expert_set.py`; see
the reproducibility notes in
[running jobs](general/running_jobs.md#reproducibility-notes).
