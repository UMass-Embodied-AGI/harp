# HARP-Bench — Task Inventory (non-base)

**40 canonical non-base tasks** (the 9 `TaskVariant.BASE` descriptors are
excluded), plus 32 compatibility names: the pre-HARP names listed in
[task names](task_framework.md#task-names).

The 37 benchmark tasks are the 7 Cueing tasks in section A (everything except
the factory rows) and the 30 variant cells in section B. The three factory
work-cell scenarios are registered extras: they reuse the `cueing` /
`contribution` variant values but are not part of the paper's four benchmark
conditions.

## A. Cueing (`cueing`) / singleton tasks (incl. factory work-cell)

| # | Task | Paper task | Notes |
|---|------|------------|-------|
| 1 | `deliver_to_human` | Deliver to Human | Deliver an object to a human's hand |
| 2 | `factory_inspect_pack` | — (not in benchmark) | Factory HRC — human inspects, robot packs (`cueing` variant) |
| 3 | `factory_kit_packing` | — (not in benchmark) | Factory HRC — human+robot pack one kit in parallel (`contribution` variant) |
| 4 | `factory_line_feeding` | — (not in benchmark) | Factory HRC — robot feeds a tool to the kneeling worker (`cueing` variant) |
| 5 | `oil_bottle_recovery` | Oil Bottle Recovery | Recover a tipped oil bottle |
| 6 | `open_microwave` | Open Microwave | Open the microwave door |
| 7 | `pour_water` | Pour Water | Robot pours water |
| 8 | `feed_documents` | Feed Documents | Robot feeds documents to the human for stamping |
| 9 | `take_from_human_easy` | Take from Human (Easy) | Take an object from a human |
| 10 | `take_from_human_safety` | Take from Human (Safety) | Take-from-human with safety |

## B. Variant families: Contribution / Interruption / Co-presence

The registry `family` (also the no-human base task name, except `categorize`,
which has no registered base task) keeps its historical identifier.

| # | Family | Paper task | Contribution | Interruption | Co-presence |
|---|--------|------------|---|---|---|
| 1 | `blocks_ranking_rgb` | Rank Blocks by Color | `rank_blocks_by_color_contribution` | `rank_blocks_by_color_interruption` | `rank_blocks_by_color_copresence` |
| 2 | `blocks_ranking_size` | Rank Blocks by Size | `rank_blocks_by_size_contribution` | `rank_blocks_by_size_interruption` | `rank_blocks_by_size_copresence` |
| 3 | `categorize` | Categorize Objects | `categorize_objects_contribution` | `categorize_objects_interruption` | `categorize_objects_copresence` |
| 4 | `dump_bin` | Dump Bin | `dump_bin_contribution` | `dump_bin_interruption` | `dump_bin_copresence` |
| 5 | `place_bread_in_basket` | Place Bread in Basket | `place_bread_in_basket_contribution` | `place_bread_in_basket_interruption` | `place_bread_in_basket_copresence` |
| 6 | `place_burger_fries` | Place Burger and Fries | `place_burger_and_fries_contribution` | `place_burger_and_fries_interruption` | `place_burger_and_fries_copresence` |
| 7 | `place_dual_shoes` | Place Pair of Shoes | `place_pair_of_shoes_contribution` | `place_pair_of_shoes_interruption` | `place_pair_of_shoes_copresence` |
| 8 | `place_food_in_skillet` | Place Food in Skillet | `place_food_in_skillet_contribution` | `place_food_in_skillet_interruption` | `place_food_in_skillet_copresence` |
| 9 | `put_object_cabinet` | Put Object in Cabinet | `put_object_in_cabinet_contribution` | `put_object_in_cabinet_interruption` | `put_object_in_cabinet_copresence` |
| 10 | `stack_bowls_three` | Stack Three Bowls | `stack_three_bowls_contribution` | `stack_three_bowls_interruption` | `stack_three_bowls_copresence` |

**Canonical totals:** contribution 10 · interruption 10 · copresence 10 = 30;
plus 10 Cueing/singleton (incl. 3 factory) = **40**.
