"""Blocks ranking size: arrange RGB cubes from largest to smallest.

Robot-only single-Franka version of RoboTwin's ``blocks_ranking_size`` task.
Three visible RGB primitive cubes have independently randomized sizes each
episode; the robot sorts them into a fixed row by size, not by color.
"""

import numpy as np

from ..base_task import BaseTask
from ..interaction.block_ranking import BlockEpisodePlan, BlockSpawnSpec
from ..utils import to_numpy
from .blocks_ranking_rgb import BlocksRankingRGB


class BlocksRankingSize(BlocksRankingRGB):
    """Robot places three differently sized RGB blocks in largest-to-smallest order."""

    INSTRUCTION = (
        "place the largest, medium, and smallest blocks in a row from left "
        "to right"
    )

    # Full side lengths: 6.0 cm, 4.8 cm, 3.6 cm.  All are within the Franka
    # gripper's comfortable top-down grasp range while remaining visually
    # distinct in camera views.
    SIZE_HALVES = {
        "large": 0.030,
        "medium": 0.024,
        "small": 0.018,
    }
    SIZE_ORDER = ("large", "medium", "small")

    SLOT_XS = {
        "large": -0.16,
        "medium": 0.0,
        "small": 0.16,
    }

    SUCCESS_DIST_XY = 0.060

    def block_keys(self) -> tuple[str, ...]:
        return self.SIZE_ORDER

    def block_half_size(self, key: str) -> float:
        return float(self.SIZE_HALVES[key])

    def block_pick_label(self, key: str) -> str:
        return f"{key} {self.block_colors.get(key, 'unknown')} block"

    def _base_block_episode_plan(self) -> BlockEpisodePlan:
        size_names = list(self.SIZE_ORDER)
        color_names = list(self.CUBE_COLORS.keys())
        color_order = np.random.permutation(color_names)
        size_order = np.random.permutation(size_names)
        xys = self._sample_spawn_xys()
        return BlockEpisodePlan(
            blocks=tuple(
                BlockSpawnSpec(
                    key=str(size_name),
                    color_name=str(color_order[index]),
                    half_size=self.block_half_size(str(size_name)),
                    xy=(float(xys[index, 0]), float(xys[index, 1])),
                )
                for index, size_name in enumerate(size_order)
            )
        )

    def _pick_and_place_block(
        self, size_name: str, arm_tag: str = "right"
    ) -> bool:
        """Preserve the size family's release-v1 helper signature."""

        return super()._pick_and_place_block(size_name, arm_tag)

    def play_once(self) -> bool:
        """Preserve the size core as the plain-expert parent boundary."""

        return super().play_once()

    def check_success(self) -> bool:
        positions = {}
        slot_ok = True
        on_table_ok = True

        for size_name in self.SIZE_ORDER:
            cube = self.blocks[size_name]
            half = float(self.SIZE_HALVES[size_name])
            p = to_numpy(cube.get_pos()).ravel()[:3]
            positions[size_name] = p
            target = self.target_slots[size_name]
            dist_xy = float(np.linalg.norm(p[:2] - target[:2]))
            in_slot = dist_xy <= self.SUCCESS_DIST_XY
            on_table = (
                p[2] >= self.TABLE_TOP_Z + half * self.ON_TABLE_HEIGHT_FRACTION
                and self._block_resting(cube, p, half)
            )
            slot_ok = slot_ok and in_slot
            on_table_ok = on_table_ok and on_table

        order_ok = (
            self._order_axis_value(positions["large"]) <
            self._order_axis_value(positions["medium"]) <
            self._order_axis_value(positions["small"])
        )
        success = bool(self.plan_success and slot_ok and on_table_ok and order_ok)
        return success

    def evaluate(self) -> dict:
        metrics = BaseTask.evaluate(self)
        positions = {}
        dist_xy = {}
        slot_ok = {}
        on_table_ok = {}
        resting_ok = {}

        for size_name in self.SIZE_ORDER:
            cube = self.blocks[size_name]
            half = float(self.SIZE_HALVES[size_name])
            p = to_numpy(cube.get_pos()).ravel()[:3]
            target = self.target_slots[size_name]
            dxy = float(np.linalg.norm(p[:2] - target[:2]))
            positions[size_name] = p.tolist()
            dist_xy[size_name] = dxy
            slot_ok[size_name] = bool(dxy <= self.SUCCESS_DIST_XY)
            on_table_ok[size_name] = bool(
                p[2] >= self.TABLE_TOP_Z + half * self.ON_TABLE_HEIGHT_FRACTION
            )
            resting_ok[size_name] = bool(self._block_resting(cube, p, half))

        order_ok = bool(
            self._order_axis_value(positions["large"]) <
            self._order_axis_value(positions["medium"]) <
            self._order_axis_value(positions["small"])
        )
        metrics.update({
            "block_positions": positions,
            "block_dist_xy": dist_xy,
            "block_slot_ok": slot_ok,
            "block_on_table_ok": on_table_ok,
            "block_all_slots_ok": bool(all(slot_ok.values())),
            "block_all_on_table_ok": bool(all(on_table_ok.values())),
            "block_order_ok": order_ok,
            "block_success_dist_xy": self.SUCCESS_DIST_XY,
            "block_resting_ok": resting_ok,
        })
        return metrics
