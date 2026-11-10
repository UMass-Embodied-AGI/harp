"""Co-presence variant of ``blocks_ranking_rgb`` (``rank_blocks_by_color_copresence``)."""

import numpy as np

from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB
from ..task_bases.block_neutral_public_grasp import BlockNeutralPublicGraspMixin
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class RankBlocksByColorCopresence(BlockNeutralPublicGraspMixin, NeutralAvatarTableWorkMixin, BlocksRankingRGB):
    """Robot orders RGB blocks while the human performs unrelated table work."""

    # Plan around the human's objects as exact boxes.
    NEUTRAL_OBJECT_PLANNER_BOXES = True

    # Keep the avatar's work footprint off the target row (issue 6a).
    NEUTRAL_SLOTS_ARE_PHYSICAL = True

    # A moving can-work obstacle may temporarily exclude the local lift branch.
    BLOCK_CAN_LIFT_WAIT_RECOVERY = True
    BLOCK_CAN_PUBLIC_GRASP_RECOVERY = True

    # Offset the clapping seat from both the table edge and the corner leg.
    NEUTRAL_CLAP_EDGE_ANCHOR_LOCAL_XY = (0.51, 0.15)

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        work_xy=(0.34, 0.10),
        mouse_work_xy=(0.34, 0.02),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the red, green, and blue blocks in left-to-right order while "
        "the human works nearby on the same table"
    )

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge"):
            return True
        return super()._neutral_avatar_body_pose_is_valid(job, xy, body_pos)

    # With NEUTRAL_SLOTS_ARE_PHYSICAL the three slot disks (default r=0.060)
    # plus the three block disks straddle the table centre, and the wipe
    # layout solver rejects many seeds outright ("no neutral layout
    # candidate", 14 resets in a 20-seed sweep).  The largest block
    # half-extent is 0.030, so r=0.040 still keeps the avatar's sweep off the
    # target row while recovering ~0.12 m of anchor space per rectangle.
    NEUTRAL_SLOT_KEEPOUT_RADIUS = 0.040

    def _neutral_avatar_forbidden_regions(self):
        r = float(self.NEUTRAL_SLOT_KEEPOUT_RADIUS)
        return [
            (center, (r if str(label).startswith("slot:") else radius), label)
            for center, radius, label in
            super()._neutral_avatar_forbidden_regions()
        ]
