"""Co-presence variant of ``blocks_ranking_size`` (``rank_blocks_by_size_copresence``)."""

import numpy as np

from ..task_bases.blocks_ranking_size import BlocksRankingSize
from ..task_bases.block_neutral_public_grasp import BlockNeutralPublicGraspMixin
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class RankBlocksBySizeCopresence(BlockNeutralPublicGraspMixin, NeutralAvatarTableWorkMixin, BlocksRankingSize):
    """Robot orders size blocks while the human performs unrelated table work."""

    # Plan around the human's objects as exact boxes.
    NEUTRAL_OBJECT_PLANNER_BOXES = True

    recording_camera_pos = [-0.85, 0.65, 1.55]
    recording_camera_lookat = [0.0, -0.22, 0.82]

    # Keep the avatar's work footprint off the target row (issue 6a).
    NEUTRAL_SLOTS_ARE_PHYSICAL = True

    # A moving can-work obstacle may temporarily exclude the local lift branch.
    BLOCK_CAN_LIFT_WAIT_RECOVERY = True
    BLOCK_CAN_PUBLIC_GRASP_RECOVERY = True

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        work_xy=(0.34, 0.10),
        mouse_work_xy=(0.34, 0.15),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the blocks from largest to smallest while the human works "
        "nearby on the same table"
    )

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge"):
            return True
        return super()._neutral_avatar_body_pose_is_valid(job, xy, body_pos)

    # Shrink the physical slot keepout (default r=0.060) to
    # r=0.040 — the largest block half-extent is 0.030 — so the wipe layout
    # solver stops rejecting seeds outright.  Same rationale as
    # rank_blocks_by_color_copresence.
    NEUTRAL_SLOT_KEEPOUT_RADIUS = 0.040

    def _neutral_avatar_forbidden_regions(self):
        r = float(self.NEUTRAL_SLOT_KEEPOUT_RADIUS)
        return [
            (center, (r if str(label).startswith("slot:") else radius), label)
            for center, radius, label in
            super()._neutral_avatar_forbidden_regions()
        ]
