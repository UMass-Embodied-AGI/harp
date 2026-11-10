"""Interruption variant for size block ranking (``rank_blocks_by_size_interruption``)
using a composed controller."""

import numpy as np

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..interaction.block_ranking import (
    BlockInterruptSampling,
    BlockRankingInterruptController,
)
from ..task_bases.blocks_ranking_size import BlocksRankingSize
from ._block_ranking_interrupt_facade import BlockRankingInterruptFacade


class RankBlocksBySizeInterruption(BlockRankingInterruptFacade, BlocksRankingSize):
    """Human inspects one size block while the robot orders all three."""

    INSTRUCTION = (
        "place the RGB blocks in largest-to-smallest order; wait while the "
        "human inspects one block, then continue"
    )
    use_avatar = True
    BLOCK_KEY_KIND = "size"
    INTERRUPT_LAYOUT_ERROR_LABEL = "blocks_size_interrupt"

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.38])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    AVATAR_REACH_SPAWN_X_RANGE = (-0.18, 0.18)
    AVATAR_REACH_SPAWN_Y_RANGE = (-0.02, 0.13)
    INTERRUPT_SPAWN_X_RANGE = AVATAR_REACH_SPAWN_X_RANGE
    INTERRUPT_SPAWN_Y_RANGE = AVATAR_REACH_SPAWN_Y_RANGE

    # Bounds of the retired random-step trigger.  Only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    # Progress-trigger interrupt contract (envs/interaction/interrupt_progress.py).
    INTERRUPT_COMMIT_RADIUS_M = 0.20
    SCRIPT_TARGET_SEED_OFFSET = 7919
    SCRIPT_TIME_SEED_OFFSET = 13
    CALIBRATION_SETTLE_STEPS = 20
    POST_CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    # Collection play_once only; eval mode returns to idle.
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    POST_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30

    def create_interaction_controller(self) -> BlockRankingInterruptController:
        return BlockRankingInterruptController(
            keys=self.block_keys(),
            sampling=BlockInterruptSampling.SIZE_V1,
        )

    def play_once(self) -> bool:
        return self.play_interrupt_once()

    def _wait_until_avatar_attach(self) -> None:
        """Keep the retreated expert arm clear until the inspection ends.

        Attachment alone does not make the other blocks safe to approach:
        the returning fingers cross that workspace after releasing the cube.
        Finish the same inspection motion before preflighting the next pickup
        against its now stationary avatar obstacle cloud.
        """
        if self.avatar is None:
            return
        super()._wait_until_avatar_attach()
        while not self.avatar.spare():
            self.step_sim()

    def evaluate(self) -> dict:
        """Append controller-owned fields in release-v1 key order."""

        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
