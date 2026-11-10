"""Contribution variant for RGB block ranking (``rank_blocks_by_color_contribution``).

The robot objective remains in :class:`BlocksRankingRGB`; the human's work is
owned by a composed :class:`BlockRankingAssistController`.
"""

import numpy as np

from ..interaction.block_ranking import (
    BlockAssistSampling,
    BlockRankingAssistController,
)
from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB


class RankBlocksByColorContribution(BlocksRankingRGB):
    """Human places one color block while the robot places the other two."""

    INSTRUCTION = (
        "place the red, green, and blue blocks in left-to-right order while "
        "the human helps place one block"
    )
    use_avatar = True
    BLOCK_KEY_KIND = "color"
    AVATAR_COLOR_CHOICES = ("red", "green", "blue")
    ASSIST_LAYOUT_ERROR_LABEL = "blocks_rgb_assist"

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 360
    AVATAR_RETRACT_FRAMES = 180
    AVATAR_START_DELAY_STEPS = (100, 200)
    AVATAR_PICK_LIFT = 0.010
    AVATAR_PLACE_LIFT = 0.075
    AVATAR_NATURAL_BODY_MARGIN = 0.34
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.28, 0.62)
    AVATAR_NATURAL_SETTLE_STEPS = 20
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200

    AVATAR_SPAWN_X = (-0.22, 0.22)
    AVATAR_SPAWN_Y = (-0.03, 0.16)
    ROBOT_SPAWN_X = (-0.24, 0.24)
    ROBOT_SPAWN_Y = (-0.38, -0.22)
    AVATAR_LAYOUT_SAMPLE_ATTEMPTS = 160

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)

    def create_interaction_controller(self) -> BlockRankingAssistController:
        return BlockRankingAssistController(
            keys=self.block_keys(),
            owner_keys=self.AVATAR_COLOR_CHOICES,
            delay_range=self.AVATAR_START_DELAY_STEPS,
            sampling=BlockAssistSampling.RGB_V1,
        )

    def play_once(self) -> bool:
        return self.play_assist_once()

    def play_blind_once(self) -> bool:
        return self.play_blind_assist_once()

    def evaluate(self) -> dict:
        # The controller reports nothing for RGB unless a blind rollout ran
        # (then blind_robot_keys / blind_contests_human_block).
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
