"""Compatibility facade for composed block-ranking Interruption tasks."""

import numpy as np

from ..interaction.block_ranking import BlockRankingInterruptController


class BlockRankingInterruptFacade:
    """Keep the legacy interrupt helper surface off robot-only family cores."""

    # Hover/dwell interrupt trigger (envs/interaction/block_ranking.py),
    # shared by the rgb and size interrupt tasks.  The demos pick 0-2 other
    # blocks first, passing within ~0.13 m of the reset-chosen one, so a
    # plain progress trigger fired on the first block approached (mid-pick).
    # The human interrupts only the reset-chosen block, once the TCP has
    # hovered over it (xy <= 6 cm, 8-20 cm above its centre) for 3
    # consecutive policy steps, plus radius 0.20 m and progress >= 0.25 (the
    # demo hover sits at progress 0.44-0.56).  Fitted on replayed demos: rgb
    # (6) fires 4 steps before to 1 step after the demo human starts, size
    # (6) 2-5 steps before.
    INTERRUPT_TRIGGER_PROGRESS = 0.25
    INTERRUPT_HOVER_DWELL_STEPS = 3
    INTERRUPT_HOVER_XY_M = 0.06
    INTERRUPT_HOVER_DZ_RANGE_M = (0.08, 0.20)

    def _block_interrupt_controller(self) -> BlockRankingInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, BlockRankingInterruptController):
            raise TypeError("task does not use block-ranking interrupt behavior")
        return controller

    @property
    def _attach_frame_scaled(self) -> int:
        return self._block_interrupt_controller().attach_frame_scaled(self)

    @property
    def _detach_frame_scaled(self) -> int:
        return self._block_interrupt_controller().detach_frame_scaled(self)

    @property
    def _inspect_attach_frame_scaled(self) -> int:
        return self._attach_frame_scaled

    @property
    def _inspect_detach_frame_scaled(self) -> int:
        return self._detach_frame_scaled

    def _resolve_inspect_motion(self) -> None:
        self._block_interrupt_controller().inspect_runtime.resolve(
            self,
            prepare_entry=self._prepare_inspect_motion_entry,
        )

    def _prepare_inspect_motion_entry(self, entry: dict) -> dict:
        return self._block_interrupt_controller().inspect_runtime.prepare_entry(
            self,
            entry,
        )

    def _play_inspect_motion(self, **kwargs):
        return self._block_interrupt_controller().play_inspect_motion(
            self,
            **kwargs,
        )

    def _dryrun_capture_palm(self):
        return self._block_interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self) -> None:
        self._block_interrupt_controller().calibrate_avatar(self)

    def _init_eval_interrupt_state(self) -> None:
        self._block_interrupt_controller().init_eval_state(self)

    def _eval_step_avatar(self) -> None:
        self._block_interrupt_controller().advance_eval_step(self)

    def _find_closest_table_block(self):
        return self._block_interrupt_controller().find_closest_table_block(self)

    def _retreat_safe(self, arm_tag: str) -> None:
        self._block_interrupt_controller().retreat_safe(self, arm_tag)

    def _wait_until_avatar_attach(self) -> None:
        self._block_interrupt_controller().wait_until_avatar_attach(self)

    # Full-state expert under the eval-mode interrupt contract (driven by
    # envs.interaction.interrupt_progress.run_full_state_interrupt_expert).
    def interrupt_expert_attempt(self) -> bool:
        return self._block_interrupt_controller().interrupt_expert_attempt(self)

    def interrupt_expert_retreat(self) -> None:
        self._block_interrupt_controller().interrupt_expert_retreat(self)

    def inspect_avatar_pos_for_object(
        self,
        object_pos,
        extra_z: float | None = None,
    ):
        return self._block_interrupt_controller().inspect_runtime.avatar_pos_for_object(
            self,
            object_pos,
            extra_z=extra_z,
        )

    def reset_avatar_for_inspect_object(
        self,
        object_pos,
        extra_z: float | None = None,
        settle_steps: int | None = None,
    ) -> bool:
        return self._block_interrupt_controller().inspect_runtime.reset_avatar_for_object(
            self,
            object_pos,
            extra_z=extra_z,
            settle_steps=settle_steps,
            avatar_pos_for_object=self.inspect_avatar_pos_for_object,
            set_motion_root_pos=self._set_inspect_motion_root_pos,
        )

    def _inspect_idle_position_for_motion_root(
        self,
        motion_root_pos,
        motion_frame: int = 0,
    ) -> np.ndarray:
        return self._block_interrupt_controller().inspect_runtime.idle_position_for_motion_root(
            self,
            motion_root_pos,
            motion_frame=motion_frame,
        )

    def _set_inspect_motion_root_pos(self, motion_root_pos) -> np.ndarray:
        return self._block_interrupt_controller().inspect_runtime.set_motion_root_pos(
            self,
            motion_root_pos,
            idle_position_for_motion_root=self._inspect_idle_position_for_motion_root,
        )
