"""Legacy inspect helper surface for composed Interruption tasks."""

import numpy as np

from ..interaction.inspect_motion import InspectMotionInterruptController


class InspectMotionInterruptFacade:
    """Delegate legacy inspect helpers to a stateless controller runtime.

    Deliberately defines neither ``reset`` nor ``play_once``: public family
    classes retain their exact release-v1 reset flow and expert MRO routing.
    """

    def _inspect_interrupt_controller(self) -> InspectMotionInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, InspectMotionInterruptController):
            raise TypeError("task does not use composed inspect interrupt behavior")
        return controller

    def _resolve_inspect_motion(self) -> None:
        return self._inspect_interrupt_controller().resolve_inspect_motion(self)

    def _prepare_inspect_motion_entry(self, entry: dict) -> dict:
        """Install a configured source-frame subclip and return runtime frames."""
        return self._inspect_interrupt_controller().inspect_runtime.prepare_entry(
            self,
            entry,
        )

    @property
    def _inspect_attach_frame_scaled(self) -> int:
        return self._inspect_interrupt_controller().inspect_runtime.inspect_attach_frame_scaled(
            self
        )

    @property
    def _inspect_detach_frame_scaled(self) -> int:
        return self._inspect_interrupt_controller().inspect_runtime.inspect_detach_frame_scaled(
            self
        )

    @property
    def _attach_frame_scaled(self) -> int:
        return self._inspect_attach_frame_scaled

    @property
    def _detach_frame_scaled(self) -> int:
        return self._inspect_detach_frame_scaled

    def _wait_until_avatar_attach(self):
        """Step physics until the inspect clip reaches its attach frame.

        Hoisted verbatim from six identical per-task copies (cleanup phase 2);
        block-ranking and stack-bowls interrupts keep their own variants.
        """
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    def inspect_avatar_pos_for_object(
        self,
        object_pos,
        extra_z: float | None = None,
    ):
        """Return the avatar root that preserves motion/object placement."""
        return self._inspect_interrupt_controller().inspect_runtime.avatar_pos_for_object(
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
        """Place the avatar using the selected motion's fixed relation."""
        return self._inspect_interrupt_controller().inspect_runtime.reset_avatar_for_object(
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
        return self._inspect_interrupt_controller().inspect_runtime.idle_position_for_motion_root(
            self,
            motion_root_pos,
            motion_frame=motion_frame,
        )

    def _set_inspect_motion_root_pos(self, motion_root_pos) -> np.ndarray:
        return self._inspect_interrupt_controller().inspect_runtime.set_motion_root_pos(
            self,
            motion_root_pos,
            idle_position_for_motion_root=self._inspect_idle_position_for_motion_root,
        )

    def _play_inspect_motion(
        self,
        attach_obj=None,
        hand_id=1,
        attach_frame=None,
        detach_frame=None,
        **kwargs,
    ):
        """Play the selected inspect motion, honoring touch-only entries."""
        return self._inspect_interrupt_controller().inspect_runtime.play(
            self,
            attach_obj=attach_obj,
            hand_id=hand_id,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            **kwargs,
        )
