"""Composable lifecycle for neutral-avatar table-work variants (step 5).

``NeutralAvatarTableWorkMixin`` remains the composition vehicle for the
neutral job catalog, clip transforms, prop layout and planner obstacles (the
*how*).  This controller owns the human-behavior lifecycle (the *when*):
per-episode state clearing, the post-reset start layout and eval trigger
draw, the eval-mode policy tick that starts the avatar's table work, the
scripted start, the avatar-side success gate, and the neutral metric surface.

Release-v1 task-facing ``_neutral_*`` attributes remain the single state
owner; draw orders are reproduced verbatim.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..utils import to_numpy
from .base import NeutralInteractionController


NEUTRAL_EVAL_TRIGGER_SALT = 7_340_021


def draw_neutral_eval_trigger_step(task, lo: int, hi: int) -> int:
    """Eval human start step in ``[lo, hi]``.

    Drawn from a private generator seeded by ``[episode seed, salt]`` so the
    global ``np.random`` stream after reset is the same in eval mode and in
    the scripted demo choreography (the scripted expert's grasp/planning
    draws read that stream; an extra global draw here measurably lowered the
    eval-mode full-state expert success rate).  ``neutral_eval_trigger_rng:
    "global"`` restores the legacy global draw for comparison.
    """
    if str((task.config or {}).get("neutral_eval_trigger_rng", "private")) == "global":
        return int(np.random.randint(lo, hi + 1))
    seed = int(getattr(task, "_neutral_avatar_reset_seed", 0) or 0)
    rng = np.random.default_rng([seed, NEUTRAL_EVAL_TRIGGER_SALT])
    return int(rng.integers(lo, hi + 1))


class NeutralTableWorkController(NeutralInteractionController):
    """Human performs unrelated table work beside the robot's task."""

    rng_namespace = "neutral_table_work.v1"
    # The mixin's take_action wrapper ticks this controller before the robot
    # action so the legacy neutral-before-robot order is kept exactly.
    handles_policy_ticks = False
    handles_scripted_rollout_ticks = False

    # -- lifecycle -------------------------------------------------------

    def on_episode_seeded(self, task, seed: int) -> None:
        """Clear neutral state before the scene is rebuilt (legacy pre-reset)."""
        if hasattr(task, "_neutral_original_step_sim"):
            task.step_sim = task._neutral_original_step_sim
            del task._neutral_original_step_sim
        if hasattr(task, "_neutral_writing_orig_step_sim"):
            task.step_sim = task._neutral_writing_orig_step_sim
            del task._neutral_writing_orig_step_sim
        task._neutral_avatar_reset_seed = int(seed)
        task._neutral_avatar_started = False
        task._neutral_laptop_hold_state = None
        task._neutral_obb_cache = {}
        task._neutral_avatar_job = None
        task._neutral_avatar_step_state = None
        task._neutral_avatar_motion_footprint = None
        task._neutral_avatar_selected_work_xy = None
        task._neutral_avatar_region_debug_records = []
        task._neutral_avatar_region_debug_entities = []
        task._neutral_avatar_can_model_ids = {}
        task._neutral_avatar_layout_ready = False
        task._neutral_eval_policy_step_count = 0
        task._neutral_eval_trigger_step = None
        task._neutral_eval_avatar_failed = False
        task._neutral_prop_disturbed = None
        task._apply_neutral_camera_overrides()

    def after_task_reset(self, task, observation: Any) -> Any:
        """Legacy post-``super().reset()`` work: start layout, neutral-object
        collision detection, and the eval trigger draw.  Called by the mixin's
        reset wrapper after every base hook has run, which keeps the
        categorize draw-burning hook ahead of the trigger draw."""
        if (
            task.avatar is not None
            and bool(task.config.get("eval_mode", False))
            and "neutral_avatar_strict_layout" not in task.config
        ):
            task.config["neutral_avatar_strict_layout"] = False
        if bool(task.config.get("neutral_avatar_skip_start_layout", False)):
            task._neutral_avatar_layout_ready = False
            return observation
        if (task.config or {}).get("scene_init"):
            task._neutral_avatar_layout_ready = False
        task._prepare_neutral_avatar_start_layout()
        task._install_neutral_object_collision_detection()
        if task.avatar is not None and bool(task.config.get("eval_mode", False)):
            lo = int(task.config.get(
                "neutral_eval_trigger_step_min",
                task.NEUTRAL_EVAL_TRIGGER_STEP_MIN,
            ))
            hi = int(task.config.get(
                "neutral_eval_trigger_step_max",
                task.NEUTRAL_EVAL_TRIGGER_STEP_MAX,
            ))
            if hi < lo:
                hi = lo
            task._neutral_eval_trigger_step = draw_neutral_eval_trigger_step(task, lo, hi)
        return observation

    # -- eval-mode tick + scripted start ---------------------------------

    @staticmethod
    def advance_eval_step(task) -> None:
        if task.avatar is None or not bool(task.config.get("eval_mode", False)):
            return
        task._neutral_eval_policy_step_count = (
            int(getattr(task, "_neutral_eval_policy_step_count", 0)) + 1
        )
        trigger = getattr(task, "_neutral_eval_trigger_step", None)
        if (
            trigger is not None
            and not getattr(task, "_neutral_avatar_started", False)
            and not getattr(task, "_neutral_eval_avatar_failed", False)
            and task._neutral_eval_policy_step_count >= int(trigger)
        ):
            try:
                task._start_neutral_avatar_table_work(add_start_delay=False)
            except Exception as exc:
                task._neutral_eval_avatar_failed = True
                if task.config.get("debug_eval_avatar", False):
                    print(
                        f"[neutral_avatar/eval] start failed: {exc}",
                        flush=True,
                    )

    @staticmethod
    def start_scripted_table_work(task) -> bool:
        return bool(task._start_neutral_avatar_table_work(add_start_delay=True))

    # -- success + metrics ----------------------------------------------

    @staticmethod
    def neutral_object_collided(task) -> bool:
        """True if any logged robot collision was with a neutral object."""
        for record in getattr(task, "avatar_collision_log", []) or []:
            for pair in record.get("pairs") or []:
                if len(pair) >= 2 and str(pair[1]).startswith("neutral_obj:"):
                    return True
        return False

    @staticmethod
    def avatar_success_gate(task) -> bool:
        """``avatar_collided`` is the sticky OR of robot-avatar-body and
        robot-neutral-object overlaps; a touch on either side fails, and so
        does a human prop the robot knocked over (``_neutral_prop_disturbed``)."""
        if getattr(task, "_neutral_prop_disturbed", None):
            return False
        return not bool(getattr(task, "avatar_collided", False))

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        metrics = {
            "avatar_collided": bool(getattr(task, "avatar_collided", False)),
            "neutral_object_collided": bool(task._neutral_object_collided()),
            "neutral_eval_trigger_step": getattr(
                task, "_neutral_eval_trigger_step", None,
            ),
            "neutral_eval_policy_step_count": int(getattr(
                task, "_neutral_eval_policy_step_count", 0,
            )),
            "neutral_avatar_started": bool(getattr(
                task, "_neutral_avatar_started", False,
            )),
            "neutral_eval_avatar_failed": bool(getattr(
                task, "_neutral_eval_avatar_failed", False,
            )),
            "neutral_prop_disturbed": getattr(task, "_neutral_prop_disturbed", None),
        }
        if hasattr(task, "neutral_avatar_prop"):
            prop_pos = to_numpy(task.neutral_avatar_prop.get_pos()).ravel()[:3]
            metrics["neutral_avatar_prop_pos"] = prop_pos.tolist()
            metrics["neutral_avatar_prop_dist_xy"] = float(
                np.linalg.norm(prop_pos[:2] - task.neutral_avatar_target[:2])
            )
        return metrics
