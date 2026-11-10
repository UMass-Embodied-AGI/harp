"""Composable human interaction for the categorize family's cooperative variant.

Simple shared-pool design (assist 50-50): two loose objects of different
categories lie in one shared strip.  At reset the human picks ONE of them
uniformly at random (task child RNG) and sorts it into its box with a FABRIK
pick-and-place, no matter what the robot does: no availability checks, no
re-targeting, no waiting.  The robot objective stays on the task.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .eval_mode import AssistEvalModeController, delayed_assist_trigger_step


class CategorizeCooperativeController(AssistEvalModeController):
    """Human sorts one object of the shared pool; robot sorts the rest."""

    rng_namespace = "categorize.cooperative.shared_pool.v1"
    handles_policy_ticks = True
    handles_scripted_rollout_ticks = False

    # Avatar FABRIK timing (sim frames).  Unchanged from the legacy
    # cooperative motion.
    APPROACH_FRAMES = 320
    TRANSPORT_FRAMES = 400
    RETRACT_FRAMES = 320
    # Palm drop height above the basket centre.
    PLACE_DZ = 0.10

    def __init__(self) -> None:
        self._reset_shared_pool()

    def _reset_shared_pool(self) -> None:
        self.human_target: int | None = None
        self.human_started = False
        self.robot_target: int | None = None

    def on_episode_seeded(self, task, seed: int) -> None:
        self._reset_shared_pool()

    # -- observations ------------------------------------------------------

    @staticmethod
    def _targets(task):
        return list(getattr(task, "_sort_targets", ()) or ())

    def object_center(self, task, idx: int) -> np.ndarray:
        actor, name, model_id, _basket = self._targets(task)[idx]
        return np.asarray(task._get_object_world_center(actor, name, model_id), dtype=float)

    # -- human behaviour ---------------------------------------------------

    def _start_reach(self, task, idx: int) -> None:
        actor, name, model_id, basket_idx = self._targets(task)[idx]
        pick_pos = task._get_object_world_center(actor, name, model_id)
        basket_pos = np.array(task.basket_poses[basket_idx].p, dtype=float)
        place_pos = basket_pos.copy()
        place_pos[2] += self.PLACE_DZ
        # Closer hand: left (0) for x > 0, right (1) otherwise.
        hand_id = 0 if float(pick_pos[0]) > 0.0 else 1
        task.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=actor,
            hand_id=hand_id,
            approach_frames=self.APPROACH_FRAMES,
            transport_frames=self.TRANSPORT_FRAMES,
            retract_frames=self.RETRACT_FRAMES,
        )
        self.human_target = int(idx)
        self._publish(task)

    def start_human(self, task) -> bool:
        """Pick one of the shared objects uniformly at random and go for it."""
        self.human_started = True
        n = len(self._targets(task))
        if getattr(task, "avatar", None) is None or n == 0:
            return False
        rng = self.episode_rng(task, "human_target")
        idx = int(rng.integers(n))
        print(f"[categorize_coop] human goes for object {idx}", flush=True)
        self._start_reach(task, idx)
        return True

    def wait_for_human(self, task, max_steps: int = 2400) -> None:
        avatar = getattr(task, "avatar", None)
        if avatar is None:
            return
        steps = 0
        while not avatar.spare() and steps < max_steps:
            task.step_sim()
            steps += 1

    # -- robot bookkeeping (records only) -------------------------------

    def note_robot_target(self, idx: int) -> None:
        if self.robot_target is None:
            self.robot_target = int(idx)

    def shared_pool_metrics(self, task=None) -> dict[str, Any]:
        names = [name for _a, name, _m, _b in self._targets(task)] if task is not None else []

        def label(idx):
            return names[idx] if idx is not None and idx < len(names) else None

        same = (self.human_target is not None and self.robot_target is not None
                and int(self.human_target) == int(self.robot_target))
        return {
            "human_target": self.human_target,
            "human_target_object": label(self.human_target),
            "robot_target": self.robot_target,
            "robot_target_object": label(self.robot_target),
            "same_target": bool(same),
        }

    def _publish(self, task) -> None:
        task._eval_avatar_target_idx = 1 if self.human_target is not None else 0
        task._shared_pool_human_target = self.human_target

    # -- legacy-compatible facade hooks -----------------------------------

    def avatar_pick_and_place(
        self, task, actor, obj_name, model_id, basket_idx, wait=True,
        approach_frames=320, transport_frames=400, retract_frames=320,
    ) -> bool:
        """Directly drive one avatar pick-and-place (debug/compat helper)."""
        if task.avatar is None:
            return False
        pick_pos = task._get_object_world_center(actor, obj_name, model_id)
        basket_pos = np.array(task.basket_poses[basket_idx].p, dtype=float)
        place_pos = basket_pos.copy()
        place_pos[2] += self.PLACE_DZ
        hand_id = 0 if float(pick_pos[0]) > 0.0 else 1
        task.avatar.pick_and_place(
            pick_pos=pick_pos, place_pos=place_pos, attach_obj=actor,
            hand_id=hand_id, approach_frames=approach_frames,
            transport_frames=transport_frames, retract_frames=retract_frames,
        )
        if wait:
            while not task.avatar.spare():
                task.step_sim()
        return True

    # -- eval-mode loop --------------------------------------------------

    def eval_reset(self, task) -> None:
        task._eval_avatar_target_idx = 0
        if hasattr(task, "_open_basket_lids"):
            task._open_basket_lids()
        delayed = delayed_assist_trigger_step(task)
        if delayed is not None:
            task._eval_trigger_step = delayed
            task._eval_avatar_started = False
            return
        task._eval_trigger_step = 0
        task._eval_avatar_started = True
        task._eval_fire_next_avatar_motion()

    def eval_step(self, task, step_idx: int) -> None:
        if (not task._eval_avatar_started
                and step_idx >= getattr(task, "_eval_trigger_step", 0)):
            task._eval_avatar_started = True
            task._eval_fire_next_avatar_motion()

    def fire_next_avatar_motion(self, task) -> None:
        if not self.human_started:
            self.start_human(task)

    def interaction_metrics(self, task) -> Mapping[str, Any]:
        metrics: dict[str, Any] = {}
        if getattr(task, "_shared_pool_active", False):
            metrics["shared_pool"] = self.shared_pool_metrics(task)
        if not bool(task.config.get("eval_mode", False)):
            return metrics
        human = self.human_target
        names = [name for _a, name, _m, _b in self._targets(task)]
        metrics.update({
            "eval_mode": True,
            "eval_trigger_step": getattr(task, "_eval_trigger_step", None),
            "eval_policy_step_count": int(getattr(
                task, "_eval_policy_step_count", 0,
            )),
            "eval_avatar_started": bool(getattr(
                task, "_eval_avatar_started", False,
            )),
            "eval_avatar_target_idx": int(getattr(
                task, "_eval_avatar_target_idx", 0,
            )),
            "eval_avatar_order": (
                [names[human]] if human is not None and human < len(names) else []
            ),
        })
        return metrics
