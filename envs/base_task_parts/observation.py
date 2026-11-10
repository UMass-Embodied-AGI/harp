"""Observation and action interface (testbed mode).

Split from BaseTask (verbatim member move); composed back into it in
envs/base_task.py.
"""


import numpy as np
from typing import Literal

from ..utils import Pose
from ..interaction import (
    PolicyDirective,
)



class _ObservationActionMixin:
    # ------------------------------------------------------------------
    # Observation and action (testbed mode)
    # ------------------------------------------------------------------

    def get_obs(self) -> dict:
        """Get current observation."""
        right_ee = self.robot.get_ee_pose("right")
        left_ee = self.robot.get_ee_pose("left") if self.robot.left_arm is not None else None
        right_pose = Pose.from_pose7(right_ee) if right_ee is not None else None
        left_pose = Pose.from_pose7(left_ee) if left_ee is not None else None
        self.cameras.update_wrist_cameras(left_pose, right_pose)
        self.cameras.render_all()

        obs = {
            "rgb": self.cameras.get_all_rgb(),
            "depth": self.cameras.get_all_depth(),
            # Per-camera 3x3 intrinsic + 4x4 cam-to-world extrinsic matrices.
            # Wrist intrinsics are static; wrist extrinsics track the EE
            # (update_wrist_cameras above ran moments ago). Consumers that
            # don't need 3D info simply ignore these keys (DP3 lifts depth
            # to a point cloud here).
            "camera_intrinsics": {
                name: self.cameras.get_intrinsic(name)
                for name in self.cameras.get_camera_names()
            },
            "camera_extrinsics": {
                name: self.cameras.get_extrinsic(name)
                for name in self.cameras.get_camera_names()
            },
            "joint_state": {
                "right": self.robot.get_right_joint_state(),
            },
            "ee_pose": {
                "right": right_ee,
            },
            "gripper_val": {
                "right": self.robot.get_right_gripper_val(),
            },
        }
        # Include left arm data if available
        if self.robot.left_arm is not None:
            obs["joint_state"]["left"] = self.robot.get_left_joint_state()
            obs["ee_pose"]["left"] = left_ee
            obs["gripper_val"]["left"] = self.robot.get_left_gripper_val()
        return obs

    def _avatar_retreat_status(self) -> tuple[bool, dict | None]:
        """Return whether active safety should override the nominal command."""
        if self.avatar_retreat_checker is None:
            return False, None
        result = self.avatar_retreat_checker.check()
        min_distance = float(result.get("min_distance", float("inf")))
        margin = float(self.avatar_retreat_checker.margin)
        release_margin = float(self.config.get(
            "avatar_retreat_release_margin",
            margin + float(self.config.get("avatar_retreat_hysteresis", 0.05)),
        ))
        hold_steps = int(self.config.get("avatar_retreat_hold_steps", 8))

        if min_distance <= margin:
            self._avatar_retreat_hold_count = max(1, hold_steps)
            active = True
        elif self._avatar_retreat_hold_count > 0 and min_distance < release_margin:
            self._avatar_retreat_hold_count -= 1
            active = True
        else:
            self._avatar_retreat_hold_count = 0
            active = False

        if active:
            self.avatar_safety_intervention_log.append({
                "frame": int(self.FRAME_IDX),
                "min_distance": min_distance,
                "margin": margin,
                "release_margin": release_margin,
                "closest_pair": result.get("closest_pair"),
            })
        return active, result

    def _closest_avatar_point_to(self, point: np.ndarray) -> np.ndarray | None:
        """Closest point on the current avatar capsule set to a world point."""
        if self.avatar_collider is None:
            return None
        capsules = self.avatar_collider.current_capsules()
        if not capsules:
            return None
        point = np.asarray(point, dtype=np.float64).ravel()[:3]
        best_point = None
        best_dist = float("inf")
        for _name, A, B, radius in capsules:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
            AB = B - A
            ab_sq = float(np.dot(AB, AB))
            if ab_sq < 1e-12:
                continue
            t = float(np.clip(np.dot(point - A, AB) / ab_sq, 0.0, 1.0))
            closest = A + t * AB
            signed = float(np.linalg.norm(point - closest) - float(radius))
            if signed < best_dist:
                best_dist = signed
                best_point = closest
        return best_point

    def _avatar_retreat_command_pose(self, arm_tag: str, pose: np.ndarray) -> bool:
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.solve_ik(arm.get_arm_qpos(), pose, num_waypoints=1, log=False)
        if result.success and result.position.size > 0:
            self.robot.set_arm_joints(result.position[-1], arm_tag)
            return True
        return False

    def _avatar_retreat_putdown(self, arm_tag: str, pose: np.ndarray) -> bool:
        if not bool(self.config.get("avatar_retreat_putdown_enabled", True)):
            return False
        arm = self.robot.get_arm(arm_tag)
        closed_threshold = float(self.config.get("avatar_retreat_holding_gripper_threshold", 0.45))
        if arm.gripper_val > closed_threshold and arm_tag not in self._avatar_retreat_release_state:
            return False

        state = self._avatar_retreat_release_state.setdefault(
            arm_tag, {"phase": "clear", "ticks": 0}
        )
        table_z = float(getattr(self, "TABLE_TOP_Z", 0.765))
        putdown_z = table_z + float(self.config.get("avatar_retreat_putdown_ee_z", 0.105))
        lower_step = float(self.config.get("avatar_retreat_putdown_step", 0.015))
        release_steps = int(self.config.get("avatar_retreat_release_steps", 80))
        release_margin = float(self.config.get(
            "avatar_retreat_release_margin",
            float(self.avatar_retreat_checker.margin)
            + float(self.config.get("avatar_retreat_hysteresis", 0.05)),
        ))

        if state["phase"] == "clear":
            active, result = self._avatar_retreat_status()
            min_distance = float((result or {}).get("min_distance", float("inf")))
            if min_distance >= release_margin:
                state["phase"] = "lower"
            else:
                return False

        if state["phase"] == "lower":
            target = pose.copy()
            target[2] = max(putdown_z, pose[2] - lower_step)
            ok = self._avatar_retreat_command_pose(arm_tag, target)
            if pose[2] <= putdown_z + 0.015:
                state["phase"] = "release"
                state["ticks"] = 0
            return ok

        if state["phase"] == "release":
            self.robot.set_gripper(1.0, arm_tag)
            if arm_tag in self._gripper_attached:
                self.detach_from_gripper(arm_tag)
            state["ticks"] += 1
            if state["ticks"] >= release_steps:
                state["phase"] = "done"
            return True

        return False

    def _apply_avatar_retreat(self, arm_tag: str) -> bool:
        """Command one arm up and away from the closest avatar capsule."""
        if self.avatar_retreat_checker is None:
            return False
        arm = self.robot.get_arm(arm_tag)
        pose = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:7]
        if pose.size < 7:
            return False
        closest = self._closest_avatar_point_to(pose[:3])
        if closest is None:
            return False

        if self._avatar_retreat_putdown(arm_tag, pose):
            return True

        away_xy = pose[:2] - closest[:2]
        norm_xy = float(np.linalg.norm(away_xy))
        if norm_xy < 1e-5:
            # Default away from the nominal robot side of the table.
            away_xy = np.array([0.0, -1.0], dtype=np.float64)
            norm_xy = 1.0
        away_xy = away_xy / norm_xy
        z_weight = float(self.config.get("avatar_retreat_z_weight", 0.25))
        direction = np.array([away_xy[0], away_xy[1], max(0.0, z_weight)], dtype=np.float64)
        direction /= float(np.linalg.norm(direction) + 1e-12)

        step = float(self.config.get("avatar_retreat_step", 0.025))
        retreat_pose = pose.copy()
        retreat_pose[:3] = pose[:3] + step * direction
        min_z = float(self.config.get("avatar_retreat_min_z", getattr(self, "TABLE_TOP_Z", 0.0) + 0.03))
        max_z = float(self.config.get("avatar_retreat_max_z", getattr(self, "TABLE_TOP_Z", 0.0) + 0.24))
        retreat_pose[2] = float(np.clip(retreat_pose[2], min_z, max_z))

        return self._avatar_retreat_command_pose(arm_tag, retreat_pose)

    def _apply_avatar_retreat_if_needed(self, arm_tags: tuple[str, ...]) -> bool:
        if self.avatar_retreat_checker is None:
            return False
        stride = max(1, int(self.config.get("avatar_retreat_control_stride", 10)))
        self._avatar_retreat_tick += 1
        if self._avatar_retreat_tick < stride:
            if not self._avatar_retreat_active_cached:
                return False
            applied = False
            for arm_tag in arm_tags:
                try:
                    applied = self._apply_avatar_retreat(arm_tag) or applied
                except Exception:
                    continue
            return applied
        self._avatar_retreat_tick = 0
        active, _result = self._avatar_retreat_status()
        if not active:
            active = any(
                self._avatar_retreat_release_state.get(tag, {}).get("phase")
                in ("clear", "lower", "release")
                for tag in arm_tags
            )
        self._avatar_retreat_active_cached = bool(active)
        if not active:
            return False
        applied = False
        for arm_tag in arm_tags:
            try:
                applied = self._apply_avatar_retreat(arm_tag) or applied
            except Exception:
                continue
        return applied

    def take_action(self, action: np.ndarray, action_type: Literal["qpos", "qpos_abs", "ee"] = "qpos"):
        """Apply action and return new observation.

        Args:
            action: action vector. For qpos: [left_arm_delta(6), left_gripper(1), right_arm_delta(6), right_gripper(1)]
                    For qpos_abs: [left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]
                    For ee: [left_ee_pose(7), right_ee_pose(7)]
            action_type: "qpos" for joint position deltas, "qpos_abs" for absolute joint targets,
                         "ee" for end-effector poses
        """
        self.advance_interaction_policy_step()
        if self.eval_success:
            return self.get_obs()

        step_index = int(self.take_action_cnt) + 1
        directive = self.interaction_controller.before_base_policy_action(
            self,
            step_index=step_index,
            action=action,
            action_type=action_type,
        )
        if not isinstance(directive, PolicyDirective):
            raise TypeError(
                "before_base_policy_action() must return PolicyDirective, got "
                f"{type(directive).__name__}"
            )
        self.last_policy_directive = directive
        if directive.disposition.value != "proceed":
            raise NotImplementedError(
                "Non-proceed PolicyDirective values are not executable in the "
                "phase-0 interaction seam; migrate the family and runner "
                "together before emitting one."
            )
        self.take_action_cnt += 1
        action = np.asarray(action, dtype=np.float64).ravel()

        interpolate_qpos = bool(self.config.get("interpolate_qpos_actions", False)) and action_type in {"qpos", "qpos_abs"}
        qpos_interp_targets = []
        gripper_close_hold_steps = 0

        def _filtered_gripper_target(current, target):
            nonlocal gripper_close_hold_steps
            if target is None:
                return None
            target = float(target)
            current = float(current)
            max_delta = self.config.get("qpos_gripper_max_delta_per_action")
            if max_delta is not None:
                max_delta = max(0.0, float(max_delta))
                target = current + float(np.clip(target - current, -max_delta, max_delta))
            if target < current:
                gripper_close_hold_steps = max(
                    gripper_close_hold_steps,
                    int(self.config.get("qpos_gripper_close_hold_steps", 0)),
                )
            return target

        if action_type in {"qpos", "qpos_abs"}:
            n_arm = self.robot.right_arm.n_arm
            if self.robot.left_arm is not None:
                left_qpos = self.robot.left_arm.get_arm_qpos()
                right_qpos = self.robot.right_arm.get_arm_qpos()
                if action_type == "qpos_abs":
                    left_target = action[:n_arm]
                    right_target = action[n_arm + 1: 2 * n_arm + 1]
                else:
                    left_target = left_qpos + action[:n_arm]
                    right_target = right_qpos + action[n_arm + 1: 2 * n_arm + 1]
                left_gripper = _filtered_gripper_target(
                    self.robot.get_arm("left").gripper_val,
                    float(action[n_arm]) if len(action) > n_arm else None,
                )
                right_gripper = _filtered_gripper_target(
                    self.robot.get_arm("right").gripper_val,
                    float(action[2 * n_arm + 1]) if len(action) > 2 * n_arm + 1 else None,
                )
                if interpolate_qpos:
                    qpos_interp_targets.extend([
                        ("left", left_qpos, left_target, self.robot.get_arm("left").gripper_val, left_gripper),
                        ("right", right_qpos, right_target, self.robot.get_arm("right").gripper_val, right_gripper),
                    ])
                else:
                    self.robot.set_arm_joints(left_target, "left")
                    self.robot.set_arm_joints(right_target, "right")
                    if left_gripper is not None:
                        self.robot.set_gripper(left_gripper, "left")
                    if right_gripper is not None:
                        self.robot.set_gripper(right_gripper, "right")
            else:
                # Single arm mode: action is [arm_delta_or_qpos(7), gripper(1)]
                right_qpos = self.robot.right_arm.get_arm_qpos()
                right_target = action[:n_arm] if action_type == "qpos_abs" else right_qpos + action[:n_arm]
                right_gripper = _filtered_gripper_target(
                    self.robot.right_arm.gripper_val,
                    float(action[n_arm]) if len(action) > n_arm else None,
                )
                if interpolate_qpos:
                    qpos_interp_targets.append(
                        ("right", right_qpos, right_target, self.robot.right_arm.gripper_val, right_gripper)
                    )
                else:
                    self.robot.set_arm_joints(right_target, "right")
                    if right_gripper is not None:
                        self.robot.set_gripper(right_gripper, "right")
        else:
            # EE pose mode: 8 per arm = [xyz, qw, qx, qy, qz, gripper].
            # Learned EE policies often emit near-zero deltas during the
            # recorded stationary prefix. Re-solving IK for a target already at
            # the current TCP can pick a slightly different null-space solution,
            # which creates drift even though the training label is no-op.
            def _move_if_not_current(pose, arm_tag: str):
                target = np.asarray(pose, dtype=np.float64).ravel()[:7]
                current = np.asarray(
                    self.robot.get_arm(arm_tag).get_ee_pose(), dtype=np.float64,
                ).ravel()[:7]
                pos_err = float(np.linalg.norm(target[:3] - current[:3]))
                q_target = target[3:7] / max(np.linalg.norm(target[3:7]), 1e-12)
                q_current = current[3:7] / max(np.linalg.norm(current[3:7]), 1e-12)
                quat_dot = abs(float(np.clip(np.dot(q_target, q_current), -1.0, 1.0)))
                ang_err = float(2.0 * np.arccos(quat_dot))
                if pos_err <= 1e-3 and ang_err <= 1e-3:
                    return
                self._move_to_pose(target, arm_tag)

            if self.robot.left_arm is not None:
                _move_if_not_current(action[:7], "left")
                _move_if_not_current(action[8:15], "right")
                if len(action) > 7:
                    self.robot.set_gripper(float(action[7]), "left")
                if len(action) > 15:
                    self.robot.set_gripper(float(action[15]), "right")
            else:
                _move_if_not_current(action[:7], "right")
                if len(action) > 7:
                    self.robot.set_gripper(float(action[7]), "right")

        # Run N physics substeps so the PD controller has time to converge to the
        # new target. Default 25 @ dt=0.002 → 50 ms per policy action (≈20 Hz ctrl).
        arm_tags = ("left", "right") if self.robot.left_arm is not None else ("right",)
        action_substeps = int(self.config.get("action_substeps", 25))
        immediate_gripper = bool(self.config.get("qpos_gripper_immediate", False))
        pause_arm_on_close = bool(self.config.get("qpos_gripper_pause_arm_on_close", False))
        for substep in range(action_substeps):
            if qpos_interp_targets:
                frac = (substep + 1) / max(1, action_substeps)
                for arm_tag, start_qpos, target_qpos, start_gripper, target_gripper in qpos_interp_targets:
                    gripper_closing = target_gripper is not None and float(target_gripper) < float(start_gripper)
                    if pause_arm_on_close and gripper_closing:
                        q = start_qpos
                    else:
                        q = start_qpos * (1.0 - frac) + target_qpos * frac
                    self.robot.set_arm_joints(q, arm_tag)
                    if target_gripper is not None:
                        g = (
                            target_gripper
                            if immediate_gripper
                            else start_gripper * (1.0 - frac) + target_gripper * frac
                        )
                        self.robot.set_gripper(float(g), arm_tag)
            self._apply_avatar_retreat_if_needed(arm_tags)
            self.step_sim()
        for _ in range(gripper_close_hold_steps):
            self._apply_avatar_retreat_if_needed(arm_tags)
            self.step_sim()

        self.interaction_controller.after_base_policy_action(
            self,
            step_index=step_index,
            directive=directive,
        )

        return self.get_obs()

