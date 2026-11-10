"""Experimental, post-exchange cube hold; no contact-physics overrides."""

import json

import numpy as np

from .take_release import _angle, _state


def establish_carry_hold(task, arm_tag):
    """Ramp a detached cube's position target while holding the measured arm pose.

    The exchange stays at .35: squeezing the avatar-constrained object more
    tightly ejects it. This separate, opt-in phase runs only after
    the lift and completed avatar retraction. It changes neither gains nor
    force limits, and a failed hold never opens or resumes transport.
    """
    if not task.config.get("take_post_handover_hold", False):
        return True
    if task.obj_name != "073_rubikscube":
        return True
    steps = 0
    try:
        if not task.plan_success:
            raise ValueError("prior planning failure")
        if getattr(task, "_gripper_attached", None):
            raise ValueError("kinematic gripper attachment is incompatible with carry hold")
        if task._avatar_retreat_enabled() or getattr(task, "avatar_retreat_checker", None) is not None:
            raise ValueError("avatar retreat is incompatible with carry hold")
        if task.avatar is not None:
            if not task.avatar.spare():
                raise ValueError("avatar retraction is incomplete")
            if any(item is not None for item in task.avatar.robot.attached_object):
                raise ValueError("avatar still has an attached object")
            if task.avatar.robot.two_hand_attached_object is not None:
                raise ValueError("avatar still has a two-hand attachment")
        arm = task.robot.get_arm(arm_tag)
        start = float(arm.gripper_val)
        if not np.isclose(start, .35, atol=1e-8, rtol=0):
            raise ValueError("unexpected cube exchange command")
        dt = float(task.SIMULATION_DT)
        if not np.isfinite(dt) or not .0001 <= dt <= .01:
            raise ValueError("unsupported simulation step")
        q = np.asarray(arm.get_arm_qpos(), dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("invalid measured arm hold target")
        initial = _state(task, arm)
        reference = initial["position"] - initial["tcp"][:3]
        ramp_steps = max(1, round(.15 / dt))
        settle_steps = max(1, round(.15 / dt))
        for index in range(ramp_steps + settle_steps):
            command = start + (.30 - start) * min(1., (index + 1) / ramp_steps)
            arm.set_arm_joints(q)
            task.robot.set_gripper(command, arm_tag)
            task.step_sim()
            steps += 1
            state = _state(task, arm)
            offset = state["position"] - state["tcp"][:3]
            if np.linalg.norm(offset - reference) > .015 or _angle(
                    state["quaternion"], initial["quaternion"]) > .15:
                raise ValueError("object moved relative to gripper during carry hold")
            if np.linalg.norm(state["tcp"][:3] - initial["tcp"][:3]) > .008 or _angle(
                    state["tcp"][3:], initial["tcp"][3:]) > .05:
                raise ValueError("arm moved during stationary carry hold")
            if not task.plan_success:
                raise ValueError("failure latched during carry hold")
        print("[take-carry-hold] " + json.dumps({
            "stage": "complete", "steps": steps, "start_command": start,
            "carry_command": float(arm.gripper_val),
            "relative_offset_change_m": (offset - reference).tolist(),
        }, allow_nan=False), flush=True)
        return True
    except Exception as exc:
        task.plan_success = False
        print("[take-carry-hold] " + json.dumps({
            "stage": "rejected_without_opening", "steps": steps,
            "reason": f"{type(exc).__name__}: {exc}",
        }, allow_nan=False), flush=True)
        return False
