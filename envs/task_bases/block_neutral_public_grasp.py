"""Can-neutral fallback through the public grasp entry, with original transfer control."""
import numpy as np

from ..grasp import GraspPose, tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..planning.base import PlanResult
from ..utils import Pose, to_numpy
from .block_grasp import (BlockGraspPlan, endpoint_matches,
                          _neighbor_endpoint_clearance_ok, refresh_block_obstacles)


_YAWS = (0., np.pi / 2, -np.pi / 2, np.pi)


def _same_target(pick, expected):
    current = np.asarray(pick.get_center(), dtype=float).ravel()
    if (current.shape != (3,) or not np.all(np.isfinite(current))
            or np.linalg.norm(current - expected) > .002):
        raise RuntimeError("block moved after public preflight; refusing a lateral grasp replan")


def _open_paths_clear(task, planner, plans, initial, center, half):
    """Check actual open jaws at every planned waypoint; mutate only the native model."""
    from mplib.pymp import Pose as NativePose
    from mplib.pymp.collision_detection.fcl import Box, CollisionObject, FCLObject

    pre, grasp = plans
    for plan in plans:
        position, velocity = np.asarray(plan.position), np.asarray(plan.velocity)
        if (not plan.success or position.ndim != 2 or position.shape[1] != 7
                or len(position) == 0 or velocity.shape != position.shape
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity))):
            return False
    if (np.max(np.abs(pre.position[0] - initial)) > 1e-6
            or np.max(np.abs(grasp.position[0] - pre.position[-1])) > 1e-6):
        return False
    state = task._block_neighbor_scene
    arm = task.robot.get_arm(state["arm_tag"])
    backend = planner._mplib
    expected_names = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1", "panda_finger_joint2"]
    if list(backend.robot.get_user_joint_names()) != expected_names:
        return False
    if (not np.isfinite(arm.gripper_open) or arm.gripper_open <= 0.
            or np.asarray(center).shape != (3,) or not np.all(np.isfinite(center))
            or not np.isfinite(half) or half <= 0.):
        return False
    quat = np.asarray(to_numpy(task.blocks[state["target"]].get_quat()), dtype=float).ravel()
    if (quat.shape != (4,) or not np.all(np.isfinite(quat))
            or not np.isclose(np.linalg.norm(quat), 1., atol=1e-4)):
        return False
    pose = NativePose(p=center, q=quat)
    target = FCLObject("block_public_target_check", pose,
        [CollisionObject(Box(2*half, 2*half, 2*half), pose)], [NativePose()])
    saved = np.asarray(backend.robot.get_qpos()).copy()
    model = backend.robot.get_fcl_model()
    try:
        for plan in plans:
            for qpos in plan.position:
                full = planner._pad_qpos(qpos).copy()
                if full.shape != (9,):
                    return False
                full[-2:] = arm.gripper_open
                limits = np.asarray(backend.joint_limits)
                if (not np.all(np.isfinite(full)) or np.any(full < limits[:, 0] - .001)
                        or np.any(full > limits[:, 1] + .001)):
                    return False
                if backend.check_for_self_collision(full) or backend.check_for_env_collision(full):
                    return False
                backend.robot.set_qpos(full, True)
                distance = float(model.distance_with(target).min_distance)
                if not np.isfinite(distance) or distance < 0.:
                    return False
        return True
    finally:
        backend.robot.set_qpos(saved, True)


class BlockNeutralPublicGraspMixin:
    # Layout recovery belongs to the two block-neutral experts, not all neutrals.
    BLOCK_NEUTRAL_LAYOUT_RECOVERY = True
    BLOCK_CAN_PUBLIC_GRASP_RECOVERY = False
    # ``execute_plan`` returns on the simulation step that commands its final
    # trajectory row, before PD tracking converges (a 9 mm / 0.060 rad transient there
    # settles to ~0.004 rad within the
    # family's ordinary 60-step recovery).  Hold the already commanded target
    # through ordinary simulation steps until the measured TCP stops moving,
    # then judge arrival.  Bounded, no new command, target or pose setter.
    BLOCK_PUBLIC_ARRIVAL_SETTLE_MIN_STEPS = 50
    BLOCK_PUBLIC_ARRIVAL_SETTLE_MAX_STEPS = 300
    BLOCK_PUBLIC_ARRIVAL_SETTLE_WINDOW = 10
    BLOCK_PUBLIC_ARRIVAL_SETTLE_POSITION_M = 5e-4
    BLOCK_PUBLIC_ARRIVAL_SETTLE_ANGLE_RAD = 1e-3

    def _settle_public_block_arrival(self, arm_tag):
        """Step the unchanged PD hold until the measured TCP is stationary.

        Returns the number of ordinary simulation steps taken.  Stops early on a
        sticky avatar contact so the caller's existing abort sees it unchanged.
        """
        arm = self.robot.get_arm(arm_tag)
        window = max(1, int(self.BLOCK_PUBLIC_ARRIVAL_SETTLE_WINDOW))
        minimum = max(0, int(self.BLOCK_PUBLIC_ARRIVAL_SETTLE_MIN_STEPS))
        maximum = max(minimum, int(self.BLOCK_PUBLIC_ARRIVAL_SETTLE_MAX_STEPS))
        previous = np.asarray(arm.get_ee_pose(), dtype=float).ravel().copy()
        steps = 0
        while steps < maximum:
            for _ in range(window):
                self.step_sim()
            steps += window
            if getattr(self, "avatar_collided", False):
                break
            current = np.asarray(arm.get_ee_pose(), dtype=float).ravel().copy()
            if (current.shape == (7,) and previous.shape == (7,)
                    and np.all(np.isfinite(current)) and np.all(np.isfinite(previous))):
                moved = float(np.linalg.norm(current[:3] - previous[:3]))
                norms = float(np.linalg.norm(current[3:]) * np.linalg.norm(previous[3:]))
                dot = abs(float(np.dot(current[3:], previous[3:]))) / max(norms, 1e-12)
                turned = float(2 * np.arccos(np.clip(dot, 0., 1.)))
                if (steps >= minimum and moved < self.BLOCK_PUBLIC_ARRIVAL_SETTLE_POSITION_M
                        and turned < self.BLOCK_PUBLIC_ARRIVAL_SETTLE_ANGLE_RAD):
                    break
            previous = current
        return steps

    def execute_plan(self, result, arm_tag, gripper_result=None):
        context = getattr(self, "_block_public_execution", None)
        if context is None:
            return super().execute_plan(result, arm_tag, gripper_result=gripper_result)
        _same_target(context["pick"], context["center"])
        index = context["executed"]
        if (arm_tag != context["arm_tag"] or gripper_result is not None
                or index >= len(context["plans"]) or result is not context["plans"][index]):
            raise RuntimeError("unexpected motion during public block acquisition")
        if getattr(self, "avatar_collided", False):
            raise RuntimeError("avatar contact before public block approach leg")
        value = super().execute_plan(result, arm_tag, gripper_result=gripper_result)
        context["executed"] += 1
        settle_steps = self._settle_public_block_arrival(arm_tag)
        _same_target(context["pick"], context["center"])
        if getattr(self, "avatar_collided", False):
            raise RuntimeError("avatar contact during public block approach leg")
        actual = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float).ravel()
        target = context["targets"][index]
        if (actual.shape != (7,) or not np.all(np.isfinite(actual))
                or not np.isclose(np.linalg.norm(actual[3:]), 1., atol=1e-4)):
            raise RuntimeError("invalid measured TCP after public block approach")
        position_error = float(np.linalg.norm(actual[:3] - target.p))
        dot = abs(float(np.dot(actual[3:] / np.linalg.norm(actual[3:]), target.q)))
        angle_error = float(2 * np.arccos(np.clip(dot, 0., 1.)))
        print(f"[blocks-public-grasp] arrival leg={index + 1} settle_steps={settle_steps} "
              f"position_error={position_error:.6f}m orientation_error={angle_error:.6f}rad", flush=True)
        if position_error > .03 or angle_error > .05:
            raise RuntimeError("public block approach missed measured TCP arrival")
        return value

    # Bounded hold-and-retry for preflights rejected while the moving avatar
    # occupies the cube (such rejections are
    # native cloud collisions of the fingers, not joint limits; no approach
    # geometry can help until the avatar moves on).
    BLOCK_PREFLIGHT_HOLD_ROUNDS = 10
    BLOCK_PREFLIGHT_HOLD_STEPS = 200

    def _neutral_block_recovery_enabled(self):
        """All neutral kinds after the avatar has started."""
        return bool(self.BLOCK_CAN_PUBLIC_GRASP_RECOVERY
                    and getattr(self, "_neutral_avatar_started", False)
                    and (getattr(self, "_neutral_avatar_job", None) or {}).get("kind"))

    # Hold at a checked home retreat, commanding a fixed target.  Holding
    # the MEASURED joints every round lets gravity sag compound under that
    # ratchet (TCP 0.838 -> 0.793 m over seven rounds) while the arm stays
    # parked over the slot row inside the avatar's carry, and the avatar's
    # thumb meets link5.  The home configuration is the pose every neutral
    # layout was gated against at reset.
    BLOCK_PREFLIGHT_HOLD_HOME_TOL_RAD = 0.02

    def _block_hold_anchor(self, arm_tag):
        """Arm target for the hold rounds: home after a checked MPLib retreat,
        else the current commanded target.  Never the measured joints."""
        arm = self.robot.get_arm(arm_tag)
        current = np.asarray(arm.get_arm_qpos(), dtype=float).ravel().copy()
        if current.shape != (7,) or not np.all(np.isfinite(current)):
            return None
        cached = getattr(arm, "_cached_target", None)
        commanded = (current.copy() if cached is None
                     else np.asarray(cached, dtype=float).ravel()[:7].copy())
        if commanded.shape != (7,) or not np.all(np.isfinite(commanded)):
            commanded = current.copy()
        home = np.asarray(list(getattr(arm, "homestate", None) or ()), dtype=float).ravel()[:7]
        if home.shape != (7,) or not np.all(np.isfinite(home)):
            return commanded
        if float(np.max(np.abs(current - home))) <= float(self.BLOCK_PREFLIGHT_HOLD_HOME_TOL_RAD):
            return home.copy()
        planner = arm.planner
        backend = getattr(planner, "_mplib", None)
        if backend is None:
            return commanded
        # Same obstacle refresh as every preflight: avatar cloud plus neighbors.
        refresh_block_obstacles(self, arm_tag)
        try:
            result = backend.plan_qpos(
                goal_qposes=[np.asarray(planner._pad_qpos(home.copy()), dtype=float)],
                current_qpos=np.asarray(planner._pad_qpos(current.copy()), dtype=float),
                time_step=1 / 250, planning_time=getattr(planner, "_timeout", None) or 10,
                rrt_range=0.3, verbose=False)
        except Exception as exc:
            print(f"[blocks-grasp] preflight hold retreat unavailable: {exc}", flush=True)
            return commanded
        position = None
        if isinstance(result, dict) and result.get("status") == "Success":
            position = np.asarray(result.get("position", ()), dtype=float)
        if (position is None or position.ndim != 2 or position.shape[0] == 0
                or position.shape[1] < 7 or not np.all(np.isfinite(position))
                or float(np.max(np.abs(position[0, :7] - current))) > 1e-6
                or float(np.max(np.abs(position[-1, :7] - home))) > 1e-3):
            print("[blocks-grasp] preflight hold retreat: no checked route home; "
                  "holding the commanded target", flush=True)
            return commanded
        position = position[:, :7].copy()
        velocity = np.asarray(result.get("velocity", np.zeros_like(position)), dtype=float)
        if velocity.ndim != 2 or velocity.shape[0] != position.shape[0] or velocity.shape[1] < 7:
            velocity = np.zeros_like(position)
        velocity = velocity[:, :7].copy()
        print(f"[blocks-grasp] preflight hold retreat home rows={len(position)}", flush=True)
        self.execute_plan(PlanResult(True, position, velocity), arm_tag)
        settle_steps = self._settle_public_block_arrival(arm_tag)
        if getattr(self, "avatar_collided", False):
            return None
        arrived = np.asarray(arm.get_arm_qpos(), dtype=float).ravel()
        error = (float(np.max(np.abs(arrived - home))) if arrived.shape == (7,)
                 and np.all(np.isfinite(arrived)) else float("inf"))
        print(f"[blocks-grasp] preflight hold retreat arrived settle_steps={settle_steps} "
              f"joint_error={error:.4f}rad", flush=True)
        return home.copy()

    def _block_preflight_hold(self, round_index, arm_tag):
        """Hold a fixed commanded target through ordinary PD steps; True permits a retry."""
        if not self._neutral_block_recovery_enabled():
            return False
        if int(round_index) >= int(self.BLOCK_PREFLIGHT_HOLD_ROUNDS):
            return False
        if getattr(self, "avatar_collided", False):
            return False
        arm = self.robot.get_arm(arm_tag)
        if int(round_index) == 0:
            self._block_preflight_hold_target = self._block_hold_anchor(arm_tag)
        anchor = getattr(self, "_block_preflight_hold_target", None)
        if (anchor is None or np.shape(anchor) != (7,) or not np.all(np.isfinite(anchor))
                or getattr(self, "avatar_collided", False)):
            return False
        # Same real PD hold as the accepted tiny-lift wait: commanded fingers
        # are preserved, nothing is teleported, physics keeps running.  The
        # target is the fixed anchor, so gravity sag cannot ratchet the arm.
        arm.set_arm_joints(np.asarray(anchor, dtype=float).copy())
        print(f"[blocks-grasp] preflight hold {int(round_index) + 1}/"
              f"{int(self.BLOCK_PREFLIGHT_HOLD_ROUNDS)}: "
              f"{int(self.BLOCK_PREFLIGHT_HOLD_STEPS)} steps", flush=True)
        for _ in range(max(0, int(self.BLOCK_PREFLIGHT_HOLD_STEPS))):
            self.step_sim()
            if getattr(self, "avatar_collided", False):
                return False
        return True

    def _block_preflight_fallback(self, pick, place, arm_tag):
        """Return None when nothing was executed; True/False after real motion."""
        if not self._neutral_block_recovery_enabled():
            return None
        if getattr(self, "avatar_collided", False):
            return None
        if getattr(self, "_block_public_execution", None) is not None:
            raise RuntimeError("nested public block acquisition")
        if getattr(self, "_active_block_grasp_plan", None) is not None:
            raise RuntimeError("public fallback encountered an active legacy grasp plan")
        if (pick.get_grasp_yaw is not None or pick.grasp_z_offset != 0.
                or np.any(np.asarray(pick.approach_xy_offset) != 0.)):
            raise ValueError("public block fallback expects ordinary block grasp geometry")
        arm = self.robot.get_arm(arm_tag)
        half = float(pick.radius)
        if not np.isfinite(half) or half <= 0.:
            raise ValueError("invalid block half size")
        self.open_gripper(arm_tag)
        for _ in range(max(0, int(pick.preopen_steps))):
            self.step_sim()
        if getattr(self, "avatar_collided", False):
            return None
        # Match the inherited pick's cache reset to measured full joint state.
        arm._cached_target = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel().copy()
        initial = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("invalid measured block arm joints")
        center = np.asarray(pick.get_center(), dtype=float).ravel().copy()
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("invalid block center")
        grasp_pos = center.copy()
        grasp_pos[2] = max(float(center[2] - .3 * half), self.TABLE_TOP_Z + .002)
        grasps = [GraspPose(f"block_yaw_{index}", grasp_pos - center,
            self._top_down_tcp(grasp_pos, yaw=yaw).q, pre_distance=.12,
            source="manual", scale_frame="runtime") for index, yaw in enumerate(_YAWS)]
        refresh_block_obstacles(self, arm_tag)
        context = {"pick": pick, "center": center, "arm_tag": arm_tag, "plans": (), "executed": 0}

        def check(plans):
            _same_target(pick, center)
            pre, grasp, grasp_link, pre_link = plans
            if not (_open_paths_clear(self, arm.planner, (pre, grasp), initial, center, half)
                    and endpoint_matches(arm.planner, pre, pre_link.to_pose7())
                    and endpoint_matches(arm.planner, grasp, grasp_link.to_pose7())
                    and _neighbor_endpoint_clearance_ok(self, arm.planner, grasp)):
                return False
            context["plans"] = (pre, grasp)
            context["targets"] = (pre_link * arm.tcp_offset, grasp_link * arm.tcp_offset)
            return True

        self._block_public_execution = context
        try:
            selected = self.select_and_execute_grasp("procedural_block", Pose(center, [1., 0., 0., 0.]),
                arm_tag, object_scale=1., candidate_grasps=grasps, max_candidates=4,
                plan_attempts=2, post_plan_check=check, table_z=self.TABLE_TOP_Z)
            if selected is None:
                if context["executed"]:
                    raise RuntimeError("public helper rejected after executing a block approach")
                return None
            if context["executed"] != 2:
                raise RuntimeError("public helper did not execute both checked block approach legs")
        finally:
            self._block_public_execution = None
        _same_target(pick, center)
        chosen = grasps.index(selected[2])
        previous = getattr(self, "_active_block_grasp_plan", None)
        if previous is not None:
            raise RuntimeError("public fallback encountered an active legacy grasp plan")
        # Empty queue: the public helper already physically executed acquisition.
        # Preserve selected yaw and existing pre-close refinement/closed motion rules.
        self._active_block_grasp_plan = BlockGraspPlan(_YAWS[chosen], [])
        try:
            print(f"[pnp] picking {pick.label} -> {place.label}, center={center}, "
                  f"radius={half:.4f}, yaw={np.degrees(_YAWS[chosen]):+.1f}deg", flush=True)
            return self._finish_public_block_transfer(pick, place, arm_tag, center, grasp_pos)
        finally:
            self._active_block_grasp_plan = previous

    def _finish_public_block_transfer(self, pick, place, arm_tag, obj_center, grasp_pos):
        """Original post-grasp behavior, kept local while the shared mixin remains root-owned."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        obj_radius = pick.radius
        approach_xy_offset = np.asarray(pick.approach_xy_offset, dtype=float)
        oriented_grasp = False  # The active family plan supplies the chosen yaw.
        grasp_yaw = 0.
        safe_z = place.transport_z if place.transport_z is not None else self.TABLE_TOP_Z + .38
        _pick_tcp = self._top_down_tcp
        def _pick_screw(pos):
            return self._move_screw(pos, arm_tag)
        _same_target(pick, obj_center)
        # Settle at grasp pose so arm reaches zero velocity before closing.
        for _ in range(100):
            self.step_sim()
        _same_target(pick, obj_center)

        # Close the loop on the reached pose before the one and only finger
        # close. IK branches track the last Cartesian descent differently, so
        # the live error here is more reliable than a fixed catalog bias or a
        # pre-grasp estimate. This is a small pose refinement, not a re-grasp.
        ee_pos = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        desired_grasp_xy = pick.get_center()[:2] + approach_xy_offset
        alignment_delta = desired_grasp_xy - ee_pos[:2]
        if float(np.linalg.norm(alignment_delta)) > 0.002:
            corrected_grasp_pos = grasp_pos.copy()
            corrected_grasp_pos[:2] += alignment_delta
            _pick_screw(corrected_grasp_pos)
            grasp_pos = corrected_grasp_pos
            for _ in range(100):
                self.step_sim()
            corrected_ee = np.asarray(arm.get_ee_pose()[:3], dtype=float)
            residual = desired_grasp_xy - corrected_ee[:2]
            print(
                f"[pnp]   pre-close xy correction={alignment_delta}, "
                f"residual={residual}"
            )

        # Verify position
        ee_pos = np.array(arm.get_ee_pose()[:3])
        print(f"[pnp]   grasp target={grasp_pos}, actual_ee={ee_pos}, "
              f"err={np.linalg.norm(ee_pos - grasp_pos):.4f}m")

        # If Cartesian descent landed the EE far from the grasp target in
        # XY, Genesis IK at each waypoint converged to a bad branch.  Refine
        # by snapping to the top-down link pose via MPLib plan_pose
        # (internal multi-start IK) and re-settling.
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        if xy_err > 0.03:
            # ONE-SHOT policy: a descent that lands >3 cm
            # off is a failed approach — retreat vertically and fail instead
            # of lurching sideways next to the object with a corrective
            # re-plan (which reads as a re-try on video and can knock the
            # object over).
            print(f"[pnp]   descent xy_err={xy_err:.3f}m > 3cm; "
                  f"aborting attempt untouched (one-shot)")
            _pick_screw(np.array([ee_pos[0], ee_pos[1], safe_z]))
            return False

        _same_target(pick, obj_center)
        # Adaptive close target.  Round/small meshes (r<3 cm, e.g. apple) need
        # a shallow squeeze (`r-0.005`) — deeper commands let fingers overpower
        # mesh contact before caging, shoving the object sideways.  Flat/large
        # meshes (r>=3 cm, e.g. wooden block) need a deep squeeze (`r*0.7`) —
        # fingers contact a flat face squarely, the block wins the PD push and
        # stops fingers at the face with ~100 N firm grip.  Verified: deliver
        # apple → 4/4 with shallow; cate-coop block → 3/3 with deep.
        if pick.close_value is not None:
            gripper_close = float(np.clip(pick.close_value, 0.0, 1.0))
        else:
            r = float(obj_radius)
            if r < 0.030:
                close_target = max(0.010, r - 0.005)
            else:
                close_target = max(0.010, r * 0.7)
            gripper_close = close_target / 0.04
        self.set_gripper(gripper_close, arm_tag)
        for _ in range(200):
            self.step_sim()
        if pick.after_close is not None:
            pick.after_close()
            for _ in range(40):
                self.step_sim()

        lift_start = grasp_pos.copy()
        lift_mid = grasp_pos.copy()
        lift_mid[2] = obj_center[2] + obj_radius  # lift to just above object top
        if oriented_grasp:
            TopDownPickPlaceMixin._move_cartesian(
                self, lift_start, lift_mid, arm_tag,
                n_steps=30, sim_per_step=30, tcp_yaw=grasp_yaw,
            )
        else:
            self._move_cartesian(
                lift_start, lift_mid, arm_tag,
                n_steps=30, sim_per_step=30,
            )

        # Debug: check grip
        from ..robot.franka_robot import _get_dof_idx, to_numpy
        qpos_after = to_numpy(arm.entity.get_qpos())
        finger_qpos = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_qpos.append(float(qpos_after[idx]))
        obj_pos_after = pick.get_center()
        print(f"[pnp]   fingers={finger_qpos}, obj={obj_pos_after}")

        if pick.max_grasp_offset is not None:
            ee_pos_after = np.asarray(arm.get_ee_pose()[:3], dtype=float)
            grasp_offset = float(np.linalg.norm(
                np.asarray(obj_pos_after, dtype=float) - ee_pos_after,
            ))
            max_grasp_offset = float(pick.max_grasp_offset)
            if grasp_offset > max_grasp_offset:
                print(
                    f"[pnp]   grasp retention failed: object/TCP offset "
                    f"{grasp_offset:.4f}m > {max_grasp_offset:.4f}m"
                )
                return False

        # Lift to safe height via screw motion (straight-line, gripper held)
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z])
        _pick_screw(lift_pos)

        obj_pos_lifted = pick.get_center()
        lift_delta = float(obj_pos_lifted[2] - obj_pos_after[2])
        print(f"[pnp]   after lift: {obj_pos_lifted} (dz={lift_delta:.4f})")
        if pick.require_lift:
            min_lift = (
                float(pick.min_lift_delta)
                if pick.min_lift_delta is not None
                else max(0.03, 0.5 * float(obj_radius))
            )
            if lift_delta < min_lift:
                print(
                    f"[pnp]   lift retention failed: dz={lift_delta:.4f}m "
                    f"< {min_lift:.4f}m"
                )
                return False

        ee_pos_lifted = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        held_offset_lift = np.asarray(obj_pos_lifted, dtype=float) - ee_pos_lifted

        if place.stop_after_lift:
            return True

        # Transport to above drop target at safe_z.
        target_tcp_xy = np.asarray(place.pos[:2], dtype=float).copy()
        place_xyz = np.array([
            target_tcp_xy[0], target_tcp_xy[1], safe_z,
        ], dtype=float)
        _pick_screw(place_xyz)

        # Settle so the object stops swinging in the gripper.
        place_settle_steps = int(getattr(
            self, "PICK_PLACE_SETTLE_ABOVE_PLACE_STEPS",
            getattr(self, "config", {}).get("pick_place_settle_above_place_steps", 100),
        ))
        for _ in range(max(0, place_settle_steps)):
            self.step_sim()

        obj_pos_transport = pick.get_center()
        print(f"[pnp]   above place: tcp={place_xyz}, obj={obj_pos_transport}")
        ee_pos_transport = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        held_offset_transport = (
            np.asarray(obj_pos_transport, dtype=float) - ee_pos_transport
        )
        if pick.transport_retention_tolerance is not None:
            retention_error = float(np.linalg.norm(
                held_offset_transport - held_offset_lift,
            ))
            tolerance = float(pick.transport_retention_tolerance)
            if retention_error > tolerance:
                print(
                    f"[pnp]   transport retention failed: offset change "
                    f"{retention_error:.4f}m > {tolerance:.4f}m"
                )
                return False
        # Descend to release height.  ``drop_from_z`` overrides place.pos[2]
        # (hover-release pattern); otherwise descend to place.pos[2].  A
        # single vertical screw — multi-segment descents have been observed
        # to flip the wrist on cate_coop and hurl the object.
        if place.support_z is not None:
            if pick.get_half_height is None:
                raise ValueError(
                    "PlaceSpec.support_z requires PickSpec.get_half_height"
                )
            object_half_height = float(pick.get_half_height())
            release_z = self.support_release_tcp_z(
                place.support_z,
                object_half_height,
                held_offset_transport[2],
                place.support_clearance,
            )
            release_z = min(float(safe_z), release_z)
            print(
                f"[pnp]   support release: support_z={place.support_z:.4f}, "
                f"half_h={object_half_height:.4f}, "
                f"held_dz={held_offset_transport[2]:+.4f}, "
                f"tcp_z={release_z:.4f}"
            )
        elif place.drop_from_z is not None:
            release_z = float(place.drop_from_z)
        elif len(place.pos) >= 3:
            release_z = float(place.pos[2])
        else:
            release_z = safe_z  # no explicit z → release at transit height
        if abs(release_z - safe_z) > 1e-4:
            descend_pos = np.array([
                target_tcp_xy[0], target_tcp_xy[1], release_z,
            ])
            if getattr(place, "descent_seeded", False):
                # Seeded IK + RRT descent (preserves the wrist branch from the
                # above-target pose) for deep targets where the screw fails.
                descend_link = tcp_to_link_pose(
                    _pick_tcp(descend_pos), tcp_offset,
                )
                self._move_seeded(descend_link.to_pose7(), arm_tag)
            else:
                _pick_screw(descend_pos)
            for _ in range(50):
                self.step_sim()

        if place.release:
            # Release.  Let the object drop under gravity.
            self.open_gripper(arm_tag)
            release_open_steps = int(getattr(
                self, "PICK_PLACE_RELEASE_OPEN_STEPS",
                getattr(self, "config", {}).get("pick_place_release_open_steps", 40),
            ))
            for _ in range(max(0, release_open_steps)):
                self.step_sim()
            after_release = getattr(self, "_after_pick_place_release", None)
            if after_release is not None:
                release_pos = np.array(
                    [place.pos[0], place.pos[1], release_z],
                    dtype=float,
                )
                after_release(pick, place, arm_tag, release_pos)
            release_settle_steps = int(getattr(
                self, "PICK_PLACE_RELEASE_SETTLE_STEPS",
                getattr(self, "config", {}).get("pick_place_release_settle_steps", 210),
            ))
            for _ in range(max(0, release_settle_steps)):
                self.step_sim()
            obj_pos_release = pick.get_center()
            print(f"[pnp]   after release: {obj_pos_release}")
        else:
            # Hold object in gripper at deliver pose (deliver-to-human).
            hold_steps = int(getattr(
                self, "PICK_PLACE_HOLD_STEPS",
                getattr(self, "config", {}).get("pick_place_hold_steps", 100),
            ))
            for _ in range(max(0, hold_steps)):
                self.step_sim()
            obj_pos_held = pick.get_center()
            print(f"[pnp]   holding (no release): {obj_pos_held}")

        return True
