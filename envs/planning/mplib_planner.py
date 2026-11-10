"""MPLib-based motion planner with TOPP-RA trajectory smoothing."""

import os
import numpy as np
import mplib
import mplib.pymp as pymp
import toppra as ta
import time

from .base import Planner, PlanResult

ta.setup_logging("CRITICAL")


def _plan_diag_enabled() -> bool:
    """``MPLIB_PLAN_DIAG=1``: log why each screw plan / RRT start fails."""
    return os.environ.get("MPLIB_PLAN_DIAG", "").strip() == "1"


def _collision_names(world) -> list[str]:
    try:
        return sorted({
            f"{c.link_name1}@{c.object_name1}|{c.link_name2}@{c.object_name2}"
            for c in world.check_collision()
        })
    except Exception as exc:  # pragma: no cover - diagnostics only
        return [f"<check_collision failed: {exc}>"]


def _screw_core(backend, goal_pose, current_qpos, *, qpos_step, time_step,
                wrt_world, verbose, finish_on_reach=False):
    """MPLib 0.2.1 ``Planner.plan_screw``, line for line, plus a failure
    reason.  Returns ``(result_dict, reason)``; ``reason`` is None on success.

    ``finish_on_reach``: when the twist left to integrate is already below
    MPLib's own 1e-4 progress floor (the previous step landed on the goal),
    finish instead of failing.  MPLib fails there because its last, scaled
    step is then shorter than that floor.  This only changes calls MPLib
    fails.
    """
    current_qpos = backend.pad_move_group_qpos(np.asarray(current_qpos).copy())
    backend.robot.set_qpos(current_qpos, True)
    if wrt_world:
        goal_pose = backend._transform_goal_to_wrt_base(goal_pose)

    def skew(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0],
        ])

    def pose2exp_coordinate(pose):
        def rot2so3(rotation):
            if np.isclose(rotation.trace(), 3):
                return np.zeros(3), 1
            if np.isclose(rotation.trace(), -1):
                return np.zeros(3), -1e6
            theta = np.arccos((rotation.trace() - 1) / 2)
            omega = 1 / 2 / np.sin(theta) * np.array([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]).T
            return omega, theta

        pose_mat = pose.to_transformation_matrix()
        omega, theta = rot2so3(pose_mat[:3, :3])
        if theta < -1e5:
            return omega, theta
        ss = skew(omega)
        inv_left_jacobian = (
            np.eye(3) / theta
            - 0.5 * ss
            + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
        )
        v = inv_left_jacobian @ pose_mat[:3, 3]
        return np.concatenate([v, omega]), theta

    pin = backend.pinocchio_model
    pin.compute_forward_kinematics(current_qpos)
    ee_index = backend.link_name_2_idx[backend.move_group]
    relative_pose = goal_pose * pin.get_link_pose(ee_index).inv()
    omega, theta = pose2exp_coordinate(relative_pose)
    if theta < -1e4:
        return {"status": "screw plan failed."}, {"kind": "rot180", "step": 0}
    omega = omega.reshape((-1, 1)) * theta
    total = float(np.linalg.norm(omega))

    move_joint_idx = backend.move_group_joint_indices
    limits = backend.joint_limits
    path = [np.copy(current_qpos[move_joint_idx])]
    step = 0
    while True:
        step += 1
        if finish_on_reach and len(path) > 1 and np.linalg.norm(omega) < 1e-4:
            out = _screw_topp(backend, path, time_step, verbose)
            out["finished_on_reach"] = True
            return out, None
        pin.compute_full_jacobian(current_qpos)
        J = pin.get_link_jacobian(ee_index, local=False)
        delta_q = np.linalg.pinv(J) @ omega
        delta_q *= qpos_step / (np.linalg.norm(delta_q))
        delta_twist = J @ delta_q

        flag = False
        if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
            ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
            delta_q = delta_q * ratio
            delta_twist = delta_twist * ratio
            flag = True

        current_qpos += delta_q.reshape(-1)
        omega -= delta_twist

        bad_joints = [
            i for i in range(len(current_qpos))
            if current_qpos[i] < limits[i][0] - 1e-3
            or current_qpos[i] > limits[i][1] + 1e-3
        ]
        backend.planning_world.set_qpos_all(current_qpos[move_joint_idx])
        collide = backend.planning_world.is_state_colliding()

        if np.linalg.norm(delta_twist) < 1e-4 or collide or bad_joints:
            reason = {"step": step, "total_twist": total,
                      "left_twist": float(np.linalg.norm(omega))}
            if bad_joints:
                reason["kind"] = "joint_limit"
                reason["joints"] = [
                    (int(i), round(float(current_qpos[i]), 4),
                     [round(float(limits[i][0]), 4), round(float(limits[i][1]), 4)])
                    for i in bad_joints
                ]
            elif collide:
                reason["kind"] = "collision"
            elif flag:
                reason["kind"] = "end_stall"
            else:
                reason["kind"] = "singular_stall"
            if collide:
                reason["collide"] = _collision_names(backend.planning_world)
            return {"status": "screw plan failed"}, reason

        path.append(np.copy(current_qpos[move_joint_idx]))

        if flag:
            return _screw_topp(backend, path, time_step, verbose), None


def _screw_topp(backend, path, time_step, verbose):
    if verbose:
        ta.setup_logging("INFO")
    else:
        ta.setup_logging("WARNING")
    times, pos, vel, acc, duration = backend.TOPP(np.vstack(path), time_step)
    return {
        "status": "Success",
        "time": times,
        "position": pos,
        "velocity": vel,
        "acceleration": acc,
        "duration": duration,
    }


def _start_collisions(backend, current_qpos) -> list[str]:
    q = np.clip(np.asarray(current_qpos, dtype=np.float64),
                backend.joint_limits[:, 0], backend.joint_limits[:, 1])
    q = backend.pad_move_group_qpos(q)
    backend.robot.set_qpos(q, True)
    return _collision_names(backend.planning_world)


def _install_backend_hooks(backend) -> None:
    """Route every ``backend.plan_screw`` call (this adapter and the task
    code that calls the MPLib planner directly) through ``_screw_core``."""
    if getattr(backend, "_r3p_hooks", False):
        return

    def plan_screw(goal_pose, current_qpos, *, qpos_step=0.1, time_step=0.1,
                   wrt_world=True, verbose=False):
        out, reason = _screw_core(
            backend, goal_pose, current_qpos, qpos_step=qpos_step,
            time_step=time_step, wrt_world=wrt_world, verbose=verbose,
            finish_on_reach=True,
        )
        if reason is None:
            if _plan_diag_enabled() and out.get("finished_on_reach"):
                print(f"[plan-diag] screw finished on reach qpos_step={qpos_step:g}", flush=True)
            return out
        if _plan_diag_enabled():
            print(f"[plan-diag] screw fail qpos_step={qpos_step:g} {reason}", flush=True)
        return out

    stock_plan_qpos = backend.plan_qpos

    def plan_qpos(goal_qposes, current_qpos, **kwargs):
        # Checked before the stock call, which resets the robot to the same
        # start state itself, so the diagnostics cannot change the plan.
        start = []
        if _plan_diag_enabled():
            try:
                start = _start_collisions(backend, current_qpos)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[plan-diag] rrt diag failed: {exc}", flush=True)
        out = stock_plan_qpos(goal_qposes, current_qpos, **kwargs)
        if start:
            try:
                pos = np.asarray(out.get("position", [])) if out else np.array([])
                jump = None
                if out and out.get("status") == "Success" and pos.ndim == 2 and len(pos):
                    cur = np.asarray(current_qpos, dtype=np.float64).ravel()
                    n = min(pos.shape[1], cur.shape[0])
                    jump = round(float(np.max(np.abs(pos[0, :n] - cur[:n]))), 4)
                print(f"[plan-diag] rrt invalid start status={out.get('status') if out else None} "
                      f"first_jump={jump} collide={start}", flush=True)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[plan-diag] rrt diag failed: {exc}", flush=True)
        return out

    backend.plan_screw = plan_screw
    backend.plan_qpos = plan_qpos
    backend._r3p_hooks = True


class MplibPlanner(Planner):
    """Motion planner using MPLib (RRT-Connect) with TOPP-RA smoothing."""

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        move_group: str,
        base_pose: np.ndarray,
        entity=None,
        n_arm: int = 6,
        num_waypoints: int = 200,
        timeout: float = None,
        max_retry: int = 5,
        arm_joint_names: list[str] = None,
        arm_joint_index: dict[str, int] = None,
        entity_ee_link_name: str = None,
        package_keyword: str = "",
    ):
        self.n_arm = n_arm
        self.num_waypoints = num_waypoints
        self.max_retry = max_retry
        self._timeout = timeout
        self._entity = entity
        self._ee_link_name = move_group
        self._entity_ee_link_name = entity_ee_link_name or move_group
        self._arm_joint_names = list(arm_joint_names) if arm_joint_names else None
        self._arm_joint_index = dict(arm_joint_index) if arm_joint_index else None

        # Build mplib planner
        links = [link.name for link in entity.links] if entity else []
        joints = [joint.name for joint in entity.joints] if entity else []

        if "base_link" not in links:
            self._mplib = mplib.Planner(
                urdf=urdf_path, srdf=srdf_path,
                move_group=move_group, use_convex=False,
                new_package_keyword=package_keyword,
            )
        else:
            self._mplib = mplib.Planner(
                urdf=urdf_path, srdf=srdf_path,
                move_group=move_group,
                user_link_names=links, user_joint_names=joints,
                use_convex=False,
                new_package_keyword=package_keyword,
            )

        # Set base pose
        pose7 = np.asarray(base_pose, dtype=np.float64).ravel()
        p = pose7[:3].astype(np.float64)
        q = pose7[3:7].astype(np.float64)
        self._mplib.set_base_pose(pymp.Pose(p=p, q=q))
        self._base_pose = pose7.copy()
        _install_backend_hooks(self._mplib)

    def _to_mplib_pose(self, target_pose) -> pymp.Pose:
        """Convert target pose to mplib Pose."""
        arr = np.asarray(target_pose, dtype=np.float64).ravel()
        if hasattr(target_pose, "p"):
            arr = np.concatenate([
                np.asarray(target_pose.p).ravel(),
                np.asarray(target_pose.q).ravel(),
            ])
        p = arr[:3].astype(np.float64)
        q = arr[3:7].astype(np.float64)
        return pymp.Pose(p=p, q=q)

    def _pad_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Pad qpos to match mplib expected size."""
        qpos = np.asarray(qpos, dtype=np.float64)
        try:
            n_user = len(self._mplib.robot.get_user_joint_names())
            if qpos.shape[0] < n_user:
                qpos = np.concatenate([qpos, np.zeros(n_user - qpos.shape[0], dtype=np.float64)])
        except Exception:
            pass
        return qpos

    @staticmethod
    def _bounded_paths_debug_enabled() -> bool:
        """Whether to use the experimental bounded-joint planning path.

        This is deliberately an environment flag rather than a default config
        option while the implementation is being validated.  With the flag
        absent, planner behavior is byte-for-byte the existing MPLib path.
        """
        return os.environ.get("DEBUG_BOUNDED_JOINT_PATHS", "").strip() == "1"

    @staticmethod
    def _joint_path_metrics(
        current_qpos: np.ndarray,
        position: np.ndarray,
        n_arm: int,
    ) -> dict[str, np.ndarray | float]:
        """Return joint-space path-quality metrics for one arm trajectory.

        Per-step continuity is not enough for a redundant arm: a path can use
        hundreds of tiny steps to wind a wrist through its complete range.
        The cumulative travel and start-relative excursion expose that failure
        without assuming which joint is redundant.
        """
        q0 = np.asarray(current_qpos, dtype=np.float64).ravel()[:n_arm]
        pos = np.asarray(position, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] < n_arm:
            return {
                "travel": np.full(n_arm, np.inf),
                "excursion": np.full(n_arm, np.inf),
                "max_step": np.inf,
                "weighted_length": np.inf,
            }
        arm = pos[:, :n_arm]
        if not np.allclose(arm[0], q0, atol=1e-6, rtol=0.0):
            arm = np.vstack([q0, arm])
        delta = np.diff(arm, axis=0)
        return {
            "travel": np.sum(np.abs(delta), axis=0),
            "excursion": np.max(np.abs(arm - q0[None, :]), axis=0),
            "max_step": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "weighted_length": float(np.sum(np.linalg.norm(delta, axis=1))),
        }

    def _debug_path_quality_ok(
        self,
        current_qpos: np.ndarray,
        position: np.ndarray,
        *,
        motion_kind: str,
        log: bool,
    ) -> bool:
        """Experimental hard gate against globally unreasonable joint paths."""
        metrics = self._joint_path_metrics(current_qpos, position, self.n_arm)
        travel = np.asarray(metrics["travel"], dtype=np.float64)
        excursion = np.asarray(metrics["excursion"], dtype=np.float64)

        # Local Cartesian motions should remain on the starting IK branch.
        # Global RRT transits get a wider allowance for obstacle detours, but
        # still may not sweep a bounded Franka joint through most of its range.
        if motion_kind == "screw":
            max_excursion = 1.75  # 100 deg on any one joint
            max_travel = 2.25     # permits modest backtracking, not winding
        else:
            max_excursion = 2.80  # 160 deg
            max_travel = 3.50     # 200 deg cumulative

        bad = np.flatnonzero(
            (excursion > max_excursion) | (travel > max_travel)
        )
        if bad.size:
            if log:
                details = ", ".join(
                    f"j{i + 1}: excursion={excursion[i]:.3f}, "
                    f"travel={travel[i]:.3f}rad"
                    for i in bad
                )
                print(
                    f"[bounded_path] reject {motion_kind} trajectory ({details})"
                )
            return False
        return True

    def _plan_bounded_screw(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        *,
        time_step: float = 1 / 250,
        qpos_step: float = 0.05,
        log: bool = True,
    ) -> PlanResult:
        """Plan a branch-preserving Cartesian path with MPLib kinematics.

        MPLib's stock ``plan_screw`` integrates ``pinv(J) @ twist``.  That is
        minimum-norm for each individual step, but it has no memory of total
        joint travel and no cost for approaching a joint limit.  On a 7-DOF
        arm it can therefore drift through the null space for several full
        radians while the TCP orientation barely changes.

        This experimental version keeps all kinematics and collision checking
        inside MPLib, but uses a joint-limit-weighted damped pseudoinverse and
        rejects any path that leaves a bounded neighborhood of its starting IK
        branch.  It is reachable only through ``DEBUG_BOUNDED_JOINT_PATHS=1``.
        """
        backend = self._mplib
        q_full = backend.pad_move_group_qpos(
            np.asarray(current_qpos, dtype=np.float64).copy()
        )
        backend.robot.set_qpos(q_full, True)

        goal = self._to_mplib_pose(target_pose)
        goal = backend._transform_goal_to_wrt_base(goal)
        pin = backend.pinocchio_model
        ee_index = backend.link_name_2_idx[backend.move_group]
        move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
        q_start = q_full.copy()

        def _skew(vec):
            return np.array([
                [0.0, -vec[2], vec[1]],
                [vec[2], 0.0, -vec[0]],
                [-vec[1], vec[0], 0.0],
            ])

        def _pose_log(pose):
            matrix = pose.to_transformation_matrix()
            rotation = matrix[:3, :3]
            trace_arg = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
            theta = float(np.arccos(trace_arg))
            if theta < 1e-9:
                return np.concatenate([matrix[:3, 3], np.zeros(3)])
            if np.pi - theta < 1e-6:
                raise ValueError("180-degree screw rotation is ill-conditioned")
            omega_axis = np.array([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]) / (2.0 * np.sin(theta))
            omega_hat = _skew(omega_axis)
            inv_left_jacobian = (
                np.eye(3) / theta
                - 0.5 * omega_hat
                + (1.0 / theta - 0.5 / np.tan(theta / 2.0))
                * omega_hat @ omega_hat
            )
            linear = inv_left_jacobian @ matrix[:3, 3]
            return np.concatenate([linear, omega_axis]) * theta

        pin.compute_forward_kinematics(q_full)
        relative = goal * pin.get_link_pose(ee_index).inv()
        try:
            remaining = _pose_log(relative).reshape(-1)
        except ValueError as exc:
            if log:
                print(f"[bounded_path] screw rejected: {exc}")
            return PlanResult(False)

        limits = np.asarray(backend.joint_limits, dtype=np.float64)
        lo = limits[move_idx, 0]
        hi = limits[move_idx, 1]
        width = np.maximum(hi - lo, 1e-6)
        start_arm = q_start[move_idx]
        q_arm = start_arm.copy()
        path = [q_arm.copy()]
        max_excursion = 1.75
        damping = 0.03

        for _ in range(2000):
            pin.compute_full_jacobian(q_full)
            jac_full = np.asarray(pin.get_link_jacobian(ee_index, local=False))
            jac = jac_full[:, move_idx]

            # Penalize joints close to either hard limit and joints that have
            # already consumed much of the local motion budget.  Weighted DLS
            # then satisfies the Cartesian twist using less constrained joints.
            limit_fraction = np.minimum(q_arm - lo, hi - q_arm) / width
            limit_weight = 1.0 + 0.02 / np.square(np.maximum(limit_fraction, 0.02))
            used_fraction = np.abs(q_arm - start_arm) / max_excursion
            excursion_weight = 1.0 + 8.0 * np.power(used_fraction, 4)
            weight_inv = np.diag(1.0 / (limit_weight * excursion_weight))
            task_matrix = jac @ weight_inv @ jac.T + damping ** 2 * np.eye(6)
            try:
                delta = weight_inv @ jac.T @ np.linalg.solve(task_matrix, remaining)
            except np.linalg.LinAlgError:
                if log:
                    print("[bounded_path] screw rejected: singular weighted Jacobian")
                return PlanResult(False)

            delta_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta_norm) or delta_norm < 1e-10:
                if log:
                    print("[bounded_path] screw rejected: no Cartesian progress")
                return PlanResult(False)
            if delta_norm > qpos_step:
                delta *= qpos_step / delta_norm

            achieved = jac @ delta
            achieved_norm = float(np.linalg.norm(achieved))
            remaining_norm = float(np.linalg.norm(remaining))
            if achieved_norm > remaining_norm and achieved_norm > 0.0:
                ratio = remaining_norm / achieved_norm
                delta *= ratio
                achieved *= ratio

            candidate = q_arm + delta
            if (
                np.any(candidate < lo - 1e-3)
                or np.any(candidate > hi + 1e-3)
                or np.any(np.abs(candidate - start_arm) > max_excursion)
            ):
                if log:
                    print("[bounded_path] screw rejected: joint branch budget exceeded")
                return PlanResult(False)

            q_full[move_idx] = candidate
            backend.planning_world.set_qpos_all(candidate)
            if backend.planning_world.is_state_colliding():
                if log:
                    print("[bounded_path] screw rejected: collision")
                return PlanResult(False)

            q_arm = candidate
            path.append(q_arm.copy())
            remaining -= achieved

            if float(np.linalg.norm(remaining)) < 1e-4:
                break
        else:
            if log:
                print("[bounded_path] screw rejected: iteration limit")
            return PlanResult(False)

        # Verify the integrated endpoint in the same MPLib FK model before
        # time-parameterizing it.  Pose.distance combines translation and
        # quaternion error and is the same metric MPLib's IK uses.
        pin.compute_forward_kinematics(q_full)
        endpoint_error = float(goal.distance(pin.get_link_pose(ee_index)))
        # The weighted DLS integration converges to about 11 mm on long
        # workspace-spanning translations in the real Franka model.  Keep the
        # debug gate well inside the shared grasp helper's 30 mm FK tolerance,
        # but do not discard a bounded Cartesian path for a 1 mm numerical
        # miss and replace it with a less constrained global transit.
        endpoint_tolerance = 0.015
        if endpoint_error > endpoint_tolerance:
            if log:
                print(
                    f"[bounded_path] screw rejected: endpoint error "
                    f"{endpoint_error:.4f} > {endpoint_tolerance:.3f}"
                )
            return PlanResult(False)

        try:
            _times, pos, vel, _acc, _duration = backend.TOPP(
                np.vstack(path), time_step
            )
        except Exception as exc:
            if log:
                print(f"[bounded_path] screw TOPP failed: {exc}")
            return PlanResult(False)

        if not self._debug_path_quality_ok(
            current_qpos, pos, motion_kind="screw", log=log
        ):
            return PlanResult(False)
        if log:
            metrics = self._joint_path_metrics(current_qpos, pos, self.n_arm)
            print(
                "[bounded_path] screw accepted: max excursion="
                f"{np.max(metrics['excursion']):.3f}rad, max travel="
                f"{np.max(metrics['travel']):.3f}rad"
            )
        return PlanResult(True, pos, vel)

    def _plan_bounded_transit(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        *,
        log: bool = True,
    ) -> PlanResult:
        """Plan to the closest reasonable MPLib IK branch in debug mode.

        Stock ``plan_pose`` passes every sampled IK goal to RRT-Connect at
        once.  RRT-Connect stops at the first branch it connects, even when a
        much closer goal configuration was sampled in the same call.  Rank
        those collision-checked MPLib IK solutions in joint space and plan to
        them one at a time so feasibility does not override path quality.
        """
        backend = self._mplib
        q_full = backend.pad_move_group_qpos(
            np.asarray(current_qpos, dtype=np.float64).copy()
        )
        goal = self._to_mplib_pose(target_pose)
        goal = backend._transform_goal_to_wrt_base(goal)

        status, goals = backend.IK(
            goal_pose=goal,
            start_qpos=q_full,
            n_init_qpos=max(20, 5 * self.max_retry),
            threshold=0.001,
            return_closest=False,
            verbose=log,
        )
        if status != "Success" or goals is None:
            if log:
                print(f"[bounded_path] transit IK failed: {status}")
            return PlanResult(False)

        goals = np.asarray(goals, dtype=np.float64)
        if goals.ndim == 1:
            goals = goals[None, :]
        move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
        q_arm = q_full[move_idx]
        limits = np.asarray(backend.joint_limits, dtype=np.float64)[move_idx]
        width = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)

        candidate_scores = []
        for index, candidate in enumerate(goals):
            candidate_arm = candidate[move_idx]
            direct = float(np.linalg.norm(candidate_arm - q_arm))
            margin = np.minimum(
                candidate_arm - limits[:, 0], limits[:, 1] - candidate_arm
            ) / width
            # Prefer nearby branches, then break close ties in favor of goals
            # with more room before a hard limit.  The limit term is small so
            # it cannot make a distant IK branch look artificially attractive.
            limit_penalty = float(np.sum(1.0 / np.maximum(margin, 0.02)))
            candidate_scores.append((direct + 0.002 * limit_penalty, index))

        for score, index in sorted(candidate_scores):
            candidate = goals[index]
            try:
                result = backend.plan_qpos(
                    goal_qposes=[candidate],
                    current_qpos=q_full,
                    time_step=1 / 250,
                    planning_time=self._timeout or 10,
                    rrt_range=0.1,
                    verbose=log,
                )
            except Exception as exc:
                if log:
                    print(f"[bounded_path] transit plan_qpos exception: {exc}")
                continue
            if not result or result.get("status") != "Success":
                continue
            pos = np.asarray(result["position"])
            if not self._debug_path_quality_ok(
                current_qpos, pos, motion_kind="transit", log=log
            ):
                continue
            vel = result.get("velocity", np.zeros_like(pos))
            if log:
                metrics = self._joint_path_metrics(current_qpos, pos, self.n_arm)
                print(
                    f"[bounded_path] transit accepted IK score={score:.3f}: "
                    f"max excursion={np.max(metrics['excursion']):.3f}rad, "
                    f"max travel={np.max(metrics['travel']):.3f}rad"
                )
            return PlanResult(True, pos, vel)

        if log:
            print("[bounded_path] no bounded transit candidate")
        return PlanResult(False)

    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Plan collision-free path using RRT-Connect.

        When DEBUG_TELEPORT=1, skips RRT and uses IK with 1 waypoint for fast debugging.
        """
        # DEBUG_TELEPORT: skip full planning, just IK teleport
        if os.environ.get("DEBUG_TELEPORT", "").strip() == "1":
            if log:
                print("[planner] DEBUG_TELEPORT=1, using IK teleport")
            return self.solve_ik(current_qpos, target_pose, num_waypoints=1, log=log)

        if self._bounded_paths_debug_enabled():
            return self._plan_bounded_transit(current_qpos, target_pose, log=log)

        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        result = None
        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                result = self._mplib.plan_pose(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    planning_time=self._timeout or 10,
                    wrt_world=True,
                    verbose=log,
                )
            except Exception as e:
                if log:
                    print(f"[planner] MPLib exception: {e}")
                result = {"status": "Fail"}
            elapsed = time.perf_counter() - t0

            if result and result.get("status") == "Success":
                pos = result["position"]
                if self._bounded_paths_debug_enabled() and not self._debug_path_quality_ok(
                    current_qpos, pos, motion_kind="transit", log=log
                ):
                    # A feasibility planner is allowed to return a needlessly
                    # circuitous RRT/IK branch.  In debug mode, treat that as a
                    # rejected candidate and let the normal retry loop sample
                    # another solution instead of executing it.
                    continue
                if log:
                    print(f"[planner] MPLib Success ({elapsed:.2f}s, attempt {attempt + 1})")
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)
            else:
                if log:
                    status = result.get("status", "Unknown") if result else "None"
                    print(f"[planner] MPLib attempt {attempt + 1} failed: {status} ({elapsed:.2f}s)")

        if log:
            print("[planner] MPLib failed all attempts — skipping (no Genesis IK fallback)")
        return PlanResult(False)

    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK using mplib (Pinocchio) and return interpolated trajectory."""
        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        # mplib.Planner.IK expects the goal in BASE frame (unlike plan_pose,
        # which exposes a wrt_world flag). Transform world→base ourselves
        # so pinocchio's CLIK seeds at the right pose; otherwise IK fails
        # whenever the robot base is offset from the world origin.
        goal = self._mplib.robot.get_base_pose().inv() * goal

        status, q_goals = self._mplib.IK(
            goal_pose=goal,
            start_qpos=qpos,
            n_init_qpos=20,
            threshold=0.001,
            return_closest=True,
        )

        if status != "Success" or q_goals is None:
            if log:
                print(f"[planner] mplib IK failed: {status}")
            return PlanResult(False)

        q_goal = np.asarray(q_goals, dtype=np.float64).ravel()[:self.n_arm]
        q_start = np.asarray(current_qpos[:self.n_arm], dtype=np.float64)

        # Wrap joint angles to shortest path (avoid 360°+ rotations)
        diff = q_goal - q_start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        q_goal = q_start + diff

        # Interpolate (min 2 waypoints to always include goal)
        n_wp = max(2, num_waypoints or self.num_waypoints)
        t = np.linspace(0, 1, n_wp)
        position = (1 - t[:, None]) * q_start + t[:, None] * q_goal
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[planner] IK Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    def plan_screw_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        qpos_step: float = None,
        **kwargs,
    ) -> PlanResult:
        """Plan a straight-line Cartesian path using MPLib's screw planner.

        Explicit ``qpos_step`` controls integration precision. Omitting it
        preserves the normal0.1 step and debug bounded-path0.05 step; an
        explicit valid step is honored by either backend.
        """
        if qpos_step is not None:
            if np.asarray(qpos_step).ndim != 0 or isinstance(qpos_step, (bool, np.bool_)):
                raise ValueError("qpos_step must be a finite positive scalar")
            qpos_step = float(qpos_step)
            if not np.isfinite(qpos_step) or qpos_step <= 0:
                raise ValueError("qpos_step must be a finite positive scalar")
        if self._bounded_paths_debug_enabled():
            return self._plan_bounded_screw(
                current_qpos,
                target_pose,
                time_step=1 / 250,
                qpos_step=0.05 if qpos_step is None else qpos_step,
                # The opt-in flag is a diagnostic mode: always expose why a
                # bounded Cartesian candidate was accepted or rejected, even
                # when the production caller normally suppresses planner logs.
                log=True,
            )

        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                result = self._mplib.plan_screw(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    qpos_step=0.1 if qpos_step is None else qpos_step,
                    wrt_world=True,
                    verbose=log,
                )
            except Exception as e:
                if log:
                    print(f"[planner] plan_screw exception: {e}")
                result = {"status": "Fail"}
            elapsed = time.perf_counter() - t0

            if result and result.get("status") == "Success":
                if log:
                    print(f"[planner] plan_screw Success ({elapsed:.2f}s, attempt {attempt + 1})")
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)
            else:
                if log:
                    status = result.get("status", "Unknown") if result else "None"
                    print(f"[planner] plan_screw attempt {attempt + 1} failed: {status} ({elapsed:.2f}s)")

        if log:
            print("[planner] plan_screw failed all attempts")
        return PlanResult(False)

    # Per-waypoint bounds for solve_ik_cartesian (same contract as the
    # task helpers' seeded MPLib IK, envs/manipulation.py _mplib_seeded_ik).
    CARTESIAN_IK_MAX_JOINT_STEP = 0.30
    CARTESIAN_IK_MAX_POS_ERR = 0.005
    CARTESIAN_IK_MAX_ROT_ERR = 0.02

    def solve_ik_cartesian(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Cartesian-space interpolation: interpolate pos/quat in task space
        (from the MPLib FK of ``current_qpos`` to ``target_pose``, world
        frame) and solve MPLib IK per waypoint, each seeded from the previous
        waypoint (``return_closest``, collision-checked against the planner
        world).  Every waypoint must pass an FK check and move no joint more
        than CARTESIAN_IK_MAX_JOINT_STEP from the previous one; otherwise the
        whole plan fails."""
        backend = self._mplib
        pin = backend.pinocchio_model
        ee_index = backend.link_name_2_idx[backend.move_group]
        base = backend.robot.get_base_pose()
        q_prev = np.asarray(current_qpos, dtype=np.float64).ravel()[:self.n_arm].copy()
        pin.compute_forward_kinematics(self._pad_qpos(q_prev))
        start = base * pin.get_link_pose(ee_index)
        start_pos = np.asarray(start.p, dtype=np.float64)
        q0 = np.asarray(start.q, dtype=np.float64)

        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        end_pos, q1 = target7[:3], target7[3:7]

        n_wp = max(2, num_waypoints or self.num_waypoints)
        position = np.zeros((n_wp, self.n_arm), dtype=np.float64)

        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))

        for i, alpha in enumerate(np.linspace(0, 1, n_wp)):
            pos_i = (1 - alpha) * start_pos + alpha * end_pos
            if abs(dot) > 0.9995:
                quat_i = q0 + alpha * (q1 - q0)
                quat_i /= np.linalg.norm(quat_i)
            else:
                theta = np.arccos(abs(dot))
                quat_i = (np.sin((1 - alpha) * theta) * q0
                          + np.sin(alpha * theta) * q1) / np.sin(theta)
            goal = base.inv() * pymp.Pose(p=pos_i, q=quat_i)
            try:
                status, q = backend.IK(goal, self._pad_qpos(q_prev), n_init_qpos=20,
                                       threshold=1e-3, return_closest=True)
            except Exception as exc:
                status, q = f"exception {exc}", None
            ok = status == "Success" and q is not None
            if ok:
                q_full = np.asarray(q, dtype=np.float64).ravel()
                pin.compute_forward_kinematics(q_full)
                reached = pin.get_link_pose(ee_index)
                pos_err = float(np.linalg.norm(np.asarray(reached.p) - np.asarray(goal.p)))
                rot_err = 2 * float(np.arccos(np.clip(abs(np.dot(reached.q, goal.q)), 0.0, 1.0)))
                q_arm = q_full[:self.n_arm]
                ok = (pos_err <= self.CARTESIAN_IK_MAX_POS_ERR
                      and rot_err <= self.CARTESIAN_IK_MAX_ROT_ERR
                      and float(np.max(np.abs(q_arm - q_prev))) <= self.CARTESIAN_IK_MAX_JOINT_STEP)
            if not ok:
                if log:
                    print(f"[planner] Cartesian MPLib IK failed at waypoint {i}/{n_wp}")
                return PlanResult(False)
            position[i] = q_arm
            q_prev = q_arm.copy()

        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[planner] Cartesian MPLib IK Success, waypoints={n_wp}")
        return PlanResult(True, position, velocity)

    def update_obstacles(self, point_cloud: np.ndarray, resolution: float = 0.02):
        self._mplib.update_point_cloud(point_cloud, resolution=resolution)
