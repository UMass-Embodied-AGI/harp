"""Pre-grasp direction sampling, general grasp selection, gripper helpers.

Split from BaseTask (verbatim member move); composed back into it in
envs/base_task.py.
"""


import numpy as np
import torch

from ..utils import Pose, to_numpy
from ..grasp import load_grasp_poses, tcp_to_link_pose, GraspPose


def mplib_world_link_pose(planner, qpos):
    """Pure planner FK; missing/invalid kinematics never touch simulator state."""
    try:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.ndim != 1 or not qpos.size or not np.all(np.isfinite(qpos)):
            return None
        backend = planner._mplib
        padded = np.asarray(planner._pad_qpos(qpos.copy()), dtype=float)
        if not np.all(np.isfinite(padded)):
            return None
        pin = backend.pinocchio_model
        pin.compute_forward_kinematics(padded)
        local = pin.get_link_pose(backend.link_name_2_idx[backend.move_group])
        world = backend.robot.get_base_pose() * local
        position = np.asarray(world.p, dtype=float)
        quaternion = np.asarray(world.q, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))
                or np.linalg.norm(quaternion) < 1e-8):
            return None
        return Pose(position.copy(), quaternion.copy())
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, IndexError):
        return None


def _validate_candidate_grasps(candidates, object_scale):
    """Validate opt-in task-local annotations before any planning/execution."""
    if not np.isfinite(object_scale) or object_scale <= 0:
        raise ValueError("candidate grasp object_scale must be finite and positive")
    candidates = list(candidates)
    for grasp in candidates:
        if not isinstance(grasp, GraspPose):
            raise TypeError("candidate_grasps entries must be GraspPose instances")
        if (grasp.pose.p.shape != (3,) or grasp.pose.q.shape != (4,)
                or not np.all(np.isfinite(grasp.pose.to_pose7()))
                or np.linalg.norm(grasp.pose.q) < 1e-8
                or not np.isfinite(grasp.pre_distance) or grasp.pre_distance <= 0
                or grasp.scale_frame not in ("runtime", "mesh_unit")):
            raise ValueError("candidate_grasps contains an invalid annotation")
    return candidates


def _validate_screw_step(value):
    if np.asarray(value).ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError("screw_qpos_step must be a finite positive scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("screw_qpos_step must be a finite positive scalar")
    return value



class _GraspSelectionMixin:
    # ------------------------------------------------------------------
    # Pre-grasp direction sampling
    # ------------------------------------------------------------------

    def find_best_pregrasp(
        self,
        grasp_tcp: "Pose",
        arm_tag: str,
        pre_dist: float = 0.12,
        constraints: list = None,
        n_az: int = 12,
        n_el: int = 8,
        bar_axis: np.ndarray = None,
    ) -> "Pose":
        """Sample pre-grasp retreat directions and return the link pose with least IK error.

        Args:
            grasp_tcp: TCP pose at the grasp point (world frame).
            arm_tag: which arm to use.
            pre_dist: how far to back off from grasp point.
            constraints: list of (normal_vec, min_dot) tuples. A candidate direction
                d is valid only if dot(d, normal) > min_dot for ALL constraints.
                Example: [([1,0,0], 0.1)] requires d to have positive X component.
            n_az: number of azimuth samples.
            n_el: number of elevation samples.
            bar_axis: optional axis for TCP +X (e.g. handle bar direction).
                If None, uses [0, 0, 1] as fallback.

        Returns:
            Link-frame Pose for the best pre-grasp (ready for move_and_execute).
        """
        from ..grasp import tcp_to_link_pose as _tcp_to_link

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        current_qpos = arm.get_arm_qpos()
        grasp_pos = grasp_tcp.p

        # Default approach direction from grasp TCP +Z
        R_grasp = grasp_tcp.to_matrix()[:3, :3]
        default_approach = R_grasp[:, 2]

        constraints = constraints or []
        best_err = float("inf")
        best_link = None
        best_dir = None

        for az_deg in np.linspace(-80, 80, n_az):
            for el_deg in np.linspace(-60, 30, n_el):
                az = np.deg2rad(az_deg)
                el = np.deg2rad(el_deg)

                # Direction in local frame where default approach = +X
                d_local = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])

                # Rotate so local +X aligns with default approach
                ax = np.array([1.0, 0.0, 0.0])
                v = np.cross(ax, default_approach)
                c = float(np.dot(ax, default_approach))
                if np.linalg.norm(v) < 1e-8:
                    R_rot = np.eye(3) if c > 0 else np.diag([-1.0, 1.0, -1.0])
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R_rot = np.eye(3) + vx + vx @ vx / (1.0 + c)

                d_world = R_rot @ d_local
                d_world /= np.linalg.norm(d_world) + 1e-12

                # Check constraints
                valid = True
                for normal, min_dot in constraints:
                    if np.dot(d_world, np.asarray(normal, dtype=np.float64)) < min_dot:
                        valid = False
                        break
                if not valid:
                    continue

                # Build pre-grasp TCP pose
                pre_pos = grasp_pos - d_world * pre_dist

                # TCP frame: +Z = d_world (approach), +X = bar_axis projected
                tcp_z = d_world.copy()
                if bar_axis is not None:
                    tcp_x = bar_axis - np.dot(bar_axis, tcp_z) * tcp_z
                    n = np.linalg.norm(tcp_x)
                    tcp_x = tcp_x / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
                else:
                    # Pick axis least aligned with tcp_z
                    up = np.array([0.0, 0.0, 1.0])
                    tcp_x = up - np.dot(up, tcp_z) * tcp_z
                    n = np.linalg.norm(tcp_x)
                    tcp_x = tcp_x / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
                tcp_y = np.cross(tcp_z, tcp_x)
                tcp_y /= np.linalg.norm(tcp_y) + 1e-8
                R_tcp = np.column_stack([tcp_x, tcp_y, tcp_z])
                import transforms3d as t3d
                q = t3d.quaternions.mat2quat(R_tcp)

                pre_tcp = Pose(pre_pos, q)
                pre_link = _tcp_to_link(pre_tcp, tcp_offset)

                # Quick IK check
                result = arm.planner.solve_ik(
                    current_qpos, pre_link.to_pose7(), num_waypoints=1, log=False,
                )
                if not result.success:
                    continue

                # Measure actual EE error
                arm.teleport_arm_joints(result.position[-1])
                self.scene.step()
                ee_pos = np.array(arm.get_ee_pose()[:3])
                err = float(np.linalg.norm(ee_pos - pre_link.p))

                if err < best_err:
                    best_err = err
                    best_link = pre_link
                    best_dir = d_world

        # Restore arm
        arm.teleport_arm_joints(current_qpos)
        self.scene.step()

        if best_link is not None:
            print(f"[pregrasp] best dir={best_dir}, error={best_err:.4f}")
        else:
            print("[pregrasp] no valid direction found, using default approach")
            pre_pos = grasp_pos - default_approach * pre_dist
            pre_tcp = Pose(pre_pos, grasp_tcp.q)
            best_link = _tcp_to_link(pre_tcp, tcp_offset)

        return best_link

    # ------------------------------------------------------------------
    # General grasp selection
    # ------------------------------------------------------------------

    def select_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        model_id: int = None,
        robot_type: str = None,
        max_candidates: int = 10,
        table_z: float = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
    ):
        """Select the best feasible grasp from saved grasp poses.

        Loads grasps from YAML, ranks them by approach-direction heuristic,
        then tries mplib planning on top candidates until one succeeds.

        Args:
            object_name: object directory name (e.g. "001_bottle").
            object_pose: current object pose in world frame.
            arm_tag: which arm to use.
            model_id: optional model variant filter.
            robot_type: optional robot type for robot-specific grasps (e.g. "franka").
            max_candidates: max grasps to try planning for.
            table_z: table surface height; grasps below this are discarded.
                     Defaults to self.TABLE_TOP_Z if available.
            pre_dist: override pre-grasp distance. If None, uses each grasp's own value.
            object_scale: mesh scale used when loading the object in the task.
                Grasp poses are stored at scale=1; this
                scales the local positions to match the loaded mesh size.

        Returns:
            (grasp_link, pre_link, grasp_pose) on success — link poses for IK
                and the GraspPose object for reference.
            None if no feasible grasp found.
        """
        import transforms3d as t3d

        if table_z is None:
            table_z = getattr(self, "TABLE_TOP_Z", 0.74)
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        robot_base = arm.origin_pose.p

        grasps = load_grasp_poses(object_name, model_id=model_id, robot_type=robot_type)
        if not grasps:
            print(f"[select_grasp] No grasp poses found for {object_name}")
            return None

        # --- Score and rank ---
        scored = []
        print(f"[select_grasp] {len(grasps)} grasps loaded, table_z={table_z}, robot_base={robot_base}")
        for g in grasps:
            tcp_world = g.to_world(object_pose, object_scale)
            R_tcp = t3d.quaternions.quat2mat(tcp_world.q)
            tcp_z = R_tcp[:, 2]  # approach direction

            # 1. Pre-grasp → grasp path must not cross through the object.
            #    The pre-grasp must be farther from object center than the grasp,
            #    otherwise the gripper sweeps through the object body.
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_pos = tcp_world.p - d * tcp_z
            dist_grasp = float(np.linalg.norm(tcp_world.p - object_pose.p))
            dist_pre = float(np.linalg.norm(pre_pos - object_pose.p))
            if dist_pre < dist_grasp:
                print(f"[select_grasp]   {g.name}: FILTERED pre-grasp closer to object than grasp "
                      f"(pre={dist_pre:.3f} < grasp={dist_grasp:.3f}), path would cross object")
                continue

            # 2. Approach alignment: tcp_z should point from robot toward object
            #    so pre-grasp (= grasp - pre_dist * tcp_z) is on the robot's side
            to_obj = object_pose.p - robot_base
            to_obj[2] = 0.0  # horizontal only
            norm = np.linalg.norm(to_obj)
            if norm > 1e-6:
                to_obj /= norm
            approach_score = float(np.dot(tcp_z[:2], to_obj[:2]))

            # 3. Height check: TCP must be above table with margin for fingers
            finger_margin = 0.01
            if tcp_world.p[2] < table_z + finger_margin:
                print(f"[select_grasp]   {g.name}: FILTERED height {tcp_world.p[2]:.3f} < {table_z + finger_margin:.3f}")
                continue

            # 4. Pre-grasp must also be above table (pre_pos computed in filter 1)
            if pre_pos[2] < table_z + finger_margin:
                print(f"[select_grasp]   {g.name}: FILTERED pre-grasp height {pre_pos[2]:.3f} < {table_z + finger_margin:.3f}")
                continue

            scored.append((approach_score, g))

        print(f"[select_grasp] {len(scored)}/{len(grasps)} passed filters")
        if not scored:
            print(f"[select_grasp] All grasps filtered out for {object_name}")
            return None

        # Manual (hand-verified) grasps are preferred over auto grasps;
        # within each source group, rank by approach-alignment score.
        scored.sort(key=lambda x: (0 if x[1].source == "manual" else 1, -x[0]))

        # --- Try planning for top candidates ---
        n_try = min(max_candidates, len(scored))
        current_qpos = arm.get_arm_qpos()

        for i in range(n_try):
            score, g = scored[i]
            tcp_world = g.to_world(object_pose, object_scale)
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_tcp = g.pre_grasp_world(object_pose, object_scale) if pre_dist is None else \
                Pose(tcp_world.p - d * t3d.quaternions.quat2mat(tcp_world.q)[:, 2], tcp_world.q)

            grasp_link = tcp_to_link_pose(tcp_world, tcp_offset)
            pre_link = tcp_to_link_pose(pre_tcp, tcp_offset)

            # Try planning: current → pre-grasp → grasp
            result_pre = arm.planner.plan_path(current_qpos, pre_link.to_pose7())
            if not result_pre.success:
                print(f"[select_grasp] {g.name} (score={score:.2f}): pre-grasp plan failed")
                continue

            # Plan grasp from end of pre-grasp trajectory
            end_qpos = result_pre.position[-1] if result_pre.position.size > 0 else current_qpos
            result_grasp = arm.planner.plan_path(end_qpos, grasp_link.to_pose7())
            if not result_grasp.success:
                print(f"[select_grasp] {g.name} (score={score:.2f}): grasp plan failed")
                continue

            # FK quality check: set qpos and read link pos without stepping physics
            grasp_final_qpos = result_grasp.position[-1] if result_grasp.position.size > 0 else end_qpos
            saved_qpos_raw = arm.entity.get_qpos()
            full_qpos = arm._build_qpos_with_arm(grasp_final_qpos)
            arm.entity.set_qpos(torch.tensor(full_qpos, dtype=torch.float32) if not hasattr(full_qpos, 'cpu') else full_qpos)
            arm.entity.get_links_pos()  # trigger FK update without physics step
            ee_link_pos = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
            fk_err = float(np.linalg.norm(ee_link_pos - grasp_link.p))
            arm.entity.set_qpos(saved_qpos_raw)  # restore without stepping
            arm.entity.get_links_pos()

            if fk_err > 0.03:
                print(f"[select_grasp] {g.name} (score={score:.2f}): FK error {fk_err:.4f}m > 0.03m, skipping")
                continue

            print(f"[select_grasp] {g.name} (score={score:.2f}): SUCCESS (FK err={fk_err:.4f}m, tried {i+1}/{n_try})")
            return grasp_link, pre_link, g

        print(f"[select_grasp] No feasible grasp found after {n_try} candidates")
        return None

    def _plan_one_grasp(
        self,
        grasp: GraspPose,
        object_pose: Pose,
        arm_tag: str,
        pre_dist: float = None,
        object_scale: float = 1.0,
        log_prefix: str = "",
        screw_qpos_step=None,
    ):
        """Plan curr→pre via RRT and pre→grasp via plan_screw for ONE grasp.
        FK-check the planned grasp endpoint.  Returns
        ``(result_pre, result_grasp, grasp_link, pre_link)`` on success or
        ``None`` if any step fails.  Does NOT execute — caller drives that.
        """
        import transforms3d as t3d

        screw_kwargs = ({} if screw_qpos_step is None else
                        {"qpos_step": _validate_screw_step(screw_qpos_step)})
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        if not np.all(np.isfinite(object_pose.to_pose7())):
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: nonfinite object pose")
            return None
        tcp_world = grasp.to_world(object_pose, object_scale)
        d = pre_dist if pre_dist is not None else grasp.pre_distance
        if pre_dist is None:
            pre_tcp = grasp.pre_grasp_world(object_pose, object_scale)
        else:
            R = t3d.quaternions.quat2mat(tcp_world.q)
            pre_tcp = Pose(tcp_world.p - d * R[:, 2], tcp_world.q)
        grasp_link = tcp_to_link_pose(tcp_world, tcp_offset)
        pre_link = tcp_to_link_pose(pre_tcp, tcp_offset)

        current_qpos = arm.get_arm_qpos()

        result_pre = arm.planner.plan_path(
            current_qpos, pre_link.to_pose7(),
        )
        if not result_pre.success:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: pre RRT failed")
            return None
        end_pre_qpos = (
            result_pre.position[-1]
            if result_pre.position.size > 0 else current_qpos
        )

        plan_screw = getattr(arm.planner, "plan_screw_path", None)
        if plan_screw is None:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: planner has no plan_screw_path")
            return None
        result_grasp = plan_screw(end_pre_qpos, grasp_link.to_pose7(), **screw_kwargs)
        if not result_grasp.success:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: screw failed "
                  f"— skipping (no RRT fallback by design)")
            return None

        grasp_final_qpos = (
            result_grasp.position[-1]
            if result_grasp.position.size > 0 else end_pre_qpos
        )
        actual_link = mplib_world_link_pose(arm.planner, grasp_final_qpos)
        if actual_link is None:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: invalid MPLib FK")
            return None
        fk_err = float(np.linalg.norm(actual_link.p - grasp_link.p))

        if not np.isfinite(fk_err) or fk_err > 0.03:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: "
                  f"FK err {fk_err:.4f}m > 3cm")
            return None

        print(f"[grasp_pick]   {log_prefix}{grasp.name}: planned OK "
              f"(FK err={fk_err:.4f}m)")
        return result_pre, result_grasp, grasp_link, pre_link

    def select_and_execute_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        model_id: int = None,
        robot_type: str = None,
        max_candidates: int = 10,
        table_z: float = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
        categories: list = None,
        plan_attempts: int = 1,
        post_plan_check=None,
        grasp_names=None,
        candidate_grasps=None,
        screw_qpos_step=None,
    ):
        """Pre-flight + execute pattern.  For each ranked grasp candidate:
        plan curr→pre via RRT, plan pre→grasp via plan_screw (straight line
        along the grasp's approach axis, NO RRT fallback).  If both legs
        plan and the FK check passes, execute the cached trajectories;
        otherwise advance to the next candidate.  This guarantees that the
        final approach segment moves exactly along the grasp z-axis (no
        wrist swing through the object) and never silently RRTs around.

        Returns ``(grasp_link, pre_link, grasp_pose)`` on success — same
        triple as :meth:`select_grasp` so callers can chain close + lift
        unchanged.  Returns ``None`` if every candidate is rejected; the
        caller should iterate to the next object or try a different grasp
        pool.

        ``grasp_names`` optionally limits the YAML pool before the normal
        geometry filters and ranking. ``None`` keeps the full existing pool.
        ``post_plan_check`` can reject a planned continuation before either
        approach leg executes; its default ``None`` preserves normal behavior.
        ``candidate_grasps`` supplies task-local, object-frame GraspPose
        annotations instead of loading YAML. The same filters, ranking and
        straight-descent checks apply; ``None`` retains the existing loader.
        ``screw_qpos_step`` optionally requests finer MPLib screw integration;
        ``None`` passes no new planner argument and preserves its default.
        """
        import transforms3d as t3d

        precision_kwargs = ({} if screw_qpos_step is None else
                            {"screw_qpos_step": _validate_screw_step(screw_qpos_step)})
        if table_z is None:
            table_z = getattr(self, "TABLE_TOP_Z", 0.74)
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        arm = self.robot.get_arm(arm_tag)
        robot_base = arm.origin_pose.p

        grasps = (
            load_grasp_poses(object_name, model_id=model_id, robot_type=robot_type)
            if candidate_grasps is None
            else _validate_candidate_grasps(candidate_grasps, object_scale)
        )
        if not grasps:
            print(f"[grasp_pick] {object_name}: no grasps in YAML")
            return None

        scored = []
        for g in grasps:
            if grasp_names is not None and g.name not in grasp_names:
                continue
            if categories is not None and g.category not in categories:
                continue
            tcp_world = g.to_world(object_pose, object_scale)
            R_tcp = t3d.quaternions.quat2mat(tcp_world.q)
            tcp_z = R_tcp[:, 2]
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_pos = tcp_world.p - d * tcp_z
            dist_grasp = float(np.linalg.norm(tcp_world.p - object_pose.p))
            dist_pre = float(np.linalg.norm(pre_pos - object_pose.p))
            if dist_pre < dist_grasp:
                continue
            to_obj = object_pose.p - robot_base
            to_obj[2] = 0.0
            n = np.linalg.norm(to_obj)
            if n > 1e-6:
                to_obj /= n
            approach_score = float(np.dot(tcp_z[:2], to_obj[:2]))
            finger_margin = 0.01
            if tcp_world.p[2] < table_z + finger_margin:
                continue
            if pre_pos[2] < table_z + finger_margin:
                continue
            scored.append((approach_score, g))

        if not scored:
            print(f"[grasp_pick] {object_name}: all {len(grasps)} grasps filtered out")
            return None

        scored.sort(key=lambda x: (0 if x[1].source == "manual" else 1, -x[0]))
        n_try = min(max_candidates, len(scored))
        print(f"[grasp_pick] {object_name}: {len(scored)} feasible candidates, "
              f"trying top {n_try}")

        for i in range(n_try):
            score, g = scored[i]
            # ``plan_attempts`` > 1 (opt-in, default 1 = unchanged): retry
            # the pre-RRT + screw PLANNING for the same candidate.  Nothing
            # has executed yet, so this is invisible — each pre-pose RRT
            # lands a different IK branch, and the screw leg's feasibility
            # depends on that arrival branch (observed on
            # oil_bottle_recovery: identical candidate, screw fails from
            # one RRT arrival and succeeds from the next).
            plans = None
            for attempt in range(max(1, int(plan_attempts))):
                if attempt > 0:
                    print(f"[grasp_pick]   {g.name}: planning retry "
                          f"{attempt + 1}/{plan_attempts} (nothing executed)")
                plans = self._plan_one_grasp(
                    g, object_pose, arm_tag,
                    pre_dist=pre_dist, object_scale=object_scale,
                    log_prefix=f"(score={score:+.2f}) ",
                    **precision_kwargs,
                )
                # ``post_plan_check`` (opt-in): caller-supplied feasibility
                # gate on the PLANNED (not yet executed) grasp — e.g. "can
                # the arrival branch also lift?".  A rejection counts as a
                # planning failure so the retry re-rolls the pre-RRT branch.
                if (plans is not None and post_plan_check is not None
                        and not post_plan_check(plans)):
                    print(f"[grasp_pick]   {g.name}: post-plan check "
                          "rejected the arrival branch (nothing executed)")
                    plans = None
                if plans is not None:
                    break
            if plans is None:
                continue
            result_pre, result_grasp, grasp_link, pre_link = plans
            print(f"[grasp_pick]   {g.name}: SUCCESS — executing "
                  f"(cand {i+1}/{n_try})")
            self.execute_plan(result_pre, arm_tag)
            self.execute_plan(result_grasp, arm_tag)
            return grasp_link, pre_link, g

        print(f"[grasp_pick] {object_name}: no candidate planned successfully "
              f"after {n_try} tries")
        return None

    def try_grasp_by_name(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        grasp_name: str,
        model_id: int = None,
        robot_type: str = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
        execute: bool = True,
    ):
        """Force a specific named grasp.  Skips the ranker; runs the same
        plan + FK check as :meth:`select_and_execute_grasp`.  When
        ``execute=True`` (default) runs ``execute_plan`` for both legs and
        returns ``(grasp_link, pre_link, grasp_pose)``.  When
        ``execute=False`` returns ``(result_pre, result_grasp, grasp_link,
        pre_link, grasp_pose)`` so the caller can interleave snapshots
        between the legs.  Returns ``None`` if the grasp isn't in the YAML
        or any planning step fails.
        """
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        grasps = load_grasp_poses(
            object_name, model_id=model_id, robot_type=robot_type,
        )
        match = next((g for g in grasps if g.name == grasp_name), None)
        if match is None:
            print(f"[grasp_pick] {object_name}: grasp_name={grasp_name!r} not in YAML "
                  f"({len(grasps)} grasps available)")
            return None

        plans = self._plan_one_grasp(
            match, object_pose, arm_tag,
            pre_dist=pre_dist, object_scale=object_scale,
            log_prefix="[forced] ",
        )
        if plans is None:
            return None
        result_pre, result_grasp, grasp_link, pre_link = plans
        if execute:
            self.execute_plan(result_pre, arm_tag)
            self.execute_plan(result_grasp, arm_tag)
            return grasp_link, pre_link, match
        return result_pre, result_grasp, grasp_link, pre_link, match

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def set_gripper(self, val: float, arm_tag: str, num_steps: int = 200):
        """Gradually set gripper value."""
        arm = self.robot.get_arm(arm_tag)
        current = arm.gripper_val
        trajectory = np.linspace(current, val, num_steps)
        for v in trajectory:
            self.robot.set_gripper(float(v), arm_tag)
            self.step_sim()

    def open_gripper(self, arm_tag: str, num_steps: int = 200):
        self.set_gripper(1.0, arm_tag, num_steps)

    def close_gripper(self, arm_tag: str, num_steps: int = 200):
        self.set_gripper(0.0, arm_tag, num_steps)
