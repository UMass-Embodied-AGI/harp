"""Human-aware full-state neutral expert: wait before a segment that would
meet the human (Co-presence).

The neutral human never adapts to the robot.  The full-state expert knows the
human's scripted clip, so before executing each planned motion segment it
predicts where the human's body capsules will be at every waypoint of that
segment (one waypoint per sim step, one clip frame per sim step) and checks
them against the robot's moving links along the plan.  If they would come
within ``neutral_aware_clearance_m`` (default 0.05 m, plus a per-link body
pad), the robot holds still and re-checks every policy tick (``action_substeps``
sim steps).  When it resumes it re-plans (MPLib ``plan_qpos`` to the same
goal joints, obstacle cloud refreshed to the human's current pose) and uses
the re-plan if it clears the predicted human; otherwise the original plan,
which was just re-validated (the arm did not move while holding).  The wait
is bounded per segment and per episode.  While holding, if the human would
reach the robot where it stands within the next policy tick, the robot
yields: it lifts its end effector 8 cm straight up (MPLib plan_screw, same
orientation; at most twice per segment).  When the budget runs out and no
plan clears the predicted human, the expert aborts (plan_success false)
instead of driving into the human.  No Genesis IK, no teleport, no
re-targeting of the human.  The blind expert never enables this.

Enabled with ``config["neutral_human_aware_expert"] = True`` (set by
``scripts/eval_expert_set.py`` for the neutral full-state expert only).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_CLEARANCE_M = 0.05
# Body pad added to the point-to-capsule distance: robot link origins sit
# inside the link geometry; the end-effector origin carries the hand and
# fingers.
DEFAULT_LINK_PAD_M = 0.07
DEFAULT_EE_PAD_M = 0.10
DEFAULT_CHECK_STRIDE = 10          # sim steps between checked waypoints
DEFAULT_MAX_WAIT_SEGMENT = 3000    # sim steps (6 s at dt 0.002)
DEFAULT_MAX_WAIT_EPISODE = 12000   # sim steps (24 s)
MOVING_LINK_MIN_TRAVEL_M = 0.02
# Detour re-plan: plan around the human with a cloud inflated to the
# gate's own margin, over the human's predicted sweep, before waiting.
# (swept over the segment?, extra capsule radius m) tried in order.
DETOUR_VARIANTS = ((True, 0.06), (False, 0.06), (True, 0.03))
DETOUR_SWEEP_STRIDE = 50           # sim steps between swept human samples
DETOUR_POINT_SPACING_M = 0.03
DETOUR_CLOUD_RES_M = 0.03


def dense_capsule_points(capsules, margin: float, spacing: float = DETOUR_POINT_SPACING_M):
    """Surface points of capsules grown by ``margin`` (shaft rings + end caps),
    spaced at most ``spacing`` so no robot link fits between them."""
    pts = []
    for _name, a, b, r in capsules:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        R = float(r) + float(margin)
        seg = b - a
        length = float(np.linalg.norm(seg))
        axis = seg / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
        tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, tmp)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(axis, u)
        n_th = max(8, int(np.ceil(2 * np.pi * R / spacing)))
        th = np.linspace(0.0, 2 * np.pi, n_th, endpoint=False)
        ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
        for t in np.linspace(0.0, 1.0, max(2, int(np.ceil(length / spacing)) + 1)):
            pts.append(a + t * seg + R * ring)
        # hemispherical caps
        for end, sgn in ((a, -1.0), (b, 1.0)):
            n_el = max(2, int(np.ceil(0.5 * np.pi * R / spacing)))
            for el in np.linspace(0.0, 0.5 * np.pi, n_el + 1)[1:]:
                n_ring = max(1, int(np.ceil(2 * np.pi * R * np.cos(el) / spacing)))
                th2 = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
                ring2 = np.cos(th2)[:, None] * u + np.sin(th2)[:, None] * v
                pts.append(end + R * (np.cos(el) * ring2 + sgn * np.sin(el) * axis))
    return np.concatenate(pts, axis=0) if pts else None


def point_capsule_distance(p, a, b, r) -> float:
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    d2 = float(ab @ ab)
    t = 0.0 if d2 < 1e-12 else float(np.clip((p - a) @ ab / d2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)) - float(r))


def path_human_clearance(link_paths: Sequence[tuple[np.ndarray, float]],
                         human_at: Sequence[Sequence[tuple]]) -> float:
    """Minimum synchronous clearance between robot link points and the human.

    ``link_paths``: (positions (T, 3), pad) per robot link, sampled at the
    same times as ``human_at`` (length T lists of capsules
    ``(name, a, b, r)``).  Clearance = capsule surface distance - pad.
    """
    best = float("inf")
    for pts, pad in link_paths:
        for t, caps in enumerate(human_at):
            if t >= len(pts):
                break
            for _name, a, b, r in caps:
                best = min(best, point_capsule_distance(pts[t], a, b, r) - float(pad))
    return best


# ----------------------------------------------------------------------
# Human prediction (privileged: the scripted clip frames)
# ----------------------------------------------------------------------

def _active_motion(task):
    avatar = getattr(task, "avatar", None)
    if avatar is None:
        return None
    state = getattr(avatar.robot, "action_state", None)
    mods = getattr(avatar, "motion_modules", {}) or {}
    motion = mods.get(state) if isinstance(state, str) else None
    if motion is None or not getattr(motion, "data", None) or not hasattr(motion, "node_data"):
        return None
    return motion


def predict_human_capsules(task, offsets: Sequence[int]) -> list[list[tuple]]:
    """Avatar capsules ``offsets`` sim steps ahead (one clip frame per sim step).

    Computed from the clip's frame data through the skin's joint transforms
    without touching the scene (the skin root node the transform helper
    writes is restored).  Before the human starts, or between clips, the
    current pose is returned for every offset.
    """
    # The collision checker keeps the real collider (the blind expert's
    # planning-blind wrapper only replaces task.avatar_collider).
    checker = getattr(task, "avatar_collision_checker", None)
    collider = getattr(checker, "collider", None) or getattr(task, "avatar_collider", None)
    collider = getattr(collider, "_collider", collider)
    if collider is None:
        return [[] for _ in offsets]
    current = [(str(n), np.asarray(a, float), np.asarray(b, float), float(r))
               for n, a, b, r in collider.current_capsules()]
    motion = _active_motion(task)
    robot = getattr(getattr(task, "avatar", None), "robot", None)
    skin = getattr(robot, "skin", None)
    if motion is None or skin is None:
        return [current for _ in offsets]

    from scipy.spatial.transform import Rotation as R
    import genesis.utils.geom as geom_utils

    n = len(motion.data)
    loop = bool((getattr(task, "_neutral_avatar_job", None) or {}).get("loop"))
    at = int(getattr(motion, "at_frame", 0))
    root_id = skin._skin_joints[0]
    root = skin._nodes[1][root_id]
    saved = (np.array(root.translation, copy=True) if root.translation is not None else None,
             None if root.rotation is None else np.array(root.rotation, copy=True))
    global_mat = getattr(motion, "global_mat", robot.global_mat)
    global_mat_inv = getattr(motion, "global_mat_inv", robot.global_mat_inv)
    euler = R.from_matrix(robot.global_rot).as_euler("xyz", degrees=True)[[0, 2, 1]]
    base_R = geom_utils.euler_to_R(euler)
    morph_z = float(np.asarray(getattr(skin._morph, "pos", (0, 0, 0)), float)[2])
    out = []
    cache: dict[int, list] = {}
    try:
        for off in offsets:
            k = at + int(off)
            k = k % n if loop else min(max(k, 0), n - 1)
            if k in cache:
                out.append(cache[k])
                continue
            pose = np.asarray(motion.data[k], dtype=np.float64)
            node_trans = motion.node_data[k]
            motion_rot = geom_utils.quat_to_R(pose[3:7])
            total = robot.global_rot @ robot.base_rot @ pose[:3] + robot.global_trans
            skin_base_rot = geom_utils.R_to_quat(base_R @ motion_rot)
            T = skin.calculate_joint_T(total[[1, 2, 0]], global_mat, global_mat_inv,
                                       skin_base_rot, node_trans)

            def bone(name):
                m = T[skin.node_findup[name]]
                p = np.array(m[-1, :3], dtype=np.float64, copy=True)
                p[:2] = p[:2][::-1]
                p[0] *= -1
                p[2] += morph_z
                return p
            caps = []
            for name, bone_a, bone_b, r, _L, _e in collider.entities:
                try:
                    caps.append((str(name), bone(bone_a), bone(bone_b), float(r)))
                except Exception:
                    continue
            cache[k] = caps
            out.append(caps)
    finally:
        if saved[0] is not None:
            root.translation = saved[0]
        if saved[1] is not None:
            root.rotation = saved[1]
    return out


# ----------------------------------------------------------------------
# Robot path along a plan (MPLib kinematics)
# ----------------------------------------------------------------------

def plan_link_paths(task, positions: np.ndarray, arm_tag: str, stride: int,
                    link_pad: float, ee_pad: float) -> tuple[list, list[int]]:
    """World positions of the moving robot links at every ``stride``-th
    waypoint (and the last), via the arm's MPLib pinocchio model."""
    arm = task.robot.get_arm(arm_tag)
    planner = arm.planner
    backend = planner._mplib
    pin = backend.pinocchio_model
    base = np.asarray(planner._base_pose, dtype=np.float64)
    import genesis.utils.geom as geom_utils
    Rb = geom_utils.quat_to_R(base[3:7])
    names = list(backend.link_name_2_idx.keys())
    ee_name = backend.move_group
    idx = list(range(0, len(positions), max(1, int(stride))))
    if idx[-1] != len(positions) - 1:
        idx.append(len(positions) - 1)
    pts = {name: [] for name in names}
    for i in idx:
        q_full = backend.pad_move_group_qpos(np.asarray(positions[i], dtype=np.float64))
        pin.compute_forward_kinematics(q_full)
        for name in names:
            pose = pin.get_link_pose(backend.link_name_2_idx[name])
            p = np.asarray(pose.p if hasattr(pose, "p") else pose.get_p(), dtype=np.float64)
            pts[name].append(Rb @ p + base[:3])
    out = []
    for name in names:
        arr = np.asarray(pts[name])
        if name != ee_name and float(np.max(np.linalg.norm(arr - arr[0], axis=1))) < MOVING_LINK_MIN_TRAVEL_M:
            continue
        out.append((arr, ee_pad if name == ee_name else link_pad))
    return out, idx


# ----------------------------------------------------------------------
# Mixin
# ----------------------------------------------------------------------

class _NeutralHumanAwareMixin:
    """Predictive wait before each planned segment (full-state expert only)."""

    def _neutral_aware_enabled(self) -> bool:
        cfg = self.config or {}
        return bool(cfg.get("neutral_human_aware_expert", False)) and getattr(self, "avatar", None) is not None

    def _neutral_aware_params(self) -> dict[str, Any]:
        cfg = self.config or {}
        return {
            "clearance": float(cfg.get("neutral_aware_clearance_m", DEFAULT_CLEARANCE_M)),
            "link_pad": float(cfg.get("neutral_aware_link_pad_m", DEFAULT_LINK_PAD_M)),
            "ee_pad": float(cfg.get("neutral_aware_ee_pad_m", DEFAULT_EE_PAD_M)),
            "stride": int(cfg.get("neutral_aware_check_stride", DEFAULT_CHECK_STRIDE)),
            "tick": max(1, int(cfg.get("action_substeps", 25))),
            "max_seg": int(cfg.get("neutral_aware_max_wait_segment", DEFAULT_MAX_WAIT_SEGMENT)),
            "max_ep": int(cfg.get("neutral_aware_max_wait_episode", DEFAULT_MAX_WAIT_EPISODE)),
        }

    def _neutral_aware_stats(self) -> dict:
        st = getattr(self, "_neutral_aware_state", None)
        if st is None or st.get("seed") != getattr(self, "_neutral_avatar_reset_seed", None):
            st = {"seed": getattr(self, "_neutral_avatar_reset_seed", None), "segments": 0,
                  "checked": 0, "waits": 0, "wait_sim_steps": 0, "budget_exhausted": 0,
                  "min_clearance_m": None, "pred_now_err_max_m": 0.0, "errors": 0,
                  "replans_used": 0, "aborted": 0, "yields": 0,
                  "detour_tries": 0, "detours_used": 0}
            self._neutral_aware_state = st
        return st

    def _neutral_aware_clearance(self, positions, arm_tag, prm) -> float:
        # A caller may announce that the robot will stay at the segment's end
        # (grasp settle/close/lift): check that dwell against the human too.
        dwell = int(getattr(self, "_neutral_aware_dwell_steps", 0) or 0)
        positions = np.asarray(positions)
        if dwell > 0 and len(positions):
            positions = np.concatenate([positions, np.repeat(positions[-1:], dwell, axis=0)], axis=0)
        paths, idx = plan_link_paths(self, positions, arm_tag, prm["stride"],
                                     prm["link_pad"], prm["ee_pad"])
        human = predict_human_capsules(self, [0] + idx)
        st = self._neutral_aware_stats()
        # self-check: offset 0 must reproduce the live capsules
        try:
            checker = getattr(self, "avatar_collision_checker", None)
            live = checker.collider.current_capsules() if checker is not None else []
            err = max((float(np.linalg.norm(np.asarray(a) - c[1])) for (_n, a, _b, _r), c
                       in zip(live, human[0])), default=0.0)
            st["pred_now_err_max_m"] = max(st["pred_now_err_max_m"], err)
        except Exception:
            pass
        return path_human_clearance(paths, human[1:])

    def _neutral_aware_robot_threat(self, arm_tag: str, prm) -> float:
        """Clearance between the robot at its CURRENT joints and the human
        over the next policy tick (the human never adapts, so a holding robot
        can still be walked into)."""
        arm = self.robot.get_arm(arm_tag)
        q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)[None, :]
        paths, _idx = plan_link_paths(self, q, arm_tag, 1, prm["link_pad"], prm["ee_pad"])
        # Look ahead long enough to finish a yield before contact.
        horizon = max(prm["tick"], int((self.config or {}).get("neutral_aware_threat_horizon", 100)))
        offs = list(range(0, horizon + 1, max(1, prm["tick"] // 5)))
        human = predict_human_capsules(self, offs)
        tiled = [(np.repeat(pts[:1], len(human), axis=0), pad) for pts, pad in paths]
        return path_human_clearance(tiled, human)

    def _neutral_aware_yield(self, arm_tag: str, prm) -> bool:
        """Yield: lift the end effector straight up (MPLib plan_screw, same
        orientation) by ``neutral_aware_yield_lift_m`` (0.08 m)."""
        import mplib.pymp as pymp
        import genesis.utils.geom as geom_utils
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        backend = planner._mplib
        q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        q_full = backend.pad_move_group_qpos(q)
        backend.pinocchio_model.compute_forward_kinematics(q_full)
        pose = backend.pinocchio_model.get_link_pose(backend.link_name_2_idx[backend.move_group])
        base = np.asarray(planner._base_pose, dtype=np.float64)
        Rb = geom_utils.quat_to_R(base[3:7])
        p_world = Rb @ np.asarray(pose.p, dtype=np.float64) + base[:3]
        q_world = geom_utils.R_to_quat(Rb @ geom_utils.quat_to_R(np.asarray(pose.q, dtype=np.float64)))
        lift = float((self.config or {}).get("neutral_aware_yield_lift_m", 0.08))
        goal = pymp.Pose(p=p_world + np.array([0.0, 0.0, lift]), q=q_world)
        for step in (0.05, 0.02):
            try:
                out = backend.plan_screw(goal_pose=goal, current_qpos=planner._pad_qpos(q),
                                         time_step=1 / 250, qpos_step=step, wrt_world=True,
                                         verbose=False)
            except Exception:
                continue
            if out and out.get("status") == "Success":
                pos = out["position"]
                self._neutral_aware_bypass = True
                try:
                    super().execute_plan(PlanResult(True, pos, out.get("velocity", np.zeros_like(pos))), arm_tag)
                finally:
                    self._neutral_aware_bypass = False
                return True
        return False

    def _neutral_aware_retreat_home(self, arm_tag: str, prm) -> bool:
        """The human keeps closing in on the waiting arm after its
        lifts: retreat to the arm's home joints (the pose every neutral
        layout is gated against at reset) along an MPLib plan that the
        predicted human does not meet.  Bypasses the gate (checked here)."""
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        home = np.asarray(list(getattr(arm, "homestate", None) or ()), dtype=np.float64).ravel()
        cur = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        if home.shape != cur.shape or float(np.max(np.abs(home - cur))) < 0.05:
            return False
        planner = arm.planner
        refresh = getattr(self, "_refresh_avatar_planner_obstacles", None)
        if refresh is not None:
            refresh(arm_tag)
        try:
            out = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(home.copy())], current_qpos=planner._pad_qpos(cur.copy()),
                time_step=1 / 250, planning_time=getattr(planner, "_timeout", None) or 10,
                rrt_range=0.3, verbose=False)
        except Exception:
            return False
        if not out or out.get("status") != "Success":
            return False
        pos = np.asarray(out["position"], dtype=np.float64)
        if pos.ndim != 2 or not len(pos) or np.max(np.abs(pos[0, :len(cur)] - cur)) > 1e-3:
            return False
        saved = getattr(self, "_neutral_aware_dwell_steps", 0)
        self._neutral_aware_dwell_steps = 0
        try:
            c = self._neutral_aware_clearance(pos, arm_tag, prm)
        finally:
            self._neutral_aware_dwell_steps = saved
        if c < 0.0:
            return False
        print(f"[neutral-aware] retreat home rows={len(pos)} clearance={c:.3f}", flush=True)
        self._neutral_aware_bypass = True
        try:
            super().execute_plan(PlanResult(True, pos, out.get("velocity", np.zeros_like(pos))), arm_tag)
        finally:
            self._neutral_aware_bypass = False
        return True

    def _neutral_aware_park(self, arm_tag: str) -> None:
        """Full-state expert: end the rollout at the home joints (through the
        gate) instead of idling beside the human's work."""
        if not self._neutral_aware_enabled() or not getattr(self, "plan_success", True):
            return
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        home = np.asarray(list(getattr(arm, "homestate", None) or ()), dtype=np.float64).ravel()
        cur = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        if home.shape != cur.shape or float(np.max(np.abs(home - cur))) < 0.05:
            return
        planner = arm.planner
        refresh = getattr(self, "_refresh_avatar_planner_obstacles", None)
        if refresh is not None:
            refresh(arm_tag)
        try:
            out = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(home.copy())], current_qpos=planner._pad_qpos(cur.copy()),
                time_step=1 / 250, planning_time=getattr(planner, "_timeout", None) or 10,
                rrt_range=0.3, verbose=False)
        except Exception:
            return
        if not out or out.get("status") != "Success":
            return
        pos = np.asarray(out["position"], dtype=np.float64)
        if pos.ndim != 2 or not len(pos) or np.max(np.abs(pos[0, :len(cur)] - cur)) > 1e-3:
            return
        # Optional: a park the gate cannot clear is skipped, not a failure.
        self._neutral_aware_optional = True
        try:
            self.execute_plan(PlanResult(True, pos, out.get("velocity", np.zeros_like(pos))), arm_tag)
        finally:
            self._neutral_aware_optional = False

    def _neutral_aware_replan(self, result, arm_tag: str):
        """MPLib plan_qpos to the segment's goal from the current joints, with
        the planner's obstacle cloud refreshed to the human's current pose."""
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        refresh = getattr(self, "_refresh_avatar_planner_obstacles", None)
        if refresh is not None:
            refresh(arm_tag)
        goal = planner._pad_qpos(np.asarray(result.position[-1], dtype=np.float64))
        cur = planner._pad_qpos(np.asarray(arm.get_arm_qpos(), dtype=np.float64))
        out = planner._mplib.plan_qpos(
            goal_qposes=[goal], current_qpos=cur, time_step=1 / 250,
            planning_time=getattr(planner, "_timeout", None) or 10,
            rrt_range=0.3, verbose=False,
        )
        if out and out.get("status") == "Success":
            pos = out["position"]
            return PlanResult(True, pos, out.get("velocity", np.zeros_like(pos)))
        return None

    def _neutral_aware_detour(self, result, arm_tag: str, prm):
        """Re-plan the segment around the human (MPLib plan_qpos to the same
        goal joints) with the human's capsules grown by the gate's margin,
        over its predicted sweep for the segment; returns a plan that passes
        the predictive clearance check, else None.  The planner cloud is
        restored afterwards.  Planning only: nothing moves here."""
        from ..planning.base import PlanResult
        if not bool((self.config or {}).get("neutral_aware_detour", True)):
            return None
        st = self._neutral_aware_stats()
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        positions = np.asarray(result.position, dtype=np.float64)
        cur_arm = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        goal = planner._pad_qpos(positions[-1].copy())
        cur = planner._pad_qpos(cur_arm.copy())
        objects = None
        obj_fn = getattr(self, "_neutral_object_obstacle_points", None)
        enabled = getattr(self, "_neutral_object_obstacles_enabled", None)
        if obj_fn is not None and (enabled is None or enabled()):
            try:
                objects = obj_fn()
            except Exception:
                objects = None
        found = None
        try:
            for swept, margin in DETOUR_VARIANTS:
                offs = (list(range(0, len(positions) + 1, DETOUR_SWEEP_STRIDE)) if swept else [0])
                caps = [c for frame in predict_human_capsules(self, offs) for c in frame]
                pts = dense_capsule_points(caps, margin)
                parts = [x for x in (pts, objects) if x is not None and np.size(x)]
                if not parts:
                    return None
                planner.update_obstacles(np.concatenate(parts, axis=0), resolution=DETOUR_CLOUD_RES_M)
                st["detour_tries"] = st.get("detour_tries", 0) + 1
                try:
                    out = planner._mplib.plan_qpos(
                        goal_qposes=[goal], current_qpos=cur, time_step=1 / 250,
                        planning_time=getattr(planner, "_timeout", None) or 10,
                        rrt_range=0.3, verbose=False,
                    )
                except Exception:
                    st["errors"] += 1
                    continue
                if not out or out.get("status") != "Success":
                    continue
                pos = np.asarray(out["position"], dtype=np.float64)
                # A perturbed (invalid) start state would make the plan jump.
                if pos.ndim != 2 or not len(pos) or np.max(np.abs(pos[0, :len(cur_arm)] - cur_arm)) > 1e-3:
                    continue
                c = self._neutral_aware_clearance(pos, arm_tag, prm)
                if c >= prm["clearance"]:
                    found = PlanResult(True, pos, out.get("velocity", np.zeros_like(pos)))
                    print(f"[neutral-aware] detour swept={swept} margin={margin} rows={len(pos)} "
                          f"clearance={c:.3f}", flush=True)
                    break
        finally:
            refresh = getattr(self, "_refresh_avatar_planner_obstacles", None)
            if refresh is not None:
                try:
                    refresh(arm_tag)
                except Exception:
                    st["errors"] += 1
        if found is not None:
            st["detours_used"] = st.get("detours_used", 0) + 1
        return found

    def _neutral_aware_gate(self, result, arm_tag: str):
        """Return the plan to execute (``result``, a re-plan) or None (abort)."""
        positions = getattr(result, "position", None)
        if positions is None or np.asarray(positions).size == 0:
            return result
        prm = self._neutral_aware_params()
        cap = getattr(self, "_neutral_aware_max_seg_override", None)
        if cap is not None:
            prm["max_seg"] = min(prm["max_seg"], int(cap))
        st = self._neutral_aware_stats()
        st["segments"] += 1
        waited = 0
        exhausted = False
        moved = False
        st["yields_this_segment"] = 0
        st["retreated_this_segment"] = False
        while True:
            try:
                c = self._neutral_aware_clearance(np.asarray(positions), arm_tag, prm)
            except Exception:
                st["errors"] += 1
                return result
            st["checked"] += 1
            st["min_clearance_m"] = c if st["min_clearance_m"] is None else min(st["min_clearance_m"], c)
            if c >= prm["clearance"]:
                break
            if waited == 0 and not moved:
                # Plan around the human first: the human may stay in the
                # corridor (seated work) longer than any wait budget.
                try:
                    detour = self._neutral_aware_detour(result, arm_tag, prm)
                except Exception:
                    st["errors"] += 1
                    detour = None
                if detour is not None:
                    return detour
            if waited + prm["tick"] > prm["max_seg"] or st["wait_sim_steps"] + prm["tick"] > prm["max_ep"]:
                st["budget_exhausted"] += 1
                exhausted = True
                break
            if waited == 0:
                st["waits"] += 1
            # Yield when the human would reach the robot where it stands:
            # lift (twice), then retreat to the home joints (once).
            try:
                threat = self._neutral_aware_robot_threat(arm_tag, prm)
            except Exception:
                threat = np.inf
                st["errors"] += 1
            if threat < prm["clearance"]:
                if st.get("yields_this_segment", 0) < int((self.config or {}).get("neutral_aware_max_yields", 2)):
                    if self._neutral_aware_yield(arm_tag, prm):
                        moved = True
                        st["yields"] = st.get("yields", 0) + 1
                        st["yields_this_segment"] = st.get("yields_this_segment", 0) + 1
                elif not st.get("retreated_this_segment", False):
                    st["retreated_this_segment"] = True
                    try:
                        retreated = self._neutral_aware_retreat_home(arm_tag, prm)
                    except Exception:
                        st["errors"] += 1
                        retreated = False
                    if retreated:
                        moved = True
                        st["retreats"] = st.get("retreats", 0) + 1
            # Hold: the arm keeps its last joint targets; only time (and the
            # human) moves.
            for _ in range(prm["tick"]):
                self.step_sim()
            waited += prm["tick"]
            st["wait_sim_steps"] += prm["tick"]
        if waited == 0 and not exhausted:
            return result
        # Resuming after a wait (or out of wait budget): re-plan on the
        # current human snapshot and use the new plan if it clears the
        # predicted human; otherwise keep the validated original, or abort
        # rather than drive a plan predicted to meet the human.
        try:
            new = self._neutral_aware_replan(result, arm_tag)
            c_new = (self._neutral_aware_clearance(np.asarray(new.position), arm_tag, prm)
                     if new is not None else -np.inf)
        except Exception:
            st["errors"] += 1
            new, c_new = None, -np.inf
        if new is not None and c_new >= prm["clearance"]:
            st["replans_used"] = st.get("replans_used", 0) + 1
            return new
        if exhausted or moved:
            try:
                detour = self._neutral_aware_detour(
                    result if new is None else new, arm_tag, prm)
            except Exception:
                st["errors"] += 1
                detour = None
            if detour is not None:
                return detour
        if moved and new is not None and not exhausted:
            # After a yield the original plan no longer starts at the robot;
            # the re-plan (from the current joints) is the only valid one.
            st["replans_used"] = st.get("replans_used", 0) + 1
            return new
        if not exhausted and not moved:
            return result
        if getattr(self, "_neutral_aware_optional", False):
            # The caller marked this segment optional (e.g. the last few cm
            # of a place descent): skip it without failing the episode.
            st["optional_skips"] = st.get("optional_skips", 0) + 1
            self._neutral_aware_skipped = True
            print(f"[neutral-aware] optional segment skipped: waited={waited}", flush=True)
            return None
        st["aborted"] = st.get("aborted", 0) + 1
        print(f"[neutral-aware] abort: waited={waited} exhausted={exhausted} moved={moved} "
              f"replan={'none' if new is None else round(float(c_new), 3)}", flush=True)
        self.plan_success = False
        return None

    def execute_plan(self, result, arm_tag: str, gripper_result=None):
        if (result is not None and self._neutral_aware_enabled()
                and not getattr(self, "_neutral_aware_bypass", False)):
            result = self._neutral_aware_gate(result, arm_tag)
            if result is None:
                return None
        return super().execute_plan(result, arm_tag, gripper_result=gripper_result)

    def _neutral_aware_plan_retry(self, attempt, arm_tag: str, label: str = "move"):
        """Full-state expert only: ``attempt()`` found no plan (returned None)
        WITHOUT moving the arm, e.g. the target pose is inside the human's
        current cloud.  Hold still (the human never adapts, it may move on),
        then try again; bounded by ``neutral_aware_plan_retry_rounds`` x
        ``neutral_aware_plan_retry_steps`` sim steps.  The blind expert and
        policies never take this path."""
        arm = self.robot.get_arm(arm_tag)
        q_start = np.asarray(arm.get_arm_qpos(), dtype=np.float64).copy()
        result = attempt()
        if result is not None or not self._neutral_aware_enabled():
            return result
        if float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=np.float64) - q_start))) > 0.02:
            return result         # the failed attempt moved the arm: no retry
        cfg = self.config or {}
        rounds = int(cfg.get("neutral_aware_plan_retry_rounds", 12))
        steps = int(cfg.get("neutral_aware_plan_retry_steps", 250))
        st = self._neutral_aware_stats()
        for k in range(rounds):
            if getattr(self, "avatar_collided", False):
                break
            q0 = np.asarray(arm.get_arm_qpos(), dtype=np.float64).copy()
            for _ in range(steps):
                self.step_sim()
            st["plan_retry_steps"] = st.get("plan_retry_steps", 0) + steps
            self.plan_success = True
            result = attempt()
            if result is not None:
                print(f"[neutral-aware] {label}: planned after {k + 1} hold round(s)", flush=True)
                st["plan_retries_ok"] = st.get("plan_retries_ok", 0) + 1
                return result
            moved = float(np.max(np.abs(np.asarray(arm.get_arm_qpos(), dtype=np.float64) - q0)))
            if moved > 0.02:      # the failed attempt executed something: stop
                break
        print(f"[neutral-aware] {label}: still no plan after hold rounds", flush=True)
        self.plan_success = False
        return None

    def neutral_aware_metrics(self) -> dict:
        if not self._neutral_aware_enabled():
            return {}
        return {"neutral_aware_expert": dict(self._neutral_aware_stats())}
