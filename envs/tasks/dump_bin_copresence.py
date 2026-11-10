"""Co-presence variant of ``dump_bin`` (``dump_bin_copresence``)."""

import numpy as np

from ..task_bases.dump_bin import DumpBin
from ..task_bases.dump_neutral_layout import initial_avatar_pose_is_clear
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class DumpBinCopresence(NeutralAvatarTableWorkMixin, DumpBin):
    """Robot dumps objects while the human performs unrelated table work."""

    # Sponge layouts that fail to initialize recover with denser anchors at
    # the original side headings; no additional avatar yaw is needed.
    NEUTRAL_LAYOUT_RECOVERY_DEGREES = (0,)
    # Plan around props as exact boxes (see _refresh_dump_obstacles).
    NEUTRAL_OBJECT_PLANNER_BOXES = True

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # The table bounds come from DumpBin.TABLE_HALF_SIZE (0.5, 0.45) @
        # centre (0, -0.35) -> x [-0.50, 0.50], y [-0.80, 0.10]
        # (`_neutral_table_xy_bounds` prefers the class attrs over a spec
        # `table_xy_bounds`).  Candidates sit inside y <= 0.06 / |x| <= 0.46
        # so none lands on or past the table edge.
        work_xy=(-0.36, 0.02),
        mouse_work_xy=(-0.36, 0.06),
        blocked_avatar_sides=("right",),
        work_candidates=(
            (-0.36, 0.02),
            (-0.42, 0.02),
            (-0.36, 0.06),
            (-0.42, 0.06),
            (-0.28, 0.06),
            (0.14, 0.05),
            (0.10, 0.04),
            (0.18, 0.04),
        ),
        mouse_work_candidates=(
            (-0.46, 0.00),
            (-0.46, 0.06),
            (-0.44, 0.02),
            (-0.44, 0.06),
        ),
    )

    INSTRUCTION = (
        "put the table objects into the bin while the human works nearby on "
        "the same table"
    )

    def reset(self, *args, **kwargs):
        self._dump_neutral_initial_clearance_counts = {
            "checked": 0, "accepted": 0, "rejected": 0,
        }
        return super().reset(*args, **kwargs)

    def _neutral_avatar_body_pose_is_valid(self, job, xy, body_pos):
        if not super()._neutral_avatar_body_pose_is_valid(job, xy, body_pos):
            return False
        if self.config.get("dump_neutral_initial_clearance_v1") is not True:
            return True
        clear = initial_avatar_pose_is_clear(self, job, xy, body_pos)
        counts = self._dump_neutral_initial_clearance_counts
        counts["checked"] += 1
        counts["accepted" if clear else "rejected"] += 1
        return clear

    def _neutral_contested_extra_places(self):
        # The bin stands on the floor beside the table, so no task region
        # lists it; objects travel to it across the right half of the table.
        return [tuple(float(v) for v in self.BIG_BIN_XY)]

    def evaluate(self):
        metrics = super().evaluate()
        if self.config.get("dump_neutral_initial_clearance_v1") is True:
            metrics["dump_neutral_initial_clearance"] = dict(
                self._dump_neutral_initial_clearance_counts
            )
        return metrics

    # Checked yield (full-state aware expert only).  The shared yield
    # lifts 8 cm with MPLib's screw planner against the full obstacle cloud
    # and no human check: when the waiting arm already sits inside the
    # human's inflated cloud MPLib rejects the start state (the robot keeps
    # waiting where the human's hand meets it), and when it does plan, the
    # lift can rise into a hand above the arm.  This yield plans straight
    # moves against the props and table only and accepts one only if the
    # predicted human capsules (the gate's own model, all links, synchronous
    # over the move) stay no closer than holding still would, and end
    # farther away.
    ESCAPE_MOVES = ((0.0, 0.0, 0.08), (0.0, -0.08, 0.08), (0.0, 0.0, 0.15), (0.0, -0.12, 0.0))
    # Clearance is capsule surface minus the gate's 7-10 cm link pads; the
    # escape path must keep at least this much (still several cm from contact).
    ESCAPE_MIN_PATH_CLEARANCE = -0.03

    def _neutral_aware_yield(self, arm_tag, prm):
        # Replaces the shared unchecked 8 cm lift (which could rise into the
        # human's hand above the arm): every yield move, the same 8 cm lift
        # first, is checked against the predicted human here.
        return self._dump_neutral_escape(arm_tag, prm)

    def _neutral_aware_retreat_home(self, arm_tag, prm):
        # The shared retreat checks only its path, not the home pose it then
        # holds while the human keeps moving (G+P run, seed 9000004: the
        # human's finger met the parked arm 65 steps after arriving).  Use
        # the checked escape, which also scores the pose after the move.
        return self._dump_neutral_escape(arm_tag, prm)

    def _dump_gate_checks_human(self):
        return self._neutral_aware_enabled()

    def _neutral_aware_clearance(self, positions, arm_tag, prm):
        # The shared gate predicts the human one clip frame per sim step and
        # assumes one plan row per sim step.  DumpBin retimes MPLib's 4 ms
        # rows onto the 2 ms sim clock (two steps per row), so without this
        # the predicted human ran twice as fast as the robot would move.
        positions = np.asarray(positions)
        per_row = int(round(0.004 / float(self.scene.sim.dt)))
        if per_row > 1 and positions.ndim == 2 and len(positions) > 1:
            positions = np.repeat(positions, per_row, axis=0)
        return super()._neutral_aware_clearance(positions, arm_tag, prm)

    # Idle guard (full-state aware expert only): while the arm stands still
    # outside a planned segment (pre-grasp waits, release and settle), check
    # every policy tick whether the human would reach it within the gate's
    # threat horizon, and if so take the checked escape (bounded per episode).
    IDLE_GUARD_MAX_ESCAPES = 6

    def step_sim(self):
        result = super().step_sim()
        if (getattr(self, "_dump_idle_guard", False) and not getattr(self, "_dump_guard_busy", False)
                and self._neutral_aware_enabled()):
            self._dump_guard_count = getattr(self, "_dump_guard_count", 0) + 1
            prm = self._neutral_aware_params()
            if self._dump_guard_count % prm["tick"] == 0:
                self._dump_guard_busy = True
                try:
                    st = self._neutral_aware_stats()
                    if st.get("idle_escapes", 0) < self.IDLE_GUARD_MAX_ESCAPES:
                        q = np.asarray(self.robot.get_arm("right").get_arm_qpos(), dtype=np.float64)
                        threat = self._dump_escape_clearance(q[None, :], "right", prm, hold=True)
                        if threat < prm["clearance"]:
                            print(f"[dump-neutral] idle guard threat={threat:.3f}", flush=True)
                            if self._dump_neutral_escape("right", prm):
                                st["idle_escapes"] = st.get("idle_escapes", 0) + 1
                except Exception as e:
                    print(f"[dump-neutral] idle guard error {e!r}", flush=True)
                finally:
                    self._dump_guard_busy = False
        return result

    def _dump_escape_clearance(self, positions, arm_tag, prm, hold=False, horizon=None, after=0) -> float:
        """Predicted clearance (capsule surface - link pad) of ALL arm links
        along ``positions`` (MPLib rows); ``hold`` checks the last row
        against the human over ``horizon`` sim steps (default: the gate's
        threat horizon) instead, starting ``after`` sim steps from now."""
        from ..task_bases.neutral_human_aware import path_human_clearance, predict_human_capsules
        import genesis.utils.geom as geom_utils
        arm = self.robot.get_arm(arm_tag)
        backend = arm.planner._mplib
        pin = backend.pinocchio_model
        base = np.asarray(arm.planner._base_pose, dtype=np.float64)
        Rb = geom_utils.quat_to_R(base[3:7])
        positions = np.asarray(positions, dtype=np.float64)
        stride = max(1, int(prm["stride"]))
        if hold:
            if horizon is None:
                horizon = max(prm["tick"], int((self.config or {}).get("neutral_aware_threat_horizon", 100)))
            offs = list(range(int(after), int(after) + horizon + 1, max(1, prm["tick"] // 5)))
            rows = [len(positions) - 1] * len(offs)
        else:
            rows = list(range(0, len(positions), stride))
            if rows[-1] != len(positions) - 1:
                rows.append(len(positions) - 1)
            # MPLib rows are 4 ms apart; DumpBin retimes them onto the sim clock.
            per_row = max(1.0, 0.004 / float(self.scene.sim.dt))
            offs = [int(round(r * per_row)) for r in rows]
        human = predict_human_capsules(self, offs)
        paths = {}
        for row in rows:
            pin.compute_forward_kinematics(backend.pad_move_group_qpos(positions[row]))
            for name, index in backend.link_name_2_idx.items():
                pose = pin.get_link_pose(index)
                p = np.asarray(pose.p if hasattr(pose, "p") else pose.get_p(), dtype=np.float64)
                paths.setdefault(name, []).append(Rb @ p + base[:3])
        links = [(np.asarray(v), prm["ee_pad"] if name == backend.move_group else prm["link_pad"])
                 for name, v in paths.items() if backend.link_name_2_idx[name] != 0]
        return path_human_clearance(links, human)

    def _dump_neutral_escape(self, arm_tag, prm) -> bool:
        import mplib.pymp as pymp
        import genesis.utils.geom as geom_utils
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        backend = planner._mplib
        q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        try:
            # Props are exact MPLib boxes when planner boxes are on.
            props = (None if self._neutral_object_planner_boxes_enabled()
                     and self._sync_neutral_object_planner_boxes(arm_tag)
                     else self._neutral_object_obstacle_points())
        except Exception:
            props = None
        try:
            planner.update_obstacles(props if props is not None and props.size else np.empty((0, 3)),
                                     resolution=.015)
            start = self._dump_escape_clearance(q[None, :], arm_tag, prm)
            backend.pinocchio_model.compute_forward_kinematics(backend.pad_move_group_qpos(q))
            pose = backend.pinocchio_model.get_link_pose(backend.link_name_2_idx[backend.move_group])
            base = np.asarray(planner._base_pose, dtype=np.float64)
            Rb = geom_utils.quat_to_R(base[3:7])
            p_world = Rb @ np.asarray(pose.p, dtype=np.float64) + base[:3]
            q_world = geom_utils.R_to_quat(Rb @ geom_utils.quat_to_R(np.asarray(pose.q, dtype=np.float64)))
            reasons = []
            best = None  # (score, move, pos, vel, stay, path, end): the safest move, not the first
            for move in self.ESCAPE_MOVES:
                goal = pymp.Pose(p=p_world + np.asarray(move), q=q_world)
                reason = "no screw"
                for step in (0.05, 0.02):
                    try:
                        out = backend.plan_screw(goal_pose=goal, current_qpos=planner._pad_qpos(q),
                                                 time_step=1 / 250, qpos_step=step, wrt_world=True,
                                                 verbose=False)
                    except Exception:
                        continue
                    if not out or out.get("status") != "Success":
                        continue
                    pos = np.asarray(out["position"], dtype=np.float64)
                    if pos.ndim != 2 or not len(pos) or np.max(np.abs(np.diff(np.vstack([q, pos]), axis=0))) > .30:
                        reason = "jump"
                        continue
                    # Compare with staying put for as long as the move takes:
                    # the human keeps moving either way.
                    steps = int(np.ceil(len(pos) * 0.004 / float(self.scene.sim.dt)))
                    stay = self._dump_escape_clearance(q[None, :], arm_tag, prm, hold=True,
                                                       horizon=steps + 100)
                    path = self._dump_escape_clearance(pos, arm_tag, prm)
                    end = self._dump_escape_clearance(pos[-1:], arm_tag, prm, hold=True, after=steps)
                    if (path < self.ESCAPE_MIN_PATH_CLEARANCE or path < stay - 0.002
                            or end < stay + 0.01):
                        reason = f"clearance stay={stay:.3f} path={path:.3f} end={end:.3f}"
                        continue
                    score = min(path, end)
                    if best is None or score > best[0]:
                        best = (score, move, pos, out.get("velocity", np.zeros_like(pos)), stay, path, end)
                    reason = None
                    break
                if reason is not None:
                    reasons.append(reason)
            if best is not None:
                _, move, pos, vel, stay, path, end = best
                print(f"[dump-neutral] escape move={move} rows={len(pos)} clearance "
                      f"start={start:.3f} stay={stay:.3f} path={path:.3f} end={end:.3f}", flush=True)
                self._neutral_aware_bypass = True
                try:
                    DumpBin.execute_plan(self, PlanResult(True, pos, vel), arm_tag)
                finally:
                    self._neutral_aware_bypass = False
                st = self._neutral_aware_stats()
                st["escapes"] = st.get("escapes", 0) + 1
                return True
            print(f"[dump-neutral] no escape: start={start:.3f} {reasons}", flush=True)
            return False
        finally:
            self._refresh_dump_obstacles(arm_tag)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(80):
                self.step_sim()

        self._start_neutral_avatar_table_work(add_start_delay=True)

        ok = True
        for i, cube in enumerate(self.objects):
            self._neutral_debug(
                f"[dump_bin_neutral] === picking object {i + 1}/{self.num_objects} ==="
            )
            cube_ok = self._pick_and_drop_cube(cube, arm_tag)
            if not cube_ok:
                self._neutral_debug(
                    f"[dump_bin_neutral] object {i + 1} failed; stopping"
                )
                return False

        return ok

    def check_success(self) -> bool:
        if not DumpBin.check_success(self):
            return False
        # The mixin's avatar_success_gate veto (this override bypasses it):
        # a human prop the robot knocked over fails the episode.
        if getattr(self, "_neutral_prop_disturbed", None):
            return False
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            self._neutral_debug(
                "[dump_bin_neutral] avatar collision detected: "
                f"first_frame={summary.get('first_collision_frame')} "
                f"deepest={summary.get('deepest_depth_m')}"
            )
            return False
        return True
