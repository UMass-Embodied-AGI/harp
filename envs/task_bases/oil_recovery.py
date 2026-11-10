"""Opt-in oil placement: checked native plans, measured arrivals, ordinary PD.

Acquisition and the existing initial lift remain the task's responsibility.
This candidate starts at the post-lift grasp-transform cache. It neither
recaches a slipping bottle nor changes contact physics or benchmark scoring.
"""
import json
import os
from contextlib import contextmanager

import numpy as np

from ..utils import to_numpy
from .oil_recovery_plan import OilRecoveryPlan, check_oil_lift_path, pose7, pose_matrix, pose_error


class OilFryingMotionClock:
    """Apply the oil task's runtime rate changes to an already loaded clip.

    AvatarController resamples a clip only when play_animation starts. Changing
    its frame_ratio later therefore cannot slow that clip on its own. Keep the
    same motion and its current frame, and advance its original sampled timeline
    at the requested rate. The controller still updates skin and held props on
    every physics step, including steps between animation samples.
    """

    def __init__(self, motion, avatar, source_frame_ratio):
        self.motion = motion
        self.avatar = avatar
        self.source_frame_ratio = float(source_frame_ratio)
        if not np.isfinite(self.source_frame_ratio) or self.source_frame_ratio <= 0:
            raise ValueError("invalid source frying frame ratio")
        self._fraction = 0.0

    def __getattr__(self, name):
        return getattr(self.motion, name)

    def step(self, skip_avatar_animation=False):
        if skip_avatar_animation:
            return self.motion.step(skip_avatar_animation=True)
        ratio = float(self.avatar.frame_ratio)
        if not np.isfinite(ratio) or ratio < self.source_frame_ratio:
            raise ValueError("oil frying clock requires a finite, nonaccelerated rate")
        self._fraction += self.source_frame_ratio / ratio
        if self._fraction >= 1.0 - 1e-12:
            self._fraction = max(0.0, self._fraction - 1.0)
            self.motion.step()


def _array(value, shape):
    result = np.asarray(to_numpy(value), dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid oil observation; expected {shape}")
    return result.copy()


def _bounds(entity):
    bounds = _array(entity.get_AABB(), (2, 3))
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("empty oil scene geometry")
    return bounds


def _pose(entity):
    return pose_matrix(np.r_[_array(entity.get_pos(), (3,)),
                             _array(entity.get_quat(), (4,))])


def oil_lift_clearance_offset(task, arm):
    """One 6 cm horizontal egress, away from the measured forearm/wrist pair.

    This is a spatial candidate, not a prediction of future avatar motion.
    Use collision vertices rather than TCP position to choose the direction;
    an ambiguous horizontal direction is rejected instead of guessed.
    """
    capsules = [c for c in task.avatar_collider.current_capsules() if c[0] == "r_forearm"]
    if len(capsules) != 1:
        raise ValueError("one live right-forearm capsule required")
    _, start, end, radius = capsules[0]
    start, end = _array(start, (3,)), _array(end, (3,))
    radius = float(radius)
    segment = end - start
    if not np.isfinite(radius) or radius <= 0 or np.dot(segment, segment) < 1e-8:
        raise ValueError("invalid live forearm geometry")
    links = [link for link in arm.entity.links if link.name in ("link6", "panda_link6")]
    if len(links) != 1:
        raise ValueError("one measured link6 required")
    parts = [np.asarray(to_numpy(g.get_verts()), dtype=float) for g in links[0]._geoms]
    if not parts or any(p.ndim != 2 or p.shape[1] != 3 or not len(p)
                        or not np.isfinite(p).all() for p in parts):
        raise ValueError("invalid measured link6 collision vertices")
    points = np.concatenate(parts)
    t = np.clip((points - start) @ segment / np.dot(segment, segment), 0., 1.)
    closest = start + t[:, None] * segment
    distances = np.linalg.norm(points - closest, axis=1)
    index = int(np.argmin(distances))
    direction = points[index] - closest[index]
    direction[2] = 0.
    length = float(np.linalg.norm(direction))
    if length < .005:
        raise ValueError("forearm/link6 pair has ambiguous horizontal egress")
    offset = direction * (.06 / length)
    return offset, {"offset_m": offset.tolist(), "forearm_start": start.tolist(),
                    "forearm_end": end.tolist(), "forearm_radius_m": radius,
                    "link6_nearest_vertex": points[index].tolist(),
                    "forearm_nearest_axis_point": closest[index].tolist(),
                    "initial_proxy_gap_m": float(distances[index] - radius)}


def plan_oil_clearance_lift(task, arm, arm_tag, lift_height):
    """Default-OFF spatial egress: one checked native screw, no fallbacks."""
    from ..planning.base import PlanResult
    from ..utils import Pose

    _require_mode(task, arm)
    offset, geometry = oil_lift_clearance_offset(task, arm)
    tcp = Pose.from_pose7(arm.get_ee_pose())
    target = Pose(tcp.p + offset + np.array([0., 0., lift_height]), tcp.q) * arm.tcp_offset.inv()
    task._oil_clearance_lift_diagnostics = dict(geometry, stage="proposed",
                                              prior_avatar_contact=False,
                                              target_link=target.to_pose7().tolist())
    print("[oil-clearance-lift] " + json.dumps(task._oil_clearance_lift_diagnostics), flush=True)
    refresh_obstacles(task, arm)
    full, finger_targets = joint_state(arm)
    native = arm.planner._mplib
    raw = native.plan_screw(goal_pose=arm.planner._to_mplib_pose(target.to_pose7()),
                            current_qpos=full.copy(), time_step=1/250,
                            qpos_step=.01, wrt_world=True, verbose=False)
    result = PlanResult(raw.get("status") == "Success", raw.get("position"), raw.get("velocity"))
    report = check_oil_lift_path(arm, result, full, finger_targets, target.to_pose7())
    task._oil_clearance_lift_diagnostics.update(stage="planned", path=report)
    print("[oil-clearance-lift] " + json.dumps(task._oil_clearance_lift_diagnostics), flush=True)
    return result, target


def checked_oil_carry_start(task, arm):
    """Refresh strictly; query actual full/finger states before native transit."""
    from ..planning.base import PlanResult

    _require_mode(task, arm)
    refresh_obstacles(task, arm)
    full, finger_targets = joint_state(arm)
    move = np.asarray(arm.planner._mplib.move_group_joint_indices, dtype=int)
    stationary = PlanResult(True, np.repeat(full[move][None, :], 2, axis=0))
    link = pose_matrix(arm.get_ee_pose()) @ np.linalg.inv(pose_matrix(arm.tcp_offset.to_pose7()))
    check_oil_lift_path(arm, stationary, full, finger_targets, link)
    return full


def fixture_bounds(task):
    """Keep legacy planning envelopes AND every actual kitchen fixture.

    In particular, the legacy cooktop proxy is not at the built stove's XY.
    Do not replace actual body/burner geometry with that proxy. The decorative
    non-colliding floor tile is the only omitted backdrop entity.
    """
    result = {"table": _bounds(task.table),
              "cabinet": np.array([[-.85, .195, 1.315], [.85, .495, 1.705]]),
              "cooktop": np.array([task._COOKTOP_X, task._COOKTOP_Y,
                                   task._COOKTOP_Z], dtype=float).T}

    def collect(prefix, value):
        if isinstance(value, dict):
            for name, child in sorted(value.items()):
                collect(prefix + "_" + name, child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                collect(prefix + "_" + str(index), child)
        else:
            result[prefix] = _bounds(value)

    if not task.kitchen_cooktop.get("body") or not task.kitchen_cooktop.get("burners"):
        raise ValueError("actual cooktop geometry missing")
    collect("actual_cooktop", task.kitchen_cooktop)
    for name, value in sorted(task.kitchen_backdrop.items()):
        if name != "floor":
            collect("actual_backdrop_" + name, value)
    if not {"wall", "cabinet"} <= set(task.kitchen_backdrop):
        raise ValueError("actual kitchen backdrop missing")
    for role, entity in sorted(task.kitchen_items.items()):
        if entity is task.oil_entity or role in result:
            raise ValueError("ambiguous oil/clutter geometry")
        result[role] = _bounds(entity)
    if not {"vinegar", "soy_sauce"} <= set(result):
        raise ValueError("required neighboring bottles missing")
    return result


def refresh_obstacles(task, arm):
    """Same three cloud sources as the task; missing avatar/errors are fatal."""
    parts = []
    for getter in (task._avatar_obstacle_points, task._cabinet_obstacle_points,
                   task._cooktop_obstacle_points):
        value = np.asarray(getter(), dtype=float)
        if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
            raise ValueError("missing or invalid oil obstacle cloud")
        parts.append(value)
    arm.planner.update_obstacles(np.concatenate(parts), resolution=task._AVATAR_OBSTACLE_RES)


def joint_state(arm):
    """Explicit simulator-to-native full9 map, including both finger sources."""
    native = arm.planner._mplib
    names = list(native.robot.get_user_joint_names())
    indices = [arm.planner._arm_joint_index[name.removeprefix("panda_")] for name in names]
    if len(names) != 9 or len(set(indices)) != 9 or any(
            not isinstance(i, (int, np.integer)) or i < 0 or i >= 9 for i in indices):
        raise ValueError("unsupported oil full-joint mapping")
    full = _array(arm.entity.get_qpos(), (9,))[indices]
    cached = _array(arm._cached_target, (9,))[indices]
    fingers = [names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")]
    scalar = arm.gripper_closed + float(arm.gripper_val) * (arm.gripper_open-arm.gripper_closed)
    return full, {"cached": cached[fingers], "scalar": np.full(2, scalar)}


def released_support(task, arm):
    """A quiet upright bottle must actually rest on the counter, away from jaws.

    The legacy task score checks XY/tilt but not support height. This candidate
    refuses completion with a bottle still carried by an opening finger.
    AABB separation is conservative, not a contact-force certificate.
    """
    bounds = _bounds(task.oil_entity)
    gap = float(bounds[0, 2] - task.TABLE_TOP_Z)
    speed = float(np.linalg.norm(_array(task.oil_entity.get_vel(), (3,))))
    spin = float(np.linalg.norm(_array(task.oil_entity.get_ang(), (3,))))
    fingers = [link for link in arm.entity.links if "finger" in link.name]
    if len(fingers) != 2 or arm.ee_link is None:
        raise ValueError("released hand geometry missing")
    separations = []
    for link in [arm.ee_link, *fingers]:
        hand = _bounds(link)
        axis_gap = np.maximum(0., np.maximum(bounds[0]-hand[1], hand[0]-bounds[1]))
        separations.append(float(np.linalg.norm(axis_gap)))
    minimum = min(separations)
    return {"supported_and_quiet": bool(-.002 <= gap <= .003 and speed < .03
                                        and spin < .3 and minimum >= .002),
            "bottom_above_table_m": gap, "linear_speed_m_s": speed,
            "angular_speed_rad_s": spin, "hand_aabb_separation_m": minimum}


def _require_mode(task, arm):
    if not task.plan_success or getattr(task, "avatar_collided", False):
        raise ValueError("earlier failure or avatar collision is latched")
    if task._avatar_retreat_enabled() or getattr(task, "avatar_retreat_checker", None) is not None:
        raise ValueError("oil candidate requires retreat-disabled evaluation")
    if getattr(task, "_gripper_attached", {}) or getattr(arm, "_attached", None):
        raise ValueError("oil candidate requires an unattached physical grasp")
    if arm.planner._bounded_paths_debug_enabled() or any(
            os.environ.get(name, "0").lower() not in ("", "0", "false")
            for name in ("DEBUG_TELEPORT", "USE_SETQPOS")):
        raise ValueError("debug execution/planning bypass enabled")


class _Placement:
    def __init__(self, task, arm_tag):
        from ..scenes.kitchen import _measure_mesh
        from ..object_catalog import get_entry

        self.task, self.tag = task, arm_tag
        self.arm = task.robot.get_arm(arm_tag)
        _require_mode(task, self.arm)
        self.dt = float(task.scene.sim.dt)
        if not np.isclose(self.dt, .002, atol=1e-12, rtol=0):
            raise ValueError("unsupported oil physics timestep")
        if not np.isclose(self.arm.gripper_val, .04, atol=1e-12, rtol=0):
            raise ValueError("oil acquisition grip changed")
        self.grip = float(self.arm.gripper_val)
        entry = get_entry("029_olive-oil")
        low, high, _ = _measure_mesh(entry.object_id, entry.model_id)
        self.payload = np.stack([low, high]) * float(task._oil_scale)
        self.held = pose_matrix(task._T_ee_bottle.to_pose7())
        self.clutter = {name: _pose(entity) for name, entity in task.kitchen_items.items()}
        self.steps, self.phase, self.opened = 0, "post_lift", False
        self.minimum_bottom = float("inf")
        self.observe(held=True)

    def emit(self, stage, **values):
        record = dict(stage=stage, physics_steps=self.steps, opened=self.opened,
                      sim_step=getattr(self.task, "_interaction_sim_step", None),
                      video_frame=max(0, len(getattr(self.task, "_video_frames", ())) - 1),
                      **values)
        self.task._oil_rotated_diagnostics = record
        print("[oil-rotated-placement] " + json.dumps(record, allow_nan=False), flush=True)

    def observe(self, *, held):
        _require_mode(self.task, self.arm)
        tcp = pose_matrix(self.arm.get_ee_pose())
        bottle = _pose(self.task.oil_entity)
        bottom = float(_bounds(self.task.oil_entity)[0, 2])
        self.minimum_bottom = min(self.minimum_bottom, bottom)
        for name, original in self.clutter.items():
            error = pose_error(_pose(self.task.kitchen_items[name]), original)
            if error["position_m"] > .002 or error["orientation_rad"] > .01:
                raise ValueError("neighbor moved during oil placement: " + name)
        fingers = [link for link in self.arm.entity.links if "finger" in link.name]
        if len(fingers) != 2 or min(_bounds(link)[0, 2] for link in fingers) < self.task.TABLE_TOP_Z + .002:
            raise ValueError("measured oil fingers entered table clearance")
        if held:
            error = pose_error(np.linalg.inv(tcp) @ bottle, self.held)
            if error["position_m"] > .008 or error["orientation_rad"] > .05:
                raise ValueError("oil slipped relative to the measured grasp: " + str(error))
            if bottom < self.task.TABLE_TOP_Z + .002:
                raise ValueError("measured held oil entered table clearance")
        elif bottom < self.task.TABLE_TOP_Z - .002:
            raise ValueError("released oil penetrated table by more than 2 mm")
        return tcp, bottle

    @contextmanager
    def context(self, *, held=True):
        refresh_obstacles(self.task, self.arm)
        fixtures = fixture_bounds(self.task)
        if not held:
            fixtures["released_bottle"] = _bounds(self.task.oil_entity)
        full, fingers = joint_state(self.arm)
        base = self.arm.planner._mplib.robot.get_base_pose()
        with OilRecoveryPlan(self.arm.planner._mplib, measured_qpos=full,
            finger_targets=fingers, gripper_open=self.arm.gripper_open,
            world_base_pose7=np.r_[base.p, base.q], hand_tcp_pose7=self.arm.tcp_offset.to_pose7(),
            tcp_payload_pose7=pose7(self.held), payload_bounds_local=self.payload,
            fixture_aabbs=fixtures, held=held) as planner:
            state = planner.check_state()
            error = pose_error(state["tcp_pose7"], self.arm.get_ee_pose())
            if error["position_m"] > .002 or error["orientation_rad"] > .01:
                raise ValueError("native/physical oil TCP frame disagreement: " + str(error))
            yield planner

    def step(self, *, held):
        _require_mode(self.task, self.arm)
        self.task.step_sim()
        self.steps += 1
        # Pose retention and impact are observed each physics step. Native
        # collision checking is discrete preflight, not a dynamic guarantee.
        self.observe(held=held)

    def execute(self, result, target_link, *, held, bottle_target=None):
        target_tcp = pose_matrix(target_link) @ pose_matrix(self.arm.tcp_offset.to_pose7())
        for waypoint in result["position"]:
            for _ in range(2):  # Native4ms samples, physical2ms; ordinary PD only.
                self.arm.set_arm_joints(waypoint)
                self.task.robot.set_gripper(self.grip if held else 1., self.tag)
                self.step(held=held)
        quiet = 0
        previous = _pose(self.task.oil_entity)
        for index in range(360):  # Preserve original endpoint-hold budget.
            self.arm.set_arm_joints(result["position"][-1])
            self.task.robot.set_gripper(self.grip if held else 1., self.tag)
            self.step(held=held)
            if (index + 1) % 10:
                continue
            tcp, bottle = self.observe(held=held)
            error = pose_error(tcp, target_tcp)
            motion = pose_error(bottle, previous)
            previous = bottle
            close = error["position_m"] <= .002 and error["orientation_rad"] <= .01
            if bottle_target is not None:
                actual = pose_error(bottle, bottle_target)
                close = close and actual["position_m"] <= .008 and actual["orientation_rad"] <= .05
            quiet = quiet + 1 if close and motion["position_m"]/.02 < .03 and motion["orientation_rad"]/.02 < .3 else 0
            if quiet >= 5:
                self.emit(self.phase + "_arrived", tcp=pose7(tcp).tolist(), bottle=pose7(bottle).tolist(),
                          target_error=error)
                return
        raise ValueError("no stable measured " + self.phase + " arrival")

    def run(self, saved_frame_ratio):
        self.emit("post_lift", fixtures={name: bounds.tolist() for name, bounds in fixture_bounds(self.task).items()},
                  held_pose7=pose7(self.held).tolist())
        self.phase = "hover"
        with self.context() as planner:
            branch = planner.select_branch(self.task._safe_target_xy, self.task._oil_upright_origin_z_on_table)
            result, report = planner.plan_to_hover(branch)
            link = planner.link_target(branch["hover_pose7"])
        self.emit("hover_plan_checked", branch=branch, plan_check=report)
        self.execute(result, link, held=True, bottle_target=branch["hover_pose7"])
        self.phase = "descent"
        # Fresh avatar cloud/full measured joints; preserve original grasp transform.
        with self.context() as planner:
            result, report = planner.plan_descent(branch)
            link = planner.link_target(branch["low_pose7"])
        self.emit("descent_plan_checked", plan_check=report)
        self.execute(result, link, held=True, bottle_target=branch["low_pose7"])
        with self.context() as planner:
            planner.check_state(opening=True, target_bottle=branch["low_pose7"])
        self.observe(held=True)
        self.emit("ready_to_open")
        self.phase, self.opened = "opening", True
        # Same200-step gradual opening as BaseTask.open_gripper; add read-only checks.
        for command in np.linspace(self.grip, 1., 200):
            self.task.robot.set_gripper(float(command), self.tag)
            self.step(held=False)
        for _ in range(150):
            self.step(held=False)
        self.phase = "retreat"
        tcp = pose_matrix(self.arm.get_ee_pose())
        tcp[2, 3] += .10
        link = tcp @ np.linalg.inv(pose_matrix(self.arm.tcp_offset.to_pose7()))
        with self.context(held=False) as planner:
            result, report = planner.plan_retreat(link)
        self.emit("retreat_plan_checked", plan_check=report)
        self.execute(result, link, held=False)
        self.task.avatar.frame_ratio = saved_frame_ratio
        quiet = 0
        for index in range(180):
            self.step(held=False)
            if (index + 1) % 10 == 0:
                support = released_support(self.task, self.arm)
                quiet = quiet + 1 if support["supported_and_quiet"] else 0
                if quiet >= 5 and self.task.avatar.spare():
                    break
        if quiet < 5:
            raise ValueError("released oil did not settle on the table apart from the hand")
        self.emit("finished", minimum_bottle_bottom_m=self.minimum_bottom, released_support=support)
        return True


def place_rotated_oil(task, arm_tag, saved_frame_ratio):
    """Fail closed; never open as a fallback to a rejected carry/descent."""
    operation = None
    try:
        operation = _Placement(task, arm_tag)
        return operation.run(saved_frame_ratio)
    except Exception as exc:
        task.plan_success = False
        if operation is not None:
            operation.emit("rejected", phase=operation.phase, reason=str(exc))
        else:
            record = {"stage": "rejected_before_carry", "reason": str(exc), "opened": False}
            task._oil_rotated_diagnostics = record
            print("[oil-rotated-placement] " + json.dumps(record), flush=True)
        return False
    finally:
        task.avatar.frame_ratio = saved_frame_ratio
