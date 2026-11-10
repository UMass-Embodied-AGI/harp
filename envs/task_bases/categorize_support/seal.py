"""Measured model-1 seal grasps; no low wrist rotation or RRT descent fallback."""

import numpy as np
import transforms3d as t3d
from dataclasses import dataclass, field
from contextlib import contextmanager

from ...avatar.collision_checker import sample_capsule_planner_points
from ...base_task_parts.grasp_select import mplib_world_link_pose
from ...grasp import GraspPose, tcp_to_link_pose
from ...utils import Pose, to_numpy
from ...planning.base import PlanResult


class SealApproachRejected(RuntimeError):
    """A physically inaccurate high approach must not start the descent."""


@dataclass
class SealSelectorExecution:
    arm_tag: str
    plans: list = field(default_factory=list)
    pre_tcp: object = None
    next_leg: int = 0


def retime_selector_plan(plan, gripper_result, physics_dt, planner_dt=0.004):
    """Replay the planner's physical duration without modifying its joint path."""
    if (not np.isfinite(physics_dt) or physics_dt <= 0
            or not np.isfinite(planner_dt) or planner_dt <= 0):
        raise SealApproachRejected("invalid selector replay clock")
    if plan is None or not plan.success:
        raise SealApproachRejected("invalid selector plan")

    def retime(values, dimensions):
        values = np.asarray(values, dtype=float)
        if values.ndim != dimensions or not np.all(np.isfinite(values)):
            raise SealApproachRejected("invalid selector trajectory")
        if len(values) < 2:
            return values.copy()
        duration = (len(values) - 1) * planner_dt
        times = np.linspace(0, duration, int(np.ceil(duration / physics_dt)) + 1)
        source_times = np.linspace(0, duration, len(values))
        if dimensions == 1:
            return np.interp(times, source_times, values)
        return np.column_stack([np.interp(times, source_times, values[:, j])
                                for j in range(values.shape[1])])

    position = retime(plan.position, 2)
    velocity = retime(plan.velocity, 2)
    if position.shape != velocity.shape or not position.size:
        raise SealApproachRejected("inconsistent selector trajectory")
    gripper = None if gripper_result is None else retime(gripper_result, 1)
    return PlanResult(True, position, velocity), gripper


def execute_seal_selector_plan(task, delegate, plan, arm_tag, gripper_result, context):
    """Task-local selector replay; other task motion retains its normal method."""
    if (arm_tag != context.arm_tag or context.next_leg >= len(context.plans)
            or context.plans[context.next_leg] is not plan):
        raise SealApproachRejected("unexpected motion inside seal selector")
    dt = float(task.scene.sim.dt)
    retimed, gripper = retime_selector_plan(plan, gripper_result, dt)
    print(f"[seal-grasp] replay leg={context.next_leg} original={len(plan.position)} "
          f"physical={len(retimed.position)} dt={dt}", flush=True)
    result = delegate(retimed, arm_tag, gripper_result=gripper)
    if context.next_leg == 0:
        arm = task.robot.get_arm(arm_tag)
        sample_steps = max(1, int(round(.02 / dt)))
        sample_dt = sample_steps * dt
        previous = Pose.from_pose7(arm.get_ee_pose())
        quiet = 0
        for count in range(sample_steps, int(np.ceil(1.0 / dt)) + 1, sample_steps):
            for _ in range(sample_steps):
                task.step_sim()
            current = Pose.from_pose7(arm.get_ee_pose())
            speed = np.linalg.norm(current.p - previous.p) / sample_dt
            angular = 2 * np.arccos(np.clip(abs(np.dot(current.q, previous.q)), 0, 1)) / sample_dt
            valid = _pose_matches(current, context.pre_tcp)
            quiet = quiet + 1 if valid and speed < .03 and angular < .3 else 0
            previous = current
            if quiet >= 5:
                print(f"[seal-grasp] high pre-pose settled after={count} "
                      f"planned={context.pre_tcp.to_pose7().tolist()} "
                      f"actual={current.to_pose7().tolist()}", flush=True)
                break
        else:
            print(f"[seal-grasp] high pre-pose rejected planned={context.pre_tcp.to_pose7().tolist()} "
                  f"actual={previous.to_pose7().tolist()}", flush=True)
            raise SealApproachRejected("high pre-pose not accurate and quiet; descent forbidden")
    context.next_leg += 1
    return result


def select_seal_with_timed_execution(task, arm_tag, **kwargs):
    """Scope task-local retiming to the selector's exact two accepted plans."""
    previous = getattr(task, "_seal_selector_execution", None)
    if previous is not None:
        raise SealApproachRejected("nested seal selector")
    context = SealSelectorExecution(arm_tag)
    check = kwargs.pop("post_plan_check")

    def capture(plans):
        if not check(plans):
            return False
        context.plans = list(plans[:2])
        context.pre_tcp = plans[3] * task.robot.get_arm(arm_tag).tcp_offset
        return True

    task._seal_selector_execution = context
    try:
        return task.select_and_execute_grasp(arm_tag=arm_tag, post_plan_check=capture, **kwargs)
    finally:
        task._seal_selector_execution = previous


def seal_candidates(object_pose, bounds, model_id, object_scale, safe_z):
    """Runtime-frame annotations derived from this selected model's live AABB.

    The jaws close across the shorter horizontal dimension. Model1's settled
    footprint is about64x76mm: closing along76mm leaves only2mm clearance per
    finger at the80mm open limit. The shorter axis leaves about8mm per side.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (model_id != 1 or not np.isfinite(object_scale) or object_scale <= 0
            or bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
            or not np.all(bounds[1] > bounds[0])
            or not np.all(np.isfinite(object_pose.to_pose7()))
            or not np.isfinite(safe_z)):
        raise ValueError("seal candidates require finite model1 live geometry")
    center = bounds.mean(axis=0)
    width = bounds[1] - bounds[0]
    center[2] = bounds[1, 2] - 0.012
    if center[2] <= bounds[0, 2] + 0.012 or safe_z <= bounds[1, 2] + 0.15:
        raise ValueError("seal has insufficient purchase or approach clearance")
    yaw = np.pi / 2 if width[0] < width[1] else 0.0
    candidates = []
    for index, angle in enumerate((yaw, yaw + np.pi)):
        rotation = t3d.axangles.axangle2mat([0, 0, 1], angle) @ np.diag([1., -1., -1.])
        world_tcp = Pose(center.copy(), t3d.quaternions.mat2quat(rotation))
        local_tcp = object_pose.inv() * world_tcp
        candidates.append(GraspPose(
            f"seal_model1_measured_short_axis_{index}", local_tcp.p, local_tcp.q,
            pre_distance=float(safe_z - center[2]), source="manual",
            scale_frame="runtime", category="measured_seal_model1",
        ))
    return candidates


def gripper_sweep_radius(arm):
    """Bound the fully opened hand/fingers about the MPLib hand link origin."""
    origin = np.asarray(to_numpy(arm.ee_link.get_pos()), dtype=float).reshape(3)
    parts = []
    for link in arm.entity.links:
        if link is not arm.ee_link and "finger" not in link.name:
            continue
        for geom in getattr(link, "_geoms", ()):
            verts = np.asarray(to_numpy(geom.get_verts()), dtype=float)
            if verts.ndim != 2 or verts.shape[1] != 3 or not np.all(np.isfinite(verts)):
                raise ValueError("invalid seal gripper geometry")
            if len(verts):
                parts.append(verts)
    if not parts:
        raise ValueError("missing seal gripper collision geometry")
    return float(np.linalg.norm(np.concatenate(parts) - origin, axis=1).max()) + 0.01


def _pose_matches(actual, target, position_tol=0.006, angle_tol=0.05):
    if actual is None:
        return False
    distance = np.linalg.norm(actual.p - target.p)
    angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1))
    return bool(np.isfinite(distance) and distance <= position_tol and angle <= angle_tol)


def seal_preflight_check(arm, bounds, sweep_radius):
    """Reject any RRT hand sweep through the seal before executing either leg."""
    bounds = np.asarray(bounds, dtype=float)

    def check(plans):
        approach, descent, grasp_link, pre_link = plans
        if not approach.position.size or not descent.position.size:
            return False
        for plan, goal in ((approach, pre_link), (descent, grasp_link)):
            if not _pose_matches(mplib_world_link_pose(arm.planner, plan.position[-1]), goal):
                return False
        # A sphere covering the opened gripper is conservative at every
        # orientation. It may rotate only while this swept volume is clear.
        previous_hand = None
        previous_qpos = None
        for qpos in approach.position:
            # Audit intermediate joint interpolation too, then connect the
            # sampled hand centers against the radius-expanded object box.
            count = (1 if previous_qpos is None else
                     max(1, int(np.ceil(np.max(np.abs(qpos - previous_qpos)) / 0.005))))
            samples = ([qpos] if previous_qpos is None else
                       np.linspace(previous_qpos, qpos, count + 1)[1:])
            for sample in samples:
                hand = mplib_world_link_pose(arm.planner, sample)
                if hand is None:
                    return False
                if previous_hand is not None and segment_intersects_box(
                        previous_hand.p, hand.p, bounds[0] - sweep_radius,
                        bounds[1] + sweep_radius):
                    print("[seal-grasp] rejected pre-grasp swept segment", flush=True)
                    return False
                previous_hand = hand
            previous_qpos = qpos
            hand = mplib_world_link_pose(arm.planner, qpos)
            if hand is None:
                return False
            close_xy = np.all(hand.p[:2] >= bounds[0, :2] - sweep_radius) and np.all(
                hand.p[:2] <= bounds[1, :2] + sweep_radius,
            )
            if close_xy and hand.p[2] - sweep_radius <= bounds[1, 2]:
                print("[seal-grasp] rejected low pre-grasp rotation sweep", flush=True)
                return False
        target_tcp = grasp_link * arm.tcp_offset
        for qpos in descent.position:
            hand = mplib_world_link_pose(arm.planner, qpos)
            if hand is None:
                return False
            tcp = hand * arm.tcp_offset
            angle = 2 * np.arccos(np.clip(abs(np.dot(tcp.q, target_tcp.q)), 0, 1))
            if (angle > 0.05 or np.linalg.norm(tcp.p[:2] - target_tcp.p[:2]) > 0.006
                    or tcp.p[2] < target_tcp.p[2] - 0.006):
                print("[seal-grasp] rejected rotating/nonvertical descent", flush=True)
                return False
        return True

    return check


def segment_intersects_box(start, end, low, high):
    """Closed segment/AABB slab test, including tangencies and zero motion."""
    lower, upper = 0.0, 1.0
    for p, d, lo, hi in zip(start, np.asarray(end) - start, low, high):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return False
            continue
        t0, t1 = sorted(((lo - p) / d, (hi - p) / d))
        lower, upper = max(lower, t0), min(upper, t1)
        if lower > upper:
            return False
    return True


def _refresh_obstacles(task, pick, arm_tag):
    """Preserve baskets, other objects and the current avatar in one cloud."""
    parts = [task._basket_aabb_points(i) for i in range(len(task.basket_poses))]
    seen = {id(pick.actor)}
    targets = list(task._sort_targets) + list(getattr(task, "_avatar_sort_targets", ()))
    for actor, name, model_id, _ in targets:
        if id(actor) not in seen:
            parts.append(task._actor_aabb_points(actor, name, model_id))
            seen.add(id(actor))
    avatar_points = getattr(task, "_avatar_obstacle_points", None)
    if avatar_points is not None:
        points = avatar_points()
        if points is not None and points.size:
            parts.append(points)
    else:
        collider = getattr(task, "avatar_collider", None)
        if collider is not None:
            points = sample_capsule_planner_points(collider.current_capsules(), inflate_factor=1.10)
            if points is not None and points.size:
                parts.append(points)
    collider = getattr(task, "avatar_collider", None)
    if collider is not None:
        for _, center, axes, half in collider.current_boxes():
            coordinates = [np.linspace(-h - 0.005, h + 0.005,
                                       max(2, int(2 * h / 0.015) + 2))
                           for h in np.asarray(half, dtype=float)]
            grid = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
            parts.append(grid.reshape(-1, 3) @ np.asarray(axes).T + center)
    if (getattr(task, "_neutral_object_obstacles_enabled", lambda: False)()):
        points = task._neutral_object_obstacle_points()
        if points is not None and points.size:
            parts.append(points)
    task.robot.get_arm(arm_tag).planner.update_obstacles(
        np.concatenate(parts) if parts else np.empty((0, 3)),
        resolution=task._OBSTACLE_RESOLUTION,
    )


def plan_straight_segments(arm, target_tcp, max_distance=0.10):
    """Bound open-loop screw integration error, planning the entire phase first."""
    current_tcp = Pose.from_pose7(arm.get_ee_pose())
    count = max(1, int(np.ceil(np.linalg.norm(target_tcp.p - current_tcp.p) / max_distance)))
    if count > 12:
        return None
    qpos = arm.get_arm_qpos()
    plans = []
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        point = current_tcp.p + fraction * (target_tcp.p - current_tcp.p)
        target = tcp_to_link_pose(Pose(point, target_tcp.q), arm.tcp_offset)
        plan = arm.planner.plan_screw_path(qpos, target.to_pose7(), qpos_step=0.025)
        if (not plan.success or not plan.position.size
                or not _pose_matches(mplib_world_link_pose(arm.planner, plan.position[-1]), target)):
            return None
        plans.append(plan)
        qpos = plan.position[-1]
    return plans


def preload_seal_hold(task, pick, arm, arm_tag):
    """Archived fixed-preload diagnostic; not used by the live seal path.

    The initial0.600 command is24mm per Franka finger, not a force fraction.
    Retain the existing gains/force limits and ramp to21mm over300ms, then
    hold100ms. The existing arm replay persists the resulting0.525 command.
    Bilateral position residual is only a stall/preload proxy, not contact
    proof; the physical lift/carry/release gates still determine success.
    """
    initial_m, target_m, residual_min = 0.024, 0.021, 0.0005

    def reject(reason):
        print(f"[seal-hold] rejected: {reason}", flush=True)
        return False

    try:
        indices = tuple(getattr(arm, "_finger_dof_indices", ()))
    except TypeError:
        return reject("missing Franka finger DOFs")
    if (len(indices) != 2
            or any(isinstance(i, (bool, np.bool_))
                   or not isinstance(i, (int, np.integer)) or i < 0 for i in indices)
            or len(set(indices)) != 2):
        return reject("expected two distinct Franka finger DOFs")
    try:
        limits = np.asarray([arm.gripper_closed, arm.gripper_open, arm.gripper_val], dtype=float)
        dt = float(task.scene.sim.dt)
        if (not np.all(np.isfinite(limits)) or not np.isfinite(dt) or dt <= 0
                or not np.isclose(limits[0], 0.0, atol=1e-9, rtol=0)
                or not np.isclose(limits[1], 0.040, atol=1e-9, rtol=0)
                or not np.isclose(limits[2], 0.600, atol=1e-9, rtol=0)):
            return reject("expected settled0.600 acquisition and finite Franka range/clock")
        ramp_steps, hold_steps = max(1, int(np.ceil(.300 / dt))), max(1, int(np.ceil(.100 / dt)))
        if ramp_steps + hold_steps > 10000:
            return reject("preload clock exceeds bounded step budget")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return reject("invalid gripper range or clock")

    def read_state():
        fingers = np.asarray(to_numpy(arm.entity.get_dofs_position(dofs_idx_local=indices)), dtype=float)
        obj, tcp = pick.actor.get_pose(), Pose.from_pose7(arm.get_ee_pose())
        if (fingers.shape != (2,) or not np.all(np.isfinite(fingers))
                or np.any(fingers < -1e-6) or np.any(fingers > .040001)
                or not np.all(np.isfinite(obj.to_pose7()))
                or not np.all(np.isfinite(tcp.to_pose7()))):
            raise ValueError("nonfinite or invalid finger/object/TCP state")
        return fingers, obj, tcp

    try:
        before_fingers, before_obj, before_tcp = read_state()
    except (AttributeError, TypeError, ValueError, IndexError):
        return reject("initial finger/object/TCP state unavailable")
    if np.any(before_fingers - initial_m <= residual_min):
        return reject("initial bilateral stall/preload proxy absent; no tighter close")
    print(f"[seal-hold] preload start target_m={initial_m:.4f}->{target_m:.4f} "
          f"fingers={before_fingers.tolist()} ramp_steps={ramp_steps} "
          f"hold_steps={hold_steps} dt={dt}", flush=True)
    task.set_gripper(0.525, arm_tag, num_steps=ramp_steps)
    for _ in range(hold_steps):
        task.step_sim()
    try:
        after_fingers, after_obj, after_tcp = read_state()
    except (AttributeError, TypeError, ValueError, IndexError):
        return reject("post-preload finger/object/TCP state unavailable")
    shift = float(np.linalg.norm(after_obj.p - before_obj.p))
    angle = float(2 * np.arccos(np.clip(abs(np.dot(after_obj.q, before_obj.q)), 0, 1)))
    print(f"[seal-hold] preload end fingers={after_fingers.tolist()} "
          f"residual_mm={((after_fingers - target_m) * 1000).tolist()} "
          f"object_shift_m={shift:.6f} object_rotation_rad={angle:.6f}", flush=True)
    if np.any(after_fingers - target_m <= residual_min):
        return reject("post-preload bilateral stall/preload proxy absent")
    # Permit ordinary settling; reject a material shove/roll without changing
    # the configured grasp pose or attempting a correction beside the seal.
    if shift > .010 or not np.isfinite(angle) or angle > np.deg2rad(15):
        return reject("preload moved object beyond10mm/15degrees")
    if not _pose_matches(after_tcp, before_tcp):
        return reject("preload moved wrist beyond existing6mm/0.05rad gate")
    return True


class SealHoldRejected(RuntimeError):
    """Invalid feedback or lost purchase must stop held arm execution."""


def seal_hold_target(fingers, velocity, kp, kv, previous_target, elapsed, *, force_n=7.5):
    """Bounded position target, with20% of the existing damping retained.

    Genesis uses u = kp*(target-q) - kv*v. Before clipping, the proposal
    q - force_n/kp + .8*kv*v/kp gives u = -force_n - .2*kv*v.
    This is an actuator-model estimate, NOT contact-force feedback. Keep the
    velocity correction within2mm, target within18--24mm, and slew50mm/s.
    The nominal setting is bounded to30N; this is not a physical force cap.
    """
    if (isinstance(force_n, (bool, np.bool_))
            or not isinstance(force_n, (int, float, np.integer, np.floating))
            or not 0 < force_n <= 30 or not np.isfinite(force_n)):
        raise SealHoldRejected("nominal seal hold force must be a finite scalar in (0,30] N")
    values = [np.asarray(value, dtype=float) for value in (fingers, velocity, kp, kv)]
    if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in values):
        raise SealHoldRejected("expected finite two-finger q/velocity/kp/kv")
    q, v, kp, kv = values
    if (np.any(q < 0) or np.any(q > .040001) or abs(q[0] - q[1]) > .002
            or np.any(np.abs(v) > .050) or np.any(kp <= 0) or np.any(kv < 0)):
        raise SealHoldRejected("finger range/asymmetry/velocity/gains guard")
    if (not np.isfinite(previous_target) or not .018 <= previous_target <= .024
            or not np.isfinite(elapsed) or elapsed <= 0 or elapsed > .040):
        raise SealHoldRejected("invalid hold target or update interval")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        compensation = np.clip(.8 * kv * v / kp, -.002, .002)
        proposals = q - float(force_n) / kp + compensation
        desired = np.clip(float(np.mean(proposals)), .018, .024)
        target = float(np.clip(desired, previous_target - .050 * elapsed,
                               previous_target + .050 * elapsed))
        effort = kp * (target - q) - kv * v
    if (not np.all(np.isfinite(proposals)) or not np.isfinite(target)
            or not np.all(np.isfinite(effort))):
        raise SealHoldRejected("nonfinite bounded target or predicted actuator effort")
    if np.any(q <= target):
        raise SealHoldRejected("bounded target lost bilateral spring preload")
    return target, effort


class SealHoldController:
    """Task-local bounded PD position feedback; no simulator or gain writes.

    command() is called just before an EXISTING physical tick, either by the
    normal execute_plan gripper sequence or a normal settle hold. No extra
    steps, arm waypoints, position teleports, or force commands are inserted.
    """

    def __init__(self, task, pick, arm, arm_tag):
        self.task, self.pick, self.arm, self.arm_tag = task, pick, arm, arm_tag
        self.force_n = task.SEAL_HOLD_FORCE_N
        try:
            self.indices = tuple(arm._finger_dof_indices)
            limits = np.asarray([arm.gripper_closed, arm.gripper_open, arm.gripper_val], dtype=float)
            dt = float(task.scene.sim.dt)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise SealHoldRejected("missing Franka acquisition state") from exc
        if (len(self.indices) != 2
                or any(isinstance(i, (bool, np.bool_))
                       or not isinstance(i, (int, np.integer)) or i < 0 for i in self.indices)
                or len(set(self.indices)) != 2):
            raise SealHoldRejected("expected two distinct Franka finger DOFs")
        if (not np.all(np.isfinite(limits))
                or not np.allclose(limits, [0., .040, .600], atol=1e-9, rtol=0)
                or not np.isfinite(dt) or not 0 < dt <= .005):
            raise SealHoldRejected("expected settled0.600 Franka acquisition and finite clock")
        self.update_steps = int(np.ceil(.005 / dt))
        if self.update_steps > 10000:
            raise SealHoldRejected("hold clock exceeds bounded update budget")
        self.update_dt = self.update_steps * dt
        self.ticks = self.updates = 0
        self.target_m = .024
        q, v, kp, kv, relative = self._read_state()
        seal_hold_target(q, v, kp, kv, self.target_m, self.update_dt, force_n=self.force_n)
        if np.any(q - self.target_m <= .0005):
            raise SealHoldRejected("initial bilateral stall/preload proxy absent")
        self.initial_relative = relative
        self.predicted_effort = kp * (self.target_m - q) - kv * v
        print(f"[seal-hold] adaptive start fingers={q.tolist()} target_m={self.target_m} "
              f"update_dt={self.update_dt} nominal_spring_N={self.force_n} beta=0.8 "
              "bounds_m=[0.018,0.024] correction_cap_m=0.002 slew_m_s=0.050", flush=True)

    def _read_state(self):
        try:
            entity = self.arm.entity
            arrays = [np.asarray(to_numpy(getattr(entity, method)(dofs_idx_local=self.indices)), dtype=float)
                      for method in ("get_dofs_position", "get_dofs_velocity", "get_dofs_kp", "get_dofs_kv")]
            obj = self.pick.actor.get_pose()
            tcp = Pose.from_pose7(self.arm.get_ee_pose())
            if not np.all(np.isfinite(np.concatenate([obj.to_pose7(), tcp.to_pose7()]))):
                raise ValueError("nonfinite object/TCP pose")
            relative = tcp.inv() * obj
            return (*arrays, relative)
        except (AttributeError, TypeError, ValueError, IndexError, RuntimeError, OverflowError) as exc:
            raise SealHoldRejected("finger/object/TCP feedback unavailable") from exc

    def command(self):
        if self.ticks >= self.update_steps:
            q, v, kp, kv, relative = self._read_state()
            if not _pose_matches(relative, self.initial_relative, .020, np.deg2rad(20)):
                raise SealHoldRejected("object/TCP offset or rotation exceeded20mm/20degrees")
            target, effort = seal_hold_target(
                q, v, kp, kv, self.target_m, self.update_dt, force_n=self.force_n,
            )
            self.target_m, self.predicted_effort = target, effort
            self.ticks = 0
            self.updates += 1
            print(f"[seal-hold] adaptive update={self.updates} target_m={target:.7f} "
                  f"fingers={q.tolist()} velocity={v.tolist()} "
                  f"predicted_actuator_effort_N={effort.tolist()}", flush=True)
        self.ticks += 1
        return self.target_m / self.arm.gripper_open


class SealHoldSequence:
    """Lazy per-tick feedback for the existing BaseTask execution loop."""

    def __init__(self, controller, length):
        self.controller, self.length = controller, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.length:
            raise IndexError(index)
        return self.controller.command()


def execute_seal_hold_plan(task, delegate, plan, arm_tag, gripper_result, controller):
    """Insert only the scoped lazy gripper input; preserve normal arm replay."""
    if (arm_tag != controller.arm_tag or gripper_result is not None
            or getattr(task, "_seal_selector_execution", None) is not None):
        raise SealHoldRejected("unexpected arm/gripper/selector motion during seal hold")
    if plan is None:
        return delegate(plan, arm_tag, gripper_result=None)
    if not plan.success or plan.position.ndim != 2 or not np.all(np.isfinite(plan.position)):
        raise SealHoldRejected("invalid held trajectory")
    return delegate(plan, arm_tag, gripper_result=SealHoldSequence(controller, len(plan.position)))


@contextmanager
def seal_hold_scope(task, controller):
    if (getattr(task, "_seal_hold_controller", None) is not None
            or getattr(task, "_seal_selector_execution", None) is not None):
        raise SealHoldRejected("nested seal hold/selector")
    task._seal_hold_controller = controller
    try:
        yield controller
    finally:
        task._seal_hold_controller = None


def settle_seal_hold(task, arm_tag, steps):
    """Maintain feedback during existing held settles without adding ticks."""
    for _ in range(steps):
        controller = getattr(task, "_seal_hold_controller", None)
        if controller is not None:
            if arm_tag != controller.arm_tag:
                raise SealHoldRejected("unexpected settling arm during seal hold")
            task.robot.set_gripper(controller.command(), arm_tag)
        task.step_sim()


def pick_and_place_seal(task, pick, place, arm_tag):
    """Bounded model1 pickup, straight lift/carry and verified physical release."""
    arm = task.robot.get_arm(arm_tag)
    model_id = next((m for a, name, m, _ in task._sort_targets
                     if a is pick.actor and name == "100_seal"), None)
    if pick.actor is None or model_id != 1:
        return False
    scale = task._get_object_scale("100_seal", model_id)
    safe_z = float(task.TABLE_TOP_Z + 0.38)

    def settle(steps):
        settle_seal_hold(task, arm_tag, steps)

    def fail(reason):
        print(f"[seal-grasp] failed: {reason}", flush=True)
        task._seal_hold_controller = None
        task.open_gripper(arm_tag)
        settle(120)
        return False

    task.open_gripper(arm_tag)
    settle(80)
    bounds = np.asarray(to_numpy(pick.actor.entity.get_AABB()), dtype=float)
    object_pose = pick.actor.get_pose()
    try:
        sweep = gripper_sweep_radius(arm)
        # Enough room for the entire opened hand to turn, without asking the
        # coarse open-loop screw integrator to cover34cm in one descent.
        approach_z = float(bounds[1, 2] + sweep + 0.03)
        candidates = seal_candidates(object_pose, bounds, model_id, scale, approach_z)
    except ValueError as exc:
        return fail(str(exc))
    _refresh_obstacles(task, pick, arm_tag)
    print(f"[seal-grasp] bounds={bounds.tolist()} sweep_radius={sweep:.4f} "
          f"pre_z={approach_z:.4f} grasp_z={bounds[1, 2] - 0.012:.4f}", flush=True)
    try:
        result = select_seal_with_timed_execution(
            task, arm_tag, object_name="100_seal", object_pose=object_pose,
            model_id=model_id, object_scale=scale,
            candidate_grasps=candidates, max_candidates=2, plan_attempts=2,
            screw_qpos_step=0.025,
            post_plan_check=seal_preflight_check(arm, bounds, sweep),
        )
    except SealApproachRejected as exc:
        return fail(str(exc))
    if result is None:
        return fail("no clear fixed-orientation descent")
    grasp_link, _, _ = result
    target_tcp = grasp_link * arm.tcp_offset
    settle(100)
    actual_tcp = Pose.from_pose7(arm.get_ee_pose())
    print(f"[seal-grasp] arrival planned={target_tcp.to_pose7().tolist()} "
          f"actual={actual_tcp.to_pose7().tolist()}", flush=True)
    current_bounds = np.asarray(to_numpy(pick.actor.entity.get_AABB()), dtype=float)
    if (not _pose_matches(actual_tcp, target_tcp)
            or np.linalg.norm((current_bounds.mean(axis=0) - bounds.mean(axis=0))[:2]) > 0.008):
        return fail("grasp arrival or object moved; no low lateral correction")
    close_m = min(0.024, float(np.min(bounds[1, :2] - bounds[0, :2]) / 2 - 0.004))
    task.set_gripper(max(0.010, close_m) / arm.gripper_open, arm_tag)
    settle(300)
    start_center = np.asarray(pick.get_center()).copy()

    def straight(point, label):
        _refresh_obstacles(task, pick, arm_tag)
        target = Pose(point, target_tcp.q)
        plans = plan_straight_segments(arm, target)
        if plans is None:
            print(f"[seal-grasp] {label} planning rejected (no RRT fallback)", flush=True)
            return False
        for plan in plans:
            task._slow_replay(plan.position, plan.velocity, arm_tag)
        settle(100)
        actual = Pose.from_pose7(arm.get_ee_pose())
        print(f"[seal-grasp] {label}: tcp={actual.p} object={pick.get_center()}", flush=True)
        return _pose_matches(actual, target, position_tol=0.012, angle_tol=0.08)

    try:
        controller = SealHoldController(task, pick, arm, arm_tag)
        with seal_hold_scope(task, controller):
            lifted_tcp = target_tcp.p.copy()
            lifted_tcp[2] = safe_z
            if not straight(lifted_tcp, "lift"):
                return fail("lift failed")
            lifted_center = np.asarray(pick.get_center())
            lift_dz = float(lifted_center[2] - start_center[2])
            print(f"[seal-grasp] measured lift_dz={lift_dz:.4f}", flush=True)
            if lift_dz < 0.04:
                return fail("object did not lift")
            held_offset = lifted_center - np.asarray(arm.get_ee_pose()[:3])
            carry = np.asarray(place.pos, dtype=float).copy()
            carry[:2] -= held_offset[:2]
            carry[2] = safe_z
            if not straight(carry, "carry"):
                return fail("carry failed")
            carried_center = np.asarray(pick.get_center())
            if (carried_center[2] - start_center[2] < 0.04
                    or np.linalg.norm(carried_center[:2] - np.asarray(place.pos)[:2]) > 0.025):
                return fail("object lost before release")
    except SealHoldRejected as exc:
        return fail(str(exc))
    # Feedback is out of scope before opening: never re-close during release.
    task.open_gripper(arm_tag)
    return settle_released_seal(task, pick, place, arm, carried_center)


def released_in_basket(carried_center, released_center, destination):
    """Release requires a real drop, not just a held object's correct XY."""
    carried_center = np.asarray(carried_center, dtype=float)
    released_center = np.asarray(released_center, dtype=float)
    destination = np.asarray(destination, dtype=float)
    return bool(np.all(np.isfinite([carried_center, released_center, destination]))
                and carried_center[2] - released_center[2] >= 0.04
                and released_center[2] >= destination[2] - 0.056
                and np.linalg.norm(released_center[:2] - destination[:2]) <= 0.10)


def release_supported(bounds, destination, table_z, half_xy=0.088):
    """Match the task's primitive basket floor, with no evaluator changes."""
    bounds, destination = np.asarray(bounds), np.asarray(destination)
    if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
            or not np.all(bounds[1] > bounds[0]) or not np.all(np.isfinite(destination))):
        return False
    # Both categorize layouts construct their floor with half-thickness5mm
    # at TABLE_TOP_Z+5mm, hence its physical top is TABLE_TOP_Z+10mm.
    floor_z = float(table_z + 0.010)
    return bool(abs(destination[2] - (table_z + 0.056)) <= 0.005
                and floor_z - 0.010 <= bounds[0, 2] <= floor_z + 0.015
                and np.all(bounds[0, :2] >= destination[:2] - half_xy)
                and np.all(bounds[1, :2] <= destination[:2] + half_xy))


def settle_released_seal(task, pick, place, arm, carried_center):
    """Require support and100ms of quiet real physics, bounded to1.5seconds."""
    dt = float(task.scene.sim.dt)
    sample_steps = max(1, int(round(0.020 / dt)))
    sample_dt = sample_steps * dt
    previous = pick.actor.get_pose()
    stable = 0
    for step in range(sample_steps, int(np.ceil(1.5 / dt)) + 1, sample_steps):
        for _ in range(sample_steps):
            task.step_sim()
        current = pick.actor.get_pose()
        bounds = np.asarray(to_numpy(pick.actor.entity.get_AABB()), dtype=float)
        center = np.asarray(pick.get_center(), dtype=float)
        tcp = np.asarray(arm.get_ee_pose(), dtype=float)
        if not np.all(np.isfinite(np.concatenate([current.to_pose7(), previous.to_pose7(), tcp, center]))):
            break
        speed = float(np.linalg.norm(current.p - previous.p) / sample_dt)
        angular = float(2 * np.arccos(np.clip(abs(np.dot(current.q, previous.q)), 0, 1)) / sample_dt)
        supported = release_supported(bounds, place.pos, task.TABLE_TOP_Z)
        clear_hand = bool(tcp[2] >= bounds[1, 2] + 0.05 and arm.gripper_val >= 0.99)
        quiet = speed < 0.03 and angular < 0.3
        stable = stable + 1 if supported and clear_hand and quiet else 0
        previous = current
        if stable >= 5 and released_in_basket(carried_center, center, place.pos):
            print(f"[seal-grasp] released-supported-settled: steps={step} "
                  f"center={center} bottom_z={bounds[0, 2]:.4f} "
                  f"speed={speed:.4f} angular_speed={angular:.4f}", flush=True)
            return True
    print("[seal-grasp] release not supported and settled within1.5s", flush=True)
    return False
