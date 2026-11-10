"""Native-only, task-local geometry gates for the opt-in oil carry candidate.

The caller supplies freshly measured full joints, held geometry and ALL scene
fixtures after refreshing the existing avatar cloud. This module never steps
physics, creates another planner, changes contact physics, or resets RNG. Use
a NEW context after every physical arrival or obstacle refresh. Its static,
discretely checked geometry is not a retention or dynamic-safety certificate.
"""
import os

import numpy as np


MARGIN = .002
HOVER_M = .13
_ARM = [f"panda_joint{i}" for i in range(1, 8)]
_FINGERS = ["panda_finger_joint1", "panda_finger_joint2"]
_LINKS = {f"panda_link{i}" for i in range(8)} | {
    "panda_hand", "panda_leftfinger", "panda_rightfinger"}
_TOUCH = {"panda_hand", "panda_leftfinger", "panda_rightfinger"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all(), f"invalid {name}")
    return result.copy()


def pose_matrix(value):
    """Validated world/local transform, from either wxyz pose7 or matrix4x4."""
    from transforms3d.quaternions import quat2mat

    value = np.asarray(value, dtype=float)
    if value.shape == (7,):
        _array(value, (7,), "pose7")
        _require(abs(np.linalg.norm(value[3:])-1.) < 1e-4, "nonunit quaternion")
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = quat2mat(value[3:]), value[:3]
    else:
        result = _array(value, (4, 4), "pose matrix")
    rotation = result[:3, :3]
    _require(np.allclose(result[3], [0., 0., 0., 1.], atol=1e-9, rtol=0) and
             np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) and
             abs(np.linalg.det(rotation)-1.) < 1e-7, "invalid rigid transform")
    return result.copy()


def pose7(value):
    from transforms3d.quaternions import mat2quat

    matrix = pose_matrix(value)
    return np.r_[matrix[:3, 3], mat2quat(matrix[:3, :3])]


def pose_error(actual, target):
    actual, target = pose_matrix(actual), pose_matrix(target)
    return {"position_m": float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            "orientation_rad": float(np.arccos(np.clip(
                (np.trace(actual[:3, :3].T @ target[:3, :3])-1.)/2., -1., 1.)))}


def upright_target(xy, origin_z):
    """180-degree world-Z yaw of the original local+Y-up bottle; same support."""
    target = np.eye(4)
    target[:3, :3] = [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]
    target[:3, 3] = [*_array(xy, (2,), "target XY"), float(origin_z)]
    return pose_matrix(target)


def dense_path(result, start_arm, goal_arm=None):
    """Read native dict or existing PlanResult; no arbitrary joint wrapping."""
    if isinstance(result, dict):
        success = result.get("status") == "Success"
        position, velocity = result.get("position"), result.get("velocity")
    else:
        success = result is not None and result.success
        position, velocity = getattr(result, "position", None), getattr(result, "velocity", None)
    _require(success, "unsuccessful native plan")
    position = np.asarray(position, dtype=float)
    _require(position.ndim == 2 and position.shape[1] == 7 and 2 <= len(position) <= 10000,
             "invalid/oversized path")
    position = _array(position, position.shape, "path positions")
    _array(velocity, position.shape, "path velocities")
    start = _array(start_arm, (7,), "path start")
    _require(np.allclose(position[0], start, atol=1e-6, rtol=0), "path changed measured start")
    if goal_arm is not None:
        goal = _array(goal_arm, (7,), "path goal")
        _require(np.allclose(position[-1], goal, atol=1e-4, rtol=0), "path changed selected branch")
    steps = np.abs(np.diff(position, axis=0))
    _require(np.max(steps) <= .10 and np.max(steps.sum(axis=0)) <= 6., "path branch/travel bound")
    samples = [position[0]]
    for first, last in zip(position, position[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(last-first))/.01)))
        _require(len(samples)+count <= 20000, "oversized densified path")
        samples.extend((1-a)*first+a*last for a in np.linspace(0., 1., count+1)[1:])
    return np.asarray(samples), {"returned_waypoints": len(position), "checked_arm_samples": len(samples),
        "max_joint_step_rad": float(np.max(steps)), "joint_travel_rad": steps.sum(axis=0).tolist(),
        "interpolation_step_rad": .01, "continuous_collision_checked": False}


def check_oil_lift_path(arm, result, measured_full, finger_targets, target_link):
    """Check the native screw from its exact measured start; restore FK state.

    Keep all existing native world objects/ACM. Check every returned waypoint
    plus <=0.01 rad interpolation, for measured and both commanded finger
    states. No simulator qpos is written and no path can be repaired here.
    """
    native = arm.planner._mplib
    names = list(native.robot.get_user_joint_names())
    move = np.asarray(native.move_group_joint_indices, dtype=int)
    fingers = [names.index(n) for n in ("panda_finger_joint1", "panda_finger_joint2")]
    full = _array(measured_full, (9,), "measured full qpos")
    if move.shape != (7,) or [names[i] for i in move] != [f"panda_joint{i}" for i in range(1, 8)]:
        raise ValueError("unsupported clearance-lift joint mapping")
    samples, report = dense_path(result, full[move])
    if len(result.position) > 800 or max(report["joint_travel_rad"]) > 1.25:
        raise ValueError("clearance lift exceeds local branch/duration bound")
    if "scene_pcd" not in native.planning_world.get_object_names():
        raise ValueError("clearance lift has no current obstacle cloud")
    saved = _array(native.robot.get_qpos(), (9,), "saved native qpos")
    limits = _array(native.joint_limits, (9, 2), "native limits")
    base = native.robot.get_base_pose()
    base = pose_matrix(np.r_[base.p, base.q])
    configurations = [full[fingers], *[finger_targets[k] for k in ("cached", "scalar")]]
    try:
        for finger_q in configurations:
            for q in samples:
                state = full.copy()
                state[move], state[fingers] = q, _array(finger_q, (2,), "finger state")
                if np.any(state < limits[:, 0] - .001) or np.any(state > limits[:, 1] + .001):
                    raise ValueError("clearance lift exceeds native joint limits")
                native.robot.set_qpos(state, True)
                native.planning_world.set_qpos_all(state[move])
                if not np.allclose(native.robot.get_qpos(), state, atol=1e-10, rtol=0):
                    raise ValueError("native world changed measured finger state")
                collisions = native.planning_world.check_collision()
                if collisions:
                    raise ValueError("clearance lift collides with native world/self geometry")
        native.pinocchio_model.compute_forward_kinematics(state)
        link = native.pinocchio_model.get_link_pose(native.link_name_2_idx[native.move_group])
        error = pose_error(base @ pose_matrix(np.r_[link.p, link.q]), target_link)
        if error["position_m"] > .005 or error["orientation_rad"] > .01:
            raise ValueError("clearance-lift FK endpoint error: " + str(error))
        return dict(report, endpoint_error=error, checked_finger_states=len(configurations))
    finally:
        native.robot.set_qpos(saved, True)
        native.planning_world.set_qpos_all(saved[move])
        native.robot.set_qpos(saved, True)
        native.pinocchio_model.compute_forward_kinematics(saved)


class OilRecoveryPlan:
    """Temporarily extend an existing native planner; restore it on every exit.

    ``measured_qpos`` is full9 in native user-joint order; ``finger_targets``
    must contain both ``cached`` and ``scalar`` two-finger targets. Pose arrays
    are wxyz pose7 (or rigid4x4). ``payload_bounds_local`` is ALREADY scaled.
    ``fixture_aabbs`` must include table/cabinet/cooktop/vinegar/soy_sauce AND
    every actual non-target kitchen/backdrop/cooktop part. No fixture whitelist
    is applied. Legacy cooktop proxies do not replace actual stove geometry.
    ``held=False`` disables only the carried payload proxy for empty retreat;
    the caller must then include the released bottle in fixture_aabbs.
    """
    def __init__(self, native, *, measured_qpos, finger_targets, gripper_open,
                 world_base_pose7, hand_tcp_pose7, tcp_payload_pose7,
                 payload_bounds_local, fixture_aabbs, held=True):
        import mplib
        from mplib.pymp import Pose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject, FCLObject, distance

        _require(type(native) is mplib.Planner and native.move_group == "panda_hand",
                 "native Franka MPLib required")
        _require(not any(os.environ.get(name, "").strip() == "1" for name in
                         ("DEBUG_BOUNDED_JOINT_PATHS", "DEBUG_TELEPORT", "USE_SETQPOS")),
                 "debug execution/planner bypass enabled")
        self.native, self.Pose, self.Box = native, Pose, Box
        self.CollisionObject, self.FCLObject, self.distance = CollisionObject, FCLObject, distance
        self.robot, self.world, self.pin = native.robot, native.planning_world, native.pinocchio_model
        names = list(self.robot.get_user_joint_names())
        _require(len(names) == 9 and set(names) == set(_ARM+_FINGERS), "unsupported full joint names")
        move = np.asarray(native.move_group_joint_indices)
        _require(move.shape == (7,) and np.issubdtype(move.dtype, np.integer) and
                 len(set(move)) == 7 and np.all((move >= 0) & (move < 9)) and
                 [names[i] for i in move] == _ARM, "unsupported move-group mapping")
        self.move, self.fingers = move.copy(), np.array([names.index(name) for name in _FINGERS])
        self.ik_mask = np.isin(np.arange(9), self.fingers)
        self.saved = _array(self.robot.get_qpos(), (9,), "saved native qpos")
        self.measured = _array(measured_qpos, (9,), "measured full qpos")
        self.start_arm = self.measured[self.move].copy()
        self.limits = _array(native.joint_limits, (9, 2), "joint limits")
        _require(np.all(self.limits[:, 1] > self.limits[:, 0]), "inverted joint limits")
        _require(set(finger_targets) == {"cached", "scalar"}, "both finger command sources required")
        self.finger_configs = [("measured", self.measured[self.fingers].copy())] + [
            (key, _array(finger_targets[key], (2,), key+" fingers")) for key in ("cached", "scalar")]
        self.aperture = float(gripper_open)
        _require(np.isfinite(self.aperture) and 0 <= self.aperture <= .04, "invalid open aperture")
        self.base, self.hand_tcp = pose_matrix(world_base_pose7), pose_matrix(hand_tcp_pose7)
        self.tcp_payload = pose_matrix(tcp_payload_pose7)
        native_base = self.robot.get_base_pose()
        _require(np.allclose(self.base, pose_matrix(np.r_[native_base.p, native_base.q]), atol=1e-9, rtol=0),
                 "measured/native base disagreement")
        self.payload_bounds = self._bounds(payload_bounds_local, "scaled payload")
        _require(type(held) is bool, "held must be explicit bool")
        self.held = held
        _require({"table", "cabinet", "cooktop", "vinegar", "soy_sauce"} <= set(fixture_aabbs),
                 "missing required fixture")
        _require(held or "released_bottle" in fixture_aabbs, "empty retreat requires released_bottle fixture")
        _require(all(isinstance(name, str) and name and not name.startswith("oil_recovery_")
                     for name in fixture_aabbs), "invalid fixture name")
        self.fixtures = [(name, self._box("oil_recovery_"+name, self._bounds(bounds, name)))
                         for name, bounds in sorted(fixture_aabbs.items())]
        self.old_names = sorted(self.world.get_object_names())
        _require("scene_pcd" in self.old_names, "original avatar scene cloud missing")
        self.original_objects = [self.world.get_object(name) for name in self.old_names]
        _require(all(obj.shapes for obj in self.original_objects), "empty existing world geometry")
        self.old_acm = str(self.world.get_allowed_collision_matrix())
        planned = list(self.world.get_planned_articulations())
        _require(len(planned) == 1 and planned[0].get_name() == self.robot.get_name(),
                 "unsupported planning-world articulation layout")
        _require(list(self.world.get_articulation_names()) == [self.robot.get_name()],
                 "unmodeled articulated obstacle in planning world")
        objects = list(self.robot.get_fcl_model().get_collision_objects())
        self.link_names = {obj.name for obj in objects}
        _require(_LINKS <= self.link_names and len(objects) == len(self.link_names) and
                 self.link_names == set(self.robot.get_fcl_model().get_collision_link_names()) and
                 all(obj.shapes for obj in objects), "incomplete robot collision geometry")
        _require(not list(self.pin.get_chain_joint_name("panda_link0")), "table-exempt link is not fixed root")
        self.owned, self.active, self.nearest, self.checked = [], False, {}, 0
        self.used = False

    @staticmethod
    def _bounds(value, name):
        result = _array(value, (2, 3), name+" bounds")
        _require(np.all(result[1] > result[0]), "empty/inverted "+name+" bounds")
        return result

    def _box(self, name, bounds, matrix=None):
        transform = np.eye(4) if matrix is None else matrix.copy()
        transform[:3, 3] += transform[:3, :3] @ bounds.mean(axis=0)
        identity = self.Pose()
        return self.FCLObject(name, self.Pose(transform),
            [self.CollisionObject(self.Box(*(bounds[1]-bounds[0])), identity)], [identity])

    def __enter__(self):
        _require(not self.used, "construct a fresh context after each physical arrival")
        self.used = True
        try:
            for name, obj in self.fixtures:
                if name == "table":
                    continue  # Direct distance query exempts the fixed mounting root only.
                _require(obj.name not in self.world.get_object_names(), "fixture name conflict: "+obj.name)
                self.world.add_object(obj)
                self.owned.append(obj.name)
            _require(str(self.world.get_allowed_collision_matrix()) == self.old_acm, "ACM changed on add")
            self.active = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        cleanup_errors = []
        try:
            for name in reversed(self.owned):
                try:
                    _require(self.world.remove_object(name), "fixture cleanup failed: "+name)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            self.owned.clear()
        finally:
            try:
                self.robot.set_qpos(self.saved.copy(), True)
                self.world.set_qpos_all(self.saved[self.move])
            finally:
                self.robot.set_qpos(self.saved.copy(), True)
                self.pin.compute_forward_kinematics(self.saved.copy())
                self.active = False
        _require(not cleanup_errors, "fixture cleanup failed: "+"; ".join(cleanup_errors))
        _require(sorted(self.world.get_object_names()) == self.old_names, "original world membership changed")
        _require(str(self.world.get_allowed_collision_matrix()) == self.old_acm, "original ACM changed")
        _require(np.array_equal(self.robot.get_qpos(), self.saved), "native full state not restored")

    def full_qpos(self, arm, fingers=None):
        full = self.measured.copy()
        full[self.move] = _array(arm, (7,), "arm qpos")
        if fingers is not None:
            full[self.fingers] = _array(fingers, (2,), "finger qpos")
        return full

    def link_target(self, bottle_pose):
        return pose_matrix(bottle_pose) @ np.linalg.inv(self.tcp_payload) @ np.linalg.inv(self.hand_tcp)

    def _finger_states(self, opening):
        for name, fingers in self.finger_configs:
            yield name, fingers
            if opening:
                count = max(1, int(np.ceil(np.max(np.abs(self.aperture-fingers))/.002)))
                for alpha in np.linspace(0., 1., count+1)[1:]:
                    yield "opening_"+name, (1-alpha)*fingers+alpha*self.aperture

    def _gap(self, first, second, label):
        gap = float(self.distance(first, second).min_distance)
        _require(np.isfinite(gap), "nonfinite distance: "+label)
        self.nearest[label] = min(self.nearest.get(label, gap), gap)
        _require(gap >= MARGIN, f"{label}: {gap:.9g}m < {MARGIN}m")

    def check_state(self, arm_qpos=None, *, opening=False, target_bottle=None, target_link=None):
        _require(self.active, "geometry context is not active")
        arm = self.start_arm if arm_qpos is None else _array(arm_qpos, (7,), "arm qpos")
        for label, fingers in self._finger_states(opening):
            full = self.full_qpos(arm, fingers)
            _require(np.all(full >= self.limits[:, 0]-.001) and
                     np.all(full <= self.limits[:, 1]+.001), "joint limit: "+label)
            self.robot.set_qpos(full, True)
            self.world.set_qpos_all(full[self.move])
            _require(np.allclose(self.robot.get_qpos(), full, atol=1e-10, rtol=0), "world changed fingers")
            collisions = self.world.check_collision()
            if collisions:
                pairs = [f"{c.object_name1}/{c.link_name1} vs {c.object_name2}/{c.link_name2}" for c in collisions]
                raise ValueError("world/self "+label+": "+"; ".join(pairs))
            self.pin.compute_forward_kinematics(full)
            hand = self.pin.get_link_pose(self.native.link_name_2_idx[self.native.move_group])
            link = self.base @ pose_matrix(np.r_[hand.p, hand.q])
            tcp, bottle = link @ self.hand_tcp, link @ self.hand_tcp @ self.tcp_payload
            for actual, target in ((link, target_link), (bottle, target_bottle)):
                if target is not None:
                    error = pose_error(actual, target)
                    _require(error["position_m"] <= .002 and error["orientation_rad"] <= .01,
                             "FK target error: "+str(error))
            objects = list(self.robot.get_fcl_model().get_collision_objects())
            _require({obj.name for obj in objects} == self.link_names and all(obj.shapes for obj in objects),
                     "robot collision geometry changed")
            payload = self._box("oil_recovery_held_payload", self.payload_bounds, bottle) if self.held else None
            for name, fixture in self.fixtures:
                if payload is not None:
                    self._gap(payload, fixture, "payload/"+name)
                for obj in objects:
                    if obj.name != "panda_link0" or name != "table":
                        self._gap(obj, fixture, obj.name+"/"+name)
            for fixture in self.original_objects:
                if payload is not None:
                    self._gap(payload, fixture, "payload/"+fixture.name)
                for obj in objects:
                    self._gap(obj, fixture, obj.name+"/"+fixture.name)
            if payload is not None:
                for obj in objects:
                    if obj.name not in _TOUCH:
                        self._gap(payload, obj, "payload/"+obj.name)
            self.checked += 1
        return {"link_pose7": pose7(link), "tcp_pose7": pose7(tcp), "bottle_pose7": pose7(bottle)}

    def diagnostics(self):
        return {"checked_finger_states": self.checked, "min_distances_m": self.nearest.copy(),
                "continuous_collision_checked": False, "table_exempt_link": "panda_link0",
                "payload_touch_links": sorted(_TOUCH), "held": self.held}

    def check_path(self, result, start_arm7, *, goal_arm=None, target_bottle=None,
                   target_link=None, opening=False, descent=False):
        samples, report = dense_path(result, start_arm7, goal_arm)
        low = pose_matrix(target_bottle) if descent else None
        for arm in samples:
            state = self.check_state(arm)
            if low is not None:
                actual = pose_matrix(state["bottle_pose7"])
                _require(np.linalg.norm(actual[:2, 3]-low[:2, 3]) <= .002 and
                         low[2, 3]-.002 <= actual[2, 3] <= low[2, 3]+HOVER_M+.002 and
                         pose_error(actual, low)["orientation_rad"] <= .01, "descent left upright corridor")
        self.check_state(samples[-1], opening=opening, target_bottle=target_bottle, target_link=target_link)
        return dict(report, **self.diagnostics())

    def _ik(self, bottle_target, start_arm, trials, closest):
        goal = self.Pose(np.linalg.inv(self.base) @ self.link_target(bottle_target))
        status, solutions = self.native.IK(goal, self.full_qpos(start_arm), mask=self.ik_mask,
            n_init_qpos=trials, threshold=1e-4, return_closest=closest)
        _require(status == "Success" and solutions is not None, "native IK: "+str(status))
        return solutions

    def select_branch(self, target_xy, origin_z):
        _require(self.held, "branch selection requires held payload")
        self.check_state()  # Never rescue an invalid measured start by planning elsewhere.
        low = upright_target(target_xy, origin_z)
        solutions = self._ik(low, self.start_arm, 32, False)
        ordered = sorted(solutions, key=lambda full: float(np.linalg.norm(
            _array(full, (9,), "IK solution")[self.move]-self.start_arm)))
        rejected = []
        for full in ordered[:8]:
            low_q = _array(full, (9,), "IK solution")[self.move]
            previous, previous_target = low_q.copy(), low.copy()
            try:
                self.check_state(low_q, opening=True, target_bottle=low)
                for dz in np.linspace(0., HOVER_M, 14)[1:]:
                    target = low.copy()
                    target[2, 3] += dz
                    solution = _array(self._ik(target, previous, 1, True), (9,), "corridor IK")
                    arm = solution[self.move]
                    count = max(1, int(np.ceil(np.max(np.abs(arm-previous))/.01)))
                    _require(count <= 16, "corridor branch jump >0.16rad")
                    for alpha in np.linspace(0., 1., count+1)[1:]:
                        between = previous_target.copy()
                        between[:3, 3] = (1-alpha)*previous_target[:3, 3]+alpha*target[:3, 3]
                        self.check_state((1-alpha)*previous+alpha*arm, target_bottle=between)
                    previous, previous_target = arm.copy(), target
                return {"yaw_degrees": 180, "low_qpos": low_q.tolist(), "hover_qpos": previous.tolist(),
                        "low_pose7": pose7(low).tolist(), "hover_pose7": pose7(previous_target).tolist(),
                        "held_pose7": pose7(self.tcp_payload).tolist(), "rejected_branches": rejected,
                        "diagnostics": self.diagnostics()}
            except (ValueError, RuntimeError) as exc:
                rejected.append(str(exc))
        raise ValueError("no checked180degree branch: "+"; ".join(rejected))

    def _branch(self, branch):
        _require(branch["yaw_degrees"] == 180 and self.held, "unsupported placement branch")
        _require(np.allclose(pose_matrix(branch["held_pose7"]), self.tcp_payload, atol=1e-8, rtol=0),
                 "branch held transform changed")
        low, hover = pose_matrix(branch["low_pose7"]), pose_matrix(branch["hover_pose7"])
        expected = low.copy()
        expected[2, 3] += HOVER_M
        _require(np.allclose(expected, hover, atol=1e-8, rtol=0), "branch hover target changed")
        return low, hover, _array(branch["low_qpos"], (7,), "low branch"), _array(branch["hover_qpos"], (7,), "hover branch")

    @staticmethod
    def _time_step(value):
        _require(np.isfinite(value) and 1/250 <= value <= .02, "unsupported native time_step")
        return float(value)

    def plan_to_hover(self, branch, *, time_step=1/250):
        low, hover, low_q, hover_q = self._branch(branch)
        self.check_state()
        self.check_state(low_q, opening=True, target_bottle=low)
        self.check_state(hover_q, target_bottle=hover)
        rejected = []
        for _ in range(3):
            try:
                result = self.native.plan_qpos(goal_qposes=[self.full_qpos(hover_q)],
                    current_qpos=self.measured.copy(), time_step=self._time_step(time_step),
                    rrt_range=.1, planning_time=2., fix_joint_limits=False, simplify=True, verbose=False)
                report = self.check_path(result, self.start_arm, goal_arm=hover_q, target_bottle=hover)
                report["rejected_attempts"] = rejected
                return result, report
            except (ValueError, RuntimeError) as exc:
                rejected.append(str(exc))
        raise ValueError("no checked hover route: "+"; ".join(rejected))

    def plan_descent(self, branch, *, time_step=1/250):
        low, hover, low_q, _ = self._branch(branch)
        self.check_state(target_bottle=hover)  # Actual measured arrival, never planned q substitution.
        self.check_state(low_q, opening=True, target_bottle=low)
        result = self.native.plan_screw(goal_pose=self.Pose(self.link_target(low)),
            current_qpos=self.measured.copy(), time_step=self._time_step(time_step),
            qpos_step=.01, wrt_world=True, verbose=False)
        samples, _ = dense_path(result, self.start_arm)
        _require(np.max(np.abs(samples[-1]-low_q)) <= .05, "descent changed selected low branch")
        return result, self.check_path(result, self.start_arm, target_bottle=low, opening=True, descent=True)

    def plan_retreat(self, target_link, *, time_step=1/250):
        """Measured open-hand screw retreat; released bottle remains a fixture."""
        _require(not self.held, "retreat requires held=False and released_bottle fixture")
        self.check_state()
        goal = pose_matrix(target_link)
        result = self.native.plan_screw(goal_pose=self.Pose(goal), current_qpos=self.measured.copy(),
            time_step=self._time_step(time_step), qpos_step=.01, wrt_world=True, verbose=False)
        return result, self.check_path(result, self.start_arm, target_link=goal)
