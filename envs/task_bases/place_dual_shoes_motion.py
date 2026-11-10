"""Native physical shoe acquisition and placement with measured fixtures."""
from contextlib import contextmanager
from functools import lru_cache

import genesis as gs
import numpy as np
import transforms3d as t3d

from ..avatar.collision_checker import sample_capsule_planner_points
from ..grasp import GraspPose, tcp_to_link_pose
from ..genesis_compat import update_dofs_force_range_compat
from ..planning.base import PlanResult
from ..utils import ASSETS_PATH, Pose, to_numpy


@lru_cache(maxsize=1)
def _shoe_mesh():
    import trimesh
    return trimesh.load(ASSETS_PATH/'objects/041_shoe/visual/base0.glb',
                        force='mesh', process=False)


@lru_cache(maxsize=1)
def _shoe_vertices():
    vertices = np.asarray(_shoe_mesh().vertices).copy()
    vertices.setflags(write=False)
    return vertices


# Pinned Genesis1.2.0 Franka hand.stl collision bounds, in the hand frame.
_PALM_LO = np.array([-.031635918, -.103989720, -.025924802])
_PALM_HI = np.array([.031615783, .100425877, .065962151])


class _ShoeMotionRejected(RuntimeError):
    """A rejected physical motion, distinct from an evaluator error."""


def _valid_pose(pose):
    try:
        p, q = np.asarray(pose.p), np.asarray(pose.q)
        return (p.shape == (3,) and q.shape == (4,)
                and np.all(np.isfinite(p)) and np.all(np.isfinite(q))
                and np.isclose(np.linalg.norm(q), 1., atol=1e-4))
    except (AttributeError, TypeError, ValueError):
        return False


def _pose_matches(actual, target, tiny=False):
    if not _valid_pose(actual) or not _valid_pose(target):
        return False
    angle = 2 * np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1))
    return (np.linalg.norm(actual.p - target.p) <= (.010 if tiny else .015)
            and angle <= (.05 if tiny else np.deg2rad(10)))


class ShoeNativeMotionMixin:
    """Shoe-local motion policy; grasp acquisition stays in the public helper."""

    def _shoe_reject(self, reason):
        self.plan_success = False
        print(f"[shoe-motion] {reason}", flush=True)
        raise _ShoeMotionRejected(reason)

    def _shoe_refresh_avatar(self, arm_tag):
        self._shoe_refresh_fixtures()
        collider = getattr(self, "avatar_collider", None)
        if collider is not None:
            points = sample_capsule_planner_points(collider.current_capsules(), inflate_factor=1.8)
            if points is not None and points.size:
                self.robot.get_arm(arm_tag).planner.update_obstacles(points, resolution=.04)

    @contextmanager
    def _shoe_motion_scene(self, shoe, arm_tag):
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, Convex, CollisionObject

        arm = self.robot.get_arm(arm_tag)
        if self.config.get("robot_type") != "franka" or not hasattr(arm.planner, "_mplib"):
            self._shoe_reject("shoe placement requires native Franka MPLib")
        backend = arm.planner._mplib
        name = "place_dual_shoes_table"
        bounds = np.asarray(to_numpy(self.table.get_AABB()), dtype=float)
        if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                or np.any(bounds[1] <= bounds[0]) or backend.planning_world.has_object(name)
                or backend.acm.has_entry(name)):
            self._shoe_reject("invalid or conflicting table fixture")
        previous = getattr(self, "_shoe_motion_active", False)
        if previous:
            self._shoe_reject("nested shoe motion")
        state = {'backend':backend, 'fixtures':{}, 'table_name':name}
        self._shoe_scene_state = state
        def add(fixture_name, geometry, pose, geom=None):
            if (backend.planning_world.has_object(fixture_name) or backend.acm.has_entry(fixture_name)
                    or not _valid_pose(pose)):
                self._shoe_reject('invalid/conflicting shoe fixture')
            backend.planning_world.add_object(fixture_name, CollisionObject(geometry, MplibPose(p=pose.p,q=pose.q)))
            state['fixtures'][fixture_name] = (geometry, geom)
        try:
            add(name, Box(*(bounds[1]-bounds[0])), Pose(bounds.mean(axis=0)))
            box_geoms = [g for link in self.shoebox.entity.links for g in link.geoms]
            if len(box_geoms) != 5:
                self._shoe_reject('expected five physical shoebox walls')
            for index, geom in enumerate(box_geoms):
                size = np.asarray(geom.data, dtype=float)[:3]
                if geom.type != gs.GEOM_TYPE.BOX or not np.all(np.isfinite(size)) or np.any(size<=0):
                    self._shoe_reject('invalid shoebox primitive')
                add(f'place_dual_shoes_wall_{index}', Box(*size), self._shoe_geom_pose(geom), geom)
            for index, other in enumerate(self.shoes):
                if other is shoe:
                    continue
                for part, geom in enumerate(g for link in other.entity.links for g in link.geoms):
                    if geom.type != gs.GEOM_TYPE.MESH or not geom.is_convex:
                        self._shoe_reject('other shoe requires convex collision pieces')
                    vertices, faces = np.asarray(geom.init_verts,dtype=float), np.asarray(geom.init_faces,dtype=np.int32)
                    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
                        self._shoe_reject('invalid other-shoe vertices')
                    shape = Convex(vertices, faces, True)
                    add(f'place_dual_shoes_other_{index}_{part}', shape, self._shoe_geom_pose(geom), geom)
            # Only the fixed mounted base intersects the tabletop by design.
            backend.acm.set_entry("panda_link0", name, True)
            self._shoe_motion_active = True
            self._shoe_refresh_avatar(arm_tag)
            yield
        finally:
            for fixture_name in state['fixtures']:
                backend.planning_world.remove_object(fixture_name)
            backend.acm.remove_entry(name)
            self._shoe_motion_active = previous
            self._shoe_scene_state = None

    def _shoe_geom_pose(self, geom):
        pose = Pose(to_numpy(geom.get_pos(relative=False)), to_numpy(geom.get_quat(relative=False)))
        if not _valid_pose(pose):
            self._shoe_reject('invalid solver-world fixture pose')
        return pose

    def _shoe_refresh_fixtures(self):
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import CollisionObject
        state = getattr(self, '_shoe_scene_state', None)
        if state is None:
            return
        updates = [(name, shape, self._shoe_geom_pose(geom))
                   for name, (shape, geom) in state['fixtures'].items() if geom is not None]
        # Re-register only owned fixtures so the native broad-phase state is
        # refreshed too. Shapes are fixed; live solver-world poses can move.
        world = state['backend'].planning_world
        for name, shape, pose in updates:
            world.remove_object(name)
            world.add_object(name, CollisionObject(shape, MplibPose(p=pose.p,q=pose.q)))

    def _shoe_fk_tcp(self, arm_tag, joints):
        arm = self.robot.get_arm(arm_tag)
        backend = arm.planner._mplib
        backend.pinocchio_model.compute_forward_kinematics(arm.planner._pad_qpos(joints))
        link = backend.pinocchio_model.get_link_pose(backend.move_group_link_id)
        if not all(_valid_pose(p) for p in (link, arm.origin_pose, arm.tcp_offset)):
            return None
        return arm.origin_pose * Pose(np.asarray(link.p), np.asarray(link.q)) * arm.tcp_offset

    def _shoe_plan_valid(self, plan, seed, target, arm_tag, *, tiny=False, tiny_bound=.30):
        def rejected(reason):
            print(f'[shoe-plan] {reason}', flush=True)
            return False
        arm = self.robot.get_arm(arm_tag)
        if plan is None or not plan.success or not _valid_pose(target):
            return rejected('plan unavailable or target invalid')
        pos, vel = np.asarray(plan.position), np.asarray(plan.velocity)
        seed = np.asarray(seed)
        if (pos.ndim != 2 or pos.shape[1] != arm.n_arm or not len(pos)
                or vel.shape != pos.shape or seed.shape != (arm.n_arm,)
                or not all(np.all(np.isfinite(x)) for x in (pos, vel, seed))):
            return rejected('path shape or finite-value check')
        path = np.vstack([seed, pos])
        if (np.linalg.norm(pos[-1] - seed) > 3.5
                or np.max(np.abs(np.diff(path, axis=0))) > .30
                or (tiny and np.max(np.abs(pos[-1] - seed)) > tiny_bound)):
            return rejected('joint continuity or endpoint bound')
        for joints in path:
            full = arm.planner._pad_qpos(joints)
            if full.shape != (9,) or not np.all(np.isfinite(full)):
                return rejected('invalid full native joints')
            full = full.copy()
            full[7:9] = .04
            collisions = arm.planner._mplib.check_for_env_collision(full)
            if collisions:
                pairs = sorted({(str(getattr(c, 'link_name1', 'unknown')),
                                 str(getattr(c, 'object_name2', 'unknown')),
                                 str(getattr(c, 'link_name2', 'unknown'))) for c in collisions})
                return rejected(f'fixture/avatar collision: {pairs}')
        if not _pose_matches(self._shoe_fk_tcp(arm_tag, pos[-1]), target, tiny):
            return rejected('endpoint FK')
        return True

    def _shoe_grasp_plan_valid(self, plans, arm_tag):
        pre, grasp, grasp_link, pre_link = plans
        arm = self.robot.get_arm(arm_tag)
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        for plan, link in ((pre, pre_link), (grasp, grasp_link)):
            if not _valid_pose(link) or not _valid_pose(arm.tcp_offset):
                return False
            if not self._shoe_plan_valid(plan, seed, link * arm.tcp_offset, arm_tag):
                return False
            seed = plan.position[-1].copy()
        return True

    def _shoe_arrived(self, target, arm_tag, label, *, tiny=False):
        raw = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(), dtype=float)
        if (raw.shape != (7,) or not np.all(np.isfinite(raw))
                or not np.isclose(np.linalg.norm(raw[3:]), 1., atol=1e-4)):
            self._shoe_reject(f"{label}: invalid measured TCP")
        actual = Pose.from_pose7(raw)
        if not _pose_matches(actual, target, tiny):
            self._shoe_reject(f"{label}: measured arrival rejected")
        return actual

    # MPLib's screw integrator fails a feasible straight path when the last
    # step's leftover twist falls under its 1e-4 stall bound (diag: a 37mm
    # tiny lift stalled at iteration 5 with twist 3.7e-5, its goal IK free
    # and in limits).  Whether that happens depends only on how qpos_step
    # quantises the path, so a failed screw is re-integrated at nearby steps.
    SHOE_SCREW_QPOS_STEPS = (.02, .0193, .0207)

    # Joint 7 (the wrist roll) often ends a shoe grasp near its 2.897 limit,
    # so a straight move can need it past the limit (diag: 9 of 10 remaining
    # screw failures).  Moves may turn the TCP about its approach axis
    # instead: carry/retreat by +-170 deg (the same parallel-jaw grasp; an
    # exact 180 deg is the screw's singular trace -1 case, and only one
    # sense unwinds the wrist), lift/descent by a small +-12 deg.
    SHOE_WRIST_TURN_DEG = 170.
    SHOE_WRIST_NUDGE_DEG = 12.
    # A descent blocked by the shoe already in the box releases higher.
    SHOE_RELEASE_RAISE_M = (.03, .06)
    # A shoe resting a few cm in front of the robot base is grasped with the
    # arm folded next to its shoulder singularity (joint 2 near 0, so joints
    # 1 and 3 nearly align).  The straight 30mm tiny lift then needs joint
    # changes above the 0.30 rad tiny endpoint bound (neutral live run: 6 of
    # 20 non-reset failures, e.g. joint 5 swings 0.8 rad for 30 mm).  Only
    # when no candidate passes the strict bound, the tiny lift is planned
    # again with this endpoint bound, and accepted only if the FK of every
    # waypoint stays on the straight segment (SHOE_TINY_PATH_TOL_M / _RAD).
    SHOE_TINY_RELAXED_BOUND = 1.2
    SHOE_TINY_PATH_TOL_M = .004
    SHOE_TINY_PATH_TOL_RAD = .05

    @staticmethod
    def _shoe_yawed(pose, degrees):
        """``pose`` turned by ``degrees`` about the TCP approach (z) axis."""
        half = np.deg2rad(float(degrees))/2
        return pose*Pose([0., 0., 0.], [np.cos(half), 0., 0., np.sin(half)])

    @staticmethod
    def _shoe_nearest_target(actual, candidates):
        """The candidate TCP target a measured arrival pose belongs to."""
        def distance(target):
            angle = 2*np.arccos(np.clip(abs(np.dot(actual.q, target.q)), 0, 1))
            return float(np.linalg.norm(actual.p-target.p)) + .1*float(angle)
        return min(candidates, key=distance)

    def _shoe_wrist_alternatives(self, pose, degrees):
        return tuple(self._shoe_yawed(pose, sign*degrees) for sign in (1., -1.))

    def _shoe_plan_screw(self, seed, command, arm_tag, label, *, tiny=False, relaxed=False):
        """Validated native screw plan to ``command`` (TCP), or None.

        ``relaxed`` (tiny moves only) uses SHOE_TINY_RELAXED_BOUND and
        additionally requires a straight TCP path."""
        arm = self.robot.get_arm(arm_tag)
        link = tcp_to_link_pose(command, arm.tcp_offset)
        bound = self.SHOE_TINY_RELAXED_BOUND if relaxed else .30
        for step in self.SHOE_SCREW_QPOS_STEPS:
            try:
                plan = arm.planner.plan_screw_path(seed.copy(), link.to_pose7(), qpos_step=step)
            except RuntimeError as exc:
                print(f"[shoe-motion] {label}: native screw rejected: {exc}", flush=True)
                continue
            if (self._shoe_plan_valid(plan, seed, command, arm_tag, tiny=tiny, tiny_bound=bound)
                    and (not relaxed or self._shoe_path_straight(plan, seed, command, arm_tag, label))):
                return plan
        return None

    def _shoe_path_straight(self, plan, seed, command, arm_tag, label):
        """Every waypoint's FK TCP lies on the straight segment from the seed's
        FK TCP to ``command`` and keeps the commanded orientation."""
        start = self._shoe_fk_tcp(arm_tag, np.asarray(seed, dtype=float))
        if not _valid_pose(start):
            return False
        a, b = np.asarray(start.p, dtype=float), np.asarray(command.p, dtype=float)
        ab = b - a
        length2 = float(ab @ ab)
        turn = 2*np.arccos(np.clip(abs(np.dot(start.q, command.q)), 0, 1))
        worst_d = worst_r = 0.
        for joints in np.asarray(plan.position, dtype=float):
            tcp = self._shoe_fk_tcp(arm_tag, joints)
            if not _valid_pose(tcp):
                return False
            t = 0. if length2 <= 0 else float(np.clip((tcp.p - a) @ ab / length2, 0, 1))
            worst_d = max(worst_d, float(np.linalg.norm(tcp.p - (a + t*ab))))
            worst_r = max(worst_r, float(2*np.arccos(np.clip(abs(np.dot(tcp.q, command.q)), 0, 1))))
        ok = worst_d <= self.SHOE_TINY_PATH_TOL_M and worst_r <= turn + self.SHOE_TINY_PATH_TOL_RAD
        print(f"[shoe-motion] {label}: relaxed-bound path deviation={worst_d:.4f}m "
              f"rot={worst_r:.3f}rad -> {'ok' if ok else 'rejected'}", flush=True)
        return ok

    def _shoe_move(self, target, arm_tag, label, *, tiny=False, settle_open=False, tracking_offset=None,
                   alternatives=(), rrt=False, relaxed_tiny=False):
        """Plan and execute a native screw move to ``target`` (TCP).

        ``alternatives`` are further TCP targets tried, in order, only when
        no valid screw reaches ``target``; with ``rrt`` an MPLib RRT plan to
        each target is tried last, under the same fixture/avatar, joint
        continuity and endpoint-FK checks.  Nothing has executed before the
        target is chosen.  Returns the measured arrival pose of the target
        that was used (``_shoe_nearest_target`` recovers which one)."""
        arm = self.robot.get_arm(arm_tag)
        if not _valid_pose(target) or not _valid_pose(arm.tcp_offset) or not hasattr(arm.planner, "_mplib"):
            self._shoe_reject(f"{label}: invalid native target")
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            self._shoe_reject(f"{label}: invalid joint seed")
        offset = None
        if tracking_offset is not None:
            offset = np.asarray(tracking_offset, dtype=float)
            if (not tiny or settle_open or offset.shape != (3,)
                    or not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > .012):
                self._shoe_reject(f'{label}: invalid bounded tracking offset')
        self._shoe_refresh_avatar(arm_tag)
        plan = None
        for index, candidate in enumerate((target, *alternatives)):
            if not _valid_pose(candidate):
                self._shoe_reject(f"{label}: invalid native target")
            command = candidate if offset is None else Pose(candidate.p+offset, candidate.q)
            plan = self._shoe_plan_screw(seed, command, arm_tag, label, tiny=tiny)
            if plan is not None:
                if index:
                    print(f"[shoe-motion] {label}: planned alternative target {index}", flush=True)
                target = candidate
                break
        if plan is None and tiny and relaxed_tiny:
            for index, candidate in enumerate((target, *alternatives)):
                command = candidate if offset is None else Pose(candidate.p+offset, candidate.q)
                plan = self._shoe_plan_screw(seed, command, arm_tag, label, tiny=True, relaxed=True)
                if plan is not None:
                    print(f"[shoe-motion] {label}: relaxed tiny bound, target {index}", flush=True)
                    target = candidate
                    break
        if plan is None and rrt and offset is None:
            for index, candidate in enumerate((target, *alternatives)):
                link = tcp_to_link_pose(candidate, arm.tcp_offset)
                try:
                    result = arm.planner.plan_path(seed.copy(), link.to_pose7(), log=False)
                except RuntimeError as exc:
                    print(f"[shoe-motion] {label}: RRT rejected: {exc}", flush=True)
                    continue
                if self._shoe_plan_valid(result, seed, candidate, arm_tag):
                    print(f"[shoe-motion] {label}: RRT plan to target {index}", flush=True)
                    plan, target = result, candidate
                    break
        if plan is None:
            self._shoe_reject(f"{label}: path, collision or FK rejected")
        self.execute_plan(plan, arm_tag)
        for _ in range(60):
            self.step_sim()
        if settle_open:
            # A short open-jaw correction can still be converging after60
            # steps. Hold the same PD target for at most150 more steps,
            # preserving the measured arrival gate and final1mm refinement.
            for _ in range(150):
                raw = np.asarray(arm.get_ee_pose(), dtype=float)
                if raw.shape != (7,) or not np.all(np.isfinite(raw)):
                    self._shoe_reject(f'{label}: invalid measured TCP')
                measured = Pose.from_pose7(raw)
                if not _valid_pose(measured):
                    self._shoe_reject(f'{label}: invalid measured TCP')
                if _pose_matches(measured, target, tiny):
                    break
                self.step_sim()
        actual = self._shoe_arrived(target, arm_tag, label, tiny=tiny)
        print(f"[shoe-motion] {label}: TCP={actual.p}", flush=True)
        return actual

    def _shoe_tiny_lift(self, tcp, arm_tag):
        # Open refinement already establishes a bounded command bias. Starting
        # the next command at measured TCP discards it and can turn a30mm
        # requested rise into a20mm rise. Carry the measured position bias
        # through this one short lift; arrival still uses the desired TCP.
        arm = self.robot.get_arm(arm_tag)
        cached = np.asarray(getattr(arm, '_cached_target', None), dtype=float)
        if (cached.shape != (arm.n_arm+2,) or not np.all(np.isfinite(cached))):
            self._shoe_reject('tiny lift: invalid cached joint target')
        commanded = self._shoe_fk_tcp(arm_tag, cached[:arm.n_arm])
        if not _valid_pose(commanded) or not _valid_pose(tcp):
            self._shoe_reject('tiny lift: invalid tracking pose')
        offset = commanded.p-tcp.p
        angle = 2*np.arccos(np.clip(abs(np.dot(commanded.q,tcp.q)),0,1))
        if np.linalg.norm(offset) > .012 or angle > .05:
            self._shoe_reject('tiny lift: tracking bias exceeds correction bound')
        print(f'[shoe-motion] tiny lift: retained tracking offset={offset}', flush=True)
        lifted = Pose(tcp.p+[0,0,.030],tcp.q)
        return self._shoe_move(lifted,arm_tag,'tiny lift',tiny=True,tracking_offset=offset,
                               alternatives=self._shoe_wrist_alternatives(lifted,self.SHOE_WRIST_NUDGE_DEG),
                               relaxed_tiny=True)

    def _shoe_scale_scalar(self):
        scale = np.asarray(self._shoe_scale, dtype=float)
        if (scale.shape not in ((), (3,)) or not np.all(np.isfinite(scale))
                or np.any(scale <= 0) or (scale.ndim and not np.all(scale == scale[0]))):
            self._shoe_reject('shoe grasp requires actual isotropic scale')
        return float(scale.ravel()[0])

    def _shoe_candidates(self, shoe, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        pose = Pose(*self._shoe_pos_quat(shoe))
        scale = self._shoe_scale_scalar()
        vertices = _shoe_vertices() * scale
        world = vertices @ pose.to_matrix()[:3,:3].T + pose.p
        candidates = []
        # The end pinch can pivot the shoe without lifting its center. The
        # actual midbody section is46.3mm wide at60mm above the sole and
        # clears the native palm by9.9mm in the upright pose. Keep the live
        # geometry gate; PUBLIC still checks the approach against fixtures.
        center_rotation = np.column_stack(([0.,0.,-1.], [1.,0.,0.], [0.,-1.,0.]))
        for flip in (0,180):
            local = Pose([0.,float(vertices[:,1].min())+.060,0.],
                         t3d.quaternions.mat2quat(center_rotation))
            if flip:
                local = local*Pose([0,0,0],[0,0,0,1])
            link = tcp_to_link_pose(pose*local,arm.tcp_offset)
            hand_vertices = (world-link.p)@link.to_matrix()[:3,:3]
            mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
            if mask.any() and float(hand_vertices[mask,2].min()-_PALM_HI[2])>=.008:
                candidates.append(GraspPose(f'shoe_midbody_{flip}',local.p/scale,local.q,
                    pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        # Horizontal jaw yaw can clear the box while keeping the fingers
        # level. The actual midbody has two crossings within the open jaws;
        # raising this rotated palm2mm preserves the existing8mm clearance.
        for degrees in (-30,30):
            angle = np.deg2rad(degrees)
            approach = np.array([0.,-1.,0.])
            closing = np.array([np.cos(angle),0.,np.sin(angle)])
            rotation = np.column_stack((np.cross(closing,approach),closing,approach))
            for flip in (0,180):
                local = Pose([0.,float(vertices[:,1].min())+.062,0.],
                             t3d.quaternions.mat2quat(rotation))
                if flip:
                    local = local*Pose([0,0,0],[0,0,0,1])
                link = tcp_to_link_pose(pose*local,arm.tcp_offset)
                hand_vertices = (world-link.p)@link.to_matrix()[:3,:3]
                mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
                if mask.any() and float(hand_vertices[mask,2].min()-_PALM_HI[2])>=.008:
                    candidates.append(GraspPose(f'shoe_yaw_{degrees}_{flip}',local.p/scale,local.q,
                        pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        # A small lateral lean moves the wide palm away from a nearby box
        # wall while preserving the same solid midbody grasp point. Actual
        # mesh rays span49.6mm between two surfaces; native palm clearance
        # is10.3mm. PUBLIC still checks each full approach against fixtures.
        for prefix, leans in (('shoe_oblique', (-15,15)), ('shoe_slight_lean', (-6,6,-10,10))):
            for degrees in leans:
                angle = np.deg2rad(degrees)
                approach = np.array([np.sin(angle),-np.cos(angle),0.])
                closing = np.array([np.cos(angle),np.sin(angle),0.])
                rotation = np.column_stack((np.cross(closing,approach),closing,approach))
                for flip in (0,180):
                    local = Pose([0.,float(vertices[:,1].min())+.060,0.],
                                 t3d.quaternions.mat2quat(rotation))
                    if flip:
                        local = local*Pose([0,0,0],[0,0,0,1])
                    link = tcp_to_link_pose(pose*local,arm.tcp_offset)
                    hand_vertices = (world-link.p)@link.to_matrix()[:3,:3]
                    mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
                    if mask.any() and float(hand_vertices[mask,2].min()-_PALM_HI[2])>=.008:
                        candidates.append(GraspPose(f'{prefix}_{degrees}_{flip}',local.p/scale,local.q,
                            pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        # A nearby box can block the open finger even after the palm leans
        # clear. Move the contact point10mm away from the measured box and
        # use22.5degree lean. Only admit paired material surfaces inside the
        #80mm opening; the opposite lean can put a surface outside one jaw.
        box = getattr(self, 'shoebox', None)
        if box is not None:
            away = pose.p - np.asarray(to_numpy(box.entity.get_pos()), dtype=float)
            away[2] = 0.
            distance = float(np.linalg.norm(away))
            if np.all(np.isfinite(away)) and distance > .05:
                shift = pose.to_matrix()[:3,:3].T @ (.010 * away / distance)
                point = np.array([0.,float(vertices[:,1].min())+.060,0.]) + shift
                for degrees in (-22.5,22.5):
                    angle = np.deg2rad(degrees)
                    approach = np.array([np.sin(angle),-np.cos(angle),0.])
                    closing = np.array([np.cos(angle),np.sin(angle),0.])
                    rotation = np.column_stack((np.cross(closing,approach),closing,approach))
                    hits,_,_ = _shoe_mesh().ray.intersects_location(
                        np.array([(point-.12*closing)/scale]),np.array([closing]),multiple_hits=True)
                    crossings = sorted(float(np.dot(hit*scale-point,closing)) for hit in hits)
                    if len(crossings)!=2 or not (-.04<crossings[0]<0.<crossings[1]<.04):
                        continue
                    for flip in (0,180):
                        local = Pose(point,t3d.quaternions.mat2quat(rotation))
                        if flip:
                            local = local*Pose([0,0,0],[0,0,0,1])
                        link = tcp_to_link_pose(pose*local,arm.tcp_offset)
                        hand_vertices = (world-link.p)@link.to_matrix()[:3,:3]
                        mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
                        if mask.any() and float(hand_vertices[mask,2].min()-_PALM_HI[2])>=.008:
                            candidates.append(GraspPose(f'shoe_offset_lean_{degrees}_{flip}',local.p/scale,local.q,
                                pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        preferred = self._choose_grip_end(shoe)
        yaw = self._object_grip_yaw(shoe)
        previous = getattr(self, '_grip_yaw_deg', 0.)
        try:
            for end, flip in ((preferred,0), (preferred,180), (-preferred,0), (-preferred,180)):
                self._grip_yaw_deg = yaw + flip
                xy = self._shoe_grip_xy(shoe, end)
                target = self._top_down_tcp([*xy, max(pose.p[2]+(.020 if end>0 else .028), self.TABLE_TOP_Z+.015)])
                link = tcp_to_link_pose(target, arm.tcp_offset)
                hand_vertices = (world-link.p) @ link.to_matrix()[:3,:3]
                mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]), axis=1)
                if not mask.any():
                    continue
                clearance = float(hand_vertices[mask,2].min()-_PALM_HI[2])
                target = Pose(target.p+[0,0,max(0.,.008-clearance)], target.q)
                local = pose.inv()*target
                candidates.append(GraspPose(f'shoe_palm_clear_{"vamp" if end>0 else "heel"}_{flip}',
                    local.p/scale,local.q,pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        finally:
            self._grip_yaw_deg = previous
        # A descending approach to the back of the heel gives the low
        # pre-grasp/table corridor more room while keeping the jaws across width.
        heel_z, sole_y = float(vertices[:,2].min())+.029, float(vertices[:,1].min())
        rotation = np.column_stack(([0.,-2**-.5,-2**-.5], [1.,0.,0.], [0.,-2**-.5,2**-.5]))
        for flip in (0,180):
            local = Pose([0,sole_y+.050,heel_z],t3d.quaternions.mat2quat(rotation))
            if flip:
                local = local*Pose([0,0,0],[0,0,0,1])
            target = pose*local
            link = tcp_to_link_pose(target,arm.tcp_offset)
            hand_vertices = (world-link.p) @ link.to_matrix()[:3,:3]
            mask = np.all((hand_vertices[:,:2]>=_PALM_LO[:2]) & (hand_vertices[:,:2]<=_PALM_HI[:2]),axis=1)
            if mask.any() and float(hand_vertices[mask,2].min()-_PALM_HI[2]) < .008:
                continue
            candidates.append(GraspPose(f'shoe_heel_side_{flip}',local.p/scale,local.q,
                pre_distance=.12,source='manual',scale_frame='mesh_unit'))
        return pose, scale, candidates

    def _shoe_anchor_world(self, shoe):
        pose = Pose(*self._shoe_pos_quat(shoe))
        point = pose.p + pose.to_matrix()[:3,:3] @ self._shoe_grasp_local
        if not np.all(np.isfinite(point)):
            self._shoe_reject('invalid live shoe grasp point')
        return point

    def _shoe_retained(self, shoe, arm_tag, label):
        tcp = Pose.from_pose7(np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(),dtype=float))
        offset = tcp.to_matrix()[:3,:3].T @ (self._shoe_anchor_world(shoe)-tcp.p)
        error = float(np.linalg.norm(offset-self._shoe_anchor_offset))
        print(f'[shoe-motion] {label}: grasp-point drift={error:.4f}m',flush=True)
        if not np.isfinite(error) or error > .030:
            self._shoe_reject(f'{label}: lost measured grasp point')

    def _shoe_refine_open_grasp(self, target, shoe, arm_tag):
        """Compensate measured position bias before the first physical close."""
        origin = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float).copy()
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            self._shoe_reject('grasp refinement: invalid shoe origin')
        correction = np.zeros(3)
        for attempt in range(4):
            actual = self._shoe_arrived(target, arm_tag, 'grasp refinement')
            current = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
            if (current.shape != (3,) or not np.all(np.isfinite(current))
                    or np.linalg.norm(current-origin) > .003):
                self._shoe_reject('grasp refinement: shoe moved before closing')
            error = target.p-actual.p
            distance = float(np.linalg.norm(error))
            angle = 2*np.arccos(np.clip(abs(np.dot(target.q,actual.q)),0,1))
            if distance <= .001 and angle <= .05:
                return actual
            if attempt == 3 or distance > .010 or angle > .05:
                self._shoe_reject('grasp refinement: alignment rejected')
            correction += error
            if np.linalg.norm(correction) > .012:
                self._shoe_reject('grasp refinement: target correction bound')
            self._shoe_move(Pose(target.p+correction,target.q),arm_tag,
                            'open grasp correction',tiny=True,settle_open=True)

    def _shoe_native_pick(self, shoe, arm_tag, label):
        if not getattr(self,'_shoe_motion_active',False):
            self._shoe_reject('shoe acquisition requires active native fixture scene')
        arm = self.robot.get_arm(arm_tag)
        result = None
        for scene_attempt in range(2):
            self._shoe_refresh_avatar(arm_tag)
            pose, scale, candidates = self._shoe_candidates(shoe, arm_tag)
            # PUBLIC ranks approach alignment, not supplied list order. Try
            # the geometry-checked center section before end pinches that can
            # pivot the shoe on the table. None means nothing has executed;
            # once either pool acquires, all close/lift failures remain final.
            center = [g for g in candidates if g.name.startswith('shoe_midbody_')]
            # A box wall next to the shoe blocks the level midbody's palm
            # (hand vs wall at the grasp pose, assist live run).  Every
            # live-run episode (v1 seeds, all variants) that executed a
            # fallback pool failed physically, so the same midbody section
            # with a few degrees of lean, which lifts the palm end over the
            # wall, is tried first.
            slight = [g for g in candidates if g.name.startswith('shoe_slight_lean_')]
            yawed = [g for g in candidates if g.name.startswith('shoe_yaw_')]
            oblique = [g for g in candidates if g.name.startswith('shoe_oblique_')]
            shifted = [g for g in candidates if g.name.startswith('shoe_offset_lean_')]
            ends = [g for g in candidates if not g.name.startswith(('shoe_midbody_','shoe_slight_lean_','shoe_yaw_',
                                                                    'shoe_oblique_','shoe_offset_lean_'))]
            for pool_name, pool in (('midbody', center), ('slight-lean midbody', slight),
                                    ('level-jaw midbody', yawed),
                                    ('offset oblique midbody', shifted),
                                    ('oblique midbody', oblique), ('end/side fallback', ends)):
                if not pool:
                    continue
                print(f'[shoe-motion] PUBLIC {pool_name} pool: {len(pool)} candidates', flush=True)
                result = self.select_and_execute_grasp(object_name='041_shoe',object_pose=pose,
                    arm_tag=arm_tag,model_id=0,object_scale=scale,candidate_grasps=pool,
                    max_candidates=len(pool),table_z=self.TABLE_TOP_Z,plan_attempts=3,
                    screw_qpos_step=.02,post_plan_check=lambda plans:self._shoe_grasp_plan_valid(plans,arm_tag))
                if result is not None:
                    break
            if result is not None:
                break
            avatar = getattr(self,'avatar',None)
            if scene_attempt or avatar is None or avatar.spare():
                break
            # No grasp has executed. Wait for the moving carried shoe to
            # clear, then plan once against its new physical geometry.
            for _ in range(2000):
                if avatar.spare():
                    break
                self.step_sim()
            if not avatar.spare():
                self._shoe_reject('avatar did not clear acquisition corridor')
        if result is None:
            self._shoe_reject(f'{label}: no PUBLIC acquisition planned')
        grasp_link, _, grasp = result
        target = grasp_link*arm.tcp_offset
        for _ in range(self.PRE_CLOSE_SETTLE_STEPS):
            self.step_sim()
        self._shoe_refine_open_grasp(target,shoe,arm_tag)
        self._close_until_contact(arm,arm_tag,shoe)
        tcp = self._shoe_arrived(target,arm_tag,'close')
        self._shoe_grasp_local = grasp.pose.p*scale
        self._shoe_anchor_offset = tcp.to_matrix()[:3,:3].T @ (self._shoe_anchor_world(shoe)-tcp.p)
        if np.linalg.norm(self._shoe_anchor_offset) > .050:
            self._shoe_reject('closed fingers missed annotated shoe grasp point')
        before_z = float(self._shoe_pos_quat(shoe)[0][2])
        tcp = self._shoe_tiny_lift(tcp,arm_tag)
        self._shoe_retained(shoe,arm_tag,'tiny lift')
        if float(self._shoe_pos_quat(shoe)[0][2])-before_z < .015:
            self._shoe_reject('shoe did not rise with tiny lift')
        target = Pose([tcp.p[0],tcp.p[1],self.TABLE_TOP_Z+.22],tcp.q)
        tcp = self._shoe_move(target,arm_tag,'lift',rrt=True,
                              alternatives=self._shoe_wrist_alternatives(target,self.SHOE_WRIST_NUDGE_DEG))
        self._shoe_retained(shoe,arm_tag,'lift')
        if float(self._shoe_pos_quat(shoe)[0][2])-before_z < .050:
            self._shoe_reject('shoe did not meet retained50mm lift')
        self._shoe_carry_quat = tcp.q.copy()
        return True

    def _shoe_native_place(self, shoe, arm_tag, label, slot_idx):
        arm = self.robot.get_arm(arm_tag)
        slot = self._box_slot_xy(slot_idx)
        above = Pose([slot[0],slot[1],self._box_rim_z+self.APPROACH_HEIGHT],self._shoe_carry_quat)
        candidates = (above, *self._shoe_wrist_alternatives(above,self.SHOE_WRIST_TURN_DEG))
        tcp = self._shoe_move(above,arm_tag,'carry',rrt=True,alternatives=candidates[1:])
        above = self._shoe_nearest_target(tcp, candidates)
        for _ in range(self.PRE_CLOSE_SETTLE_STEPS*3):
            self.step_sim()
        self._shoe_arrived(above,arm_tag,'carry settle')
        self._shoe_retained(shoe,arm_tag,'carry')
        tcp = Pose.from_pose7(np.asarray(arm.get_ee_pose(),dtype=float))
        bottom = float(self._shoe_world_aabb(shoe)[0][2])
        # The primitive floor is10mm thick above the shoebox origin.
        floor = float(to_numpy(self.shoebox.entity.get_AABB())[0,2])+.010
        drop_z = max(tcp.p[2]-(bottom-floor-.010),self._box_rim_z+.050)
        if drop_z > tcp.p[2]+.015:
            self._shoe_reject('shoe already below intended release support')
        release = Pose([tcp.p[0],tcp.p[1],min(drop_z,tcp.p[2])],tcp.q)
        higher = tuple(Pose(release.p+[0,0,dz],release.q) for dz in self.SHOE_RELEASE_RAISE_M
                       if release.p[2]+dz < tcp.p[2]-.010)
        candidates = (release, *self._shoe_wrist_alternatives(release,self.SHOE_WRIST_NUDGE_DEG), *higher)
        tcp = self._shoe_move(release,arm_tag,'descent',alternatives=candidates[1:])
        release = self._shoe_nearest_target(tcp, candidates)
        self._shoe_retained(shoe,arm_tag,'descent')
        if self.RELEASE_FORCE_N is not None:
            force = float(self.RELEASE_FORCE_N)
            update_dofs_force_range_compat(arm.entity,arm._finger_dof_indices,
                                          lower_value=-force,upper_value=force)
        self.open_gripper(arm_tag,num_steps=self.RELEASE_GRIPPER_STEPS)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()
        self._shoe_arrived(release,arm_tag,'release')
        retreat = Pose([release.p[0],release.p[1],above.p[2]],release.q)
        tcp = self._shoe_move(retreat,arm_tag,'release retreat',rrt=True,
                              alternatives=self._shoe_wrist_alternatives(retreat,self.SHOE_WRIST_TURN_DEG))
        if np.linalg.norm(self._shoe_anchor_world(shoe)-tcp.p) < .050:
            self._shoe_reject('released shoe followed the withdrawing fingers')
        for _ in range(self.POST_PLACE_SETTLE_STEPS):
            self.step_sim()
        return True

    def execute_plan(self, result, arm_tag, gripper_result=None):
        if result is not None and getattr(self, "_shoe_motion_active", False):
            dt = float(self.scene.sim.dt)
            if not np.isfinite(dt) or dt <= 0:
                self._shoe_reject("invalid execution clock")
            ratio = .004 / dt
            repeats = int(round(ratio))
            if repeats < 1 or repeats > 16 or not np.isclose(ratio, repeats, rtol=0, atol=1e-8):
                self._shoe_reject("physical clock does not divide native4ms samples")
            pos, vel = np.asarray(result.position), np.asarray(result.velocity)
            if (not result.success or pos.ndim != 2 or not len(pos) or vel.shape != pos.shape
                    or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel))):
                self._shoe_reject("invalid execution trajectory")
            if gripper_result is not None:
                gripper_result = np.asarray(gripper_result)
                if gripper_result.shape != (len(pos),) or not np.all(np.isfinite(gripper_result)):
                    self._shoe_reject("invalid gripper trajectory")
                gripper_result = np.repeat(gripper_result, repeats)
            result = PlanResult(True, np.repeat(pos, repeats, axis=0), np.repeat(vel, repeats, axis=0))
        return super().execute_plan(result, arm_tag, gripper_result=gripper_result)
