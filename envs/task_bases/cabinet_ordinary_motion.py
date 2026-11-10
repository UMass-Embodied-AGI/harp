"""Native motion adapters for ordinary cabinet placement and assist/neutral."""
from contextlib import contextmanager

import numpy as np

from ..grasp import GraspPose, tcp_to_link_pose
from ..planning.base import PlanResult
from ..utils import Pose, to_numpy


class _OrdinaryCabinetRejected(RuntimeError):
    """A physical motion rejection propagated through shared placement steps."""


class CabinetOrdinaryMotionMixin:
    """Public acquisition and checked native transfer for ordinary cabinet."""

    # Further object ids that use the width grip (opt-in per variant).
    ORDINARY_WIDTH_GRIP_EXTRA_OBJECTS = ()
    ORDINARY_WIDTH_GRIP_EXTRA_SQUEEZE_M = .008

    def _ordinary_reject(self, reason):
        self.plan_success = False
        print(f'[cabinet-ordinary] {reason}', flush=True)
        raise _OrdinaryCabinetRejected(reason)

    def _ordinary_pose_valid(self, pose):
        from .put_object_cabinet import _cabinet_motion_pose_valid
        return _cabinet_motion_pose_valid(pose)

    def _ordinary_arrived(self, target, arm_tag, *, tiny=False):
        raw = np.asarray(self.robot.get_arm(arm_tag).get_ee_pose(),dtype=float)
        if raw.shape != (7,) or not np.all(np.isfinite(raw)):
            self._ordinary_reject('invalid measured TCP')
        actual = Pose.from_pose7(raw)
        if not self._ordinary_pose_valid(actual):
            self._ordinary_reject('invalid measured TCP quaternion')
        distance = float(np.linalg.norm(actual.p-target.p))
        angle = 2*float(np.arccos(np.clip(abs(np.dot(actual.q,target.q)),0,1)))
        if distance > (.010 if tiny else .015) or angle > (.05 if tiny else np.deg2rad(10)):
            self._ordinary_reject(f'measured arrival rejected: {distance:.4f}m/{angle:.4f}rad')
        return actual

    def _ordinary_path(self, plan, seed, target, arm_tag, *, tiny=False, goal_only=False):
        from .put_object_cabinet import _cabinet_motion_path
        arm = self.robot.get_arm(arm_tag)
        if plan is None or not plan.success or not self._ordinary_pose_valid(target):
            self._ordinary_reject('native plan unavailable or target invalid')
        path = _cabinet_motion_path(plan.position, getattr(plan,'velocity',None),arm.n_arm)
        if path is None:
            self._ordinary_reject('invalid native path')
        pos, vel = path
        seed = np.asarray(seed,dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            self._ordinary_reject('invalid native path seed')
        full_path = np.vstack([seed,pos])
        if (np.linalg.norm(pos[-1]-seed)>3.5
                or (not goal_only and np.max(np.abs(np.diff(full_path,axis=0)))>.30)
                or (tiny and np.max(np.abs(pos[-1]-seed))>.30)):
            self._ordinary_reject('native path continuity rejected')
        # IK is only a proposed endpoint for RRT, not an executed straight
        # interpolation. Check the real planned route after RRT returns.
        collision_samples = np.vstack([seed,pos[-1]]) if goal_only else full_path
        for q in collision_samples:
            full = np.asarray(arm.planner._pad_qpos(q),dtype=float).copy()
            if full.shape != (9,) or not np.all(np.isfinite(full)):
                self._ordinary_reject('invalid full native joints')
            full[7:9] = .04
            if arm.planner._mplib.check_for_env_collision(full):
                self._ordinary_reject('native path fixture/avatar collision')
        actual = self._drawer_plan_endpoint_tcp(arm_tag,pos[-1])
        if not self._ordinary_pose_valid(actual):
            self._ordinary_reject('invalid native FK')
        distance = float(np.linalg.norm(actual.p-target.p))
        angle = 2*float(np.arccos(np.clip(abs(np.dot(actual.q,target.q)),0,1)))
        if distance > (.010 if tiny else .015) or angle > (.05 if tiny else np.deg2rad(10)):
            self._ordinary_reject('native endpoint FK rejected')
        return PlanResult(True,pos,vel)

    def _ordinary_arm(self, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        if (self.config.get('robot_type')!='franka' or not hasattr(arm.planner,'_mplib')
                or not self._ordinary_pose_valid(arm.tcp_offset)):
            self._ordinary_reject('ordinary cabinet requires native Franka MPLib')
        seed = np.asarray(arm.get_arm_qpos(),dtype=float)
        if seed.shape != (arm.n_arm,) or not np.all(np.isfinite(seed)):
            self._ordinary_reject('invalid measured native joints')
        self._refresh_avatar_planner_obstacles(arm_tag)
        return arm, seed.copy()

    @contextmanager
    def _ordinary_fixtures(self, arm_tag):
        from mplib.pymp import Pose as MplibPose
        from mplib.pymp.collision_detection.fcl import Box, CollisionObject
        arm,_ = self._ordinary_arm(arm_tag)
        backend = arm.planner._mplib
        names = []
        self._ordinary_fixture_state = (backend,names)
        try:
            for suffix,entity in (('table',self.table),('cabinet',self.cabinet.entity)):
                name = f'cabinet_ordinary_{suffix}'
                bounds = np.asarray(to_numpy(entity.get_AABB()),dtype=float)
                if (bounds.shape!=(2,3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1]<=bounds[0])
                        or backend.planning_world.has_object(name) or backend.acm.has_entry(name)):
                    self._ordinary_reject('invalid/conflicting ordinary fixture')
                backend.planning_world.add_object(name,CollisionObject(Box(*(bounds[1]-bounds[0])),
                    MplibPose(p=bounds.mean(axis=0),q=np.array([1.,0,0,0]))))
                names.append(name)
            backend.acm.set_entry('panda_link0',names[0],True)
            yield
        finally:
            for name in names:
                backend.planning_world.remove_object(name)
                backend.acm.remove_entry(name)
            self._ordinary_fixture_state = None

    def _ordinary_clear_cabinet_envelope(self):
        state = getattr(self,'_ordinary_fixture_state',None)
        if state is None:
            return
        backend,names = state
        name = 'cabinet_ordinary_cabinet'
        if name in names:
            backend.planning_world.remove_object(name)
            backend.acm.remove_entry(name)
            names.remove(name)

    def pick_and_place(self, pick, place, arm_tag):
        if getattr(self,'_ordinary_pick_active',False):
            self._ordinary_reject('nested ordinary pick')
        self._ordinary_pick_active = True
        self._ordinary_last_target = None
        self._ordinary_tiny_lift_complete = False
        self._ordinary_transport_z = float(place.transport_z if place.transport_z is not None else self.TABLE_TOP_Z+.38)
        try:
            with self._ordinary_fixtures(arm_tag):
                return self._ordinary_public_pick_and_place(pick,place,arm_tag)
        except _OrdinaryCabinetRejected:
            self.plan_success = False
            return False
        finally:
            self._ordinary_pick_active = False
            self._ordinary_last_target = None

    def _ordinary_grasp_plans_valid(self, plans, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        pre, grasp, grasp_link, pre_link = plans
        seed = np.asarray(arm.get_arm_qpos(), dtype=float)
        try:
            for plan, link in ((pre, pre_link), (grasp, grasp_link)):
                if not self._ordinary_pose_valid(link):
                    return False
                checked = self._ordinary_path(plan, seed, link * arm.tcp_offset, arm_tag)
                seed = checked.position[-1].copy()
        except _OrdinaryCabinetRejected:
            return False
        return True

    def _ordinary_grasp_candidate(self, pick, object_pose):
        center = np.asarray(pick.get_center(), dtype=float)
        offset = np.asarray(pick.approach_xy_offset, dtype=float)
        radius = float(pick.radius)
        scale = float(self._object_scale)
        yaw = 0. if pick.get_grasp_yaw is None else float(pick.get_grasp_yaw())
        if (center.shape != (3,) or offset.shape != (2,)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(offset))
                or not np.isfinite(radius) or radius <= 0
                or not np.isfinite(scale) or scale <= 0 or not np.isfinite(yaw)
                or not np.isfinite(pick.grasp_z_offset)
                or not self._ordinary_pose_valid(object_pose)):
            self._ordinary_reject('invalid public grasp geometry')
        position = center.copy()
        position[:2] += offset
        position[2] = max(center[2] - .3 * radius + pick.grasp_z_offset, self.TABLE_TOP_Z + .002)
        grip = self._ordinary_width_grip()
        if grip is not None and grip[0] is not None:
            position[2] = grip[0]
        target = self._top_down_tcp(position, yaw=yaw)
        local = object_pose.inv() * target
        return GraspPose('cabinet_ordinary_topdown', local.p / scale, local.q,
                         pre_distance=.15, source='manual', scale_frame='mesh_unit')

    def _ordinary_width_grip(self):
        """Grasp height and close for payloads the catalog grip drops on lift.

        The dump family's validated Franka rule: close to the live
        cross-section minus 8 mm (4 mm per pad). The stapler keeps its plain
        radius height (the catalog's +20 mm offset gripped only its top
        ~10 mm); apple model 1 grips its broadest band, 28 mm below the top.
        Returns ``(tcp_z, close)`` (``tcp_z`` None: keep the ordinary grasp
        height) or ``None`` for every other payload."""
        key = (getattr(self, '_object_id', None), getattr(self, '_object_model_id', None))
        extra = key[0] in tuple(self.ORDINARY_WIDTH_GRIP_EXTRA_OBJECTS)
        if ((key not in (('048_stapler', 0), ('035_apple', 1)) and not extra)
                or self.config.get('robot_type') != 'franka'):
            return None
        entity = getattr(getattr(self, 'object', None), 'entity', None)
        if entity is None:
            return None
        bounds = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
            self._ordinary_reject('invalid payload bounds for width grip')
        extent = bounds[1] - bounds[0]
        if extra:
            # Opt-in upright bodies (jar, carton): the same closing rule across
            # the jaw axis (world Y for the ordinary top-down grasp, as for the
            # stapler). The grasp height stays the ordinary candidate's (None).
            width = float(extent[1])
            close = float(np.clip((width - float(self.ORDINARY_WIDTH_GRIP_EXTRA_SQUEEZE_M)) / .08, 0., 1.))
            return None, close
        if key[0] == '048_stapler':
            # Both stapler yaws close along world Y (as in dump_bin).
            width = float(extent[1])
            center_z = float(np.asarray(self._live_object_center(), dtype=float)[2])
            tcp_z = center_z - .3 * float(self._object_radius)
        else:
            width = float(np.min(extent[:2]))
            tcp_z = float(bounds[1, 2]) - .028
        close = float(np.clip((width - .008) / .08, 0., 1.))
        return max(tcp_z, float(self.TABLE_TOP_Z) + .002), close

    def _ordinary_grasp_candidates(self, pick, object_pose):
        """The nominal candidate, then the same grasp turned a half turn.

        A parallel-jaw top-down grasp turned about the vertical through its
        TCP closes on the same contact line. The caller tries the turned
        one only when the nominal fails to plan (typically a wrist-limit
        descent)."""
        nominal = self._ordinary_grasp_candidate(pick, object_pose)
        tcp = nominal.to_world(object_pose, float(self._object_scale))
        turned = Pose(tcp.p, (Pose(np.zeros(3), tcp.q) * Pose(np.zeros(3), [0., 0., 0., 1.])).q)
        local = object_pose.inv() * turned
        return [nominal, GraspPose(nominal.name + '_turned', local.p / float(self._object_scale), local.q,
                                   pre_distance=nominal.pre_distance, source='manual',
                                   scale_frame='mesh_unit')]

    def _ordinary_public_pick_and_place(self, pick, place, arm_tag):
        arm, _ = self._ordinary_arm(arm_tag)
        self._reseed_cached_arm_target(arm)
        self.open_gripper(arm_tag)
        for _ in range(max(0, int(pick.preopen_steps))):
            self.step_sim()
        object_pose = self.object.get_pose()
        # One candidate per call: the selector sorts by an approach score that
        # is ~0 for both top-down yaws, so float noise could put the turned
        # grasp first. Nothing has executed when a call returns None.
        result = None
        for candidate in self._ordinary_grasp_candidates(pick, object_pose):
            result = self.select_and_execute_grasp(
                object_name=self._object_id, object_pose=object_pose, arm_tag=arm_tag,
                model_id=self._object_model_id, object_scale=self._object_scale,
                candidate_grasps=[candidate], max_candidates=1, table_z=self.TABLE_TOP_Z,
                plan_attempts=3, screw_qpos_step=.02,
                post_plan_check=lambda plans: self._ordinary_grasp_plans_valid(plans, arm_tag),
            )
            if result is not None:
                break
        if result is None:
            return False
        grasp_link, _, _ = result
        target = grasp_link * arm.tcp_offset
        for _ in range(100):
            self.step_sim()
        self._ordinary_last_target = target
        self._ordinary_arrived(target, arm_tag)
        radius = float(pick.radius)
        close = (max(.010, radius - .005 if radius < .030 else radius * .7) / .04
                 if pick.close_value is None else float(np.clip(pick.close_value, 0., 1.)))
        grip = self._ordinary_width_grip()
        if grip is not None:
            close = grip[1]
        if not np.isfinite(close):
            self._ordinary_reject('invalid public grasp closing target')
        self.set_gripper(close, arm_tag)
        for _ in range(200):
            self.step_sim()
        if pick.after_close is not None:
            pick.after_close()
            for _ in range(40):
                self.step_sim()
        start = self._ordinary_arrived(target, arm_tag)

        def center():
            value = np.asarray(pick.get_center(), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                self._ordinary_reject('invalid measured payload center')
            return value

        initial_offset = start.to_matrix()[:3, :3].T @ (center() - start.p)

        def retained(actual, label):
            offset = center() - actual.p
            if pick.max_grasp_offset is not None and np.linalg.norm(offset) > float(pick.max_grasp_offset):
                self._ordinary_reject(f'{label}: payload outside grasp')
            local = actual.to_matrix()[:3, :3].T @ offset
            if (pick.transport_retention_tolerance is not None
                    and np.linalg.norm(local - initial_offset) > float(pick.transport_retention_tolerance)):
                self._ordinary_reject(f'{label}: payload lost purchase')
            return offset

        retained(start, 'close')
        tiny = self._ordinary_move_pose(Pose(start.p + [0, 0, .030], start.q), arm_tag, tiny=True)
        retained(tiny, 'tiny lift')
        after_tiny = center().copy()
        self._ordinary_tiny_lift_complete = True
        height = self._ordinary_transport_z
        lifted = self._ordinary_move_pose(Pose([tiny.p[0], tiny.p[1], height], tiny.q), arm_tag)
        retained(lifted, 'lift')
        minimum = float(pick.min_lift_delta) if pick.min_lift_delta is not None else max(.030, .5 * radius)
        if pick.require_lift and center()[2] - after_tiny[2] < minimum:
            self._ordinary_reject('payload did not lift from support')
        if place.stop_after_lift:
            return True
        destination = np.asarray(place.pos, dtype=float)
        if destination.ndim != 1 or len(destination) < 2 or not np.all(np.isfinite(destination)):
            self._ordinary_reject('invalid destination')
        carried = self._ordinary_move_pose(Pose([*destination[:2], height], lifted.q), arm_tag)
        for _ in range(max(0, int(getattr(self, 'PICK_PLACE_SETTLE_ABOVE_PLACE_STEPS',
                                        self.config.get('pick_place_settle_above_place_steps', 100))))):
            self.step_sim()
        carried = self._ordinary_arrived(carried, arm_tag)
        held = retained(carried, 'carry')
        if place.support_z is not None:
            if pick.get_half_height is None:
                self._ordinary_reject('support placement requires half height')
            release_z = min(height, self.support_release_tcp_z(place.support_z,
                float(pick.get_half_height()), held[2], place.support_clearance))
            if self._ordinary_palm_clearance_applies():
                release_z = self._ordinary_release_height(release_z, height, carried, arm_tag)
        else:
            release_z = (float(place.drop_from_z) if place.drop_from_z is not None
                         else float(destination[2]) if len(destination) >= 3 else height)
        release = self._ordinary_move_pose(Pose([*destination[:2], release_z], carried.q), arm_tag)
        for _ in range(50):
            self.step_sim()
        actual_release = self._ordinary_arrived(release, arm_tag)
        retained(actual_release, 'descent')
        if place.support_z is not None and self._ordinary_palm_clearance_applies():
            if self._ordinary_palm_bottom(actual_release, arm_tag) < self._drawer_rim_z + .002:
                self._ordinary_reject('measured palm remains too close to cabinet rim')
        if place.release:
            self.open_gripper(arm_tag)
            for _ in range(max(0, int(getattr(self, 'PICK_PLACE_RELEASE_OPEN_STEPS',
                                            self.config.get('pick_place_release_open_steps', 40))))):
                self.step_sim()
            self._after_pick_place_release(pick, place, arm_tag, release.p)
            for _ in range(max(0, int(getattr(self, 'PICK_PLACE_RELEASE_SETTLE_STEPS',
                                            self.config.get('pick_place_release_settle_steps', 210))))):
                self.step_sim()
        else:
            for _ in range(max(0, int(getattr(self, 'PICK_PLACE_HOLD_STEPS',
                                            self.config.get('pick_place_hold_steps', 100))))):
                self.step_sim()
        return True

    def _ordinary_palm_clearance_applies(self):
        return (self.config.get('robot_type') == 'franka'
                and getattr(self, '_object_id', None) != '100_seal')

    def _ordinary_palm_bottom(self, tcp, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        if not self._ordinary_pose_valid(tcp) or not self._ordinary_pose_valid(arm.tcp_offset):
            self._ordinary_reject('invalid palm clearance pose')
        # Outward-rounded bounds of the identical native/physical hand.stl
        # (SHA94493e94...). The204mm palm cannot descend inside this drawer.
        # Rotate its bounding corners to include the measured held tilt.
        corners = np.array([[x, y, z] for x in (-.032, .032)
                            for y in (-.104, .101) for z in (-.026, .066)])
        hand = tcp_to_link_pose(tcp, arm.tcp_offset)
        vertices = corners @ hand.to_matrix()[:3, :3].T + hand.p
        return float(vertices[:, 2].min())

    def _ordinary_release_height(self, support_height, transport_height, carried, arm_tag):
        values = np.asarray([support_height, transport_height, self._drawer_rim_z], dtype=float)
        if not np.all(np.isfinite(values)):
            self._ordinary_reject('invalid palm release heights')
        palm_offset = self._ordinary_palm_bottom(carried, arm_tag) - carried.p[2]
        release_height = max(float(support_height), float(self._drawer_rim_z) + .008 - palm_offset)
        if release_height > transport_height:
            self._ordinary_reject('palm release clearance exceeds transport height')
        return release_height

    def _move_seeded(self, link_pose7, arm_tag):
        raw = np.asarray(link_pose7,dtype=float)
        if raw.shape!=(7,) or not np.all(np.isfinite(raw)):
            self._ordinary_reject('invalid seeded target')
        link = Pose.from_pose7(raw)
        if not self._ordinary_pose_valid(link):
            self._ordinary_reject('invalid seeded quaternion')
        arm,seed = self._ordinary_arm(arm_tag)
        target = link*arm.tcp_offset
        try:
            ik = arm.planner.solve_ik(seed.copy(),raw,num_waypoints=2,log=False)
            checked = self._ordinary_path(ik,seed,target,arm_tag,goal_only=True)
            result = arm.planner._mplib.plan_qpos(goal_qposes=[arm.planner._pad_qpos(checked.position[-1])],
                current_qpos=arm.planner._pad_qpos(seed),time_step=.004,planning_time=10,
                rrt_range=.15,verbose=False)
        except _OrdinaryCabinetRejected:
            raise
        except RuntimeError as exc:
            self._ordinary_reject(f'native seeded planning failed: {exc}')
        if not result or result.get('status')!='Success':
            self._ordinary_reject('native seeded route unavailable')
        plan = PlanResult(True,result.get('position'),result.get('velocity'))
        plan = self._ordinary_path(plan,seed,target,arm_tag)
        self.execute_plan(plan,arm_tag)
        for _ in range(60):
            self.step_sim()
        self._ordinary_arrived(target,arm_tag)
        self._reseed_cached_arm_target(arm)
        self._ordinary_last_target = target
        return True

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=0.):
        target = self._top_down_tcp(target_pos,yaw=tcp_yaw)
        self._ordinary_move_pose(target, arm_tag)
        return True

    def _ordinary_move_pose(self, target, arm_tag, *, tiny=False):
        if not self._ordinary_pose_valid(target):
            self._ordinary_reject('invalid screw target')
        arm,seed = self._ordinary_arm(arm_tag)
        try:
            plan = arm.planner.plan_screw_path(seed,tcp_to_link_pose(target,arm.tcp_offset).to_pose7(),qpos_step=.02)
        except RuntimeError as exc:
            self._ordinary_reject(f'native screw failed: {exc}')
        plan = self._ordinary_path(plan,seed,target,arm_tag,tiny=tiny)
        self.execute_plan(plan,arm_tag)
        for _ in range(60):
            self.step_sim()
        actual = self._ordinary_arrived(target,arm_tag,tiny=tiny)
        self._reseed_cached_arm_target(arm)
        self._ordinary_last_target = target
        if (getattr(self,'_ordinary_tiny_lift_complete',False)
                and abs(target.p[2]-self._ordinary_transport_z)<1e-6):
            # The acquisition envelope fills the drawer cavity. Once the
            # payload reaches transport height, remove this box; keep the
            # independent table fixture and current avatar cloud.
            self._ordinary_clear_cabinet_envelope()
        return actual

    def _solve_ik(self, tcp_pose, arm_tag):
        if not self._ordinary_pose_valid(tcp_pose):
            self._ordinary_reject('invalid tiny-lift pose')
        arm,measured = self._ordinary_arm(arm_tag)
        seed = np.asarray(getattr(arm,'_cached_target',measured),dtype=float)
        if seed.ndim!=1 or len(seed)<arm.n_arm or not np.all(np.isfinite(seed)):
            self._ordinary_reject('invalid tiny-lift cached seed')
        seed = seed[:arm.n_arm].copy()
        try:
            plan = arm.planner.solve_ik(seed,tcp_to_link_pose(tcp_pose,arm.tcp_offset).to_pose7(),
                                      num_waypoints=2,log=False)
        except RuntimeError as exc:
            self._ordinary_reject(f'native tiny-lift IK failed: {exc}')
        plan = self._ordinary_path(plan,seed,tcp_pose,arm_tag,tiny=True)
        return plan.position[-1].copy()

    def _move_cartesian(self, start_pos, end_pos, arm_tag, n_steps=50, sim_per_step=20, tcp_yaw=0.):
        result = super()._move_cartesian(start_pos,end_pos,arm_tag,n_steps=n_steps,
                                        sim_per_step=sim_per_step,tcp_yaw=tcp_yaw)
        target = self._top_down_tcp(end_pos,yaw=tcp_yaw)
        self._ordinary_arrived(target,arm_tag,tiny=True)
        self._ordinary_last_target = target
        self._ordinary_tiny_lift_complete = True
        return result

    def set_gripper(self, value, arm_tag, *args, **kwargs):
        target = getattr(self,'_ordinary_last_target',None)
        if getattr(self,'_ordinary_pick_active',False) and target is not None:
            arm = self.robot.get_arm(arm_tag)
            if float(value)<float(arm.gripper_val):
                self._ordinary_arrived(target,arm_tag)
        return super().set_gripper(value,arm_tag,*args,**kwargs)

    def _after_pick_place_release(self, pick, place, arm_tag, release_pos):
        if not getattr(self,'_ordinary_pick_active',False):
            return
        arm = self.robot.get_arm(arm_tag)
        start = Pose.from_pose7(np.asarray(arm.get_ee_pose(),dtype=float))
        height = float(place.transport_z if place.transport_z is not None else self.TABLE_TOP_Z+.38)
        self._ordinary_move_pose(Pose([start.p[0],start.p[1],height],start.q),arm_tag)
        distance = float(np.linalg.norm(np.asarray(pick.get_center())-np.asarray(arm.get_ee_pose()[:3])))
        if distance < .050:
            self._ordinary_reject('payload still follows withdrawing open fingers')
        if place.support_z is not None:
            bounds = np.asarray(to_numpy(self.object.entity.get_AABB()),dtype=float)
            if bounds.shape!=(2,3) or not np.all(np.isfinite(bounds)):
                self._ordinary_reject('invalid released payload geometry')
            if bounds[0,2] > float(place.support_z)+.020:
                self._ordinary_reject('released payload remains above support after withdrawal')

    def execute_plan(self, result, arm_tag, gripper_result=None):
        if (result is not None and getattr(self,'_ordinary_pick_active',False)
                and not getattr(self,'_cabinet_plan_is_physical_clock',False)):
            dt = float(self.scene.sim.dt)
            if not np.isfinite(dt) or dt<=0:
                self._ordinary_reject('invalid execution clock')
            ratio = .004/dt
            repeats = int(round(ratio))
            if repeats<1 or repeats>16 or not np.isclose(ratio,repeats,rtol=0,atol=1e-8):
                self._ordinary_reject('physical clock does not divide native4ms samples')
            pos,vel = np.asarray(result.position),np.asarray(result.velocity)
            if (not result.success or pos.ndim!=2 or not len(pos) or pos.shape[1]!=7
                    or vel.shape!=pos.shape or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel))):
                self._ordinary_reject('invalid execution trajectory')
            if gripper_result is not None:
                gripper_result = np.asarray(gripper_result)
                if gripper_result.shape!=(len(pos),) or not np.all(np.isfinite(gripper_result)):
                    self._ordinary_reject('invalid gripper timeline')
                gripper_result = np.repeat(gripper_result,repeats)
            result = PlanResult(True,np.repeat(pos,repeats,axis=0),np.repeat(vel,repeats,axis=0))
        return super().execute_plan(result,arm_tag,gripper_result=gripper_result)
