"""Shared frying-scene choreography extracted verbatim from the retired
``frying_with_robot_pour`` task (no longer part of the benchmark).

The avatar holds pan + fork over the cooktop and runs the frying shake
loop; ``oil_bottle_recovery`` composes this around its own robot script.
Do not re-register this as a task.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d

from ..utils import Pose, to_numpy

# Two `frying` motions existed during retargeting:
#   - retarget/output_motion/frying.pkl       — 345 frames, root_z ∈ [-0.01, 0.33].
#     This is a squat-style motion where the avatar *moves vertically*; the
#     left-hand pose at frame 0 is far to the avatar's side and below the
#     counter — pan ends up out of Franka reach.  Not what we want here.
#   - assets/avatars/motions/generated_motions.pkl — `frying` key, 2120 frames,
#     root_z ∈ [0.29, 0.32] (essentially stationary upright stance).  This
#     is the standing shake-pan loop.  Use this.


# Two `frying` motions existed during retargeting:
#   - retarget/output_motion/frying.pkl       — 345 frames, root_z ∈ [-0.01, 0.33].
#     This is a squat-style motion where the avatar *moves vertically*; the
#     left-hand pose at frame 0 is far to the avatar's side and below the
#     counter — pan ends up out of Franka reach.  Not what we want here.
#   - assets/avatars/motions/generated_motions.pkl — `frying` key, 2120 frames,
#     root_z ∈ [0.29, 0.32] (essentially stationary upright stance).  This
#     is the standing shake-pan loop.  Use this.
_FRYING_MOTION_PKL = "avatars/motions/generated_motions.pkl"

_PAN_ASSET = "106_skillet"

# The stirring utensil is a cooking spatula (Kenney spatula), not a fork;
# the ``fork`` names in this module refer to it.
# Mesh axes (verified by parsing source OBJ materials):
#   +X side (X ∈ [+0.094, +0.341]) = brown HANDLE
#   -X side (X ∈ [-0.341, +0.094]) = white slotted BLADE
_FORK_ASSET = "cooking_spatula"

# Target longest-world-axis length after scaling.
_PAN_TARGET_LEN_M = 0.26          # matches kitchen-layout skillet scale

_FORK_TARGET_LEN_M = 0.25         # ~25 cm spatula

# Slow the 2120-frame "frying" motion. step_sim() advances the motion one
# frame per physics tick (dt=0.002, 500 Hz), so frame_ratio=4 stretches it
# to ~17 s of real-time sim — enough time for the robot to pick + pour
# while the avatar's shake-pan cycles play in the background.
_AVATAR_FRAME_RATIO = 4.0

# Kitchen-layout roles to drop so the cooktop (0, 0) footprint is clear.
_DROP_ROLES = {"skillet", "mug", "wineglass", "plate", "bread"}

# Bottle longest-axis target — drives how snugly the 8 cm Franka gripper
# wraps around the body.  At 0.14 the cross-section is ~5.4×6.2 cm so
# fingers close past the bottle surface and hold it firmly.  At 0.18
# (6.9×8.0 cm) the fingers grip only the surface and the bottle squeezes out.
_OIL_TARGET_DIM_M = 0.14

# Palm offset from the avatar's wrist along the finger direction.  Used
# by the fork driver so its handle lands at the palm (not the wrist).
_PALM_FROM_WRIST_M = 0.05

# Fork extra forward bias toward the *centre* of the palm (vs the edge).
# Without it the fork's grip sits at the inner edge of the palm, not the
# centre.  Push it forward along the finger direction for the
# fork only — the pan grip stays at the inner edge so the pan handle
# reads as being held, not pulled forward.  0.03 still leaves it
# slightly behind the palm; 0.06 centres it.
_FORK_FORWARD_BIAS_M = 0.06

# How far below the palm to anchor the pan.  Without it the pan floats
# ~10 cm *above* the avatar's visible hand because the hand
# joint returned by ``_get_hand_frame`` sits above the rendered skin's
# fingers (and the e2_L * 0.05 palm offset adds a bit more upward bias).
# Drop the pan kinematic anchor by this amount in world-Z to seat the
# handle inside the rendered grip.
_PAN_DROP_FROM_PALM_M = 0.10

def _upright_q() -> np.ndarray:
    """Rotation that maps object local +Y → world +Z (matches DEFAULT_KITCHEN_LAYOUT)."""
    return _euler_deg_to_q((90.0, 0.0, 0.0))

def _euler_deg_to_q(euler_deg) -> np.ndarray:
    rx, ry, rz = (float(a) * np.pi / 180.0 for a in euler_deg)
    R = t3d.euler.euler2mat(rx, ry, rz, axes="sxyz")
    return np.asarray(t3d.quaternions.mat2quat(R), dtype=np.float64)

def _mesh_fit_scale(asset: str, model_id: int, target_len_m: float) -> float:
    """Isotropic scale factor so the mesh's longest bounding-box axis
    measures ``target_len_m`` in world units."""
    from ..scenes.kitchen import _measure_mesh
    bmin, bmax, _ = _measure_mesh(asset, model_id)
    longest = float((bmax - bmin).max())
    return target_len_m / longest if longest > 1e-6 else 1.0


class FryingSceneMixin:
    """Verbatim closure of the frying-scene members oil_bottle_recovery uses."""

    def _get_entity_pose(self, entity, fallback: Pose) -> Pose:
        if entity is None:
            return fallback
        p = to_numpy(entity.get_pos()).ravel()[:3]
        q = to_numpy(entity.get_quat()).ravel()[:4]
        return Pose(p, q)

    # Skillet mesh (`106_skillet`) local frame, from trimesh analysis:
    #   bounds x ∈ [-0.558, +0.558]  (body/handle width)
    #   bounds y ∈ [ 0.000, +0.238]  (rim depth — local +Y is up)
    #   bounds z ∈ [-0.950, +0.949]  (long axis; HANDLE on -Z, BODY on +Z)
    # Vertex-density histogram confirms +Z = dense pan body, -Z = sparse handle.
    # Grip at z ≈ -0.6 (well along the handle, 63% of the way to the tip).
    _PAN_GRIP_LOCAL_Z = -0.6

    # Spatula mesh (`cooking_spatula`) local frame (verified via OBJ+mtl):
    # long axis is mesh +X (handle at +X side, blade at -X side); +Y is
    # the small "up" direction when held flat (extents only +0.000..+0.072).
    # Grip at mesh-X = +0.30 (4 cm in from the +0.341 handle end).
    _FORK_GRIP_LOCAL_X = +0.30

    # World-frame nudges applied by `_drive_fork_at_pan` so subclasses can
    # tune the spatula's visual presentation without duplicating the driver.
    # Defaults of 0 preserve the existing FryingWithRobotPour behavior.
    _FORK_NE_SHIFT_M = 0.0

    _FORK_UP_SHIFT_M = 0.0

    _FORK_EXTRA_DOWNTILT_DEG = 0.0

    _FORK_FORWARD_GRIP_SHIFT_M = 0.03

    _FORK_RIGHT_GRIP_SHIFT_M = 0.03

    def _attach_items_to_hands(self):
        """Ease avatar into the frying start pose, snap pan to left palm
        (handle in hand, body forward, rim up) and fork to right palm
        (handle in hand, tines forward, flat), then ``attach_object_to_hand``.
        """
        if self.avatar is None:
            return

        # Ease into frame 0 of the frying motion so the hands are in the
        # approximate "holding pan" posture.  The 2120-frame frying loop
        # then plays later from this starting pose.
        self.avatar.play_ease_into_animation("frying", frames=60)
        while not self.avatar.spare():
            self.step_sim()

        left_hand, R_left = self.avatar.robot._get_hand_frame(0)
        right_hand, R_right = self.avatar.robot._get_hand_frame(1)
        # e1 = thumb−pinky axis across palm
        # e2 = wrist→middle-knuckle (finger / forward direction)
        # e3 = e1 × e2 = palm normal
        e1_L, e2_L, e3_L = R_left[:, 0], R_left[:, 1], R_left[:, 2]
        e1_R, e2_R, e3_R = R_right[:, 0], R_right[:, 1], R_right[:, 2]

        # --- Pan: handle gripped in left palm, body extending forward ---
        #   pan +Y_world (rim up) → palm normal e3
        #   pan +Z_world (handle→body direction) → finger direction e2
        #   pan +X_world (width)  = pan +Y × pan +Z = e3 × e2 = -e1
        R_pan = np.column_stack([-e1_L, e3_L, e2_L])
        # Pan origin sits forward of the palm by (−grip_z) × scale along e2:
        #   origin - R_pan @ (0, 0, grip_z*scale) = hand
        # ⇒ origin = hand + R_pan[:, 2] × grip_z × scale × (−1)
        #          = hand + e2 × (−grip_z) × scale
        pan_origin = (left_hand
                      + e2_L * (-self._PAN_GRIP_LOCAL_Z) * self._pan_scale)
        pan_q = t3d.quaternions.mat2quat(R_pan)

        self.pan_entity.set_pos(pan_origin.astype(np.float64))
        self.pan_entity.set_quat(np.asarray(pan_q, dtype=np.float64))

        # --- Spatula: handle gripped in right palm, blade extending forward ---
        # Spatula handle is at mesh +X side, blade tip at mesh -X.  We want:
        #   spatula -X (blade tip)            → +e2  (forward = finger fwd)
        #   ⇒ spatula +X (handle direction)   → -e2  (back from finger fwd)
        #   spatula +Y (blade flat-side normal) → -e3 (up; opposite of palm-down e3)
        #   spatula +Z (blade width)          = +e1  (= -e2 × -e3 = e2 × e3)
        R_fork = np.column_stack([-e2_R, -e3_R, e1_R])
        # Origin: mesh point (_FORK_GRIP_LOCAL_X, 0, 0) lands at the palm:
        #   palm = origin + R[:,0] * (grip_x * scale)
        #   origin = palm - R[:,0] * (grip_x * scale)
        #          = right_hand - (-e2_R) * (grip_x * scale)
        #          = right_hand + e2_R * (grip_x * scale)
        fork_origin = (right_hand
                       - e2_R * self._FORK_FORWARD_GRIP_SHIFT_M
                       - e1_R * self._FORK_RIGHT_GRIP_SHIFT_M
                       + e2_R * self._FORK_GRIP_LOCAL_X * self._fork_scale)
        fork_q = t3d.quaternions.mat2quat(R_fork)

        self.fork_entity.set_pos(fork_origin.astype(np.float64))
        self.fork_entity.set_quat(np.asarray(fork_q, dtype=np.float64))

        # One cheap physics tick so set_pos/set_quat take effect before the
        # attach captures the hand-local offsets.
        self.scene.step()

        # Pan + fork are both driven manually from hand positions each step
        # (see step_sim override + _drive_pan_horizontal / _drive_fork_at_pan
        # below).  The default `attach_object_to_hand` carries the items
        # with the wrist's full rigid-body motion, which the frying shake
        # tilts ~45° (pan) and rotates leftward (fork tines).
        self._pan_drive = True
        self._fork_drive = True

    def _drive_pan_horizontal(self):
        """Re-position the pan at the avatar's left palm with rim-up
        orientation.  Called once per ``step_sim`` (after the avatar's
        own update) so the pan tracks hand translation but not hand
        rotation — the frying motion's wrist swing would otherwise tip
        the pan ~45°."""
        if not getattr(self, "_pan_drive", False) or self.pan_entity is None:
            return
        if self.avatar is None:
            return
        hand_pos, R_hand = self.avatar.robot._get_hand_frame(0)
        hand_pos = np.asarray(hand_pos, dtype=np.float64)
        e2_L = np.asarray(R_hand[:, 1], dtype=np.float64)
        # Forward direction in world horizontal plane (pan handle→body).
        forward = np.array([float(e2_L[0]), float(e2_L[1]), 0.0])
        n = np.linalg.norm(forward)
        forward = forward / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        # R_z(+30°) is the leftward swing (R_z(−30°) turns the wrong way).
        ang = np.radians(30.0)
        R_z = t3d.axangles.axangle2mat(np.array([0.0, 0.0, 1.0]), ang)
        forward = R_z @ forward
        # Pan world frame: +Y → world +Z (rim up), +Z → forward, +X = +Y × +Z.
        pan_y = np.array([0.0, 0.0, 1.0])
        pan_z = forward
        pan_x = np.cross(pan_y, pan_z)
        R_pan = np.column_stack([pan_x, pan_y, pan_z])
        # Pan grip point (mesh-local z=_PAN_GRIP_LOCAL_Z) should sit at the
        # avatar's PALM, not the wrist.  Palm is
        # ~5 cm distal from the wrist along the finger direction; also
        # drop the anchor 10 cm in world-Z so the rendered fingers wrap
        # the handle (the bone joint sits ~10 cm above the visible skin's
        # grip area).
        palm_pos = hand_pos + e2_L * _PALM_FROM_WRIST_M
        palm_pos = palm_pos + np.array([0.0, 0.0, -_PAN_DROP_FROM_PALM_M])
        pan_origin = (palm_pos
                      + pan_z * (-self._PAN_GRIP_LOCAL_Z) * self._pan_scale)
        pan_q = t3d.quaternions.mat2quat(R_pan)
        self.pan_entity.set_pos(pan_origin.astype(np.float64))
        self.pan_entity.set_quat(np.asarray(pan_q, dtype=np.float64))

    def _drive_fork_at_pan(self):
        """Re-position the fork so its tines point at the pan center
        regardless of right-wrist rotation.  Tines along world horizontal
        from the right hand toward the pan body center; thickness axis
        kept world-up so the fork sits flat."""
        if not getattr(self, "_fork_drive", False) or self.fork_entity is None:
            return
        if self.avatar is None:
            return
        right_hand, R_hand_R = self.avatar.robot._get_hand_frame(1)
        right_hand = np.asarray(right_hand, dtype=np.float64)
        e1_R = np.asarray(R_hand_R[:, 0], dtype=np.float64)
        e2_R = np.asarray(R_hand_R[:, 1], dtype=np.float64)
        # Fork uses a deeper palm anchor (palm-centre, not palm-edge) so the
        # handle sits inside the visible grip.
        palm_pos = (
            right_hand
            + e2_R * (
                _PALM_FROM_WRIST_M
                + _FORK_FORWARD_BIAS_M
                - self._FORK_FORWARD_GRIP_SHIFT_M
            )
            - e1_R * self._FORK_RIGHT_GRIP_SHIFT_M
        )
        # World-frame nudge for subclass presentation tuning.
        palm_pos = palm_pos + np.array([
            float(self._FORK_NE_SHIFT_M),
            float(self._FORK_NE_SHIFT_M),
            float(self._FORK_UP_SHIFT_M),
        ])
        # Tines should point mostly DOWNWARD into the pan, not
        # horizontally toward the right.  Use the full 3D
        # vector from palm to the live pan body center: with the palm
        # held above the pan during the frying motion, this aim has a
        # negative-Z component so the fork dives into the pan.  Forced
        # ≥10 cm downward bias if the palm happens to be below pan.
        pan_pose = self._get_entity_pose(
            self.pan_entity, Pose([0, 0, self.TABLE_TOP_Z + 0.1]),
        )
        R_pan = t3d.quaternions.quat2mat(np.asarray(pan_pose.q, dtype=np.float64))
        pan_body_center = (np.asarray(pan_pose.p, dtype=np.float64)
                           + R_pan[:, 2] * self._PAN_BODY_CENTER_LOCAL_Z
                           * self._pan_scale)
        aim = np.asarray(pan_body_center, dtype=np.float64) - palm_pos
        if aim[2] > -0.10:
            aim = np.array([float(aim[0]), float(aim[1]), -0.10])
        n = np.linalg.norm(aim)
        aim_unit = aim / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        # Subclass-tunable extra downward tilt: rotate aim_unit around the
        # horizontal axis perpendicular to its xy projection so the blade
        # tip dips further into the pan.
        extra_deg = float(self._FORK_EXTRA_DOWNTILT_DEG)
        if abs(extra_deg) > 1e-6:
            horiz = np.array([float(aim_unit[0]), float(aim_unit[1]), 0.0])
            hn = np.linalg.norm(horiz)
            if hn > 1e-6:
                horiz_unit = horiz / hn
                axis = np.array([-horiz_unit[1], horiz_unit[0], 0.0])
                R_extra = t3d.axangles.axangle2mat(
                    axis, np.deg2rad(extra_deg)
                )
                aim_unit = R_extra @ aim_unit
                an = np.linalg.norm(aim_unit)
                if an > 1e-9:
                    aim_unit = aim_unit / an
        # Spatula axes:
        #   mesh +X (handle direction) → -aim_unit  (handle points AWAY from pan)
        #   mesh +Y (blade flat-side)  → up-ish     (blade flat-side faces up)
        #   mesh +Z (blade width)      → horizontal perpendicular
        spatula_x = -aim_unit
        world_up = np.array([0.0, 0.0, 1.0])
        spatula_z = np.cross(spatula_x, world_up)
        sz_n = np.linalg.norm(spatula_z)
        if sz_n < 1e-6:
            spatula_z = np.array([0.0, 1.0, 0.0])
        else:
            spatula_z = spatula_z / sz_n
        spatula_y = np.cross(spatula_z, spatula_x)
        sy_n = np.linalg.norm(spatula_y)
        spatula_y = spatula_y / sy_n if sy_n > 1e-9 else np.array([0.0, 0.0, 1.0])
        R_fork = np.column_stack([spatula_x, spatula_y, spatula_z])
        # Anchor the spatula's handle grip (mesh-X = +_FORK_GRIP_LOCAL_X) at the palm:
        #   palm = origin + R[:,0] * grip_x * scale
        #   origin = palm - R[:,0] * grip_x * scale
        #          = palm - (-aim_unit) * grip_x * scale
        #          = palm + aim_unit * grip_x * scale
        fork_origin = palm_pos + aim_unit * (
            self._FORK_GRIP_LOCAL_X * self._fork_scale
        )
        fork_q = t3d.quaternions.mat2quat(R_fork)
        self.fork_entity.set_pos(fork_origin.astype(np.float64))
        self.fork_entity.set_quat(np.asarray(fork_q, dtype=np.float64))

    def step_sim(self):
        """Same physics tick as ``BaseTask.step_sim`` plus pan/fork
        re-drives between ``avatar.step()`` (which would tip the pan with
        the wrist + drift the fork tines) and ``capture_frame()``."""
        self.scene.step()
        self.SIM_STEP_IDX += 1
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            self.avatar.step()
            self._drive_pan_horizontal()
            self._drive_fork_at_pan()
        self._notify_interaction_sim_step()
        # Executed-motion spill tracker — during carry phases record
        # the worst world-Z component of the bottle's +Y (long) axis so
        # check_success can fail episodes whose bottle tipped far enough
        # mid-transit that real oil would have poured out.
        if (getattr(self, "_tilt_track_active", False)
                and getattr(self, "oil_entity", None) is not None):
            try:
                R_oil = t3d.quaternions.quat2mat(
                    to_numpy(self.oil_entity.get_quat()).ravel()[:4],
                )
                axis_z = float(R_oil[2, 1])
                prev = getattr(self, "_transit_min_axis_z", None)
                if prev is None or axis_z < prev:
                    self._transit_min_axis_z = axis_z
            except Exception:
                pass
        self._tick_avatar_checks()
        if self.vla_recorder is not None:
            self.vla_recorder.tick(self)
        self.capture_frame()
        if self._record_stride is not None:
            if self._record_tick % self._record_stride == 0:
                self.record_frame()
            self._record_tick += 1

    # Pan-body center in local mesh frame is roughly z ≈ +0.45 (vertex
    # density peaks around there; pan body extends z ∈ [0, +0.95]).
    _PAN_BODY_CENTER_LOCAL_Z = 0.45

    # mplib `update_obstacles` resolution.  Each point is treated as a
    # voxel of half this size, so 0.04 → ±2 cm inflation around every
    # capsule sample.  Coarse enough that planning stays fast, fine
    # enough that the avatar's arm cylinders form an obstacle wall.
    _AVATAR_OBSTACLE_RES = 0.035

    # Perimeter samples are placed at capsule_radius × this
    # factor so the planner sees a fatter obstacle and routes the
    # robot elbow further away; without it the elbow grazes the
    # avatar's elbow during the pour-hold.  1.6× ≈ +3 cm extra clearance
    # on a typical 5 cm-radius forearm capsule (combined with mplib's
    # built-in ±2 cm voxel pad → ~+5 cm total clearance margin).
    # Do not raise it to 2.0: at 2.0 (+0.03 arm extra +mplib ±2 cm voxels)
    # the inflated pcd swallows the whole carry corridor: the robot's CURRENT
    # config registers as "Invalid start state" (screw dead on arrival)
    # and every near-branch IK solution is "in collision", leaving only
    # wrist-flipped far branches that swing the bottle.  Avatar avoidance
    # beyond this belongs in explicit via-waypoints, not heavier obstacle
    # inflation; the live avatar-collision tracker remains the ultimate
    # contact gate.
    _AVATAR_INFLATE_FACTOR = 1.6

    # The frying animation moves the avatar's right arm while the robot is
    # executing an already-planned path.  Add a task-local buffer around those
    # arm/hand capsules so the static mplib snapshot covers the near-future
    # shake motion instead of only the exact current frame.
    # Kept at 0.02: a fatter arm margin (e.g. 0.045; the pour-hold graze is
    # handled by the pour-point z-shift) blocks the IK
    # feasibility of EVERY staging candidate on some seeds at every shake
    # snapshot: the wrist volume at the goal pose reaches into the
    # inflated shell even with the goal bubble.  Real hand geometry is
    # ~20 cm from the staging zone; the live tracker guards contact.
    _AVATAR_ACTIVE_ARM_EXTRA_INFLATE_M = 0.02

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        """Sample points along the avatar's bone-cylinder rig so the
        planner can treat the avatar as a static obstacle for the
        duration of one plan.  Returns ``None`` if the collider isn't
        live."""
        if self.avatar_collider is None:
            return None
        pts = []
        for _name, pa, pb, r in self.avatar_collider.current_capsules():
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            seg = pb - pa
            L = float(np.linalg.norm(seg))
            r_inflated = float(r) * self._AVATAR_INFLATE_FACTOR
            if _name.startswith(("r_", "l_")):
                # A larger per-phase margin here does not prevent avatar
                # contact during the return carry: planning-time margin
                # can't stop the avatar's LIVE shake from swinging into an
                # already-executing trajectory.  See avatar_retreat_enabled
                # for the per-step fix.
                r_inflated += self._AVATAR_ACTIVE_ARM_EXTRA_INFLATE_M
            if L < 1e-6:
                pts.append(pa)
                continue
            axis = seg / L
            # Pick two orthogonal vectors perpendicular to axis.
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 \
                else np.array([1.0, 0.0, 0.0])
            u = np.cross(axis, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(axis, u)
            v = v / (np.linalg.norm(v) + 1e-12)
            n_axis = max(2, int(L / 0.04) + 1)
            for t in np.linspace(0.0, 1.0, n_axis):
                c = pa + t * seg
                pts.append(c)
                # Perimeter samples at the inflated radius so the
                # obstacle covers a ring fatter than the actual
                # cylinder — gives the planner more clearance margin.
                # 8 azimuth samples (was 4) so the inflated ring is
                # dense enough that 4 cm voxel inflation can't slip
                # the elbow between samples.
                for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    pts.append(c + r_inflated * (np.cos(theta) * u + np.sin(theta) * v))
        return np.asarray(pts, dtype=np.float64) if pts else None

    # Sample the open-front wall cabinet's bottom shelf as a
    # static planner obstacle so mplib routes the elbow under it during
    # the lift+pour swing.  Cabinet geometry from `build_kitchen_backdrop`
    # in `envs/scenes/kitchen.py`: width 1.70 m (x), depth 0.30 m (y),
    # bottom-shelf surface at z = TABLE_TOP_Z + 0.55 = 1.315 m, centered
    # at y ≈ 0.345 (front face y ≈ 0.195, back face y ≈ 0.495).
    _CABINET_X_HALF = 0.85

    _CABINET_Y_RANGE = (0.195, 0.495)

    _CABINET_Z_RANGE = (1.315, 1.665)  # bottom-shelf z up to top-panel z

    def _cabinet_obstacle_points(self) -> np.ndarray | None:
        """Sample points on the cabinet's bottom-shelf face + the inside
        volume so mplib treats the open-front wall cabinet as a no-go
        region.  Spacing matches `_AVATAR_OBSTACLE_RES` so the points
        get merged into adjacent voxels."""
        if not getattr(self, "kitchen_backdrop", {}).get("cabinet"):
            return None
        step = self._AVATAR_OBSTACLE_RES
        xs = np.arange(-self._CABINET_X_HALF, self._CABINET_X_HALF + step, step)
        ys = np.arange(self._CABINET_Y_RANGE[0], self._CABINET_Y_RANGE[1] + step, step)
        zs = np.arange(self._CABINET_Z_RANGE[0], self._CABINET_Z_RANGE[1] + step, step)
        # Bottom-shelf surface (z = lower bound of cabinet) — this is the
        # plane the elbow would clip when reaching forward+up.
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        bottom = np.stack(
            [XX.ravel(), YY.ravel(),
             np.full(XX.size, self._CABINET_Z_RANGE[0])], axis=-1,
        )
        # Sample the front face (y = lower bound of cabinet y range)
        # so the elbow can't poke INTO the cabinet from below+south.
        XX2, ZZ2 = np.meshgrid(xs, zs, indexing="ij")
        front = np.stack(
            [XX2.ravel(),
             np.full(XX2.size, self._CABINET_Y_RANGE[0]),
             ZZ2.ravel()], axis=-1,
        )
        return np.concatenate([bottom, front], axis=0).astype(np.float64)

    def _bottle_to_link(self, T_world_bottle: Pose, arm_tag: str) -> Pose:
        arm = self.robot.get_arm(arm_tag)
        T_world_ee = T_world_bottle * self._T_ee_bottle.inv()
        return T_world_ee * arm.tcp_offset.inv()

