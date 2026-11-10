"""Co-presence variant of ``put_object_cabinet`` (``put_object_in_cabinet_copresence``)."""

import operator

from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.put_object_cabinet import PutObjectCabinet


class PutObjectInCabinetCopresence(NeutralAvatarTableWorkMixin, PutObjectCabinet):
    """Robot puts an object in the cabinet while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Cabinet uses the full front/back lane: object starts near y=-0.2,
        # the drawer opens at y=-0.55, and the Franka hand parks above that
        # corridor. Match PutObjectCabinet._create_table exactly: center
        # (0,-0.35), half_size=(0.50,0.45), so tabletop y ends at +0.10.
        work_xy=(0.42, 0.12),
        mouse_work_xy=(0.42, 0.16),
        table_xy_bounds=(-0.50, 0.50, -0.80, 0.10),
        work_candidates=(
            (0.42, 0.12),
            (0.42, 0.20),
            (0.34, 0.16),
            (0.50, 0.12),
        ),
        # The cabinet (r=0.14), its object, and the Franka base all occupy
        # the right band; without blocking it the sponge/wipe layout solver
        # discards that band and raises on ~7 of 28 seeds.  Steer the
        # wipe work to the back/left bands instead.
        blocked_avatar_sides=("right",),
    )

    INSTRUCTION = (
        "put the table object in the cabinet drawer while the human works "
        "nearby on the same table"
    )

    # Sparse anchors can miss the narrow left work band beside the cabinet.
    # Search measured full-clip footprints more densely only after the ordinary
    # candidates fail, keeping the original heading and all strict layout gates.
    NEUTRAL_LAYOUT_RECOVERY_DEGREES = (0,)

    # Full-state expert: the seated writer leans over the drawer corridor and
    # the laptop/bottle props sit in front of the cabinet, so the horizontal
    # handle pre-grasp often has no collision-free IK. Fall back to the turned
    # or tilted bar grasps (see CabinetTaskCore.HANDLE_APPROACH_VARIANTS).
    HANDLE_APPROACH_VARIANTS = True

    # The neutral pool keeps the milk carton (dropped from assist/interrupt
    # for its fragile grasp) and the jam jar: their catalog close (.55)
    # squeezes them out on lift/carry. Close to the live jaw-axis width
    # minus 8 mm instead (the family's width grip): 3/20 -> 8/20 seeds.
    ORDINARY_WIDTH_GRIP_EXTRA_OBJECTS = ("038_milk-box", "031_jam-jar")

    def __init__(self, config=None):
        cfg = dict(config or {})
        # The laptop's many visual nodes can abort native NYX scene import.
        # This existing lossless visual-only cache leaves collision geometry,
        # articulated links, materials, and all non-laptop props unchanged.
        cfg.setdefault("nyx_laptop_mesh_pack_v1", True)
        super().__init__(cfg)

    def _neutral_avatar_use_layout_recovery(self, job):
        if (job.get("kind") == "sponge"
                and self.config.get("cabinet_neutral_layout_recovery_v1", True) is True):
            return True
        return super()._neutral_avatar_use_layout_recovery(job)

    def _open_drawer_physics(self, arm_tag):
        job = getattr(self, "_neutral_avatar_job", None) or {}
        if job.get("kind") not in ("bottle", "writing"):
            return super()._open_drawer_physics(arm_tag)
        # The ordinary 80 mm approach places link6 in the nearby human work
        # area. A 40 mm stand-off retains the same handle pose and orientation,
        # while the ordinary native IK, route, FK, and arrival gates still
        # check both approach legs and every subsequent physical pull.
        had_override = "_HANDLE_PRE_DIST" in self.__dict__
        previous = self._HANDLE_PRE_DIST
        self._HANDLE_PRE_DIST = 0.04
        try:
            print("[cabinet-neutral] compact 40 mm handle pre-grasp", flush=True)
            return super()._open_drawer_physics(arm_tag)
        finally:
            if had_override:
                self._HANDLE_PRE_DIST = previous
            else:
                del self._HANDLE_PRE_DIST

    def pick_and_place(self, pick, place, arm_tag):
        # The can-transfer hand crosses the table payload's pickup region.
        # Keep drawer opening concurrent, then let that finite clip finish
        # before bringing the robot hand down into the shared pickup area.
        job = getattr(self, "_neutral_avatar_job", None) or {}
        avatar = getattr(self, "avatar", None)
        if (job.get("kind") == "put_objects_in_bowl" and not job.get("loop")
                and avatar is not None and getattr(self, "_neutral_avatar_started", False)
                and not avatar.spare()):
            motion = avatar.motion_modules.get(job.get("motion"))
            try:
                count = len(motion.data)
                frame = operator.index(motion.at_frame)
            except (AttributeError, TypeError):
                self.plan_success = False
                print("[cabinet-neutral] invalid can-transfer clip; pickup rejected", flush=True)
                return False
            if not 0 <= frame < count <= 20000:
                self.plan_success = False
                print("[cabinet-neutral] invalid can-transfer frame bounds; pickup rejected", flush=True)
                return False
            budget = min(20000, count - frame + 100)
            waited = 0
            while waited < budget and not avatar.spare():
                self.step_sim()
                waited += 1
            print(f"[cabinet-neutral] waited {waited} physical steps for can transfer", flush=True)
            if not avatar.spare():
                self.plan_success = False
                print("[cabinet-neutral] can transfer did not finish; pickup rejected", flush=True)
                return False
        return super().pick_and_place(pick, place, arm_tag)
