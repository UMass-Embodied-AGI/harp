"""Deliver to Human (Cueing): pick up an object and deliver it to the human's hand.

Per-object grasp method (selected by ``OBJECT_GRASP_METHOD``):
  - ``annotated``: read grasp pose from ``grasp_poses_franka.yml``.  Object
    is spawned with ``save_object_quat`` so the annotation is geometrically
    consistent.  Uses the shared annotated-grasp delivery pipeline.
  - ``primitive``: top-down approach via ``TopDownPickPlaceMixin``.  Object
    spawned with identity quat; gripper descends straight down with
    ``radius * 0.7`` close target.  Works well for objects whose smallest
    cross-section fits the Franka gripper.

Each object uses whichever method succeeded on at least 50% of per-object
coverage runs.

Object selection is sampled by default from the shared
``human_transfer_pool``.  Config ``object_name`` / ``model_id`` remains an
explicit override, and ``random_object: false`` selects the pool's first item
for deterministic compatibility runs.

Both methods end with the gripper held at the deliver pose (no release) —
the arm extends toward the avatar's right hand and stays there.
"""

import json

import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask, TargetSpec
from ..interaction import IntentEvalModeController
from ..interaction.cue_wait import await_cue
from ..manipulation import TopDownPickPlaceMixin, PickSpec, PlaceSpec
from ..utils import (
    ASSETS_PATH,
    Pose,
    load_object,
    to_numpy,
)
from ..grasp import read_save_object_quat, tcp_to_link_pose
from ..task_bases.human_transfer_planning import (
    refresh_avatar_obstacles, move_seeded, endpoint_reaches_goal,
)
from ..task_bases.resting_guard import avatar_holds
from ..interaction.interrupt_progress import robot_tcp


class DeliverToHuman(TopDownPickPlaceMixin, BaseTask):
    """Pick up an object and deliver to static avatar; per-object grasp method."""

    INSTRUCTION = "pick up the {object} from the table and deliver it to the human"
    OBJECT_SET = "human_transfer_pool"
    use_avatar = True
    avatar_init_pos = np.array([0.0, 1.5, -0.18])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Preserve the avatar-side edge at y=+0.87, but extend the robot-side edge
    # to y=-0.50 so the Franka base at y=-0.35 sits 0.15 m inboard.
    TABLE_HALF_SIZE = (0.53, 0.685)
    table_offset = np.array([0.0, 0.185])

    recording_camera_pos = [1.3, 0.6, 2.5]
    recording_camera_lookat = [0.0, 0.4, 0.75]

    # 035_apple is not in human_transfer_pool (mid-carry drops on the round
    # mesh even after full close/PD/friction tuning); the per-object apple
    # entries below remain for explicit ``object_name`` overrides, but the
    # pool does not offer it.
    DEFAULT_OBJ_NAME = "073_rubikscube"
    DEFAULT_OBJ_MODEL_ID = 1

    # Per-object method selection — empirical per-object coverage runs of
    # both methods (annotated grasps augmented with synthetic grasps, and
    # max_candidates 40 so RRT can fall back to negative-score arm-side
    # grasps).  Only objects with >= 50% success are kept.  Synthetic
    # grasps come from ``preprocess_grasp_flying_gripper.py``.  Below
    # threshold: 046_alarm-clock (1/3), 018_microphone (0/3 — synth grasps
    # below the table-height filter).
    OBJECT_GRASP_METHOD = {
        "048_stapler": "annotated",       # 3/3
        "024_scanner": "annotated",       # 1/1 (small sample)
        "023_tissue-box": "annotated",    # 3/3 (synthetic-augmented)
        "065_soy-sauce": "annotated",     # 2/3 (synthetic + max_cand=40)
        "113_coffee-box": "annotated",    # 2/3 (10x synthetic resample)
        "005_french-fries": "primitive",  # 2/3
        "052_dumbbell": "annotated",      # 2/3 (manual_topdown_X/Y_center)
        "039_mug": "annotated",           # 3/3 (manual_topdown_X/Y_center)
        "047_mouse": "annotated",         # 3/3 (manual_topdown_yaw45_*)
        "086_woodenblock": "annotated",   # 3/3 (m1 + manual_topdown_yaw45_*)
        "020_hammer": "annotated",        # 2/3 (lay-flat save_quat + manual_topdown_X_center)
        # Apple uses the primitive path.  The annotated path drops the apple
        # mid-carry (even with seeded carry + kept orientation + friction +
        # PD + close 0.45), while this task's primitive path is 100% on the same-size
        # rubikscube and take_from_human carries the same apple@1 fine on
        # the mixin primitives.
        "035_apple": "primitive",

        "073_rubikscube": "primitive",    # compact box; radius-based top-down pick
        "029_olive-oil": "annotated",     # 2/3 (m1 + lay-flat + B_higher)
        "066_vinegar": "annotated",       # 2/3 (m1 + lay-flat + A_center)
        "095_glue": "annotated",          # 2/3 (m6 + lay-flat + A_center)
        "100_seal": "annotated",          # 3/3 (m2 + tight spawn + A_center)
        "055_small-speaker": "annotated", # 2/3 (m1 + A_center)
        "038_milk-box": "annotated",      # object-aug: grasp 3/3
    }

    # Per-object model_id override (default 0).  Some objects have a
    # smaller variant that fits the franka 8 cm gripper better.
    OBJECT_MODEL_ID = {
        "035_apple": 1,        # 5.4 cm (vs 6.6 cm at model 0)
        "086_woodenblock": 1,  # 7.1 × 10.2 × 7.1 cm (vs 10.2 cube at model 0)
        "029_olive-oil": 1,    # body 7.7 cm (vs 13.3 cm at m0)
        "066_vinegar": 1,      # body 8.9 cm (vs 16.6 cm at m0)
        "030_drill": 1,        # 5.3 × 15.4 × 15.4
        "100_seal": 2,         # 3.6 × 7.8 × 3.2 — small handle stamp
        "095_glue": 6,         # 4.3 × 15.5 × 4.3 — slim glue stick
        "055_small-speaker": 1,
    }

    # Per-object gripper close target (None = full close).  Some objects
    # need a partial close so the gripper doesn't squeeze them out.
    OBJECT_CLOSE_TARGET = {
        # 0.45: apple carry failures are mid-carry drops at 0.55; 0.45 holds
        # the identical 035_apple@1 reliably in take_from_human_easy.
        "035_apple": 0.45,
    }

    # Per-object spawn (x_lo, x_hi, y_lo, y_hi). Default applies when missing.
    # 100_seal: small stamp, IK marginal at edges of default range; this tight
    # corner is where the A_center grasp succeeds reliably.
    OBJECT_SPAWN_RANGE = {
        "100_seal":     ( 0.00, 0.05, -0.10, -0.07),
    }

    TASK_OBJECTS = ("073_rubikscube",)
    # Success: the object is at the PRESENTED palm.  The human's
    # take_from_human clip has started and finished (hand held out), and the
    # object centre is within this of the live palm (the delivery pose is
    # 0.21 m in front of and 5 cm below it: ~0.22 m).  A wide (0.65 m)
    # sphere around the live palm with no presentation check would score a
    # cube held near the demo delivery spot on the step the human starts
    # moving.
    SUCCESS_DIST_3D_M = 0.30
    # The hand is above the tabletop; a table-resting object can be within
    # the success sphere just because the avatar stands nearby. Require the final
    # object to be no more than 40 cm below the palm, which still allows the
    # reachable deliver pose but rejects dropped/unpicked table objects.
    SUCCESS_DZ_MIN_M = -0.40
    SUCCESS_TABLE_CLEARANCE_M = 0.05
    FINAL_HOLD_SETTLE_STEPS = 300
    # At 18 cm the gripper grazed the offered index finger on seeds 0/17;
    # allow a physical approach margin without changing collision scoring.
    DELIVER_HAND_STANDOFF_M = 0.21
    DELIVER_HAND_Z_OFFSET_M = -0.05

    def __init__(self, config):
        super().__init__(config)
        # The actual entry is sampled in load_actors, after reset(seed) has
        # seeded numpy.  ``random_object`` is retained as a compatibility knob;
        # random selection is now the default.
        self.obj_name = None
        self.obj_model_id = self.DEFAULT_OBJ_MODEL_ID
        self.random_object = bool(config.get("random_object", True))
        self.grasp_method_override = config.get("grasp_method")

    def create_interaction_controller(self) -> IntentEvalModeController:
        return IntentEvalModeController()

    def _refresh_planner_obstacles(self, arm_tag):
        refresh_avatar_obstacles(self, arm_tag)

    def move_to_pose(self, pose, arm_tag):
        self._refresh_planner_obstacles(arm_tag)
        plan = super().move_to_pose(pose, arm_tag)
        if plan is not None and not endpoint_reaches_goal(
            self.robot.get_arm(arm_tag).planner, plan, pose,
        ):
            self.plan_success = False
            return None
        return plan

    def _move_seeded(self, link_pose7, arm_tag):
        return move_seeded(self, link_pose7, arm_tag)

    # mplib's multi-start IK often returns a far joint branch (1-3 rad jumps)
    # for a millimetre lift waypoint; re-solve a few times for a solution on
    # the current branch before skipping the waypoint.
    LIFT_IK_ATTEMPTS = 6

    def _solve_ik(self, tcp_pose, arm_tag):
        """Use collision-aware MPLib for the primitive's short lift too.

        A waypoint with no same-branch solution is skipped (the mixin's
        Cartesian contract); it does not fail the episode.  The legs after
        the lift re-plan from the measured qpos and verify their arrival.
        (Latching plan_success=False here would fail episodes whose pick
        and handover then succeed.)
        """
        arm = self.robot.get_arm(arm_tag)
        self._refresh_planner_obstacles(arm_tag)
        link_pose7 = tcp_to_link_pose(tcp_pose, arm.tcp_offset).to_pose7()
        seed = np.asarray(
            getattr(arm, "_cached_target", arm.get_arm_qpos()), dtype=float,
        ).ravel()[:arm.n_arm]
        measured = np.asarray(arm.get_arm_qpos(), dtype=float).ravel()[:arm.n_arm]
        for attempt in range(int(self.LIFT_IK_ATTEMPTS)):
            start = seed if attempt % 2 == 0 else measured
            result = arm.planner.solve_ik(start, link_pose7, log=False)
            if not endpoint_reaches_goal(arm.planner, result, link_pose7):
                continue
            goal = np.asarray(result.position[-1], dtype=float)[:arm.n_arm]
            # Millimetre-spaced lift waypoints must not cause branch flips.
            if float(np.max(np.abs(goal - seed))) <= 0.30:
                return goal
        self._lift_ik_skips = int(getattr(self, "_lift_ik_skips", 0)) + 1
        return None

    def _move_screw(self, target_pos, arm_tag, tcp_yaw=0.0):
        self._refresh_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag, tcp_yaw=tcp_yaw)

    def _grasp_method(self) -> str:
        if self.grasp_method_override:
            return self.grasp_method_override
        return self.OBJECT_GRASP_METHOD.get(self.obj_name, "annotated")

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.35, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _create_table(self, table_height=0.74):
        dx, dy = self.table_offset
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=(dx, dy),
        )

    def _settle_scene(self, n_steps: int = None):
        if not getattr(self, "_avatar_head_hand_aligned", False):
            target = self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id
            )
            self._avatar_head_hand_yaw_deg = self._align_avatar_head_hand_to_xy(
                target[:2], hand_id=1, motion_name="take_from_human",
            )
            self._avatar_head_hand_aligned = True
        super()._settle_scene(n_steps)

    def load_actors(self):
        entry = self.resolve_target_object(randomize=self.random_object)
        self.obj_name = entry.object_id or entry.key
        self.obj_model_id = int(self.config.get("model_id", entry.model_id))

        table_top = self.TABLE_TOP_Z + 0.02
        # Annotated method needs the spawn quat that matches the grasp's
        # frame-of-reference; primitive top-down doesn't care, identity is fine.
        if self._grasp_method() == "annotated":
            obj_q = read_save_object_quat(self.obj_name, robot_type="franka")
        else:
            obj_q = np.array([1.0, 0.0, 0.0, 0.0])

        # Spawn region: per-object override falls back to default range that
        # stays inside the dexterous workspace at any annotated grasp orientation.
        spawn = self.OBJECT_SPAWN_RANGE.get(self.obj_name, (-0.10, 0.15, -0.15, -0.05))
        obj_x = np.random.uniform(spawn[0], spawn[1])
        obj_y = np.random.uniform(spawn[2], spawn[3])
        self.obj_actor = load_object(
            self.scene, Pose([obj_x, obj_y, table_top], obj_q), self.obj_name,
            model_id=self.obj_model_id, convex=True, is_static=False,
            # Explicit friction 4.0 (the catalog value): on the Genesis
            # DEFAULT friction the apple slips out mid-carry.
            friction=4.0,
        )

        self.target = TargetSpec(
            object=self.obj_actor,
            position=lambda: self.avatar.robot.get_palm_center(hand_id=1),
            label="right_palm",
            object_pos_fn=lambda: self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id,
            ),
            success_dist_3d=self.SUCCESS_DIST_3D_M,
            success_dz_min=self.SUCCESS_DZ_MIN_M,
            success_require_no_avatar_collision=True,
        )

    def _object_bottom_z(self) -> float:
        md_path = (
            ASSETS_PATH / "objects" / self.obj_name
            / f"model_data{self.obj_model_id}.json"
        )
        with open(md_path) as f:
            md = json.load(f)
        extents = np.asarray(md["extents"], dtype=np.float64)
        scale = np.asarray(md.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        center = np.asarray(md.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        half = 0.5 * extents * scale
        center = center * scale
        corners = np.array([
            center + np.array([sx * half[0], sy * half[1], sz * half[2]])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ])
        pose = self.obj_actor.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        world = np.asarray(pose.p, dtype=np.float64) + corners @ R.T
        return float(np.min(world[:, 2]))

    def _object_table_clearance(self) -> float:
        return self._object_bottom_z() - float(self.TABLE_TOP_Z)

    def _object_still_held(self) -> bool:
        return self._object_table_clearance() > float(self.SUCCESS_TABLE_CLEARANCE_M)

    def _clear_avatar_collision_state(self) -> None:
        """Start grading after the avatar's presentation pre-roll."""
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0

    # ------------------------------------------------------------------
    # Deliver-pose helpers
    # ------------------------------------------------------------------
    def _deliver_target_and_direction(self, arm_tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Return target xyz and horizontal gripper direction toward the palm."""
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.asarray(arm.origin_pose.p, dtype=np.float64)
        palm = np.asarray(self.avatar.robot.get_palm_center(hand_id=1), dtype=np.float64)

        palm_to_robot = arm_base[:2] - palm[:2]
        n = np.linalg.norm(palm_to_robot)
        if n < 1e-6:
            palm_to_robot = np.array([0.0, -1.0], dtype=np.float64)
        else:
            palm_to_robot = palm_to_robot / n

        target = palm.copy()
        target[:2] = palm[:2] + palm_to_robot * float(self.DELIVER_HAND_STANDOFF_M)
        target[2] = palm[2] + float(self.DELIVER_HAND_Z_OFFSET_M)

        toward_palm = palm[:2] - target[:2]
        n = np.linalg.norm(toward_palm)
        toward_palm = toward_palm / n if n > 1e-6 else -palm_to_robot
        return target, toward_palm

    def _deliver_pos(self, arm_tag: str) -> np.ndarray:
        """Object-center target for the primitive top-down delivery path."""
        target, _ = self._deliver_target_and_direction(arm_tag)
        return target

    def _compute_deliver_pose(
        self, arm_tag: str, keep_current_orientation: bool = False,
    ) -> Pose:
        """Link pose with TCP z=down and TCP x pointing toward the palm.

        Used by the annotated path via the guarded seeded move.
        ``keep_current_orientation``: reuse the measured TCP orientation
        instead of the palm-facing frame — for round objects the handoff
        needs no specific wrist roll, and the authored reorientation can be
        a 4+ rad wound-seed jump from the annotated grasp branch that the
        branch gate refuses (unseeded plan_pose then spins/drops the
        object).
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        deliver_pos, toward_palm = self._deliver_target_and_direction(arm_tag)

        if keep_current_orientation:
            current = Pose.from_pose7(arm.get_ee_pose())
            deliver_tcp = Pose(deliver_pos, current.q)
            return tcp_to_link_pose(deliver_tcp, tcp_offset)

        tcp_z = np.array([0.0, 0.0, -1.0])
        tcp_x = np.array([toward_palm[0], toward_palm[1], 0.0])
        tcp_x /= np.linalg.norm(tcp_x) + 1e-8
        tcp_y = np.cross(tcp_z, tcp_x)
        tcp_y /= np.linalg.norm(tcp_y) + 1e-8

        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        q = t3d.quaternions.mat2quat(R)
        deliver_tcp = Pose(deliver_pos, q)
        return tcp_to_link_pose(deliver_tcp, tcp_offset)

    def _move_screw_to_link(self, link_pose7, arm_tag: str) -> bool:
        """Straight-line Cartesian move to a link pose (mplib plan_screw).

        Falls back to ``move_and_execute`` (RRT plan_pose) if screw planning
        fails. Returns True on success.
        """
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        plan_screw_path = getattr(planner, "plan_screw_path", None)
        if plan_screw_path is None:
            return self.move_and_execute(link_pose7, arm_tag) is not None
        result = plan_screw_path(arm.get_arm_qpos(), np.asarray(link_pose7).ravel()[:7])
        if result is not None and getattr(result, "success", False):
            self.execute_plan(result, arm_tag)
            return True
        # Guarded fallback (j7 unwrap + branch gate) instead of raw
        # unseeded plan_pose while holding.
        return self._move_seeded(link_pose7, arm_tag) is not None

    # ------------------------------------------------------------------
    # Per-method play_once
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Eval-mode human.  The human starts the take_from_human presentation
    # (holds out an open hand) at a per-seed random policy step (25 sim
    # steps) from [EVAL_TRIGGER_STEP_MIN, EVAL_TRIGGER_STEP_MAX], during the
    # policy; the robot must see the hand come out before it delivers.  The
    # clip plays over EVAL_PRESENTATION_SECONDS (30 policy steps).  In the
    # training demos the hand is out from
    # the first frames and the robot starts its delivery leg ~150-175 policy
    # steps in (demo median 206, p95 220); the window straddles that moment,
    # so a robot that never watches the hand delivers to the hanging arm on
    # roughly half the seeds, and "always wait for the latest presentation"
    # plus the handover needs up to ~358 steps (max_steps 448).  Before the
    # trigger the
    # human stands at the clip's start pose, as at the demos' first frame.
    # ``eval_avatar_pre_policy=True`` instead finishes the presentation
    # before the first policy step (no cue to read).
    # ------------------------------------------------------------------
    EVAL_AVATAR_PRE_POLICY = False
    EVAL_TRIGGER_STEP_MIN = 80
    EVAL_TRIGGER_STEP_MAX = 240
    _EVAL_RNG_SALT = 0xDE11E

    # Eval-mode presentation at physical speed.  The demos play
    # take_from_human at one mocap frame per 2 ms sim step (~0.1 s) before
    # the recording starts; at that rate the eval human would snap into the
    # presented pose within ~2 policy steps.  In eval mode the same clip (same start and
    # end poses) is stretched to this duration, so the cue is a visible reach.
    EVAL_PRESENTATION_SECONDS = 1.5
    _SIM_DT_S = 0.002

    # Eval-mode success is a real handover: the robot opens
    # the gripper while the object is within HANDOVER_RADIUS_M of the
    # presented palm; the human then holds it (attached to the right hand) and
    # success is "the human holds it" (for the N-step success hold).  Release
    # = the commanded gripper value rises HANDOVER_OPEN_DELTA above its
    # tightest value while the object was in the gripper (TCP within
    # HANDOVER_GRIP_TCP_M, object lifted off the table).
    HANDOVER_RADIUS_M = 0.12
    # On a handover the human takes the cube: it glides (kinematically, this
    # task's object only) from the release point into the palm-down hand's
    # grasp pose -- around the fingertips, then under the fingers -- over
    # HANDOVER_GLIDE_SIM_STEPS (0.4 s = 8 policy steps), and is attached
    # there.  ``_handover_done`` (success) is set when the glide ends.
    HANDOVER_GLIDE_SIM_STEPS = 200
    # Before the glide the human holds the cube still where it was released
    # while the gripper finishes opening and backs off (0.25 s).
    HANDOVER_GLIDE_DELAY_SIM_STEPS = 125
    # Grasp pose in the hand frame: this far from the knuckle centre along
    # the fingers, and the object's half height + this below the palm side.
    HANDOVER_GRASP_ALONG_FINGERS_M = 0.03
    HANDOVER_GRASP_PALM_GAP_M = 0.012
    # Way round the fingertips: knuckles + finger length + half size + margin.
    HANDOVER_FINGER_LENGTH_M = 0.09
    HANDOVER_OPEN_DELTA = 0.15
    HANDOVER_GRIP_TCP_M = 0.07
    # Full-state / blind expert handover pose: object centre this far from the
    # palm (knuckle centre) toward the robot (xy) and this far above it (the
    # highest candidate with IK).  The presented hand is held out palm-down,
    # fingers toward the robot, near the arm's reach limit: a diagonal
    # approach from the demo pose can touch the index finger, so the expert
    # rises first and comes in level; targets 3 cm behind the knuckles
    # often have no IK.
    HANDOVER_OBJ_CANDIDATES_M = (   # (standoff toward the robot, height)
        (0.02, 0.08), (0.02, 0.065), (0.05, 0.08), (0.05, 0.065),
        (0.02, 0.05), (0.05, 0.05),
        # Over the offered fingertips: nearer the robot and clear of the
        # finger obstacle cloud that blocks the targets above (no-obstacle IK
        # succeeds there).  Object-palm distance stays <= 0.113 m, inside
        # HANDOVER_RADIUS_M.
        (0.08, 0.08), (0.08, 0.065), (0.08, 0.05),
    )
    HANDOVER_RETREAT_DZ_M = 0.05
    HANDOVER_RETREAT_BACK_M = 0.15

    def step_sim(self):
        super().step_sim()
        # Sim-step clock for the blind_intent metrics only.
        self._blind_sim_step = int(getattr(self, "_blind_sim_step", 0)) + 1
        if getattr(self, "_handover_glide", None) is not None:
            self._advance_handover_glide()

    def _eval_at_reset(self):
        self._blind_sim_step = 0
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        self._blind_human_started_sim_step = None
        self._blind_human_done_sim_step = None
        self._full_eval_cue_waits = None
        self._handover_done = False
        self._handover_sim_step = None
        self._handover_dist_m = None
        self._handover_grip_min = None
        self._handover_last_release_dist_m = None
        self._handover_glide = None
        self._handover_taken = False
        if self._eval_pre_policy_enabled():
            self._eval_trigger_step = None
            self.avatar.play_animation("take_from_human")
            return
        lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
        hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
        if hi < lo:
            hi = lo
        # Private per-seed stream: the draw never shifts the global RNG.
        rng = np.random.default_rng(
            [int(getattr(self, "_episode_seed", 0)), self._EVAL_RNG_SALT])
        self._eval_trigger_step = int(rng.integers(lo, hi + 1))
        if self._eval_trigger_step <= 0:
            self._start_blind_presentation()

    def _start_blind_presentation(self):
        self._play_presentation_physical_time()
        self._blind_human_started_sim_step = int(getattr(self, "_blind_sim_step", 0))

    def _play_presentation_physical_time(self):
        """Play take_from_human stretched to EVAL_PRESENTATION_SECONDS."""
        avatar = self.avatar
        saved = getattr(avatar, "frame_ratio", None)
        n_frames = self._presentation_clip_frames() if saved is not None else None
        if n_frames:
            seconds = float(self.config.get(
                "eval_presentation_seconds", self.EVAL_PRESENTATION_SECONDS))
            avatar.frame_ratio = max(1.0, seconds / self._SIM_DT_S / n_frames)
        try:
            avatar.play_animation("take_from_human")
        finally:
            if saved is not None:
                avatar.frame_ratio = saved

    def _presentation_clip_frames(self):
        md = (getattr(self.avatar, "motion_data", None) or {}).get("take_from_human")
        if md is None:
            try:
                from ..avatar.controller import AvatarController
                full = AvatarController.full_motion_data
                if full is None and AvatarController.full_motion_data_path:
                    import pickle
                    with open(AvatarController.full_motion_data_path, "rb") as f:
                        AvatarController.full_motion_data = full = pickle.load(f)
                md = (full or {}).get("take_from_human")
            except Exception:
                md = None
        try:
            return int(len(md["trans"])) if md is not None else None
        except Exception:
            return None

    def _eval_at_step(self, step):
        trigger = getattr(self, "_eval_trigger_step", None)
        if trigger is not None:
            if self._blind_human_started_sim_step is None:
                if int(step) >= int(trigger):
                    self._start_blind_presentation()
            elif self._blind_human_done_sim_step is None and self.avatar.spare():
                # Metric only (policy-step resolution): presentation finished.
                self._blind_human_done_sim_step = int(getattr(self, "_blind_sim_step", 0))
        self._update_handover()

    # ------------------------------------------------------------------
    # Eval-mode handover (ticked once per policy step)
    # ------------------------------------------------------------------
    def _handover_enabled(self) -> bool:
        return bool(self.config.get("eval_mode", False)) and self.avatar is not None

    def _object_center(self) -> np.ndarray:
        return np.asarray(self._get_object_world_center(
            self.obj_actor, self.obj_name, self.obj_model_id), dtype=float).ravel()[:3]

    def _palm(self) -> np.ndarray:
        return np.asarray(
            self.avatar.robot.get_palm_center(hand_id=1), dtype=float).ravel()[:3]

    def _update_handover(self):
        """Attach the object to the human's hand when the robot releases it
        at the presented palm.  The human never moves toward the robot."""
        if not self._handover_enabled() or getattr(self, "_handover_taken", False):
            return
        if getattr(self, "obj_actor", None) is None:
            return
        try:
            center = self._object_center()
            grip = float(self.robot.get_arm("right").gripper_val)
            tcp = robot_tcp(self, "right")
            in_gripper = (
                float(np.linalg.norm(tcp - center)) < float(self.HANDOVER_GRIP_TCP_M)
                and self._object_still_held()
            )
        except Exception:
            return
        grip_min = getattr(self, "_handover_grip_min", None)
        if grip_min is not None and grip >= grip_min + float(self.HANDOVER_OPEN_DELTA):
            # One release event per grip: released away from the presented
            # palm (or before the presentation finished), the object is
            # dropped; it can be picked up again and re-delivered.
            self._handover_grip_min = None
            dist = float(np.linalg.norm(center - self._palm()))
            self._handover_last_release_dist_m = dist
            if self._eval_presentation_done() and dist <= float(self.HANDOVER_RADIUS_M):
                self._handover_taken = True
                self._handover_sim_step = int(getattr(self, "_blind_sim_step", 0))
                self._handover_dist_m = dist
                self._start_handover_glide()
            return
        if in_gripper:
            self._handover_grip_min = grip if grip_min is None else min(grip_min, grip)
        else:
            self._handover_grip_min = None

    def _hand_grasp_waypoints(self, start: np.ndarray) -> list:
        """Glide path from ``start`` into the palm-down hand's grasp pose:
        out past the fingertips at the start height, down to the grasp
        height, then back under the fingers."""
        _, R = self.avatar.robot._get_hand_frame(1)
        fingers = np.asarray(R[:, 1], dtype=float)
        normal = np.asarray(R[:, 2], dtype=float)
        # Palm side = the side facing down for the presented (palm-down) hand.
        palm_side = normal if normal[2] <= 0.0 else -normal
        knuckles = self._palm()
        half = self._object_half_height()
        grasp = (knuckles + fingers * float(self.HANDOVER_GRASP_ALONG_FINGERS_M)
                 + palm_side * (half + float(self.HANDOVER_GRASP_PALM_GAP_M)))
        tip_out = float(self.HANDOVER_FINGER_LENGTH_M) + half + 0.015
        beyond_hi = knuckles + fingers * tip_out
        beyond_hi[2] = max(float(start[2]), float(beyond_hi[2]))
        beyond_lo = beyond_hi.copy()
        beyond_lo[2] = float(grasp[2])
        return [np.asarray(start, dtype=float), beyond_hi, beyond_lo, grasp]

    def _object_half_height(self) -> float:
        try:
            aabb = np.asarray(to_numpy(
                getattr(self.obj_actor, "entity", self.obj_actor).get_AABB()), dtype=float)
            return float(0.5 * (aabb.reshape(-1, 2, 3)[0][1, 2] - aabb.reshape(-1, 2, 3)[0][0, 2]))
        except Exception:
            return 0.03

    def _start_handover_glide(self):
        entity = getattr(self.obj_actor, "entity", self.obj_actor)
        start = np.asarray(to_numpy(entity.get_pos()), dtype=float).ravel()[:3]
        # Waypoints are for the object centre; glide the entity origin.
        offset = start - self._object_center()
        pts = [p + offset for p in self._hand_grasp_waypoints(self._object_center())]
        seg = np.array([np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])])
        self._handover_glide = {"pts": pts, "cum": np.concatenate([[0.0], np.cumsum(seg)]),
                                "k": -int(self.HANDOVER_GLIDE_DELAY_SIM_STEPS),
                                "n": max(1, int(self.HANDOVER_GLIDE_SIM_STEPS))}

    def _glide_point(self, glide, u: float) -> np.ndarray:
        cum, pts = glide["cum"], glide["pts"]
        total = float(cum[-1])
        if total <= 1e-9:
            return pts[-1].copy()
        d = (3 * u * u - 2 * u * u * u) * total        # smoothstep along the path
        i = int(min(np.searchsorted(cum, d, side="right") - 1, len(pts) - 2))
        f = (d - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
        return pts[i] + (pts[i + 1] - pts[i]) * float(np.clip(f, 0.0, 1.0))

    def _advance_handover_glide(self):
        glide = self._handover_glide
        entity = getattr(self.obj_actor, "entity", self.obj_actor)
        glide["k"] += 1
        u = min(1.0, max(0.0, glide["k"] / glide["n"]))
        entity.set_pos(self._glide_point(glide, u).astype(float))
        try:
            entity.set_dofs_velocity(np.zeros(6, dtype=float))
        except Exception:
            pass
        if u >= 1.0:
            self._handover_glide = None
            self.avatar.attach_object_to_hand(entity, hand_id=1)
            self._handover_done = True
            self._handover_glide_end_sim_step = int(getattr(self, "_blind_sim_step", 0))

    def _eval_pre_policy_enabled(self) -> bool:
        return bool(self.config.get("eval_avatar_pre_policy", self.EVAL_AVATAR_PRE_POLICY))

    def _eval_presentation_done(self) -> bool:
        started = (getattr(self, "_eval_trigger_step", None) is None
                   or getattr(self, "_blind_human_started_sim_step", None) is not None)
        return bool(started and self.avatar.spare())

    def _mark_robot_interaction(self):
        """Record when the robot plans the human-dependent (delivery) leg."""
        if not bool(self.config.get("eval_mode", False)) or self.avatar is None:
            return
        # Full-state eval expert only (no-op otherwise): wait for the open hand.
        self._await_cue("presentation", self._eval_presentation_done)
        self._blind_robot_interaction_sim_step = int(getattr(self, "_blind_sim_step", 0))
        if getattr(self, "_eval_trigger_step", None) is None:
            started = True  # presentation queued at reset (VLA default path)
        else:
            started = getattr(self, "_blind_human_started_sim_step", None) is not None
        self._blind_human_ready_before_robot = bool(started and self.avatar.spare())

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not self._eval_pre_policy_enabled()
        ):
            return self.get_obs()
        while not self.avatar.spare():
            self.step_sim()
        for _ in range(int(self.config.get("eval_avatar_post_warmup_settle_steps", 50))):
            self.step_sim()
        self._clear_avatar_collision_state()
        return self.get_obs()

    def play_once(self) -> bool:
        if self.avatar is not None:
            self.avatar.play_animation("take_from_human")
            while not self.avatar.spare():
                self.step_sim()

        for _ in range(50):
            self.step_sim()
        # The opening human-only "take_from_human" presentation can move
        # randomized avatar hands/forearms through the parked robot before
        # the robot has started the delivery.  That pre-roll is scene setup,
        # not a robot-avatar safety violation for the scripted rollout.
        self._clear_avatar_collision_state()

        method = self._grasp_method()
        if method == "annotated":
            return self._play_once_annotated()
        else:
            return self._play_once_primitive()

    def _play_once_annotated(self) -> bool:
        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        obj_pose = self.obj_actor.get_pose()
        self._z0 = float(obj_pose.p[2])
        self._stage_failed = None

        with open(
            ASSETS_PATH / "objects" / self.obj_name
            / f"model_data{self.obj_model_id}.json"
        ) as f:
            raw_scale = json.load(f).get("scale", 1.0)
        object_scale = float(
            raw_scale[0] if isinstance(raw_scale, (list, tuple)) else raw_scale
        )

        self.open_gripper(arm_tag)
        result = self.select_and_execute_grasp(
            self.obj_name, obj_pose, arm_tag,
            max_candidates=40, model_id=self.obj_model_id,
            robot_type="franka",
            object_scale=object_scale,
        )
        if result is None:
            self._stage_failed = "select_grasp"
            return False

        grasp_link, pre_link, grasp = result
        self._grasp_used = grasp.name
        grasp_tcp = grasp.to_world(obj_pose, object_scale)

        # Boost the finger PD (prevents mid-carry drops)
        # to the mixin-standard carry profile before closing — the stock MJCF
        # finger force range is too weak for the carry.
        self.configure_finger_pd(
            arm_tag, force_limit=200.0, kp=9000.0, kv=250.0,
        )
        close_target = self.OBJECT_CLOSE_TARGET.get(self.obj_name)
        if close_target is None:
            self.close_gripper(arm_tag)
        else:
            self.robot.set_gripper(close_target, arm_tag)
            for _ in range(60):
                self.step_sim()

        # Raw move_and_execute on the carry legs is UNSEEDED
        # mplib plan_pose (IK branches can span
        # j1 -2.33..+2.38 with RRT picking one arbitrarily), replayed at 2x
        # the planned speed.  The pure +z lift rides the screw planner; the
        # deliver leg goes through the guarded `_move_seeded` (j7 unwrap +
        # branch gate + post-fallback reseed).
        lift_tcp = Pose(grasp_tcp.p + np.array([0.0, 0.0, 0.12]), grasp_tcp.q)
        lift_link = tcp_to_link_pose(lift_tcp, tcp_offset)
        if not self._move_screw_to_link(lift_link.to_pose7(), arm_tag):
            self._stage_failed = "move_lift"
            return False

        self._mark_robot_interaction()
        deliver_link = self._compute_deliver_pose(
            arm_tag, keep_current_orientation=True,
        )
        if self._move_seeded(deliver_link.to_pose7(), arm_tag) is None:
            self._stage_failed = "move_deliver"
            return False

        if self._handover_enabled():
            return self._handover_release(arm_tag, top_down=False)
        for _ in range(self.FINAL_HOLD_SETTLE_STEPS):
            self.step_sim()

        final_pose = self.obj_actor.get_pose()
        self._z1 = float(final_pose.p[2])
        self._held = self._object_still_held()
        return self._held

    def _play_once_primitive(self) -> bool:
        arm_tag = "right"
        obj_pose = self.obj_actor.get_pose()
        self._z0 = float(obj_pose.p[2])

        pick = PickSpec(
            get_center=lambda: self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id
            ),
            radius=self._get_object_cross_section_radius(self.obj_name, self.obj_model_id),
            label=self.obj_name,
        )
        deliver_pos = self._deliver_pos(arm_tag)
        transport_z = max(float(deliver_pos[2]), self.TABLE_TOP_Z + 0.20) + 0.05
        place = PlaceSpec(
            pos=deliver_pos,
            label="hand",
            transport_z=transport_z,
            release=False,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False
        for _ in range(self.FINAL_HOLD_SETTLE_STEPS):
            self.step_sim()

        final_pose = self.obj_actor.get_pose()
        self._z1 = float(final_pose.p[2])
        self._held = self._object_still_held()
        return self._held

    # ------------------------------------------------------------------
    # Blind ("no intent") expert: play_once's robot motions on the robot's
    # own clock.  The human is driven only by the eval hooks above; the
    # delivery target is the palm observed when the delivery leg is planned.
    # ------------------------------------------------------------------
    def play_full_eval_once(self) -> bool:
        """Full-state expert under the eval-mode human: play_blind_once's
        robot script, waiting for the open hand before the delivery leg."""
        self._full_eval_active = True
        try:
            return self.play_blind_once()
        finally:
            self._full_eval_active = False

    def _await_cue(self, label, ready):
        return await_cue(self, ready, label)

    def play_blind_once(self) -> bool:
        for _ in range(50):
            self.step_sim()
        if self._grasp_method() == "annotated":
            return self._play_once_annotated()
        return self._play_blind_primitive()

    def _play_blind_primitive(self) -> bool:
        """``_play_once_primitive`` split at the lift.

        The pick legs use the same specs as the full-state path (transport
        height from the palm observed at pick time); the delivery target is
        re-read from the live palm after the lift, when the delivery leg is
        planned.  The delivery legs mirror ``pick_and_place``'s no-release
        place phase.
        """
        arm_tag = "right"
        obj_pose = self.obj_actor.get_pose()
        self._z0 = float(obj_pose.p[2])

        pick = PickSpec(
            get_center=lambda: self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id
            ),
            radius=self._get_object_cross_section_radius(self.obj_name, self.obj_model_id),
            label=self.obj_name,
        )
        pick_time_pos = self._deliver_pos(arm_tag)
        transport_z = max(float(pick_time_pos[2]), self.TABLE_TOP_Z + 0.20) + 0.05
        lift_only = PlaceSpec(
            pos=pick_time_pos,
            label="hand",
            transport_z=transport_z,
            release=False,
            stop_after_lift=True,
        )
        if not self.pick_and_place(pick, lift_only, arm_tag):
            return False

        self._mark_robot_interaction()
        deliver_pos = self._deliver_pos(arm_tag)
        above_pos = np.array([deliver_pos[0], deliver_pos[1], transport_z])
        # The observed palm may be somewhere the arm cannot reach (e.g. the
        # hand still hanging at the human's side).  mplib's plan_screw toward
        # an unreachable pose can return a ~200k-step "success" trajectory,
        # so check IK for both delivery poses first; no IK is an honest
        # failure (no retry, no waiting).
        if self._handover_enabled():
            # Eval mode: rise above the demo delivery point to the handover
            # height (above the offered fingers), then go level to the
            # handover over the palm; the demo's final hold pose is skipped.
            # The highest handover height with IK for both poses is used.
            # Check IK against the live human (the cloud was
            # last refreshed during the pick, with the hand elsewhere).
            try:
                self._refresh_planner_obstacles(arm_tag)
            except Exception as exc:  # planner/collider unavailable: keep the old cloud
                print(f"[deliver] obstacle refresh skipped: {exc!r}", flush=True)
            for standoff, dz in self.HANDOVER_OBJ_CANDIDATES_M:
                target = self._handover_tcp_target(arm_tag, standoff=standoff, dz=dz)
                approach = np.array([above_pos[0], above_pos[1],
                                     max(float(above_pos[2]), float(target[2]))])
                # The target is reached from the approach pose: seed its IK
                # with the approach solution (the lift pose is a far seed).
                q_app = self._deliver_pose_has_ik(
                    approach, arm_tag, seed=self.robot.get_arm(arm_tag).get_arm_qpos())
                ok = [q_app is not False, q_app is not False and self._deliver_pose_has_ik(
                    target, arm_tag, seed=q_app) is not False]
                print(f"[deliver] handover standoff={standoff} dz={dz}: approach ik={ok[0]} "
                      f"{np.round(approach, 3).tolist()} target ik={ok[1]} "
                      f"{np.round(target, 3).tolist()}", flush=True)
                if all(ok):
                    break
            else:
                self._stage_failed = "deliver_ik"
                self.plan_success = False
                return False
            self._move_screw(approach, arm_tag)
            for _ in range(50):
                self.step_sim()
            return self._handover_release(arm_tag, top_down=True, target=target)
        for pos in (above_pos, np.asarray(deliver_pos, dtype=float)):
            if not self._deliver_pose_has_ik(pos, arm_tag):
                self._stage_failed = "deliver_ik"
                self.plan_success = False
                return False
        self._move_screw(above_pos, arm_tag)
        for _ in range(100):
            self.step_sim()
        if abs(float(deliver_pos[2]) - transport_z) > 1e-4:
            self._move_screw(np.asarray(deliver_pos, dtype=float), arm_tag)
            for _ in range(50):
                self.step_sim()
        for _ in range(100):
            self.step_sim()

        for _ in range(self.FINAL_HOLD_SETTLE_STEPS):
            self.step_sim()
        final_pose = self.obj_actor.get_pose()
        self._z1 = float(final_pose.p[2])
        self._held = self._object_still_held()
        return self._held

    def _handover_tcp_target(self, arm_tag: str, standoff: float | None = None,
                             dz: float | None = None) -> np.ndarray:
        """TCP position that puts the held object's centre at the handover
        point: ``standoff`` from the live palm toward the robot (xy) and
        ``dz`` above it (defaults: the first HANDOVER_OBJ_CANDIDATES_M)."""
        if standoff is None or dz is None:
            standoff, dz = self.HANDOVER_OBJ_CANDIDATES_M[0]
        arm = self.robot.get_arm(arm_tag)
        palm = self._palm()
        to_robot = np.asarray(arm.origin_pose.p, dtype=float)[:2] - palm[:2]
        n = float(np.linalg.norm(to_robot))
        to_robot = to_robot / n if n > 1e-6 else np.array([0.0, -1.0])
        obj_goal = palm.copy()
        obj_goal[:2] += to_robot * float(standoff)
        obj_goal[2] += float(dz)
        tcp = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        return obj_goal + (tcp - self._object_center())

    def _handover_release(self, arm_tag: str, top_down: bool, target=None) -> bool:
        """Eval-mode end of the delivery: bring the object over the live palm,
        open the gripper, let the human take it, then lift the gripper away.
        Success is scored by check_success (the human holds the object)."""
        arm = self.robot.get_arm(arm_tag)
        if target is None:
            target = self._handover_tcp_target(arm_tag)
        print(f"[deliver] handover: palm={np.round(self._palm(), 3).tolist()} "
              f"obj={np.round(self._object_center(), 3).tolist()} "
              f"tcp_goal={np.round(target, 3).tolist()}", flush=True)
        if top_down:
            if not self._deliver_pose_has_ik(target, arm_tag):
                self._stage_failed = "handover_ik"
                self.plan_success = False
                return False
            self._move_screw(target, arm_tag)
        else:
            q = np.asarray(arm.get_ee_pose()[3:7], dtype=float)
            link = tcp_to_link_pose(Pose(target, q), arm.tcp_offset)
            if not self._move_screw_to_link(link.to_pose7(), arm_tag):
                self._stage_failed = "handover_move"
                return False
        for _ in range(50):
            self.step_sim()
        self.set_gripper(1.0, arm_tag, 100)
        for _ in range(25):
            self.step_sim()
        if not getattr(self, "_handover_taken", False):
            self._stage_failed = "handover_not_taken"
            print(f"[deliver] handover not taken: obj-palm "
                  f"{np.linalg.norm(self._object_center() - self._palm()):.3f} m",
                  flush=True)
            return False
        # Retreat: up out of the object, then back toward the robot.  A leg
        # without IK is skipped.  The object is already the human's, so a
        # failed retreat plan does not undo the delivery (plan_success is
        # restored; the motion itself is still collision-scored).
        plan_ok = self.plan_success
        q = np.asarray(arm.get_ee_pose()[3:7], dtype=float)
        pos = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        to_robot = np.asarray(arm.origin_pose.p, dtype=float)[:2] - pos[:2]
        to_robot /= float(np.linalg.norm(to_robot)) + 1e-9
        def leg_link7(leg):
            return tcp_to_link_pose(
                self._top_down_tcp(leg) if top_down else Pose(leg, q), arm.tcp_offset,
            ).to_pose7()

        def has_ik(leg):
            link7 = leg_link7(leg)
            ik = arm.planner.solve_ik(
                np.asarray(arm.get_arm_qpos(), dtype=float), link7, log=False)
            return endpoint_reaches_goal(arm.planner, ik, link7)

        legs = []
        for dz in (float(self.HANDOVER_RETREAT_DZ_M), 0.6 * float(self.HANDOVER_RETREAT_DZ_M)):
            up = pos + np.array([0.0, 0.0, dz])
            if has_ik(up):
                legs.append(up)
                break
        start = legs[-1] if legs else pos
        legs.append(start + np.array([*(to_robot * float(self.HANDOVER_RETREAT_BACK_M)), 0.0]))
        for leg in legs:
            if not has_ik(leg):
                print(f"[deliver] retreat leg {np.round(leg, 3).tolist()} has no IK; "
                      "skipped", flush=True)
                break
            if top_down:
                self._move_screw(leg, arm_tag)
            else:
                self._move_screw_to_link(leg_link7(leg), arm_tag)
            if not self.plan_success:
                print("[deliver] retreat plan failed; robot stays", flush=True)
                self._handover_retreat_failed = True
                break
        self.plan_success = plan_ok
        for _ in range(int(self.HANDOVER_GLIDE_DELAY_SIM_STEPS)
                       + int(self.HANDOVER_GLIDE_SIM_STEPS) + 25):
            if getattr(self, "_handover_done", False):
                break
            self.step_sim()
        for _ in range(25):
            self.step_sim()
        self._held = bool(avatar_holds(self, self.obj_actor))
        return self._held

    def _deliver_pose_has_ik(self, tcp_pos, arm_tag: str, seed=None):
        """Top-down IK check (mplib).  ``seed`` defaults to the current arm
        qpos.  Returns False, or True / the solution when ``seed`` is given."""
        arm = self.robot.get_arm(arm_tag)
        link_pose7 = tcp_to_link_pose(
            self._top_down_tcp(tcp_pos), arm.tcp_offset,
        ).to_pose7()
        q0 = np.asarray(arm.get_arm_qpos() if seed is None else seed, dtype=float)
        # Near the reach limit mplib's 20-sample IK misses
        # reachable handover poses at random (such deliver_ik
        # failures solve on a re-query); re-query before declaring no IK.
        measured = np.asarray(arm.get_arm_qpos(), dtype=float)
        for attempt in range(int(self.HANDOVER_IK_ATTEMPTS)):
            start = q0 if attempt % 2 == 0 else measured
            result = arm.planner.solve_ik(start, link_pose7, log=False)
            ok = endpoint_reaches_goal(arm.planner, result, link_pose7)
            if ok:
                break
        if not ok or seed is None:
            return bool(ok)
        return np.asarray(result.position[-1], dtype=float)[:arm.n_arm]

    HANDOVER_IK_ATTEMPTS = 4

    def check_success(self) -> bool:
        # `self._held` is set only by the scripted `play_once` path. Under
        # policy eval `play_once` never runs, so the attribute is absent and
        # a `getattr(..., False)` gate would make `check_success` return
        # False for *every* eval episode regardless of the policy. Gate on
        # the live hold check (`_object_still_held`) instead — it recomputes
        # the same thing from the current scene and works in both flows.
        if self._handover_enabled():
            # Eval mode: a real handover.  The robot released the object at
            # the presented palm and the human now holds it.
            return bool(
                self.plan_success
                and getattr(self, "_handover_done", False)
                and avatar_holds(self, self.obj_actor)
                and not self.avatar_collision_summary().get("any_collision")
            )
        # Scripted demo collection (eval_mode off): the demo choreography
        # ends holding the object near the presented palm.
        if not self.plan_success or not self._object_still_held():
            return False
        if self.avatar is None or not self._eval_presentation_done():
            return False
        if self.avatar_collision_summary().get("any_collision"):
            return False
        metrics = self._compute_target_metrics()
        if metrics is None:
            return False
        if metrics["dist_3d"] > float(self.SUCCESS_DIST_3D_M):
            return False
        if metrics["dz"] < float(self.SUCCESS_DZ_MIN_M):
            return False
        return True

    def evaluate(self) -> dict:
        out = super().evaluate()
        out["object_name"] = self.obj_name
        out["object_model_id"] = int(self.obj_model_id)
        out["grasp_method"] = self._grasp_method()
        if hasattr(self, "_stage_failed"):
            out["stage_failed"] = self._stage_failed
        if hasattr(self, "_grasp_used"):
            out["grasp_used"] = self._grasp_used
        if bool(self.config.get("eval_mode", False)) and self.avatar is not None:
            started = getattr(self, "_blind_human_started_sim_step", None)
            trigger = getattr(self, "_eval_trigger_step", None)
            if getattr(self, "_full_eval_cue_waits", None) is not None:
                out["full_eval_cue_waits"] = self._full_eval_cue_waits
            out["handover"] = {
                "done": bool(getattr(self, "_handover_done", False)),
                "sim_step": getattr(self, "_handover_sim_step", None),
                "obj_palm_dist_m": getattr(self, "_handover_dist_m", None),
                "last_release_obj_palm_dist_m": getattr(
                    self, "_handover_last_release_dist_m", None),
                "radius_m": float(self.HANDOVER_RADIUS_M),
                "retreat_failed": bool(getattr(self, "_handover_retreat_failed", False)),
                "glide_end_sim_step": getattr(self, "_handover_glide_end_sim_step", None),
            }
            out["blind_intent"] = {
                "trigger_step": trigger,
                "human_started_sim_step": started,
                "human_done_sim_step": getattr(self, "_blind_human_done_sim_step", None),
                "human_presentation_done": bool(
                    (trigger is None or started is not None) and self.avatar.spare()
                ),
                "robot_interaction_sim_step": getattr(
                    self, "_blind_robot_interaction_sim_step", None),
                "human_ready_before_robot": getattr(
                    self, "_blind_human_ready_before_robot", None),
                "sim_steps_total": int(getattr(self, "_blind_sim_step", 0)),
            }
        try:
            out["object_bottom_z"] = self._object_bottom_z()
            out["object_table_clearance"] = self._object_table_clearance()
            out["object_table_clearance_threshold"] = self.SUCCESS_TABLE_CLEARANCE_M
            out["object_still_held"] = self._object_still_held()
        except Exception as e:
            out["object_hold_metric_error"] = str(e)
        return out
