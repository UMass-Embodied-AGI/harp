"""Shared layout, object, and manipulation settings for the categorize family."""

import numpy as np

from ...object_catalog import object_set_pairs


class CategorizeTaskProfile:
    """Defaults inherited by the core and overridden by public scenarios."""

    # Opt-in until matched original-process-shape validation is complete.
    SEAL_MEASURED_GRASP = False
    # Nominal inward actuator effort per finger during the seal-only hold.
    # Position/slew bounds still apply; this is not a measured contact force.
    SEAL_HOLD_FORCE_N = 7.5

    use_avatar = False

    # Family profile. Public scenarios select the cooperative free-XY layout
    # or the interrupt/neutral column layout without inheriting one another.
    _CATEGORIZE_COLUMN_LAYOUT = False
    _CATEGORIZE_ROBOT_Y = -0.35
    _CATEGORIZE_FINGER_KV = 250.0
    _CATEGORIZE_OPEN_LIDS = True
    # Avatar stands ~0.22 m behind the table edge (y=0.45) so the body is not
    # clipping the table and the motion is visually comfortable.  With the
    # body at y=0.67 the right shoulder sits at world y≈0.69 and the left at
    # the mirror.  Each loose object on the avatar side (free XY in
    # x∈[-0.32,+0.32], y∈[+0.32,+0.42]) is reached by the closer hand
    # (see `_avatar_pick_and_place`): right hand for x≤0, left hand for x>0.
    # Worst-case shoulder→target ≈ 0.39 m at the far-Y corners — well inside
    # the ~0.65 m palm reach.
    avatar_init_pos = np.array([0.0, 0.67, -0.18])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Table cropped toward the robot side but with a small band behind the
    # baskets on the avatar side to hold the avatar-side loose objects
    # (half_size_y=0.45, dy=0.0 → y ∈ [-0.45, 0.45]).  Baskets at y=0.22,
    # avatar-side loose objects at y=0.40 (6 cm beyond basket rear edge).
    table_offset = np.array([0.0, 0.0])
    _TABLE_HALF_SIZE_Y = 0.45

    # VLA primary view: near top-down over the full 1.60 x 0.90 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, 0.005, 2.20],
        "forward": [0.0, -0.005, -1.30],
    }]

    recording_camera_pos = [1.1, -0.3, 1.7]
    recording_camera_lookat = [0.0, 0.00, 0.75]
    # Top-down view of table — so we can see inside the baskets.
    side_camera_pos = [0.0, -0.05, 2.2]
    side_camera_lookat = [0.0, -0.05, 0.80]

    # Pure top-down gripper orientation — Franka 7-DOF has no singularity here.
    _R_DOWN = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    MIN_TCP_CLEARANCE = 0.0

    # Push an inflated point-cloud of all baskets + sort_targets to mplib
    # before each transit / descent.  Required for the interrupt variant's
    # column transits.  Without it, cross-table transits can sweep the arm
    # wrist over the remaining objects and knock the woodenblock off the
    # table edge to y=-0.85 (floor), making the final pick unrecoverable.
    # The chase fix only catches small displacements, not floor-knock.
    # Inflation forces plan_pose to detour around all spawned objects; on
    # the longer diagonal transits of the random-XY layout this can push
    # plan_pose RRT into bad IK branches, which the branch-flip joint-delta
    # heuristic + chase fix are meant to handle.
    _USE_OBSTACLE_INFLATION = True

    # Override DEFAULT_AVATAR_POOL with a fixed 3-avatar subset so categorize
    # results stay comparable across evaluations.  Celebrity models are
    # excluded from the benchmark store entirely (envs/avatar/model_list.py).
    AVATAR_POOL = [
        "avatars/models/custom_Adrian_Keller.glb",
        "avatars/models/custom_Clara_Voss.glb",
        "avatars/models/custom_Malik_Okoro.glb",
    ]

    # Per-object Z offset (positive = grasp HIGHER than center).  Replaces the
    # default `obj_center[2] - radius * 0.3` formula for objects that need a
    # specific grasp height.  Seal is a tall slender cylinder.  A higher
    # offset (+0.022) put the grip at the seal's top and the heavier bottom
    # pivoted out under transit g-load; +0.015 grips the mid-upper body
    # (~7 mm below top).
    OBJECT_GRASP_Z_OFFSET = {
        "100_seal": 0.015,
        # Tall jar body: without a lift the radius scoop grips the lower
        # third near the base.  +0.035 reached the tapered lid and slipped
        # on lift (dz -0.017); the catalog's
        # +0.02 lands mid-body and is used here explicitly.
        "031_jam-jar": 0.020,
    }

    # Compensate for PD controller Z overshoot — command this far below the
    # true target so the actual TCP lands at the desired height.
    PD_Z_COMPENSATION = 0.03

    # Route the robot pick-side approach through the shared
    # ``BaseTask.select_and_execute_grasp`` helper using the manually
    # annotated YAML grasps.  The annotations live in mesh-origin frame,
    # so the helper drops the TCP at the right grasping point even when
    # mesh origin differs from bbox centre.  Trust the YAML and skip the
    # "chase" toward bbox centre on this path: a chase at the same Z
    # plows the gripper through the object.
    USE_YAML_GRASPS = True

    # All robot-side picks route through BaseTask.select_and_execute_grasp.
    # The wooden block asset has center top-down Franka grasps, so no
    # object needs the legacy synthetic fallback.
    YAML_GRASP_SKIP = set()

    # Subclasses can prefer task-validated grasp names before falling back to
    # the generic ranker.  Cooperative keeps the generic standard-helper order:
    # the full preferred object set regressed cooperative rollouts, but the
    # wooden block should avoid body-center yaw0/yaw90 grasps that can carry
    # the block loosely and drop it during transport.
    PREFERRED_GRASP_NAMES = {
        "035_apple": (
            "manual_center_p0_yaw0",
            "manual_center_p0_yaw2",
        ),
        "086_woodenblock": (
            "manual_topdown_yaw45_center",
            "manual_topdown_yaw45_higher",
        ),
    }
    # Object-specific grasp allowlists.  These are exclusive: once an object
    # is listed here, the generic YAML ranker must not fall back to any other
    # annotation.
    EXCLUSIVE_GRASP_NAMES = {
        # The milk carton's folded top is slippery, so all top-down annotations
        # are banned.  These four candidates approach its upright body from the
        # side and were validated through the full categorize placement path.
        "038_milk-box": (
            "grasp_065",  # validated full categorize pick/place
            "grasp_058",  # validated full categorize pick/place
            "grasp_005",  # validated full categorize pick/place
            "grasp_078",  # validated full categorize pick/place
        ),
        # The radius path descends below the cube's equator.  At some table
        # locations its lateral pose correction contacts a lower corner and
        # chases the cube across the table.  Use the asset's face-centred,
        # top-down body grasps instead; try both closing axes and finger flips
        # before giving up.
        "073_rubikscube": (
            "manual_topdown_close_x",
            "manual_topdown_close_x_flipfinger",
            "manual_topdown_close_y",
            "manual_topdown_close_y_flipfinger",
        ),
    }
    YAML_ONLY_GRASP_LABELS = frozenset(EXCLUSIVE_GRASP_NAMES)

    # Staged opt-in until the unchanged five-episode release shards validate
    # the continuation gate. It never expands the four authored side grasps.
    MILK_CONTINUATION_PREFLIGHT = False

    # The stapler rests with its long dimension in the yaw-0 jaw direction.
    # Closing there pushes on the two ends and ejects it.  Rotate the top-down
    # parallel jaws by 90 degrees so they close across the short dimension;
    # yaw 0 is deliberately no longer a candidate for this object.
    RADIUS_GRASP_YAW = {
        "048_stapler": 0.5 * np.pi,
    }

    # Apple pose selection is workspace-sensitive.  The authored yaw-0 pose is
    # unreachable on the left side, where the radius pose is validated.  On
    # the right, the lower underhand grasp retains mid-workspace carries while
    # the center-height grasp is stable at the far edge.  These thresholds are
    # evaluated from the live object center, not the episode seed.
    APPLE_YAML_MIN_X = 0.0
    APPLE_CENTER_MIN_X = 0.20
    APPLE_CENTER_GRASP_NAMES = (
        "manual_center_p0_yaw0",
        "manual_underhand_p0_yaw0",
    )
    APPLE_UNDERHAND_GRASP_NAMES = (
        "manual_underhand_p0_yaw0",
        "manual_center_p0_yaw0",
    )

    # Disabled by default.  A column-aligned cooperative spawn was tested but
    # regressed the rollout, so keep the original free-XY layout here.
    ROBOT_MATCH_BASKET_COLUMNS = False
    ROBOT_COLUMN_X_JITTER = 0.015
    ROBOT_COLUMN_Y_RANGE = (-0.16, -0.08)

    # Eval/testbed mode: policies call take_action(action)->obs while the
    # avatar begins its cooperative sorting sequence at a seeded random policy
    # step.  This keeps baseline rollouts from depending on scripted play_once.
    EVAL_TRIGGER_STEP_MIN = 40
    EVAL_TRIGGER_STEP_MAX = 100

    # Objects are sampled per-episode from the shared catalog pool
    # ``tabletop_pick_pool`` (NUM_CATEGORIES distinct draws, one per basket).
    # CATEGORIES holds a default subset for any pre-load reference; load_actors
    # re-samples ``self._cats`` each episode.
    OBJECT_SET = "tabletop_pick_pool"
    NUM_CATEGORIES = 3

    # Mesh loading preserves the authored Y-up frame. Map local +Y to world +Z
    # so jars, blocks, and seals stand on their intended bases.
    SPAWN_UPRIGHT_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)

    # Grasp strategy.  The per-object YAML grasp annotations (``_pick_and_place_yaml``)
    # are object-local and angled, so under Genesis 1.2.0 their approach sweeps
    # light objects out of the jaws (stapler/block knocked 8–45 cm) or closes on
    # air.  The orientation-agnostic top-down radius grasp from
    # ``TopDownPickPlaceMixin`` — the exact path that lifts the same
    # apple/woodenblock/seal/rubikscube 3/3 in ``put_object_cabinet`` — descends
    # straight down onto the object's live-AABB centre and is far more robust.
    USE_RADIUS_GRASP = True

    # Objects whose resting AABB is taller than this (m) are skipped by the
    # radius grasp. With the authored-frame upright spawn the objects stand on
    # their base, so a top-down grasp handles them up to ~10 cm tall (jar
    # 9.5cm, block 7cm,
    # seal/rubik/cube all pass + lift+place).  Only the 12cm milk-box carton is
    # too tall to purchase top-down, so it is left standing on the table.
    RADIUS_MAX_GRASP_HEIGHT = 0.10

    # Object-level emergency bans for the radius path.  Keep this empty: an
    # object sampled as a required task target must not be silently left on the
    # table.  Apple model 1 is handled by the task-local 0.58 close below; the
    # task-native grasp/lift/hold probe retained it with <0.5 mm drift.
    RADIUS_SKIP_LABELS = set()

    # Categorize-local spawn friction overrides (else catalog friction).
    # Optional (object_name, model_id) keys take priority over name-only keys,
    # allowing a model-specific experiment without changing sibling models.
    # The 086_woodenblock@1 catalog friction (5.0, tuned for put_object_cabinet's
    # radius grasp) makes the block's contact solver blow up when it is RELEASED
    # into the basket (z→-40 / NaN).  At 2.5 the release is stable and the block
    # still grips + lifts + places (2/3), so categorize uses the gentler value.
    SPAWN_FRICTION_OVERRIDE = {"086_woodenblock": 2.5}

    # Fixed, task-local contact correction from the tested held-friction
    # candidate. Keep the literal opt-in until fresh training pilots pass;
    # other models and the procedural cube retain their existing materials.
    PAYLOAD_FRICTION_V1_MODELS = frozenset({
        ("038_milk-box", 0), ("086_woodenblock", 1),
        ("035_apple", 1), ("073_rubikscube", 1),
        ("031_jam-jar", 0), ("048_stapler", 0),
    })
    PAYLOAD_FRICTION_V1_VALUE = 1.0

    # Categorize-local gripper close fraction overrides for the radius grasp
    # (0=closed, 1=open-ish).  Tuned for this pool's models + the below-equator
    # scoop; takes priority over the shared catalog ``close_value`` so cabinet
    # is not regressed.  Round/domed objects need a LOOSE cage (the scoop +
    # palm holds them; a tight close ejects them).
    RADIUS_CLOSE_OVERRIDE = {
        # apple@1 ~5.2 cm round: 0.35 (catalog) crush-ejects, 0.70 is too loose
        # to contact (fingers stop at 0.028 > 0.026 radius).  ~0.58 lets the
        # jaws touch + lightly squeeze so the below-equator scoop holds it.
        "035_apple": 0.58,
    }
    CATEGORIES = object_set_pairs(
        ["035_apple@1", "086_woodenblock@1", "100_seal@2"]
    )

    # Basket geometry (built from primitives in load_actors) + obstacle
    # inflation margins fed to mplib so RRT routes around basket walls.
    _BASKET_HALF_X = 0.096
    _BASKET_HALF_Y = 0.096
    _BASKET_HALF_Z = 0.056
    _BASKET_INNER_HALF_XY = 0.086
    _BASKET_FOOTPRINT_MARGIN = 0.002
    _OBSTACLE_MARGIN_XY = 0.020
    _OBSTACLE_MARGIN_Z = 0.010
    _OBSTACLE_RESOLUTION = 0.015
