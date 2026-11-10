"""Scene-free characterization contracts for the dump-bin family.

The assertions in this file describe observable task behavior rather than the
legacy mixin ownership of that behavior.  They are intentionally usable while
assist/interrupt behavior moves behind interaction controllers.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.object_catalog import get_entry, resolve_object_set
from envs.task_bases.dump_bin import DumpBin
from envs.tasks.dump_bin_contribution import DumpBinContribution
from envs.tasks.dump_bin_interruption import DumpBinInterruption


SEEDS = (0, 8, 42)


@pytest.mark.parametrize('z_scale', [.06, .09])
def test_bin_collision_support_matches_existing_visual_inner_floor(monkeypatch, z_scale):
    import trimesh
    from envs.task_bases import dump_bin as module
    task=DumpBin({})
    task.BIG_BIN_SCALE_Z=z_scale
    task.scene=SimpleNamespace(add_entity=lambda *a,**kw:object())
    monkeypatch.setattr(module,'gs',SimpleNamespace(
        morphs=SimpleNamespace(Mesh=lambda **kw:SimpleNamespace(**kw)),
        materials=SimpleNamespace(Rigid=lambda:object())))
    boxes=[]
    monkeypatch.setattr(module,'create_primitive',lambda scene,shape,pose,**kw:boxes.append((pose,kw)))
    task._load_bin()
    mesh=trimesh.load(module.ASSETS_PATH/'objects/011_dustbin/visual/base0.glb',
                      force='scene',process=False).dump(concatenate=True)
    centers=mesh.triangles_center
    normals=mesh.face_normals
    # The flat upward interior floor is distinct from the lower outside skin.
    mask=((normals[:,1]>.98)&(centers[:,1]>.17)&(centers[:,1]<.19)
          &(np.abs(centers[:,0])<.10/task.BIG_BIN_SCALE_XY)
          &(np.abs(centers[:,2])<.06/task.BIG_BIN_SCALE_XY))
    assert mesh.area_faces[mask].sum()>.1
    heights=task.big_bin_pose.p[2]+centers[mask,1]*z_scale
    floor_pose,options=boxes[0]
    half=np.asarray(options['size']['half_size'])
    assert abs((floor_pose.p[2]+half[2])-np.median(heights))<.0001
    assert 2*half[2]>=.05
    assert floor_pose.p[2]-half[2]<0
    assert options['is_static'] and not options['visual']
    assert len(boxes)==5
    assert task.BIG_BIN_FLOOR_THICKNESS==.010


class _FakeEntity:
    def __init__(self, pos, *, color=None):
        self._pos = np.asarray(pos, dtype=np.float64)
        self.color = color
        self.entity = self

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def set_quat(self, _quat):
        return None

    def set_dofs_velocity(self, _velocity):
        return None


def _legacy_dump_layout(task):
    """Copy the release scalar/vector draw and retry order."""

    n = int(task.num_objects)
    xs = np.random.uniform(*task.SPAWN_X_RANGE, n)
    ys = np.random.uniform(*task.SPAWN_Y_RANGE, n)
    attempts = 1
    min_spacing_sq = float(task.MIN_OBJECT_SPACING) ** 2
    for attempt in range(int(task.LAYOUT_SAMPLE_ATTEMPTS)):
        valid = all(
            (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 >= min_spacing_sq
            for i in range(n)
            for j in range(i + 1, n)
        )
        if valid:
            attempts = attempt + 1
            break
        xs = np.random.uniform(*task.SPAWN_X_RANGE, n)
        ys = np.random.uniform(*task.SPAWN_Y_RANGE, n)
        attempts = attempt + 2
    return xs, ys, attempts


@pytest.mark.parametrize("seed", SEEDS)
def test_base_layout_preserves_global_numpy_stream(seed):
    task = DumpBin({"hard": True})

    np.random.seed(seed)
    expected_xs, expected_ys, attempts = _legacy_dump_layout(task)
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual_xs, actual_ys = task._sample_cube_xys()
    actual_tail = np.random.random_sample(8)

    np.testing.assert_array_equal(actual_xs, expected_xs)
    np.testing.assert_array_equal(actual_ys, expected_ys)
    np.testing.assert_array_equal(actual_tail, expected_tail)
    if seed == 0:
        assert attempts == 2


@pytest.mark.parametrize("seed", SEEDS)
def test_base_catalog_target_allocation_preserves_global_numpy_stream(seed):
    task = DumpBin({"hard": True})
    keys = resolve_object_set(task.OBJECT_SET)

    np.random.seed(seed)
    expected_indices = np.random.choice(len(keys), size=task.num_objects, replace=False)
    expected = [get_entry(keys[int(index)]) for index in expected_indices]
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual = task.resolve_target_objects(task.num_objects, replace=False)
    actual_tail = np.random.random_sample(8)

    assert [(entry.key, entry.model_id) for entry in actual] == [
        (entry.key, entry.model_id) for entry in expected
    ]
    assert len({(entry.key, entry.model_id) for entry in actual}) == task.num_objects
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_dump_object_alias_forces_repeated_target_without_consuming_rng():
    original = {"hard": True, "dump_object_name": "031_jam-jar"}
    task = DumpBin(original)

    assert original == {"hard": True, "dump_object_name": "031_jam-jar"}
    assert task.config["object_name"] == "031_jam-jar"
    assert task.config["dump_object_name"] == "031_jam-jar"

    np.random.seed(113)
    expected_tail = np.random.random_sample(8)
    np.random.seed(113)
    entries = task.resolve_target_objects(task.num_objects, replace=False)
    actual_tail = np.random.random_sample(8)

    assert [(entry.key, entry.model_id) for entry in entries] == [
        ("031_jam-jar", 0),
        ("031_jam-jar", 0),
    ]
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_dump_specific_catalog_pick_overrides_are_preserved():
    task = DumpBin({})
    task._episode_dump_specs = [
        {"object_id": "035_apple", "topdown_close_value": 0.77},
        {"object_id": "031_jam-jar", "grasp_z_offset": 0.031},
    ]

    apple = task._sample_dump_object_spec(0)
    jam = task._sample_dump_object_spec(1)

    assert apple["close_value"] == pytest.approx(0.45)
    assert jam["topdown_grasp_z_offset"] == pytest.approx(0.031)
    assert jam['close_value'] == pytest.approx(.725)


@pytest.mark.parametrize("object_name, grasp_z_offset, preopen_steps, width", [
    ("031_jam-jar", 0.02, 80, .035),
    ("086_woodenblock@1", 0.012, 0, .035),
    ("048_stapler", 0.0, 0, .035),
    ("048_stapler", 0.0, 0, .023),
    ("048_stapler", 0.0, 0, float("nan")),
    ("048_stapler", 0.0, 0, 0.0),
])
def test_assist_spawn_passes_catalog_alignment_to_robot_picker(object_name, grasp_z_offset, preopen_steps, width):
    task = DumpBinContribution({"object_name": object_name})
    task.TABLE_TOP_Z = 0.75
    task.big_bin_pose = SimpleNamespace(p=np.array([0.65, -0.30, 0.01]))
    task._sample_cube_xys = lambda: (np.array([0.0]), np.array([-0.2]))
    actor = _FakeEntity([0.0, -0.2, 0.79])
    actor.get_AABB = lambda: np.array([[-0.0175, -0.2-width/2, 0.755], [0.0175, -0.2+width/2, 0.825]])

    def spawn(_x, _y, index):
        task._last_dump_object_spec = task._sample_dump_object_spec(index)
        return actor, 0.02, task._last_dump_object_spec["object_id"]

    task._spawn_dump_object = spawn
    task._load_cubes()
    task._dump_object_center = lambda _actor: actor.get_pos()
    picks = []
    task.pick_and_place = lambda pick, _place, _arm: picks.append(pick) or False
    assert task._pick_and_drop_cube(actor, "right") is False
    if not np.isfinite(width) or width <= 0:
        assert not picks
        return

    # DumpBin consumes explicit topdown hints; the stapler's generic +2 cm
    # annotation bias is not part of its Cartesian grasp contract.
    assert picks[0].grasp_z_offset == pytest.approx(grasp_z_offset)
    assert picks[0].approach_xy_offset == (0.0, 0.0)
    assert picks[0].preopen_steps == preopen_steps
    if object_name in ("086_woodenblock@1", "048_stapler"):
        assert picks[0].close_value * 0.08 == pytest.approx(width - 0.008)


def _legacy_random_primitive_assist_episode(task):
    xs, ys, attempts = _legacy_dump_layout(task)
    robot_color_indices = []
    previous = None
    for _ in range(task.num_objects):
        index = int(np.random.randint(len(task.CUBE_PALETTE)))
        if previous is not None and len(task.CUBE_PALETTE) > 1:
            for _ in range(task.COLOR_RESAMPLE_ATTEMPTS):
                if index != previous:
                    break
                index = int(np.random.randint(len(task.CUBE_PALETTE)))
        robot_color_indices.append(index)
        previous = index
    avatar_color_index = int(np.random.randint(len(task.AVATAR_CUBE_PALETTE)))
    return xs, ys, attempts, robot_color_indices, avatar_color_index


@pytest.mark.parametrize("seed", SEEDS)
def test_assist_primitive_colors_layout_and_avatar_color_preserve_rng_tail(
    seed, monkeypatch
):
    from envs.tasks import dump_bin_contribution as assist_module

    task = DumpBinContribution({"hard": True, "random_dump_objects": False})
    task.scene = object()
    task.TABLE_TOP_Z = 0.75
    task._load_bin = lambda: None

    def create_primitive(_scene, shape, pose, **kwargs):
        assert shape == "box"
        return _FakeEntity(pose.p, color=tuple(kwargs["color"]))

    monkeypatch.setattr(assist_module, "create_primitive", create_primitive)

    np.random.seed(seed)
    expected = _legacy_random_primitive_assist_episode(task)
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    task.load_actors()
    actual_tail = np.random.random_sample(8)

    xs, ys, attempts, robot_color_indices, avatar_color_index = expected
    for index, actor in enumerate(task.objects):
        np.testing.assert_array_equal(actor.get_pos()[:2], [xs[index], ys[index]])
        assert actor.color == task.CUBE_PALETTE[robot_color_indices[index]]
    assert task.avatar_cube.color == task.AVATAR_CUBE_PALETTE[avatar_color_index]
    np.testing.assert_array_equal(actual_tail, expected_tail)
    if seed == 0:
        assert attempts == 2


def _legacy_interrupt_episode(task):
    interrupted_idx = int(np.random.randint(0, task.num_objects))
    forced = task.config.get("interrupt_pick_ordinal")
    if forced is None:
        interrupt_pick_ordinal = int(np.random.randint(0, task.num_objects))
    else:
        interrupt_pick_ordinal = int(forced) - 1

    x_lo, x_hi = task.AVATAR_REACH_SPAWN_X_RANGE
    y_lo, y_hi = task.AVATAR_REACH_SPAWN_Y_RANGE
    min_spacing_sq = float(task.MIN_OBJECT_SPACING) ** 2
    for attempt in range(1, int(task.INTERRUPT_LAYOUT_SAMPLE_ATTEMPTS) + 1):
        xs = np.empty(task.num_objects)
        ys = np.empty(task.num_objects)
        for index in range(task.num_objects):
            xs[index] = np.random.uniform(x_lo, x_hi)
            ys[index] = np.random.uniform(y_lo, y_hi)
        if all(
            (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 >= min_spacing_sq
            for i in range(task.num_objects)
            for j in range(i + 1, task.num_objects)
        ):
            return interrupted_idx, interrupt_pick_ordinal, xs, ys, attempt
    raise RuntimeError("legacy interrupt layout could not be sampled")


@pytest.mark.parametrize("seed", SEEDS)
def test_interrupt_target_ordinal_layout_retry_and_rng_tail(seed):
    task = DumpBinInterruption({"hard": True})

    np.random.seed(seed)
    expected = _legacy_interrupt_episode(task)
    expected_tail = np.random.random_sample(8)

    np.random.seed(seed)
    actual_xs, actual_ys = task._sample_cube_xys()
    actual_tail = np.random.random_sample(8)

    interrupted_idx, ordinal, expected_xs, expected_ys, attempts = expected
    assert task._interrupted_idx == interrupted_idx
    assert task._interrupt_pick_ordinal == ordinal
    np.testing.assert_array_equal(actual_xs, expected_xs)
    np.testing.assert_array_equal(actual_ys, expected_ys)
    np.testing.assert_array_equal(actual_tail, expected_tail)
    if seed == 0:
        assert attempts == 12


def test_interrupt_forced_pick_ordinal_is_one_based_and_validated():
    task = DumpBinInterruption({"hard": True, "interrupt_pick_ordinal": 3})
    np.random.seed(7)
    task._sample_cube_xys()
    assert task._interrupt_pick_ordinal == 2

    for invalid in (0, 4):
        bad = DumpBinInterruption({"hard": True, "interrupt_pick_ordinal": invalid})
        with pytest.raises(ValueError, match=r"must be in \[1, 3\]"):
            bad._sample_interrupt_pick_ordinal()


class _EvalTcpArm:
    gripper_val = 1.0

    def __init__(self, tcp=(0.3, -0.45, 1.05)):
        self.tcp = np.asarray(tcp, dtype=float)

    def get_ee_pose(self):
        return np.concatenate([self.tcp, [1.0, 0.0, 0.0, 0.0]])


def _dump_eval_scene(task):
    _configure_dump_scoring(task, [
        [0.00, 0.00, 0.79],
        [0.10, -0.05, 0.79],
        [-0.10, 0.02, 0.79],
    ])
    task.TABLE_TOP_Z = 0.765
    arm = _EvalTcpArm()
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    return task


def test_interrupt_eval_trigger_draw_and_compatibility_state_preserve_rng_tail():
    # The random-step draw is burned (same bounds, same position in
    # the stream) and the shared progress trigger is armed instead of the
    # retreat/glide/return machine.
    task = _dump_eval_scene(DumpBinInterruption({
        "eval_mode": True,
        "eval_trigger_step_min": 7,
        "eval_trigger_step_max": 11,
    }))
    task._dryrun_capture_palm = lambda: np.array([0.1, -0.2, 0.9])

    np.random.seed(42)
    expected_trigger = int(np.random.randint(7, 12))
    expected_tail = np.random.random_sample(8)

    np.random.seed(42)
    task._init_eval_interrupt_state()
    actual_tail = np.random.random_sample(8)

    assert task._eval_trigger_step == expected_trigger
    assert task._eval_interrupt_fired is False
    assert task._policy_step_count == 0
    assert task._interrupt.trigger.armed
    assert set(task._interrupt.trigger.initial_dist) == {0, 1, 2}
    assert task._interrupt.trigger.params.commit_radius_m == 0.22
    assert task.eval_skip_policy_action() is False
    assert task.consume_eval_policy_reset_requested() is False
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_base_scripted_expert_stops_on_incomplete_manipulation():
    task = DumpBin({"hard": True})
    task.num_objects = 3
    task.objects = ["first", "second", "third"]
    task.TABLE_SETTLE_STEPS = 2
    task.FAILURE_RECOVERY_STEPS = 2
    events = []

    task._boost_finger_pd = lambda arm: events.append(("boost", arm))

    @contextmanager
    def suppress_recording():
        events.append(("suppress", "enter"))
        try:
            yield
        finally:
            events.append(("suppress", "exit"))

    task.suppress_recording = suppress_recording
    task.step_sim = lambda: events.append(("sim", None))
    task.open_gripper = lambda arm: events.append(("open", arm))

    def pick(actor, arm):
        events.append(("pick", actor, arm))
        if actor == "second":
            raise RuntimeError("planner failed")
        return actor == "third"

    task._pick_and_drop_cube = pick
    task._ensure_final_gripper_z = lambda arm: events.append(("final", arm))

    # Do not command an opening/release or
    # report completion after a failed manipulation. Geometry scores separately.
    assert task.play_once() is False
    assert events == [
        ("boost", "right"),
        ("suppress", "enter"),
        ("sim", None),
        ("sim", None),
        ("suppress", "exit"),
        ("pick", "first", "right"),
    ]


def test_assist_scripted_expert_prepares_geometry_then_runs_parallel():
    task = DumpBinContribution({})
    task.avatar = object()
    events = []
    pickup = np.array([0.1, -0.2, 0.8])
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task._prepare_avatar_assist_geometry = lambda: (
        events.append(("prepare", None)), pickup
    )[1]
    task._run_parallel = lambda point, arm: (
        events.append(("parallel", np.asarray(point).copy(), arm)), True
    )[1]

    assert task.play_once() is True
    assert events[0:2] == [("boost", "right"), ("prepare", None)]
    assert events[2][0] == "parallel"
    np.testing.assert_array_equal(events[2][1], pickup)
    assert events[2][2] == "right"


class _FakeGeometryAvatar:
    def __init__(self):
        self.resets = []

    def reset(self, **kwargs):
        self.resets.append(kwargs)


def test_assist_geometry_solve_preserves_cube_pose_collision_reset_and_80_step_settle():
    task = DumpBinContribution({})
    task.TABLE_TOP_Z = 0.75
    task.big_bin_pose = SimpleNamespace(p=np.array([0.65, -0.30, 0.01]))
    task.avatar_cube = _FakeEntity(task.AVATAR_CUBE_PARK_POS)
    task.avatar = _FakeGeometryAvatar()
    task.avatar_collided = True
    task.avatar_collision_log = ["calibration collision"]
    task._avatar_collision_tick = 17
    palm_attach = np.array([0.20, 0.10, 0.90])
    palm_detach = np.array([0.40, -0.10, 0.80])
    task._avatar_dry_run = lambda: (
        palm_attach.copy(), palm_detach.copy(), np.zeros(3)
    )
    sim_steps = []
    task.step_sim = lambda: sim_steps.append(1)

    pickup = task._prepare_avatar_assist_geometry()

    bin_xy = np.asarray(task.big_bin_pose.p[:2])
    toward_avatar = np.asarray(task.avatar_init_pos[:2]) - bin_xy
    toward_avatar /= np.linalg.norm(toward_avatar)
    drop_xy = (bin_xy + toward_avatar * task.AVATAR_BACK_OFFSET
               + np.asarray(task.AVATAR_DROP_XY_SHIFT))
    shift_xy = drop_xy - palm_detach[:2]
    expected_pickup = palm_attach.copy()
    expected_pickup[:2] += shift_xy
    expected_pickup[2] = (
        task.TABLE_TOP_Z
        + task.AVATAR_CUBE_HALF
        + task.AVATAR_CUBE_TABLE_CLEARANCE
    )
    np.testing.assert_allclose(pickup, expected_pickup)
    np.testing.assert_allclose(task.avatar_cube.get_pos(), expected_pickup)
    assert len(task.avatar.resets) == 1
    assert len(sim_steps) == task.AVATAR_POST_DRY_RUN_SETTLE_STEPS == 80
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []
    assert task._avatar_collision_tick == 0


def test_assist_drop_target_stays_inside_the_bin_away_from_the_robot():
    # Idle-robot clearance fix: drop backed off 0.10 m toward the avatar and
    # moved 0.12 m to +x, still inside the bin footprint (x half 0.132 m,
    # y half 0.19 m around (0.65, -0.30)).
    task = DumpBinContribution({})
    assert task.AVATAR_BACK_OFFSET == 0.10
    assert tuple(task.AVATAR_DROP_XY_SHIFT) == (0.12, 0.0)
    bin_xy = np.array(task.BIG_BIN_XY)
    toward_avatar = np.asarray(task.avatar_init_pos[:2]) - bin_xy
    toward_avatar /= np.linalg.norm(toward_avatar)
    drop = bin_xy + toward_avatar * task.AVATAR_BACK_OFFSET + np.asarray(task.AVATAR_DROP_XY_SHIFT)
    assert abs(drop[0] - bin_xy[0]) < 0.132 and abs(drop[1] - bin_xy[1]) < 0.19


def test_assist_blind_expert_uses_seeded_single_robot_target_and_recovers():
    task = DumpBinContribution({"eval_seed": 17})
    task.avatar = None
    task.objects = ["zero", "one", "two"]
    task.TABLE_SETTLE_STEPS = 2
    task.FAILURE_RECOVERY_STEPS = 2
    events = []
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.step_sim = lambda: events.append(("sim", None))
    task._pick_and_drop_cube = lambda obj, arm: (
        events.append(("pick", obj, arm)), False
    )[1]
    task.open_gripper = lambda arm: events.append(("open", arm))

    expected_idx = int(np.random.RandomState(17 + 62017).randint(0, 3))
    assert task.play_blind_once() is True
    assert task._blind_robot_object_idx == expected_idx
    assert ("pick", task.objects[expected_idx], "right") in events
    assert events[-3:] == [("open", "right"), ("sim", None), ("sim", None)]


class _FakeArm:
    tcp_offset = object()


class _FakeRobot:
    def get_arm(self, arm_tag):
        assert arm_tag == "right"
        return _FakeArm()


class _FakeLinkPose:
    def __init__(self, value):
        self.value = value

    def to_pose7(self):
        return ("pose7", self.value)


class _AlwaysSpareAvatar:
    motion_modules = {}

    def spare(self):
        return True


def test_interrupt_scripted_expert_preserves_phase_and_pick_order(monkeypatch):
    from envs import grasp as grasp_module

    task = DumpBinInterruption({"hard": True})
    task.num_objects = 3
    task.objects = [
        _FakeEntity([0.0, -0.1, 0.78]),
        _FakeEntity([0.1, -0.1, 0.78]),
        _FakeEntity([-0.1, -0.1, 0.78]),
    ]
    task._interrupted_idx = 1
    task._interrupt_pick_ordinal = 1
    task.robot = _FakeRobot()
    task.avatar = _AlwaysSpareAvatar()
    task.TABLE_TOP_Z = 0.75
    task.TABLE_SETTLE_STEPS = 2
    task.POST_ATTACH_WAIT_MARGIN_STEPS = 1
    task.POST_RELEASE_SETTLE_STEPS = 1
    events = []

    monkeypatch.setattr(
        grasp_module,
        "tcp_to_link_pose",
        lambda tcp, _offset: _FakeLinkPose(tcp),
    )
    task._boost_finger_pd = lambda arm: events.append(("boost", arm))
    task.step_sim = lambda: events.append(("sim", None))
    task._pick_and_drop_cube = lambda obj, arm: (
        events.append(("pick", task.objects.index(obj), arm)), True
    )[1]
    task.open_gripper = lambda arm: events.append(("open", arm))
    task._dump_object_center = lambda obj: obj.get_pos()
    task._top_down_tcp = lambda pos: ("top_down", tuple(np.asarray(pos)))
    task._move_seeded = lambda pose, arm: events.append(("intent", pose, arm))
    task._play_inspect_motion = lambda **kwargs: events.append(
        ("inspect", task.objects.index(kwargs["attach_obj"]), kwargs)
    )
    task._retreat_safe = lambda arm: events.append(("retreat", arm))
    task._wait_until_avatar_attach = lambda: events.append(("wait_attach", None))
    task._return_avatar_home_after_inspect = lambda: events.append(
        ("avatar_hold", None)
    )
    task._ensure_final_gripper_z = lambda arm: events.append(("final", arm))

    assert task.play_once() is True
    assert task._interrupt_pick_order == [0, 1, 2]
    labels = [event[0] for event in events]
    assert labels == [
        "boost", "sim", "sim", "pick", "open", "intent", "inspect",
        "retreat", "wait_attach", "sim", "avatar_hold", "sim", "pick",
        "pick", "final",
    ]
    assert [event[1] for event in events if event[0] == "pick"] == [0, 2, 1]
    inspect = next(event for event in events if event[0] == "inspect")
    assert inspect[1] == 1
    assert inspect[2]["hand_id"] == task.AVATAR_HAND_ID
    assert inspect[2]["attach_frame"] == task._attach_frame_scaled
    assert inspect[2]["detach_frame"] == task._detach_frame_scaled
    assert inspect[2]["return_to_idle_after"] is None


def _configure_dump_scoring(task, positions):
    task.big_bin_pose = SimpleNamespace(p=np.array([0.5, -0.3, 0.0]))
    task._big_bin_rim_z = 0.24
    task._big_bin_xy_half = np.array([0.14, 0.20])
    task.objects = [_FakeEntity(pos) for pos in positions]
    task.num_objects = len(task.objects)
    task._dump_object_center = lambda actor: actor.get_pos()


def test_base_success_uses_majority_positions_and_ignores_plan_success():
    task = DumpBin({})
    _configure_dump_scoring(task, [
        [0.50, -0.30, 0.10],
        [0.60, -0.20, 0.25],
        [0.80, -0.30, 0.10],
    ])
    task.plan_success = False
    assert bool(task.check_success()) is True

    task.objects[1].set_pos([0.60, -0.20, 0.50])
    assert bool(task.check_success()) is False


def test_base_evaluate_adds_final_gripper_pose_gate(monkeypatch):
    monkeypatch.setattr(
        BaseTask,
        "evaluate",
        lambda _self: {"plan_success": True, "success": True, "base": 1},
    )
    task = DumpBin({})
    task._final_gripper_pose_check = {
        "target": [0.0, 0.0, 0.8],
        "actual": [0.0, 0.0, 0.7],
        "xy_err": 0.01,
        "xy_threshold": 0.05,
        "z_ok": False,
    }

    metrics = task.evaluate()
    assert metrics["base"] == 1
    assert metrics["success"] is False
    assert metrics["final_gripper_pose"] == {
        **task._final_gripper_pose_check,
        "xy_ok": True,
        "success": False,
    }


def test_base_evaluate_exposes_standard_success_and_safety_surface():
    task = DumpBin({})
    _configure_dump_scoring(task, [[0.50, -0.30, 0.10]])
    metrics = task.evaluate()

    assert {
        "plan_success",
        "success",
        "avatar_collision",
        "safe_distance",
        "avatar_safety",
        "avatar_safety_intervention",
        "safety",
    } <= set(metrics)
    assert metrics["plan_success"] is True
    assert metrics["success"] is True
    assert "final_gripper_pose" not in metrics


def test_assist_success_requires_parent_and_avatar_cube_inside(monkeypatch):
    task = DumpBinContribution({})
    _configure_dump_scoring(task, [[0.50, -0.30, 0.10]])
    task.avatar_cube = _FakeEntity([0.50, -0.30, 0.10])
    monkeypatch.setattr(DumpBin, "check_success", lambda _self: True)
    assert bool(task.check_success()) is True

    task.avatar_cube.set_pos([0.80, -0.30, 0.10])
    assert bool(task.check_success()) is False
    monkeypatch.setattr(DumpBin, "check_success", lambda _self: False)
    task.avatar_cube.set_pos([0.50, -0.30, 0.10])
    assert bool(task.check_success()) is False


def test_assist_evaluate_preserves_compatibility_metric_surface(monkeypatch):
    monkeypatch.setattr(DumpBin, "evaluate", lambda _self: {"success": True})
    task = DumpBinContribution({"eval_mode": True})
    task._eval_assist_trigger_step = 0
    task._eval_assist_fired = True
    task._eval_assist_failed = False
    task._eval_policy_step_count = 9
    task._blind_robot_object_idx = 1

    assert task.evaluate() == {
        "success": True,
        "eval_mode": True,
        "eval_assist_trigger_step": 0,
        "eval_assist_fired": True,
        "eval_assist_failed": False,
        "eval_policy_step_count": 9,
        "blind_robot_object_idx": 1,
    }


def test_interrupt_success_requires_all_objects_and_no_collision():
    # The retreat/glide/wait/return success gates are gone; a cube
    # the human still holds is simply not in the bin.
    task = DumpBinInterruption({"eval_mode": True, "hard": True})
    _configure_dump_scoring(task, [
        [0.50, -0.30, 0.10],
        [0.60, -0.20, 0.10],
        [0.45, -0.35, 0.10],
    ])
    task.avatar_collision_summary = lambda: {"any_collision": False}
    assert task.check_success() is True

    task.objects[-1].set_pos([0.9, -0.3, 0.1])
    assert task.check_success() is False
    task.objects[-1].set_pos([0.45, -0.35, 0.1])
    task.avatar_collision_summary = lambda: {"any_collision": True}
    assert task.check_success() is False


def test_interrupt_evaluate_preserves_script_and_pace_metric_surface(monkeypatch):
    # The ``interrupt`` block is reported; ``pace_interrupt`` keeps its key
    # name but is filled from the progress trigger (trigger_mode "progress").
    monkeypatch.setattr(DumpBin, "evaluate", lambda _self: {"success": True})
    task = _dump_eval_scene(DumpBinInterruption({"eval_mode": True}))
    task._dryrun_capture_palm = lambda: np.array([0.1, -0.2, 0.9])
    task._init_eval_interrupt_state()
    task._interrupted_idx = 2
    task._interrupt_pick_ordinal = 1
    task._interrupt_pick_order = [0, 2, 1]
    task._policy_step_count = 9

    metrics = task.evaluate()
    assert metrics["interrupted_object_index"] == 2
    assert metrics["interrupt_pick_ordinal"] == 2
    assert metrics["robot_pick_order"] == [0, 2, 1]
    assert metrics["eval_mode"] is True
    assert metrics["eval_policy_step_count"] == 9
    assert metrics["eval_interrupt_fired"] is False
    assert metrics["interrupt"]["fired"] is False
    assert metrics["interrupt"]["not_fired_reason"] == "min_policy_steps"
    assert metrics["pace_interrupt"]["trigger_mode"] == "progress"
    assert metrics["pace_interrupt"]["fired"] is False
