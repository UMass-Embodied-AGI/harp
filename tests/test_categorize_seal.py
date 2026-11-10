from types import SimpleNamespace

import numpy as np
import pytest

from envs.planning.base import PlanResult
from envs.task_bases.categorize_support import seal
from envs.task_bases.categorize import CategorizeTaskCore
from envs.utils import Pose


BOUNDS = np.array([[-0.267355, 0.151885, 0.763361], [-0.203410, 0.227900, 0.817124]])
OBJECT_POSE = Pose([-0.235374, 0.189768, 0.763364], [2**-0.5, 2**-0.5, 0, 0])


def test_seal_candidates_use_actual_model1_short_axis_and_high_vertical_approach():
    candidates = seal.seal_candidates(OBJECT_POSE, BOUNDS, 1, 0.04, 1.145)
    assert len(candidates) == 2
    for grasp in candidates:
        tcp = grasp.to_world(OBJECT_POSE, 0.04)
        pre = grasp.pre_grasp_world(OBJECT_POSE, 0.04)
        np.testing.assert_allclose(tcp.p[:2], BOUNDS.mean(axis=0)[:2], atol=1e-9)
        assert np.isclose(tcp.p[2], BOUNDS[1, 2] - 0.012)
        np.testing.assert_allclose(pre.p, [*tcp.p[:2], 1.145], atol=1e-9)
        rotation = tcp.to_matrix()[:3, :3]
        np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-9)
        assert abs(rotation[0, 1]) > 0.999  # finger closure axis is worldX
        assert grasp.scale_frame == "runtime"
        # Annotation already measures the scaled world mesh: no second scale.
        np.testing.assert_allclose(grasp.to_world(OBJECT_POSE, 1).p, tcp.p)


@pytest.mark.parametrize("model,scale", [(0, .04), (2, .04), (1, 0), (1, np.nan)])
def test_seal_rejects_wrong_model_or_invalid_scale(model, scale):
    with pytest.raises(ValueError):
        seal.seal_candidates(OBJECT_POSE, BOUNDS, model, scale, 1.145)


def test_seal_rejects_low_rotation_height_and_invalid_bounds():
    with pytest.raises(ValueError):
        seal.seal_candidates(OBJECT_POSE, BOUNDS, 1, .04, .90)
    bad = BOUNDS.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        seal.seal_candidates(OBJECT_POSE, bad, 1, .04, 1.145)


def plans_for_check(monkeypatch):
    tcp = seal.seal_candidates(OBJECT_POSE, BOUNDS, 1, .04, 1.145)[0].to_world(OBJECT_POSE, .04)
    arm = SimpleNamespace(planner=object(), tcp_offset=Pose([0, 0, .1034]))
    grasp_link = tcp * arm.tcp_offset.inv()
    pre_link = Pose([*grasp_link.p[:2], 1.2484], grasp_link.q)
    middle = Pose((pre_link.p + grasp_link.p) / 2, grasp_link.q)
    approach = PlanResult(True, np.stack([pre_link.to_pose7(), pre_link.to_pose7()]))
    descent = PlanResult(True, np.stack([pre_link.to_pose7(), middle.to_pose7(), grasp_link.to_pose7()]))
    monkeypatch.setattr(seal, "mplib_world_link_pose", lambda _, q: Pose.from_pose7(q))
    return arm, (approach, descent, grasp_link, pre_link)


def test_clear_high_rotation_and_constant_orientation_descent_are_accepted(monkeypatch):
    arm, plans = plans_for_check(monkeypatch)
    assert seal.seal_preflight_check(arm, BOUNDS, .13)(plans)


def test_low_rrt_hand_sweep_is_rejected_before_execution(monkeypatch):
    arm, plans = plans_for_check(monkeypatch)
    plans[0].position[0, 2] = BOUNDS[1, 2] + .12
    assert not seal.seal_preflight_check(arm, BOUNDS, .13)(plans)


@pytest.mark.parametrize("bad_motion", ["rotation", "sideways"])
def test_low_descent_cannot_rotate_or_slide_sideways(monkeypatch, bad_motion):
    arm, plans = plans_for_check(monkeypatch)
    if bad_motion == "rotation":
        plans[1].position[1, 3:] = [1, 0, 0, 0]
    else:
        plans[1].position[1, 0] += .01
    assert not seal.seal_preflight_check(arm, BOUNDS, .13)(plans)


def test_invalid_fk_rejects_seal_continuation(monkeypatch):
    arm, plans = plans_for_check(monkeypatch)
    monkeypatch.setattr(seal, "mplib_world_link_pose", lambda *_: None)
    assert not seal.seal_preflight_check(arm, BOUNDS, .13)(plans)


def test_rotation_sphere_contains_open_finger_geometry():
    geom = SimpleNamespace(get_verts=lambda: np.array([[0, 0, .10], [.04, 0, .11]]))
    hand = SimpleNamespace(name="hand", get_pos=lambda: np.zeros(3), _geoms=[geom])
    finger = SimpleNamespace(name="left_finger", _geoms=[geom])
    arm = SimpleNamespace(ee_link=hand, entity=SimpleNamespace(links=[hand, finger]))
    assert seal.gripper_sweep_radius(arm) > np.linalg.norm([.04, 0, .11])


def test_held_center_over_basket_is_not_completed_release():
    assert not seal.released_in_basket([0, 0, 1.1], [0, 0, 1.1], [0, 0, .82])
    assert not seal.released_in_basket([0, 0, 1.1], [.3, 0, .82], [0, 0, .82])
    assert not seal.released_in_basket([0, 0, 1.1], [0, 0, .1], [0, 0, .82])
    assert seal.released_in_basket([0, 0, 1.1], [.01, .01, .82], [0, 0, .82])


def test_seal_experiment_does_not_change_default_or_milk_policy():
    assert CategorizeTaskCore.SEAL_MEASURED_GRASP is False
    assert CategorizeTaskCore.SEAL_HOLD_FORCE_N == 7.5
    assert CategorizeTaskCore.MILK_CONTINUATION_PREFLIGHT is False


@pytest.mark.parametrize("reject_later", [False, True])
def test_seal_continuation_is_preflighted_in_bounded_straight_segments(monkeypatch, reject_later):
    initial = Pose([0, 0, .80], [0, 1, 0, 0])
    targets = []

    def screw(qpos, target, **kwargs):
        assert kwargs == {"qpos_step": .025}
        targets.append(np.asarray(target))
        if reject_later and len(targets) == 3:
            return PlanResult(False)
        return PlanResult(True, np.stack([qpos, target]))

    arm = SimpleNamespace(
        planner=SimpleNamespace(plan_screw_path=screw), tcp_offset=Pose(),
        get_ee_pose=lambda: initial.to_pose7(), get_arm_qpos=lambda: initial.to_pose7(),
        set_arm_joints=lambda *_: pytest.fail("continuation executed before all plans accepted"),
    )
    monkeypatch.setattr(seal, "mplib_world_link_pose", lambda _, q: Pose.from_pose7(q))
    result = seal.plan_straight_segments(arm, Pose([0, 0, 1.145], initial.q))
    if reject_later:
        assert result is None and len(targets) == 3
    else:
        assert len(result) == 4
        np.testing.assert_allclose(np.asarray(targets)[:, :2], 0)
        assert np.max(np.diff([.80, *np.asarray(targets)[:, 2]])) <= .10


def test_clear_endpoints_do_not_hide_sweep_through_object():
    low, high = BOUNDS[0] - .13, BOUNDS[1] + .13
    left = np.array([low[0] - .1, *BOUNDS.mean(axis=0)[1:]])
    right = np.array([high[0] + .1, *BOUNDS.mean(axis=0)[1:]])
    assert seal.segment_intersects_box(left, right, low, high)
    left[2] = right[2] = high[2] + .01
    assert not seal.segment_intersects_box(left, right, low, high)


def test_cooperative_human_targets_remain_planner_obstacles():
    own, human = object(), object()
    seen, clouds = [], []
    task = SimpleNamespace(
        basket_poses=[], _sort_targets=[(own, "seal", 1, 0)],
        _avatar_sort_targets=[(human, "human_object", 0, 0)],
        _actor_aabb_points=lambda actor, *_: seen.append(actor) or np.ones((1, 3)),
        _OBSTACLE_RESOLUTION=.015,
        robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(planner=SimpleNamespace(
            update_obstacles=lambda p, **_: clouds.append(p)))),
    )
    seal._refresh_obstacles(task, SimpleNamespace(actor=own), "right")
    assert seen == [human] and clouds[0].shape == (1, 3)


@pytest.mark.parametrize("mode,expected", [("supported", True), ("floor", False), ("falling", False), ("held", False)])
def test_release_requires_supported_quiet_samples_with_open_clear_hand(mode, expected):
    state = {"steps": 0}

    def position():
        z = .798 if mode != "floor" else .1
        x = .02 * np.sin(state["steps"] * .05) if mode == "falling" else 0
        return np.array([x, 0, z])

    actor = SimpleNamespace(get_pose=lambda: Pose(position()))
    actor.entity = SimpleNamespace(get_AABB=lambda: np.array([position() - [.02, .03, .027], position() + [.02, .03, .027]]))
    task = SimpleNamespace(
        scene=SimpleNamespace(sim=SimpleNamespace(dt=.001)), TABLE_TOP_Z=.765,
        step_sim=lambda: state.update(steps=state["steps"] + 1),
    )
    arm = SimpleNamespace(gripper_val=0 if mode == "held" else 1,
                          get_ee_pose=lambda: [0, 0, 1.145, 0, 1, 0, 0])
    result = seal.settle_released_seal(
        task, SimpleNamespace(actor=actor, get_center=position),
        SimpleNamespace(pos=np.array([0, 0, .821])), arm, np.array([0, 0, 1.1]),
    )
    assert result is expected
    assert 100 <= state["steps"] <= 1500


def test_selector_retiming_preserves_physical_duration_and_gripper_timeline():
    position = np.arange(28, dtype=float).reshape(4, 7)
    plan = PlanResult(True, position.copy())
    retimed, fingers = seal.retime_selector_plan(plan, [0, 1], .001)
    assert len(retimed.position) == 13  # 3*4ms intervals at1ms
    assert len(fingers) == 5
    np.testing.assert_allclose(retimed.position[::4], position)
    np.testing.assert_allclose(plan.position, position)
    np.testing.assert_allclose(fingers, [0, .25, .5, .75, 1])


def test_default_categorize_execute_delegates_original_plan_unchanged(monkeypatch):
    from envs.base_task import BaseTask
    calls = []
    monkeypatch.setattr(BaseTask, "execute_plan", lambda self, *a, **kw: calls.append((a, kw)) or "unchanged")
    task = object.__new__(CategorizeTaskCore)
    plan, gripper = object(), [1, 0]
    assert task.execute_plan(plan, "right", gripper) == "unchanged"
    assert calls == [((plan, "right"), {"gripper_result": gripper})]


@pytest.mark.parametrize("raises", [False, True])
def test_seal_selector_flag_is_restored_on_return_or_exception(raises):
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(tcp_offset=Pose())))

    def select(**kwargs):
        assert task._seal_selector_execution is not None
        if raises:
            raise seal.SealApproachRejected("high pose failed")
        return "selected"

    task.select_and_execute_grasp = select
    if raises:
        with pytest.raises(seal.SealApproachRejected):
            seal.select_seal_with_timed_execution(task, "right", post_plan_check=lambda _: True)
    else:
        assert seal.select_seal_with_timed_execution(task, "right", post_plan_check=lambda _: True) == "selected"
    assert task._seal_selector_execution is None


@pytest.mark.parametrize("arrived", [True, False])
def test_high_pre_pose_must_be_quiet_and_accurate_before_descent(arrived):
    target = Pose([0, 0, 1.0], [0, 1, 0, 0])
    actual = target if arrived else Pose(target.p, [1, 0, 0, 0])
    arm = SimpleNamespace(get_ee_pose=lambda: actual.to_pose7())
    steps, executed = [], []
    task = SimpleNamespace(scene=SimpleNamespace(sim=SimpleNamespace(dt=.001)),
                           robot=SimpleNamespace(get_arm=lambda _: arm), step_sim=lambda: steps.append(1))
    plan = PlanResult(True, np.ones((2, 7)))
    context = seal.SealSelectorExecution("right", plans=[plan, object()], pre_tcp=target)
    delegate = lambda *a, **kw: executed.append((a, kw))
    if arrived:
        seal.execute_seal_selector_plan(task, delegate, plan, "right", None, context)
        assert context.next_leg == 1 and len(steps) == 100
    else:
        with pytest.raises(seal.SealApproachRejected, match="descent forbidden"):
            seal.execute_seal_selector_plan(task, delegate, plan, "right", None, context)
        assert context.next_leg == 0 and len(steps) == 1000
    assert len(executed) == 1  # only the high approach, never a descent


def preload_fixture(dt=.001, initial=(.0264, .0264), final=(.0249, .0249),
                    object_shift=(0, 0, 0), object_angle=0, wrist_shift=0):
    state = {"steps": 0, "commands": [], "ramp": []}
    arm = SimpleNamespace(
        gripper_closed=0., gripper_open=.04, gripper_val=.6,
        _finger_dof_indices=[7, 8],
    )

    def fingers(*, dofs_idx_local):
        assert tuple(dofs_idx_local) == (7, 8)
        return np.asarray(final if state["steps"] else initial)

    forbidden = lambda *_a, **_k: pytest.fail("hold changed gains, force caps, simulator state, or arm targets")
    arm.entity = SimpleNamespace(
        get_dofs_position=fingers, set_qpos=forbidden,
        set_dofs_kp=forbidden, set_dofs_kv=forbidden, set_dofs_force_range=forbidden,
    )
    arm.set_arm_joints = forbidden
    initial_pose = Pose([0, 0, .8])

    def object_pose():
        if not state["steps"]:
            return initial_pose
        return Pose(initial_pose.p + np.asarray(object_shift),
                    [np.cos(object_angle / 2), np.sin(object_angle / 2), 0, 0])

    arm.get_ee_pose = lambda: Pose([wrist_shift if state["steps"] else 0, 0, .81]).to_pose7()
    task = SimpleNamespace(scene=SimpleNamespace(sim=SimpleNamespace(dt=dt)))
    task.step_sim = lambda: state.update(steps=state["steps"] + 1)

    def set_gripper(value, arm_tag, num_steps=200):
        state["commands"].append((value, arm_tag, num_steps))
        for value_i in np.linspace(arm.gripper_val, value, num_steps):
            arm.gripper_val = float(value_i)
            state["ramp"].append(float(value_i))
            task.step_sim()

    task.set_gripper = set_gripper
    pick = SimpleNamespace(actor=SimpleNamespace(get_pose=object_pose))
    return task, pick, arm, state


@pytest.mark.parametrize("dt,ramp_steps,hold_steps", [(.001, 300, 100), (.002, 150, 50), (.004, 75, 25)])
def test_seal_preload_changes_only_hold_command_with_physical_ramp_and_duration(dt, ramp_steps, hold_steps):
    task, pick, arm, state = preload_fixture(dt=dt)
    assert seal.preload_seal_hold(task, pick, arm, "right")
    assert state["commands"] == [(.525, "right", ramp_steps)]
    assert state["steps"] == ramp_steps + hold_steps
    assert state["ramp"][0] == .6 and state["ramp"][-1] == .525
    assert np.all(np.diff(state["ramp"]) <= 0)
    assert arm.gripper_val == .525  # Existing execute_plan will hold this value.
    assert arm.gripper_val * arm.gripper_open == pytest.approx(.021)


@pytest.mark.parametrize("initial", [(np.nan, .0264), (.0264,), (-.001, .0264),
                                     (.05, .0264), (.0243, .0264), (.024, .024)])
def test_seal_preload_never_closes_tighter_without_finite_bilateral_stall_proxy(initial):
    task, pick, arm, state = preload_fixture(initial=initial)
    assert not seal.preload_seal_hold(task, pick, arm, "right")
    assert state["commands"] == [] and state["steps"] == 0
    assert arm.gripper_val == .6


@pytest.mark.parametrize("indices", [None, [], [7], [7, 7], [True, 8], [7.0, 8], [-1, 8], [[7], 8]])
def test_seal_preload_rejects_malformed_finger_mapping_before_execution(indices):
    task, pick, arm, state = preload_fixture()
    arm._finger_dof_indices = indices
    assert not seal.preload_seal_hold(task, pick, arm, "right")
    assert state["commands"] == [] and state["steps"] == 0


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf, 1e-8])
def test_seal_preload_invalid_clock_fails_closed(dt):
    task, pick, arm, state = preload_fixture(dt=dt)
    assert not seal.preload_seal_hold(task, pick, arm, "right")
    assert state["commands"] == [] and state["steps"] == 0


@pytest.mark.parametrize("attribute,value", [("gripper_closed", .01), ("gripper_open", .08),
                                             ("gripper_val", .5), ("gripper_val", np.nan)])
def test_seal_preload_is_scoped_to_expected_franka_acquisition(attribute, value):
    task, pick, arm, state = preload_fixture()
    setattr(arm, attribute, value)
    assert not seal.preload_seal_hold(task, pick, arm, "right")
    assert state["commands"] == [] and state["steps"] == 0


@pytest.mark.parametrize("kwargs,expected", [
    ({"object_shift": (.001, -.001, .001), "object_angle": np.deg2rad(3)}, True),
    ({"object_shift": (.009, 0, 0), "object_angle": np.deg2rad(14)}, True),
    ({"object_shift": (.011, 0, 0)}, False),
    ({"object_angle": np.deg2rad(16)}, False),
    ({"object_shift": (np.nan, 0, 0)}, False),
    ({"wrist_shift": .007}, False),
    ({"final": (.0213, .024)}, False),
    ({"final": (.024, np.nan)}, False),
])
def test_seal_preload_allows_small_settling_but_rejects_loss_or_material_shove(kwargs, expected):
    task, pick, arm, state = preload_fixture(**kwargs)
    assert seal.preload_seal_hold(task, pick, arm, "right") is expected
    assert state["commands"] == [(.525, "right", 300)]
    assert state["steps"] == 400  # Fixed bound, no repeated squeezing or regrasp.


def test_seal_acquisition_is_unchanged_before_opt_in_feedback():
    import inspect
    source = inspect.getsource(seal.pick_and_place_seal)
    initial_close = "task.set_gripper(max(0.010, close_m) / arm.gripper_open, arm_tag)"
    assert initial_close + "\n    settle(300)\n    start_center" in source
    assert "preload_seal_hold" not in source
    assert source.index("SealHoldController") < source.index('straight(lifted_tcp, "lift")')


def test_pinned_genesis_position_reader_accepts_local_finger_index_tuple():
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    # get_dofs_position delegates its tuple directly to this pure index mapper.
    actual = RigidEntity._get_global_idx(SimpleNamespace(), (7, 8), 9, 13, unsafe=True)
    assert actual == [20, 21]


def hold_fixture(dt=.001):
    state = dict(q=np.array([.0264, .0264]), v=np.zeros(2), kp=np.full(2, 2500.),
                 kv=np.full(2, 200.), obj=Pose([0, 0, .8]), tcp=Pose([0, 0, .81]),
                 reads=0, steps=0, commands=[])
    forbidden = lambda *_a, **_k: pytest.fail("hold changed gains/caps/simulator state/arm target")

    def read(key, dofs_idx_local):
        assert dofs_idx_local == (7, 8)
        state["reads"] += 1
        return state[key].copy()

    entity = SimpleNamespace(
        get_dofs_position=lambda **k: read("q", **k),
        get_dofs_velocity=lambda **k: read("v", **k),
        get_dofs_kp=lambda **k: read("kp", **k),
        get_dofs_kv=lambda **k: read("kv", **k),
        set_qpos=forbidden, set_dofs_kp=forbidden, set_dofs_kv=forbidden,
        set_dofs_force_range=forbidden, control_dofs_force=forbidden,
    )
    arm = SimpleNamespace(entity=entity, _finger_dof_indices=(7, 8),
                          gripper_closed=0., gripper_open=.04, gripper_val=.6,
                          get_ee_pose=lambda: state["tcp"].to_pose7(),
                          set_arm_joints=forbidden)
    pick = SimpleNamespace(actor=SimpleNamespace(get_pose=lambda: state["obj"]),
                           get_center=lambda: state["obj"].p.copy())
    task = object.__new__(CategorizeTaskCore)
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=dt))
    task.step_sim = lambda: state.update(steps=state["steps"] + 1)

    def set_gripper(value, arm_tag):
        assert arm_tag == "right"
        state["commands"].append(float(value))
        arm.gripper_val = float(value)

    task.robot = SimpleNamespace(get_arm=lambda _: arm, set_gripper=set_gripper)
    return task, pick, arm, state


@pytest.mark.parametrize("velocity,target,effort", [(-.02, .02072, -6.7), (0, .022, -7.5),
                                                    (.02, .02328, -8.3)])
def test_hold_formula_retains20percent_damping_with_signed_actuator_estimate(velocity, target, effort):
    actual, predicted = seal.seal_hold_target(
        [.025, .025], [velocity, velocity], [2500, 2500], [200, 200], target, .005)
    assert actual == pytest.approx(target)
    np.testing.assert_allclose(predicted, effort, atol=1e-12)


def test_hold_velocity_compensation_is_capped_and_target_slew_is_bounded():
    target, predicted = seal.seal_hold_target(
        [.025, .025], [-.05, -.05], [2500, 2500], [200, 200], .020, .005)
    assert target == pytest.approx(.020)  # -.002m cap, not uncapped -.0032m correction
    np.testing.assert_allclose(predicted, -2.5, atol=1e-12)
    target, predicted = seal.seal_hold_target(
        [.0264, .0264], [0, 0], [2500, 2500], [200, 200], .024, .005)
    assert target == pytest.approx(.02375)  # maximum250um per5ms
    np.testing.assert_allclose(predicted, -6.625, atol=1e-12)


@pytest.mark.parametrize("force", [None, True, np.bool_(False), "25", [25], np.array(25),
                                   np.nan, np.inf, -1, 0, 30.001, 10**100])
def test_hold_force_setting_rejects_invalid_or_out_of_range_values(force):
    with pytest.raises(seal.SealHoldRejected, match="nominal seal hold force"):
        seal.seal_hold_target([.0312, .0312], [0, 0], [2500, 2500], [200, 200],
                              .024, .005, force_n=force)


def test_firmer_hold_slews_to25N_without_gains_or_bounds_change():
    target = .024
    efforts = []
    for _ in range(12):
        previous = target
        target, effort = seal.seal_hold_target(
            [.0312, .0312], [0, 0], [2500, 2500], [200, 200], target, .005, force_n=25,
        )
        assert .018 <= target <= .024
        assert 0 <= previous - target <= .00025 + 1e-12
        efforts.append(effort)
    assert target == pytest.approx(.0212)
    np.testing.assert_allclose(efforts[-1], -25.)
    assert all(np.all(-25. - 1e-9 <= e) and np.all(e < -18.) for e in efforts)


def test_explicit_default_force_matches_legacy_call():
    args = ([.0264, .0264], [-.01, -.01], [2500, 2500], [200, 200], .024, .005)
    implicit, explicit = seal.seal_hold_target(*args), seal.seal_hold_target(*args, force_n=7.5)
    assert implicit[0] == explicit[0]
    np.testing.assert_array_equal(implicit[1], explicit[1])


def test_firmer_hold_retains_damping_compensation_and_floor():
    target, effort = seal.seal_hold_target(
        [.0312, .0312], [-.01, -.01], [2500, 2500], [200, 200], .02056, .005, force_n=25,
    )
    assert target == pytest.approx(.02056)
    np.testing.assert_allclose(effort, -24.6)
    target, effort = seal.seal_hold_target(
        [.027, .027], [0, 0], [2500, 2500], [200, 200], .018, .005, force_n=25,
    )
    assert target == .018
    np.testing.assert_allclose(effort, -22.5)  # physical position floor, not forced25N


def test_controller_latches_task_local_force_after_unchanged_acquisition():
    task, pick, arm, state = hold_fixture()
    task.SEAL_HOLD_FORCE_N = 25.
    state["q"][:] = .0312
    controller = seal.SealHoldController(task, pick, arm, "right")
    assert controller.target_m == .024 and arm.gripper_val == .6
    assert not state["commands"] and not state["steps"]
    assert CategorizeTaskCore.SEAL_HOLD_FORCE_N == 7.5
    task.SEAL_HOLD_FORCE_N = 7.5  # an in-flight hold keeps its chosen setting
    for _ in range(12):
        controller.ticks = controller.update_steps
        controller.command()
    assert controller.force_n == 25.
    assert controller.target_m == pytest.approx(.0212)
    np.testing.assert_allclose(controller.predicted_effort, -25.)
    assert not state["commands"] and not state["steps"]


def test_v6_pre_guard_state_keeps_more_inward_effort_even_at18mm_floor():
    # Recorded V6 step6640. These are command-model calculations only: not a
    # claim that a changed controller would reproduce this physical state.
    q = np.array([.0223626047372818, .022371355444192886])
    v = np.array([-.02437405474483967, -.023734260350465775])
    previous = .019084004685282704
    old_effort = 2500 * (previous - q) - 200 * v
    first, first_effort = seal.seal_hold_target(q, v, [2500, 2500], [200, 200], previous, .005)
    assert first == pytest.approx(previous - .00025)
    assert np.all(first_effort < old_effort)  # strictly more inward, same gains
    target = first
    for _ in range(4):
        target, effort = seal.seal_hold_target(q, v, [2500, 2500], [200, 200], target, .005)
    assert target == .018
    np.testing.assert_allclose(effort, [-6.031700894236568, -6.1815365403890645], atol=1e-10)
    assert np.all(effort < old_effort) and np.all(effort > -7.5)


@pytest.mark.parametrize("q,previous,expected", [(.020, .0184, .018), (.031, .0236, .024)])
def test_hold_keeps_absolute_position_target_bounds(q, previous, expected):
    target, _ = seal.seal_hold_target([q, q], [0, 0], [2500, 2500], [200, 200], previous, .02)
    assert target == pytest.approx(expected)


@pytest.mark.parametrize("key,value", [
    ("q", [np.nan, .0264]), ("q", [.0264]), ("q", [.0264, .029]),
    ("q", [-.001, -.001]), ("q", [.041, .041]), ("q", [.017, .017]),
    ("v", [np.inf, 0]), ("v", [.0501, 0]), ("v", [-.0501, 0]),
    ("kp", [0, 2500]), ("kp", [-1, 2500]), ("kp", [np.nan, 2500]),
    ("kp", [1e-320, 1e-320]), ("kv", [-1, 200]), ("kv", [200, np.inf]),
])
def test_hold_rejects_nonfinite_malformed_or_unsafe_feedback(key, value):
    values = dict(q=[.0264, .0264], v=[0, 0], kp=[2500, 2500], kv=[200, 200])
    values[key] = value
    with pytest.raises(seal.SealHoldRejected):
        seal.seal_hold_target(values["q"], values["v"], values["kp"], values["kv"], .024, .02)


@pytest.mark.parametrize("dt", [0, -.001, np.nan, np.inf, .006, .021, 1e-8])
def test_hold_invalid_clock_rejected_before_any_command(dt):
    task, pick, arm, state = hold_fixture(dt)
    with pytest.raises(seal.SealHoldRejected):
        seal.SealHoldController(task, pick, arm, "right")
    assert state["steps"] == 0 and not state["commands"]


@pytest.mark.parametrize("indices", [None, [], [7], [7, 7], [True, 8], [7.0, 8], [-1, 8], [[7], 8]])
def test_hold_malformed_finger_mapping_rejected(indices):
    task, pick, arm, state = hold_fixture()
    arm._finger_dof_indices = indices
    with pytest.raises(seal.SealHoldRejected):
        seal.SealHoldController(task, pick, arm, "right")
    assert state["steps"] == 0 and not state["commands"]


def test_hold_needs_original_acquisition_and_bilateral_stall_proxy():
    task, pick, arm, state = hold_fixture()
    arm.gripper_val = .525
    with pytest.raises(seal.SealHoldRejected):
        seal.SealHoldController(task, pick, arm, "right")
    arm.gripper_val = .6
    state["q"][:] = .0243
    with pytest.raises(seal.SealHoldRejected):
        seal.SealHoldController(task, pick, arm, "right")
    assert not state["commands"] and not state["steps"]


def test_hold_sequence_is_lazy_live_and_updates_every5ms_without_extra_steps():
    task, pick, arm, state = hold_fixture()
    controller = seal.SealHoldController(task, pick, arm, "right")
    sequence = seal.SealHoldSequence(controller, 50)
    initial_reads = state["reads"]
    assert len(sequence) == 50 and state["reads"] == initial_reads
    assert [sequence[i] for i in range(5)] == [.6] * 5
    assert state["reads"] == initial_reads
    # Change the real state after sequence creation; do not precompute feedback.
    state["q"][:] = .0262
    assert sequence[5] == pytest.approx(.59375)
    assert state["reads"] == initial_reads + 4
    assert controller.predicted_effort[0] == pytest.approx(-6.125)
    assert state["steps"] == 0 and not state["commands"]
    with pytest.raises(IndexError):
        sequence[50]


@pytest.mark.parametrize("shift,angle,expected", [(.009, 14, True), (.019, 19, True),
                                                  (.021, 0, False), (0, 21, False)])
def test_hold_relative_object_guard_allows_settling_and_unchanged_rigid_transport(shift, angle, expected):
    task, pick, arm, state = hold_fixture()
    controller = seal.SealHoldController(task, pick, arm, "right")
    state["tcp"] = Pose([.3, .2, 1.2])
    state["obj"] = Pose([.3 + shift, .2, 1.19],
                        [np.cos(np.deg2rad(angle) / 2), np.sin(np.deg2rad(angle) / 2), 0, 0])
    controller.ticks = controller.update_steps
    if expected:
        assert np.isfinite(controller.command())
    else:
        with pytest.raises(seal.SealHoldRejected, match="offset or rotation"):
            controller.command()
    assert not state["commands"] and not state["steps"]


def test_hold_updates_across_replay_and_settle_without_extra_physics(monkeypatch):
    from envs.base_task import BaseTask
    task, pick, arm, state = hold_fixture()
    controller = seal.SealHoldController(task, pick, arm, "right")
    plan = PlanResult(True, np.zeros((15, 7)))
    seen = []

    def delegate(self, result, arm_tag, gripper_result=None):
        assert result is plan
        assert isinstance(gripper_result, seal.SealHoldSequence)
        seen.append(result)
        for i in range(len(gripper_result)):
            task.robot.set_gripper(gripper_result[i], arm_tag)
            task.step_sim()

    monkeypatch.setattr(BaseTask, "execute_plan", delegate)
    with seal.seal_hold_scope(task, controller):
        task.execute_plan(plan, "right")
        seal.settle_seal_hold(task, "right", 10)
        task.execute_plan(plan, "right")
    assert len(seen) == 2 and state["steps"] == 40 and len(state["commands"]) == 40
    assert state["commands"][:5] == [.6] * 5
    assert state["commands"][5:10] == pytest.approx([.59375] * 5)
    assert state["commands"][10:15] == pytest.approx([.5875] * 5)
    assert state["commands"][15:] == pytest.approx([.585] * 25)
    assert controller.updates == 7 and controller.update_dt == .005
    assert task._seal_hold_controller is None
    seal.settle_seal_hold(task, "right", 10)
    assert state["steps"] == 50 and len(state["commands"]) == 40


@pytest.mark.parametrize("bad", ["arm", "gripper", "selector"])
def test_hold_dispatch_rejects_conflicting_contract_before_delegate(bad):
    task, pick, arm, state = hold_fixture()
    controller = seal.SealHoldController(task, pick, arm, "right")
    if bad == "selector":
        task._seal_selector_execution = object()
    with pytest.raises(seal.SealHoldRejected):
        seal.execute_seal_hold_plan(
            task, lambda *_a, **_k: pytest.fail("invalid hold was executed"),
            PlanResult(True, np.zeros((2, 7))), "left" if bad == "arm" else "right",
            [0, 0] if bad == "gripper" else None, controller)


@pytest.mark.parametrize("raises", [False, True])
def test_hold_scope_cleans_on_normal_or_exceptional_exit(raises):
    task, pick, arm, state = hold_fixture()
    controller = seal.SealHoldController(task, pick, arm, "right")
    try:
        with seal.seal_hold_scope(task, controller):
            assert task._seal_hold_controller is controller
            if raises:
                raise RuntimeError("unrelated replay failure")
    except RuntimeError:
        assert raises
    assert task._seal_hold_controller is None


def test_hold_lifecycle_clears_before_open_and_rejects_cleanly(monkeypatch):
    task, pick, arm, state = hold_fixture()
    task.TABLE_TOP_Z = .76
    task._sort_targets = [(pick.actor, "100_seal", 1, 0)]
    task._get_object_scale = lambda *_: .04
    bounds = np.array([[-.032, -.038, .78], [.032, .038, .82]])
    pick.actor.entity = SimpleNamespace(get_AABB=lambda: bounds.copy())
    arm.tcp_offset = Pose()
    state["tcp"] = Pose([0, 0, .808])
    events = []

    def opening(_):
        assert getattr(task, "_seal_hold_controller", None) is None
        events.append("open")
        arm.gripper_val = 1

    def closing(value, tag):
        assert value == pytest.approx(.6)
        assert state["steps"] == 180  # original80+100 settle; no new acquisition ticks
        arm.gripper_val = value
        events.append("close")

    task.open_gripper, task.set_gripper = opening, closing
    monkeypatch.setattr(seal, "gripper_sweep_radius", lambda _: .13)
    monkeypatch.setattr(seal, "_refresh_obstacles", lambda *_: None)
    monkeypatch.setattr(seal, "select_seal_with_timed_execution", lambda *_a, **_k: (state["tcp"], None, None))

    def reject_plan(*_):
        assert state["steps"] == 480  # original300 settle, no fixedV5 ramp
        assert task._seal_hold_controller is not None
        raise seal.SealHoldRejected("diagnostic grip loss")

    monkeypatch.setattr(seal, "plan_straight_segments", reject_plan)
    assert not seal.pick_and_place_seal(task, pick, SimpleNamespace(pos=[0, 0, .816]), "right")
    assert events == ["open", "close", "open"] and state["steps"] == 600
    assert task._seal_hold_controller is None
