"""Scene-free contracts for the opt-in, measured TAKE release descent.

The fake arm follows commanded waypoints exactly unless a test injects lag or
object motion. Native planner/preflight behavior has separate tests; here their
interfaces are stubbed, while the production target/tracking helper is real.
"""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from envs.task_bases import take_release as release
from envs.tasks.take_from_human_easy import TakeFromHumanEasy


PLATE = np.array([[-.1, -.1, .765], [.1, .1, .790]])
TABLE = np.array([[-.5, -.7, .715], [.5, .5, .765]])


def _initial():
    return dict(
        tcp=np.array([.02, -.01, 1.04, 1., 0., 0., 0.]),
        position=np.array([.021, -.012, .995]),
        quaternion=np.array([1., 0., 0., 0.]),
        bounds=np.array([[0., -.03, .960], [.04, .01, 1.000]]),
        fingers=np.array([[[0., -.04, .980], [.01, -.03, 1.06]],
                          [[.03, .01, .980], [.04, .02, 1.06]]]),
    )


def _rot(angle):
    return np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)])


@pytest.fixture
def runtime(monkeypatch):
    initial = _initial()
    clock = SimpleNamespace(steps=0, q=np.zeros(7), target=None, commands=[],
                            grips=[], plans=[], validations=[], refreshed=[],
                            adjust=lambda state, step: state)

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected opening, pose setter, IK, or branch fallback")

    def state():
        result = {key: value.copy() for key, value in initial.items()}
        if np.any(clock.q):
            delta = clock.target[:3] - initial["tcp"][:3]
            result["tcp"][:3] += delta
            result["position"] += delta
            result["bounds"] += delta
            result["fingers"] += delta
        return clock.adjust(result, clock.steps)

    def plan(q, target, **kwargs):
        clock.plans.append((q.copy(), target.copy(), kwargs))
        clock.target = np.asarray(target).copy()
        return SimpleNamespace(success=True, position=np.array([q, np.full(7, .1)]))

    def validate(arm, path, target, **kwargs):
        clock.validations.append((path, target.copy(), kwargs))
        if path is None or not path.success:
            raise ValueError("unsuccessful plan")
        return {"stub_preflight": True}

    def command(q):
        clock.commands.append((np.asarray(q).copy(), clock.steps))
        clock.q = np.asarray(q).copy()

    def grip(value, tag):
        clock.grips.append((value, tag, clock.steps))
        assert value == .35

    planner = SimpleNamespace(_bounded_paths_debug_enabled=lambda: False,
                              plan_screw_path=plan, plan_path=forbidden,
                              plan_qpos=forbidden, solve_ik=forbidden)
    entity = SimpleNamespace(
        links=[SimpleNamespace(name="panda_leftfinger", get_AABB=lambda: state()["fingers"][0]),
               SimpleNamespace(name="panda_rightfinger", get_AABB=lambda: state()["fingers"][1])],
        get_qpos=lambda: np.r_[clock.q, .014, .014], set_qpos=forbidden,
        set_pos=forbidden, inverse_kinematics=forbidden,
    )
    arm = SimpleNamespace(planner=planner, entity=entity, gripper_val=.35,
                          tcp_offset=None, get_arm_qpos=lambda: clock.q.copy(),
                          get_ee_pose=lambda: state()["tcp"], set_arm_joints=command)
    task = SimpleNamespace(
        plan_success=True, SIMULATION_DT=.001,
        robot=SimpleNamespace(get_arm=lambda tag: arm, set_gripper=grip),
        apple_actor=SimpleNamespace(entity=SimpleNamespace(
            get_pos=lambda: state()["position"], get_quat=lambda: state()["quaternion"],
            get_AABB=lambda: state()["bounds"], set_pos=forbidden, set_quat=forbidden)),
        plate_actor=SimpleNamespace(entity=SimpleNamespace(get_AABB=lambda: PLATE.copy())),
        table=SimpleNamespace(get_AABB=lambda: TABLE.copy()),
        _refresh_planner_obstacles=lambda tag: clock.refreshed.append(tag),
        _avatar_retreat_enabled=lambda: False,
        _apply_avatar_retreat_if_needed=forbidden,
        _move_screw=forbidden, _move_seeded=forbidden, _move_cartesian=forbidden,
        open_gripper=forbidden, execute_plan=forbidden,
    )

    def step():
        clock.steps += 1

    task.step_sim = step
    monkeypatch.setattr(release, "validate_release_plan", validate)
    monkeypatch.setattr(release, "tcp_to_link_pose", lambda pose, offset: pose)
    return SimpleNamespace(task=task, arm=arm, clock=clock, initial=initial)


def _rejected(runtime, reason):
    assert release.lower_for_release(runtime.task, "right") is False
    assert runtime.task.plan_success is False
    diagnostic = runtime.task._guarded_release_diagnostics
    assert diagnostic["stage"] == "rejected_without_opening"
    assert reason in diagnostic["reason"]
    return diagnostic


def test_benchmark_default_is_opt_out():
    assert TakeFromHumanEasy.GUARDED_RELEASE is False


def test_target_uses_live_bounds_not_offset_object_origin():
    state = _initial()
    expected = state["tcp"].copy()
    expected[:3] = [0., 0., .875]
    np.testing.assert_allclose(release.release_target(state, PLATE), expected, atol=1e-12)
    state["position"] += [2., -3., .4]
    np.testing.assert_allclose(release.release_target(state, PLATE), expected, atol=1e-12)


def test_lower_finger_sets_clearance_and_raises_object_target():
    state = _initial()
    state["fingers"][:, :, 2] -= .035  # Fingers now 15 mm below object bottom.
    target = release.release_target(state, PLATE)
    delta = target[:3] - state["tcp"][:3]
    assert state["fingers"][:, 0, 2].min() + delta[2] == pytest.approx(.795)
    assert state["bounds"][0, 2] + delta[2] - PLATE[1, 2] == pytest.approx(.020)


def test_finger_clearance_cannot_justify_another_high_release():
    state = _initial()
    state["fingers"][:, :, 2] -= .05
    with pytest.raises(ValueError, match="another high release"):
        release.release_target(state, PLATE)


@pytest.mark.parametrize("translation", [[0., 0., -.20], [0., 0., .10], [.10, 0., 0.]])
def test_release_target_rejects_upward_too_long_or_lateral_moves(translation):
    state = _initial()
    state["bounds"] += translation
    state["fingers"] += translation
    with pytest.raises(ValueError, match="bounded downward corridor"):
        release.release_target(state, PLATE)


@pytest.mark.parametrize("dt", [.001, .002, .004])
def test_native_path_six_pd_substeps_and_five_quiet_arrivals(runtime, dt):
    task, arm, clock = runtime.task, runtime.arm, runtime.clock
    task.SIMULATION_DT = dt
    assert release.lower_for_release(task, "right") is True
    assert task.plan_success is True
    assert len(clock.plans) == len(clock.validations) == 1
    assert clock.plans[0][2] == {"qpos_step": .02, "log": True}
    assert clock.refreshed == ["right"]
    assert [step for _, step in clock.commands] == list(range(12))
    np.testing.assert_allclose([q for q, _ in clock.commands[:6]], np.zeros((6, 7)))
    np.testing.assert_allclose([q for q, _ in clock.commands[6:]], np.full((6, 7), .1))
    assert clock.grips == [(.35, "right", step) for step in range(12)]
    assert arm.gripper_val == .35
    assert clock.steps == 12 + 5 * max(1, round(.02 / dt))
    diagnostic = task._guarded_release_diagnostics
    assert diagnostic["stage"] == "ready_to_open"
    assert diagnostic["samples"] == 5
    assert diagnostic["tcp_error"] == pytest.approx(0.)
    assert diagnostic["bottom_above_rim"] == pytest.approx(.005)
    np.testing.assert_allclose(clock.validations[0][2]["table_bounds"], TABLE)
    np.testing.assert_allclose(clock.validations[0][2]["plate_bounds"], PLATE)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_arrival_tracks_measured_xyz_not_only_commanded_target(runtime, axis):
    def lag(state, step):
        if step >= 6:
            delta = np.zeros(3)
            delta[axis] = .01
            state["tcp"][:3] += delta
            state["position"] += delta
            state["bounds"] += delta
            state["fingers"] += delta
        return state

    runtime.clock.adjust = lag
    _rejected(runtime, "no stable measured low-release arrival")
    assert runtime.clock.steps == 12 + 600
    assert len(runtime.clock.plans) == 1


def test_arrival_requires_object_centering_even_if_tcp_arrived(runtime):
    def off_center(state, step):
        if step >= 6:
            state["position"][0] += .01
            state["bounds"][:, 0] += .01
        return state

    runtime.clock.adjust = off_center
    _rejected(runtime, "no stable measured low-release arrival")


def test_motion_resets_five_consecutive_quiet_arrival_samples(runtime):
    def nudge(state, step):
        if step >= 92:  # Fourth sample, after 12 PD steps and 4 x 20 ms.
            state["position"][0] += .004
            state["bounds"][:, 0] += .004
        return state

    runtime.clock.adjust = nudge
    assert release.lower_for_release(runtime.task, "right") is True
    assert runtime.task._guarded_release_diagnostics["samples"] == 9
    assert runtime.clock.steps == 12 + 180


@pytest.mark.parametrize("fault,reason", [
    ("drift", "held object moved"),
    ("object_rotation", "held object moved"),
    ("tcp_rotation", "gripper orientation"),
    ("object_clearance", "plate-rim clearance"),
    ("finger_clearance", "plate-rim clearance"),
])
def test_each_physical_substep_rejects_retention_or_clearance_loss(runtime, fault, reason):
    def disturbed(state, step):
        if step == 7:  # One-step disturbance, gone by the next observation window.
            if fault == "drift":
                state["position"][0] += .016
            elif fault == "object_rotation":
                state["quaternion"] = _rot(.151)
            elif fault == "tcp_rotation":
                state["tcp"][3:] = _rot(.051)
            elif fault == "object_clearance":
                state["bounds"][:, 2] -= .004
            else:
                state["fingers"][:, :, 2] -= .024
        return state

    runtime.clock.adjust = disturbed
    _rejected(runtime, reason)
    assert runtime.clock.steps == 7


def test_prior_planning_failure_cannot_be_cleared(runtime):
    runtime.task.plan_success = False
    _rejected(runtime, "earlier planning failure")
    assert not runtime.clock.plans and not runtime.clock.commands
    assert runtime.clock.steps == 0


def test_new_planning_failure_cannot_be_cleared_by_quiet_arrival(runtime):
    original_step = runtime.task.step_sim

    def fail_during_wait():
        original_step()
        if runtime.clock.steps == 40:
            runtime.task.plan_success = False

    runtime.task.step_sim = fail_during_wait
    _rejected(runtime, "planning failure latched during descent")


def test_avatar_retreat_aborts_stale_preflight_without_opening(runtime):
    runtime.task._avatar_retreat_enabled = lambda: runtime.clock.steps == 3
    _rejected(runtime, "avatar retreat is incompatible")
    assert runtime.clock.steps == 3
    assert len(runtime.clock.commands) == len(runtime.clock.grips) == 3


@pytest.mark.parametrize("mode", ["enabled", "checker", "quiet_wait"])
def test_retreat_actuator_cannot_open_before_rejection(runtime, mode):
    calls = []

    def puts_down(tags):
        calls.append(tags)
        runtime.arm.gripper_val = 1.
        return True

    runtime.task._apply_avatar_retreat_if_needed = puts_down
    if mode == "checker":
        runtime.task.avatar_retreat_checker = object()
    else:
        runtime.task._avatar_retreat_enabled = lambda: mode == "enabled" or runtime.clock.steps >= 20
    _rejected(runtime, "avatar retreat is incompatible")
    assert not calls and runtime.arm.gripper_val == .35
    assert runtime.clock.steps == (20 if mode == "quiet_wait" else 0)


def test_debug_bounded_planner_is_never_a_fallback(runtime):
    runtime.arm.planner._bounded_paths_debug_enabled = lambda: True
    _rejected(runtime, "native MPLib")
    assert not runtime.clock.plans and not runtime.clock.commands


@pytest.mark.parametrize("stage", ["planner", "validator"])
def test_planner_or_preflight_exception_latches_failure_without_execution(runtime, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise ValueError("injected rejection")

    if stage == "planner":
        runtime.arm.planner.plan_screw_path = fail
    else:
        monkeypatch.setattr(release, "validate_release_plan", fail)
    _rejected(runtime, "injected rejection")
    assert not runtime.clock.commands and runtime.clock.steps == 0


@pytest.mark.parametrize("plan", [None, SimpleNamespace(success=False)])
def test_unsuccessful_native_plan_never_executes_or_opens(runtime, plan):
    runtime.arm.planner.plan_screw_path = lambda *args, **kwargs: plan
    _rejected(runtime, "unsuccessful plan")
    assert not runtime.clock.commands and not runtime.clock.grips


@pytest.mark.parametrize("field", ["position", "quaternion", "bounds", "fingers", "tcp"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_observed_geometry_fails_before_motion(runtime, field, invalid):
    runtime.initial[field].flat[0] = invalid
    _rejected(runtime, "invalid release observation")
    assert not runtime.clock.plans and runtime.clock.steps == 0


@pytest.mark.parametrize("field", ["bounds", "fingers"])
def test_inverted_geometry_fails_before_motion(runtime, field):
    if field == "bounds":
        runtime.initial[field][0, 0] = runtime.initial[field][1, 0]
    else:
        runtime.initial[field][0, 0, 0] = runtime.initial[field][0, 1, 0]
    _rejected(runtime, "empty release geometry")
    assert not runtime.clock.commands


def test_missing_finger_geometry_fails_closed(runtime):
    runtime.arm.entity.links.pop()
    _rejected(runtime, "finger geometry")
    assert not runtime.clock.plans


@pytest.mark.parametrize("field,value", [("gripper_val", np.nan), ("dt", 0.), ("dt", np.inf)])
def test_invalid_command_or_timestep_fails_closed(runtime, field, value):
    if field == "gripper_val":
        runtime.arm.gripper_val = value
    else:
        runtime.task.SIMULATION_DT = value
    _rejected(runtime, "invalid")
    assert not runtime.clock.plans


@pytest.mark.parametrize("field", ["arm", "full"])
def test_nonfinite_joint_readings_do_not_poison_rejection_diagnostics(runtime, field):
    if field == "arm":
        runtime.arm.get_arm_qpos = lambda: np.full(7, np.nan)
    else:
        runtime.arm.entity.get_qpos = lambda: np.full(9, np.nan)
    _rejected(runtime, "")
    assert not runtime.clock.commands


def test_missing_arm_latches_failure_instead_of_escaping(runtime):
    def missing(tag):
        raise ValueError("arm unavailable")

    runtime.task.robot.get_arm = missing
    _rejected(runtime, "arm unavailable")
    assert runtime.clock.steps == 0


def _enable_terminal_diagnostics(runtime):
    runtime.task.config = {"guarded_release_diagnostics": True}
    runtime.task._video_frames = [None] * 17
    runtime.arm._cached_target = np.r_[np.full(7, .1), .014, .014]


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_diagnostics_are_opt_in_and_do_not_change_execution(runtime, capsys, enabled):
    if enabled:
        _enable_terminal_diagnostics(runtime)
    assert release.lower_for_release(runtime.task, "right") is True
    assert runtime.clock.steps == 112
    assert len(runtime.clock.commands) == len(runtime.clock.grips) == 12
    assert len(runtime.clock.plans) == len(runtime.clock.validations) == 1
    rows = [json.loads(line.split("] ", 1)[1]) for line in capsys.readouterr().out.splitlines()
            if line.startswith("[take-guarded-release]")]
    assert [row["stage"] for row in rows] == ["before_plan", "plan_checked", "ready_to_open"]
    assert all("terminal_snapshot" not in row for row in rows[:-1])
    assert ("terminal_snapshot" in rows[-1]) is enabled
    if enabled:
        snapshot = rows[-1]["terminal_snapshot"]
        assert snapshot["progress"]["physics_steps"] == 112
        assert snapshot["progress"]["phase"] == "quiet_wait"
        assert snapshot["progress"]["waypoint_count"] == 2
        assert snapshot["physical_time_s"] == pytest.approx(.112)
        assert snapshot["video_frames"] == 17
        assert snapshot["native_endpoint_fk"]["available"] is False
        assert not snapshot["observation_errors"]
        np.testing.assert_allclose(snapshot["measured_tcp_minus_target_xyz_m"], 0., atol=1e-12)


def test_terminal_snapshot_preserves_first_rejected_state_not_later_reread(runtime, monkeypatch):
    _enable_terminal_diagnostics(runtime)

    def disturbed(state, step):
        if step == 7:
            state["position"][2] -= .004
            state["bounds"][:, 2] -= .004
        return state

    runtime.clock.adjust = disturbed
    original = release._retained

    def retained(state, initial, plate):
        try:
            return original(state, initial, plate)
        except ValueError:
            # A diagnostic re-read would now miss the first bad sample.
            runtime.clock.adjust = lambda state, step: state
            raise

    monkeypatch.setattr(release, "_retained", retained)
    diagnostic = _rejected(runtime, "plate-rim clearance")
    snapshot = diagnostic["terminal_snapshot"]
    assert snapshot["progress"] == {"phase": "descent", "physics_steps": 7,
                                     "waypoint_index": 1, "waypoint_count": 2, "substep": 0}
    assert snapshot["bottom_above_reference_rim_m"] == pytest.approx(.001)
    np.testing.assert_allclose(snapshot["origin_offset_drift_world_m"], [0., 0., -.004], atol=1e-12)
    np.testing.assert_allclose(snapshot["measured_tcp_minus_target_xyz_m"], 0., atol=1e-12)
    np.testing.assert_array_equal(snapshot["commanded_waypoint"], np.full(7, .1))
    np.testing.assert_array_equal(snapshot["full_qpos"], np.r_[np.full(7, .1), .014, .014])
    np.testing.assert_array_equal(snapshot["cached_qpos"], runtime.arm._cached_target)
    np.testing.assert_array_equal(snapshot["live_plate_bounds"], PLATE)
    assert set(snapshot["measured"]) == {"tcp", "position", "quaternion", "bounds", "fingers"}
    assert runtime.clock.steps == len(runtime.clock.commands) == len(runtime.clock.grips) == 7
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("reject", [False, True])
def test_terminal_diagnostic_failure_cannot_change_controller_outcome(runtime, monkeypatch, reject):
    _enable_terminal_diagnostics(runtime)

    def unavailable(*args):
        raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(release, "_terminal_snapshot", unavailable)
    if reject:
        runtime.task.plan_success = False
        _rejected(runtime, "earlier planning failure")
        assert runtime.clock.steps == 0
    else:
        assert release.lower_for_release(runtime.task, "right") is True
        assert runtime.task.plan_success is True
        assert runtime.clock.steps == 112
    snapshot = runtime.task._guarded_release_diagnostics["terminal_snapshot"]
    assert "diagnostics unavailable" in snapshot["observation_errors"]["snapshot"]


def test_terminal_partial_getter_errors_preserve_other_fields_and_success(runtime):
    _enable_terminal_diagnostics(runtime)
    calls = []

    def plate_bounds():
        calls.append(None)
        if len(calls) > 1:
            raise RuntimeError("plate observation unavailable")
        return PLATE.copy()

    runtime.task.plate_actor.entity.get_AABB = plate_bounds
    runtime.arm._cached_target[0] = np.nan
    assert release.lower_for_release(runtime.task, "right") is True
    snapshot = runtime.task._guarded_release_diagnostics["terminal_snapshot"]
    assert set(snapshot["observation_errors"]) == {"live_plate_bounds", "cached_qpos"}
    assert "measured" in snapshot and "full_qpos" in snapshot
    assert runtime.clock.steps == 112
    json.dumps(snapshot, allow_nan=False)


def test_terminal_signed_native_endpoint_comparisons_use_tcp_offset_without_fk(runtime, monkeypatch):
    _enable_terminal_diagnostics(runtime)
    runtime.arm.tcp_offset = release.Pose([0., 0., .1034], [1., 0., 0., 0.])
    bias = np.array([.001, -.002, .003])

    def published_endpoint(arm, plan, target, **kwargs):
        link = target.copy()
        link[:3] += bias
        link[2] -= .1034
        return {"endpoint_world_link_pose7": link.tolist()}

    monkeypatch.setattr(release, "validate_release_plan", published_endpoint)
    assert release.lower_for_release(runtime.task, "right") is True
    native = runtime.task._guarded_release_diagnostics["terminal_snapshot"]["native_endpoint_fk"]
    assert native["available"] is True
    assert "not instantaneous" in native["comparison"]
    np.testing.assert_allclose(native["native_endpoint_tcp_minus_target_xyz_m"], bias, atol=1e-12)
    np.testing.assert_allclose(native["measured_tcp_minus_native_endpoint_xyz_m"], -bias, atol=1e-12)
    assert runtime.clock.steps == 112
