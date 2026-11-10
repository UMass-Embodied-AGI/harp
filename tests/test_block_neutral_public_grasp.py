"""Public fallback preserves checked execution, sticky failures and transfer control."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs.manipulation import PickSpec, PlaceSpec
from envs.planning.base import PlanResult
from envs.task_bases import block_neutral_public_grasp as public
from envs.task_bases import blocks_ranking_rgb as family
from envs.utils import Pose


@pytest.fixture
def trial(monkeypatch):
    events = []
    state = {"center": np.array([.1, -.3, .785]), "q": np.zeros(9), "steps": 0,
             "move_after_first": False, "contact_after_first": False, "decline": False,
             "real_finish": False, "helper_calls": 0, "held": False, "ee": np.array([.1, -.3, .779]),
             "ee_quat": np.array([1., 0., 0., 0.])}
    state["retreat"] = "success"
    state["plan_qpos_calls"] = []

    def plan_qpos(goal_qposes, current_qpos, **kwargs):
        state["plan_qpos_calls"].append((np.asarray(goal_qposes[0]).copy(), np.asarray(current_qpos).copy(), kwargs))
        if state["retreat"] == "raise":
            raise RuntimeError("planner unavailable")
        if state["retreat"] == "fail":
            return {"status": "Fail"}
        rows = np.linspace(np.asarray(current_qpos, float), np.asarray(goal_qposes[0], float), 5)
        if state["retreat"] == "short":
            rows = rows[:-1]
        return {"status": "Success", "position": rows, "velocity": np.zeros_like(rows)}

    arm = SimpleNamespace(entity=SimpleNamespace(get_qpos=lambda: state["q"].copy()),
        get_arm_qpos=lambda: state["q"][:7].copy(),
        planner=SimpleNamespace(_mplib=SimpleNamespace(plan_qpos=plan_qpos), _pad_qpos=lambda q: np.asarray(q, float), _timeout=None),
        homestate=[0., .19634954, 0., -2.61799388, 0., 2.94159265, .78539816],
        gripper_val=0., gripper_open=.04, tcp_offset=Pose([0., 0., .1034], [1., 0., 0., 0.]),
        get_ee_pose=lambda: np.r_[state["ee"], state["ee_quat"]], finger_joints=[])

    class Parent:
        def execute_plan(self, result, arm_tag, gripper_result=None):
            events.append(("execute", result, arm_tag, gripper_result))
            state["q"][:7] = result.position[-1]
            context = getattr(self, "_block_public_execution", None)
            if context is not None:
                target = context["targets"][context["executed"]]
                state["ee"], state["ee_quat"] = target.p.copy(), target.q.copy()
                if state.get("miss_position"):
                    state["ee"][2] += .04
                if state.get("miss_angle"):
                    state["ee_quat"] = np.array([1., 0., 0., 0.])
                if state.get("invalid_tcp"):
                    state["ee"][0] = np.nan
                if state.get("lag_steps"):
                    # PD lag: the measured TCP is still short of the commanded
                    # pose when execute_plan returns and converges over steps.
                    state["lag_goal"] = (state["ee"].copy(), state["ee_quat"].copy())
                    state["lag_remaining"] = int(state["lag_steps"])
                    state["ee"] = state["ee"] + np.array([.0, .0, .02])
                    state["ee_quat"] = np.array([.05, np.sqrt(1 - .05 ** 2), 0., 0.])
            if len([e for e in events if e[0] == "execute"]) == 1:
                if state["move_after_first"]:
                    state["center"][0] += state.get("shift_m", .003)
                if state["contact_after_first"]:
                    self.avatar_collided = True
            return result

    class Task(public.BlockNeutralPublicGraspMixin, Parent, family.BlocksRankingRGB):
        def __init__(self):
            pass

        def step_sim(self):
            state["steps"] += 1
            if state.get("move_during_settle") and state["steps"] == 350:
                state["center"][0] += .003
            remaining = state.get("lag_remaining", 0)
            if remaining > 0:
                goal_p, goal_q = state["lag_goal"]
                state["ee"] = state["ee"] + (goal_p - state["ee"]) / remaining
                state["ee_quat"] = state["ee_quat"] + (goal_q - state["ee_quat"]) / remaining
                state["ee_quat"] = state["ee_quat"] / np.linalg.norm(state["ee_quat"])
                state["lag_remaining"] = remaining - 1
            if state.get("drift_per_step"):
                state["ee"] = state["ee"] + np.array([state["drift_per_step"], 0., 0.])
            if state.get("contact_at_step") == state["steps"]:
                self.avatar_collided = True

        def open_gripper(self, tag):
            events.append(("open", tag))
            arm.gripper_val = 1.
            state["q"][-2:] = .04
            state["held"] = False
            for _ in range(200):
                self.step_sim()

        def set_gripper(self, value, tag):
            events.append(("close", value, tag))
            arm.gripper_val = value
            state["held"] = True

        def _move_cartesian(self, start, end, tag, **kwargs):
            events.append(("tiny", np.asarray(start).copy(), np.asarray(end).copy(), kwargs))
            for _ in range(kwargs["n_steps"] * kwargs["sim_per_step"]):
                self.step_sim()
            state["ee"] = np.asarray(end).copy()
            state["center"] = state["ee"] + [0., 0., .006]

        def _move_screw(self, pos, tag):
            events.append(("screw", np.asarray(pos).copy(), self._active_block_grasp_plan.yaw))
            state["ee"] = np.asarray(pos).copy()
            if state["held"]:
                state["center"] = state["ee"] + [0., 0., .006]
            return True

        def select_and_execute_grasp(self, name, pose, tag, **kwargs):
            state["helper_calls"] += 1
            state["helper_args"] = (name, pose, tag, kwargs)
            assert kwargs["object_scale"] == 1. and kwargs["plan_attempts"] == 2
            assert kwargs["max_candidates"] == 4
            assert all(g.pre_distance == .12 and g.scale_frame == "runtime" for g in kwargs["candidate_grasps"])
            assert not np.shares_memory(arm._cached_target, state["q"])
            np.testing.assert_array_equal(arm._cached_target, state["q"])
            if state["decline"]:
                return None
            q = arm.get_arm_qpos()
            pre = PlanResult(True, np.stack([q, q + .01]))
            grasp = PlanResult(True, np.stack([q + .01, q + .02]))
            chosen = kwargs["candidate_grasps"][2]
            tcp = chosen.to_world(pose)
            link = public.tcp_to_link_pose(tcp, arm.tcp_offset)
            plans = (pre, grasp, link, link)
            if not kwargs["post_plan_check"](plans):
                return None
            self.execute_plan(pre, tag)
            self.execute_plan(grasp, tag)
            state["ee"] = tcp.p.copy()
            return link, link, chosen

        def _finish_public_block_transfer(self, pick, place, tag, center, grasp):
            events.append(("finish", self._active_block_grasp_plan.yaw,
                           list(self._active_block_grasp_plan.legs)))
            if state["real_finish"]:
                return public.BlockNeutralPublicGraspMixin._finish_public_block_transfer(
                    self, pick, place, tag, center, grasp)
            return True

    task = Task()
    task.BLOCK_CAN_PUBLIC_GRASP_RECOVERY = True
    task._neutral_avatar_started = True
    task._neutral_avatar_job = {"kind": "put_objects_in_bowl"}
    task.TABLE_TOP_Z = .765
    task.config = {}
    task.robot = SimpleNamespace(get_arm=lambda tag: arm)
    task.plan_success = False  # An earlier failure must never be reset by this helper.
    task.avatar_collided = False
    task._active_block_grasp_plan = None
    task._block_neighbor_scene = {"target": "red", "arm_tag": "right"}
    task.blocks = {"red": SimpleNamespace(get_quat=lambda: np.array([1., 0., 0., 0.]))}
    pick = PickSpec(lambda: state["center"].copy(), .02, "red block")
    place = PlaceSpec(np.array([-.16, -.08, .845]), "red slot", transport_z=.965)
    monkeypatch.setattr(public, "refresh_block_obstacles", lambda *a: events.append(("refresh", state["steps"])))
    monkeypatch.setattr(public, "endpoint_matches", lambda *a: True)
    monkeypatch.setattr(public, "_neighbor_endpoint_clearance_ok", lambda *a: True)
    monkeypatch.setattr(public, "_open_paths_clear", lambda *a: True)
    return SimpleNamespace(task=task, arm=arm, state=state, pick=pick, place=place, events=events)


def run(t):
    return t.task._block_preflight_fallback(t.pick, t.place, "right")


@pytest.mark.parametrize("field,value", [("BLOCK_CAN_PUBLIC_GRASP_RECOVERY", False),
    ("_neutral_avatar_started", False), ("_neutral_avatar_job", None),
    ("_neutral_avatar_job", {}), ("avatar_collided", True)])
def test_inactive_scope_has_no_commands_or_helper_calls(trial, field, value):
    setattr(trial.task, field, value)
    assert run(trial) is None
    assert not trial.events and trial.state["helper_calls"] == 0
    assert not trial.task.plan_success
    assert trial.task._block_preflight_hold(0, "right") is False


@pytest.mark.parametrize("kind", ["writing", "sponge", "mouse", "sitting_clap", "bottle", "put_objects_in_bowl"])
def test_every_started_neutral_kind_is_eligible(trial, kind):
    trial.task._neutral_avatar_job = {"kind": kind}
    assert run(trial) is True
    assert [e[0] for e in trial.events] == ["open", "refresh", "execute", "execute", "finish"]


def test_declined_and_no_route_return_none_without_motion(trial):
    trial.state["decline"] = True
    assert run(trial) is None
    assert [e[0] for e in trial.events] == ["open", "refresh"]
    trial.events.clear()
    trial.state["decline"] = False
    trial.task.avatar_collided = True
    assert run(trial) is None
    assert not trial.events


def test_preflight_hold_retreats_home_once_then_holds_the_fixed_home_target(trial):
    holds = []
    trial.arm.set_arm_joints = lambda q: holds.append(np.asarray(q).copy())
    trial.state["q"][:7] = [.1, .2, .3, -2., .0, 2.5, .8]
    home = np.asarray(trial.arm.homestate)
    before = trial.state["steps"]
    assert trial.task._block_preflight_hold(0, "right") is True
    # One checked plan from the measured joints to home, obstacles refreshed first,
    # executed through the ordinary execute_plan, then the arrival settle (50 steps).
    assert len(trial.state["plan_qpos_calls"]) == 1
    goal, start, kwargs = trial.state["plan_qpos_calls"][0]
    np.testing.assert_array_equal(goal, home)
    np.testing.assert_array_equal(start, [.1, .2, .3, -2., .0, 2.5, .8])
    assert kwargs["rrt_range"] == .3 and kwargs["planning_time"] == 10
    executed = [e for e in trial.events if e[0] == "execute"]
    assert len(executed) == 1 and executed[0][2] == "right"
    np.testing.assert_array_equal(executed[0][1].position[-1], home)
    refreshes = [e for e in trial.events if e[0] == "refresh"]
    assert refreshes and refreshes[0][1] == before
    assert trial.state["steps"] - before == trial.task.BLOCK_PREFLIGHT_HOLD_STEPS + 50
    assert len(holds) == 1
    np.testing.assert_array_equal(holds[0], home)
    # Later rounds never re-plan and never re-target measured joints, even when
    # the measured arm has sagged below the commanded home.
    trial.state["q"][:7] = home - .03
    before = trial.state["steps"]
    assert trial.task._block_preflight_hold(1, "right") is True
    assert trial.state["steps"] - before == trial.task.BLOCK_PREFLIGHT_HOLD_STEPS
    assert len(trial.state["plan_qpos_calls"]) == 1 and len(holds) == 2
    np.testing.assert_array_equal(holds[1], home)
    assert trial.task._block_preflight_hold(trial.task.BLOCK_PREFLIGHT_HOLD_ROUNDS - 1, "right") is True
    assert trial.task._block_preflight_hold(trial.task.BLOCK_PREFLIGHT_HOLD_ROUNDS, "right") is False
    assert not trial.task.plan_success


@pytest.mark.parametrize("retreat", ["fail", "raise", "short"])
def test_preflight_hold_without_a_checked_route_holds_the_commanded_target(trial, retreat):
    holds = []
    trial.arm.set_arm_joints = lambda q: holds.append(np.asarray(q).copy())
    trial.state["retreat"] = retreat
    commanded = np.array([.1, .2, .3, -2., .0, 2.5, .8, .04, .04])
    trial.arm._cached_target = commanded.copy()
    trial.state["q"][:7] = commanded[:7] - .03  # measured sag below the command
    before = trial.state["steps"]
    assert trial.task._block_preflight_hold(0, "right") is True
    assert trial.state["steps"] - before == trial.task.BLOCK_PREFLIGHT_HOLD_STEPS
    assert not [e for e in trial.events if e[0] == "execute"]
    assert len(holds) == 1
    np.testing.assert_array_equal(holds[0], commanded[:7])
    assert trial.task._block_preflight_hold(1, "right") is True
    np.testing.assert_array_equal(holds[1], commanded[:7])
    assert len(trial.state["plan_qpos_calls"]) == 1


def test_preflight_hold_skips_the_retreat_when_already_home(trial):
    holds = []
    trial.arm.set_arm_joints = lambda q: holds.append(np.asarray(q).copy())
    home = np.asarray(trial.arm.homestate)
    trial.state["q"][:7] = home + .01
    assert trial.task._block_preflight_hold(0, "right") is True
    assert not trial.state["plan_qpos_calls"]
    assert not [e for e in trial.events if e[0] == "execute"]
    np.testing.assert_array_equal(holds[0], home)


def test_preflight_hold_stops_on_sticky_avatar_contact(trial):
    holds = []
    trial.arm.set_arm_joints = lambda q: holds.append(np.asarray(q).copy())
    trial.state["q"][:7] = [.1, .2, .3, -2., .0, 2.5, .8]
    # Contact while settling after the retreat: nothing is held, False is sticky.
    trial.state["contact_at_step"] = trial.state["steps"] + 37
    assert trial.task._block_preflight_hold(0, "right") is False
    assert trial.task.avatar_collided and not holds
    assert trial.task._block_preflight_hold(1, "right") is False


def test_preflight_hold_stops_on_contact_during_the_hold(trial):
    trial.arm.set_arm_joints = lambda q: None
    trial.state["q"][:7] = np.asarray(trial.arm.homestate) + .01
    trial.state["contact_at_step"] = trial.state["steps"] + 37
    assert trial.task._block_preflight_hold(0, "right") is False
    assert trial.task.avatar_collided
    assert trial.task._block_preflight_hold(1, "right") is False


@pytest.mark.parametrize("fallback_outcomes,holds,expected,calls", [
    ([None, None], [True, False], False, ["fallback", "hold", "fallback", "hold"]),
    ([None], [True], True, ["fallback", "hold", "original"]),
    ([False], [], False, ["fallback"]),
    ([True], [], True, ["fallback"]),
    ([None], [], False, ["fallback"]),
])
def test_family_dispatch_retries_preflight_between_bounded_holds(monkeypatch, fallback_outcomes, holds, expected, calls):
    seen = []
    cube = SimpleNamespace(get_pos=lambda: np.array([.1, -.3, .785]))
    task = SimpleNamespace(blocks={"red": cube}, target_slots={"red": [-.16, -.08]},
        block_half_size=lambda key: .02, block_pick_label=lambda key: key,
        TABLE_TOP_Z=.765, TRANSPORT_Z_ABOVE_TABLE=.2, RELEASE_Z_ABOVE_TABLE=.08,
        BLOCK_GRASP_PREFLIGHT=True, pick_and_place=lambda *a: seen.append("original") or True)
    outcomes = list(fallback_outcomes)
    task._block_preflight_fallback = lambda *a: seen.append("fallback") or outcomes.pop(0)
    hold_results = list(holds)
    if hold_results:
        task._block_preflight_hold = lambda r, tag: seen.append("hold") or hold_results.pop(0)
    plans = [None] * (len(fallback_outcomes)) + [object()]
    monkeypatch.setattr(family, "block_neighbor_obstacles", lambda *a: nullcontext())
    monkeypatch.setattr(family, "plan_block_grasp", lambda *a: plans.pop(0))
    assert family.BlocksRankingRGB._pick_and_place_block(task, "red") == expected
    assert seen == calls


def test_public_entry_and_yaw_preserve_sticky_failure(trial):
    assert run(trial)
    assert [e[0] for e in trial.events] == ["open", "refresh", "execute", "execute", "finish"]
    assert trial.events[1] == ("refresh", 200)
    assert trial.events[-1][1:] == (-np.pi / 2, [])
    assert not trial.task.plan_success and not trial.task.avatar_collided
    assert trial.task._block_public_execution is None
    assert trial.task._active_block_grasp_plan is None


@pytest.mark.parametrize("reject", ["endpoint_matches", "_neighbor_endpoint_clearance_ok", "_open_paths_clear"])
def test_failed_gate_never_executes(trial, monkeypatch, reject):
    monkeypatch.setattr(public, reject, lambda *a: False)
    assert not run(trial)
    assert not any(e[0] in ("execute", "finish", "close") for e in trial.events)
    assert trial.task._block_public_execution is None


@pytest.mark.parametrize("kind", ["move_after_first", "contact_after_first"])
def test_first_leg_disturbance_stops_second_leg_and_close(trial, kind):
    trial.state[kind] = True
    with pytest.raises(RuntimeError):
        run(trial)
    assert len([e for e in trial.events if e[0] == "execute"]) == 1
    assert not any(e[0] in ("finish", "close") for e in trial.events)
    assert trial.task._block_public_execution is None
    assert trial.task._active_block_grasp_plan is None
    if kind == "contact_after_first":
        assert trial.task.avatar_collided


def test_no_selected_grasp_cleans_context_without_execution(trial):
    trial.state["decline"] = True
    assert not run(trial)
    assert trial.task._block_public_execution is None
    assert [e[0] for e in trial.events] == ["open", "refresh"]


def test_subthreshold_target_motion_retains_original_tolerance(trial):
    trial.state.update(move_after_first=True, shift_m=.0019)
    assert run(trial)
    assert len([e for e in trial.events if e[0] == "execute"]) == 2


def test_nonfinite_measured_joints_never_reach_public_planning(trial):
    trial.state["q"][0] = np.nan
    with pytest.raises(ValueError, match="measured block arm"):
        run(trial)
    assert trial.state["helper_calls"] == 0
    assert not trial.task.plan_success


def test_active_legacy_plan_rejected_before_physical_commands(trial):
    sentinel = object()
    trial.task._active_block_grasp_plan = sentinel
    with pytest.raises(RuntimeError, match="active legacy"):
        run(trial)
    assert not trial.events
    assert trial.task._active_block_grasp_plan is sentinel


def test_post_plan_exception_cleans_context_preserving_identity(trial, monkeypatch):
    error = RuntimeError("path checker failed")
    def broken(*args):
        raise error
    monkeypatch.setattr(public, "_open_paths_clear", broken)
    with pytest.raises(RuntimeError) as caught:
        run(trial)
    assert caught.value is error
    assert trial.task._block_public_execution is None
    assert not any(e[0] == "execute" for e in trial.events)


@pytest.mark.parametrize("failure", ["miss_position", "miss_angle", "invalid_tcp"])
def test_measured_arrival_failure_stops_descent_before_close(trial, failure):
    trial.state[failure] = True
    with pytest.raises(RuntimeError, match="TCP"):
        run(trial)
    assert len([e for e in trial.events if e[0] == "execute"]) == 1
    assert not any(e[0] in ("close", "finish") for e in trial.events)
    assert trial.task._block_public_execution is None
    assert not trial.task.plan_success


def test_default_execute_plan_delegates_exact_object_and_result(trial):
    result = PlanResult(True, np.zeros((2, 7)))
    fingers = object()
    assert trial.task.execute_plan(result, "right", fingers) is result
    assert trial.events == [("execute", result, "right", fingers)]


def test_transfer_keeps_partial_close_tiny_lift_yaw_and_release(trial):
    trial.state["real_finish"] = True
    # The post-release rise is covered in test_block_post_release.py.
    trial.task.BLOCK_RETREAT_AFTER_RELEASE = False
    assert run(trial)
    assert [e for e in trial.events if e[0] == "close"] == [("close", .375, "right")]
    tiny = [e for e in trial.events if e[0] == "tiny"]
    assert len(tiny) == 1 and tiny[0][3] == {"n_steps": 30, "sim_per_step": 30}
    screws = [e for e in trial.events if e[0] == "screw"]
    assert len(screws) == 3 and all(e[2] == -np.pi / 2 for e in screws)
    np.testing.assert_allclose(screws[0][1], [.1, -.3, .965])
    np.testing.assert_allclose(screws[1][1], [-.16, -.08, .965])
    np.testing.assert_allclose(screws[2][1], [-.16, -.08, .845])
    assert len([e for e in trial.events if e[0] == "open"]) == 2
    # 2000 original steps plus the two minimum 50-step arrival settles.
    assert trial.state["steps"] == 2100
    assert not trial.task.plan_success and trial.task._active_block_grasp_plan is None


def _settle_steps(trial, monkeypatch):
    taken = []
    original = trial.task._settle_public_block_arrival

    def record(arm_tag):
        steps = original(arm_tag)
        taken.append(steps)
        return steps

    monkeypatch.setattr(trial.task, "_settle_public_block_arrival", record)
    return taken


def test_arrival_settle_is_minimum_when_tcp_is_stationary(trial, monkeypatch):
    taken = _settle_steps(trial, monkeypatch)
    assert run(trial)
    assert taken == [50, 50]
    assert [e[0] for e in trial.events] == ["open", "refresh", "execute", "execute", "finish"]


def test_arrival_settle_absorbs_pd_lag_before_judging_arrival(trial, monkeypatch):
    taken = _settle_steps(trial, monkeypatch)
    trial.state["lag_steps"] = 80  # 2 cm / 0.1 rad transient, gone after 80 steps
    assert run(trial)
    assert len([e for e in trial.events if e[0] == "execute"]) == 2
    assert all(80 <= steps <= 100 for steps in taken), taken
    # 200 fixture open steps plus both settles; the stubbed transfer takes none.
    assert trial.state["steps"] == 200 + sum(taken)


def test_arrival_settle_is_bounded_and_still_rejects_a_drifting_tcp(trial, monkeypatch):
    taken = _settle_steps(trial, monkeypatch)
    trial.state["drift_per_step"] = .001
    with pytest.raises(RuntimeError, match="TCP"):
        run(trial)
    assert taken == [300]
    assert len([e for e in trial.events if e[0] == "execute"]) == 1
    assert not any(e[0] in ("close", "finish") for e in trial.events)
    assert trial.task._block_public_execution is None


def test_arrival_settle_stops_on_sticky_avatar_contact(trial, monkeypatch):
    taken = _settle_steps(trial, monkeypatch)
    trial.state["contact_at_step"] = 215  # during the first leg's settle
    with pytest.raises(RuntimeError, match="avatar contact"):
        run(trial)
    assert taken == [20]
    assert trial.task.avatar_collided
    assert len([e for e in trial.events if e[0] == "execute"]) == 1
    assert not any(e[0] in ("close", "finish") for e in trial.events)


def test_target_motion_during_settle_aborts_before_close(trial):
    trial.state.update(real_finish=True, move_during_settle=True)
    with pytest.raises(RuntimeError, match="block moved"):
        run(trial)
    assert not any(e[0] in ("close", "tiny") for e in trial.events)
    assert trial.task._active_block_grasp_plan is None


@pytest.mark.parametrize("preflight,fallback", [(None, None), (None, True), (object(), True)])
def test_family_hook_only_runs_after_original_preflight_exhausts(monkeypatch, preflight, fallback):
    calls = []
    cube = SimpleNamespace(get_pos=lambda: np.array([.1, -.3, .785]))
    task = SimpleNamespace(blocks={"red": cube}, target_slots={"red": [-.16, -.08]},
        block_half_size=lambda key: .02, block_pick_label=lambda key: key,
        TABLE_TOP_Z=.765, TRANSPORT_Z_ABOVE_TABLE=.2, RELEASE_Z_ABOVE_TABLE=.08,
        BLOCK_GRASP_PREFLIGHT=True, pick_and_place=lambda *a: calls.append("original") or True)
    if fallback is not None:
        task._block_preflight_fallback = lambda *a: calls.append("fallback") or True
    monkeypatch.setattr(family, "block_neighbor_obstacles", lambda *a: nullcontext())
    monkeypatch.setattr(family, "plan_block_grasp", lambda *a: preflight)
    expected = preflight is not None or fallback is not None
    assert family.BlocksRankingRGB._pick_and_place_block(task, "red") == expected
    assert calls == (["original"] if preflight is not None else ["fallback"] if fallback is not None else [])


def test_native_open_jaw_target_guard_and_model_restoration():
    from envs.planning.mplib_planner import MplibPlanner
    asset = Path(__file__).resolve().parents[1] / "assets/embodiments/franka-panda"
    planner = MplibPlanner(str(asset / "panda.urdf"), str(asset / "panda.srdf"),
        "panda_hand", [0., 0., 0., 1., 0., 0., 0.], n_arm=7)
    q = np.array([0., .2, 0., -2.62, 0., 2.94, .785])
    plans = [PlanResult(True, np.stack([q, q])) for _ in range(2)]
    arm = SimpleNamespace(gripper_open=.04)
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda tag: arm),
        _block_neighbor_scene={"arm_tag": "right", "target": "red"},
        blocks={"red": SimpleNamespace(get_quat=lambda: np.array([1., 0., 0., 0.]))})
    before = np.asarray(planner._mplib.robot.get_qpos()).copy()
    assert public._open_paths_clear(task, planner, plans, q, np.array([2., 2., 2.]), .02)
    np.testing.assert_array_equal(planner._mplib.robot.get_qpos(), before)
    assert not public._open_paths_clear(task, planner, plans, q, np.array([0., 0., .4]), .5)
    np.testing.assert_array_equal(planner._mplib.robot.get_qpos(), before)
    original = planner._mplib.check_for_env_collision
    def broken(*args):
        raise RuntimeError("native check failed")
    planner._mplib.check_for_env_collision = broken
    try:
        with pytest.raises(RuntimeError, match="native check failed"):
            public._open_paths_clear(task, planner, plans, q, np.array([2., 2., 2.]), .02)
        np.testing.assert_array_equal(planner._mplib.robot.get_qpos(), before)
    finally:
        planner._mplib.check_for_env_collision = original


@pytest.mark.parametrize("bad", ["nan", "join", "velocity"])
def test_invalid_path_rejected_before_native_calls(bad):
    q = np.zeros(7)
    pre, grasp = [PlanResult(True, np.zeros((2, 7))) for _ in range(2)]
    if bad == "nan":
        grasp.position[1, 0] = np.nan
    elif bad == "join":
        grasp.position[0, 0] = .1
    else:
        grasp.velocity[0, 0] = np.nan
    assert not public._open_paths_clear(None, None, (pre, grasp), q, np.zeros(3), .02)
