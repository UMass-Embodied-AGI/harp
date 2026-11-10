"""Optional measured annotations keep the selector's normal safety contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task_parts import grasp_select
from envs.grasp import GraspPose
from envs.planning.base import PlanResult
from envs.utils import Pose


def candidate():
    return GraspPose("measured", [0, 0, 0], [0, 1, 0, 0], scale_frame="runtime")


def test_explicit_candidates_skip_loader_but_use_normal_preflight(monkeypatch):
    monkeypatch.setattr(grasp_select, "load_grasp_poses", lambda *_a, **_k: pytest.fail("YAML loaded"))
    planned, executed = [], []
    result = PlanResult(True, np.ones((2, 7)))
    task = SimpleNamespace(
        config={}, TABLE_TOP_Z=0.765,
        robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(origin_pose=Pose([0, -0.5, 0]))),
        _plan_one_grasp=lambda g, *_a, **_k: planned.append(g) or (result, result, Pose([0, 0, 1]), Pose([0, 0, 1.12])),
        execute_plan=lambda *args: executed.append(args),
    )
    grasp = candidate()
    chosen = grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "100_seal", Pose([0, 0, 1]), "right", candidate_grasps=[grasp],
        post_plan_check=lambda _: False,
    )
    assert chosen is None and planned == [grasp] and executed == []


def test_none_candidates_preserves_default_loader_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(grasp_select, "load_grasp_poses", lambda *a, **kw: calls.append((a, kw)) or [])
    task = SimpleNamespace(config={"robot_type": "franka"}, TABLE_TOP_Z=0.765,
                           robot=SimpleNamespace(get_arm=lambda _: SimpleNamespace(origin_pose=Pose())))
    assert grasp_select._GraspSelectionMixin.select_and_execute_grasp(
        task, "100_seal", Pose([0, 0, 1]), "right", model_id=1, candidate_grasps=None,
    ) is None
    assert calls == [(("100_seal",), {"model_id": 1, "robot_type": "franka"})]


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_supplied_annotation_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        grasp_select._validate_candidate_grasps([candidate()], scale)


def test_supplied_annotation_requires_grasp_pose():
    with pytest.raises(TypeError):
        grasp_select._validate_candidate_grasps([object()], 1.0)


@pytest.mark.parametrize("field,value", [
    ("pre_distance", np.nan), ("pre_distance", 0), ("scale_frame", "unknown"),
])
def test_supplied_annotation_rejects_invalid_contract(field, value):
    grasp = candidate()
    setattr(grasp, field, value)
    with pytest.raises(ValueError):
        grasp_select._validate_candidate_grasps([grasp], 0.04)


def test_supplied_annotation_rejects_nonfinite_pose():
    grasp = candidate()
    grasp.pose.p[0] = np.nan
    with pytest.raises(ValueError):
        grasp_select._validate_candidate_grasps([grasp], 1.0)


def planner_with_fk(local_pose):
    calls = []
    backend = SimpleNamespace(
        pinocchio_model=SimpleNamespace(
            compute_forward_kinematics=lambda q: calls.append(q.copy()),
            get_link_pose=lambda _: local_pose,
        ),
        link_name_2_idx={"hand": 0}, move_group="hand",
        robot=SimpleNamespace(get_base_pose=lambda: Pose([0.1, -0.5, 0.7])),
    )
    plan = PlanResult(True, np.ones((2, 7)))
    return SimpleNamespace(
        _mplib=backend, _pad_qpos=lambda q: q.copy(),
        plan_path=lambda *_a, **_k: plan,
        plan_screw_path=lambda *_a, **_k: plan,
    ), calls


def test_mplib_fk_converts_base_frame_without_simulator_writes():
    planner, calls = planner_with_fk(Pose([0.2, 0.3, 0.4]))
    actual = grasp_select.mplib_world_link_pose(planner, np.zeros(7))
    np.testing.assert_allclose(actual.p, [0.3, -0.2, 1.1])
    assert len(calls) == 1


def test_mplib_fk_rotated_base_pads_without_mutating_joint_input():
    planner, calls = planner_with_fk(Pose([0.2, 0.3, 0.4]))
    planner._mplib.robot.get_base_pose = lambda: Pose([0.1, -0.5, 0.7], [2**-0.5, 0, 0, 2**-0.5])
    planner._pad_qpos = lambda q: np.concatenate([q, [0., 0.]])
    qpos = np.arange(7, dtype=float)
    original = qpos.copy()
    actual = grasp_select.mplib_world_link_pose(planner, qpos)
    np.testing.assert_allclose(actual.p, [-0.2, -0.3, 1.1], atol=1e-9)
    np.testing.assert_allclose(calls[0], np.concatenate([original, [0, 0]]))
    np.testing.assert_array_equal(qpos, original)


def test_plan_one_grasp_nonfinite_target_does_not_pass_nan_gate():
    planner, _ = planner_with_fk(Pose([-0.1, 0.5, 0.3], [0, 1, 0, 0]))
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), get_arm_qpos=lambda: np.zeros(7))
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: arm))
    assert grasp_select._GraspSelectionMixin._plan_one_grasp(
        task, candidate(), Pose([np.nan, 0, 1]), "right",
    ) is None


@pytest.mark.parametrize("requested,expected", [(None, {}), (.025, {"qpos_step": .025})])
def test_plan_one_grasp_precision_plumbing_preserves_omitted_arguments(requested, expected):
    planner, _ = planner_with_fk(Pose([-0.1, .5, .3], [0, 1, 0, 0]))
    calls = []
    planner.plan_screw_path = lambda *a, **kw: calls.append(kw) or PlanResult(True, np.ones((2, 7)))
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), get_arm_qpos=lambda: np.zeros(7))
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: arm))
    assert grasp_select._GraspSelectionMixin._plan_one_grasp(
        task, candidate(), Pose([0, 0, 1]), "right", screw_qpos_step=requested,
    ) is not None
    assert calls == [expected]


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, True, [.025], "invalid"])
def test_selector_rejects_invalid_precision_before_any_planning(invalid):
    with pytest.raises(ValueError):
        grasp_select._GraspSelectionMixin.select_and_execute_grasp(
            SimpleNamespace(), "seal", Pose(), "right", screw_qpos_step=invalid,
        )


@pytest.mark.parametrize("bad", [np.full(7, np.nan), np.empty(0), np.zeros((1, 7))])
def test_invalid_fk_joint_input_is_rejected_before_computation(bad):
    planner, calls = planner_with_fk(Pose())
    assert grasp_select.mplib_world_link_pose(planner, bad) is None
    assert calls == []


def test_missing_fk_has_no_simulator_fallback():
    assert grasp_select.mplib_world_link_pose(SimpleNamespace(), np.zeros(7)) is None


def test_plan_one_grasp_fk_never_calls_simulator_set_qpos():
    planner, calls = planner_with_fk(Pose([-0.1, 0.5, 0.3], [0, 1, 0, 0]))
    arm = SimpleNamespace(
        planner=planner, tcp_offset=Pose(), get_arm_qpos=lambda: np.zeros(7),
        entity=SimpleNamespace(set_qpos=lambda *_: pytest.fail("simulator qpos mutated")),
    )
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: arm))
    plans = grasp_select._GraspSelectionMixin._plan_one_grasp(
        task, candidate(), Pose([0, 0, 1]), "right",
    )
    assert plans is not None and len(calls) == 1


@pytest.mark.parametrize("offset", [0.031, np.nan])
def test_plan_one_grasp_preserves_position_gate_and_rejects_nonfinite(offset):
    local = Pose([-0.1 + offset, 0.5, 0.3], [0, 1, 0, 0])
    planner, _ = planner_with_fk(local)
    arm = SimpleNamespace(planner=planner, tcp_offset=Pose(), get_arm_qpos=lambda: np.zeros(7))
    task = SimpleNamespace(robot=SimpleNamespace(get_arm=lambda _: arm))
    assert grasp_select._GraspSelectionMixin._plan_one_grasp(
        task, candidate(), Pose([0, 0, 1]), "right",
    ) is None
