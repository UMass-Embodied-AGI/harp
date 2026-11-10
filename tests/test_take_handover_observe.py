"""The take checkpoints must copy observations without affecting execution."""

import importlib.util
import ast
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "envs/task_bases/take_handover_observe.py"
SPEC = importlib.util.spec_from_file_location("take_handover_observe_test", PATH)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


@pytest.fixture
def runtime():
    calls = []
    bounds = np.array([[-.03, -.03, .95], [.03, .03, 1.01]])
    qpos, cached = np.r_[np.zeros(7), .018, .019], np.r_[np.zeros(7), .014, .014]
    contacts = dict(geom_a=np.array([12]), geom_b=np.array([20]),
                    link_a=np.array([0]), link_b=np.array([1]),
                    penetration=np.array([.002]), position=np.array([[0., 0., 1.]]),
                    normal=np.array([[0., 1., 0.]]), force_a=np.array([[0., -3., 0.]]),
                    force_b=np.array([[0., 3., 0.]]))

    def forbidden(*args, **kwargs):
        pytest.fail("observer attempted simulation, control, rendering, planning, or collision detection")

    def get_contacts(label):
        calls.append(label)
        return contacts

    robot = SimpleNamespace(idx=1, geom_start=10, geom_end=20,
                            get_qpos=lambda: qpos, get_AABB=lambda: bounds,
                            get_contacts=lambda: get_contacts("robot"),
                            set_qpos=forbidden, control_dofs_position=forbidden,
                            detect_collision=forbidden)
    obj = SimpleNamespace(idx=2, geom_start=20, geom_end=21,
                          get_pos=lambda: np.array([0., 0., .98]),
                          get_quat=lambda: np.array([1., 0., 0., 0.]),
                          get_AABB=lambda: bounds, get_contacts=lambda: get_contacts("object"),
                          set_pos=forbidden, set_quat=forbidden, detect_collision=forbidden)
    left = SimpleNamespace(name="left_finger", idx=0, entity=robot, get_AABB=lambda: bounds)
    payload = SimpleNamespace(name="cube", idx=1, entity=obj)
    right = SimpleNamespace(name="right_finger", idx=2, entity=robot, get_AABB=lambda: bounds)
    solver = SimpleNamespace(n_envs=0, links=[left, payload, right])
    robot._solver = obj._solver = solver
    robot.links = [left, right]
    arm = SimpleNamespace(entity=robot, _cached_target=cached, gripper_val=.35,
                          get_ee_pose=lambda: np.array([0., 0., 1., 1., 0., 0., 0.]),
                          finger_joints=[SimpleNamespace(name=f"finger{i}", qs_idx_local=[i],
                                                         dofs_idx_local=[i]) for i in (7, 8)],
                          set_arm_joints=forbidden, set_gripper=forbidden,
                          planner=SimpleNamespace(solve_ik=forbidden, plan_path=forbidden))
    task = SimpleNamespace(config={"take_handover_diagnostics": True},
                           _interaction_sim_step=42, SIMULATION_DT=.001, _video_frames=[0, 1, 2],
                           apple_actor=SimpleNamespace(entity=obj),
                           robot=SimpleNamespace(get_arm=lambda tag: arm, set_gripper=forbidden),
                           step_sim=forbidden, scene=SimpleNamespace(step=forbidden, render=forbidden),
                           plan_success=True)
    return SimpleNamespace(task=task, arm=arm, obj=obj, contacts=contacts,
                           qpos=qpos, cached=cached, bounds=bounds, calls=calls)


def test_snapshot_is_read_only_and_serializable(runtime, capsys):
    before = dict(runtime.task.__dict__)
    row = observer.observe_handover(runtime.task, "right", "after_lift")
    assert runtime.calls == ["object", "robot"]
    assert runtime.task.__dict__ == before
    assert row["observation_errors"] == {}
    assert row["sim_step"] == 42 and row["sim_time_s"] == .042
    assert row["video_frames"] == 3 and row["last_video_frame"] == 2
    assert row["measured_finger_qpos"] == [.018, .019]
    assert row["commanded_finger_qpos"] == [.014, .014]
    contact = row["object_contacts"]["rows"][0]
    assert contact["geom_a"] == 12 and contact["geom_b"] == 20
    assert contact["link_names"] == ["left_finger", "cube"]
    assert contact["entity_indices"] == [1, 2]
    assert contact["penetration"] == .002 and contact["force_a"] == [0., -3., 0.]
    printed = capsys.readouterr().out
    assert printed.startswith("[take-handover-state] ")
    assert json.loads(printed.split(" ", 1)[1]) == row
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("config", [{}, {"take_handover_diagnostics": False},
                                    {"take_handover_diagnostics": "true"}, None])
def test_disabled_does_not_touch_robot_or_object(config, capsys):
    class Task:
        def __getattr__(self, name):
            raise AssertionError(f"disabled observer read {name}")
    task = Task()
    task.config = config
    assert observer.observe_handover(task, "right", "ignored") is None
    assert capsys.readouterr().out == ""


def test_no_aliases_survive_snapshot(runtime):
    row = observer.observe_handover(runtime.task, "right", "test")
    row["full_qpos"][7] = 99
    row["cached_qpos"][7] = 99
    row["object_aabb"][0][0] = 99
    row["object_contacts"]["rows"][0]["position"][0] = 99
    assert runtime.qpos[7] == .018 and runtime.cached[7] == .014
    assert runtime.bounds[0, 0] == -.03 and runtime.contacts["position"][0, 0] == 0
    runtime.contacts["force_a"][:] = 77
    assert row["robot_contacts"]["rows"][0]["force_a"] == [0., -3., 0.]


@pytest.mark.parametrize("field,value", [
    ("penetration", np.array([np.nan])), ("position", np.zeros((1, 2))),
    ("geom_a", np.array([12.5])), ("link_b", np.array([-1])),
    ("geom_b", np.array([np.inf])), ("normal", np.zeros((1, 1, 3))),
])
def test_invalid_contacts_report_errors_without_affecting_state(runtime, field, value):
    runtime.contacts[field] = value
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "object_contacts" in row["observation_errors"]
    assert "robot_contacts" in row["observation_errors"]
    assert row["gripper_command"] == .35 and runtime.task.plan_success is True
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("failure", ["missing_contact_key", "getter", "batched", "mask", "link_identity"])
def test_contact_schema_and_getter_failures_are_separate(runtime, failure):
    if failure == "missing_contact_key":
        del runtime.contacts["penetration"]
    elif failure == "getter":
        def fail():
            raise RuntimeError("contact getter failed")
        runtime.obj.get_contacts = fail
    elif failure == "batched":
        runtime.obj._solver.n_envs = 1
    elif failure == "mask":
        runtime.contacts["valid_mask"] = np.array([True])
    else:
        runtime.obj._solver.links[1].idx = 3
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "object_contacts" in row["observation_errors"]
    assert row["object_pose7"][2] == .98 and row["full_qpos"][7] == .018
    json.dumps(row, allow_nan=False)


def test_empty_contacts_are_valid(runtime):
    runtime.contacts = {key: value[:0].copy() for key, value in runtime.contacts.items()}
    runtime.obj.get_contacts = lambda: runtime.contacts
    row = observer.observe_handover(runtime.task, "right", "test")
    assert row["object_contacts"]["count"] == 0
    assert row["object_contacts"]["rows"] == []


@pytest.mark.parametrize("failure", ["qpos", "cached", "quat", "tcp", "bounds", "command", "dt", "step", "finger"])
def test_invalid_telemetry_cannot_emit_nonfinite_json(runtime, failure):
    expected = {"qpos": "full_qpos", "cached": "cached_qpos", "quat": "object_pose7",
                "tcp": "tcp_pose7", "bounds": "object_aabb", "command": "gripper_command",
                "dt": "simulation_dt_s", "step": "sim_step", "finger": "finger_joints"}[failure]
    if failure == "qpos":
        runtime.qpos[0] = np.nan
    elif failure == "cached":
        runtime.cached[7] = np.inf
    elif failure == "quat":
        runtime.obj.get_quat = lambda: np.zeros(4)
    elif failure == "tcp":
        runtime.arm.get_ee_pose = lambda: np.zeros(6)
    elif failure == "bounds":
        runtime.bounds[1] = runtime.bounds[0]
    elif failure == "command":
        runtime.arm.gripper_val = float("nan")
    elif failure == "dt":
        runtime.task.SIMULATION_DT = float("inf")
    elif failure == "step":
        runtime.task._interaction_sim_step = -1
    else:
        runtime.arm.finger_joints[0].qs_idx_local = [30]
    row = observer.observe_handover(runtime.task, "right", "test")
    assert expected in row["observation_errors"]
    assert runtime.task.plan_success is True
    json.dumps(row, allow_nan=False)


def test_missing_arm_does_not_hide_object_and_missing_object_does_not_hide_arm(runtime):
    runtime.task.apple_actor = None
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "object" in row["observation_errors"] and "robot_contacts" in row
    runtime.task.apple_actor = SimpleNamespace(entity=runtime.obj)
    runtime.task.robot = None
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "arm" in row["observation_errors"] and "object_contacts" in row


def test_missing_cache_and_video_are_explicit(runtime):
    runtime.arm._cached_target = None
    runtime.task._video_frames = []
    row = observer.observe_handover(runtime.task, "right", "test")
    assert row["cached_qpos"] is None and "commanded_finger_qpos" not in row
    assert row["last_video_frame"] is None


def test_output_failure_does_not_change_control_flow(runtime, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("closed output stream")
    monkeypatch.setattr("builtins.print", fail)
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "emit" in row["observation_errors"] and runtime.task.plan_success is True


@pytest.mark.parametrize("field,value,error", [
    ("SIMULATION_DT", 0., "simulation_dt_s"),
    ("SIMULATION_DT", -.001, "simulation_dt_s"),
    ("gripper_val", -.1, "gripper_command"),
    ("gripper_val", 1.1, "gripper_command"),
])
def test_out_of_range_scalar_observations(runtime, field, value, error):
    setattr(runtime.arm if field == "gripper_val" else runtime.task, field, value)
    row = observer.observe_handover(runtime.task, "right", "test")
    assert error in row["observation_errors"]
    json.dumps(row, allow_nan=False)


def test_observer_has_no_mutating_or_planning_calls():
    tree = ast.parse(PATH.read_text())
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    forbidden = {"step", "step_sim", "render", "detect_collision", "detection",
                 "set_qpos", "set_pos", "set_quat", "control_dofs_position",
                 "set_arm_joints", "set_gripper", "solve_ik", "plan_path",
                 "compute_forward_kinematics", "seed", "random", "uniform"}
    assert not calls & forbidden


def test_does_not_change_numpy_random_state(runtime):
    before = np.random.get_state()
    observer.observe_handover(runtime.task, "right", "test")
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.fixture
def native_buffers(runtime, monkeypatch):
    """Nontrivial int32 order, both entity sides, self and unrelated contacts."""
    state = SimpleNamespace(n_contacts=np.array([4], dtype=np.int32),
                            contact_sort_idx=np.array([[2, 0, 3, 4, 1]], dtype=np.int32))
    fields = {
        "geom_a": np.array([[20, -1, 12, 12, 30]], dtype=np.int32),
        "geom_b": np.array([[13, -1, 20, 13, 31]], dtype=np.int32),
        "link_a": np.array([[1, -1, 0, 0, 3]], dtype=np.int32),
        "link_b": np.array([[2, -1, 1, 2, 4]], dtype=np.int32),
        "penetration": np.array([[.001, np.nan, .003, .004, .005]]),
    }
    for field in ("position", "normal", "force"):
        fields[field] = np.arange(15, dtype=float).reshape(1, 5, 3)
        fields[field][0, 1] = np.nan  # Unused storage is not live telemetry.
    collider = SimpleNamespace(_solver=runtime.obj._solver, _collider_state=state,
                               _contact_data=fields,
                               _contact_data_cache=MappingProxyType({"sentinel": object()}))
    reads = []

    def read(value, *, transpose=False, copy=True):
        reads.append((value is state.contact_sort_idx, transpose, copy))
        return value

    def forbidden():
        pytest.fail("native reader called the public/cache-mutating getter")

    runtime.obj._solver.collider = collider
    runtime.obj.get_contacts = runtime.arm.entity.get_contacts = forbidden
    collider.get_contacts = forbidden
    monkeypatch.setattr(observer, "_read_quadrants", read)
    return SimpleNamespace(collider=collider, state=state, fields=fields, reads=reads)


def test_copied_reader_preserves_permutation_normals_depths_and_native_state(native_buffers):
    native = native_buffers
    before = {key: value.copy() for key, value in native.fields.items()}
    count, order = native.state.n_contacts.copy(), native.state.contact_sort_idx.copy()
    cache = native.collider._contact_data_cache
    for value in (*native.fields.values(), native.state.n_contacts, native.state.contact_sort_idx):
        value.flags.writeable = False
    raw = observer._copied_single_scene_contacts(native.collider)
    assert native.reads == [(False, False, True), (True, True, True)]
    assert raw["geom_a"].tolist() == [12, 20, 12, 30]
    assert raw["geom_a"].dtype == np.int64
    assert raw["penetration"].tolist() == [.003, .001, .004, .005]
    np.testing.assert_array_equal(raw["normal"], before["normal"][0, [2, 0, 3, 4]])
    np.testing.assert_array_equal(raw["force_a"], -before["force"][0, [2, 0, 3, 4]])
    np.testing.assert_array_equal(raw["force_b"], before["force"][0, [2, 0, 3, 4]])
    for value in raw.values():
        value[...] = 0
    for key, value in native.fields.items():
        np.testing.assert_array_equal(value, before[key])
    np.testing.assert_array_equal(native.state.n_contacts, count)
    np.testing.assert_array_equal(native.state.contact_sort_idx, order)
    assert native.state.contact_sort_idx.dtype == np.int32
    assert native.collider._contact_data_cache is cache


def test_native_observer_filters_either_side_and_keeps_self_pairs(runtime, native_buffers):
    row = observer.observe_handover(runtime.task, "right", "after_close")
    assert row["observation_errors"] == {}
    assert row["object_contacts"]["count"] == 2
    assert row["robot_contacts"]["count"] == 3
    contacts = row["object_contacts"]["rows"]
    assert [contact["entity_indices"] for contact in contacts] == [[1, 2], [2, 1]]
    assert [contact["penetration"] for contact in contacts] == [.003, .001]
    assert row["robot_contacts"]["rows"][-1]["entity_indices"] == [1, 1]
    assert row["object_contacts"]["source"] == "copied_int64_contact_permutation; last scene.step"
    assert runtime.calls == [] and runtime.task.plan_success is True
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_shared_reader_matches_observer_schema_without_shared_storage(native_buffers, float_dtype):
    native = native_buffers
    for key in ("penetration", "position", "normal", "force"):
        native.fields[key] = native.fields[key].astype(float_dtype)
    shared = observer.copied_single_scene_contacts(
        native.collider, field_reader=observer._read_quadrants)
    telemetry = observer._copied_single_scene_contacts(native.collider)
    assert shared.keys() == telemetry.keys()
    for key, value in shared.items():
        np.testing.assert_array_equal(value, telemetry[key])
        integer = key in ("geom_a", "geom_b", "link_a", "link_b")
        assert value.dtype == (np.int64 if integer else float_dtype)
        assert telemetry[key].dtype == (np.int64 if integer else np.float64)
        assert value.flags.owndata and telemetry[key].flags.owndata
        assert not np.shares_memory(value, telemetry[key])
        assert not any(np.shares_memory(value, field) for field in native.fields.values())
        assert not any(np.shares_memory(telemetry[key], field) for field in native.fields.values())
    shared["force_a"][:] = 999
    assert not np.any(telemetry["force_a"] == 999)
    assert not np.any(native.fields["force"] == 999)


@pytest.mark.parametrize("failure", [
    "negative_count", "count_shape", "float_count", "count_exceeds_capacity",
    "sort_shape", "float_sort", "sort_negative", "sort_oob", "sort_duplicate",
    "sort_overflow", "missing_normal", "missing_penetration", "missing_force",
    "field_shape", "float_identifier", "negative_identifier", "nonfinite_normal",
    "nonfinite_penetration", "batched",
])
def test_native_schema_failures_are_explicit_incomplete(runtime, native_buffers, failure):
    native = native_buffers
    if failure == "negative_count":
        native.state.n_contacts[0] = -1
    elif failure == "count_shape":
        native.state.n_contacts = np.array([[4]], dtype=np.int32)
    elif failure == "float_count":
        native.state.n_contacts = np.array([4.])
    elif failure == "count_exceeds_capacity":
        native.state.n_contacts[0] = 6
    elif failure == "sort_shape":
        native.state.contact_sort_idx = np.array([2, 0, 3, 4, 1], dtype=np.int32)
    elif failure == "float_sort":
        native.state.contact_sort_idx = native.state.contact_sort_idx.astype(float)
    elif failure == "sort_negative":
        native.state.contact_sort_idx[0, 0] = -1
    elif failure == "sort_oob":
        native.state.contact_sort_idx[0, 0] = 5
    elif failure == "sort_duplicate":
        native.state.contact_sort_idx[0, 0] = 0
    elif failure == "sort_overflow":
        native.state.contact_sort_idx = np.full((1, 5), 2**63, dtype=np.uint64)
    elif failure.startswith("missing_"):
        del native.fields[failure.removeprefix("missing_")]
    elif failure == "field_shape":
        native.fields["normal"] = np.zeros((5, 3))
    elif failure == "float_identifier":
        native.fields["geom_a"] = native.fields["geom_a"].astype(float)
    elif failure == "negative_identifier":
        native.fields["geom_a"][0, 2] = -1
    elif failure == "nonfinite_normal":
        native.fields["normal"][0, 2, 0] = np.inf
    elif failure == "nonfinite_penetration":
        native.fields["penetration"][0, 2] = np.nan
    else:
        native.collider._solver.n_envs = 1
    row = observer.observe_handover(runtime.task, "right", "test")
    for key in ("object_contacts", "robot_contacts"):
        assert key not in row and key in row["observation_errors"]
        if failure.startswith("missing_"):
            assert "incomplete native contact data: missing" in row["observation_errors"][key]
    assert row["gripper_command"] == .35 and runtime.task.plan_success is True
    json.dumps(row, allow_nan=False)


def test_native_zero_contacts_return_correct_empty_shapes(native_buffers):
    native_buffers.state.n_contacts[0] = 0
    raw = observer._copied_single_scene_contacts(native_buffers.collider)
    for key, value in raw.items():
        assert value.shape == ((0, 3) if key in ("normal", "position", "force_a", "force_b") else (0,))


def test_native_field_getter_failure_is_not_hidden(runtime, native_buffers, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cannot copy native field")
    monkeypatch.setattr(observer, "_read_quadrants", fail)
    row = observer.observe_handover(runtime.task, "right", "test")
    assert "cannot copy native field" in row["observation_errors"]["object_contacts"]
    assert "object_contacts" not in row


@pytest.mark.skipif(os.environ.get("TAKE_CONTACT_NATIVE_SMOKE") != "1",
                    reason="opt-in tiny native scene; run only inside GPU srun")
def test_tiny_native_contact_scene_without_state_or_cache_mutation(monkeypatch):
    import genesis as gs

    assert gs.__version__ == "1.2.0"
    gs.init(backend=gs.cpu, logging_level="warning")
    try:
        scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=.001))
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(gs.morphs.Box(size=(.04, .04, .04), pos=(0, 0, .019)))
        scene.build()
        for _ in range(20):
            scene.step()
        collider = box._solver.collider

        def snapshot():
            return {
                "qpos": observer._copy_array(box.get_qpos()),
                "velocity": observer._copy_array(box.get_dofs_velocity()),
                "count": observer._copy_array(observer._read_quadrants(collider._collider_state.n_contacts)),
                "order": observer._copy_array(observer._read_quadrants(
                    collider._collider_state.contact_sort_idx, transpose=True)),
                **{key: observer._copy_array(value) for key, value in collider._contact_data.items()},
            }

        before = snapshot()
        cache = collider._contact_data_cache
        cache_before = {key: dict(value) for key, value in cache.items()}

        def forbidden(*args, **kwargs):
            pytest.fail("native contact observation attempted a public getter, step or collision detection")

        monkeypatch.setattr(box, "get_contacts", forbidden)
        monkeypatch.setattr(collider, "get_contacts", forbidden)
        monkeypatch.setattr(scene, "step", forbidden)
        monkeypatch.setattr(collider, "detection", forbidden)
        shared = observer.copied_single_scene_contacts(collider)
        telemetry = observer._copied_single_scene_contacts(collider)
        # Reconstruct the pre-refactor reader's logical rows independently.
        count = int(before["count"][0])
        indices = before["order"][0, :count].astype(np.int64, copy=True)
        for key, value in shared.items():
            source = "force" if key in ("force_a", "force_b") else key
            expected = before[source][0, indices].copy()
            if key == "force_a":
                expected = -expected
            np.testing.assert_array_equal(value, expected)
            np.testing.assert_array_equal(telemetry[key], expected)
            assert value.flags.owndata and telemetry[key].flags.owndata
            assert not np.shares_memory(value, telemetry[key])
            assert not np.shares_memory(value, before[source])
        row = observer._contacts(box)
        assert row["source"] == "copied_int64_contact_permutation; last scene.step"
        assert row["count"] > 0
        assert all(np.isfinite(contact["penetration"]) for contact in row["rows"])
        assert all(np.isclose(np.linalg.norm(contact["normal"]), 1., atol=1e-4)
                   for contact in row["rows"])
        assert all(contact["geom_a"] in range(box.geom_start, box.geom_end)
                   or contact["geom_b"] in range(box.geom_start, box.geom_end)
                   for contact in row["rows"])
        after = snapshot()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])
        assert collider._contact_data_cache is cache and cache.keys() == cache_before.keys()
        for key, fields in cache_before.items():
            assert cache[key].keys() == fields.keys()
            assert all(cache[key][field] is value for field, value in fields.items())
        json.dumps(row, allow_nan=False)
        print("[take-contact-native-smoke] " + json.dumps({
            "source": row["source"], "count": row["count"],
            "index_dtype": str(before["order"].dtype),
            "penetrations": [contact["penetration"] for contact in row["rows"]],
            "native_state_and_cache_unchanged": True,
            "shared_and_observer_match_pre_refactor_rows": True,
        }, allow_nan=False))
        monkeypatch.setattr(gs, "__version__", "unsupported")
        with pytest.raises(ValueError, match="requires Genesis 1.2.0"):
            observer._contacts(box)
    finally:
        gs.destroy()
