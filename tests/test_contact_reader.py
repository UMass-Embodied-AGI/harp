"""Contact snapshots preserve pinned signs/order without engine mutations."""

import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "envs/contact.py"
SPEC = importlib.util.spec_from_file_location("contact_reader_test", PATH)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


@pytest.fixture
def native(monkeypatch):
    """Global IDs, A/B/self pairs, a nonidentity order and dead native storage."""
    solver = SimpleNamespace(n_envs=0)
    state = SimpleNamespace(n_contacts=np.array([4], dtype=np.int32),
                            contact_sort_idx=np.array([[2, 0, 3, 4, 1]], dtype=np.int32))
    fields = {
        "geom_a": np.array([[20, -1, 12, 12, 30]], dtype=np.int32),
        "geom_b": np.array([[13, -1, 20, 13, 31]], dtype=np.int32),
        "link_a": np.array([[1, -1, 0, 0, 3]], dtype=np.int32),
        "link_b": np.array([[2, -1, 1, 2, 4]], dtype=np.int32),
        "penetration": np.array([[.001, np.nan, .003, .004, .005]]),
    }
    for key in ("position", "normal", "force"):
        fields[key] = np.arange(15, dtype=float).reshape(1, 5, 3)
        fields[key][0, 1] = np.nan
    collider = SimpleNamespace(_solver=solver, _collider_state=state, _contact_data=fields,
                               _contact_data_cache=MappingProxyType({"sentinel": object()}))
    solver.collider = collider
    calls = []

    def get_contacts():
        calls.append("public")
        raise RuntimeError(reader._GATHER_DTYPE_ERROR)

    def read(value, *, transpose=False, copy=True):
        calls.append((value is state.contact_sort_idx, transpose, copy))
        return value

    monkeypatch.setattr(reader, "_read_quadrants", read)
    entity = SimpleNamespace(_solver=solver, geom_start=10, geom_end=20,
                             get_contacts=get_contacts)
    return SimpleNamespace(entity=entity, solver=solver, collider=collider, state=state,
                           fields=fields, calls=calls)


def test_fallback_filters_global_either_side_ids_preserves_order_and_force_sign(native):
    result = reader.entity_contacts(native.entity)
    assert native.calls == ["public", (False, False, True), (True, True, True)]
    assert result["geom_a"].tolist() == [12, 20, 12]
    assert result["geom_b"].tolist() == [20, 13, 13]
    assert result["geom_a"].dtype == np.int64
    assert result["link_a"].tolist() == [0, 1, 0]
    assert result["penetration"].tolist() == [.003, .001, .004]
    np.testing.assert_array_equal(result["normal"], native.fields["normal"][0, [2, 0, 3]])
    np.testing.assert_array_equal(result["force_a"], -native.fields["force"][0, [2, 0, 3]])
    np.testing.assert_array_equal(result["force_b"], native.fields["force"][0, [2, 0, 3]])


def test_fallback_returns_owned_arrays_and_never_changes_native_fields_or_cache(native):
    buffers = [*native.fields.values(), native.state.n_contacts, native.state.contact_sort_idx]
    before = [value.copy() for value in buffers]
    cache = native.collider._contact_data_cache
    for value in buffers:
        value.flags.writeable = False
    result = reader.entity_contacts(native.entity)
    for value in result.values():
        assert value.flags.owndata
        value[...] = 0
    for value, expected in zip(buffers, before):
        np.testing.assert_array_equal(value, expected)
    assert native.state.contact_sort_idx.dtype == np.int32
    assert native.collider._contact_data_cache is cache


def test_passive_scene_reader_never_calls_public_getters_or_filters_contacts(native):
    def forbidden():
        pytest.fail("passive snapshot called the public contact getter")

    native.entity.get_contacts = native.collider.get_contacts = forbidden
    result = reader.copied_single_scene_contacts(native.collider)
    assert native.calls == [(False, False, True), (True, True, True)]
    assert result["geom_a"].tolist() == [12, 20, 12, 30]
    assert result["geom_b"].tolist() == [20, 13, 13, 31]
    for key, value in result.items():
        assert value.flags.owndata
        assert not any(np.shares_memory(value, source) for source in native.fields.values())


@pytest.mark.parametrize("n_envs", [-1, 1, 0., np.array([0])])
def test_passive_scene_reader_rejects_invalid_scene_metadata_before_native_reads(native, n_envs):
    native.solver.n_envs = n_envs
    with pytest.raises(ValueError):
        reader.copied_single_scene_contacts(native.collider)
    assert native.calls == []


def test_public_getter_preferred_and_returns_owned_arrays_without_fallback(native, monkeypatch):
    expected = reader.entity_contacts(native.entity)
    before = {key: value.copy() for key, value in expected.items()}
    native.entity.get_contacts = lambda: expected

    def forbidden(*args, **kwargs):
        pytest.fail("working public getter must not trigger native reads")

    monkeypatch.setattr(reader, "_copied_contacts", forbidden)
    del native.solver.collider
    result = reader.entity_contacts(native.entity)
    for key, value in result.items():
        np.testing.assert_array_equal(value, before[key])
        assert value.flags.owndata
        value[...] = 0
        np.testing.assert_array_equal(expected[key], before[key])


@pytest.mark.parametrize("error", [
    RuntimeError("CUDA device lost"), RuntimeError("gather(): index out of bounds"),
    RuntimeError(reader._GATHER_DTYPE_ERROR + " unexpected suffix"),
    ValueError(reader._GATHER_DTYPE_ERROR), AttributeError("no contact data"),
])
def test_only_exact_runtime_dtype_error_can_trigger_fallback(native, error):
    def fail():
        raise error

    native.entity.get_contacts = fail
    with pytest.raises(type(error)) as caught:
        reader.entity_contacts(native.entity)
    assert caught.value is error
    assert native.calls == []


@pytest.mark.parametrize("failure", [
    "negative_count", "count_shape", "float_count", "count_exceeds_capacity",
    "sort_shape", "float_sort", "sort_negative", "sort_oob", "sort_duplicate",
    "sort_overflow", "missing_normal", "missing_penetration", "missing_force",
    "field_shape", "float_identifier", "negative_identifier", "nonfinite_normal",
    "nonfinite_penetration", "batched", "solver_mismatch",
])
def test_malformed_native_data_raises_instead_of_reporting_no_contacts(native, failure):
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
    elif failure == "solver_mismatch":
        native.collider._solver = SimpleNamespace(n_envs=0)
    else:
        native.solver.n_envs = 1
    with pytest.raises(ValueError):
        reader.entity_contacts(native.entity)


@pytest.mark.parametrize("failure", ["missing", "mask", "shape", "nonfinite", "float_id", "not_dict"])
def test_malformed_public_data_does_not_trigger_fallback(native, failure):
    raw = reader.entity_contacts(native.entity)
    native.calls.clear()
    if failure == "missing":
        del raw["force_a"]
    elif failure == "mask":
        raw["valid_mask"] = np.ones(3, dtype=bool)
    elif failure == "shape":
        raw["position"] = np.zeros((3, 2))
    elif failure == "nonfinite":
        raw["force_a"][0, 0] = np.inf
    elif failure == "float_id":
        raw["geom_a"] = raw["geom_a"].astype(float)
    else:
        raw = []
    native.entity.get_contacts = lambda: raw
    with pytest.raises(ValueError):
        reader.entity_contacts(native.entity)
    assert native.calls == []


@pytest.mark.parametrize("empty_scene", [True, False])
def test_empty_result_has_valid_shapes_only_when_data_proves_no_contacts(native, empty_scene):
    if empty_scene:
        native.state.n_contacts[0] = 0
    else:
        native.entity.geom_start, native.entity.geom_end = 50, 51
    result = reader.entity_contacts(native.entity)
    for key, value in result.items():
        assert value.shape == ((0, 3) if key in reader._VECTORS else (0,))


def test_native_read_failure_is_not_hidden(native, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cannot copy native field")

    monkeypatch.setattr(reader, "_read_quadrants", fail)
    with pytest.raises(RuntimeError, match="cannot copy native field"):
        reader.entity_contacts(native.entity)


@pytest.mark.parametrize(("name", "value"), [
    ("n_envs", -1), ("n_envs", 0.), ("n_envs", np.array([0])),
    ("geom_start", -1), ("geom_start", 10.), ("geom_end", 9),
])
def test_invalid_entity_metadata_fails_before_contact_access(native, name, value):
    target = native.solver if name == "n_envs" else native.entity
    setattr(target, name, value)
    with pytest.raises(ValueError):
        reader.entity_contacts(native.entity)
    assert native.calls == []


def test_private_layout_is_version_pinned(monkeypatch):
    fake_misc = SimpleNamespace(qd_to_torch=lambda *args, **kwargs: pytest.fail("unsupported version"))
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace(__version__="1.3.0"))
    monkeypatch.setitem(sys.modules, "genesis.utils.misc", fake_misc)
    with pytest.raises(ValueError, match="requires Genesis 1.2.0"):
        reader._read_quadrants(object())


def test_quadrants_reads_explicitly_copy(monkeypatch):
    calls = []

    def read(value, **kwargs):
        calls.append((value, kwargs))
        return value

    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace(__version__="1.2.0"))
    monkeypatch.setitem(sys.modules, "genesis.utils.misc", SimpleNamespace(qd_to_torch=read))
    value = object()
    assert reader._read_quadrants(value, transpose=True) is value
    assert calls == [(value, {"transpose": True, "copy": True})]
