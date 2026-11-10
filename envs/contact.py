"""Owned, read-only contact snapshots for one non-batched Genesis scene.

The entity reader prefers the public getter. Genesis 1.2.0's getter can pass an
int32 permutation to torch.gather; only that exact error enables a copied
reader of the pinned contact layout. Passive observers can use the copied
single-scene reader directly to avoid the public getter's cache. Neither path
detects contacts again or changes simulation state. Unsupported or incomplete
data raises, not an empty contact result, so callers can fail closed.
"""

import numpy as np


_IDENTIFIERS = ("geom_a", "geom_b", "link_a", "link_b")
_VECTORS = ("position", "normal", "force_a", "force_b")
_FIELDS = (*_IDENTIFIERS, "penetration", *_VECTORS)
_GATHER_DTYPE_ERROR = "gather(): Expected dtype int64 for index"


def _copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _integers(value, name):
    copied = _copy(value)
    if copied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    if copied.size and np.any(copied > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64 range")
    return copied.astype(np.int64, copy=True)


def _nonnegative_integer(value, name):
    copied = _integers(value, name)
    if copied.shape != () or copied < 0:
        raise ValueError(f"{name} must be a nonnegative scalar integer")
    return int(copied)


def _validated_contacts(raw):
    if not isinstance(raw, dict) or "valid_mask" in raw:
        raise ValueError("expected a non-batched contact dictionary")
    if any(key not in raw for key in _FIELDS):
        raise ValueError("incomplete contact fields")
    result = {key: (_integers(raw[key], key) if key in _IDENTIFIERS
                    else _copy(raw[key])) for key in _FIELDS}
    if result["geom_a"].ndim != 1:
        raise ValueError("expected one-dimensional contact identifiers")
    count = len(result["geom_a"])
    for key, value in result.items():
        shape = (count, 3) if key in _VECTORS else (count,)
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"invalid contact field shape or dtype: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite live contact field: {key}")
        if key in _IDENTIFIERS and np.any(value < 0):
            raise ValueError(f"negative live contact identifier: {key}")
    return result


def _read_quadrants(value, *, transpose=False, copy=True):
    import genesis as gs
    from genesis.utils.misc import qd_to_torch

    if gs.__version__ != "1.2.0":
        raise ValueError("copied contact fallback requires Genesis 1.2.0")
    return qd_to_torch(value, transpose=transpose, copy=copy)


def copied_single_scene_contacts(collider, *, field_reader=None):
    """Copy last-step contacts without a public getter or cache mutation.

    Apply the pinned collider's logical permutation to owned arrays, retaining
    global identifiers and native floating dtypes. The injected field reader is
    a test seam; production reads always copy the underlying Quadrants fields.
    Only live rows are validated: unused native storage can contain sentinels.
    """
    if _nonnegative_integer(collider._solver.n_envs, "n_envs") != 0:
        raise ValueError("batched contact snapshots are unsupported")
    read = _read_quadrants if field_reader is None else field_reader
    state = collider._collider_state
    counts = _integers(read(state.n_contacts, copy=True), "n_contacts")
    if counts.shape != (1,) or counts[0] < 0:
        raise ValueError("invalid single-scene contact count")
    count = int(counts[0])
    order = _integers(read(state.contact_sort_idx, transpose=True, copy=True),
                      "contact_sort_idx")
    if order.ndim != 2 or order.shape[0] != 1 or count > order.shape[1]:
        raise ValueError("invalid single-scene contact permutation shape/count")
    indices, capacity = order[0, :count].copy(), order.shape[1]
    if (np.any(indices < 0) or np.any(indices >= capacity)
            or np.unique(indices).size != count):
        raise ValueError("invalid live contact permutation")
    selected = {}
    for key in (*_IDENTIFIERS, "penetration", "position", "normal", "force"):
        if key not in collider._contact_data:
            raise ValueError(f"incomplete native contact data: missing {key}")
        value = collider._contact_data[key]
        data = _integers(value, key) if key in _IDENTIFIERS else _copy(value)
        shape = (1, capacity, 3) if key in ("position", "normal", "force") else (1, capacity)
        if data.shape != shape:
            raise ValueError(f"invalid native contact field shape: {key}")
        # Only live rows are valid: unused storage may contain sentinels/NaNs.
        selected[key] = data[0, indices].copy()
    force = selected.pop("force")
    selected["force_a"], selected["force_b"] = -force, force.copy()
    return _validated_contacts(selected)


def _copied_contacts(collider):
    """Compatibility seam for the entity reader's narrowly gated fallback."""
    return copied_single_scene_contacts(collider)


def entity_contacts(entity):
    """Return owned NumPy contact arrays with global geom/link identifiers.

    Matches ``entity.get_contacts()``'s either-side filtering, including self
    contacts, with ``force_a`` acting on geom A and ``force_b`` on geom B.
    Includes position, normal and penetration from the most recent scene step.
    Batched scenes and missing/malformed data raise. The fallback never modifies
    engine fields, permutation indices, caches, or global zero-copy settings.
    """
    solver = entity._solver
    if _nonnegative_integer(solver.n_envs, "n_envs") != 0:
        raise ValueError("batched contact snapshots are unsupported")
    first = _nonnegative_integer(entity.geom_start, "geom_start")
    end = _nonnegative_integer(entity.geom_end, "geom_end")
    if end < first:
        raise ValueError("invalid entity geom range")
    try:
        raw = entity.get_contacts()
    except RuntimeError as exc:
        if str(exc) != _GATHER_DTYPE_ERROR:
            raise
        collider = solver.collider
        if collider._solver is not solver:
            raise ValueError("collider and entity belong to different solvers") from exc
        raw = _copied_contacts(collider)
    contacts = _validated_contacts(raw)
    mask = (((contacts["geom_a"] >= first) & (contacts["geom_a"] < end))
            | ((contacts["geom_b"] >= first) & (contacts["geom_b"] < end)))
    return {key: value[mask].copy() for key, value in contacts.items()}
