"""Scene-free characterization of neutral specs and motion-clip transforms.

These tests intentionally exercise the historical mixin entry points.  A
later extraction may move the pure data transforms into a smaller helper, as
long as the mixin keeps thin delegates with the same observable contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases import neutral_avatar_table_work as neutral_module
from envs.task_bases import neutral_avatar_spec as spec_module
from envs.task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


def _numbered_motion(num_frames: int) -> dict[str, np.ndarray]:
    """Return arrays whose every scalar and every source frame is unique."""
    return {
        "joint_pose": (
            np.arange(num_frames * 4, dtype=np.int64).reshape(num_frames, 2, 2)
            + 100_000
        ),
        "trans": (
            np.arange(num_frames * 3, dtype=np.float64).reshape(num_frames, 3)
            + 0.125
        ),
    }


def _frame_ids(motion: dict[str, np.ndarray]) -> tuple[int, ...]:
    return tuple(
        ((np.asarray(motion["trans"])[:, 0] - 0.125) / 3.0)
        .astype(np.int64)
        .tolist()
    )


def _motion_digest(motion: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(motion):
        value = np.ascontiguousarray(motion[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _clip_task(
    source_name: str,
    num_frames: int,
    *,
    config: dict | None = None,
    **attributes,
) -> tuple[NeutralAvatarTableWorkMixin, dict[str, np.ndarray]]:
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    source = _numbered_motion(num_frames)
    task.avatar = SimpleNamespace(motion_data={source_name: source})
    task.config = dict(config or {})
    for name, value in attributes.items():
        setattr(task, name, value)
    return task, source


def _assert_motion_unchanged(
    source: dict[str, np.ndarray],
    before: dict[str, np.ndarray],
) -> None:
    assert tuple(source) == tuple(before)
    for name in source:
        np.testing.assert_array_equal(source[name], before[name])


def _assert_independent_arrays(
    source: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
) -> None:
    for name in source:
        assert not np.shares_memory(source[name], derived[name])


def _assert_preserves_global_numpy_rng(callback) -> None:
    old_state = np.random.get_state()
    try:
        np.random.seed(7_214_909)
        expected_tail = np.random.randint(0, 2**31 - 1, size=8)
        np.random.seed(7_214_909)
        callback()
        actual_tail = np.random.randint(0, 2**31 - 1, size=8)
        np.testing.assert_array_equal(actual_tail, expected_tail)
    finally:
        np.random.set_state(old_state)


def _assert_second_install_is_noop(task, method_name: str, job: dict) -> None:
    before_job = deepcopy(job)
    before_motion = {
        name: _motion_digest(motion)
        for name, motion in task.avatar.motion_data.items()
    }
    _assert_preserves_global_numpy_rng(lambda: getattr(task, method_name)(job))
    assert job == before_job
    assert {
        name: _motion_digest(motion)
        for name, motion in task.avatar.motion_data.items()
    } == before_motion


def test_repo_paths_and_motion_constants_are_portable_and_stable(tmp_path: Path) -> None:
    expected_root = Path(neutral_module.__file__).resolve().parents[2]
    assert neutral_module.NeutralTaskSpec is spec_module.NeutralTaskSpec
    assert neutral_module.ROOT_PATH is spec_module.ROOT_PATH
    assert neutral_module._resolve_repo_path is spec_module.resolve_repo_path
    assert tuple(inspect.signature(neutral_module._resolve_repo_path).parameters) == (
        "p",
    )
    assert neutral_module._NEUTRAL_MOTION_PKL is spec_module.NEUTRAL_MOTION_PKL
    assert neutral_module._NEUTRAL_MOTION_META is spec_module.NEUTRAL_MOTION_META
    assert neutral_module.ROOT_PATH == expected_root
    assert neutral_module._NEUTRAL_MOTION_PKL == str(
        expected_root
        / "assets"
        / "neutral_motions"
        / "raw_blender_interaction_clips.pkl"
    )
    assert neutral_module._NEUTRAL_MOTION_META == str(
        expected_root
        / "assets"
        / "neutral_motions"
        / "raw_blender_interaction_clips.json"
    )

    relative = Path("assets/neutral_motions/../neutral_motions/custom.json")
    assert neutral_module._resolve_repo_path(relative) == str(expected_root / relative)

    absolute = tmp_path / "nested" / "motion.pkl"
    assert neutral_module._resolve_repo_path(absolute) == str(absolute)
    assert neutral_module._resolve_repo_path(str(absolute)) == str(absolute)


def test_neutral_task_spec_hydrates_every_legacy_attribute_without_aliasing() -> None:
    spec = NeutralTaskSpec(
        work_xy=(0.11, 0.12),
        mouse_work_xy=(0.21, 0.22),
        sponge_work_xy=(0.31, 0.32),
        table_xy_bounds=(-0.5, 0.5, -0.7, 0.3),
        max_table_edge_gap=0.61,
        min_table_edge_gap=0.17,
        avatar_as_planner_obstacle=False,
        use_region_placement=True,
        blocked_avatar_sides=("left", "back"),
        work_candidates=((0.1, 0.2), (0.3, 0.4)),
        mouse_work_candidates=((0.5, 0.6),),
        sponge_work_candidates=((0.7, 0.8),),
        work_region=(-0.1, 0.2, -0.3, 0.4),
        mouse_work_region=(-0.2, 0.3, -0.4, 0.5),
        sponge_work_region=(-0.3, 0.4, -0.5, 0.6),
        sponge_side_samples=9,
    )

    class HydratedA(NeutralAvatarTableWorkMixin):
        NEUTRAL_TASK_SPEC = spec

    class HydratedB(NeutralAvatarTableWorkMixin):
        NEUTRAL_TASK_SPEC = spec

    assert HydratedA.NEUTRAL_TASK_SPEC is spec
    np.testing.assert_array_equal(HydratedA.NEUTRAL_WORK_XY, [0.11, 0.12])
    np.testing.assert_array_equal(HydratedA.NEUTRAL_MOUSE_WORK_XY, [0.21, 0.22])
    np.testing.assert_array_equal(HydratedA.NEUTRAL_SPONGE_WORK_XY, [0.31, 0.32])
    assert HydratedA.NEUTRAL_WORK_XY.dtype == np.dtype(np.float64)
    assert HydratedA.NEUTRAL_TABLE_XY_BOUNDS == (-0.5, 0.5, -0.7, 0.3)
    assert HydratedA.NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP == 0.61
    assert HydratedA.NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP == 0.17
    assert HydratedA.NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE is False
    assert HydratedA.NEUTRAL_USE_REGION_PLACEMENT is True
    assert HydratedA.NEUTRAL_AVATAR_BLOCKED_SIDES == ("left", "back")
    assert isinstance(HydratedA.NEUTRAL_WORK_CANDIDATES, list)
    assert [xy.tolist() for xy in HydratedA.NEUTRAL_WORK_CANDIDATES] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert [xy.tolist() for xy in HydratedA.NEUTRAL_MOUSE_WORK_CANDIDATES] == [
        [0.5, 0.6]
    ]
    assert [xy.tolist() for xy in HydratedA.NEUTRAL_SPONGE_WORK_CANDIDATES] == [
        [0.7, 0.8]
    ]
    assert HydratedA.NEUTRAL_WORK_REGION == (-0.1, 0.2, -0.3, 0.4)
    assert HydratedA.NEUTRAL_MOUSE_WORK_REGION == (-0.2, 0.3, -0.4, 0.5)
    assert HydratedA.NEUTRAL_SPONGE_WORK_REGION == (-0.3, 0.4, -0.5, 0.6)
    assert HydratedA.NEUTRAL_SPONGE_SIDE_SAMPLES == 9

    assert HydratedA.NEUTRAL_WORK_XY is not HydratedB.NEUTRAL_WORK_XY
    assert (
        HydratedA.NEUTRAL_WORK_CANDIDATES[0]
        is not HydratedB.NEUTRAL_WORK_CANDIDATES[0]
    )
    HydratedA.NEUTRAL_WORK_XY[0] = 99.0
    HydratedA.NEUTRAL_WORK_CANDIDATES[0][0] = 88.0
    assert spec.work_xy == (0.11, 0.12)
    assert spec.work_candidates == ((0.1, 0.2), (0.3, 0.4))

    with pytest.raises(FrozenInstanceError):
        spec.work_xy = (1.0, 2.0)


def test_neutral_task_spec_dict_normalization_none_inheritance_and_validation() -> None:
    class DictHydrated(NeutralAvatarTableWorkMixin):
        NEUTRAL_TASK_SPEC = {"work_xy": (0.41, 0.09)}

    assert isinstance(DictHydrated.NEUTRAL_TASK_SPEC, NeutralTaskSpec)
    assert DictHydrated.NEUTRAL_TASK_SPEC.work_xy == (0.41, 0.09)
    np.testing.assert_array_equal(DictHydrated.NEUTRAL_WORK_XY, [0.41, 0.09])
    assert "NEUTRAL_MOUSE_WORK_XY" not in DictHydrated.__dict__
    assert (
        DictHydrated.NEUTRAL_MOUSE_WORK_XY
        is NeutralAvatarTableWorkMixin.NEUTRAL_MOUSE_WORK_XY
    )

    with pytest.raises(TypeError, match="must be NeutralTaskSpec or dict"):

        class InvalidSpec(NeutralAvatarTableWorkMixin):
            NEUTRAL_TASK_SPEC = "not-a-spec"


def _write_resolver_meta(path: Path) -> dict[str, dict]:
    clips = {
        "alpha": {
            "events": [
                {"object": "mouse", "type": "attach", "clip_frame": 1},
                {"object": "mouse", "type": "detach", "clip_frame": 4},
            ],
            "kind": "mouse",
            "neutral_placement": {"task": {"tag": "alpha"}},
            "num_frames": 5,
        },
        "alpha_variant": {
            "events": [
                {"object": "phone", "type": "attach", "clip_frame": 2},
            ],
            "kind": "phone",
            "num_frames": 6,
        },
        "beta": {
            "events": [
                {"object": "sponge", "type": "attach", "clip_frame": 2},
                {"object": "sponge", "type": "detach", "clip_frame": 7},
            ],
            "kind": "sponge",
            "num_frames": 9,
        },
    }
    path.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return clips


def _resolver_task(meta_path: Path, **config) -> NeutralAvatarTableWorkMixin:
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = {"neutral_avatar_motion_meta": str(meta_path), **config}
    task._neutral_avatar_job = None
    task._neutral_avatar_reset_seed = 23
    return task


def test_forced_resolver_is_deterministic_copies_events_and_does_not_use_rng(
    tmp_path: Path,
) -> None:
    clips = _write_resolver_meta(tmp_path / "clips.json")
    task = _resolver_task(tmp_path / "clips.json", neutral_avatar_motion="alpha")

    _assert_preserves_global_numpy_rng(task._resolve_neutral_avatar_job)

    assert task._neutral_avatar_job == {
        "motion": "alpha",
        "base_motion": "alpha",
        "clip": clips["alpha"],
        "events": clips["alpha"]["events"],
        "objects": ["mouse"],
        "kind": "mouse",
        "placement": {"task": {"tag": "alpha"}},
    }
    assert task._neutral_avatar_job["events"] is not task._neutral_avatar_job["clip"]["events"]
    assert all(
        resolved is not authored
        for resolved, authored in zip(
            task._neutral_avatar_job["events"],
            task._neutral_avatar_job["clip"]["events"],
        )
    )
    task._neutral_avatar_job["events"][0]["clip_frame"] = 99
    assert task._neutral_avatar_job["clip"]["events"][0]["clip_frame"] == 1


def test_exclusion_resolver_uses_private_seeded_generator_without_global_rng(
    tmp_path: Path,
) -> None:
    clips = _write_resolver_meta(tmp_path / "clips.json")
    choices = ["alpha", "beta"]
    task = _resolver_task(
        tmp_path / "clips.json",
        neutral_avatar_motion_choices=choices,
    )
    expected_idx = int(np.random.default_rng(23 + 904_219).integers(len(choices)))

    _assert_preserves_global_numpy_rng(
        lambda: task._resolve_neutral_avatar_job(excluded_target_ids={"not_a_prop"})
    )

    assert task._neutral_avatar_job["motion"] == choices[expected_idx]
    assert task._neutral_avatar_job["clip"] == clips[choices[expected_idx]]


def test_unforced_resolver_consumes_exactly_one_global_randint(tmp_path: Path) -> None:
    _write_resolver_meta(tmp_path / "clips.json")
    choices = ["alpha", "beta"]
    task = _resolver_task(
        tmp_path / "clips.json",
        neutral_avatar_motion_choices=choices,
    )
    old_state = np.random.get_state()
    try:
        np.random.seed(88_031)
        expected_idx = int(np.random.randint(len(choices)))
        expected_tail = np.random.randint(0, 2**31 - 1, size=8)
        np.random.seed(88_031)

        task._resolve_neutral_avatar_job()
        actual_tail = np.random.randint(0, 2**31 - 1, size=8)
    finally:
        np.random.set_state(old_state)

    assert task._neutral_avatar_job["motion"] == choices[expected_idx]
    np.testing.assert_array_equal(actual_tail, expected_tail)


def test_laptop_pingpong_transform_exact_contract() -> None:
    task, source = _clip_task(
        "fax",
        12,
        NEUTRAL_LAPTOP_CLIP_TRIM=(2, 7),
        NEUTRAL_LAPTOP_PINGPONG_LOOPS=3,
    )
    source_before = deepcopy(source)
    source_clip = {"num_frames": 12, "tag": "laptop"}
    job = {"base_motion": "fax", "motion": "fax", "clip": source_clip}
    expected_indices = (
        2,
        3,
        4,
        5,
        6,
        7,
        6,
        5,
        4,
        3,
        2,
        3,
        4,
        5,
        6,
        7,
        6,
        5,
        4,
        3,
        2,
        3,
        4,
        5,
        6,
        7,
    )

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_laptop_pingpong_subclip(job)
    )

    destination = "fax_laptop_pingpong_3"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == expected_indices
    assert _motion_digest(derived) == (
        "1d6fc2ce5cfa45454ec109abdc829944c415f17b22cd12ee813984293afc8517"
    )
    assert job == {
        "base_motion": "fax",
        "motion": destination,
        "clip": {"num_frames": 26, "tag": "laptop"},
        "laptop_pingpong": 3,
    }
    assert job["clip"] is not source_clip
    assert source_clip == {"num_frames": 12, "tag": "laptop"}
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(
        task, "_install_neutral_laptop_pingpong_subclip", job
    )


def test_typing_trim_transform_exact_contract() -> None:
    task, source = _clip_task(
        "Typing",
        12,
        NEUTRAL_TYPING_CLIP_TRIM=(2, 7),
    )
    source_before = deepcopy(source)
    source_clip = {"num_frames": 12, "tag": "typing"}
    job = {"base_motion": "Typing", "motion": "Typing", "clip": source_clip}

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_typing_subclip(job)
    )

    destination = "Typing_typing_trim"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == (2, 3, 4, 5, 6, 7)
    assert _motion_digest(derived) == (
        "2eaf722f864f7151452a0919dc9cb956646af94c666dd6afc9a8469a2090cb52"
    )
    assert job == {
        "base_motion": "Typing",
        "motion": destination,
        "clip": {"num_frames": 6, "tag": "typing"},
        "typing_trim": True,
    }
    assert job["clip"] is not source_clip
    assert source_clip == {"num_frames": 12, "tag": "typing"}
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(task, "_install_neutral_typing_subclip", job)


def test_clap_trim_transform_exact_contract() -> None:
    task, source = _clip_task(
        "clap",
        12,
        NEUTRAL_CLAP_CLIP_TRIM=(2, 7),
    )
    source_before = deepcopy(source)
    source_clip = {"num_frames": 12, "tag": "clap"}
    job = {"base_motion": "clap", "motion": "clap", "clip": source_clip}

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_clap_subclip(job)
    )

    destination = "clap_clap_trim_2_7"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == (2, 3, 4, 5, 6, 7)
    assert _motion_digest(derived) == (
        "2eaf722f864f7151452a0919dc9cb956646af94c666dd6afc9a8469a2090cb52"
    )
    assert job == {
        "base_motion": "clap",
        "motion": destination,
        "clip": {
            "num_frames": 6,
            "tag": "clap",
            "clip_start": 2,
            "clip_end": 7,
        },
        "clap_trim": True,
    }
    assert job["clip"] is not source_clip
    assert source_clip == {"num_frames": 12, "tag": "clap"}
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(task, "_install_neutral_clap_subclip", job)


def test_mouse_trim_and_attached_pingpong_exact_contract() -> None:
    task, source = _clip_task(
        "mouse",
        75,
        NEUTRAL_MOUSE_CLIP_TRIM=(3, 72),
        NEUTRAL_MOUSE_ATTACHED_FORWARD_LOOPS=2,
    )
    source_before = deepcopy(source)
    source_clip = {"num_frames": 75, "tag": "mouse"}
    job = {
        "base_motion": "mouse",
        "motion": "mouse",
        "clip": source_clip,
        "events": [
            {"object": "mouse", "type": "attach", "clip_frame": 60},
            {"object": "mouse", "type": "detach", "clip_frame": 69},
            {"object": "cursor", "type": "mark", "clip_frame": 70},
            {"object": "old", "type": "mark", "clip_frame": 1},
            {"object": "late", "type": "mark", "clip_frame": 72},
        ],
    }
    expected_indices = (
        tuple(range(3, 67))
        + (67, 68, 69)
        + (68, 67, 66)
        + (67, 68, 69)
        + (70, 71)
    )

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_avatar_subclip(job)
    )

    destination = "mouse_neutral_trim_3_72_mouse_pingpong_2"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == expected_indices
    assert _motion_digest(derived) == (
        "424e0feeec549b8244234ecc4f0a9bc8b827b74f0a61fb289b900ae09666aa9d"
    )
    assert job == {
        "base_motion": "mouse",
        "motion": destination,
        "clip": {"num_frames": 75, "tag": "mouse"},
        "events": [
            {"object": "mouse", "type": "attach", "clip_frame": 63},
            {"object": "mouse", "type": "detach", "clip_frame": 72},
            {"object": "cursor", "type": "mark", "clip_frame": 73},
        ],
        "trim_start": 3,
        "mouse_attached_forward_loops": 2,
    }
    assert job["clip"] is not source_clip
    assert source_clip == {"num_frames": 75, "tag": "mouse"}
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(task, "_install_neutral_avatar_subclip", job)


def test_sponge_attached_pingpong_and_event_remap_exact_contract() -> None:
    task, source = _clip_task("sponge", 12)
    source_before = deepcopy(source)
    source_clip = {
        "num_frames": 12,
        "tag": "sponge",
        "neutral_placement": {"task": {"sponge_attached_forward_loops": 3}},
    }
    job = {
        "base_motion": "sponge",
        "motion": "sponge",
        "clip": source_clip,
        "placement": source_clip["neutral_placement"],
        "events": [
            {"object": "before", "type": "mark", "clip_frame": 1},
            {"object": "sponge", "type": "attach", "clip_frame": 2},
            {"object": "sponge", "type": "detach", "clip_frame": 5},
            {"object": "after", "type": "mark", "clip_frame": 8},
        ],
    }
    expected_indices = (
        0,
        1,
        2,
        3,
        4,
        5,
        4,
        3,
        2,
        3,
        4,
        5,
        4,
        3,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    )

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_avatar_sponge_pingpong_subclip(job)
    )

    destination = "sponge_sponge_pingpong_3"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == expected_indices
    assert _motion_digest(derived) == (
        "4fd3f9efb0d13239bf9361b490b6b30034f579492acc1b3d7ceac73735c241e9"
    )
    assert job == {
        "base_motion": "sponge",
        "motion": destination,
        "clip": {
            "num_frames": 24,
            "tag": "sponge",
            "neutral_placement": {
                "task": {"sponge_attached_forward_loops": 3}
            },
        },
        "placement": {"task": {"sponge_attached_forward_loops": 3}},
        "events": [
            {"object": "before", "type": "mark", "clip_frame": 1},
            {"object": "sponge", "type": "attach", "clip_frame": 2},
            {"object": "sponge", "type": "detach", "clip_frame": 17},
            {"object": "after", "type": "mark", "clip_frame": 20},
        ],
        "sponge_pingpong": 3,
    }
    assert job["clip"] is not source_clip
    assert source_clip["num_frames"] == 12
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(
        task, "_install_neutral_avatar_sponge_pingpong_subclip", job
    )


def test_reverse_transform_exact_contract() -> None:
    task, source = _clip_task("put", 7)
    source_before = deepcopy(source)
    source_clip = {
        "reverse": True,
        "source_motion": "put",
        "num_frames": 999,
        "tag": "take",
    }
    source_events = [{"object": "object_1", "type": "pick", "clip_frame": 1}]
    job = {
        "base_motion": "take",
        "motion": "take",
        "clip": source_clip,
        "events": source_events,
    }

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_reverse_subclip(job)
    )

    destination = "put_neutral_reverse"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == (6, 5, 4, 3, 2, 1, 0)
    assert _motion_digest(derived) == (
        "25d94ed3d44069aa17830047b16965be7f39f4713ed06d40ad75ebbb981a9c08"
    )
    assert job == {
        "base_motion": "take",
        "motion": destination,
        "clip": {
            "reverse": True,
            "source_motion": "put",
            "num_frames": 7,
            "tag": "take",
        },
        "events": source_events,
        "reversed_installed": True,
    }
    assert job["clip"] is not source_clip
    assert source_clip["num_frames"] == 999
    assert job["events"] is source_events
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(task, "_install_neutral_reverse_subclip", job)


def test_generic_loop_transform_exact_contract() -> None:
    task, source = _clip_task("mouse_stage", 6)
    source_before = deepcopy(source)
    source_clip = {"num_frames": 6, "tag": "generic"}
    source_events = [{"object": "mouse", "type": "attach", "clip_frame": 2}]
    job = {
        "base_motion": "mouse",
        "motion": "mouse_stage",
        "kind": "mouse",
        "clip": source_clip,
        "events": source_events,
    }

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_loop_subclip(job)
    )

    destination = "mouse_stage_neutral_loop"
    derived = task.avatar.motion_data[destination]
    assert _frame_ids(derived) == (0, 1, 2, 3, 4, 5, 4, 3, 2, 1)
    assert _motion_digest(derived) == (
        "7b700ffd2bb5a703ef8a6ad358045fd42c2c713a6b97e34d3e0f052ad7a84f6d"
    )
    assert job == {
        "base_motion": "mouse",
        "motion": destination,
        "kind": "mouse",
        "clip": {"num_frames": 10, "tag": "generic"},
        "events": source_events,
        "loop": True,
    }
    assert job["clip"] is not source_clip
    assert source_clip == {"num_frames": 6, "tag": "generic"}
    assert job["events"] is source_events
    _assert_motion_unchanged(source, source_before)
    _assert_independent_arrays(source, derived)
    _assert_second_install_is_noop(task, "_install_neutral_loop_subclip", job)


@pytest.mark.parametrize(
    "job,config,num_frames",
    (
        (
            {
                "motion": "put",
                "kind": "put_objects_in_bowl",
                "clip": {"reverse": False},
            },
            {},
            6,
        ),
        (
            {"motion": "short", "kind": "mouse", "clip": {}},
            {},
            2,
        ),
        (
            {"motion": "mouse", "kind": "mouse", "clip": {}},
            {"neutral_avatar_loop_motion": False},
            6,
        ),
    ),
)
def test_generic_loop_guard_paths_are_rng_free_noops(job, config, num_frames) -> None:
    task, source = _clip_task(job["motion"], num_frames, config=config)
    source_before = deepcopy(source)
    job_before = deepcopy(job)

    _assert_preserves_global_numpy_rng(
        lambda: task._install_neutral_loop_subclip(job)
    )

    assert job == job_before
    assert tuple(task.avatar.motion_data) == (job["motion"],)
    _assert_motion_unchanged(source, source_before)
