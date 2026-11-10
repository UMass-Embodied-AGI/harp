"""Scene-free contracts for inspect-motion runtime extraction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.avatar.inspect_motion_mixin import (
    INSPECT_POOL_DEFAULT,
    InspectMotionMixin,
    InspectMotionRuntime,
)


def _task(*, pool=None, config=None, avatar=None):
    return SimpleNamespace(
        INSPECT_POOL=pool,
        config=dict(config or {}),
        avatar=avatar,
        AVATAR_HAND_ID=7,
        AVATAR_MOTION_SLOW=2.0,
    )


def test_runtime_is_stateless_and_default_pool_selects_first_without_rng(
    monkeypatch,
    capsys,
):
    runtime = InspectMotionRuntime()
    task = _task(pool=None)

    def unexpected_randint(*_args, **_kwargs):
        raise AssertionError("default inspect selection must not sample RNG")

    monkeypatch.setattr(np.random, "randint", unexpected_randint)
    runtime.resolve(task)

    first = INSPECT_POOL_DEFAULT[0]
    assert not hasattr(runtime, "__dict__")
    assert task.INSPECT_MOTION_NAME == first["name"]
    assert task.INSPECT_ATTACH_FRAME == first["attach"]
    assert task.INSPECT_DETACH_FRAME == first["detach"]
    assert task.AVATAR_HAND_ID == first["hand_id"]
    assert task.INSPECT_ATTACH_OBJECT is True
    assert task.AVATAR_MOTION_SLOW == 2.0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "forced_name,expected_name",
    (("Inspect", "Inspect1"), ("Inspect2", "Inspect2")),
)
def test_forced_selection_supports_legacy_alias_and_direct_name(
    forced_name,
    expected_name,
    capsys,
):
    pool = (
        {"name": "Inspect1", "attach": 34, "detach": 187},
        {"name": "Inspect2", "attach": 31, "detach": 84},
    )
    task = _task(pool=pool, config={"inspect_motion_name": forced_name})

    InspectMotionRuntime().resolve(task)

    assert task.INSPECT_MOTION_NAME == expected_name
    assert capsys.readouterr().out == (
        f"[inspect] forced motion={expected_name} (config override)\n"
    )


def test_forced_selection_rejects_name_outside_pool():
    pool = ({"name": "Inspect2", "attach": 31, "detach": 84},)
    task = _task(pool=pool, config={"inspect_motion_name": "Inspect"})

    with pytest.raises(
        ValueError,
        match=r"inspect_motion_name.*Inspect1.*not in pool.*Inspect2",
    ):
        InspectMotionRuntime().resolve(task)


def test_random_selection_uses_legacy_global_numpy_draw(monkeypatch, capsys):
    pool = (
        {"name": "InspectA", "attach": 1, "detach": 2},
        {"name": "InspectB", "attach": 3, "detach": 4},
        {"name": "InspectC", "attach": 5, "detach": 6},
    )
    task = _task(pool=pool, config={"randomize_inspect": True})
    calls = []

    def legacy_randint(upper):
        calls.append(upper)
        return 1

    monkeypatch.setattr(np.random, "randint", legacy_randint)
    InspectMotionRuntime().resolve(task)

    assert calls == [len(pool)]
    assert task.INSPECT_MOTION_NAME == "InspectB"
    assert task.INSPECT_ATTACH_FRAME == 3
    assert task.INSPECT_DETACH_FRAME == 4
    assert capsys.readouterr().out == "[inspect] randomized motion=InspectB\n"


def test_entry_metadata_preserves_casts_and_optional_overrides():
    entry = {
        "name": "InspectCustom",
        "attach": "11",
        "detach": "29",
        "hand_id": "4",
        "slow": "3.5",
        "attach_object": 0,
    }
    task = _task(pool=(entry,))

    InspectMotionRuntime().resolve(task)

    assert task.INSPECT_MOTION_NAME == "InspectCustom"
    assert task.INSPECT_ATTACH_FRAME == 11
    assert task.INSPECT_DETACH_FRAME == 29
    assert task.AVATAR_HAND_ID == 4
    assert task.AVATAR_MOTION_SLOW == 3.5
    assert task.INSPECT_ATTACH_OBJECT is False
    assert entry == {
        "name": "InspectCustom",
        "attach": "11",
        "detach": "29",
        "hand_id": "4",
        "slow": "3.5",
        "attach_object": 0,
    }


def _motion_data(length: int) -> dict[str, np.ndarray]:
    return {
        "trans": np.arange(length * 3, dtype=np.float64).reshape(length, 3),
        "rot": np.arange(length * 4, dtype=np.float64).reshape(length, 4),
        "joint": np.arange(length * 2, dtype=np.float64).reshape(length, 2),
        "mat": np.arange(length * 4, dtype=np.float64).reshape(length, 2, 2),
    }


def test_clip_conversion_installs_independent_clip_and_publishes_local_frames():
    source = _motion_data(12)
    avatar = SimpleNamespace(motion_data={"InspectClip": source})
    entry = {
        "name": "InspectClip",
        "attach": 4,
        "detach": 8,
        "clip": (2, 9),
    }
    task = _task(pool=(entry,), avatar=avatar)

    InspectMotionRuntime().resolve(task)

    clip_name = "InspectClip_clip_2_9"
    assert task.INSPECT_MOTION_NAME == clip_name
    assert task.INSPECT_ATTACH_FRAME == 2
    assert task.INSPECT_DETACH_FRAME == 6
    assert entry["name"] == "InspectClip"
    assert entry["attach"] == 4
    assert entry["detach"] == 8
    for key, value in source.items():
        clipped = avatar.motion_data[clip_name][key]
        np.testing.assert_array_equal(clipped, value[2:10])
        assert not np.shares_memory(clipped, value)


def test_clip_is_left_in_source_units_when_avatar_or_motion_is_missing():
    entry = {
        "name": "InspectMissing",
        "attach": 7,
        "detach": 9,
        "clip": (5, 10),
    }
    runtime = InspectMotionRuntime()

    no_avatar = _task(pool=(entry,), avatar=None)
    runtime.resolve(no_avatar)
    assert no_avatar.INSPECT_MOTION_NAME == "InspectMissing"
    assert no_avatar.INSPECT_ATTACH_FRAME == 7
    assert no_avatar.INSPECT_DETACH_FRAME == 9

    missing_motion = _task(
        pool=(entry,),
        avatar=SimpleNamespace(motion_data={}),
    )
    runtime.resolve(missing_motion)
    assert missing_motion.INSPECT_MOTION_NAME == "InspectMissing"
    assert missing_motion.INSPECT_ATTACH_FRAME == 7
    assert missing_motion.INSPECT_DETACH_FRAME == 9


class _RecordingAvatar:
    def __init__(self, motion_data=None):
        self.motion_data = dict(motion_data or {})
        self.robot = SimpleNamespace(base_rot=np.eye(3, dtype=np.float64))
        self.reset_calls = []
        self.play_calls = []
        self.play_result = object()

    def reset(self, pos, rot):
        self.reset_calls.append(
            (
                np.asarray(pos, dtype=np.float64).copy(),
                np.asarray(rot, dtype=np.float64).copy(),
            )
        )

    def play_animation(self, name, **kwargs):
        self.play_calls.append((name, kwargs))
        return self.play_result


def test_touch_only_play_clears_attachment_and_restores_motion_root_first():
    avatar = _RecordingAvatar()
    attach_obj = object()
    root = np.array([0.2, -0.1, 0.7], dtype=np.float64)
    rotation = np.arange(9, dtype=np.float64).reshape(3, 3)
    task = SimpleNamespace(
        avatar=avatar,
        avatar_init_rot=rotation,
        INSPECT_MOTION_NAME="InspectTouch",
        INSPECT_ATTACH_OBJECT=False,
        _inspect_motion_root_pos=root,
    )

    result = InspectMotionRuntime().play(
        task,
        attach_obj=attach_obj,
        hand_id=5,
        attach_frame=12,
        detach_frame=34,
        return_to_idle_after=None,
        custom_flag=True,
    )

    assert result is avatar.play_result
    assert len(avatar.reset_calls) == 1
    np.testing.assert_array_equal(avatar.reset_calls[0][0], root)
    np.testing.assert_array_equal(avatar.reset_calls[0][1], rotation)
    assert avatar.play_calls == [
        (
            "InspectTouch",
            {
                "attach_obj": None,
                "hand_id": 5,
                "attach_frame": None,
                "detach_frame": None,
                "return_to_idle_after": None,
                "custom_flag": True,
            },
        )
    ]


class _ResetBase:
    def reset(self, seed=0):
        self.reset_events.append(("base_reset", seed))
        return {"seed": seed}


class _MixedInspectTask(InspectMotionMixin, _ResetBase):
    INSPECT_POOL = (
        {"name": "InspectMixed", "attach": 3, "detach": 5, "slow": 2.5},
    )

    def __init__(self):
        self.config = {}
        self.avatar = None
        self.AVATAR_HAND_ID = 1
        self.AVATAR_MOTION_SLOW = 1.0
        self.reset_events = []


def test_legacy_mixin_reset_and_scaled_properties_delegate_to_runtime():
    task = _MixedInspectTask()

    observation = task.reset(seed=17)

    assert observation == {"seed": 17}
    assert task.reset_events == [("base_reset", 17)]
    assert task.INSPECT_MOTION_NAME == "InspectMixed"
    assert task._inspect_attach_frame_scaled == 8
    assert task._attach_frame_scaled == 8
    assert task._inspect_detach_frame_scaled == 12
    assert task._detach_frame_scaled == 12


def test_legacy_geometry_and_reset_helpers_preserve_relative_root_behavior():
    motion_data = {
        "InspectMixed": {"trans": np.array([[0.4, 0.2, 0.1]])},
        "idle": {"trans": np.array([[0.1, 0.1, 0.1]])},
    }
    avatar = _RecordingAvatar(motion_data)
    task = _MixedInspectTask()
    task.avatar = avatar
    task.INSPECT_MOTION_NAME = "InspectMixed"
    task.avatar_init_pos = np.array([1.0, 2.0, 3.0])
    task.avatar_init_rot = np.eye(3, dtype=np.float64)
    task.AVATAR_EXTRA_Z = 0.25
    task.CALIBRATION_SETTLE_STEPS = 2
    task.avatar_collided = True
    task.avatar_collision_log = ["old"]
    task._dryrun_capture_palm = lambda: np.array([0.2, 0.3, 0.4])
    sim_steps = []
    task.step_sim = lambda: sim_steps.append("step")

    object_pos = np.array([0.7, 0.9, 1.1])
    expected_motion_root = np.array([1.5, 2.6, 3.95])
    expected_idle_root = expected_motion_root + np.array([0.3, 0.1, 0.0])

    np.testing.assert_allclose(
        task.inspect_avatar_pos_for_object(object_pos),
        expected_motion_root,
    )
    assert task.reset_avatar_for_inspect_object(object_pos) is True

    np.testing.assert_allclose(task._inspect_motion_root_pos, expected_motion_root)
    np.testing.assert_allclose(task._inspect_idle_start_pos, expected_idle_root)
    np.testing.assert_allclose(avatar.reset_calls[-1][0], expected_idle_root)
    np.testing.assert_array_equal(avatar.reset_calls[-1][1], np.eye(3))
    assert sim_steps == ["step", "step"]
    assert task.avatar_collided is False
    assert task.avatar_collision_log == []
