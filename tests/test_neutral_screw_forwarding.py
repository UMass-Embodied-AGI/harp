"""Neutral obstacle refresh must not change the downstream screw-call contract."""

from __future__ import annotations

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.tasks.registry import TASK_DESCRIPTORS, TaskRegistration, TaskVariant


_CALLS = [
    pytest.param((), {}, id="omitted"),
    pytest.param((0.73,), {}, id="positional-yaw"),
    pytest.param((), {"tcp_yaw": -0.41}, id="keyword-yaw"),
    pytest.param((None,), {}, id="positional-none"),
    pytest.param((), {"tcp_yaw": None}, id="keyword-none"),
]


def _recording_task(*, delegate_error=None, refresh_error=None):
    class Delegate:
        def _move_screw(self, target_pos, arm_tag, *args, **kwargs):
            self.events.append(("delegate", target_pos, arm_tag, args, kwargs))
            if delegate_error is not None:
                raise delegate_error
            return self.result

    class Task(NeutralAvatarTableWorkMixin, Delegate):
        def __init__(self):
            self.events = []
            self.result = object()

        def _refresh_avatar_planner_obstacles(self, arm_tag):
            self.events.append(("refresh", arm_tag))
            if refresh_error is not None:
                raise refresh_error

    return Task()


@pytest.mark.parametrize("args, kwargs", _CALLS)
def test_screw_forwards_exact_call_after_refresh(args, kwargs):
    task = _recording_task()
    target = object()

    result = task._move_screw(target, "left", *args, **kwargs)

    assert result is task.result
    assert task.events == [
        ("refresh", "left"),
        ("delegate", target, "left", args, kwargs),
    ]


def test_screw_does_not_filter_additional_downstream_arguments():
    task = _recording_task()
    target, extra, payload = object(), object(), object()

    result = task._move_screw(
        target, "right", 0.73, extra, payload=payload,
    )

    assert result is task.result
    assert task.events == [
        ("refresh", "right"),
        ("delegate", target, "right", (0.73, extra), {"payload": payload}),
    ]


@pytest.mark.parametrize("default_yaw", [None, 0.0], ids=["none", "zero"])
def test_omitted_yaw_preserves_the_downstream_default(default_yaw):
    class Delegate:
        def _move_screw(self, target_pos, arm_tag, tcp_yaw=default_yaw):
            return tcp_yaw

    class Task(NeutralAvatarTableWorkMixin, Delegate):
        def _refresh_avatar_planner_obstacles(self, arm_tag):
            pass

    # Do not initialize a simulator for a transparent delegation unit test.
    task = Task.__new__(Task)

    result = task._move_screw(object(), "right")

    assert result is default_yaw


def test_delegate_exception_propagates_unchanged_after_refresh():
    error = RuntimeError("downstream planner failure")
    task = _recording_task(delegate_error=error)
    target = object()

    with pytest.raises(RuntimeError) as caught:
        task._move_screw(target, "left", tcp_yaw=None)

    assert caught.value is error
    assert task.events == [
        ("refresh", "left"),
        ("delegate", target, "left", (), {"tcp_yaw": None}),
    ]


def test_refresh_exception_prevents_delegation_and_propagates_unchanged():
    error = RuntimeError("obstacle refresh failure")
    task = _recording_task(refresh_error=error)

    with pytest.raises(RuntimeError) as caught:
        task._move_screw(object(), "right", tcp_yaw=0.73)

    assert caught.value is error
    assert task.events == [("refresh", "right")]


_NEUTRAL_CLASSES = {
    descriptor.name: descriptor.task_cls
    for descriptor in TASK_DESCRIPTORS
    if descriptor.variant is TaskVariant.COPRESENCE
    and descriptor.registration is TaskRegistration.CANONICAL
    and issubclass(descriptor.task_cls, NeutralAvatarTableWorkMixin)
}
_WITHOUT_SCREW_DELEGATE = {
    "place_bread_in_basket_copresence",
    "stack_three_bowls_copresence",
}
_SCREW_TASKS = [
    pytest.param(name, cls, id=name)
    for name, cls in _NEUTRAL_CLASSES.items()
    if name not in _WITHOUT_SCREW_DELEGATE
]


def _screw_delegate(task_cls):
    mro = task_cls.__mro__
    neutral_index = mro.index(NeutralAvatarTableWorkMixin)
    return next(
        (cls for cls in mro[neutral_index + 1:] if "_move_screw" in cls.__dict__),
        None,
    )


def test_all_ten_registered_neutral_mixin_inheritors_are_covered():
    assert set(_NEUTRAL_CLASSES) == {
        "rank_blocks_by_color_copresence",
        "rank_blocks_by_size_copresence",
        "categorize_objects_copresence",
        "dump_bin_copresence",
        "place_bread_in_basket_copresence",
        "place_burger_and_fries_copresence",
        "place_pair_of_shoes_copresence",
        "place_food_in_skillet_copresence",
        "put_object_in_cabinet_copresence",
        "stack_three_bowls_copresence",
    }


def test_only_bread_and_bowls_lack_a_downstream_screw_delegate():
    assert {
        name for name, cls in _NEUTRAL_CLASSES.items()
        if _screw_delegate(cls) is None
    } == _WITHOUT_SCREW_DELEGATE


@pytest.mark.parametrize("task_name", sorted(_WITHOUT_SCREW_DELEGATE))
def test_missing_delegate_attribute_error_propagates_after_refresh(
    monkeypatch, task_name,
):
    task_cls = _NEUTRAL_CLASSES[task_name]
    assert _screw_delegate(task_cls) is None
    task = task_cls.__new__(task_cls)
    events = []
    monkeypatch.setattr(
        task, "_refresh_avatar_planner_obstacles",
        lambda arm_tag: events.append(("refresh", arm_tag)),
    )

    with pytest.raises(AttributeError, match="'super' object has no attribute '_move_screw'"):
        task._move_screw(object(), "right")

    assert events == [("refresh", "right")]


@pytest.mark.parametrize("task_name, task_cls", _SCREW_TASKS)
@pytest.mark.parametrize("args, kwargs", _CALLS)
def test_actual_neutral_mro_accepts_forwarded_yaw(
    monkeypatch, task_name, task_cls, args, kwargs,
):
    """Bind the forwarded call to each real downstream consumer signature.

    Replace only that consumer's body; do not initialize or command a robot.
    Binding against its original signature rejects unsupported pass-through
    arguments and preserves its real None-versus-zero default semantics.
    """
    consumer = _screw_delegate(task_cls)
    assert consumer is not None, f"{task_name} unexpectedly lost its screw delegate"
    signature = inspect.signature(consumer.__dict__["_move_screw"])
    expected_default = None if task_name == "categorize_objects_copresence" else 0.0
    assert signature.parameters["tcp_yaw"].default == expected_default

    events = []
    result = object()
    task = task_cls.__new__(task_cls)
    target = object()

    def record_delegate(self, *forwarded_args, **forwarded_kwargs):
        assert self is task
        assert events == [("refresh", "right")]
        bound = signature.bind(self, *forwarded_args, **forwarded_kwargs)
        bound.apply_defaults()
        assert bound.arguments["target_pos"] is target
        assert bound.arguments["arm_tag"] == "right"
        yaw = args[0] if args else kwargs.get("tcp_yaw", expected_default)
        assert bound.arguments["tcp_yaw"] == yaw
        events.append(("delegate", forwarded_args, forwarded_kwargs))
        return result

    monkeypatch.setattr(consumer, "_move_screw", record_delegate)
    monkeypatch.setattr(
        task, "_refresh_avatar_planner_obstacles",
        lambda arm_tag: events.append(("refresh", arm_tag)),
    )

    assert task._move_screw(target, "right", *args, **kwargs) is result
    assert events == [
        ("refresh", "right"),
        ("delegate", (target, "right", *args), kwargs),
    ]
