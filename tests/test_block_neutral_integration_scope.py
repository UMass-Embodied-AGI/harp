"""The block-neutral recovery does not opt sibling experts into new layouts."""
import pytest

from envs.task_bases.neutral_avatar_layout import _NeutralAvatarLayoutMixin
from envs.tasks import resolve_task_class
from envs.tasks.registry import TASK_DESCRIPTORS, TaskRegistration
from test_neutral_pose_fallback import Selected, acceptance_task
from test_neutral_can_pose_fallback import can_task


def test_only_block_neutral_tasks_enable_layout_recovery():
    expected = {"rank_blocks_by_color_copresence", "rank_blocks_by_size_copresence"}
    actual = {
        descriptor.name for descriptor in TASK_DESCRIPTORS
        if descriptor.registration is TaskRegistration.CANONICAL
        and getattr(resolve_task_class(descriptor.name)[1], "BLOCK_NEUTRAL_LAYOUT_RECOVERY", False)
    }
    assert actual == expected


def test_default_layout_does_not_enter_fallback():
    task, calls, _ = acceptance_task()
    del task.BLOCK_NEUTRAL_LAYOUT_RECOVERY
    with pytest.raises(RuntimeError, match="no neutral layout candidate"):
        task._prepare_neutral_avatar_start_layout()
    assert calls == []


def test_direct_fallback_is_disabled_without_task_opt_in():
    task = _NeutralAvatarLayoutMixin()
    assert list(task._neutral_avatar_fallback_work_candidates(
        {"kind": "put_objects_in_bowl"}, {}, [],
    )) == []


def test_nonblock_can_acceptance_keeps_its_previous_gate():
    task, calls, _ = can_task(rejected_gate="region")
    del task.BLOCK_NEUTRAL_LAYOUT_RECOVERY
    with pytest.raises(Selected):
        task._prepare_neutral_avatar_start_layout()
    assert calls == []
