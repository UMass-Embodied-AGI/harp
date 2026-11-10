"""Blind-expert ablation hides the avatar from planning but not from scoring."""

from types import SimpleNamespace

from envs.base_task import BaseTask, _PlanningBlindCollider
from scripts.eval_expert_set import hides_avatar_from_planning


class _Collider:
    def __init__(self):
        self.updates = 0

    def current_capsules(self):
        return [("torso", (0, 0, 0), (0, 0, 1), 0.1)]

    def current_boxes(self):
        return [("palm", None)]

    def update(self):
        self.updates += 1


def test_planning_view_is_empty_but_scoring_keeps_real_collider():
    real = _Collider()
    checker = SimpleNamespace(collider=real)
    task = SimpleNamespace(avatar_collider=real, avatar_collision_checker=checker)
    BaseTask.hide_avatar_from_planning(task)
    assert isinstance(task.avatar_collider, _PlanningBlindCollider)
    assert task.avatar_collider.current_capsules() == []
    assert task.avatar_collider.current_boxes() == []
    task.avatar_collider.update()
    assert real.updates == 1
    assert checker.collider.current_capsules()
    BaseTask.hide_avatar_from_planning(task)
    assert task.avatar_collider._collider is real


def test_only_unprivileged_expert_on_copresence_rows_is_blinded():
    assert hides_avatar_from_planning({"category": "copresence"}, "expert")
    assert not hides_avatar_from_planning({"category": "cueing"}, "expert")
    assert not hides_avatar_from_planning({"category": "contribution"}, "expert")
    assert not hides_avatar_from_planning({"category": "copresence"}, "expert_full_state")
    # Pre-HARP manifests stored "neutral".
    assert hides_avatar_from_planning({"category": "neutral"}, "expert")
