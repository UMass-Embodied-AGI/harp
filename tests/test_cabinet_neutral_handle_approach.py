"""Neutral compact pre-grasp preserves native drawer motion and sibling scope."""
import pytest

from envs.task_bases.put_object_cabinet import PutObjectCabinet
from envs.tasks.put_object_in_cabinet_contribution import PutObjectInCabinetContribution
from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
from envs.tasks.put_object_in_cabinet_copresence import PutObjectInCabinetCopresence


@pytest.mark.parametrize("kind, expected", [
    ("bottle", .04), ("writing", .04), ("sponge", .08),
    ("mouse", .08), ("put_objects_in_bowl", .08), (None, .08),
])
def test_compact_approach_uses_unchanged_native_drawer_motion(monkeypatch, kind, expected):
    task = PutObjectInCabinetCopresence.__new__(PutObjectInCabinetCopresence)
    task._neutral_avatar_job = {"kind": kind}
    calls = []

    def native(owner, arm_tag):
        calls.append((owner, arm_tag, owner._HANDLE_PRE_DIST))
        assert PutObjectCabinet._HANDLE_PRE_DIST == .08
        assert PutObjectInCabinetContribution._HANDLE_PRE_DIST == .08
        assert PutObjectInCabinetInterruption._HANDLE_PRE_DIST == .08
        return False

    monkeypatch.setattr(PutObjectCabinet, "_open_drawer_physics", native)
    assert task._open_drawer_physics("right") is False
    assert calls == [(task, "right", expected)]
    assert "_HANDLE_PRE_DIST" not in task.__dict__


def test_compact_approach_restores_existing_override_on_native_failure(monkeypatch):
    task = PutObjectInCabinetCopresence.__new__(PutObjectInCabinetCopresence)
    task._neutral_avatar_job = {"kind": "bottle"}
    task._HANDLE_PRE_DIST = .075

    def native(owner, _):
        assert owner._HANDLE_PRE_DIST == .04
        raise RuntimeError("native collision rejection")

    monkeypatch.setattr(PutObjectCabinet, "_open_drawer_physics", native)
    with pytest.raises(RuntimeError, match="native collision rejection"):
        task._open_drawer_physics("right")
    assert task._HANDLE_PRE_DIST == .075
