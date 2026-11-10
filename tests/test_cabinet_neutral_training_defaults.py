"""Cabinet-only training repairs must retain the common layout/scoring gates."""
import pytest

from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.tasks.put_object_in_cabinet_copresence import PutObjectInCabinetCopresence


def task(config=None):
    result = PutObjectInCabinetCopresence.__new__(PutObjectInCabinetCopresence)
    result.config = config or {}
    return result


@pytest.mark.parametrize("override", [None, True, False])
def test_laptop_visual_pack_is_cabinet_default_with_explicit_opt_out(monkeypatch, override):
    received = []
    monkeypatch.setattr(NeutralAvatarTableWorkMixin, "__init__",
                        lambda self, config: received.append(config))
    supplied = {"renderer": "nyx", "neutral_avatar_strict_layout": True}
    if override is not None:
        supplied["nyx_laptop_mesh_pack_v1"] = override
    original = supplied.copy()
    PutObjectInCabinetCopresence(supplied)
    assert received == [dict(original, nyx_laptop_mesh_pack_v1=(override is not False))]
    assert supplied == original


def test_sponge_fallback_is_default_and_same_heading_only():
    assert task()._neutral_avatar_use_layout_recovery({"kind": "sponge"})
    assert PutObjectInCabinetCopresence.NEUTRAL_LAYOUT_RECOVERY_DEGREES == (0,)


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_only_literal_true_enables_cabinet_fallback(value):
    assert not task({"cabinet_neutral_layout_recovery_v1": value})._neutral_avatar_use_layout_recovery(
        {"kind": "sponge"})


@pytest.mark.parametrize("kind", ["put_objects_in_bowl", "laptop", "mouse", "phone", "writing", "open_bottle"])
def test_non_sponge_jobs_keep_parent_policy(kind):
    candidate = task()
    job = {"kind": kind}
    assert candidate._neutral_avatar_use_layout_recovery(job) == (
        NeutralAvatarTableWorkMixin._neutral_avatar_use_layout_recovery(candidate, job))


def test_dense_search_and_acceptance_are_unmodified_shared_implementations():
    for name in ("_neutral_avatar_fallback_work_candidates", "_prepare_neutral_avatar_start_layout",
                 "_neutral_avatar_body_pose_is_valid"):
        assert getattr(PutObjectInCabinetCopresence, name) is getattr(NeutralAvatarTableWorkMixin, name)
    assert "evaluate" not in PutObjectInCabinetCopresence.__dict__
