"""CPU-only campaign admission, resume, and dry-run regression tests.

Every scheduler interaction is mocked; no job or simulator is started.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.vla_data import nyx_campaign as campaign


def _state():
    return {"attempts": {}, "next_seed": {}, "batches": {}, "sequence": 0}


def _spec(tmp_path, *, status="ready", target=300):
    return {
        "status": status,
        "target": target,
        "source": str(tmp_path / "frozen_source"),
        "config": str(tmp_path / "collection.json"),
    }


def _manifest(tmp_path):
    return {
        "root": str(tmp_path / "campaign"),
        "log_root": str(tmp_path / "logs"),
        "campaign": "test_campaign",
        "user": "test_user",
        "job_prefix": "nyx_test",
        "python": "/unused/python",
        "utility": "/unused/nyx_campaign.py",
        "constraint": "l40s",
        "exclude": "unused-node",
        "parallel_by_gpu": {},
        "max_processes": 1,
        "max_gpu_jobs": 40,
        "pilot_successes": 3,
        "seed_start": 1000000,
        "tasks": {"pour_water": _spec(tmp_path)},
    }


def _row(seed, status, *, task="pour_water", attempt=0):
    ident = f"{task}__s{seed}__a{attempt}"
    return {
        "id": ident,
        "task": task,
        "seed": seed,
        "attempt": attempt,
        "status": status,
    }


def _put(state, row):
    state["attempts"][row["id"]] = row
    return row


def _gpu_jobs(count):
    return {
        str(index): {"gpu": True, "name": f"external_{index}"}
        for index in range(count)
    }


def _marked_episode(path, *, task="pour_water", seed=1000000):
    path.mkdir(parents=True)
    fingerprints = {}
    for name in ("steps.h5", "meta.json", "verdict.json"):
        item = path / name
        item.write_bytes(b"validated artifact")
        stat = item.stat()
        fingerprints[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    campaign.atomic_json(path / "validated.json", {
        "validated": True, "renderer": "nyx", "files": fingerprints,
        "task": task, "seed": seed, "jpeg_validation": "all",
    })
    return path


def _tree_bytes(root):
    """Ignore the advisory lock, which may be opened by a read-only preview."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (
            ("link", str(path.readlink())) if path.is_symlink()
            else ("file", path.read_bytes()) if path.is_file()
            else ("directory", None)
        )
        for path in root.rglob("*")
        if path.name != "dispatcher.lock"
    }


def test_queue_expands_pending_arrays_and_counts_them_toward_global_cap(monkeypatch):
    lines = [f"{i}|gpu|RUNNING|gres/gpu:1|external_{i}" for i in range(46)]
    lines += [f"200_{i}|gpu-preempt|PENDING|gres/gpu:1|external_array" for i in range(4)]
    lines += ["300|cpu|RUNNING|N/A|cpu_job"]
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(stdout="\n".join(lines))

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    jobs = campaign.queue_jobs("test_user")

    assert "-r" in observed[0][0]
    assert observed[0][1]["check"] is True
    assert len(jobs) == 51
    assert campaign.available_gpu_jobs(jobs, set()) == 0
    del jobs["200_3"]
    assert campaign.available_gpu_jobs(jobs, set()) == 1


def test_campaign_limit_and_global_limit_both_apply():
    jobs = _gpu_jobs(48)
    assert campaign.available_gpu_jobs(jobs, {str(i) for i in range(39)}) == 1
    assert campaign.available_gpu_jobs(jobs, set()) == 2
    assert campaign.available_gpu_jobs(jobs, {str(i) for i in range(40)}) == 0
    assert campaign.available_gpu_jobs(_gpu_jobs(55), set()) == 0


def test_queue_failure_is_not_treated_as_an_empty_queue(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "squeue")

    monkeypatch.setattr(campaign.subprocess, "run", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.queue_jobs("test_user")


def test_allocation_stops_at_300_including_assigned_attempts(tmp_path):
    state = _state()
    manifest = _manifest(tmp_path)
    for seed in range(298):
        _put(state, _row(seed, "accepted"))
    _put(state, _row(9000, "assigned"))

    item = campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest)
    assert item is not None
    assert item["seed"] == 1000000
    assert campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest) is None
    counts = campaign.task_counts(state, "pour_water")
    assert counts["accepted"] == 298
    assert counts["assigned"] == 2


@pytest.mark.parametrize("status", ["pending", "blocked", "", None])
def test_only_explicit_ready_status_admits_new_work(tmp_path, status):
    state = _state()
    assert campaign.allocate_attempt(state, "pour_water", _spec(tmp_path, status=status), _manifest(tmp_path)) is None
    assert state == _state()


def test_paused_ready_task_remains_unassigned(tmp_path):
    state = _state()
    state["paused"] = {"pour_water": "Inspect pilot failure"}
    assert campaign.allocate_attempt(state, "pour_water", _spec(tmp_path), _manifest(tmp_path)) is None
    assert state["attempts"] == {}
    assert state["next_seed"] == {}


def test_pilot_admits_only_two_inflight_until_three_successes(tmp_path):
    state = _state()
    manifest = _manifest(tmp_path)
    spec = manifest["tasks"]["pour_water"]
    first = campaign.allocate_attempt(state, "pour_water", spec, manifest)
    second = campaign.allocate_attempt(state, "pour_water", spec, manifest)
    assert first is not None and second is not None
    assert campaign.allocate_attempt(state, "pour_water", spec, manifest) is None

    first["status"] = "accepted"
    third = campaign.allocate_attempt(state, "pour_water", spec, manifest)
    assert third is not None
    assert campaign.allocate_attempt(state, "pour_water", spec, manifest) is None

    second["status"] = third["status"] = "accepted"
    for _ in range(4):
        assert campaign.allocate_attempt(state, "pour_water", spec, manifest) is not None
    assert campaign.task_counts(state, "pour_water")["assigned"] == 4


def test_retry_reuses_seed_with_a_new_attempt_identity(tmp_path):
    state = _state()
    state["next_seed"]["pour_water"] = 1000100
    old = _put(state, _row(1000007, "retry", attempt=1))
    manifest = _manifest(tmp_path)

    item = campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest)
    assert item["seed"] == old["seed"]
    assert item["attempt"] == 2
    assert item["id"] != old["id"]
    assert old["status"] == "infra_failed"
    assert state["next_seed"]["pour_water"] == 1000100
    assert len(state["attempts"]) == 2


@pytest.mark.parametrize("name", ["steps.h5", "meta.json", "verdict.json"])
def test_mutated_artifact_invalidates_validation_marker(tmp_path, name):
    episode = _marked_episode(tmp_path / "episode")
    assert campaign.marker_valid(episode)
    (episode / name).write_bytes(b"changed after validation")
    assert not campaign.marker_valid(episode)


def test_rasterizer_marker_cannot_admit_an_episode(tmp_path):
    episode = _marked_episode(tmp_path / "episode")
    marker = campaign.read_json(episode / "validated.json")
    marker["renderer"] = "rasterizer"
    campaign.atomic_json(episode / "validated.json", marker)
    assert not campaign.marker_valid(episode)


def test_visible_frame_marker_requires_explicit_check_and_exact_threshold(tmp_path):
    episode = _marked_episode(tmp_path / "episode")
    assert campaign.marker_valid(episode)
    assert not campaign.marker_valid(episode, require_visible_frames=True)
    marker = campaign.read_json(episode / "validated.json")
    marker.update(visible_frames_checked=True, visible_frames_max_rgb_threshold=3,
                  jpeg_validation="all")
    campaign.atomic_json(episode / "validated.json", marker)
    assert campaign.marker_valid(episode, require_visible_frames=True)
    marker["visible_frames_max_rgb_threshold"] = 1
    campaign.atomic_json(episode / "validated.json", marker)
    assert not campaign.marker_valid(episode, require_visible_frames=True)


@pytest.mark.parametrize("require_visible_frames", [False, True])
@pytest.mark.parametrize("with_receipt", [False, True])
def test_worker_cannot_skip_strict_validation_using_old_marker(tmp_path, require_visible_frames, with_receipt):
    batch = _manifest(tmp_path)
    batch.update(require_visible_frames=require_visible_frames,
                 preflight_error="strict validation must not skip directly to acceptance")
    item = {**_row(1000000, "assigned"), **_spec(tmp_path)}
    episode = _marked_episode(Path(batch["root"]) / "attempts" / item["id"] / "nyx" / "seed_1000000")
    marker = campaign.read_json(episode / "validated.json")
    marker.update(task=item["task"], seed=item["seed"], jpeg_validation="all")
    campaign.atomic_json(episode / "validated.json", marker)
    if with_receipt:
        campaign.atomic_json(Path(batch["root"]) / "receipts" / (item["id"] + ".json"),
                             {**item, "status": "accepted", "episode": str(episode)})
    result = campaign.run_attempt(item, batch)
    assert result["status"] == ("infra_failed" if require_visible_frames else "accepted")


def test_strict_reconcile_does_not_admit_old_receipts(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["require_visible_frames"] = True
    state = _state()
    item = _put(state, _row(1000000, "assigned"))
    episode = _marked_episode(tmp_path / "episode")
    campaign.atomic_json(Path(manifest["root"]) / "receipts" / (item["id"] + ".json"),
                         {**item, "status": "accepted", "episode": str(episode)})
    campaign.reconcile(state, manifest, {})
    assert item["status"] == "retry"
    assert not (Path(manifest["root"]) / "accepted").exists()


@pytest.mark.parametrize("require_visible_frames", [False, True])
def test_manifest_visible_frame_flag_reaches_saved_batch(tmp_path, monkeypatch, require_visible_frames):
    manifest = _manifest(tmp_path)
    manifest.update(require_visible_frames=require_visible_frames, submissions_per_poll=1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(campaign, "queue_jobs", lambda _user: {})
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_kw:
                        SimpleNamespace(stdout="123456\n", stderr=""))
    assert campaign.dispatch(manifest_path, once=True) == 0
    paths = list((Path(manifest["root"]) / "batches").glob("*.json"))
    assert len(paths) == 1
    assert campaign.read_json(paths[0])["require_visible_frames"] is require_visible_frames


@pytest.mark.parametrize("require_visible_frames", [False, True])
def test_worker_passes_optional_visible_frame_flag_to_validator(tmp_path, monkeypatch, require_visible_frames):
    batch = _manifest(tmp_path)
    batch["require_visible_frames"] = require_visible_frames
    item = {**_row(1000000, "assigned"), **_spec(tmp_path)}
    source = Path(item["source"])
    (source / "scripts/vla_data").mkdir(parents=True)
    (source / "scripts/vla_data/collect_vla.py").write_text("# mocked collector\n")
    Path(item["config"]).write_text("{}")
    episode = Path(batch["root"]) / "attempts" / item["id"] / "nyx" / "seed_1000000"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text("{}")
    (episode / "verdict.json").write_text('{"success": true}')
    observed = []

    def validate(command, **_kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_a, **_kw:
                        SimpleNamespace(wait=lambda **_kwargs: 0, poll=lambda: 0))
    monkeypatch.setattr(campaign.subprocess, "run", validate)
    result = campaign.run_attempt(item, batch)
    assert result["status"] == "infra_failed"
    assert len(observed) == 1
    assert ("--require-visible-frames" in observed[0]) is require_visible_frames


@pytest.mark.parametrize("tampered", [False, True])
def test_reconcile_rechecks_accepted_receipt_before_publishing(tmp_path, tampered):
    state = _state()
    manifest = _manifest(tmp_path)
    item = _put(state, _row(1000000, "assigned"))
    episode = _marked_episode(tmp_path / "episode")
    if tampered:
        (episode / "steps.h5").unlink()
    campaign.atomic_json(Path(manifest["root"]) / "receipts" / (item["id"] + ".json"), {
        **item, "status": "accepted", "episode": str(episode),
    })

    campaign.reconcile(state, manifest, {})

    selected = Path(manifest["root"]) / "accepted" / "pour_water" / "nyx" / "seed_1000000"
    if tampered:
        assert item["status"] == "retry"
        assert not selected.is_symlink()
    else:
        assert item["status"] == "accepted"
        assert selected.is_symlink()
        assert selected.resolve() == episode


@pytest.mark.parametrize("attempt,expected", [(0, "retry"), (2, "exhausted")])
def test_disappeared_job_obeys_retry_budget(tmp_path, monkeypatch, attempt, expected):
    state = _state()
    item = _put(state, _row(1000000, "assigned", attempt=attempt))
    item.update(job_id="gone", submitted=100.0)
    monkeypatch.setattr(campaign.time, "time", lambda: 281.0)
    campaign.reconcile(state, _manifest(tmp_path), {})
    assert item["status"] == "assigned"
    assert item["missing_since"] == 281.0
    monkeypatch.setattr(campaign.time, "time", lambda: 461.0)
    campaign.reconcile(state, _manifest(tmp_path), {})
    assert item["status"] == expected


@pytest.mark.parametrize("gpu_count", [0, 50])
def test_dry_run_never_writes_campaign_state_or_attempts(tmp_path, monkeypatch, gpu_count):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    root = Path(manifest["root"])
    before = _tree_bytes(root)
    monkeypatch.setattr(campaign, "queue_jobs", lambda _user: _gpu_jobs(gpu_count))
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_a, **_kw: pytest.fail("dry-run submitted a job"))

    assert campaign.dispatch(manifest_path, once=True, dry_run=True) == 0

    assert _tree_bytes(root) == before


def test_dry_run_does_not_publish_receipts_or_modify_persisted_state(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest["tasks"]["pour_water"]["target"] = 1
    root = Path(manifest["root"])
    state = _state()
    item = _put(state, _row(1000000, "assigned"))
    episode = _marked_episode(tmp_path / "episode")
    campaign.atomic_json(root / "state.json", state)
    campaign.atomic_json(root / "receipts" / (item["id"] + ".json"), {
        **item, "status": "accepted", "episode": str(episode),
    })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    before = _tree_bytes(root)
    monkeypatch.setattr(campaign, "queue_jobs", lambda _user: {})
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_a, **_kw: pytest.fail("dry-run submitted a job"))

    assert campaign.dispatch(manifest_path, once=True, dry_run=True) == 0

    assert _tree_bytes(root) == before


def _accepted_receipt(manifest, item):
    root = Path(manifest["root"])
    episode = _marked_episode(root / "attempts" / item["id"] / "nyx" / f"seed_{item['seed']}",
                              task=item["task"], seed=item["seed"])
    if manifest.get("require_visible_frames"):
        marker = campaign.read_json(episode / "validated.json")
        marker.update(visible_frames_checked=True, visible_frames_max_rgb_threshold=3)
        campaign.atomic_json(episode / "validated.json", marker)
    receipt = {**item, "status": "accepted", "episode": str(episode)}
    campaign.atomic_json(root / "receipts" / (item["id"] + ".json"), receipt)
    return episode, receipt


def test_queue_reappearance_resets_disappearance_grace(tmp_path, monkeypatch):
    state = _state()
    item = _put(state, _row(1000000, "assigned"))
    item.update(job_id="123", submitted=0)
    manifest = _manifest(tmp_path)
    for now, live in [(1000, {}), (1100, {"123": {}}), (1200, {}), (1379, {})]:
        monkeypatch.setattr(campaign.time, "time", lambda: now)
        campaign.reconcile(state, manifest, live)
        assert item["status"] == "assigned"
        if live:
            assert "missing_since" not in item
    assert item["missing_since"] == 1200
    monkeypatch.setattr(campaign.time, "time", lambda: 1380)
    campaign.reconcile(state, manifest, {})
    assert item["status"] == "retry"


@pytest.mark.parametrize("status", ["assigned", "retry", "infra_failed", "exhausted"])
def test_late_accepted_receipt_is_consumed_from_every_prior_status(tmp_path, status):
    manifest = _manifest(tmp_path)
    manifest["require_visible_frames"] = True
    state = _state()
    item = _put(state, _row(1000000, status))
    episode, receipt = _accepted_receipt(manifest, item)
    receipt_path = Path(manifest["root"]) / "receipts" / (item["id"] + ".json")
    before = receipt_path.read_bytes()

    campaign.reconcile(state, manifest, {})
    campaign.reconcile(state, manifest, {})

    assert item["status"] == "accepted"
    assert all(item[key] == value for key, value in receipt.items())
    assert receipt_path.read_bytes() == before
    assert campaign.task_counts(state, item["task"])["accepted"] == 1
    assert (Path(manifest["root"]) / "accepted" / item["task"] / "nyx" / "seed_1000000").resolve() == episode


def test_late_success_supersedes_assigned_retry_without_duplicate_count_or_allocation(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["tasks"]["pour_water"]["target"] = 2
    state = _state()
    original = _put(state, _row(1000000, "infra_failed"))
    retry = _put(state, _row(1000000, "assigned", attempt=1))
    retry.update(job_id="retry_job", submitted=0)
    episode, _ = _accepted_receipt(manifest, original)

    campaign.reconcile(state, manifest, {"retry_job": {}})
    assert original["status"] == "accepted"
    assert retry["status"] == "superseded"
    assert campaign.task_counts(state, "pour_water")["assigned"] == 0
    next_item = campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest)
    assert next_item["seed"] == 1000001
    assert campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest) is None

    # An immutable retry batch can still finish. Its success must not replace
    # the existing winner, inflate totals, or create a second published episode.
    _accepted_receipt(manifest, retry)
    campaign.reconcile(state, manifest, {"retry_job": {}})
    campaign.reconcile(state, manifest, {"retry_job": {}})
    assert original["status"] == "accepted"
    assert retry["status"] == "superseded"
    counts = campaign.task_counts(state, "pour_water")
    assert counts["accepted"] == counts["assigned"] == 1
    assert (Path(manifest["root"]) / "accepted/pour_water/nyx/seed_1000000").resolve() == episode


def test_existing_accepted_retry_remains_winner_when_earlier_receipt_arrives(tmp_path):
    manifest = _manifest(tmp_path)
    state = _state()
    original = _put(state, _row(1000000, "infra_failed"))
    retry = _put(state, _row(1000000, "assigned", attempt=1))
    episode, receipt = _accepted_receipt(manifest, retry)
    campaign.reconcile(state, manifest, {})
    _accepted_receipt(manifest, original)
    campaign.reconcile(state, manifest, {})
    assert original["status"] == "superseded"
    assert retry["status"] == "accepted"
    assert all(retry[key] == value for key, value in receipt.items())
    assert (Path(manifest["root"]) / "accepted/pour_water/nyx/seed_1000000").resolve() == episode


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_simultaneous_successes_choose_deterministic_winner(tmp_path, order):
    manifest = _manifest(tmp_path)
    state = _state()
    for attempt in order:
        item = _put(state, _row(1000000, "infra_failed", attempt=attempt))
        _accepted_receipt(manifest, item)
    campaign.reconcile(state, manifest, {})
    winners = [row for row in state["attempts"].values() if row["status"] == "accepted"]
    assert len(winners) == 1
    assert winners[0]["attempt"] == 0


def test_late_success_at_target_is_preserved_as_surplus_without_extra_symlink(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["tasks"]["pour_water"]["target"] = 1
    state = _state()
    admitted = _put(state, _row(1000001, "assigned"))
    _accepted_receipt(manifest, admitted)
    campaign.reconcile(state, manifest, {})
    late = _put(state, _row(1000000, "exhausted"))
    episode, _ = _accepted_receipt(manifest, late)
    campaign.reconcile(state, manifest, {})
    campaign.reconcile(state, manifest, {})
    assert late["status"] == "surplus"
    assert admitted["status"] == "accepted"
    assert episode.is_dir()
    assert not (Path(manifest["root"]) / "accepted/pour_water/nyx/seed_1000000").exists()
    assert campaign.task_counts(state, "pour_water")["accepted"] == 1
    assert campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest) is None


@pytest.mark.parametrize("field", ["id", "task", "seed", "attempt", "source", "config"])
def test_receipt_identity_mismatch_cannot_overwrite_assigned_identity(tmp_path, field):
    manifest = _manifest(tmp_path)
    state = _state()
    item = _put(state, {**_spec(tmp_path), **_row(1000000, "assigned")})
    identity = {key: item[key] for key in ("id", "task", "seed", "attempt", "source", "config")}
    _, receipt = _accepted_receipt(manifest, item)
    receipt[field] = "wrong identity"
    campaign.atomic_json(Path(manifest["root"]) / "receipts" / (item["id"] + ".json"), receipt)
    campaign.reconcile(state, manifest, {})
    assert {key: item[key] for key in identity} == identity
    assert item["status"] == "retry"
    assert "identity" in item["error"]
    assert not (Path(manifest["root"]) / "accepted").exists()


@pytest.mark.parametrize("field,value", [("task", "another_task"), ("seed", 1000001), ("jpeg_validation", "representative")])
def test_matching_marker_identity_and_all_frame_validation_are_required(tmp_path, field, value):
    manifest = _manifest(tmp_path)
    state = _state()
    item = _put(state, _row(1000000, "infra_failed"))
    episode, _ = _accepted_receipt(manifest, item)
    marker = campaign.read_json(episode / "validated.json")
    marker[field] = value
    campaign.atomic_json(episode / "validated.json", marker)
    campaign.reconcile(state, manifest, {})
    assert item["status"] == "retry"
    assert not (Path(manifest["root"]) / "accepted").exists()


def test_repeated_failed_receipt_does_not_requeue_an_older_attempt(tmp_path):
    manifest = _manifest(tmp_path)
    state = _state()
    original = _put(state, _row(1000000, "assigned"))
    receipt = {**original, "status": "infra_failed"}
    campaign.atomic_json(Path(manifest["root"]) / "receipts" / (original["id"] + ".json"), receipt)
    campaign.reconcile(state, manifest, {})
    retry = campaign.allocate_attempt(state, "pour_water", manifest["tasks"]["pour_water"], manifest)
    retry["job_id"] = "123"
    for _ in range(3):
        campaign.reconcile(state, manifest, {"123": {}})
    assert original["status"] == "superseded"
    assert retry["status"] == "assigned"
    assert campaign.task_counts(state, "pour_water")["retry"] == 0
    assert campaign.task_counts(state, "pour_water")["assigned"] == 1


@pytest.mark.parametrize("count,status,reason", [
    (12, "infra_failed", "12 consecutive infrastructure failures"),
    (20, "task_failed", "initial 20 completed attempts"),
])
def test_failure_guard_includes_real_failures_from_superseded_attempts(tmp_path, count, status, reason):
    state = _state()
    for index in range(count):
        _put(state, _row(1000000 + index // 3, status, attempt=index % 3))
    campaign.reconcile(state, _manifest(tmp_path), {})
    assert reason in state["paused"]["pour_water"]
    assert set(campaign.task_counts(state, "pour_water")) == {
        "accepted", "assigned", "retry", "task_failed", "infra_failed", "exhausted",
    }


@pytest.mark.parametrize("stdout,expected", [
    ("123456\n", "123456"),
    ("123456;cluster-name\n", "123456"),
    ("sbatch: informational memory notice\n123456\n", "123456"),
])
def test_parsable_submission_id_tolerates_notices_and_cluster_suffix(stdout, expected):
    assert campaign.parse_submission_id(stdout) == expected


@pytest.mark.parametrize("stdout", ["", "sbatch: notice only\n", "123\n456\n", "123;cluster;extra\n"])
def test_ambiguous_submission_id_is_not_guessed(stdout):
    with pytest.raises(ValueError):
        campaign.parse_submission_id(stdout)


def test_submission_stderr_notice_does_not_stop_dispatch_wave(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest.update(pilot_successes=0, submissions_per_poll=2)
    manifest_path = tmp_path / "manifest.json"
    campaign.atomic_json(manifest_path, manifest)
    monkeypatch.setattr(campaign, "queue_jobs", lambda _user: {})
    submitted = []

    def fake_run(command, **kwargs):
        assert command[0] == "sbatch"
        assert kwargs["capture_output"] and kwargs["check"]
        submitted.append(command)
        return SimpleNamespace(stdout=f"{123450 + len(submitted)};cluster\n",
                               stderr="sbatch: informational memory notice\n")

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    assert campaign.dispatch(manifest_path, once=True) == 0
    assert len(submitted) == 2
    state = campaign.read_json(Path(manifest["root"]) / "state.json")
    assert {batch["job_id"] for batch in state["batches"].values()} == {"123451", "123452"}


@pytest.mark.parametrize("use_profiles", [False, True])
@pytest.mark.parametrize("initial_sequence", [0, 5])
def test_submission_profiles_alternate_across_resume_and_preserve_defaults(
    tmp_path, monkeypatch, use_profiles, initial_sequence,
):
    manifest = _manifest(tmp_path)
    manifest.update(pilot_successes=0, submissions_per_poll=4, cpus_per_job=4,
                    constraint="2080_ti|l40s|a40", memory_per_job="31G",
                    time_per_job="01:15:00")
    if use_profiles:
        manifest["submission_profiles"] = [
            {"name": "rtx2080ti", "cpus_per_job": 2, "constraint": "2080_ti"},
            {},
        ]
    manifest_path = tmp_path / "manifest.json"
    campaign.atomic_json(manifest_path, manifest)
    state = _state()
    state["sequence"] = initial_sequence
    campaign.atomic_json(Path(manifest["root"]) / "state.json", state)
    monkeypatch.setattr(campaign, "queue_jobs", lambda _user: {})
    submitted = []

    def fake_run(command, **_kwargs):
        assert command[0] == "sbatch"
        submitted.append(command)
        return SimpleNamespace(stdout=f"{123450 + len(submitted)}\n", stderr="")

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    assert campaign.dispatch(manifest_path, once=True) == 0
    assert len(submitted) == 4
    for offset, command in enumerate(submitted):
        sequence = initial_sequence + offset + 1
        special = use_profiles and sequence % 2 == 1
        cpus = 2 if special else 4
        constraint = "2080_ti" if special else manifest["constraint"]
        assert f"--cpus-per-task={cpus}" in command
        assert f"--constraint={constraint}" in command
        assert "--mem=31G" in command
        assert "--time=01:15:00" in command
        assert "--partition=gpu,gpu-preempt" in command
        assert "--gres=gpu:1" in command
        batch = campaign.read_json(
            Path(manifest["root"]) / "batches" / f"nyx_test_{sequence:06d}.json")
        if use_profiles:
            expected = {"index": (sequence - 1) % 2,
                        "cpus_per_job": cpus, "constraint": constraint}
            if special:
                expected["name"] = "rtx2080ti"
            assert batch["submission_profile"] == expected
        else:
            assert "submission_profile" not in batch


@pytest.mark.parametrize("profiles", [
    None, {}, [], [None], ["rtx2080ti"],
    [{"cpus_per_job": 0}], [{"cpus_per_job": -2}],
    [{"cpus_per_job": True}], [{"cpus_per_job": 2.5}], [{"cpus_per_job": "2"}],
    [{"constraint": ""}], [{"constraint": "  "}], [{"constraint": None}],
    [{"name": ""}], [{"name": 2}],
    [{"cpu_per_job": 2}], [{"memory_per_job": "16G"}],
    [{}, {"constraint": ""}],
])
def test_malformed_submission_profiles_fail_before_saving_intents_or_submitting(
    tmp_path, monkeypatch, profiles,
):
    manifest = _manifest(tmp_path)
    manifest["submission_profiles"] = profiles
    manifest_path = tmp_path / "manifest.json"
    campaign.atomic_json(manifest_path, manifest)
    root = Path(manifest["root"])
    campaign.atomic_json(root / "state.json", _state())
    before = _tree_bytes(root)
    monkeypatch.setattr(campaign, "queue_jobs",
                        lambda _user: pytest.fail("Malformed profiles reached queue lookup"))
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda *_a, **_kw: pytest.fail("Malformed profiles submitted work"))

    with pytest.raises(ValueError, match="submission_profiles"):
        campaign.dispatch(manifest_path, once=True)
    assert _tree_bytes(root) == before
