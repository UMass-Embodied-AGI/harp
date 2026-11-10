"""Durable, bounded SLURM collection campaign for an explicit NYX task list.

The coordinator is the only state writer. Jobs receive immutable attempt lists;
each child runs one scene, writes its own receipt, and exits. No shared SQLite
database or concurrent mutable episode directory is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def queue_jobs(user):
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", user, "-o", "%i|%P|%T|%b|%j"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError(f"Unrecognized squeue row: {line!r}")
        job_id, partition, status, tres, name = fields
        jobs[job_id] = dict(partition=partition, status=status, tres=tres,
                            name=name, gpu=("gpu" in tres.lower() or "gpu" in partition.lower()))
    return jobs


def available_gpu_jobs(jobs, own_ids, normal_limit=40, hard_limit=50):
    total = sum(row["gpu"] for row in jobs.values())
    own = sum(row["gpu"] for key, row in jobs.items() if key in own_ids)
    return max(0, min(normal_limit - own, hard_limit - total))


def marker_valid(episode, *, require_visible_frames=False):
    episode = Path(episode)
    try:
        marker = read_json(episode / "validated.json")
        if not marker.get("validated") or marker.get("renderer") != "nyx":
            return False
        if require_visible_frames and (
            marker.get("visible_frames_checked") is not True
            or marker.get("visible_frames_max_rgb_threshold") != 3
            or marker.get("jpeg_validation") != "all"
        ):
            return False
        for name in ("steps.h5", "meta.json", "verdict.json"):
            stat = (episode / name).stat()
            fingerprint = marker["files"][name]
            if fingerprint != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_attempt(item, batch):
    root = Path(batch["root"])
    receipt = root / "receipts" / (item["id"] + ".json")
    output = root / "attempts" / item["id"] / "nyx"
    episode = output / f"seed_{item['seed']}"
    log = Path(batch["log_root"]) / (item["id"] + ".log")
    started = time.time()
    result = dict(id=item["id"], task=item["task"], seed=item["seed"],
                  attempt=item["attempt"], job_id=os.environ.get("SLURM_JOB_ID"),
                  node=os.environ.get("SLURMD_NODENAME"), episode=str(episode),
                  source=item["source"], config=item["config"], started=started,
                  runtime=batch.get("runtime_preflight"), log=str(log))
    proc = None

    def matching_marker():
        if not marker_valid(episode, require_visible_frames=batch.get("require_visible_frames", False)):
            return False
        marker = read_json(episode / "validated.json")
        return (marker.get("task") == item["task"] and marker.get("seed") == item["seed"]
                and marker.get("jpeg_validation") == "all")

    def stop_child():
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        except ProcessLookupError:
            pass

    try:
        if receipt.exists():
            previous = read_json(receipt)
            if (previous.get("id") != item["id"] or previous.get("task") != item["task"]
                    or previous.get("seed") != item["seed"]):
                raise ValueError("Existing receipt identity does not match the attempt")
            if previous.get("status") == "accepted" and not matching_marker():
                raise ValueError("Existing accepted receipt has no valid matching completion marker")
            return previous
        if matching_marker():
            result.update(status="accepted", elapsed=0.0)
            atomic_json(receipt, result)
            return result
        if batch.get("preflight_error"):
            raise RuntimeError(batch["preflight_error"])
        log.parent.mkdir(parents=True, exist_ok=True)
        source = Path(item["source"]).resolve()
        config = Path(item["config"]).resolve()
        result["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        result["collector_sha256"] = hashlib.sha256(
            (source / "scripts/vla_data/collect_vla.py").read_bytes()).hexdigest()
        env = dict(os.environ)
        env.update(GENESIS_BACKEND="gpu", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", PYOPENGL_PLATFORM="egl", MESA_GL_VERSION_OVERRIDE="3.3",
                   PYTHONPATH=str(source))
        threads = str(batch.get("threads_per_process", 1))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[name] = threads
        with tempfile.TemporaryDirectory(prefix="nyx-episode-", dir=os.environ.get("TMPDIR", "/tmp")) as temporary:
            scratch = Path(temporary)
            cwd = scratch / "cwd"
            cwd.mkdir()
            excluded = {"data", "outputs", "output", "job_logs", "logs", "runs",
                        "__nyx_cache__", "__pycache__"}
            for entry in source.iterdir():
                if entry.name not in excluded and not entry.name.startswith("."):
                    (cwd / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
            for name, directory in {
                "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "NUMBA_CACHE_DIR": "numba",
                "XDG_CACHE_HOME": "xdg", "CUDA_CACHE_PATH": "cuda",
                "TI_OFFLINE_CACHE_FILE_PATH": "taichi", "TORCH_EXTENSIONS_DIR": "torch_extensions",
                "TRITON_CACHE_DIR": "triton",
            }.items():
                location = scratch / directory
                location.mkdir(exist_ok=True)
                env[name] = str(location)
            cmd = [batch["python"], str(source / "scripts/vla_data/collect_vla.py"),
                   "--task", item["task"], "--seed", str(item["seed"]), "--mode", "record",
                   "--renderer", "nyx", "--config", str(config), "--exact-config",
                   "--cams", "head_camera,right_wrist", "--out-root", str(output),
                   "--failure-root", str(root / "failed")]
            with log.open("a") as stream:
                stream.write(json.dumps(dict(command=cmd, provenance=result)) + "\n")
                stream.flush()
                proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    try:
                        code = proc.wait(timeout=batch.get("episode_timeout_s", 3600))
                    except subprocess.TimeoutExpired:
                        stop_child()
                        code = 124
                    result["collector_returncode"] = code
                    # Independently decode every saved image before acceptance.
                    if code == 0 and (episode / "meta.json").exists():
                        validation_command = [
                            batch["python"], str(source / "scripts/vla_data/validate_nyx.py"),
                            str(episode), "--task", item["task"], "--seed", str(item["seed"]),
                            "--decode-all", "--write-marker",
                        ]
                        if batch.get("require_visible_frames", False):
                            validation_command.append("--require-visible-frames")
                        validation = subprocess.run(
                            validation_command,
                            cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                            timeout=600, check=False,
                        )
                        code = validation.returncode
                        result["validation_returncode"] = code
                finally:
                    stop_child()
        verdict_path = episode / "verdict.json"
        if not verdict_path.exists():
            verdict_path = root / "failed" / item["id"] / "nyx" / f"seed_{item['seed']}" / "verdict.json"
        verdict = read_json(verdict_path) if verdict_path.exists() else {}
        if code == 0 and matching_marker():
            status = "accepted"
        elif (code != 124 and verdict.get("success") is False
              and not verdict.get("infra_failure_reason") and not verdict.get("error")):
            status = "task_failed"
        else:
            status = "infra_failed"
        result.update(status=status, returncode=code, elapsed=time.time() - started,
                      verdict=str(verdict_path), error=verdict.get("error"),
                      infra_failure_reason=verdict.get("infra_failure_reason"))
    except Exception as exc:
        result.update(status="infra_failed", elapsed=time.time() - started,
                      returncode=124 if isinstance(exc, subprocess.TimeoutExpired) else 1,
                      error=repr(exc), infra_failure_reason="worker_exception")
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        except OSError:
            pass
    finally:
        stop_child()
    atomic_json(receipt, result)
    return result


def worker(batch_path):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run workers through srun inside a GPU allocation")
    batch = read_json(batch_path)
    # Probe the selected interpreter's CUDA device 0. nvidia-smi's first board
    # may be an unallocated device with a different model on a mixed GPU node.
    try:
        packages = {"genesis-world": "1.2.0", "gs-nyx": "0.1.3", "gs-nyx-plugin": "0.1.4"}
        probe_code = (
            "import importlib.metadata as m,json,torch; n=torch.cuda.device_count(); "
            "p=torch.cuda.get_device_properties(0) if n else None; "
            "print(json.dumps({'packages':{k:m.version(k) for k in "
            + repr(list(packages)) + "},'cuda_device_count':n,"
            "'gpu_name':p.name if p else None,'gpu_uuid':str(getattr(p,'uuid','')) if p else None}))"
        )
        probe_env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                         OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        probe_env.pop("PYTHONPATH", None)
        probe = subprocess.run([batch["python"], "-c", probe_code], capture_output=True,
                               text=True, env=probe_env, timeout=90, check=False)
        if probe.returncode:
            raise RuntimeError(f"NYX environment preflight failed: {probe.stderr[-4000:]}")
        runtime = json.loads(probe.stdout.strip().splitlines()[-1])
        if runtime["packages"] != packages or runtime["cuda_device_count"] != 1:
            raise RuntimeError(f"Requires pinned NYX packages and one visible CUDA GPU: {runtime}")
        batch["runtime_preflight"] = runtime
        gpu_name = runtime["gpu_name"]
        slots = 1
        for model, count in batch.get("parallel_by_gpu", {}).items():
            if model.lower() in gpu_name.lower():
                slots = int(count)
        threads = int(batch.get("threads_per_process", 1))
        cpus = len(os.sched_getaffinity(0))
        if threads < 1 or slots < 1 or threads > cpus or int(batch.get("max_processes", 1)) < 1:
            raise ValueError("Invalid process/thread request for available CPU affinity")
        slots = min(slots, int(batch.get("max_processes", 1)), len(batch["items"]), cpus // threads)
    except Exception as exc:
        batch["preflight_error"] = repr(exc)
        results = [run_attempt(item, batch) for item in batch["items"]]
        print(json.dumps(dict(preflight_error=repr(exc), results=results)), flush=True)
        raise
    print(json.dumps(dict(batch=str(batch_path), gpu=gpu_name, runtime=runtime,
                          processes=slots, threads_per_process=threads)), flush=True)
    with ThreadPoolExecutor(max_workers=max(1, slots)) as pool:
        results = list(pool.map(lambda item: run_attempt(item, batch), batch["items"]))
    print(json.dumps(results), flush=True)


def task_counts(state, task):
    rows = [item for item in state["attempts"].values() if item["task"] == task]
    accepted = {item["seed"] for item in rows if item["status"] == "accepted"}
    counts = {name: sum(item["status"] == name for item in rows)
              for name in ("retry", "task_failed", "infra_failed", "exhausted")}
    counts["accepted"] = len(accepted)
    counts["assigned"] = len({item["seed"] for item in rows
                              if item["status"] == "assigned" and item["seed"] not in accepted})
    return counts


def allocate_attempt(state, task, spec, manifest):
    counts = task_counts(state, task)
    target = int(spec.get("target", 300))
    if (spec.get("status") != "ready" or task in state.get("paused", {})
            or counts["accepted"] + counts["assigned"] >= target):
        return None
    # Require a small successful recording pilot before opening the full pool.
    if counts["accepted"] < manifest.get("pilot_successes", 3) and counts["assigned"] >= 2:
        return None
    rows = [row for row in state["attempts"].values() if row["task"] == task]
    occupied = {row["seed"] for row in rows if row["status"] in ("accepted", "assigned")}
    previous = next((row for row in rows
                     if row["status"] == "retry" and row["seed"] not in occupied), None)
    if previous:
        previous["status"] = "infra_failed"
        seed = previous["seed"]
        attempt = max(row["attempt"] for row in rows if row["seed"] == seed) + 1
    else:
        seed = state["next_seed"].get(task, manifest.get("seed_start", 1000000))
        used = {row["seed"] for row in rows}
        while seed in used:
            seed += 1
        state["next_seed"][task] = seed + 1
        attempt = 0
    ident = f"{task}__s{seed}__a{attempt}"
    item = dict(id=ident, task=task, seed=seed, attempt=attempt, status="assigned",
                source=spec["source"], config=spec["config"])
    state["attempts"][ident] = item
    return item


def reconcile(state, manifest, live, publish=True):
    root = Path(manifest["root"])
    now = time.time()
    require_visible_frames = manifest.get("require_visible_frames", False)
    previous_winners = {item["id"] for item in state["attempts"].values()
                        if item["status"] == "accepted"}
    previous_seeds = {(item["task"], item["seed"]) for item in state["attempts"].values()
                      if item["status"] == "accepted"}
    groups = {}

    def matching_output(item):
        episode = item.get("episode", "")
        if not marker_valid(episode, require_visible_frames=require_visible_frames):
            return False
        try:
            marker = read_json(Path(episode) / "validated.json")
        except (OSError, ValueError):
            return False
        return (marker.get("task") == item["task"] and marker.get("seed") == item["seed"]
                and marker.get("jpeg_validation") == "all")

    def suppress(item, status):
        item["superseded_status"] = item["status"]
        item["status"] = status

    for ident, item in state["attempts"].items():
        # Keep the underlying outcome so a valid alternative can replace an
        # invalidated winner, without counting duplicate or surplus episodes.
        if item["status"] in ("superseded", "surplus"):
            item["status"] = item.pop("superseded_status", "infra_failed")
        receipt = root / "receipts" / (ident + ".json")
        if receipt.exists():
            stat = receipt.stat()
            fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if item.get("receipt_fingerprint") != fingerprint:
                try:
                    result = read_json(receipt)
                    if not isinstance(result, dict):
                        raise ValueError("Receipt must be an object")
                    identity = ("id", "task", "seed", "attempt", "source", "config")
                    if any(result.get(key) != item[key] for key in identity if key in item):
                        raise ValueError("Receipt identity does not match the assigned attempt")
                    if result.get("status") not in ("accepted", "task_failed", "infra_failed"):
                        raise ValueError("Receipt has an invalid outcome")
                    item.update(result)
                    item.pop("missing_since", None)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    item.update(status="infra_failed", error=f"Invalid receipt: {exc}")
                item["receipt_fingerprint"] = fingerprint
        if item["status"] == "accepted" and not matching_output(item):
            item.update(status="infra_failed",
                        error="Accepted output has no valid matching completion marker")
        if item["status"] == "assigned":
            if item.get("job_id") in live:
                item.pop("missing_since", None)
            else:
                # Queue/receipt visibility can lag completion. An old submitted
                # timestamp must not bypass the first-observed-absence grace.
                missing_since = item.setdefault("missing_since", now)
                if now - missing_since >= manifest.get("missing_receipt_grace_s", 180):
                    item.update(status="infra_failed", error="Job disappeared without a receipt")
        groups.setdefault((item["task"], item["seed"]), []).append(item)

    winners = {}
    for key, rows in groups.items():
        accepted = [row for row in rows if row["status"] == "accepted"]
        if accepted:
            # Preserve an already published choice. Otherwise break ties by
            # attempt number/id, independently of dict or receipt scan order.
            winner = min(accepted, key=lambda row: (
                row["id"] not in previous_winners, row["attempt"], row["id"]))
            winners[key] = winner
            for row in rows:
                if row is not winner:
                    suppress(row, "superseded")
        else:
            latest = max(rows, key=lambda row: (row["attempt"], row["id"]))
            for row in rows:
                if row is not latest:
                    suppress(row, "superseded")
            if latest["status"] == "infra_failed":
                latest["status"] = ("retry" if latest["attempt"] < manifest.get("max_retries", 2)
                                    else "exhausted")

    for task, spec in manifest["tasks"].items():
        candidates = sorted((key for key in winners if key[0] == task), key=lambda key: (
            key not in previous_seeds, key[1], winners[key]["id"]))
        for key in candidates[int(spec.get("target", 300)):]:
            suppress(winners.pop(key), "surplus")

    if publish:
        for (task, seed), rows in groups.items():
            destination = root / "accepted" / task / "nyx" / f"seed_{seed}"
            winner = winners.get((task, seed))
            if winner:
                episode = Path(winner["episode"]).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink() and destination.resolve() != episode:
                    temporary_link = destination.with_name(destination.name + ".new")
                    temporary_link.unlink(missing_ok=True)
                    temporary_link.symlink_to(episode, target_is_directory=True)
                    os.replace(temporary_link, destination)
                elif not destination.is_symlink() and not destination.exists():
                    destination.symlink_to(episode, target_is_directory=True)
            elif destination.is_symlink() and destination.resolve() in {
                Path(row["episode"]).resolve() for row in rows if row.get("episode")
            }:
                destination.unlink()
    for task in manifest["tasks"]:
        # Admission counts unique seeds, but the failure guard still counts real
        # completed attempts, including failures preceding a retry. Duplicate
        # accepted outcomes must not dilute the infrastructure-failure streak.
        rows = []
        for row in state["attempts"].values():
            if row["task"] != task:
                continue
            outcome = row.get("superseded_status", row["status"])
            if outcome in ("assigned", "retry"):
                continue
            if outcome == "accepted" and row["status"] != "accepted":
                continue
            rows.append(dict(row, status=outcome))
        successes = sum(row["status"] == "accepted" for row in rows)
        reason = None
        if len(rows) >= 20 and successes == 0:
            reason = "No accepted episodes in the initial 20 completed attempts; inspect pilot failures"
        elif len(rows) >= 12 and all(row["status"] in ("infra_failed", "exhausted") for row in rows[-12:]):
            reason = "12 consecutive infrastructure failures; inspect environment/node failures"
        elif len({row["seed"] for row in rows}) >= manifest.get("max_seeds_per_task", 1500):
            reason = "Seed budget reached; inspect yield before increasing the budget"
        if reason:
            state.setdefault("paused", {})[task] = reason


def write_summary(state, manifest):
    summary = dict(updated=time.time(), campaign=manifest["campaign"],
                   tasks={task: task_counts(state, task) for task in manifest["tasks"]})
    summary["accepted_total"] = sum(counts["accepted"] for counts in summary["tasks"].values())
    summary["paused"] = state.get("paused", {})
    atomic_json(Path(manifest["root"]) / "status.json", summary)
    return summary


def parse_submission_id(stdout):
    matches = [match.group(1) for line in stdout.splitlines()
               if (match := re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", line.strip()))]
    if len(matches) != 1:
        raise ValueError(f"Unexpected submission response: {stdout!r}")
    return matches[0]


def validated_submission_profiles(manifest):
    """Resolve the optional CPU/constraint profiles before any submission intent."""
    if "submission_profiles" not in manifest:
        return None
    profiles = manifest["submission_profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("submission_profiles must be a nonempty list")
    resolved = []
    for index, profile in enumerate(profiles):
        prefix = f"submission_profiles[{index}]"
        if not isinstance(profile, dict):
            raise ValueError(f"{prefix} must be an object")
        unknown = set(profile) - {"name", "cpus_per_job", "constraint"}
        if unknown:
            raise ValueError(f"{prefix} has unsupported overrides: {sorted(unknown)}")
        options = {
            "cpus_per_job": profile.get("cpus_per_job", manifest.get("cpus_per_job", 4)),
            "constraint": profile.get("constraint", manifest["constraint"]),
        }
        cpus = options["cpus_per_job"]
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            raise ValueError(f"{prefix}.cpus_per_job must be a positive integer")
        constraint = options["constraint"]
        if not isinstance(constraint, str) or not constraint.strip():
            raise ValueError(f"{prefix}.constraint must be a nonempty string")
        if "name" in profile:
            if not isinstance(profile["name"], str) or not profile["name"].strip():
                raise ValueError(f"{prefix}.name must be a nonempty string")
            options["name"] = profile["name"]
        resolved.append(options)
    return resolved


def dispatch(manifest_path, once=False, dry_run=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    root = Path(manifest["root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "dispatcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "state.json"
    state = read_json(state_path) if state_path.exists() else dict(attempts={}, next_seed={}, batches={}, sequence=0)
    while True:
        manifest = read_json(manifest_path)
        profiles = validated_submission_profiles(manifest)
        try:
            live = queue_jobs(manifest["user"])
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            print(f"Queue unavailable; submitting nothing: {exc}", flush=True)
            if once:
                return 2
            time.sleep(30)
            continue
        # Recover submissions made just before a coordinator interruption.
        names = {row["name"]: job_id for job_id, row in live.items()}
        for batch in state["batches"].values():
            if batch.get("job_id") is None and batch["name"] in names:
                batch["job_id"] = names[batch["name"]]
                for ident in batch["attempts"]:
                    state["attempts"][ident]["job_id"] = batch["job_id"]
        reconcile(state, manifest, live, publish=not dry_run)
        own_ids = {batch["job_id"] for batch in state["batches"].values() if batch.get("job_id")}
        room = available_gpu_jobs(live, own_ids, manifest.get("max_gpu_jobs", 40), 50)
        for _ in range(min(room, manifest.get("submissions_per_poll", 4))):
            tasks = sorted(manifest["tasks"], key=lambda task: (
                task_counts(state, task)["assigned"], task_counts(state, task)["accepted"], task))
            items = []
            for _round in range(manifest.get("attempts_per_job", 4)):
                for task in tasks:
                    item = allocate_attempt(state, task, manifest["tasks"][task], manifest)
                    if item:
                        items.append(item)
                        break
                if not item:
                    break
            if not items:
                break
            state["sequence"] += 1
            name = f"{manifest['job_prefix']}_{state['sequence']:06d}"
            batch_path = root / "batches" / (name + ".json")
            batch = {key: manifest[key] for key in
                     ("root", "log_root", "python", "parallel_by_gpu", "max_processes")}
            batch.update(items=items, threads_per_process=manifest.get("threads_per_process", 2),
                         episode_timeout_s=manifest.get("episode_timeout_s", 3600),
                         require_visible_frames=manifest.get("require_visible_frames", False))
            resources = dict(cpus_per_job=manifest.get("cpus_per_job", 4),
                             constraint=manifest["constraint"])
            if profiles is not None:
                profile_index = (state["sequence"] - 1) % len(profiles)
                resources = profiles[profile_index]
                batch["submission_profile"] = dict(index=profile_index, **resources)
            if not dry_run:
                atomic_json(batch_path, batch)
            state["batches"][name] = dict(name=name, job_id=None, attempts=[item["id"] for item in items])
            for item in items:
                item["submitted"] = time.time()
                item["batch"] = name
            if not dry_run:
                atomic_json(state_path, state)
            command = ["sbatch", "--parsable", "--partition=gpu,gpu-preempt", "--gres=gpu:1",
                       f"--cpus-per-task={resources['cpus_per_job']}",
                       f"--mem={manifest.get('memory_per_job', '48G')}",
                       f"--time={manifest.get('time_per_job', '03:00:00')}",
                       "--job-name=" + name, "--constraint=" + resources["constraint"],
                       "--exclude=" + manifest["exclude"],
                       "--output=" + str(Path(manifest["log_root"]) / (name + "_%j.log")),
                       "--error=" + str(Path(manifest["log_root"]) / (name + "_%j.log")),
                       "--wrap=" + shlex.join(["srun", manifest["python"], manifest["utility"], "worker", str(batch_path)])]
            if dry_run:
                print(shlex.join(command), flush=True)
                return 0
            try:
                # Recheck immediately before sbatch, allowing for other submitters.
                fresh = queue_jobs(manifest["user"])
                if sum(row["gpu"] for row in fresh.values()) >= 50:
                    print("GPU job cap reached; deferring the saved submission intent", flush=True)
                    break
                submitted = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
                if submitted.stderr:
                    print(submitted.stderr.rstrip(), flush=True)
                job_id = parse_submission_id(submitted.stdout)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                # Keep the saved intent: next queue scan recovers an ambiguous submission.
                print(f"Submission unresolved; stopping this dispatch wave: {exc}", flush=True)
                break
            state["batches"][name]["job_id"] = job_id
            for item in items:
                item["job_id"] = job_id
            atomic_json(state_path, state)
            print(f"Submitted {job_id}: {len(items)} attempts", flush=True)
        if dry_run:
            print("Dry run: no eligible submissions", flush=True)
            return 0
        summary = write_summary(state, manifest)
        atomic_json(state_path, state)
        print(json.dumps(summary), flush=True)
        if all(summary["tasks"][task]["accepted"] >= spec.get("target", 300)
               for task, spec in manifest["tasks"].items() if spec.get("status") == "ready"):
            atomic_json(root / "complete.json", summary)
            accepted = [row for row in state["attempts"].values() if row["status"] == "accepted"]
            atomic_json(root / "dataset_manifest.json", dict(
                schema="genesis_hr_bench.nyx_collection.v1", campaign=manifest["campaign"],
                tasks=manifest["tasks"], episodes=accepted))
            return 0
        if once:
            return 0
        time.sleep(manifest.get("poll_seconds", 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coordinator = commands.add_parser("dispatch")
    coordinator.add_argument("manifest")
    coordinator.add_argument("--once", action="store_true")
    coordinator.add_argument("--dry-run", action="store_true")
    executor = commands.add_parser("worker")
    executor.add_argument("batch")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.batch)
        return 0
    return dispatch(args.manifest, args.once, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
