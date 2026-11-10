#!/usr/bin/env python3
"""Run one batch of train37 VLA eval episodes on one GPU (locally or in SLURM).

One LeRobot policy server per batch (pi0/pi05 take ~3 min to load), and one
fresh ``eval_vla.py`` process per episode because NYX can build only one
scene per process. Each episode gets the eval-set row's ``config_overrides``
verbatim (the VLA NYX collection config), the task's exact training
instruction, ``action_type=qpos_abs`` and the row's ``action_substeps``.

Batch JSON::

  {"model": "act", "server_device": "cuda", "parallel": 2,
   "timeout_s": 3600, "video": true,
   "items": [{"task": ..., "seed": ..., "config_overrides": {...},
              "action_substeps": 25, "max_steps": 600, "out": "<dir>"}]}

Per episode ``<out>/`` holds result.json (eval_vla --output-json), config.yml,
run_meta.json, episode.log and the video. Finished episodes are skipped.

Usage: <nyx python> run_batch.py <batch.json>

Runs on any machine with one NYX-capable (Vulkan) GPU; under SLURM,
run_batch.sbatch calls it once per array task. Build batch files with
make_batches.py or launch_full.sh.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import gpu_total_mib, slots_for_gpu  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("TRAIN37_REPO", HERE.parents[1]))
# LeRobot server interpreter: from the environment (run_batch*.sbatch export
# T37_LEROBOT_PY or env.sh's LEROBOT_PY; see scripts/env.local.sh.example).
LEROBOT_PY = os.environ.get("LEROBOT_PY", "")
CKPT_ROOT = Path(os.environ.get("TRAIN37_CKPT_ROOT", REPO / "checkpoints" / "train37_eval"))
TMP_ROOT = Path(os.environ.get("TRAIN37_TMP_ROOT", REPO / "data" / "tmp" / "vla_train37"))
LOG_ROOT = Path(os.environ.get("TRAIN37_LOG_ROOT", REPO / "job_logs" / "train37"))

# eval_vla client name and lerobot_server --policy-type per checkpoint.
MODELS = {
    "act": ("act", "act"),
    "diffusion": ("lerobot_diffusion", "diffusion"),
    "pi0": ("pi0", "pi0"),
    "pi05": ("pi05", "pi05"),
    "smolvla": ("smolvla", "smolvla"),
}


def _legacy_task_names():
    """Pre-HARP task name -> current name, from envs/tasks/paper_labels.py."""
    import importlib.util
    path = HERE.parents[1] / "envs" / "tasks" / "paper_labels.py"
    spec = importlib.util.spec_from_file_location("harp_paper_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LEGACY_TASK_NAMES


INSTRUCTIONS = json.loads((HERE / "train_instructions.json").read_text())["instructions"]
# Old runs and manifests may still carry pre-HARP task names.
INSTRUCTIONS.update({old: INSTRUCTIONS[new] for old, new in _legacy_task_names().items() if new in INSTRUCTIONS})
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def done(out: Path) -> bool:
    res = out / "result.json"
    if not res.is_file():
        return False
    try:
        return bool(json.loads(res.read_text()).get("episode_results"))
    except Exception:
        return False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def health(port, timeout=5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def start_server(model, port, device, job):
    _, policy_type = MODELS[model]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"server_{model}_{job}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [LEROBOT_PY, str(REPO / "baseline/servers/lerobot_server.py"),
           "--policy-type", policy_type, "--checkpoint", str(CKPT_ROOT / model),
           "--port", str(port), "--device", device]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}; see {log_path}")
        h = health(port)
        if h and h.get("ok"):
            print(f"[batch] server up in {time.time() - t0:.0f}s: {h}", flush=True)
            return proc, h, str(log_path)
        time.sleep(5)
    os.killpg(proc.pid, signal.SIGKILL)
    raise RuntimeError(f"server not healthy after 900s; see {log_path}")


def episode_config(item, batch):
    cfg = dict(item.get("config_overrides") or {})
    cfg["action_substeps"] = int(item.get("action_substeps", 25))
    cfg["renderer"] = item.get("renderer", cfg.get("renderer", "nyx"))
    # Training collection used random_object=false for every task; eval_vla
    # would otherwise default it to True for deliver_to_human_easy.
    cfg.setdefault("random_object", False)
    # eval_vla defaults planner_override=genesis_ik for EE-delta servoing.
    # qpos_abs actions drive joint PD targets directly (no IK), and several
    # task resets (take_from_human_easy, oil, microwave, ...) need the
    # default mplib planner the training data was collected with.
    cfg.setdefault("planner_override", None)
    # Robot-avatar distance/contact check every 5 sim steps (task default 30)
    # for metric-quality safety numbers. It also feeds the collision veto on
    # success, so success is slightly stricter than a stride-30 expert run.
    cfg["collision_check_stride"] = int(item.get("collision_check_stride",
                                             batch.get("collision_check_stride", 5)))
    cfg.update(item.get("extra_config") or {})
    return cfg


def run_episode(item, batch, server_url, job):
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    client, _ = MODELS[batch["model"]]
    cfg = episode_config(item, batch)
    want_video = bool(batch.get("video", False))
    if want_video:
        cfg["disable_video"] = False
        cfg["save_video"] = True
    (out / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    vla_cfg = {"vla": {"image_pipeline": batch.get("image_pipeline", "jpeg92_bilinear"),
                       "primary_camera": "head_camera", "wrist_camera": "right_wrist"}}
    (out / "vla.yml").write_text(yaml.safe_dump(vla_cfg, sort_keys=True))

    scratch = TMP_ROOT / f"{job}_{item['task']}_{item['seed']}_{os.getpid()}"
    shutil.rmtree(scratch, ignore_errors=True)
    env = dict(os.environ, GENESIS_BACKEND="gpu", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYOPENGL_PLATFORM="egl",
               MESA_GL_VERSION_OVERRIDE="3.3",
               PYTHONPATH=os.pathsep.join(
                   [str(REPO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    for name in THREAD_VARS:
        env[name] = "1"
    for name, sub in {"TMPDIR": "tmp", "NUMBA_CACHE_DIR": "numba", "XDG_CACHE_HOME": "xdg",
                      "CUDA_CACHE_PATH": "cuda", "TI_OFFLINE_CACHE_FILE_PATH": "taichi",
                      "TRITON_CACHE_DIR": "triton"}.items():
        (scratch / sub).mkdir(parents=True, exist_ok=True)
        env[name] = str(scratch / sub)

    cmd = [sys.executable, str(REPO / "scripts/eval_vla.py"),
           "--task", item["task"], "--model", client,
           "--server-url", server_url, "--action-type", "qpos_abs",
           "--episodes", "1", "--start-seed", str(item["seed"]),
           "--max-steps", str(item["max_steps"]),
           "--instruction", INSTRUCTIONS[item["task"]],
           "--config", str(out / "config.yml"), "--vla-config", str(out / "vla.yml"),
           "--output-json", str(out / "result.json")]
    if want_video:
        # video_stride counts SIM steps; 25 = one frame per policy step (20 fps).
        # Stride 5 buffered ~17 GB of frames on 3.7k-step episodes (OOM).
        cmd += ["--video-dir", str(out), "--video-stride", str(batch.get("video_stride", 25))]
    cmd += list(batch.get("eval_args", []))
    meta = dict(item, model=batch["model"], host=socket.gethostname(), slurm_job=job,
                command=cmd, started=time.time())
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    with open(out / "episode.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=float(item.get("timeout_s", batch.get("timeout_s", 3600))))
            meta["timeout"] = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
            meta["timeout"] = True
    meta.update(returncode=rc, finished=time.time(), elapsed=time.time() - meta["started"])
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(scratch, ignore_errors=True)
    return rc, meta


def _slot_worker(slot, queue, batch, job, lock):
    """One policy server per slot: the server keeps per-episode action-chunk
    state, so concurrent episodes must not share one."""
    port = free_port()
    server, _, server_log = start_server(batch["model"], port,
                                         batch.get("server_device", "cuda"), f"{job}_s{slot}")
    try:
        while True:
            with lock:
                if not queue:
                    return
                item = queue.pop(0)
            if server.poll() is not None:
                raise RuntimeError(f"server died rc={server.returncode}; see {server_log}")
            print(f"[batch] s{slot} start {item['task']} seed={item['seed']}", flush=True)
            rc, meta = run_episode(item, batch, f"http://127.0.0.1:{port}", f"{job}_s{slot}")
            if rc != 0 and not meta["timeout"] and meta["elapsed"] < 300 and not done(Path(item["out"])):
                # Transient NYX init crashes (e.g. "Invalid shader uuid") die within
                # ~1 min; retry once in a fresh process.
                print(f"[batch] retry {item['task']} seed={item['seed']} after rc={rc} "
                      f"{meta['elapsed']:.0f}s", flush=True)
                rc, meta = run_episode(item, batch, f"http://127.0.0.1:{port}", f"{job}_s{slot}r")
            res = Path(item["out"]) / "result.json"
            succ = None
            if res.is_file():
                eps = json.loads(res.read_text()).get("episode_results") or [{}]
                succ = eps[0].get("success")
            print(f"[batch] end {item['task']} seed={item['seed']} rc={rc} success={succ} "
                  f"timeout={meta['timeout']} {meta['elapsed']:.0f}s", flush=True)
    finally:
        os.killpg(server.pid, signal.SIGTERM)


def _gpu_mem_logger(stop):
    """Print GPU memory use once a minute (sizing parallel slots)."""
    while not stop.wait(60):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=30).stdout.strip()
            print(f"[gpu] {out}", flush=True)
        except Exception:
            pass


def main():
    import threading
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_batch.py <batch.json>")
    batch = json.loads(Path(sys.argv[1]).read_text())
    # Job tag for log and scratch names: the SLURM job when there is one.
    job = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        job = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID')}"
    if not LEROBOT_PY or not Path(LEROBOT_PY).is_file():
        raise SystemExit(f"LEROBOT_PY={LEROBOT_PY!r} is not an interpreter; set T37_LEROBOT_PY "
                         "(or LEROBOT_PY) in scripts/env.local.sh")
    pending = [it for it in batch["items"] if not done(Path(it["out"]))]
    slots = max(1, min(int(batch.get("parallel", 1)), len(pending) or 1))
    # pi0/pi05 never run on an 11 GB GPU, and get no more slots than fit.
    total_mib = gpu_total_mib()
    fit = slots_for_gpu(batch["model"], slots, total_mib)
    if fit < slots:
        print(f"[batch] {batch['model']}: {total_mib} MiB GPU fits {fit} of {slots} slots",
              flush=True)
        slots = fit
    print(f"[batch] host={socket.gethostname()} job={job} model={batch['model']} "
          f"pending={len(pending)}/{len(batch['items'])} slots={slots}", flush=True)
    if not pending:
        return
    stop = threading.Event()
    threading.Thread(target=_gpu_mem_logger, args=(stop,), daemon=True).start()
    lock = threading.Lock()
    errors = []

    def run_slot(slot):
        try:
            _slot_worker(slot, pending, batch, job, lock)
        except Exception as exc:  # noqa: BLE001 - report and let other slots finish
            errors.append(f"slot {slot}: {exc}")
            print(f"[batch] slot {slot} FAILED: {exc}", flush=True)

    threads = [threading.Thread(target=run_slot, args=(k,)) for k in range(slots)]
    for t in threads:
        t.start()
        time.sleep(20)  # stagger server loads / NYX inits
    for t in threads:
        t.join()
    stop.set()
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
