#!/usr/bin/env python3
"""Build run_batch.py batch files from eval-set rows.

  make_batches.py --rows <jsonl> --run-dir data/vla_train37_eval/<run> \
      --models act,diffusion,pi0,pi05,smolvla [--select task:seed,...] \
      [--per-batch 5] [--video]

Rows use the eval-set schema (task, seed, config_overrides, action_substeps).
max_steps per task = ceil(MAX_STEPS_FACTOR x p95 training-demo length), from
the dataset manifest (policy steps at capture_stride 25), unless --max-steps.
Writes <run-dir>/batches/<model>/b_<i>.json and prints the array sizes.
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _legacy_task_names():
    """Pre-HARP task name -> current name, from envs/tasks/paper_labels.py."""
    import importlib.util
    path = HERE.parents[1] / "envs" / "tasks" / "paper_labels.py"
    spec = importlib.util.spec_from_file_location("harp_paper_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LEGACY_TASK_NAMES


DEMO_STEPS_P95 = json.loads((HERE / "train_demo_steps.json").read_text())["p95"]
# Old runs and manifests may still carry pre-HARP task names.
DEMO_STEPS_P95.update({old: DEMO_STEPS_P95[new] for old, new in _legacy_task_names().items() if new in DEMO_STEPS_P95})
MAX_STEPS_FACTOR = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default="act,diffusion,pi0,pi05,smolvla")
    ap.add_argument("--select", default=None, help="comma list of task:seed to keep")
    ap.add_argument("--per-batch", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent episodes per GPU job (one policy server each)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.rows) if line.strip()]
    if args.select:
        keep = {(t, int(s)) for t, s in (x.split(":") for x in args.select.split(","))}
        rows = [r for r in rows if (r["task"], int(r["seed"])) in keep]
        missing = keep - {(r["task"], int(r["seed"])) for r in rows}
        if missing:
            raise SystemExit(f"rows not found: {sorted(missing)}")
    run_dir = Path(args.run_dir).resolve()
    for model in args.models.split(","):
        bdir = run_dir / "batches" / model
        bdir.mkdir(parents=True, exist_ok=True)
        items = []
        for r in rows:
            ms = args.max_steps or math.ceil(MAX_STEPS_FACTOR * DEMO_STEPS_P95[r["task"]])
            items.append({
                "task": r["task"], "seed": int(r["seed"]),
                "config_overrides": r.get("config_overrides", {}),
                "action_substeps": int(r.get("action_substeps", 25)),
                "renderer": r.get("renderer", "nyx"),
                "max_steps": ms, "eval_row_id": r.get("id"),
                # reset/build + ~2.5 s per policy step (2 slots/GPU), never below --timeout-s
                "timeout_s": max(args.timeout_s, 900 + 2.5 * ms),
                "out": str(run_dir / "episodes" / model / r["task"] / f"seed_{r['seed']}"),
            })
        # Round-robin across tasks so every batch mixes short and long tasks
        # (keeps per-job wall time even; output paths are unchanged).
        by_task = {}
        for it in items:
            by_task.setdefault(it["task"], []).append(it)
        queues = [by_task[t] for t in sorted(by_task)]
        items = []
        while any(queues):
            for q in queues:
                if q:
                    items.append(q.pop(0))
        n = 0
        for i in range(0, len(items), args.per_batch):
            batch = {"model": model, "server_device": "cuda", "timeout_s": args.timeout_s,
                     "video": bool(args.video), "video_stride": 25,
                     "parallel": args.parallel,
                     "items": items[i:i + args.per_batch]}
            (bdir / f"b_{n}.json").write_text(json.dumps(batch, indent=1))
            n += 1
        print(f"{model}: {len(items)} episodes in {n} batches -> --array=0-{n - 1}")


if __name__ == "__main__":
    main()
