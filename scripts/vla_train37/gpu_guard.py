#!/usr/bin/env python3
"""GPU sizing for train37 VLA eval jobs (NYX + a LeRobot policy server).

pi0 and pi05 need ~10-11 GB of GPU memory per policy-server slot on top of
NYX's Vulkan buffers, so they must not run on an 11 GB card (e.g. an RTX
2080 Ti): there the server or NYX runs out of memory.  A 16 GB card fits one
slot; 48 GB cards fit several.

Two layers use this module:

* runtime (``run_batch.py``, local or under SLURM): ``slots_for_gpu`` refuses
  a large model on a GPU under ``LARGE_MIN_MIB`` and caps the parallel slots
  to what the GPU holds, so a large model on a small GPU fails fast with a
  clear message instead of an NYX/CUDA OOM;
* optional SLURM submission (``submit.sh``, ``launch_full.sh``):
  ``constraint <model> [parallel]`` prints the ``--constraint`` to submit
  with.  GPU feature names are cluster-specific, so the constraints come from
  environment variables (set them in ``scripts/env.local.sh``; see
  ``docs/cluster.md``).  Unset or empty means "no --constraint":

  - ``NYX_GPU_CONSTRAINT``: Vulkan-capable GPUs that run NYX (compute-only
    A100/H100/H200 cannot);
  - ``NYX_LARGE_GPU_CONSTRAINT``: the >=16 GB subset, for pi0/pi05
    (default: ``NYX_GPU_CONSTRAINT``);
  - ``NYX_LARGE_MULTI_GPU_CONSTRAINT``: the ~48 GB subset, for pi0/pi05 jobs
    with >=2 slots (default: ``NYX_LARGE_GPU_CONSTRAINT``);
  - ``NYX_SMALL_GPU_FEATURES``: ``|``-separated feature names of <16 GB GPUs;
    ``check`` refuses a pi0/pi05 constraint that names one.

CLI::

    gpu_guard.py constraint <model> [parallel]
    gpu_guard.py model <batch_dir>        # model of <batch_dir>/b_0.json
    gpu_guard.py parallel <batch_dir>     # slots per job of <batch_dir>/b_0.json
    gpu_guard.py check <model> <constraint>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

NYX_CONSTRAINT = os.environ.get("NYX_GPU_CONSTRAINT", "")
LARGE_CONSTRAINT = os.environ.get("NYX_LARGE_GPU_CONSTRAINT", "") or NYX_CONSTRAINT
LARGE_MULTI_CONSTRAINT = (
    os.environ.get("NYX_LARGE_MULTI_GPU_CONSTRAINT", "") or LARGE_CONSTRAINT)
SMALL_GPU_FEATURES = frozenset(
    f.strip() for f in os.environ.get("NYX_SMALL_GPU_FEATURES", "").split("|") if f.strip())
LARGE_MODELS = frozenset({"pi0", "pi05"})
LARGE_MIN_MIB = 15000       # a 16 GB card reports ~15356 MiB; an 11 GB card ~11264 MiB
LARGE_SLOT_MIB = 11000      # one pi0 server + one NYX episode (~10.0 GB measured)


def is_large(model: str) -> bool:
    return str(model) in LARGE_MODELS


def constraint_for(model: str, parallel: int = 1) -> str:
    if not is_large(model):
        return NYX_CONSTRAINT
    return LARGE_MULTI_CONSTRAINT if int(parallel) >= 2 else LARGE_CONSTRAINT


def constraint_ok(model: str, constraint: str) -> bool:
    """A user-given constraint is acceptable unless it admits a GPU listed in
    ``NYX_SMALL_GPU_FEATURES`` for a large model."""
    if not is_large(model):
        return True
    features = {f.strip() for f in str(constraint).replace("&", "|").split("|")}
    return not (features & SMALL_GPU_FEATURES)


def gpu_total_mib() -> int | None:
    """Smallest total memory over the visible GPUs (MiB); None if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout
        vals = [int(float(x)) for x in out.split() if x.strip()]
    except Exception:
        return None
    return min(vals) if vals else None


def slots_for_gpu(model: str, wanted: int, total_mib: int | None) -> int:
    """Parallel slots this GPU can host for ``model``; raises SystemExit when
    a large model is on a GPU smaller than ``LARGE_MIN_MIB``."""
    wanted = max(1, int(wanted))
    if not is_large(model) or total_mib is None:
        return wanted
    if total_mib < LARGE_MIN_MIB:
        hint = constraint_for(model, wanted)
        raise SystemExit(
            f"[gpu_guard] {model} needs a >=16 GB GPU, got {total_mib} MiB. "
            "Run it on a larger GPU"
            + (f" (e.g. submit with --constraint='{hint}')" if hint else "") + ".")
    return max(1, min(wanted, total_mib // LARGE_SLOT_MIB))


def batch_model(batch_dir: str | Path) -> str:
    return json.loads((Path(batch_dir) / "b_0.json").read_text())["model"]


def batch_parallel(batch_dir: str | Path) -> int:
    return int(json.loads((Path(batch_dir) / "b_0.json").read_text()).get("parallel", 1))


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "constraint":
        print(constraint_for(argv[1], int(argv[2]) if len(argv) > 2 else 1))
        return 0
    if len(argv) == 2 and argv[0] == "model":
        print(batch_model(argv[1]))
        return 0
    if len(argv) == 2 and argv[0] == "parallel":
        print(batch_parallel(argv[1]))
        return 0
    if len(argv) == 3 and argv[0] == "check":
        return 0 if constraint_ok(argv[1], argv[2]) else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
