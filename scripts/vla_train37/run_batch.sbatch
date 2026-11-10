#!/bin/bash
# Optional SLURM wrapper around run_batch.py: one array task per batch file.
# Locally, run `"$NYX_PY" scripts/vla_train37/run_batch.py <batch_dir>/b_<i>.json`
# instead (see docs/vla-inference-control-pipeline.md).
#
# Partition, GPU constraint and node exclusions are cluster-specific and are
# not set here: pass them on the command line or let launch_full.sh/submit.sh
# add them from scripts/env.local.sh (see docs/cluster.md). NYX needs a
# Vulkan-capable GPU; compute-only A100/H100/H200 cannot run it, and pi0/pi05
# need >=16 GB (run_batch.py refuses smaller GPUs).
#
# Usage: sbatch --array=0-N%C --job-name=t37_<run>_<model> \
#          [-p <partition>] [--constraint=<gpu-constraint>] \
#          scripts/vla_train37/run_batch.sbatch <batch_dir>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=job_logs/train37/%x_%A_%a.log
#SBATCH --error=job_logs/train37/%x_%A_%a.log
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# Interpreters come from the environment contract (scripts/env.sh +
# gitignored scripts/env.local.sh): NYX_PY (NYX-enabled Genesis python;
# defaults to MAWM_PY) and the policy-server python, T37_LEROBOT_PY if set
# (a host override that does not change env.sh's baseline-root derivation),
# else LEROBOT_PY.
# shellcheck source=../env.sh
source scripts/env.sh
set -u
export LEROBOT_PY="${T37_LEROBOT_PY:-$LEROBOT_PY}"
NYX_PY="${NYX_PY:-$MAWM_PY}"
echo "host=$(hostname) date=$(date -u +%FT%TZ) job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "commit=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l)"
"$NYX_PY" --version
"$NYX_PY" scripts/vla_train37/run_batch.py "$1/b_${SLURM_ARRAY_TASK_ID:-0}.json"
date -u +%FT%TZ
