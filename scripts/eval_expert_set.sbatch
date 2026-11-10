#!/bin/bash
# Optional SLURM wrapper for scripts/eval_expert_set.py (one shard per job).
# The local equivalent is
#   source scripts/env.sh && python scripts/eval_expert_set.py <args>
# Partition and GPU constraint are cluster-specific and are not set here; pass
# them on the command line (see docs/cluster.md), e.g.
#   sbatch -p <partition> --constraint=<nyx-gpu-constraint> \
#     scripts/eval_expert_set.sbatch --eval-set eval_sets/eval_v3_sampled \
#     --policy expert_full_state --start-index 0 --end-index 500
# eval_v3_sampled rows render with NYX, which needs GENESIS_BACKEND=gpu (the
# default here) on a Vulkan-capable GPU. For a CPU-only rasterizer run, submit
# with GENESIS_BACKEND=cpu and pass --renderer rasterizer.
#SBATCH --job-name=expert_eval_set
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o job_logs/expert_eval_set_%j.log
#SBATCH -e job_logs/expert_eval_set_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p job_logs

REQUESTED_GENESIS_BACKEND="${GENESIS_BACKEND:-gpu}"
# shellcheck source=env.sh
source scripts/env.sh

export SETUPTOOLS_USE_DISTUTILS=stdlib
export GENESIS_BACKEND="$REQUESTED_GENESIS_BACKEND"

echo "=== expert eval set job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "args: $*"
echo "python=${MAWM_PY}"
echo "GENESIS_BACKEND=${GENESIS_BACKEND}"

srun "${MAWM_PY}" scripts/eval_expert_set.py "$@"
