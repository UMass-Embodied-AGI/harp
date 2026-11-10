#!/bin/bash
# SLURM wrapper for LeRobot ACT finetuning on genesis-hr-bench.
#
# Submit with:
#   sbatch --time=1:00:00  scripts/finetune_act.sbatch --sanity
#   sbatch --time=8:00:00  scripts/finetune_act.sbatch --full
#   sbatch --time=12:00:00 scripts/finetune_act.sbatch --full -- --batch_size=64
#
# The local equivalent is scripts/finetune_act.sh. Partition and GPU
# constraint are cluster-specific: add `-p <partition>` / `--constraint=...`
# to sbatch (see docs/cluster.md).
#
# Notes:
#   * Reuses the pi0 venv (lerobot already installed). No bootstrap needed.
#   * ACT is small (~50M params); any recent >=24 GB GPU is comfortable at
#     batch=32 (training does not render, so compute-only GPUs work). A
#     single-GPU full run takes ~5h for 100k steps on an A100.
#   * Preempt-resume: $SLURM_JOB_ID is stable across requeues, so the
#     output_dir is shared and lerobot's --resume picks up the last ckpt.

#SBATCH --job-name=act_finetune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=job_logs/finetune_act_%j.log
#SBATCH --error=job_logs/finetune_act_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
cd "$REPO_ROOT"
# shellcheck source=env.sh
source scripts/env.sh
mkdir -p job_logs runs/act/finetune

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

exec srun scripts/finetune_act.sh "$@"
