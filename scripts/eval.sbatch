#!/bin/bash
# Generic SLURM wrapper for VLA / imitation-policy evaluation through Genesis.
# Spawns the matching baseline server (per scripts/run_vla_eval.sh's dispatch
# table) and runs scripts/eval_vla.py from the MAWM env against it.
#
# The local equivalent is scripts/run_vla_eval.sh. Partition and GPU
# constraint are cluster-specific and not set here: add `-p <partition>
# --constraint=<nyx-gpu-constraint>` to sbatch (see docs/cluster.md). NYX
# rendering needs a Vulkan-capable GPU (not compute-only A100/H100/H200).
#
# Usage:
#   sbatch [-p <partition>] [--constraint=<gpu-constraint>] scripts/eval.sbatch <model> [<checkpoint>] [extra run_vla_eval.sh args...]
#
# Models (matches `run_vla_eval.sh --model ...`):
#   act lerobot_diffusion pi0 pi05 smolvla
#
# Examples (every model needs its finetuned checkpoint):
#   sbatch scripts/eval.sbatch act runs/act/finetune/<exp>/checkpoints/last/pretrained_model \
#       --task pour_water
#   sbatch scripts/eval.sbatch pi05 runs/pi05/finetune/<exp>/checkpoints/last/pretrained_model \
#       --task pour_water --episodes 3
#
#   # forward eval_vla.py args after `--` (e.g. --output-json, --vla-config):
#   sbatch scripts/eval.sbatch act <ckpt> --task pour_water -- --output-json runs/act/eval/job_$.json
#
# Output:
#   runs/<model>/eval/job_<jobid>/  ← videos + optional results.json
#   job_logs/<model>_server.<jobid>.log  ← server diagnostics

#SBATCH --job-name=eval_vla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=job_logs/eval_%j.log
#SBATCH --error=job_logs/eval_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
cd "$REPO_ROOT"

MODEL="${1:?usage: sbatch $0 <model> [<checkpoint>] [extra run_vla_eval.sh args...]}"
shift

CKPT_ARG=()
# Heuristic: if the next arg exists and is not a flag, treat it as the checkpoint.
if [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; then
  CKPT_ARG=(--checkpoint "$1")
  shift
fi

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo "model:    $MODEL"
echo "ckpt:     ${CKPT_ARG[*]:-(default)}"
echo "extra:    $*"
echo

# Per-baseline output layout: runs/<model>/eval/job_<jobid>_<model>/
OUT="runs/${MODEL}/eval/job_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Persistent server diagnostics follow the repository-wide log convention.
export SERVER_LOG_DIR="$PWD/job_logs"

exec srun scripts/run_vla_eval.sh \
    --model "$MODEL" "${CKPT_ARG[@]}" \
    --video-dir "$OUT" \
    "$@"
