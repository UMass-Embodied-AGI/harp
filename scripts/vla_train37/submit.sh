#!/bin/bash
# Submit a train37 VLA batch array with the GPU constraint its model needs.
#
#   scripts/vla_train37/submit.sh <run_batch.sbatch> <batch_dir> [sbatch args...]
#
# Reads the model (and slots per job) from <batch_dir>/b_0.json and prepends
# --constraint from gpu_guard.py (NYX_*GPU_CONSTRAINT in scripts/env.local.sh;
# omitted when unset): pi0/pi05 get >=16 GB NYX GPUs (48 GB ones when a job
# runs >=2 slots); the other models keep the full NYX pool.  An explicit
# --constraint in the extra args is kept, but refused if it names a GPU listed
# in NYX_SMALL_GPU_FEATURES for pi0/pi05.  T37_SBATCH_ARGS (e.g.
# "-p <partition>") is added before the extra args.  Pass --array/--job-name
# etc. as extra args.  DRY_RUN=1 prints the command instead of submitting.
# See docs/cluster.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -f scripts/env.local.sh ]]; then
  # Export the host knobs (NYX_*GPU_CONSTRAINT, T37_SBATCH_ARGS) for gpu_guard.py.
  set -a
  # shellcheck source=/dev/null
  source scripts/env.local.sh
  set +a
fi
SCRIPT="$1"; BDIR="$2"; shift 2
GUARD=scripts/vla_train37/gpu_guard.py
MODEL=$(python3 "$GUARD" model "$BDIR")
PAR=$(python3 "$GUARD" parallel "$BDIR")
read -r -a SBATCH_EXTRA <<< "${T37_SBATCH_ARGS:-}"
USER_CONSTRAINT=""
for a in "$@"; do
  case "$a" in --constraint=*) USER_CONSTRAINT="${a#--constraint=}";; esac
done
if [[ -n "$USER_CONSTRAINT" ]]; then
  python3 "$GUARD" check "$MODEL" "$USER_CONSTRAINT" || {
    echo "refusing --constraint='$USER_CONSTRAINT' for $MODEL (it admits a GPU under 16 GB)" >&2; exit 2; }
  cmd=(sbatch "${SBATCH_EXTRA[@]}" "$@" "$SCRIPT" "$BDIR")
else
  CONSTRAINT="$(python3 "$GUARD" constraint "$MODEL" "$PAR")"
  CFLAG=(); [[ -n "$CONSTRAINT" ]] && CFLAG=(--constraint="$CONSTRAINT")
  cmd=(sbatch "${CFLAG[@]}" "${SBATCH_EXTRA[@]}" "$@" "$SCRIPT" "$BDIR")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
