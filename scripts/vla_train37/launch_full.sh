#!/bin/bash
# Build batches for a full train37 VLA eval over an eval-set manifest, then
# run them locally, or print (default) / submit per-model SLURM arrays.
#
#   scripts/vla_train37/launch_full.sh <eval_set.jsonl> <run_name> [--local|--submit]
#
#   (no flag)  build batches and print the sbatch commands (dry run)
#   --local    build batches and run every batch on this machine, one after
#              another, with run_batch.py (needs one NYX-capable GPU)
#   --submit   build batches and submit one SLURM array per model
#
# Env knobs:
#   MODELS          comma list (default: act,diffusion,pi0,pi05,smolvla)
#   PER_BATCH       episodes per batch file / array task (default: 37)
#   PARALLEL        concurrent episodes per GPU, one policy server each (default: 2)
#   VIDEO           1 = save episode videos (default: 1)
#   CONC_PER_MODEL  SLURM array concurrency per model (default: 2, so at most
#                   2 x 5 models = 10 GPU jobs at once)
#   JOB_TIME        SLURM --time per array task (default: 20:00:00)
#   T37_SBATCH_ARGS extra sbatch arguments, e.g. "-p <partition> --exclude=<nodes>"
# --constraint comes from scripts/vla_train37/gpu_guard.py, which reads the
# NYX_*GPU_CONSTRAINT variables (empty: no --constraint). Set these in
# scripts/env.local.sh; see docs/cluster.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROWS="$1"; RUN="$2"; MODE="${3:-}"
case "$MODE" in ""|--local|--submit) ;; *) echo "unknown mode: $MODE (use --local or --submit)" >&2; exit 2 ;; esac
if [[ -f scripts/env.local.sh ]]; then
  # Export the host knobs (NYX_*GPU_CONSTRAINT, T37_SBATCH_ARGS) for gpu_guard.py.
  set -a
  # shellcheck source=/dev/null
  source scripts/env.local.sh
  set +a
fi
MODELS="${MODELS:-act,diffusion,pi0,pi05,smolvla}"
PER_BATCH="${PER_BATCH:-37}"; PARALLEL="${PARALLEL:-2}"
CONC="${CONC_PER_MODEL:-2}"; VIDEO="${VIDEO:-1}"
read -r -a SBATCH_EXTRA <<< "${T37_SBATCH_ARGS:-}"
RUN_DIR="data/vla_train37_eval/$RUN"
[[ -e "$RUN_DIR/batches" && "${REBUILD:-0}" != 1 ]] && { echo "refusing to overwrite $RUN_DIR/batches (REBUILD=1 to rebuild; finished episodes are kept)" >&2; exit 2; }
rm -rf "$RUN_DIR/batches"
mkdir -p "$RUN_DIR" job_logs/train37
cp "$ROWS" "$RUN_DIR/eval_set_rows.jsonl"
sha256sum "$RUN_DIR/eval_set_rows.jsonl" > "$RUN_DIR/eval_set_rows.sha256"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git diff HEAD > "$RUN_DIR/source_uncommitted.diff"
VFLAG=(); [[ "$VIDEO" == 1 ]] && VFLAG=(--video)
python3 scripts/vla_train37/make_batches.py --rows "$RUN_DIR/eval_set_rows.jsonl" \
  --run-dir "$RUN_DIR" --models "$MODELS" --per-batch "$PER_BATCH" \
  --parallel "$PARALLEL" "${VFLAG[@]}" | tee "$RUN_DIR/batches.txt"

IFS=',' read -r -a MLIST <<< "$MODELS"
if [[ "$MODE" == --local ]]; then
  # shellcheck source=../env.sh
  source scripts/env.sh
  export LEROBOT_PY="${T37_LEROBOT_PY:-$LEROBOT_PY}"
  NYX_PY="${NYX_PY:-$MAWM_PY}"
  for m in "${MLIST[@]}"; do
    for b in "$RUN_DIR/batches/$m"/b_*.json; do
      echo "[launch_full] local batch $b"
      "$NYX_PY" scripts/vla_train37/run_batch.py "$b"
    done
  done
  exit 0
fi
for m in "${MLIST[@]}"; do
  n=$(ls "$RUN_DIR/batches/$m"/b_*.json | wc -l)
  # GPU constraint from gpu_guard.py (pi0/pi05 need >=16 GB, 48 GB ones at
  # >=2 slots per job); omitted when the NYX_*GPU_CONSTRAINT knobs are unset.
  extra=()
  constraint="$(python3 scripts/vla_train37/gpu_guard.py constraint "$m" "$PARALLEL")"
  [[ -n "$constraint" ]] && extra+=(--constraint="$constraint")
  if python3 -c "import sys; sys.path.insert(0, 'scripts/vla_train37'); import gpu_guard; sys.exit(0 if gpu_guard.is_large('$m') else 1)"; then
    extra+=(--mem=96G)
  fi
  cmd=(sbatch --parsable --array=0-$((n - 1))%"$CONC" --time="${JOB_TIME:-20:00:00}" --job-name="t37_${RUN}_$m"
       "${SBATCH_EXTRA[@]}" "${extra[@]}" scripts/vla_train37/run_batch.sbatch "$RUN_DIR/batches/$m")
  if [[ "$MODE" == --submit ]]; then
    jid=$("${cmd[@]}")
    echo "{\"model\": \"$m\", \"job\": \"$jid\", \"batches\": $n}" | tee -a "$RUN_DIR/submissions.jsonl"
  else
    printf '%q ' "${cmd[@]}"; echo
  fi
done
