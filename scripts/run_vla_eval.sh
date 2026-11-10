#!/bin/bash
# One-command VLA baseline eval. Spawns the matching model server in its own
# env, waits for /health, runs scripts/eval_vla.py in the MAWM env, and kills
# the server on exit (normal, error, or Ctrl-C).
#
# Usage:
#   scripts/run_vla_eval.sh --model {act|lerobot_diffusion|pi0|pi05|smolvla}
#                           [--task NAME] [--episodes N] [--max-steps N]
#                           [--port P] [--action-type ee|qpos|qpos_abs]
#                           [--checkpoint PATH] [--unnorm-key KEY] [--video-dir DIR]
#                           [-- extra eval_vla.py args...]
#
# Defaults per model (override with flags above). Every model is served by
# baseline/servers/lerobot_server.py and needs --checkpoint:
#   pi0:               port 8767, action-type qpos_abs (lerobot venv)
#   pi05:              port 8768, action-type qpos_abs (lerobot venv)
#   act:               port 8775, action-type qpos_abs (pi0 venv; ACT_SERVER_ENV=lerobot for the lerobot venv)
#   lerobot_diffusion: port 8777, action-type ee       (lerobot venv)
#   smolvla:           port 8778, action-type ee       (lerobot venv)
#
# Shared defaults: --task pour_water --episodes 1 --max-steps 50
#
# Args after `--` are forwarded verbatim to eval_vla.py.

set -euo pipefail

MODEL=""
TASK="pour_water"
EPISODES=1
MAX_STEPS=50
PORT=""
ACTION_TYPE="${ACTION_TYPE:-}"
VIDEO_DIR=""
CHECKPOINT=""
UNNORM_KEY=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --unnorm-key) UNNORM_KEY="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to pass through to eval_vla.py)" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "required: --model {act|lerobot_diffusion|pi0|pi05|smolvla}" >&2
  exit 2
fi

# Source per-host config (REPO_ROOT, MAWM_PY, per-baseline *_PY, ...).
# To override, create scripts/env.local.sh — see scripts/env.local.sh.example.
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"

# Per-model dispatch table. Keep in sync with baseline/servers/lerobot_server.py
# — these fields are load-bearing:
#   SERVER_PY          env python that can import the model
#   SERVER_SCRIPT      the server module to run
#   DEFAULT_PORT       baseline port (non-overlapping so you can run side-by-side)
#   SERVER_ARGS        any non-port CLI flags the server needs (config/checkpoint)
#   DEFAULT_ACTION_TYPE  matches what the model was trained to emit
case "$MODEL" in
  pi0|pi05)
    # pi0 / pi0.5 are finetuned via lerobot v0.5.2 (see
    # scripts/finetune_lerobot_v2.sh) and served by the generic
    # lerobot_server.py from the lerobot venv — same path as smolvla.
    SERVER_PY="$LEROBOT_PY"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    case "$MODEL" in
      pi0)      DEFAULT_PORT=8767 ;;
      pi05)     DEFAULT_PORT=8768 ;;
    esac
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for $MODEL (finetuned via lerobot, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type "$MODEL" --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos_abs"
    ;;
  act)
    # The LeRobot imitation baselines (act/lerobot_diffusion) run through
    # baseline/servers/lerobot_server.py with a different --policy-type.
    # Reuse the legacy pi0 venv for older ACT checkpoints. Newer LeRobot
    # checkpoints with saved policy processors can opt into the v0.5 env with
    # ACT_SERVER_ENV=lerobot.
    if [[ "${ACT_SERVER_ENV:-pi0}" == "lerobot" ]]; then
      SERVER_PY="$LEROBOT_PY"
    else
      SERVER_PY="$PI0_PY"
    fi
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8775
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for act (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type act --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos_abs"
    ;;
  lerobot_diffusion)
    SERVER_PY="$LEROBOT_PY"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8777
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for lerobot_diffusion (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type diffusion --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  smolvla)
    # SmolVLA only exists in lerobot >= 0.4 (submodule v0.5.2); the pi0
    # venv's v0.1.0 doesn't have it. Use the dedicated lerobot venv.
    SERVER_PY="$LEROBOT_PY"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8778
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for smolvla (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type smolvla --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  *) echo "--model must be act|lerobot_diffusion|pi0|pi05|smolvla, got: $MODEL" >&2; exit 2 ;;
esac

PORT="${PORT:-$DEFAULT_PORT}"
ACTION_TYPE="${ACTION_TYPE:-$DEFAULT_ACTION_TYPE}"

if [[ ! -x "$SERVER_PY" ]]; then
  echo "missing $MODEL server interpreter: $SERVER_PY" >&2
  echo "Set the matching *_PY value in scripts/env.local.sh." >&2
  exit 2
fi

# MAWM_PY already set by env.sh; nothing to do. Keep server diagnostics in the
# repository log area by default; callers may select another persistent log
# directory explicitly.
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$REPO_ROOT/job_logs}"
mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG="${SERVER_LOG_DIR}/${MODEL}_server.${SLURM_JOB_ID:-$$}.log"

echo "[run_vla_eval] model=$MODEL port=$PORT action_type=$ACTION_TYPE"
echo "[run_vla_eval] server log: $SERVER_LOG"

"$SERVER_PY" "$SERVER_SCRIPT" \
    "${SERVER_ARGS[@]}" --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_vla_eval] stopping $MODEL server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Cold-cache first runs download multi-GB checkpoints — allow 30 min. Warm
# reruns come up in ~30-60s.
DEADLINE=1800
for ((i=1; i<=DEADLINE; i++)); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[run_vla_eval] server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_vla_eval] $MODEL server died during startup. Log tail:" >&2
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
  fi
  if (( i % 30 == 0 )); then
    echo "[run_vla_eval] still waiting ... (${i}s) — last log line:"
    tail -n 1 "$SERVER_LOG" || true
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[run_vla_eval] server did not come up within ${DEADLINE}s. Log tail:" >&2
  tail -n 50 "$SERVER_LOG" >&2
  exit 1
fi

EVAL_ARGS=(
  --task "$TASK" --model "$MODEL"
  --server-url "http://127.0.0.1:$PORT"
  --action-type "$ACTION_TYPE"
  --episodes "$EPISODES" --max-steps "$MAX_STEPS"
)
[[ -n "$UNNORM_KEY" ]] && EVAL_ARGS+=(--unnorm-key "$UNNORM_KEY")
[[ -n "$VIDEO_DIR" ]] && EVAL_ARGS+=(--video-dir "$VIDEO_DIR")
EVAL_ARGS+=("${EXTRA_ARGS[@]}")

echo "[run_vla_eval] running eval: ${EVAL_ARGS[*]}"
# VLA eval is simulation/render heavy; use Genesis GPU backend unless the
# caller explicitly overrides it (e.g. GENESIS_BACKEND=cpu for debugging).
export GENESIS_BACKEND="${GENESIS_BACKEND:-gpu}"
echo "[run_vla_eval] GENESIS_BACKEND=$GENESIS_BACKEND"
SETUPTOOLS_USE_DISTUTILS=stdlib "$MAWM_PY" scripts/eval_vla.py "${EVAL_ARGS[@]}"
