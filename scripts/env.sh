#!/usr/bin/env bash
# Shared, portable environment contract for repository launchers.
#
# Host-specific paths belong in scripts/env.local.sh (gitignored).  Values
# already present in the caller environment or env.local.sh always win.

# This file is a sourced environment contract. It must not mutate the caller's
# shell option state; executable launchers enable their own strict mode.

_GENESIS_HR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GENESIS_HR_BOOTSTRAP_ROOT="$(cd "$_GENESIS_HR_SCRIPT_DIR/.." && pwd)"

# Load host overrides before deriving any paths from them.
if [[ -f "$_GENESIS_HR_SCRIPT_DIR/env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_GENESIS_HR_SCRIPT_DIR/env.local.sh"
fi

: "${REPO_ROOT:=$_GENESIS_HR_BOOTSTRAP_ROOT}"
: "${GENESIS_HR_LATEST_ROOT:=$REPO_ROOT}"
: "${GENESIS_LATEST_VENV:=$GENESIS_HR_LATEST_ROOT/.venv-genesis-latest}"
: "${MAWM_PY:=$GENESIS_LATEST_VENV/bin/python}"

# Parent directory containing optional per-baseline environments. Preserve the
# historical `<root>/MAWM/bin/python` layout when detected; a normal repo-local
# simulator venv instead defaults baseline environments to `<repo>/.venvs`.
if [[ -z "${VLA_ENV_ROOT+x}" ]]; then
  _GENESIS_HR_MAWM_ENV_DIR="$(dirname "$(dirname "$MAWM_PY")")"
  if [[ "$(basename "$_GENESIS_HR_MAWM_ENV_DIR")" == "MAWM" ]]; then
    VLA_ENV_ROOT="$(dirname "$_GENESIS_HR_MAWM_ENV_DIR")"
  else
    VLA_ENV_ROOT="$REPO_ROOT/.venvs"
  fi
fi
: "${LEROBOT_PY:=$VLA_ENV_ROOT/lerobot/bin/python}"
: "${PI0_PY:=$VLA_ENV_ROOT/pi0/bin/python}"
# NYX-rendered VLA batch evaluation (scripts/vla_train37/) runs NYX_PY; the
# pinned release environment includes NYX, so it defaults to MAWM_PY.
: "${NYX_PY:=$MAWM_PY}"

# Optional SLURM knobs for `scripts/finetune.sh --slurm` (docs/cluster.md).
# Partition, QoS, constraint and account names are cluster-specific, so they
# default to empty, which omits the corresponding sbatch option.
FT_PARTITION="${FT_PARTITION-}"
FT_QOS="${FT_QOS-}"
FT_CONSTRAINT="${FT_CONSTRAINT-}"
FT_ACCOUNT="${FT_ACCOUNT-}"
: "${FT_GRES:=gpu}"
: "${FT_CPUS_PER_GPU:=8}"
: "${FT_MEM_GB_PER_GPU:=80}"

if [[ ! -x "$MAWM_PY" ]]; then
  echo "Missing simulator Python: $MAWM_PY" >&2
  echo "Create $GENESIS_LATEST_VENV or set MAWM_PY in scripts/env.local.sh." >&2
  return 1 2>/dev/null || exit 1
fi

# Keep the convenient activation promised by the README for the standard
# repo-local environment.  An explicitly selected external interpreter must
# not have its environment silently replaced.
if [[ "$MAWM_PY" == "$GENESIS_LATEST_VENV/bin/python" \
      && -f "$GENESIS_LATEST_VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$GENESIS_LATEST_VENV/bin/activate"
fi

export REPO_ROOT GENESIS_HR_LATEST_ROOT GENESIS_LATEST_VENV MAWM_PY
export VLA_ENV_ROOT LEROBOT_PY PI0_PY NYX_PY
export FT_PARTITION FT_QOS FT_CONSTRAINT FT_ACCOUNT FT_GRES
export FT_CPUS_PER_GPU FT_MEM_GB_PER_GPU
# Optional cuRobo interpreter for embodiments configured with planner: curobo.
if [[ -n "${CUROBO_PY:-}" ]]; then export CUROBO_PY; fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GENESIS_BACKEND="${GENESIS_BACKEND:-gpu}"

unset _GENESIS_HR_SCRIPT_DIR _GENESIS_HR_BOOTSTRAP_ROOT
unset _GENESIS_HR_MAWM_ENV_DIR
