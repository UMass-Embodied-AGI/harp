#!/bin/bash
#SBATCH --job-name=evalset_init
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o job_logs/evalset_init_%j.log
#SBATCH -e job_logs/evalset_init_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p job_logs

REQUESTED_GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"
# shellcheck source=env.sh
source scripts/env.sh

export SETUPTOOLS_USE_DISTUTILS=stdlib
export GENESIS_BACKEND="$REQUESTED_GENESIS_BACKEND"

echo "=== eval-set init validation ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "args: $*"
echo "python=${MAWM_PY}"
echo "GENESIS_BACKEND=${GENESIS_BACKEND}"

srun "${MAWM_PY}" scripts/validate_eval_set_init.py --subprocess-per-episode "$@"
