#!/bin/bash
# Generic rasterizer launcher.
#
# Usage:
#   sbatch scripts/launch/rasterizer.sh <task> [config.yml]
#
# The task name is passed to collect.py --task; the optional config yaml is
# passed to --config if given.  Output lands at data/<task>/rasterizer/.
# Override sbatch defaults (time, constraint, etc.) with `sbatch -t ... ...`.
#
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -J rast
#SBATCH -o job_logs/rasterizer_%j.log
#SBATCH -e job_logs/rasterizer_%j.log

set -euo pipefail

TASK="${1:?usage: sbatch scripts/launch/rasterizer.sh <task> [config.yml]}"
CFG="${2:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}}"
cd "$REPO_ROOT"
SAVE="data/${TASK}/rasterizer"
mkdir -p "$SAVE"

source scripts/env.sh
export GENESIS_BACKEND=cpu
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3
export PYTHONUNBUFFERED=1

echo "=== Rasterizer: $TASK ${CFG:+(config=$CFG)} ==="
if [ -n "$CFG" ]; then
    exec srun "$MAWM_PY" scripts/collect.py --task "$TASK" --config "$CFG" \
        --renderer rasterizer --episodes 1 --save-dir "$SAVE"
else
    exec srun "$MAWM_PY" scripts/collect.py --task "$TASK" \
        --renderer rasterizer --episodes 1 --save-dir "$SAVE"
fi
