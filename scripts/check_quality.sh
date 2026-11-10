#!/usr/bin/env bash
# Run from a compute allocation on shared clusters; see docs/release_readiness.md.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  printf 'Usage: bash scripts/check_quality.sh\n'
  printf 'Runs Ruff and the complete test suite, including strict release inventory checks.\n'
  if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    exit 0
  fi
  exit 2
fi

quality_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$quality_root"
source "$quality_root/scripts/env.sh"

# Prefer the linter installed with requirements-dev.txt. An explicit executable
# permits read-only validation in an existing pinned simulator environment.
if [[ -n "${RUFF_BIN:-}" ]]; then
  quality_linter=("$RUFF_BIN")
else
  quality_linter=("$MAWM_PY" -m ruff)
fi

quality_status=0
# Audit repository-owned adapters, not the vendored baseline repositories.
quality_sources=(envs scripts tests baseline/servers)
printf 'Checking Python correctness (%s)...\n' "${quality_sources[*]}"
"${quality_linter[@]}" check "${quality_sources[@]}" || quality_status=1
printf 'Running all regression and release-inventory checks...\n'
"$MAWM_PY" -m pytest -q tests || quality_status=1

if [[ "$quality_status" -eq 0 ]]; then
  printf 'Source quality checks passed. Simulation qualification is a separate gate.\n'
else
  printf 'Source quality checks failed; see the diagnostics above.\n' >&2
fi
exit "$quality_status"
