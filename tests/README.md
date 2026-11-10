# Test organization

Run the complete suite with `GENESIS_BACKEND=cpu bash scripts/check_quality.sh`
in the pinned environment (no GPU needed). See [release readiness](../docs/release_readiness.md)
for the separate packaging and native-simulation gates.

| Location | Purpose |
| --- | --- |
| `contracts/` | Registry/public API, release inventory, source hygiene, docs, and launcher contracts |
| `test_<family>*.py` | Task-family mechanics, characterization, and targeted repair regressions |
| `test_<utility>*.py` | Shared geometry, planners, collection, conversion, and server utilities |
| `test_pytest_latest_genesis*.py` | Native simulator integration checks |
| `fixtures/` | Small versioned deterministic test inputs |

Keep family tests beside their related tests while they share fixtures and
import seams; do not move modules solely to reduce directory counts. New
cross-repository structure checks belong in `contracts/`, not a task family.

Tests must not load executable dependencies from ignored `job_logs/`, `data/`,
or a personal checkout. Asset-backed and opt-in native tests retain their
explicit prerequisites.

Characterization tests preserve behavior, not implementation layout. When
reorganizing a module, update import/source paths without removing assertions
or altering expected physics, scoring, timing, or seeded draw order. A green
mock-based suite is not an expert success-rate measurement.
