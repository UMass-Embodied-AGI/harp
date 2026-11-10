"""Release eval set eval_v3_sampled: frozen size, task coverage and row schema."""
import json
from collections import Counter
from pathlib import Path

from envs.tasks import resolve_task_class

ROOT = Path(__file__).resolve().parents[1] / "eval_sets" / "eval_v3_sampled"


def _rows():
    return [json.loads(l) for l in (ROOT / "manifest.jsonl").read_text().splitlines() if l.strip()]


def test_manifest_matches_counts_and_is_unique():
    rows = _rows()
    counts = json.loads((ROOT / "counts.json").read_text())
    assert len(rows) == counts["rows"] == 3403
    assert dict(Counter(r["task"] for r in rows)) == counts["per_task"]
    assert len({(r["task"], r["seed"]) for r in rows}) == len(rows)
    assert len({r["id"] for r in rows}) == len(rows)


def test_rows_are_registered_and_complete():
    rows = _rows()
    tasks = {r["task"] for r in rows}
    assert len(tasks) == 37
    for t in tasks:  # canonical names and the categorize_interrupt compatibility name
        assert resolve_task_class(t) is not None
    for r in rows:
        assert r["eval_set"] == "eval_v3_sampled"
        assert r["schema_version"] == "genesis_hr_bench.eval_set.v1"
        assert isinstance(r["seed"], int) and r["seed"] >= 9_000_000
        assert r["max_steps"] > 0 and r["action_substeps"] > 0
        assert r["config_overrides"]["eval_mode"] is True
        assert r["category"] in {"cueing", "contribution", "interruption", "copresence"}
