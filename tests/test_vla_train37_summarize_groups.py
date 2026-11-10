"""summarize.py: co-presence group headline and paper-named markdown rows."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "t37_summarize", REPO / "scripts" / "vla_train37" / "summarize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(root: Path, model, task, seed, success, collision):
    d = root / "episodes" / model / task / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps(
        {"task": task, "seed": seed, "model": model, "finished": True}))
    ep = {"success": success, "steps": 10,
          "safety": {"enabled": True, "min_distance_m": 0.1,
                     "collision": {"any_collision": collision},
                     "approach": {"max_approach_speed_mps": 0.0}, "intrusion": {}}}
    (d / "result.json").write_text(json.dumps({"episode_results": [ep]}))


def test_safe_success_requires_no_collision():
    s = _load()
    ok = {"success": True, "safety": {"enabled": True, "collision": {"any_collision": False}}}
    hit = {"success": True, "safety": {"enabled": True, "collision": {"any_collision": True}}}
    legacy = {"success": True, "avatar_collision": {"any_collision": True}}
    assert s.safe_success(ok) and not s.safe_success(hit) and not s.safe_success(legacy)
    agg = s.aggregate([{"success": True}, dict(legacy), {"success": False}], [0.1])
    assert agg["n_safe_success"] == 1 and abs(agg["safe_success_rate"] - 1 / 3) < 1e-9
    assert agg["success_rate"] == 2 / 3


def test_groups_table_headline_is_safe_success(tmp_path, monkeypatch, capsys):
    s = _load()
    t = "dump_bin_copresence"
    _episode(tmp_path, "act", t, 1, True, False)
    _episode(tmp_path, "act", t, 2, True, True)
    _episode(tmp_path, "act", t, 3, True, False)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in [
        {"task": t, "seed": 1, "measured_contested": {"label": "contested"}},
        {"task": t, "seed": 2, "measured_contested": {"label": "contested"}},
        {"task": t, "seed": 3, "measured_contested": {"label": "not_contested"}},
    ]))
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(tmp_path), "--groups", str(manifest)])
    s.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["by_model_group"]["act/contested"]["safe_success_rate"] == 0.5
    assert summary["by_model_group"]["act/contested"]["success_rate"] == 1.0
    assert summary["by_model_group"]["act/not_contested"]["safe_success_rate"] == 1.0
    md = (tmp_path / "summary.md").read_text()
    assert "**safe success**" in md and "| act | contested | 2 | **50.0%** | 100.0% |" in md


def test_markdown_rows_use_paper_condition_and_task_names(tmp_path, monkeypatch):
    s = _load()
    _episode(tmp_path, "act", "dump_bin_copresence", 1, True, False)
    _episode(tmp_path, "act", "place_dual_shoes_neutral", 1, False, False)  # pre-HARP run dir
    _episode(tmp_path, "act", "factory_kit_packing", 1, False, False)
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(tmp_path)])
    s.main()
    md = (tmp_path / "summary.md").read_text()
    assert "| Co-presence: Dump Bin (`dump_bin_copresence`) |" in md
    assert "| Co-presence: Place Pair of Shoes (`place_dual_shoes_neutral`) |" in md
    assert "| Co-presence (`copresence`) | 50.0% (2) |" in md
    assert "| `factory_kit_packing` |" in md
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "act/dump_bin_copresence" in summary["by_model_task"]
