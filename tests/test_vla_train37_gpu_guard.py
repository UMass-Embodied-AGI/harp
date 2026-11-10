"""pi0/pi05 never run on an 11 GB GPU."""
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vla_train37" / "gpu_guard.py"

# Example cluster feature names (see docs/cluster.md); the module itself ships
# no cluster-specific defaults.
_ENV = {
    "NYX_GPU_CONSTRAINT": "small11gb|mid16gb|big48gb",
    "NYX_LARGE_GPU_CONSTRAINT": "mid16gb|big48gb",
    "NYX_LARGE_MULTI_GPU_CONSTRAINT": "big48gb",
    "NYX_SMALL_GPU_FEATURES": "small11gb",
}


def _load(monkeypatch, env):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("t37_gpu_guard", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gpu_guard(monkeypatch):
    return _load(monkeypatch, _ENV)


def test_defaults_emit_no_cluster_specific_constraint(monkeypatch):
    guard = _load(monkeypatch, {})
    for model in ("act", "pi0"):
        assert guard.constraint_for(model, 2) == ""
        assert guard.constraint_ok(model, "anything")


@pytest.mark.parametrize("model", ["pi0", "pi05"])
def test_large_models_never_get_a_small_gpu(gpu_guard, model):
    for parallel in (1, 2, 4):
        c = gpu_guard.constraint_for(model, parallel)
        assert "small11gb" not in c
        assert gpu_guard.constraint_ok(model, c)
    assert "mid16gb" in gpu_guard.constraint_for(model, 1)
    assert gpu_guard.constraint_for(model, 2) == "big48gb"
    assert not gpu_guard.constraint_ok(model, "small11gb|big48gb")


@pytest.mark.parametrize("model", ["act", "diffusion", "smolvla"])
def test_small_models_keep_the_full_nyx_pool(gpu_guard, model):
    assert gpu_guard.constraint_for(model, 2) == gpu_guard.NYX_CONSTRAINT
    assert gpu_guard.constraint_ok(model, "small11gb")
    assert gpu_guard.slots_for_gpu(model, 2, 11264) == 2


def test_runtime_refuses_pi0_on_11gb_and_caps_slots(gpu_guard):
    with pytest.raises(SystemExit, match=">=16 GB"):
        gpu_guard.slots_for_gpu("pi0", 1, 11264)    # 11 GB card
    assert gpu_guard.slots_for_gpu("pi0", 2, 15356) == 1   # 16 GB card
    assert gpu_guard.slots_for_gpu("pi05", 2, 46068) == 2  # 48 GB card
    assert gpu_guard.slots_for_gpu("pi0", 2, None) == 2    # unknown: no cap


def test_cli_reads_model_from_batch_dir(gpu_guard, tmp_path, capsys):
    (tmp_path / "b_0.json").write_text(json.dumps({"model": "pi0", "parallel": 2, "items": []}))
    assert gpu_guard.main(["model", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "pi0"
    assert gpu_guard.main(["constraint", "pi0", "1"]) == 0
    assert "small11gb" not in capsys.readouterr().out
    assert gpu_guard.main(["check", "pi0", "small11gb|big48gb"]) == 1
