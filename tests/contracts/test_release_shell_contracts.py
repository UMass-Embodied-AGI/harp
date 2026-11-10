import os
import re
import stat
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _working_tree_shell_entrypoints(repo: Path = REPO) -> list[str]:
    """Check the candidate checkout, including additions before they are staged.

    Git's index alone still lists locally removed launchers and misses new
    release scripts. Ignore only Git-reported deletions and ignored local
    experiments; tracked scripts remain in scope even if an ignore rule
    matches them. Tracking completeness has its own release-manifest tests.
    """

    def paths(*options: str) -> set[str]:
        result = subprocess.run(
            ["git", "ls-files", *options, "-z", "--", "*.sh", "*.sbatch"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        )
        return set(filter(None, result.stdout.split("\0")))

    candidates = paths("--cached", "--others", "--exclude-standard")
    return sorted(candidates - paths("--deleted"))


def test_shell_inventory_covers_the_current_release_candidate(tmp_path):
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    (tmp_path / ".gitignore").write_text("local/\n")
    for relative in ("retained.sh", "removed.sh", "local/tracked.sh"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n")
    subprocess.run(
        ["git", "add", "-f", "retained.sh", "removed.sh", "local/tracked.sh"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "removed.sh").unlink()
    (tmp_path / "new job.sbatch").write_text("#!/bin/bash\n")
    (tmp_path / "local/experiment.sh").write_text("#!/bin/bash\n")
    (tmp_path / "not_a_launcher.txt").write_text("not a shell script\n")

    assert _working_tree_shell_entrypoints(tmp_path) == [
        "local/tracked.sh",
        "new job.sbatch",
        "retained.sh",
    ]


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_env_contract_preserves_external_simulator_python(tmp_path):
    simulator_python = _make_executable(tmp_path / "sim" / "bin" / "python")
    baseline_root = tmp_path / "baselines"
    env = dict(os.environ)
    env.update(
        MAWM_PY=str(simulator_python),
        VLA_ENV_ROOT=str(baseline_root),
        GENESIS_BACKEND="cpu",
    )
    # Exercise derivation from VLA_ENV_ROOT even when the test runner itself
    # was launched after sourcing scripts/env.sh.
    for name in (
        "LEROBOT_PY",
        "PI0_PY",
        "VIRTUAL_ENV",
    ):
        env.pop(name, None)

    command = r'''
        source scripts/env.sh
        printf '%s\n' "$REPO_ROOT" "$MAWM_PY" "$VLA_ENV_ROOT" \
            "$LEROBOT_PY" "$PI0_PY" "$FT_GRES" \
            "$FT_CPUS_PER_GPU" "$FT_MEM_GB_PER_GPU" "$GENESIS_BACKEND" \
            "${VIRTUAL_ENV-}"
    '''
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    values = result.stdout.splitlines()
    assert values == [
        str(REPO),
        str(simulator_python),
        str(baseline_root),
        str(baseline_root / "lerobot/bin/python"),
        str(baseline_root / "pi0/bin/python"),
        "gpu",
        "8",
        "80",
        "cpu",
        "",
    ]


def test_env_contract_uses_repo_baseline_root_for_nonlegacy_simulator(tmp_path):
    simulator_python = _make_executable(tmp_path / "simulator" / "bin" / "python")
    env = dict(os.environ)
    env["MAWM_PY"] = str(simulator_python)
    for name in (
        "VLA_ENV_ROOT",
        "LEROBOT_PY",
        "PI0_PY",
        "VIRTUAL_ENV",
    ):
        env.pop(name, None)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/env.sh; printf "%s\\n" "$VLA_ENV_ROOT" "$LEROBOT_PY" "$PI0_PY"',
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    baseline_root = REPO / ".venvs"
    assert result.stdout.splitlines() == [
        str(baseline_root),
        str(baseline_root / "lerobot/bin/python"),
        str(baseline_root / "pi0/bin/python"),
    ]


def test_sourcing_env_contract_preserves_caller_shell_options(tmp_path):
    simulator_python = _make_executable(tmp_path / "simulator/bin/python")
    env = dict(os.environ, MAWM_PY=str(simulator_python))
    result = subprocess.run(
        [
            "bash",
            "-c",
            r'''
                set +e +u
                set +o pipefail
                before_flags=$-
                shopt -qo pipefail; before_pipe=$?
                source scripts/env.sh
                after_flags=$-
                shopt -qo pipefail; after_pipe=$?
                printf '%s\n' "$before_flags" "$after_flags" \
                    "$before_pipe" "$after_pipe"
            ''',
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    before_flags, after_flags, before_pipe, after_pipe = result.stdout.splitlines()
    assert after_flags == before_flags
    assert after_pipe == before_pipe == "1"


def test_generated_directory_ignores_are_root_anchored():
    def ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=REPO,
            check=False,
        )
        return result.returncode == 0

    # The release asset bundle is commonly mounted as a root symlink, so test
    # the path itself instead of traversing beyond it.
    assert ignored("assets")
    assert ignored("data/episode.pkl")
    assert ignored("outputs/frame.png")
    assert not ignored("envs/robot/assets/planner.urdf")
    assert not ignored("docs/assets/architecture.svg")
    assert ignored("scripts/launch/new_backend.sh")


def test_all_release_shell_entrypoints_parse():
    entrypoints = _working_tree_shell_entrypoints()
    assert entrypoints
    for relative_path in entrypoints:
        subprocess.run(
            ["bash", "-n", relative_path],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=True,
        )


def test_release_launchers_do_not_embed_an_absolute_repo_checkout():
    absolute_checkout = re.compile(
        r"(?<![A-Za-z0-9_.-])/(?:[^/\s\"']+/)+genesis-hr-bench"
        r"(?=[/\s\"']|$)"
    )

    offenders = []
    for relative_path in _working_tree_shell_entrypoints():
        text = (REPO / relative_path).read_text(
            encoding="utf-8", errors="replace"
        )
        if absolute_checkout.search(text):
            offenders.append(relative_path)

    assert offenders == [], (
        "release launchers must resolve the repository at runtime: "
        + ", ".join(offenders)
    )


def test_default_pytest_discovery_stays_inside_release_tests():
    config = (REPO / "pytest.ini").read_text(encoding="utf-8")
    assert re.search(r"(?m)^testpaths\s*=\s*tests\s*$", config)
    for generated_tree in ("assets", "data", "outputs", "runs", "vla_data"):
        assert re.search(
            rf"(?m)^\s+{re.escape(generated_tree)}\s*$", config
        ), f"pytest.ini must exclude generated tree {generated_tree!r}"


def test_slurm_submission_logs_use_the_tracked_job_log_parent():
    destinations = []
    pattern = re.compile(
        r"^#SBATCH\s+(?:--output|--error|-o|-e)(?:=|\s+)(\S+)"
    )
    for relative_path in _working_tree_shell_entrypoints():
        for line in (REPO / relative_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            match = pattern.match(line)
            if match:
                destinations.append((relative_path, match.group(1)))

    assert destinations
    for relative_path, destination in destinations:
        assert destination.startswith("job_logs/"), (
            f"{relative_path} writes Slurm output before the job starts to "
            f"an unsafe or missing parent: {destination}"
        )
    assert (REPO / "job_logs" / ".gitkeep").is_file()

    # The generic finetune launcher passes command-line destinations to
    # sbatch, which override the directives in finetune.sbatch.
    finetune = (REPO / "scripts" / "finetune.sh").read_text(encoding="utf-8")
    for match in re.finditer(r"--(?:output|error)=\"([^\"]+)\"", finetune):
        assert "/job_logs/" in match.group(1)
        assert "/runs/" not in match.group(1)


def test_local_launchers_and_vla_experiments_stay_opt_in():
    def ignored(path: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=REPO,
            check=False,
        ).returncode == 0

    assert ignored("scripts/launch/local_experiment.sbatch")
    assert not ignored("scripts/launch/rasterizer.sh")
    assert ignored("scripts/vla_data/materialize_hf_ready.py")
    assert ignored("scripts/vla_data/select_successful_subset.py")
    assert ignored("scripts/vla_data/select_successful_subset.sbatch")
    assert ignored(".vla_data_hf_ready_local/steps.h5")
    assert ignored("vla_cpu_record_canary_local/steps.h5")
    for tracked_utility in (
        "rewrite_instructions.py",
        "audit_vla_hz.py",
        "normalize_vla_hz.py",
    ):
        assert not ignored(f"scripts/vla_data/{tracked_utility}")


def test_release_install_uses_the_complete_requirements_entrypoint():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    status = (REPO / "docs/latest_genesis_status.md").read_text(
        encoding="utf-8"
    )

    assert "pip install -r requirements.txt" in readme
    assert "requirements.txt" in status
    for text in (readme, status):
        assert "requirements-genesis-latest.txt" in text
        assert "simulator" in text.lower() and "lock" in text.lower()


def test_generic_render_launchers_use_release_runtime_and_job_logs():
    rasterizer = (REPO / "scripts/launch/rasterizer.sh").read_text(
        encoding="utf-8"
    )

    assert "source scripts/env.sh" in rasterizer
    assert "export GENESIS_BACKEND=cpu" in rasterizer
    assert rasterizer.index("source scripts/env.sh") < rasterizer.index(
        "export GENESIS_BACKEND=cpu"
    )
    assert 'srun "$MAWM_PY" scripts/collect.py' in rasterizer
    assert "--renderer rasterizer" in rasterizer
    assert "robotwin" not in rasterizer
    assert "run.log" not in rasterizer
    assert "#SBATCH -o job_logs/" in rasterizer
    assert "#SBATCH -e job_logs/" in rasterizer
    assert "SLURM_SUBMIT_DIR" in rasterizer


def test_primary_finetune_drivers_use_portable_interpreters():
    expected = {
        "scripts/finetune_lerobot.sh": 'ENV_PY="${LEROBOT_TRAIN_PY:-$PI0_PY}"',
    }
    for relative_path, assignment in expected.items():
        source = (REPO / relative_path).read_text(encoding="utf-8")
        assert "source scripts/env.sh" in source
        assert assignment in source
        assert "/scratch4/" not in source


def test_direct_sbatch_entrypoints_do_not_resolve_from_slurm_spool():
    """BASH_SOURCE points at Slurm's copied script, not the checkout."""

    for relative_path in (
        "scripts/launch/rasterizer.sh",
    ):
        source = (REPO / relative_path).read_text(encoding="utf-8")
        repo_assignment = next(
            line
            for line in source.splitlines()
            if line.startswith('REPO_ROOT="${REPO_ROOT:-')
        )
        assert "SLURM_SUBMIT_DIR" in repo_assignment, (
            f"{relative_path} would derive the checkout from Slurm's spool copy"
        )


def test_nyx_compatibility_doc_stays_minimal():
    compatibility_doc = (REPO / "docs/nyx_render.md").read_text(
        encoding="utf-8"
    )
    assert "general/nyx_render_howto.md" in compatibility_doc
    assert "outputs/" not in compatibility_doc
    assert len(compatibility_doc.splitlines()) < 30


def test_generic_finetune_submission_is_independent_of_caller_cwd():
    submitter = (REPO / "scripts/finetune.sh").read_text(encoding="utf-8")
    wrapper = (REPO / "scripts/finetune.sbatch").read_text(encoding="utf-8")

    assert '--chdir="$REPO_ROOT"' in submitter
    assert 'source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"' in wrapper


def test_vla_eval_dispatch_honors_per_baseline_interpreter_overrides():
    source = (REPO / "scripts/run_vla_eval.sh").read_text(encoding="utf-8")

    for variable in (
        "LEROBOT_PY",
        "PI0_PY",
    ):
        assert f'SERVER_PY="${variable}"' in source
    assert 'SERVER_PY="$ENV_ROOT/' not in source
    assert 'if [[ ! -x "$SERVER_PY" ]]' in source

    example = (REPO / "scripts/env.local.sh.example").read_text(
        encoding="utf-8"
    )
    for variable in (
        "LEROBOT_PY",
        "PI0_PY",
    ):
        assert f"{variable}=" in example


def test_direct_finetune_wrappers_honor_host_overrides_and_use_srun():
    wrappers = (
        "act",
    )
    for model in wrappers:
        source = (REPO / f"scripts/finetune_{model}.sbatch").read_text(
            encoding="utf-8"
        )
        assert "SLURM_SUBMIT_DIR" in source
        assert "source scripts/env.sh" in source
        assert f'exec srun scripts/finetune_{model}.sh "$@"' in source
        assert "/scratch4/" not in source
        assert re.search(r"(?m)^#SBATCH --output=job_logs/", source)
        assert re.search(r"(?m)^#SBATCH --error=job_logs/", source)


def test_eval_wrappers_preserve_requested_backend_across_env_loading():
    # The expert eval wrapper defaults to the GPU backend so NYX rows render
    # with NYX; the init validator only resets scenes and defaults to CPU.
    for relative_path, default in (
        ("scripts/eval_expert_set.sbatch", "gpu"),
        ("scripts/validate_eval_set_init.sbatch", "cpu"),
    ):
        source = (REPO / relative_path).read_text(encoding="utf-8")
        assert f'REQUESTED_GENESIS_BACKEND="${{GENESIS_BACKEND:-{default}}}"' in source
        assert 'export GENESIS_BACKEND="$REQUESTED_GENESIS_BACKEND"' in source


def test_user_run_sbatch_wrappers_carry_no_cluster_specific_resources():
    for relative_path in (
        "scripts/eval.sbatch",
        "scripts/eval_expert_set.sbatch",
        "scripts/validate_eval_set_init.sbatch",
        "scripts/finetune_act.sbatch",
        "scripts/launch/rasterizer.sh",
        "scripts/vla_train37/run_batch.sbatch",
    ):
        source = (REPO / relative_path).read_text(encoding="utf-8")
        assert not re.search(
            r"(?m)^#SBATCH\s+(?:-p|--partition|--constraint|-C|--exclude|-x)\b", source
        ), relative_path


def test_vla_sbatch_wrappers_run_server_and_eval_inside_srun_steps():
    source = (REPO / "scripts/eval.sbatch").read_text(encoding="utf-8")
    assert "srun scripts/run_vla_eval.sh" in source


def test_vla_server_logs_default_to_the_persistent_job_log_directory():
    source = (REPO / "scripts/run_vla_eval.sh").read_text(encoding="utf-8")
    assert 'SERVER_LOG_DIR="${SERVER_LOG_DIR:-$REPO_ROOT/job_logs}"' in source
    assert "mktemp" not in source
