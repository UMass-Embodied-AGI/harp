import sys
from argparse import Namespace
from pathlib import Path

import pytest

from envs.tasks import TASK_ALIASES, TASK_MAP, resolve_task_class
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.tasks.take_from_human_easy import TakeFromHumanEasy
from envs.tasks.take_from_human_safety import TakeFromHumanSafety
from scripts import check_latest_genesis_integration as integration
from scripts import probe_latest_genesis_robot_tasks as task_probe
from scripts import probe_latest_genesis_robot_only as robot_probe
from scripts.vla_client import get_task_instruction


def test_default_task_probe_only_contains_registered_unique_names():
    assert len(task_probe.DEFAULT_TASKS) == len(set(task_probe.DEFAULT_TASKS))
    for task_name in task_probe.DEFAULT_TASKS:
        resolved, task_cls = resolve_task_class(task_name)
        assert resolved == task_name
        assert task_cls is not None


def test_probe_settle_override_is_passed_through_config():
    default_config = task_probe.build_probe_config(None)
    assert "settle_steps" not in default_config
    assert default_config["renderer"] == "rasterizer"
    assert task_probe.build_probe_config(1)["settle_steps"] == 1


def test_probe_shards_partition_registry_without_overlap():
    tasks = task_probe.registered_task_names()
    shards = [task_probe.select_task_shard(tasks, 4, i) for i in range(4)]
    assert sum(len(shard) for shard in shards) == len(tasks)
    assert set().union(*map(set, shards)) == set(tasks)
    assert all(
        set(left).isdisjoint(right)
        for i, left in enumerate(shards)
        for right in shards[i + 1 :]
    )


@pytest.mark.parametrize(
    "task_class",
    [CategorizeObjectsContribution, TakeFromHumanEasy, TakeFromHumanSafety],
)
def test_custom_timestep_scene_builders_resolve_effective_renderer(task_class):
    assert "_resolve_renderer" in task_class._setup_scene.__code__.co_names


def test_all_registered_probe_names_match_resolver_surface():
    assert TASK_MAP.keys().isdisjoint(TASK_ALIASES)
    expected = sorted({*TASK_MAP, *TASK_ALIASES})
    assert task_probe.registered_task_names() == expected
    for task_name in expected:
        expected_class = TASK_MAP.get(task_name, TASK_ALIASES.get(task_name))
        assert resolve_task_class(task_name) == (task_name, expected_class)


def test_compatibility_registry_names_resolve_real_instructions():
    assert get_task_instruction("categorize_interrupt") == (
        TASK_ALIASES["categorize_interrupt"].default_instruction()
    )


def test_readme_registry_counts_match_registry():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    expected = (
        f"Ships {len(TASK_MAP)} canonical manipulation tasks plus "
        f"{len(TASK_ALIASES)} compatibility alias"
    )
    assert expected in readme


def test_full_task_sweep_has_no_stale_outer_timeout(monkeypatch):
    args = Namespace(
        python=Path("/bin/true"),
        robots=["franka"],
        tasks=["blocks_ranking_rgb", "put_object_cabinet"],
        full_robot_sweep=False,
        full_task_sweep=True,
        skip_task_sweep=False,
        task_timeout=120.0,
        robot_timeout=120.0,
        steps=1,
        settle_steps=1,
    )
    calls = []

    monkeypatch.setattr(integration, "parse_args", lambda: args)

    def fake_run(label, cmd, timeout, env):
        calls.append((label, cmd, timeout))
        return True

    monkeypatch.setattr(integration, "_run", fake_run)
    assert integration.main() == 0

    label, cmd, timeout = calls[-1]
    assert label == "task reset/step smoke"
    assert "--all-registered" in cmd
    assert cmd[cmd.index("--settle-steps") + 1] == "1"
    assert timeout is None


def test_full_robot_sweep_gets_one_timeout_budget_per_embodiment(monkeypatch):
    args = Namespace(
        python=Path("/bin/true"),
        robots=["franka"],
        tasks=["blocks_ranking_rgb"],
        full_robot_sweep=True,
        full_task_sweep=False,
        skip_task_sweep=True,
        task_timeout=300.0,
        robot_timeout=120.0,
        steps=1,
        settle_steps=1,
    )
    calls = []
    monkeypatch.setattr(integration, "parse_args", lambda: args)
    monkeypatch.setattr(
        integration,
        "_run",
        lambda label, cmd, timeout, env: calls.append((label, cmd, timeout)) or True,
    )

    assert integration.main() == 0
    label, cmd, timeout = calls[1]
    assert label == "direct robot smoke"
    assert "--robots" not in cmd
    assert set(integration.FULL_ROBOT_NAMES) == set(robot_probe.ROBOTS)
    assert timeout == 120.0 * len(robot_probe.ROBOTS) + 30.0


def test_integration_defaults_allow_cold_avatar_task_resets(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_latest_genesis_integration.py"])
    args = integration.parse_args()
    assert args.task_timeout == 300.0
    assert args.settle_steps == 1


def test_integration_version_gate_matches_release_pins():
    requirements = (
        Path(__file__).resolve().parents[2]
        / "requirements-genesis-latest.txt"
    ).read_text(encoding="utf-8")
    for package, version in integration.REQUIRED_RUNTIME_VERSIONS.items():
        assert f"{package}=={version}" in requirements

    command = integration._version_cmd(Path("/release/python"))
    assert command[0] == "/release/python"
    assert "VERSION_MISMATCH" in command[-1]
    for package, version in integration.REQUIRED_RUNTIME_VERSIONS.items():
        assert package in command[-1]
        assert version in command[-1]


def test_robot_probe_covers_every_base_task_embodiment():
    assert set(robot_probe.ROBOTS) == {
        "piper",
        "franka",
        "stretch",
        "xarm7",
        "arx_x5",
        "ur5_wsg",
    }
