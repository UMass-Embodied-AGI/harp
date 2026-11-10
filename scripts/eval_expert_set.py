#!/usr/bin/env python3
"""Evaluate scripted expert baselines on an initialization-only eval set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.interaction.interrupt_progress import (
    ScriptedRolloutDriver,
    interrupt_metrics,
    run_full_state_interrupt_expert,
)
from envs.manifest_io import load_eval_set_manifest
from envs.success_hold import success_hold_steps
from envs.tasks import (
    BENCHMARK_TASKS,
    PAPER_CATEGORY_ORDER,
    TASK_DESCRIPTOR_MAP,
    TaskVariant,
    canonical_category,
    canonical_task_name,
    category_display,
    resolve_task_class,
)


POLICIES = ("expert_full_state", "expert")

# For these categories the unprivileged expert's rollout is the same scripted
# rollout as expert_full_state; it is made blind by hiding the avatar's live
# body from planning (scoring still sees the avatar).
BLIND_PLANNING_CATEGORIES = frozenset({"copresence"})

# Intent tasks: the unprivileged expert never waits for the human's cue.  Its
# play_blind_once runs the robot script from reset while the human runs the
# task's eval-mode hooks, ticked at policy cadence.  The robot plans on the
# avatar body and objects as observed at plan time (not play_once with a
# hidden avatar, which would still wait on the scripted cue).
BLIND_INTENT_TASKS = frozenset({
    "deliver_to_human",
    "oil_bottle_recovery",
    "open_microwave",
    "pour_water",
    "feed_documents",
    "take_from_human_easy",
    "take_from_human_safety",
})

# Intent tasks whose eval-mode human acts at a random step during the policy
# (no pre-policy cue).  Their full-state expert runs in eval mode against
# that same human: play_full_eval_once, the blind robot script plus a wait
# for the human's cue.  The other intent tasks keep the scripted play_once.
FULL_EVAL_INTENT_TASKS = frozenset({
    "deliver_to_human",
    "oil_bottle_recovery",
    "feed_documents",
    "take_from_human_safety",
})

# Contribution tasks whose blind-expert human starts at a random step from
# the task's trigger window instead of at reset.  Empty: in every
# Contribution task the human shows its intent before the robot can commit,
# so the blind expert faces the same at-reset human as the policies.  The
# opt-in (``eval_assist_trigger_delayed``) stays available for ablations.
DELAYED_ASSIST_TRIGGER_TASKS: frozenset[str] = frozenset()


# Co-presence rows: both experts run in eval mode against the
# policies' human (the eval trigger step 0-5, ticked once per policy step,
# same layout/anchor), instead of the scripted play_once human with its demo
# start delay.  The robot script is unchanged; the blind expert additionally
# hides the avatar body from planning.
NEUTRAL_EVAL_CATEGORIES = frozenset({"copresence"})


def is_neutral_eval_row(episode: dict[str, Any]) -> bool:
    return canonical_category(episode.get("category")) in NEUTRAL_EVAL_CATEGORIES


def is_interrupt_task(task_name: str) -> bool:
    """Interruption task, by registry variant (pre-HARP names accepted)."""
    descriptor = TASK_DESCRIPTOR_MAP.get(task_name)
    return descriptor is not None and descriptor.variant is TaskVariant.INTERRUPTION


# Tasks whose blind expert may fall back to the parent class's play_once: the
# former ``*_interrupt`` / ``*_assist`` names (so not categorize contribution,
# whose pre-HARP name was ``categorize_cooperative``).
PARENT_PLAY_FALLBACK_TASKS = frozenset(
    name
    for name, (family, category) in BENCHMARK_TASKS.items()
    if category == "interruption"
    or (category == "contribution" and family != "categorize")
)


def hides_avatar_from_planning(episode: dict[str, Any], policy: str) -> bool:
    return (
        policy == "expert"
        and canonical_category(episode.get("category")) in BLIND_PLANNING_CATEGORIES
    )


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper around the versioned manifest loader."""
    return load_eval_set_manifest(path)


# Optional CLI renderer override (--renderer). None = use each row's renderer.
# main() also exports it through this environment variable so that worker
# processes started with the "spawn" context (which re-import this module and
# do not see the parent's globals) apply the same override.
_RENDERER_OVERRIDE_ENV = "GENESIS_HR_EVAL_RENDERER_OVERRIDE"
_RENDERER_OVERRIDE: str | None = os.environ.get(_RENDERER_OVERRIDE_ENV) or None
RENDERER_CHOICES = ("nyx", "rasterizer")


def build_config(episode: dict[str, Any], policy: str) -> dict[str, Any]:
    cfg = deepcopy(episode.get("config_overrides", {}))
    for key in ("renderer", "setting", "action_type", "action_substeps"):
        if key in episode:
            cfg[key] = episode[key]
    if _RENDERER_OVERRIDE:
        cfg["renderer"] = _RENDERER_OVERRIDE
    # Same default as the VLA harness (scripts/vla_train37/run_batch.py):
    # rows without random_object run the collection setting (false).  Some
    # tasks default it to true in code (place_food_in_skillet_copresence),
    # which would give the experts a different food than the policies; the
    # eval_v3_sampled rows pin the key explicitly.
    cfg.setdefault("random_object", False)
    if is_neutral_eval_row(episode):
        cfg["eval_mode"] = True
        cfg["neutral_eval_human_from_ticks"] = True
        # The full-state expert waits before a segment that would meet the
        # human's upcoming clip poses; the blind expert never does.  A row may
        # set it false to run the snapshot-only full-state expert.
        cfg["neutral_human_aware_expert"] = (
            policy == "expert_full_state"
            and bool(cfg.get("neutral_human_aware_expert", True))
        )
    elif (policy == "expert_full_state"
            and not is_interrupt_task(episode.get("task", ""))
            and canonical_task_name(episode.get("task", "")) not in FULL_EVAL_INTENT_TASKS):
        # Interrupt tasks run both experts in eval mode under the interrupt
        # contract (same trigger and human as the policies).  Every other
        # category keeps the scripted play_once choreography.
        cfg["eval_mode"] = False
    return cfg


def resolve_for_policy(task_name: str, policy: str):
    resolved_name, task_cls = resolve_task_class(task_name, no_human=False)
    return resolved_name, task_cls, False, None


def _unprivileged_parent_play_once(task_cls):
    """Return a parent play_once for blind interrupt expert fallback, if any."""
    for cls in task_cls.__mro__[1:]:
        if cls.__name__ == "BaseTask":
            break
        if cls.__module__.endswith("inspect_motion_mixin"):
            continue
        play_once = cls.__dict__.get("play_once")
        if play_once is not None:
            return play_once
    return None


def policy_play_mode(task_name: str, task_cls, policy: str) -> tuple[str, object | None]:
    if policy == "expert_full_state":
        if is_interrupt_task(task_name):
            if not hasattr(task_cls, "interrupt_expert_attempt"):
                raise TypeError(
                    f"{task_name} lacks interrupt_expert_attempt(); every "
                    "interrupt task must provide the eval-mode full-state expert"
                )
            return "interrupt_full_state", None
        if canonical_task_name(task_name) in FULL_EVAL_INTENT_TASKS:
            if not hasattr(task_cls, "play_full_eval_once"):
                raise TypeError(f"{task_name} lacks play_full_eval_once()")
            return "play_full_eval_once", None
        return "play_once", None
    if policy != "expert":
        return "play_once", None
    if hasattr(task_cls, "play_blind_once"):
        return "play_blind_once", None
    if canonical_task_name(task_name) in PARENT_PLAY_FALLBACK_TASKS:
        parent_play = _unprivileged_parent_play_once(task_cls)
        if parent_play is not None:
            return "parent_play_once", parent_play
    return "play_once", None


def configure_parent_rollout_eval(cfg: dict, task_name: str, play_mode: str) -> None:
    """Harness-side eval config for scripted rollouts.

    Interrupt tasks get no overrides: the progress trigger belongs to the
    task, so both experts face exactly the trigger the policies face.
    """
    if play_mode not in {"parent_play_once", "play_blind_once", "play_full_eval_once"}:
        return
    task_name = canonical_task_name(task_name)
    if task_name in BLIND_INTENT_TASKS:
        cfg.setdefault("eval_avatar_pre_policy", False)
        cfg.setdefault("eval_force_policy_ticks", True)
        # One human tick per VLA policy step (action_substeps sim steps).
        cfg.setdefault("eval_parent_rollout_tick_interval",
                       int(cfg.get("action_substeps", 25)))
        return
    if task_name in DELAYED_ASSIST_TRIGGER_TASKS:
        cfg.setdefault("eval_assist_trigger_delayed", True)
        cfg.setdefault("eval_force_policy_ticks", True)


def rollout_tick_interval(cfg: dict) -> int:
    """Sim steps per eval tick for scripted rollouts: the policy cadence
    (``action_substeps``, 25 = 20 Hz), unless explicitly overridden."""
    value = cfg.get("eval_parent_rollout_tick_interval")
    if value is None:
        value = cfg.get("action_substeps", 25)
    return max(1, int(value))


def run_with_eval_step_driver(task, fn, cfg: dict) -> bool:
    """Drive eval-mode avatar hooks during scripted parent rollouts.

    Parent play_once implementations call step_sim directly, not take_action.
    The eval-mode human still ticks once per ``action_substeps`` sim steps,
    the cadence a policy sees.
    """
    driver = ScriptedRolloutDriver(
        task,
        tick_interval=rollout_tick_interval(cfg),
        force_policy_ticks=bool(cfg.get("eval_force_policy_ticks", False)),
    )
    with driver:
        return bool(fn())


def run_neutral_eval(task, cfg: dict) -> bool:
    """Scripted neutral rollout facing the eval-mode human.

    The policies start the neutral human from ``take_action`` (one
    ``advance_eval_step`` per policy step, before the action).  The scripted
    robot calls ``step_sim`` directly, so this wraps it: one tick before the
    first sim step, then one per ``action_substeps`` sim steps.  (The
    controller's ``on_scripted_rollout_tick`` hook stays the base no-op; the
    rollout-dispatch contract test freezes it.)
    """
    controller = task._neutral_controller()
    interval = rollout_tick_interval(cfg)
    orig = task.step_sim
    count = {"sim": 0}

    def driven_step_sim(*a, **k):
        out = orig(*a, **k)
        count["sim"] += 1
        if count["sim"] % interval == 0:
            controller.advance_eval_step(task)
        return out

    # Install the driver BEFORE the first tick.  When the human's trigger
    # step is 0 or 1 that tick starts the human, which wraps whatever
    # ``task.step_sim`` is current with its prop-handling patch; installing
    # the driver afterwards overwrote that patch (task audit X2: the human
    # lost prop handling on ~1/3 of seeds in both expert columns).  The
    # human's patch now chains to this driver, which chains to the task's
    # original step_sim.
    task.step_sim = driven_step_sim
    try:
        controller.advance_eval_step(task)
        return bool(task.play_once())
    finally:
        task.step_sim = orig


def run_interrupt_full_state(task, cfg: dict) -> bool:
    """Full-state expert under the interrupt contract (yields on the
    privileged fired flag, waits for the human, resumes)."""
    driver = ScriptedRolloutDriver(
        task,
        tick_interval=rollout_tick_interval(cfg),
        force_policy_ticks=True,
    )
    with driver:
        return run_full_state_interrupt_expert(task, driver)


def _hold_action(task):
    """qpos_abs action that holds every arm at its measured qpos and its
    current gripper value (the screening hold-still policy's action)."""
    parts = []
    arms = ("left", "right") if task.robot.left_arm is not None else ("right",)
    for tag in arms:
        arm = task.robot.get_arm(tag)
        parts.extend(np.asarray(arm.get_arm_qpos(), dtype=np.float64).ravel().tolist())
        parts.append(float(arm.gripper_val))
    return np.asarray(parts, dtype=np.float64)


def post_play_success_hold(task, cfg: dict) -> dict[str, Any]:
    """The policies' success rule (envs/success_hold.py) for a scripted
    expert: after its rollout the robot holds still for N policy steps
    (``take_action``, so the human keeps ticking as it does for a policy)
    and the success predicate must hold on every one of them.

    ``eval_success_hold_measure_steps`` (config) holds longer, for
    measurement only; the verdict still uses the first N steps.
    """
    n = success_hold_steps(cfg)
    total = max(n, int(cfg.get("eval_success_hold_measure_steps", 0) or 0))
    action = _hold_action(task)
    trace = []
    for _ in range(total):
        task.take_action(action, action_type="qpos_abs")
        trace.append(bool(task.evaluate().get("success", False)))
        if len(trace) >= n and not all(trace[:n]):
            break
    streak = 0
    for ok in trace:
        if not ok:
            break
        streak += 1
    return {
        "success_hold_steps": n,
        "success_hold_ok": bool(len(trace) >= n and all(trace[:n])),
        "success_hold_streak": streak,
        "success_hold_trace": trace,
    }


def run_episode(
    episode: dict[str, Any],
    policy: str,
    output_dir: Path,
    save_video: bool,
) -> dict[str, Any]:
    task_name = episode["task"]
    seed = int(episode["seed"])
    cfg = build_config(episode, policy)
    cfg["eval_seed"] = seed
    resolved_name, task_cls, no_human_applied, no_human_note = resolve_for_policy(task_name, policy)
    play_mode, parent_play_once = policy_play_mode(task_name, task_cls, policy)
    if is_neutral_eval_row(episode):
        play_mode = "neutral_eval_play_once"
    if play_mode == "neutral_eval_play_once":
        pass  # build_config already set eval_mode + neutral_eval_human_from_ticks
    elif play_mode == "interrupt_full_state":
        cfg["eval_mode"] = True
    elif play_mode in {"play_blind_once", "parent_play_once", "play_full_eval_once"}:
        cfg["eval_mode"] = True
        configure_parent_rollout_eval(cfg, task_name, play_mode)
    elif policy == "expert":
        # Fallback experts call play_once(); keep that path internally
        # consistent with scripted calibration instead of mixing it with
        # eval-mode avatar state.
        cfg["eval_mode"] = False

    if no_human_applied:
        cfg["no_human"] = True
        cfg["randomize_avatar"] = False
        cfg["track_avatar_collision"] = False
        cfg["avatar_retreat_enabled"] = False
        cfg["eval_mode"] = False

    video_path = None
    if save_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{episode['id']}__{policy}.mp4"

    t0 = time.time()
    result = {
        "id": episode["id"],
        "eval_set": episode.get("eval_set"),
        "category": episode.get("category"),
        "task": task_name,
        "resolved_task": resolved_name,
        "seed": seed,
        "policy": policy,
        "no_human_applied": no_human_applied,
        "no_human_note": no_human_note,
        "avatar_hidden_from_planning": hides_avatar_from_planning(episode, policy),
        "play_mode": play_mode,
        "success": False,
        "play_success": False,
        "elapsed_s": None,
        "error": None,
        "metrics": {},
        "video": str(video_path) if video_path else None,
    }

    task = None
    try:
        task = task_cls(cfg)
        task.reset(seed=seed)
        if hides_avatar_from_planning(episode, policy):
            task.hide_avatar_from_planning()
        if video_path is not None:
            task.start_video(str(video_path))
        if play_mode == "neutral_eval_play_once":
            play_success = run_neutral_eval(task, cfg)
        elif play_mode == "interrupt_full_state":
            play_success = run_interrupt_full_state(task, cfg)
        elif play_mode == "play_blind_once" and (
            cfg.get("eval_force_policy_ticks") or is_interrupt_task(task_name)
        ):
            play_success = run_with_eval_step_driver(task, task.play_blind_once, cfg)
        elif play_mode == "play_blind_once":
            play_success = bool(task.play_blind_once())
        elif play_mode == "play_full_eval_once":
            play_success = run_with_eval_step_driver(task, task.play_full_eval_once, cfg)
        elif play_mode == "parent_play_once":
            play_success = run_with_eval_step_driver(
                task, lambda: parent_play_once(task), cfg,
            )
        else:
            play_success = bool(task.play_once())
        if not play_success:
            task.plan_success = False
        metrics = task.evaluate()
        hold = None
        if metrics.get("success", False):
            hold = post_play_success_hold(task, cfg)
            metrics = task.evaluate()
            metrics["success"] = bool(hold["success_hold_ok"])
            metrics["success_hold"] = hold
        if video_path is not None:
            task.save_video()
        if is_interrupt_task(task_name) and "interrupt" not in metrics:
            metrics["interrupt"] = interrupt_metrics(task)
        result["play_success"] = play_success
        result["success"] = bool(metrics.get("success", False))
        result["metrics"] = metrics
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.time() - t0
        task = None  # drop the scene reference for GC before the next episode
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    by_category = {}
    by_task = {}
    for key, bucket in (("category", by_category), ("task", by_task)):
        for r in results:
            name = r.get(key) or "unknown"
            item = bucket.setdefault(name, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] += 1
            item["success"] += int(bool(r.get("success")))
        for item in bucket.values():
            item["success_rate"] = item["success"] / item["total"] if item["total"] else 0.0
    return {
        "total": total,
        "success": success,
        "success_rate": success / total if total else 0.0,
        "by_category": by_category,
        "by_task": by_task,
        "errors": sum(1 for r in results if r.get("error")),
    }


def format_category_summary(summary: dict[str, Any]) -> list[str]:
    """Per-condition success lines using the paper's condition names.

    ``summary["by_category"]`` keeps the manifest ``category`` values as keys;
    only this console text shows the paper names.
    """
    by_category = summary.get("by_category", {})
    order = [c for c in PAPER_CATEGORY_ORDER if c in by_category]
    order += sorted(c for c in by_category if c not in PAPER_CATEGORY_ORDER)
    return [
        f"  {category_display(c)} ({c}): {by_category[c]['success']}/"
        f"{by_category[c]['total']} = {by_category[c]['success_rate']:.3f}"
        for c in order
    ]


def write_outputs(policy: str, output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in results),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "task",
                "resolved_task",
                "seed",
                "policy",
                "no_human_applied",
                "success",
                "play_success",
                "elapsed_s",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(
        f"{policy}: {summary['success']}/{summary['total']} "
        f"= {summary['success_rate']:.3f}; errors={summary['errors']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("eval_sets/eval_v0_smoke"),
        help="Eval-set directory or manifest.jsonl path.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        action="append",
        default=None,
        help="Policy to evaluate. Repeat to run more than one. Default: expert_full_state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results (one subdirectory per policy). Default: "
             "runs/expert_eval_set/<eval-set name>, plus a "
             "shard_<start>_<end> suffix when --start-index/--end-index select "
             "a shard.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save debug videos.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N episodes.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based manifest row to start from (inclusive). Use with "
             "--end-index to split an eval set into shards that run in "
             "parallel, e.g. rows 0-500, 500-1000, ...",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based manifest row to stop at (exclusive). Default: end of "
             "the eval set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of episode worker processes per policy. Default: 1.",
    )
    parser.add_argument(
        "--renderer",
        choices=RENDERER_CHOICES,
        default=None,
        help="Override the renderer of every row. Default: use each row's "
             "manifest value (eval_v3_sampled rows use 'nyx', which needs "
             "GENESIS_BACKEND=gpu on a Vulkan-capable GPU). Pass 'rasterizer' "
             "(with GENESIS_BACKEND=cpu) for a CPU-only expert check.",
    )
    parser.add_argument(
        "--subprocess-per-episode",
        choices=("auto", "on", "off"),
        default="auto",
        help="Run every episode in a fresh Python process. NYX can build only "
             "one scene per process, so 'auto' (default) turns this on when "
             "more than one episode would render with NYX on the GPU backend.",
    )
    parser.add_argument("--episode-result-json", type=Path, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def eval_set_name(path: Path) -> str:
    """Eval-set name for default output paths: the set's directory name."""
    path = Path(path)
    return (path.parent if path.suffix == ".jsonl" else path).name or "eval_set"


def default_output_dir(eval_set: Path, start_index: int, end_index: int | None) -> Path:
    out = Path("runs/expert_eval_set") / eval_set_name(eval_set)
    if start_index or end_index is not None:
        out = out / f"shard_{start_index}_{'end' if end_index is None else end_index}"
    return out


def renderer_notice(episodes: list[dict[str, Any]], override: str | None) -> str | None:
    """One-line notice when NYX rows will silently run on the rasterizer."""
    if override:
        return None
    wants_nyx = any(
        str(ep.get("renderer") or ep.get("config_overrides", {}).get("renderer") or "").lower()
        == "nyx"
        for ep in episodes
    )
    backend = (os.environ.get("GENESIS_BACKEND") or "").strip().lower()
    if wants_nyx and backend == "cpu":
        return (
            "NOTICE: rows request renderer 'nyx' but GENESIS_BACKEND=cpu; NYX "
            "needs the GPU backend, so these rows fall back to the rasterizer "
            "(CPU physics). Set GENESIS_BACKEND=gpu on a Vulkan-capable GPU for "
            "the reference NYX results, or pass --renderer rasterizer to make "
            "the fallback explicit."
        )
    return None



def uses_nyx(episodes: list[dict[str, Any]], override: str | None) -> bool:
    """Whether any selected row would render with NYX on the GPU backend."""
    if (os.environ.get("GENESIS_BACKEND") or "").strip().lower() == "cpu":
        return False
    if override:
        return override == "nyx"
    return any(
        str(ep.get("renderer") or ep.get("config_overrides", {}).get("renderer") or "").lower()
        == "nyx"
        for ep in episodes
    )


def run_episode_in_subprocess(
    args: argparse.Namespace,
    row_index: int,
    episode: dict[str, Any],
    policy: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one manifest row in a fresh interpreter and return its result."""
    result_dir = output_dir / policy / "episodes"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"row_{row_index:05d}.json"
    result_path.unlink(missing_ok=True)
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--eval-set", str(args.eval_set), "--policy", policy,
        "--start-index", str(row_index), "--end-index", str(row_index + 1),
        "--output-dir", str(output_dir), "--subprocess-per-episode", "off",
        "--episode-result-json", str(result_path),
    ]
    if args.renderer:
        cmd += ["--renderer", args.renderer]
    if args.save_video:
        cmd.append("--save-video")
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode == 0 and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    return {
        "id": episode.get("id"),
        "eval_set": episode.get("eval_set"),
        "category": episode.get("category"),
        "task": episode.get("task"),
        "resolved_task": None,
        "seed": episode.get("seed"),
        "policy": policy,
        "success": False,
        "play_success": False,
        "elapsed_s": time.time() - t0,
        "error": f"episode subprocess exited with code {proc.returncode}\n{tail}",
        "metrics": {},
    }

def main() -> None:
    args = parse_args()
    from envs.debug_switches import refuse_debug_switches
    refuse_debug_switches("scripts/eval_expert_set.py")
    global _RENDERER_OVERRIDE
    _RENDERER_OVERRIDE = args.renderer
    if args.renderer:
        os.environ[_RENDERER_OVERRIDE_ENV] = args.renderer
    else:
        os.environ.pop(_RENDERER_OVERRIDE_ENV, None)
    policies = args.policy or ["expert_full_state"]
    episodes = load_eval_set(args.eval_set)
    start_index = max(0, int(args.start_index))
    end_index = args.end_index if args.end_index is None else max(start_index, int(args.end_index))
    episodes = episodes[start_index:end_index]
    if args.limit is not None:
        episodes = episodes[: max(0, args.limit)]
    output_dir = args.output_dir or default_output_dir(args.eval_set, start_index, end_index)
    print(
        f"Loaded {len(episodes)} eval-set initializations from {args.eval_set} "
        f"(start_index={start_index}, end_index={end_index}); "
        f"renderer={args.renderer or 'per row'}; output_dir={output_dir}"
    )
    notice = renderer_notice(episodes, args.renderer)
    if notice:
        print(notice, flush=True)

    if args.episode_result_json is not None:
        # Child of --subprocess-per-episode: run the one row and hand back its
        # result; the parent writes the results files.
        if len(episodes) != 1 or len(policies) != 1:
            raise SystemExit("--episode-result-json needs exactly one row and one policy")
        result = run_episode(episodes[0], policies[0], output_dir / policies[0],
                             save_video=args.save_video)
        args.episode_result_json.write_text(json.dumps(result, default=str), encoding="utf-8")
        return

    fresh = args.subprocess_per_episode == "on" or (
        args.subprocess_per_episode == "auto"
        and len(episodes) > 1
        and uses_nyx(episodes, args.renderer)
    )
    if fresh:
        print("Running each episode in a fresh process (NYX builds one scene per process).",
              flush=True)

    for policy in policies:
        policy_dir = output_dir / policy
        results = []
        workers = max(1, int(args.workers))
        if args.save_video:
            workers = 1
        if fresh:
            indexed = list(enumerate(episodes, start=1))
            completed = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_episode_in_subprocess, args, start_index + i - 1,
                                    episode, policy, output_dir): i
                    for i, episode in indexed
                }
                for future in as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    completed[i] = result
                    status = "SUCCESS" if result["success"] else "FAIL"
                    if result.get("error"):
                        status = "ERROR"
                    print(
                        f"[{policy}] {i}/{len(episodes)} "
                        f"{result['task']} seed={result['seed']} "
                        f"{status} elapsed={result['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    if result.get("error"):
                        print(result["error"], flush=True)
                    results = [completed[j] for j in sorted(completed)]
                    write_outputs(policy, policy_dir, results)
        elif workers == 1 or len(episodes) <= 1:
            for i, episode in enumerate(episodes, start=1):
                print(
                    f"[{policy}] {i}/{len(episodes)} "
                    f"{episode['task']} seed={episode['seed']}",
                    flush=True,
                )
                result = run_episode(episode, policy, policy_dir, save_video=args.save_video)
                results.append(result)
                status = "SUCCESS" if result["success"] else "FAIL"
                if result.get("error"):
                    status = "ERROR"
                print(f"  {status} elapsed={result['elapsed_s']:.1f}s", flush=True)
                if result.get("error"):
                    print(result["error"], flush=True)
                write_outputs(policy, policy_dir, results)
        else:
            print(f"[{policy}] running with workers={workers}", flush=True)
            indexed = list(enumerate(episodes, start=1))
            ctx = get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(run_episode, episode, policy, policy_dir, args.save_video): i
                    for i, episode in indexed
                }
                completed = {}
                for future in as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    completed[i] = result
                    status = "SUCCESS" if result["success"] else "FAIL"
                    if result.get("error"):
                        status = "ERROR"
                    print(
                        f"[{policy}] {i}/{len(episodes)} "
                        f"{result['task']} seed={result['seed']} "
                        f"{status} elapsed={result['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    if result.get("error"):
                        print(result["error"], flush=True)
                    results = [completed[j] for j in sorted(completed)]
                    write_outputs(policy, policy_dir, results)
        write_outputs(policy, policy_dir, results)
        for line in format_category_summary(summarize(results)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
