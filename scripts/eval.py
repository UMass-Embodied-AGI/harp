"""Policy evaluation: run a policy on a task and report success rate."""

import argparse
import hashlib
import os
import sys
import time
import numpy as np
import json
import traceback
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.success_hold import success_hold_steps as success_hold_steps_for
from envs.tasks import resolve_task_class
from envs.utils import load_config
from envs.scene_init import (
    episode_scene_init,
    load_scene_init_file,
    prepare_config_for_scene_init,
)


def _resolve_eval_task_class(task_name: str, config: dict):
    static_avatar = bool((config or {}).get("static_avatar", False))
    resolved_task_name, TaskClass = resolve_task_class(
        task_name,
        no_human=bool((config or {}).get("no_human", False) or static_avatar),
    )
    if static_avatar:
        from scripts.vla_data.collect_vla import with_static_avatar
        TaskClass = with_static_avatar(TaskClass)
    return resolved_task_name, TaskClass


def _array_digest(arr) -> dict:
    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _host_array(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _vec3(value):
    return np.asarray(_host_array(value), dtype=float).ravel()[:3].tolist()


def _merge_config(base: dict, override: dict | None) -> dict:
    out = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_config(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _summarize_policy_obs(obs: dict) -> dict:
    summary = {}
    if not isinstance(obs, dict):
        return summary
    rgb = obs.get("rgb")
    if isinstance(rgb, dict):
        summary["rgb"] = {
            str(k): _array_digest(v)
            for k, v in rgb.items()
            if v is not None
        }
    for key in (
        "qpos", "joint_pos", "gripper_val", "ee_pose", "state",
        "robot_state",
    ):
        if key in obs:
            try:
                summary[key] = np.asarray(obs[key], dtype=float).ravel().tolist()
            except Exception:
                pass
    return summary


def _reserve_video_path(video_dir: str, filename: str, max_videos: int | None) -> tuple[str | None, str | None]:
    if max_videos is None or max_videos < 0:
        return os.path.join(video_dir, filename), None

    os.makedirs(video_dir, exist_ok=True)
    lock_dir = os.path.join(video_dir, ".video_slots.lock")
    for _ in range(200):
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        return None, None

    marker = None
    try:
        videos = [name for name in os.listdir(video_dir) if name.endswith(".mp4")]
        reservations = [name for name in os.listdir(video_dir) if name.startswith(".video_reserve_")]
        if len(videos) + len(reservations) >= max_videos:
            return None, None
        marker = os.path.join(video_dir, f".video_reserve_{filename}")
        with open(marker, "w") as f:
            f.write(str(time.time()))
        return os.path.join(video_dir, filename), marker
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def _release_video_reservation(marker: str | None) -> None:
    if not marker:
        return
    try:
        os.remove(marker)
    except OSError:
        pass


class ExpertFullStatePolicy:
    """Privileged scripted expert used for data rollouts.

    This intentionally receives the task object instead of only observation
    tensors: the baseline is the full-state scripted policy, not a learned
    observation-only controller.
    """

    def reset(self):
        pass

    def run_episode(self, task, obs, max_steps: int, action_type: str):
        return bool(task.play_once()), None


def _should_save_video(ep_index: int, video_dir: str | None, video_limit: int | None) -> bool:
    if not video_dir:
        return False
    if video_limit is None:
        return True
    return ep_index < max(0, int(video_limit))


def _reset_failure_result(seed: int, elapsed_s: float, error: str) -> dict:
    return {
        "seed": seed,
        "success": False,
        "steps": 0,
        "elapsed_s": elapsed_s,
        "error": error,
        "stage_failed": "reset",
    }


def _scene_init_failure_result(
    seed: int,
    scene_init: dict | None,
    elapsed_s: float,
    error: str,
) -> dict:
    result = _reset_failure_result(seed, elapsed_s, error)
    if scene_init:
        result["scene_init_id"] = scene_init.get("id")
        result["scene_init_source_seed"] = scene_init.get("source_seed")
    return result


def _policy_metrics_dict(policy_fn):
    """Return serializable diagnostics from privileged baseline policies."""
    metrics = getattr(policy_fn, "last_metrics", None)
    if metrics is None:
        return None
    if hasattr(metrics, "as_dict"):
        return metrics.as_dict()
    if hasattr(metrics, "__dataclass_fields__"):
        out = {}
        for key in metrics.__dataclass_fields__:
            value = getattr(metrics, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, (list, tuple)):
                out[key] = list(value)
            elif isinstance(value, dict):
                out[key] = dict(value)
            elif isinstance(value, set):
                out[key] = sorted(value)
            else:
                out[key] = repr(value)
        return out
    return None


def _print_policy_metrics(policy_name: str, policy_fn):
    metrics = _policy_metrics_dict(policy_fn)
    if not metrics:
        return
    keys = (
        "success",
        "steps",
        "fallback_used",
        "time_progress_fallback_used",
        "first_assist_step",
        "robot_idle_time",
        "human_idle_time",
        "ticks",
        "skill_calls",
        "successes",
        "failures",
        "recoveries",
    )
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]}")
    if not parts:
        parts.append(f"keys={sorted(metrics)}")
    print(f"  {policy_name}-metrics: " + ", ".join(parts))


def _privileged_policy_step_count(policy_fn):
    """Best-effort episode step count for privileged full-state baselines."""
    last_metrics = getattr(policy_fn, "last_metrics", None)
    if last_metrics is not None:
        steps = getattr(last_metrics, "steps", None)
        if steps:
            return int(steps)
        events = getattr(last_metrics, "events", None)
        if events:
            if isinstance(events, (list, tuple, set, dict)):
                return len(events)
            return int(events)
        event_log = getattr(last_metrics, "event_log", None)
        if event_log:
            return len(event_log)

    last_context = getattr(policy_fn, "last_context", None)
    steps = getattr(last_context, "step_count", None) if last_context else None
    return int(steps) if steps else None


def eval_policy(
    task_name: str,
    config: dict,
    policy_fn,
    num_episodes: int,
    max_steps: int = 300,
    start_seed: int = 0,
    action_type: str = "qpos",
    video_dir: str = None,
    video_limit: int | None = None,
    return_details: bool = False,
    scene_inits: list[dict] | None = None,
    scene_init_start_index: int = 0,
):
    """Evaluate a policy on a task.

    Args:
        task_name: task name from TASK_MAP
        config: task config dict
        policy_fn: callable(obs) -> action
        num_episodes: number of evaluation episodes
        max_steps: max steps per episode
        start_seed: starting seed
        action_type: "qpos" or "ee"
        video_dir: if set, write one MP4 per episode to this directory
        video_limit: when video_dir is set, save videos for only the first N episodes

    Returns:
        success_rate: float
    """
    success_count = 0
    episode_results = []

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    max_videos_per_dir = config.get("max_videos_per_dir")
    if max_videos_per_dir is not None:
        max_videos_per_dir = int(max_videos_per_dir)

    for ep in range(num_episodes):
        seed = start_seed + ep
        scene_init_entry = None
        scene_init = None
        if scene_inits is not None:
            idx = int(scene_init_start_index) + ep
            if idx >= len(scene_inits):
                raise IndexError(
                    f"Scene-init index {idx} out of range for {len(scene_inits)} entries"
                )
            scene_init_entry = scene_inits[idx]
            ep_base_config = _merge_config(config, scene_init_entry.get("config_overrides"))
            resolved_task_name, TaskClass = _resolve_eval_task_class(
                task_name,
                ep_base_config,
            )
            entry_task = scene_init_entry.get("task")
            if entry_task and entry_task not in {task_name, resolved_task_name}:
                raise ValueError(
                    f"Scene-init task mismatch at index {idx}: "
                    f"{entry_task!r} not in {{{task_name!r}, {resolved_task_name!r}}}"
                )
            scene_init = episode_scene_init(scene_init_entry)
            seed = int(scene_init.get("source_seed", scene_init_entry.get("seed", seed)))
        else:
            ep_base_config = config
            resolved_task_name, TaskClass = _resolve_eval_task_class(
                task_name,
                ep_base_config,
            )
        label = f"scene_init={scene_init.get('id')}" if scene_init else f"seed={seed}"
        print(f"\nEpisode {ep + 1}/{num_episodes} ({label})")

        reset_t0 = time.time()
        try:
            ep_config = prepare_config_for_scene_init(ep_base_config, scene_init)
            task = TaskClass(ep_config)
            obs = task.reset(seed=seed)
        except Exception:
            elapsed = time.time() - reset_t0
            err = traceback.format_exc()
            print(f"  ERROR during reset ({elapsed:.1f}s)")
            print(err, flush=True)
            episode_results.append(_scene_init_failure_result(seed, scene_init, elapsed, err))
            continue

        save_episode_video = _should_save_video(ep, video_dir, video_limit)
        video_marker = None
        video_path = None
        if save_episode_video:
            video_path, video_marker = _reserve_video_path(
                video_dir,
                f"{resolved_task_name}_seed_{seed}.mp4",
                max_videos_per_dir,
            )
        if video_path:
            task.start_video(video_path)

        if ep_config.get("eval_avatar_pre_policy", True) and hasattr(task, "eval_pre_policy_warmup"):
            print("  pre-policy warmup: begin", flush=True)
            maybe_obs = task.eval_pre_policy_warmup()
            obs = maybe_obs if maybe_obs is not None else task.get_obs()
            print("  pre-policy warmup: done", flush=True)

        policy_name = getattr(policy_fn, "baseline_name", None)
        if policy_name:
            print(f"  warning: ignoring removed privileged baseline policy {policy_name!r}")

        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        # Feed the per-episode instruction to the policy: tasks that sample
        # their target object each reset (object catalog) template the object
        # name into task.instruction, which must reach the language-conditioned
        # model for this episode.
        if hasattr(policy_fn, "set_instruction"):
            policy_fn.set_instruction(getattr(task, "instruction", None))


        t0 = time.time()
        step = 0
        episode_error = None
        # Success rule (envs/success_hold.py): the predicate must hold for
        # ``success_hold_steps`` consecutive policy steps.  None = the loop
        # below did not evaluate per step (privileged run_episode policies,
        # or stop-on-success disabled): the final state decides alone.
        success_hold_steps = None
        consecutive_success = 0
        if hasattr(policy_fn, "run_episode"):
            try:
                play_success, obs = policy_fn.run_episode(
                    task, obs, max_steps=max_steps, action_type=action_type,
                )
                policy_steps = _privileged_policy_step_count(policy_fn)
                step = max(0, int(policy_steps) - 1) if policy_steps else 0
                if not play_success:
                    task.plan_success = False
            except Exception:
                episode_error = traceback.format_exc()
                print(f"  ERROR during rollout ({time.time() - t0:.1f}s)")
                print(episode_error, flush=True)
                task.plan_success = False
        else:
            trace_rows = []
            trace_dir = ep_config.get("trace_actions_dir")
            trace_eval_metrics = bool(ep_config.get("trace_eval_metrics", False))
            stop_on_success = bool(ep_config.get("eval_stop_on_success", True))
            success_hold_steps = success_hold_steps_for(ep_config)
            consecutive_success = 0
            last_action = None
            progress_interval = int(ep_config.get("debug_eval_progress_interval", 0) or 0)
            for step in range(max_steps):
                if progress_interval > 0 and step % progress_interval == 0:
                    print(f"  step {step}/{max_steps}", flush=True)
                if (
                    hasattr(task, "consume_eval_policy_reset_requested")
                    and task.consume_eval_policy_reset_requested()
                    and hasattr(policy_fn, "reset")
                ):
                    policy_fn.reset()
                skip_policy = (
                    hasattr(task, "eval_skip_policy_action")
                    and task.eval_skip_policy_action()
                )
                try:
                    action_source = "policy"
                    if skip_policy:
                        if last_action is None:
                            action_dim = int(getattr(policy_fn, "action_dim", 8))
                            last_action = np.zeros(action_dim, dtype=np.float64)
                            action = np.asarray(last_action, dtype=np.float64).copy()
                            action_source = "zero_hold"
                        else:
                            action = np.asarray(last_action, dtype=np.float64).copy()
                            action_source = "last_action_hold"
                    else:
                        action = policy_fn(obs)
                        last_action = np.asarray(action, dtype=np.float64).copy()
                except Exception:
                    episode_error = traceback.format_exc()
                    print(f"  ERROR during policy step {step} ({time.time() - t0:.1f}s)")
                    print(episode_error, flush=True)
                    task.plan_success = False
                    break
                if trace_dir:
                    row = {
                        "step": int(step),
                        "policy_skipped": bool(skip_policy),
                        "action_source": action_source,
                        "eval_policy_step_count": int(getattr(task, "_policy_step_count", -1)),
                        "eval_interrupt_fired": bool(getattr(task, "_eval_interrupt_fired", False)),
                        "eval_glide_active": bool(getattr(task, "_eval_glide_active", False)),
                        "eval_retreat_active": bool(getattr(task, "_eval_retreat_active", False)),
                        "eval_return_active": bool(getattr(task, "_eval_return_active", False)),
                        "eval_waiting_avatar_done": bool(getattr(task, "_eval_waiting_avatar_done", False)),
                        "eval_pending_interrupt_idx": getattr(task, "_eval_pending_interrupt_idx", None),
                        "pace_trigger_step": getattr(task, "_pace_trigger_step", None),
                        "pace_trigger_reason": getattr(task, "_pace_trigger_reason", None),
                        "policy_obs_before": _summarize_policy_obs(obs),
                        "action": np.asarray(action, dtype=float).ravel().tolist(),
                        "ee_pose_before": np.asarray(
                            task.robot.right_arm.get_ee_pose(), dtype=float,
                        ).ravel().tolist(),
                        "qpos_before": np.asarray(
                            task.robot.right_arm.get_arm_qpos(), dtype=float,
                        ).ravel().tolist(),
                        "gripper_before": float(task.robot.right_arm.gripper_val),
                    }
                    if hasattr(task, "objects"):
                        row["object_positions_before"] = [
                            _vec3(task._object_pos(obj))
                            if hasattr(task, "_object_pos")
                            else _vec3(obj.get_pos())
                            for obj in getattr(task, "objects", [])
                        ]
                    if hasattr(task, "bowls"):
                        row["bowl_positions_before"] = [
                            _vec3(bowl.entity.get_pos())
                            for bowl in getattr(task, "bowls", [])
                            if bowl is not None
                        ]
                    if hasattr(task, "object") and getattr(task, "object") is not None:
                        entity = getattr(task.object, "entity", task.object)
                        row["object_position_before"] = _vec3(entity.get_pos())
                try:
                    obs = task.take_action(action, action_type=action_type)
                except Exception:
                    episode_error = traceback.format_exc()
                    print(f"  ERROR during env step {step} ({time.time() - t0:.1f}s)")
                    print(episode_error, flush=True)
                    task.plan_success = False
                    break
                if trace_dir:
                    row["qpos_after"] = np.asarray(
                        task.robot.right_arm.get_arm_qpos(), dtype=float,
                    ).ravel().tolist()
                    row["ee_pose_after"] = np.asarray(
                        task.robot.right_arm.get_ee_pose(), dtype=float,
                    ).ravel().tolist()
                    row["gripper_after"] = float(task.robot.right_arm.gripper_val)
                    if getattr(task, "target", None) is not None:
                        metrics_now = task.evaluate()
                        for key in (
                            "target_dist_xy",
                            "target_dist_3d",
                            "target_dz",
                            "target_tilt_deg",
                        ):
                            if key in metrics_now:
                                row[key] = float(metrics_now[key])
                    if hasattr(task, "objects"):
                        row["object_positions_after"] = [
                            _vec3(task._object_pos(obj))
                            if hasattr(task, "_object_pos")
                            else _vec3(obj.get_pos())
                            for obj in getattr(task, "objects", [])
                        ]
                    if hasattr(task, "bowls"):
                        row["bowl_positions_after"] = [
                            _vec3(bowl.entity.get_pos())
                            for bowl in getattr(task, "bowls", [])
                            if bowl is not None
                        ]
                    if hasattr(task, "object") and getattr(task, "object") is not None:
                        entity = getattr(task.object, "entity", task.object)
                        row["object_position_after"] = _vec3(entity.get_pos())
                    if trace_eval_metrics:
                        step_metrics = task.evaluate()
                        for key, value in step_metrics.items():
                            if isinstance(value, (bool, int, float, str)) or value is None:
                                row[f"metric_{key}"] = value
                    trace_rows.append(row)

                if task.eval_success:
                    break
                if stop_on_success:
                    step_metrics = task.evaluate()
                    if step_metrics.get("success", False):
                        consecutive_success += 1
                        if consecutive_success >= success_hold_steps:
                            task.eval_success = True
                            print(
                                f"  early-stop: success stable for "
                                f"{consecutive_success} step(s) at step={step + 1}",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_success = 0
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(
                    trace_dir,
                    f"{resolved_task_name}_seed_{seed}_trace.json",
                )
                with open(trace_path, "w") as f:
                    json.dump(trace_rows, f, indent=2)
                print(f"  trace saved: {trace_path} ({len(trace_rows)} rows)")

        metrics = task.evaluate()
        if success_hold_steps is not None and stop_on_success:
            metrics["success_hold_steps"] = int(success_hold_steps)
            metrics["success_streak_steps"] = int(consecutive_success)
            if (metrics.get("success", False) and not task.eval_success
                    and consecutive_success < success_hold_steps):
                # Horizon reached while the predicate had held for fewer
                # than N steps (e.g. an object passing through the goal).
                metrics["success"] = False
                metrics["success_hold_unmet"] = True
        if episode_error is not None:
            metrics["stage_failed"] = "rollout"
        task_success = bool(metrics.get("success", False)) and episode_error is None
        policy_metrics = _policy_metrics_dict(policy_fn)
        if policy_metrics is not None:
            metrics["hrc_baseline"] = policy_metrics
        elapsed = time.time() - t0

        if video_path:
            task.save_video()
            _release_video_reservation(video_marker)

        _print_policy_metrics(getattr(policy_fn, "__class__", type(policy_fn)).__name__, policy_fn)

        if "target_dist_xy" in metrics:
            tilt = metrics.get("target_tilt_deg")
            tilt_s = f", tilt={tilt:.1f}°" if tilt is not None else ""
            print(f"  target[{metrics.get('target_label','?')}]: "
                  f"dist_xy={metrics['target_dist_xy']:.3f}m, "
                  f"dist_3d={metrics['target_dist_3d']:.3f}m, "
                  f"dz={metrics['target_dz']:+.3f}m{tilt_s}")
        # Per-episode avatar-collision report (if tracking enabled).
        coll = metrics.get("avatar_collision", {})
        if coll.get("enabled"):
            n = coll["n_checks"]; hits = coll["n_collisions"]
            dd = coll.get("deepest_depth_m")
            dd_str = f"{dd*100:+.1f} cm" if dd is not None else "—"
            print(f"  avatar-collision: {hits}/{n} checks hit, "
                  f"deepest={dd_str}, pair={coll.get('deepest_pair')}")
        safe_dist = metrics.get("safe_distance", {})
        if safe_dist.get("enabled"):
            sd = safe_dist.get("safe_distance_m")
            sd_str = f"{sd*100:+.1f} cm" if sd is not None else "—"
            print(f"  safe-distance: min={sd_str}, "
                  f"pair={safe_dist.get('closest_pair')}")
        intervention = metrics.get("avatar_safety_intervention", {})
        if intervention.get("enabled"):
            md = intervention.get("min_distance_m")
            md_str = f"{md*100:+.1f} cm" if md is not None else "—"
            print(f"  avatar-retreat: {intervention['n_interventions']} interventions, "
                  f"min_dist={md_str}")
        pace = metrics.get("pace_interrupt", {})
        if pace:
            dist = pace.get("trigger_dist_xy")
            dist_s = f"{dist:.3f}m" if dist is not None else "—"
            prog = pace.get("trigger_progress")
            prog_s = f"{prog:.2f}" if prog is not None else "—"
            print(f"  pace-interrupt: mode={pace.get('trigger_mode')}, "
                  f"fired={pace.get('fired')}, step={pace.get('trigger_step')}, "
                  f"reason={pace.get('trigger_reason')}, "
                  f"progress={prog_s}, dist_xy={dist_s}")
        handoff = metrics.get("eval_handoff_debug")
        if handoff is not None:
            print(f"  eval-handoff: step={handoff.get('step_idx')}, "
                  f"gripper={handoff.get('gripper'):.3f}, "
                  f"ee_apple_dist={handoff.get('ee_apple_dist'):.3f}m")
        elif "eval_apple_attached_to_avatar" in metrics:
            print(f"  eval-handoff: no detach, "
                  f"attached_to_avatar={metrics['eval_apple_attached_to_avatar']}")

        if task_success:
            success_count += 1
            print(f"  SUCCESS (step={step + 1}, {elapsed:.1f}s)")
        else:
            print(f"  FAIL (step={step + 1}, {elapsed:.1f}s)")
        detail = {
            "seed": seed,
            "success": task_success,
            "steps": step + 1,
            "elapsed_s": elapsed,
            "avatar_collision": metrics.get("avatar_collision"),
            "safe_distance": metrics.get("safe_distance"),
            "avatar_safety": metrics.get("avatar_safety"),
            "safety": metrics.get("safety"),
            "plan_success": metrics.get("plan_success"),
            "avatar_collision_success_veto": bool(
                (metrics.get("avatar_collision") or {}).get("any_collision")),
            "avatar_safety_intervention": metrics.get("avatar_safety_intervention"),
            "pace_interrupt": metrics.get("pace_interrupt"),
            # Eval-mode Interruption contract block (trigger + human sequence).
            "interrupt": metrics.get("interrupt"),
            "hrc_baseline": policy_metrics,
        }
        if scene_init:
            detail["scene_init_id"] = scene_init.get("id")
            detail["scene_init_source_seed"] = scene_init.get("source_seed")
            detail["scene_init_replay"] = getattr(task, "scene_init_replay", None)
        if episode_error is not None:
            detail["error"] = episode_error
        detail_keys = (
            "target_label",
            "target_dist_xy",
            "target_dist_3d",
            "target_dz",
            "target_tilt_deg",
            "object_name",
            "object_model_id",
            "grasp_method",
            "stage_failed",
            "grasp_used",
            "object_bottom_z",
            "object_table_clearance",
            "object_table_clearance_threshold",
            "object_still_held",
            "object_hold_metric_error",
            "object_in_drawer",
            "object_in_drawer_xy",
            "object_in_drawer_aabb_xy",
            "object_in_drawer_exposed",
            "object_in_drawer_z",
            "object_center",
            "drawer_center",
            "drawer_aabb_min",
            "drawer_aabb_max",
            "drawer_floor_z",
            "drawer_rim_z",
            "drawer_qpos",
            "cabinet_front_y",
            "ee_pose",
            "ee_in_drawer_aabb",
            "ee_to_drawer_center",
            "gripper_open_value",
            "door_angle",
            "max_door_angle",
            "success_open_angle",
            "robot_plate_collision",
            "robot_plate_collision_events",
            "robot_plate_min_clearance",
            "robot_blocks_load_path",
            "robot_load_path_clearance",
            "avatar_plate_attached",
            "avatar_plate_two_hand_attached",
            "avatar_plate_hand_distance",
            "avatar_plate_hand_max_dist",
            "avatar_plate_carried_ok",
            "eval_avatar_trigger_step",
            "eval_avatar_started",
            "eval_avatar_failed",
            # Co-presence: contested layout flag + anchor,
            # and when the eval human started its table work.
            "neutral_contested",
            "neutral_contested_requested",
            "neutral_contested_reason",
            "neutral_layout",
            "neutral_object_collided",
            "neutral_prop_disturbed",
            "neutral_eval_trigger_step",
            "neutral_avatar_started",
            "neutral_eval_avatar_failed",
            # Success-hold rule (envs/success_hold.py).
            "success_hold_steps",
            "success_streak_steps",
            "success_hold_unmet",
            # place_food_in_skillet_assist: human-initiated contact during
            # the human's drop with the robot holding still (not scored).
            "exempt_human_contact",
            # deliver_to_human_easy: handover (release at the palm, glide,
            # human holds) and the random-start human's timing.
            "handover",
            "blind_intent",
        )
        for key in detail_keys:
            if key in metrics:
                detail[key] = metrics[key]
        episode_results.append(detail)

    success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
    print(f"\nSuccess rate: {success_count}/{num_episodes} = {success_rate:.1%}")
    if return_details:
        return {
            "success_rate": success_rate,
            "successes": success_count,
            "episodes": num_episodes,
            "episode_results": episode_results,
        }
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy on a task")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--policy", type=str, default="random",
                        choices=[
                            "random",
                            "expert_full_state",
                            "expert",
                            "expert_no_human",
                        ],
                        help="Policy baseline. expert_full_state is the scripted "
                             "rollout policy with privileged task state; expert "
                             "runs the same scripted policy with the avatar "
                             "disabled. expert_no_human is kept as an alias.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--scene-init-file", type=str, default=None,
                        help="YAML/JSON/JSONL deterministic scene-init file. "
                             "When set, eval episodes are selected by index "
                             "instead of start-seed-only initialization.")
    parser.add_argument("--scene-init-start-index", type=int, default=0,
                        help="First scene-init entry to evaluate.")
    parser.add_argument("--renderer", type=str, default=None,
                        choices=["nyx", "rasterizer"],
                        help="Override the renderer. nyx needs the GPU backend on a "
                             "Vulkan-capable GPU (not compute-only A100/H100); "
                             "rasterizer also runs on CPU.")
    parser.add_argument("--no-human", action="store_true",
                        help="Collapse a Contribution/Interruption/Co-presence task "
                             "to its family's robot-only no-human task. Cueing tasks "
                             "do not support this flag.")
    parser.add_argument("--table-random-objects",
                        "--task-irrelevant-objects",
                        "--with-irrelevant-objects",
                        dest="table_random_objects", action="store_true",
                        help="Enable random background tabletop objects for tasks "
                             "that define decorative object regions. Default: disabled.")
    parser.add_argument("--hard", action="store_true",
                        help="Use the legacy multi-object task variant where supported.")
    parser.add_argument("--no-table-random-objects",
                        "--no-task-irrelevant-objects",
                        dest="no_table_random_objects",
                        action="store_true",
                        help="Disable random background tabletop objects, overriding config.")
    parser.add_argument("--track-avatar-collision", action="store_true",
                        help="Run the analytic robot↔avatar-capsule collision checker "
                             "and print per-episode summary. Physics untouched.")
    parser.add_argument("--collision-stride", type=int, default=None,
                        help="Collision check every N sim steps (default 30).")
    parser.add_argument("--collision-margin", type=float, default=None,
                        help="Flag within M metres of capsule surface (default 0).")
    parser.add_argument("--avatar-safety-margin", type=float, default=None,
                        help="Log inflated avatar safety envelope violations at this margin.")
    parser.add_argument("--avatar-retreat-enabled", action="store_true",
                        help="Override nominal robot motion with an up-and-away retreat "
                             "when the avatar safety threshold is breached.")
    parser.add_argument("--avatar-retreat-margin", type=float, default=None,
                        help="Unsafe distance threshold for active retreat. Defaults to "
                             "--avatar-safety-margin, then 0.10m.")
    parser.add_argument("--show-collider", action="store_true",
                        help="Render the bone cylinders — debug only.")
    parser.add_argument("--eval-mode", action="store_true",
                        help="Set config['eval_mode']=True. Tasks that distinguish "
                             "scripted vs eval flow (e.g. dump_bin_interruption) drive "
                             "the avatar via a per-step state machine instead of "
                             "the choreographed play_once timeline.")
    parser.add_argument("--disable-stop-on-success", action="store_true",
                        help="Do not terminate rollout early when check_success() "
                             "becomes true.")
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="Require success for N consecutive policy steps "
                             "(envs/success_hold.py). Default: 10.")
    parser.add_argument("--inspect-motion-name", type=str, default=None,
                        help="(Interruption tasks) Force a specific inspect motion by name.")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="If set, save one MP4 per episode under this directory.")
    parser.add_argument("--video-limit", type=int, default=None,
                        help="When --video-dir is set, save videos only for the first N episodes.")
    parser.add_argument("--video-stride", type=int, default=None,
                        help="Capture 1 video frame every N sim steps when "
                             "--video-dir is set (default: task VIDEO_STRIDE).")
    parser.add_argument("--output-json", type=str, default=None,
                        help="If set, write aggregate and per-episode eval results to this JSON file.")
    parser.add_argument("--random-action-scale", type=float, default=0.0,
                        help="Magnitude of random arm-delta per step in the "
                             "default random_policy (only used if no real "
                             "policy is supplied). 0 = zero policy. Small "
                             "values like 0.02 produce visible joint jitter.")
    args = parser.parse_args()
    from envs.debug_switches import refuse_debug_switches
    refuse_debug_switches("scripts/eval.py")

    # config/default.yml is the base (renderer, nyx spp/denoise/env_texture,
    # camera resolutions); --config overlays it.
    config = load_config(args.config)

    if args.renderer:
        config["renderer"] = args.renderer
        if args.renderer == "nyx":
            os.environ.setdefault("GENESIS_BACKEND", "gpu")
            config.setdefault("nyx", {"spp": 1, "denoise": False, "open_window": False})
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.policy == "expert_no_human":
        args.no_human = True

    if args.no_human:
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
        config["eval_mode"] = False
    else:
        # Policy eval bypasses scripted play_once(), so avatar tasks need
        # their eval-mode state machines to reproduce assist / interrupt /
        # neutral avatar behaviour during take_action() rollouts.
        config.setdefault("eval_mode", True)
        if args.policy == "expert_full_state":
            # Full-state expert is allowed privileged scheduling / safety
            # feedback.  Keep plain `expert` as the nominal scripted baseline.
            config.setdefault("track_avatar_collision", True)
            config.setdefault("use_avatar_collider", True)
            config.setdefault("collision_check_stride", 5)
            config.setdefault("neutral_avatar_start_delay_steps", 2500)
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False

    if args.track_avatar_collision:
        config["track_avatar_collision"] = True
    if args.collision_stride is not None:
        config["collision_check_stride"] = max(1, args.collision_stride)
    if args.collision_margin is not None:
        config["collision_margin"] = float(args.collision_margin)
    if args.avatar_safety_margin is not None:
        config["avatar_safety_margin"] = float(args.avatar_safety_margin)
    if args.avatar_retreat_enabled:
        config["avatar_retreat_enabled"] = True
    if args.avatar_retreat_margin is not None:
        config["avatar_retreat_margin"] = float(args.avatar_retreat_margin)
    if args.show_collider:
        config["use_avatar_collider"] = True
        config["collider_visualization"] = True
    if args.eval_mode:
        config["eval_mode"] = True
    if args.disable_stop_on_success:
        config["eval_stop_on_success"] = False
    if args.success_hold_steps is not None:
        config["eval_success_hold_steps"] = max(1, int(args.success_hold_steps))
    if args.inspect_motion_name:
        config["inspect_motion_name"] = args.inspect_motion_name
    if args.video_stride is not None:
        config["video_stride"] = max(1, args.video_stride)

    # Random policy for testing — replace with actual policy.
    rng = np.random.default_rng(0)

    def random_policy(obs):
        joint_state = obs.get("joint_state") if isinstance(obs, dict) else None
        if not isinstance(joint_state, dict):
            joint_state = {}
        right_raw = joint_state.get("right")
        left_q = joint_state.get("left")
        if right_raw is None:
            right_q = np.zeros(8, dtype=np.float64)
        else:
            right_q = np.asarray(right_raw, dtype=np.float64).ravel()
        # Observed joint_state includes arm qpos plus the scalar gripper
        # value; BaseTask.take_action expects arm deltas and separate
        # absolute gripper targets.
        n_right = max(0, int(right_q.size) - 1)
        if left_q is not None:
            n_left = max(0, int(np.asarray(left_q, dtype=np.float64).ravel().size) - 1)
            a = np.zeros(n_left + 1 + n_right + 1, dtype=np.float64)
        else:
            a = np.zeros(n_right + 1, dtype=np.float64)
        if args.random_action_scale > 0:
            if left_q is not None:
                a[:n_left] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_left,
                )
                a[n_left + 1:n_left + 1 + n_right] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_right,
                )
            else:
                a[:n_right] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_right,
                )
        # Gripper slots are absolute targets in qpos mode. Hold open so
        # random-policy smoke tests perturb only the arm joints.
        if left_q is not None:
            a[n_left] = 1.0
            a[n_left + 1 + n_right] = 1.0
        else:
            a[n_right] = 1.0
        return a

    if args.policy in {"expert_full_state", "expert", "expert_no_human"}:
        policy = ExpertFullStatePolicy()
    else:
        policy = random_policy

    scene_inits = load_scene_init_file(args.scene_init_file) if args.scene_init_file else None
    result = eval_policy(
        args.task,
        config,
        policy,
        args.episodes,
        args.max_steps,
        args.start_seed,
        video_dir=args.video_dir,
        video_limit=args.video_limit,
        return_details=bool(args.output_json),
        scene_inits=scene_inits,
        scene_init_start_index=args.scene_init_start_index,
    )
    if args.output_json:
        if isinstance(result, dict) and args.scene_init_file:
            result["scene_init_file"] = args.scene_init_file
            result["scene_init_start_index"] = args.scene_init_start_index
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=_json_default)
        print(f"Results JSON: {args.output_json}")


if __name__ == "__main__":
    main()
