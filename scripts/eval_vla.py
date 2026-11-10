"""Evaluate a VLA model on Genesis HR Bench tasks.

Usage:
    python scripts/eval_vla.py --task pour_water --model pi0 \
        --server-url http://127.0.0.1:8767 --action-type qpos_abs --episodes 10
"""

import argparse
import os
import sys
import yaml
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval import eval_policy
from scripts.vla_client import VLA_REGISTRY, get_task_instruction
from envs.tasks import canonical_task_name, resolve_task_class
from envs.scene_init import load_scene_init_file


VLA_DT_001_TASKS = {
    "categorize_objects_contribution",
    "take_from_human_easy",
    "take_from_human_safety",
}


VLA_COLLECTION_TASK_DEFAULTS = {
    # Matched to local VLA meta.json config_snapshot values where available.
    "pour_water": {"randomize_avatar": False, "randomize_table": False, "random_object": False},
    "take_from_human_easy": {"random_object": False},
    "deliver_to_human": {"random_object": True},
    "categorize_objects_contribution": {"random_object": False},
    "put_object_in_cabinet_contribution": {"random_object": False},
}


def _default_vla_action_substeps(task_name: str, target_hz: float = 20.0) -> int:
    sim_dt = 0.001 if canonical_task_name(task_name) in VLA_DT_001_TASKS else 0.002
    return max(1, int(round(1.0 / (sim_dt * target_hz))))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLA policy")
    parser.add_argument("--task", required=True, help="Task name from TASK_MAP")
    parser.add_argument("--model", required=True,
                        help="VLA model name — a key in scripts.vla_client.VLA_REGISTRY "
                             "(act, lerobot_diffusion, pi0, pi05, smolvla, template)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path — unused for remote policies "
                             "(the server holds the checkpoint); required for "
                             "in-process policies.")
    parser.add_argument("--server-url", default=None,
                        help="For remote VLAs: HTTP URL of the model server, "
                             "e.g. http://127.0.0.1:8767")
    parser.add_argument("--action-type", default="qpos", choices=["qpos", "qpos_abs", "ee"])
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--instruction", default=None, help="Override task instruction")
    parser.add_argument("--config", default=None, help="Base task config YAML")
    parser.add_argument("--vla-config", default=None, help="VLA config YAML override")
    parser.add_argument("--unnorm-key", default=None,
                        help="Action/proprio normalization stats key for remote VLA servers.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--scene-init-file", default=None,
                        help="YAML/JSON/JSONL deterministic scene-init file. "
                             "When set, eval episodes are selected by index "
                             "instead of start-seed-only initialization.")
    parser.add_argument("--scene-init-start-index", type=int, default=0)
    parser.add_argument("--output-json", default=None, help="Save results to JSON")
    parser.add_argument("--video-dir", default=None,
                        help="If set, write one MP4 per episode to this directory")
    parser.add_argument("--video-limit", type=int, default=None,
                        help="When --video-dir is set, save videos only for the first N episodes.")
    parser.add_argument("--video-stride", type=int, default=None,
                        help="Capture one video frame every N sim steps when "
                             "--video-dir is set. Defaults to 10 for video "
                             "evals to avoid recording every physics substep.")
    parser.add_argument("--max-videos-per-dir", type=int, default=None,
                        help="When --video-dir is set, render at most this many "
                             "videos in the output directory; remaining episodes "
                             "still run without video capture.")
    parser.add_argument("--trace-actions-dir", default=None,
                        help="If set, write per-step action/qpos/object traces there.")
    parser.add_argument("--interpolate-qpos-actions", action="store_true",
                        help="Linearly interpolate qpos/qpos_abs arm and "
                             "gripper targets across each action_substeps window")
    parser.add_argument("--qpos-gripper-max-delta", type=float, default=None,
                        help="Clamp qpos/qpos_abs gripper command change per "
                             "policy action, e.g. 0.08 closes over many steps.")
    parser.add_argument("--qpos-gripper-close-hold-steps", type=int, default=None,
                        help="Extra sim steps to hold after each closing qpos "
                             "gripper command.")
    parser.add_argument("--qpos-gripper-immediate", action="store_true",
                        help="With --interpolate-qpos-actions, apply gripper "
                             "targets immediately while still interpolating arm qpos.")
    parser.add_argument("--qpos-gripper-pause-arm-on-close", action="store_true",
                        help="With qpos/qpos_abs actions, keep the arm fixed "
                             "during action substeps whose gripper target is closing.")
    parser.add_argument("--policy-gripper-kp", type=float, default=None,
                        help="Override eval-mode policy gripper finger kp.")
    parser.add_argument("--policy-gripper-kv", type=float, default=None,
                        help="Override eval-mode policy gripper finger kv.")
    parser.add_argument("--disable-eval-avatar-pre-policy", action="store_true",
                        help="For tasks that support it, do not run avatar "
                             "pre-policy warmup before the robot policy starts.")
    parser.add_argument("--disable-stop-on-success", action="store_true",
                        help="Do not terminate rollout early when check_success() "
                             "becomes true.")
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="Require success for N consecutive policy steps "
                             "(envs/success_hold.py). Default: 10.")
    parser.add_argument("--act-gripper-closed-cmd", type=float, default=None,
                        help="Override ACT qpos/qpos_abs snapped closed command.")
    parser.add_argument("--track-avatar-collision", action="store_true")
    parser.add_argument("--no-human", action="store_true",
                        help="Collapse a Contribution/Interruption/Co-presence task "
                             "to its family's robot-only no-human task. Cueing tasks "
                             "do not support this flag.")
    parser.add_argument("--static-avatar", action="store_true",
                        help="Evaluate base/no-human task behavior while rendering "
                             "a still avatar in the scene.")
    parser.add_argument("--static-avatar-pos", type=float, nargs=3,
                        default=[0.0, 0.55, -0.18],
                        metavar=("X", "Y", "Z"),
                        help="Still avatar body position for --static-avatar.")
    parser.add_argument("--hard", action="store_true",
                        help="Use the legacy multi-object task variant where supported.")
    parser.add_argument("--table-random-objects",
                        "--task-irrelevant-objects",
                        "--with-irrelevant-objects",
                        dest="table_random_objects",
                        action="store_true",
                        help="Enable random background tabletop objects. Default: disabled.")
    parser.add_argument("--no-table-random-objects",
                        "--no-task-irrelevant-objects",
                        dest="no_table_random_objects",
                        action="store_true",
                        help="Disable random background tabletop objects, overriding config.")
    parser.add_argument("--avatar-safety-margin", type=float, default=None)
    parser.add_argument("--avatar-retreat-enabled", action="store_true")
    parser.add_argument("--avatar-retreat-margin", type=float, default=None)
    parser.add_argument("--pace-trigger-progress", type=float, default=None,
                        help="Override the Interruption progress threshold in "
                             "[0,1] (config interrupt_trigger_progress). "
                             "Default is task-specific.")
    parser.add_argument("--interrupt-commit-radius-m", type=float, default=None,
                        help="Override the 3-D TCP-object commit radius of "
                             "the Interruption trigger. Default is task-specific.")
    parser.add_argument("--pace-min-policy-steps", type=int, default=None,
                        help="Override the minimum policy steps before an "
                             "Interruption can fire (config "
                             "interrupt_min_policy_steps). Default 20.")
    parser.add_argument("--pace-no-pan-ready", action="store_true",
                        help="For supported skillet tasks, allow PACE to "
                             "trigger before the pan is near the cooktop.")
    parser.add_argument("--pace-pan-ready-xy", type=float, default=None,
                        help="For supported skillet tasks, xy tolerance in "
                             "meters for considering the pan ready.")
    parser.add_argument("--static-pan", action="store_true",
                        help="For supported skillet tasks, start the pan on "
                             "the cooktop instead of forcing a dynamic pan.")
    parser.add_argument("--no-vla-train-defaults", action="store_true",
                        help="Do not apply VLA collection defaults such as "
                             "20 Hz action_substeps, randomized layout, and "
                             "VLA top-down head camera.")
    args = parser.parse_args()
    from envs.debug_switches import refuse_debug_switches
    refuse_debug_switches("scripts/eval_vla.py")

    config = {}
    default_config = os.path.join(os.path.dirname(__file__), "..", "config", "default.yml")
    if os.path.exists(default_config):
        with open(default_config) as f:
            config = yaml.safe_load(f) or {}
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f) or {})

    vla_config = {}
    vla_default = os.path.join(os.path.dirname(__file__), "..", "config", "vla.yml")
    if os.path.exists(vla_default):
        with open(vla_default) as f:
            vla_config = (yaml.safe_load(f) or {}).get("vla", {})
    if args.vla_config:
        with open(args.vla_config) as f:
            vla_config.update((yaml.safe_load(f) or {}).get("vla", {}))

    vla_config["model_name"] = args.model
    vla_config["checkpoint_path"] = args.checkpoint
    vla_config["action_type"] = args.action_type
    vla_config["dual_arm"] = args.dual_arm
    if args.server_url:
        vla_config["server_url"] = args.server_url
    if args.unnorm_key is not None:
        vla_config["unnorm_key"] = args.unnorm_key
    if args.act_gripper_closed_cmd is not None:
        vla_config["act_gripper_closed_cmd"] = float(args.act_gripper_closed_cmd)

    # VLA baselines always plan with Genesis damped-least-squares IK: it stays
    # near current qpos on every single-step EE move, which keeps the
    # benchmark's PD loop from amplifying branch jumps (cuRobo/mplib pick
    # globally-optimal IK solutions that can be far from the current pose,
    # producing runaway under high-frequency servoing). BaseTask only applies
    # this override when `planner_override` is set, so `scripts/collect.py`
    # (which never sets it) is unaffected.
    config.setdefault("planner_override", "genesis_ik")
    config.setdefault("table_random_objects", False)
    config.setdefault("task_irrelevant_objects", False)
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False

    if not args.no_vla_train_defaults:
        # Match the VLA training-data collection surface. The raw VLA dataset
        # records at 20 Hz, but some tasks use dt=0.001 while BaseTask defaults
        # to dt=0.002; action_substeps must therefore be task-dependent.
        config.setdefault("action_substeps", _default_vla_action_substeps(args.task))
        # Apply task-specific collection metadata before generic defaults.
        # Explicit config files still win, while known task metadata (for
        # example pour_water's fixed avatar/table layout) can override broad
        # VLA defaults.
        task_defaults = VLA_COLLECTION_TASK_DEFAULTS.get(canonical_task_name(args.task), {})
        for key, value in task_defaults.items():
            config.setdefault(key, value)
        # Collection used the VLA top-down head camera via
        # vla_recording.opposite_head_camera. During eval we need the same
        # camera without enabling a recorder.
        config.setdefault("vla_head_camera", True)
        config.setdefault("randomize_avatar", True)
        config.setdefault("randomize_table", True)
        config.setdefault("track_avatar_collision", True)

    if args.static_avatar:
        config["static_avatar"] = True
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["randomize_table"] = False
        config["random_object"] = False
        config["track_avatar_collision"] = False
        config["use_avatar_collider"] = False
        config["static_avatar_as_planner_obstacle"] = False
        config["static_avatar_pos"] = list(args.static_avatar_pos)
        config["eval_mode"] = False

    if args.no_human:
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
        config["eval_mode"] = False

    # Enable each task's composed or task-local eval-mode avatar state machine.
    # Without it the avatar stands idle during policy evaluation, so an assist,
    # interrupt, or intent scenario would not match its demonstration.
    if not args.no_human and not args.static_avatar:
        config.setdefault("eval_mode", True)

    resolved_task_name, TaskClass = resolve_task_class(
        args.task,
        no_human=bool(config.get("no_human", False) or config.get("static_avatar", False)),
    )
    instruction = (
        args.instruction
        or (TaskClass.default_instruction() if TaskClass is not None
            and hasattr(TaskClass, "default_instruction") else None)
        or get_task_instruction(resolved_task_name)
    )
    # Only force the task's instruction on an explicit override; otherwise let
    # the task template its per-episode object name into self.instruction, which
    # the eval loop feeds to the policy each reset via set_instruction.
    if args.instruction:
        config["instruction"] = args.instruction
    if (
        not args.no_human
        and (args.track_avatar_collision
             or args.avatar_safety_margin is not None
             or args.avatar_retreat_enabled)
    ):
        config["track_avatar_collision"] = True
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.avatar_safety_margin is not None:
        config["avatar_safety_margin"] = float(args.avatar_safety_margin)
    if args.avatar_retreat_enabled:
        config["avatar_retreat_enabled"] = True
    if args.avatar_retreat_margin is not None:
        config["avatar_retreat_margin"] = float(args.avatar_retreat_margin)
    if args.pace_trigger_progress is not None:
        config["interrupt_trigger_progress"] = float(args.pace_trigger_progress)
    if args.interrupt_commit_radius_m is not None:
        config["interrupt_commit_radius_m"] = float(args.interrupt_commit_radius_m)
    if args.pace_min_policy_steps is not None:
        config["interrupt_min_policy_steps"] = max(0, int(args.pace_min_policy_steps))
    if args.pace_no_pan_ready:
        config["pace_require_pan_ready"] = False
    if args.pace_pan_ready_xy is not None:
        config["pace_pan_ready_xy"] = float(args.pace_pan_ready_xy)
    if args.static_pan:
        config["force_dynamic_pan"] = False
    if args.video_stride is not None:
        config["video_stride"] = max(1, int(args.video_stride))
    elif args.video_dir:
        config.setdefault("video_stride", 10)
    if args.max_videos_per_dir is not None:
        config["max_videos_per_dir"] = max(0, int(args.max_videos_per_dir))
    if args.interpolate_qpos_actions:
        config["interpolate_qpos_actions"] = True
    if args.qpos_gripper_max_delta is not None:
        config["qpos_gripper_max_delta_per_action"] = float(args.qpos_gripper_max_delta)
    if args.qpos_gripper_close_hold_steps is not None:
        config["qpos_gripper_close_hold_steps"] = max(0, int(args.qpos_gripper_close_hold_steps))
    if args.qpos_gripper_immediate:
        config["qpos_gripper_immediate"] = True
    if args.qpos_gripper_pause_arm_on_close:
        config["qpos_gripper_pause_arm_on_close"] = True
    if args.policy_gripper_kp is not None:
        config["policy_gripper_kp"] = float(args.policy_gripper_kp)
    if args.policy_gripper_kv is not None:
        config["policy_gripper_kv"] = float(args.policy_gripper_kv)
    if args.disable_eval_avatar_pre_policy:
        config["eval_avatar_pre_policy"] = False
    if args.disable_stop_on_success:
        config["eval_stop_on_success"] = False
    if args.success_hold_steps is not None:
        config["eval_success_hold_steps"] = max(1, int(args.success_hold_steps))
    if args.trace_actions_dir:
        config["trace_actions_dir"] = args.trace_actions_dir

    if args.model not in VLA_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(VLA_REGISTRY.keys())}")

    PolicyClass = VLA_REGISTRY[args.model]
    policy = PolicyClass(
        checkpoint_path=args.checkpoint,
        task_instruction=instruction,
        config=vla_config,
        action_type=args.action_type,
        dual_arm=args.dual_arm,
    )
    policy.load_model()

    scene_inits = load_scene_init_file(args.scene_init_file) if args.scene_init_file else None
    eval_results = eval_policy(
        resolved_task_name, config, policy,
        args.episodes, args.max_steps, args.start_seed,
        action_type=args.action_type,
        video_dir=args.video_dir,
        video_limit=args.video_limit,
        return_details=True,
        scene_inits=scene_inits,
        scene_init_start_index=args.scene_init_start_index,
    )
    success_rate = float(eval_results["success_rate"])

    if args.output_json:
        results = {
            "task": resolved_task_name,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "action_type": args.action_type,
            "dual_arm": args.dual_arm,
            "episodes": args.episodes,
            "success_rate": success_rate,
            "successes": eval_results.get("successes"),
            "episode_results": eval_results.get("episode_results", []),
        }
        if args.scene_init_file:
            results["scene_init_file"] = args.scene_init_file
            results["scene_init_start_index"] = args.scene_init_start_index
        output_parent = os.path.dirname(args.output_json)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
