"""VLA training-data collection: stage-1 (verdict) + stage-2 (record).

Two modes:

  --mode verdict   (stage 1, rasterizer)
      Run ``play_once()`` for one seed, write
      ``<out_dir>/seed_<N>/verdict.json`` with
        { success, plan_success, infra_failure_reason,
          evaluate, wall_time, task, seed }
      No image recording, ``--track-avatar-collision`` always on so
      ``evaluate()`` can gate on it.

  --mode record    (NYX or rasterizer)
      Run ``play_once()`` and stream qpos/ee/gripper + JPEGs into
      ``<out_dir>/seed_<N>/{steps.h5, meta.json}`` via VLARecorder.
      Existing recordings are skipped only after successful validation.

Verdict JSON is also written in record mode so a downstream consumer can
treat the output dir uniformly.

Outer-layer wall-clock cap is set by the sbatch wrapper (`timeout`); this
script just runs and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from envs.tasks import resolve_task_class
from envs.utils import ASSETS_PATH, load_config
from scripts.vla_client import get_task_instruction
from scripts.vla_data.collection_contract import (
    REQUIRED_ACT_H5_KEYS as REQUIRED_ACT_H5_KEYS,
    _collection_avatar_meta,
    _json_default,
    _neutral_meta,
    _require_reusable_collection_contract,
    _success_from_evaluation,
    validate_recorded_h5,
)
from scripts.vla_data.validate_nyx import validate_episode


_STATIC_AVATAR_BASE_ROT = None


def _static_avatar_base_rot():
    global _STATIC_AVATAR_BASE_ROT
    if _STATIC_AVATAR_BASE_ROT is None:
        import numpy as np
        _STATIC_AVATAR_BASE_ROT = np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
    return _STATIC_AVATAR_BASE_ROT


class StaticAvatarMixin:
    """Force an idle avatar into base/no-human rollouts.

    `config["no_human"]` stays true so inherited task code uses the base-task
    object layout and scripted robot behavior.  This mixin bypasses only the
    avatar creation/collision gates, then refreshes the avatar capsule point
    cloud before robot plans so demonstrations route around the still human.
    """

    use_avatar = True
    avatar_init_pos = None
    avatar_init_rot = None
    _STATIC_AVATAR_OBSTACLE_RES = 0.04
    _STATIC_AVATAR_INFLATE_FACTOR = 1.6

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg["static_avatar"] = True
        cfg["no_human"] = True
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        if self.avatar_init_pos is None:
            self.avatar_init_pos = np.asarray(
                cfg.get("static_avatar_pos", [0.0, 0.55, -0.18]), dtype=np.float64
            )
        if self.avatar_init_rot is None:
            self.avatar_init_rot = _static_avatar_base_rot().copy()
        super().__init__(cfg)

    def _init_avatar(self):
        try:
            from envs.avatar import AvatarController
        except ImportError:
            print("[static_avatar] AvatarController import failed")
            return

        avatar_cfg = self.config.get("avatar", {})
        self.avatar = AvatarController(
            scene=self.scene,
            motion_data_path=avatar_cfg.get("motion_data", "avatars/motions/motion.pkl"),
            skin_options=self._resolve_avatar_skin(),
            frame_ratio=avatar_cfg.get("frame_ratio", 1.0),
            name="human",
            assets_dir=str(ASSETS_PATH),
            generated_motion_path=avatar_cfg.get(
                "generated_motion_data", "avatars/motions/generated_motions.pkl"
            ),
        )
        self.static_avatar = self.avatar

    def reset(self, seed: int = 0):
        obs = super().reset(seed)
        self.static_avatar = self.avatar
        # Keep the avatar entity in the scene for rendering and analytic
        # collision checking, but hide it from task scripts so no-human
        # play_once branches stay identical to base-task collection.
        self.avatar = None
        return obs

    def _avatar_collision_enabled(self) -> bool:
        return bool(self.config.get("track_avatar_collision", False))

    def _avatar_obstacle_points(self):
        if self.avatar_collider is None:
            return None
        pts = []
        for _name, pa, pb, r in self.avatar_collider.current_capsules():
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            seg = pb - pa
            length = float(np.linalg.norm(seg))
            r_inflated = float(r) * self._STATIC_AVATAR_INFLATE_FACTOR
            if length < 1e-6:
                pts.append(pa)
                continue
            axis = seg / length
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(axis, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(axis, u)
            v = v / (np.linalg.norm(v) + 1e-12)
            n_axis = max(2, int(length / self._STATIC_AVATAR_OBSTACLE_RES) + 1)
            for t in np.linspace(0.0, 1.0, n_axis):
                center = pa + t * seg
                pts.append(center)
                for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    pts.append(center + r_inflated * (np.cos(theta) * u + np.sin(theta) * v))
        return np.asarray(pts, dtype=np.float64) if pts else None

    def _refresh_static_avatar_planner_obstacles(self, arm_tag: str) -> None:
        if not bool(self.config.get("static_avatar_as_planner_obstacle", True)):
            return
        if self.avatar_collider is not None:
            try:
                self.avatar_collider.update()
            except Exception:
                pass
        pts = self._avatar_obstacle_points()
        if pts is None or pts.size == 0:
            return
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._STATIC_AVATAR_OBSTACLE_RES)
        except Exception as e:
            print(f"[static_avatar] obstacle update failed: {e}")

    def move_to_pose(self, pose, arm_tag: str):
        self._refresh_static_avatar_planner_obstacles(arm_tag)
        return super().move_to_pose(pose, arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        self._refresh_static_avatar_planner_obstacles(arm_tag)
        return super()._move_seeded(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag):
        self._refresh_static_avatar_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag)


def with_static_avatar(TaskClass):
    return type(f"StaticAvatar{TaskClass.__name__}", (StaticAvatarMixin, TaskClass), {})


def _classify_infra_failure(err: BaseException) -> str | None:
    """Return a short tag for an exception that is not a task-logic
    failure -- IK/planner/CUDA/OOM all retry-worthy.  Return None if
    the exception looks like a task-side bug that retrying won't fix."""
    msg = str(err)
    if isinstance(err, MemoryError):
        return "oom"
    if "PTX" in msg or "CUDA" in msg or "cuda" in msg.lower():
        return "cuda"
    if "mplib" in msg.lower() or "ompl" in msg.lower():
        return "planner"
    if "IK" in msg or "ik" in msg.lower():
        return "ik"
    return None


def _failure_out_root(out_root: Path, failure_root: str | None = None) -> Path:
    """Return the sibling per-task renderer root for rejected episodes."""
    out_root = Path(out_root)
    if failure_root:
        return Path(failure_root) / out_root.parent.name / out_root.name
    if not out_root.parent.name:
        return out_root.with_name(f"{out_root.name}_failed")
    data_root = out_root.parent.parent
    return data_root.with_name(f"{data_root.name}_failed") / out_root.parent.name / out_root.name


def _move_failed_episode(episode_dir: Path, out_root: Path, failure_root: str | None = None) -> Path:
    """Move a rejected episode out of the VLA data tree and return its new path."""
    failed_root = _failure_out_root(out_root, failure_root)
    failed_root.mkdir(parents=True, exist_ok=True)
    dest = failed_root / episode_dir.name
    if dest.exists():
        suffix = 1
        while True:
            candidate = failed_root / f"{episode_dir.name}_dup{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            suffix += 1
    shutil.move(str(episode_dir), str(dest))
    return dest


def _build_config(args) -> dict:
    # config/default.yml is the base (renderer, nyx spp/denoise/env_texture,
    # camera resolutions); --config overlays it unless exact parity is requested.
    if args.exact_config:
        import yaml

        with Path(args.config).open() as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("--exact-config requires a YAML mapping")
        config["eval_seed"] = args.seed
    else:
        config = load_config(args.config)

    if args.renderer:
        config["renderer"] = args.renderer
        if args.renderer == "nyx":
            os.environ.setdefault("GENESIS_BACKEND", "gpu")
            config.setdefault("nyx", {"spp": 1, "denoise": False, "open_window": False})

    # Always on per the data-pipeline contract: avatar-collision gates
    # success on tasks with success_require_no_avatar_collision.
    config["track_avatar_collision"] = True
    config.setdefault("table_random_objects", False)
    config.setdefault("task_irrelevant_objects", False)

    if args.randomize_avatar:
        config["randomize_avatar"] = True
    if args.static_avatar:
        config["static_avatar"] = True
        config["no_human"] = True
        config["use_avatar_collider"] = True
        config["track_avatar_collision"] = True
        config["static_avatar_as_planner_obstacle"] = not args.static_avatar_visual_only
        config["static_avatar_pos"] = list(args.static_avatar_pos)
    if args.no_human:
        config["no_human"] = True
        config["track_avatar_collision"] = False
        config["randomize_avatar"] = False
    if args.randomize_table:
        config["randomize_table"] = True
        config["debug_table_randomization"] = True
    if args.table_variant_name:
        config["randomize_table"] = True
        config["debug_table_randomization"] = True
        config["table_variant_name"] = args.table_variant_name
    if args.random_object:
        config["random_object"] = True
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False
    if args.random_interrupt_target:
        config["random_interrupt_target"] = True
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.setting is not None:
        config["setting"] = args.setting

    # Stage 2 only: turn on VLA recorder.
    if args.mode == "record":
        config["vla_recording"] = {
            "enabled": True,
            "out_dir": str(args.episode_dir),
            "cams": args.cams.split(","),
            "jpeg_quality": args.jpeg_quality,
            "opposite_head_camera": not args.keep_original_head_camera,
        }

    return config


def _actual_renderer(task, requested_renderer: str) -> tuple[str, bool, str]:
    """Require actual NYX cameras before recording can be called NYX."""
    actual = str(task.config.get("renderer", "unknown")).lower()
    manager = type(task.cameras).__name__
    verified = False
    if requested_renderer == "nyx":
        from envs.camera import NyxCameraManager

        if actual != "nyx" or not isinstance(task.cameras, NyxCameraManager):
            raise RuntimeError(
                f"NYX renderer required; resolved renderer={actual!r}, cameras={manager!r}"
            )
        verified = True
        manager = "NyxCameraManager"
    return actual, verified, manager


def _existing_record_is_complete(episode_dir, task_name, seed, renderer, config):
    """Never accept a metadata file left by an empty or interrupted write."""
    with (Path(episode_dir) / "meta.json").open() as stream:
        meta = json.load(stream)
    with (Path(episode_dir) / "verdict.json").open() as stream:
        verdict = json.load(stream)
    for record in (meta, verdict):
        _require_reusable_collection_contract(record, config, require_play_success=True)
        if (record.get("task") != task_name or record.get("seed") != seed
                or record.get("renderer") != renderer
                or record.get("success") is not True
                or record.get("plan_success") is not True
                or record.get("infra_failure_reason") or record.get("error")):
            raise ValueError("existing recording does not match a successful requested episode")
    if config.get("randomize_avatar") is True and meta["avatar_model"] != verdict["avatar_model"]:
        raise ValueError("existing recording avatar identities do not match")
    if renderer == "nyx":
        return validate_episode(episode_dir, expected_task=task_name,
                                expected_seed=seed, write_marker=True)
    return validate_recorded_h5(episode_dir)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", required=True,
                   help="Per-task per-renderer dir, e.g. vla_data/<task>/rasterizer")
    p.add_argument("--mode", choices=["verdict", "record"], required=True)
    p.add_argument("--renderer", choices=["nyx", "rasterizer"], default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--exact-config", action="store_true",
                   help="Read only --config without merging default.yml; preserve evaluator settings.")
    p.add_argument("--randomize-avatar", action="store_true")
    p.add_argument("--static-avatar", action="store_true",
                   help="Use base/no-human task behavior while forcing an idle avatar into the scene.")
    p.add_argument("--static-avatar-visual-only", action="store_true",
                   help="Do not add the idle avatar capsule cloud to the robot planner.")
    p.add_argument("--static-avatar-pos", type=float, nargs=3, default=[0.0, 0.55, -0.18],
                   metavar=("X", "Y", "Z"),
                   help="Idle avatar body position for --static-avatar.")
    p.add_argument("--no-human", action="store_true",
                   help="Collapse a Contribution/Interruption/Co-presence task to "
                        "its family's robot-only no-human task. Cueing tasks do "
                        "not support this flag.")
    p.add_argument("--randomize-table", action="store_true",
                   help="Randomize the SAPIEN table mesh variant for tasks "
                        "that support table variants. Enables the table "
                        "debug gate required by envs.table_assets.")
    p.add_argument("--table-variant-name", default=None,
                   help="Force a specific SAPIEN table variant.")
    p.add_argument("--random-object", action="store_true",
                   help="Set config['random_object']=True for tasks that "
                        "support object identity randomization.")
    p.add_argument("--table-random-objects",
                   "--task-irrelevant-objects",
                   "--with-irrelevant-objects",
                   dest="table_random_objects",
                   action="store_true",
                   help="Enable random background tabletop objects. Default: disabled.")
    p.add_argument("--no-table-random-objects",
                   "--no-task-irrelevant-objects",
                   dest="no_table_random_objects",
                   action="store_true",
                   help="Disable random background tabletop objects, overriding config.")
    p.add_argument("--random-interrupt-target", action="store_true")
    p.add_argument("--hard", action="store_true",
                   help="Use the legacy multi-object task variant where supported.")
    p.add_argument("--cams", default="head_camera,side,right_wrist",
                   help="Comma-separated cam names to record (record mode only)")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--keep-original-head-camera", action="store_true",
                   help="Do not mirror VLA head_camera to the opposite side.")
    p.add_argument("--keep-failures", action="store_true",
                   help="Keep failed record-mode episodes as steps.h5/meta.json for debugging.")
    p.add_argument("--failure-root", default=None,
                   help="Dataset root for rejected episodes. Defaults to sibling '<out_root_root>_failed'.")
    p.add_argument("--instruction", default=None,
                   help="Override the per-task language instruction")
    p.add_argument("--setting", choices=["sim", "real"], default=None,
                   help="Collection setting. 'real' uses the real-world "
                        "interaction axis where the robot arm is at world x- "
                        "and the human is at world x+ for tasks that support it.")
    args = p.parse_args()
    from envs.debug_switches import refuse_debug_switches
    refuse_debug_switches("scripts/vla_data/collect_vla.py")
    if args.exact_config and not args.config:
        p.error("--exact-config requires --config")

    try:
        resolved_task_name, TaskClass = resolve_task_class(
            args.task,
            no_human=bool(getattr(args, "no_human", False) or getattr(args, "static_avatar", False)),
        )
    except ValueError as e:
        print(f"[collect_vla] {e}", file=sys.stderr)
        return 2
    if args.static_avatar:
        TaskClass = with_static_avatar(TaskClass)

    episode_dir = Path(args.out_root) / f"seed_{args.seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    args.episode_dir = episode_dir

    verdict_path = episode_dir / "verdict.json"
    meta_path = episode_dir / "meta.json"

    config = _build_config(args)
    requested_renderer = str(config.get("renderer", "rasterizer")).lower()

    # A process can die after writing metadata but before completing the H5.
    # Revalidate existing products and quarantine incomplete attempts before
    # replaying so old frames/markers cannot survive a fresh attempt.
    if args.mode == "verdict" and verdict_path.exists():
        try:
            prior = json.loads(verdict_path.read_text())
            _require_reusable_collection_contract(prior, config, require_play_success=False)
            complete = (
                prior.get("task") == resolved_task_name and prior.get("seed") == args.seed
                and prior.get("mode") == "verdict"
                and prior.get("renderer") == requested_renderer
                and type(prior.get("success")) is bool
                and not prior.get("infra_failure_reason") and not prior.get("error")
                and (requested_renderer != "nyx" or prior.get("renderer_verified") is True)
            )
        except (OSError, ValueError, AttributeError):
            complete = False
        if complete:
            print(f"[collect_vla] complete verdict already exists, skipping: {verdict_path}")
            return 0
        failed_dir = _move_failed_episode(episode_dir, Path(args.out_root), args.failure_root)
        print(f"[collect_vla] preserved incompatible verdict at {failed_dir}")
        episode_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "record" and any(episode_dir.iterdir()):
        try:
            _existing_record_is_complete(episode_dir, resolved_task_name,
                                         args.seed, requested_renderer, config)
        except Exception as exc:
            print(f"[collect_vla] existing recording is incomplete: {exc}")
            failed_dir = _move_failed_episode(episode_dir, Path(args.out_root), args.failure_root)
            print(f"[collect_vla] preserved incomplete attempt at {failed_dir}")
            episode_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[collect_vla] validated recording already exists, skipping: {episode_dir}")
            return 0
    instruction = (
        args.instruction
        or (TaskClass.default_instruction() if TaskClass is not None
            and hasattr(TaskClass, "default_instruction") else None)
        or get_task_instruction(resolved_task_name)
    )
    # Only force the task's instruction on an explicit override; otherwise let
    # the task template its per-episode object name into self.instruction (the
    # saved record reads task.instruction, so the dataset gets the real object).
    if args.instruction:
        config["instruction"] = args.instruction

    print(f"[collect_vla] task={resolved_task_name} seed={args.seed} mode={args.mode} "
          f"renderer={config.get('renderer','rasterizer')}")

    t0 = time.time()
    success = False
    play_success = False
    plan_success = False
    infra_failure_reason: str | None = None
    evaluate_dict: dict = {}
    error_msg: str | None = None
    task = None
    actual_renderer = "unknown"
    renderer_verified = False
    camera_manager = None

    try:
        task = TaskClass(config)
        task.reset(seed=args.seed)
        observed_avatar = _collection_avatar_meta(task, task.config)
        if observed_avatar["randomize_avatar"] and not (
            observed_avatar["avatar_randomized"] and observed_avatar["avatar_initialized"]
            and observed_avatar["avatar_model"]
        ):
            raise RuntimeError("Requested randomized avatar was not initialized and verified")
        # Record the fallback's actual identity even when the strict check fails.
        actual_renderer = str(task.config.get("renderer", "unknown")).lower()
        camera_manager = type(task.cameras).__name__
        actual_renderer, renderer_verified, camera_manager = _actual_renderer(
            task, requested_renderer)
        instruction = str(getattr(task, "instruction", None) or instruction)
        try:
            play_success = bool(task.play_once())
            success = play_success
        except Exception as e:
            traceback.print_exc()
            error_msg = repr(e)
            infra_failure_reason = _classify_infra_failure(e)
            success = False
        plan_success = bool(getattr(task, "plan_success", False))
        try:
            evaluate_dict = task.evaluate()
            success = _success_from_evaluation(task, evaluate_dict, success)
        except Exception as e:
            evaluate_dict = {"error": repr(e)}
            error_msg = repr(e)
            success = False
        if error_msg or infra_failure_reason or not plan_success or not play_success:
            success = False
    except Exception as e:
        traceback.print_exc()
        error_msg = repr(e)
        infra_failure_reason = (
            "renderer_mismatch" if "NYX renderer required" in str(e)
            else _classify_infra_failure(e) or "task_init"
        )
        success = False

    elapsed = time.time() - t0
    effective_config = getattr(task, "config", config)
    if task is not None:
        instruction = str(getattr(task, "instruction", None) or instruction)
    neutral_meta = _neutral_meta(task, config) if task is not None else {}
    avatar_meta = _collection_avatar_meta(task, effective_config)

    verdict = {
        "task": resolved_task_name,
        "seed": args.seed,
        "mode": args.mode,
        "renderer": actual_renderer,
        "requested_renderer": requested_renderer,
        "renderer_verified": renderer_verified,
        "camera_manager": camera_manager,
        "instruction": instruction,
        "setting": effective_config.get("setting", "sim"),
        "success": success,
        "plan_success": plan_success,
        "play_success": play_success,
        "collection_contract_version": 2,
        "infra_failure_reason": infra_failure_reason,
        "error": error_msg,
        "wall_time_s": elapsed,
        "evaluate": evaluate_dict,
        **neutral_meta,
        **avatar_meta,
    }
    verdict_path.write_text(json.dumps(verdict, indent=2, default=_json_default))
    print(f"[collect_vla] verdict: success={success} plan={plan_success} "
          f"infra={infra_failure_reason} wall={elapsed:.1f}s -> {verdict_path}")

    recorder = getattr(task, "vla_recorder", None) if task is not None else None
    if args.mode == "record" and not success and not args.keep_failures:
        try:
            n_captures = 0
            if recorder is not None:
                stats = recorder.discard()
                n_captures = int(stats.get("n_captures", 0))
            failed_dir = _move_failed_episode(episode_dir, Path(args.out_root), args.failure_root)
            print(
                f"[collect_vla] rejected failed recording "
                f"captures={n_captures} -> {failed_dir}"
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[collect_vla] recorder discard failed: {e}", file=sys.stderr)
        return 1

    if args.mode == "record" and recorder is not None:
        config = effective_config
        robot_type = str(config.get("robot_type", "franka"))
        meta = {
            "task": resolved_task_name,
            "seed": args.seed,
            "instruction": instruction,
            "success": success,
            "plan_success": plan_success,
            "play_success": play_success,
            "collection_contract_version": 2,
            "renderer": actual_renderer,
            "requested_renderer": requested_renderer,
            "renderer_verified": renderer_verified,
            "camera_manager": camera_manager,
            "infra_failure_reason": infra_failure_reason,
            "error": error_msg,
            "setting": config.get("setting", "sim"),
            "wall_time_s": elapsed,
            "evaluate": evaluate_dict,
            "embodiment": robot_type,
            "robot_type": robot_type,
            **neutral_meta,
            **avatar_meta,
            "config_snapshot": {
                "robot_type": robot_type,
                "action_substeps": config.get("action_substeps", 25),
                "setting": config.get("setting", "sim"),
                "track_avatar_collision": config.get("track_avatar_collision", False),
                "randomize_avatar": config.get("randomize_avatar", False),
                "randomize_table": config.get("randomize_table", False),
                "table_variant_name": config.get("table_variant_name"),
                "random_object": getattr(task, "config", config).get(
                    "random_object", False,
                ),
                "random_interrupt_target": config.get("random_interrupt_target", False),
                "neutral_avatar_motion": config.get("neutral_avatar_motion"),
                "neutral_avatar_motion_choices": config.get("neutral_avatar_motion_choices"),
                "opposite_head_camera": config.get("vla_recording", {}).get(
                    "opposite_head_camera", False,
                ),
            },
        }
        try:
            stats = recorder.close(meta)
            if success and requested_renderer == "nyx":
                validation = validate_episode(
                    episode_dir, expected_task=resolved_task_name,
                    expected_seed=args.seed, write_marker=True)
            else:
                validation = validate_recorded_h5(episode_dir)
            print(
                f"[collect_vla] recorded {stats['n_steps']} (obs,action) pairs to {episode_dir} "
                f"schema={validation['schema_version']} cams={validation['cams']}"
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[collect_vla] recorder close failed: {e}", file=sys.stderr)
            (episode_dir / "validated.json").unlink(missing_ok=True)
            verdict.update(success=False, infra_failure_reason="recording_validation", error=repr(e))
            verdict_path.write_text(json.dumps(verdict, indent=2, default=_json_default))
            return 1

    if args.mode == "record" and recorder is None:
        verdict.update(success=False, infra_failure_reason="missing_recorder")
        verdict_path.write_text(json.dumps(verdict, indent=2, default=_json_default))
        return 1

    return 0 if (success or args.mode == "verdict") else 1


if __name__ == "__main__":
    sys.exit(main())
