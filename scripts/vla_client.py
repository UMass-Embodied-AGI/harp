"""Small runtime helpers for VLA evaluation.

This keeps the benchmark-facing HTTP client shims in one place without the
larger historical ``policies/`` package.
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


def get_task_instruction(task_name, override=None):
    if override:
        return override
    try:
        from envs.tasks import resolve_task_class
        _, task_cls = resolve_task_class(task_name, no_human=False)
        # default_instruction() fills any {object} template slot with a
        # representative object so static lookups never surface a raw
        # placeholder (per-episode the real object is set via set_instruction).
        if task_cls is not None and hasattr(task_cls, "default_instruction"):
            instruction = task_cls.default_instruction()
        else:
            instruction = getattr(task_cls, "INSTRUCTION", None) if task_cls else None
        if instruction:
            return instruction
    except Exception:
        pass
    return f"Complete the {task_name.replace('_', ' ')} task."


def encode_png(img_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def resize_uint8(img_uint8: np.ndarray, hw) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img_uint8).resize((hw[1], hw[0]), Image.LANCZOS),
        dtype=np.uint8,
    )


def resize_like_training(img_uint8: np.ndarray, hw, pipeline=None) -> np.ndarray:
    """Resize a camera frame for the policy.

    pipeline="jpeg92_bilinear" reproduces the train37 dataset path (VLA
    recorder JPEG q92 -> decode -> PIL BILINEAR squash; the later h264
    video stage is not reproduced). Default keeps LANCZOS.
    """
    if pipeline == "jpeg92_bilinear":
        buf = io.BytesIO()
        Image.fromarray(np.asarray(img_uint8, dtype=np.uint8)).save(buf, format="JPEG", quality=92)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return np.asarray(img.resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.uint8)
    if pipeline not in (None, "", "lanczos"):
        raise ValueError(f"unknown image_pipeline {pipeline!r}")
    return resize_uint8(img_uint8, hw)


def extract_proprio(obs, action_type, dual_arm):
    parts = []
    arms = ["left", "right"] if dual_arm else ["right"]
    for arm in arms:
        if action_type in {"qpos", "qpos_abs"}:
            joint_state = list(obs["joint_state"][arm])
            gripper = obs.get("gripper_val", {}).get(arm)
            if gripper is not None and len(joint_state) in {6, 7}:
                joint_state.append(float(gripper))
            parts.extend(joint_state)
        else:
            parts.extend(obs["ee_pose"][arm])
    return np.array(parts, dtype=np.float32)


class BaseVLAPolicy(ABC):
    def __init__(self, checkpoint_path, task_instruction, config, action_type, dual_arm):
        self.checkpoint_path = checkpoint_path
        self.task_instruction = task_instruction
        self.config = config
        self.action_type = action_type
        self.dual_arm = dual_arm
        self.action_dim = 8 * (2 if dual_arm else 1)
        self._action_queue = []

    def set_instruction(self, instruction):
        """Update the language instruction (called per episode by the eval
        loop so tasks that sample their target object each reset feed the
        correct per-episode instruction to the model).  No-op on None/empty."""
        if instruction:
            self.task_instruction = instruction

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def predict_raw(self, processed_obs: dict) -> np.ndarray:
        ...

    def preprocess_obs(self, obs: dict) -> dict:
        cameras = self.config.get("cameras", ["head_camera"])
        image_size = tuple(self.config.get("image_size", [224, 224]))
        images = {}
        for name in cameras:
            images[name] = np.asarray(
                Image.fromarray(obs["rgb"][name]).resize(
                    (image_size[1], image_size[0]), Image.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        proprio = extract_proprio(obs, self.action_type, self.dual_arm)
        return {"images": images, "proprio": proprio, "instruction": self.task_instruction}

    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        return raw_action.astype(np.float64)

    def reset(self):
        self._action_queue = []

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._action_queue:
            raw = np.atleast_2d(self.predict_raw(self.preprocess_obs(obs)))
            for action in raw:
                self._action_queue.append(self.postprocess_action(action))
        action = self._action_queue.pop(0)
        assert action.shape == (self.action_dim,), (
            f"Expected ({self.action_dim},), got {action.shape}"
        )
        return action


def _quat_wxyz_to_euler(q):
    import transforms3d as t3d

    return np.asarray(t3d.euler.quat2euler(q, axes="sxyz"), dtype=np.float32)


def _lerobot_delta_to_absolute_ee(delta_action, current_pose) -> np.ndarray:
    import transforms3d as t3d
    from scipy.spatial.transform import Rotation as R

    delta_action = np.asarray(delta_action, dtype=np.float64).ravel()
    current_pose = np.asarray(current_pose, dtype=np.float64).ravel()
    dxyz = delta_action[:3]
    drpy = delta_action[3:6]
    gripper_open = 1.0 - float(delta_action[6])

    cur_xyz = current_pose[:3]
    cur_quat_wxyz = current_pose[3:7]
    cur_rot = R.from_quat([
        cur_quat_wxyz[1],
        cur_quat_wxyz[2],
        cur_quat_wxyz[3],
        cur_quat_wxyz[0],
    ])
    dq_wxyz = t3d.euler.euler2quat(
        float(drpy[0]), float(drpy[1]), float(drpy[2]), axes="sxyz"
    )
    delta_rot = R.from_quat([dq_wxyz[1], dq_wxyz[2], dq_wxyz[3], dq_wxyz[0]])
    qx, qy, qz, qw = (cur_rot * delta_rot).as_quat()
    new_xyz = cur_xyz + dxyz
    return np.array(
        [new_xyz[0], new_xyz[1], new_xyz[2], qw, qx, qy, qz, gripper_open],
        dtype=np.float64,
    )


class RemoteVLAPolicy(BaseVLAPolicy):
    IMG_PRIMARY = (224, 224)
    IMG_WRIST = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ee_command_pose = None

    def reset(self):
        super().reset()
        self._ee_command_pose = None

    def load_model(self):
        import requests

        self.server_url = self.config.get("server_url", "http://127.0.0.1:8765").rstrip("/")
        self.request_timeout = float(self.config.get("server_timeout", 60))
        r = requests.get(f"{self.server_url}/health", timeout=5)
        r.raise_for_status()
        print(f"[{type(self).__name__}] connected to {self.server_url}: {r.json()}")

    def _ee_delta_base_pose(self, current_pose) -> np.ndarray:
        current_pose = np.asarray(current_pose, dtype=np.float64).ravel()[:7]
        mode = str(self.config.get("ee_delta_integration", "target")).lower()
        if mode in {"measured", "current", "feedback"}:
            return current_pose
        if self._ee_command_pose is None:
            self._ee_command_pose = current_pose.copy()
        return self._ee_command_pose

    def _remember_ee_command_pose(self, absolute_action) -> np.ndarray:
        absolute_action = np.asarray(absolute_action, dtype=np.float64).ravel()
        if absolute_action.size >= 7:
            self._ee_command_pose = absolute_action[:7].copy()
        return absolute_action

    def preprocess_obs(self, obs):
        primary_cam = self.config.get("primary_camera", "head_camera")
        pipeline = self.config.get("image_pipeline")
        images = {
            "image_primary": encode_png(
                resize_like_training(obs["rgb"][primary_cam], self.IMG_PRIMARY, pipeline)
            )
        }
        wrist_cam = self.config.get("wrist_camera")
        if wrist_cam and self.IMG_WRIST and wrist_cam in obs["rgb"]:
            images["image_wrist"] = encode_png(
                resize_like_training(obs["rgb"][wrist_cam], self.IMG_WRIST, pipeline)
            )
        secondary_cam = self.config.get("secondary_camera")
        if secondary_cam and secondary_cam in obs["rgb"]:
            images["image_secondary"] = encode_png(
                resize_uint8(obs["rgb"][secondary_cam], self.IMG_PRIMARY)
            )
        return {
            "images": images,
            "proprio": extract_proprio(obs, self.action_type, self.dual_arm),
            "instruction": self.task_instruction,
        }

    def predict_raw(self, processed_obs):
        import requests

        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "unnorm_key": self.config.get("unnorm_key"),
        }
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["action"], dtype=np.float64)


class LeRobotRemotePolicy(RemoteVLAPolicy):
    IMG_PRIMARY = (256, 256)
    IMG_WRIST = (256, 256)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_call = True
        self._gripper_open_cmd = 1.0

    def reset(self):
        super().reset()
        self._first_call = True
        self._gripper_open_cmd = 1.0

    def preprocess_obs(self, obs):
        processed = super().preprocess_obs(obs)
        if self.action_type == "ee":
            if self.dual_arm:
                raise ValueError(f"{type(self).__name__} is single-arm only for EE eval.")
            current_pose = np.asarray(obs["ee_pose"]["right"], dtype=np.float64).ravel()[:7]
            gripper_closed = 1.0 - float(obs["gripper_val"]["right"])
            processed["current_ee_pose"] = current_pose
            processed["proprio"] = np.concatenate(
                [current_pose[:3], _quat_wxyz_to_euler(current_pose[3:7]), [gripper_closed]]
            ).astype(np.float32)
        return processed

    def predict_raw(self, processed_obs):
        import requests

        if self.dual_arm:
            raise ValueError(f"{type(self).__name__} is single-arm only.")
        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "proprio": [float(x) for x in np.asarray(processed_obs["proprio"]).tolist()],
            "reset": self._first_call,
        }
        self._first_call = False
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        delta = np.asarray(r.json()["action"], dtype=np.float64)
        if self.action_type == "ee":
            action = _lerobot_delta_to_absolute_ee(
                delta, self._ee_delta_base_pose(processed_obs["current_ee_pose"])
            )
            return self._remember_ee_command_pose(action)
        return delta

    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = super().postprocess_action(raw_action)
        if self.action_type not in {"qpos", "qpos_abs"} or "act_gripper_closed_cmd" not in self.config:
            return action
        close_threshold = float(self.config.get("act_gripper_close_threshold", 0.75))
        open_threshold = float(self.config.get("act_gripper_open_threshold", 0.90))
        closed_cmd = float(self.config["act_gripper_closed_cmd"])
        if action[-1] <= close_threshold:
            self._gripper_open_cmd = closed_cmd
        elif action[-1] >= open_threshold:
            self._gripper_open_cmd = 1.0
        action[-1] = self._gripper_open_cmd
        return action


class ACTPolicy(LeRobotRemotePolicy):
    pass


class Pi0Policy(LeRobotRemotePolicy):
    pass


class Pi05Policy(Pi0Policy):
    pass


class LeRobotDiffusionPolicy(LeRobotRemotePolicy):
    pass


class SmolVLAPolicy(LeRobotRemotePolicy):
    pass


class TemplatePolicy(BaseVLAPolicy):
    def load_model(self):
        pass

    def predict_raw(self, processed_obs):
        return np.zeros(self.action_dim, dtype=np.float64)


VLA_REGISTRY = {
    "act": ACTPolicy,
    "lerobot_diffusion": LeRobotDiffusionPolicy,
    "pi0": Pi0Policy,
    "pi05": Pi05Policy,
    "smolvla": SmolVLAPolicy,
    "template": TemplatePolicy,
}
