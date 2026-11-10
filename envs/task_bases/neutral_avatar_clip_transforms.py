"""Deterministic array transforms for neutral-avatar motion clips.

The helpers in this module know nothing about task lifecycle, scene layout,
Genesis, or avatar stepping.  They accept the historical task adapter so
subclass constants, config overrides, and motion-data stores are still looked
up dynamically at the exact point the legacy methods did.  Each transform
mutates only ``job`` and the destination entry in ``avatar.motion_data``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _indexed_motion_copy(
    motion: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: values[indices].copy() for name, values in motion.items()}


def install_laptop_pingpong_subclip(task: Any, job: dict) -> None:
    """Trim to the stationary laptop-typing window and ping-pong it."""
    if job.get("laptop_pingpong") is not None:
        return
    source_name = str(job["base_motion"])
    motion = task.avatar.motion_data[source_name]
    num_frames = int(motion["trans"].shape[0])
    start, end = task.NEUTRAL_LAPTOP_CLIP_TRIM
    start = int(np.clip(start, 0, num_frames - 2))
    end = int(np.clip(end, start + 1, num_frames - 1))
    loops = int(
        task.config.get(
            "neutral_avatar_laptop_pingpong_loops",
            task.NEUTRAL_LAPTOP_PINGPONG_LOOPS,
        )
    )
    loops = int(np.clip(loops, 1, 6))
    forward = list(range(start, end + 1))
    backward = list(range(end - 1, start, -1))
    indices = list(forward)
    for _ in range(loops - 1):
        indices.extend(backward)
        indices.extend(forward)
    indices = np.asarray(indices, dtype=int)
    destination = f"{source_name}_laptop_pingpong_{loops}"
    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["laptop_pingpong"] = loops
    job["clip"] = dict(job["clip"])
    job["clip"]["num_frames"] = int(indices.shape[0])


def install_typing_subclip(task: Any, job: dict) -> None:
    """Trim the typing clip's T-pose blend frames."""
    if job.get("typing_trim"):
        return
    source_name = str(job["base_motion"])
    motion = task.avatar.motion_data[source_name]
    num_frames = int(motion["trans"].shape[0])
    start, end = task.NEUTRAL_TYPING_CLIP_TRIM
    start = int(np.clip(start, 0, num_frames - 2))
    end = int(np.clip(end, start + 1, num_frames - 1))
    indices = np.arange(start, end + 1, dtype=int)
    destination = f"{source_name}_typing_trim"
    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["typing_trim"] = True
    job["clip"] = dict(job["clip"])
    job["clip"]["num_frames"] = int(indices.shape[0])


def install_clap_subclip(task: Any, job: dict) -> None:
    """Trim to the selected clapping period before the central loop wrap."""
    if job.get("clap_trim"):
        return
    source_name = str(job["base_motion"])
    motion = task.avatar.motion_data[source_name]
    num_frames = int(motion["trans"].shape[0])
    start, end = task.NEUTRAL_CLAP_CLIP_TRIM
    start = int(np.clip(start, 0, num_frames - 2))
    end = int(np.clip(end, start + 1, num_frames - 1))
    indices = np.arange(start, end + 1, dtype=int)
    destination = f"{source_name}_clap_trim_{start}_{end}"
    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["clap_trim"] = True
    job["clip"] = dict(job["clip"])
    job["clip"]["clip_start"] = start
    job["clip"]["clip_end"] = end
    job["clip"]["num_frames"] = int(indices.shape[0])


def install_mouse_subclip(task: Any, job: dict) -> None:
    """Trim mouse setup frames and repeat its attached forward segment."""
    if job.get("trim_start") is not None:
        return
    start, end = task.NEUTRAL_MOUSE_CLIP_TRIM
    source_name = str(job["base_motion"])
    motion = task.avatar.motion_data[source_name]
    num_frames = int(motion["trans"].shape[0])
    start = int(np.clip(start, 0, num_frames - 1))
    end = int(np.clip(end, start + 1, num_frames))
    attach_frame = 63
    detach_frame = None
    for event in job["events"]:
        if str(event.get("object")) == "mouse" and event.get("type") == "detach":
            detach_frame = int(event["clip_frame"]) - start
            break
    trimmed_frames = end - start
    if detach_frame is None:
        detach_frame = trimmed_frames - 1
    attach_frame = int(np.clip(attach_frame, 0, trimmed_frames - 2))
    detach_frame = int(
        np.clip(detach_frame, attach_frame + 1, trimmed_frames - 1)
    )

    forward_loops = int(
        task.config.get(
            "neutral_avatar_mouse_attached_forward_loops",
            task.NEUTRAL_MOUSE_ATTACHED_FORWARD_LOOPS,
        )
    )
    forward_loops = int(np.clip(forward_loops, 1, 5))
    destination = (
        f"{source_name}_neutral_trim_{int(start)}_{int(end)}"
        f"_mouse_pingpong_{forward_loops}"
    )

    prefix = list(range(start, start + attach_frame + 1))
    forward = list(
        range(start + attach_frame + 1, start + detach_frame + 1)
    )
    backward = list(
        range(start + detach_frame - 1, start + attach_frame - 1, -1)
    )
    attached = []
    for loop_index in range(forward_loops):
        if loop_index > 0:
            attached.extend(backward)
        attached.extend(forward)
    suffix = list(range(start + detach_frame + 1, end))
    indices = np.asarray(prefix + attached + suffix, dtype=int)

    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["trim_start"] = start
    job["mouse_attached_forward_loops"] = forward_loops
    attached_length = detach_frame - attach_frame
    expanded_detach_frame = (
        attach_frame
        + attached_length * forward_loops
        + attached_length * (forward_loops - 1)
    )
    for event in job["events"]:
        event["clip_frame"] = int(event["clip_frame"]) - start
        if str(event.get("object")) == "mouse" and event.get("type") == "attach":
            event["clip_frame"] = attach_frame
        elif (
            str(event.get("object")) == "mouse"
            and event.get("type") == "detach"
        ):
            event["clip_frame"] = expanded_detach_frame
        elif int(event["clip_frame"]) > detach_frame:
            event["clip_frame"] = int(event["clip_frame"]) + (
                expanded_detach_frame - detach_frame
            )
    job["events"] = [
        event
        for event in job["events"]
        if 0 <= int(event["clip_frame"]) < int(indices.shape[0])
    ]
    job["clip"] = dict(job["clip"])
    job["clip"]["num_frames"] = int(indices.shape[0])


def install_sponge_pingpong_subclip(task: Any, job: dict) -> None:
    """Repeat the sponge's attached wiping segment as a ping-pong clip."""
    if job.get("sponge_pingpong") is not None:
        return
    source_name = str(job["base_motion"])
    motion = task.avatar.motion_data[source_name]
    num_frames = int(motion["trans"].shape[0])
    attach_frame = None
    detach_frame = None
    for event in job["events"]:
        if str(event.get("object")) != "sponge":
            continue
        if event.get("type") == "attach":
            attach_frame = int(event["clip_frame"])
        elif event.get("type") == "detach":
            detach_frame = int(event["clip_frame"])
            break
    if attach_frame is None or detach_frame is None:
        return
    attach_frame = int(np.clip(attach_frame, 0, num_frames - 2))
    detach_frame = int(np.clip(detach_frame, attach_frame + 1, num_frames - 1))
    task_placement = task._neutral_avatar_task_placement(job)
    forward_loops = int(
        task.config.get(
            "neutral_avatar_sponge_attached_forward_loops",
            task_placement.get("sponge_attached_forward_loops", 3),
        )
    )
    forward_loops = int(np.clip(forward_loops, 1, 5))
    destination = f"{source_name}_sponge_pingpong_{forward_loops}"

    prefix = list(range(0, attach_frame + 1))
    forward = list(range(attach_frame + 1, detach_frame + 1))
    backward = list(range(detach_frame - 1, attach_frame - 1, -1))
    attached = []
    for loop_index in range(forward_loops):
        if loop_index > 0:
            attached.extend(backward)
        attached.extend(forward)
    suffix = list(range(detach_frame + 1, num_frames))
    indices = np.asarray(prefix + attached + suffix, dtype=int)

    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["sponge_pingpong"] = forward_loops
    attached_length = detach_frame - attach_frame
    expanded_detach_frame = (
        attach_frame
        + attached_length * forward_loops
        + attached_length * (forward_loops - 1)
    )
    for event in job["events"]:
        if (
            str(event.get("object")) == "sponge"
            and event.get("type") == "detach"
        ):
            event["clip_frame"] = expanded_detach_frame
        elif int(event["clip_frame"]) > detach_frame:
            event["clip_frame"] = int(event["clip_frame"]) + (
                expanded_detach_frame - detach_frame
            )
    job["clip"] = dict(job["clip"])
    job["clip"]["num_frames"] = int(indices.shape[0])


def install_reverse_subclip(task: Any, job: dict) -> None:
    """Install a reversed source clip for a take-objects neutral motion."""
    clip = job.get("clip", {}) or {}
    if not bool(clip.get("reverse")) or job.get("reversed_installed"):
        return
    source_name = str(clip.get("source_motion") or job["base_motion"])
    motion = task.avatar.motion_data.get(source_name)
    if motion is None:
        raise RuntimeError(
            f"reverse neutral motion: source clip {source_name!r} is missing "
            "from avatar.motion_data"
        )
    num_frames = int(motion["trans"].shape[0])
    indices = np.arange(num_frames - 1, -1, -1, dtype=int)
    destination = f"{source_name}_neutral_reverse"
    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["reversed_installed"] = True
    job["clip"] = dict(clip)
    job["clip"]["num_frames"] = int(num_frames)


def install_loop_subclip(task: Any, job: dict) -> None:
    """Wrap a neutral motion in one forward/reverse ping-pong pass."""
    if job.get("loop"):
        return
    if job.get("kind") == "put_objects_in_bowl" and not bool(
        (job.get("clip") or {}).get("reverse")
    ):
        return
    if not bool(
        task.config.get(
            "neutral_avatar_loop_motion",
            task.NEUTRAL_AVATAR_LOOP_MOTION,
        )
    ):
        return
    source_name = str(job["motion"])
    motion = task.avatar.motion_data.get(source_name)
    if motion is None:
        return
    num_frames = int(motion["trans"].shape[0])
    if num_frames < 3:
        return
    indices = np.asarray(
        list(range(num_frames)) + list(range(num_frames - 2, 0, -1)),
        dtype=int,
    )
    destination = f"{source_name}_neutral_loop"
    task.avatar.motion_data[destination] = _indexed_motion_copy(motion, indices)
    job["motion"] = destination
    job["clip"] = dict(job.get("clip", {}))
    job["clip"]["num_frames"] = int(indices.shape[0])
    job["loop"] = True


__all__ = (
    "install_clap_subclip",
    "install_laptop_pingpong_subclip",
    "install_loop_subclip",
    "install_mouse_subclip",
    "install_reverse_subclip",
    "install_sponge_pingpong_subclip",
    "install_typing_subclip",
)
