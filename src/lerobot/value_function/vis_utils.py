#!/usr/bin/env python
"""Shared visualization utilities for value and advantage function scripts."""

import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import ImageDraw, ImageFont


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class EpisodeInfo:
    ds_idx: int
    repo_id: str
    episode_index: int
    task_name: str
    length: int
    dataset_from_index: int
    dataset_to_index: int
    max_length: int = 0
    frame_start: int = 0
    frame_end: int = 0


# ── Output path helpers ─────────────────────────────────────────────


def derive_output_dir(checkpoint_path: str, head_type: str = "value_head") -> Path:
    """Derive default output dir from checkpoint path.

    Returns: visualization/{head_type}/{model_name}_{step}/
    Example: visualization/value_head/value_head_baseline_qwen3vl_20260413_150840_015000/
    """
    ckpt = Path(checkpoint_path).resolve()
    ckpt_dir = ckpt.parent if ckpt.is_file() else ckpt
    model_name = ckpt_dir.parent.name
    step = ckpt_dir.name.replace("checkpoint_", "")
    return Path("visualization") / head_type / f"{model_name}_{step}"


def video_filename(episode_info: EpisodeInfo, prefix: str = "") -> str:
    """Flat filename: [prefix_]<dataset_name>_ep<NNN>.mp4"""
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"{episode_info.repo_id}_ep{episode_info.episode_index:03d}")
    return "_".join(parts) + ".mp4"


# ── Image / video helpers ───────────────────────────────────────────


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


# ── Plot helpers ────────────────────────────────────────────────────


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def xy_points(values: np.ndarray, left: int, top: int, width: int, height: int, vmin: float = -1.0, vmax: float = 0.0) -> list[tuple[int, int]]:
    """Map (index, value) pairs to pixel coordinates."""
    if len(values) == 1:
        xs = np.array([left + width // 2], dtype=np.int32)
    else:
        xs = np.linspace(left, left + width, len(values)).round().astype(np.int32)
    frac = (values - vmin) / (vmax - vmin + 1e-12)
    ys = (top + height - frac * height).round().astype(np.int32)
    ys = np.clip(ys, top, top + height)
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def history_points(values: np.ndarray, current_index: int, left: int, top: int, width: int, height: int, vmin: float = -1.0, vmax: float = 0.0) -> list[tuple[int, int]]:
    """Like xy_points but only up to current_index."""
    history = values[: current_index + 1]
    if len(values) <= 1:
        xs = np.array([left], dtype=np.int32)
    else:
        xs = np.linspace(left, left + width, len(values)).round().astype(np.int32)[: current_index + 1]
    frac = (history - vmin) / (vmax - vmin + 1e-12)
    ys = (top + height - frac * height).round().astype(np.int32)
    ys = np.clip(ys, top, top + height)
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def draw_camera_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(70, 70, 70, 210))
    draw.text((x, y), label, fill=(245, 245, 245), font=font)


# ── Episode selection ───────────────────────────────────────────────


def episode_group_key(info: EpisodeInfo) -> str:
    match = re.search(r"(task\d+)", info.repo_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return info.task_name


def select_episode_infos(episode_infos: list[EpisodeInfo], selection_mode: str, episodes_per_group: int = 1) -> list[EpisodeInfo]:
    if selection_mode == "all":
        return list(episode_infos)

    if selection_mode == "per_task_first":
        chosen: dict[str, EpisodeInfo] = {}
        for info in episode_infos:
            chosen.setdefault(episode_group_key(info), info)
        return [chosen[k] for k in sorted(chosen)]

    if selection_mode == "per_task_longest":
        task_infos: dict[str, list[EpisodeInfo]] = {}
        for info in episode_infos:
            task_infos.setdefault(episode_group_key(info), []).append(info)
        return [
            max(infos, key=lambda i: (i.length, -i.ds_idx, -i.episode_index, i.repo_id))
            for infos in (task_infos[k] for k in sorted(task_infos))
        ]

    if selection_mode == "per_task_topk":
        episodes_per_group = max(1, int(episodes_per_group))
        task_infos: dict[str, list[EpisodeInfo]] = {}
        for info in episode_infos:
            task_infos.setdefault(episode_group_key(info), []).append(info)
        selected = []
        for k in sorted(task_infos):
            ranked = sorted(
                task_infos[k],
                key=lambda i: (i.length, -i.ds_idx, -i.episode_index, i.repo_id),
                reverse=True,
            )
            selected.extend(ranked[:episodes_per_group])
        return selected

    raise ValueError(f"Unsupported selection_mode: {selection_mode}")
